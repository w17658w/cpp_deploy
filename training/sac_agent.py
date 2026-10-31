import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


def _build_mlp(input_dim, output_dim, hidden_size):
    """构建一个两层隐藏层的 MLP，使用 ReLU 激活。"""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_dim),
    )


class GaussianPolicy(nn.Module):
    """高斯策略网络（Actor）。

    输入状态，输出一个高斯分布的均值和对数标准差（对角协方差），
    通过重参数化技巧采样动作，并加上 tanh 压缩到 [-1, 1]。
    """

    def __init__(self, obs_dim, action_dim, hidden_size, action_scale, action_bias,
                 log_std_min=-20.0, log_std_max=2.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_scale = action_scale
        self.action_bias = action_bias

        self.net = _build_mlp(obs_dim, action_dim * 2, hidden_size)
        # 最后一层输出 2*action_dim：前一半是均值，后一半是对数标准差

    def forward(self, state):
        x = self.net(state)
        mean, log_std = x.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        """采样动作，返回 (动作, 动作的对数概率, 原始均值)。"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()                 # 重参数化采样
        y_t = torch.tanh(x_t)                  # tanh 压缩到 [-1,1]
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # tanh 压缩后的 log 概率修正
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean


class QNetwork(nn.Module):
    """Q 价值网络（Critic）：评估 (state, action) 的价值。"""

    def __init__(self, obs_dim, action_dim, hidden_size):
        super().__init__()
        self.net = _build_mlp(obs_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class soft_actor_critic_agent:
    """SAC 智能体。

    接口严格对齐 SAC-main.py 的调用方式：
        - soft_actor_critic_agent(obs_dim, action_space, device=..., hidden_size=...,
                                  seed=..., lr=..., gamma=..., tau=..., alpha=...)
        - select_action(state, eval=False) -> numpy 动作
        - update_parameters(memory, batch_size, updates) -> None
        - .policy  (Actor 网络)  /  .critic  (Critic 网络)

    采用自动温度调节（alpha 为可学习参数），目标熵默认为 -action_dim。
    """

    def __init__(self, obs_dim, action_space, device="cpu", hidden_size=256, seed=0,
                 lr=1e-4, gamma=0.99, tau=0.005, alpha=0.2, target_entropy=None):
        # 记录初始传入的 alpha，仅作为自动温度调节的初始值
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)
        self.seed = seed

        # 从 gym 的 Box 动作空间提取维度与缩放/偏置，用于把 [-1,1] 映射到真实动作范围
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        action_dim = low.shape[0]
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.action_scale = torch.FloatTensor((high - low) / 2.0).to(self.device)
        self.action_bias = torch.FloatTensor((high + low) / 2.0).to(self.device)

        # 固定随机种子（保证复现）
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 四个网络
        self.policy = GaussianPolicy(obs_dim, action_dim, hidden_size,
                                     self.action_scale, self.action_bias).to(self.device)
        self.critic = QNetwork(obs_dim, action_dim, hidden_size).to(self.device)          # Q1
        self.critic_2 = QNetwork(obs_dim, action_dim, hidden_size).to(self.device)        # Q2
        self.critic_target = QNetwork(obs_dim, action_dim, hidden_size).to(self.device)
        self.critic_target_2 = QNetwork(obs_dim, action_dim, hidden_size).to(self.device)

        # 目标网络初始化与源网络一致，且不参与梯度
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False
        for p in self.critic_target_2.parameters():
            p.requires_grad = False

        # 自动温度调节：alpha = exp(log_alpha)
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
        if target_entropy is None:
            target_entropy = -action_dim
        self.target_entropy = target_entropy

        # 优化器
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.critic_optim = Adam(list(self.critic.parameters()) + list(self.critic_2.parameters()), lr=lr)
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

        self._updates = 0

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, eval=False):
        """根据策略选择动作。

        state: numpy 数组（obs_dim,）
        eval:  True 表示评估模式（确定性动作，取均值）；False 表示训练模式（加探索噪声）
        返回: numpy 数组 (action_dim,)
        """
        state_t = torch.FloatTensor(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if eval:
                mean, _ = self.policy(state_t)
                action = torch.tanh(mean) * self.action_scale + self.action_bias
            else:
                action, _, _ = self.policy.sample(state_t)
        return action.cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        """从记忆库采样一批数据，执行一次 SAC 参数更新。"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)

        state = torch.FloatTensor(state_batch).to(self.device)
        action = torch.FloatTensor(action_batch).to(self.device)
        reward = torch.FloatTensor(reward_batch).to(self.device)
        next_state = torch.FloatTensor(next_state_batch).to(self.device)
        mask = torch.FloatTensor(mask_batch).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            target_q1 = self.critic_target(next_state, next_action)
            target_q2 = self.critic_target_2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target_value = reward + mask * self.gamma * target_q

        # ---- 更新两个 Critic ----
        q1 = self.critic(state, action)
        q2 = self.critic_2(state, action)
        critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---- 更新 Actor ----
        new_action, log_prob, _ = self.policy.sample(state)
        q_new = torch.min(self.critic(state, new_action), self.critic_2(state, new_action))
        policy_loss = (self.alpha.detach() * log_prob - q_new).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # ---- 更新温度 alpha（自动调节）----
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ---- 软更新目标网络 ----
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self._updates += 1
