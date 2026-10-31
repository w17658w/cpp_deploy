# -*- coding: utf-8 -*-
"""
SAC_vec_train.py —— 并行（n_envs）SAC 训练脚本（Genesis 批量版）
==================================================================
与 SAC-main.py 的关系：
  * 复用 sac_agent.py / replay_memory.py（零改动）
  * 训练循环改为向量化：一次推进 N 个环境，批量采样 N 组经验入库
  * done 的环境在轮内异步重置（reset_dones），环境不空转
  * 超参数与原 SAC-main.py 保持一致（可对比训练曲线）

诊断（每 100 轮）：XLS 落盘 scores/avg/test/max_y/alpha/六项分解/外置/回合轨迹/随机基线
      + PNG 2x4 面板（含随机基线/目标参考线、奖励分解与外置条形、回合内 y(t)/俯仰(t) 轨迹）。
"""
import os
import sys
import time
import numpy as np
import torch

from sac_agent import soft_actor_critic_agent
from replay_memory import ReplayMemory
from env_genesis_vec import VecRobotEnv

# ---------------- 超参数（与原 SAC-main.py 对齐） ----------------
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 命令行：python SAC_vec_train.py <mode> <num_episodes> <n_envs> [view]
#   mode: train(从零训练) | resume(断点续训) | play(可视化评估)
#   view: 第 4 参任意非空 → render=True 边训边看（train/resume 模式）
MODE = sys.argv[1].lower() if len(sys.argv) > 1 else 'train'
N_ENVS = int(sys.argv[3]) if len(sys.argv) > 3 else 8    # 并行环境数
VIEW = len(sys.argv) > 4                                  # 边训边看开关
batch_size = 256
start_steps = 8000      # 随机动作收集步数（总跨环境步数）
replay_size = 1000000
LEARNING_RATE = 0.00001
alpha = 0.2
gamma = 0.99
tau = 0.005
TARGET_ENTROPY = -34    # 与原版一致（补偿 action_scale≈0.04 的 log_prob 底座）

max_steps = 500         # 每轮控制步数（0.125s×500=62.5s/轮；终点改远+翻墙，从 300 加到 500）
num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 6000   # 轮数（train/resume=总轮数；play=评估轮数）
update_steps = N_ENVS   # 每窗梯度更新次数 = 环境数，使更新/样本比 ≈ 1.0（对齐单环境 PyBullet 的 1 更新/样本）
ACTION_REPEAT = 8       # 动作保持步数：8×0.125s=1s（1Hz 控制；原版 16 步=2s，此处按用户要求改 1Hz）
test_interval = 10      # 每 10 轮做一次确定性评估（eval）
test_episodes = 2       # 每次评估跑 2 轮
TRACE_INTERVAL = 100    # 捕获回合内轨迹的间隔（与诊断面板同频，画 轨迹y/轨迹俯仰）

CHECKPOINT_PATH = os.path.join('record data - net', 'checkpoint_vec.pth')

# ---------------- 环境与智能体 ----------------
# 课程学习：先 "flat"（平地无障碍）练走路，练好后再改成 "obstacle" 续训翻墙
TERRAIN = "flat"   # "flat"=平地 | "obstacle"=平地+两个10cm墙 | "wave"=波浪地形（原版）
env = VecRobotEnv(n_envs=N_ENVS, render=(MODE == 'play' or VIEW), terrain=TERRAIN)
agent = soft_actor_critic_agent(env_state_dim := 34, env.action_space,
                                device=device, hidden_size=256, seed=seed,
                                lr=LEARNING_RATE, gamma=gamma, tau=tau,
                                alpha=alpha, target_entropy=TARGET_ENTROPY)
memory = ReplayMemory(seed, replay_size)

print(f"并行环境: {N_ENVS} | 状态维 {env_state_dim} | 动作维 {env.action_space.shape[0]} | device {device}")


def select_actions_batch(states, eval=False):
    """批量采样动作：states (N,34) -> actions (N,12)。eval=True 用均值（确定性）。"""
    s = torch.FloatTensor(np.asarray(states, dtype=np.float32)).to(device)
    with torch.no_grad():
        if eval:
            mean, _ = agent.policy(s)
            act = torch.tanh(mean) * agent.action_scale + agent.action_bias
        else:
            act, _, _ = agent.policy.sample(s)
    return act.cpu().numpy()


def save_checkpoint(extra="", episode=None, total_steps=None):
    os.makedirs('record data - net', exist_ok=True)
    torch.save({
        'policy': agent.policy.state_dict(),
        'critic': agent.critic.state_dict(),
        'critic_2': agent.critic_2.state_dict(),
        'critic_target': agent.critic_target.state_dict(),
        'critic_target_2': agent.critic_target_2.state_dict(),
        'log_alpha': agent.log_alpha,
        'policy_optim': agent.policy_optim.state_dict(),
        'critic_optim': agent.critic_optim.state_dict(),
        'alpha_optim': agent.alpha_optim.state_dict(),
        'updates': agent._updates,
        'total_numsteps': total_steps,
        'episode': episode,
    }, CHECKPOINT_PATH + extra)
    print(f"  [save] checkpoint -> {CHECKPOINT_PATH}{extra} (ep {episode}, {total_steps} steps)")


def load_checkpoint(path):
    """加载 checkpoint，返回 (total_numsteps, episode)。失败返回 None。"""
    try:
        ckpt = torch.load(path, map_location=device)
        agent.policy.load_state_dict(ckpt['policy'])
        agent.critic.load_state_dict(ckpt['critic'])
        agent.critic_2.load_state_dict(ckpt['critic_2'])
        agent.critic_target.load_state_dict(ckpt['critic_target'])
        agent.critic_target_2.load_state_dict(ckpt['critic_target_2'])
        agent.log_alpha.data.copy_(ckpt['log_alpha'].data)
        agent.policy_optim.load_state_dict(ckpt['policy_optim'])
        agent.critic_optim.load_state_dict(ckpt['critic_optim'])
        agent.alpha_optim.load_state_dict(ckpt['alpha_optim'])
        agent._updates = ckpt.get('updates', 0)
        return ckpt.get('total_numsteps', 0), ckpt.get('episode', 0)
    except (RuntimeError, KeyError) as e:
        print('[load] checkpoint 加载失败（可能结构不匹配）:', e)
        return None


def run_eval(n_eps):
    """确定性评估：跑 n_eps 轮 eval（均值动作、无噪声），返回平均每轮奖励。"""
    total = 0.0
    for _ in range(n_eps):
        t_state, _ = env.reset()
        t_r = 0.0
        for step in range(max_steps):
            if step % ACTION_REPEAT == 0:
                t_action = select_actions_batch(t_state, eval=True)
            t_next, t_rewards, t_dones, _ = env.step(t_action)
            t_r += float(t_rewards.mean())
            t_state = env.reset_dones(t_dones)
            if np.all(t_dones):
                break
        total += t_r
    return total / n_eps


def _mini_plot(img, x0, y0, x1, y1, data, color, title, hline=None, hlabel=None):
    """OpenCV 单子图：一维序列，独立 min/max 缩放。hline: 水平参考虚线（数值），如随机基线/目标线。"""
    import cv2
    cv2.rectangle(img, (x0, y0), (x1, y1), (235, 235, 235), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 1)
    n = len(data)
    if n == 0:
        cv2.putText(img, title, (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return
    vmin, vmax = min(data), max(data)
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    pts = []
    for i, v in enumerate(data):
        px = x0 + int(i / (n - 1) * (x1 - x0)) if n > 1 else (x0 + x1) // 2
        py = y0 + int((1.0 - (v - vmin) / (vmax - vmin)) * (y1 - y0))
        pts.append((px, py))
    if n == 1:
        cv2.circle(img, pts[0], 3, color, -1)
    else:
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, color, 1)
    if hline is not None:
        hy = y0 + int((1.0 - (hline - vmin) / (vmax - vmin)) * (y1 - y0))
        hy = max(y0 + 2, min(y1 - 2, hy))
        dash = 6
        for x in range(x0, x1, dash * 2):
            cv2.line(img, (x, hy), (min(x + dash, x1), hy), (110, 110, 110), 1)
        cv2.putText(img, hlabel or ('%.1f' % hline), (x0 + 6, hy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)
    cv2.putText(img, title, (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, 'min %.2f max %.2f' % (vmin, vmax), (x0 + 6, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)


def _bar_plot(img, x0, y0, x1, y1, labels, values, title):
    """OpenCV 水平条形图：正负着色 + 数值标签（用于奖励分解/外置诊断的近 100 轮均值）。"""
    import cv2
    cv2.rectangle(img, (x0, y0), (x1, y1), (240, 240, 240), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 1)
    cv2.putText(img, title, (x0 + 6, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    n = len(values)
    if n == 0:
        return
    vmax = max(abs(v) for v in values) or 1.0
    bw = (y1 - y0 - 26) / n
    mid = x0 + 56
    span = (x1 - mid - 66)
    for i, v in enumerate(values):
        by = int(y0 + 24 + i * bw)
        bh = max(3, int(bw) - 4)
        w = max(1, int(abs(v) / vmax * span))
        color = (40, 160, 60) if v >= 0 else (200, 60, 60)
        sx = mid if v >= 0 else mid - w
        cv2.rectangle(img, (sx, by), (sx + w, by + bh), color, -1)
        cv2.putText(img, labels[i], (x0 + 6, by + bh - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv2.putText(img, '%.2f' % v, (x1 - 64, by + bh - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)


def write_diagnostics(ep):
    """每 100 轮：写 process_vec_{ep}.xls + 画诊断 PNG（2x4，含参考线/奖励分解/外置/回合轨迹）。

    XLS 表：scores / avg_scores / test_scores / max_y / alpha
          / comp(六项分解,6列) / extras(五项外置,5列) / trace(回合轨迹,4列) / random_base(随机基线)
    PNG 面板（2x4）：
      Avg(100)｜Test(eval)｜MaxY(m)｜Alpha
      Comp(近100)｜Extras(近100)｜轨迹y(t)｜轨迹俯仰(t)
    """
    import xlwt
    import cv2
    os.makedirs('record data - reward', exist_ok=True)

    wb = xlwt.Workbook()
    def _sheet(name, data):
        sh = wb.add_sheet(name, cell_overwrite_ok=True)
        for i, v in enumerate(data):
            sh.write(i, 0, float(v))
    def _sheet_cols(name, arr):
        sh = wb.add_sheet(name, cell_overwrite_ok=True)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        for j in range(arr.shape[1]):
            for i in range(arr.shape[0]):
                v = arr[i, j]
                sh.write(i, j, float(v) if np.isfinite(v) else '')
    _sheet('scores', scores_array)
    _sheet('avg_scores', [float(np.mean(scores_array[max(0, i - 99):i + 1])) for i in range(len(scores_array))])
    _sheet('test_scores', test_scores_array)
    _sheet('max_y', max_y_array)
    _sheet('alpha', alpha_array)
    _sheet_cols('comp', comp_array)
    _sheet_cols('extras', extra_array)
    if trace_buf is not None:
        ty, ta, tp, tr = trace_buf
        L = max(len(ty), len(ta), len(tp), len(tr))
        rows = [[(ty[k] if k < len(ty) else float('nan')),
                 (ta[k] if k < len(ta) else float('nan')),
                 (tp[k] if k < len(tp) else float('nan')),
                 (tr[k] if k < len(tr) else float('nan'))] for k in range(L)]
        _sheet_cols('trace', rows)
    if random_baseline is not None:
        _sheet('random_base', [random_baseline])
    wb.save(f'record data - reward/process_vec_{ep}.xls')

    W, H = 1500, 900
    img = np.full((H, W, 3), 255, np.uint8)
    m = 30
    sw, sh = (W - 5 * m) // 4, (H - 5 * m) // 2
    # 8 个格子坐标（2 行 × 4 列）
    cells = []
    for r in range(2):
        for c in range(4):
            cells.append((m + c * (sw + m), m + r * (sh + m),
                          m + (c + 1) * sw + c * m, m + (r + 1) * sh + r * m))
    avg_arr = [float(np.mean(scores_array[max(0, i - 99):i + 1])) for i in range(len(scores_array))]
    _mini_plot(img, *cells[0], avg_arr, (200, 60, 60), 'Avg(100)', hline=random_baseline, hlabel='随机基线')
    _mini_plot(img, *cells[1], test_scores_array, (60, 140, 60), 'Test(eval)', hline=random_baseline, hlabel='随机基线')
    _mini_plot(img, *cells[2], max_y_array, (200, 140, 0), 'MaxY(m)', hline=2.0, hlabel='目标2.0')
    _mini_plot(img, *cells[3], alpha_array, (200, 80, 200), 'Alpha')
    comp_np = np.asarray(comp_array, dtype=np.float64)
    extra_np = np.asarray(extra_array, dtype=np.float64)
    comp_avg = np.mean(comp_np[-100:], axis=0) if len(comp_np) else np.zeros(6)
    extra_avg = np.mean(extra_np[-100:], axis=0) if len(extra_np) else np.zeros(5)
    _bar_plot(img, *cells[4], ["alive", "progress", "distance", "velocity", "fin", "milestone"], list(comp_avg), 'Comp(近100均值)')
    _bar_plot(img, *cells[5], ["force", "x_line", "height", "back", "partial"], list(extra_avg), 'Extras(近100均值)')
    if trace_buf is not None:
        ty, ta, tp, tr = trace_buf
        _mini_plot(img, *cells[6], list(ty), (30, 110, 210), '轨迹y(t)')
        _mini_plot(img, *cells[7], list(tp), (210, 110, 60), '轨迹俯仰(t)')
    else:
        _mini_plot(img, *cells[6], [], (30, 110, 210), '轨迹y(t)')
        _mini_plot(img, *cells[7], [], (210, 110, 60), '轨迹俯仰(t)')
    ok, buf = cv2.imencode('.png', img)
    if ok:
        buf.tofile(f'record data - reward/diagnostic_vec_{ep}.png')


def play(num_eps):
    """可视化评估：用当前权重跑确定性动作，render 观察（需 MODE='play'，env render=True）。"""
    for ep in range(1, num_eps + 1):
        state, _ = env.reset()
        total_r = 0.0
        max_y = 0.0
        for step in range(max_steps):
            if step % ACTION_REPEAT == 0:
                action = select_actions_batch(state, eval=True)
            next_state, reward, done, info = env.step(action)
            total_r += float(reward.mean())
            max_y = max(max_y, float(info["max_y"].max()))
            state = env.reset_dones(done)
            time.sleep(1 / 240)
            if np.all(done):
                break
        print(f"[play] ep{ep:3d} 总奖励 {total_r:8.2f} 峰值Y {max_y:.3f}m")


# ---------------- 训练 ----------------
time_start = time.time()
round_t0 = time.time()
total_numsteps = 0          # 跨环境累计步数（用于随机探索阶段判断）
updates = 0
scores_deque = []           # 每轮 N 个环境的平均奖励
scores_array = []           # 每轮均奖（诊断）
max_y_array = []            # 每轮峰值 Y（诊断）
test_scores_array = []      # 每 test_interval 轮的 eval 均奖
alpha_array = []            # 每 100 轮的 alpha
best_max_y = 0.0            # 已存档最佳峰值 Y
comp_array = []             # 每轮六项奖励分解均值 (ep,6)
extra_array = []            # 每轮五项外置诊断均值 (ep,5)
random_scores = []          # 随机探索期每轮均奖（用于随机基线）
random_baseline = None      # 随机基线：首个非随机轮结束后固定
trace_buf = None            # 最近一次捕获的回合内轨迹 (y, act, pitch, roll)

# resume：MODE='resume' 时断点续训
RESUME = (MODE == 'resume')
start_episode = 1
if RESUME and os.path.exists(CHECKPOINT_PATH):
    loaded = load_checkpoint(CHECKPOINT_PATH)
    if loaded is not None:
        total_numsteps, ep = loaded
        start_episode = ep + 1
        updates = agent._updates
        print(f"[resume] 从 ep{start_episode} 续训（{total_numsteps} 步，{updates} 更新）")
    else:
        print('[resume] 无可兼容 checkpoint，从零开始')

# play 模式：加载 checkpoint 可视化评估后退出
if MODE == 'play':
    if os.path.exists(CHECKPOINT_PATH):
        if load_checkpoint(CHECKPOINT_PATH) is None:
            print('[play] checkpoint 结构不匹配，退出')
            sys.exit()
        print('[play] 已从 checkpoint 加载权重，开始可视化评估')
    else:
        print('[play] 无 checkpoint，请先运行 train')
        sys.exit()
    play(num_episodes)
    print('play 结束')
    sys.exit()

states, _ = env.reset()
for i_episode in range(start_episode, num_episodes + 1):
    episode_reward = np.zeros(N_ENVS, dtype=np.float32)
    ep_comp = np.zeros((N_ENVS, 6), dtype=np.float32)   # 六项奖励分解累计（alive/progress/distance/velocity/fin/milestone）
    ep_extra = np.zeros((N_ENVS, 5), dtype=np.float32)  # 五项外置诊断累计（不进总奖励）
    max_y_this = np.zeros(N_ENVS, dtype=np.float32)
    in_random = total_numsteps < start_steps            # 本轮是否处于随机探索阶段（随机基线）
    capture_trace = (i_episode % TRACE_INTERVAL == 0)   # 本轮捕获回合内轨迹（供诊断面板）
    if capture_trace:
        trace_y, trace_act, trace_pitch, trace_roll = [], [], [], []

    for step in range(max_steps):
        if step % ACTION_REPEAT == 0:    # 每 1s（8 步）采样一次动作，中间沿用
            if total_numsteps < start_steps:
                actions = np.random.uniform(0.01, 0.09, (N_ENVS, 12)).astype(np.float32)
            else:
                actions = select_actions_batch(states, eval=False)

        next_states, rewards, dones, info = env.step(actions)

        # 批量入库（N 条）
        for i in range(N_ENVS):
            memory.push(states[i], actions[i], rewards[i], next_states[i], float(1.0 - dones[i]))
        total_numsteps += N_ENVS
        episode_reward += rewards
        ep_comp += info["reward_comp"]
        ep_extra += info["penalty_extras"]
        if capture_trace:
            trace_y.append(float(info["net_y"][0]))
            trace_act.append(float(actions[0].mean()))
            trace_roll.append(float(next_states[0, 12]))
            trace_pitch.append(float(next_states[0, 13]))
        max_y_this = np.maximum(max_y_this, info["max_y"])

        # SAC 更新
        if len(memory) > batch_size:
            for _ in range(update_steps):
                agent.update_parameters(memory, batch_size, updates)
                updates += 1

        # done 的环境异步重置，继续参与本轮
        states = env.reset_dones(dones)

        if np.all(dones):
            break

    # 每轮结束：全部重置进入下一轮
    states, _ = env.reset()

    # 统计
    round_time = time.time() - round_t0
    round_t0 = time.time()
    mean_r = float(np.mean(episode_reward))
    scores_deque.append(mean_r)
    avg100 = float(np.mean(scores_deque[-100:])) if len(scores_deque) >= 100 else float(np.mean(scores_deque))
    scores_array.append(mean_r)
    max_y_array.append(float(max_y_this.max()))
    # 诊断数据记账：六项分解 / 外置 / 随机基线 / 回合轨迹
    comp_array.append(np.mean(ep_comp, axis=0))
    extra_array.append(np.mean(ep_extra, axis=0))
    if in_random:
        random_scores.append(mean_r)
    elif random_baseline is None and random_scores:
        random_baseline = float(np.mean(random_scores))
    if capture_trace:
        trace_buf = (np.asarray(trace_y, dtype=np.float32), np.asarray(trace_act, dtype=np.float32),
                     np.asarray(trace_pitch, dtype=np.float32), np.asarray(trace_roll, dtype=np.float32))

    # best 存档：本轮峰值 Y 创新高
    if max_y_array[-1] > best_max_y:
        best_max_y = max_y_array[-1]
        try:
            torch.save({'policy': agent.policy.state_dict(), 'critic': agent.critic.state_dict()},
                       f"record data - net/vec_weights_best_ep{i_episode}_{best_max_y:.2f}.pth")
            print(f"  [best] 峰值 Y 新高 {best_max_y:.3f}m @ ep{i_episode}")
        except Exception as e:
            print("  [warn] best 保存失败:", repr(e))

    if i_episode % 10 == 0 or i_episode == 1:
        s = int(time.time() - time_start)
        print(f"[ep {i_episode:5d}] 总步数 {total_numsteps:8d} | 本轮均奖 {mean_r:8.2f} | "
              f"近100轮均奖 {avg100:8.2f} | 峰值Y {float(max_y_this.max()):.3f}m | "
              f"本轮耗时 {round_time:5.1f}s | 决策产能 {N_ENVS*300/round_time:7.0f}决策/s | "
              f"alpha {float(agent.alpha.detach()):.4f} | 时间 {s//3600}:{s%3600//60:02d}:{s%60:02d}")

    # 确定性评估（eval）：每 test_interval 轮
    if i_episode % test_interval == 0:
        avg_test = run_eval(test_episodes)
        test_scores_array.append(avg_test)
        print(f"  [eval] ep{i_episode} 确定性均奖 {avg_test:8.2f}")
        states, _ = env.reset()   # eval 结束后恢复训练初始状态

    if i_episode % 20 == 0:   # 每 20 轮存一次 checkpoint（方便断点续训，不丢进度）
        save_checkpoint(episode=i_episode, total_steps=total_numsteps)

    if i_episode % 100 == 0:
        save_checkpoint(episode=i_episode, total_steps=total_numsteps)
        try:
            torch.save({'policy': agent.policy.state_dict(), 'critic': agent.critic.state_dict()},
                       f"record data - net/vec_weights_ep{i_episode}.pth")
        except Exception as e:
            print("  [warn] 权重保存失败:", repr(e))
        alpha_array.append(float(agent.alpha.detach()))
        try:
            write_diagnostics(i_episode)
            comp_last = np.mean(np.asarray(comp_array)[-100:], axis=0)
            extra_last = np.mean(np.asarray(extra_array)[-100:], axis=0)
            print("  [diag] 六项均: " + " | ".join(f"{n}={v:.2f}" for n, v in
                  zip(["alive", "progress", "distance", "velocity", "fin", "milestone"], comp_last)))
            print("  [diag] 外置均: " + " | ".join(f"{n}={v:.2f}" for n, v in
                  zip(["force", "x_line", "height", "back", "partial"], extra_last)))
            if random_baseline is not None:
                print(f"  [diag] 随机基线 {random_baseline:.2f} vs 近100均奖 {avg100:.2f}")
        except Exception as e:
            print("  [warn] 诊断面板生成失败:", repr(e))

print("训练完成")
save_checkpoint(episode=num_episodes, total_steps=total_numsteps)
