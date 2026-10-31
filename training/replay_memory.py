import numpy as np
import random


class ReplayMemory:
    """经验回放缓冲区：存储 (state, action, reward, next_state, mask) 五元组。

    使用环形缓冲区实现，容量满时覆盖最旧的数据。训练时随机采样一个 batch，
    打破样本之间的时间相关性，这是深度强化学习稳定的关键。
    """

    def __init__(self, seed, capacity):
        """初始化缓冲区。

        参数：
            seed:     随机种子，用于保证采样可复现
            capacity: 缓冲区最大容量（SAC-main.py 中为 1000000）
        """
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0  # 下一个写入位置
        self._seed = seed
        self._rng = random.Random(seed)

    def push(self, state, action, reward, next_state, mask):
        """存入一条经验。mask 表示该状态是否为终止状态（done 时 mask=0，否则为 1）。

        注意：这里统一把标量 reward / mask 转成 numpy 数组，方便后续批量拼接。
        """
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        reward = np.asarray([reward], dtype=np.float32)
        mask = np.asarray([mask], dtype=np.float32)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """随机采样 batch_size 条经验，返回按列拆分的 numpy 数组。"""
        batch = self._rng.sample(self.buffer, batch_size)
        state, action, reward, next_state, mask = map(np.stack, zip(*batch))
        return state, action, reward, next_state, mask

    def __len__(self):
        return len(self.buffer)
