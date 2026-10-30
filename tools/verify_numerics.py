#!/usr/bin/env python3
"""
verify_numerics.py — 校验「.pth → .bin → C++ 前向」数值链路与 PyTorch 完全一致。

原理: 用 numpy 严格复刻 C++ mlp.cpp 的前向（同样的权重布局、ReLU、tanh*scale+bias），
     与 PyTorch 加载原始 .pth 做 forward 对比，两者误差应 < 1e-5。

用法:
    python verify_numerics.py <权重.pth> <导出.bin>
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

OBS, HIDDEN, ACT = 34, 256, 12
OUT = ACT * 2
SCALE = (0.09 - 0.01) / 2.0   # 0.04
BIAS = (0.09 + 0.01) / 2.0    # 0.05


def make_realistic_obs(seed=42):
    """构造一个贴近真实 34 维状态（量纲与 env_genesis_vec.py 一致）"""
    rng = np.random.default_rng(seed)
    obs = np.zeros(OBS, dtype=np.float32)
    obs[0:12] = rng.uniform(-1, 1, 12)          # 命令腿长归一化
    obs[12:15] = rng.uniform(-0.5, 0.5, 3)      # 姿态 rad
    obs[15:18] = rng.uniform(-2, 2, 3)          # 角速度 rad/s
    obs[18:30] = rng.integers(0, 2, 12)         # 触地 0/1
    obs[30:33] = rng.uniform(-2, 2, 3)          # 比力 g（静止约 ±1，动态 ±2~3）
    obs[33] = rng.uniform(0, 1)                 # 卡死量
    return obs


def torch_forward(pth_path, obs):
    data = torch.load(pth_path, map_location='cpu')
    sd = data['policy'] if 'policy' in data else data
    w1 = sd['net.0.weight']; b1 = sd['net.0.bias']
    w2 = sd['net.2.weight']; b2 = sd['net.2.bias']
    w3 = sd['net.4.weight']; b3 = sd['net.4.bias']

    x = torch.from_numpy(obs).float()
    h1 = F.relu(x @ w1.t() + b1)
    h2 = F.relu(h1 @ w2.t() + b2)
    o = h2 @ w3.t() + b3
    mean = o[:ACT]
    return (torch.tanh(mean) * SCALE + BIAS).numpy()


def cpp_forward(bin_path, obs):
    d = np.fromfile(bin_path, dtype=np.float32)
    i = 0
    w1 = d[i:i + HIDDEN * OBS].reshape(HIDDEN, OBS); i += HIDDEN * OBS
    b1 = d[i:i + HIDDEN]; i += HIDDEN
    w2 = d[i:i + HIDDEN * HIDDEN].reshape(HIDDEN, HIDDEN); i += HIDDEN * HIDDEN
    b2 = d[i:i + HIDDEN]; i += HIDDEN
    w3 = d[i:i + OUT * HIDDEN].reshape(OUT, HIDDEN); i += OUT * HIDDEN
    b3 = d[i:i + OUT]; i += OUT
    scale = d[i:i + ACT]; i += ACT
    bias = d[i:i + ACT]; i += ACT
    assert i == len(d), f'文件长度不匹配: 读 {i} 期望 {len(d)}'

    h1 = np.maximum(0.0, w1 @ obs + b1)
    h2 = np.maximum(0.0, w2 @ h1 + b2)
    o = w3 @ h2 + b3
    return np.tanh(o[:ACT]) * scale + bias


def main():
    pth, binpath = sys.argv[1], sys.argv[2]
    obs = make_realistic_obs()

    a_t = torch_forward(pth, obs)
    a_c = cpp_forward(binpath, obs)

    diff = np.max(np.abs(a_t - a_c))
    print(f'[check] 输入 34 维（真实量纲）')
    print(f'  PyTorch: {np.array2string(a_t, precision=6)}')
    print(f'  C++(numpy模拟): {np.array2string(a_c, precision=6)}')
    print(f'  最大绝对误差: {diff:.3e}')
    print('[PASS] 数值链路一致' if diff < 1e-5 else '[FAIL] 存在不一致，请检查导出布局或前向实现')


if __name__ == '__main__':
    main()
