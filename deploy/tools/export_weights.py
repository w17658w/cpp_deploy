#!/usr/bin/env python3
"""
export_weights.py — 把训练好的 SAC Actor 权重(.pth)导出为 C++ 可读的二进制 float32 裸数组。

用法（在 sac_env 里跑，训练程序暂停时）:
    python export_weights.py <输入.pth> [-o 输出.bin]

输入 .pth 支持两种格式:
    1. 直接是 policy.state_dict()  —— 如 weights_actor_*.pth / weights_best_actor_*.pth
    2. checkpoint  —— 含 'policy' key 的 dict（脚本自动取 ckpt['policy']）

导出布局（全部 float32，小端，与 C++ 端按同一顺序读取）:
    [W1]   256*36  = 9216    net.0.weight   (行主序 [out][in])
    [b1]   256               net.0.bias
    [W2]   256*256 = 65536   net.2.weight
    [b2]   256               net.2.bias
    [W3]   24*256  = 6144    net.4.weight
    [b3]   24                net.4.bias
    [scale] 12               action_scale = 0.04   (不在 state_dict 里，这里写死)
    [bias]  12               action_bias  = 0.05
总计 81456 个 float32 = 325824 字节

网络结构（GaussianPolicy, sac_agent.py，hidden_size=256）:
    输入 36 -> Linear(36,256) -> ReLU -> Linear(256,256) -> ReLU -> Linear(256,24)
    前 12 维是均值 mean，后 12 维是 log_std
    评估动作: action = tanh(mean) * action_scale + action_bias  ∈ [0.01, 0.09] m

注意: 本脚本只读 .pth、写 .bin，不 import/修改任何训练代码。
"""
import argparse
import os
import numpy as np
import torch

# 动作空间 gym.spaces.Box(low=[0.01]*12, high=[0.09]*12) 决定（见 env.py）
ACTION_SCALE = (0.09 - 0.01) / 2.0   # 0.04
ACTION_BIAS = (0.09 + 0.01) / 2.0    # 0.05
ACTION_DIM = 12

# 网络维度（与 sac_agent.py 的 hidden_size=256, obs_dim=36 一致）
OBS_DIM, HIDDEN = 36, 256
OUT_DIM = ACTION_DIM * 2             # 24


def to_np(t):
    """torch tensor -> numpy float32，显式搬到 CPU 并 detach。"""
    return t.detach().cpu().to(torch.float32).numpy()


def resolve_policy_state_dict(data):
    """兼容 checkpoint({'policy': ...}) 与纯 policy state_dict 两种输入。"""
    if isinstance(data, dict) and 'policy' in data:
        return data['policy']
    return data


def main():
    ap = argparse.ArgumentParser(description='SAC Actor 权重 -> C++ float32 裸数组')
    ap.add_argument('input', help='输入 .pth 路径（policy state_dict 或 checkpoint）')
    ap.add_argument('-o', '--output', help='输出 .bin 路径（默认与输入同名、.bin 后缀）')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f'[err] 输入文件不存在: {args.input}')

    data = torch.load(args.input, map_location='cpu')
    sd = resolve_policy_state_dict(data)
    # 去掉可能的 'policy.' 前缀，统一成裸 key
    clean = {k[7:] if k.startswith('policy.') else k: v for k, v in sd.items()}

    w1 = to_np(clean['net.0.weight'])   # [256, 36]
    b1 = to_np(clean['net.0.bias'])     # [256]
    w2 = to_np(clean['net.2.weight'])   # [256, 256]
    b2 = to_np(clean['net.2.bias'])     # [256]
    w3 = to_np(clean['net.4.weight'])   # [24, 256]
    b3 = to_np(clean['net.4.bias'])     # [24]

    # 维度校验：对不上会直接报错，防止拿到旧结构/错误文件
    assert w1.shape == (HIDDEN, OBS_DIM), f'[err] W1 形状 {w1.shape} != ({HIDDEN},{OBS_DIM})，obs_dim 对不上？'
    assert b1.shape == (HIDDEN,), f'[err] b1 形状 {b1.shape}'
    assert w2.shape == (HIDDEN, HIDDEN), f'[err] W2 形状 {w2.shape}'
    assert b2.shape == (HIDDEN,), f'[err] b2 形状 {b2.shape}'
    assert w3.shape == (OUT_DIM, HIDDEN), f'[err] W3 形状 {w3.shape} != ({OUT_DIM},{HIDDEN})，action_dim 对不上？'
    assert b3.shape == (OUT_DIM,), f'[err] b3 形状 {b3.shape}'

    scale = np.full(ACTION_DIM, ACTION_SCALE, dtype=np.float32)
    bias = np.full(ACTION_DIM, ACTION_BIAS, dtype=np.float32)

    out = args.output or (os.path.splitext(args.input)[0] + '.bin')
    with open(out, 'wb') as f:
        for arr in (w1, b1, w2, b2, w3, b3, scale, bias):
            f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())

    n = sum(a.size for a in (w1, b1, w2, b2, w3, b3, scale, bias))
    print(f'[ok] 导出完成 -> {out}')
    print(f'     布局: W1{w1.shape} b1{b1.shape} W2{w2.shape} b2{b2.shape} '
          f'W3{w3.shape} b3{b3.shape} scale{scale.shape} bias{bias.shape}')
    print(f'     共 {n} 个 float32 = {n * 4} 字节')
    print(f'     参考: 应与 C++ 端维度 {OBS_DIM}->{HIDDEN}->{HIDDEN}->{OUT_DIM} 一致')


if __name__ == '__main__':
    main()
