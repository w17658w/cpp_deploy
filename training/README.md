# robot12/training — 12 腿机器人 SAC 训练

Genesis 物理引擎 + 手写 SAC（`sac_agent.py`）训练 12 腿棱柱腿机器人走平地并翻越障碍。

## 文件说明

| 文件 | 作用 |
|---|---|
| `SAC_vec_train.py` | 训练入口（`train`/`resume`/`play` 三种模式，向量化并行） |
| `env_genesis_vec.py` | Genesis 环境（平地/障碍地形、奖励、34 维状态） |
| `sac_agent.py` | 手写 SAC（Actor/Critic/Alpha 自动调温） |
| `replay_memory.py` | 经验回放缓冲 |
| `requirements.txt` | pip 依赖清单 |
| `../contract/policy_contract.json` | 训练↔部署共享的状态/动作/权重格式契约 |

## 环境搭建（可复现）

本训练环境较特殊：**Genesis 从源码 editable 装 + 一套 pip 依赖 + torch 按 CUDA 装**，且有一个 numpy 版本坑，务必按顺序。

### 1. 前置
- Python 3.10
- NVIDIA GPU + CUDA（torch 需 cu130 版）

### 2. 装 Genesis（从源码 editable）
```bash
# 拿到 genesis-world 1.4.5 源码后
cd genesis-world-1.4.5
pip install -e .
```

### 3. 建 venv + 装 pip 依赖
```bash
python -m venv .venv
# Linux/Mac: source .venv/bin/activate
# Windows:   .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. 装 torch（CUDA 版）
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

### 5. ⚠️ 关键坑：numpy 必须 pip/OpenBLAS 版，不能是 conda 的 MKL 版

如果 torch 是从 conda 环境带出来的（conda 的 numpy 是 MKL 版），会导致 Genesis 初始化渲染器时在
`trimesh.util.unitize` 崩 `0xc06d007f`（ERROR_MOD_NOT_FOUND）。

解决：在 venv 里用 pip 覆盖装 numpy/scipy（不碰 conda 环境）：
```bash
pip install --ignore-installed numpy==2.2.6 scipy==1.15.3
```

### 6. 运行
```bash
# 平地学走路（TERRAIN 在 SAC_vec_train.py 顶部，默认 "flat"）
python SAC_vec_train.py train 6000 8

# 续训 / 回放
python SAC_vec_train.py resume 6000 8
python SAC_vec_train.py play 5 1
```

## 训练产物

默认写到运行目录下的 `record data - net/`（权重/checkpoint）和 `record data - reward/`（诊断 XLS/PNG），
已加入 `.gitignore`（仓库根），不会进 git。
