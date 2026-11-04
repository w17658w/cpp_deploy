import math
import os
import numpy as np
from gym import spaces

import yaml

import genesis as gs

_gs_inited = False


def _load_robot_params():
    """加载 sim-to-real 机器人物理参数（robot_params.yaml，与部署端 config.yaml 共用同一物理量）。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robot_params.yaml')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


ROBOT_PARAMS = _load_robot_params()


def _ensure_gs_init(backend):
    global _gs_inited
    if not _gs_inited:
        gs.init(backend=backend)
        _gs_inited = True


def _build_wave_heightfield(size_m=12.0, h_scale=0.25, v_scale=0.005, num_waves=12.0, amplitude=0.05):   #生成高度场
    n = int(size_m / h_scale) + 1
    A = 0.5 * amplitude / v_scale
    div = (n - 1) / (num_waves * 2 * np.pi)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return (A * np.cos(yy / div) + A * np.sin(xx / div)).astype(np.float32)


def _quat_to_euler_batch(q):
    """批量四元数 (N,4) (w,x,y,z) -> 欧拉角 (N,3)，与 PyBullet getEulerFromQuaternion 同约定。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def _quat_to_rotmat_batch(q):
    """批量四元数 (N,4) (w,x,y,z) -> 旋转矩阵 (N,3,3)，body->world（R^T 即 world->body）。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r = np.stack([
        np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=-1),
        np.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], axis=-1),
        np.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], axis=-1),
    ], axis=-2)
    return r


def _quat_multiply_batch(q1, q2):
    """批量四元数乘法（Hamilton 积）q1 ⊗ q2，(N,4) (w,x,y,z)。"""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


# spawn 四元数的共轭（= base_link 系 → 自然机身系的旋转；状态姿态乘它后，静止水平时=0，与实机归零后的姿态对齐）
SPAWN_QUAT_INV = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 新版 base_link 出生即自然姿态（单位四元数），无需再旋转


# ---- 障碍翻越任务参数（终点/障碍布局，训练与部署共用）----
TARGET_Y = 10000.0               # 一直往前走：目标点设远，让"朝目标"≈"朝+y"（原 2.4 = 走到固定点停下）
FIN_Y = 10000.0                  # 一直往前走：终点设远，fin 永不触发（原 2.2）
OBSTACLE_YS = [1.0, 2.0]         # 两个 10cm 障碍中心 y
OBSTACLE_SIZE = (2.0, 0.1, 0.1)  # (x 2m, y 0.1m, z 0.1m)
OBSTACLE_TOP_Z = 0.1             # 障碍顶高
SPAWN_Z_FLAT = float(ROBOT_PARAMS['robot']['spawn_z'])   # 平地初始 z（从 robot_params.yaml 读）
SPAWN_Z_WAVE = 0.219             # 波浪地形初始 z


class VecRobotEnv:
    def __init__(self, n_envs=8, render=False, terrain="obstacle"):   # 并行数 / 可视化 / 地形
        self.n_envs = n_envs
        self.render = render
        self.terrain = terrain    # "flat"=平地无障碍 | "obstacle"=平地+两个10cm墙 | "wave"=波浪地形（原版）

        # 空间定义
        self.action_space = spaces.Box(low=np.array([0.01] * 12), high=np.array([0.09] * 12), dtype=np.float32)
        # 36 维状态（真机可观测对齐版）：12 命令腿长 + 3 欧拉 + 3 角速度 + 12 触地 + 3 比力 + 3 姿态变化
        self.state_space = spaces.Box(np.array([-5.0] * 36), np.array([5.0] * 36), dtype=np.float32)

        _ensure_gs_init(gs.gpu)

        # 场景（并行数在 build 时开启）
        self.scene = gs.Scene(              #整个大的场景
            show_viewer=render,
            sim_options=gs.options.SimOptions(dt=1 / 240, gravity=(0, 0, -9.8)), #每秒推进240次 ，重力
            rigid_options=gs.options.RigidOptions(enable_torsional_friction=True),   # 接触法线的扭转摩擦
        )
        if self.terrain == "wave":
            # 波浪地形：12m 网格整体平移到 (-6,-6,0)，覆盖世界 [-6,6]×[-6,6]
            self.floor = self.scene.add_entity(gs.morphs.Terrain(    #往场景里添加地板
                pos=(-6.0, -6.0, 0.0),
                horizontal_scale=0.25,
                vertical_scale=0.005,
                height_field=_build_wave_heightfield(num_waves=12.0, amplitude=0.05),
                name="wave_12_005_flat6",
            ))
            self.spawn_z = SPAWN_Z_WAVE
        else:
            # 平地（flat=无障碍 / obstacle=加两个 10cm 墙，横跨 x 逼翻越而非绕行）
            self.floor = self.scene.add_entity(gs.morphs.Plane())
            self.obstacles = []
            if self.terrain == "obstacle":
                for yc in OBSTACLE_YS:
                    self.obstacles.append(self.scene.add_entity(gs.morphs.Box(
                        pos=(0.0, yc, OBSTACLE_TOP_Z / 2.0),
                        size=OBSTACLE_SIZE,
                        fixed=True,
                    )))
            self.spawn_z = SPAWN_Z_FLAT

        self.robot = self.scene.add_entity(gs.morphs.URDF(    #往场景里添加机器人（URDF）
            file=ROBOT_PARAMS['robot']['urdf_file'],
            pos=(-0.046, 0.005, self.spawn_z),
            quat=(1.0, 0.0, 0.0, 0.0),
        ))
        self.scene.build(n_envs=n_envs, env_spacing=(13.0, 13.0))   # 创建场景 8个并行，各场景间隔13m，避免碰撞

        # 关节
        self._joint_dofs = list(range(6, self.robot.n_dofs))  #关节列表
        assert len(self._joint_dofs) == 24, f"期望 24 关节 dof，实际 {len(self._joint_dofs)}"

        # 动力学参数）
        _kp = ROBOT_PARAMS['joint']['kp']
        _kv = ROBOT_PARAMS['joint']['kv']
        _f = ROBOT_PARAMS['joint']['force_limit_N']
        self.robot.set_dofs_kp(_kp, dofs_idx_local=self._joint_dofs)   #控制的Kp（从 robot_params.yaml 读）
        self.robot.set_dofs_kv(_kv, dofs_idx_local=self._joint_dofs)    #控制的Kv
        self.robot.set_dofs_force_range(-_f, _f, dofs_idx_local=self._joint_dofs)  #控制的力范围

        # ---- 物理参数----
        _base_link = next(l for l in self.robot.links if l.name == "base_link")  #找urdf文件中的base_link
        self._base_link_idx = _base_link.idx_local   # 供 _get_specific_force 读 base_link 线加速度
        _base_link.set_mass(ROBOT_PARAMS['robot']['base_mass_kg'])  #基座的质量（从 robot_params.yaml 读）
        self.robot.set_friction(ROBOT_PARAMS['robot']['friction'])  #滑动摩擦系数（从 robot_params.yaml 读）
        self.robot.set_friction_torsional(0.005)  #扭转摩擦（单位是【米】=接触斑半径；参考 spinningFriction=0.1 是无量纲，不能照抄 0.1，否则=10cm 巨脚扭死；Genesis 默认 0.005m=5mm）
        self.robot.set_friction_rolling(0.0)    #滚动摩擦系数（对齐参考 PyBullet rollingFriction=0）
        try:
            self.floor.set_friction(1.0)   # 地板摩擦对齐参考 PyBullet lateralFriction=1.0
        except Exception as e:
            print("[warn] floor.set_friction 失败（跳过）:", repr(e))

        # ---- 每环境状态缓冲（N 维）----
        self.flag = np.zeros(n_envs, dtype=np.int32)
        self.step_count = np.zeros(n_envs, dtype=np.int32)
        self.window_start_pos = np.zeros((n_envs, 3), dtype=np.float32)   # 初始占位（首个窗口建立前不用）
        self.last_cmd = np.full((n_envs, 12), 0.03, dtype=np.float32)
        self.last_target = np.zeros((n_envs, 12), dtype=np.float32)   # 限速用：上一窗口实际下发目标（reset 后关节位置=0）
        self.max_leg_velocity = ROBOT_PARAMS['joint']['velocity_limit_m_s']   # 速度上限（从 robot_params.yaml 读，对齐 L16 电机）
        self.max_y_reached = np.zeros(n_envs, dtype=np.float32)
        self.milestone_count = np.zeros(n_envs, dtype=np.int32)
        self.fin_given = np.zeros(n_envs, dtype=bool)        # fin(+200) 一次性标志（与原版 fin_given 一致）
        self.action_old = np.zeros((n_envs, 24), dtype=np.float32)
        self.action_24 = np.zeros((n_envs, 24), dtype=np.float32)
        self._motion_change = None          # (N,3) 三欧拉角原始弧度变化（带正负，step 缓存，_get_state 读）
        self.index = []
        self.index_old = []
        self.hand_control = 1

        self.reset()

    # ------------------------------------------------------------------
    def _reset_envs(self, envs_idx):
        """重置指定环境（或全部）到初始构型并清零计数器。"""
        idx = np.asarray(envs_idx)
        if idx.ndim == 0:
            idx = idx[None]
        if idx.size == self.n_envs:  # 全部
            self.scene.reset()
        else:
            self.scene.reset(envs_idx=idx.tolist())

        pos = np.tile(np.array([-0.046, 0.005, self.spawn_z], dtype=np.float32), (idx.size, 1))
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (idx.size, 1))
        self.robot.set_pos(pos, envs_idx=idx.tolist(), zero_velocity=True)
        self.robot.set_quat(quat, envs_idx=idx.tolist(), zero_velocity=True)
        self.robot.set_dofs_position(np.zeros((idx.size, 24), dtype=np.float32),
                                     dofs_idx_local=self._joint_dofs,
                                     envs_idx=idx.tolist(), zero_velocity=True)

        self.flag[idx] = 0
        self.step_count[idx] = 0
        self.last_cmd[idx] = 0.03
        self.last_target[idx] = 0.0
        self.max_y_reached[idx] = 0.0
        self.milestone_count[idx] = 0
        self.fin_given[idx] = False
        self.action_old[idx] = 0.0
        self.action_24[idx] = 0.0
        if self._motion_change is not None:
            self._motion_change[idx] = 0.0

    def reset(self, envs_idx=None):
        """全部或部分重置。返回 (N,36) 状态。"""
        if envs_idx is None:
            self._reset_envs(np.arange(self.n_envs))
        else:
            self._reset_envs(envs_idx)
        return self._get_state(), self.hand_control

    def reset_dones(self, dones):
        """将 done=True 的环境单独重置（异步重置），返回更新后的 (N,36) 状态。"""
        done_idx = np.nonzero(dones)[0]
        if done_idx.size > 0:
            self._reset_envs(done_idx)
        return self._get_state()

    # ------------------------------------------------------------------
    def step(self, actions):
        """批量推进：actions (N,12) -> (N,36) 状态 / (N,) reward / (N,) done / info。"""
        actions = np.clip(np.asarray(actions, dtype=np.float32), 0.01, 0.09)
        self.last_cmd = actions.copy()   # [0:12] 用「原始目标值」（限速前），与部署端 last_cmd_m 一致
        # 限速（对齐原版 maxVelocity=0.032）：每 env.step 推进 30 物理步=0.125s，最大增量 0.004m
        max_delta = self.max_leg_velocity * (30.0 / 240.0)
        actions = np.clip(actions, self.last_target - max_delta, self.last_target + max_delta)
        self.last_target = actions.copy()

        body_old = self._get_pos()
        body_velocity_old = self._get_vel()
        ori_old = _quat_to_euler_batch(self._get_quat())   # 物理前姿态，算 motion_change 用

        # 24 关节目标：每条腿两段同目标（对应单环境版的 24 个位置控制）
        targets = np.repeat(actions, 2, axis=1)   # (N,24)
        self.robot.control_dofs_position(targets, dofs_idx_local=self._joint_dofs)

        # 推进 30 步
        for _ in range(30):
            self.scene.step()

        self.action_old = self._get_dof_pos()
        self.action_24 = self.action_old.copy()
        body_new = self._get_pos()
        # motion_change：本步姿态变化，在 step 里算一次并缓存；_get_state 直接读，
        #   避免 step + reset_dones 重复调 _get_state 把 _prev_ori 刷新、导致 state[33] 恒 0
        ori_new = _quat_to_euler_batch(self._get_quat())
        delta = ori_new - ori_old
        delta = (delta + np.pi) % (2 * np.pi) - np.pi   # 角度差 wrap 到 (-π, π]（防 roll/yaw 跨 ±π 跳变）
        self._motion_change = delta                     # (N,3) 三欧拉角原始弧度变化（带正负，不归一化）

        # 卡死 flag：y 无位移 且 z 不上升 才累计（翻墙时顶墙抬升 dz>0 不算卡死）
        dy = body_new[:, 1] - body_old[:, 1]
        dz = body_new[:, 2] - body_old[:, 2]
        self.flag += ((np.abs(dy) < 1e-6) & (dz < 1e-4)).astype(np.int32)
        self.step_count += 1

        # ---------------- 奖励（向量化，逻辑与 env_genesis.py 一致） ----------------
        # alive：每 8 步一个窗口，投影到目标方向
        window = 8
        sc = self.step_count
        new_window = (sc % window == 1)
        self.window_start_pos = np.where(new_window[:, None], body_old, self.window_start_pos)
        alive = np.zeros(self.n_envs, dtype=np.float32)
        end_window = sc % window == 0
        if np.any(end_window):
            to_target = np.stack([0.0 - self.window_start_pos[:, 0],
                                  TARGET_Y - self.window_start_pos[:, 1]], axis=-1)   # (N,2)
            n = np.linalg.norm(to_target, axis=-1)
            disp = np.stack([body_new[:, 0] - self.window_start_pos[:, 0],
                             body_new[:, 1] - self.window_start_pos[:, 1]], axis=-1)
            proj = np.einsum('nd,nd->n', disp, to_target) / np.maximum(n, 1e-9)
            alive = np.where(n < 1e-9, 0.0, np.where(proj > 0.01, 0.5, -0.2))
            alive = np.where(end_window, alive, 0.0)
        alive_r = alive

        # progress：y 前进量
        progress_r = dy * 100.0

        abs_x = np.abs(body_new[:, 0])   # 供 fin / 诊断使用（x 误差奖励已按用户要求移除）

        # velocity：目标方向速度投影（从机身指向目标 (0,TARGET_Y)）
        direction_rad = np.arctan2(TARGET_Y - body_old[:, 1], 0.0 - body_old[:, 0])   # (N,)
        v = self._get_vel()
        v_new = v[:, 0] * np.cos(direction_rad) + v[:, 1] * np.sin(direction_rad)
        velocity_penalty = v_new * 10.0

        # fin：到达终点（一次性：与原版 fin_given 语义一致，防停留每步刷 200）
        fin_cond = (abs_x <= 0.15) & (body_new[:, 1] >= FIN_Y)
        reward_fin = np.where(fin_cond & ~self.fin_given, 200.0, 0.0)
        self.fin_given = self.fin_given | fin_cond

        # 里程碑：每跨 0.15m +10（基于历史最高 y，单调递增防退回去重复给）
        self.max_y_reached = np.maximum(self.max_y_reached, body_new[:, 1])
        new_mc = (self.max_y_reached // 0.15).astype(np.int32)
        milestone_r = (new_mc - self.milestone_count).astype(np.float32) * 10.0
        self.milestone_count = np.maximum(self.milestone_count, new_mc)

        # 接触力 / 出界 / 高度 / 后退 / 事件奖励：仅计算供诊断（与原版 env.py 一致——
        #   原版这些项全部不在 self.rewards 里，属于历史遗留的注释奖励，不进总奖励）
        force_penalty = -np.abs(self._contact_force_penalty() * 0.1)
        x_line_penality = np.where(abs_x >= 0.3, -5.0, 0.0)
        height_penalty = np.where(body_new[:, 2] >= 0.40, -0.1, 0.0)
        back_penalty = np.where(body_new[:, 1] < 0, -100.0, 0.0)
        reward_partial = np.where((abs_x <= 0.022) & (body_new[:, 1] >= 0.5), 30.0, 0.0) \
            + np.where(body_new[:, 1] >= 1.0, 30.0, 0.0)

        # reward_distance：密集"保持在直线路径(x=0)上"惩罚（原 PyBullet 版有，Genesis 移植时被 climb 顶替丢了）
        #   到直线 x=0（y∈[0,3]）的最近距离，超 0.15m 死区后 -0.5×距离；给策略"贴直线走"的稠密梯度
        x = body_new[:, 0]
        y = body_new[:, 1]
        dist_to_line = np.abs(x)
        dist_to_line = np.where(y < 0.0, np.sqrt(x ** 2 + y ** 2), dist_to_line)
        dist_to_line = np.where(y > 3.0, np.sqrt(x ** 2 + (y - 3.0) ** 2), dist_to_line)
        reward_distance = -0.5 * np.maximum(0.0, dist_to_line - 0.15)

        # 奖励：6 项 = alive + progress + distance + velocity + fin + milestone
        rewards = (alive_r + progress_r + reward_distance + velocity_penalty
                   + reward_fin + milestone_r)

        # 奖励分解（训练诊断用：alive/progress/distance/velocity/fin/milestone）
        reward_comp = np.stack([alive_r, progress_r, reward_distance, velocity_penalty,
                                reward_fin, milestone_r], axis=-1)   # (N,6)

        states = self._get_state()
        dones = self._get_done()
        info = {"reward_comp": reward_comp, "max_y": self.max_y_reached.copy(),
                "net_y": body_new[:, 1].copy(), "flag": self.flag.copy(),
                "penalty_extras": np.stack([force_penalty, x_line_penality,
                                            height_penalty, back_penalty,
                                            reward_partial], axis=-1)}  # 诊断：5 项原版未计入奖励的分量

        return states, rewards, dones, info

    # ------------------------------------------------------------------
    # 状态读取（批量）
    def _get_pos(self):
        return np.asarray(self.robot.get_pos().cpu().numpy(), dtype=np.float32)      # (N,3)

    def _get_quat(self):
        return np.asarray(self.robot.get_quat().cpu().numpy(), dtype=np.float32)     # (N,4)

    def _get_vel(self):
        return np.asarray(self.robot.get_vel().cpu().numpy(), dtype=np.float32)      # (N,3)

    def _get_ang(self):
        return np.asarray(self.robot.get_ang().cpu().numpy(), dtype=np.float32)      # (N,3)

    def _get_specific_force(self, R_body2world):
        """机体系比力 specific force = 线加速度 - 重力（= 加速度计原始读数，等价原生 IMU 的 lin_acc）。
        先取 base_link 世界系经典线加速度 a（不含重力），减重力得世界系比力，再用 R^T 转机体系。"""
        a = self.robot.get_links_acc(links_idx_local=[self._base_link_idx], relative=False)   # (N,1,3) 世界系
        a = np.asarray(a.cpu().numpy(), dtype=np.float32).reshape(self.n_envs, 3)
        g = np.array([0.0, 0.0, -9.8], dtype=np.float32)
        sf_world = a - g                                                     # (N,3) 世界系比力
        sf_body = np.einsum('nij,nj->ni', np.transpose(R_body2world, (0, 2, 1)), sf_world)  # R^T @ sf_world
        return sf_body                                                       # (N,3)

    def _get_dof_pos(self):
        return np.asarray(self.robot.get_dofs_position().cpu().numpy(),
                          dtype=np.float32)[:, 6:]                                   # (N,24)

    def _get_contact_links(self):
        """批量触地：返回 (N, 列表[局部 link 索引]) 的紧凑表示。
        用 with_entity=None 把「地板 + 障碍」的接触都算作触地（翻越时脚踩在障碍上也应记为触地）。
        """
        contacts = self.robot.get_contacts(with_entity=None, exclude_self_contact=True)
        valid = contacts["valid_mask"].cpu().numpy()          # (N, Cmax) bool
        link_a = contacts["link_a"].cpu().numpy()             # (N, Cmax)
        link_b = contacts["link_b"].cpu().numpy()
        local = []
        for i in range(self.n_envs):
            links = set()
            va = link_a[i][valid[i]]
            vb = link_b[i][valid[i]]
            for v in va:
                if self.robot.link_start <= v < self.robot.link_end:
                    links.add(int(v) - self.robot.link_start)
            for v in vb:
                if self.robot.link_start <= v < self.robot.link_end:
                    links.add(int(v) - self.robot.link_start)
            local.append(links)
        return local

    def _contact_force_penalty(self):
        """批量接触力惩罚（(N,)）：每个接触点模长 <80N 累加；无接触罚 100。"""
        contacts = self.robot.get_contacts(with_entity=None, exclude_self_contact=True)
        valid = contacts["valid_mask"].cpu().numpy()          # (N,C)
        force_b = contacts["force_b"].cpu().numpy()           # (N,C,3)
        f = np.linalg.norm(force_b, axis=-1)                  # (N,C)
        f = np.where(valid, f, 0.0)
        penalty = np.where(f < 80.0, f, 0.0).sum(axis=-1)     # (N,)
        n_contacts = valid.sum(axis=-1)
        return np.where(n_contacts > 0, penalty, 100.0)

    def _get_done(self):
        return (self.flag > 100).copy()

    # ------------------------------------------------------------------
    def _get_state(self):
        """批量 36 维状态（真机可观测对齐版）：
        [0:12] 命令腿长 | [12:15] 欧拉角 | [15:18] 机体系角速度 | [18:30] 触地 | [30:33] 比力 | [33:36] 三欧拉角姿态变化。"""
        # 12 命令腿长归一化（-R 无位置反馈，状态只保留"上一命令"，不用真实 dof_pos）
        last_cmd_norm = 2.0 * (self.last_cmd - 0.05) / 0.1     # (N,12)

        # 姿态转到「自然机身系」（base_link 系乘 spawn⁻¹），静止水平时=0，与实机归零后的姿态对齐
        quat = _quat_multiply_batch(
            np.tile(SPAWN_QUAT_INV, (self.n_envs, 1)), self._get_quat())  # (N,4) 自然机身系四元数
        body_ori = _quat_to_euler_batch(quat)                  # (N,3)
        R = _quat_to_rotmat_batch(quat)                        # (N,3,3) 自然机身系 -> world
        R_t = np.transpose(R, (0, 2, 1))                       # world -> 自然机身系

        # 机体系角速度：世界系角速度用完整四元数 R^T 转机体系（与陀螺仪读数一致）
        w = np.einsum('nij,nj->ni', R_t, self._get_ang())      # (N,3) 机体系角速度

        # 触地杆 12 维
        contact_link = np.zeros((self.n_envs, 12), dtype=np.float32)
        for i, links in enumerate(self._get_contact_links()):
            for link_idx in links:
                if link_idx >= 2 and link_idx % 2 == 0:        # 触地杆 = 偶数 link（已校准）
                    leg = link_idx // 2 - 1
                    if 0 <= leg < 12:
                        contact_link[i, leg] = 1.0
        self.index_old = self.index
        self.index = [np.nonzero(c)[0].tolist() for c in contact_link]

        # 机体系比力（加速度计原始读数，含重力项，归一化 /9.8），与角速度共用同一 R
        sf_body = self._get_specific_force(R) / 9.8            # (N,3)

        # motion_change：读 step() 里算好的本步姿态变化（避免重复调 _get_state 刷新导致恒 0）
        motion_change = (self._motion_change if self._motion_change is not None
                         else np.zeros((self.n_envs, 3), dtype=np.float32))

        return np.concatenate([last_cmd_norm, body_ori, w, contact_link,
                               sf_body, motion_change], axis=-1)   # (N, 36)


if __name__ == "__main__":
    env = VecRobotEnv(n_envs=8)
    obs, hand = env.reset()
    print("reset 状态形状:", obs.shape, "| hand_control:", hand)
    for t in range(50):
        actions = np.random.uniform(0.01, 0.09, (env.n_envs, 12)).astype(np.float32)
        states, rewards, dones, info = env.step(actions)
        if t % 10 == 0:
            print(f"t={t} reward均值={rewards.mean():.2f} done数={int(dones.sum())} 触地腿数均值={int((states[:,18:30]>0.5).sum(1).mean())}")
        env.reset_dones(dones)
    print("并行环境验证完成")
