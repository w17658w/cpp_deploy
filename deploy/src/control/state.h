// state.h — 36 维状态组装（量纲/归一化严格对齐 env_genesis_vec.py 的 _get_state）
//
// 36 维分解（顺序与训练完全一致）:
//   [0..11]  命令腿长: norm = 2*(last_cmd_m - 0.05)/0.1   ∈ [-1,1]
//   [12..14] 姿态:     roll/pitch/yaw  原始 rad            （不归一化！）
//   [15..17] 角速度:   wx/wy/wz        原始 rad/s          （不归一化！）
//   [18..29] 触地:     0/1
//   [30..32] 比力:     ax/ay/az        g 单位（=线加速度-重力，加速度计原始读数）
//   [33..35] 姿态变化: Δroll/Δpitch/Δyaw 原始 rad（带正负，wrapped 到 (-π,π]）
//
// 注意：-R 无位置反馈，状态里没有"真实腿长"，只有"上一命令腿长"。
#pragma once

#include <cstdint>

struct RobotState {
    double roll = 0, pitch = 0, yaw = 0;   // 姿态 rad，来自 WT901（校正后）
    double wx = 0, wy = 0, wz = 0;         // 角速度 rad/s，来自 WT901（校正后）
    double ax_g = 0, ay_g = 0, az_g = 0;   // 比力 g，来自 WT901 加速度计（原始读数）
    uint16_t contact = 0;                  // 12 bit 触地（GPIO 直读）
    double last_cmd_m[12] = {0};           // 上一帧动作 (m)
};

class StateBuilder {
public:
    void init(double leg_mid, double leg_span);

    // 组装 36 维状态；内部维护上一帧姿态用于姿态变化量计算
    void build(const RobotState& s, float obs[36]);

    // 重置卡死量历史（开机/复位时）
    void resetMotion();

private:
    double mid_ = 0.05, span_ = 0.1;

    double prev_roll_ = 0, prev_pitch_ = 0, prev_yaw_ = 0;
    bool has_prev_ = false;
};
