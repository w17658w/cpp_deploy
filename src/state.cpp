// state.cpp — 状态组装实现
#include "state.h"

#include <cmath>
#include <algorithm>

void StateBuilder::init(double leg_mid, double leg_span, double ori_ref) {
    mid_ = leg_mid;
    span_ = leg_span;
    ori_ref_ = ori_ref;
}

void StateBuilder::resetMotion() {
    has_prev_ = false;
}

void StateBuilder::build(const RobotState& s, float obs[34]) {
    // [0..11] 命令腿长归一化（-R 无位置反馈，用上一命令）
    for (int i = 0; i < 12; ++i)
        obs[i] = (float)(2.0 * (s.last_cmd_m[i] - mid_) / span_);

    // [12..14] 姿态（原始 rad，不归一化）
    obs[12] = (float)s.roll;
    obs[13] = (float)s.pitch;
    obs[14] = (float)s.yaw;

    // [15..17] 角速度（原始 rad/s，不归一化）
    obs[15] = (float)s.wx;
    obs[16] = (float)s.wy;
    obs[17] = (float)s.wz;

    // [18..29] 触地 0/1
    for (int i = 0; i < 12; ++i)
        obs[18 + i] = (float)((s.contact >> i) & 1u);

    // [30..32] 比力（g 单位，原始读数；已等于 sf_body/9.8，不额外归一化）
    obs[30] = (float)s.ax_g;
    obs[31] = (float)s.ay_g;
    obs[32] = (float)s.az_g;

    // [33] 卡死量（仅姿态变化项 d_ori/ori_ref；真机无定位，无位置项）
    double motion = 0.0;
    if (has_prev_) {
        double d_ori = std::fabs(s.roll - prev_roll_)
                     + std::fabs(s.pitch - prev_pitch_)
                     + std::fabs(s.yaw - prev_yaw_);
        motion = std::min(1.0, d_ori / ori_ref_);
    }
    obs[33] = (float)motion;

    // 更新上一帧姿态
    prev_roll_ = s.roll;
    prev_pitch_ = s.pitch;
    prev_yaw_ = s.yaw;
    has_prev_ = true;
}
