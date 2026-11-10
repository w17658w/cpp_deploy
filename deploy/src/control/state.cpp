// state.cpp — 状态组装实现
#include "state.h"

#include <cmath>
#include <algorithm>
void StateBuilder::init(double leg_mid, double leg_span) {
    mid_ = leg_mid;
    span_ = leg_span;
}

void StateBuilder::resetMotion() {
    has_prev_ = false;
}

static double wrapPi(double a) {
    const double PI = 3.14159265358979323846;
    while (a >  PI) a -= 2.0 * PI;
    while (a < -PI) a += 2.0 * PI;
    return a;
}

void StateBuilder::build(const RobotState& s, float obs[38]) {
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

    // [33..35] 三欧拉角原始弧度变化（带正负，wrapped 到 (-π,π]）
    if (has_prev_) {
        obs[33] = (float)wrapPi(s.roll  - prev_roll_);
        obs[34] = (float)wrapPi(s.pitch - prev_pitch_);
        obs[35] = (float)wrapPi(s.yaw   - prev_yaw_);
    } else {
        obs[33] = obs[34] = obs[35] = 0.0f;
    }

    // [36..37] 目标方向【相对机体系】：把世界系 (cosθ, sinθ) 用 yaw 旋转成机体系
    // 与 env_genesis_vec.py 的 _get_state() 一致：
    //   c_rel = cosθ*cos(yaw) + sinθ*sin(yaw)
    //   s_rel = sinθ*cos(yaw) - cosθ*sin(yaw)
    // 注：这两个数 hypot 恒为 1，不做归一化。
    {
        const double cy = std::cos(s.yaw), sy = std::sin(s.yaw);
        obs[36] = (float)(s.cmd_cos * cy + s.cmd_sin * sy);
        obs[37] = (float)(s.cmd_sin * cy - s.cmd_cos * sy);
    }

    // 更新上一帧姿态
    prev_roll_ = s.roll;
    prev_pitch_ = s.pitch;
    prev_yaw_ = s.yaw;
    has_prev_ = true;
}
