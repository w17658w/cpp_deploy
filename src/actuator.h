// actuator.h — L16 线性舵机「长度 <-> 脉宽」映射
//
// L16-100-35-6-R：1.0ms=全缩回(0mm)，2.0ms=全伸出(100mm)，线性映射。
//   脉宽(μs) = pulse_min + (len_m / stroke_m) * (pulse_max - pulse_min)
#pragma once

#include <cstdint>

class ActuatorMap {
public:
    void init(double stroke_m, double pulse_min_us, double pulse_max_us);

    // 目标长度(m) -> 脉宽(μs)，自动 clamp 到行程两端
    uint16_t lengthToPulse(double length_m) const;

    // 脉宽(μs) -> 长度(m)，用于回读换算
    double pulseToLength(uint16_t pulse_us) const;

private:
    double stroke_ = 0.1;
    double pmin_ = 1000, pmax_ = 2000;
};
