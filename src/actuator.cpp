// actuator.cpp — 长度/脉宽映射实现
#include "actuator.h"

#include <algorithm>

void ActuatorMap::init(double stroke_m, double pulse_min_us, double pulse_max_us) {
    stroke_ = stroke_m;
    pmin_ = pulse_min_us;
    pmax_ = pulse_max_us;
}

uint16_t ActuatorMap::lengthToPulse(double length_m) const {
    double ratio = length_m / stroke_;
    ratio = std::max(0.0, std::min(1.0, ratio));          // clamp 到 [0,1]
    double us = pmin_ + ratio * (pmax_ - pmin_);
    return (uint16_t)(us + 0.5);                           // 四舍五入
}

double ActuatorMap::pulseToLength(uint16_t pulse_us) const {
    double us = (double)pulse_us;
    us = std::max(pmin_, std::min(pmax_, us));
    return (us - pmin_) / (pmax_ - pmin_) * stroke_;
}
