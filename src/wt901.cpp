// wt901.cpp — WT901 协议解析与零偏校准实现
#include "wt901.h"

#include <cmath>

int16_t Wt901::s16(const uint8_t* p) {
    uint16_t u = (uint16_t)((p[1] << 8) | p[0]);
    return (int16_t)u;
}

void Wt901::feed(uint8_t byte) {
    switch (st_) {
        case WAIT_HEAD:
            if (byte == 0x55) { buf_[0] = byte; idx_ = 1; st_ = COLLECT; }
            break;
        case COLLECT:
            buf_[idx_++] = byte;
            if (idx_ >= FRAME_LEN) {   // 收满 11 字节，校验
                uint8_t sum = 0;
                for (int i = 0; i < FRAME_LEN - 1; ++i) sum += buf_[i];
                if (sum == buf_[FRAME_LEN - 1]) parseFrame();
                st_ = WAIT_HEAD;
            }
            break;
    }
}

void Wt901::parseFrame() {
    uint8_t type = buf_[1];
    const uint8_t* d = &buf_[2];   // 指向 D1L
    switch (type) {
        case 0x51:  // 加速度（比力）g
            raw_.ax_g = s16(&d[0]) / 32768.0 * accel_range_g_;
            raw_.ay_g = s16(&d[2]) / 32768.0 * accel_range_g_;
            raw_.az_g = s16(&d[4]) / 32768.0 * accel_range_g_;
            got_acc_ = true;
            break;
        case 0x52:  // 角速度 °/s
            raw_.wx_dps = s16(&d[0]) / 32768.0 * gyro_range_dps_;
            raw_.wy_dps = s16(&d[2]) / 32768.0 * gyro_range_dps_;
            raw_.wz_dps = s16(&d[4]) / 32768.0 * gyro_range_dps_;
            got_gyro_ = true;
            break;
        case 0x53:  // 角度 °
            raw_.roll_deg  = s16(&d[0]) / 32768.0 * angle_range_deg_;
            raw_.pitch_deg = s16(&d[2]) / 32768.0 * angle_range_deg_;
            raw_.yaw_deg   = s16(&d[4]) / 32768.0 * angle_range_deg_;
            got_angle_ = true;
            break;
        default:
            break;  // 忽略其他帧（磁场/四元数/时间等）
    }
}

// ---- 零偏校准 ----
void Wt901::beginCalib(int samples) {
    calib_samples_ = samples;
    calib_count_ = 0;
    gx_sum_ = gy_sum_ = gz_sum_ = 0;
    calib_on_ = false;
}

bool Wt901::calibDone() const { return calib_count_ >= calib_samples_; }

void Wt901::feedCalibSample() {
    if (calibDone() || !got_gyro_) return;
    gx_sum_ += wxRawRadS();
    gy_sum_ += wyRawRadS();
    gz_sum_ += wzRawRadS();
    ++calib_count_;
}

void Wt901::applyCalib() {
    if (calib_count_ > 0) {
        gx_bias_ = gx_sum_ / calib_count_;
        gy_bias_ = gy_sum_ / calib_count_;
        gz_bias_ = gz_sum_ / calib_count_;
    }
    // 初始姿态基准 = 校准完成时的当前角度
    r0_ = raw_.roll_deg  * DEG2RAD;
    p0_ = raw_.pitch_deg * DEG2RAD;
    y0_ = raw_.yaw_deg   * DEG2RAD;
    calib_on_ = true;
}

// ---- 校正后物理量 ----
double Wt901::rollRad()  const { return (raw_.roll_deg  * DEG2RAD) - r0_; }
double Wt901::pitchRad() const { return (raw_.pitch_deg * DEG2RAD) - p0_; }
double Wt901::yawRad()   const { return (raw_.yaw_deg   * DEG2RAD) - y0_; }

double Wt901::wxRadS() const { return calib_on_ ? wxRawRadS() - gx_bias_ : wxRawRadS(); }
double Wt901::wyRadS() const { return calib_on_ ? wyRawRadS() - gy_bias_ : wyRawRadS(); }
double Wt901::wzRadS() const { return calib_on_ ? wzRawRadS() - gz_bias_ : wzRawRadS(); }
