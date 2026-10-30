// wt901.h — WT901 九轴 IMU 的 0x55 协议解析 + 零偏校准
//
// 协议（维特智能标准，每帧 11 字节）:
//   0x55 TYPE D1L D1H D2L D2H D3L D3H X1 X2 SUM
//   TYPE=0x51 加速度(g), 0x52 角速度(°/s), 0x53 角度(°)
//   数据 int16 有符号、低字节在前；SUM = 前 10 字节累加取低 8 位
//   量程由 config 注入（默认 ±16g / ±4000°/s / ±180°，对应 WT9011G4K）
#pragma once

#include <cstdint>

struct ImuRaw {
    double ax_g = 0, ay_g = 0, az_g = 0;              // 加速度（比力）g
    double wx_dps = 0, wy_dps = 0, wz_dps = 0;        // 角速度 °/s
    double roll_deg = 0, pitch_deg = 0, yaw_deg = 0;  // 角度 °
};

class Wt901 {
public:
    // 逐字节喂入，内部状态机解析并校验
    void feed(uint8_t byte);

    // ---- 量程注入（须在喂数据前设置；默认按 WT9011G4K）----
    void setAccelRange(double g)   { accel_range_g_ = g; }
    void setGyroRange(double dps)  { gyro_range_dps_ = dps; }
    void setAngleRange(double deg) { angle_range_deg_ = deg; }

    // 是否已收到完整角度 + 角速度 + 加速度数据
    bool ready() const { return got_angle_ && got_gyro_ && got_acc_; }

    // ---- 零偏校准 ----
    void beginCalib(int samples);        // 开始采集零偏样本
    bool calibDone() const;              // 是否采集够样本
    void feedCalibSample();              // 采集当前原始值作为零偏样本（由调用方按节拍调用）
    void applyCalib();                   // 用样本均值计算零偏并启用

    // ---- 校正后物理量（rad / rad/s）----
    double rollRad()  const;   // 减去初始姿态角
    double pitchRad() const;
    double yawRad()   const;
    double wxRadS()   const;   // 减去陀螺仪零偏
    double wyRadS()   const;
    double wzRadS()   const;

    // ---- 加速度（比力，g 单位，原始读数）----
    double axG() const { return raw_.ax_g; }
    double ayG() const { return raw_.ay_g; }
    double azG() const { return raw_.az_g; }

    // ---- 原始物理量（rad / rad/s，未校正，调试用）----
    double rollRawRad()  const { return raw_.roll_deg  * DEG2RAD; }
    double pitchRawRad() const { return raw_.pitch_deg * DEG2RAD; }
    double yawRawRad()   const { return raw_.yaw_deg   * DEG2RAD; }
    double wxRawRadS()   const { return raw_.wx_dps    * DEG2RAD; }
    double wyRawRadS()   const { return raw_.wy_dps    * DEG2RAD; }
    double wzRawRadS()   const { return raw_.wz_dps    * DEG2RAD; }

private:
    static constexpr double DEG2RAD = 3.14159265358979323846 / 180.0;
    static constexpr int FRAME_LEN = 11;

    // 解析状态机
    enum State { WAIT_HEAD, COLLECT } st_ = WAIT_HEAD;
    uint8_t buf_[FRAME_LEN] = {0};
    int idx_ = 0;

    // 量程（config 注入）
    double accel_range_g_ = 16.0;    // ±16g（手册确认）
    double gyro_range_dps_ = 4000.0; // WT9011G4K ±4000°/s（标准 WT901 为 2000）
    double angle_range_deg_ = 180.0; // X/Z ±180°（Y ±90°，编码刻度统一 180）

    ImuRaw raw_;
    bool got_angle_ = false, got_gyro_ = false, got_acc_ = false;

    // 零偏
    bool calib_on_ = false;
    int calib_samples_ = 0, calib_count_ = 0;
    double gx_sum_ = 0, gy_sum_ = 0, gz_sum_ = 0;
    double gx_bias_ = 0, gy_bias_ = 0, gz_bias_ = 0;      // 陀螺仪零偏 (rad/s)
    double r0_ = 0, p0_ = 0, y0_ = 0;                      // 初始姿态 (rad)

    void parseFrame();
    static int16_t s16(const uint8_t* p);
};
