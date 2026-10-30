// axis_calib.h — IMU 安装角补偿（把传感器系量转回机身系）
//
// 原理：传感器装在机身上有固定安装旋转 R_mount（传感器系 → 机身系）。
//   标定时机器人摆到"零姿态"（水平+朝前），读传感器欧拉角 (roll0,pitch0,yaw0)，
//   用它构造 R_mount。之后每帧把传感器量转回机身系。
//
// 欧拉角约定（对齐 WT901，Z-Y-X 内旋）：
//   R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
#pragma once

#include <array>
#include <cmath>

// 3x3 矩阵（行主序：[r0c0,r0c1,r0c2, r1c0,r1c1,r1c2, r2c0,r2c1,r2c2]）
using Mat3 = std::array<double, 9>;

// Z-Y-X 内旋欧拉角 → 旋转矩阵（WT901 约定）
inline Mat3 eulerToRotmat(double roll, double pitch, double yaw) {
    double cr = std::cos(roll), sr = std::sin(roll);
    double cp = std::cos(pitch), sp = std::sin(pitch);
    double cy = std::cos(yaw), sy = std::sin(yaw);
    return {
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp,     cp * sr,                cp * cr
    };
}

// 转置
inline Mat3 transpose(const Mat3& R) {
    return {R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]};
}

// 矩阵乘向量：out = R @ v
inline void matVec(const Mat3& R, const double v[3], double out[3]) {
    for (int i = 0; i < 3; ++i)
        out[i] = R[i * 3 + 0] * v[0] + R[i * 3 + 1] * v[1] + R[i * 3 + 2] * v[2];
}

// 旋转矩阵 → Z-Y-X 内旋欧拉角（WT901 约定）
inline void rotmatToEuler(const Mat3& R, double& roll, double& pitch, double& yaw) {
    roll  = std::atan2(R[7], R[8]);   // atan2(R21, R22)
    pitch = std::asin(-R[6]);         // asin(-R20)，WT901 俯仰 ±90° 内
    yaw   = std::atan2(R[3], R[0]);   // atan2(R10, R00)
}

// 安装补偿器：保存 R_mount^T，提供把传感器量转机身系的接口
class MountCalib {
public:
    // 标定：传入零姿态下传感器的欧拉角
    void calibrate(double roll0, double pitch0, double yaw0) {
        R_mount_T_ = transpose(eulerToRotmat(roll0, pitch0, yaw0));
        enabled_ = true;
    }

    // 姿态补偿：传感器欧拉角 → 机身欧拉角
    void applyOrientation(double roll, double pitch, double yaw,
                          double& roll_out, double& pitch_out, double& yaw_out) {
        if (!enabled_) { roll_out = roll; pitch_out = pitch; yaw_out = yaw; return; }
        Mat3 R_sensor = eulerToRotmat(roll, pitch, yaw);
        Mat3 R_robot = {  // R_sensor @ R_mount^T（行主序矩阵乘）
            R_sensor[0] * R_mount_T_[0] + R_sensor[1] * R_mount_T_[3] + R_sensor[2] * R_mount_T_[6],
            R_sensor[0] * R_mount_T_[1] + R_sensor[1] * R_mount_T_[4] + R_sensor[2] * R_mount_T_[7],
            R_sensor[0] * R_mount_T_[2] + R_sensor[1] * R_mount_T_[5] + R_sensor[2] * R_mount_T_[8],
            R_sensor[3] * R_mount_T_[0] + R_sensor[4] * R_mount_T_[3] + R_sensor[5] * R_mount_T_[6],
            R_sensor[3] * R_mount_T_[1] + R_sensor[4] * R_mount_T_[4] + R_sensor[5] * R_mount_T_[7],
            R_sensor[3] * R_mount_T_[2] + R_sensor[4] * R_mount_T_[5] + R_sensor[5] * R_mount_T_[8],
            R_sensor[6] * R_mount_T_[0] + R_sensor[7] * R_mount_T_[3] + R_sensor[8] * R_mount_T_[6],
            R_sensor[6] * R_mount_T_[1] + R_sensor[7] * R_mount_T_[4] + R_sensor[8] * R_mount_T_[7],
            R_sensor[6] * R_mount_T_[2] + R_sensor[7] * R_mount_T_[5] + R_sensor[8] * R_mount_T_[8],
        };
        rotmatToEuler(R_robot, roll_out, pitch_out, yaw_out);
    }

    // 向量补偿（比力/角速度）：传感器系 → 机身系
    void applyVec(const double v[3], double out[3]) {
        if (!enabled_) { out[0] = v[0]; out[1] = v[1]; out[2] = v[2]; return; }
        matVec(R_mount_T_, v, out);
    }

    bool enabled() const { return enabled_; }

private:
    Mat3 R_mount_T_ = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // 默认单位阵（未标定）
    bool enabled_ = false;
};
