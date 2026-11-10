// cmd_link.h — WiFiLink 方向指令 UDP 接收（香橙派端 bind，外部设备发）
//
// 线格式（6 字节，小端）：
//   [0] 0xA5 帧头
//   [1] 0x01 类型（=设置世界系目标方向）
//   [2] seq（发送端递增，便于丢包统计）
//   [3] int8(cosθ * 127)
//   [4] int8(sinθ * 127)
//   [5] CRC8 over [0..4]，poly=0x07
//
// 语义：收端 hypot 归一化后存为世界系单位向量 (cosθ, sinθ)。
// 本类只负责"收包"，不负责转成相对机体系；相对化在 StateBuilder 里用 IMU yaw 算。
#pragma once

#include <cstdint>

struct CmdFrame {
    double cos_world = -1.0;   // 默认 180°（当前唯一能走 4m 的方向）
    double sin_world =  0.0;
    uint64_t recv_ms  =  0;     // 最近一帧到达时间（steady_clock ms）
    bool     fresh    = false;  // 是否在 timeout_ms 内
    int      seq       = -1;
    long     lost      =  0;    // 校验失败/长度错误计数（诊断）
};

class CmdLink {
public:
    bool open(uint16_t port);
    void close();
    bool ok() const { return fd_ >= 0; }

    // 阻塞最多 timeout_ms 收一帧；合法就更新 out。一般在独立线程里循环调用。
    // TODO(预留): WiFiLink 方案未最终拍板，本头文件仅保留接口设计，
    //             暂不提供 .cpp、不接入 CMake、不被 main.cpp 引用。
    void recvLoop(CmdFrame& out, uint64_t now_ms, int timeout_ms, bool& running);

private:
    int fd_ = -1;
};
