// gpio.h — 12 路触地 GPIO 读取（libgpiod v1，平台中立接口）
//
// 触地信号由压力传感器比较器模块直接输出 0/1，香橙派 GPIO 直读。
// 在 Linux 上走 libgpiod（/dev/gpiochipN 的 chip+line 编号）；
// 无 libgpiod / 非 Linux 时编译为 stub（read() 恒返回 0）。
//
// A733 有两个 gpiochip：
//   gpiochip0 = GPIO 0~351 (PORTA~PORTK)
//   gpiochip1 = GPIO 352~415 (PORTL)
// 本类自动按 line 号选择 chip，上层传入全局 GPIO 号即可。
#pragma once

#include <cstdint>
#include <vector>
#include <string>

class ContactGpio {
public:
    ContactGpio() = default;
    ~ContactGpio();

    // chip_path: 默认 "/dev/gpiochip0"；lines: 12 个全局 GPIO 号；active_high: 有效电平
    void init(const std::string& chip_path, const std::vector<int>& lines, bool active_high);
    void close();

    // 读 12 路，返回 12bit（bit i = 触点 i 有效）
    uint16_t read();

    bool ok() const { return ok_; }

private:
    void* chip0_ = nullptr;              // struct gpiod_chip* (gpiochip0, GPIO 0~351)
    void* chip1_ = nullptr;               // struct gpiod_chip* (gpiochip1, GPIO 352~415)
    std::vector<void*> lines_;           // struct gpiod_line*（每路一个，跨两个 chip）
    std::vector<int> line_chip_idx_;     // 每路属于哪个 chip（0 或 1）
    bool active_high_ = true;
    bool ok_ = false;

    // A733: gpiochip0 覆盖 GPIO 0~351，gpiochip1 覆盖 GPIO 352~415
    static constexpr int CHIP1_OFFSET = 352;
};
