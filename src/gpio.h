// gpio.h — 12 路触地 GPIO 读取（libgpiod v1，平台中立接口）
//
// 触地信号由压力传感器比较器模块直接输出 0/1，香橙派 GPIO 直读。
// 在 Linux 上走 libgpiod（/dev/gpiochipN 的 chip+line 编号）；
// 无 libgpiod / 非 Linux 时编译为 stub（read() 恒返回 0）。
#pragma once

#include <cstdint>
#include <vector>
#include <string>

class ContactGpio {
public:
    ContactGpio() = default;
    ~ContactGpio();

    // chip_path: "/dev/gpiochip0"；lines: 12 个 line 号；active_high: 有效电平
    void init(const std::string& chip_path, const std::vector<int>& lines, bool active_high);
    void close();

    // 读 12 路，返回 12bit（bit i = 触点 i 有效）
    uint16_t read();

    bool ok() const { return ok_; }

private:
    void* chip_ = nullptr;               // struct gpiod_chip*
    std::vector<void*> lines_;           // struct gpiod_line*
    bool active_high_ = true;
    bool ok_ = false;
};
