// gpio.cpp — 触地 GPIO 读取实现（libgpiod v1；无 libgpiod 时编译为 stub）
#include "gpio.h"

#include <cstdio>

#if defined(HAVE_GPIOD)
// ==================== Linux + libgpiod v1 ====================
#include <gpiod.h>

ContactGpio::~ContactGpio() { close(); }

void ContactGpio::init(const std::string& chip_path, const std::vector<int>& lines, bool active_high) {
    close();
    active_high_ = active_high;
    chip_ = gpiod_chip_open(chip_path.c_str());
    if (!chip_) {
        std::fprintf(stderr, "[gpio] 打开 %s 失败（检查路径/权限）\n", chip_path.c_str());
        ok_ = false;
        return;
    }
    lines_.resize(lines.size(), nullptr);
    for (size_t i = 0; i < lines.size(); ++i) {
        struct gpiod_line* l = gpiod_chip_get_line((struct gpiod_chip*)chip_, lines[i]);
        if (!l) {
            std::fprintf(stderr, "[gpio] 获取 line %d 失败\n", lines[i]);
            ok_ = false;
            return;
        }
        if (gpiod_line_request_input(l, "contact") < 0) {
            std::fprintf(stderr, "[gpio] 请求 line %d 为输入失败（可能被占用/需权限）\n", lines[i]);
            ok_ = false;
            return;
        }
        lines_[i] = l;
    }
    ok_ = true;
}

void ContactGpio::close() {
    if (chip_) {
        gpiod_chip_close((struct gpiod_chip*)chip_);
        chip_ = nullptr;
    }
    lines_.clear();
    ok_ = false;
}

uint16_t ContactGpio::read() {
    if (!ok_) return 0;
    uint16_t contact = 0;
    for (size_t i = 0; i < lines_.size() && i < 12; ++i) {
        int v = gpiod_line_get_value((struct gpiod_line*)lines_[i]);
        if (v < 0) continue;
        if ((v == 1) == active_high_) contact |= (uint16_t)(1u << i);
    }
    return contact;
}

#else
// ==================== stub（无 libgpiod / 非 Linux）====================
ContactGpio::~ContactGpio() { close(); }

void ContactGpio::init(const std::string&, const std::vector<int>&, bool) {
    std::fprintf(stderr, "[gpio] 未编译 libgpiod，触地 GPIO 不可用（read 恒返回 0）\n");
    ok_ = false;
}

void ContactGpio::close() { chip_ = nullptr; lines_.clear(); ok_ = false; }

uint16_t ContactGpio::read() { return 0; }
#endif
