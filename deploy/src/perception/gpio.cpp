// gpio.cpp — 触地 GPIO 读取实现（libgpiod v1；无 libgpiod 时编译为 stub）
#include "gpio.h"

#include <cstdio>
#include <cstring>

#if defined(HAVE_GPIOD)
// ==================== Linux + libgpiod v1 ====================
#include <gpiod.h>

ContactGpio::~ContactGpio() { close(); }

void ContactGpio::init(const std::string& chip_path, const std::vector<int>& lines, bool active_high) {
    close();
    active_high_ = active_high;

    // 打开 gpiochip0
    chip0_ = gpiod_chip_open(chip_path.c_str());
    if (!chip0_) {
        std::fprintf(stderr, "[gpio] 打开 %s 失败（检查路径/权限）\n", chip_path.c_str());
        ok_ = false;
        return;
    }

    // 推导 gpiochip1 路径（把路径末尾的数字替换为 1）
    std::string chip1_path = chip_path;
    size_t last_slash = chip1_path.find_last_of('/');
    if (last_slash != std::string::npos) {
        chip1_path = chip1_path.substr(0, last_slash + 1) + "gpiochip1";
    } else {
        chip1_path = "/dev/gpiochip1";
    }
    chip1_ = gpiod_chip_open(chip1_path.c_str());
    if (!chip1_) {
        std::fprintf(stderr, "[gpio] 警告：打开 %s 失败（>=%d 的 line 将读不到）\n",
                     chip1_path.c_str(), CHIP1_OFFSET);
    } else {
        std::fprintf(stderr, "[gpio] 已打开 %s (chip0) 和 %s (chip1, GPIO>=%d)\n",
                     chip_path.c_str(), chip1_path.c_str(), CHIP1_OFFSET);
    }

    lines_.resize(lines.size(), nullptr);
    line_chip_idx_.resize(lines.size(), 0);

    for (size_t i = 0; i < lines.size(); ++i) {
        int global_line = lines[i];
        int chip_idx = (global_line >= CHIP1_OFFSET) ? 1 : 0;
        int local_line = (chip_idx == 1) ? (global_line - CHIP1_OFFSET) : global_line;

        struct gpiod_chip* chip = (chip_idx == 1)
            ? static_cast<struct gpiod_chip*>(chip1_)
            : static_cast<struct gpiod_chip*>(chip0_);

        if (!chip) {
            std::fprintf(stderr, "[gpio] line %d 属于 chip%d 但该 chip 未打开\n",
                         global_line, chip_idx);
            ok_ = false;
            return;
        }

        struct gpiod_line* l = gpiod_chip_get_line(chip, local_line);
        if (!l) {
            std::fprintf(stderr, "[gpio] 获取 line %d (local %d on chip%d) 失败\n",
                         global_line, local_line, chip_idx);
            ok_ = false;
            return;
        }
        if (gpiod_line_request_input(l, "contact") < 0) {
            std::fprintf(stderr, "[gpio] 请求 line %d 为输入失败（可能被占用/需权限）\n",
                         global_line);
            ok_ = false;
            return;
        }
        lines_[i] = l;
        line_chip_idx_[i] = chip_idx;
    }
    ok_ = true;
}

void ContactGpio::close() {
    if (chip0_) {
        gpiod_chip_close(static_cast<struct gpiod_chip*>(chip0_));
        chip0_ = nullptr;
    }
    if (chip1_) {
        gpiod_chip_close(static_cast<struct gpiod_chip*>(chip1_));
        chip1_ = nullptr;
    }
    lines_.clear();
    line_chip_idx_.clear();
    ok_ = false;
}

uint16_t ContactGpio::read() {
    if (!ok_) return 0;
    uint16_t contact = 0;
    for (size_t i = 0; i < lines_.size() && i < 12; ++i) {
        int v = gpiod_line_get_value(static_cast<struct gpiod_line*>(lines_[i]));
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

void ContactGpio::close() { chip0_ = nullptr; chip1_ = nullptr; lines_.clear(); ok_ = false; }

uint16_t ContactGpio::read() { return 0; }
#endif
