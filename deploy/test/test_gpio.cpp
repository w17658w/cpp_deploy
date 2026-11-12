// test_gpio.cpp — 触地 GPIO 测试（支持模拟信号粗测）
//
// 用法:
//   ./test_gpio [chip] [line0,line1,...] [active_high]
//   默认: /dev/gpiochip0  41,139,96,357,356,140,141,36,97,99,38,98  true
//
// 现象: 每 50ms 打印一次电平，变化时高亮显示。
//       模拟信号接 GPIO 也能测：按压时电压跨过阈值（~1.65V）会翻转。
//
// 退出: Ctrl+C
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "gpio.h"

static std::atomic<bool> g_run{true};
static void onSig(int) { g_run = false; }

static std::vector<int> parseIntList(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::atoi(item.c_str()));
    }
    return out;
}

int main(int argc, char** argv) {
    std::string chip = (argc > 1) ? argv[1] : "/dev/gpiochip0";
    std::string lines_str = (argc > 2) ? argv[2]
        : "41,139,96,357,356,140,141,36,97,99,38,98";
    bool active_high = (argc > 3) ? (std::string(argv[3]) != "false") : true;
    std::vector<int> lines = parseIntList(lines_str);

    signal(SIGINT, onSig);
    signal(SIGTERM, onSig);

    std::printf("[test_gpio] chip=%s  %zu 路 lines=[%s]  active_high=%d\n",
                chip.c_str(), lines.size(), lines_str.c_str(), (int)active_high);

    ContactGpio gpio;
    gpio.init(chip, lines, active_high);
    if (!gpio.ok()) {
        std::fprintf(stderr, "[err] GPIO 打开失败（检查: 1.libgpiod-dev 装了吗  2.引脚号对吗  3.权限 sudo 了吗）\n");
        return 1;
    }
    std::printf("[ok] GPIO 已打开，循环读取（Ctrl+C 退出）\n");
    std::printf("     按压传感器，对应位应翻转（模拟信号接 GPIO 也能粗测）\n\n");

    // 表头
    std::printf("  time(s)  | ");
    for (size_t i = 0; i < lines.size(); ++i) {
        std::printf("L%02d ", (int)(i + 1));
    }
    std::printf(" | raw\n");
    std::printf("-----------+");
    for (size_t i = 0; i < lines.size(); ++i) std::printf("----");
    std::printf("-+------\n");

    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_v = 0xFFFF;  // 初始化成不可能的值，第一帧必打印
    int tick = 0;
    while (g_run.load()) {
        uint16_t v = gpio.read();
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        // 只有变化时才打印，避免刷屏
        if (v != last_v) {
            std::printf("  %6.2f   | ", t);
            for (size_t i = 0; i < lines.size(); ++i) {
                int bit = (v >> i) & 1;
                int last_bit = (last_v >> i) & 1;
                // 变化的位高亮显示
                if (bit != last_bit) {
                    std::printf(" \033[1;31m%d\033[0m  ", bit);  // 红色高亮
                } else {
                    std::printf(" %d  ", bit);
                }
            }
            std::printf(" | 0x%03x\r\n", v);
            std::fflush(stdout);
            last_v = v;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        tick++;
    }

    gpio.close();
    std::printf("\n[exit] 共读 %d 次\n", tick);
    return 0;
}
