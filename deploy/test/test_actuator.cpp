// test_actuator.cpp — Arduino Mega2560 + 12 路舵机通信测试
//
// 用法:
//   ./test_actuator [port] [baud]
//   默认: /dev/ttyACM0 115200
//
// 流程（安全第一，先架空机器人！）:
//   1. 全部舵机回中 1500us (50mm)
//   2. 等待 2 秒
//   3. 依次单独扫每一路舵机: 1500 -> 1000 -> 1500 -> 2000 -> 1500，每步 1 秒
//      这样能看出哪路舵机对应哪条腿、方向对不对
//   4. 全部回中，结束
//
// 退出: 中途 Ctrl+C（舵机停在当前位置，不会回中）
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <thread>

#include "serial.h"
#include "protocol.h"
#include "actuator.h"

static std::atomic<bool> g_run{true};
static void onSig(int) { g_run = false; }

static void sendAll(Serial& ard, uint16_t us) {
    uint16_t pulse[12];
    for (int i = 0; i < 12; ++i) pulse[i] = us;
    uint8_t frame[DOWN_LEN];
    buildDownFrame(pulse, frame);
    ard.write(frame, DOWN_LEN);
}

static void sendOne(Serial& ard, int idx, uint16_t us, uint16_t rest_us) {
    uint16_t pulse[12];
    for (int i = 0; i < 12; ++i) pulse[i] = (i == idx) ? us : rest_us;
    uint8_t frame[DOWN_LEN];
    buildDownFrame(pulse, frame);
    ard.write(frame, DOWN_LEN);
}

int main(int argc, char** argv) {
    std::string port = (argc > 1) ? argv[1] : "/dev/ttyACM0";
    int baud = (argc > 2) ? std::atoi(argv[2]) : 115200;

    signal(SIGINT, onSig);
    signal(SIGTERM, onSig);

    std::printf("[test_actuator] 打开 %s @ %d ...\n", port.c_str(), baud);
    Serial ard;
    try {
        ard.open(port, baud);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[err] %s\n", e.what());
        return 1;
    }
    std::printf("[ok] Arduino 串口已打开\n");
    std::printf("========================================\n");
    std::printf(" 确认机器人已架空！按回车开始...\n");
    std::printf("========================================\n");
    std::getchar();

    // 1. 全部回中
    std::printf("[step1] 全部舵机回中 1500us (50mm)\n");
    sendAll(ard, 1500);
    for (int i = 0; i < 20 && g_run.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!g_run.load()) { ard.close(); return 0; }

    // 2. 依次扫每一路
    for (int leg = 0; leg < 12 && g_run.load(); ++leg) {
        std::printf("[step2] 测试 leg_%02d: 1500 -> 1000 -> 1500 -> 2000 -> 1500\n", leg + 1);

        sendOne(ard, leg, 1000, 1500);   // 全缩
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (!g_run.load()) break;

        sendOne(ard, leg, 1500, 1500);    // 回中
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (!g_run.load()) break;

        sendOne(ard, leg, 2000, 1500);    // 全伸
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (!g_run.load()) break;

        sendOne(ard, leg, 1500, 1500);    // 回中
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // 3. 全部回中
    if (g_run.load()) {
        std::printf("[step3] 全部回中 1500us\n");
        sendAll(ard, 1500);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    ard.close();
    std::printf("[exit] 舵机测试结束（舵机停在中位 1500us）\n");
    return 0;
}
