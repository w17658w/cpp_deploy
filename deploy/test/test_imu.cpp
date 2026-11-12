// test_imu.cpp — WT901 IMU 串口通信测试
//
// 用法:
//   ./test_imu [port] [baud]
//   默认: /dev/ttyUSB0 115200
//
// 现象: 每 100ms 打印一次 roll/pitch/yaw、角速度、加速度。
//       手动转动 IMU，数值应跟随变化；静止时角速度≈0，加速度 z≈1g。
//
// 退出: Ctrl+C
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <signal.h>
#include <thread>

#include "serial.h"
#include "wt901.h"

static std::atomic<bool> g_run{true};
static void onSig(int) { g_run = false; }

int main(int argc, char** argv) {
    std::string port = (argc > 1) ? argv[1] : "/dev/ttyUSB0";
    int baud = (argc > 2) ? std::atoi(argv[2]) : 115200;

    signal(SIGINT, onSig);
    signal(SIGTERM, onSig);

    std::printf("[test_imu] 打开 %s @ %d ...\n", port.c_str(), baud);
    Serial ser;
    try {
        ser.open(port, baud);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[err] %s\n", e.what());
        return 1;
    }
    std::printf("[ok] 串口已打开，开始读数据（Ctrl+C 退出）\n");
    std::printf("%-8s %8s %8s %8s | %8s %8s %8s | %7s %7s %7s\n",
                "frame", "roll°", "pitch°", "yaw°",
                "wx°/s", "wy°/s", "wz°/s", "ax_g", "ay_g", "az_g");

    Wt901 wt;
    wt.setAccelRange(16.0);
    wt.setGyroRange(4000.0);
    wt.setAngleRange(180.0);

    uint8_t buf[512];
    int frames = 0;
    auto last_print = std::chrono::steady_clock::now();

    while (g_run.load()) {
        int n = ser.read(buf, sizeof(buf));
        if (n > 0) {
            for (int i = 0; i < n; ++i) wt.feed(buf[i]);
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print).count() >= 100) {
            last_print = now;
            if (wt.ready()) {
                frames++;
                std::printf("%-8d %8.2f %8.2f %8.2f | %8.2f %8.2f %8.2f | %7.3f %7.3f %7.3f\n",
                            frames,
                            wt.rollRad() * 57.2958,
                            wt.pitchRad() * 57.2958,
                            wt.yawRad() * 57.2958,
                            wt.wxRadS() * 57.2958,
                            wt.wyRadS() * 57.2958,
                            wt.wzRadS() * 57.2958,
                            wt.axG(), wt.ayG(), wt.azG());
            } else {
                std::printf("等待数据... (已收字节中)\r");
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    ser.close();
    std::printf("\n[exit] 共收到 %d 帧完整数据\n", frames);
    return 0;
}
