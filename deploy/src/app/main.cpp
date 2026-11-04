// main.cpp — 双线程推理主程序（Zero 3W）
//
// 线程 A（采集）: 读 IMU（含加速度）+ 读触地 GPIO，刷新共享 latest_state
// 线程 B（主线程/决策）: 按 control_hz 快照 -> 组装 36 维 -> 前向 -> 12 动作 -> 下行帧(仅脉宽)
//
// 运行:  ./cpp_deploy [config.yaml]   （默认当前目录 config.yaml）
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>

#include "config.h"
#include "serial.h"
#include "wt901.h"
#include "protocol.h"
#include "state.h"
#include "mlp.h"
#include "actuator.h"
#include "gpio.h"
#include "axis_calib.h"

// ---------- 全局运行标志与共享状态 ----------
static std::atomic<bool> g_running{true};

struct SharedState {
    std::mutex mtx;
    RobotState st;
    bool imu_ok = false;
};

// ---------- 线程 A：采集 ----------
static void acquireLoop(Serial* imu, Wt901* wt, ContactGpio* gpio, MountCalib* mount,
                        SharedState* sh, double imu_hz, std::atomic<bool>* run) {
    auto period = std::chrono::microseconds((long long)(1e6 / imu_hz));
    while (run->load()) {
        auto next = std::chrono::steady_clock::now() + period;

        uint8_t b[512];
        int n = imu->read(b, sizeof(b));
        if (n > 0) {
            for (int i = 0; i < n; ++i) wt->feed(b[i]);
            std::lock_guard<std::mutex> g(sh->mtx);
            // 安装补偿：姿态用「原始角(绝对姿态)」做完整旋转；角速度/比力用去零偏后的量
            mount->applyOrientation(wt->rollRawRad(), wt->pitchRawRad(), wt->yawRawRad(),
                                    sh->st.roll, sh->st.pitch, sh->st.yaw);
            double w[3] = {wt->wxRadS(), wt->wyRadS(), wt->wzRadS()};
            double f[3] = {wt->axG(), wt->ayG(), wt->azG()};
            double w_r[3], f_r[3];
            mount->applyVec(w, w_r);
            mount->applyVec(f, f_r);
            sh->st.wx    = w_r[0];
            sh->st.wy    = w_r[1];
            sh->st.wz    = w_r[2];
            sh->st.ax_g  = f_r[0];
            sh->st.ay_g  = f_r[1];
            sh->st.az_g  = f_r[2];
            sh->imu_ok   = wt->ready();
        }

        // 触地直接读 GPIO（不再走 Arduino 上行）
        if (gpio && gpio->ok()) {
            std::lock_guard<std::mutex> g(sh->mtx);
            sh->st.contact = gpio->read();
        }

        std::this_thread::sleep_until(next);
    }
}

// ---------- 主程序 ----------
int main(int argc, char** argv) {
    std::string cfg_path = (argc > 1) ? argv[1] : "config.yaml";

    Config cfg;
    try {
        cfg.load(cfg_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[err] %s\n", e.what());
        return 1;
    }

    std::printf("[init] 打开 IMU 串口 %s @ %d\n",
                cfg.getString("imu_port").c_str(), cfg.getInt("imu_baud"));
    Serial imu, ard;
    try {
        imu.open(cfg.getString("imu_port"), cfg.getInt("imu_baud"));
        std::printf("[init] 打开 Arduino 串口 %s @ %d\n",
                    cfg.getString("arduino_port").c_str(), cfg.getInt("arduino_baud"));
        ard.open(cfg.getString("arduino_port"), cfg.getInt("arduino_baud"));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[err] %s\n", e.what());
        return 1;
    }

    // 网络
    int obs = cfg.getInt("obs_dim"), hid = cfg.getInt("hidden_size"), act = cfg.getInt("action_dim");
    Mlp net;
    net.init(obs, hid, act);
    if (!net.load(cfg.getString("weights_path"))) {
        std::fprintf(stderr, "[err] 权重加载失败: %s\n", cfg.getString("weights_path").c_str());
        return 1;
    }
    std::printf("[init] 权重加载成功（%d->%d->%d->%d）\n", obs, hid, hid, act * 2);

    // 状态组装
    StateBuilder sb;
    sb.init(cfg.getDouble("leg_length_mid"), cfg.getDouble("leg_length_span"));

    // 舵机映射
    ActuatorMap act_map;
    act_map.init(cfg.getDouble("actuator_stroke_m"),
                 cfg.getDouble("pulse_min_us"), cfg.getDouble("pulse_max_us"));

    // 触地 GPIO（香橙派直读）
    std::vector<int> gpio_lines = cfg.getIntList("contact_gpio_lines");
    ContactGpio gpio;
    gpio.init(cfg.getString("contact_gpio_chip"), gpio_lines,
              cfg.getBool("contact_active_high"));
    std::printf("[init] 触地 GPIO %s（%zu 路）: %s\n",
                cfg.getString("contact_gpio_chip").c_str(), gpio_lines.size(),
                gpio.ok() ? "OK" : "不可用（检查引脚/权限/是否装 libgpiod）");

    // IMU（量程注入）
    Wt901 wt;
    wt.setAccelRange(cfg.getDouble("accel_range_g"));
    wt.setGyroRange(cfg.getDouble("gyro_range_dps"));
    wt.setAngleRange(cfg.getDouble("angle_range_deg"));

    // ---- 零偏校准：静止采 N 帧 ----
    std::printf("[calib] 请保持机器人静止，采集零偏...\n");
    wt.beginCalib(cfg.getInt("zero_bias_samples"));
    {
        uint8_t b[512];
        auto t0 = std::chrono::steady_clock::now();
        while (!wt.calibDone() && g_running.load()) {
            int n = imu.read(b, sizeof(b));
            for (int i = 0; i < n; ++i) wt.feed(b[i]);
            if (wt.ready()) wt.feedCalibSample();
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) {
                std::fprintf(stderr, "[warn] 校准超时（10s），用已采样本继续\n");
                break;
            }
        }
        wt.applyCalib();
        std::printf("[calib] 零偏校准完成（陀螺仪偏置 %.4f %.4f %.4f rad/s）\n",
                    wt.wxRawRadS() - wt.wxRadS(), wt.wyRawRadS() - wt.wyRadS(),
                    wt.wzRawRadS() - wt.wzRadS());
    }

    // ---- 安装角补偿：机器人摆到零姿态后，用「原始欧拉角」做安装基准 R_mount ----
    MountCalib mount;
    if (cfg.getBool("enable_axis_calib")) {
        mount.calibrate(wt.rollRawRad(), wt.pitchRawRad(), wt.yawRawRad());
        std::printf("[calib] 安装角补偿已启用（原始角 roll=%.1f° pitch=%.1f° yaw=%.1f°）\n",
                    wt.rollRawRad() * 57.2958, wt.pitchRawRad() * 57.2958, wt.yawRawRad() * 57.2958);
    } else {
        std::printf("[calib] 安装角补偿未启用\n");
    }

    // ---- 初始动作 / 上一命令 ----
    SharedState sh;
    double init_len = cfg.getDouble("init_leg_length_m");
    for (int i = 0; i < 12; ++i) sh.st.last_cmd_m[i] = init_len;

    double ctrl_hz = cfg.getDouble("control_hz");
    double imu_hz = cfg.getDouble("imu_read_hz");

    // ---- 启动采集线程 A ----
    std::thread t(acquireLoop, &imu, &wt, &gpio, &mount, &sh, imu_hz, &g_running);

    std::printf("[run] 启动决策循环 @ %.1f Hz（Ctrl+C 退出）\n", ctrl_hz);
    auto ctrl_period = std::chrono::microseconds((long long)(1e6 / ctrl_hz));

    // ---- 决策循环（线程 B）----
    float obs[36];
    float action[12];
    uint16_t pulse[12];
    uint8_t down[DOWN_LEN];
    uint64_t tick = 0;

    while (g_running.load()) {
        auto next = std::chrono::steady_clock::now() + ctrl_period;

        // 快照最新状态
        RobotState snap;
        {
            std::lock_guard<std::mutex> g(sh.mtx);
            snap = sh.st;
        }

        // 组装 36 维 -> 前向 -> 12 动作
        sb.build(snap, obs);
        net.forward(obs, action);

        // 动作(m) -> 脉宽(μs) -> 下行帧 -> 写 Arduino
        for (int i = 0; i < 12; ++i) {
            double a = action[i];
            a = std::max(cfg.getDouble("action_low"), std::min(cfg.getDouble("action_high"), a));
            pulse[i] = act_map.lengthToPulse(a);
            snap.last_cmd_m[i] = a;   // 更新上一命令（供下帧状态）
        }
        buildDownFrame(pulse, down);
        ard.write(down, DOWN_LEN);

        // 回写 last_cmd
        {
            std::lock_guard<std::mutex> g(sh.mtx);
            for (int i = 0; i < 12; ++i) sh.st.last_cmd_m[i] = snap.last_cmd_m[i];
        }

        if ((++tick) % 10 == 1) {
            std::printf("[run] tick=%llu  leg0=%.3fm  roll=%.2f pitch=%.2f yaw=%.2f  acc=(%.2f,%.2f,%.2f)g  contact=0x%03x\n",
                        (unsigned long long)tick, action[0], snap.roll, snap.pitch, snap.yaw,
                        snap.ax_g, snap.ay_g, snap.az_g, snap.contact);
        }

        std::this_thread::sleep_until(next);
    }

    g_running.store(false);
    if (t.joinable()) t.join();
    gpio.close();
    imu.close();
    ard.close();
    std::printf("[exit] 正常退出\n");
    return 0;
}
