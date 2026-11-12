# deploy/test/ — 硬件通信测试程序

三个独立测试程序，分别验证香橙派与 IMU、Arduino/舵机、触地 GPIO 的通信。
**先单独测通每个硬件，再跑主程序。**

## 编译

在香橙派上（项目根目录）：

```bash
cd deploy
cmake -B build_test -DCMAKE_BUILD_TYPE=Release
cmake --build build_test -j
```

产物在 `build_test/test_*/`。

## 测试顺序

### 1. test_imu — WT901 IMU

```bash
cd build_test/test_imu
sudo ./test_imu /dev/ttyUSB0 115200
```

**预期现象：**
- 每 100ms 打印一行 roll/pitch/yaw、角速度、加速度
- 静止平放：roll≈0, pitch≈0, yaw 不动；角速度≈0；az≈1.0g
- 手动转 IMU：数值跟随变化，无跳变

**不通时排查：**
- `ls /dev/ttyUSB*` 看设备在不在
- `dmesg | tail` 看 USB 转串口有没有识别
- 波特率对不对（WT901 默认 115200）
- 权限：`sudo usermod -aG dialout $USER` 后重登

### 2. test_actuator — Arduino + 12 路舵机

**⚠️ 先把机器人架空！舵机会动！**

```bash
cd build_test/test_actuator
sudo ./test_actuator /dev/ttyACM0 115200
```

**流程：**
1. 全部舵机回中 1500μs（50mm）
2. 等回车开始
3. 依次单独扫每一路 leg_01 ~ leg_12：1500→1000→1500→2000→1500
4. 全部回中结束

**预期现象：**
- 每路舵机单独伸缩，能看出对应哪条腿
- 方向正确：1000μs=全缩，2000μs=全伸
- 如果某路不动：查 Arduino 接线、舵机电源

**不通时排查：**
- `ls /dev/ttyACM*` 看 Arduino 在不在
- Arduino IDE 里串口监视器能收到数据吗（115200）
- 舵机电源够不够（12 路舵机不能只靠 USB 供电）

### 3. test_gpio — 12 路触地 GPIO

```bash
cd build_test/test_gpio
sudo ./test_gpio /dev/gpiochip0 41,139,96,357,356,140,141,36,97,99,38,98
```

**预期现象：**
- 每 200ms 打印 12 路电平
- 用手按压力传感器（或短接 DO 到 GND/3.3V），对应 bit 翻转

**不通时排查：**
- `gpioinfo` 看 chip 和 line 号对不对
- `sudo apt install libgpiod-dev`
- 引脚有没有被 SPI/I2C/UART 占用
- 电平：传感器 DO 是 3.3V 还是 5V（5V 必须电平转换）

## 三个都测通后

跑主程序：
```bash
cd build
./robot12_deploy ../config/config.yaml
```
