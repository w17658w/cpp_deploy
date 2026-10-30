// mega2560_actuator.ino — 12 路 L16-R 舵机驱动（纯 PWM 中继）
//
// 下行帧(上位机->本机, 26字节): 0xAA | 12×uint16 脉宽(μs,低字节在前) | 校验(累加低8位)
//
// 触地改由香橙派 GPIO 直读、腿长无真实反馈（-R），故本机不再采集/上行。
// 只负责：收下行 12 脉宽 -> 立即 writeMicroseconds 给 12 路舵机（舵机内部闭环自行逼近）。
//
// 串口: USB Serial, 115200 8N1
#include <Servo.h>

// ================= 引脚定义（按实际接线修改）=================
const int SERVO_PINS[12] = { 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13};  // 12 路舵机 PWM

// ================= 常量 =================
const uint8_t DOWN_HEAD = 0xAA;
const int   DOWN_LEN = 26;

Servo servos[12];

// 下行帧累积缓冲
uint8_t down_buf[DOWN_LEN];
int     down_idx = 0;

// ---------- 校验 ----------
uint8_t sum8(const uint8_t* d, int n) {
  uint8_t s = 0;
  for (int i = 0; i < n; ++i) s += d[i];
  return s;
}

// ---------- 下行帧逐字节解析 ----------
void onDownByte(uint8_t b) {
  if (down_idx == 0) {
    if (b == DOWN_HEAD) down_buf[down_idx++] = b;
  } else {
    down_buf[down_idx++] = b;
    if (down_idx >= DOWN_LEN) {
      down_idx = 0;
      if (sum8(down_buf, DOWN_LEN - 1) == down_buf[DOWN_LEN - 1]) {
        for (int i = 0; i < 12; ++i) {
          uint16_t us = (uint16_t)(down_buf[1 + 2*i] | (down_buf[1 + 2*i + 1] << 8));
          servos[i].writeMicroseconds(us);
        }
      }
    }
  }
}

// ================= 主流程 =================
void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 12; ++i) {
    servos[i].attach(SERVO_PINS[i]);
    servos[i].writeMicroseconds(1500);     // 上电中位（50mm）
  }
}

void loop() {
  while (Serial.available()) onDownByte((uint8_t)Serial.read());
}
