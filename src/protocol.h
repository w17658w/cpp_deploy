// protocol.h — 上位机 -> Arduino 下行帧协议
//
// 下行（Zero3W -> Arduino，每控制节拍一帧，26 字节）:
//   [0] 0xAA | [1..24] 12×uint16 脉宽(μs, 低字节在前) | [25] 校验(累加低8位)
//
// 触地改由香橙派 GPIO 直读，腿长无真实反馈（-R），故不再有上行帧。
#pragma once

#include <cstdint>

constexpr uint8_t  DOWN_HEAD = 0xAA;
constexpr int      DOWN_LEN  = 26;

// 累加校验（取低 8 位）
uint8_t checksum(const uint8_t* data, int len);

// 打包下行帧（12 个脉宽 -> 26 字节）
void buildDownFrame(const uint16_t pulse_us[12], uint8_t out[DOWN_LEN]);
