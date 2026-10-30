// protocol.cpp — 下行帧打包实现
#include "protocol.h"

uint8_t checksum(const uint8_t* data, int len) {
    uint8_t s = 0;
    for (int i = 0; i < len; ++i) s = (uint8_t)(s + data[i]);
    return s;
}

void buildDownFrame(const uint16_t pulse_us[12], uint8_t out[DOWN_LEN]) {
    out[0] = DOWN_HEAD;
    for (int i = 0; i < 12; ++i) {
        out[1 + 2 * i]     = (uint8_t)(pulse_us[i] & 0xFF);
        out[1 + 2 * i + 1] = (uint8_t)(pulse_us[i] >> 8);
    }
    out[DOWN_LEN - 1] = checksum(out, DOWN_LEN - 1);
}
