// serial.h — 串口封装（平台抽象：Windows Win32 / Linux termios）
//
// 用法:
//   Serial s;
//   s.open("/dev/ttyUSB0", 115200);   // Linux
//   s.open("COM9", 115200);           // Windows
//   s.read(buf, n);  s.write(buf, n); s.close();
#pragma once

#include <cstdint>
#include <string>

class Serial {
public:
    Serial();
    ~Serial();

    // 打开串口；失败抛 std::runtime_error
    void open(const std::string& port, int baud);
    void close();
    bool isOpen() const { return fd_ >= 0; }

    // 读取：非阻塞，返回实际读到的字节数（0 表示当前无数据，<0 表示错误）
    int read(uint8_t* buf, int maxlen);
    // 写入：返回实际写入字节数（<0 表示错误）
    int write(const uint8_t* buf, int len);

private:
    int fd_;   // Windows: HANDLE 转 int 存储；Linux: 文件描述符

#ifdef _WIN32
    void* handle_;  // 实际 HANDLE
#else
    // Linux 无需额外成员
#endif
};
