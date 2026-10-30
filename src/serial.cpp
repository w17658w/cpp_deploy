// serial.cpp — 串口封装实现（Windows Win32 / Linux termios 双平台）
#include "serial.h"

#include <stdexcept>
#include <cstring>

#ifdef _WIN32
// ==================== Windows (Win32 API) ====================
#include <windows.h>

Serial::Serial() : fd_(-1), handle_(nullptr) {}
Serial::~Serial() { close(); }

void Serial::open(const std::string& port, int baud) {
    close();
    HANDLE h = CreateFileA(port.c_str(),
                           GENERIC_READ | GENERIC_WRITE,
                           0, NULL, OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE)
        throw std::runtime_error("打开串口失败: " + port + " (错误码 " + std::to_string(GetLastError()) + ")");

    DCB dcb;
    memset(&dcb, 0, sizeof(dcb));
    dcb.DCBlength = sizeof(dcb);
    if (!GetCommState(h, &dcb)) { CloseHandle(h); throw std::runtime_error("GetCommState 失败: " + port); }
    dcb.BaudRate = baud;
    dcb.ByteSize = 8;
    dcb.Parity   = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary  = TRUE;
    if (!SetCommState(h, &dcb)) { CloseHandle(h); throw std::runtime_error("SetCommState 失败: " + port); }

    COMMTIMEOUTS to;
    memset(&to, 0, sizeof(to));
    to.ReadIntervalTimeout         = MAXDWORD;  // 非阻塞：立即返回当前已有字节
    to.ReadTotalTimeoutMultiplier  = 0;
    to.ReadTotalTimeoutConstant    = 0;
    SetCommTimeouts(h, &to);

    handle_ = h;
    fd_ = 0;  // 仅作"已打开"标记
}

void Serial::close() {
    if (handle_) { CloseHandle((HANDLE)handle_); handle_ = nullptr; }
    fd_ = -1;
}

int Serial::read(uint8_t* buf, int maxlen) {
    if (!handle_) return -1;
    DWORD got = 0;
    if (!ReadFile((HANDLE)handle_, buf, (DWORD)maxlen, &got, NULL)) return -1;
    return (int)got;
}

int Serial::write(const uint8_t* buf, int len) {
    if (!handle_) return -1;
    DWORD sent = 0;
    if (!WriteFile((HANDLE)handle_, buf, (DWORD)len, &sent, NULL)) return -1;
    return (int)sent;
}

#else
// ==================== Linux (termios) ====================
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>

Serial::Serial() : fd_(-1) {}
Serial::~Serial() { close(); }

static speed_t baud_to_speed(int baud) {
    switch (baud) {
        case 4800:   return B4800;
        case 9600:   return B9600;
        case 19200:  return B19200;
        case 38400:  return B38400;
        case 57600:  return B57600;
        case 115200: return B115200;
        case 230400: return B230400;
        default: throw std::runtime_error("不支持的波特率: " + std::to_string(baud));
    }
}

void Serial::open(const std::string& port, int baud) {
    close();
    int fd = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0)
        throw std::runtime_error("打开串口失败: " + port + " (" + strerror(errno) + ")");

    struct termios tio;
    memset(&tio, 0, sizeof(tio));
    if (tcgetattr(fd, &tio) != 0) { ::close(fd); throw std::runtime_error("tcgetattr 失败: " + port); }

    speed_t sp = baud_to_speed(baud);
    cfsetispeed(&tio, sp);
    cfsetospeed(&tio, sp);

    // 8N1，无流控，非阻塞读
    tio.c_cflag &= ~PARENB;   // 无校验
    tio.c_cflag &= ~CSTOPB;   // 1 停止位
    tio.c_cflag &= ~CSIZE;
    tio.c_cflag |=  CS8;      // 8 数据位
    tio.c_cflag &= ~CRTSCTS;  // 无硬件流控
    tio.c_cflag |=  CREAD | CLOCAL;

    tio.c_iflag &= ~(IXON | IXOFF | IXANY);  // 无软件流控
    tio.c_iflag &= ~(INLCR | ICRNL | IGNCR); // 不做换行转换（透传二进制）
    tio.c_oflag &= ~OPOST;                    // 原始输出
    tio.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);  // 原始模式

    tio.c_cc[VMIN]  = 0;   // 非阻塞：read 立即返回
    tio.c_cc[VTIME] = 0;

    if (tcsetattr(fd, TCSANOW, &tio) != 0) { ::close(fd); throw std::runtime_error("tcsetattr 失败: " + port); }

    tcflush(fd, TCIOFLUSH);
    fd_ = fd;
}

void Serial::close() {
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

int Serial::read(uint8_t* buf, int maxlen) {
    if (fd_ < 0) return -1;
    int n = (int)::read(fd_, buf, maxlen);
    return n;  // 0=无数据, -1=错误
}

int Serial::write(const uint8_t* buf, int len) {
    if (fd_ < 0) return -1;
    return (int)::write(fd_, buf, len);
}

#endif
