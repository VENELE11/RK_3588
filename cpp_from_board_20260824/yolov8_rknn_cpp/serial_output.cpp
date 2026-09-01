#include "serial_output.h"
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>

SerialOutput::SerialOutput() : fd_(-1), isOpen_(false) {}

SerialOutput::~SerialOutput() {
    close();
}

bool SerialOutput::open(const std::string& device, int baudRate) {
    // 打开串口设备
    fd_ = ::open(device.c_str(), O_WRONLY | O_NOCTTY | O_NDELAY);
    if (fd_ == -1) {
        lastError_ = "Failed to open device: " + std::string(strerror(errno));
        return false;
    }

    // 配置串口参数
    if (!configure(baudRate)) {
        close();
        return false;
    }

    isOpen_ = true;
    return true;
}

void SerialOutput::close() {
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
    isOpen_ = false;
}

bool SerialOutput::write(const std::string& data) {
    if (!isOpen_) {
        lastError_ = "Serial port is not open";
        return false;
    }

    ssize_t written = ::write(fd_, data.c_str(), data.length());
    if (written == -1) {
        lastError_ = "Write failed: " + std::string(strerror(errno));
        return false;
    }

    return true;
}

bool SerialOutput::configure(int baudRate) {
    struct termios options;
    
    // 获取当前配置
    if (tcgetattr(fd_, &options) == -1) {
        lastError_ = "Failed to get port attributes: " + std::string(strerror(errno));
        return false;
    }

    // 设置波特率
    speed_t baud;
    switch (baudRate) {
        case 9600:   baud = B9600;   break;
        case 19200:  baud = B19200;  break;
        case 38400:  baud = B38400;  break;
        case 57600:  baud = B57600;  break;
        case 115200: baud = B115200; break;
        default:
            lastError_ = "Unsupported baud rate";
            return false;
    }

    cfsetospeed(&options, baud);

    // 设置其他参数
    options.c_cflag |= (CLOCAL | CREAD);    // 启用接收器并忽略调制解调器控制线
    options.c_cflag &= ~PARENB;             // 无奇偶校验
    options.c_cflag &= ~CSTOPB;             // 1个停止位
    options.c_cflag &= ~CSIZE;              // 清除数据位掩码
    options.c_cflag |= CS8;                 // 8位数据
    
    // 应用新配置
    if (tcsetattr(fd_, TCSANOW, &options) == -1) {
        lastError_ = "Failed to set port attributes: " + std::string(strerror(errno));
        return false;
    }

    return true;
}

std::string SerialOutput::getLastError() const {
    return lastError_;
}