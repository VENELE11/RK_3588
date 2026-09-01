#pragma once
#include <string>

class SerialOutput {
public:
    SerialOutput();
    ~SerialOutput();

    // 打开串口
    bool open(const std::string& device, int baudRate = 9600);
    
    // 关闭串口
    void close();
    
    // 发送字符串
    bool write(const std::string& data);
    
    // 获取最后的错误信息
    std::string getLastError() const;

private:
    // 配置串口参数
    bool configure(int baudRate);
    
    int fd_;                     // 文件描述符
    std::string lastError_;      // 最后的错误信息
    bool isOpen_;               // 串口是否打开
};