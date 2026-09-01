#include "queued_rknn_worker.hpp"
#include "serial_output.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <map>
#include <mutex>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <thread>
#include <filesystem>

#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/registry.hpp>  // OpenCV 4.x 需要包含这个头文件
#include <chrono>  // 包含 chrono 库

//用于数据交互的库
#include <cstring>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>

using namespace cv;

int source_size[2];

std::map<int, cv::Mat> framesMap;
std::mutex framesMapMutex;

std::vector<std::string> labels;
SerialOutput serialOutput;

struct ProgramOptions {
  std::string input_source;
  std::string input_path;
  std::string model_path;
  std::string serial_port;
  bool show_display = false;
  bool save_video = false;
  bool use_hw_decode = false;
  bool use_hw_encode = false;
  std::string labels_path;
  bool print_results = false;  
  int continuous_frames = 0;  
  std::string save_dir;     
};

void print_usage(const char *program_name) {
  std::cerr << "用法: " << program_name << " [选项]\n"
            << "选项:\n"
            << "  --video <路径>      输入视频文件路径\n"
            << "  --image <路径>      输入图片文件路径\n"
            << "  --camera <路径>     输入摄像头设备路径\n"
            << "  --model <路径>      模型文件路径(必需)\n"
            << "  --imshow           在显示器上显示结果\n"
            << "  --saveVideo        保存视频到output.ts\n"
            << "  --serialPort <端口:波特率> 将结果输出到串口\n"
            << "  --labels <路径>     标签文件路径\n"
            << "  --hw-decode        使用硬件解码器处理输入(可能需要比较新的系统)\n"
            << "  --hw-encode        使用硬件编码器处理输出(可能需要比较新的系统)\n"
            << "  --print           将检测结果输出到终端\n"
            << "  --continuous <N>    当连续N帧检测到物体时保存图片到指定目录\n"
            << "  --save-dir <路径>   保存图片的目录路径\n";
}

ProgramOptions parse_arguments(int argc, char *argv[]) {
  ProgramOptions options;
  static struct option long_options[] = {
      {"video", required_argument, 0, 'v'},
      {"image", required_argument, 0, 'i'},
      {"camera", required_argument, 0, 'c'},
      {"model", required_argument, 0, 'm'},
      {"imshow", no_argument, 0, 'd'},
      {"saveVideo", no_argument, 0, 's'},
      {"serialPort", required_argument, 0, 'p'},
      {"hw-decode", no_argument, 0, 'D'},
      {"hw-encode", no_argument, 0, 'E'},
      {"labels", required_argument, 0, 'l'},
      {"print", no_argument, 0, 't'},  // 新增选项
      {"continuous", required_argument, 0, 'n'},
      {"save-dir", required_argument, 0, 'o'},
      {0, 0, 0, 0}};

  int input_source_count = 0;
  int opt;
  while ((opt = getopt_long(argc, argv, "v:i:c:m:dsp:l:DEtno:", long_options,
                            nullptr)) != -1) {
    switch (opt) {
    case 'v':
      options.input_source = "video";
      options.input_path = optarg;
      input_source_count++;
      break;
    case 'i':
      options.input_source = "image";
      options.input_path = optarg;
      input_source_count++;
      break;
    case 'c':
      options.input_source = "camera";
      options.input_path = optarg;
      input_source_count++;
      break;
    case 'm':
      options.model_path = optarg;
      break;
    case 'd':
      options.show_display = true;
      break;
    case 's':
      options.save_video = true;
      break;
    case 'p':
      options.serial_port = optarg;
      break;
    case 'D':
      options.use_hw_decode = true;
      break;
    case 'E':
      options.use_hw_encode = true;
      break;
    case 'l':
      options.labels_path = optarg;
      break;
    case 't':
      options.print_results = true;
      break;
    case 'n':
      options.continuous_frames = std::stoi(optarg);
      break;
    case 'o':
      options.save_dir = optarg;
      break;
    default:
      print_usage(argv[0]);
      exit(1);
    }
  }

  if (input_source_count != 1) {
    std::cerr << "Error: Must specify exactly one input source (--video, "
                 "--image, or --camera)\n";
    print_usage(argv[0]);
    exit(1);
  }

  if (options.model_path.empty()) {
    std::cerr << "Error: Model path (--model) is required\n";
    print_usage(argv[0]);
    exit(1);
  }

  return options;
}

static auto getLabelColor(const std::string &label) -> cv::Scalar {
  std::mt19937 gen(std::hash<std::string>{}(label));
  std::uniform_int_distribution<int> dis(0, 255);
  return cv::Scalar(dis(gen), dis(gen), dis(gen));
}

void yoloDrawResults(cv::Mat &img, const object_detect_result_list &results, 
                    float xscale, float yscale) {
  for (int i = 0; i < results.count; i++) {
    auto result = results.results[i];
    cv::Scalar color = getLabelColor(labels.at(result.cls_id));
    cv::Scalar txt_color(255, 255, 255);
    int lw = 2;
    double sf = 1.0;
    int tf = 2;

    // 缩放检测框坐标
    cv::Point p1(result.box.left * xscale, result.box.top * yscale);
    cv::Point p2(result.box.right * xscale, result.box.bottom * yscale);
    cv::rectangle(img, p1, p2, color, lw, cv::LINE_AA);

    int baseLine;
    cv::Size labelSize = cv::getTextSize(
        labels.at(result.cls_id), cv::FONT_HERSHEY_SIMPLEX, sf, tf, &baseLine);
    int w = labelSize.width;
    int h = labelSize.height;

    bool outside = p1.y - h >= 3;  // 使用缩放后的 y 坐标
    cv::Point p2_text;
    if (outside) {
      p2_text = cv::Point(p1.x + w, p1.y - h - 3);
    } else {
      p2_text = cv::Point(p1.x + w, p1.y + h + 3);
    }

    cv::rectangle(img, p1, p2_text, color, -1, cv::LINE_AA);

    cv::Point textOrigin;
    if (outside) {
      textOrigin = cv::Point(p1.x, p1.y - 2);
    } else {
      textOrigin = cv::Point(p1.x, p1.y + h + 2);
    }

    cv::putText(img, labels.at(result.cls_id), textOrigin,
                cv::FONT_HERSHEY_SIMPLEX, sf, txt_color, tf, cv::LINE_AA);
  }
}

bool loadLabels(const std::string &filepath) {
  if (filepath.empty()) {
    return false;
  }

  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open labels file: " << filepath << std::endl;
    return false;
  }

  labels.clear();
  std::string line;
  while (std::getline(file, line)) {
    // 去除可能的回车符和空
    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
    line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
    if (!line.empty()) {
      labels.push_back(line);
    }
  }

  return !labels.empty();
}



int main(int argc, char *argv[]) {


    std::cout << "OpenCV version: " << cv::getVersionString() << std::endl;
    
    key_t key = ftok("shmfile", 65); // 生成唯一的key
    
    // 检查共享内存是否存在，并获取它的ID
    int shmid = shmget(key, 1024, 0666);
    if (shmid != -1) {
        // 如果共享内存存在，删除它
        if (shmctl(shmid, IPC_RMID, NULL) == -1) {
            perror("shmctl");
            exit(1);
        }
        printf("Deleted existing shared memory segment.\n");
    }

    // 现在创建新的共享内存
    shmid = shmget(key, 102400, 0666 | IPC_CREAT);
    if (shmid == -1) {
        perror("shmget");
        exit(1);
    }

    int *str = static_cast<int*>(shmat(shmid, NULL, 0)); // 将共享内存附加到进程地址空间

    if (str == reinterpret_cast<int*>(-1)) {
        perror("shmat");
        exit(1);
    }
    
    
    std::cout << "Producer is writing to shared memory..." << std::endl;
    
    //strcpy(str, "Hello, Consumer!"); // 写入数据到共享内存

    //std::cout << "Data written to shared memory." << std::endl;

    //sleep(10); // 保持程序运行，以便消费者有时间读取数据

    









  auto options = parse_arguments(argc, argv);

  auto modelPath = options.model_path;
  auto inputVideo = options.input_path;
  auto usehwcodec = options.use_hw_decode;

  setenv("GST_VIDEO_CONVERT_USE_RGA", "1", 1);

  // 加载标签文件
  if (!options.labels_path.empty() && !loadLabels(options.labels_path)) {
    std::cerr << "Failed to load labels file" << std::endl;
    return -1;
  }

  if (!options.serial_port.empty()) {
    auto path = options.serial_port.substr(0, options.serial_port.find(':'));
    auto baudRate = std::stoi(options.serial_port.substr(options.serial_port.find(':') + 1));
    auto res = serialOutput.open(path, baudRate);
    if (!res) {
      std::cerr << "Failed to open serial port: " << serialOutput.getLastError() << std::endl;
      return -1;
    }
  }

  // 打开输入
  cv::VideoCapture *cap = nullptr;
  if (options.input_source == "video") {
    if (options.use_hw_decode) {
      cap = new cv::VideoCapture(
          "filesrc location=" + std::string(inputVideo) +
              " ! qtdemux !  video/x-h264 ! h264parse ! video/x-h264 ! "
              "mppvideodec arm-afbc=0 ! video/x-raw, format=NV12 ! "
              "videoconvert "
              "! video/x-raw, format=BGR ! appsink sync=false",
          cv::CAP_GSTREAMER);
    } else {
      cap = new cv::VideoCapture(inputVideo);
    }
  } else if (options.input_source == "camera") {
    cap = new cv::VideoCapture(options.input_path);
  } else {
    cap = new cv::VideoCapture(options.input_path);
  }
  if (!cap->isOpened()) {
    std::cerr << "Error opening video stream or file" << std::endl;
    return -1;
  }

  // 获取当前视频后端的名称
    // 获取当前视频后端的名称
    std::cout << "Video backend: " << (*cap).getBackendName() << std::endl;

  source_size[0] = cap->get(cv::CAP_PROP_FRAME_WIDTH);
  source_size[1] = cap->get(cv::CAP_PROP_FRAME_HEIGHT);

  auto worker = new queued_rknn_worker<true>(modelPath.c_str(),
                                             RKNN_NPU_CORE_0_1_2, 2, 0);

  std::thread resultThread([&worker, &options, &str]() {
    int shareMemFlag = 0; //用于标志共享内存写了多少位
    auto count = 0;
    auto start = std::chrono::system_clock::now();
    cv::VideoWriter *video = nullptr;
    if (options.save_video) {
      if (options.use_hw_encode) {
        video = new cv::VideoWriter(
            "appsrc ! videoconvert ! video/x-raw, format=NV12 ! mpph264enc ! "
            " h264parse ! mpegtsmux ! filesink location=output.ts",
            cv::CAP_GSTREAMER, 0, 30, cv::Size(source_size[0], source_size[1]));
      } else {
        video = new cv::VideoWriter(
            "output.ts", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 30,
            cv::Size(source_size[0], source_size[1]));
      }
    }
    int continuous_detect_count = 0;  // 连续检测计数器
    int save_count = 0;              // 保存图片计数器
    
    while (true) {
      auto [taskId, data] = worker->waitForResult();
      if (taskId == -1) {
        break;
      }else{
        printf("id is %d\n",taskId);
      }
      // std::cout << "Got result " << data->count << std::endl;
      framesMapMutex.lock();
      auto frame = framesMap[taskId];
      framesMap.erase(taskId);
      framesMapMutex.unlock();
      // 缩放结果
      float xscale = (float)source_size[0] / worker->inputShape[1];
      float yscale = (float)source_size[1] / worker->inputShape[2];
      yoloDrawResults(frame, *data, xscale, yscale);



      //这里把检测结果写入共享内存
      
      auto result_num = data->count;
      
      str[shareMemFlag] = result_num;
      shareMemFlag++;

      int i = 0;
      do {
          // 循环体至少执行一次
          auto result = data->results[i];
      
          cv::Point p1(result.box.left * xscale, result.box.top * yscale);
          cv::Point p2(result.box.right * xscale, result.box.bottom * yscale);
          //printf("-----------%f %f\n",xscale,yscale);
          str[shareMemFlag + 0] = 2;
          str[shareMemFlag + 1] = 3;
          str[shareMemFlag + 2] = p1.x;
          str[shareMemFlag + 3] = p1.y;
          str[shareMemFlag + 4] = p2.x;
          str[shareMemFlag + 5] = p2.y;

          shareMemFlag = shareMemFlag + 6;
          i++;
      } while (i < result_num);

     



      // 处理连续检测逻辑
      if (options.continuous_frames > 0 && !options.save_dir.empty()) {
          if (data->count > 0) {
              continuous_detect_count++;
              if (continuous_detect_count >= options.continuous_frames) {
                  // 创建保存目录（如果不存在）
                  std::filesystem::create_directories(options.save_dir);
                  
                  // 生成文件名（使用时间戳和计数器）
                  auto now = std::chrono::system_clock::now();
                  auto timestamp = std::chrono::system_clock::to_time_t(now);
                  std::stringstream ss;
                  ss << options.save_dir << "/detect_" 
                     << std::put_time(std::localtime(&timestamp), "%Y%m%d_%H%M%S")
                     << "_" << save_count++ << ".jpg";
                  
                  // 保存图片
                  cv::imwrite(ss.str(), frame);
                  std::cout << "Saved detection image: " << ss.str() << std::endl;
                  
                  // 重置计数器
                  continuous_detect_count = 0;
              }
          } else {
              // 如果当前帧没有检测到物体，重置计数器
              continuous_detect_count = 0;
          }
      }

      if (options.input_source == "image") {
        cv::imwrite("output.jpg", frame);
        exit(0);
      }

      if (video) {
        video->write(frame);
      }
      if (options.show_display) {
        cv::imshow("YOLO", frame);
        cv::waitKey(1);
      }

      if (!options.serial_port.empty()) {
        for (int i = 0; i < data->count; i++) {
          auto result = data->results[i];
          serialOutput.write(std::to_string(result.cls_id) + "," + std::to_string(result.prop) + "," + std::to_string(result.box.left) + "," + std::to_string(result.box.top) + "," + std::to_string(result.box.right) + "," + std::to_string(result.box.bottom) + "\n");
        }
        serialOutput.write("\n");
      }

      if (options.print_results) {
        std::cout << "Frame " << count << " detected " << data->count
                  << " objects:" << std::endl;
        for (int i = 0; i < data->count; i++) {
          auto result = data->results[i];
          std::cout << "  " << labels.at(result.cls_id)
                    << ": conf=" << result.prop << ", box=(" << result.box.left
                    << "," << result.box.top << "," << result.box.right << ","
                    << result.box.bottom << ")" << std::endl;
        }
        std::cout << std::endl;
      }

      frame.release();

      // print fps
      count++;
      if (count % 100 == 0) {
        auto end = std::chrono::system_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "fps: " << 1000.0 * count / duration.count() << std::endl;
        start = end;
        count = 0;
      }

      worker->free(taskId);
    }
  });


  while (true) {
    cv::Mat frame;

auto start = std::chrono::high_resolution_clock::now();

    cap->read(frame);

    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    // 计算执行时间
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
// 打印执行时间
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    if (frame.empty()) {
      break;
    }
    auto frame_ = frame.clone();

    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    // resize to 640x640, if need faster, this should be in a separate thread +
    // hw scale

    //printf("---------frame size is %d %d",frame.cols,frame.rows);



    cv::Mat resized;
    cv::resize(frame, resized,
               cv::Size(worker->inputShape[1], worker->inputShape[2]), 0, 0,
               cv::INTER_NEAREST);
    auto id = worker->submit(resized.data);
    framesMapMutex.lock();
    framesMap[id] = frame_;
    framesMapMutex.unlock();
    while (worker->getInputQueueSize() > 10 ||
           worker->getResultQueueSize() > 10) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(10000));

  delete worker;
  resultThread.join();

  shmdt(str); // 分离共享内存
  shmctl(shmid, IPC_RMID, NULL); // 删除共享内存段

  return 0;
}
