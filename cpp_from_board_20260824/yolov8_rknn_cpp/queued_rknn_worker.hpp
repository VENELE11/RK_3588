#pragma once

#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <type_traits>
#include <vector>

#include "postprocess.h"
#include "rknn_api.h"
#include "rknn_dtype.hpp"

template <bool sorted_output = false> class queued_rknn_worker {
  template <typename T> using queue = std::queue<T>;
  template <typename T> using vector = std::vector<T>;

public:
  using taskId = size_t;
  int inputShape[4];
  int inputCount = 1;
  using inputDtype = uint8_t;
  using outputDtype = object_detect_result_list;

  struct TaskResources {
    const inputDtype *input1;
    rknn_output *rawOutputs;
    const outputDtype *output;
  };

  struct ThreadResources {
    std::thread *workerThread;
    std::condition_variable workerCond;
    rknn_context modelContext;
  };

private:
  unsigned char *model_data;
  rknn_sdk_version version;
  rknn_input_output_num io_num;
  rknn_tensor_attr *input_attrs;
  rknn_tensor_attr *output_attrs;

  uint32_t rknnFlag = 0;

  int threadsPerCore = 1;

  std::vector<ThreadResources *> threadResources;
  std::queue<taskId> inputQueue;
  std::mutex inputQueueMutex;
  using queue_type =
      typename std::conditional<sorted_output, std::priority_queue<taskId, std::vector<taskId>, std::greater<taskId>>,
                                std::queue<taskId>>::type;
  queue_type resultQueue;
  std::mutex resultQueueMutex;
  std::condition_variable resultCond;

  std::atomic<bool> requestStop{false};
  std::map<taskId, TaskResources> taskResourcesMap;
  std::mutex taskResourceAllocMutex;

  taskId lastId = 0;
  taskId lastOutputId = 0;

  static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz) {
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp) {
      return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0) {
      printf("blob seek failure.\n");
      return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL) {
      printf("buffer malloc failure.\n");
      return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
  }

  static unsigned char *load_model(const char *filename, int *model_size) {
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
      printf("Open file %s failed.\n", filename);
      return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
  }
  
  auto postprocess(const rknn_output *outputs) -> object_detect_result_list * {
    auto letter_box = letterbox_t{0, 0, 1.0};
    auto od_results = new object_detect_result_list();
    const int8_t *outputs_[] = {
        (int8_t *)outputs[0].buf,
        (int8_t *)outputs[1].buf,
        (int8_t *)outputs[2].buf,
        (int8_t *)outputs[3].buf,
        (int8_t *)outputs[4].buf,
        (int8_t *)outputs[5].buf,
    };

    int32_t zps[] = {
        output_attrs[0].zp,
        output_attrs[1].zp,
        output_attrs[2].zp,
        output_attrs[3].zp,
        output_attrs[4].zp,
        output_attrs[5].zp,
    };
    
    float scales[] = {
        output_attrs[0].scale,
        output_attrs[1].scale,
        output_attrs[2].scale,
        output_attrs[3].scale,
        output_attrs[4].scale,
        output_attrs[5].scale,
    };
    
    int32_t grid_hs[] = {
        output_attrs[0].dims[2],
        output_attrs[2].dims[2],
        output_attrs[4].dims[2],
    };
    
    int32_t grid_ws[] = {
        output_attrs[0].dims[3],
        output_attrs[2].dims[3],
        output_attrs[4].dims[3],
    };
    
    post_process(outputs_, &letter_box, BOX_THRESH, NMS_THRESH, od_results,
                 inputShape[1], inputShape[2], output_attrs[0].dims[1] / 4,
                 zps, scales, grid_hs, grid_ws,
                 output_attrs[1].dims[1]);

    return od_results;
  }
  void workerThreadFunc(const int threadId) {
    static int inferenceCount = 0;
    auto &threadResource = *threadResources[threadId];
    while (true) {
      taskId id = 0;
      {
        std::unique_lock<std::mutex> lock(inputQueueMutex);
        threadResource.workerCond.wait(
            lock, [&] { return !inputQueue.empty() || requestStop; });
        if (requestStop) {
          break;
        }
        id = inputQueue.front();
        inputQueue.pop();
      }
      TaskResources *r_;
      {
        std::lock_guard<std::mutex> lock(taskResourceAllocMutex);
        r_ = &taskResourcesMap.at(id);
      }
      auto &item = *r_;

      // 设置rknn的输入数据
      vector<rknn_input> inputs;
      for (int i = 0; i < io_num.n_input; i++) {
        rknn_input input;
        input.index = i;
        input.type = RknnDtype<inputDtype>::value;
        input.size = sizeof(inputDtype) * inputShape[0] * inputShape[1] *
                     inputShape[2] * inputShape[3];
        input.fmt = RKNN_TENSOR_NHWC;
        input.buf = (void *)item.input1;
        input.pass_through = false;
        inputs.push_back(input);
      }

      rknn_inputs_set(threadResource.modelContext, io_num.n_input,
                      inputs.data());

      // 设置输出
      auto *outputs = new rknn_output[io_num.n_output]();
      for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 0;
      }
      int ret;
      // 调用npu进行推演
      ret = rknn_run(threadResource.modelContext, NULL);
      // 获取npu的推演输出结果
      ret = rknn_outputs_get(threadResource.modelContext, io_num.n_output,
                             outputs, NULL);
      inferenceCount++;
      if (((rknnFlag & RKNN_FLAG_COLLECT_PERF_MASK) != 0) &&
          inferenceCount % 100 == 0) {
        // 获取性能信息
        rknn_perf_detail perf;
        ret = rknn_query(threadResource.modelContext, RKNN_QUERY_PERF_DETAIL,
                         &perf, sizeof(perf));
        if (ret == RKNN_SUCC) {
          std::cout << "==========性能信息==========" << std::endl;
          std::cout << perf.perf_data << std::endl;
        } else {
          std::cout << "获取性能信息失败" << std::endl;
        }
      }
      // 将结果放入队列
      item.rawOutputs = outputs;
      item.output = postprocess(outputs);
      {
        std::unique_lock<std::mutex> lock(resultQueueMutex);
        resultQueue.push(id);
        resultCond.notify_all();
      }
    }
  }

public:
  auto submit(const inputDtype *input1) -> taskId {
    std::lock_guard<std::mutex> lock(taskResourceAllocMutex);
    auto input1_buf = new inputDtype[inputShape[0] * inputShape[1] *
                                     inputShape[2] * inputShape[3]];
    memcpy(input1_buf, input1,
           sizeof(inputDtype) * inputShape[0] * inputShape[1] * inputShape[2] *
               inputShape[3]);
    auto tid = lastId++;
    TaskResources r{
        .input1 = input1_buf,
        .rawOutputs = nullptr,
    };
    taskResourcesMap.insert(std::make_pair(tid, r));
    inputQueueMutex.lock();
    inputQueue.push(tid);
    inputQueueMutex.unlock();
    threadResources[tid % threadResources.size()]->workerCond.notify_all();
    return tid;
  }
  auto getLastResult() -> std::pair<taskId, const outputDtype *> {
    if constexpr (sorted_output) {
      std::unique_lock<std::mutex> lock(resultQueueMutex);
      resultCond.wait(lock, [&] {
        return resultQueue.top() == lastOutputId || requestStop;
      });
      auto id = resultQueue.top();
      resultQueue.pop();
      auto &r = taskResourcesMap.at(id);
      lastOutputId = id + 1;
      return std::make_pair(id, r.output);
    } else {
      std::lock_guard<std::mutex> lock(resultQueueMutex);
      auto id = resultQueue.front();
      resultQueue.pop();
      auto &r = taskResourcesMap.at(id);
      return std::make_pair(id, r.output);
    }
  }

  auto waitForResult() -> std::pair<taskId, const outputDtype *> {
    if constexpr (sorted_output) {
      std::unique_lock<std::mutex> lock(resultQueueMutex);
      resultCond.wait(lock, [&] {
        return !resultQueue.empty() && resultQueue.top() == lastOutputId ||
               requestStop;
      });
      if(resultQueue.empty()){
        return std::make_pair(-1, nullptr);
      }
      auto id = resultQueue.top();
      resultQueue.pop();
      auto &r = taskResourcesMap.at(id);
      lastOutputId = id + 1;
      return std::make_pair(id, r.output);

    } else {
      std::unique_lock<std::mutex> lock(resultQueueMutex);
      resultCond.wait(lock,
                      [&] { return !resultQueue.empty() || requestStop; });
      if (requestStop) {
        return std::make_pair(-1, nullptr);
      }
      auto id = resultQueue.front();
      resultQueue.pop();
      auto &r = taskResourcesMap.at(id);
      return std::make_pair(id, r.output);
    }
  }
  auto getInputQueueSize() -> size_t {
    std::lock_guard<std::mutex> lock(inputQueueMutex);
    return inputQueue.size();
  }
  auto getResultQueueSize() -> size_t {
    std::lock_guard<std::mutex> lock(resultQueueMutex);
    return resultQueue.size();
  }
  void free(taskId id) {
    std::lock_guard<std::mutex> lock(taskResourceAllocMutex);
    auto &item = taskResourcesMap.at(id);
    auto &threadResource = threadResources[id % threadResources.size()];
    rknn_outputs_release(threadResource->modelContext, io_num.n_output,
                         item.rawOutputs);
    delete[] item.input1;
    delete item.output;
    taskResourcesMap.erase(id);
  }
  queued_rknn_worker(const char *model_name, rknn_core_mask target_core_mask,
                     int threads_per_core, uint32_t flag = 0)
      : rknnFlag(flag) {
    using namespace std;
    /* Create the neural network */
    printf("Loading mode...\n");
    int model_data_size = 0;

    // 计算总线程数
    vector<int> use_npu_cores;
    if ((target_core_mask & RKNN_NPU_CORE_0) != 0) {
      use_npu_cores.push_back(0);
    }
    if ((target_core_mask & RKNN_NPU_CORE_1) != 0) {
      use_npu_cores.push_back(1);
    }
    if ((target_core_mask & RKNN_NPU_CORE_2) != 0) {
      use_npu_cores.push_back(2);
    }
    if (use_npu_cores.empty()) {
      throw std::runtime_error("没有选择NPU核心");
    }
    int use_npu_cores_count = use_npu_cores.size();
    int thread_count = use_npu_cores_count * threads_per_core;
    for (int i = 0; i < thread_count; i++) {
      auto *threadResource = new ThreadResources();
      threadResources.push_back(threadResource);
    }
    auto &context0 = threadResources[0]->modelContext;
    // 读取模型文件数据
    model_data = load_model(model_name, &model_data_size);
    // 通过模型文件初始化第一个线程的context
    int ret;
    ret = rknn_init(&context0, model_data, model_data_size, rknnFlag, NULL);
    if (ret < 0) {
      printf("rknn_init error ret=%d\n", ret);
      exit(-1);
    }
    // 复制到其他线程
    for (int i = 1; i < thread_count; i++) {
      ret = rknn_dup_context(&context0, &threadResources[i]->modelContext);
      if (ret < 0) {
        printf("rknn_dup_context error ret=%d\n", ret);
        exit(-1);
      }
    }
    // 设置线程的核心关联
    for (int i = 0; i < threads_per_core; i++) {
      for (int j = 0; j < use_npu_cores_count; j++) {
        int thread_index = i * use_npu_cores_count + j;
        cout << "thread_index:" << thread_index << endl;
        cout << "use_npu_cores[j]:" << use_npu_cores[j] << endl;
        ret = rknn_set_core_mask(
            threadResources[thread_index]->modelContext,
            static_cast<rknn_core_mask>(1 << use_npu_cores[j]));
        if (ret < 0) {
          printf("rknn_set_core_mask error ret=%d\n", ret);
          exit(-1);
        }
      }
    }

    // 初始化rknn类的版本
    ret = rknn_query(context0, RKNN_QUERY_SDK_VERSION, &version,
                     sizeof(rknn_sdk_version));
    if (ret < 0) {
      printf("rknn_init error ret=%d\n", ret);
      exit(-1);
    }

    // 获取模型的输入参数
    ret = rknn_query(context0, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
      printf("rknn_init error ret=%d\n", ret);
      exit(-1);
    }

    // 设置输入数组
    input_attrs = new rknn_tensor_attr[io_num.n_input];
    memset(input_attrs, 0, sizeof(*input_attrs) * io_num.n_input);
    assert(io_num.n_input == 1 && "输入个数必须为1");
    for (int i = 0; i < io_num.n_input; i++) {
      input_attrs[i].index = i;
      ret = rknn_query(context0, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                       sizeof(rknn_tensor_attr));
      std::ostringstream oss;
      oss << "Input[" << i << "] name:" << input_attrs[i].name
          << " dtype:" << get_type_string(input_attrs[i].type) << " shape:["
          << input_attrs[i].dims[0] << ", " << input_attrs[i].dims[1] << ", "
          << input_attrs[i].dims[2] << ", " << input_attrs[i].dims[3]
          << "] fmt:" << get_format_string(input_attrs[i].fmt)
          << " wstride:" << input_attrs[i].w_stride;
      std::cout << oss.str() << std::endl;
      inputShape[0] = input_attrs[i].dims[0];
      inputShape[1] = input_attrs[i].dims[1];
      inputShape[2] = input_attrs[i].dims[2];
      inputShape[3] = input_attrs[i].dims[3];
      if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
      }
    }

    // 设置输出数组
    output_attrs = new rknn_tensor_attr[io_num.n_output];
    memset(output_attrs, 0, sizeof(*output_attrs) * io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++) {
      output_attrs[i].index = i;
      ret = rknn_query(context0, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                       sizeof(rknn_tensor_attr));
      std::ostringstream oss;
      oss << "Output[" << i << "] name:" << output_attrs[i].name
          << " dtype:" << get_type_string(output_attrs[i].type) << " shape:["
          << output_attrs[i].dims[0] << ", " << output_attrs[i].dims[1] << ", "
          << output_attrs[i].dims[2] << ", " << output_attrs[i].dims[3]
          << "] fmt:" << get_format_string(output_attrs[i].fmt)
          << " wstride:" << output_attrs[i].w_stride;

      std::cout << oss.str() << std::endl;
    }

    // 设置输入参数
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
      printf("model is NCHW input fmt\n");
    } else {
      printf("model is NHWC input fmt\n");
    }

    // 启动线程
    for (int i = 0; i < thread_count; i++) {
      threadResources[i]->workerThread =
          new std::thread(&queued_rknn_worker::workerThreadFunc, this, i);
    }
  }

  ~queued_rknn_worker() {
    // 等待处理完毕
    while (!inputQueue.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    requestStop = true;
    resultCond.notify_all();
    for (int i = 0; i < threadResources.size(); i++) {
      threadResources[i]->workerCond.notify_all();
      threadResources[i]->workerThread->join();
      delete threadResources[i]->workerThread;
      rknn_destroy(threadResources[i]->modelContext);
      delete threadResources[i];
    }
    delete[] input_attrs;
    delete[] output_attrs;
    if (model_data != nullptr) {
      ::free(model_data);
    }
    vector<taskId> ids;
    for (auto &item : taskResourcesMap) {
      ids.push_back(item.first);
    }
    for (auto &id : ids) {
      free(id);
    }
  }
};