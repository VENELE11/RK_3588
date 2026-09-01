#define main single_entry_point
#include "main.cpp"
#undef main

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace {
struct Worker {
    rknn_context ctx = 0;
    std::vector<rknn_tensor_attr> attrs;
    std::vector<std::pair<int, int>> grids;
    std::vector<uint8_t> model;
    int frames = 0;

    explicit Worker(const std::vector<uint8_t>& model_data, int core_index) : model(model_data) {
        check(rknn_init(&ctx, model.data(), model.size(), 0, nullptr), "rknn_init");
        const rknn_core_mask mask = core_index == 0 ? RKNN_NPU_CORE_0 : (core_index == 1 ? RKNN_NPU_CORE_1 : RKNN_NPU_CORE_2);
        check(rknn_set_core_mask(ctx, mask), "rknn_set_core_mask");
        rknn_input_output_num io{};
        check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "query io num");
        if (io.n_input != 1 || io.n_output != 3) throw std::runtime_error("multi mode currently supports v5 3-head models");
        attrs.resize(io.n_output); grids.reserve(io.n_output);
        for (uint32_t i = 0; i < io.n_output; ++i) {
            attrs[i].index = i;
            check(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attrs[i], sizeof(attrs[i])), "query output");
            grids.emplace_back(static_cast<int>(attrs[i].dims[2]), static_cast<int>(attrs[i].dims[3]));
        }
    }
    ~Worker() { if (ctx) rknn_destroy(ctx); }

    size_t infer(const cv::Mat& bgr) {
        LetterboxInfo lb{}; cv::Mat rgb = letterbox_rgb_rga(bgr, lb);
        rknn_input input{}; input.index = 0; input.type = RKNN_TENSOR_UINT8; input.fmt = RKNN_TENSOR_NHWC;
        input.size = static_cast<uint32_t>(rgb.total() * rgb.elemSize()); input.buf = rgb.data; input.pass_through = 0;
        check(rknn_inputs_set(ctx, 1, &input), "inputs_set"); check(rknn_run(ctx, nullptr), "run");
        rknn_input_output_num io{}; check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "query io num");
        std::vector<rknn_output> outputs(io.n_output); for (auto& o : outputs) o.want_float = 0;
        check(rknn_outputs_get(ctx, io.n_output, outputs.data(), nullptr), "outputs_get");
        std::vector<TensorView> views(io.n_output);
        for (uint32_t i = 0; i < io.n_output; ++i) views[i] = {outputs[i].buf, attrs[i].type, attrs[i].zp, attrs[i].scale};
        auto dets = decode_v5(views, grids, lb, bgr.size());
        check(rknn_outputs_release(ctx, io.n_output, outputs.data()), "outputs_release");
        ++frames;
        return dets.size();
    }
};
}

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::cerr << "usage: " << argv[0] << " model.rknn video.mp4 threads max_frames\n";
        return 2;
    }
    try {
        const auto model = load_file(argv[1]);
        const std::string video_path = argv[2];
        const int thread_count = std::max(1, std::stoi(argv[3]));
        const int max_frames = std::stoi(argv[4]);
        std::vector<std::unique_ptr<Worker>> workers;
        for (int i = 0; i < thread_count; ++i) workers.emplace_back(new Worker(model, i % 3));

        std::deque<cv::Mat> queue; std::mutex mutex; std::condition_variable ready, space;
        bool finished = false; size_t total_dets = 0; int submitted = 0;
        const auto start = std::chrono::steady_clock::now();
        std::vector<std::thread> threads;
        for (int i = 0; i < thread_count; ++i) {
            threads.emplace_back([&, i] {
                while (true) {
                    cv::Mat frame;
                    { std::unique_lock<std::mutex> lock(mutex); ready.wait(lock, [&] { return finished || !queue.empty(); });
                      if (queue.empty() && finished) return; frame = std::move(queue.front()); queue.pop_front(); space.notify_one(); }
                    const size_t n = workers[i]->infer(frame);
                    { std::lock_guard<std::mutex> lock(mutex); total_dets += n; }
                }
            });
        }
        cv::VideoCapture cap(video_path); if (!cap.isOpened()) throw std::runtime_error("cannot open video: " + video_path);
        cv::Mat frame;
        while ((max_frames < 0 || submitted < max_frames) && cap.read(frame)) {
            std::unique_lock<std::mutex> lock(mutex);
            space.wait(lock, [&] { return queue.size() < static_cast<size_t>(thread_count * 2); });
            queue.push_back(std::move(frame)); ++submitted; ready.notify_one();
        }
        { std::lock_guard<std::mutex> lock(mutex); finished = true; } ready.notify_all();
        for (auto& t : threads) t.join();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << std::fixed << std::setprecision(3) << "threads=" << thread_count << " frames=" << submitted
                  << " elapsed_s=" << seconds << " fps=" << submitted / std::max(seconds, 1e-9)
                  << " avg_detections=" << (submitted ? static_cast<double>(total_dets) / submitted : 0.0) << "\n";
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
    return 0;
}
