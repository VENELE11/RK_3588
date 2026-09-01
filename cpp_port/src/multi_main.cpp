#define main single_entry_point
#include "main.cpp"
#undef main

#define CPP_PORT_MULTI_NO_MAIN
#include "modern_main.cpp"
#undef CPP_PORT_MULTI_NO_MAIN

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace {

enum class ModelKind { V5, V8, V11, V11Norm };

ModelKind parse_model_kind(const std::string& name) {
    if (name == "v5" || name == "v5s") return ModelKind::V5;
    if (name == "v8" || name == "v8s") return ModelKind::V8;
    if (name == "v11") return ModelKind::V11;
    if (name == "v11_norm" || name == "v11s_norm2") return ModelKind::V11Norm;
    throw std::runtime_error("unknown model type: " + name +
                             " (expected v5, v8, v11, or v11_norm)");
}

const char* model_kind_name(ModelKind kind) {
    switch (kind) {
        case ModelKind::V5: return "v5";
        case ModelKind::V8: return "v8";
        case ModelKind::V11: return "v11";
        case ModelKind::V11Norm: return "v11_norm";
    }
    return "unknown";
}

rknn_core_mask core_mask_for(int index) {
    return index % 3 == 0 ? RKNN_NPU_CORE_0 :
           (index % 3 == 1 ? RKNN_NPU_CORE_1 : RKNN_NPU_CORE_2);
}

struct Worker {
    ModelKind kind;
    rknn_context ctx = 0;
    std::vector<rknn_tensor_attr> attrs;
    std::vector<std::pair<int, int>> grids;
    std::vector<uint8_t> model;
    std::unique_ptr<RknnIoBuffers> io_buffers;
    PreprocessBuffers preprocess;
    TimingStats timing_stats;

    Worker(ModelKind model_kind, const std::vector<uint8_t>& model_data, int core_index)
        : kind(model_kind), model(model_data) {
        try {
            check(rknn_init(&ctx, model.data(), model.size(), 0, nullptr), "rknn_init");
            check(rknn_set_core_mask(ctx, core_mask_for(core_index)), "rknn_set_core_mask");
            rknn_input_output_num io{};
            check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "query io num");
            rknn_tensor_attr input_attr{}; input_attr.index = 0;
            check(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr)), "query input");
            attrs.resize(io.n_output);
            for (uint32_t i = 0; i < io.n_output; ++i) {
                attrs[i].index = i;
                check(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attrs[i], sizeof(attrs[i])), "query output");
            }
            validate_layout(io);
            io_buffers.reset(new RknnIoBuffers(ctx, input_attr, attrs));
        } catch (...) {
            io_buffers.reset();
            if (ctx) { rknn_destroy(ctx); ctx = 0; }
            throw;
        }
    }

    ~Worker() {
        io_buffers.reset();
        if (ctx) rknn_destroy(ctx);
    }

    void validate_layout(const rknn_input_output_num& io) {
        if (io.n_input != 1) throw std::runtime_error("multi mode expects one input tensor");
        if (kind == ModelKind::V5) {
            if (io.n_output != 3) throw std::runtime_error("v5 expects three output tensors");
            for (const auto& attr : attrs) {
                if (attr.n_dims < 4 || attr.dims[1] != 255)
                    throw std::runtime_error("v5 output is not NCHW [1,255,H,W]");
                grids.emplace_back(static_cast<int>(attr.dims[2]), static_cast<int>(attr.dims[3]));
            }
        } else if (kind == ModelKind::V8) {
            if (io.n_output != 6) throw std::runtime_error("v8 expects six output tensors");
        } else {
            if (io.n_output != 1 || attrs[0].n_dims != 3)
                throw std::runtime_error("v11 expects one 3-D output tensor");
        }
    }

    size_t infer(const cv::Mat& bgr) {
        StageTimes timing;
        const auto total_begin = ProfileClock::now();
        LetterboxInfo lb{};
        cv::Mat input = io_buffers->input_frame();
        preprocess.prepare(bgr, lb, true, input);
        const auto preprocess_end = ProfileClock::now();
        io_buffers->sync_input();
        const auto input_set_end = ProfileClock::now();
        check(rknn_run(ctx, nullptr), "run");
        const auto inference_end = ProfileClock::now();
        const auto& views = io_buffers->views();
        const auto output_get_end = ProfileClock::now();

        std::vector<Detection> detections;
        if (kind == ModelKind::V5)
            detections = decode_v5(views, grids, lb, bgr.size());
        else if (kind == ModelKind::V8)
            detections = decode_v8(views, attrs, lb, bgr.size());
        else if (kind == ModelKind::V11)
            detections = decode_v11(views[0], attrs[0], lb, bgr.size());
        else
            detections = decode_v11_norm(views[0], attrs[0], lb, bgr.size());
        const auto postprocess_end = ProfileClock::now();
        const auto release_end = postprocess_end;

        timing.preprocess_ms = elapsed_ms(total_begin, preprocess_end);
        timing.input_set_ms = elapsed_ms(preprocess_end, input_set_end);
        timing.inference_ms = elapsed_ms(input_set_end, inference_end);
        timing.output_get_ms = elapsed_ms(inference_end, output_get_end);
        timing.postprocess_ms = elapsed_ms(output_get_end, postprocess_end);
        timing.output_release_ms = elapsed_ms(postprocess_end, release_end);
        timing.total_ms = elapsed_ms(total_begin, release_end);
        timing_stats.add(timing);
        return detections.size();
    }
};

} // namespace

int main(int argc, char** argv) {
    const bool legacy_v5 = argc == 5;
    const bool image_mode = argc == 8 && std::string(argv[3]) == "--images";
    if ((!legacy_v5 && argc != 6 && !image_mode) || argc > 8) {
        std::cerr << "usage: " << argv[0]
                  << " v5|v8|v11|v11_norm model.rknn video.mp4 threads max_frames\n"
                  << "images: " << argv[0]
                  << " v5|v8|v11|v11_norm model.rknn --images image_dir image_list.txt threads max_frames\n"
                  << "legacy v5: " << argv[0] << " model.rknn video.mp4 threads max_frames\n";
        return 2;
    }
    try {
        const ModelKind kind = parse_model_kind(legacy_v5 ? "v5" : argv[1]);
        const std::string model_path = legacy_v5 ? argv[1] : argv[2];
        const std::string video_path = legacy_v5 ? argv[2] : (image_mode ? "" : argv[3]);
        const std::string image_dir = image_mode ? argv[4] : "";
        const std::string image_list = image_mode ? argv[5] : "";
        const int thread_count = std::max(1, std::stoi(
            legacy_v5 ? argv[3] : (image_mode ? argv[6] : argv[4])));
        const int max_frames = std::stoi(
            legacy_v5 ? argv[4] : (image_mode ? argv[7] : argv[5]));
        const auto model = load_file(model_path);

        std::vector<std::unique_ptr<Worker>> workers;
        workers.reserve(thread_count);
        for (int i = 0; i < thread_count; ++i)
            workers.emplace_back(new Worker(kind, model, i));

        std::deque<cv::Mat> queue;
        std::mutex mutex;
        std::condition_variable ready, space;
        bool finished = false;
        std::atomic<bool> failed{false};
        std::string error_message;
        size_t total_dets = 0;
        int submitted = 0;
        const auto start = std::chrono::steady_clock::now();

        std::vector<std::thread> threads;
        threads.reserve(thread_count);
        for (int i = 0; i < thread_count; ++i) {
            threads.emplace_back([&, i] {
                try {
                    while (true) {
                        cv::Mat frame;
                        {
                            std::unique_lock<std::mutex> lock(mutex);
                            ready.wait(lock, [&] { return finished || failed || !queue.empty(); });
                            if (failed || (queue.empty() && finished)) return;
                            frame = std::move(queue.front());
                            queue.pop_front();
                            space.notify_one();
                        }
                        const size_t count = workers[i]->infer(frame);
                        {
                            std::lock_guard<std::mutex> lock(mutex);
                            total_dets += count;
                        }
                    }
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (!failed) error_message = e.what();
                    failed = true;
                    finished = true;
                    ready.notify_all();
                    space.notify_all();
                }
            });
        }

        auto submit_frame = [&](cv::Mat&& frame) {
            std::unique_lock<std::mutex> lock(mutex);
            space.wait(lock, [&] {
                return failed || queue.size() < static_cast<size_t>(thread_count * 2);
            });
            if (failed) return false;
            queue.push_back(std::move(frame));
            ++submitted;
            ready.notify_one();
            return true;
        };
        if (image_mode) {
            std::ifstream list(image_list);
            if (!list) throw std::runtime_error("cannot open image list: " + image_list);
            std::string name;
            while (!failed && (max_frames < 0 || submitted < max_frames) && std::getline(list, name)) {
                if (name.empty()) continue;
                cv::Mat image = cv::imread(join_path(image_dir, name), cv::IMREAD_COLOR);
                if (image.empty()) throw std::runtime_error("cannot read image: " + join_path(image_dir, name));
                if (!submit_frame(std::move(image))) break;
            }
        } else {
            cv::VideoCapture cap(video_path);
            if (!cap.isOpened()) throw std::runtime_error("cannot open video: " + video_path);
            cv::Mat frame;
            while (!failed && (max_frames < 0 || submitted < max_frames) && cap.read(frame)) {
                if (!submit_frame(std::move(frame))) break;
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            finished = true;
        }
        ready.notify_all();
        space.notify_all();
        for (auto& thread : threads) thread.join();
        if (failed) throw std::runtime_error(error_message.empty() ? "worker failed" : error_message);

        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        TimingStats aggregate;
        for (const auto& worker : workers) aggregate.merge(worker->timing_stats);
        std::cout << std::fixed << std::setprecision(3)
                  << "model=" << model_kind_name(kind)
                  << " threads=" << thread_count
                  << " " << (image_mode ? "images=" : "frames=") << submitted
                  << " elapsed_s=" << seconds
                  << " fps=" << submitted / std::max(seconds, 1e-9)
                  << " avg_detections=" << (submitted ? static_cast<double>(total_dets) / submitted : 0.0)
                  << '\n';
        print_timing_stats(std::string("multi_") + model_kind_name(kind), aggregate);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
