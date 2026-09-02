#define main single_entry_point
#include "main.cpp"
#undef main

namespace {
float softmax16(const TensorView& t, size_t base, int channel) {
    float sum = 0.0f, out = 0.0f;
    for (int k = 0; k < 16; ++k) {
        const float logit = std::max(-30.0f, std::min(30.0f, tensor_value(t, base + k * channel)));
        const float e = std::exp(logit); sum += e; out += k * e;
    }
    return out / std::max(sum, 1e-9f);
}

struct V8Candidate {
    int branch, y, x;
    float score;
    int cls;
    uint32_t order;
};

template <typename Q>
void collect_v8_quantized(const TensorView& tensor, int branch, int gh, int gw,
                          std::vector<V8Candidate>& candidates, uint32_t& candidate_order) {
    const Q* data = static_cast<const Q*>(tensor.data);
    const size_t cells = static_cast<size_t>(gh) * gw;
    std::vector<int> best_q(cells); std::vector<uint8_t> best_cls(cells);
#if defined(__aarch64__) || defined(__ARM_NEON)
    if constexpr (std::is_same_v<Q, int8_t>)
        scan_argmax_neon(reinterpret_cast<const int8_t*>(data), cells, cells, best_q, best_cls);
    else if constexpr (std::is_same_v<Q, uint8_t>)
        scan_argmax_neon(reinterpret_cast<const uint8_t*>(data), cells, cells, best_q, best_cls);
    else
#endif
    {
        for (size_t cell = 0; cell < cells; ++cell) {
            int cls = 0; int value = static_cast<int>(data[cell]);
            for (int c = 1; c < kClasses; ++c) {
                const int q = static_cast<int>(data[static_cast<size_t>(c) * cells + cell]);
                if (q > value) { value = q; cls = c; }
            }
            best_q[cell] = value; best_cls[cell] = static_cast<uint8_t>(cls);
        }
    }
    for (size_t cell = 0; cell < cells; ++cell) {
        const float score = sigmoid((best_q[cell] - tensor.zp) * tensor.scale);
        if (score > kObjThreshold) {
            const int y = static_cast<int>(cell / gw);
            const int x = static_cast<int>(cell % gw);
            candidates.push_back({branch, y, x, score, static_cast<int>(best_cls[cell]), candidate_order++});
        }
    }
}

void collect_v8_float(const TensorView& tensor, int branch, int gh, int gw,
                      std::vector<V8Candidate>& candidates, uint32_t& candidate_order) {
    const float* data = static_cast<const float*>(tensor.data);
    const size_t cells = static_cast<size_t>(gh) * gw;
    for (size_t cell = 0; cell < cells; ++cell) {
        int cls = 0;
        float best_logit = data[cell];
        for (int c = 1; c < kClasses; ++c) {
            const float value = data[static_cast<size_t>(c) * cells + cell];
            if (value > best_logit) { best_logit = value; cls = c; }
        }
        const float score = sigmoid(best_logit);
        if (score > kObjThreshold) {
            const int y = static_cast<int>(cell / gw);
            const int x = static_cast<int>(cell % gw);
            candidates.push_back({branch, y, x, score, cls, candidate_order++});
        }
    }
}

std::vector<Detection> decode_v8(const std::vector<TensorView>& v, const std::vector<rknn_tensor_attr>& a, const LetterboxInfo& lb, const cv::Size& original) {
    std::vector<V8Candidate> candidates;
    candidates.reserve(4096);
    uint32_t candidate_order = 0;
    for (int branch = 0; branch < 3; ++branch) {
        const auto& box_attr = a[branch * 2];
        const int gh = box_attr.dims[2], gw = box_attr.dims[3];
        const auto& cls_tensor = v[branch * 2 + 1];
        if (cls_tensor.type == RKNN_TENSOR_INT8)
            collect_v8_quantized<int8_t>(cls_tensor, branch, gh, gw, candidates, candidate_order);
        else if (cls_tensor.type == RKNN_TENSOR_UINT8)
            collect_v8_quantized<uint8_t>(cls_tensor, branch, gh, gw, candidates, candidate_order);
        else
            collect_v8_float(cls_tensor, branch, gh, gw, candidates, candidate_order);
    }
    keep_top_detections(candidates);
    std::vector<Detection> raw;
    raw.reserve(candidates.size());
    for (const auto& c : candidates) {
        const auto& box_attr = a[c.branch * 2];
        const int gh = box_attr.dims[2], gw = box_attr.dims[3], stride = 640 / gh;
        const size_t cells = static_cast<size_t>(gh) * gw;
        const size_t cell = static_cast<size_t>(c.y) * gw + c.x;
        float d[4]{};
        for (int side = 0; side < 4; ++side)
            d[side] = softmax16(v[c.branch * 2], static_cast<size_t>(side) * 16 * cells + cell, cells);
        const float cx = (c.x + 0.5f) * stride, cy = (c.y + 0.5f) * stride;
        raw.push_back({cx - d[0] * stride, cy - d[1] * stride,
                       cx + d[2] * stride, cy + d[3] * stride,
                       c.score, c.cls, c.order});
    }
    auto out = nms(std::move(raw), true);
    for (auto& d : out) {
        d.x1 = std::max(0.0f, std::min((d.x1 - lb.dx) / lb.scale, static_cast<float>(original.width)));
        d.y1 = std::max(0.0f, std::min((d.y1 - lb.dy) / lb.scale, static_cast<float>(original.height)));
        d.x2 = std::max(0.0f, std::min((d.x2 - lb.dx) / lb.scale, static_cast<float>(original.width)));
        d.y2 = std::max(0.0f, std::min((d.y2 - lb.dy) / lb.scale, static_cast<float>(original.height)));
    }
    return out;
}

struct FusedCandidate {
    int index;
    float score;
    int cls;
    uint32_t order;
};

template <typename Q>
void collect_fused_quantized(const TensorView& tensor, bool channels_first, int n,
                             bool apply_sigmoid, std::vector<FusedCandidate>& candidates) {
    const Q* data = static_cast<const Q*>(tensor.data);
#if defined(__aarch64__) || defined(__ARM_NEON)
    if (channels_first && (std::is_same_v<Q, int8_t> || std::is_same_v<Q, uint8_t>)) {
        std::vector<int> best_q(n); std::vector<uint8_t> best_cls(n);
        if constexpr (std::is_same_v<Q, int8_t>)
            scan_argmax_neon(reinterpret_cast<const int8_t*>(data + static_cast<size_t>(4) * n), n, n, best_q, best_cls);
        else
            scan_argmax_neon(reinterpret_cast<const uint8_t*>(data + static_cast<size_t>(4) * n), n, n, best_q, best_cls);
        for (int i = 0; i < n; ++i) {
            const float value = (best_q[i] - tensor.zp) * tensor.scale;
            const float score = apply_sigmoid ? sigmoid(value) : value;
            if (score > kObjThreshold)
                candidates.push_back({i, score, static_cast<int>(best_cls[i]), static_cast<uint32_t>(i)});
        }
        return;
    }
#endif
    for (int i = 0; i < n; ++i) {
        const auto index = [&](int c) -> size_t {
            return channels_first ? static_cast<size_t>(4 + c) * n + i
                                  : static_cast<size_t>(i) * 84 + 4 + c;
        };
        int cls = 0;
        int best_q = static_cast<int>(data[index(0)]);
        for (int c = 1; c < kClasses; ++c) {
            const int q = static_cast<int>(data[index(c)]);
            if (q > best_q) { best_q = q; cls = c; }
        }
        const float value = (best_q - tensor.zp) * tensor.scale;
        const float score = apply_sigmoid ? sigmoid(value) : value;
        if (score > kObjThreshold)
            candidates.push_back({i, score, cls, static_cast<uint32_t>(i)});
    }
}

void collect_fused_float(const TensorView& tensor, bool channels_first, int n,
                         bool apply_sigmoid, std::vector<FusedCandidate>& candidates) {
    const float* data = static_cast<const float*>(tensor.data);
    for (int i = 0; i < n; ++i) {
        const auto index = [&](int c) -> size_t {
            return channels_first ? static_cast<size_t>(4 + c) * n + i
                                  : static_cast<size_t>(i) * 84 + 4 + c;
        };
        int cls = 0;
        float best = tensor_value(tensor, index(0));
        for (int c = 1; c < kClasses; ++c) {
            const float value = tensor_value(tensor, index(c));
            if (value > best) { best = value; cls = c; }
        }
        const float score = apply_sigmoid ? sigmoid(best) : best;
        if (score > kObjThreshold)
            candidates.push_back({i, score, cls, static_cast<uint32_t>(i)});
    }
}

void collect_fused_candidates(const TensorView& tensor, bool channels_first, int n,
                              bool apply_sigmoid, std::vector<FusedCandidate>& candidates) {
    if (tensor.type == RKNN_TENSOR_INT8)
        collect_fused_quantized<int8_t>(tensor, channels_first, n, apply_sigmoid, candidates);
    else if (tensor.type == RKNN_TENSOR_UINT8)
        collect_fused_quantized<uint8_t>(tensor, channels_first, n, apply_sigmoid, candidates);
    else
        collect_fused_float(tensor, channels_first, n, apply_sigmoid, candidates);
}

std::vector<Detection> decode_v11(const TensorView& t, const rknn_tensor_attr& attr, const LetterboxInfo& lb, const cv::Size& original) {
    const int d1 = attr.dims[1], d2 = attr.dims[2]; const bool cn = d1 == 84; const int n = cn ? d2 : d1;
    std::vector<FusedCandidate> candidates;
    candidates.reserve(n);
    auto at = [&](int row, int col) { return tensor_value(t, cn ? static_cast<size_t>(col) * n + row : static_cast<size_t>(row) * 84 + col); };
    collect_fused_candidates(t, cn, n, true, candidates);
    keep_top_detections(candidates);
    std::vector<Detection> raw;
    raw.reserve(candidates.size());
    for (const auto& c : candidates) {
        const float cx = at(c.index, 0), cy = at(c.index, 1);
        const float w = at(c.index, 2), h = at(c.index, 3);
        raw.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
                       c.score, c.cls, c.order});
    }
    auto out = nms(std::move(raw), true);
    for (auto& d : out) {
        d.x1 = std::max(0.0f, std::min((d.x1 - lb.dx) / lb.scale, static_cast<float>(original.width)));
        d.y1 = std::max(0.0f, std::min((d.y1 - lb.dy) / lb.scale, static_cast<float>(original.height)));
        d.x2 = std::max(0.0f, std::min((d.x2 - lb.dx) / lb.scale, static_cast<float>(original.width)));
        d.y2 = std::max(0.0f, std::min((d.y2 - lb.dy) / lb.scale, static_cast<float>(original.height)));
    }
    return out;
}
std::vector<Detection> decode_v11_norm(const TensorView& t, const rknn_tensor_attr& attr, const LetterboxInfo& lb, const cv::Size& original) {
    const int d1 = attr.dims[1], d2 = attr.dims[2]; const bool cn = d1 == 84; const int n = cn ? d2 : d1;
    std::vector<FusedCandidate> candidates;
    candidates.reserve(n);
    auto at = [&](int row, int col) { return tensor_value(t, cn ? static_cast<size_t>(col) * n + row : static_cast<size_t>(row) * 84 + col); };
    collect_fused_candidates(t, cn, n, false, candidates);
    keep_top_detections(candidates);
    std::vector<Detection> raw;
    raw.reserve(candidates.size());
    for (const auto& c : candidates) {
        const float cx = at(c.index, 0) * 640.0f, cy = at(c.index, 1) * 640.0f;
        const float w = at(c.index, 2) * 640.0f, h = at(c.index, 3) * 640.0f;
        raw.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
                       c.score, c.cls, c.order});
    }
    auto out = nms(std::move(raw), true);
    for (auto& d : out) {
        d.x1 = std::max(0.0f, std::min((d.x1 - lb.dx) / lb.scale, static_cast<float>(original.width)));
        d.y1 = std::max(0.0f, std::min((d.y1 - lb.dy) / lb.scale, static_cast<float>(original.height)));
        d.x2 = std::max(0.0f, std::min((d.x2 - lb.dx) / lb.scale, static_cast<float>(original.width)));
        d.y2 = std::max(0.0f, std::min((d.y2 - lb.dy) / lb.scale, static_cast<float>(original.height)));
    }
    return out;
}
}

#ifndef CPP_PORT_MULTI_NO_MAIN
int main(int argc, char** argv) {
    const bool video_mode = argc >= 5 && std::string(argv[3]) == "--video";
    if (argc != 4 && argc != 7 && argc != 5 && argc != 6) {
        std::cerr << "usage:\n  " << argv[0] << " v8|v11|v11_norm model.rknn image.jpg\n"
                  << "  " << argv[0] << " v8|v11|v11_norm model.rknn --video video.mp4 [max_frames]\n"
                  << "  " << argv[0] << " v8|v11|v11_norm model.rknn --eval images_dir image_list.txt out.json\n";
        return 2;
    }
    try {
        const std::string type = argv[1]; auto model = load_file(argv[2]);
        rknn_context ctx = 0; check(rknn_init(&ctx, model.data(), model.size(), 0, nullptr), "rknn_init");
        rknn_input_output_num io{}; check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "query io");
        rknn_tensor_attr in_attr{}; in_attr.index = 0;
        check(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr)), "query input");
        std::vector<rknn_tensor_attr> attrs(io.n_output);
        for (uint32_t i = 0; i < io.n_output; ++i) { attrs[i].index = i; check(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attrs[i], sizeof(attrs[i])), "query output"); std::cout << "output" << i << ": dims=" << attrs[i].n_dims << " [" << attrs[i].dims[0] << "," << attrs[i].dims[1] << "," << attrs[i].dims[2] << "," << attrs[i].dims[3] << "]\n"; }
        auto io_buffers = std::make_unique<RknnIoBuffers>(ctx, in_attr, attrs);
        PreprocessBuffers preprocess;
        auto infer_one = [&](const cv::Mat& bgr, bool use_rga, TimingStats* stats) {
            StageTimes timing;
            const auto total_begin = ProfileClock::now();
            LetterboxInfo lb{};
            cv::Mat rgb = io_buffers->input_frame();
            preprocess.prepare(bgr, lb, use_rga, rgb);
            const auto preprocess_end = ProfileClock::now();
            io_buffers->sync_input();
            const auto input_set_end = ProfileClock::now();
            check(rknn_run(ctx, nullptr), "run");
            const auto inference_end = ProfileClock::now();
            const auto& current_views = io_buffers->views();
            const auto output_get_end = ProfileClock::now();
            std::vector<Detection> dets;
            if (type == "v8") dets = decode_v8(current_views, attrs, lb, bgr.size());
            else if (type == "v11" && io.n_output == 1) dets = decode_v11(current_views[0], attrs[0], lb, bgr.size());
            else if (type == "v11_norm" && io.n_output == 1) dets = decode_v11_norm(current_views[0], attrs[0], lb, bgr.size());
            else throw std::runtime_error("unsupported output layout for selected model");
            const auto postprocess_end = ProfileClock::now();
            const auto release_end = ProfileClock::now();
            if (stats) {
                timing.preprocess_ms = elapsed_ms(total_begin, preprocess_end);
                timing.input_set_ms = elapsed_ms(preprocess_end, input_set_end);
                timing.inference_ms = elapsed_ms(input_set_end, inference_end);
                timing.output_get_ms = elapsed_ms(inference_end, output_get_end);
                timing.postprocess_ms = elapsed_ms(output_get_end, postprocess_end);
                timing.output_release_ms = elapsed_ms(postprocess_end, release_end);
                timing.total_ms = elapsed_ms(total_begin, release_end);
                stats->add(timing);
            }
            return dets;
        };
        if (std::string(argv[3]) == "--eval") {
            const auto names = read_lines(argv[5]);
            std::vector<std::pair<long long, std::vector<Detection>>> results;
            results.reserve(names.size()); size_t total_dets = 0; TimingStats timing_stats;
            for (size_t i = 0; i < names.size(); ++i) {
                const cv::Mat bgr = cv::imread(join_path(argv[4], names[i]));
                if (bgr.empty()) { std::cerr << "WARN missing image: " << names[i] << '\n'; continue; }
                auto dets = infer_one(bgr, false, &timing_stats); total_dets += dets.size();
                results.emplace_back(image_id_from_name(names[i]), std::move(dets));
                if ((i + 1) % 10 == 0) std::cout << "processed " << (i + 1) << '/' << names.size()
                                                  << ", total dets=" << total_dets << '\n';
            }
            write_coco_json(argv[6], results);
            print_timing_stats(type + "_eval", timing_stats);
            std::cout << "DONE: " << results.size() << " images -> " << total_dets
                      << " predictions -> " << argv[6] << '\n';
        } else if (video_mode) {
            cv::VideoCapture cap(argv[4]);
            if (!cap.isOpened()) throw std::runtime_error("cannot open video: " + std::string(argv[4]));
            const int max_frames = argc == 6 ? std::stoi(argv[5]) : -1;
            cv::Mat frame; int frames = 0; size_t total_dets = 0; TimingStats timing_stats;
            const auto start = ProfileClock::now();
            while ((max_frames < 0 || frames < max_frames) && cap.read(frame)) {
                total_dets += infer_one(frame, true, &timing_stats).size();
                ++frames;
            }
            const double seconds = std::chrono::duration<double>(ProfileClock::now() - start).count();
            std::cout << std::fixed << std::setprecision(3)
                      << "video_frames=" << frames << " elapsed_s=" << seconds
                      << " fps=" << frames / std::max(seconds, 1e-9)
                      << " avg_detections=" << (frames ? static_cast<double>(total_dets) / frames : 0.0) << '\n';
            print_timing_stats(type + "_video", timing_stats);
        } else {
            const cv::Mat bgr = cv::imread(argv[3]);
            if (bgr.empty()) throw std::runtime_error("cannot read image");
            TimingStats timing_stats;
            const auto dets = infer_one(bgr, true, &timing_stats);
            print_timing_stats(type + "_single", timing_stats);
            std::cout << "detections=" << dets.size() << "\n";
            for (const auto& d : dets) std::cout << d.x1 << ' ' << d.y1 << ' ' << d.x2 << ' ' << d.y2 << ' ' << d.score << ' ' << d.cls << '\n';
        }
        io_buffers.reset();
        rknn_destroy(ctx);
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
    return 0;
}
#endif // CPP_PORT_MULTI_NO_MAIN
