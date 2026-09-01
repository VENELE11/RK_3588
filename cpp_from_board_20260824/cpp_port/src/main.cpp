#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include "im2d.h"
#include "rga.h"

namespace {
constexpr int kImageSize = 640;
constexpr int kClasses = 80;
constexpr float kObjThreshold = 0.01f;
constexpr float kNmsThreshold = 0.45f;
constexpr float kObjLogitThreshold = -4.59511985f; // sigmoid(x) > 0.01
constexpr int kMaxDetections = 500;
const int kAnchors[9][2] = {{10, 13}, {16, 30}, {33, 23}, {30, 61}, {62, 45},
                            {59, 119}, {116, 90}, {156, 198}, {373, 326}};
const int kMasks[3][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};

struct Detection { float x1, y1, x2, y2, score; int cls; };
struct LetterboxInfo { float scale; int dx, dy; };
struct TensorView { const void* data; rknn_tensor_type type; int32_t zp; float scale; };

float sigmoid(float x) {
    x = std::max(-50.0f, std::min(50.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<uint8_t> load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open model: " + path);
    const auto n = f.tellg(); f.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(data.data()), n);
    return data;
}

cv::Mat letterbox_rgb(const cv::Mat& bgr, LetterboxInfo& info) {
    const int h = bgr.rows, w = bgr.cols;
    info.scale = std::min(static_cast<float>(kImageSize) / h,
                          static_cast<float>(kImageSize) / w);
    const int nh = static_cast<int>(std::round(h * info.scale));
    const int nw = static_cast<int>(std::round(w * info.scale));
    cv::Mat resized, canvas(kImageSize, kImageSize, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(bgr, resized, cv::Size(nw, nh));
    info.dy = (kImageSize - nh) / 2;
    info.dx = (kImageSize - nw) / 2;
    resized.copyTo(canvas(cv::Rect(info.dx, info.dy, nw, nh)));
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
    return canvas;
}

cv::Mat letterbox_rgb_rga(const cv::Mat& bgr, LetterboxInfo& info) {
    const int h = bgr.rows, w = bgr.cols;
    info.scale = std::min(static_cast<float>(kImageSize) / h, static_cast<float>(kImageSize) / w);
    const int nh = static_cast<int>(std::round(h * info.scale));
    const int nw = static_cast<int>(std::round(w * info.scale));
    info.dy = (kImageSize - nh) / 2; info.dx = (kImageSize - nw) / 2;
    cv::Mat rgb, canvas(kImageSize, kImageSize, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rga_buffer_t src = wrapbuffer_virtualaddr(rgb.data, w, h, w, h, RK_FORMAT_RGB_888);
    auto* dst_ptr = canvas.data + (static_cast<size_t>(info.dy) * kImageSize + info.dx) * 3;
    rga_buffer_t dst = wrapbuffer_virtualaddr(dst_ptr, nw, nh, kImageSize, kImageSize, RK_FORMAT_RGB_888);
    im_rect sr = {0, 0, w, h}; im_rect dr = {0, 0, nw, nh};
    const int checked = imcheck(src, dst, sr, dr);
    if (checked == IM_STATUS_NOERROR && imresize(src, dst) == IM_STATUS_NOERROR) return canvas;
    return letterbox_rgb(bgr, info);
}

float iou(const Detection& a, const Detection& b) {
    const float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    const float inter = std::max(0.0f, x2 - x1 + 1.0f) * std::max(0.0f, y2 - y1 + 1.0f);
    const float aa = std::max(0.0f, a.x2 - a.x1 + 1.0f) * std::max(0.0f, a.y2 - a.y1 + 1.0f);
    const float ab = std::max(0.0f, b.x2 - b.x1 + 1.0f) * std::max(0.0f, b.y2 - b.y1 + 1.0f);
    return inter / (aa + ab - inter + 1e-9f);
}

std::vector<Detection> nms(std::vector<Detection> dets) {
    if (dets.size() > kMaxDetections) {
        std::partial_sort(dets.begin(), dets.begin() + kMaxDetections, dets.end(),
                          [](const Detection& a, const Detection& b) { return a.score > b.score; });
        dets.resize(kMaxDetections);
    }
    std::map<int, std::vector<Detection>> by_class;
    for (const auto& d : dets) by_class[d.cls].push_back(d);
    std::vector<Detection> out;
    for (auto& [cls, items] : by_class) {
        std::sort(items.begin(), items.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
        std::vector<bool> removed(items.size(), false);
        for (size_t i = 0; i < items.size(); ++i) {
            if (removed[i]) continue;
            out.push_back(items[i]);
            for (size_t j = i + 1; j < items.size(); ++j)
                if (iou(items[i], items[j]) > kNmsThreshold) removed[j] = true;
        }
    }
    std::sort(out.begin(), out.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    return out;
}

std::vector<Detection> decode_v5(const std::vector<TensorView>& outs,
                                 const std::vector<std::pair<int, int>>& grids,
                                 const LetterboxInfo& lb, const cv::Size& original) {
    std::vector<Detection> raw;
    for (int b = 0; b < 3; ++b) {
        const int gh = grids[b].first, gw = grids[b].second;
        const int stride = kImageSize / gh;
        auto at = [&](int a, int c, int y, int x) -> float {
            const size_t i = ((a * 85 + c) * gh + y) * gw + x;
            const auto& o = outs[b];
            if (o.type == RKNN_TENSOR_INT8) return (static_cast<const int8_t*>(o.data)[i] - o.zp) * o.scale;
            if (o.type == RKNN_TENSOR_UINT8) return (static_cast<const uint8_t*>(o.data)[i] - o.zp) * o.scale;
            return static_cast<const float*>(o.data)[i];
        };
        for (int a = 0; a < 3; ++a) for (int y = 0; y < gh; ++y) for (int x = 0; x < gw; ++x) {
            const float bx = sigmoid(at(a, 0, y, x)), by = sigmoid(at(a, 1, y, x));
            const float bw = sigmoid(at(a, 2, y, x)), bh = sigmoid(at(a, 3, y, x));
            const float obj_logit = at(a, 4, y, x);
            if (obj_logit <= kObjLogitThreshold) continue;
            float best_logit = -1e30f; int cls = 0;
            for (int c = 0; c < kClasses; ++c) {
                const float p = at(a, 5 + c, y, x);
                if (p > best_logit) { best_logit = p; cls = c; }
            }
            const float score = sigmoid(obj_logit) * sigmoid(best_logit);
            if (score <= kObjThreshold) continue;
            const float cx = (bx * 2.0f - 0.5f + x) * stride;
            const float cy = (by * 2.0f - 0.5f + y) * stride;
            const float ww = std::pow(bw * 2.0f, 2.0f) * kAnchors[kMasks[b][a]][0];
            const float hh = std::pow(bh * 2.0f, 2.0f) * kAnchors[kMasks[b][a]][1];
            raw.push_back({cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2, score, cls});
        }
    }
    auto out = nms(std::move(raw));
    for (auto& d : out) {
        d.x1 = std::max(0.0f, std::min((d.x1 - lb.dx) / lb.scale, static_cast<float>(original.width - 1)));
        d.y1 = std::max(0.0f, std::min((d.y1 - lb.dy) / lb.scale, static_cast<float>(original.height - 1)));
        d.x2 = std::max(0.0f, std::min((d.x2 - lb.dx) / lb.scale, static_cast<float>(original.width - 1)));
        d.y2 = std::max(0.0f, std::min((d.y2 - lb.dy) / lb.scale, static_cast<float>(original.height - 1)));
    }
    return out;
}

void check(int ret, const std::string& what) { if (ret != RKNN_SUCC) throw std::runtime_error(what + " failed: " + std::to_string(ret)); }

} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "usage:\n  " << argv[0] << " model.rknn image.jpg [input_rgb.raw]\n"
                  << "  " << argv[0] << " model.rknn --video video.mp4 [max_frames]\n";
        return 2;
    }
    try {
        const std::string model_path = argv[1];
        auto model = load_file(model_path);
        rknn_context ctx = 0; check(rknn_init(&ctx, model.data(), model.size(), 0, nullptr), "rknn_init");
        rknn_input_output_num io{}; check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "query io num");
        if (io.n_input != 1 || io.n_output != 3) throw std::runtime_error("expected 1 input and 3 outputs");
        rknn_tensor_attr in_attr{}; in_attr.index = 0; check(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr)), "query input");
        std::vector<rknn_tensor_attr> attrs(io.n_output); std::vector<std::pair<int, int>> grids;
        for (uint32_t i = 0; i < io.n_output; ++i) {
            attrs[i].index = i; check(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attrs[i], sizeof(attrs[i])), "query output");
            if (attrs[i].n_dims < 4 || attrs[i].dims[1] != 255) throw std::runtime_error("output is not NCHW [1,255,H,W]");
            grids.emplace_back(static_cast<int>(attrs[i].dims[2]), static_cast<int>(attrs[i].dims[3]));
            std::cout << "output" << i << ": [" << attrs[i].dims[0] << "," << attrs[i].dims[1] << "," << attrs[i].dims[2] << "," << attrs[i].dims[3] << "]\n";
        }
        auto infer_one = [&](const cv::Mat& bgr, const std::string& raw_path) {
            LetterboxInfo lb{}; cv::Mat rgb = letterbox_rgb_rga(bgr, lb);
            if (!raw_path.empty()) {
                std::ofstream raw(raw_path, std::ios::binary);
                raw.write(reinterpret_cast<const char*>(rgb.data), static_cast<std::streamsize>(rgb.total() * rgb.elemSize()));
            }
            rknn_input input{}; input.index = 0; input.type = RKNN_TENSOR_UINT8; input.fmt = RKNN_TENSOR_NHWC;
            input.size = static_cast<uint32_t>(rgb.total() * rgb.elemSize()); input.buf = rgb.data; input.pass_through = 0;
            check(rknn_inputs_set(ctx, 1, &input), "inputs_set"); check(rknn_run(ctx, nullptr), "run");
            std::vector<rknn_output> outputs(io.n_output); for (auto& o : outputs) o.want_float = 0;
            check(rknn_outputs_get(ctx, io.n_output, outputs.data(), nullptr), "outputs_get");
            std::vector<TensorView> out_data(io.n_output);
            for (uint32_t i = 0; i < io.n_output; ++i) {
                out_data[i] = {outputs[i].buf, attrs[i].type, attrs[i].zp, attrs[i].scale};
            }
            auto detections = decode_v5(out_data, grids, lb, bgr.size());
            check(rknn_outputs_release(ctx, io.n_output, outputs.data()), "outputs_release");
            return detections;
        };

        if (std::string(argv[2]) == "--video") {
            if (argc < 4) throw std::runtime_error("--video needs a video path");
            cv::VideoCapture cap(argv[3]);
            if (!cap.isOpened()) throw std::runtime_error("cannot open video: " + std::string(argv[3]));
            const int max_frames = argc == 5 ? std::stoi(argv[4]) : -1;
            cv::Mat frame; int frames = 0; size_t total_dets = 0;
            const auto start = std::chrono::steady_clock::now();
            while ((max_frames < 0 || frames < max_frames) && cap.read(frame)) {
                total_dets += infer_one(frame, {}).size();
                ++frames;
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << std::fixed << std::setprecision(3)
                      << "video_frames=" << frames << " elapsed_s=" << seconds
                      << " fps=" << (frames / std::max(seconds, 1e-9))
                      << " avg_detections=" << (frames ? static_cast<double>(total_dets) / frames : 0.0) << "\n";
        } else {
            const cv::Mat bgr = cv::imread(argv[2]);
            if (bgr.empty()) throw std::runtime_error("cannot read image: " + std::string(argv[2]));
            const std::string raw_path = argc == 4 ? argv[3] : "";
            auto detections = infer_one(bgr, raw_path);
            std::cout << std::fixed << std::setprecision(6) << "detections=" << detections.size() << "\n";
            for (const auto& d : detections) std::cout << d.x1 << ' ' << d.y1 << ' ' << d.x2 << ' ' << d.y2 << ' ' << d.score << ' ' << d.cls << '\n';
        }
        rknn_destroy(ctx);
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
    return 0;
}
