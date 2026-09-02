#ifndef CPP_PORT_COMMON_INCLUDED
#define CPP_PORT_COMMON_INCLUDED

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include "im2d.h"
#include "rga.h"
#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

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
const int kCocoCategoryIds[kClasses] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90};

struct Detection { float x1, y1, x2, y2, score; int cls; uint32_t order = 0; };
struct LetterboxInfo { float scale; int dx, dy; };
struct TensorView { const void* data; rknn_tensor_type type; int32_t zp; float scale; };
using ProfileClock = std::chrono::steady_clock;

void check(int ret, const std::string& what);

struct StageTimes {
    double preprocess_ms = 0.0;
    double input_set_ms = 0.0;
    double inference_ms = 0.0;
    double output_get_ms = 0.0;
    double postprocess_ms = 0.0;
    double output_release_ms = 0.0;
    double total_ms = 0.0;
};

struct TimingStats {
    size_t count = 0;
    StageTimes sum;

    void add(const StageTimes& t) {
        ++count;
        sum.preprocess_ms += t.preprocess_ms;
        sum.input_set_ms += t.input_set_ms;
        sum.inference_ms += t.inference_ms;
        sum.output_get_ms += t.output_get_ms;
        sum.postprocess_ms += t.postprocess_ms;
        sum.output_release_ms += t.output_release_ms;
        sum.total_ms += t.total_ms;
    }

    void merge(const TimingStats& other) {
        count += other.count;
        sum.preprocess_ms += other.sum.preprocess_ms;
        sum.input_set_ms += other.sum.input_set_ms;
        sum.inference_ms += other.sum.inference_ms;
        sum.output_get_ms += other.sum.output_get_ms;
        sum.postprocess_ms += other.sum.postprocess_ms;
        sum.output_release_ms += other.sum.output_release_ms;
        sum.total_ms += other.sum.total_ms;
    }
};

struct RknnIoBuffers {
    rknn_context ctx = 0;
    rknn_tensor_attr input_attr{};
    std::vector<rknn_tensor_attr> output_attrs;
    rknn_tensor_mem* input_mem = nullptr;
    std::vector<rknn_tensor_mem*> output_mems;
    std::vector<TensorView> output_views;

    RknnIoBuffers(rknn_context context, rknn_tensor_attr input,
                  const std::vector<rknn_tensor_attr>& outputs)
        : ctx(context), input_attr(input), output_attrs(outputs), output_mems(outputs.size(), nullptr),
          output_views(outputs.size()) {
        input_attr.type = RKNN_TENSOR_UINT8;
        input_attr.fmt = RKNN_TENSOR_NHWC;
        input_attr.pass_through = 0;
        const uint32_t input_size = input_attr.size_with_stride ? input_attr.size_with_stride : input_attr.size;
        input_mem = rknn_create_mem(ctx, input_size);
        if (!input_mem) throw std::runtime_error("rknn_create_mem input failed");
        for (size_t i = 0; i < output_attrs.size(); ++i) {
            output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size);
            if (!output_mems[i]) throw std::runtime_error("rknn_create_mem output failed");
        }
        check(rknn_set_io_mem(ctx, input_mem, &input_attr), "set_io_mem input");
        for (size_t i = 0; i < output_attrs.size(); ++i)
            check(rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]), "set_io_mem output");
    }

    ~RknnIoBuffers() {
        for (auto* mem : output_mems) if (mem) rknn_destroy_mem(ctx, mem);
        if (input_mem) rknn_destroy_mem(ctx, input_mem);
    }

    cv::Mat input_frame() const {
        const int width = static_cast<int>(input_attr.dims[2]);
        const int height = static_cast<int>(input_attr.dims[1]);
        const int channels = static_cast<int>(input_attr.dims[3]);
        const int stride = input_attr.w_stride ? static_cast<int>(input_attr.w_stride) : width;
        const size_t row_bytes = static_cast<size_t>(stride) * channels;
        if (row_bytes * height > input_mem->size)
            throw std::runtime_error("zero-copy input buffer is too small");
        return cv::Mat(height, width, CV_8UC3, input_mem->virt_addr, row_bytes);
    }

    void sync_input() {
        check(rknn_mem_sync(ctx, input_mem, RKNN_MEMORY_SYNC_TO_DEVICE), "sync input");
    }

    const std::vector<TensorView>& views() {
        for (size_t i = 0; i < output_mems.size(); ++i) {
            check(rknn_mem_sync(ctx, output_mems[i], RKNN_MEMORY_SYNC_FROM_DEVICE), "sync output");
            output_views[i] = {output_mems[i]->virt_addr, output_attrs[i].type,
                               output_attrs[i].zp, output_attrs[i].scale};
        }
        return output_views;
    }
};

double elapsed_ms(ProfileClock::time_point begin, ProfileClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void print_timing_stats(const std::string& label, const TimingStats& stats) {
    if (stats.count == 0) return;
    const double n = static_cast<double>(stats.count);
    std::cout << std::fixed << std::setprecision(3)
              << "PROFILE " << label << " images=" << stats.count
              << " preprocess_ms=" << stats.sum.preprocess_ms / n
              << " input_set_ms=" << stats.sum.input_set_ms / n
              << " inference_ms=" << stats.sum.inference_ms / n
              << " output_get_ms=" << stats.sum.output_get_ms / n
              << " postprocess_ms=" << stats.sum.postprocess_ms / n
              << " output_release_ms=" << stats.sum.output_release_ms / n
              << " total_ms=" << stats.sum.total_ms / n
              << " fps=" << (1000.0 * n / stats.sum.total_ms) << '\n';
}

std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open image list: " + path);
    std::vector<std::string> lines;
    for (std::string line; std::getline(f, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

std::string join_path(const std::string& dir, const std::string& name) {
    return dir.empty() || dir.back() == '/' ? dir + name : dir + "/" + name;
}

long long image_id_from_name(const std::string& name) {
    const size_t slash = name.find_last_of("/\\");
    const size_t begin = slash == std::string::npos ? 0 : slash + 1;
    const size_t dot = name.find('.', begin);
    return std::stoll(name.substr(begin, dot == std::string::npos ? dot : dot - begin));
}

void write_coco_json(const std::string& path,
                     const std::vector<std::pair<long long, std::vector<Detection>>>& results) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write predictions: " + path);
    out << std::setprecision(17) << "{\"predictions\":[";
    bool first = true;
    for (const auto& item : results) {
        for (const auto& d : item.second) {
            if (!first) out << ',';
            first = false;
            out << "{\"image_id\":" << item.first
                << ",\"category_id\":" << kCocoCategoryIds[d.cls]
                << ",\"bbox\":[" << d.x1 << ',' << d.y1 << ','
                << std::max(0.0f, d.x2 - d.x1) << ',' << std::max(0.0f, d.y2 - d.y1)
                << "],\"score\":" << d.score << '}';
        }
    }
    out << "]}\n";
}

float sigmoid(float x) {
    x = std::max(-50.0f, std::min(50.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}

float fp16_to_float(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t bits;
    if (exp == 0) { if (mant == 0) bits = sign; else { uint32_t m = mant; int e = -14; while ((m & 0x400u) == 0) { m <<= 1; --e; } m &= 0x3FFu; bits = sign | (static_cast<uint32_t>(e + 127) << 23) | (m << 13); } }
    else if (exp == 31) bits = sign | 0x7F800000u | (mant << 13);
    else bits = sign | ((exp + 112u) << 23) | (mant << 13);
    float out; std::memcpy(&out, &bits, sizeof(out)); return out;
}
float tensor_value(const TensorView& t, size_t i) {
    if (t.type == RKNN_TENSOR_FLOAT16)
        return fp16_to_float(static_cast<const uint16_t*>(t.data)[i]);
    if (t.type == RKNN_TENSOR_INT8)
        return (static_cast<const int8_t*>(t.data)[i] - t.zp) * t.scale;
    if (t.type == RKNN_TENSOR_UINT8)
        return (static_cast<const uint8_t*>(t.data)[i] - t.zp) * t.scale;
    return static_cast<const float*>(t.data)[i];
}

template <typename T>
bool score_order_better(const T& a, const T& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.order < b.order;
}

template <typename T>
void keep_top_detections(std::vector<T>& values) {
    const auto better = [](const T& a, const T& b) { return score_order_better(a, b); };
    if (values.size() > kMaxDetections) {
        std::partial_sort(values.begin(), values.begin() + kMaxDetections, values.end(), better);
        values.resize(kMaxDetections);
    } else {
        std::sort(values.begin(), values.end(), better);
    }
}

std::vector<uint8_t> load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open model: " + path);
    const auto n = f.tellg(); f.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(data.data()), n);
    return data;
}

void letterbox_geometry(const cv::Mat& bgr, LetterboxInfo& info, int& nw, int& nh) {
    const int h = bgr.rows, w = bgr.cols;
    info.scale = std::min(static_cast<float>(kImageSize) / h,
                          static_cast<float>(kImageSize) / w);
    nh = static_cast<int>(std::round(h * info.scale));
    nw = static_cast<int>(std::round(w * info.scale));
    info.dy = (kImageSize - nh) / 2;
    info.dx = (kImageSize - nw) / 2;
}

struct PreprocessBuffers {
    cv::Mat bgr_canvas;
    cv::Mat rgb_source;

    void prepare(const cv::Mat& bgr, LetterboxInfo& info, bool use_rga, cv::Mat& output) {
        int nw = 0, nh = 0;
        letterbox_geometry(bgr, info, nw, nh);
        const cv::Rect roi(info.dx, info.dy, nw, nh);
        if (!use_rga) {
            bgr_canvas.create(kImageSize, kImageSize, CV_8UC3);
            bgr_canvas.setTo(cv::Scalar(114, 114, 114));
            cv::resize(bgr, bgr_canvas(roi), roi.size());
            cv::cvtColor(bgr_canvas, output, cv::COLOR_BGR2RGB);
            return;
        }

        output.setTo(cv::Scalar(114, 114, 114));
        rgb_source.create(bgr.rows, bgr.cols, CV_8UC3);
        cv::cvtColor(bgr, rgb_source, cv::COLOR_BGR2RGB);
        const int w = bgr.cols, h = bgr.rows;
        rga_buffer_t src = wrapbuffer_virtualaddr(rgb_source.data, w, h, w, h, RK_FORMAT_RGB_888);
        auto* dst_ptr = output.ptr<uint8_t>(info.dy) + static_cast<size_t>(info.dx) * 3;
        const int dst_stride = static_cast<int>(output.step / 3);
        rga_buffer_t dst = wrapbuffer_virtualaddr(dst_ptr, nw, nh, dst_stride, output.rows, RK_FORMAT_RGB_888);
        im_rect sr = {0, 0, w, h}; im_rect dr = {0, 0, nw, nh};
        const int checked = imcheck(src, dst, sr, dr);
        if (checked == IM_STATUS_NOERROR && imresize(src, dst) == IM_STATUS_NOERROR) return;

        cv::resize(rgb_source, output(roi), roi.size());
    }
};

std::vector<Detection> nms(std::vector<Detection> dets, bool already_top_sorted = false) {
    if (!already_top_sorted) keep_top_detections(dets);
    std::array<std::vector<Detection>, kClasses> by_class;
    for (const auto& d : dets) by_class[d.cls].push_back(d);
    std::vector<Detection> out;
    out.reserve(dets.size());
    for (auto& items : by_class) {
        std::vector<uint8_t> removed(items.size(), 0);
        std::vector<float> areas(items.size());
        for (size_t i = 0; i < items.size(); ++i) {
            areas[i] = std::max(0.0f, items[i].x2 - items[i].x1 + 1.0f) *
                       std::max(0.0f, items[i].y2 - items[i].y1 + 1.0f);
        }
        for (size_t i = 0; i < items.size(); ++i) {
            if (removed[i]) continue;
            out.push_back(items[i]);
            for (size_t j = i + 1; j < items.size(); ++j) {
                if (removed[j]) continue;
                const float x1 = std::max(items[i].x1, items[j].x1);
                const float y1 = std::max(items[i].y1, items[j].y1);
                const float x2 = std::min(items[i].x2, items[j].x2);
                const float y2 = std::min(items[i].y2, items[j].y2);
                const float inter = std::max(0.0f, x2 - x1 + 1.0f) *
                                    std::max(0.0f, y2 - y1 + 1.0f);
                const float overlap = inter / (areas[i] + areas[j] - inter + 1e-9f);
                if (overlap > kNmsThreshold) removed[j] = 1;
            }
        }
    }
    return out;
}

struct V5Candidate {
    int branch, anchor, y, x;
    float score;
    int cls;
    uint32_t order;
};

#if defined(__aarch64__) || defined(__ARM_NEON)
[[maybe_unused]] void scan_argmax_neon(const int8_t* data, size_t cells, size_t channel_stride,
                                       std::vector<int>& best_q, std::vector<uint8_t>& best_cls) {
    size_t i = 0;
    for (; i + 16 <= cells; i += 16) {
        int8x16_t best = vld1q_s8(data + i);
        uint8x16_t cls = vdupq_n_u8(0);
        for (int c = 1; c < kClasses; ++c) {
            const int8x16_t value = vld1q_s8(data + static_cast<size_t>(c) * channel_stride + i);
            const uint8x16_t mask = vcgtq_s8(value, best);
            best = vbslq_s8(mask, value, best);
            cls = vbslq_u8(mask, vdupq_n_u8(static_cast<uint8_t>(c)), cls);
        }
        alignas(16) int8_t q[16]; alignas(16) uint8_t c[16];
        vst1q_s8(q, best); vst1q_u8(c, cls);
        for (int j = 0; j < 16; ++j) { best_q[i + j] = q[j]; best_cls[i + j] = c[j]; }
    }
    for (; i < cells; ++i) {
        int q_best = static_cast<int>(data[i]); int cls_best = 0;
        for (int c = 1; c < kClasses; ++c) {
            const int q = static_cast<int>(data[static_cast<size_t>(c) * channel_stride + i]);
            if (q > q_best) { q_best = q; cls_best = c; }
        }
        best_q[i] = q_best; best_cls[i] = static_cast<uint8_t>(cls_best);
    }
}

[[maybe_unused]] void scan_argmax_neon(const uint8_t* data, size_t cells, size_t channel_stride,
                                       std::vector<int>& best_q, std::vector<uint8_t>& best_cls) {
    size_t i = 0;
    for (; i + 16 <= cells; i += 16) {
        uint8x16_t best = vld1q_u8(data + i);
        uint8x16_t cls = vdupq_n_u8(0);
        for (int c = 1; c < kClasses; ++c) {
            const uint8x16_t value = vld1q_u8(data + static_cast<size_t>(c) * channel_stride + i);
            const uint8x16_t mask = vcgtq_u8(value, best);
            best = vbslq_u8(mask, value, best);
            cls = vbslq_u8(mask, vdupq_n_u8(static_cast<uint8_t>(c)), cls);
        }
        alignas(16) uint8_t q[16]; alignas(16) uint8_t c[16];
        vst1q_u8(q, best); vst1q_u8(c, cls);
        for (int j = 0; j < 16; ++j) { best_q[i + j] = q[j]; best_cls[i + j] = c[j]; }
    }
    for (; i < cells; ++i) {
        int q_best = static_cast<int>(data[i]); int cls_best = 0;
        for (int c = 1; c < kClasses; ++c) {
            const int q = static_cast<int>(data[static_cast<size_t>(c) * channel_stride + i]);
            if (q > q_best) { q_best = q; cls_best = c; }
        }
        best_q[i] = q_best; best_cls[i] = static_cast<uint8_t>(cls_best);
    }
}
#endif

template <typename Q>
void collect_v5_quantized(const TensorView& tensor, int branch, int gh, int gw,
                          std::vector<V5Candidate>& candidates, uint32_t& candidate_order) {
    const Q* data = static_cast<const Q*>(tensor.data);
    const size_t cells = static_cast<size_t>(gh) * gw;
    for (int a = 0; a < 3; ++a) for (int y = 0; y < gh; ++y) for (int x = 0; x < gw; ++x) {
        const size_t cell = static_cast<size_t>(y) * gw + x;
        const int obj_q = static_cast<int>(data[(a * 85 + 4) * cells + cell]);
        const float obj_logit = (obj_q - tensor.zp) * tensor.scale;
        if (obj_logit <= kObjLogitThreshold) continue;
        int cls = 0;
        int best_q = static_cast<int>(data[(a * 85 + 5) * cells + cell]);
        for (int c = 1; c < kClasses; ++c) {
            const int q = static_cast<int>(data[(a * 85 + 5 + c) * cells + cell]);
            if (q > best_q) { best_q = q; cls = c; }
        }
        const float best_logit = (best_q - tensor.zp) * tensor.scale;
        const float score = sigmoid(obj_logit) * sigmoid(best_logit);
        if (score > kObjThreshold)
            candidates.push_back({branch, a, y, x, score, cls, candidate_order++});
    }
}

void collect_v5_float(const TensorView& tensor, int branch, int gh, int gw,
                      std::vector<V5Candidate>& candidates, uint32_t& candidate_order) {
    const float* data = static_cast<const float*>(tensor.data);
    const size_t cells = static_cast<size_t>(gh) * gw;
    for (int a = 0; a < 3; ++a) for (int y = 0; y < gh; ++y) for (int x = 0; x < gw; ++x) {
        const size_t cell = static_cast<size_t>(y) * gw + x;
        const float obj_logit = data[(a * 85 + 4) * cells + cell];
        if (obj_logit <= kObjLogitThreshold) continue;
        int cls = 0;
        float best_logit = data[(a * 85 + 5) * cells + cell];
        for (int c = 1; c < kClasses; ++c) {
            const float value = data[(a * 85 + 5 + c) * cells + cell];
            if (value > best_logit) { best_logit = value; cls = c; }
        }
        const float score = sigmoid(obj_logit) * sigmoid(best_logit);
        if (score > kObjThreshold)
            candidates.push_back({branch, a, y, x, score, cls, candidate_order++});
    }
}

std::vector<Detection> decode_v5(const std::vector<TensorView>& outs,
                                 const std::vector<std::pair<int, int>>& grids,
                                 const LetterboxInfo& lb, const cv::Size& original) {
    std::vector<V5Candidate> candidates;
    candidates.reserve(2048);
    uint32_t candidate_order = 0;
    for (int b = 0; b < 3; ++b) {
        const int gh = grids[b].first, gw = grids[b].second;
        if (outs[b].type == RKNN_TENSOR_INT8)
            collect_v5_quantized<int8_t>(outs[b], b, gh, gw, candidates, candidate_order);
        else if (outs[b].type == RKNN_TENSOR_UINT8)
            collect_v5_quantized<uint8_t>(outs[b], b, gh, gw, candidates, candidate_order);
        else
            collect_v5_float(outs[b], b, gh, gw, candidates, candidate_order);
    }
    keep_top_detections(candidates);
    std::vector<Detection> raw;
    raw.reserve(candidates.size());
    for (const auto& c : candidates) {
        const int gh = grids[c.branch].first, gw = grids[c.branch].second;
        const int stride = kImageSize / gh;
        auto at = [&](int channel) {
            const size_t i = ((c.anchor * 85 + channel) * gh + c.y) * gw + c.x;
            return tensor_value(outs[c.branch], i);
        };
        const float bx = sigmoid(at(0)), by = sigmoid(at(1));
        const float bw = sigmoid(at(2)), bh = sigmoid(at(3));
        const float cx = (bx * 2.0f - 0.5f + c.x) * stride;
        const float cy = (by * 2.0f - 0.5f + c.y) * stride;
        const float ww = std::pow(bw * 2.0f, 2.0f) * kAnchors[kMasks[c.branch][c.anchor]][0];
        const float hh = std::pow(bh * 2.0f, 2.0f) * kAnchors[kMasks[c.branch][c.anchor]][1];
        raw.push_back({cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2,
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

void check(int ret, const std::string& what) { if (ret != RKNN_SUCC) throw std::runtime_error(what + " failed: " + std::to_string(ret)); }

} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "usage:\n  " << argv[0] << " model.rknn image.jpg [input_rgb.raw]\n"
                  << "  " << argv[0] << " model.rknn --video video.mp4 [max_frames]\n"
                  << "  " << argv[0] << " model.rknn --eval images_dir image_list.txt out.json\n";
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
        auto io_buffers = std::make_unique<RknnIoBuffers>(ctx, in_attr, attrs);
        PreprocessBuffers preprocess;
        auto infer_one = [&](const cv::Mat& bgr, const std::string& raw_path, bool use_rga,
                             TimingStats* stats) {
            StageTimes timing;
            const auto total_begin = ProfileClock::now();
            const auto preprocess_begin = total_begin;
            LetterboxInfo lb{};
            cv::Mat rgb = io_buffers->input_frame();
            preprocess.prepare(bgr, lb, use_rga, rgb);
            if (!raw_path.empty()) {
                std::ofstream raw(raw_path, std::ios::binary);
                for (int y = 0; y < rgb.rows; ++y)
                    raw.write(reinterpret_cast<const char*>(rgb.ptr<uint8_t>(y)),
                              static_cast<std::streamsize>(rgb.cols * rgb.elemSize()));
            }
            const auto preprocess_end = ProfileClock::now();
            io_buffers->sync_input();
            const auto input_set_end = ProfileClock::now();
            check(rknn_run(ctx, nullptr), "run");
            const auto inference_end = ProfileClock::now();
            const auto& out_data = io_buffers->views();
            const auto output_get_end = ProfileClock::now();
            auto detections = decode_v5(out_data, grids, lb, bgr.size());
            const auto postprocess_end = ProfileClock::now();
            const auto release_end = ProfileClock::now();
            if (stats) {
                timing.preprocess_ms = elapsed_ms(preprocess_begin, preprocess_end);
                timing.input_set_ms = elapsed_ms(preprocess_end, input_set_end);
                timing.inference_ms = elapsed_ms(input_set_end, inference_end);
                timing.output_get_ms = elapsed_ms(inference_end, output_get_end);
                timing.postprocess_ms = elapsed_ms(output_get_end, postprocess_end);
                timing.output_release_ms = elapsed_ms(postprocess_end, release_end);
                timing.total_ms = elapsed_ms(total_begin, release_end);
                stats->add(timing);
            }
            return detections;
        };

        if (std::string(argv[2]) == "--eval") {
            if (argc != 6) throw std::runtime_error("--eval needs images_dir image_list.txt out.json");
            const auto names = read_lines(argv[4]);
            std::vector<std::pair<long long, std::vector<Detection>>> results;
            results.reserve(names.size());
            size_t total_dets = 0; TimingStats timing_stats;
            for (size_t i = 0; i < names.size(); ++i) {
                const cv::Mat bgr = cv::imread(join_path(argv[3], names[i]));
                if (bgr.empty()) { std::cerr << "WARN missing image: " << names[i] << '\n'; continue; }
                auto detections = infer_one(bgr, {}, false, &timing_stats);
                total_dets += detections.size();
                results.emplace_back(image_id_from_name(names[i]), std::move(detections));
                if ((i + 1) % 10 == 0) std::cout << "processed " << (i + 1) << '/' << names.size()
                                                  << ", total dets=" << total_dets << '\n';
            }
            write_coco_json(argv[5], results);
            print_timing_stats("v5_eval", timing_stats);
            std::cout << "DONE: " << results.size() << " images -> " << total_dets
                      << " predictions -> " << argv[5] << '\n';
        } else if (std::string(argv[2]) == "--video") {
            if (argc < 4) throw std::runtime_error("--video needs a video path");
            cv::VideoCapture cap(argv[3]);
            if (!cap.isOpened()) throw std::runtime_error("cannot open video: " + std::string(argv[3]));
            const int max_frames = argc == 5 ? std::stoi(argv[4]) : -1;
            cv::Mat frame; int frames = 0; size_t total_dets = 0; TimingStats timing_stats;
            const auto start = std::chrono::steady_clock::now();
            while ((max_frames < 0 || frames < max_frames) && cap.read(frame)) {
                total_dets += infer_one(frame, {}, true, &timing_stats).size();
                ++frames;
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << std::fixed << std::setprecision(3)
                      << "video_frames=" << frames << " elapsed_s=" << seconds
                      << " fps=" << (frames / std::max(seconds, 1e-9))
                      << " avg_detections=" << (frames ? static_cast<double>(total_dets) / frames : 0.0) << "\n";
            print_timing_stats("v5_video", timing_stats);
        } else {
            const cv::Mat bgr = cv::imread(argv[2]);
            if (bgr.empty()) throw std::runtime_error("cannot read image: " + std::string(argv[2]));
            const std::string raw_path = argc == 4 ? argv[3] : "";
            TimingStats timing_stats;
            auto detections = infer_one(bgr, raw_path, true, &timing_stats);
            print_timing_stats("v5_single", timing_stats);
            std::cout << std::fixed << std::setprecision(6) << "detections=" << detections.size() << "\n";
            for (const auto& d : detections) std::cout << d.x1 << ' ' << d.y1 << ' ' << d.x2 << ' ' << d.y2 << ' ' << d.score << ' ' << d.cls << '\n';
        }
        io_buffers.reset();
        rknn_destroy(ctx);
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
    return 0;
}

#endif // CPP_PORT_COMMON_INCLUDED
