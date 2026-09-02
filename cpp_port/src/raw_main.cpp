// Standalone YOLO11 raw-head runner.  The graph must expose three tensors
// [1, 4*reg_max+nc, H, W] (normally [1,144,H,W]).  DFL, sigmoid, decode and
// NMS intentionally stay in float32 here.
#define main legacy_main
#include "main.cpp"
#undef main

namespace {
std::vector<Detection> decode_yolo11_raw(const std::vector<TensorView>& views,
                                         const std::vector<rknn_tensor_attr>& attrs,
                                         const LetterboxInfo& lb,
                                         const cv::Size& original) {
    std::vector<Detection> raw;
    uint32_t order = 0;
    for (size_t branch = 0; branch < views.size(); ++branch) {
        const auto& a = attrs[branch];
        if (a.n_dims < 4 || a.dims[1] < 5 || a.dims[2] <= 0 || a.dims[3] <= 0)
            throw std::runtime_error("raw head must be NCHW [1,C,H,W]");
        const int c = a.dims[1], gh = a.dims[2], gw = a.dims[3];
        const int reg = 64;
        if (c <= reg || c - reg != kClasses)
            throw std::runtime_error("expected raw YOLO11 channels=144");
        const size_t cells = static_cast<size_t>(gh) * gw;
        auto at = [&](int channel, size_t cell) {
            return tensor_value(views[branch], static_cast<size_t>(channel) * cells + cell);
        };
        const int stride = kImageSize / gh;
        for (size_t cell = 0; cell < cells; ++cell) {
            int cls = 0;
            float best = at(reg, cell);
            for (int j = 1; j < kClasses; ++j) {
                const float v = at(reg + j, cell);
                if (v > best) { best = v; cls = j; }
            }
            const float score = sigmoid(best);
            if (score <= kObjThreshold) continue;
            const int y = static_cast<int>(cell / gw), x = static_cast<int>(cell % gw);
            float d[4]{};
            for (int side = 0; side < 4; ++side) {
                float max_logit = at(side * 16, cell), sum = 0.0f, weighted = 0.0f;
                for (int k = 1; k < 16; ++k) max_logit = std::max(max_logit, at(side * 16 + k, cell));
                for (int k = 0; k < 16; ++k) {
                    const float e = std::exp(std::max(-30.0f, std::min(30.0f,
                        at(side * 16 + k, cell) - max_logit)));
                    sum += e; weighted += k * e;
                }
                d[side] = weighted / std::max(sum, 1e-9f);
            }
            const float cx = (x + 0.5f) * stride, cy = (y + 0.5f) * stride;
            raw.push_back({cx - d[0] * stride, cy - d[1] * stride,
                           cx + d[2] * stride, cy + d[3] * stride,
                           score, cls, order++});
        }
    }
    keep_top_detections(raw);
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

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " raw_model.rknn image.jpg\n";
        return 2;
    }
    try {
        auto model = load_file(argv[1]);
        rknn_context ctx = 0; check(rknn_init(&ctx, model.data(), model.size(), 0, nullptr), "rknn_init");
        rknn_input_output_num io{}; check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "query io");
        if (io.n_input != 1 || io.n_output != 3) throw std::runtime_error("raw model needs 1 input and 3 outputs");
        rknn_tensor_attr in{}; in.index = 0; check(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in, sizeof(in)), "query input");
        std::cout << "input type=" << in.type << " scale=" << in.scale << " zp=" << in.zp << "\n";
        std::vector<rknn_tensor_attr> attrs(io.n_output);
        for (uint32_t i = 0; i < io.n_output; ++i) {
            attrs[i].index = i; check(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attrs[i], sizeof(attrs[i])), "query output");
            std::cout << "output" << i << " type=" << attrs[i].type << " scale=" << attrs[i].scale << " zp=" << attrs[i].zp
                      << " dims=[" << attrs[i].dims[0] << "," << attrs[i].dims[1] << "," << attrs[i].dims[2] << "," << attrs[i].dims[3] << "]\n";
        }
        RknnIoBuffers io_buffers(ctx, in, attrs); PreprocessBuffers prep;
        cv::Mat bgr = cv::imread(argv[2]); if (bgr.empty()) throw std::runtime_error("cannot read image");
        LetterboxInfo lb{}; cv::Mat rgb = io_buffers.input_frame(); prep.prepare(bgr, lb, false, rgb);
        io_buffers.sync_input(); check(rknn_run(ctx, nullptr), "run");
        auto dets = decode_yolo11_raw(io_buffers.views(), attrs, lb, bgr.size());
        std::cout << "detections=" << dets.size() << "\n";
        for (const auto& d : dets) std::cout << d.x1 << ' ' << d.y1 << ' ' << d.x2 << ' ' << d.y2 << ' ' << d.score << ' ' << d.cls << '\n';
        rknn_destroy(ctx); return 0;
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
}
