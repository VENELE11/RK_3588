#define main single_entry_point
#include "main.cpp"
#undef main

namespace {
float qvalue(const TensorView& t, size_t i) {
    if (t.type == RKNN_TENSOR_INT8) return (static_cast<const int8_t*>(t.data)[i] - t.zp) * t.scale;
    if (t.type == RKNN_TENSOR_UINT8) return (static_cast<const uint8_t*>(t.data)[i] - t.zp) * t.scale;
    return static_cast<const float*>(t.data)[i];
}
float softmax16(const TensorView& t, size_t base, int channel) {
    float mx = -1e30f; for (int k = 0; k < 16; ++k) mx = std::max(mx, qvalue(t, base + k * channel));
    float sum = 0.0f, out = 0.0f;
    for (int k = 0; k < 16; ++k) { const float e = std::exp(qvalue(t, base + k * channel) - mx); sum += e; out += k * e; }
    return out / std::max(sum, 1e-9f);
}
std::vector<Detection> decode_v8(const std::vector<TensorView>& v, const std::vector<rknn_tensor_attr>& a, const LetterboxInfo& lb, const cv::Size& original) {
    std::vector<Detection> raw;
    for (int branch = 0; branch < 3; ++branch) {
        const auto& box_attr = a[branch * 2];
        const int gh = box_attr.dims[2], gw = box_attr.dims[3], stride = 640 / gh;
        const size_t cells = static_cast<size_t>(gh) * gw;
        for (int y = 0; y < gh; ++y) for (int x = 0; x < gw; ++x) {
            const size_t cell = static_cast<size_t>(y) * gw + x;
            float best = 0.0f; int cls = 0;
            float best_logit = -1e30f;
            for (int c = 0; c < 80; ++c) { const float p = qvalue(v[branch * 2 + 1], c * cells + cell); if (p > best_logit) best_logit = p, cls = c; }
            best = 1.0f / (1.0f + std::exp(-best_logit));
            if (best <= kObjThreshold) continue;
            float d[4]{};
            for (int side = 0; side < 4; ++side) d[side] = softmax16(v[branch * 2], static_cast<size_t>(side) * 16 * cells + cell, cells);
            const float cx = (x + 0.5f) * stride, cy = (y + 0.5f) * stride;
            raw.push_back({cx - d[0] * stride, cy - d[1] * stride, cx + d[2] * stride, cy + d[3] * stride, best, cls});
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
std::vector<Detection> decode_v11(const TensorView& t, const rknn_tensor_attr& attr, const LetterboxInfo& lb, const cv::Size& original) {
    const int d1 = attr.dims[1], d2 = attr.dims[2]; const bool cn = d1 == 84; const int n = cn ? d2 : d1;
    std::vector<Detection> raw;
    auto at = [&](int row, int col) { return qvalue(t, cn ? static_cast<size_t>(col) * n + row : static_cast<size_t>(row) * 84 + col); };
    for (int i = 0; i < n; ++i) {
        float best = 0.0f; int cls = 0;
        float best_logit = -1e30f;
        for (int c = 0; c < 80; ++c) { const float p = at(i, 4 + c); if (p > best_logit) best_logit = p, cls = c; }
        best = 1.0f / (1.0f + std::exp(-best_logit));
        if (best <= kObjThreshold) continue;
        const float cx = at(i, 0), cy = at(i, 1), w = at(i, 2), h = at(i, 3);
        raw.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, cls});
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
std::vector<Detection> decode_v11_norm(const TensorView& t, const rknn_tensor_attr& attr, const LetterboxInfo& lb, const cv::Size& original) {
    const int d1 = attr.dims[1], d2 = attr.dims[2]; const bool cn = d1 == 84; const int n = cn ? d2 : d1;
    std::vector<Detection> raw;
    auto at = [&](int row, int col) { return qvalue(t, cn ? static_cast<size_t>(col) * n + row : static_cast<size_t>(row) * 84 + col); };
    for (int i = 0; i < n; ++i) {
        float best = 0.0f; int cls = 0;
        for (int c = 0; c < 80; ++c) { const float p = at(i, 4 + c); if (p > best) best = p, cls = c; }
        if (best <= kObjThreshold) continue;
        const float cx = at(i, 0) * 640.0f, cy = at(i, 1) * 640.0f;
        const float w = at(i, 2) * 640.0f, h = at(i, 3) * 640.0f;
        raw.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, cls});
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
}

int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: " << argv[0] << " v8|v11|v11_norm model.rknn image.jpg\n"; return 2; }
    try {
        const std::string type = argv[1]; auto model = load_file(argv[2]); cv::Mat bgr = cv::imread(argv[3]);
        if (bgr.empty()) throw std::runtime_error("cannot read image");
        rknn_context ctx = 0; check(rknn_init(&ctx, model.data(), model.size(), 0, nullptr), "rknn_init");
        rknn_input_output_num io{}; check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "query io");
        std::vector<rknn_tensor_attr> attrs(io.n_output); std::vector<TensorView> views(io.n_output);
        for (uint32_t i = 0; i < io.n_output; ++i) { attrs[i].index = i; check(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attrs[i], sizeof(attrs[i])), "query output"); std::cout << "output" << i << ": dims=" << attrs[i].n_dims << " [" << attrs[i].dims[0] << "," << attrs[i].dims[1] << "," << attrs[i].dims[2] << "," << attrs[i].dims[3] << "]\n"; }
        LetterboxInfo lb{}; cv::Mat rgb = letterbox_rgb_rga(bgr, lb);
        rknn_input in{}; in.index = 0; in.type = RKNN_TENSOR_UINT8; in.fmt = RKNN_TENSOR_NHWC; in.size = rgb.total() * rgb.elemSize(); in.buf = rgb.data;
        check(rknn_inputs_set(ctx, 1, &in), "inputs_set"); check(rknn_run(ctx, nullptr), "run");
        std::vector<rknn_output> outs(io.n_output); for (auto& o : outs) o.want_float = 0; check(rknn_outputs_get(ctx, io.n_output, outs.data(), nullptr), "outputs_get");
        for (uint32_t i = 0; i < io.n_output; ++i) views[i] = {outs[i].buf, attrs[i].type, attrs[i].zp, attrs[i].scale};
        std::vector<Detection> dets;
        if (type == "v8") dets = decode_v8(views, attrs, lb, bgr.size());
        else if (type == "v11" && io.n_output == 1) dets = decode_v11(views[0], attrs[0], lb, bgr.size());
        else if (type == "v11_norm" && io.n_output == 1) dets = decode_v11_norm(views[0], attrs[0], lb, bgr.size());
        else throw std::runtime_error("unsupported output layout for selected model");
        rknn_outputs_release(ctx, io.n_output, outs.data()); rknn_destroy(ctx);
        std::cout << "detections=" << dets.size() << "\n";
        for (const auto& d : dets) std::cout << d.x1 << ' ' << d.y1 << ' ' << d.x2 << ' ' << d.y2 << ' ' << d.score << ' ' << d.cls << '\n';
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
    return 0;
}
