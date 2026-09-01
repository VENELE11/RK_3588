#pragma once

#include "rknn_api.h"

template <typename T>
struct RknnDtype{};

template <>
struct RknnDtype<float>{
    static constexpr rknn_tensor_type value = RKNN_TENSOR_FLOAT32;
};

template <>
struct RknnDtype<__fp16>{
    static constexpr rknn_tensor_type value = RKNN_TENSOR_FLOAT16;
};

template <>
struct RknnDtype<int8_t>{
    static constexpr rknn_tensor_type value = RKNN_TENSOR_INT8;
};

template <>
struct RknnDtype<uint8_t>{
    static constexpr rknn_tensor_type value = RKNN_TENSOR_UINT8;
};

template <>
struct RknnDtype<int16_t>{
    static constexpr rknn_tensor_type value = RKNN_TENSOR_INT16;
};

template <>
struct RknnDtype<uint16_t>{
    static constexpr rknn_tensor_type value = RKNN_TENSOR_UINT16;
};

template <>
struct RknnDtype<int32_t>{
    static constexpr rknn_tensor_type value = RKNN_TENSOR_INT32;
};

template <>
struct RknnDtype<uint32_t>{
    static constexpr rknn_tensor_type value = RKNN_TENSOR_UINT32;
};

template <>
struct RknnDtype<int64_t>{
    static constexpr rknn_tensor_type value = RKNN_TENSOR_INT64;
};

template <>
struct RknnDtype<bool>{
    static constexpr rknn_tensor_type value = RKNN_TENSOR_BOOL;
};
