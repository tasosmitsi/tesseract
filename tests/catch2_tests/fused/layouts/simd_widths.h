#pragma once

#include "config.h"

// SIMD widths, in elements, for padding policies built with an explicit width,
// so a test checks the same padding and strides on every machine.

enum SimdWidth : my_size_t
{
    SCALAR = 1,
    SSE_DOUBLE = 2,
    SSE_FLOAT = 4,
    AVX_DOUBLE = 4,
    AVX_FLOAT = 8,
    AVX512_DOUBLE = 8,
    AVX512_FLOAT = 16
};

template <my_size_t SW>
struct SimdWidthTag
{
    static constexpr my_size_t value = SW;
};

using ScalarWidth = SimdWidthTag<SCALAR>;
using SSEFloatWidth = SimdWidthTag<SSE_FLOAT>;
using SSEDoubleWidth = SimdWidthTag<SSE_DOUBLE>;
using AVXFloatWidth = SimdWidthTag<AVX_FLOAT>;
using AVXDoubleWidth = SimdWidthTag<AVX_DOUBLE>;
using AVX512FloatWidth = SimdWidthTag<AVX512_FLOAT>;
using AVX512DoubleWidth = SimdWidthTag<AVX512_DOUBLE>;
