// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>

namespace ov::reference {

/// Clamp \p a to [0, 255]. For integral types the value is rounded first.
template <typename T, typename U>
T clip(U a) {
    if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(std::min(std::max(std::round(a), static_cast<U>(0)), static_cast<U>(255)));
    } else {
        return static_cast<T>(std::min(std::max(a, static_cast<U>(0)), static_cast<U>(255)));
    }
}

/// Cast \p a to T, rounding first for integral types.
template <typename T, typename U>
T round_cast(U a) {
    if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(std::round(a));
    } else {
        return static_cast<T>(a);
    }
}

/// Convert a single YUV pixel (BT.601 limited range) to (R, G, B).
template <typename T, typename U = float>
std::tuple<T, T, T> yuv_pixel_to_rgb(U y_val, U u_val, U v_val) {
    const auto c = y_val - static_cast<U>(16);
    const auto d = u_val - static_cast<U>(128);
    const auto e = v_val - static_cast<U>(128);
    return {clip<T>(static_cast<U>(1.164) * c + static_cast<U>(1.596) * e),
            clip<T>(static_cast<U>(1.164) * c - static_cast<U>(0.391) * d - static_cast<U>(0.813) * e),
            clip<T>(static_cast<U>(1.164) * c + static_cast<U>(2.018) * d)};
}

/// Convert a single RGB pixel to (Y, U, V).
template <typename T>
std::tuple<T, T, T> rgb_pixel_to_yuv(T r_val, T g_val, T b_val) {
    const double r = static_cast<double>(r_val);
    const double g = static_cast<double>(g_val);
    const double b = static_cast<double>(b_val);
    return {clip<T>(0.257 * r + 0.504 * g + 0.098 * b + 16.0),
            clip<T>(-0.148 * r - 0.291 * g + 0.439 * b + 128.0),
            clip<T>(0.439 * r - 0.368 * g - 0.071 * b + 128.0)};
}

}  // namespace ov::reference
