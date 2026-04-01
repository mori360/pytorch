#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/core/ScalarType.h>
#include <cstdint>

namespace at::native {

using philox_uniform_fill_fn = void(*)(
    void* output, int64_t base, int64_t numel,
    uint64_t seed, uint64_t key_offset,
    double low, double high, at::ScalarType dtype);

using philox_normal_fill_fn = void(*)(
    void* output, int64_t base, int64_t numel,
    uint64_t seed, uint64_t key_offset,
    double mean, double stddev, at::ScalarType dtype);

DECLARE_DISPATCH(philox_uniform_fill_fn, philox_uniform_fill_stub);
DECLARE_DISPATCH(philox_normal_fill_fn, philox_normal_fill_stub);

} // namespace at::native
