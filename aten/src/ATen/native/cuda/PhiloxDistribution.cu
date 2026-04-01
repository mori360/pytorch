#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/core/TransformationHelper.h>
#include <curand_kernel.h>
#include <type_traits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_normal_native.h>
#include <ATen/ops/_philox_uniform_native.h>
#endif

namespace at::native {

namespace {

// Philox outputs per scalar element: double consumes 2 uint32 outputs, others 1.
template <typename scalar_t>
constexpr int OPV = std::is_same_v<scalar_t, double> ? 2 : 1;

// Elements produced per curand4 call: 4 for float/half/bfloat16, 2 for double.
template <typename scalar_t>
constexpr int EPC = 4 / OPV<scalar_t>;

// Generate elements in [elem_start, elem_end) from a Philox stream defined
// by (seed, key_offset). Handles Box-Muller alignment (controlled by needs_alignment)
// and 64-bit offset wrapping.
template <typename scalar_t, bool needs_alignment, typename dist_t, typename transform_t>
__device__ void generate_range(
    scalar_t* out, int64_t elem_start, int64_t elem_end,
    uint64_t seed, uint64_t key_offset,
    const dist_t& dist_func, const transform_t& transform_func) {

  unsigned long long raw_offset = key_offset +
      static_cast<unsigned long long>(elem_start) * OPV<scalar_t>;
  unsigned long long philox_offset = raw_offset;
  int skip = 0;

  if constexpr (needs_alignment) {
    // Round down to a 4-output boundary so Box-Muller always pairs
    // the same absolute Philox stream positions.
    int misalign = static_cast<int>(raw_offset & 3);
    if (misalign > 0 && (misalign % OPV<scalar_t>) == 0) {
      skip = misalign / OPV<scalar_t>;
      philox_offset = raw_offset - misalign;
    }
  }

  // Detect 64-bit offset wrap within this range.
  auto range_outputs =
      static_cast<unsigned long long>(elem_end - elem_start) * OPV<scalar_t>;
  bool wraps =
      raw_offset != 0 && (raw_offset + range_outputs < raw_offset);
  int64_t wrap_at = wraps
      ? elem_start + static_cast<int64_t>((0ULL - raw_offset) / OPV<scalar_t>)
      : elem_end;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, /*subsequence=*/0, /*offset=*/philox_offset, &state);

  int64_t gen_end = min(wrap_at, elem_end);
  int64_t elem = elem_start;

  if constexpr (needs_alignment) {
    if (skip > 0 && elem < gen_end) {
      auto rand = dist_func(&state);
      for (int j = skip; j < EPC<scalar_t> && elem < gen_end; j++, elem++) {
        out[elem] = transform_func((&rand.x)[j]);
      }
    }
  }

  for (; elem < gen_end; elem += EPC<scalar_t>) {
    auto rand = dist_func(&state);
    #pragma unroll
    for (int j = 0; j < EPC<scalar_t> && elem + j < gen_end; j++) {
      out[elem + j] = transform_func((&rand.x)[j]);
    }
  }

  // Handle 64-bit offset wrap: reinitialize at offset 0 and continue.
  if (wraps) {
    curand_init(seed, /*subsequence=*/0, /*offset=*/0ULL, &state);
    for (elem = wrap_at; elem < elem_end; elem += EPC<scalar_t>) {
      auto rand = dist_func(&state);
      #pragma unroll
      for (int j = 0; j < EPC<scalar_t> && elem + j < elem_end; j++) {
        out[elem + j] = transform_func((&rand.x)[j]);
      }
    }
  }
}

// Single-key kernel. Uses a vectorized grid-stride loop when the Philox
// offset is 4-aligned (the common case). Falls back to contiguous per-thread
// generation for Box-Muller misalignment or 64-bit offset wrapping.
template <typename scalar_t, bool needs_alignment, typename dist_t, typename transform_t>
C10_LAUNCH_BOUNDS_2(256, 4)
__global__ void philox_single_key_kernel(
    scalar_t* __restrict__ output,
    const uint64_t* __restrict__ key,
    int64_t numel,
    dist_t dist_func,
    transform_t transform_func) {

  uint64_t seed = key[0];
  uint64_t key_offset = key[1];

  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  // Fall back to contiguous per-thread generation when the vectorized
  // path can't be used due to Box-Muller misalignment or 64-bit offset wrap.
  bool misaligned = needs_alignment && (key_offset & 3) != 0;
  bool could_wrap = key_offset != 0 &&
      (key_offset + static_cast<unsigned long long>(numel) * OPV<scalar_t> < key_offset);
  if (misaligned || could_wrap) {
    int64_t per_thread =
        ((numel + total_threads - 1) / total_threads + EPC<scalar_t> - 1) / EPC<scalar_t> * EPC<scalar_t>;
    int64_t start = tid * per_thread;
    if (start < numel) {
      generate_range<scalar_t, needs_alignment>(
          output, start, min(start + per_thread, numel),
          seed, key_offset, dist_func, transform_func);
    }
    return;
  }

  // Vectorized grid-stride loop usable when Philox offset is 4-aligned
  // so we're guaranteed to get correctly paired Box-Muller normals.
  int64_t vec_end = numel / EPC<scalar_t> * EPC<scalar_t>;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, /*subsequence=*/0,
              key_offset + static_cast<unsigned long long>(tid) * EPC<scalar_t> * OPV<scalar_t>,
              &state);
  for (int64_t pos = tid * EPC<scalar_t>; pos < vec_end;
       pos += total_threads * EPC<scalar_t>) {
    auto rand = dist_func(&state);
    constexpr int vec_bytes = EPC<scalar_t> * sizeof(scalar_t);
    memory::Vec<vec_bytes> v;
    auto* vals = reinterpret_cast<scalar_t*>(&v);
    #pragma unroll
    for (int i = 0; i < EPC<scalar_t>; i++) {
      vals[i] = transform_func((&rand.x)[i]);
    }
    memory::st_vec<vec_bytes>(output + pos, v);
    skipahead(static_cast<unsigned long long>(total_threads - 1) * 4, &state);
  }

  // Scalar tail for remaining elements that don't fill a full vector.
  if (tid == 0 && vec_end < numel) {
    generate_range<scalar_t, needs_alignment>(
        output, vec_end, numel, seed, key_offset,
        dist_func, transform_func);
  }
}

// Multi-key: each thread handles a fixed-size chunk of elements for one key.
// Threads are indexed over (key_idx, chunk_idx) via grid-stride loop.
template <typename scalar_t, bool needs_alignment, typename dist_t, typename transform_t>
__global__ void philox_multi_key_kernel(
    scalar_t* __restrict__ output,
    const uint64_t* __restrict__ keys,
    int64_t num_keys,
    int64_t event_numel,
    int64_t elems_per_thread,
    dist_t dist_func,
    transform_t transform_func,
    OffsetCalculator<1> key_offset_calc) {
  int64_t chunks_per_key =
      (event_numel + elems_per_thread - 1) / elems_per_thread;
  int64_t total_work = num_keys * chunks_per_key;

  for (int64_t work_idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work_idx < total_work;
       work_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t key_idx = work_idx / chunks_per_key;
    int64_t chunk_idx = work_idx % chunks_per_key;

    auto elem_offset = key_offset_calc.get(key_idx)[0];
    uint64_t seed = keys[elem_offset];
    uint64_t key_offset = keys[elem_offset + 1];

    int64_t elem_start = chunk_idx * elems_per_thread;
    int64_t elem_end = min(elem_start + elems_per_thread, event_numel);

    generate_range<scalar_t, needs_alignment>(
        output + key_idx * event_numel, elem_start, elem_end,
        seed, key_offset, dist_func, transform_func);
  }
}

// Shared distribution kernel dispatches to single-key or multi-key kernels.
template <typename scalar_t, bool needs_alignment, typename dist_t, typename transform_t>
void philox_distribution_kernel(
    const char* op_name,
    Tensor& self, const Tensor& key,
    const dist_t& dist_func, const transform_t& transform_func) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      op_name, ": key must have shape (2,) or (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      op_name, ": key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(self.is_floating_point(),
      op_name, ": self must be a floating point tensor, got ",
      self.scalar_type());
  TORCH_CHECK(self.device() == key.device(),
      op_name, ": self and key must be on the same device, got ",
      self.device(), " and ", key.device());

  if (self.numel() == 0) {
    return;
  }

  int64_t ndim = self.dim();
  int64_t elems_per_key = 1;
  int64_t key_dims = 0;

  if (key.dim() > 1) {
    TORCH_CHECK(key.dim() == ndim + 1,
        op_name, ": batched key must have ndim == output ndim + 1, "
        "got key shape ", key.sizes(), " with output shape ", self.sizes());

    for (int64_t i = 0; i < ndim; i++) {
      TORCH_CHECK(key.size(i) == 1 || key.size(i) == self.size(i),
          op_name, ": key dim ", i, " (size ", key.size(i),
          ") is not broadcastable with output dim ", i,
          " (size ", self.size(i), ")");
    }

    key_dims = ndim;
    for (int64_t i = ndim - 1; i >= 0; i--) {
      if (key.size(i) != 1) break;
      elems_per_key *= self.size(i);
      key_dims--;
    }
  } else {
    elems_per_key = self.numel();
  }

  int64_t num_keys = self.numel() / elems_per_key;
  // ensure output contiguity for simplicity
  auto output = self.contiguous();

  constexpr int block_size = 256;
  int max_blocks =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount *
      (at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor /
       block_size);

  auto key_arg = num_keys == 1 ? key.contiguous() : key;
  const uint64_t* key_ptr = key_arg.data_ptr<uint64_t>();

  if (num_keys == 1) {
    // === Launch single key kernel. ===
    int num_blocks = std::min(
        static_cast<int>((elems_per_key + block_size - 1) / block_size),
        max_blocks);

    philox_single_key_kernel<scalar_t, needs_alignment>
        <<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        output.mutable_data_ptr<scalar_t>(),
        key_ptr, elems_per_key, dist_func, transform_func);
  } else {
    // === Launch batched key kernel. ===
    // Handle key, self broadcasting via OffsetCalculator
    c10::SmallVector<int64_t, MAX_DIMS> oc_sizes(key_dims);
    c10::SmallVector<int64_t, MAX_DIMS> oc_strides(key_dims);
    for (int64_t i = 0; i < key_dims; i++) {
      int64_t dim = key_dims - 1 - i;
      oc_sizes[i] = self.size(dim);
      oc_strides[i] = key.size(dim) > 1 ? key.stride(dim) : 0;
    }
    const int64_t* oc_strides_ptr = oc_strides.data();
    auto key_offset_calc = OffsetCalculator<1>(
        key_dims, oc_sizes.data(), &oc_strides_ptr);

    constexpr int64_t elems_per_thread = 16;
    int64_t chunks_per_key =
        (elems_per_key + elems_per_thread - 1) / elems_per_thread;
    int64_t total_work = num_keys * chunks_per_key;
    int num_blocks = std::min(
        static_cast<int>((total_work + block_size - 1) / block_size),
        max_blocks);

    philox_multi_key_kernel<scalar_t, needs_alignment>
        <<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        output.mutable_data_ptr<scalar_t>(),
        key_ptr, num_keys, elems_per_key, elems_per_thread,
        dist_func, transform_func, key_offset_calc);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (output.data_ptr() != self.data_ptr()) {
    self.copy_(output);
  }
}

} // anonymous namespace

Tensor& _philox_uniform_cuda_(
    Tensor& self, const Tensor& key, double low, double high) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, self.scalar_type(), "_philox_uniform_", [&] {
    auto lo = static_cast<scalar_t>(low);
    auto hi = static_cast<scalar_t>(high);
    auto transform_func = [lo, hi] __device__ (auto rand) {
      return static_cast<scalar_t>(
          at::transformation::uniform_real(rand, lo, hi));
    };
    auto dist_func = []() {
      if constexpr (std::is_same_v<scalar_t, double>) {
        return [] __device__ (curandStatePhilox4_32_10_t* state) {
          // double needs 53 bits of randomness; pack pairs of uint32 into uint64.
          uint4 r = curand4(state);
          ulonglong2 result;
          result.x = (static_cast<unsigned long long>(r.x) << 32) | r.y;
          result.y = (static_cast<unsigned long long>(r.z) << 32) | r.w;
          return result;
        };
      } else {
        return [] __device__ (curandStatePhilox4_32_10_t* state) {
          return curand4(state);
        };
      }
    }();
    philox_distribution_kernel<scalar_t, /*needs_alignment=*/false>(
        "_philox_uniform_", self, key, dist_func, transform_func);
  });
  return self;
}

Tensor& _philox_normal_cuda_(
    Tensor& self, const Tensor& key, double mean, double stddev) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, self.scalar_type(), "_philox_normal_", [&] {
    using compute_t = std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
    auto mu = static_cast<compute_t>(mean);
    auto sigma = static_cast<compute_t>(stddev);
    auto transform_func = [mu, sigma] __device__ (compute_t rand) {
      return static_cast<scalar_t>(rand * sigma + mu);
    };
    auto dist_func = []() {
      if constexpr (std::is_same_v<scalar_t, double>) {
        return [] __device__ (curandStatePhilox4_32_10_t* state) {
          return curand_normal2_double(state);
        };
      } else {
        return [] __device__ (curandStatePhilox4_32_10_t* state) {
          return curand_normal4(state);
        };
      }
    }();
    philox_distribution_kernel<scalar_t, /*needs_alignment=*/true>(
        "_philox_normal_", self, key, dist_func, transform_func);
  });
  return self;
}

} // namespace at::native
