#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>

#include <curand_kernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_key_fold_in_native.h>
#include <ATen/ops/_philox_key_split_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

namespace at::native {

namespace {

// Each thread handles a contiguous chunk of splits for one key. Threads are
// indexed over (key_idx, chunk_idx) where chunk_idx partitions the splits
// dimension, so we get parallelism across both keys and splits while calling
// curand_init only once per thread.
__global__ void philox_key_split_kernel(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ output,
    int64_t num_keys,
    int64_t num_splits,
    int64_t splits_per_thread) {
  int64_t total_threads = num_keys * ((num_splits + splits_per_thread - 1) / splits_per_thread);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t num_chunks = (num_splits + splits_per_thread - 1) / splits_per_thread;
  for (; tid < total_threads; tid += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t key_idx = tid / num_chunks;
    int64_t chunk_idx = tid % num_chunks;
    int64_t split_start = chunk_idx * splits_per_thread;
    int64_t split_end = min(split_start + splits_per_thread, num_splits);

    uint64_t seed = input[key_idx * 2];
    uint64_t offset = input[key_idx * 2 + 1];

    curandStatePhilox4_32_10_t state;
    curand_init(seed, /*subsequence=*/0, /*offset=*/offset, &state);
    if (split_start > 0) {
      skipahead(static_cast<unsigned long long>(split_start) * 4, &state);
    }

    for (int64_t split_idx = split_start; split_idx < split_end; split_idx++) {
      // Sample randomness to derive each new (seed, offset) pair.
      uint4 r = curand4(&state);

      uint64_t new_seed = static_cast<uint64_t>(r.x) | (static_cast<uint64_t>(r.y) << 32);
      uint64_t new_offset = static_cast<uint64_t>(r.z) | (static_cast<uint64_t>(r.w) << 32);

      output[(split_idx * num_keys + key_idx) * 2] = new_seed;
      output[(split_idx * num_keys + key_idx) * 2 + 1] = new_offset;
    }
  }
}

__global__ void philox_key_fold_in_kernel(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ output,
    int64_t num_keys,
    int64_t data) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; idx < num_keys; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    uint64_t seed = input[idx * 2];
    uint64_t offset = input[idx * 2 + 1];

    curandStatePhilox4_32_10_t state;
    curand_init(seed, /*subsequence=*/0, /*offset=*/offset, &state);
    skipahead(static_cast<unsigned long long>(data) * 4, &state);

    // Sample randomness to derive a new (seed, offset) pair.
    uint4 r = curand4(&state);

    uint64_t new_seed = static_cast<uint64_t>(r.x) | (static_cast<uint64_t>(r.y) << 32);
    uint64_t new_offset = static_cast<uint64_t>(r.z) | (static_cast<uint64_t>(r.w) << 32);

    output[idx * 2] = new_seed;
    output[idx * 2 + 1] = new_offset;
  }
}

} // anonymous namespace

Tensor _philox_key_split_cuda(const Tensor& key, int64_t num_splits) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_key_split: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_key_split: key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(num_splits > 0,
      "_philox_key_split: num_splits must be positive, got ",
      num_splits);

  // Output shape: (num_splits, *batch, 2)
  auto batch_sizes = key.sizes().slice(0, key.dim() - 1);
  std::vector<int64_t> output_sizes;
  output_sizes.reserve(batch_sizes.size() + 2);
  output_sizes.push_back(num_splits);
  for (auto s : batch_sizes) {
    output_sizes.push_back(s);
  }
  output_sizes.push_back(2);

  Tensor output = at::empty(output_sizes, key.options());
  int64_t num_keys = key.numel() / 2;
  if (num_keys == 0) {
    return output;
  }

  auto key_contig = key.contiguous();

  // Each thread generates splits_per_thread consecutive splits for one key,
  // amortizing curand_init over many sequential curand calls.
  constexpr int64_t splits_per_thread = 16;
  int64_t num_chunks = (num_splits + splits_per_thread - 1) / splits_per_thread;
  int64_t total_threads = num_keys * num_chunks;
  constexpr int block_size = 256;
  int num_blocks = std::min(
      static_cast<int>((total_threads + block_size - 1) / block_size),
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 4);

  philox_key_split_kernel<<<num_blocks, block_size, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      key_contig.data_ptr<uint64_t>(),
      output.data_ptr<uint64_t>(),
      num_keys, num_splits, splits_per_thread);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

Tensor _philox_key_fold_in_cuda(const Tensor& key, int64_t data) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_key_fold_in: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_key_fold_in: key must have dtype uint64, got ",
      key.scalar_type());

  Tensor output = at::empty_like(key);
  int64_t num_keys = key.numel() / 2;
  if (num_keys == 0) {
    return output;
  }

  auto key_contig = key.contiguous();

  constexpr int block_size = 256;
  int num_blocks = std::min(
      static_cast<int>((num_keys + block_size - 1) / block_size),
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 4);

  philox_key_fold_in_kernel<<<num_blocks, block_size, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      key_contig.data_ptr<uint64_t>(),
      output.data_ptr<uint64_t>(),
      num_keys, data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

} // namespace at::native
