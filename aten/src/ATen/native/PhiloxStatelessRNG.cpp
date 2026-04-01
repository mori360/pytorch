#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Dispatch.h>
#include <ATen/native/PhiloxStatelessRNG.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_key_fold_in_native.h>
#include <ATen/ops/_philox_key_split_native.h>
#include <ATen/ops/_philox_normal_native.h>
#include <ATen/ops/_philox_uniform_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

#include <cmath>

namespace at::native {

DEFINE_DISPATCH(philox_uniform_fill_stub);
DEFINE_DISPATCH(philox_normal_fill_stub);

namespace {

// curand's offset counts individual uint32 outputs; philox_engine's
// constructor offset counts 128-bit blocks (groups of 4 uint32).
// We restrict to subsequence=0 and let the 64-bit offset wrap naturally,
// so grid_split's modular offset arithmetic stays consistent.
inline philox_engine make_philox(uint64_t seed, uint64_t offset) {
  uint64_t block = offset / 4;
  philox_engine engine(seed, /*subsequence=*/0, /*offset=*/block);
  for (uint64_t i = 0; i < offset % 4; i++) {
    engine();
  }
  return engine;
}

} // anonymous namespace

Tensor _philox_key_split_cpu(const Tensor& key, int64_t num_splits) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_key_split: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_key_split: key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(num_splits > 0,
      "_philox_key_split: num_splits must be positive, got ",
      num_splits);

  auto key_contig = key.contiguous();
  int64_t num_keys = key.numel() / 2;

  auto batch_sizes = key.sizes().slice(0, key.dim() - 1);
  std::vector<int64_t> output_sizes;
  output_sizes.reserve(batch_sizes.size() + 2);
  output_sizes.push_back(num_splits);
  for (auto s : batch_sizes) {
    output_sizes.push_back(s);
  }
  output_sizes.push_back(2);

  Tensor output = at::empty(output_sizes, key.options());

  if (num_keys == 0) {
    return output;
  }

  const uint64_t* input = key_contig.const_data_ptr<uint64_t>();
  uint64_t* out_ptr = output.data_ptr<uint64_t>();

  for (int64_t key_idx = 0; key_idx < num_keys; key_idx++) {
    uint64_t seed = input[key_idx * 2];
    uint64_t offset = input[key_idx * 2 + 1];
    auto engine = make_philox(seed, offset);

    for (int64_t split_idx = 0; split_idx < num_splits; split_idx++) {
      uint32_t r0 = engine(), r1 = engine(), r2 = engine(), r3 = engine();
      uint64_t new_seed = static_cast<uint64_t>(r0) |
          (static_cast<uint64_t>(r1) << 32);
      uint64_t new_offset = static_cast<uint64_t>(r2) |
          (static_cast<uint64_t>(r3) << 32);
      out_ptr[(split_idx * num_keys + key_idx) * 2] = new_seed;
      out_ptr[(split_idx * num_keys + key_idx) * 2 + 1] = new_offset;
    }
  }

  return output;
}

Tensor _philox_key_fold_in_cpu(const Tensor& key, int64_t data) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_key_fold_in: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_key_fold_in: key must have dtype uint64, got ",
      key.scalar_type());

  auto key_contig = key.contiguous();
  int64_t num_keys = key.numel() / 2;

  Tensor output = at::empty_like(key_contig);

  if (num_keys == 0) {
    return output;
  }

  const uint64_t* input = key_contig.const_data_ptr<uint64_t>();
  uint64_t* out_ptr = output.data_ptr<uint64_t>();

  for (int64_t idx = 0; idx < num_keys; idx++) {
    uint64_t seed = input[idx * 2];
    uint64_t offset = input[idx * 2 + 1];

    // Match CUDA: curand_init(seed, 0, offset), skipahead(data * 4).
    auto engine = make_philox(seed, offset + static_cast<uint64_t>(data) * 4);

    uint32_t r0 = engine(), r1 = engine(), r2 = engine(), r3 = engine();
    uint64_t new_seed = static_cast<uint64_t>(r0) |
        (static_cast<uint64_t>(r1) << 32);
    uint64_t new_offset = static_cast<uint64_t>(r2) |
        (static_cast<uint64_t>(r3) << 32);
    out_ptr[idx * 2] = new_seed;
    out_ptr[idx * 2 + 1] = new_offset;
  }

  return output;
}

Tensor& _philox_uniform_cpu_(Tensor& self, const Tensor& key, double low, double high, bool portable) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_uniform: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_uniform: key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(self.is_floating_point(),
      "_philox_uniform: self must be a floating point tensor, got ",
      self.scalar_type());
  TORCH_CHECK(self.device() == key.device(),
      "_philox_uniform: self and key must be on the same device, got ",
      self.device(), " and ", key.device());

  if (!portable) {
    TORCH_CHECK(key.dim() == 1 && key.size(0) == 2,
        "_philox_uniform: portable=False does not support batched keys");
    auto key_contig = key.contiguous();
    const uint64_t* key_data = key_contig.const_data_ptr<uint64_t>();
    auto gen = at::detail::createCPUGenerator(
        key_data[0] ^ (key_data[1] * 0x9E3779B97F4A7C15ULL));
    self.uniform_(low, high, gen);
    return self;
  }

  int64_t ndim = self.dim();
  int64_t elems_per_key = 1;

  if (key.dim() > 1) {
    int64_t key_batch_ndim = key.dim() - 1;
    TORCH_CHECK(key_batch_ndim == ndim,
        "_philox_uniform: batched key must have ndim == output ndim + 1, "
        "got key shape ", key.sizes(), " with output shape ", self.sizes());

    for (int64_t i = 0; i < ndim; i++) {
      TORCH_CHECK(key.size(i) == 1 || key.size(i) == self.size(i),
          "_philox_uniform: key dim ", i, " (size ", key.size(i),
          ") is not broadcastable with output dim ", i,
          " (size ", self.size(i), ")");
    }

    // Trailing suffix of size-1 key dims forms the generation axis.
    for (int64_t i = ndim - 1; i >= 0; i--) {
      if (key.size(i) != 1) break;
      elems_per_key *= self.size(i);
    }
  } else {
    elems_per_key = self.numel();
  }

  int64_t num_keys = self.numel() / elems_per_key;
  int64_t key_dims = key.dim() > 1 ? ndim : 0;
  if (key.dim() > 1) {
    for (int64_t i = ndim - 1; i >= 0; i--) {
      if (key.size(i) != 1) break;
      key_dims--;
    }
  }

  Tensor key_flat;
  if (key.dim() > 1) {
    std::vector<int64_t> expand_sizes;
    expand_sizes.reserve(key.dim());
    for (int64_t i = 0; i < key_dims; i++) {
      expand_sizes.push_back(self.size(i));
    }
    for (int64_t i = key_dims; i < ndim; i++) {
      expand_sizes.push_back(1);
    }
    expand_sizes.push_back(2);
    key_flat = key.expand(expand_sizes).reshape({num_keys, 2}).contiguous();
  } else {
    key_flat = key.contiguous();
  }

  if (num_keys == 0 || elems_per_key == 0) {
    return self;
  }

  const uint64_t* keys_ptr = key_flat.const_data_ptr<uint64_t>();
  void* out_ptr = self.data_ptr();
  auto dtype = self.scalar_type();

  for (int64_t key_idx = 0; key_idx < num_keys; key_idx++) {
    uint64_t seed = keys_ptr[key_idx * 2];
    uint64_t key_offset = keys_ptr[key_idx * 2 + 1];
    int64_t base = key_idx * elems_per_key;
    philox_uniform_fill_stub(kCPU, out_ptr, base, elems_per_key,
                             seed, key_offset, low, high, dtype);
  }

  return self;
}

Tensor& _philox_normal_cpu_(Tensor& self, const Tensor& key, double mean, double stddev, bool portable) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_normal: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_normal: key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(self.is_floating_point(),
      "_philox_normal: self must be a floating point tensor, got ",
      self.scalar_type());
  TORCH_CHECK(self.device() == key.device(),
      "_philox_normal: self and key must be on the same device, got ",
      self.device(), " and ", key.device());

  if (!portable) {
    TORCH_CHECK(key.dim() == 1 && key.size(0) == 2,
        "_philox_normal: portable=False does not support batched keys");
    auto key_contig = key.contiguous();
    const uint64_t* key_data = key_contig.const_data_ptr<uint64_t>();
    auto gen = at::detail::createCPUGenerator(
        key_data[0] ^ (key_data[1] * 0x9E3779B97F4A7C15ULL));
    self.normal_(mean, stddev, gen);
    return self;
  }

  int64_t ndim = self.dim();
  int64_t elems_per_key = 1;

  if (key.dim() > 1) {
    int64_t key_batch_ndim = key.dim() - 1;
    TORCH_CHECK(key_batch_ndim == ndim,
        "_philox_normal: batched key must have ndim == output ndim + 1, "
        "got key shape ", key.sizes(), " with output shape ", self.sizes());

    for (int64_t i = 0; i < ndim; i++) {
      TORCH_CHECK(key.size(i) == 1 || key.size(i) == self.size(i),
          "_philox_normal: key dim ", i, " (size ", key.size(i),
          ") is not broadcastable with output dim ", i,
          " (size ", self.size(i), ")");
    }

    // Trailing suffix of size-1 key dims forms the generation axis.
    for (int64_t i = ndim - 1; i >= 0; i--) {
      if (key.size(i) != 1) break;
      elems_per_key *= self.size(i);
    }
  } else {
    elems_per_key = self.numel();
  }

  int64_t num_keys = self.numel() / elems_per_key;
  int64_t key_dims = key.dim() > 1 ? ndim : 0;
  if (key.dim() > 1) {
    for (int64_t i = ndim - 1; i >= 0; i--) {
      if (key.size(i) != 1) break;
      key_dims--;
    }
  }

  // Expand the key dims portion against self's sizes, collapse trailing
  // generation dims, and make contiguous to get shape (num_keys, 2).
  Tensor key_flat;
  if (key.dim() > 1) {
    std::vector<int64_t> expand_sizes;
    expand_sizes.reserve(key.dim());
    for (int64_t i = 0; i < key_dims; i++) {
      expand_sizes.push_back(self.size(i));
    }
    for (int64_t i = key_dims; i < ndim; i++) {
      expand_sizes.push_back(1);
    }
    expand_sizes.push_back(2);
    key_flat = key.expand(expand_sizes).reshape({num_keys, 2}).contiguous();
  } else {
    key_flat = key.contiguous();
  }

  if (num_keys == 0 || elems_per_key == 0) {
    return self;
  }

  const uint64_t* keys_ptr = key_flat.const_data_ptr<uint64_t>();
  void* out_ptr = self.data_ptr();
  auto dtype = self.scalar_type();

  for (int64_t key_idx = 0; key_idx < num_keys; key_idx++) {
    uint64_t seed = keys_ptr[key_idx * 2];
    uint64_t key_offset = keys_ptr[key_idx * 2 + 1];
    int64_t base = key_idx * elems_per_key;
    philox_normal_fill_stub(kCPU, out_ptr, base, elems_per_key,
                            seed, key_offset, mean, stddev, dtype);
  }

  return self;
}

} // namespace at::native
