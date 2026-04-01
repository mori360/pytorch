#include <ATen/core/PhiloxRNGEngine.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/PhiloxStatelessRNG.h>

#ifdef CPU_CAPABILITY_AVX2
#include <ATen/native/cpu/avx_mathfun.h>
#endif

#include <cmath>

namespace at::native {
namespace {

// Constants matching curand's conversion formulas.
constexpr float CURAND_2POW32_INV = 2.3283064e-10f;
constexpr double CURAND_2POW32_INV_DOUBLE = 2.3283064365386963e-10;
constexpr float CURAND_2POW32_INV_2PI = 2.3283064e-10f * 6.2831855f;
constexpr double CURAND_2POW53_INV_DOUBLE = 1.1102230246251565e-16;
constexpr double CURAND_PI_DOUBLE = 3.1415926535897932;

inline philox_engine make_philox(uint64_t seed, uint64_t offset) {
  uint64_t block = offset / 4;
  philox_engine engine(seed, /*subsequence=*/0, /*offset=*/block);
  for (uint64_t i = 0; i < offset % 4; i++) {
    engine();
  }
  return engine;
}

inline float philox_uniform_float(uint32_t x) {
  return x * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
}

inline double philox_uniform_double(uint32_t x0, uint32_t x1) {
  auto z = static_cast<unsigned long long>(x0) ^
      (static_cast<unsigned long long>(x1) << (53 - 32));
  return z * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE / 2.0);
}

inline int64_t wrap_point(uint64_t key_offset, int64_t numel, int outputs_per_elem) {
  if (key_offset == 0) return numel;
  uint64_t outputs_before_wrap = static_cast<uint64_t>(0) - key_offset;
  uint64_t elems_before_wrap = outputs_before_wrap / outputs_per_elem;
  return (elems_before_wrap < static_cast<uint64_t>(numel))
      ? static_cast<int64_t>(elems_before_wrap)
      : numel;
}

// --------------- Uniform generation ---------------

template <typename scalar_t>
void uniform_fill(
    scalar_t* output, int64_t base, int64_t numel,
    uint64_t seed, uint64_t key_offset, double low, double high) {
  float flow = static_cast<float>(low);
  float frange = static_cast<float>(high - low);
  int64_t wp = wrap_point(key_offset, numel, /*outputs_per_elem=*/1);
  auto engine = make_philox(seed, key_offset);
  for (int64_t i = 0; i < wp; i++) {
    float u = philox_uniform_float(engine());
    output[base + i] = static_cast<scalar_t>(flow + frange * u);
  }
  if (wp < numel) {
    engine = make_philox(seed, 0);
    for (int64_t i = wp; i < numel; i++) {
      float u = philox_uniform_float(engine());
      output[base + i] = static_cast<scalar_t>(flow + frange * u);
    }
  }
}

template <>
void uniform_fill<double>(
    double* output, int64_t base, int64_t numel,
    uint64_t seed, uint64_t key_offset, double low, double high) {
  double range = high - low;
  int64_t wp = wrap_point(key_offset, numel, /*outputs_per_elem=*/2);
  auto engine = make_philox(seed, key_offset);
  for (int64_t i = 0; i < wp; i++) {
    uint32_t r0 = engine(), r1 = engine();
    double u = philox_uniform_double(r0, r1);
    output[base + i] = low + range * u;
  }
  if (wp < numel) {
    engine = make_philox(seed, 0);
    for (int64_t i = wp; i < numel; i++) {
      uint32_t r0 = engine(), r1 = engine();
      double u = philox_uniform_double(r0, r1);
      output[base + i] = low + range * u;
    }
  }
}

// --------------- Normal generation ---------------

// Scalar Box-Muller for 4 normals from one Philox block.
inline void box_muller_4(philox_engine& engine, float* out) {
  uint32_t r0 = engine(), r1 = engine(), r2 = engine(), r3 = engine();
  float u0 = r0 * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
  float v0 = r1 * CURAND_2POW32_INV_2PI + (CURAND_2POW32_INV_2PI / 2.0f);
  float s0 = std::sqrt(-2.0f * std::log(u0));
  float u1 = r2 * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
  float v1 = r3 * CURAND_2POW32_INV_2PI + (CURAND_2POW32_INV_2PI / 2.0f);
  float s1 = std::sqrt(-2.0f * std::log(u1));
  out[0] = s0 * std::sin(v0);
  out[1] = s0 * std::cos(v0);
  out[2] = s1 * std::sin(v1);
  out[3] = s1 * std::cos(v1);
}

#ifdef CPU_CAPABILITY_AVX2

// AVX2 Box-Muller on 16 interleaved uniforms in-place.
// Preserves (r[2i], r[2i+1]) pairing: even elements are radius inputs,
// odd elements are angle inputs. Deinterleaves, transforms, interleaves back.
inline void box_muller_16_avx2(
    float* data,
    const __m256& minus_two, const __m256& two_pi,
    const __m256& mean_v, const __m256& std_v,
    const __m256i& deinterleave_perm) {
  __m256 a = _mm256_loadu_ps(data);
  __m256 b = _mm256_loadu_ps(data + 8);

  __m256 u = _mm256_permutevar8x32_ps(
      _mm256_shuffle_ps(a, b, 0x88), deinterleave_perm);
  __m256 v = _mm256_permutevar8x32_ps(
      _mm256_shuffle_ps(a, b, 0xDD), deinterleave_perm);

  __m256 radius = _mm256_sqrt_ps(_mm256_mul_ps(minus_two, log256_ps(u)));
  __m256 sin_t, cos_t;
  sincos256_ps(_mm256_mul_ps(two_pi, v), &sin_t, &cos_t);

  __m256 n_sin = _mm256_fmadd_ps(
      _mm256_mul_ps(radius, sin_t), std_v, mean_v);
  __m256 n_cos = _mm256_fmadd_ps(
      _mm256_mul_ps(radius, cos_t), std_v, mean_v);

  __m256 lo = _mm256_unpacklo_ps(n_sin, n_cos);
  __m256 hi = _mm256_unpackhi_ps(n_sin, n_cos);
  _mm256_storeu_ps(data, _mm256_permute2f128_ps(lo, hi, 0x20));
  _mm256_storeu_ps(data + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
}

#endif

// Generate a contiguous range [start, end) of normal values from Philox.
// The caller ensures the offset doesn't wrap within this range.
template <typename scalar_t>
void normal_fill_vec(
    scalar_t* output, int64_t base, int64_t start, int64_t end,
    uint64_t seed, uint64_t philox_offset, float fmean, float fstd) {
  int misalign = static_cast<int>(philox_offset & 3);
  if (misalign > 0) {
    philox_offset -= misalign;
  }
  auto engine = make_philox(seed, philox_offset);
  int64_t elem = start;

  // Handle misaligned prefix (partial Philox block).
  if (misalign > 0 && elem < end) {
    float normals[4];
    box_muller_4(engine, normals);
    for (int j = misalign; j < 4 && elem < end; j++, elem++) {
      output[base + elem] = static_cast<scalar_t>(fmean + fstd * normals[j]);
    }
  }

#ifdef CPU_CAPABILITY_AVX2
  int64_t count = end - elem;
  int64_t full16 = (count / 16) * 16;

  if (full16 > 0) {
    const __m256 minus_two = _mm256_set1_ps(-2.0f);
    const __m256 two_pi = _mm256_set1_ps(
        2.0f * static_cast<float>(3.14159265358979323846));
    const __m256 mean_v = _mm256_set1_ps(fmean);
    const __m256 std_v = _mm256_set1_ps(fstd);
    const __m256i deinterleave_perm =
        _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);

    if constexpr (std::is_same_v<scalar_t, float>) {
      // Two-pass in-place: fill output with uniforms, then transform.
      // Separating generation from SIMD math improves CPU pipelining.
      int64_t data_start = elem;
      float* data = output + base + elem;
      for (int64_t i = 0; i < count; i++) {
        data[i] = engine() * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
      }
      for (int64_t i = 0; i < full16; i += 16) {
        box_muller_16_avx2(data + i, minus_two, two_pi,
                           mean_v, std_v, deinterleave_perm);
      }
      elem += full16;
      // Remaining < 16 elements: transform uniforms in-place.
      for (int64_t i = full16; i + 4 <= count; i += 4) {
        float u0 = data[i], v0 = data[i + 1];
        float u1 = data[i + 2], v1 = data[i + 3];
        float s0 = std::sqrt(-2.0f * std::log(u0));
        float th0 = static_cast<float>(
            2.0 * 3.14159265358979323846) * v0;
        float s1 = std::sqrt(-2.0f * std::log(u1));
        float th1 = static_cast<float>(
            2.0 * 3.14159265358979323846) * v1;
        data[i] = fmean + fstd * s0 * std::sin(th0);
        data[i + 1] = fmean + fstd * s0 * std::cos(th0);
        data[i + 2] = fmean + fstd * s1 * std::sin(th1);
        data[i + 3] = fmean + fstd * s1 * std::cos(th1);
        elem += 4;
      }
      // Final partial block (1-3 remaining elements).
      if (elem < end) {
        int64_t ri = elem - data_start;
        int remaining = static_cast<int>(end - elem);
        float tail[4];
        for (int j = 0; j < remaining; j++) {
          tail[j] = data[ri + j];
        }
        // If remaining is odd, generate the missing pair element.
        if (remaining & 1) {
          tail[remaining] = engine() * CURAND_2POW32_INV
              + (CURAND_2POW32_INV / 2.0f);
        }
        float s0 = std::sqrt(-2.0f * std::log(tail[0]));
        float th0 = static_cast<float>(
            2.0 * 3.14159265358979323846) * tail[1];
        float normals[4] = {
          fmean + fstd * s0 * std::sin(th0),
          fmean + fstd * s0 * std::cos(th0), 0.0f, 0.0f};
        if (remaining > 2) {
          float s1 = std::sqrt(-2.0f * std::log(tail[2]));
          float th1 = static_cast<float>(
              2.0 * 3.14159265358979323846) * tail[3];
          normals[2] = fmean + fstd * s1 * std::sin(th1);
          normals[3] = fmean + fstd * s1 * std::cos(th1);
        }
        for (int j = 0; elem < end; j++, elem++) {
          output[base + elem] = normals[j];
        }
      }
      return;
    } else {
      // Chunked two-pass for non-float types.
      constexpr int64_t CHUNK = 256;
      float buf[CHUNK];

      while (elem + CHUNK <= end) {
        for (int i = 0; i < CHUNK; i++) {
          buf[i] = engine() * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
        }
        for (int i = 0; i < CHUNK; i += 16) {
          box_muller_16_avx2(buf + i, minus_two, two_pi,
                             mean_v, std_v, deinterleave_perm);
        }
        for (int i = 0; i < CHUNK; i++) {
          output[base + elem + i] = static_cast<scalar_t>(buf[i]);
        }
        elem += CHUNK;
      }

      int64_t remaining = end - elem;
      int64_t rem16 = (remaining / 16) * 16;
      for (int64_t i = 0; i < rem16; i++) {
        buf[i] = engine() * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
      }
      for (int64_t i = 0; i < rem16; i += 16) {
        box_muller_16_avx2(buf + i, minus_two, two_pi,
                           mean_v, std_v, deinterleave_perm);
      }
      for (int64_t i = 0; i < rem16; i++) {
        output[base + elem + i] = static_cast<scalar_t>(buf[i]);
      }
      elem += rem16;
    }
  }
#endif

  // Scalar tail for remaining < 16 elements.
  for (; elem + 4 <= end; elem += 4) {
    float normals[4];
    box_muller_4(engine, normals);
    output[base + elem + 0] = static_cast<scalar_t>(fmean + fstd * normals[0]);
    output[base + elem + 1] = static_cast<scalar_t>(fmean + fstd * normals[1]);
    output[base + elem + 2] = static_cast<scalar_t>(fmean + fstd * normals[2]);
    output[base + elem + 3] = static_cast<scalar_t>(fmean + fstd * normals[3]);
  }

  if (elem < end) {
    float normals[4];
    box_muller_4(engine, normals);
    for (int j = 0; elem < end; j++, elem++) {
      output[base + elem] = static_cast<scalar_t>(fmean + fstd * normals[j]);
    }
  }
}

// Double normal: uses curand's double Box-Muller (4 uint32 → 2 normals).
inline void normal_fill_double_range(
    double* output, int64_t base, int64_t start, int64_t end,
    uint64_t seed, uint64_t philox_offset, double mean, double stddev) {
  int misalign = static_cast<int>(philox_offset & 3);
  int skip = 0;
  if (misalign > 0 && (misalign % 2) == 0) {
    skip = misalign / 2;
    philox_offset -= misalign;
  }
  auto engine = make_philox(seed, philox_offset);
  int64_t elem = start;

  if (skip > 0 && elem < end) {
    uint32_t r0 = engine(), r1 = engine(), r2 = engine(), r3 = engine();
    auto zx = static_cast<unsigned long long>(r0) ^
        (static_cast<unsigned long long>(r1) << (53 - 32));
    double u = zx * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE / 2.0);
    auto zy = static_cast<unsigned long long>(r2) ^
        (static_cast<unsigned long long>(r3) << (53 - 32));
    double v = zy * (CURAND_2POW53_INV_DOUBLE * 2.0) + CURAND_2POW53_INV_DOUBLE;
    double s = std::sqrt(-2.0 * std::log(u));
    // skip == 1, output only the second normal.
    if (elem < end) {
      output[base + elem] = mean + stddev * s * std::cos(v * CURAND_PI_DOUBLE);
      elem++;
    }
  }

  for (; elem + 2 <= end; elem += 2) {
    uint32_t r0 = engine(), r1 = engine(), r2 = engine(), r3 = engine();
    auto zx = static_cast<unsigned long long>(r0) ^
        (static_cast<unsigned long long>(r1) << (53 - 32));
    double u = zx * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE / 2.0);
    auto zy = static_cast<unsigned long long>(r2) ^
        (static_cast<unsigned long long>(r3) << (53 - 32));
    double v = zy * (CURAND_2POW53_INV_DOUBLE * 2.0) + CURAND_2POW53_INV_DOUBLE;
    double s = std::sqrt(-2.0 * std::log(u));
    output[base + elem + 0] = mean + stddev * s * std::sin(v * CURAND_PI_DOUBLE);
    output[base + elem + 1] = mean + stddev * s * std::cos(v * CURAND_PI_DOUBLE);
  }

  if (elem < end) {
    uint32_t r0 = engine(), r1 = engine(), r2 = engine(), r3 = engine();
    auto zx = static_cast<unsigned long long>(r0) ^
        (static_cast<unsigned long long>(r1) << (53 - 32));
    double u = zx * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE / 2.0);
    auto zy = static_cast<unsigned long long>(r2) ^
        (static_cast<unsigned long long>(r3) << (53 - 32));
    double v = zy * (CURAND_2POW53_INV_DOUBLE * 2.0) + CURAND_2POW53_INV_DOUBLE;
    double s = std::sqrt(-2.0 * std::log(u));
    output[base + elem] = mean + stddev * s * std::sin(v * CURAND_PI_DOUBLE);
    elem++;
  }
}

template <typename scalar_t>
void normal_fill(
    scalar_t* output, int64_t base, int64_t numel,
    uint64_t seed, uint64_t key_offset, double mean, double stddev) {
  float fmean = static_cast<float>(mean);
  float fstd = static_cast<float>(stddev);
  int64_t wp = wrap_point(key_offset, numel, /*outputs_per_elem=*/1);
  normal_fill_vec(output, base, 0, wp, seed, key_offset, fmean, fstd);
  if (wp < numel) {
    normal_fill_vec(output, base, wp, numel, seed, 0, fmean, fstd);
  }
}

template <>
void normal_fill<double>(
    double* output, int64_t base, int64_t numel,
    uint64_t seed, uint64_t key_offset, double mean, double stddev) {
  int64_t wp = wrap_point(key_offset, numel, /*outputs_per_elem=*/2);
  normal_fill_double_range(output, base, 0, wp, seed, key_offset, mean, stddev);
  if (wp < numel) {
    normal_fill_double_range(output, base, wp, numel, seed, 0, mean, stddev);
  }
}

// --------------- Dispatch entry points ---------------

void philox_uniform_fill_kernel(
    void* output, int64_t base, int64_t numel,
    uint64_t seed, uint64_t key_offset,
    double low, double high, at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      uniform_fill(static_cast<float*>(output), base, numel,
                   seed, key_offset, low, high);
      break;
    case at::kDouble:
      uniform_fill(static_cast<double*>(output), base, numel,
                   seed, key_offset, low, high);
      break;
    case at::kHalf:
      uniform_fill(static_cast<at::Half*>(output), base, numel,
                   seed, key_offset, low, high);
      break;
    case at::kBFloat16:
      uniform_fill(static_cast<at::BFloat16*>(output), base, numel,
                   seed, key_offset, low, high);
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype for philox_uniform_fill");
  }
}

void philox_normal_fill_kernel(
    void* output, int64_t base, int64_t numel,
    uint64_t seed, uint64_t key_offset,
    double mean, double stddev, at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      normal_fill(static_cast<float*>(output), base, numel,
                  seed, key_offset, mean, stddev);
      break;
    case at::kDouble:
      normal_fill(static_cast<double*>(output), base, numel,
                  seed, key_offset, mean, stddev);
      break;
    case at::kHalf:
      normal_fill(static_cast<at::Half*>(output), base, numel,
                  seed, key_offset, mean, stddev);
      break;
    case at::kBFloat16:
      normal_fill(static_cast<at::BFloat16*>(output), base, numel,
                  seed, key_offset, mean, stddev);
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype for philox_normal_fill");
  }
}

} // anonymous namespace

REGISTER_DISPATCH(philox_uniform_fill_stub, &philox_uniform_fill_kernel);
REGISTER_DISPATCH(philox_normal_fill_stub, &philox_normal_fill_kernel);

} // namespace at::native
