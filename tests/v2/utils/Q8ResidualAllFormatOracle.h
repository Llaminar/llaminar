/**
 * @file Q8ResidualAllFormatOracle.h
 * @brief Test-only independent FP64 arithmetic over original and actual
 * prepared native weights.
 *
 * Element decoding reuses the canonical format/LUT decoder, never a GEMM, dot
 * microkernel, quantizer or output. Scalar coefficient reconstruction and FP64
 * dots are diagnostic oracles only. Original-source approximation and prepared
 * contribution correctness are separate because preparation may round factors.
 */
#pragma once
#include "QuantizedVerifierFormats.h"
#include "kernels/common/DeviceNativeVNNIMatrixDesc.h"
#include "kernels/cpu/gemm/CPUNativeVNNIDecode.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>
namespace llaminar2::test::q8_all_formats {
/** @brief Snapshot original source bytes before any preparation can reclaim
 * storage. */
inline std::vector<uint8_t> originalBytes(const TensorBase &source) {
  const auto *p = static_cast<const uint8_t *>(source.raw_data());
  if (!p || !source.size_bytes())
    throw std::invalid_argument("Original native source is absent");
  return {p, p + source.size_bytes()};
}
/** @brief Independently decode the original source, without consulting a
 * prepared GEMM. */
inline std::vector<double> originalWeights(const TensorBase &source) {
  std::vector<double> out(source.numel());
  std::vector<float> row(source.cols());
  for (int n = 0; n < int(source.rows()); ++n) {
    source.to_fp32_row(n, row.data());
    for (int k = 0; k < int(source.cols()); ++k) {
      if (!std::isfinite(row[k]))
        throw std::invalid_argument("Oracle source is nonfinite");
      out[std::size_t(n) * source.cols() + k] = row[k];
    }
  }
  return out;
}
/** @brief Preserve the sole inventory; strengthen only actual native minimum
 * witnesses before preparation. */
inline std::unique_ptr<TensorBase>
source(const QuantizedVerifierFormatCase &format, int n, int k) {
  auto creator = format.create;
  if (format.tensor_type == TensorType::IQ4_XS)
    creator = createBoundedMoEIQ4XS;
  if (format.tensor_type == TensorType::IQ3_S)
    creator = createNonzeroMoEIQ3S;
  auto result = creator({std::size_t(n), std::size_t(k)},
                        801u + unsigned(format.tensor_type));
  // The base Q4_K/Q5_K creators intentionally have dmin0. These authored binary
  // factors make minima mandatory and exactly representable, without
  // requantizing.
  if (format.tensor_type == TensorType::Q4_K) {
    auto *p = static_cast<Q4_KBlock *>(result->raw_mutable_data());
    for (std::size_t i = 0; i < result->size_bytes() / sizeof(*p); ++i) {
      p[i].d = fp32_to_fp16(.0078125f);
      p[i].dmin = fp32_to_fp16(.00390625f);
    }
  }
  if (format.tensor_type == TensorType::Q5_K) {
    auto *p = static_cast<Q5_KBlock *>(result->raw_mutable_data());
    for (std::size_t i = 0; i < result->size_bytes() / sizeof(*p); ++i) {
      p[i].d = fp32_to_fp16(.0078125f);
      p[i].dmin = fp32_to_fp16(.00390625f);
    }
  }
  if (format.tensor_type == TensorType::Q2_K) {
    auto *p = static_cast<Q2_KBlock *>(result->raw_mutable_data());
    for (std::size_t i = 0; i < result->size_bytes() / sizeof(*p); ++i) {
      p[i].d = fp32_to_fp16(.015625f);
      p[i].dmin = fp32_to_fp16(.0078125f);
      for (auto &v : p[i].scales)
        v |= 0x10u;
    }
  }
  return result;
}
/** @brief Seven distinct authored rows (one in tiled mode) bound diagnostic
 * oracle cost. Repeated rows remain explicit physical work; consumers prove
 * exact row-byte identity before reusing an M1 expectation. Outliers leave a
 * signed activation residual. */
inline std::vector<float> input(int m, int k, bool tile = false) {
  std::vector<float> out(std::size_t(m) * k);
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < k; ++c) {
      const int row = tile ? 0 : r % 7;
      out[std::size_t(r) * k + c] =
          c % 32 == 0 ? float(row % 7 + 1) * 17.25f
                      : std::sin(float(c * 7 + row * 11)) * .31f;
    }
  return out;
}
/** @brief One independent activation observation; no inferred live count is
 * retained. */
struct Banks {
  std::vector<int8_t> primary, residual;
  std::vector<float> d0, d1;
};
/** @brief Decode actual CPU Q8 banks and verify each term's independent integer
 * compensation. */
inline Banks cpuBanks(const Q8_1Block *a, const Q8_1Block *b, int rows, int k) {
  Banks out;
  out.primary.resize(std::size_t(rows) * k);
  out.residual.resize(out.primary.size());
  out.d0.resize(std::size_t(rows) * k / 32);
  out.d1.resize(out.d0.size());
  for (std::size_t block = 0; block < out.d0.size(); ++block) {
    if (a[block].sum_qs != std::accumulate(std::begin(a[block].qs),
                                           std::end(a[block].qs), 0) ||
        b[block].sum_qs !=
            std::accumulate(std::begin(b[block].qs), std::end(b[block].qs), 0))
      throw std::runtime_error("Independent primary/residual sum_qs mismatch");
    std::copy_n(a[block].qs, 32, out.primary.data() + block * 32);
    std::copy_n(b[block].qs, 32, out.residual.data() + block * 32);
    out.d0[block] = fp16_to_fp32(a[block].d);
    out.d1[block] = fp16_to_fp32(b[block].d);
  }
  return out;
}
/** @brief Decode only a canonical integer codebook element, not its GEMM
 * contribution tree. */
inline std::array<int8_t, 32> integerPayload(uint8_t codebook,
                                             std::span<const uint8_t> bytes) {
  std::array<int8_t, 32> out{};
  if (codebook == 19 || codebook == kNativeVnniExpandedInt8MinCodebook) {
    if (bytes.size() < 32)
      throw std::invalid_argument("Short expanded payload");
    std::memcpy(out.data(), bytes.data(), 32);
  } else if (codebook == 8) {
    if (bytes.size() < 24)
      throw std::invalid_argument("Short Q6 payload");
    for (int i = 0; i < 32; ++i) {
      const int half = i / 16, j = i % 16;
      const int low = (bytes[j] >> (half * 4)) & 15;
      const int high = (bytes[16 + half * 4 + j / 4] >> (2 * (j % 4))) & 3;
      out[i] = int8_t((low | (high << 4)) - 32);
    }
  } else
    cpu::native_vnni::decode_native_block(
        codebook == kNativeVnniIQ4XSSourceScaleCodebook ? 4 : codebook,
        bytes.data(), out.data());
  return out;
}
/** @brief FP64 coefficients from actual compact device planes, including
 * additive and source-factor metadata. */
inline std::array<double, 32> coefficients(uint8_t cb,
                                           std::span<const uint8_t> bytes,
                                           uint16_t primary, uint16_t secondary,
                                           uint32_t emins) {
  const auto q = integerPayload(cb, bytes);
  std::array<double, 32> out{};
  double lo = fp16_to_fp32(primary), hi = fp16_to_fp32(secondary);
  if (cb == 8) {
    hi = lo * int8_t(secondary >> 8);
    lo *= int8_t(secondary & 255);
  }
  if (cb == 13) {
    hi = lo * (.5 + ((secondary >> 4) & 15)) * .25;
    lo *= .5 + (secondary & 15);
    lo *= .25;
  }
  if (cb == kNativeVnniIQ4XSSourceScaleCodebook)
    lo *= int(secondary & 63) - 32;
  if (cb == 11)
    lo *= 1 + 2 * int(secondary & 15);
  const bool dual =
      cb == 8 || cb == 9 || cb == 10 || cb == 13 || cb == 14 || cb == 17;
  for (int i = 0; i < 32; ++i) {
    const double factor = dual && i >= 16 ? hi : lo;
    out[i] = factor * q[i];
    if (cb == 5 || cb == 7 || cb == 16 ||
        cb == kNativeVnniExpandedInt8MinCodebook)
      out[i] += hi;
    if (cb == 10)
      out[i] += fp16_to_fp32(uint16_t(emins >> (i < 16 ? 0 : 16)));
    if (cb == 17) {
      const int group = i / 8;
      const auto word = bytes[4 + group / 2];
      out[i] += factor * ((word & (group % 2 ? 128 : 8)) ? -.125 : .125);
    }
  }
  return out;
}
/** @brief Extract actual CPU physical coefficients without calling any
 * projection or contribution kernel. */
inline std::vector<double>
cpuWeights(const cpu::native_vnni::CPUNativeVNNIPackedWeights &p) {
  std::vector<double> out(std::size_t(p.N) * p.K);
  for (int n = 0; n < p.N; ++n)
    for (int b = 0; b < p.blocks_per_row; ++b) {
      const int chunk = n / 64, col = n % 64, z = col / 16, lane = col % 16;
      std::array<double, 32> decoded{};
      if (p.usesCompactMultiScale()) {
        std::array<uint8_t, 32> payload{};
        for (int j = 0; j < p.payload_bytes; ++j)
          payload[j] = p.interleavedB(chunk, b, j / 4, z)[lane * 4 + j % 4];
        decoded = coefficients(
            p.codebook_id,
            std::span<const uint8_t>(payload.data(), p.payload_bytes),
            p.chunkScales(chunk, b)[col], p.chunkMins(chunk, b)[col],
            p.codebook_id == 10 ? p.chunkEffectiveMins(chunk, b)[col] : 0);
      } else
        for (int i = 0; i < 32; ++i) {
          int value;
          if (p.usesQ6KNativeDualScale()) {
            const auto byte =
                p.interleavedB(chunk, b, (i % 16) / 4, z)[lane * 4 + i % 4];
            const int low = (byte >> (i < 16 ? 0 : 4)) & 15;
            const auto *planes = p.q6KHighBitplanes(chunk, b, i / 4, z);
            uint64_t a, c;
            std::memcpy(&a, planes, 8);
            std::memcpy(&c, planes + 8, 8);
            const int bit = lane * 4 + i % 4;
            value =
                (low | (((a >> bit) & 1) << 4) | (((c >> bit) & 1) << 5)) - 32;
          } else if (p.usesNibbleLUT()) {
            const auto byte =
                p.interleavedB(chunk, b, (i % 16) / 4, z)[lane * 4 + i % 4];
            const int q = (byte >> (i < 16 ? 0 : 4)) & 15;
            value = p.codebook_id == 0   ? q - 8
                    : p.codebook_id == 5 ? q
                                         : kvalues_iq4nl[q];
          } else
            value =
                int8_t(p.interleavedB(chunk, b, i / 4, z)[lane * 4 + i % 4]);
          const double scale = p.usesQ6KNativeDualScale() && i >= 16
                                   ? p.blockMin(chunk, b, col)
                                   : p.blockScale(chunk, b, col);
          decoded[i] = scale * value;
          if (p.is_asymmetric && !p.usesQ6KNativeDualScale())
            decoded[i] += p.blockMin(chunk, b, col);
        }
      std::copy(decoded.begin(), decoded.end(),
                out.begin() + std::size_t(n) * p.K + b * 32);
    }
  return out;
}
/** @brief Independent FP64 dot; each term folds separately before the final
 * sum. */
inline std::vector<double> project(std::span<const double> weights, int n,
                                   int k, const Banks &banks, int rows,
                                   bool corrected) {
  std::vector<double> out(std::size_t(rows) * n);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < n; ++c) {
      double a = 0, b = 0;
      for (int j = 0; j < k; ++j) {
        const auto offset = std::size_t(r) * k + j, block = offset / 32;
        a += weights[std::size_t(c) * k + j] * banks.d0[block] *
             banks.primary[offset];
        if (corrected)
          b += weights[std::size_t(c) * k + j] * banks.d1[block] *
               banks.residual[offset];
      }
      out[std::size_t(r) * n + c] = a + b;
    }
  return out;
}
/** @brief Record true original-source FP64 error independently of native
 * preparation. */
inline std::vector<double> exact(std::span<const double> weights, int n, int k,
                                 std::span<const float> values, int rows) {
  std::vector<double> out(std::size_t(rows) * n);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < n; ++c)
      for (int j = 0; j < k; ++j)
        out[std::size_t(r) * n + c] += weights[std::size_t(c) * k + j] *
                                       double(values[std::size_t(r) * k + j]);
  return out;
}
/** @brief Derive a forward-error budget from FP32 operations and the
 * independent absolute dot mass. */
inline double roundingBound(std::span<const double> weights, int c, int k,
                            const Banks &banks, int row) {
  double mass = 0;
  for (int j = 0; j < k; ++j) {
    const auto i = std::size_t(row) * k + j, b = i / 32;
    mass += std::abs(weights[std::size_t(c) * k + j]) *
            (std::abs(double(banks.d0[b]) * banks.primary[i]) +
             std::abs(double(banks.d1[b]) * banks.residual[i]));
  }
  const double operations = 16.0 * (k / 32 + 4),
               epsilon = std::numeric_limits<float>::epsilon();
  return operations * epsilon / (1 - operations * epsilon) *
         std::max(mass, double(std::numeric_limits<float>::min()));
}
/**
 * @brief Bound original-source error by independently measured packing and
 * operand errors.
 *
 * The two nonnegative masses are explicit diagnostics, not a replacement
 * accuracy tolerance: actual native coefficient rounding and actual Q8 operand
 * error are separated. Existing Q8_0 golden accuracy limits remain in their
 * original tests.
 */
inline double originalProjectionBound(std::span<const double> original,
                                      std::span<const double> prepared, int c,
                                      int k, const Banks &banks, int row,
                                      std::span<const float> input) {
  double packing = 0, activation = 0;
  for (int j = 0; j < k; ++j) {
    const auto i = std::size_t(row) * k + j, b = i / 32,
               w = std::size_t(c) * k + j;
    const double x = input[i];
    const double operand = double(banks.d0[b]) * banks.primary[i] +
                           double(banks.d1[b]) * banks.residual[i];
    packing += std::abs(original[w] - prepared[w]) * std::abs(x);
    activation += std::abs(prepared[w]) * std::abs(x - operand);
  }
  return packing + activation + roundingBound(prepared, c, k, banks, row);
}
} // namespace llaminar2::test::q8_all_formats
