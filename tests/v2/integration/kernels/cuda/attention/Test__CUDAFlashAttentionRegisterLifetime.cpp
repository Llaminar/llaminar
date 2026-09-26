/**
 * @file Test__CUDAFlashAttentionRegisterLifetime.cpp
 * @brief Captured all-native-format proof for register-resident HD256
 * attention.
 *
 * Relocatable CUDA compilation can lower a row-maximum shuffle to a device
 * helper call. Keeping 128 output accumulators across that call spilled the
 * one-warp P@V specialization. The compiler gate rejects any recurrence; this
 * fixture separately proves that removing the broadcast preserves every output
 * byte across physical tiles, warp striping, and context-partition publication.
 * Q8 decode additionally crosses every compiled head width and changing
 * device-owned KV length, checking an independent FP64 oracle and poisoned
 * replay against eager bytes. Buffers are persistent for each captured replay
 * and no timing gate lives here.
 */
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "backends/cuda/CUDAGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/attention/AttentionDeviceParams.h"
#include "kernels/cuda/attention/CUDAFlashAttentionLaunchPolicy.h"
#include "tensors/FP16Utils.h"
#include "tensors/BlockStructures.h"
#include "utils/DebugEnv.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" {
/** @brief Production Q8 split decode; all scratch is caller-owned. */
int cudaFlashAttn_decode_q8_1(
    const float *, const void *, const void *, float *, float *, float *, float *,
    int, int, int, int, int, int,
    const llaminar2::attention::AttentionDeviceParams *, void *, int, int, int);
/** @brief Production direct FA2 launchers; only native K/V storage differs. */
int cudaFlashAttn_prefill_fa2(
    const float *, const float *, const float *, float *, int, int, int, int,
    int, int, bool, int, int,
    const llaminar2::attention::AttentionDeviceParams *, const float *, void *,
    int, int, int);
int cudaFlashAttn_prefill_fa2_fp16kv(
    const float *, const void *, const void *, float *, int, int, int, int, int,
    int, bool, int, int, const llaminar2::attention::AttentionDeviceParams *,
    const float *, void *, int, int, int);
decltype(cudaFlashAttn_prefill_fa2_fp16kv) cudaFlashAttn_prefill_fa2_bf16kv;
/** @brief Production fixed-context launchers, using caller-owned partials. */
int cudaFlashAttn_prefill_fa2_context_parallel(
    const float *, const float *, const float *, float *, float *, float *,
    float *, int, int, int, int, int, int, bool, int, int,
    const llaminar2::attention::AttentionDeviceParams *, const float *, int,
    int, int, int, int, void *, void *, int, int, int);
int cudaFlashAttn_prefill_fa2_fp16kv_context_parallel(
    const float *, const void *, const void *, float *, float *, float *,
    float *, int, int, int, int, int, int, bool, int, int,
    const llaminar2::attention::AttentionDeviceParams *, const float *, int,
    int, int, int, int, void *, void *, int, int, int);
decltype(cudaFlashAttn_prefill_fa2_fp16kv_context_parallel)
    cudaFlashAttn_prefill_fa2_bf16kv_context_parallel;
}

namespace {
/** @brief Surface an asynchronous or admission error at its originating
 * operation. */
void checked(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

/** @brief Test-only storage allocated before capture and retired after replay.
 */
template <class T> class Storage final {
public:
  /** @brief Allocate the exact physical fixture envelope. */
  explicit Storage(size_t count) {
    checked(cudaMalloc(&data, count * sizeof(T)));
  }
  /** @brief Free after the explicit terminal join in the fixture. */
  ~Storage() { (void)cudaFree(data); }
  Storage(const Storage &) = delete;
  Storage &operator=(const Storage &) = delete;
  T *data = nullptr;
};

/** @brief Scope exact diagnostic launch geometry without leaking it to another
 * test. */
class LaunchPolicy final {
public:
  /** @brief Save the production policy and create one non-default test stream.
   */
  LaunchPolicy()
      : previous_{llaminar2::debugEnv().attention.cuda_fa2_hd256_pv_warps,
                  llaminar2::debugEnv().attention.cuda_fa2_tile_kv,
                  llaminar2::debugEnv().attention.cuda_fa2_q_warp_groups} {
    checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
  /** @brief Join work, restore defaults, and retire the stream in that order.
   */
  ~LaunchPolicy() {
    (void)cudaStreamSynchronize(stream);
    auto &policy = llaminar2::mutableDebugEnv().attention;
    policy.cuda_fa2_hd256_pv_warps = previous_[0];
    policy.cuda_fa2_tile_kv = previous_[1];
    policy.cuda_fa2_q_warp_groups = previous_[2];
    (void)cudaStreamDestroy(stream);
  }
  /** @brief Select only an existing compiled geometry before graph recording.
   */
  void select(int warps, int tile) {
    auto &policy = llaminar2::mutableDebugEnv().attention;
    policy.cuda_fa2_hd256_pv_warps = warps;
    policy.cuda_fa2_tile_kv = tile;
    policy.cuda_fa2_q_warp_groups = 1;
  }
  LaunchPolicy(const LaunchPolicy &) = delete;
  LaunchPolicy &operator=(const LaunchPolicy &) = delete;
  cudaStream_t stream = nullptr;

private:
  std::array<int, 3> previous_;
};

/** @brief The three native FA2 storage families; activations remain FP32. */
enum class KVFormat { FP32, FP16, BF16 };
} // namespace

/** @brief Prove every Q8 register geometry with empty/tail KV and TP head offsets. */
TEST(CUDAFlashAttentionRegisterLifetime,
     Q8DecodeAllHeadWidthsRetainCapturedArithmetic) {
  using llaminar2::Q8_1Block;
  using llaminar2::attention::AttentionDeviceParams;
  int devices = 0;
  checked(cudaGetDeviceCount(&devices));
  ASSERT_GT(devices, 0);
  checked(cudaSetDevice(0));
  LaunchPolicy policy;
  const auto stream = policy.stream;
  constexpr int batches = 2, heads = 4, kv_heads = 3, capacity = 289;
  constexpr int head_start = 2, gqa = 2, splits = 8, guard = 8;
  constexpr std::array lengths{0, 1, 7, 16, 17, 129, 273};
  for (int width = 32; width <= 256; width += 32) {
    SCOPED_TRACE(width);
    const int count = batches * heads * width;
    const int blocks_per_head = width / 32;
    const int kv_blocks = batches * capacity * kv_heads * blocks_per_head;
    Storage<float> q(count), output(count + guard),
        partial(count * splits), maxima(batches * heads * splits),
        sums(batches * heads * splits);
    Storage<Q8_1Block> k(kv_blocks), v(kv_blocks);
    Storage<AttentionDeviceParams> parameters(1);
    std::vector<float> query(count), eager(count + guard), actual(count + guard);
    std::vector<Q8_1Block> keys(kv_blocks), values(kv_blocks);
    for (int i = 0; i < count; ++i)
      query[i] = std::sin(i * 0.13f) * 0.3f;
    // Exact power-of-two scales make the independently decoded FP64 oracle
    // insensitive to host half-conversion implementation differences.
    for (int block = 0; block < kv_blocks; ++block) {
      keys[block].d = values[block].d = llaminar2::fp32_to_fp16(1.0f / 128);
      for (int j = 0; j < 32; ++j) {
        keys[block].qs[j] = (block * 13 + j * 7) % 251 - 125;
        values[block].qs[j] = (block * 17 + j * 11) % 247 - 123;
      }
    }
    checked(cudaMemcpyAsync(q.data, query.data(), count * sizeof(float),
                            cudaMemcpyHostToDevice, stream));
    checked(cudaMemcpyAsync(k.data, keys.data(), kv_blocks * sizeof(Q8_1Block),
                            cudaMemcpyHostToDevice, stream));
    checked(cudaMemcpyAsync(v.data, values.data(), kv_blocks * sizeof(Q8_1Block),
                            cudaMemcpyHostToDevice, stream));
    const auto launch = [&]() {
      return cudaFlashAttn_decode_q8_1(
          q.data, k.data, v.data, output.data, partial.data, maxima.data,
          sums.data, batches, capacity, heads, kv_heads, width, splits,
          parameters.data, stream, 0, head_start, gqa);
    };
    checked(cudaStreamSynchronize(stream));
    llaminar2::CUDAGraphCapture graph(stream, 0);
    llaminar2::ScopedBackendGraphCapture recording(graph, "Q8 register lifetime");
    ASSERT_TRUE(recording.begin());
    // Both valid and inactive partials start poisoned on every replay. A
    // changing logical length must never consume a stale captured split.
    checked(cudaMemsetAsync(partial.data, 0x5a, count * splits * sizeof(float), stream));
    checked(cudaMemsetAsync(output.data, 0x5a, (count + guard) * sizeof(float), stream));
    ASSERT_EQ(launch(), 0);
    recording.finish();
    ASSERT_TRUE(graph.instantiate());
    for (int replay = 0; replay < 20; ++replay) {
      const int length = lengths[replay % lengths.size()];
      SCOPED_TRACE(::testing::Message() << "replay=" << replay << " KV=" << length);
      const AttentionDeviceParams params{.kv_len = length, .kv_stride = capacity};
      checked(cudaMemcpyAsync(parameters.data, &params, sizeof(params),
                              cudaMemcpyHostToDevice, stream));
      checked(cudaMemsetAsync(output.data, 0x5a, (count + guard) * sizeof(float), stream));
      ASSERT_EQ(launch(), 0);
      checked(cudaMemcpyAsync(eager.data(), output.data, eager.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
      ASSERT_TRUE(graph.launch());
      checked(cudaMemcpyAsync(actual.data(), output.data, actual.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
      checked(cudaStreamSynchronize(stream));
      ASSERT_EQ(std::memcmp(eager.data(), actual.data(), actual.size() * sizeof(float)), 0);
      for (int i = count; i < count + guard; ++i)
        ASSERT_EQ(std::bit_cast<uint32_t>(actual[i]), 0x5a5a5a5aU);
      for (int batch = 0; batch < batches; ++batch)
        for (int head = 0; head < heads; ++head) {
          const int kv_head = (head_start + head) / gqa;
          const int query_offset = (batch * heads + head) * width;
          std::vector<double> numerators(width, 0.0);
          double denominator = 0;
          for (int key = 0; key < length; ++key) {
            const int block_base = ((batch * capacity + key) * kv_heads + kv_head) * blocks_per_head;
            double dot = 0;
            for (int d = 0; d < width; ++d)
              dot += static_cast<double>(query[query_offset + d]) *
                     keys[block_base + d / 32].qs[d % 32] / 128.0;
            const double probability = std::exp(dot / std::sqrt(static_cast<double>(width)));
            denominator += probability;
            for (int d = 0; d < width; ++d)
              numerators[d] += probability * values[block_base + d / 32].qs[d % 32] / 128.0;
          }
          for (int d = 0; d < width; ++d) {
            const double expected = length ? numerators[d] / denominator : 0.0;
            ASSERT_NEAR(actual[query_offset + d], expected, 2.0e-6);
          }
        }
    }
  }
}

/** @brief Cross all spilling specializations with poisoned twenty-replay byte
 * proofs. */
TEST(CUDAFlashAttentionRegisterLifetime,
     CapturedNativeFormatsAndWarpStripesAreByteExact) {
  int devices = 0;
  checked(cudaGetDeviceCount(&devices));
  ASSERT_GT(devices, 0);
  checked(cudaSetDevice(0));
  int shared_limit = 0;
  checked(cudaDeviceGetAttribute(&shared_limit,
                                 cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  constexpr int rows = 33, keys = 273, heads = 2, head_dim = 256;
  constexpr int outputs = rows * heads * head_dim;
  constexpr int kv_elements = keys * head_dim;
  constexpr int summaries = rows * heads * 2;
  LaunchPolicy policy;
  const auto stream = policy.stream;
  Storage<float> q(outputs), k(kv_elements), v(kv_elements),
      output(outputs + 8);
  Storage<float> partial(summaries * head_dim), maxima(summaries),
      sums(summaries);
  Storage<float> mask(rows * keys);
  std::vector<float> q_host(outputs), k_host(kv_elements), v_host(kv_elements);
  std::vector<float> mask_host(rows * keys);
  for (int i = 0; i < outputs; ++i)
    q_host[i] = std::sin(i * 0.13f) * 0.3f;
  for (int i = 0; i < kv_elements; ++i) {
    k_host[i] = std::cos(i * 0.07f) * 0.2f;
    v_host[i] = std::sin(i * 0.03f);
  }
  for (int row = 0; row < rows; ++row)
    for (int key = 0; key < keys; ++key)
      // One neutral microtile, finite biases, a partition tail, and query
      // rows whose entire second partition is still causally invisible.
      mask_host[row * keys + key] =
          key < 16 ? -1.0e30f : (key % 7 == 0 ? -0.125f : 0.0f);
  checked(cudaMemcpyAsync(q.data, q_host.data(), outputs * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
  checked(cudaMemcpyAsync(mask.data, mask_host.data(),
                          mask_host.size() * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
  std::vector<uint16_t> k_packed(kv_elements), v_packed(kv_elements);
  for (const auto format : {KVFormat::FP32, KVFormat::FP16, KVFormat::BF16}) {
    SCOPED_TRACE(static_cast<int>(format));
    for (int i = 0; i < kv_elements; ++i) {
      k_packed[i] = format == KVFormat::FP16
                        ? llaminar2::fp32_to_fp16(k_host[i])
                        : std::bit_cast<uint32_t>(k_host[i]) >> 16;
      v_packed[i] = format == KVFormat::FP16
                        ? llaminar2::fp32_to_fp16(v_host[i])
                        : std::bit_cast<uint32_t>(v_host[i]) >> 16;
    }
    const bool fp32 = format == KVFormat::FP32;
    const size_t bytes =
        kv_elements * (fp32 ? sizeof(float) : sizeof(uint16_t));
    checked(cudaMemcpyAsync(
        k.data, fp32 ? static_cast<void *>(k_host.data()) : k_packed.data(),
        bytes, cudaMemcpyHostToDevice, stream));
    checked(cudaMemcpyAsync(
        v.data, fp32 ? static_cast<void *>(v_host.data()) : v_packed.data(),
        bytes, cudaMemcpyHostToDevice, stream));
    checked(cudaStreamSynchronize(stream));
    std::vector<float> expected, actual(outputs + 8);
    for (int warps : {4, 2, 1})
      for (int tile : {16, 32, 64})
        for (bool context : {false, true}) {
          SCOPED_TRACE(::testing::Message() << "PV=" << warps << " tile="
                                            << tile << " context=" << context);
          policy.select(warps, tile);
          const auto launch = [&]() -> int {
            if (context) {
              if (fp32)
                return cudaFlashAttn_prefill_fa2_context_parallel(
                    q.data, k.data, v.data, output.data, partial.data,
                    maxima.data, sums.data, 1, rows, keys, heads, 1, head_dim,
                    true, -1, keys - rows, nullptr, mask.data, 256, 2, 2, 0, 1,
                    nullptr, stream, 0, 0, 0);
              auto fn = format == KVFormat::FP16
                            ? cudaFlashAttn_prefill_fa2_fp16kv_context_parallel
                            : cudaFlashAttn_prefill_fa2_bf16kv_context_parallel;
              return fn(q.data, k.data, v.data, output.data, partial.data,
                        maxima.data, sums.data, 1, rows, keys, heads, 1,
                        head_dim, true, -1, keys - rows, nullptr, mask.data,
                        256, 2, 2, 0, 1, nullptr, stream, 0, 0, 0);
            }
            if (fp32)
              return cudaFlashAttn_prefill_fa2(
                  q.data, k.data, v.data, output.data, 1, rows, keys, heads, 1,
                  head_dim, true, -1, keys - rows, nullptr, mask.data, stream,
                  0, 0, 0);
            auto fn = format == KVFormat::FP16
                          ? cudaFlashAttn_prefill_fa2_fp16kv
                          : cudaFlashAttn_prefill_fa2_bf16kv;
            return fn(q.data, k.data, v.data, output.data, 1, rows, keys, heads,
                      1, head_dim, true, -1, keys - rows, nullptr, mask.data,
                      stream, 0, 0, 0);
          };
          // Prime function attributes before capture; all proof runs
          // below then execute this exact retained production graph.
          // The compiled 64-key HD256 tile needs more shared memory
          // than GA102 owns. Prove exact admission failure there;
          // on a larger device it joins the ordinary captured proof.
          if (llaminar2::cuda::fa2_policy::fa2DynamicSharedMemoryBytes(
                  head_dim, 1, tile) > static_cast<size_t>(shared_limit)) {
            ASSERT_EQ(launch(), -1);
            continue;
          }
          ASSERT_EQ(launch(), 0);
          checked(cudaStreamSynchronize(stream));
          llaminar2::CUDAGraphCapture graph(stream, 0);
          llaminar2::ScopedBackendGraphCapture recording(
              graph, "FA2 register lifetime");
          ASSERT_TRUE(recording.begin());
          checked(cudaMemsetAsync(output.data, 0x5a,
                                  (outputs + 8) * sizeof(float), stream));
          ASSERT_EQ(launch(), 0);
          recording.finish();
          ASSERT_TRUE(graph.instantiate());
          for (int replay = 0; replay < 20; ++replay) {
            ASSERT_TRUE(graph.launch());
            checked(cudaMemcpyAsync(actual.data(), output.data,
                                    actual.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));
            checked(cudaStreamSynchronize(stream));
            if (expected.empty())
              expected.assign(actual.begin(), actual.begin() + outputs);
            ASSERT_EQ(std::memcmp(expected.data(), actual.data(),
                                  outputs * sizeof(float)),
                      0)
                << "replay=" << replay;
            for (int i = 0; i < outputs; ++i)
              ASSERT_TRUE(std::isfinite(actual[i]));
            for (int i = outputs; i < outputs + 8; ++i)
              ASSERT_EQ(std::bit_cast<uint32_t>(actual[i]), 0x5a5a5a5aU);
          }
        }
  }
}
