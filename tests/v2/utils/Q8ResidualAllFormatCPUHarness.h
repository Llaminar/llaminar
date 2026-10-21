/**
 * @file Q8ResidualAllFormatCPUHarness.h
 * @brief Actual prepared CPU stage fixture shared by tiny Unit and larger
 * integration cases.
 *
 * Every expected numerical value comes from original/native coefficient
 * decoding and independent FP64 arithmetic or a separately invoked production
 * M1 oracle. No alternate production path, serial replay implementation or
 * source policy is supplied.
 */
#pragma once
#include "CPUProjectionTestWorkspace.h"
#include "PreparedWeightTestHarness.h"
#include "Q8ResidualAllFormatOracle.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include <gtest/gtest.h>
#include <iostream>
#include <omp.h>
namespace llaminar2::test::q8_all_formats {
/** @brief Restore the invoking test's exact OpenMP policy even after assertion
 * unwind. */
struct Threads {
  int threads = omp_get_max_threads(), dynamic = omp_get_dynamic();
  explicit Threads(int n) {
    if (n <= 0)
      throw std::invalid_argument("Positive CPU test budget required");
    omp_set_dynamic(0);
    omp_set_num_threads(n);
  }
  ~Threads() {
    omp_set_num_threads(threads);
    omp_set_dynamic(dynamic);
  }
};
/** @brief Actual stage, source/prepared owners and two immutable row
 * geometries. */
struct CPUFixture {
  int m, n, k, count;
  std::unique_ptr<TensorBase> weight;
  std::vector<uint8_t> raw;
  std::vector<double> original, prepared;
  PreparedGemmFixture binding;
  ITensorGemm *kernel;
  FP32Tensor input, output, baseline, rowInput, rowOutput;
  std::vector<float> values;
  std::unique_ptr<CPUProjectionTestWorkspace> workspace;
  std::unique_ptr<GEMMStage> stage;
  /** @brief Prepare the original source once and pay only actual named
   * stage/ordinary scratch. */
  CPUFixture(const QuantizedVerifierFormatCase &format, int rows, int columns,
             int reduction, bool tile)
      : m(rows), n(columns), k(reduction), count(rows),
        weight(source(format, n, k)), raw(originalBytes(*weight)),
        original(originalWeights(*weight)),
        binding(makePreparedGemmFixture(weight.get(), DeviceId::cpu(),
                                        "blk.2.hc_ffn_up.weight",
                                        ModelContextId{871001})),
        kernel(binding.store->gemmKernel(binding.ref)),
        input({std::size_t(m), std::size_t(k)}, DeviceId::cpu()),
        output({std::size_t(m), std::size_t(n)}, DeviceId::cpu()),
        baseline({std::size_t(m), std::size_t(n)}, DeviceId::cpu()),
        rowInput({1, std::size_t(k)}, DeviceId::cpu()),
        rowOutput({1, std::size_t(n)}, DeviceId::cpu()),
        values(q8_all_formats::input(m, k, tile)) {
    if (!kernel)
      throw std::runtime_error("Actual prepared CPU engine absent");
    const auto *packed = kernel->exportCPUNativeVNNIPackedWeights();
    if (!packed)
      throw std::runtime_error("Actual CPU native source export absent");
    prepared = cpuWeights(*packed);
    std::copy(values.begin(), values.end(), input.mutable_data());
    stage = std::make_unique<GEMMStage>(GEMMStage::Params{
        .device_id = DeviceId::cpu(),
        .A = &input,
        .B = weight.get(),
        .C = &output,
        .m = m,
        .n = n,
        .k = k,
        .a_buffer_id = BufferId::NORMALIZED,
        .c_buffer_id = BufferId::GATE_PROJ,
        .force_decode_equivalent_verifier_prefill = true,
        .verifier_row_range = DeviceRowRange::deviceCounted(m, &count),
        .prepared_ref = binding.ref,
        .prepared_store = binding.store.get(),
        .quantized_operand_precision =
            QuantizedActivationOperandPrecision::TwoTermQ8});
    auto requirements = stage->getWorkspaceRequirements(m, n, k);
    auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
    if (!consumer)
      throw std::runtime_error(
          "Actual prepared CPU workspace declaration absent");
    requirements.merge(consumer->getWorkspaceRequirements(m, n, k));
    workspace = std::make_unique<CPUProjectionTestWorkspace>(requirements);
    stage->bindWorkspace(workspace->get());
  }
  /** @brief End the exact stage borrow before arena or immutable engine
   * retirement. */
  ~CPUFixture() {
    if (stage)
      stage->unbindWorkspace();
  }
};

/** @brief Freeze independent M1 outputs/banks once per distinct authored input.
 */
struct CPUOracle {
  std::vector<float> output;
  std::vector<Q8_1Block> primary, residual;
  int rows = 0;
};
/**
 * @brief Check every unique native M1 contribution against independent FP64
 * math.
 * @param f Original source/prepared owner and ordinary primary baseline.
 * @param expected Frozen serial observations, never an inference
 * implementation.
 */
inline void checkCPUOracle(const CPUFixture &f, const CPUOracle &expected) {
  const auto banks = cpuBanks(expected.primary.data(), expected.residual.data(),
                              expected.rows, f.k);
  const auto contribution =
      project(f.prepared, f.n, f.k, banks, expected.rows, true);
  const auto unquantized = exact(f.prepared, f.n, f.k, f.values, expected.rows);
  const auto original = exact(f.original, f.n, f.k, f.values, expected.rows);
  double preparedError = 0, sourceError = 0, singleError = 0;
  for (int r = 0; r < expected.rows; ++r)
    for (int c = 0; c < f.n; ++c) {
      const auto i = std::size_t(r) * f.n + c;
      ASSERT_TRUE(std::isfinite(expected.output[i]));
      EXPECT_NEAR(double(expected.output[i]), contribution[i],
                  roundingBound(f.prepared, c, f.k, banks, r));
      EXPECT_NEAR(double(expected.output[i]), original[i],
                  originalProjectionBound(f.original, f.prepared, c, f.k, banks,
                                          r, f.values));
      preparedError += std::pow(double(expected.output[i]) - unquantized[i], 2);
      singleError += std::pow(double(f.baseline.data()[i]) - unquantized[i], 2);
      sourceError += std::pow(double(expected.output[i]) - original[i], 2);
    }
  // This outlier witness isolates activation correction. Original-source
  // packing error is retained separately and is never treated as an exact bit
  // oracle.
  EXPECT_GT(singleError, 0);
  EXPECT_LT(preparedError, singleError / 16);
  std::cout << "Q8_TWO_TERM_ORACLE scope=unique_actual_prepared_cpu_rows "
               "source_format="
            << int(f.weight->native_type()) << " unique_rows=" << expected.rows
            << " original_fp64_squared_error=" << sourceError
            << " prepared_fp64_squared_error=" << preparedError
            << " single_prepared_squared_error=" << singleError << "\n";
}
/**
 * @brief Reuse one actual prepared source/arena across independent positive
 * teams.
 * @param teams Explicit positive worker counts, all within the original caller
 * budget.
 * @param tile True for the separately labeled one-row tiled workload.
 * @param requestedLives Optional live-prefix sequence; physical geometry is
 * unchanged.
 *
 * Preparation and FP64/M1 oracles run once. Every team still executes every
 * physical grouped launch, live row and poisoned suffix. Large repeated rows
 * therefore test actual dispatch without repeating identical diagnostic GEMVs.
 */
inline void proveCPUThreads(const QuantizedVerifierFormatCase &format, int m,
                            int n, int k, std::span<const int> teams,
                            bool tile = false,
                            std::span<const int> requestedLives = {}) {
  SCOPED_TRACE(format.label);
  SCOPED_TRACE(::testing::Message()
               << "M=" << m << " N=" << n << " K=" << k << " tile=" << tile);
  ASSERT_FALSE(teams.empty());
  ASSERT_TRUE(
      std::all_of(teams.begin(), teams.end(), [](int n) { return n > 0; }));
  Threads capacity(*std::max_element(teams.begin(), teams.end()));
  CPUFixture f(format, m, n, k, tile);
  const auto *packed = f.kernel->exportCPUNativeVNNIPackedWeights();
  ASSERT_NE(packed, nullptr);
  EXPECT_EQ(packed->codebook_id, format.source_codebook_id);
  EXPECT_EQ(packed->is_superblock, format.source_is_superblock);
  const auto *sourcePointer = packed->interleavedBase();
  const std::vector<uint8_t> nativeBefore(
      sourcePointer,
      sourcePointer + cpu::native_vnni::interleavedWorkspaceSize(*packed));
  const auto signature = f.stage->graphCaptureVariantSignature();
  ASSERT_TRUE(f.kernel->multiply_tensor(&f.input, &f.baseline, m, n, k, true, 1,
                                        0, nullptr, nullptr, -1,
                                        f.workspace->get()));
  auto *primary =
      static_cast<Q8_1Block *>(f.workspace->get()->getBuffer(kCPUProjectionQ8));
  auto *residual = static_cast<Q8_1Block *>(
      f.workspace->get()->getBuffer(kCPUProjectionResidualQ8));
  ASSERT_NE(primary, nullptr);
  ASSERT_NE(residual, nullptr);
  const auto rowBlocks = std::size_t(k) / 32,
             blocks = std::size_t(m) * rowBlocks;
  const std::vector<Q8_1Block> ordinary(primary, primary + blocks);
  CPUOracle expected;
  expected.rows = tile ? 1 : std::min(m, 7);
  expected.output.resize(std::size_t(expected.rows) * n);
  expected.primary.resize(std::size_t(expected.rows) * rowBlocks);
  expected.residual.resize(expected.primary.size());
  // This proof permits bounded oracle reuse; no approximate row match is
  // accepted.
  for (int r = 0; r < m; ++r)
    ASSERT_EQ(std::memcmp(f.values.data() + std::size_t(r) * k,
                          f.values.data() + std::size_t(r % expected.rows) * k,
                          std::size_t(k) * sizeof(float)),
              0);
  {
    Threads serialTeam(1);
    for (int r = 0; r < expected.rows; ++r) {
      std::copy_n(f.values.data() + std::size_t(r) * k, k,
                  f.rowInput.mutable_data());
      f.kernel->multiply_q8_residual_corrected(&f.rowInput, &f.rowOutput,
                                               DeviceRowRange::fullyActive(1),
                                               n, k, f.workspace->get());
      std::copy_n(f.rowOutput.data(), n,
                  expected.output.data() + std::size_t(r) * n);
      std::copy_n(primary, rowBlocks,
                  expected.primary.data() + std::size_t(r) * rowBlocks);
      std::copy_n(residual, rowBlocks,
                  expected.residual.data() + std::size_t(r) * rowBlocks);
    }
  }
  checkCPUOracle(f, expected);
  ASSERT_FALSE(::testing::Test::HasFailure());
  std::vector<int> liveRows(requestedLives.begin(), requestedLives.end());
  if (liveRows.empty())
    liveRows = {m, std::min(m, 9), std::min(m, 1), 0, std::min(m, 15), m};
  ASSERT_TRUE(std::all_of(liveRows.begin(), liveRows.end(),
                          [=](int v) { return v >= 0 && v <= m; }));
  for (int team : teams) {
    SCOPED_TRACE(team);
    Threads active(team);
    CPUDeviceContext context(DeviceId::cpu(), team);
    for (int live : liveRows) {
      SCOPED_TRACE(live);
      f.count = live;
      std::copy(f.values.begin(), f.values.end(), f.input.mutable_data());
      std::fill(f.input.mutable_data() + std::size_t(live) * k,
                f.input.mutable_data() + f.input.numel(),
                std::numeric_limits<float>::quiet_NaN());
      std::fill_n(f.output.mutable_data(), f.output.numel(), -12345.f);
      std::memset(primary, 0x5a, blocks * sizeof(*primary));
      std::memset(residual, 0x5a, blocks * sizeof(*residual));
      ASSERT_TRUE(f.stage->execute(&context));
      EXPECT_EQ(std::memcmp(primary, ordinary.data(),
                            std::size_t(live) * rowBlocks * sizeof(Q8_1Block)),
                0);
      // Decode every actual bank to independently verify primary/residual
      // sum_qs.
      (void)cpuBanks(primary, residual, live, k);
      for (int r = 0; r < live; ++r) {
        const auto frozen = r % expected.rows;
        ASSERT_EQ(std::memcmp(f.output.data() + std::size_t(r) * n,
                              expected.output.data() + std::size_t(frozen) * n,
                              std::size_t(n) * sizeof(float)),
                  0)
            << "row=" << r;
        ASSERT_EQ(std::memcmp(primary + std::size_t(r) * rowBlocks,
                              expected.primary.data() +
                                  std::size_t(frozen) * rowBlocks,
                              rowBlocks * sizeof(Q8_1Block)),
                  0)
            << "primary row=" << r;
        ASSERT_EQ(std::memcmp(residual + std::size_t(r) * rowBlocks,
                              expected.residual.data() +
                                  std::size_t(frozen) * rowBlocks,
                              rowBlocks * sizeof(Q8_1Block)),
                  0)
            << "residual row=" << r;
      }
      for (std::size_t i = std::size_t(live) * n; i < f.output.numel(); ++i)
        ASSERT_EQ(f.output.data()[i], -12345.f);
      const auto *a = reinterpret_cast<const uint8_t *>(primary),
                 *b = reinterpret_cast<const uint8_t *>(residual);
      for (std::size_t i = std::size_t(live) * rowBlocks * sizeof(Q8_1Block);
           i < blocks * sizeof(Q8_1Block); ++i) {
        ASSERT_EQ(a[i], 0x5a);
        ASSERT_EQ(b[i], 0x5a);
      }
      EXPECT_EQ(f.stage->graphCaptureVariantSignature(), signature);
      EXPECT_EQ(packed->interleavedBase(), sourcePointer);
      ASSERT_FALSE(::testing::Test::HasFailure());
    }
  }
  EXPECT_TRUE(
      std::equal(nativeBefore.begin(), nativeBefore.end(), sourcePointer));
  EXPECT_EQ(originalBytes(*f.weight), f.raw);
  std::copy(f.values.begin(), f.values.end(), f.input.mutable_data());
  FP32Tensor again({std::size_t(m), std::size_t(n)}, DeviceId::cpu());
  ASSERT_TRUE(f.kernel->multiply_tensor(&f.input, &again, m, n, k, true, 1, 0,
                                        nullptr, nullptr, -1,
                                        f.workspace->get()));
  EXPECT_EQ(std::memcmp(again.data(), f.baseline.data(),
                        again.numel() * sizeof(float)),
            0);
}
/** @brief Compatibility wrapper for tiny Unit and one-team component clients.
 */
inline void proveCPU(const QuantizedVerifierFormatCase &format, int m, int n,
                     int k, int workers, bool tile = false) {
  const std::array teams{workers};
  proveCPUThreads(format, m, n, k, teams, tile);
}
} // namespace llaminar2::test::q8_all_formats
