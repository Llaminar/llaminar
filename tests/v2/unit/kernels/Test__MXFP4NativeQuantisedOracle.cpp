/**
 * @file Test__MXFP4NativeQuantisedOracle.cpp
 * @brief OCP-spec reference oracle for MXFP4 through the CPU NativeQuantised kernel.
 *
 * The oracle is written against the OCP MXFP4 definition directly: each value
 * is sign * magnitude * 2^(e-127) with magnitude in {0, 0.5, 1, 1.5, 2, 3, 4, 6}
 * selected by the three-bit E2M1 field (bit 3 is the sign). It deliberately
 * does not reuse mxfp4::kValues or the implemented scale helper, so a doubled
 * table or a biased exponent shift in the implementation cannot silently match.
 *
 * Coverage: each exponent is exercised in isolation (minimum 2, mid-band, and
 * the admission maximum 252 from MXFP4Format.h) with a power-of-two activation
 * scale chosen so every output is finite and normal; distinct codes per row
 * and distinct activations per grouped row. Kernel outputs must be finite,
 * nonzero and within 2e-6 of an exact double-precision oracle; grouped rows
 * must be byte-identical to serial rows. Also: the all-zero zero marker and
 * rejection of 1/253/254/255.
 */
#include "kernels/cpu/gemm/CPUNativeQuantisedExecutionContract.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemv.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h"
#include "tensors/BlockStructures.h"
#include "tensors/TensorClasses.h"
#include "tensors/MXFP4Format.h"
#include "../../utils/NativeQuantisedTestPartialStorage.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

namespace llaminar2::test
{
namespace
{
using cpu::native_quantised::CPUNativeQuantisedPackedWeights;
using cpu::native_quantised::ISAPath;
using cpu::native_quantised::DecodeSchedulePolicy;
using cpu::native_quantised::VerifierRowsPolicy;

/** @brief OCP E2M1 magnitudes for the three magnitude bits. */
constexpr float kOCPE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};


/**
 * @brief Build a raw MXFP4 matrix with one exponent and row-distinct codes.
 * @param e E8M0 exponent of every block.
 * @param n Matrix rows (output width N of the projection).
 * @param k Matrix columns (reduction width K).
 * @return Row-major blocks with nonzero magnitudes (codes 1..7) and one sign
 *         per row (negative on odd rows), so no column dot can cancel to zero;
 *         patterns shift with row and block so columns differ.
 */
std::vector<uint8_t> rawMatrix(uint8_t e, int n, int k)
{
    const int blocks_per_row = (k + MXFP4Block::BLOCK_SIZE - 1) / MXFP4Block::BLOCK_SIZE;
    std::vector<uint8_t> raw(static_cast<size_t>(n) * blocks_per_row * sizeof(MXFP4Block));
    auto *out = reinterpret_cast<MXFP4Block *>(raw.data());
    for (int row = 0; row < n; ++row)
        for (int kb = 0; kb < blocks_per_row; ++kb)
        {
            const size_t index = static_cast<size_t>(row) * blocks_per_row + kb;
            out[index].e = e;
            for (int j = 0; j < 16; ++j)
            {
                const int sign = (row & 1) ? 8 : 0;
                const int low = ((j + row + kb) % 7 + 1) | sign;
                const int high = ((5 * j + 3 * row + kb) % 7 + 1) | sign;
                out[index].qs[j] = static_cast<uint8_t>(low | (high << 4));
            }
        }
    return raw;
}

/**
 * @brief Power-of-two FP16 activation scale keeping exponent @p e finite.
 * @return 2^p with p in [-14, 15] (normal FP16) such that 2^(e-128+p) times a
 *         bounded integer dot stays a normal FP32 value.
 */
int activationExponentFor(uint8_t e) noexcept
{
    return std::clamp(128 - int(e), -14, 15);
}

/**
 * @brief One Q8 activation row: small positive values and a power-of-two scale.
 * @param k Reduction width.
 * @param row Grouped row index (0..3); shifts every value so rows differ.
 * @param scale_exponent Activation scale 2^scale_exponent (normal FP16).
 * @details 1 <= q <= 6 keeps each column dot nonzero and below 6*6*K, so the
 *          exponent-252 case stays below 2^123 and the exponent-2 case above
 *          2^-112: finite, normal FP32 outputs at both admission bounds.
 */
std::vector<Q8_1Block> activationRow(int k, int row, int scale_exponent)
{
    std::vector<Q8_1Block> blocks((k + 31) / 32);
    for (size_t b = 0; b < blocks.size(); ++b)
    {
        blocks[b].d = static_cast<uint16_t>((scale_exponent + 15) << 10);
        int32_t sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            blocks[b].qs[i] = static_cast<int8_t>((i * 7 + int(b) * 3) % 3 + 1 + row);
            sum += blocks[b].qs[i];
        }
        blocks[b].sum_qs = static_cast<int16_t>(sum);
    }
    return blocks;
}

/**
 * @brief Independent oracle in double: sum_k OCP(code, e) * q[k] * d per column.
 * @details Uses OCP magnitudes and 2^(e-127) only; exact for these fixtures.
 */
std::vector<double> oracleOutput(
    const MXFP4Block *blocks, int n, int k, const Q8_1Block *act, int scale_exponent)
{
    std::vector<double> out(n, 0.0);
    const int blocks_per_row = (k + 31) / 32;
    for (int col = 0; col < n; ++col)
        for (int kb = 0; kb < blocks_per_row; ++kb)
        {
            const MXFP4Block &block = blocks[col * blocks_per_row + kb];
            for (int lane = 0; lane < 32; ++lane)
            {
                // Low half: nibbles of byte lane; high half: byte lane-16.
                const int code = lane < 16 ? block.qs[lane] & 15 : block.qs[lane - 16] >> 4;
                const double value = block.e == 0 ? 0.0
                    : ((code & 8) ? -1.0 : 1.0) * double(kOCPE2M1[code & 7]) * std::ldexp(1.0, int(block.e) - 127);
                out[col] += value * double(act[kb].qs[lane]) * std::ldexp(1.0, scale_exponent);
            }
        }
    return out;
}

/** @brief Exponents exercised in isolation: minimum, FP16-exact band, maximum. */
constexpr std::array<uint8_t, 7> kIsolatedExponents{2, 64, 104, 128, 143, 200, mxfp4::kMaxNativeE8M0Exponent};

/** @brief Prepare through the production CPU packer; returns success, logs nothing. */
bool prepareWeights(const TensorBase *tensor, CPUNativeQuantisedPackedWeights *packed)
{
    try
    {
        return packWeightsCPUNativeQuantised(tensor, *packed);
    }
    catch (const std::invalid_argument &)
    {
        return false;
    }
    catch (const std::runtime_error &)
    {
        return false;
    }
}

/** @brief Construct and prepare one matrix; any admission throw is a rejection. */
bool prepareRaw(const std::vector<uint8_t> &raw, CPUNativeQuantisedPackedWeights *packed)
{
    try
    {
        auto weights = std::make_unique<MXFP4Tensor>(
            std::vector<size_t>{32, 32}, raw);
        return prepareWeights(weights.get(), packed);
    }
    catch (const std::invalid_argument &)
    {
        return false;
    }
    catch (const std::runtime_error &)
    {
        return false;
    }
}
} // namespace

/** @test The OCP oracle and the production kernel agree on each admitted exponent in isolation. */
TEST(MXFP4NativeQuantisedOracle, KernelMatchesOCPReferenceAcrossAdmittedExponents)
{
    constexpr int n = 64;
    constexpr int k = 128;
    for (const uint8_t e : kIsolatedExponents)
    {
        SCOPED_TRACE("e=" + std::to_string(int(e)));
        const auto raw = rawMatrix(e, n, k);
        auto weights = std::make_unique<MXFP4Tensor>(
            std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, raw);
        CPUNativeQuantisedPackedWeights packed;
        ASSERT_TRUE(prepareWeights(weights.get(), &packed));

        const int scale_exponent = activationExponentFor(e);
        const auto act = activationRow(k, 0, scale_exponent);
        std::vector<float> output(n, 0.0f);
        NativeQuantisedTestPartialStorage partial(packed, 1);
        gemv_native_quantised_preq(packed, act.data(), output.data(), partial.span(),
            ISAPath::AUTO, DecodeSchedulePolicy::Auto);

        const auto expected = oracleOutput(
            reinterpret_cast<const MXFP4Block *>(raw.data()), n, k, act.data(), scale_exponent);
        int distinct = 0;
        for (int col = 0; col < n; ++col)
        {
            ASSERT_TRUE(std::isnormal(expected[col]) && std::isnormal(float(expected[col])))
                << "fixture must keep column " << col << " finite and normal: " << expected[col];
            EXPECT_TRUE(std::isfinite(output[col])) << "column " << col;
            EXPECT_LE(std::fabs(double(output[col]) - expected[col]), 2e-6 * std::fabs(expected[col]))
                << "column " << col << " kernel=" << output[col] << " oracle=" << expected[col];
            distinct += col > 0 && output[col] != output[col - 1];
        }
        EXPECT_GT(distinct, n / 2) << "row-distinct codes must yield distinct columns";
    }
}

/** @test The zero-marker block contributes exactly zero in every position. */
TEST(MXFP4NativeQuantisedOracle, ZeroMarkerBlockStaysZeroThroughTheKernel)
{
    std::vector<uint8_t> raw(static_cast<size_t>(32) * sizeof(MXFP4Block), 0);
    constexpr int n = 32;
    constexpr int k = 32;
    auto weights = std::make_unique<MXFP4Tensor>(
        std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, raw);
    CPUNativeQuantisedPackedWeights packed;
    ASSERT_TRUE(prepareWeights(weights.get(), &packed));

    const auto act = activationRow(k, 0, 0);
    std::vector<float> output(n, 123.0f);
    NativeQuantisedTestPartialStorage partial(packed, 1);
    gemv_native_quantised_preq(packed, act.data(), output.data(), partial.span(),
        ISAPath::AUTO, DecodeSchedulePolicy::Auto);
    for (float value : output)
        EXPECT_EQ(value, 0.0f) << "zero-marker block must publish exact zero";
}

/** @test Both sides of the E8M0 admission boundary; 1/253/254/255 are rejected. */
TEST(MXFP4NativeQuantisedOracle, AdmissionContractAcceptsZeroMarkerAnd252Only)
{
    struct AdmittedCase
    {
        uint8_t e;
        bool admitted;
    };
    const std::array<AdmittedCase, 11> cases{{
        {0, true},   // zero marker (all-zero codes)
        {1, false},  // FP32 subnormal scale: rejected
        {2, true},   // minimum admitted
        {103, true}, {104, true}, {128, true}, {143, true},
        {252, true}, // derived maximum: kMaxNativeE8M0Exponent
        {253, false}, {254, false}, {255, false},
    }};

    for (const auto &entry : cases)
    {
        SCOPED_TRACE("e=" + std::to_string(int(entry.e)));
        constexpr int shape_n = 32;
        constexpr int shape_k = 32;
        std::vector<uint8_t> raw(static_cast<size_t>(shape_n) * sizeof(MXFP4Block));
        std::fill(raw.begin(), raw.end(), 0);
        if (entry.e != 0)
        {
            auto *blocks = reinterpret_cast<MXFP4Block *>(raw.data());
            for (int row = 0; row < shape_n; ++row)
            {
                blocks[row].e = entry.e;
                for (int j = 0; j < 16; ++j)
                    blocks[row].qs[j] = static_cast<uint8_t>(j | (j << 4));
            }
        }
        if (entry.admitted)
        {
            // Valid blocks prepare through the production packer.
            CPUNativeQuantisedPackedWeights packed;
            EXPECT_TRUE(prepareRaw(raw, &packed))
                << "admitted exponent must prepare: e=" << int(entry.e);
            continue;
        }
        // A rejected exponent is fatal inside the packer's OpenMP parallel
        // region (repo maxim 29: fail fast and fatally), so it cannot be
        // caught by a unit test. The catchable contract is the admission
        // predicate itself plus the load-time screen, both consumed by the
        // packer and the loader.
        const std::array<uint8_t, 16> nonzero{{
            0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
            0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01}};
        EXPECT_FALSE(mxfp4::blockFitsNativeE8M0(entry.e, nonzero.data()))
            << "predicate must reject e=" << int(entry.e);
        EXPECT_NE(mxfp4::firstInadmissibleNativeE8M0Block(raw.data(), raw.size()),
                  raw.size())
            << "load-time screen must flag e=" << int(entry.e);
    }
}

/** @test Grouped decode rows stay byte-identical to finite serial M=1 rows at every exponent. */
TEST(MXFP4NativeQuantisedOracle, GroupedRowsAreByteIdenticalToSerialRows)
{
    constexpr int n = 64;
    constexpr int k = 64;
    constexpr int m = 4;
    constexpr int blocks_per_row = (k + 31) / 32;
    for (const uint8_t e : kIsolatedExponents)
    {
        SCOPED_TRACE("e=" + std::to_string(int(e)));
        const auto raw = rawMatrix(e, n, k);
        auto weights = std::make_unique<MXFP4Tensor>(
            std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, raw);
        CPUNativeQuantisedPackedWeights packed;
        ASSERT_TRUE(prepareWeights(weights.get(), &packed));

        const int scale_exponent = activationExponentFor(e);
        std::vector<Q8_1Block> act(static_cast<size_t>(m) * blocks_per_row);
        for (int row = 0; row < m; ++row)
        {
            const auto row_act = activationRow(k, row, scale_exponent);
            std::copy(row_act.begin(), row_act.end(), act.begin() + row * blocks_per_row);
        }

        std::vector<float> serial(static_cast<size_t>(m) * n, 0.0f);
        NativeQuantisedTestPartialStorage serial_partial(packed, 1);
        for (int row = 0; row < m; ++row)
            gemv_native_quantised_preq(packed, act.data() + row * blocks_per_row,
                serial.data() + row * n, serial_partial.span(), ISAPath::AUTO, DecodeSchedulePolicy::Auto);

        std::vector<float> grouped(static_cast<size_t>(m) * n, 0.0f);
        NativeQuantisedTestPartialStorage grouped_partial(packed, m);
        gemm_native_quantised_preq_decode_equivalent_rows(packed, act.data(), grouped.data(),
            grouped_partial.span(), m, n, ISAPath::AUTO, VerifierRowsPolicy::Auto,
            /*publish_verifier_route=*/false);

        for (int row = 0; row < m; ++row)
        {
            const auto expected = oracleOutput(reinterpret_cast<const MXFP4Block *>(raw.data()), n, k,
                act.data() + row * blocks_per_row, scale_exponent);
            for (int col = 0; col < n; ++col)
            {
                const float value = serial[static_cast<size_t>(row) * n + col];
                ASSERT_TRUE(std::isnormal(value)) << "row " << row << " column " << col << " = " << value;
                EXPECT_LE(std::fabs(double(value) - expected[col]), 2e-6 * std::fabs(expected[col]));
            }
            if (row > 0)
                EXPECT_NE(std::memcmp(serial.data() + row * n, serial.data() + (row - 1) * n, n * sizeof(float)), 0)
                    << "grouped rows must differ so byte identity is meaningful";
        }
        EXPECT_EQ(std::memcmp(grouped.data(), serial.data(), static_cast<size_t>(m) * n * sizeof(float)), 0)
            << "grouped decode rows must be byte-identical to serial rows";
    }
}

} // namespace llaminar2::test
