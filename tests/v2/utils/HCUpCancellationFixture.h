/**
 * @file HCUpCancellationFixture.h
 * @brief Small authored native-Q8 source and independent HC gate/read oracle.
 *
 * Paired positive activations multiply opposite signed source bytes. Their
 * small nonzero difference is sensitive to activation quantization although
 * the original weight decode is exact. Nothing is copied from model payloads.
 * The oracle decodes the authored original Q8_0 bytes directly in FP64; it does
 * not reproduce either native activation quantizer or integer-dot reduction.
 */
#pragma once
#include "tensors/Tensors.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace llaminar2::test::hc_up_cancellation
{
inline constexpr int kColumns = 320; ///< Ten original32-element Q8_0 blocks.
inline constexpr float kSentinel = -12345.f; ///< Diagnostic inactive destination.
// Motif-specific absolute bounds verified on CPU, CUDA and ROCm.
// Each exact block contributes1/8; no relative ratio or model tolerance is used.
inline constexpr double kLogitAbsoluteError = .02;
inline constexpr double kReadAbsoluteError = .006;
inline constexpr double kSingleReadErrorFloor = .1;

/**
 * @brief Author native source bytes with opposite signed large contributions.
 * @param outputs Four complete equal residual branches.
 * @return Unmodified Q8_0 source blocks; production packers consume these bytes.
 * @throws std::invalid_argument For incomplete residual branch geometry.
 */
inline std::vector<std::uint8_t> weights(int outputs)
{
    if (outputs <= 0 || outputs % 4 != 0)
        throw std::invalid_argument("HC cancellation needs four complete branches");
    std::vector<std::uint8_t> raw(std::size_t(outputs) * kColumns / 32 * sizeof(Q8_0Block));
    for (int column = 0; column < outputs; ++column)
    {
        const int sign = column / (outputs / 4) < 2 ? 1 : -1;
        for (int block = 0; block < kColumns / 32; ++block)
        {
            Q8_0Block source{};
            source.d = fp32_to_fp16(1.f / 64.f);
            source.qs[0] = static_cast<std::int8_t>(sign * 64);
            source.qs[1] = static_cast<std::int8_t>(-sign * 64);
            source.qs[2] = static_cast<std::int8_t>(sign * 32);
            std::memcpy(raw.data() + (std::size_t(column) * (kColumns / 32) + block) * sizeof(source),
                        &source, sizeof(source));
        }
    }
    return raw;
}

/**
 * @brief Author positive low-rank inputs; dyadic row scaling varies live rows.
 * @param rows Positive physical capacity, independent of replay live counts.
 * @return Original FP32 input bank, not quantized activation bytes.
 * @throws std::invalid_argument For a nonpositive capacity.
 */
inline std::vector<float> activations(int rows)
{
    if (rows <= 0)
        throw std::invalid_argument("HC cancellation needs positive physical rows");
    std::vector<float> values(std::size_t(rows) * kColumns);
    for (int row = 0; row < rows; ++row)
    {
        const float scale = 1.f + float(row) / 16.f;
        for (int block = 0; block < kColumns / 32; ++block)
        {
            auto *x = values.data() + std::size_t(row) * kColumns + block * 32;
            x[0] = 32.f * scale;
            x[1] = 31.9375f * scale;
            x[2] = .125f * scale;
        }
    }
    return values;
}

/**
 * @brief Author signed branch operands without introducing an RMS oracle.
 * @param rows Positive physical capacity retained through captured replay.
 * @param outputs Four complete residual branches, matching source output rows.
 * @return Original FP32 normalized-boundary operands for the real read stage.
 * @throws std::invalid_argument For incomplete geometry.
 */
inline std::vector<float> normalized(int rows, int outputs)
{
    if (rows <= 0 || outputs <= 0 || outputs % 4 != 0)
        throw std::invalid_argument("HC cancellation normalized geometry is invalid");
    std::vector<float> values(std::size_t(rows) * outputs);
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < outputs; ++column)
            values[std::size_t(row) * outputs + column] = column / (outputs / 4) < 2 ? 1.f : -1.f;
    return values;
}

/** @brief Independent original-source FP64 logits and nonzero signed read signal. */
struct Oracle
{
    std::vector<double> logits; ///< [M,4H], exact source decode and signed FP64 dot.
    std::vector<double> read; ///< [M,H], scalar sigmoid and original four-branch mean.
};

/**
 * @brief Evaluate the equation from original signed Q8_0 source bytes in FP64.
 * @param raw Authored original source, never prepared or requantized weights.
 * @param input Exact immutable low-rank FP32 input rows.
 * @param branches Exact signed normalized operands supplied to ResidualRead.
 * @param rows Physical row capacity evaluated only outside native execution.
 * @param outputs Four-branch original projection width.
 * @return Independent logits and read; no native operand-bank arithmetic is copied.
 */
inline Oracle reference(const std::vector<std::uint8_t>& raw,
                        const std::vector<float>& input,
                        const std::vector<float>& branches, int rows, int outputs)
{
    if (rows <= 0 || outputs <= 0 || outputs % 4 != 0 ||
        raw.size() != std::size_t(outputs) * kColumns / 32 * sizeof(Q8_0Block) ||
        input.size() != std::size_t(rows) * kColumns || branches.size() != std::size_t(rows) * outputs)
        throw std::invalid_argument("HC cancellation oracle extent mismatch");
    Oracle result{std::vector<double>(std::size_t(rows) * outputs),
                  std::vector<double>(std::size_t(rows) * (outputs / 4))};
    for (int row = 0; row < rows; ++row)
    {
        for (int column = 0; column < outputs; ++column)
        {
            double dot = 0;
            for (int block = 0; block < kColumns / 32; ++block)
            {
                Q8_0Block source;
                std::memcpy(&source, raw.data() + (std::size_t(column) * (kColumns / 32) + block) * sizeof(source), sizeof(source));
                const double d = fp16_to_fp32(source.d);
                for (int lane = 0; lane < 32; ++lane)
                    dot += double(input[std::size_t(row) * kColumns + block * 32 + lane]) * d * source.qs[lane];
            }
            result.logits[std::size_t(row) * outputs + column] = dot;
        }
        for (int channel = 0; channel < outputs / 4; ++channel)
        {
            double sum = 0;
            for (int branch = 0; branch < 4; ++branch)
            {
                const auto index = std::size_t(row) * outputs + branch * (outputs / 4) + channel;
                sum += double(branches[index]) / (1.0 + std::exp(-result.logits[index]));
            }
            result.read[std::size_t(row) * (outputs / 4) + channel] = sum / 4.0;
        }
    }
    return result;
}
} // namespace llaminar2::test::hc_up_cancellation
