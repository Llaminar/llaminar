/**
 * @file Test__CPUSwiGLUGroupedVerifier.cpp
 * @brief CPU all-format grouped standalone SwiGLU byte-equivalence gate.
 *
 * The production CPU activation path supports FP32, BF16, FP16, and Q8_1.
 * This suite proves runtime-M output is byte-identical to production M=1 decode at
 * both an odd 17-block row width and a real dense Qwen FFN width. Route counters
 * additionally require one grouped primitive call rather than row replay.
 */

#include <gtest/gtest.h>

#include "kernels/cpu/ops/CPUSwiGLUKernelT.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "utils/VerifierRowTestInventory.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Enable perfstats for one suite and restore the caller environment. */
    class ScopedPerfStats
    {
    public:
        ScopedPerfStats()
        {
            if (const char *old = std::getenv("LLAMINAR_PERF_STATS_SUMMARY"))
            {
                had_old_value_ = true;
                old_value_ = old;
            }
            setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ~ScopedPerfStats()
        {
            if (had_old_value_)
                setenv("LLAMINAR_PERF_STATS_SUMMARY", old_value_.c_str(), 1);
            else
                unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ScopedPerfStats(const ScopedPerfStats &) = delete;
        ScopedPerfStats &operator=(const ScopedPerfStats &) = delete;

    private:
        bool had_old_value_ = false;
        std::string old_value_;
    };

    /** @brief Generate deterministic bounded values for gate and up operands. */
    std::vector<float> makeValues(size_t count, uint32_t seed)
    {
        std::vector<float> values(count);
        uint32_t state = seed;
        for (size_t index = 0; index < count; ++index)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            const int centered = static_cast<int>(state % 4093u) - 2046;
            values[index] = static_cast<float>(centered) / 1536.0f;
        }
        return values;
    }

    /**
     * @brief Materialize one supported CPU activation tensor in native bytes.
     *
     * Q8_1 uses the production tensor quantizer. Widths in this suite are exact
     * multiples of 32, so every logical row has an integral native block span.
     */
    template <ActivationPrecision Precision>
    std::shared_ptr<TensorBase> makeNativeTensor(
        const std::vector<size_t> &shape,
        const float *source)
    {
        if constexpr (Precision == ActivationPrecision::FP32)
        {
            auto tensor = std::make_shared<FP32Tensor>(shape);
            std::copy_n(source, tensor->numel(), tensor->mutable_data());
            return tensor;
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            auto tensor = std::make_shared<BF16Tensor>(shape);
            simd::convert_fp32_to_bf16(
                source, tensor->mutable_typed_data(), tensor->numel());
            return tensor;
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            auto tensor = std::make_shared<FP16Tensor>(shape);
            simd::convert_fp32_to_fp16(
                source, tensor->mutable_typed_data(), tensor->numel());
            return tensor;
        }
        else
        {
            static_assert(Precision == ActivationPrecision::Q8_1);
            return Q8_1Tensor::quantize_from_fp32(source, shape);
        }
    }

    /** @brief Create a native zero-filled activation output. */
    template <ActivationPrecision Precision>
    std::shared_ptr<TensorBase> makeZeroNativeTensor(
        const std::vector<size_t> &shape)
    {
        size_t count = 1;
        for (size_t extent : shape)
            count *= extent;
        const std::vector<float> zeros(count, 0.0f);
        return makeNativeTensor<Precision>(shape, zeros.data());
    }

    /** @brief Return a required route tag or a visible missing marker. */
    std::string tag(const PerfStatRecord &record, const char *name)
    {
        const auto it = record.tags.find(name);
        return it == record.tags.end()
                   ? std::string("<missing:") + name + '>'
                   : it->second;
    }

    /** @brief Require native output byte equality and report the first mismatch. */
    void expectNativeBytesEqual(
        const TensorBase &grouped,
        const std::vector<uint8_t> &serial,
        const std::string &label)
    {
        ASSERT_EQ(grouped.size_bytes(), serial.size()) << label;
        const auto *grouped_bytes = static_cast<const uint8_t *>(grouped.raw_data());
        if (std::memcmp(grouped_bytes, serial.data(), serial.size()) == 0)
            return;

        for (size_t index = 0; index < serial.size(); ++index)
        {
            if (grouped_bytes[index] != serial[index])
            {
                ADD_FAILURE() << label << " first native mismatch at byte " << index
                              << " grouped=" << static_cast<unsigned>(grouped_bytes[index])
                              << " serial=" << static_cast<unsigned>(serial[index]);
                return;
            }
        }
    }

    /**
     * @brief Sweep one CPU precision over verifier depths and FFN widths.
     *
     * @tparam Precision Native activation precision.
     * @tparam Kernel Matching CPUSwiGLUKernelT specialization.
     * @param format_label Expected telemetry format.
     * @param schedule_label Expected native work partition.
     */
    template <ActivationPrecision Precision, typename Kernel>
    void runNativeFormat(
        const char *format_label,
        const char *schedule_label)
    {
        constexpr std::array<int, 2> column_counts = {544, 4864};
        constexpr int max_rows = test::kGroupedVerifierRuntimeRows.back();
        constexpr const char *counter_name =
            "cpu_swiglu_grouped_verifier_rows_calls";

        for (int cols : column_counts)
        {
            const auto gate_values = makeValues(
                static_cast<size_t>(max_rows) * cols,
                0x91A10000u ^ static_cast<uint32_t>(cols));
            const auto up_values = makeValues(
                static_cast<size_t>(max_rows) * cols,
                0xA2B20000u ^ static_cast<uint32_t>(cols));

            for (int verifier_rows : test::kGroupedVerifierRuntimeRows)
            {
                SCOPED_TRACE(
                    std::string("CPU format=") + format_label +
                    " M=" + std::to_string(verifier_rows) +
                    " cols=" + std::to_string(cols));

                const std::vector<size_t> grouped_shape = {
                    static_cast<size_t>(verifier_rows),
                    static_cast<size_t>(cols),
                };
                auto grouped_gate = makeNativeTensor<Precision>(
                    grouped_shape, gate_values.data());
                auto grouped_up = makeNativeTensor<Precision>(
                    grouped_shape, up_values.data());
                auto grouped_output = makeZeroNativeTensor<Precision>(grouped_shape);

                std::vector<uint8_t> native_gate(grouped_gate->size_bytes());
                std::vector<uint8_t> native_up(grouped_up->size_bytes());
                std::memcpy(
                    native_gate.data(), grouped_gate->raw_data(), native_gate.size());
                std::memcpy(native_up.data(), grouped_up->raw_data(), native_up.size());
                const size_t row_bytes = native_gate.size() /
                                         static_cast<size_t>(verifier_rows);
                std::vector<uint8_t> serial_output(grouped_output->size_bytes(), 0u);

                Kernel kernel;
                for (int row = 0; row < verifier_rows; ++row)
                {
                    const std::vector<size_t> row_shape = {
                        1u, static_cast<size_t>(cols)};
                    auto row_gate = makeNativeTensor<Precision>(
                        row_shape,
                        gate_values.data() + static_cast<size_t>(row) * cols);
                    auto row_up = makeNativeTensor<Precision>(
                        row_shape,
                        up_values.data() + static_cast<size_t>(row) * cols);
                    auto row_output = makeZeroNativeTensor<Precision>(row_shape);
                    std::memcpy(
                        row_gate->raw_mutable_data(),
                        native_gate.data() + static_cast<size_t>(row) * row_bytes,
                        row_bytes);
                    std::memcpy(
                        row_up->raw_mutable_data(),
                        native_up.data() + static_cast<size_t>(row) * row_bytes,
                        row_bytes);

                    ASSERT_TRUE(kernel.apply_tensor(
                        row_gate.get(), row_up.get(), row_output.get(),
                        1, cols, false, nullptr, -1));
                    std::memcpy(
                        serial_output.data() + static_cast<size_t>(row) * row_bytes,
                        row_output->raw_data(), row_bytes);
                }

                // The obsolete residual flag has no operand in ITensorSwiGLU.
                // It must fail visibly rather than silently changing semantics.
                PerfStatsCollector::reset();
                EXPECT_FALSE(kernel.apply_tensor(
                    grouped_gate.get(), grouped_up.get(), grouped_output.get(),
                    verifier_rows, cols, true, nullptr, -1));
                const std::string metric = std::string("kernel.") + counter_name;
                EXPECT_TRUE(PerfStatsCollector::snapshot({metric}).empty());

                PerfStatsCollector::reset();
                ASSERT_TRUE(kernel.apply_tensor(
                    grouped_gate.get(), grouped_up.get(), grouped_output.get(),
                    verifier_rows, cols, false, nullptr, -1));
                expectNativeBytesEqual(
                    *grouped_output, serial_output,
                    std::string(format_label) + " grouped SwiGLU");

                const auto records = PerfStatsCollector::snapshot({metric});
                ASSERT_EQ(records.size(), 1u)
                    << PerfStatsCollector::summaryString({metric}, 20);
                const auto &record = records.front();
                EXPECT_EQ(record.count, 1u);
                EXPECT_EQ(tag(record, "tensor_format"), format_label);
                EXPECT_EQ(
                    tag(record, "verifier_rows"),
                    std::to_string(verifier_rows));
                EXPECT_EQ(tag(record, "cols"), std::to_string(cols));
                EXPECT_EQ(tag(record, "element_schedule"), schedule_label);
                EXPECT_EQ(
                    tag(record, "invocation_policy"),
                    "single_grouped_workshare");
            }
        }
    }
} // namespace

TEST(Test__CPUSwiGLUGroupedVerifier, AllNativeFormatsMatchSerialDecodeBytes)
{
    ScopedPerfStats perfstats;
    runNativeFormat<ActivationPrecision::FP32,
                    CPUSwiGLUKernelT<ActivationPrecision::FP32>>(
        "FP32", "fp32_row_chunks");
    runNativeFormat<ActivationPrecision::BF16,
                    CPUSwiGLUKernelT<ActivationPrecision::BF16>>(
        "BF16", "native_elements");
    runNativeFormat<ActivationPrecision::FP16,
                    CPUSwiGLUKernelT<ActivationPrecision::FP16>>(
        "FP16", "native_elements");
    runNativeFormat<ActivationPrecision::Q8_1,
                    CPUSwiGLUKernelT<ActivationPrecision::Q8_1>>(
        "Q8_1", "native_q8_blocks");
}
