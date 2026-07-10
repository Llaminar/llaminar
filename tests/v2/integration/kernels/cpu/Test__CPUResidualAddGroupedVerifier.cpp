/**
 * @file Test__CPUResidualAddGroupedVerifier.cpp
 * @brief CPU all-format grouped residual-add decode-equivalence integration gate.
 *
 * Residual addition is row-independent, so the economical CPU implementation
 * is one flat OpenMP workshare over all active verifier values. This suite
 * compares FP32, BF16, and FP16 native output bytes against production M=1
 * decode at M=2..4 and requires exactly one grouped workshare counter.
 */

#include <gtest/gtest.h>

#include "kernels/cpu/ops/CPUResidualAddKernelT.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

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
    /** @brief Enable perfstats for this integration test and restore the caller. */
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

    /** @brief Generate deterministic finite values with row-distinct patterns. */
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
            values[index] = static_cast<float>(centered) / 1024.0f;
        }
        return values;
    }

    /** @brief Materialize one CPU activation tensor in its native format. */
    template <ActivationPrecision Precision>
    std::unique_ptr<TensorBase> makeNativeTensor(
        const std::vector<size_t> &shape,
        const float *source)
    {
        if constexpr (Precision == ActivationPrecision::FP32)
        {
            auto tensor = std::make_unique<FP32Tensor>(shape);
            std::copy_n(source, tensor->numel(), tensor->mutable_data());
            return tensor;
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            auto tensor = std::make_unique<BF16Tensor>(shape);
            simd::convert_fp32_to_bf16(
                source, tensor->mutable_typed_data(), tensor->numel());
            return tensor;
        }
        else
        {
            static_assert(Precision == ActivationPrecision::FP16);
            auto tensor = std::make_unique<FP16Tensor>(shape);
            simd::convert_fp32_to_fp16(
                source, tensor->mutable_typed_data(), tensor->numel());
            return tensor;
        }
    }

    /** @brief Create a zero-filled native result tensor. */
    template <ActivationPrecision Precision>
    std::unique_ptr<TensorBase> makeZeroNativeTensor(
        const std::vector<size_t> &shape)
    {
        size_t count = 1;
        for (size_t extent : shape)
            count *= extent;
        const std::vector<float> zeros(count, 0.0f);
        return makeNativeTensor<Precision>(shape, zeros.data());
    }

    /** @brief Return one required telemetry tag or a visible missing marker. */
    std::string tag(const PerfStatRecord &record, const char *name)
    {
        const auto it = record.tags.find(name);
        return it == record.tags.end()
                   ? std::string("<missing:") + name + '>'
                   : it->second;
    }

    /** @brief Require native byte equality and report the first mismatch. */
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
     * @brief Sweep one CPU precision across all verifier depths and row widths.
     *
     * @tparam Precision Native CPU activation precision.
     * @tparam Kernel Corresponding CPUResidualAddKernelT specialization.
     * @param format_label Expected native telemetry label.
     */
    template <ActivationPrecision Precision, typename Kernel>
    void runNativeFormat(const char *format_label)
    {
        constexpr std::array<int, 2> column_counts = {128, 4096};
        constexpr const char *counter_name =
            "cpu_residual_add_grouped_verifier_rows_calls";

        for (int cols : column_counts)
        {
            const auto input_values = makeValues(
                static_cast<size_t>(4) * cols,
                0x13570000u ^ static_cast<uint32_t>(cols));
            const auto residual_values = makeValues(
                static_cast<size_t>(4) * cols,
                0x24680000u ^ static_cast<uint32_t>(cols));

            for (int verifier_rows : {2, 3, 4})
            {
                SCOPED_TRACE(
                    std::string("CPU format=") + format_label +
                    " M=" + std::to_string(verifier_rows) +
                    " cols=" + std::to_string(cols));

                const std::vector<size_t> grouped_shape = {
                    static_cast<size_t>(verifier_rows),
                    static_cast<size_t>(cols),
                };
                auto grouped_input = makeNativeTensor<Precision>(
                    grouped_shape, input_values.data());
                auto grouped_residual = makeNativeTensor<Precision>(
                    grouped_shape, residual_values.data());
                auto grouped_output = makeZeroNativeTensor<Precision>(grouped_shape);

                std::vector<uint8_t> native_input(grouped_input->size_bytes());
                std::vector<uint8_t> native_residual(grouped_residual->size_bytes());
                std::memcpy(
                    native_input.data(), grouped_input->raw_data(), native_input.size());
                std::memcpy(
                    native_residual.data(), grouped_residual->raw_data(),
                    native_residual.size());
                const size_t row_bytes = native_input.size() /
                                         static_cast<size_t>(verifier_rows);
                std::vector<uint8_t> serial_output(grouped_output->size_bytes(), 0u);

                Kernel kernel;
                for (int row = 0; row < verifier_rows; ++row)
                {
                    const std::vector<size_t> row_shape = {
                        1u, static_cast<size_t>(cols)};
                    auto row_input = makeNativeTensor<Precision>(
                        row_shape,
                        input_values.data() + static_cast<size_t>(row) * cols);
                    auto row_residual = makeNativeTensor<Precision>(
                        row_shape,
                        residual_values.data() + static_cast<size_t>(row) * cols);
                    auto row_output = makeZeroNativeTensor<Precision>(row_shape);
                    std::memcpy(
                        row_input->raw_mutable_data(),
                        native_input.data() + static_cast<size_t>(row) * row_bytes,
                        row_bytes);
                    std::memcpy(
                        row_residual->raw_mutable_data(),
                        native_residual.data() + static_cast<size_t>(row) * row_bytes,
                        row_bytes);

                    ASSERT_TRUE(kernel.apply_tensor(
                        row_input.get(), row_residual.get(), row_output.get(),
                        static_cast<size_t>(cols), nullptr, -1));
                    std::memcpy(
                        serial_output.data() + static_cast<size_t>(row) * row_bytes,
                        row_output->raw_data(), row_bytes);
                }

                const size_t active_elements =
                    static_cast<size_t>(verifier_rows) * cols;
                PerfStatsCollector::reset();
                ASSERT_TRUE(kernel.apply_tensor(
                    grouped_input.get(), grouped_residual.get(), grouped_output.get(),
                    active_elements, nullptr, -1));
                expectNativeBytesEqual(
                    *grouped_output, serial_output,
                    std::string(format_label) + " grouped residual add");

                const std::string metric = std::string("kernel.") + counter_name;
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
                EXPECT_EQ(
                    tag(record, "active_elements"),
                    std::to_string(active_elements));
                EXPECT_EQ(
                    tag(record, "invocation_policy"),
                    "single_flat_workshare");
            }
        }
    }
} // namespace

TEST(Test__CPUResidualAddGroupedVerifier, AllNativeFormatsMatchSerialDecodeBytes)
{
    ScopedPerfStats perfstats;
    runNativeFormat<ActivationPrecision::FP32,
                    CPUResidualAddKernelT<ActivationPrecision::FP32>>("FP32");
    runNativeFormat<ActivationPrecision::BF16,
                    CPUResidualAddKernelT<ActivationPrecision::BF16>>("BF16");
    runNativeFormat<ActivationPrecision::FP16,
                    CPUResidualAddKernelT<ActivationPrecision::FP16>>("FP16");
}
