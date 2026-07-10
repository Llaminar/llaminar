/**
 * @file Test__CPUFusedResidualNormGroupedVerifier.cpp
 * @brief CPU grouped fused-residual RMSNorm decode-equivalence integration gate.
 *
 * The graph uses FusedResidualNormStage at attention and FFN residual boundaries,
 * so proving ResidualAdd and RMSNorm separately is insufficient. This suite enters
 * the production stage itself and compares both mutable outputs - the updated
 * residual stream and normalized activation - with independent production M=1
 * executions. FP32, BF16, and FP16 are swept at M=2..4 and at widths that cover
 * small per-head reductions and real hidden-state reductions.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/FusedResidualNormStage.h"
#include "execution/local_execution/device/DeviceContext.h"
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
    /** @brief Enable perfstats for the test lifetime and restore the caller. */
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

    /** @brief Generate deterministic finite values with row-distinct bit patterns. */
    std::vector<float> makeValues(size_t count, uint32_t seed)
    {
        std::vector<float> values(count);
        uint32_t state = seed;
        for (size_t index = 0; index < count; ++index)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            const int centered = static_cast<int>(state % 3079u) - 1539;
            values[index] = static_cast<float>(centered) / 1024.0f;
        }
        return values;
    }

    /** @brief Generate non-uniform positive FP32 RMSNorm scale weights. */
    std::vector<float> makeGamma(size_t cols)
    {
        std::vector<float> gamma(cols);
        for (size_t col = 0; col < cols; ++col)
            gamma[col] = 0.625f + static_cast<float>((col * 53u) % 383u) / 512.0f;
        return gamma;
    }

    /**
     * @brief Materialize one CPU tensor in the requested native format.
     *
     * Conversion occurs once before the serial/grouped split. Tests subsequently
     * copy native bytes into each M=1 witness so conversion cannot hide drift.
     */
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

    /** @brief Create a native output tensor initialized to deterministic zero. */
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

    /** @brief Read a required perfstats tag with a visible missing marker. */
    std::string tag(const PerfStatRecord &record, const char *name)
    {
        const auto it = record.tags.find(name);
        return it == record.tags.end()
                   ? std::string("<missing:") + name + '>'
                   : it->second;
    }

    /** @brief Require byte equality and identify the first native mismatch. */
    void expectNativeBytesEqual(
        const TensorBase &grouped,
        const std::vector<uint8_t> &serial,
        const std::string &label)
    {
        ASSERT_EQ(grouped.size_bytes(), serial.size()) << label;
        const auto *grouped_bytes =
            static_cast<const uint8_t *>(grouped.raw_data());
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
     * @brief Build the exact production stage used by Qwen attention/FFN graphs.
     */
    FusedResidualNormStage makeStage(
        const TensorBase *input,
        TensorBase *residual,
        const TensorBase *gamma,
        TensorBase *norm_output,
        int rows,
        int cols)
    {
        FusedResidualNormStage::Params params{};
        params.device_id = DeviceId::cpu();
        params.input = input;
        params.residual = residual;
        params.gamma = gamma;
        params.norm_output = norm_output;
        params.eps = 1e-6f;
        params.seq_len = rows;
        params.hidden_dim = cols;
        return FusedResidualNormStage(std::move(params));
    }

    /**
     * @brief Sweep one CPU native format over M=2..4 and two reduction widths.
     *
     * Each serial row begins from the exact native grouped input/residual bytes.
     * The grouped call is then made once over the full M-row tensors. Both stage
     * outputs and the grouped route counter are blocking acceptance criteria.
     */
    template <ActivationPrecision Precision>
    void runNativeFormat(
        IDeviceContext *context,
        const char *format_label,
        const char *implementation)
    {
        constexpr std::array<int, 2> column_counts = {128, 4096};
        constexpr const char *counter_name =
            "cpu_fused_residual_rmsnorm_grouped_verifier_rows_calls";

        for (int cols : column_counts)
        {
            const auto input_values = makeValues(
                static_cast<size_t>(4) * cols,
                0xF0010000u ^ static_cast<uint32_t>(cols));
            const auto residual_values = makeValues(
                static_cast<size_t>(4) * cols,
                0xF0020000u ^ static_cast<uint32_t>(cols));
            const auto gamma_values = makeGamma(static_cast<size_t>(cols));

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
                auto grouped_norm = makeZeroNativeTensor<Precision>(grouped_shape);
                auto gamma = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(cols)});
                std::copy(
                    gamma_values.begin(), gamma_values.end(), gamma->mutable_data());

                std::vector<uint8_t> native_input(grouped_input->size_bytes());
                std::vector<uint8_t> native_residual(grouped_residual->size_bytes());
                std::memcpy(
                    native_input.data(), grouped_input->raw_data(), native_input.size());
                std::memcpy(
                    native_residual.data(), grouped_residual->raw_data(),
                    native_residual.size());

                const size_t row_bytes = native_input.size() /
                                         static_cast<size_t>(verifier_rows);
                std::vector<uint8_t> serial_residual(grouped_residual->size_bytes(), 0u);
                std::vector<uint8_t> serial_norm(grouped_norm->size_bytes(), 0u);

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
                    auto row_norm = makeZeroNativeTensor<Precision>(row_shape);
                    std::memcpy(
                        row_input->raw_mutable_data(),
                        native_input.data() + static_cast<size_t>(row) * row_bytes,
                        row_bytes);
                    std::memcpy(
                        row_residual->raw_mutable_data(),
                        native_residual.data() + static_cast<size_t>(row) * row_bytes,
                        row_bytes);

                    auto serial_stage = makeStage(
                        row_input.get(), row_residual.get(), gamma.get(),
                        row_norm.get(), 1, cols);
                    ASSERT_TRUE(serial_stage.execute(context));
                    std::memcpy(
                        serial_residual.data() + static_cast<size_t>(row) * row_bytes,
                        row_residual->raw_data(), row_bytes);
                    std::memcpy(
                        serial_norm.data() + static_cast<size_t>(row) * row_bytes,
                        row_norm->raw_data(), row_bytes);
                }

                PerfStatsCollector::reset();
                auto grouped_stage = makeStage(
                    grouped_input.get(), grouped_residual.get(), gamma.get(),
                    grouped_norm.get(), verifier_rows, cols);
                ASSERT_TRUE(grouped_stage.execute(context));

                expectNativeBytesEqual(
                    *grouped_residual, serial_residual,
                    std::string(format_label) + " grouped residual publication");
                expectNativeBytesEqual(
                    *grouped_norm, serial_norm,
                    std::string(format_label) + " grouped normalized publication");

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
                EXPECT_EQ(tag(record, "implementation"), implementation);
                EXPECT_EQ(tag(record, "capture_mode"), "direct");
                EXPECT_EQ(tag(record, "row_mapping"), "independent_rows");
                EXPECT_EQ(
                    tag(record, "invocation_policy"),
                    "single_grouped_stage_call");
            }
        }
    }
} // namespace

TEST(Test__CPUFusedResidualNormGroupedVerifier,
     AllNativeFormatsAndReductionWidthsMatchSerialDecodeBytes)
{
    ScopedPerfStats perfstats;
    auto context = IDeviceContext::create(DeviceId::cpu());
    ASSERT_NE(context, nullptr);

    runNativeFormat<ActivationPrecision::FP32>(
        context.get(), "FP32", "fused_cache_resident_rows");
    runNativeFormat<ActivationPrecision::BF16>(
        context.get(), "BF16", "typed_residual_then_rmsnorm");
    runNativeFormat<ActivationPrecision::FP16>(
        context.get(), "FP16", "typed_residual_then_rmsnorm");
}
