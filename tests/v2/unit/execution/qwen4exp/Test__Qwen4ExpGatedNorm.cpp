/**
 * @file Test__Qwen4ExpGatedNorm.cpp
 * @brief Flash Next's sigmoid gate follows per-head normalization on CPU.
 *
 * Independent double-precision equations distinguish sigmoid from SiLU and
 * verify shared 128-element gains, extreme gates, in-place execution and
 * grouped/serial byte equivalence at the released 48-head geometry.
 */
#include "execution/compute_stages/stages/GatedRMSNormStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/Tensors.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

using namespace llaminar2;

TEST(Qwen4ExpGatedNorm, SigmoidAfterPerHeadNormalizationMatchesEquationAndSerialRows)
{
    constexpr std::size_t heads = 48, width = 128, columns = heads * width;
    CPUDeviceContext context(DeviceId::cpu(), 1);
    FP32Tensor gain({width}, DeviceId::cpu());
    for (std::size_t column = 0; column < width; ++column)
        gain.mutable_data()[column] = 0.3f + static_cast<float>(column % 17) / 11.0f;
    for (const std::size_t rows : {1u, 3u, 17u})
    {
        FP32Tensor input({rows, columns}, DeviceId::cpu()), gate({rows, columns}, DeviceId::cpu()),
            output({rows, columns}, DeviceId::cpu());
        for (std::size_t index = 0; index < input.numel(); ++index)
        {
            input.mutable_data()[index] = static_cast<float>(static_cast<int>(index % 173) - 86) / 37.0f;
            gate.mutable_data()[index] = static_cast<float>(static_cast<int>(index % 201) - 100) / 5.0f;
        }
        GatedRMSNormStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = &input;
        params.output = &output;
        params.gate = &gate;
        params.gamma = &gain;
        params.seq_len = static_cast<int>(rows);
        params.norm_dim = width;
        params.feature_dim = columns;
        params.gate_activation = GatedNormActivation::Sigmoid;
        GatedRMSNormStage grouped(params);
        ASSERT_TRUE(grouped.execute(&context));
        for (std::size_t row = 0; row < rows; ++row)
        {
            for (std::size_t head = 0; head < heads; ++head)
            {
                const auto offset = row * columns + head * width;
                double square_sum = 0;
                for (std::size_t column = 0; column < width; ++column)
                {
                    const double value = input.data()[offset + column];
                    square_sum += value * value;
                }
                const double inv = 1.0 / std::sqrt(square_sum / width + params.eps);
                for (std::size_t column = 0; column < width; ++column)
                {
                    const auto index = offset + column;
                    const double expected = input.data()[index] * inv * gain.data()[column] /
                        (1.0 + std::exp(-static_cast<double>(gate.data()[index])));
                    EXPECT_NEAR(output.data()[index], expected, 1e-5 * std::max(1.0, std::abs(expected)))
                        << "rows=" << rows << " row=" << row << " head=" << head << " column=" << column;
                }
            }
            FP32Tensor serial_input({1, columns}, DeviceId::cpu()), serial_gate({1, columns}, DeviceId::cpu());
            std::memcpy(serial_input.mutable_data(), input.data() + row * columns, columns * sizeof(float));
            std::memcpy(serial_gate.mutable_data(), gate.data() + row * columns, columns * sizeof(float));
            params.input = &serial_input;
            params.output = &serial_input; // In-place normalization retains the exact serial arithmetic.
            params.gate = &serial_gate;
            params.seq_len = 1;
            GatedRMSNormStage serial(params);
            ASSERT_TRUE(serial.execute(&context));
            EXPECT_EQ(std::memcmp(serial_input.data(), output.data() + row * columns,
                                  columns * sizeof(float)), 0);
        }
    }
}

TEST(Qwen4ExpGatedNorm, InvalidActivationCannotEnterCapturedStage)
{
    GatedRMSNormStage::Params params;
    params.gate_activation = static_cast<GatedNormActivation>(255);
    EXPECT_THROW((void)GatedRMSNormStage(params), std::invalid_argument);
}
