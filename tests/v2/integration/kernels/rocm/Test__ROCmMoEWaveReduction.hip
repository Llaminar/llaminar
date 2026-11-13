/**
 * @file Test__ROCmMoEWaveReduction.hip
 * @brief Captured replay of the actual production router sum and its original portable-tree defect.
 *
 * Recorded finite FP32 lanes distinguish the established gfx906 ascending XOR
 * tree from the former descending down-shuffle tree without exp/div or transport.
 * A retained graph replays full/partial/empty/full live masks and poisoned output.
 * The original gfx906 DPP implementation and current generic implementation are
 * authenticated by the same host expression tree on every visible real device.
 */
#include "kernels/rocm/moe/ROCmMoEWaveReduction.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace llaminar2::test
{
    namespace
    {
        constexpr unsigned lanes = 64, cases = 4, elements = lanes * cases;
        using Inputs = std::array<float, elements>;
        /** @brief Native failures retain their exact operation and original status. */
        void checked(hipError_t error, const char *operation)
        {
            if (error != hipSuccess)
                throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
        }
        /** @brief Produce the actual production sum and the frozen original generic DOWN control. */
        __global__ void replayProductionSum(const float *input, const int *live,
            float *production, float *original_generic, unsigned *wave)
        {
            const unsigned lane = threadIdx.x, row = blockIdx.x;
            const float value = lane < static_cast<unsigned>(*live) ? input[row * lanes + lane] : 0.0f;
            production[row * lanes + lane] =
                rocm::moe_wave_contract::rocm_moe_wave_reduce_sum_f32(value);
            float old = value;
            // Exact predecessor branch; only lane zero is its defined full-wave reduction result.
            for (int offset = warpSize / 2; offset > 0; offset >>= 1)
                old += __shfl_down(old, offset);
            if (lane == 0)
            {
                original_generic[row] = old;
                wave[row] = warpSize;
            }
        }
        /** @return Recorded normal, finite inputs; first two cases distinguish the old trees exactly. */
        Inputs recordedInputs()
        {
            Inputs input{};
            input[0] = 1.0f;
            input[2] = input[3] = 0x1p-24f;
            input[lanes] = 0x1p70f;
            input[lanes + 1] = -0x1p70f;
            input[lanes + 2] = 3.0f;
            input[lanes + 3] = 4.0f;
            for (unsigned lane = 0; lane < lanes; ++lane)
            {
                input[2 * lanes + lane] = float(int(lane * 37 % 101) - 50) * 0x1p-8f;
                input[3 * lanes + lane] = float(int(lane % 7) - 3) * 0x1p-12f;
            }
            return input;
        }
        /**
         * @brief Evaluate the established fixed FP32 expression tree, independently of device instructions.
         * @param input Original recorded lanes. @param row Independent wave index.
         * @param live Admitted prefix; absent lanes contribute positive zero.
         * @return Lane-zero result after adjacent-pair-first groups2/4/8/16/32/64.
         */
        float canonicalReference(const Inputs &input, unsigned row, int live)
        {
            std::array<float, lanes> values{};
            for (int lane = 0; lane < live; ++lane) values[lane] = input[row * lanes + lane];
            for (unsigned offset = 1; offset < lanes; offset <<= 1)
            {
                const auto previous = values;
                for (unsigned lane = 0; lane < lanes; ++lane)
                    values[lane] = previous[lane] + previous[lane ^ offset];
            }
            return values[0];
        }
        /** @brief Retain exact test-native handles until every replay and observation completes. */
        struct Resources
        {
            int ordinal;
            hipStream_t stream = nullptr;
            hipGraph_t graph = nullptr;
            hipGraphExec_t execution = nullptr;
            bool recording = false;
            float *input = nullptr, *production = nullptr, *original = nullptr;
            int *live = nullptr;
            unsigned *wave = nullptr;
            /** @brief Complete owner-selected native cleanup; never hide incomplete release. */
            ~Resources()
            {
                const auto release = [&](hipError_t error, const char *operation) {
                    if (error == hipSuccess) return;
                    std::fprintf(stderr, "[ERROR] router wave replay cleanup device=%d %s: %s\n",
                        ordinal, operation, hipGetErrorString(error));
                    std::terminate();
                };
                release(hipSetDevice(ordinal), "hipSetDevice");
                if (recording) release(hipStreamEndCapture(stream, &graph), "abort incomplete capture");
                if (execution) release(hipGraphExecDestroy(execution), "hipGraphExecDestroy");
                if (graph) release(hipGraphDestroy(graph), "hipGraphDestroy");
                if (input) release(hipFree(input), "hipFree(input)");
                if (production) release(hipFree(production), "hipFree(production)");
                if (original) release(hipFree(original), "hipFree(original)");
                if (live) release(hipFree(live), "hipFree(live)");
                if (wave) release(hipFree(wave), "hipFree(wave)");
                if (stream) release(hipStreamDestroy(stream), "hipStreamDestroy");
            }
        };
    }

    /** @test Every real native target retains the original gfx906 sum expression across captured replay. */
    TEST(ROCmMoEWaveReductionNative, CapturedProductionSumUsesCanonicalAscendingXorOrder)
    {
        auto process_lifetime = GPUDeviceProcessLifetime::acquireNativeContext();
        initializeROCmHardwareMetadata();
        const auto &inventory = rocmHardwareMetadata();
        if (inventory.size() == 0)
            GTEST_SKIP() << "UNQUALIFIED: no ROCm hardware for production router wave reduction replay";
        const auto input = recordedInputs();
        ASSERT_EQ(std::bit_cast<std::uint32_t>(canonicalReference(input, 0, 64)), 0x3f800001u);
        for (int ordinal = 0; ordinal < static_cast<int>(inventory.size()); ++ordinal)
        {
            SCOPED_TRACE(ordinal);
            checked(hipSetDevice(ordinal), "select replay owner");
            Resources resources{ordinal};
            checked(hipStreamCreateWithFlags(&resources.stream, hipStreamNonBlocking), "create replay stream");
            checked(hipMalloc(&resources.input, sizeof(input)), "allocate replay input");
            checked(hipMalloc(&resources.production, sizeof(input)), "allocate production observation");
            checked(hipMalloc(&resources.original, cases * sizeof(float)), "allocate original observation");
            checked(hipMalloc(&resources.live, sizeof(int)), "allocate replay mask");
            checked(hipMalloc(&resources.wave, cases * sizeof(unsigned)), "allocate compiled wave observation");
            checked(hipMemcpyAsync(resources.input, input.data(), sizeof(input), hipMemcpyHostToDevice,
                resources.stream), "publish recorded input");
            checked(hipStreamSynchronize(resources.stream), "close initial publication");
            hipFuncAttributes attributes{};
            checked(hipFuncGetAttributes(&attributes, reinterpret_cast<const void *>(replayProductionSum)), "query replay resources");
            int occupancy = 0;
            checked(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, replayProductionSum, lanes, 0), "query replay occupancy");
            ASSERT_GT(occupancy, 0);
            std::cout << "[ROCmRouterWaveTree] ordinal=" << ordinal << " isa=" << inventory.at(ordinal).isa_target
                      << " threads=" << lanes << " regs=" << attributes.numRegs
                      << " lds=" << attributes.sharedSizeBytes << " local=" << attributes.localSizeBytes
                      << " resident_blocks=" << occupancy << '\n';
            checked(hipStreamBeginCapture(resources.stream, hipStreamCaptureModeThreadLocal), "begin replay capture");
            resources.recording = true;
            hipLaunchKernelGGL(replayProductionSum, dim3(cases), dim3(lanes), 0, resources.stream,
                resources.input, resources.live, resources.production, resources.original, resources.wave);
            checked(hipGetLastError(), "record production sum replay");
            const auto ended = hipStreamEndCapture(resources.stream, &resources.graph);
            if (ended != hipSuccess)
            {
                hipStreamCaptureStatus status{};
                if (hipStreamIsCapturing(resources.stream, &status) != hipSuccess || status != hipStreamCaptureStatusNone)
                    std::terminate();
            }
            resources.recording = false;
            checked(ended, "end replay capture");
            checked(hipGraphInstantiate(&resources.execution, resources.graph, nullptr, nullptr, 0), "instantiate replay");
            for (const int live : {64, 17, 3, 0, 64})
                for (int repeat = 0; repeat < 3; ++repeat)
                {
                    checked(hipMemcpyAsync(resources.live, &live, sizeof(live), hipMemcpyHostToDevice, resources.stream), "publish live mask");
                    checked(hipMemsetAsync(resources.production, 0xa5, sizeof(input), resources.stream), "poison production output");
                    checked(hipMemsetAsync(resources.original, 0xa5, cases * sizeof(float), resources.stream), "poison original output");
                    checked(hipGraphLaunch(resources.execution, resources.stream), "launch retained replay");
                    Inputs actual{};
                    std::array<float, cases> original{};
                    std::array<unsigned, cases> wave{};
                    checked(hipMemcpyAsync(actual.data(), resources.production, sizeof(actual), hipMemcpyDeviceToHost, resources.stream), "observe production sum");
                    checked(hipMemcpyAsync(original.data(), resources.original, sizeof(original), hipMemcpyDeviceToHost, resources.stream), "observe original control");
                    checked(hipMemcpyAsync(wave.data(), resources.wave, sizeof(wave), hipMemcpyDeviceToHost, resources.stream), "observe compiled width");
                    checked(hipStreamSynchronize(resources.stream), "terminal replay observation");
                    for (unsigned row = 0; row < cases; ++row)
                    {
                        EXPECT_EQ(wave[row], lanes);
                        const auto expected = std::bit_cast<std::uint32_t>(canonicalReference(input, row, live));
                        for (unsigned lane = 0; lane < lanes; ++lane)
                            EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[row * lanes + lane]), expected)
                                << "live=" << live << " row=" << row << " lane=" << lane;
                    }
                    if (live >= 17)
                    {
                        EXPECT_EQ(std::bit_cast<std::uint32_t>(original[0]), 0x3f800000u);
                        EXPECT_NE(std::bit_cast<std::uint32_t>(original[0]), std::bit_cast<std::uint32_t>(actual[0]));
                    }
                }
        }
    }
}
