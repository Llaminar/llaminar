/**
 * @file Test__ROCmSignedDot4.hip
 * @brief Captured native signed-dot replay across live extents and reset epochs.
 *
 * Every lane consumes distinct packed operands, signed extremes and accumulator
 * overflow cases. A scalar unsigned-bit oracle avoids host signed overflow.
 * One retained graph replays full, partial, empty and reset input epochs; dirty
 * inactive outputs must remain untouched. Synchronization is test-only terminal
 * collection, outside every captured production-helper invocation.
 */
#include "kernels/rocm/common/ROCmSignedDot4.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "backends/BackendManager.h"
#include "backends/rocm/ROCmRuntimeStartup.h"
#include "kernels/common/ROCmCompiledWaveContract.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <cstdio>
#include <dlfcn.h>
#include <bit>
#include <cstdint>
#include <limits>

namespace
{
    std::atomic<unsigned> hip_attribute_observations{0}; ///< Test-only actual runtime query observer, not a device-state cache.
}

/** @brief Observe the selected SDK runtime without changing native query behavior or status. */
extern "C" hipError_t hipDeviceGetAttribute(int *value, hipDeviceAttribute_t attribute, int ordinal)
{
    ++hip_attribute_observations;
    static const auto native = reinterpret_cast<decltype(&hipDeviceGetAttribute)>(dlsym(RTLD_NEXT, "hipDeviceGetAttribute"));
    if (!native)
    {
        std::fputs("[ERROR] HIP attribute observer cannot bind the selected native runtime\n", stderr);
        std::terminate();
    }
    return native(value, attribute, ordinal);
}

namespace llaminar2::test
{
    constexpr unsigned kDotLanes = 256;
    /** @brief Immutable launch geometry with device-read live input state. */
    struct DotReplayInput
    {
        std::array<std::int32_t, kDotLanes> a, b, initial;
        unsigned live;
    };
    /** @brief Each lane owns one exact native dot and its observed wave width. */
    __global__ void replaySignedDot(const DotReplayInput *input, std::int32_t *output,
                                    unsigned *observed_wave)
    {
        const unsigned lane = threadIdx.x;
        if (lane == 0) *observed_wave = warpSize;
        if (lane < input->live)
            output[lane] = rocm_integer_contract::signedDot4(
                input->a[lane], input->b[lane], input->initial[lane]);
    }
    /**
     * @brief Independently decode signed bytes and add using unsigned modulo arithmetic.
     * @param a First packed signed-byte input.
     * @param b Second packed signed-byte input.
     * @param initial Original accumulator bit pattern.
     * @return Exact expected unsaturated accumulator bit pattern.
     */
    std::uint32_t referenceDot(std::int32_t a, std::int32_t b, std::int32_t initial)
    {
        auto result = std::bit_cast<std::uint32_t>(initial);
        const auto ua = std::bit_cast<std::uint32_t>(a);
        const auto ub = std::bit_cast<std::uint32_t>(b);
        for (unsigned byte = 0; byte < 4; ++byte)
        {
            const int av = static_cast<int>((ua >> (8 * byte)) & 255u);
            const int bv = static_cast<int>((ub >> (8 * byte)) & 255u);
            result += static_cast<std::uint32_t>((av < 128 ? av : av - 256) *
                                                 (bv < 128 ? bv : bv - 256));
        }
        return result;
    }
    /** @brief Scoped ownership keeps failures from leaking native test resources. */
    struct DotReplayResources
    {
        hipStream_t stream = nullptr;
        hipGraph_t graph = nullptr;
        hipGraphExec_t executable = nullptr;
        DotReplayInput *input = nullptr;
        std::int32_t *output = nullptr;
        unsigned *wave = nullptr;
        /** @brief Retire retained graph resources after terminal test collection. */
        ~DotReplayResources()
        {
            if (executable) (void)hipGraphExecDestroy(executable);
            if (graph) (void)hipGraphDestroy(graph);
            if (input) (void)hipFree(input);
            if (output) (void)hipFree(output);
            if (wave) (void)hipFree(wave);
            if (stream) (void)hipStreamDestroy(stream);
        }
    };
    /** @test Native signed dot4 preserves every lane's exact modulo result during replay. */
    TEST(ROCmSignedDot4, CapturedSignedModuloLaneReplay)
    {
        requireROCmRuntimeStartup();
        initializeROCmHardwareMetadata();
        int devices = 0;
        ASSERT_EQ(hipGetDeviceCount(&devices), hipSuccess);
        ASSERT_GT(devices, 0);
        const auto &hardware = rocmHardwareMetadata();
        ASSERT_EQ(hardware.size(), static_cast<std::size_t>(devices));
        for (int device = 0; device < devices; ++device)
        {
            SCOPED_TRACE(device);
            ASSERT_EQ(hipSetDevice(device), hipSuccess);
            // The native fixture independently authenticates the cached facts.
            // Device preference remains separate from the captured program's
            // wave width, checked below by the actual kernel output.
            hipDeviceProp_t observed{};
            ASSERT_EQ(hipGetDeviceProperties(&observed, device), hipSuccess);
            const auto &cached = hardware.at(device);
            EXPECT_EQ(cached.isa_target, observed.gcnArchName);
            EXPECT_EQ(cached.native_wave_width, observed.warpSize);
            EXPECT_EQ(cached.compute_units, observed.multiProcessorCount);
            EXPECT_EQ(cached.lds_per_block, observed.sharedMemPerBlock);
            EXPECT_EQ(cached.maximum_threads_per_block, observed.maxThreadsPerBlock);
            int wall_clock_rate_khz = 0;
            ASSERT_EQ(hipDeviceGetAttribute(&wall_clock_rate_khz,
                hipDeviceAttributeWallClockRate, device), hipSuccess);
            EXPECT_EQ(cached.wall_clock_rate_khz, wall_clock_rate_khz);
            EXPECT_EQ(rocmServiceWallClockRateKHz(cached),
                static_cast<std::uint64_t>(wall_clock_rate_khz));
            EXPECT_EQ(cached.total_memory, observed.totalGlobalMem);
            for (std::size_t byte = 0; byte < cached.uuid.size(); ++byte)
                EXPECT_EQ(cached.uuid[byte], static_cast<std::uint8_t>(observed.uuid.bytes[byte]));
            DotReplayResources resources;
            ASSERT_EQ(hipStreamCreateWithFlags(&resources.stream, hipStreamNonBlocking), hipSuccess);
            ASSERT_NE(resources.stream, nullptr);
            ASSERT_EQ(hipMalloc(&resources.input, sizeof(DotReplayInput)), hipSuccess);
            ASSERT_EQ(hipMalloc(&resources.output, kDotLanes * sizeof(std::int32_t)), hipSuccess);
            ASSERT_EQ(hipMalloc(&resources.wave, sizeof(unsigned)), hipSuccess);
            ASSERT_EQ(hipStreamBeginCapture(resources.stream, hipStreamCaptureModeThreadLocal), hipSuccess);
            hipLaunchKernelGGL(replaySignedDot, dim3(1), dim3(kDotLanes), 0, resources.stream,
                               resources.input, resources.output, resources.wave);
            ASSERT_EQ(hipGetLastError(), hipSuccess);
            ASSERT_EQ(hipStreamEndCapture(resources.stream, &resources.graph), hipSuccess);
            ASSERT_EQ(hipGraphInstantiate(&resources.executable, resources.graph, nullptr, nullptr, 0), hipSuccess);
            constexpr std::array<std::uint32_t, 8> patterns = {
                0x80808080u, 0x7f7f7f7fu, 0xff00ff00u, 0x017f80ffu,
                0x00000000u, 0xffffffffu, 0x01010101u, 0x80ff007fu};
            constexpr std::array<unsigned, 6> extents = {256, 129, 1, 0, 255, 256};
            for (unsigned epoch = 0; epoch < extents.size(); ++epoch)
            {
                DotReplayInput input{};
                input.live = extents[epoch];
                std::array<std::int32_t, kDotLanes> output;
                for (unsigned lane = 0; lane < kDotLanes; ++lane)
                {
                    input.a[lane] = std::bit_cast<std::int32_t>(patterns[(lane + epoch) % patterns.size()]);
                    input.b[lane] = std::bit_cast<std::int32_t>(patterns[(lane / 8 + epoch) % patterns.size()]);
                    // Extremes deliberately cross both positive and negative overflow boundaries.
                    input.initial[lane] = lane % 2 ? std::numeric_limits<std::int32_t>::max() - int(lane)
                                                   : std::numeric_limits<std::int32_t>::min() + int(lane);
                }
                for (unsigned repeat = 0; repeat < 2; ++repeat)
                {
                    SCOPED_TRACE(epoch);
                    SCOPED_TRACE(repeat);
                    output.fill(std::bit_cast<std::int32_t>(0xdeadbeefu));
                    ASSERT_EQ(hipMemcpyAsync(resources.input, &input, sizeof(input), hipMemcpyHostToDevice, resources.stream), hipSuccess);
                    ASSERT_EQ(hipMemcpyAsync(resources.output, output.data(), sizeof(output), hipMemcpyHostToDevice, resources.stream), hipSuccess);
                    ASSERT_EQ(hipGraphLaunch(resources.executable, resources.stream), hipSuccess);
                    unsigned wave = 0;
                    ASSERT_EQ(hipMemcpyAsync(output.data(), resources.output, sizeof(output), hipMemcpyDeviceToHost, resources.stream), hipSuccess);
                    ASSERT_EQ(hipMemcpyAsync(&wave, resources.wave, sizeof(wave), hipMemcpyDeviceToHost, resources.stream), hipSuccess);
                    ASSERT_EQ(hipStreamSynchronize(resources.stream), hipSuccess);
                    EXPECT_EQ(wave, rocmOrderedWaveWidth);
                    for (unsigned lane = 0; lane < kDotLanes; ++lane)
                        EXPECT_EQ(std::bit_cast<std::uint32_t>(output[lane]), lane < input.live ?
                            referenceDot(input.a[lane], input.b[lane], input.initial[lane]) : 0xdeadbeefu) << lane;
                }
            }
        }
    }
}

namespace llaminar2::test
{
    /** @test Native capability matches each actual visible device and repeated getters perform no native query or selection. */
    TEST(ROCmHardwareMetadataNative, StreamWaitValueCapabilityUsesAdmittedFactsWithoutRediscovery)
    {
        initializeROCmHardwareMetadata();
        const auto &inventory = rocmHardwareMetadata();
        if (inventory.size() == 0)
            GTEST_SKIP() << "UNQUALIFIED: no visible ROCm hardware for stream-wait capability authentication";
        auto *backend = getROCmBackend();
        ASSERT_NE(backend, nullptr);
        for (int ordinal = 0; ordinal < static_cast<int>(inventory.size()); ++ordinal)
        {
            SCOPED_TRACE(ordinal);
            int actual = -1, before = -1, after = -1;
            ASSERT_EQ(hipDeviceGetAttribute(&actual, hipDeviceAttributeCanUseStreamWaitValue, ordinal), hipSuccess);
            ASSERT_TRUE(actual == 0 || actual == 1);
            const auto &hardware = inventory.at(ordinal);
            EXPECT_EQ(hardware.can_use_stream_wait_value, actual == 1);
            ASSERT_EQ(hipGetDevice(&before), hipSuccess);
            const auto observations = hip_attribute_observations.load();
            EXPECT_GT(observations, 0u); // Actual observer must be active, never a vacuous constant counter.
            for (unsigned replay = 0; replay < 64; ++replay)
            {
                EXPECT_EQ(backend->supportsStreamTimelineSignal32(ordinal), actual == 1);
                EXPECT_EQ(backend->supportsStreamTimelineSignal64(ordinal), actual == 1);
                EXPECT_EQ(&rocmHardwareMetadata().at(ordinal), &hardware);
            }
            EXPECT_EQ(hip_attribute_observations.load(), observations);
            ASSERT_EQ(hipGetDevice(&after), hipSuccess);
            EXPECT_EQ(after, before);
        }
    }
}
