/**
 * @file Test__GDNProjectionCapture.cpp
 * @brief Retained mixed-decoder GDN graph and dirty-output regression.
 *
 * CUDA and ROCm compile this fixture independently so each process owns one
 * runtime. Shared row/format inventories exercise both ordinary prefill and
 * serial-equivalent verification, including full and TP-sharded model shapes.
 * Route counters prove shared quantization and persistent projection streams;
 * repeated native replay proves their output bytes and untouched capacity.
 */
#include "../../utils/GDNProjectionCaptureHarness.h"
#ifdef LLAMINAR_GDN_CAPTURE_CUDA
#include "backends/cuda/CUDAGraphCapture.h"
#else
#include "backends/rocm/HIPGraphCapture.h"
#endif
#include "backends/ComputeBackend.h"
#include "utils/DebugEnv.h"
#include "utils/MPIContext.h"

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /**
     * @brief Run one backend's complete retained-family regression.
     * @param device Physical device owning every prepared engine and stream.
     * @param all_formats Select the shared registry or full/sharded model shapes.
     */
    void exercise(DeviceId device, bool all_formats)
    {
        DeviceManager::instance().initialize(-1);
        ScopedGPUStream selection(device);
        std::unique_ptr<IDeviceContext> context;
        GDNProjectionGraphFactory factory;
#ifdef LLAMINAR_GDN_CAPTURE_CUDA
        {
            context = std::make_unique<CUDADeviceContext>(device, device.ordinal);
            factory = [device](void *stream) {
                return std::make_unique<CUDAGraphCapture>(
                    static_cast<cudaStream_t>(stream), device.ordinal);
            };
        }
#else
        {
            context = std::make_unique<ROCmDeviceContext>(device, device.ordinal);
            factory = [device](void *stream) {
                return std::make_unique<HIPGraphCapture>(
                    static_cast<hipStream_t>(stream), device.ordinal);
            };
        }
#endif
        const auto &formats = quantizedVerifierFormats();
        const auto &q5 = *std::find_if(formats.begin(), formats.end(),
            [](const auto &format) { return format.tensor_type == TensorType::Q5_K; });
        const auto &iq4 = *std::find_if(formats.begin(), formats.end(),
            [](const auto &format) { return format.tensor_type == TensorType::IQ4_XS; });
        using Purpose = GDNProjectionRowsPurpose;
        std::vector<GDNProjectionCaptureRows> rows{{1, Purpose::Ordinary}};
        if (all_formats)
        {
            for (int count : kGroupedVerifierRuntimeRows)
                rows.push_back({count, Purpose::Verifier});
            std::vector<int> ordinary_rows{2, 16, 17, 32, 63, 65};
            for (int count : kSupportedPrefillGraphBucketSizes)
                if (count <= kDefaultPrefillGraphMaxBucketSize)
                    ordinary_rows.push_back(count);
            std::sort(ordinary_rows.begin(), ordinary_rows.end());
            ordinary_rows.erase(std::unique(ordinary_rows.begin(),
                ordinary_rows.end()), ordinary_rows.end());
            for (int count : ordinary_rows)
                rows.push_back({count, Purpose::Ordinary});
            for (const auto &format : formats)
            {
                const auto &other = format.device_execution_codebook_id ==
                    q5.device_execution_codebook_id ? iq4 : q5;
                ASSERT_NO_FATAL_FAILURE(runGDNProjectionCaptureCase(device,
                    *context, factory, format, other, 512, {384, 256, 12, 12}, rows));
            }
        }
        else
        {
            rows.push_back({4, Purpose::Verifier});
            rows.push_back({16, Purpose::Verifier});
            for (int count : {17, 64, 448, 512})
                rows.push_back({count, Purpose::Ordinary});
            for (int degree : {1, 2})
                ASSERT_NO_FATAL_FAILURE(runGDNProjectionCaptureCase(device,
                    *context, factory, q5, iq4, 5120,
                    {10240 / degree, 6144 / degree, 48 / degree, 48 / degree}, rows));
        }
    }
}

#ifdef LLAMINAR_GDN_CAPTURE_CUDA
TEST(GDNMixedProjectionCapture, CUDAProductionShapes) { exercise(DeviceId::cuda(0), false); }
TEST(GDNMixedProjectionCapture, CUDAAllFormats) { exercise(DeviceId::cuda(0), true); }
#else
TEST(GDNMixedProjectionCapture, ROCmProductionShapes) { exercise(DeviceId::rocm(0), false); }
TEST(GDNMixedProjectionCapture, ROCmAllFormats) { exercise(DeviceId::rocm(0), true); }
#endif

/** @brief Initialize the process and route counters before native graph tests. */
int main(int argc, char **argv)
{
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS)
        return 1;
    if (provided < MPI_THREAD_MULTIPLE)
    {
        MPI_Finalize();
        return 1;
    }
    setenv("LLAMINAR_PERF_STATS_JSON", "1", 1);
    mutableDebugEnv().reload();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
