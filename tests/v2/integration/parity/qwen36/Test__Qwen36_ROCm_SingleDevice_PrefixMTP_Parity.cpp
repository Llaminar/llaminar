/**
 * @file Test__Qwen36_ROCm_SingleDevice_PrefixMTP_Parity.cpp
 * @brief Native MPI fixture with an explicit process-entry backing owner.
 *
 * The executable enters the shared ancestor session before parsing arguments,
 * starting test environments, or initializing MPI. Direct binary runs retain
 * their actual descendants and runtime backing through terminal retirement.
 */
#include "backends/GPUDeviceProcessLifetime.h"
#include "utils/MPISessionOwner.h"
#include "Qwen36DenseParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    DensePrefixRestoreParityCase rocmSingleDeviceCase()
    {
        auto test_case = qwen36DensePrefixParityCase(
            "Qwen3.6 dense ROCm SingleDevice parity",
            DensePrefixParityTopology::SingleDevice);
        test_case.devices = {GlobalDeviceAddress::rocm(0)};
        test_case.required_cuda_devices = 0;
        test_case.required_rocm_devices = 1;
        return test_case;
    }
}

#define QWEN36_DENSE_PREFIX_MTP_SUITE Qwen36ROCmSingleDevicePrefixMTPParity
#define QWEN36_DENSE_PREFIX_MTP_CASE rocmSingleDeviceCase
#include "Qwen36DenseSingleDevicePrefixMTPParityTests.inc"

int main(int argc, char **argv)
{
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv))
        return *terminal;
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    std::string retirement_error;
    if (!releaseDenseMTPModelContextCampaignCache(&retirement_error))
    {
        std::cerr << retirement_error << '\n';
        result = 1;
    }

    GlobalBackendRouter::shutdown();
    llaminar2::GPUDeviceProcessLifetime::retireActive();

    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
