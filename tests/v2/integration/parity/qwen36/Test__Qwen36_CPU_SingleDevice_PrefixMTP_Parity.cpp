#include "Qwen36DenseParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    DensePrefixRestoreParityCase cpuSingleDeviceCase()
    {
        auto test_case = qwen36DensePrefixParityCase(
            "Qwen3.6 dense CPU SingleDevice parity",
            DensePrefixParityTopology::SingleDevice);
        test_case.devices = {GlobalDeviceAddress::cpu()};
        test_case.required_rocm_devices = 0;
        return test_case;
    }
}

#define QWEN36_DENSE_PREFIX_MTP_SUITE Qwen36CPUSingleDevicePrefixMTPParity
#define QWEN36_DENSE_PREFIX_MTP_CASE cpuSingleDeviceCase
#include "Qwen36DenseSingleDevicePrefixMTPParityTests.inc"

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
