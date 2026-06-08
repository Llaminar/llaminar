#include "Qwen36MoEParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    MoEPrefixRestoreParityCase rocmSingleDeviceCase()
    {
        auto test_case = qwen36MoEPrefixParityCase(
            "Qwen3.6 MoE ROCm SingleDevice parity",
            MoEPrefixParityTopology::SingleDevice);
        test_case.devices = {GlobalDeviceAddress::rocm(0)};
        test_case.required_cuda_devices = 0;
        test_case.required_rocm_devices = 1;
        test_case.decode_steps = 2;
        return test_case;
    }

    MoEPrefixRestoreParityCase rocmSingleDeviceBenchmarkPromptCase()
    {
        auto test_case = rocmSingleDeviceCase();
        test_case.name = "Qwen3.6 MoE ROCm SingleDevice benchmark-prompt MTP diagnostic";
        test_case.prompt = qwen36MoEBenchmarkPrompt();
        test_case.metadata_envs = {"LLAMINAR_QWEN36_MOE_ROCM_MTP_DIAGNOSTIC_METADATA"};
        test_case.default_metadata_path =
            "pytorch_qwen36_moe_rocm_mtp_diagnostic_snapshots/metadata.txt";
        test_case.decode_steps = 4;
        test_case.max_seq_len = 768;
        return test_case;
    }

    MoEPrefixRestoreParityCase rocmSingleDeviceDepth3Case()
    {
        auto test_case = rocmSingleDeviceBenchmarkPromptCase();
        test_case.name = "Qwen3.6 MoE ROCm SingleDevice depth-3 MTP parity";
        test_case.decode_steps = 4;
        return test_case;
    }
} // namespace

#define QWEN36_MOE_PREFIX_MTP_SUITE Qwen36MoEROCmSingleDevicePrefixMTPParity
#define QWEN36_MOE_PREFIX_MTP_CASE rocmSingleDeviceCase
#define QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE rocmSingleDeviceBenchmarkPromptCase
#define QWEN36_MOE_PREFIX_MTP_DEPTH3_CASE rocmSingleDeviceDepth3Case
#include "Qwen36MoESingleDevicePrefixMTPParityTests.inc"

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
