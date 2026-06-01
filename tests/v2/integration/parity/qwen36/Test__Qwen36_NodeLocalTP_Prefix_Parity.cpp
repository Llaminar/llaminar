#include "Qwen36DenseParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    DensePrefixRestoreParityCase nodeLocalTPCase()
    {
        return qwen36DensePrefixParityCase(
            "Qwen3.6 dense NodeLocalTP parity",
            DensePrefixParityTopology::NodeLocalTP);
    }
}

TEST(Qwen36NodeLocalTPPrefixParity, PrefixRestoreFullHit)
{
    runDensePrefixRestoreParity(nodeLocalTPCase(), PrefixRestoreParityMode::FullHit);
}

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
