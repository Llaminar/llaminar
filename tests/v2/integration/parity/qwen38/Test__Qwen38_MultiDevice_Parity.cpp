/**
 * @file Test__Qwen38_MultiDevice_Parity.cpp
 * @brief Canonical dense Qwen3.8 TP, PP and cross-vendor TP/PP model matrix.
 *
 * One typed definition owns each topology and expands all ordinary precision,
 * MTP and prefix proofs. HTTP E2E projects dynamic-depth cells into automatic
 * planner constraints; no second HTTP-only topology list or runner is built.
 */
#include "../qwen35/Qwen35ParityTestBase.h"
#include "Qwen38ModelParityDefinitions.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include <iterator>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35;

namespace
{
    /** @return The single canonical expander's complete diagnostic matrix. */
    std::vector<ModelParityCase> cases()
    {
        std::vector<ModelParityCase> result;
        for (const auto &definition : qwen38::qwen38DenseMultiDeviceDefinitions())
        {
            auto expanded = expandModelParityDefinition(definition);
            result.insert(result.end(), std::make_move_iterator(expanded.begin()),
                          std::make_move_iterator(expanded.end()));
        }
        return result;
    }
}

/** @brief Reuse the live hybrid-GDN/HF diagnostic fixture without custom execution. */
class Qwen38DenseMultiDeviceParityTest
    : public Qwen35ConfigDrivenParityTest<Qwen38DenseMultiDeviceParityTest>,
      public ModelParityCaseParameter
{};

TEST_P(Qwen38DenseMultiDeviceParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

INSTANTIATE_TEST_SUITE_P(Qwen38Dense, Qwen38DenseMultiDeviceParityTest,
    ::testing::ValuesIn(cases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });

/** @brief Retire backend contexts before MPI while preserving the test result. */
int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();
    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
