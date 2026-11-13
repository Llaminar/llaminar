/**
 * @file Test__TensorFactoryAggregateNUMAPolicy.cpp
 * @brief Actual aggregate factory policy and new-worker inheritance regression.
 *
 * This CPU-only integration executable has its own aggregate backend lifetime.
 * Kernel policy operations must be permitted; denial is failure rather than a
 * fake pass. It uses no native GPU, model, source payload or large allocation.
 */
#include "TensorFactoryNUMAPolicyTestSupport.h"
#include "v2/backends/BackendManager.h"

namespace llaminar2::test
{
namespace
{
using namespace tensor_factory_numa;
/**
 * @brief Exercise construction, external allocation hook, failure and reset.
 * @param expected Exact incoming policy which aggregate operations must preserve.
 */
void expectAggregateOperationsPreserve(const Policy &expected)
{
    initCPUBackend(-1);
    ASSERT_EQ(cpuBackendNUMANode(), -1);
    EXPECT_EQ(observe(), expected);
    for (const int rank : {0, 1, 7})
    {
        MPIContext context(rank, 8, MPI_COMM_NULL);
        for (int repeat = 0; repeat < 3; ++repeat)
        {
            {
                TensorFactory factory(context);
                ASSERT_EQ(factory.getNumaNode(), -1);
                EXPECT_EQ(factory.getMPIRank(), rank);
                EXPECT_EQ(observe(), expected) << "Constructor changed aggregate policy";
                factory.ensureNumaBinding();
                EXPECT_EQ(observe(), expected) << "External allocation hook changed policy";
                auto fp32 = factory.createFP32({32u});
                auto fp16 = factory.createFP16({32u});
                auto quantized = factory.createActivation(
                    {1u, 32u}, ActivationPrecision::Q8_1, DeviceId::cpu());
                ASSERT_NE(fp32, nullptr);
                ASSERT_NE(fp16, nullptr);
                ASSERT_NE(quantized, nullptr);
                EXPECT_EQ(observe(), expected) << "Native CPU construction changed policy";
                EXPECT_THROW(factory.createQuantized(static_cast<TensorType>(-1), {32u}, {}),
                             std::runtime_error);
                EXPECT_EQ(observe(), expected) << "Failed construction changed policy";
            }
            EXPECT_EQ(observe(), expected) << "Destruction changed incoming policy";
        }
    }
    expectFreshWorkerPolicy(expected);
}
} // namespace
/** @test Aggregate admission must not install singleton MPOL_BIND at any factory entry. */
TEST(TensorFactoryAggregateNUMAPolicy, DefaultPolicyAndNewWorkersRemainAggregate)
{
    RestorePolicy restore;
    install(Policy{});
    const auto expected = observe();
    ASSERT_EQ(expected.mode, MPOL_DEFAULT);
    expectAggregateOperationsPreserve(expected);
}
/** @test Aggregate intent preserves an external bind; it does not reset caller policy. */
TEST(TensorFactoryAggregateNUMAPolicy, ExistingExternalBindIsPreservedExactly)
{
    RestorePolicy restore;
    install(bindPolicy(allowedNode()));
    const auto expected = observe();
    ASSERT_EQ(expected.mode, MPOL_BIND);
    expectAggregateOperationsPreserve(expected);
}
} // namespace llaminar2::test
