/**
 * @file Test__TensorFactoryBoundNUMAPolicy.cpp
 * @brief Explicitly admitted bound-node factory policy remains first-class.
 *
 * Build this CPU-only integration source as a separate executable from the
 * aggregate-policy tests: BackendManager correctly rejects a domain switch in
 * one process. The fixture restores its incoming thread policy, while production
 * explicitly bound factories retain their documented persistent-bind behavior.
 */
#include "TensorFactoryNUMAPolicyTestSupport.h"
#include "v2/backends/BackendManager.h"

namespace llaminar2::test
{
/** @test The admitted bound node wins over diagnostic rank and current affinity. */
TEST(TensorFactoryBoundNUMAPolicy, ExplicitBackendNodeRemainsTheFactoryAuthority)
{
    using namespace tensor_factory_numa;
    RestorePolicy restore;
    const int node = allowedNode();
    initCPUBackend(node);
    ASSERT_EQ(cpuBackendNUMANode(), node);
    const auto bound = bindPolicy(node);
    MPIContext context(node + 1, node + 2, MPI_COMM_NULL);
    for (int repeat = 0; repeat < 3; ++repeat)
    {
        install(Policy{});
        EXPECT_EQ(observe().mode, MPOL_DEFAULT);
        {
            TensorFactory factory(context);
            ASSERT_EQ(factory.getNumaNode(), node);
            EXPECT_EQ(factory.getMPIRank(), node + 1);
            EXPECT_EQ(observe(), bound);
            // Explicit ensure must re-establish the admitted bound policy even
            // when a caller deliberately changed its own policy between calls.
            install(Policy{});
            factory.ensureNumaBinding();
            EXPECT_EQ(observe(), bound);
            auto tensor = factory.createFP32({32u});
            ASSERT_NE(tensor, nullptr);
            EXPECT_EQ(observe(), bound);
        }
        EXPECT_EQ(observe(), bound) << "This slice preserves persistent bound semantics";
    }
    expectFreshWorkerPolicy(bound);
}
} // namespace llaminar2::test
