/**
 * @file Test__ClusterInventoryPublication.cpp
 * @brief Device-free failure/lifetime proofs for context-owned discovery.
 *
 * Invalid communicator identity rejects before any MPI or hardware call. Its
 * retained failure must survive sequential and concurrent readers, including
 * topology construction. Real successful publications belong to MPI preflight.
 */
#include <gtest/gtest.h>
#include "utils/MPIContext.h"
#include "planning/ClusterInventoryGatherer.h"
#include "../../mocks/MockMPIContext.h"
#include <atomic>
#include <thread>
#include <type_traits>

using namespace llaminar2;

namespace
{
    /** @brief Device-free retained publication, without discovering or entering MPI. */
    class PublishedDiscoveryContext final : public test::MockMPIContext
    {
    public:
        /** @brief Supply the exact already-published immutable owner. */
        explicit PublishedDiscoveryContext(std::shared_ptr<const ClusterInventory> inventory)
            : inventory_(std::move(inventory)) {}
        /** @return Same publication for every reader; no mutable snapshot copy. */
        std::shared_ptr<const ClusterInventory> clusterInventory() const override { return inventory_; }
    private:
        std::shared_ptr<const ClusterInventory> inventory_;
    };

    /** @brief Count attempted discovery before rejecting an invalid communicator. */
    class InvalidDiscoveryContext final : public MPIContext
    {
    public:
        /** @brief A multi-rank identity with no live communicator cannot discover. */
        InvalidDiscoveryContext() : MPIContext(0, 2, MPI_COMM_NULL) {}
        /** @return Declared size while counting each attempted observation. */
        int world_size() const override { ++observations; return 2; }
        mutable std::atomic<int> observations{0};
    };

    /** @brief Execute independent readers and join before inspecting their results. */
    template <class Reader>
    void parallelReaders(Reader reader)
    {
        std::vector<std::jthread> threads;
        for (int i = 0; i < 20; ++i) threads.emplace_back(reader);
    }
}

TEST(ClusterInventoryPublication, GatherRetainsTheImmutableOwnerInsteadOfCopyingItsFacts)
{
    auto publication = std::make_shared<const ClusterInventory>();
    std::weak_ptr<const ClusterInventory> lifetime = publication;
    auto context = std::make_shared<PublishedDiscoveryContext>(publication);
    auto retained = gatherClusterInventory(context);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(*retained)>>);
    EXPECT_EQ(retained, publication);
    EXPECT_EQ(gatherClusterInventory(context), retained);
    context.reset();
    publication.reset();
    EXPECT_FALSE(lifetime.expired());
    retained.reset();
    EXPECT_TRUE(lifetime.expired());
}

TEST(ClusterInventoryPublication, FailedDiscoveryIsNotRetriedBySequentialReaders)
{
    auto context = std::make_shared<InvalidDiscoveryContext>();
    for (int i = 0; i < 20; ++i)
        EXPECT_THROW(gatherClusterInventory(context), std::invalid_argument);
    EXPECT_EQ(context->observations.load(), 1);
}

TEST(ClusterInventoryPublication, ConcurrentReadersObserveOneOriginalFailure)
{
    auto context = std::make_shared<InvalidDiscoveryContext>();
    std::exception_ptr original;
    try { context->clusterInventory(); }
    catch (...) { original = std::current_exception(); }
    ASSERT_TRUE(original);
    parallelReaders([&] {
        try { context->clusterInventory(); ADD_FAILURE() << "Failed discovery became usable"; }
        catch (...) { EXPECT_EQ(std::current_exception(), original); }
    });
    EXPECT_EQ(context->observations.load(), 1);
}

TEST(ClusterInventoryPublication, FirstConcurrentPublicationOnlyAttemptsDiscoveryOnce)
{
    InvalidDiscoveryContext context;
    parallelReaders([&] {
        EXPECT_THROW(context.clusterInventory(), std::invalid_argument);
    });
    EXPECT_EQ(context.observations.load(), 1);
}

TEST(ClusterInventoryPublication, TopologyFailureCannotRestartItsConstruction)
{
    InvalidDiscoveryContext context;
    parallelReaders([&] { EXPECT_THROW(context.topology(), std::invalid_argument); });
    EXPECT_EQ(context.observations.load(), 1);
    EXPECT_THROW(context.concrete_topology(), std::invalid_argument);
    EXPECT_EQ(context.observations.load(), 1);
}

TEST(ClusterInventoryPublication, MockWithoutDiscoveryCannotInitializeRealDevices)
{
    auto context = std::make_shared<llaminar2::test::MockMPIContext>();
    EXPECT_THROW(context->clusterInventory(), std::logic_error);
    EXPECT_THROW(gatherClusterInventory(context), std::logic_error);
}

TEST(ClusterInventoryPublication, NullCommunicatorIsNotLocalDiscovery)
{
    MPIContext context(0, 1, MPI_COMM_NULL);
    EXPECT_THROW(context.clusterInventory(), std::invalid_argument);
}
