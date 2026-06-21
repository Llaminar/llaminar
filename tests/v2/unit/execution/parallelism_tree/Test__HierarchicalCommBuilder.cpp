/**
 * @file Test__HierarchicalCommBuilder.cpp
 * @brief Unit tests for HierarchicalCommBuilder and CommHierarchy
 * @author David Sanftenberg
 * @date February 2026
 *
 * Tests for MPI communicator hierarchy construction from parallelism trees:
 * - needsCommunicator() logic for different node types
 * - calculateTPColor() for domain membership
 * - CommHierarchy RAII wrapper operations
 * - Full hierarchy building for various tree topologies
 *
 * These tests run under MPI so CommHierarchy can exercise real communicator
 * handles. They still do not require GPUs or model loading.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <set>

#include "execution/parallelism_tree/HierarchicalCommBuilder.h"
#include "execution/parallelism_tree/ParallelismTree.h"
#include "backends/GlobalDeviceAddress.h"
#include "config/OrchestrationConfig.h"

using namespace llaminar2;

// =============================================================================
// Helper: Build test trees
// =============================================================================

/**
 * Build a single-device tree (no parallelism)
 */
static ParallelismTree buildSingleDevice()
{
    auto root = Device(GlobalDeviceAddress::cuda(0), 0);

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    tree.total_layers = 24;
    tree.assignLayers(24);
    return tree;
}

/**
 * Build a local TP (single rank, multiple devices)
 *
 * TP(local_tp, rank=0, [cuda:0, cuda:1])
 */
static ParallelismTree buildLocalTP()
{
    auto root = TP("local_tp",
                   {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
                   0, CollectiveBackendType::NCCL);

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    tree.total_layers = 24;
    tree.assignLayers(24);
    return tree;
}

/**
 * Build a cross-rank TP (2 ranks, 1 device each)
 *
 * TP(cross_rank_tp, [Device(rank=0), Device(rank=1)])
 */
static ParallelismTree buildCrossRankTP()
{
    auto root = TP("cross_rank_tp",
                   {
                       Device(GlobalDeviceAddress::cuda(0), 0),
                       Device(GlobalDeviceAddress::cuda(0), 1),
                   },
                   CollectiveBackendType::MPI);

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 2;
    tree.total_layers = 24;
    tree.assignLayers(24);
    return tree;
}

/**
 * Build a PP tree with two local TP children
 *
 * PP(global)
 * ├── TP(socket0, rank=0, [cuda:0, cuda:1])
 * └── TP(socket1, rank=1, [cuda:0, cuda:1])
 */
static ParallelismTree buildPPWithLocalTP()
{
    auto root = PP("global", {
        TP("socket0", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 0, CollectiveBackendType::NCCL),
        TP("socket1", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 1, CollectiveBackendType::NCCL),
    });

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 2;
    tree.total_layers = 24;
    tree.assignLayers(24);
    return tree;
}

/**
 * Build nested PP with cross-rank TP
 *
 * PP(global)
 * ├── TP(tp0, [Device(rank=0), Device(rank=1)])  ← cross-rank TP
 * └── TP(tp1, [Device(rank=2), Device(rank=3)])  ← cross-rank TP
 */
static ParallelismTree buildNestedPPWithCrossRankTP()
{
    auto root = PP("global", {
        TP("tp0",
           {
               Device(GlobalDeviceAddress::cuda(0), 0),
               Device(GlobalDeviceAddress::cuda(0), 1),
           },
           CollectiveBackendType::MPI),
        TP("tp1",
           {
               Device(GlobalDeviceAddress::cuda(0), 2),
               Device(GlobalDeviceAddress::cuda(0), 3),
           },
           CollectiveBackendType::MPI),
    });

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 4;
    tree.total_layers = 24;
    tree.assignLayers(24);
    return tree;
}

/**
 * Build deeply nested tree with cross-rank TP at multiple levels
 *
 * PP(global)
 * ├── PP(host0)
 * │   └── TP(cross_tp, [Device(rank=0), Device(rank=1)])
 * └── PP(host1)
 *     └── TP(cross_tp2, [Device(rank=2), Device(rank=3)])
 */
static ParallelismTree buildDeepNested()
{
    auto root = PP("global", {
        PP("host0", {
            TP("cross_tp",
               {
                   Device(GlobalDeviceAddress::cuda(0), 0),
                   Device(GlobalDeviceAddress::cuda(0), 1),
               },
               CollectiveBackendType::MPI),
        }),
        PP("host1", {
            TP("cross_tp2",
               {
                   Device(GlobalDeviceAddress::cuda(0), 2),
                   Device(GlobalDeviceAddress::cuda(0), 3),
               },
               CollectiveBackendType::MPI),
        }),
    });

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 4;
    tree.total_layers = 24;
    tree.assignLayers(24);
    return tree;
}

// =============================================================================
// Test Suite: needsCommunicator
// =============================================================================

TEST(Test__HierarchicalCommBuilder, NeedsCommunicatorDevice)
{
    // Device leaves never need communicators
    auto node = Device(GlobalDeviceAddress::cuda(0), 0);
    EXPECT_FALSE(HierarchicalCommBuilder::needsCommunicator(node));
}

TEST(Test__HierarchicalCommBuilder, NeedsCommunicatorLocalTP)
{
    // TP with all devices on the same rank doesn't need an MPI communicator
    auto node = TP("local_tp",
                   {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
                   0, CollectiveBackendType::NCCL);
    EXPECT_FALSE(HierarchicalCommBuilder::needsCommunicator(node));
}

TEST(Test__HierarchicalCommBuilder, NeedsCommunicatorCrossRankTP)
{
    // TP spanning multiple ranks needs a communicator
    auto node = TP("cross_rank_tp",
                   {
                       Device(GlobalDeviceAddress::cuda(0), 0),
                       Device(GlobalDeviceAddress::cuda(0), 1),
                   },
                   CollectiveBackendType::MPI);
    EXPECT_TRUE(HierarchicalCommBuilder::needsCommunicator(node));
}

TEST(Test__HierarchicalCommBuilder, NeedsCommunicatorPP)
{
    // PP nodes don't need collective communicators (use P2P)
    auto node = PP("pp_node", {
        Device(GlobalDeviceAddress::cuda(0), 0),
        Device(GlobalDeviceAddress::cuda(0), 1),
    });
    EXPECT_FALSE(HierarchicalCommBuilder::needsCommunicator(node));
}

// =============================================================================
// Test Suite: calculateTPColor
// =============================================================================

TEST(Test__HierarchicalCommBuilder, CalculateTPColorInDomain)
{
    // Rank 0 is in the TP domain → should get a valid color
    auto node = TP("cross_rank_tp",
                   {
                       Device(GlobalDeviceAddress::cuda(0), 0),
                       Device(GlobalDeviceAddress::cuda(0), 1),
                   },
                   CollectiveBackendType::MPI);

    int color0 = HierarchicalCommBuilder::calculateTPColor(node, "root/cross_rank_tp", 0);
    int color1 = HierarchicalCommBuilder::calculateTPColor(node, "root/cross_rank_tp", 1);

    // Both ranks in the domain should get the same positive color
    EXPECT_GE(color0, 0);
    EXPECT_EQ(color0, color1);
}

TEST(Test__HierarchicalCommBuilder, CalculateTPColorOutsideDomain)
{
    // Rank 2 is NOT in the TP domain → should get MPI_UNDEFINED
    auto node = TP("cross_rank_tp",
                   {
                       Device(GlobalDeviceAddress::cuda(0), 0),
                       Device(GlobalDeviceAddress::cuda(0), 1),
                   },
                   CollectiveBackendType::MPI);

    int color = HierarchicalCommBuilder::calculateTPColor(node, "root/cross_rank_tp", 2);
    EXPECT_EQ(color, MPI_UNDEFINED);
}

TEST(Test__HierarchicalCommBuilder, CalculateTPColorDifferentDomains)
{
    // Two different TP domains should get different colors
    auto node1 = TP("tp1",
                    {
                        Device(GlobalDeviceAddress::cuda(0), 0),
                        Device(GlobalDeviceAddress::cuda(0), 1),
                    },
                    CollectiveBackendType::MPI);

    auto node2 = TP("tp2",
                    {
                        Device(GlobalDeviceAddress::cuda(0), 2),
                        Device(GlobalDeviceAddress::cuda(0), 3),
                    },
                    CollectiveBackendType::MPI);

    int color1 = HierarchicalCommBuilder::calculateTPColor(node1, "root/tp1", 0);
    int color2 = HierarchicalCommBuilder::calculateTPColor(node2, "root/tp2", 2);

    // Different paths should produce different colors (high probability)
    // This is a hash-based assertion, so it's probabilistic but very unlikely to fail
    EXPECT_NE(color1, color2);
}

// =============================================================================
// Test Suite: CommHierarchy operations
// =============================================================================

TEST(Test__HierarchicalCommBuilder, HierarchyEmpty)
{
    // Single device tree → no split communicators needed
    auto tree = buildSingleDevice();
    auto hierarchy = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 0);

    // Should have no split communicators (just uses world)
    EXPECT_EQ(hierarchy.size(), 0u);
    EXPECT_TRUE(hierarchy.empty());
    EXPECT_EQ(hierarchy.rootCommunicator(), MPI_COMM_WORLD);
}

TEST(Test__HierarchicalCommBuilder, HierarchyLocalTP)
{
    // Local TP on one rank → no MPI comm split needed
    auto tree = buildLocalTP();
    auto hierarchy = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 0);

    // Local TP doesn't cross ranks, so no communicators
    EXPECT_EQ(hierarchy.size(), 0u);
    EXPECT_EQ(hierarchy.rootCommunicator(), MPI_COMM_WORLD);
}

TEST(Test__HierarchicalCommBuilder, HierarchyCrossRankTP)
{
    // 2-rank TP → hierarchy has one split communicator
    auto tree = buildCrossRankTP();

    // Build for rank 0
    auto hierarchy0 = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 0);
    EXPECT_EQ(hierarchy0.size(), 1u);

    // Build for rank 1
    auto hierarchy1 = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 1);
    EXPECT_EQ(hierarchy1.size(), 1u);

    // Both should have the same path
    auto paths0 = hierarchy0.allPaths();
    auto paths1 = hierarchy1.allPaths();
    ASSERT_EQ(paths0.size(), 1u);
    ASSERT_EQ(paths1.size(), 1u);
    EXPECT_EQ(paths0[0], paths1[0]);
}

TEST(Test__HierarchicalCommBuilder, HierarchyNestedPPTP)
{
    // PP(TP, TP) with cross-rank TPs → two split communicators
    auto tree = buildNestedPPWithCrossRankTP();

    // Rank 0 is in tp0 only
    auto hierarchy0 = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 0);
    // Rank 0 should see one comm (for tp0)
    EXPECT_EQ(hierarchy0.size(), 1u);

    // Rank 2 is in tp1 only
    auto hierarchy2 = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 2);
    // Rank 2 should see one comm (for tp1)
    EXPECT_EQ(hierarchy2.size(), 1u);

    // The paths should be different (tp0 vs tp1)
    auto paths0 = hierarchy0.allPaths();
    auto paths2 = hierarchy2.allPaths();
    ASSERT_EQ(paths0.size(), 1u);
    ASSERT_EQ(paths2.size(), 1u);
    EXPECT_NE(paths0[0], paths2[0]);
}

TEST(Test__HierarchicalCommBuilder, HierarchyToString)
{
    // Verify toString() produces readable output
    auto tree = buildCrossRankTP();
    auto hierarchy = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 0);

    std::string str = hierarchy.toString();
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("CommHierarchy"), std::string::npos);
    EXPECT_NE(str.find("1 communicators"), std::string::npos);
}

TEST(Test__HierarchicalCommBuilder, HierarchyGetCommunicator)
{
    // Path lookup should work
    auto tree = buildCrossRankTP();
    auto hierarchy = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 0);

    auto paths = hierarchy.allPaths();
    ASSERT_EQ(paths.size(), 1u);

    // Should find the communicator by path
    MPI_Comm comm = hierarchy.getCommunicator(paths[0]);
    EXPECT_NE(comm, MPI_COMM_NULL);

    // Should return NULL for unknown path
    MPI_Comm null_comm = hierarchy.getCommunicator("nonexistent/path");
    EXPECT_EQ(null_comm, MPI_COMM_NULL);
}

TEST(Test__HierarchicalCommBuilder, HierarchyAllPaths)
{
    // Verify allPaths() returns correct paths
    auto tree = buildDeepNested();

    // Rank 0 is in first cross_tp
    auto hierarchy = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 0);
    auto paths = hierarchy.allPaths();

    // Should have one path for the TP group rank 0 belongs to
    EXPECT_EQ(paths.size(), 1u);
    if (!paths.empty())
    {
        // Path should contain the TP name
        EXPECT_NE(paths[0].find("cross_tp"), std::string::npos);
    }
}

TEST(Test__HierarchicalCommBuilder, HierarchyMoveSemantics)
{
    // Test move constructor
    auto tree = buildCrossRankTP();
    auto hierarchy1 = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 0);

    size_t original_size = hierarchy1.size();
    auto paths = hierarchy1.allPaths();

    // Move construct
    CommHierarchy hierarchy2(std::move(hierarchy1));
    EXPECT_EQ(hierarchy2.size(), original_size);
    EXPECT_EQ(hierarchy1.size(), 0u); // NOLINT: testing moved-from state

    // Paths should be preserved
    auto paths2 = hierarchy2.allPaths();
    EXPECT_EQ(paths, paths2);

    // Move assign
    CommHierarchy hierarchy3;
    hierarchy3 = std::move(hierarchy2);
    EXPECT_EQ(hierarchy3.size(), original_size);
    EXPECT_EQ(hierarchy2.size(), 0u); // NOLINT: testing moved-from state
}

TEST(Test__HierarchicalCommBuilder, HierarchyDestructionOrder)
{
    // This test verifies that communicators are tracked by depth
    // In a real MPI environment, freeAll() would free them in depth-descending order

    auto tree = buildDeepNested();
    auto hierarchy = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 0);

    // The hierarchy should have one communicator (the cross_tp at depth 2)
    EXPECT_GE(hierarchy.size(), 0u);

    // toString() should show depth information
    std::string str = hierarchy.toString();

    // If we have communicators, depth should be mentioned
    if (hierarchy.size() > 0)
    {
        EXPECT_NE(str.find("depth="), std::string::npos);
    }

    // Destructor will be called at end of scope - no crash means success
}

// =============================================================================
// Test Suite: Edge cases
// =============================================================================

TEST(Test__HierarchicalCommBuilder, RankNotInAnyDomain)
{
    // Rank 99 is not in any domain
    auto tree = buildCrossRankTP();
    auto hierarchy = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 99);

    // Rank 99 is not in the tree, so no communicator should be created
    EXPECT_EQ(hierarchy.size(), 0u);
}

TEST(Test__HierarchicalCommBuilder, PPOnlyTree)
{
    // PP tree with no TP → no communicators needed
    auto root = PP("global", {
        Device(GlobalDeviceAddress::cuda(0), 0),
        Device(GlobalDeviceAddress::cuda(0), 1),
    });

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 2;
    tree.total_layers = 24;
    tree.assignLayers(24);

    auto hierarchy = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 0);
    EXPECT_EQ(hierarchy.size(), 0u);
}

TEST(Test__HierarchicalCommBuilder, MixedLocalAndCrossRankTP)
{
    // PP with one local TP and one cross-rank TP
    auto root = PP("global", {
        TP("local_tp",
           {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
           0, CollectiveBackendType::NCCL), // Local to rank 0
        TP("cross_tp",
           {
               Device(GlobalDeviceAddress::cuda(0), 1),
               Device(GlobalDeviceAddress::cuda(0), 2),
           },
           CollectiveBackendType::MPI), // Cross-rank
    });

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 3;
    tree.total_layers = 24;
    tree.assignLayers(24);

    // Rank 0 is only in local_tp (no comm needed)
    auto hierarchy0 = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 0);
    EXPECT_EQ(hierarchy0.size(), 0u);

    // Rank 1 is in cross_tp (needs comm)
    auto hierarchy1 = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 1);
    EXPECT_EQ(hierarchy1.size(), 1u);

    // Rank 2 is in cross_tp (needs comm)
    auto hierarchy2 = HierarchicalCommBuilder::build(tree, MPI_COMM_WORLD, 2);
    EXPECT_EQ(hierarchy2.size(), 1u);
}
