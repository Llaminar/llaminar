/**
 * @file Test__ParallelismTree.cpp
 * @brief Unit tests for ParallelismTree recursive data structure
 * @author David Sanftenberg
 * @date February 2026
 *
 * Tests for parallelism tree construction, layer assignment, validation,
 * fluent builders, and tree queries:
 * - Construction with fluent builders (PP, TP, Device)
 * - Layer assignment for equal and proportional splits
 * - Validation catches invalid trees (TP with 1 child, PP with gaps, etc.)
 * - leafRanks() and leafDevices() correctness
 * - isCrossRank() for local vs distributed subtrees
 * - toString() produces expected structure
 * - Edge cases: single device, degenerate trees
 *
 * All tests are pure data-structure tests — no MPI, no GPU, no model loading.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <set>

#include "execution/parallelism_tree/ParallelismTree.h"
#include "backends/GlobalDeviceAddress.h"
#include "config/OrchestrationConfig.h"

using namespace llaminar2;

// =============================================================================
// Helper: build the canonical 4-machine example from the plan
// =============================================================================

/**
 * Build the 4-machine × 2-socket × 2-GPU topology from the plan document:
 *
 * PP(global)
 * ├── PP(host0)
 * │   ├── TP(socket0, rank=0, [cuda:0, cuda:1])
 * │   └── TP(socket1, rank=1, [cuda:0, cuda:1])
 * └── PP(host1)
 *     ├── TP(socket2, rank=2, [cuda:0, cuda:1])
 *     └── TP(socket3, rank=3, [cuda:0, cuda:1])
 */
static ParallelismTree build4Machine2Socket2GPU()
{
    auto root = PP("global", {
                                 PP("host0", {
                                                 TP("socket0", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 0, CollectiveBackendType::NCCL),
                                                 TP("socket1", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 1, CollectiveBackendType::NCCL),
                                             }),
                                 PP("host1", {
                                                 TP("socket2", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 2, CollectiveBackendType::NCCL),
                                                 TP("socket3", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 3, CollectiveBackendType::NCCL),
                                             }),
                             });

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 4;
    return tree;
}

/**
 * Build a simple 2-rank PP with local TP:
 *
 * PP(cross_socket)
 * ├── TP(socket0, rank=0, [cuda:0, cuda:1])
 * └── TP(socket1, rank=1, [rocm:0, rocm:1])
 */
static ParallelismTree buildSimple2RankPPTP()
{
    auto root = PP("cross_socket", {
                                       TP("socket0", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 0, CollectiveBackendType::NCCL),
                                       TP("socket1", {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)}, 1, CollectiveBackendType::RCCL),
                                   });

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 2;
    return tree;
}

/**
 * Build a mixed CUDA+ROCm TP domain using HETEROGENEOUS backend:
 *
 * TP(mixed_tp, rank=0, [cuda:0, rocm:0], backend=HETEROGENEOUS)
 *
 * This is the canonical mixed-vendor TP case: two GPUs from different vendors
 * on the same PCIe fabric, using host-staged allreduce.
 */
static ParallelismTree buildMixedTPPcieBar()
{
    auto root = TP("mixed_tp",
                   {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)},
                   0, CollectiveBackendType::HETEROGENEOUS);

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    return tree;
}

/**
 * Build a PP topology with mixed-vendor TP on each socket:
 *
 * PP(cross_socket)
 * ├── TP(socket0, rank=0, [cuda:0, rocm:0], backend=HETEROGENEOUS)
 * └── TP(socket1, rank=1, [cuda:1, rocm:1], backend=HETEROGENEOUS)
 */
static ParallelismTree buildPPWithMixedTP()
{
    auto root = PP("cross_socket", {
                                       TP("socket0", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)}, 0, CollectiveBackendType::HETEROGENEOUS),
                                       TP("socket1", {GlobalDeviceAddress::cuda(1), GlobalDeviceAddress::rocm(1)}, 1, CollectiveBackendType::HETEROGENEOUS),
                                   });

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 2;
    return tree;
}

/**
 * Build a heterogeneous TP domain with 4 GPUs (2 CUDA + 2 ROCm):
 *
 * TP(hetero_tp, rank=0, [cuda:0, cuda:1, rocm:0, rocm:1], backend=HETEROGENEOUS)
 *
 * The HETEROGENEOUS backend orchestrates NCCL among CUDA devices, RCCL among
 * ROCm devices, and HETEROGENEOUS for cross-vendor pairs.
 */
static ParallelismTree buildHeterogeneousTP()
{
    auto root = TP("hetero_tp",
                   {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1),
                    GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)},
                   0, CollectiveBackendType::HETEROGENEOUS);

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    return tree;
}

// =============================================================================
// Test Suite: Construction with Fluent Builders
// =============================================================================

TEST(Test__ParallelismTree, DeviceLeafCreation)
{
    auto node = Device(GlobalDeviceAddress::cuda(0), 0);

    EXPECT_EQ(node.type, ParallelismNodeType::DEVICE);
    EXPECT_EQ(node.owning_rank, 0);
    EXPECT_TRUE(node.isLeaf());
    EXPECT_FALSE(node.isInterior());
    EXPECT_EQ(node.leafCount(), 1);
    EXPECT_TRUE(node.children.empty());
}

TEST(Test__ParallelismTree, TPNodeFromDevices)
{
    auto node = TP("my_tp",
                   {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
                   0, CollectiveBackendType::NCCL);

    EXPECT_EQ(node.type, ParallelismNodeType::TENSOR_PARALLEL);
    EXPECT_EQ(node.name, "my_tp");
    EXPECT_EQ(node.backend, CollectiveBackendType::NCCL);
    EXPECT_FALSE(node.isLeaf());
    EXPECT_TRUE(node.isInterior());
    EXPECT_EQ(node.children.size(), 2u);
    EXPECT_EQ(node.tpDegree(), 2);

    // Children should be DEVICE leaves
    for (const auto &child : node.children)
    {
        EXPECT_EQ(child.type, ParallelismNodeType::DEVICE);
        EXPECT_EQ(child.owning_rank, 0);
    }

    EXPECT_EQ(node.leafCount(), 2);
}

TEST(Test__ParallelismTree, TPNodeFromChildren)
{
    auto node = TP("cross_rank_tp",
                   {Device(GlobalDeviceAddress::cuda(0), 0),
                    Device(GlobalDeviceAddress::cuda(0), 1)},
                   CollectiveBackendType::MPI);

    EXPECT_EQ(node.type, ParallelismNodeType::TENSOR_PARALLEL);
    EXPECT_EQ(node.children.size(), 2u);
    EXPECT_EQ(node.backend, CollectiveBackendType::MPI);
    EXPECT_TRUE(node.isCrossRank());
}

TEST(Test__ParallelismTree, PPNodeCreation)
{
    auto node = PP("pipeline", {
                                   Device(GlobalDeviceAddress::cuda(0), 0),
                                   Device(GlobalDeviceAddress::cuda(1), 0),
                               });

    EXPECT_EQ(node.type, ParallelismNodeType::PIPELINE_PARALLEL);
    EXPECT_EQ(node.name, "pipeline");
    EXPECT_EQ(node.children.size(), 2u);
    EXPECT_EQ(node.tpDegree(), 1); // PP nodes always return 1
    EXPECT_EQ(node.leafCount(), 2);
}

TEST(Test__ParallelismTree, NestedTreeConstruction)
{
    auto tree = build4Machine2Socket2GPU();

    // Root is PP
    EXPECT_EQ(tree.root.type, ParallelismNodeType::PIPELINE_PARALLEL);
    EXPECT_EQ(tree.root.name, "global");
    EXPECT_EQ(tree.root.children.size(), 2u);

    // Each host is PP with 2 TP children
    for (const auto &host : tree.root.children)
    {
        EXPECT_EQ(host.type, ParallelismNodeType::PIPELINE_PARALLEL);
        EXPECT_EQ(host.children.size(), 2u);

        for (const auto &socket : host.children)
        {
            EXPECT_EQ(socket.type, ParallelismNodeType::TENSOR_PARALLEL);
            EXPECT_EQ(socket.children.size(), 2u);
        }
    }

    // 4 sockets × 2 GPUs = 8 leaf devices
    EXPECT_EQ(tree.root.leafCount(), 8);
}

// =============================================================================
// Test Suite: Layer Assignment
// =============================================================================

TEST(Test__ParallelismTree, AssignLayersEqualSplit)
{
    auto tree = buildSimple2RankPPTP();
    tree.assignLayers(24);

    EXPECT_EQ(tree.total_layers, 24);

    // Root gets all layers
    EXPECT_EQ(tree.root.first_layer, 0);
    EXPECT_EQ(tree.root.last_layer, 23);
    EXPECT_TRUE(tree.root.has_embedding);
    EXPECT_TRUE(tree.root.has_lm_head);

    // First PP child (socket0): layers 0-11
    const auto &s0 = tree.root.children[0];
    EXPECT_EQ(s0.first_layer, 0);
    EXPECT_EQ(s0.last_layer, 11);
    EXPECT_TRUE(s0.has_embedding);
    EXPECT_FALSE(s0.has_lm_head);

    // Second PP child (socket1): layers 12-23
    const auto &s1 = tree.root.children[1];
    EXPECT_EQ(s1.first_layer, 12);
    EXPECT_EQ(s1.last_layer, 23);
    EXPECT_FALSE(s1.has_embedding);
    EXPECT_TRUE(s1.has_lm_head);

    // TP children get same range as parent
    for (const auto &leaf : s0.children)
    {
        EXPECT_EQ(leaf.first_layer, 0);
        EXPECT_EQ(leaf.last_layer, 11);
    }
    for (const auto &leaf : s1.children)
    {
        EXPECT_EQ(leaf.first_layer, 12);
        EXPECT_EQ(leaf.last_layer, 23);
    }
}

TEST(Test__ParallelismTree, AssignLayers4Machine48Layers)
{
    auto tree = build4Machine2Socket2GPU();
    tree.assignLayers(48);

    // 48 layers, 2 top-level PP children: host0 gets 24, host1 gets 24
    EXPECT_EQ(tree.root.children[0].first_layer, 0);
    EXPECT_EQ(tree.root.children[0].last_layer, 23);
    EXPECT_EQ(tree.root.children[1].first_layer, 24);
    EXPECT_EQ(tree.root.children[1].last_layer, 47);

    // host0: 2 sockets, 24 layers → 12 each
    const auto &host0 = tree.root.children[0];
    EXPECT_EQ(host0.children[0].first_layer, 0);
    EXPECT_EQ(host0.children[0].last_layer, 11);
    EXPECT_EQ(host0.children[1].first_layer, 12);
    EXPECT_EQ(host0.children[1].last_layer, 23);

    // host1: 2 sockets, 24 layers → 12 each
    const auto &host1 = tree.root.children[1];
    EXPECT_EQ(host1.children[0].first_layer, 24);
    EXPECT_EQ(host1.children[0].last_layer, 35);
    EXPECT_EQ(host1.children[1].first_layer, 36);
    EXPECT_EQ(host1.children[1].last_layer, 47);

    // Exactly the last socket's last TP device has has_lm_head
    auto leaves = tree.root.leafDevices();
    int emb_count = 0, lm_count = 0;
    for (auto *leaf : leaves)
    {
        if (leaf->has_embedding)
            emb_count++;
        if (leaf->has_lm_head)
            lm_count++;
    }
    // All TP siblings share flags, so 2 leaves have embedding (socket0's 2 devices)
    // and 2 have lm_head (socket3's 2 devices)
    EXPECT_EQ(emb_count, 2);
    EXPECT_EQ(lm_count, 2);
}

TEST(Test__ParallelismTree, AssignLayersOddSplit)
{
    // 7 layers across 3 PP children → 3, 2, 2 (extra goes to first children)
    auto root = PP("pipeline", {
                                   Device(GlobalDeviceAddress::cuda(0), 0),
                                   Device(GlobalDeviceAddress::cuda(1), 0),
                                   Device(GlobalDeviceAddress::cuda(2), 0),
                               });

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    tree.assignLayers(7);

    EXPECT_EQ(tree.root.children[0].first_layer, 0);
    EXPECT_EQ(tree.root.children[0].last_layer, 2); // 3 layers
    EXPECT_EQ(tree.root.children[1].first_layer, 3);
    EXPECT_EQ(tree.root.children[1].last_layer, 4); // 2 layers
    EXPECT_EQ(tree.root.children[2].first_layer, 5);
    EXPECT_EQ(tree.root.children[2].last_layer, 6); // 2 layers
}

TEST(Test__ParallelismTree, AssignLayersSingleDevice)
{
    auto node = Device(GlobalDeviceAddress::cpu(), 0);

    ParallelismTree tree;
    tree.root = std::move(node);
    tree.world_size = 1;
    tree.assignLayers(24);

    EXPECT_EQ(tree.root.first_layer, 0);
    EXPECT_EQ(tree.root.last_layer, 23);
    EXPECT_TRUE(tree.root.has_embedding);
    EXPECT_TRUE(tree.root.has_lm_head);
}

// =============================================================================
// Test Suite: Validation
// =============================================================================

TEST(Test__ParallelismTree, ValidCorrectTree)
{
    auto tree = buildSimple2RankPPTP();
    tree.assignLayers(24);
    auto errors = tree.validate();
    EXPECT_TRUE(errors.empty()) << "Validation errors: " << errors[0];
}

TEST(Test__ParallelismTree, Valid4MachineTree)
{
    auto tree = build4Machine2Socket2GPU();
    tree.assignLayers(48);
    auto errors = tree.validate();
    EXPECT_TRUE(errors.empty()) << "Validation errors: " << errors[0];
}

TEST(Test__ParallelismTree, InvalidTPWith1Child)
{
    // TP with only 1 child is invalid
    ParallelismNode tp_node;
    tp_node.type = ParallelismNodeType::TENSOR_PARALLEL;
    tp_node.name = "bad_tp";
    tp_node.children.push_back(Device(GlobalDeviceAddress::cuda(0), 0));

    ParallelismTree tree;
    tree.root = std::move(tp_node);
    tree.world_size = 1;
    tree.assignLayers(24);

    auto errors = tree.validate();
    EXPECT_FALSE(errors.empty());
    bool found_tp_error = false;
    for (const auto &e : errors)
    {
        if (e.find("TP node must have") != std::string::npos)
        {
            found_tp_error = true;
        }
    }
    EXPECT_TRUE(found_tp_error) << "Expected TP child count error";
}

TEST(Test__ParallelismTree, InvalidRankOutOfRange)
{
    auto node = Device(GlobalDeviceAddress::cuda(0), 5); // rank 5 with world_size=2

    ParallelismTree tree;
    tree.root = std::move(node);
    tree.world_size = 2;
    tree.assignLayers(24);

    auto errors = tree.validate();
    EXPECT_FALSE(errors.empty());
    bool found_rank_error = false;
    for (const auto &e : errors)
    {
        if (e.find("owning_rank") != std::string::npos &&
            e.find("out of range") != std::string::npos)
        {
            found_rank_error = true;
        }
    }
    EXPECT_TRUE(found_rank_error) << "Expected rank out of range error";
}

TEST(Test__ParallelismTree, InvalidZeroLayers)
{
    auto tree = buildSimple2RankPPTP();
    tree.total_layers = 0;
    // Don't assign layers — validate should catch total_layers = 0

    auto errors = tree.validate();
    EXPECT_FALSE(errors.empty());
    bool found = false;
    for (const auto &e : errors)
    {
        if (e.find("total_layers must be > 0") != std::string::npos)
            found = true;
    }
    EXPECT_TRUE(found);
}

TEST(Test__ParallelismTree, InvalidZeroWorldSize)
{
    auto tree = buildSimple2RankPPTP();
    tree.world_size = 0;
    tree.assignLayers(24);

    auto errors = tree.validate();
    EXPECT_FALSE(errors.empty());
    bool found = false;
    for (const auto &e : errors)
    {
        if (e.find("world_size must be > 0") != std::string::npos)
            found = true;
    }
    EXPECT_TRUE(found);
}

TEST(Test__ParallelismTree, InvalidEmptyChildren)
{
    ParallelismNode pp_node;
    pp_node.type = ParallelismNodeType::PIPELINE_PARALLEL;
    pp_node.name = "empty_pp";
    // No children

    ParallelismTree tree;
    tree.root = std::move(pp_node);
    tree.world_size = 1;
    tree.total_layers = 24;

    auto errors = tree.validate();
    EXPECT_FALSE(errors.empty());
    bool found = false;
    for (const auto &e : errors)
    {
        if (e.find("no children") != std::string::npos)
            found = true;
    }
    EXPECT_TRUE(found);
}

// =============================================================================
// Test Suite: Query Methods
// =============================================================================

TEST(Test__ParallelismTree, LeafRanksSingleRank)
{
    auto tp = TP("tp0",
                 {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
                 0, CollectiveBackendType::NCCL);

    auto ranks = tp.leafRanks();
    EXPECT_EQ(ranks.size(), 1u);
    EXPECT_TRUE(ranks.count(0));
}

TEST(Test__ParallelismTree, LeafRanksMultiRank)
{
    auto tree = build4Machine2Socket2GPU();

    auto ranks = tree.root.leafRanks();
    EXPECT_EQ(ranks.size(), 4u);
    EXPECT_TRUE(ranks.count(0));
    EXPECT_TRUE(ranks.count(1));
    EXPECT_TRUE(ranks.count(2));
    EXPECT_TRUE(ranks.count(3));
}

TEST(Test__ParallelismTree, LeafRanksPerHost)
{
    auto tree = build4Machine2Socket2GPU();

    // host0 has ranks 0 and 1
    auto host0_ranks = tree.root.children[0].leafRanks();
    EXPECT_EQ(host0_ranks.size(), 2u);
    EXPECT_TRUE(host0_ranks.count(0));
    EXPECT_TRUE(host0_ranks.count(1));

    // host1 has ranks 2 and 3
    auto host1_ranks = tree.root.children[1].leafRanks();
    EXPECT_EQ(host1_ranks.size(), 2u);
    EXPECT_TRUE(host1_ranks.count(2));
    EXPECT_TRUE(host1_ranks.count(3));
}

TEST(Test__ParallelismTree, LeafDevicesCount)
{
    auto tree = build4Machine2Socket2GPU();

    auto leaves = tree.root.leafDevices();
    EXPECT_EQ(leaves.size(), 8u);

    // All leaves are DEVICE type
    for (const auto *leaf : leaves)
    {
        EXPECT_EQ(leaf->type, ParallelismNodeType::DEVICE);
    }
}

TEST(Test__ParallelismTree, IsCrossRankLocal)
{
    auto tp = TP("local_tp",
                 {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
                 0, CollectiveBackendType::NCCL);

    EXPECT_FALSE(tp.isCrossRank());
}

TEST(Test__ParallelismTree, IsCrossRankDistributed)
{
    auto tree = build4Machine2Socket2GPU();

    EXPECT_TRUE(tree.root.isCrossRank());
    EXPECT_TRUE(tree.root.children[0].isCrossRank()); // host0 has ranks 0 and 1

    // Individual TP domains are single-rank
    EXPECT_FALSE(tree.root.children[0].children[0].isCrossRank()); // socket0, rank 0
    EXPECT_FALSE(tree.root.children[0].children[1].isCrossRank()); // socket1, rank 1
}

TEST(Test__ParallelismTree, LeafCountNested)
{
    auto tree = build4Machine2Socket2GPU();

    EXPECT_EQ(tree.root.leafCount(), 8);
    EXPECT_EQ(tree.root.children[0].leafCount(), 4);             // host0
    EXPECT_EQ(tree.root.children[0].children[0].leafCount(), 2); // socket0
}

// =============================================================================
// Test Suite: toString
// =============================================================================

TEST(Test__ParallelismTree, ToStringContainsStructure)
{
    auto tree = buildSimple2RankPPTP();
    tree.assignLayers(24);

    auto str = tree.toString();

    // Should contain key structural elements
    EXPECT_NE(str.find("ParallelismTree"), std::string::npos);
    EXPECT_NE(str.find("layers=24"), std::string::npos);
    EXPECT_NE(str.find("world_size=2"), std::string::npos);
    EXPECT_NE(str.find("PP(cross_socket)"), std::string::npos);
    EXPECT_NE(str.find("TP(socket0)"), std::string::npos);
    EXPECT_NE(str.find("TP(socket1)"), std::string::npos);
    EXPECT_NE(str.find("nccl"), std::string::npos);
    EXPECT_NE(str.find("rccl"), std::string::npos);
}

TEST(Test__ParallelismTree, ToStringContainsLayerRanges)
{
    auto tree = buildSimple2RankPPTP();
    tree.assignLayers(24);

    auto str = tree.toString();

    EXPECT_NE(str.find("layers 0-23"), std::string::npos);
    EXPECT_NE(str.find("layers 0-11"), std::string::npos);
    EXPECT_NE(str.find("layers 12-23"), std::string::npos);
}

// =============================================================================
// Test Suite: Mixed CUDA+ROCm TP (HETEROGENEOUS backends)
// =============================================================================

TEST(Test__ParallelismTree, MixedTPPcieBarConstruction)
{
    auto tree = buildMixedTPPcieBar();

    EXPECT_EQ(tree.root.type, ParallelismNodeType::TENSOR_PARALLEL);
    EXPECT_EQ(tree.root.name, "mixed_tp");
    EXPECT_EQ(tree.root.backend, CollectiveBackendType::HETEROGENEOUS);
    EXPECT_EQ(tree.root.children.size(), 2u);
    EXPECT_EQ(tree.root.tpDegree(), 2);
    EXPECT_EQ(tree.root.leafCount(), 2);

    // Children should be one CUDA and one ROCm device
    EXPECT_EQ(tree.root.children[0].device.device_type, DeviceType::CUDA);
    EXPECT_EQ(tree.root.children[1].device.device_type, DeviceType::ROCm);
}

TEST(Test__ParallelismTree, MixedTPHeterogeneousConstruction)
{
    auto tree = buildHeterogeneousTP();

    EXPECT_EQ(tree.root.type, ParallelismNodeType::TENSOR_PARALLEL);
    EXPECT_EQ(tree.root.backend, CollectiveBackendType::HETEROGENEOUS);
    EXPECT_EQ(tree.root.children.size(), 4u);
    EXPECT_EQ(tree.root.tpDegree(), 4);
    EXPECT_EQ(tree.root.leafCount(), 4);

    // Should have 2 CUDA + 2 ROCm leaves
    auto types = tree.root.leafDeviceTypes();
    EXPECT_EQ(types.size(), 2u);
    EXPECT_TRUE(types.count(DeviceType::CUDA));
    EXPECT_TRUE(types.count(DeviceType::ROCm));
}

TEST(Test__ParallelismTree, MixedTPIsMixedVendor)
{
    auto mixed_tree = buildMixedTPPcieBar();
    EXPECT_TRUE(mixed_tree.root.isMixedVendor());

    auto hetero_tree = buildHeterogeneousTP();
    EXPECT_TRUE(hetero_tree.root.isMixedVendor());

    // Same-vendor should NOT be mixed
    auto same_vendor = TP("cuda_only",
                          {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
                          0, CollectiveBackendType::NCCL);
    EXPECT_FALSE(same_vendor.isMixedVendor());
}

TEST(Test__ParallelismTree, MixedTPLeafDeviceTypes)
{
    auto tree = buildMixedTPPcieBar();

    auto types = tree.root.leafDeviceTypes();
    EXPECT_EQ(types.size(), 2u);
    EXPECT_TRUE(types.count(DeviceType::CUDA));
    EXPECT_TRUE(types.count(DeviceType::ROCm));

    // Each child has only its own type
    auto child0_types = tree.root.children[0].leafDeviceTypes();
    EXPECT_EQ(child0_types.size(), 1u);
    EXPECT_TRUE(child0_types.count(DeviceType::CUDA));

    auto child1_types = tree.root.children[1].leafDeviceTypes();
    EXPECT_EQ(child1_types.size(), 1u);
    EXPECT_TRUE(child1_types.count(DeviceType::ROCm));
}

TEST(Test__ParallelismTree, MixedTPPcieBarValidation)
{
    auto tree = buildMixedTPPcieBar();
    tree.assignLayers(24);

    auto errors = tree.validate();
    EXPECT_TRUE(errors.empty()) << "Validation errors: " << errors[0];

    // Both devices get same layer range (TP)
    EXPECT_EQ(tree.root.children[0].first_layer, 0);
    EXPECT_EQ(tree.root.children[0].last_layer, 23);
    EXPECT_EQ(tree.root.children[1].first_layer, 0);
    EXPECT_EQ(tree.root.children[1].last_layer, 23);
}

TEST(Test__ParallelismTree, MixedTPHeterogeneousValidation)
{
    auto tree = buildHeterogeneousTP();
    tree.assignLayers(24);

    auto errors = tree.validate();
    EXPECT_TRUE(errors.empty()) << "Validation errors: " << errors[0];
}

TEST(Test__ParallelismTree, PPWithMixedTPValidation)
{
    auto tree = buildPPWithMixedTP();
    tree.assignLayers(24);

    auto errors = tree.validate();
    EXPECT_TRUE(errors.empty()) << "Validation errors: " << errors[0];

    // PP splits layers: socket0 gets 0-11, socket1 gets 12-23
    const auto &s0 = tree.root.children[0];
    const auto &s1 = tree.root.children[1];
    EXPECT_EQ(s0.first_layer, 0);
    EXPECT_EQ(s0.last_layer, 11);
    EXPECT_EQ(s1.first_layer, 12);
    EXPECT_EQ(s1.last_layer, 23);

    // Both sockets are mixed-vendor
    EXPECT_TRUE(s0.isMixedVendor());
    EXPECT_TRUE(s1.isMixedVendor());
    EXPECT_EQ(s0.backend, CollectiveBackendType::HETEROGENEOUS);
    EXPECT_EQ(s1.backend, CollectiveBackendType::HETEROGENEOUS);
}

TEST(Test__ParallelismTree, InvalidNCCLWithROCmDevice)
{
    // NCCL backend with mixed CUDA+ROCm devices should fail validation
    auto root = TP("bad_nccl_tp",
                   {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)},
                   0, CollectiveBackendType::NCCL);

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    tree.assignLayers(24);

    auto errors = tree.validate();
    EXPECT_FALSE(errors.empty());
    bool found_backend_error = false;
    for (const auto &e : errors)
    {
        if (e.find("NCCL backend") != std::string::npos &&
            e.find("ROCm") != std::string::npos)
        {
            found_backend_error = true;
        }
    }
    EXPECT_TRUE(found_backend_error) << "Expected NCCL+ROCm incompatibility error";
}

TEST(Test__ParallelismTree, InvalidRCCLWithCUDADevice)
{
    // RCCL backend with CUDA device should fail validation
    auto root = TP("bad_rccl_tp",
                   {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::cuda(0)},
                   0, CollectiveBackendType::RCCL);

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    tree.assignLayers(24);

    auto errors = tree.validate();
    EXPECT_FALSE(errors.empty());
    bool found_backend_error = false;
    for (const auto &e : errors)
    {
        if (e.find("RCCL backend") != std::string::npos &&
            e.find("CUDA") != std::string::npos)
        {
            found_backend_error = true;
        }
    }
    EXPECT_TRUE(found_backend_error) << "Expected RCCL+CUDA incompatibility error";
}

TEST(Test__ParallelismTree, MixedTPToStringContainsBackend)
{
    auto tree = buildMixedTPPcieBar();
    tree.assignLayers(24);

    auto str = tree.toString();

    EXPECT_NE(str.find("heterogeneous"), std::string::npos)
        << "toString should show heterogeneous backend";
    EXPECT_NE(str.find("mixed_tp"), std::string::npos);
    EXPECT_NE(str.find("cuda"), std::string::npos);
    EXPECT_NE(str.find("rocm"), std::string::npos);
}

TEST(Test__ParallelismTree, HeterogeneousTPToStringContainsBackend)
{
    auto tree = buildHeterogeneousTP();
    tree.assignLayers(24);

    auto str = tree.toString();

    EXPECT_NE(str.find("heterogeneous"), std::string::npos)
        << "toString should show heterogeneous backend";
    EXPECT_NE(str.find("hetero_tp"), std::string::npos);
}

TEST(Test__ParallelismTree, NestedPPWithMixedAndSameVendorTP)
{
    // Real-world scenario: one socket has same-vendor TP (NCCL),
    // another socket has mixed-vendor TP (HETEROGENEOUS)
    auto root = PP("cross_socket", {
                                       TP("socket0_nccl", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 0, CollectiveBackendType::NCCL),
                                       TP("socket1_pciebar", {GlobalDeviceAddress::cuda(2), GlobalDeviceAddress::rocm(0)}, 1, CollectiveBackendType::HETEROGENEOUS),
                                   });

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 2;
    tree.assignLayers(24);

    auto errors = tree.validate();
    EXPECT_TRUE(errors.empty()) << "Validation errors: " << errors[0];

    // Root has mixed vendors overall (CUDA + ROCm across children)
    EXPECT_TRUE(tree.root.isMixedVendor());

    // socket0 is same-vendor
    EXPECT_FALSE(tree.root.children[0].isMixedVendor());

    // socket1 is mixed-vendor
    EXPECT_TRUE(tree.root.children[1].isMixedVendor());
}

// =============================================================================
// Test Suite: Edge Cases
// =============================================================================

TEST(Test__ParallelismTree, SingleDeviceTree)
{
    auto node = Device(GlobalDeviceAddress::cpu(), 0);

    ParallelismTree tree;
    tree.root = std::move(node);
    tree.world_size = 1;
    tree.assignLayers(24);

    auto errors = tree.validate();
    EXPECT_TRUE(errors.empty()) << "Errors: " << errors[0];

    EXPECT_EQ(tree.root.leafCount(), 1);
    EXPECT_FALSE(tree.root.isCrossRank());
    EXPECT_EQ(tree.root.first_layer, 0);
    EXPECT_EQ(tree.root.last_layer, 23);
}

TEST(Test__ParallelismTree, TPOnlyTreeNopp)
{
    // Pure TP, no PP — 2 devices on same rank
    auto tp = TP("pure_tp",
                 {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
                 0, CollectiveBackendType::NCCL);

    ParallelismTree tree;
    tree.root = std::move(tp);
    tree.world_size = 1;
    tree.assignLayers(24);

    auto errors = tree.validate();
    EXPECT_TRUE(errors.empty()) << "Errors: " << errors[0];

    // Both devices get all layers (TP = replicated range)
    EXPECT_EQ(tree.root.children[0].first_layer, 0);
    EXPECT_EQ(tree.root.children[0].last_layer, 23);
    EXPECT_EQ(tree.root.children[1].first_layer, 0);
    EXPECT_EQ(tree.root.children[1].last_layer, 23);
}

TEST(Test__ParallelismTree, DeepNesting3Levels)
{
    // PP → PP → TP → DEVICE (3 levels of nesting)
    auto root = PP("global", {
                                 PP("host0", {
                                                 TP("s0", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 0),
                                             }),
                                 PP("host1", {
                                                 TP("s1", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 1),
                                             }),
                             });

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 2;
    tree.assignLayers(24);

    auto errors = tree.validate();
    EXPECT_TRUE(errors.empty()) << "Errors: " << errors[0];

    // Verify deep nesting works
    EXPECT_EQ(tree.root.leafCount(), 4);

    // host0 gets layers 0-11, host1 gets 12-23
    // host0 has 1 PP child (TP), which has 1 PP child = layers 0-11
    // Actually with only 1 PP child per host, the single child gets all host layers
    EXPECT_EQ(tree.root.children[0].children[0].first_layer, 0);
    EXPECT_EQ(tree.root.children[0].children[0].last_layer, 11);
    EXPECT_EQ(tree.root.children[1].children[0].first_layer, 12);
    EXPECT_EQ(tree.root.children[1].children[0].last_layer, 23);
}

TEST(Test__ParallelismTree, NodeEquality)
{
    auto a = Device(GlobalDeviceAddress::cuda(0), 0);
    auto b = Device(GlobalDeviceAddress::cuda(0), 0);
    auto c = Device(GlobalDeviceAddress::cuda(1), 0);

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(Test__ParallelismTree, ParallelismNodeTypeNames)
{
    EXPECT_STREQ(parallelismNodeTypeName(ParallelismNodeType::PIPELINE_PARALLEL), "PP");
    EXPECT_STREQ(parallelismNodeTypeName(ParallelismNodeType::TENSOR_PARALLEL), "TP");
    EXPECT_STREQ(parallelismNodeTypeName(ParallelismNodeType::DEVICE), "DEVICE");
}
