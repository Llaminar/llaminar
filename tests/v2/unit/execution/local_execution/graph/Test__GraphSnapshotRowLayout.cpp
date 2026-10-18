/**
 * @file Test__GraphSnapshotRowLayout.cpp
 * @brief Preserve typed diagnostic ownership through the real captured manifest.
 *
 * These tests use inert backend-qualified source views and an already acquired
 * CPU arena mirror. They enter production descriptor preparation, recording's
 * drift rejection and terminal publication without allocating or calling a GPU.
 * The old omission is reproduced by erasing only the retained row layout.
 */
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/debug/TPSnapshot.h"
#include "snapshots/SnapshotCapture.h"
#include "tensors/GpuTensorView.h"
#include "tensors/Tensors.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
    /** @brief Enter exact production boundaries without a native transfer. */
    struct GraphSnapshotRowLayoutTestAccess
    {
        /** @brief Freeze or reject descriptor drift before any backend copy.
         * @param executor Real executor under test.
         * @param node Stable diagnostic producer.
         * @param device Inert backend-qualified device identity.
         * @param record Whether to validate native-recording identity.
         * @param manifest Exact retained descriptor owner.
         * @return Production preparation/recording result. */
        static bool describe(DeviceGraphExecutor &executor, ComputeNode &node,
                             DeviceId device, bool record,
                             DeviceGraphExecutor::GraphSnapshotManifest &manifest)
        {
            return executor.prepareOrRecordGraphSnapshotCopies(
                node, device, reinterpret_cast<void *>(0x1234), record, manifest);
        }

        /** @brief Publish an already acquired mirror through production projection.
         * @param executor Real executor under test.
         * @param manifest Frozen slots backed by a CPU-only test mirror.
         * @param logical Terminal observed logical request geometry.
         * @return Production publication result; no native API is entered. */
        static bool publish(DeviceGraphExecutor &executor,
                            DeviceGraphExecutor::GraphSnapshotManifest &manifest,
                            const DeviceGraphExecutor::GraphSnapshotLogicalRows &logical)
        {
            return executor.publishGraphSnapshotCopies(
                "layer0_moe_routing_owned", reinterpret_cast<void *>(0x1234),
                manifest, &logical,
                DeviceGraphExecutor::GraphSnapshotPublicationMode::ArenaAlreadyAcquired);
        }
    };
}

using namespace llaminar2;
namespace
{
    /** @brief Metadata-only producer; execute must never run in these tests. */
    class DescriptorStage final : public IComputeStage
    {
    public:
        /** @brief Bind one inert source and its typed publication layout. */
        DescriptorStage(GpuTensorView &source, DeviceId device, SnapshotRowLayout layout)
            : IComputeStage(device), layout(std::move(layout)), source_(source) {}
        /** @return False: the fixture cannot execute model arithmetic. */
        bool execute(IDeviceContext *) override { return false; }
        /** @return A concrete stage classification for the diagnostic-only node. */
        ComputeStageType type() const override { return ComputeStageType::GEMM; }
        /** @return True for inert backend metadata; no native context is used. */
        bool supportsBackend(ComputeBackendType) const override { return true; }
        /** @return The producer descriptor used by actual capture preparation. */
        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo info;
            info.addOutput("router_logits", &source_, source_.rows(), source_.cols());
            info.outputs.back().row_layout = layout;
            return info;
        }
        SnapshotRowLayout layout;
    private:
        GpuTensorView &source_;
    };

    /** @brief Own synthetic source bytes, frozen slots and an acquired CPU mirror. */
    struct FrozenRouter
    {
        /** @brief Freeze the same descriptor path used by CUDA and ROCm graphs.
         * @param device Inert CUDA/HIP identity; never selects an actual device.
         * @param values Physical source values in producer storage order.
         * @param cols Features per physical source row.
         * @param layout Exact compact/replicated diagnostic ownership.
         * @throws std::runtime_error If production descriptor preparation fails. */
        FrozenRouter(DeviceId device, std::vector<float> values, size_t cols,
                     SnapshotRowLayout layout)
            : source(std::move(values)),
              view(source.data(), source.size()/cols, cols, TensorType::FP32, device)
        {
            executor.setSnapshotCallback([this](const std::string &name, const StageDumpInfo &dump) {
                capture.captureStage(name, dump);
            });
            auto owned = std::make_unique<DescriptorStage>(view, device, std::move(layout));
            stage = owned.get();
            graph.addNode("layer0_moe_routing_owned", std::move(owned), device);
            manifest.beginDescriptorCollection();
            if (!GraphSnapshotRowLayoutTestAccess::describe(executor, *node(), device, false, manifest))
                throw std::runtime_error("Cannot prepare synthetic captured descriptor");
            // This fixture supplies only the already acquired host mirror. It
            // never binds a native allocation or enters the transfer boundary.
            manifest.storage_arena = std::make_shared<FP32Tensor>(
                std::vector<size_t>{1, source.size()}, DeviceId::cpu());
            std::memcpy(manifest.storage_arena->mutable_data(), source.data(), source.size()*sizeof(float));
            manifest.storage_device = device;
            manifest.storage_capacity_bytes = manifest.storage_required_bytes = source.size()*sizeof(float);
            manifest.bound_slot_count = 1;
            manifest.storage_state = DeviceGraphExecutor::GraphSnapshotManifest::StorageState::StorageBound;
        }
        /** @return The sole stable producer whose descriptor was frozen. */
        ComputeNode *node() { return graph.getNode("layer0_moe_routing_owned"); }
        /** @return The immutable slot from which publication reconstructs metadata. */
        DeviceGraphExecutor::GraphSnapshotOutputCopy &slot()
        { return manifest.stage_copies.at("layer0_moe_routing_owned").outputs.at(0); }
        DeviceGraphExecutor executor;
        SnapshotCapture capture;
        std::vector<float> source;
        GpuTensorView view;
        DescriptorStage *stage = nullptr;
        ComputeGraph graph;
        DeviceGraphExecutor::GraphSnapshotManifest manifest;
    };
}

/** @test Compact and replicated storage retain every live row exactly once. */
TEST(GraphSnapshotRowLayout, CapturedPublicationPreservesOwnedRowsAcrossBackends)
{
    constexpr int cols = 3;
    for (DeviceId device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const int capacity : {65, 512})
    for (const int degree : {2, 3, 8})
    for (const int live : {1, 3, 65, capacity})
    for (auto storage : {SnapshotRowStorage::CompactOwned, SnapshotRowStorage::CompleteReplicated})
    {
        SCOPED_TRACE(::testing::Message() << "capacity=" << capacity << " degree=" << degree << " live=" << live);
        std::vector<float> oracle(static_cast<size_t>(live)*cols);
        for (size_t i = 0; i < oracle.size(); ++i) oracle[i] = float(i)+0.125f;
        oracle[0] = -0.0f;
        std::vector<std::vector<float>> values(degree);
        std::vector<SnapshotOwnedRows> ownership(degree);
        std::vector<SnapshotRowSource> sources;
        for (int participant = 0; participant < degree; ++participant)
        {
            const auto partition = DeviceRowPartition::balanced(participant, degree);
            const auto owned = partition.resolveFor(capacity, live);
            const size_t physical_rows = storage == SnapshotRowStorage::CompleteReplicated
                ? capacity : partition.capacityFor(capacity);
            std::vector<float> physical(physical_rows*cols, -123456.f);
            if (storage == SnapshotRowStorage::CompleteReplicated)
                std::copy(oracle.begin(), oracle.end(), physical.begin());
            else if (owned.count > 0)
                std::copy_n(oracle.data()+owned.first*cols, owned.count*cols, physical.data());
            FrozenRouter fixture(device, physical, cols, SnapshotCompactRows{partition,capacity,storage});
            EXPECT_EQ(fixture.slot().row_layout, fixture.stage->layout);
            // Publication must read the retained descriptor, never mutable stage
            // state queried after execution or before a different graph reuses it.
            fixture.stage->layout = std::monostate{};
            const DeviceGraphExecutor::GraphSnapshotLogicalRows logical{size_t(capacity),{size_t(live)}};
            ASSERT_TRUE(GraphSnapshotRowLayoutTestAccess::publish(fixture.executor, fixture.manifest, logical));
            auto snapshot = fixture.capture.getShared("layer0_MOE_ROUTER_OUTPUT");
            ASSERT_TRUE(snapshot);
            ASSERT_EQ(snapshot->publication, SnapshotPublication::RowPartition);
            ASSERT_EQ(snapshot->rows, size_t(live));
            EXPECT_EQ(snapshot->cols, size_t(cols));
            values[participant] = snapshot->data;
            ownership[participant] = snapshot->row_ownership;
            EXPECT_EQ(std::memcmp(fixture.source.data(), physical.data(), physical.size()*sizeof(float)),0);
        }
        for (int participant = degree-1; participant >= 0; --participant)
            sources.push_back({values[participant],size_t(live),cols,&ownership[participant]});
        const auto assembled = assembleSnapshotOwnedRows(sources);
        ASSERT_EQ(assembled.size(), oracle.size());
        EXPECT_EQ(std::memcmp(assembled.data(),oracle.data(),oracle.size()*sizeof(float)),0);
    }
}

/** @test The previous omitted layout publishes half a bucket as a replica. */
TEST(GraphSnapshotRowLayout, OldOmittedLayoutCannotPublishTheLogicalBucket)
{
    constexpr int capacity=512,cols=3;
    FrozenRouter fixture(DeviceId::cuda(0), std::vector<float>((capacity/2)*cols,0.25f),cols,
                         SnapshotCompactRows{DeviceRowPartition::balanced(0,2),capacity});
    const auto retained=fixture.slot().row_layout;
    const DeviceGraphExecutor::GraphSnapshotLogicalRows logical{capacity,{capacity}};
    fixture.slot().row_layout=std::monostate{};
    ASSERT_TRUE(GraphSnapshotRowLayoutTestAccess::publish(fixture.executor,fixture.manifest,logical));
    auto wrong=fixture.capture.getShared("layer0_MOE_ROUTER_OUTPUT");
    ASSERT_TRUE(wrong);
    EXPECT_EQ(wrong->publication,SnapshotPublication::SchemaPartition);
    EXPECT_EQ(wrong->rows,size_t(capacity/2));
    fixture.slot().row_layout=retained;
    ASSERT_TRUE(GraphSnapshotRowLayoutTestAccess::publish(fixture.executor,fixture.manifest,logical));
    auto correct=fixture.capture.getShared("layer0_MOE_ROUTER_OUTPUT");
    ASSERT_TRUE(correct);
    EXPECT_EQ(correct->publication,SnapshotPublication::RowPartition);
    EXPECT_EQ(correct->rows,size_t(capacity));
}

/** @test Recording rejects every ownership change before entering a GPU copy. */
TEST(GraphSnapshotRowLayout, RecordingRejectsOwnershipDescriptorDrift)
{
    const auto original=SnapshotCompactRows{DeviceRowPartition::balanced(0,2),512};
    const std::array<SnapshotRowLayout,6> changed{
        SnapshotCompactRows{DeviceRowPartition::balanced(1,2),512},
        SnapshotCompactRows{DeviceRowPartition::balanced(0,3),512},
        SnapshotCompactRows{DeviceRowPartition::balanced(0,2),511},
        SnapshotCompactRows{DeviceRowPartition::balanced(0,2),512,SnapshotRowStorage::CompleteReplicated},
        SnapshotOwnedRows{{{0,256}}},std::monostate{}};
    for (DeviceId device : {DeviceId::cuda(0),DeviceId::rocm(0)})
    for (const auto &layout : changed)
    {
        FrozenRouter fixture(device,std::vector<float>(256*3,1.f),3,original);
        fixture.stage->layout=layout;
        EXPECT_FALSE(GraphSnapshotRowLayoutTestAccess::describe(
            fixture.executor,*fixture.node(),device,true,fixture.manifest));
        EXPECT_EQ(fixture.slot().row_layout,SnapshotRowLayout(original));
    }
    EXPECT_NE((SnapshotRowLayout{SnapshotOwnedRows{{{0,2},{4,1}}}}),
              (SnapshotRowLayout{SnapshotOwnedRows{{{0,2},{5,1}}}}));
}
