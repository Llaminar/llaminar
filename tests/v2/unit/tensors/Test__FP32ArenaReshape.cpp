/**
 * @file Test__FP32ArenaReshape.cpp
 * @brief Device-free exact owner/event/shape admission and stage-lifetime regressions.
 *
 * GPU semantics use the existing host-memory MockBackend, never a native device.
 * Original host-only create_view is an explicit failing ownership control.
 * CPU arena roots use an explicit model-free fixture admission authority.
 */
#include "../../utils/CPUActivationArenaFixture.h"
#include "tensors/FP32ArenaReshape.h"
#include "execution/compute_stages/stages/GEMMArenaReshapeStage.h"
#include "memory/BufferArena.h"
#include "tensors/TensorFactory.h"
#include "transfer/TransferEngine.h"
#include "utils/MPIContext.h"
#include "mocks/MockBackend.h"
#include "mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
using namespace llaminar2;
namespace
{
/** @brief Existing arena and tensor factory lifetime for one mutable physical root. */
struct Arena
{
    MPIContext mpi{0, 1};
    TensorFactory factory{mpi};
    BufferArena arena{::llaminar2::test::cpuArenaFixtureConfig(&factory)};
    /** @brief Allocate ordinary host storage only, including logical GPU participants. */
    explicit Arena(DeviceId device = DeviceId::cpu(), const char *type = "FP32")
    {
        if (!arena.registerBuffer(BufferId::MTP_CONCAT, 2, 16, type, device) ||
            !arena.registerBuffer(BufferId::MTP_PROJECTED, 2, 8, "FP32", device) || !arena.allocate())
            throw std::runtime_error("Test arena allocation failed");
    }
};
/** @brief Host-memory backend with explicit runtime generation, without device discovery. */
struct Backend : test::MockBackend
{
    explicit Backend(DeviceType type) : MockBackend(type) {}
    std::uint64_t generation = 1;
    std::uint64_t deviceRuntimeGeneration(int) const override { return generation; }
};
void *stream() { return reinterpret_cast<void *>(0x12340); }
}
TEST(FP32ArenaReshape, SharesCompleteHostStorageAndLogicalRowDecode)
{
    Arena fixture;
    auto root = fixture.arena.getSharedTensor(BufferId::MTP_CONCAT);
    const auto stats = fixture.arena.stats();
    const auto view = FP32ArenaReshape::fromArena(fixture.arena, BufferId::MTP_CONCAT, {8, 4}, DeviceId::cpu());
    EXPECT_EQ(view.tensor().transferStorageOwner(), root.get());
    EXPECT_EQ(view.tensor().raw_data(), root->raw_data());
    EXPECT_EQ(view.tensor().shape(), (std::vector<size_t>{8, 4}));
    EXPECT_NE(dynamic_cast<FP32Tensor *>(&view.tensor()), nullptr);
    EXPECT_EQ(view.byteOffset(), 0);
    EXPECT_EQ(fixture.arena.stats().total_bytes, stats.total_bytes);
    EXPECT_EQ(fixture.arena.stats().total_buffers, stats.total_buffers);
    for (size_t i = 0; i < 32; ++i) view.tensor().mutable_data()[i] = static_cast<float>(i);
    EXPECT_EQ(root->data()[31], 31);
    std::array<float, 4> row{};
    view.tensor().to_fp32_row(3, row.data());
    EXPECT_EQ(row, (std::array<float, 4>{12, 13, 14, 15}));
    view.sealStorage();
    EXPECT_EQ(view.captureIdentity().root, root.get());
    EXPECT_EQ(view.captureIdentity().bytes, 128);
    EXPECT_EQ(view.captureIdentity(), view.captureIdentity());
    const auto second = FP32ArenaReshape::fromArena(fixture.arena, BufferId::MTP_CONCAT, {4, 8}, DeviceId::cpu());
    second.sealStorage();
    EXPECT_NE(view.captureIdentity(), second.captureIdentity());
}
TEST(FP32ArenaReshape, RejectsInvalidOrUnretainedAdmission)
{
    Arena fixture;
    EXPECT_THROW(FP32ArenaReshape::fromArena(fixture.arena, BufferId::MTP_CONCAT, {}, DeviceId::cpu()), std::invalid_argument);
    EXPECT_THROW(FP32ArenaReshape::fromArena(fixture.arena, BufferId::MTP_CONCAT, {2, 4, 4}, DeviceId::cpu()), std::invalid_argument);
    EXPECT_THROW(FP32ArenaReshape::fromArena(fixture.arena, BufferId::MTP_CONCAT, {0, 32}, DeviceId::cpu()), std::invalid_argument);
    EXPECT_THROW(FP32ArenaReshape::fromArena(fixture.arena, BufferId::MTP_CONCAT, {3, 4}, DeviceId::cpu()), std::invalid_argument);
    EXPECT_THROW(FP32ArenaReshape::fromArena(fixture.arena, BufferId::MTP_CONCAT, {std::numeric_limits<size_t>::max(), 2}, DeviceId::cpu()), std::overflow_error);
    EXPECT_THROW(FP32ArenaReshape::fromArena(fixture.arena, BufferId::MTP_CONCAT, {8, 4}, DeviceId::cuda(0)), std::invalid_argument);
    auto root = fixture.arena.getSharedTensor(BufferId::MTP_CONCAT);
    auto host_view = root->create_view({8, 4});
    ASSERT_TRUE(host_view);
    EXPECT_TRUE(fixture.arena.bindExternalBuffer(BufferId::Q_PROJ, host_view.get()));
    EXPECT_THROW(FP32ArenaReshape::fromArena(fixture.arena, BufferId::Q_PROJ, {8, 4}, DeviceId::cpu()), std::invalid_argument);
    EXPECT_TRUE(fixture.arena.bindExternalBuffer(BufferId::K_PROJ, root.get()));
    EXPECT_THROW(FP32ArenaReshape::fromArena(fixture.arena, BufferId::K_PROJ, {8, 4}, DeviceId::cpu()), std::invalid_argument);
    Arena integer(DeviceId::cpu(), "INT32");
    EXPECT_THROW(FP32ArenaReshape::fromArena(integer.arena, BufferId::MTP_CONCAT, {8, 4}, DeviceId::cpu()), std::invalid_argument);
}
TEST(FP32ArenaReshape, RejectsReleaseResizeAndPolicyMutationThroughAnyBase)
{
    Arena fixture;
    auto view = FP32ArenaReshape::fromArena(fixture.arena, BufferId::MTP_CONCAT, {8, 4}, DeviceId::cpu());
    TensorBase &base = view.tensor();
    EXPECT_THROW(base.release_raw_data(), std::logic_error);
    EXPECT_THROW(base.release_host_weight_data(), std::logic_error);
    EXPECT_THROW(base.releaseDeviceMemory(), std::logic_error);
    EXPECT_THROW(base.create_view({4, 4}, 1), std::logic_error);
    EXPECT_THROW(base.setGpuOnly(), std::logic_error);
    EXPECT_THROW(base.setHostResident(), std::logic_error);
    EXPECT_FALSE(fixture.arena.getSharedTensor(BufferId::MTP_CONCAT)->is_raw_data_released());
}
TEST(FP32ArenaReshape, AdmissionPrecedesGPUPreparationAndSealTracksOneOwnerAndEvent)
{
    llaminar2::testing::installHardwareFreeGPUContextFactories();
    for (DeviceId device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        Backend backend(device.type);
        Arena fixture(device);
        auto root = fixture.arena.getSharedTensor(BufferId::MTP_CONCAT);
        root->setBackendForTesting(&backend);
        auto view = FP32ArenaReshape::fromArena(fixture.arena, BufferId::MTP_CONCAT, {8, 4}, device);
        const auto declaration = view.declarationIdentity();
        EXPECT_EQ(declaration.root, root.get());
        EXPECT_EQ(declaration.bytes, 128);
        EXPECT_FALSE(view.isSealed());
        EXPECT_EQ(view.tensor().gpu_data_ptr(), nullptr);
        EXPECT_EQ(backend.getAllocationCount(), 0);
        EXPECT_THROW(view.sealStorage(), std::logic_error);
        EXPECT_THROW(view.captureIdentity(), std::logic_error);
        TransferEngine::prepareDeviceInput(&view.tensor(), device, stream());
        EXPECT_EQ(backend.getAllocationCount(), 1);
        EXPECT_EQ(backend.getEventCreateCount(), 1);
        EXPECT_EQ(backend.getTransferStats().h2d_bytes, 128);
        EXPECT_EQ(view.tensor().gpu_data_ptr(), root->gpu_data_ptr());
        view.sealStorage();
        const auto identity = view.captureIdentity();
        EXPECT_EQ(view.declarationIdentity(), declaration);
        EXPECT_EQ(identity.runtime_generation, 1);
        EXPECT_EQ(identity.address, root->gpu_data_ptr());
        TransferEngine::prepareDeviceOutput(&view.tensor(), device, stream());
        EXPECT_EQ(backend.getAllocationCount(), 1);
        EXPECT_EQ(view.captureIdentity(), identity);
        TransferEngine::publishDeviceWrite(&view.tensor(), device, stream());
        EXPECT_EQ(view.tensor().getAuthoritativeDevice(), root->getAuthoritativeDevice());
        EXPECT_EQ(view.tensor().coherenceState(), root->coherenceState());
        const auto producer_events = backend.getEventRecords();
        const auto producer = std::find_if(producer_events.rbegin(), producer_events.rend(), [](const auto &event) {
            return event.type == Backend::EventRecord::RECORD;
        });
        ASSERT_NE(producer, producer_events.rend());
        void *consumer_stream = reinterpret_cast<void *>(0x12360);
        TransferEngine::requireDeviceInput(&view.tensor(), device, consumer_stream);
        const auto consumer_events = backend.getEventRecordsForStream(consumer_stream);
        ASSERT_FALSE(consumer_events.empty());
        EXPECT_EQ(consumer_events.back().type, Backend::EventRecord::WAIT);
        EXPECT_EQ(consumer_events.back().event, producer->event);
        EXPECT_EQ(backend.getAllocationCount(), 1);
        backend.generation = 2;
        EXPECT_THROW(view.validateStorage(), std::logic_error);
        EXPECT_THROW(view.declarationIdentity(), std::logic_error);
        EXPECT_THROW(view.tensor().gpu_data_ptr(), std::logic_error);
        backend.generation = 1;
        EXPECT_TRUE(root->releaseDeviceMemory());
        EXPECT_THROW(view.validateStorage(), std::logic_error);
    }
}
TEST(FP32ArenaReshape, OriginalHostViewHasIndependentGPUBackingNegativeControl)
{
    llaminar2::testing::installHardwareFreeGPUContextFactories();
    Backend backend(DeviceType::CUDA);
    auto root = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 16}, DeviceId::cuda(0));
    root->setBackendForTesting(&backend);
    auto old_view = root->create_view({8, 4});
    ASSERT_TRUE(old_view);
    old_view->setBackendForTesting(&backend);
    EXPECT_NE(old_view->transferStorageOwner(), root.get());
    TransferEngine::prepareDeviceInput(root.get(), DeviceId::cuda(0), stream());
    TransferEngine::prepareDeviceInput(old_view.get(), DeviceId::cuda(0), stream());
    EXPECT_EQ(backend.getAllocationCount(), 2);
    EXPECT_NE(old_view->gpu_data_ptr(), root->gpu_data_ptr());
}
/** @test Legacy shape-handle placement retains the public exact-stream contract. */
TEST(FP32ArenaReshape, DirectPlacementUsesCanonicalTransferAuthority)
{
    llaminar2::testing::installHardwareFreeGPUContextFactories();
    for (DeviceId device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        SCOPED_TRACE(device.toString());
        Backend backend(device.type);
        Arena fixture(device);
        auto root = fixture.arena.getSharedTensor(BufferId::MTP_CONCAT);
        root->setBackendForTesting(&backend);
        auto view = FP32ArenaReshape::fromArena(fixture.arena, BufferId::MTP_CONCAT, {8, 4}, device);
        EXPECT_THROW(view.tensor().ensureOnDevice(device, nullptr), std::invalid_argument);
        EXPECT_THROW(view.tensor().ensureOnDevice(DeviceId::cpu(), stream()), std::logic_error);
        EXPECT_EQ(backend.getAllocationCount(), 0);
        EXPECT_EQ(backend.getTransferStats().h2d_bytes, 0);
        ASSERT_TRUE(view.tensor().ensureOnDevice(device, stream()));
        EXPECT_EQ(backend.getAllocationCount(), 1);
        EXPECT_EQ(backend.getTransferStats().h2d_bytes, 128);
        EXPECT_EQ(view.tensor().transferStorageOwner(), root.get());
        EXPECT_EQ(view.tensor().gpu_data_ptr(), root->gpu_data_ptr());
        view.sealStorage();
        const auto identity = view.captureIdentity();
        ASSERT_TRUE(view.tensor().ensureOnDevice(device, stream()));
        EXPECT_EQ(view.captureIdentity(), identity);
        EXPECT_EQ(backend.getAllocationCount(), 1);
        EXPECT_EQ(backend.getTransferStats().h2d_bytes, 128);
    }
    Arena host;
    auto view = FP32ArenaReshape::fromArena(host.arena, BufferId::MTP_CONCAT, {8, 4}, DeviceId::cpu());
    EXPECT_TRUE(view.tensor().ensureOnDevice(DeviceId::cpu()));
    EXPECT_TRUE(view.tensor().isHostValid());
    EXPECT_EQ(view.tensor().raw_data(), host.arena.getTensor(BufferId::MTP_CONCAT)->raw_data());
}
TEST(FP32ArenaReshape, MovedHandleRejectsUseAndCanonicalHostReleaseInvalidatesSeal)
{
    Arena fixture;
    auto original = FP32ArenaReshape::fromArena(fixture.arena, BufferId::MTP_CONCAT, {8, 4}, DeviceId::cpu());
    auto moved = std::move(original);
    EXPECT_FALSE(original.isSealed());
    EXPECT_THROW(original.tensor(), std::logic_error);
    moved.sealStorage();
    fixture.arena.getSharedTensor(BufferId::MTP_CONCAT)->release_raw_data();
    EXPECT_THROW(moved.validateStorage(), std::logic_error);
}
TEST(FP32ArenaReshape, StageOwnsOperandsAndPreservesPhysicalBufferContracts)
{
    std::unique_ptr<GEMMArenaReshapeStage> stage;
    std::weak_ptr<TensorBase> weak;
    {
        Arena fixture;
        weak = fixture.arena.getSharedTensor(BufferId::MTP_CONCAT);
        auto input = FP32ArenaReshape::fromArena(fixture.arena, BufferId::MTP_CONCAT, {8, 4}, DeviceId::cpu());
        auto output = FP32ArenaReshape::fromArena(fixture.arena, BufferId::MTP_PROJECTED, {8, 2}, DeviceId::cpu());
        GEMMStage::Params params{.device_id = DeviceId::cpu(), .m = 8, .n = 2, .k = 4};
        EXPECT_THROW(GEMMArenaReshapeStage((GEMMStage::Params{.device_id = DeviceId::cpu(), .m = 8, .n = 2, .k = 4, .a_buffer_id = BufferId::Q_PROJ}), input, output), std::invalid_argument);
        EXPECT_THROW(GEMMArenaReshapeStage((GEMMStage::Params{.device_id = DeviceId::cpu(), .m = 8, .n = 4, .k = 4}), input, input), std::invalid_argument);
        EXPECT_THROW(GEMMArenaReshapeStage((GEMMStage::Params{.device_id = DeviceId::cpu(), .m = 9, .n = 2, .k = 4}), input, output), std::invalid_argument);
        EXPECT_THROW(GEMMArenaReshapeStage((GEMMStage::Params{.device_id = DeviceId::cuda(0), .m = 8, .n = 2, .k = 4}), input, output), std::invalid_argument);
        stage = std::make_unique<GEMMArenaReshapeStage>(params, input, output);
        EXPECT_EQ(stage->graphLaunchPreparationPolicy(), GraphLaunchPreparationPolicy::CaptureOnly);
        EXPECT_TRUE(stage->prepareGraphLaunch(nullptr, nullptr));
        EXPECT_EQ(stage->inputCaptureIdentity().buffer, BufferId::MTP_CONCAT);
        EXPECT_EQ(stage->outputCaptureIdentity().buffer, BufferId::MTP_PROJECTED);
    }
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(stage->inputCaptureIdentity().shape[0], 8);
    stage.reset();
    EXPECT_TRUE(weak.expired());
}
