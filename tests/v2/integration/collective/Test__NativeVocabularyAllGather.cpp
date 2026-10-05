/**
 * @file Test__NativeVocabularyAllGather.cpp
 * @brief Captured CUDA/ROCm regression for complete sharded terminal logits.
 *
 * Reversed physical devices exercise communicator ownership. One retained
 * graph replays large, partial, empty and growing prefixes, preserving every
 * FP32 bit and inactive output byte. Native useful-byte receipts prove wire
 * traffic follows live rows rather than the admitted maximum capacity.
 * The serial-family variants publish transpose storage from a one-row graph
 * and use that same address for the retained 16-row graph without growth.
 * Fixed serial publication additionally ignores stale prompt/verifier counts;
 * only the later grouped graph borrows that device count authority.
 */
#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/LocalTPContext.h"
#include "collective/VocabularyGatherWorkspaceContract.h"
#include "execution/compute_stages/stages/NativeVocabularyAllGatherStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/device/WorkspaceAllocator.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include <algorithm>
#include <array>
#include <barrier>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Raise a fixture error while all native owners can still abort and retire. */
    void require(bool value, const char *message)
    { if (!value) throw std::runtime_error(message); }

    /** @return Distinct raw representations, including signed zero and NaN payload bytes. */
    std::uint8_t byte(int rank, int replay, std::size_t coordinate)
    {
        constexpr std::array<std::uint8_t, 16> special{0, 0, 0, 128, 1, 0, 192, 127,
            0, 0, 128, 255, 255, 255, 255, 255};
        return coordinate < special.size() ? special[(coordinate + rank + replay) % special.size()]
            : static_cast<std::uint8_t>(coordinate * 37 + rank * 101 + replay * 17);
    }

    /** @brief Explicit graph-family ownership exercised by this fixture. */
    enum class PublicationFixture { Counted, SerialFamily, FixedSerial };

    /** @brief Authenticate real native graph output and byte economy without a model.
     * @param first Backend whose two physical devices execute the graph.
     * @param fixture Counted publication or a retained serial/grouped family. */
    void verify(DeviceId first, PublicationFixture fixture = PublicationFixture::Counted)
    {
        const bool serial_family = fixture != PublicationFixture::Counted;
        auto *backend = getBackendFor(first);
        ASSERT_NE(backend, nullptr);
        ASSERT_GE(backend->deviceCount(), 2);
        const std::array devices{first.is_cuda() ? DeviceId::cuda(1) : DeviceId::rocm(1), first};
        auto tp = createLocalTPContext({GlobalDeviceAddress::fromLocalDeviceId(devices[0]),
            GlobalDeviceAddress::fromLocalDeviceId(devices[1])}, {},
            first.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        ASSERT_NE(tp, nullptr);
        std::barrier rendezvous(2);
        std::array<std::exception_ptr, 2> errors{};
        std::array<unsigned long long, 2> receipts{};
        std::array<std::thread, 2> workers;
        for (int participant = 0; participant < 2; ++participant)
            workers[participant] = std::thread([&, participant] {
                try
                {
                    auto &gpu = GPUDeviceContextPool::instance().getContext(devices[participant]);
                    gpu.submitAndWait([&] {
                        const auto device = devices[participant];
                        const int ordinal = device.gpu_ordinal();
                        void *stream = gpu.getOrCreateAuxiliaryStream("native_vocabulary_regression");
                        auto context = IDeviceContext::create(device, 1);
                        constexpr int columns = 257; // Odd width also catches truncation/alignment assumptions.
                        WorkspaceAllocator family;
                        DeviceWorkspaceManager *published_workspace = nullptr;
                        void *published_bank = nullptr;
                        std::uint64_t published_generation = 0;
                        if (serial_family)
                        {
                            FP32Tensor serial_local({1, columns}, device);
                            FP32Tensor serial_full({1, 2 * columns}, device);
                            NativeVocabularyAllGatherStage::Params serial;
                            serial.device_id = device;
                            serial.tp_ctx = tp.get();
                            serial.local_logits = &serial_local;
                            serial.full_logits = &serial_full;
                            serial.rows = DeviceRowRange::fullyActive(1);
                            serial.local_vocabulary = columns;
                            serial.vocabulary = 2 * columns;
                            serial.participant = participant;
                            serial.stage_name = "lm_head_allgather";
                            ComputeGraph serial_graph;
                            serial_graph.addNode(serial.stage_name,
                                std::make_unique<NativeVocabularyAllGatherStage>(serial), device);
                            WorkspaceSizingHints hints;
                            hints.max_seq_len = 1;
                            hints.serial_family_max_rows = 512;
                            hints.serial_family_max_compact_rows = 16;
                            hints.graph_family_policy = WorkspaceGraphFamilyPolicy::SerialDeviceFamilyLargestParticipant;
                            WorkspaceBudgetConfig budget;
                            budget.min_budget = budget.max_budget = 1024 * 1024;
                            require(family.allocateForGraph(serial_graph, hints, {}, budget), "publish serial family workspace");
                            published_workspace = family.getDeviceWorkspace(device);
                            require(published_workspace && published_workspace->getBufferSize(
                                VocabularyGatherWorkspaceContract::rankMajorBank) == 16u * 2 * columns * sizeof(float),
                                "serial setup omitted the retained verifier bank");
                            published_bank = published_workspace->getBuffer(VocabularyGatherWorkspaceContract::rankMajorBank);
                            require(published_bank != nullptr, "publish resident family bank");
                            published_generation = family.deviceGeneration(device);
                        }
                        for (const int capacity : {1, 16})
                        {
                            const bool fixed_serial = fixture == PublicationFixture::FixedSerial && capacity == 1;
                            FP32Tensor local({static_cast<std::size_t>(capacity + 3), columns}, device);
                            FP32Tensor full({static_cast<std::size_t>(capacity + 3), 2 * columns}, device);
                            FP32Tensor metadata({4}, device);
                            for (auto *tensor : {&local, &full, &metadata})
                            {
                                std::fill_n(tensor->mutable_data(), tensor->numel(), 0.0f);
                                require(tensor->ensureOnDevice(device, stream), "fixture upload");
                            }
                            auto *count = static_cast<int32_t *>(metadata.gpu_data_ptr());
                            auto *counter = reinterpret_cast<unsigned long long *>(count + 2);
                            NativeVocabularyAllGatherStage::Params declaration;
                            declaration.device_id = device;
                            declaration.tp_ctx = tp.get();
                            declaration.local_logits = &local;
                            declaration.full_logits = &full;
                            declaration.rows = fixed_serial ? DeviceRowRange::fullyActive(1)
                                : DeviceRowRange::deviceCounted(capacity, count);
                            declaration.local_vocabulary = columns;
                            declaration.vocabulary = 2 * columns;
                            declaration.participant = participant;
                            declaration.stage_name = "lm_head_allgather";
                            declaration.payload_bytes = fixed_serial ? nullptr : counter;
                            NativeVocabularyAllGatherStage stage(declaration);
                            DeviceWorkspaceManager workspace(device, 1024 * 1024);
                            if (!serial_family)
                                require(workspace.allocate(stage.getWorkspaceRequirements(capacity)), "admit transpose bank");
                            stage.bindWorkspace(serial_family ? published_workspace : &workspace);
                            require(stage.prepareGraphLaunch(context.get(), stream), "bind exact stage stream");
                            require(stage.isGraphCapturable(), "native terminal capture contract");
                            require(gpu.synchronizeStreamChecked(stream), "fixture upload completion");
                            TransferEngine::requireDeviceInput(&local, device, stream);
                            GraphCaptureDependencyLedger::StagePlan plan;
                            plan.stage_identity = &stage;
                            plan.stage_name = declaration.stage_name;
                            plan.external_inputs = {&local};
                            plan.outputs = {&full};
                            GraphCaptureDependencyLedger ledger(device, stream, {plan}, "native_vocabulary");
                            auto capture = gpu.createGraphCapture(stream);
                            rendezvous.arrive_and_wait();
                            ScopedBackendGraphCapture recording(gpu, *capture, "native_vocabulary", &ledger);
                            require(recording.begin(), "begin terminal collective graph");
                            {
                                ScopedGraphCaptureStage scope(&stage);
                                require(stage.execute(context.get()), "record terminal collective");
                                scope.complete();
                            }
                            recording.finish();
                            require(capture->instantiate(), "instantiate retained terminal graph");
                            constexpr std::array lengths{16, 7, 1, 0, 3, 16, 2, 0, 15, 1,
                                4, 5, 6, 8, 9, 10, 11, 12, 13, 14};
                            for (int replay = 0; replay < 20; ++replay)
                            {
                                const int live = fixed_serial ? 1 : std::min(capacity, lengths[replay % lengths.size()]);
                                std::vector<std::uint8_t> sent(local.size_bytes(), 0x5a);
                                std::vector<std::uint8_t> received(full.size_bytes(), 0xa5);
                                for (std::size_t i = 0; i < std::size_t(live) * columns * sizeof(float); ++i)
                                    sent[i] = byte(participant, replay, i);
                                // A serial head always publishes its one committed
                                // row, even when the separate source counter names
                                // no rows, a verifier window, or a longer prompt.
                                const int published_count = fixed_serial
                                    ? (replay % 3 == 0 ? 37 : lengths[replay % lengths.size()]) : live;
                                const std::array<std::uint32_t, 4> words{static_cast<std::uint32_t>(published_count), 0, 0, 0};
                                require(backend->hostToDeviceOnStream(local.gpu_data_ptr(), sent.data(), sent.size(), ordinal, stream), "reset local logits");
                                require(backend->hostToDeviceOnStream(full.gpu_data_ptr(), received.data(), received.size(), ordinal, stream), "poison output capacity");
                                require(backend->hostToDeviceOnStream(metadata.gpu_data_ptr(), words.data(), sizeof(words), ordinal, stream), "publish live count");
                                rendezvous.arrive_and_wait();
                                require(capture->launch(), "replay unchanged terminal graph");
                                require(backend->deviceToHostOnStream(received.data(), full.gpu_data_ptr(), received.size(), ordinal, stream), "observe logits");
                                require(backend->deviceToHostOnStream(&receipts[participant], counter, sizeof(receipts[participant]), ordinal, stream), "observe wire receipt");
                                require(gpu.synchronizeStreamChecked(stream), "terminal fixture observation");
                                std::vector<std::uint8_t> expected(full.size_bytes(), 0xa5);
                                for (int row = 0; row < live; ++row)
                                    for (int rank = 0; rank < 2; ++rank)
                                        for (std::size_t col = 0; col < std::size_t(columns) * sizeof(float); ++col)
                                            expected[(std::size_t(row) * 2 + rank) * columns * sizeof(float) + col] =
                                                byte(rank, replay, std::size_t(row) * columns * sizeof(float) + col);
                                require(received == expected, "terminal vocabulary bits, row order or inactive tail changed");
                                rendezvous.arrive_and_wait();
                                require(receipts[0] + receipts[1] == (fixed_serial ? 0 : 2ULL * live * columns * sizeof(float)),
                                    "terminal collective communicated inactive capacity");
                                rendezvous.arrive_and_wait();
                            }
                            rendezvous.arrive_and_wait();
                            capture->reset();
                            if (serial_family)
                            {
                                require(family.deviceGeneration(device) == published_generation,
                                    "retained graph changed published workspace generation");
                                require(published_workspace->getBuffer(VocabularyGatherWorkspaceContract::rankMajorBank) == published_bank,
                                    "retained graph changed published transpose address");
                            }
                        }
                    });
                }
                catch (...)
                {
                    errors[participant] = std::current_exception();
                    tp->requestAbort();
                    rendezvous.arrive_and_drop();
                }
            });
        for (auto &worker : workers) worker.join();
        for (const auto &error : errors)
            if (error)
            {
                try { std::rethrow_exception(error); }
                catch (const std::exception &failure) { ADD_FAILURE() << failure.what(); }
            }
    }
}
#ifdef HAVE_CUDA
TEST(NativeVocabularyAllGather, CUDA) { verify(DeviceId::cuda(0)); }
TEST(NativeVocabularyAllGather, SerialFamilyCUDA) { verify(DeviceId::cuda(0), PublicationFixture::SerialFamily); }
TEST(NativeVocabularyAllGather, FixedSerialCUDA) { verify(DeviceId::cuda(0), PublicationFixture::FixedSerial); }
#endif
#ifdef HAVE_ROCM
TEST(NativeVocabularyAllGather, ROCm) { verify(DeviceId::rocm(0)); }
TEST(NativeVocabularyAllGather, SerialFamilyROCm) { verify(DeviceId::rocm(0), PublicationFixture::SerialFamily); }
TEST(NativeVocabularyAllGather, FixedSerialROCm) { verify(DeviceId::rocm(0), PublicationFixture::FixedSerial); }
#endif
