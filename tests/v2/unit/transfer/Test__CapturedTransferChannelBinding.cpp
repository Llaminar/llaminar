/**
 * @file Test__CapturedTransferChannelBinding.cpp
 * @brief Device-free proofs of captured-channel admission, ownership and submission.
 *
 * Independent backend spies make physical aliases and exact native submission
 * order observable without touching accelerators. These tests complement the
 * protocol state-machine sweep; they do not certify GPU memory visibility.
 * CPU arena roots use an explicit model-free fixture admission authority.
 */
#include "../../utils/CPUActivationArenaFixture.h"
#include "transfer/CapturedTransferChannel.h"
#include "transfer/CapturedTransferKernelPlan.h"
#include "transfer/CapturedTransferChannelBatch.h"
#include "execution/moe/MoECountedTransferPolicy.h"
#include "execution/local_execution/orchestrators/PipelineActivationExchange.h"
#include "execution/local_execution/orchestrators/PipelineMetadataExchange.h"
#include "execution/local_execution/orchestrators/PipelineForwardGraphEdges.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "memory/BufferArena.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
#include "utils/PerfStatsCollector.h"
#include "../../mocks/MockBackend.h"
#include "../../mocks/MockWorkerGPUContext.h"
#include "../../mocks/MockComputeStage.h"
#include "../../mocks/MockLocalTPContext.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <optional>
#include <limits>
#include <set>
#include <string>
#include <vector>
#include <functional>

using namespace llaminar2;
using llaminar2::test::MockBackend;

namespace
{
    /** @brief One ordered native submission with its physical extent and stream. */
    struct Call { std::string operation; void *stream; size_t bytes; };

    /** @brief Host-only backend observing native order, identity and fault unwind. */
    class ChannelBackend final : public MockBackend
    {
    public:
        /** @brief Construct a backend family without probing real devices. */
        explicit ChannelBackend(DeviceType type) : MockBackend(type) { setDeviceCount(8); }
        /** @brief Register the shared host mapping exactly once for this family. */
        bool registerExternalMappedHostMemory(void *ptr, size_t, int, MappedHostRegistrationScope) override
        { EXPECT_TRUE(registered.insert(ptr).second); ++registrations; return true; }
        /** @brief Return a usable mock address while still checking allocation identity. */
        bool externalMappedHostDevicePointer(void *ptr, int, void **alias) override
        { if (!registered.contains(ptr) || !alias) return false; *alias = ptr; return true; }
        /** @brief Count release before the host mapping is destroyed. */
        bool unregisterExternalMappedHostMemory(void *ptr, int) override
        {
            if(forbid_early_retirement && initialization_attempted && !initialization_complete)std::_Exit(76);
            if(initialization_attempted)EXPECT_TRUE(initialization_complete);
            EXPECT_EQ(registered.erase(ptr),1u);++unregistrations;return true;
        }
        /** @brief Freeze timer geometry during setup, with explicit failure injection. */
        bool prepareCapturedTransferChannelKernels(int, int timeout_ms, std::uint64_t *ticks) override
        { ++preparations; EXPECT_EQ(timeout_ms, 30000); *ticks = 42; return !fail_prepare; }
        /** @brief Observe one complete operation; logical phases are not native kernel counts. */
        bool enqueueCapturedTransferChannel(const CapturedTransferChannelDeviceBinding &binding,
            void *dst, const void *src, int, void *stream) override
        {
            EXPECT_TRUE(binding.valid());
            ++complete_operations;
            native_bindings.push_back(binding);
            calls.push_back({"acquire", stream, static_cast<size_t>(binding.message.bytes)});
            if (fail_boundary) return false;
            const bool counted = binding.extent_source != CapturedTransferExtentSource::FixedMessage;
            const bool producer = binding.role == CapturedTransferEndpoint::Producer;
            auto &control = *binding.control;
            if (counted)
            {
                const auto slice = binding.source_slice ? *binding.source_slice : CapturedTransferSourceSlice{};
                const auto decision = CapturedTransferChannelProtocol::acquire(binding.role, binding.expected,
                    control.identity, *binding.cursor, producer ? control.consumer : control.producer,
                    binding.message, binding.extent_source, producer ? *binding.device_extent : 0,
                    slice.byte_offset, binding.source_span_bytes, binding.source_offset_origin);
                EXPECT_TRUE(decision.ready());
                if (!decision.ready()) return false;
                if (!producer) *binding.device_extent = binding.cursor->acquired_message.bytes;
            }
            const auto bytes = counted ? binding.cursor->acquired_message.bytes : binding.message.bytes;
            calls.push_back({counted ? "counted_copy" : "copy", stream, static_cast<size_t>(bytes)});
            std::memcpy(dst, static_cast<const std::byte *>(src) + binding.cursor->acquired_source_offset, bytes);
            calls.push_back({"publish", stream, static_cast<size_t>(binding.message.bytes)});
            if (counted)
                EXPECT_TRUE(CapturedTransferChannelProtocol::publish(*binding.cursor,
                    binding.cursor->completed_epoch + 1, producer ? control.producer : control.consumer).ready());
            return true;
        }
        /** @brief Observe packet boundaries while reusing the original pure per-message protocol spy. */
        bool enqueueCapturedTransferChannelBatch(const CapturedTransferChannelBatch &packet,
            int device, void *stream) override
        {
            EXPECT_TRUE(packet.valid());
            batch_sizes.push_back(packet.count);
            if (fail_boundary) return false;
            for (unsigned i = 0; i < packet.count; ++i)
            {
                const auto &entry = packet.entries[i];
                if (!enqueueCapturedTransferChannel(entry.binding, entry.destination, entry.source, device, stream)) return false;
            }
            return true;
        }
        /** @brief Fail a queued cold cursor initialization without manufacturing an event. */
        bool memset(void* ptr,int value,size_t bytes,int device,void* stream)override{
            if(track_initialization_lifetime){initialization_attempted=true;initialization_pointer=ptr;initialization_complete=false;}
            return fail_initialization?false:MockBackend::memset(ptr,value,bytes,device,stream);
        }
        /** @return Host-owned original runtime generation, without a device query. */
        std::uint64_t deviceRuntimeGeneration(int)const override{return runtime_generation;}
        std::uint64_t runtime_generation=1;
        unsigned event_creators=0;
        std::vector<void*> initialization_streams;
        std::vector<GPUStreamExecutionState> cold_states;
        std::size_t cold_queries=0;
        std::function<void(void*,GPUStreamExecutionState)> observe_cold_query;
        bool track_initialization_lifetime=false,initialization_attempted=false,initialization_complete=false;
        bool throw_cold_query=false,forbid_early_retirement=false;
        void* initialization_pointer=nullptr;
        std::vector<unsigned> batch_sizes;

        /** @brief Copy exact bytes, retaining sentinels outside the requested message. */
        bool copyDeviceVisibleRegionByKernelOnStream(void *dst, const void *src, size_t bytes, int, void *stream) override
        { calls.push_back({"copy", stream, bytes}); std::memcpy(dst, src, bytes); return true; }
        /** @brief Import an event into the exact stream, never wait on the host. */
        bool streamWaitEvent(void *stream, void *event, int device) override
        { calls.push_back({"event", stream, 0}); return MockBackend::streamWaitEvent(stream, event, device); }
        /** @brief Exercise rollback after some of the physical owners have materialized. */
        void *createEvent(int device) override
        { ++event_creators;return MockBackend::createEvent(device); }
        /** @brief Complete setup only; ordinary enqueue must never query this event again. */
        bool queryEvent(void *, int, bool *ready) override
        { ++setup_queries; *ready = true; return true; }

        /** @brief Borrow the one original native record selected by a focused tensor-extent replay. */
        NativeAllocationCaptureLease retainNativeAllocation(const void *pointer, size_t bytes, int ordinal) override
        {
            ++capture_acquisitions;
            if (capture_sealed) throw std::logic_error("Replay reacquired native allocation authority");
            if (pointer == second_capture_base)
                return NativeAllocationCaptureLease::acquire(second_capture_identity, second_capture_base, second_capture_bytes,
                    capture_device, pointer, bytes, DeviceId(backendDeviceType(), ordinal));
            return NativeAllocationCaptureLease::acquire(capture_identity, capture_base, capture_bytes,
                capture_device, pointer, bytes, DeviceId(backendDeviceType(), ordinal));
        }
        /** @brief Model the allocator's fatal pre-free borrow check without releasing protected bytes. */
        void free(void *pointer, int ordinal) override
        {
            if(pointer && pointer==initialization_pointer){
                if(forbid_early_retirement && !initialization_complete)std::_Exit(76);
                EXPECT_TRUE(initialization_complete);
            }
            if (pointer && pointer == capture_base)
                NativeAllocationCaptureLease::requireUnborrowed(capture_identity);
            if (pointer && pointer == second_capture_base)
                NativeAllocationCaptureLease::requireUnborrowed(second_capture_identity);
            MockBackend::free(pointer, ordinal);
        }
        std::shared_ptr<const NativeAllocationCaptureIdentity> capture_identity =
            std::make_shared<NativeAllocationCaptureIdentity>();
        std::shared_ptr<const NativeAllocationCaptureIdentity> second_capture_identity =
            std::make_shared<NativeAllocationCaptureIdentity>();
        void *second_capture_base = nullptr;
        size_t second_capture_bytes = 0;
        void *capture_base = nullptr;
        size_t capture_bytes = 0;
        DeviceId capture_device = DeviceId::invalid();
        int capture_acquisitions = 0;
        bool capture_sealed = false;

        std::set<void *> registered;
        int registrations = 0, unregistrations = 0, preparations = 0, setup_queries = 0;
        size_t complete_operations = 0;
        bool fail_prepare = false, fail_initialization = false, fail_boundary = false;
        std::vector<Call> calls;
        std::vector<CapturedTransferChannelDeviceBinding> native_bindings;
    };

    /** @brief Native-domain spy preserving the leader-before-broadcast data dependency. */
    class DomainCollective final : public llaminar2::test::MockLocalTPContext
    {
    public:
        /** @brief Record native broadcast order and copy only the admitted physical extent. */
        bool broadcastRawOnStream(const void *send, void *receive, size_t count, CollectiveDataType dtype,
            int root, int participant, void *stream, const std::string &) override
        {
            EXPECT_EQ(dtype, CollectiveDataType::FLOAT32);
            EXPECT_EQ(root, 0);
            EXPECT_EQ(send, receive);
            EXPECT_NE(stream, nullptr);
            calls->push_back({"broadcast", stream, count * sizeof(float)});
            if (fail) return false;
            if (participant == 0) leader = send;
            EXPECT_NE(leader, nullptr);
            if (participant != 0) std::memcpy(receive, leader, count * sizeof(float));
            return true;
        }
        std::vector<Call> *calls = nullptr;
        const void *leader = nullptr;
        bool fail = false;
    };

    /** @brief Inspect complete metadata fanout without a device or a live protocol clock. */
    class MetadataCollective final : public llaminar2::test::MockLocalTPContext
    {
    public:
        /** @brief Prove native metadata remains INT32 and bound to the exact local stream. */
        bool collectiveSidebandSpanOnStream(std::span<const LocalTPCollectiveSidebandBuffer> fields,
            int, void *stream, const std::string &) override
        {
            for (const auto &field : fields)
            {
                EXPECT_EQ(field.kind, LocalTPCollectiveSidebandKind::Broadcast);
                EXPECT_EQ(field.dtype, CollectiveDataType::INT32);
                EXPECT_EQ(field.root_device_index, root);
                EXPECT_EQ(field.send_buffer, field.recv_buffer);
                calls->push_back({"metadata_broadcast", stream, field.element_count * sizeof(int32_t)});
            }
            return !fail;
        }
        std::vector<Call> *calls = nullptr;
        int root = 0;
        bool fail = false;
    };

    /** @brief One isolated pair of backend families and canonical test admission. */
    /** @brief Original fake pool worker observes only the exact cold setup stream. */
    class ChannelSetupWorker final:public llaminar2::testing::MockWorkerGPUContext {
    public:
        ChannelSetupWorker(int ordinal,ChannelBackend* backend):MockWorkerGPUContext(ordinal),backend_(backend){}
        bool ownsCurrentThread()const noexcept override{return true;}
        GPUStreamExecutionState queryStreamExecutionState(void* stream,std::string_view boundary)override{
            if(reinterpret_cast<std::uintptr_t>(stream)<=2)throw std::invalid_argument("Invalid original setup stream");
            ++backend_->setup_queries;backend_->initialization_streams.push_back(stream);
            if(boundary!="captured channel cold initialization")return GPUStreamExecutionState::Complete;
            const auto index=backend_->cold_queries++;
            const auto state=index<backend_->cold_states.size()?backend_->cold_states[index]:GPUStreamExecutionState::Complete;
            if(backend_->observe_cold_query)backend_->observe_cold_query(stream,state);
            if(backend_->throw_cold_query)throw std::runtime_error("recorded cold query failure");
            backend_->initialization_complete=state==GPUStreamExecutionState::Complete;return state;
        }
    private:ChannelBackend* backend_;
    };

    class CapturedTransferChannelBindingTest : public ::testing::Test
    {
    protected:
        /** @brief Install only mock authorities, with no backend-manager discovery. */
        void SetUp() override {
            cuda_=&cuda;rocm_=&rocm;auto& pool=GPUDeviceContextPool::instance();pool.shutdown();
            pool.registerNvidiaFactory([](int ordinal){return std::make_unique<ChannelSetupWorker>(ordinal,static_cast<ChannelBackend*>(resolve(DeviceId::cuda(ordinal))));},8);
            pool.registerAMDFactory([](int ordinal){return std::make_unique<ChannelSetupWorker>(ordinal,static_cast<ChannelBackend*>(resolve(DeviceId::rocm(ordinal))));},8);
            for(int ordinal=0;ordinal<8;++ordinal){(void)pool.retainContext(DeviceId::cuda(ordinal));(void)pool.retainContext(DeviceId::rocm(ordinal));}
        }
        /** @brief Clear borrowed test pointers after all local owners have retired. */
        void TearDown() override { GPUDeviceContextPool::instance().shutdown();cuda_ = nullptr; rocm_ = nullptr; }
        /** @return Exact-family spy through TransferEngine's dependency-injection surface. */
        static IBackend *resolve(DeviceId device) { return device.is_cuda() ? cuda_ : device.is_rocm() ? rocm_ : nullptr; }
        /** @return An exact-sized, all-or-nothing canonical memory ledger for a channel. */
        std::unique_ptr<PhysicalMemoryAuthority> admission(DeviceId producer, DeviceId consumer, size_t capacity,
            size_t consumer_shortfall = 0, size_t channels = 1)
        {
            const auto geometry = CapturedTransferChannel::memoryFor(capacity);
            PhysicalMemoryPlanBuilder builder;
            for (const auto device : {DeviceId::cpu(), producer, consumer})
            {
                const auto bytes = channels * (device.is_cpu() ? geometry.mapped_host_bytes : geometry.cursor_bytes_per_device);
                builder.add({.world_rank = 0, .device = device, .total_bytes = bytes, .admission_available_bytes = bytes},
                    PhysicalMemoryOwner::ActivationTransportStaging, bytes - (device == consumer ? consumer_shortfall : 0));
            }
            return std::make_unique<PhysicalMemoryAuthority>(
                std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
        }
        /** @return Remaining canonical capacity; no parallel fixture-side live ledger. */
        size_t remaining(const PhysicalMemoryAuthority &authority, DeviceId device)
        { return authority.remainingAdmittedNewAllocationBytes(device, PhysicalMemoryOwner::ActivationTransportStaging); }

        inline static ChannelBackend *cuda_ = nullptr, *rocm_ = nullptr;
        ChannelBackend cuda{DeviceType::CUDA}, rocm{DeviceType::ROCm};
        TransferEngine transfer{resolve};
        void *producer_stream = reinterpret_cast<void *>(0x101);
        void *consumer_stream = reinterpret_cast<void *>(0x202);
    };

    TEST_F(CapturedTransferChannelBindingTest, ExactBOMAndBindingsRetainPhysicalOwnersInBothVendorOrders)
    {
        for (const bool reverse : {false, true})
        {
            const auto producer = reverse ? DeviceId::rocm(2) : DeviceId::cuda(1);
            const auto consumer = reverse ? DeviceId::cuda(1) : DeviceId::rocm(2);
            auto authority = admission(producer, consumer, 257);
            auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), producer,
                producer_stream, consumer, consumer_stream, 257);
            EXPECT_EQ(remaining(*authority, DeviceId::cpu()), 0u);
            EXPECT_EQ(remaining(*authority, producer), 0u);
            EXPECT_EQ(remaining(*authority, consumer), 0u);
            auto source = transfer.allocateDeviceTransferBuffer(512, producer);
            auto destination = transfer.allocateDeviceTransferBuffer(512, consumer);
            auto *source_bytes = static_cast<unsigned char *>(source->mutableDeviceData());
            for (size_t i = 0; i < 512; ++i) source_bytes[i] = static_cast<unsigned char>((i * 71) ^ (i >> 3));
            std::memset(destination->mutableDeviceData(), 0xa5, 512);
            std::weak_ptr<CapturedTransferChannel> weak = channel;
            {
                auto send = transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer, {7, 257}, source, 3);
                auto receive = transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Consumer, {7, 257}, destination, 11);
                const auto nonce = channel->identity().nonce;
                EXPECT_GT(nonce, 0u);
                channel.reset();
                ASSERT_FALSE(weak.expired());
                cuda.calls.clear(); rocm.calls.clear();
                const int queries = cuda.setup_queries + rocm.setup_queries;
                transfer.enqueueCapturedTransfer(send, producer_stream);
                transfer.enqueueCapturedTransfer(receive, consumer_stream);
                EXPECT_EQ(cuda.setup_queries + rocm.setup_queries, queries);
                for (auto *backend : {&cuda, &rocm})
                {
                    ASSERT_EQ(backend->calls.size(), 3u);
                    EXPECT_EQ(backend->calls[0].operation, "acquire");
                    EXPECT_EQ(backend->calls[1].operation, "copy");
                    EXPECT_EQ(backend->calls[2].operation, "publish");
                    EXPECT_EQ(backend->calls[1].bytes, 257u);
                    for (const auto &call : backend->calls)
                        EXPECT_EQ(call.stream, backend == resolve(producer) ? producer_stream : consumer_stream);
                    for (const auto &event : backend->getEventRecords())
                        if (event.type == MockBackend::EventRecord::WAIT) EXPECT_NE(event.stream, nullptr);
                    EXPECT_EQ(backend->getSyncCount(), 0u);
                    EXPECT_EQ(backend->getStreamSyncCount(), 0u);
                    EXPECT_EQ(backend->native_bindings.back().expected.nonce, nonce);
                }
                const auto *bytes = static_cast<const unsigned char *>(destination->deviceData());
                for (size_t i = 0; i < 512; ++i) EXPECT_EQ(bytes[i], i >= 11 && i < 268 ? source_bytes[i - 11 + 3] : 0xa5);
            }
            EXPECT_TRUE(weak.expired());
            EXPECT_EQ(remaining(*authority, producer), sizeof(CapturedTransferCursor));
            EXPECT_EQ(remaining(*authority, consumer), sizeof(CapturedTransferCursor));
            EXPECT_EQ(remaining(*authority, DeviceId::cpu()), CapturedTransferChannel::memoryFor(257).mapped_host_bytes);
            EXPECT_EQ(cuda.registrations, cuda.unregistrations);
            EXPECT_EQ(rocm.registrations, rocm.unregistrations);
        }
    }

    TEST_F(CapturedTransferChannelBindingTest, BadGeometryAndAdmissionFailBeforePhysicalAllocation)
    {
        EXPECT_THROW((void)CapturedTransferChannel::memoryFor(0), std::invalid_argument);
        EXPECT_THROW((void)CapturedTransferChannel::memoryFor(std::numeric_limits<size_t>::max()), std::overflow_error);
        auto authority = admission(DeviceId::cuda(0), DeviceId::rocm(0), 256, 1);
        EXPECT_THROW((void)transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), DeviceId::cuda(0),
            producer_stream, DeviceId::rocm(0), consumer_stream, 256), std::exception);
        EXPECT_EQ(cuda.getAllocationCount(), 0u);
        EXPECT_EQ(rocm.getAllocationCount(), 0u);
        EXPECT_EQ(cuda.registrations, 0);
        EXPECT_EQ(rocm.registrations, 0);
        EXPECT_EQ(remaining(*authority, DeviceId::cuda(0)), sizeof(CapturedTransferCursor));
        EXPECT_GT(remaining(*authority, DeviceId::cpu()), 0u);
    }

    TEST_F(CapturedTransferChannelBindingTest, CountedBindingsRetainCountOwnersAndNeverCopyPadding)
    {
        for (const bool reverse : {false, true})
        {
            const auto first = reverse ? DeviceId::rocm(0) : DeviceId::cuda(0);
            const auto second = reverse ? DeviceId::cuda(0) : DeviceId::rocm(0);
            auto authority = admission(first, second, 257);
            auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), first,
                producer_stream, second, consumer_stream, 257);
            auto source_count = transfer.allocateDeviceTransferBuffer(64, first);
            auto destination_count = transfer.allocateDeviceTransferBuffer(64, second);
            auto *input_count = static_cast<std::uint64_t *>(source_count->mutableDeviceData());
            auto *output_count = static_cast<std::uint64_t *>(destination_count->mutableDeviceData());
            auto source = transfer.allocateDeviceTransferBuffer(300, first);
            auto destination = transfer.allocateDeviceTransferBuffer(300, second);
            const auto capacity_send = transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer, {31, 257}, source, 3);
            const auto capacity_receive = transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Consumer, {31, 257}, destination, 11);
            const auto send = transfer.bindDeviceCountedTransfer(capacity_send, source_count);
            const auto receive = transfer.bindDeviceCountedTransfer(capacity_receive, destination_count);
            EXPECT_EQ(send.extentSource(), CapturedTransferExtentSource::ProducerDevice);
            EXPECT_EQ(receive.extentSource(), CapturedTransferExtentSource::ProducerPublication);
            EXPECT_THROW((void)transfer.bindDeviceCountedTransfer(send, source_count), std::invalid_argument);
            EXPECT_THROW((void)transfer.bindDeviceCountedTransfer(capacity_send, destination_count), std::invalid_argument);
            EXPECT_THROW((void)transfer.bindDeviceCountedTransfer(capacity_send, source_count, 1), std::invalid_argument);
            EXPECT_THROW((void)transfer.bindDeviceCountedTransfer(capacity_send, source_count, 64), std::out_of_range);
            const auto overlapping = transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer, {31, 32}, source_count);
            EXPECT_THROW((void)transfer.bindDeviceCountedTransfer(overlapping, source_count), std::invalid_argument);
            source_count.reset(); destination_count.reset();
            const auto queries = cuda.setup_queries + rocm.setup_queries;
            for (const auto bytes : {257u, 0u, 1u, 255u, 16u, 0u, 257u})
            {
                cuda.calls.clear(); rocm.calls.clear();
                std::memset(source->mutableDeviceData(), static_cast<int>(bytes % 251), 300);
                std::memset(destination->mutableDeviceData(), 0xa5, 300);
                *input_count = bytes;
                *output_count = ~std::uint64_t{0};
                const auto operations_before = cuda.complete_operations + rocm.complete_operations;
                transfer.enqueueCapturedTransfer(send, producer_stream);
                // A producer may update its next count after publication. The
                // receiver must still take the just-published original extent.
                *input_count = ~std::uint64_t{0};
                transfer.enqueueCapturedTransfer(receive, consumer_stream);
                EXPECT_EQ(cuda.complete_operations + rocm.complete_operations, operations_before + 2);
                EXPECT_EQ(*output_count, bytes);
                for (auto *backend : {&cuda, &rocm})
                {
                    ASSERT_EQ(backend->calls.size(), 3u);
                    EXPECT_EQ(backend->calls[1].operation, "counted_copy");
                    EXPECT_EQ(backend->calls[1].bytes, bytes);
                }
                const auto *actual = static_cast<const unsigned char *>(destination->deviceData());
                for (size_t i = 0; i < 300; ++i)
                    ASSERT_EQ(actual[i], i >= 11 && i < 11 + bytes ? bytes % 251 : 0xa5);
            }
            EXPECT_EQ(cuda.setup_queries + rocm.setup_queries, queries);
        }
    }

    /** @test Actual tensor count roots survive retained replay without a copied mailbox or registry scan. */
    TEST_F(CapturedTransferChannelBindingTest, TensorExtentReplayRetainsOriginalOwnersAndPublishesExactBytes)
    {
        for (const bool reverse : {false, true})
        for (const bool batched : {false, true})
        {
            const auto first = reverse ? DeviceId::rocm(0) : DeviceId::cuda(0);
            const auto second = reverse ? DeviceId::cuda(0) : DeviceId::rocm(0);
            auto &producer = reverse ? rocm : cuda;
            auto &consumer = reverse ? cuda : rocm;
            producer.capture_sealed = consumer.capture_sealed = false;
            auto authority = admission(first, second, 257);
            auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), first,
                producer_stream, second, consumer_stream, 257);
            const auto count = [&](DeviceId device, ChannelBackend &backend, void *stream) {
                auto root = std::make_shared<INT32Tensor>(std::vector<size_t>{16});
                root->setBackendForTesting(&backend);
                TransferEngine::prepareDeviceInput(root.get(), device, stream);
                backend.capture_identity = std::make_shared<NativeAllocationCaptureIdentity>();
                backend.capture_base = root->gpu_data_ptr();
                backend.capture_bytes = root->size_bytes();
                backend.capture_device = device;
                return root;
            };
            auto source_count = count(first, producer, producer_stream);
            auto destination_count = count(second, consumer, consumer_stream);
            std::weak_ptr<INT32Tensor> source_lifetime = source_count, destination_lifetime = destination_count;
            auto *source_owner = source_count.get();
            auto *destination_owner = destination_count.get();
            auto *input = reinterpret_cast<std::uint64_t *>(static_cast<std::byte *>(source_count->gpu_data_ptr()) + 8);
            auto *output = reinterpret_cast<std::uint64_t *>(static_cast<std::byte *>(destination_count->gpu_data_ptr()) + 16);
            auto source = transfer.allocateDeviceTransferBuffer(300, first);
            auto destination = transfer.allocateDeviceTransferBuffer(300, second);
            {
                auto send = transfer.bindDeviceCountedTransfer(transfer.bindCapturedTransfer(channel,
                    CapturedTransferEndpoint::Producer, {71, 257}, source, 3), source_count, 8);
                const auto receive = transfer.bindDeviceCountedTransfer(transfer.bindCapturedTransfer(channel,
                    CapturedTransferEndpoint::Consumer, {71, 257}, destination, 11), destination_count, 16);
                source_count.reset(); destination_count.reset();
                EXPECT_FALSE(source_lifetime.expired()); EXPECT_FALSE(destination_lifetime.expired());
                EXPECT_THROW(NativeAllocationCaptureLease::requireUnborrowed(producer.capture_identity), std::logic_error);
                EXPECT_THROW(NativeAllocationCaptureLease::requireUnborrowed(consumer.capture_identity), std::logic_error);
                EXPECT_THROW(producer.free(producer.capture_base, first.ordinal), std::logic_error);
                EXPECT_THROW(consumer.free(consumer.capture_base, second.ordinal), std::logic_error);
                const auto acquisitions = producer.capture_acquisitions + consumer.capture_acquisitions;
                producer.capture_sealed = consumer.capture_sealed = true;
                for (const auto bytes : {257u, 0u, 1u, 255u, 16u, 0u, 257u})
                {
                    *input = bytes; *output = ~std::uint64_t{0};
                    std::memset(source->mutableDeviceData(), 0x3c, 300);
                    std::memset(destination->mutableDeviceData(), 0xa5, 300);
                    TransferEngine::publishDeviceWrite(source_owner, first, producer_stream);
                    producer.calls.clear(); consumer.calls.clear();
                    const auto publications = consumer.getEventRecordCount();
                    if (batched) transfer.enqueueCapturedTransferBatch(std::span(&send, 1), producer_stream);
                    else transfer.enqueueCapturedTransfer(send, producer_stream);
                    *input = ~std::uint64_t{0}; // Receiver adopts the authenticated published extent.
                    if (batched) transfer.enqueueCapturedTransferBatch(std::span(&receive, 1), consumer_stream);
                    else transfer.enqueueCapturedTransfer(receive, consumer_stream);
                    EXPECT_EQ(*output, bytes);
                    EXPECT_EQ(consumer.getEventRecordCount(), publications + 1);
                    for (const auto *backend : {&producer, &consumer})
                    {
                        const auto copy = std::find_if(backend->calls.begin(), backend->calls.end(),
                            [](const Call &call) { return call.operation == "counted_copy"; });
                        ASSERT_NE(copy, backend->calls.end()); EXPECT_EQ(copy->bytes, bytes);
                    }
                    const auto *actual = static_cast<const unsigned char *>(destination->deviceData());
                    for (size_t i = 0; i < 300; ++i)
                        ASSERT_EQ(actual[i], i >= 11 && i < 11 + bytes ? 0x3c : 0xa5);
                    // A separate downstream stream must acquire this exact scalar producer.
                    EXPECT_NO_THROW(TransferEngine::requireDeviceInput(destination_owner, second,
                        reinterpret_cast<void *>(0x303)));
                }
                EXPECT_EQ(producer.capture_acquisitions + consumer.capture_acquisitions, acquisitions);
                EXPECT_EQ(producer.getSyncCount() + consumer.getSyncCount(), 0u);
                EXPECT_EQ(producer.getStreamSyncCount() + consumer.getStreamSyncCount(), 0u);
                // Copy-and-swap replacement must retire the old borrow before
                // dropping its last original Tensor owner, just like destruction.
                send = receive;
                EXPECT_TRUE(source_lifetime.expired());
                EXPECT_NO_THROW(NativeAllocationCaptureLease::requireUnborrowed(producer.capture_identity));
            }
            EXPECT_TRUE(source_lifetime.expired()); EXPECT_TRUE(destination_lifetime.expired());
            EXPECT_NO_THROW(NativeAllocationCaptureLease::requireUnborrowed(producer.capture_identity));
            EXPECT_NO_THROW(NativeAllocationCaptureLease::requireUnborrowed(consumer.capture_identity));
        }
    }

    /** @test Original tensor locality, exact bounds, aliases and backend authority fail before native enqueue. */
    TEST_F(CapturedTransferChannelBindingTest, TensorExtentRejectsForeignUnpreparedAndOverlappingRoots)
    {
        auto authority = admission(DeviceId::cuda(0), DeviceId::rocm(0), 64);
        auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), DeviceId::cuda(0),
            producer_stream, DeviceId::rocm(0), consumer_stream, 64);
        auto source = transfer.allocateDeviceTransferBuffer(64, DeviceId::cuda(0));
        const auto capacity = transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer, {72, 64}, source);
        auto count = std::make_shared<INT32Tensor>(std::vector<size_t>{16});
        count->setBackendForTesting(&cuda);
        EXPECT_THROW((void)transfer.bindDeviceCountedTransfer(capacity, count), std::invalid_argument);
        EXPECT_THROW((void)transfer.bindDeviceCountedTransfer(capacity, std::shared_ptr<ITensor>{}), std::invalid_argument);
        TransferEngine::prepareDeviceInput(count.get(), DeviceId::cuda(0), producer_stream);
        cuda.capture_base = count->gpu_data_ptr(); cuda.capture_bytes = count->size_bytes(); cuda.capture_device = DeviceId::cuda(0);
        EXPECT_THROW((void)transfer.bindDeviceCountedTransfer(capacity, count, 1), std::invalid_argument);
        EXPECT_THROW((void)transfer.bindDeviceCountedTransfer(capacity, count, 60), std::out_of_range);
        EXPECT_THROW((void)transfer.bindDeviceCountedTransfer(capacity, count, std::numeric_limits<size_t>::max()), std::out_of_range);
        auto foreign = std::make_shared<INT32Tensor>(std::vector<size_t>{16});
        foreign->setBackendForTesting(&rocm);
        TransferEngine::prepareDeviceInput(foreign.get(), DeviceId::rocm(0), consumer_stream);
        EXPECT_THROW((void)transfer.bindDeviceCountedTransfer(capacity, foreign), std::invalid_argument);
        const auto overlapping = transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer, {72, 64}, count);
        EXPECT_THROW((void)transfer.bindDeviceCountedTransfer(overlapping, count, 8), std::invalid_argument);
        const auto counted = transfer.bindDeviceCountedTransfer(capacity, count, 8);
        EXPECT_THROW((void)transfer.bindDeviceCountedTransfer(counted, count, 8), std::invalid_argument);
        {
            GraphCaptureGuard capture;
            EXPECT_THROW((void)transfer.bindDeviceCountedTransfer(capacity, count, 8), std::invalid_argument);
        }
        // Replacing the tensor's backend cannot retain the old native storage identity.
        count->setBackendForTesting(&rocm);
        cuda.calls.clear(); rocm.calls.clear();
        EXPECT_THROW(transfer.enqueueCapturedTransfer(counted, producer_stream), std::logic_error);
        EXPECT_TRUE(cuda.calls.empty()); EXPECT_TRUE(rocm.calls.empty());
        count->setBackendForTesting(&cuda);
    }

    /** @test Forked counts require the actual acquired companion and an existing paired output publication. */
    TEST_F(CapturedTransferChannelBindingTest, TensorExtentForkRejectsUnacquiredCompanionAndUnjoinedOutput)
    {
        // The production fork records/waits through the worker pool as well
        // as the injected backend. Install both hardware-free interfaces;
        // never let this Unit path discover a real CUDA/HIP context.
        llaminar2::testing::installHardwareFreeGPUContextFactories();
        const auto first = DeviceId::cuda(0), second = DeviceId::rocm(0);
        auto authority = admission(first, second, 64, 0, 2);
        auto forward = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), first,
            producer_stream, second, consumer_stream, 64);
        auto backward = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), second,
            consumer_stream, first, producer_stream, 64);
        const auto tensor = [&] {
            auto root = std::make_shared<INT32Tensor>(std::vector<size_t>{32});
            root->setBackendForTesting(&cuda);
            TransferEngine::prepareDeviceInput(root.get(), first, producer_stream);
            return root;
        };
        auto source = tensor(), destination = tensor(), count = tensor();
        cuda.capture_base = count->gpu_data_ptr(); cuda.capture_bytes = count->size_bytes(); cuda.capture_device = first;
        const std::array sends{transfer.bindDeviceCountedTransfer(transfer.bindCapturedTransfer(forward,
            CapturedTransferEndpoint::Producer, {73, 64}, source), count)};
        const auto receive = transfer.bindCapturedTransfer(backward, CapturedTransferEndpoint::Consumer, {74, 64}, destination);
        const std::array receives{receive};
        void *ready = cuda.createEvent(0);
        void *auxiliary = reinterpret_cast<void *>(0x404);
        const auto incomplete = transfer.acquireDeviceInputFork(transfer.recordDeviceInputFork(
            DeviceTransferForkSources::single(*source), first, producer_stream, ready), auxiliary);
        cuda.calls.clear();
        EXPECT_THROW(transfer.enqueueForkedCapturedExchange(sends, receives, incomplete, destination.get(),
            CapturedTransferEndpointScheduling::Individual), std::invalid_argument);
        EXPECT_TRUE(cuda.calls.empty());
        const auto acquired = transfer.acquireDeviceInputFork(transfer.recordDeviceInputFork(
            DeviceTransferForkSources::withCompanion(*source, *count), first, producer_stream, ready), auxiliary);
        const std::array unjoined{transfer.bindDeviceCountedTransfer(receive, count, 8)};
        cuda.calls.clear();
        EXPECT_THROW(transfer.enqueueForkedCapturedExchange(sends, unjoined, acquired, destination.get(),
            CapturedTransferEndpointScheduling::Batched), std::invalid_argument);
        EXPECT_TRUE(cuda.calls.empty());
        cuda.destroyEvent(ready, 0);
    }

    /** @test Device-live source regions retain one original authority and transmit no unused bank capacity. */
    TEST_F(CapturedTransferChannelBindingTest, SourceSliceReplayRetainsOriginalTensorAndExactRegion)
    {
        for (const bool reverse : {false, true})
        for (const bool batched : {false, true})
        {
            const auto first = reverse ? DeviceId::rocm(0) : DeviceId::cuda(0);
            const auto second = reverse ? DeviceId::cuda(0) : DeviceId::rocm(0);
            auto &backend = reverse ? rocm : cuda;
            backend.capture_sealed = false;
            auto authority = admission(first, second, 257);
            auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), first,
                producer_stream, second, consumer_stream, 257);
            auto root = std::make_shared<INT32Tensor>(std::vector<size_t>{256});
            root->setBackendForTesting(&backend);
            TransferEngine::prepareDeviceInput(root.get(), first, producer_stream);
            backend.capture_identity = std::make_shared<NativeAllocationCaptureIdentity>();
            backend.capture_base = root->gpu_data_ptr(); backend.capture_bytes = root->size_bytes(); backend.capture_device = first;
            auto *original = root.get();
            auto *source = static_cast<std::uint8_t *>(root->gpu_data_ptr());
            auto *slice = reinterpret_cast<CapturedTransferSourceSlice *>(source + 768);
            auto destination = transfer.allocateDeviceTransferBuffer(512, second);
            auto count = transfer.allocateDeviceTransferBuffer(8, second);
            auto *received = static_cast<std::uint64_t *>(count->mutableDeviceData());
            const auto before = backend.capture_acquisitions;
            const auto send = transfer.bindDeviceSlicedTransfer(transfer.bindCapturedTransfer(channel,
                CapturedTransferEndpoint::Producer, {81, 257}, root), root, 768, 512);
            const auto receive = transfer.bindDeviceCountedTransfer(transfer.bindCapturedTransfer(channel,
                CapturedTransferEndpoint::Consumer, {81, 257}, destination, 11), count);
            EXPECT_EQ(backend.capture_acquisitions, before + 1); // Descriptor/source share the original native record.
            const auto identity = send.captureIdentity();
            const auto copy = send; EXPECT_EQ(copy.captureIdentity(), identity);
            EXPECT_THROW(backend.free(backend.capture_base, first.ordinal), std::logic_error);
            std::weak_ptr<INT32Tensor> lifetime = root;
            root.reset(); ASSERT_FALSE(lifetime.expired()); backend.capture_sealed = true;
            for (std::uint64_t replay = 0; replay < 128; ++replay)
            {
                const auto local = replay % 16; // Reset request data without resetting transfer epochs.
                const auto bytes = local % 4 == 0 ? 257u : local % 4 == 1 ? 0u : local % 4 == 2 ? 1u : 17u;
                const auto offset = bytes ? (local * 31) % 256 : 512u;
                *slice = {offset, bytes};
                EXPECT_EQ(send.captureIdentity(), identity); // Live descriptor values are never host cache identity.

                for (size_t i = 0; i < 512; ++i) source[i] = static_cast<std::uint8_t>((i * 71 + local) % 251);
                std::memset(destination->mutableDeviceData(), 0xa5, 512);
                TransferEngine::publishDeviceWrite(original, first, producer_stream);
                if (batched) transfer.enqueueCapturedTransferBatch(std::span(&send, 1), producer_stream);
                else transfer.enqueueCapturedTransfer(send, producer_stream);
                *slice = {~std::uint64_t{0}, ~std::uint64_t{0}};
                if (batched) transfer.enqueueCapturedTransferBatch(std::span(&receive, 1), consumer_stream);
                else transfer.enqueueCapturedTransfer(receive, consumer_stream);
                EXPECT_EQ(*received, bytes);
                const auto *actual = static_cast<const std::uint8_t *>(destination->deviceData());
                for (size_t i = 0; i < 512; ++i)
                    ASSERT_EQ(actual[i], i >= 11 && i < 11 + bytes ? source[offset + i - 11] : 0xa5);
                ASSERT_FALSE(backend.native_bindings.empty());
                EXPECT_EQ(backend.native_bindings.back().cursor->completed_epoch, replay + 1);
            }
            EXPECT_EQ(backend.capture_acquisitions, before + 1);
            EXPECT_EQ(backend.getSyncCount(), 0u); EXPECT_EQ(backend.getStreamSyncCount(), 0u);
        }
    }

    /** @test Slices cannot grant access to another region, metadata, role or allocation owner. */
    TEST_F(CapturedTransferChannelBindingTest, SourceSliceRejectsUnboundedAliasedAndForeignDescriptors)
    {
        auto authority = admission(DeviceId::cuda(0), DeviceId::rocm(0), 257);
        auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), DeviceId::cuda(0),
            producer_stream, DeviceId::rocm(0), consumer_stream, 257);
        auto root = std::make_shared<INT32Tensor>(std::vector<size_t>{256});
        root->setBackendForTesting(&cuda); TransferEngine::prepareDeviceInput(root.get(), DeviceId::cuda(0), producer_stream);
        cuda.capture_base = root->gpu_data_ptr(); cuda.capture_bytes = root->size_bytes(); cuda.capture_device = DeviceId::cuda(0);
        const auto capacity = transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer, {82, 257}, root);
        EXPECT_THROW((void)transfer.bindDeviceSlicedTransfer(capacity, root, 768, 0), std::invalid_argument);
        EXPECT_THROW((void)transfer.bindDeviceSlicedTransfer(capacity, root, 768, 256), std::invalid_argument);
        EXPECT_THROW((void)transfer.bindDeviceSlicedTransfer(capacity, root, 768, 512, ~std::uint64_t{0}), std::invalid_argument);
        EXPECT_THROW((void)transfer.bindDeviceSlicedTransfer(capacity, root, 768, 1025), std::out_of_range);
        EXPECT_THROW((void)transfer.bindDeviceSlicedTransfer(capacity, root, 769, 512), std::invalid_argument);
        EXPECT_THROW((void)transfer.bindDeviceSlicedTransfer(capacity, root, 1016, 512), std::out_of_range);
        // This aliases the wider source span beyond the smaller wire maximum.
        EXPECT_THROW((void)transfer.bindDeviceSlicedTransfer(capacity, root, 504, 512), std::invalid_argument);
        const auto sliced = transfer.bindDeviceSlicedTransfer(capacity, root, 768, 512);
        EXPECT_THROW((void)transfer.bindDeviceSlicedTransfer(sliced, root, 768, 512), std::invalid_argument);
        auto foreign = std::make_shared<INT32Tensor>(std::vector<size_t>{256});
        foreign->setBackendForTesting(&rocm); TransferEngine::prepareDeviceInput(foreign.get(), DeviceId::rocm(0), consumer_stream);
        EXPECT_THROW((void)transfer.bindDeviceSlicedTransfer(capacity, foreign, 768, 512), std::invalid_argument);
        const auto consumer = transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Consumer, {82, 257}, foreign);
        EXPECT_THROW((void)transfer.bindDeviceSlicedTransfer(consumer, foreign, 768, 512), std::invalid_argument);
        {
            GraphCaptureGuard capture;
            EXPECT_THROW((void)transfer.bindDeviceSlicedTransfer(capacity, root, 768, 512), std::invalid_argument);
        }
        root->setBackendForTesting(&rocm); cuda.calls.clear(); rocm.calls.clear();
        EXPECT_THROW(transfer.enqueueCapturedTransfer(sliced, producer_stream), std::logic_error);
        EXPECT_TRUE(cuda.calls.empty()); EXPECT_TRUE(rocm.calls.empty()); root->setBackendForTesting(&cuda);
    }

    /** @brief Expose declaration-only byte drift while preserving the original physical allocation in negative replays. */
    class SourceSliceGeometryTensor final : public INT32Tensor
    {
    public:
        /** @brief Construct an ordinary typed root. @param elements Exact original INT32 element count. */
        explicit SourceSliceGeometryTensor(size_t elements) : INT32Tensor({elements}) {}
        /** @brief Change only reported geometry for a rejected-binding negative. @param bytes Zero restores the original declaration. */
        void reportBytesForNegative(size_t bytes) noexcept { reported_bytes_ = bytes; }
    protected:
        /** @return Original physical byte declaration unless the test explicitly injects drift. */
        size_t byte_size() const override { return reported_bytes_ ? reported_bytes_ : INT32Tensor::byte_size(); }
    private:
        size_t reported_bytes_ = 0;
    };

    /** @test Distinct original Tensor roots retain two native records and preserve nonzero origin through bind/enqueue. */
    TEST_F(CapturedTransferChannelBindingTest, SourceSliceDistinctTensorRootsPreserveOriginAndRejectIndependentDrift)
    {
        for (const bool reverse : {false, true})
        for (const bool batched : {false, true})
        {
            const auto first = reverse ? DeviceId::rocm(0) : DeviceId::cuda(0);
            const auto second = reverse ? DeviceId::cuda(0) : DeviceId::rocm(0);
            auto &backend = reverse ? rocm : cuda;
            backend.capture_sealed = false;
            auto authority = admission(first, second, 257);
            auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), first,
                producer_stream, second, consumer_stream, 257);
            auto source = std::make_shared<SourceSliceGeometryTensor>(256);
            auto descriptor = std::make_shared<SourceSliceGeometryTensor>(16);
            for (const auto &root : {source, descriptor})
            {
                root->setBackendForTesting(&backend);
                TransferEngine::prepareDeviceInput(root.get(), first, producer_stream);
            }
            backend.capture_identity = std::make_shared<NativeAllocationCaptureIdentity>();
            backend.second_capture_identity = std::make_shared<NativeAllocationCaptureIdentity>();
            backend.capture_base = source->gpu_data_ptr(); backend.capture_bytes = source->size_bytes(); backend.capture_device = first;
            backend.second_capture_base = descriptor->gpu_data_ptr(); backend.second_capture_bytes = descriptor->size_bytes();
            const auto before = backend.capture_acquisitions;
            auto destination = transfer.allocateDeviceTransferBuffer(512, second);
            auto count = transfer.allocateDeviceTransferBuffer(8, second);
            auto *received = static_cast<std::uint64_t *>(count->mutableDeviceData());
            auto *payload = static_cast<std::uint8_t *>(source->gpu_data_ptr()) + 16;
            auto *slice = reinterpret_cast<CapturedTransferSourceSlice *>(static_cast<std::byte *>(descriptor->gpu_data_ptr()) + 8);
            constexpr std::uint64_t origin = 4000, span = 512;
            std::weak_ptr<SourceSliceGeometryTensor> source_lifetime = source, descriptor_lifetime = descriptor;
            {
                const auto send = transfer.bindDeviceSlicedTransfer(transfer.bindCapturedTransfer(channel,
                    CapturedTransferEndpoint::Producer, {83, 257}, source, 16), descriptor, 8, span, origin);
                const auto receive = transfer.bindDeviceCountedTransfer(transfer.bindCapturedTransfer(channel,
                    CapturedTransferEndpoint::Consumer, {83, 257}, destination, 11), count);
                EXPECT_EQ(backend.capture_acquisitions, before + 2);
                EXPECT_THROW(backend.free(backend.capture_base, first.ordinal), std::logic_error);
                EXPECT_THROW(backend.free(backend.second_capture_base, first.ordinal), std::logic_error);
                const auto identity = send.captureIdentity();
                const auto rejectsWithoutSubmission = [&] {
                    const auto native_operations = cuda.complete_operations + rocm.complete_operations;
                    EXPECT_THROW(transfer.enqueueCapturedTransfer(send, producer_stream), std::logic_error);
                    EXPECT_EQ(cuda.complete_operations + rocm.complete_operations, native_operations);
                };
                source->reportBytesForNegative(1028); rejectsWithoutSubmission(); source->reportBytesForNegative(0);
                descriptor->reportBytesForNegative(68); rejectsWithoutSubmission(); descriptor->reportBytesForNegative(0);
                auto *foreign_backend = reverse ? static_cast<IBackend *>(&cuda) : static_cast<IBackend *>(&rocm);
                source->setBackendForTesting(foreign_backend); rejectsWithoutSubmission(); source->setBackendForTesting(&backend);
                descriptor->setBackendForTesting(foreign_backend); rejectsWithoutSubmission(); descriptor->setBackendForTesting(&backend);
                auto *source_owner = source.get(); auto *descriptor_owner = descriptor.get();
                source.reset(); descriptor.reset();
                ASSERT_FALSE(source_lifetime.expired()); ASSERT_FALSE(descriptor_lifetime.expired());
                backend.capture_sealed = true;
                for (std::uint64_t replay = 0; replay < 64; ++replay)
                {
                    const auto local = replay % 16;
                    const auto bytes = local % 4 == 0 ? 257u : local % 4 == 1 ? 0u : local % 4 == 2 ? 1u : 17u;
                    const auto relative = bytes ? (local * 31) % 256 : span;
                    *slice = {origin + relative, bytes};
                    const auto original_slice = *slice;
                    for (size_t i = 0; i < span; ++i) payload[i] = static_cast<std::uint8_t>((i * 71 + local) % 251);
                    std::memset(destination->mutableDeviceData(), 0xa5, 512);
                    TransferEngine::publishDeviceWrite(source_owner, first, producer_stream);
                    TransferEngine::publishDeviceWrite(descriptor_owner, first, producer_stream);
                    EXPECT_EQ(send.captureIdentity(), identity);
                    if (batched) transfer.enqueueCapturedTransferBatch(std::span(&send, 1), producer_stream);
                    else transfer.enqueueCapturedTransfer(send, producer_stream);
                    ASSERT_FALSE(backend.native_bindings.empty());
                    const auto native = backend.native_bindings.back();
                    EXPECT_EQ(native.source_offset_origin, origin);
                    // Original dropped-origin forwarding control uses actual
                    // bound native arguments, changing only the lost scalar.
                    CapturedTransferCursor rejected{};
                    EXPECT_EQ(CapturedTransferChannelProtocol::acquire(native.role, native.expected,
                        native.control->identity, rejected, {}, native.message, native.extent_source,
                        original_slice.live_bytes, original_slice.byte_offset, native.source_span_bytes, 0).status,
                        CapturedTransferStatus::InvalidBinding);
                    *slice = {~std::uint64_t{0}, ~std::uint64_t{0}};
                    if (batched) transfer.enqueueCapturedTransferBatch(std::span(&receive, 1), consumer_stream);
                    else transfer.enqueueCapturedTransfer(receive, consumer_stream);
                    EXPECT_EQ(*received, bytes);
                    const auto *actual = static_cast<const std::uint8_t *>(destination->deviceData());
                    for (size_t i = 0; i < 512; ++i)
                        ASSERT_EQ(actual[i], i >= 11 && i < 11 + bytes ? payload[relative + i - 11] : 0xa5);
                }
                EXPECT_EQ(backend.capture_acquisitions, before + 2);
            }
            EXPECT_TRUE(source_lifetime.expired()); EXPECT_TRUE(descriptor_lifetime.expired());
            EXPECT_NO_THROW(NativeAllocationCaptureLease::requireUnborrowed(backend.capture_identity));
            EXPECT_NO_THROW(NativeAllocationCaptureLease::requireUnborrowed(backend.second_capture_identity));
        }
    }

    TEST_F(CapturedTransferChannelBindingTest, BatchesRetainExactOriginalOwnersAndRejectBeforeAnySubmission)
    {
        constexpr size_t lanes = CapturedTransferChannelBatch::kCapacity + 1;
        constexpr size_t capacity = 257;
        auto authority = admission(DeviceId::cuda(0), DeviceId::rocm(0), capacity, 0, lanes);
        auto source = transfer.allocateDeviceTransferBuffer(capacity, DeviceId::cuda(0));
        auto destination = transfer.allocateDeviceTransferBuffer(lanes * capacity, DeviceId::rocm(0));
        std::memset(source->mutableDeviceData(), 0x37, capacity);
        std::memset(destination->mutableDeviceData(), 0xa5, lanes * capacity);
        std::vector<CapturedTransferBinding> sends, receives;
        std::vector<std::weak_ptr<CapturedTransferChannel>> owners;
        for (size_t i = 0; i < lanes; ++i)
        {
            auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(),
                DeviceId::cuda(0), producer_stream, DeviceId::rocm(0), consumer_stream, capacity);
            owners.push_back(channel);
            sends.push_back(transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer,
                {71 + i, capacity}, source));
            receives.push_back(transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Consumer,
                {71 + i, capacity}, destination, i * capacity));
        }
        const auto attempts = cuda.complete_operations + rocm.complete_operations;
        EXPECT_THROW(transfer.enqueueCapturedTransferBatch({}, producer_stream), std::invalid_argument);
        EXPECT_THROW(transfer.enqueueCapturedTransferBatch(sends, nullptr), std::invalid_argument);
        auto malformed = sends;
        malformed.back() = malformed.front(); // Duplicate in a later native packet.
        EXPECT_THROW(transfer.enqueueCapturedTransferBatch(malformed, producer_stream), std::invalid_argument);
        malformed = sends;
        malformed.back() = receives.back();
        EXPECT_THROW(transfer.enqueueCapturedTransferBatch(malformed, producer_stream), std::invalid_argument);
        auto overlap = receives;
        // A different original channel writing the same receive range is not independent.
        auto extra = transfer.bindCapturedTransfer(owners.back().lock(), CapturedTransferEndpoint::Consumer,
            {71 + lanes - 1, capacity}, destination, 0);
        overlap.back() = extra;
        EXPECT_THROW(transfer.enqueueCapturedTransferBatch(overlap, consumer_stream), std::invalid_argument);
        EXPECT_EQ(cuda.complete_operations + rocm.complete_operations, attempts);
        const int observations = cuda.setup_queries + rocm.setup_queries;
        transfer.enqueueCapturedTransferBatch(sends, producer_stream);
        transfer.enqueueCapturedTransferBatch(receives, consumer_stream);
        EXPECT_EQ(cuda.batch_sizes, (std::vector<unsigned>{CapturedTransferChannelBatch::kCapacity, 1}));
        EXPECT_EQ(rocm.batch_sizes, (std::vector<unsigned>{CapturedTransferChannelBatch::kCapacity, 1}));
        EXPECT_EQ(cuda.setup_queries + rocm.setup_queries, observations);
        const auto *bytes = static_cast<const unsigned char *>(destination->deviceData());
        for (size_t i = 0; i < lanes * capacity; ++i) EXPECT_EQ(bytes[i], 0x37);
        for (const auto &owner : owners) EXPECT_FALSE(owner.expired());
        cuda.fail_boundary = true;
        const auto accepted = cuda.complete_operations;
        EXPECT_THROW(transfer.enqueueCapturedTransferBatch(sends, producer_stream), std::runtime_error);
        EXPECT_EQ(cuda.complete_operations, accepted); // No individual-endpoint retry.
        sends.clear(); receives.clear(); malformed.clear(); overlap.clear();
        // extra is intentionally still a real original binding, so its owner remains.
        EXPECT_FALSE(owners.back().expired());
        for (size_t i = 0; i + 1 < lanes; ++i) EXPECT_TRUE(owners[i].expired());
        EXPECT_EQ(cuda.getSyncCount() + rocm.getSyncCount(), 0u);
        EXPECT_EQ(cuda.getStreamSyncCount() + rocm.getStreamSyncCount(), 0u);
    }

    /** @brief Collect only accepted batch host submissions and restore exact prior policy. */
    class BatchSubmissionObservation final
    {
        std::optional<std::string> json_, filter_;
        /** @return Exact prior value, distinguishing absence from an empty string. */
        static std::optional<std::string> prior(const char *name)
        {
            const char *value = std::getenv(name);
            return value ? std::optional<std::string>{value} : std::nullopt;
        }
        /** @brief Restore one owned environment key; a restoration failure is fatal. */
        static void restore(const char *name, const std::optional<std::string> &value)
        {
            if ((value ? setenv(name, value->c_str(), 1) : unsetenv(name)) != 0) std::terminate();
        }
    public:
        /** @brief Enable this exact existing counter family without enabling timers or probes. */
        BatchSubmissionObservation()
            : json_(prior("LLAMINAR_PERF_STATS_JSON")), filter_(prior("LLAMINAR_PERF_STATS_FILTER"))
        {
            if (setenv("LLAMINAR_PERF_STATS_JSON", "1", 1) != 0 ||
                setenv("LLAMINAR_PERF_STATS_FILTER", "transfer.captured_endpoint_batch_submissions", 1) != 0)
            {
                restore("LLAMINAR_PERF_STATS_FILTER", filter_);
                restore("LLAMINAR_PERF_STATS_JSON", json_);
                throw std::runtime_error("Cannot select batch submission observation");
            }
            PerfStatsCollector::reloadConfigurationFromEnvironment();
            PerfStatsCollector::reset();
        }
        /** @brief Retire fixture evidence and restore the original collector policy exactly. */
        ~BatchSubmissionObservation()
        {
            PerfStatsCollector::reset();
            restore("LLAMINAR_PERF_STATS_FILTER", filter_);
            restore("LLAMINAR_PERF_STATS_JSON", json_);
            PerfStatsCollector::reloadConfigurationFromEnvironment();
        }
        BatchSubmissionObservation(const BatchSubmissionObservation &) = delete;
        BatchSubmissionObservation &operator=(const BatchSubmissionObservation &) = delete;
    };

    TEST_F(CapturedTransferChannelBindingTest, Batch16KiBSubmissionCountersDescribeAcceptedPacketsOnly)
    {
        BatchSubmissionObservation observation;
        constexpr size_t lanes = CapturedTransferChannelBatch::kCapacity + 1;
        for (const bool reverse : {false, true})
        for (const size_t capacity : {8193u, 10240u, 16384u, 16385u})
        {
            SCOPED_TRACE(::testing::Message() << "reverse=" << reverse << " capacity=" << capacity);
            PerfStatsCollector::reset();
            const auto producer = reverse ? DeviceId::rocm(2) : DeviceId::cuda(1);
            const auto consumer = reverse ? DeviceId::cuda(1) : DeviceId::rocm(2);
            auto authority = admission(producer, consumer, capacity, 0, lanes);
            auto source = transfer.allocateDeviceTransferBuffer(capacity, producer);
            auto destination = transfer.allocateDeviceTransferBuffer(lanes * capacity, consumer);
            std::memset(source->mutableDeviceData(), 0x37, capacity);
            std::memset(destination->mutableDeviceData(), 0xa5, lanes * capacity);
            std::vector<CapturedTransferBinding> sends, receives;
            for (size_t i = 0; i < lanes; ++i)
            {
                auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(),
                    producer, producer_stream, consumer, consumer_stream, capacity);
                sends.push_back(transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer,
                    {71 + i, capacity}, source));
                receives.push_back(transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Consumer,
                    {71 + i, capacity}, destination, i * capacity));
            }
            EXPECT_THROW(transfer.enqueueCapturedTransferBatch(sends, nullptr), std::invalid_argument);
            EXPECT_TRUE(PerfStatsCollector::snapshot({"transfer.captured_endpoint_batch_submissions"}).empty());
            // Two public host submissions per role merge into each packet's existing key.
            for (int repetition = 0; repetition < 2; ++repetition)
            {
                transfer.enqueueCapturedTransferBatch(sends, producer_stream);
                transfer.enqueueCapturedTransferBatch(receives, consumer_stream);
            }
            auto *producer_backend = producer.is_cuda() ? &cuda : &rocm;
            producer_backend->fail_boundary = true;
            EXPECT_THROW(transfer.enqueueCapturedTransferBatch(sends, producer_stream), std::runtime_error);
            producer_backend->fail_boundary = false;
            const auto records = PerfStatsCollector::snapshot({"transfer.captured_endpoint_batch_submissions"});
            ASSERT_EQ(records.size(), 4u); // Producer/consumer times full/partial packet.
            std::set<std::pair<std::string, std::string>> keys;
            for (const auto &record : records)
            {
                EXPECT_EQ(record.kind, PerfStatRecord::Kind::Counter);
                EXPECT_EQ(record.domain, "transfer");
                EXPECT_EQ(record.name, "captured_endpoint_batch_submissions");
                EXPECT_TRUE(record.phase.empty());
                EXPECT_EQ(record.count, 2u);
                EXPECT_EQ(record.value, 2.0); // Failed enqueue must not increment accepted observations.
                ASSERT_TRUE(record.tags.contains("role"));
                ASSERT_TRUE(record.tags.contains("endpoints"));
                const auto &role = record.tags.at("role");
                const auto &endpoints = record.tags.at("endpoints");
                EXPECT_TRUE(role == "producer" || role == "consumer");
                EXPECT_TRUE(endpoints == "3" || endpoints == "1");
                EXPECT_TRUE(keys.emplace(role, endpoints).second);
                EXPECT_EQ(record.device, (role == "producer" ? producer : consumer).toString());
                const PerfStatsCollector::Tags expected{{"role", role}, {"endpoints", endpoints},
                    {"lowering", capacity <= 16384 ? "fused" : "acquire_copy_publish"},
                    {"accounting", "host_submission_or_graph_template"}};
                EXPECT_EQ(record.tags, expected);
            }
        }
    }

    TEST_F(CapturedTransferChannelBindingTest, BatchConsumerCountsRejectAllCrossEndpointAliasesBeforeEnqueue)
    {
        constexpr size_t lanes = CapturedTransferChannelBatch::kCapacity + 1;
        constexpr size_t capacity = 257;
        auto authority = admission(DeviceId::cuda(0), DeviceId::rocm(0), capacity, 0, lanes);
        auto source = transfer.allocateDeviceTransferBuffer(capacity, DeviceId::cuda(0));
        auto destination = transfer.allocateDeviceTransferBuffer(lanes * capacity, DeviceId::rocm(0));
        auto producer_count = transfer.allocateDeviceTransferBuffer(sizeof(std::uint64_t), DeviceId::cuda(0));
        auto consumer_counts = transfer.allocateDeviceTransferBuffer(lanes * sizeof(std::uint64_t), DeviceId::rocm(0));
        *static_cast<std::uint64_t *>(producer_count->mutableDeviceData()) = 2;
        std::memset(source->mutableDeviceData(), 0x39, capacity);
        std::memset(destination->mutableDeviceData(), 0xa5, lanes * capacity);
        std::memset(consumer_counts->mutableDeviceData(), 0x7f, lanes * sizeof(std::uint64_t));
        std::vector<CapturedTransferBinding> sends, receives, counted_receives;
        for (size_t i = 0; i < lanes; ++i)
        {
            auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(),
                DeviceId::cuda(0), producer_stream, DeviceId::rocm(0), consumer_stream, capacity);
            sends.push_back(transfer.bindDeviceCountedTransfer(transfer.bindCapturedTransfer(channel,
                CapturedTransferEndpoint::Producer, {81 + i, capacity}, source), producer_count));
            receives.push_back(transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Consumer,
                {81 + i, capacity}, destination, i * capacity));
            counted_receives.push_back(transfer.bindDeviceCountedTransfer(receives.back(), consumer_counts,
                i * sizeof(std::uint64_t)));
        }
        const auto attempts = cuda.complete_operations + rocm.complete_operations;
        auto malformed = counted_receives;
        // Distinct original channels and disjoint payloads, but duplicate
        // mutable counts across the first/second native packet boundary.
        malformed.back() = transfer.bindDeviceCountedTransfer(receives.back(), consumer_counts, 0);
        EXPECT_THROW(transfer.enqueueCapturedTransferBatch(malformed, consumer_stream), std::invalid_argument);
        malformed = counted_receives;
        // Both original per-binding own-payload checks pass. Only complete
        // cohort validation can see a count inside another payload capacity.
        constexpr size_t in_second_payload = (capacity + 7) / 8 * 8;
        malformed.front() = transfer.bindDeviceCountedTransfer(receives.front(), destination, in_second_payload);
        EXPECT_THROW(transfer.enqueueCapturedTransferBatch(malformed, consumer_stream), std::invalid_argument);
        malformed = counted_receives;
        malformed.back() = transfer.bindDeviceCountedTransfer(receives.back(), destination, 8);
        EXPECT_THROW(transfer.enqueueCapturedTransferBatch(malformed, consumer_stream), std::invalid_argument);
        EXPECT_EQ(cuda.complete_operations + rocm.complete_operations, attempts);
        // Shared producer counts are intentionally read-only; do not reject
        // multicast from one original authority. Separate consumer scalars
        // remain outputs and each must adopt the exact published size.
        transfer.enqueueCapturedTransferBatch(sends, producer_stream);
        transfer.enqueueCapturedTransferBatch(counted_receives, consumer_stream);
        for (size_t i = 0; i < lanes; ++i)
        {
            EXPECT_EQ(*static_cast<const std::uint64_t *>(consumer_counts->deviceData(i * sizeof(std::uint64_t))), 2u);
            const auto *bytes = static_cast<const unsigned char *>(destination->deviceData(i * capacity));
            for (size_t byte = 0; byte < capacity; ++byte) EXPECT_EQ(bytes[byte], byte < 2 ? 0x39 : 0xa5);
        }
    }

    /** @brief Resolve the original graph role, not an inferred captured row bucket. */
    TEST(CapturedTransferScheduling, ExplicitMoEPhaseAndMessageIdentity)
    {
        using Phase = MoEOverlayServicePhaseHint;
        using Scheduling = CapturedTransferEndpointScheduling;
        EXPECT_EQ(moeCountedTransferScheduling(Phase::Prefill), Scheduling::Individual);
        EXPECT_EQ(moeCountedTransferScheduling(Phase::Decode), Scheduling::Batched);
        EXPECT_EQ(moeCountedTransferScheduling(Phase::GroupedVerifier), Scheduling::Individual);
        EXPECT_THROW(moeCountedTransferScheduling(Phase::Auto), std::invalid_argument);
        EXPECT_THROW(moeCountedTransferScheduling(static_cast<Phase>(99)), std::invalid_argument);
        EXPECT_THROW(capturedTransferEndpointSchedulingName(static_cast<Scheduling>(0)), std::invalid_argument);
        EXPECT_THROW(capturedTransferEndpointSchedulingKey(0, Scheduling::Individual), std::invalid_argument);
        EXPECT_NE(capturedTransferEndpointSchedulingKey(31, Scheduling::Individual),
            capturedTransferEndpointSchedulingKey(31, Scheduling::Batched));
        EXPECT_EQ(capturedTransferEndpointSchedulingKey(31, Scheduling::Batched),
            capturedTransferEndpointSchedulingKey(31, Scheduling::Batched));
    }

    /** @brief A bad receive must reject before any send under either explicit policy. */
    TEST_F(CapturedTransferChannelBindingTest, ScheduledExchangePrevalidatesBothPhases)
    {
        using Scheduling = CapturedTransferEndpointScheduling;
        constexpr size_t capacity = 257;
        for (const auto scheduling : {Scheduling::Individual, Scheduling::Batched})
        {
            auto authority = admission(DeviceId::cuda(0), DeviceId::rocm(0), capacity, 0, 2);
            auto forward = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(),
                DeviceId::cuda(0), producer_stream, DeviceId::rocm(0), consumer_stream, capacity);
            auto backward = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(),
                DeviceId::rocm(0), consumer_stream, DeviceId::cuda(0), producer_stream, capacity);
            auto local = transfer.allocateDeviceTransferBuffer(capacity, DeviceId::cuda(0));
            auto remote = transfer.allocateDeviceTransferBuffer(capacity, DeviceId::rocm(0));
            auto output = transfer.allocateDeviceTransferBuffer(capacity, DeviceId::cuda(0));
            std::memset(local->mutableDeviceData(), 0x32, capacity);
            std::memset(remote->mutableDeviceData(), 0x64, capacity);
            std::memset(output->mutableDeviceData(), 0xa5, capacity);
            const std::array sends{transfer.bindCapturedTransfer(forward,
                CapturedTransferEndpoint::Producer, {31, capacity}, local)};
            const std::array receives{transfer.bindCapturedTransfer(backward,
                CapturedTransferEndpoint::Consumer, {41, capacity}, output)};
            const std::array peer_sends{transfer.bindCapturedTransfer(backward,
                CapturedTransferEndpoint::Producer, {41, capacity}, remote)};
            const auto before_calls = cuda.complete_operations + rocm.complete_operations;
            const std::array wrong_role{sends.front()};
            EXPECT_THROW(transfer.enqueueCapturedExchange(sends, wrong_role, scheduling, producer_stream),
                std::invalid_argument);
            auto duplicate = std::vector{receives.front(), receives.front()};
            EXPECT_THROW(transfer.enqueueCapturedExchange(sends, duplicate, scheduling, producer_stream),
                std::invalid_argument);
            EXPECT_THROW(transfer.enqueueCapturedExchange(sends, receives,
                static_cast<Scheduling>(0), producer_stream), std::invalid_argument);
            EXPECT_EQ(cuda.complete_operations + rocm.complete_operations, before_calls);
            // Prime only the remote original publication, then observe the
            // production same-device send/receive transaction with real spies.
            transfer.enqueueCapturedTransfer(peer_sends.front(), consumer_stream);
            cuda.batch_sizes.clear();
            const auto first_binding = cuda.native_bindings.size();
            transfer.enqueueCapturedExchange(sends, receives, scheduling, producer_stream);
            ASSERT_EQ(cuda.native_bindings.size(), first_binding + 2);
            EXPECT_EQ(cuda.native_bindings[first_binding].role, CapturedTransferEndpoint::Producer);
            EXPECT_EQ(cuda.native_bindings[first_binding + 1].role, CapturedTransferEndpoint::Consumer);
            EXPECT_EQ(cuda.batch_sizes.size(), scheduling == Scheduling::Batched ? 2u : 0u);
            const auto *bytes = static_cast<const unsigned char *>(output->deviceData());
            for (size_t byte = 0; byte < capacity; ++byte) ASSERT_EQ(bytes[byte], 0x64);
            EXPECT_EQ(cuda.getSyncCount() + rocm.getSyncCount(), 0u);
            EXPECT_EQ(cuda.getStreamSyncCount() + rocm.getStreamSyncCount(), 0u);
        }
    }

    TEST(CapturedTransferKernelPlan, CompleteDispatchUsesCaptureCapacityAndKeepsCursorBOM)
    {
        EXPECT_THROW((void)capturedTransferKernelPlan(0), std::invalid_argument);
        for (const auto bytes : {std::uint64_t{1}, std::uint64_t{4099}, kCapturedTransferSingleBlockBytes})
            EXPECT_EQ(capturedTransferKernelPlan(bytes), CapturedTransferKernelPlan::SingleBlock);
        for (const auto bytes : {kCapturedTransferSingleBlockBytes + 1, std::numeric_limits<std::uint64_t>::max()})
            EXPECT_EQ(capturedTransferKernelPlan(bytes), CapturedTransferKernelPlan::ParallelCopy);
        EXPECT_EQ(sizeof(CapturedTransferCursor), 40u);
        EXPECT_EQ(CapturedTransferChannel::memoryFor(1024).cursor_bytes_per_device, 40u);
    }

    /** @test Batch capacity owns dispatch at both boundaries; individual controls stay unchanged. */
    TEST(CapturedTransferKernelPlan, Batch16KiBBoundariesRetainIndividual8KiBAndCursorBOM)
    {
        EXPECT_THROW((void)capturedTransferBatchKernelPlan(0), std::invalid_argument);
        for (const auto bytes : {std::uint64_t{1}, std::uint64_t{8191}, std::uint64_t{8192},
                std::uint64_t{8193}, std::uint64_t{10240}, std::uint64_t{16383}, std::uint64_t{16384}})
            EXPECT_EQ(capturedTransferBatchKernelPlan(bytes), CapturedTransferKernelPlan::SingleBlock);
        for (const auto bytes : {std::uint64_t{16385}, std::numeric_limits<std::uint64_t>::max()})
            EXPECT_EQ(capturedTransferBatchKernelPlan(bytes), CapturedTransferKernelPlan::ParallelCopy);
        EXPECT_EQ(capturedTransferKernelPlan(8192), CapturedTransferKernelPlan::SingleBlock);
        EXPECT_EQ(capturedTransferKernelPlan(8193), CapturedTransferKernelPlan::ParallelCopy);
        EXPECT_EQ(capturedTransferKernelPlan(10240), CapturedTransferKernelPlan::ParallelCopy);
        EXPECT_EQ(capturedTransferKernelPlan(16384), CapturedTransferKernelPlan::ParallelCopy);
        EXPECT_EQ(sizeof(CapturedTransferCursor), 40u);
        EXPECT_EQ(CapturedTransferChannel::memoryFor(16384).cursor_bytes_per_device, 40u);
    }

    /** @test Actual cold readiness polling retains all original resources through repeated Pending. */
    TEST_F(CapturedTransferChannelBindingTest, ColdInitializationPendingTwiceThenCompleteRetainsAllOriginalClaims)
    {
        for(unsigned replay=0;replay<20;++replay){
            auto authority=admission(DeviceId::cuda(0),DeviceId::rocm(0),256);
            cuda.track_initialization_lifetime=rocm.track_initialization_lifetime=true;
            cuda.cold_states=rocm.cold_states={GPUStreamExecutionState::Pending,GPUStreamExecutionState::Pending,GPUStreamExecutionState::Complete};
            cuda.cold_queries=rocm.cold_queries=0;
            const auto assertLive=[&](ChannelBackend& backend,void* stream,void* expected){
                EXPECT_EQ(stream,expected);EXPECT_FALSE(backend.registered.empty());
                EXPECT_EQ(remaining(*authority,DeviceId::cpu()),0u);
                EXPECT_EQ(remaining(*authority,DeviceId::cuda(0)),0u);
                EXPECT_EQ(remaining(*authority,DeviceId::rocm(0)),0u);
            };
            cuda.observe_cold_query=[&](void* stream,GPUStreamExecutionState){assertLive(cuda,stream,producer_stream);};
            rocm.observe_cold_query=[&](void* stream,GPUStreamExecutionState){assertLive(rocm,stream,consumer_stream);};
            auto channel=transfer.createCapturedTransferChannel(*authority,DeviceId::cpu(),DeviceId::cuda(0),producer_stream,DeviceId::rocm(0),consumer_stream,256);
            EXPECT_EQ(cuda.cold_queries,3u);EXPECT_EQ(rocm.cold_queries,3u);
            EXPECT_TRUE(cuda.initialization_complete);EXPECT_TRUE(rocm.initialization_complete);
            EXPECT_EQ(cuda.event_creators,0u);EXPECT_EQ(rocm.event_creators,0u);
            cuda.observe_cold_query={};rocm.observe_cold_query={};channel.reset();
            EXPECT_TRUE(cuda.registered.empty());EXPECT_TRUE(rocm.registered.empty());
            EXPECT_EQ(remaining(*authority,DeviceId::cpu()),CapturedTransferChannel::memoryFor(256).mapped_host_bytes);
            EXPECT_EQ(remaining(*authority,DeviceId::cuda(0)),sizeof(CapturedTransferCursor));
            EXPECT_EQ(remaining(*authority,DeviceId::rocm(0)),sizeof(CapturedTransferCursor));
        }
    }

    /** @test Consumer setup failure must drain already queued producer work before any physical release. */
    TEST_F(CapturedTransferChannelBindingTest, ColdInitializationPartialFailureDrainsPendingProducerBeforeCleanup)
    {
        auto authority=admission(DeviceId::cuda(0),DeviceId::rocm(0),256);
        cuda.track_initialization_lifetime=rocm.track_initialization_lifetime=true;
        cuda.cold_states={GPUStreamExecutionState::Pending,GPUStreamExecutionState::Pending,GPUStreamExecutionState::Complete};
        rocm.fail_initialization=true;
        cuda.observe_cold_query=[&](void* stream,GPUStreamExecutionState){
            EXPECT_EQ(stream,producer_stream);EXPECT_EQ(cuda.unregistrations,0);EXPECT_EQ(rocm.unregistrations,0);
            EXPECT_EQ(remaining(*authority,DeviceId::cpu()),0u);
            EXPECT_EQ(remaining(*authority,DeviceId::cuda(0)),0u);
            EXPECT_EQ(remaining(*authority,DeviceId::rocm(0)),0u);
        };
        EXPECT_THROW(transfer.createCapturedTransferChannel(*authority,DeviceId::cpu(),DeviceId::cuda(0),producer_stream,DeviceId::rocm(0),consumer_stream,256),std::runtime_error);
        cuda.observe_cold_query={};
        EXPECT_EQ(cuda.cold_queries,3u);EXPECT_EQ(rocm.cold_queries,1u);
        EXPECT_TRUE(cuda.initialization_complete);EXPECT_TRUE(rocm.initialization_complete);
        EXPECT_EQ(cuda.registrations,cuda.unregistrations);EXPECT_EQ(rocm.registrations,rocm.unregistrations);
        EXPECT_EQ(remaining(*authority,DeviceId::cpu()),CapturedTransferChannel::memoryFor(256).mapped_host_bytes);
        EXPECT_EQ(remaining(*authority,DeviceId::cuda(0)),sizeof(CapturedTransferCursor));
        EXPECT_EQ(remaining(*authority,DeviceId::rocm(0)),sizeof(CapturedTransferCursor));
    }

    /** @test Unprovable immediate native-query failure terminates before any pending owner frees. */
    TEST_F(CapturedTransferChannelBindingTest, ColdInitializationQueryFailureIsFatalBeforeUnprovableCleanup)
    {
        GTEST_FLAG_SET(death_test_style,"threadsafe");
        EXPECT_DEATH({
            auto authority=admission(DeviceId::cuda(0),DeviceId::rocm(0),256);
            cuda.track_initialization_lifetime=rocm.track_initialization_lifetime=true;
            cuda.throw_cold_query=true;cuda.forbid_early_retirement=true;rocm.forbid_early_retirement=true;
            (void)transfer.createCapturedTransferChannel(*authority,DeviceId::cpu(),DeviceId::cuda(0),producer_stream,DeviceId::rocm(0),consumer_stream,256);
        },"recorded cold query failure");
    }

    TEST_F(CapturedTransferChannelBindingTest, ColdInitializationRequiresExistingOriginalWorkersBeforePhysicalCreation)
    {
        GPUDeviceContextPool::instance().shutdown();
        auto authority=admission(DeviceId::cuda(0),DeviceId::rocm(0),256);
        EXPECT_THROW(transfer.createCapturedTransferChannel(*authority,DeviceId::cpu(),DeviceId::cuda(0),producer_stream,DeviceId::rocm(0),consumer_stream,256),std::logic_error);
        EXPECT_EQ(cuda.registrations,0);EXPECT_EQ(rocm.registrations,0);
        EXPECT_EQ(cuda.event_creators,0u);EXPECT_EQ(rocm.event_creators,0u);
        EXPECT_EQ(remaining(*authority,DeviceId::cpu()),CapturedTransferChannel::memoryFor(256).mapped_host_bytes);
    }

    TEST_F(CapturedTransferChannelBindingTest, ColdInitializationUsesOriginalStreamsWithoutEventsAndRejectsRuntimeDrift)
    {
        auto authority=admission(DeviceId::cuda(0),DeviceId::rocm(0),256);
        auto channel=transfer.createCapturedTransferChannel(*authority,DeviceId::cpu(),DeviceId::cuda(0),producer_stream,DeviceId::rocm(0),consumer_stream,256);
        EXPECT_EQ(cuda.event_creators,0u);EXPECT_EQ(rocm.event_creators,0u);
        EXPECT_EQ(cuda.initialization_streams,(std::vector<void*>{producer_stream,producer_stream}));
        EXPECT_EQ(rocm.initialization_streams,(std::vector<void*>{consumer_stream,consumer_stream}));
        auto source=transfer.allocateDeviceTransferBuffer(256,DeviceId::cuda(0));
        const auto binding=transfer.bindCapturedTransfer(channel,CapturedTransferEndpoint::Producer,{1,32},source);
        const auto identity=binding.captureIdentity();const auto queries=cuda.setup_queries+rocm.setup_queries;
        for(unsigned replay=0;replay<20;++replay)EXPECT_EQ(binding.captureIdentity(),identity);
        EXPECT_EQ(cuda.setup_queries+rocm.setup_queries,queries);
        ++cuda.runtime_generation;
        EXPECT_THROW(binding.captureIdentity(),std::logic_error);
        EXPECT_THROW(transfer.enqueueCapturedTransfer(binding,producer_stream),std::logic_error);
        EXPECT_TRUE(cuda.calls.empty());--cuda.runtime_generation;
    }

    TEST_F(CapturedTransferChannelBindingTest, PartialSetupRetiresOriginalStreamsBuffersRegistrationsAndClaims)
    {
        auto authority = admission(DeviceId::cuda(0), DeviceId::rocm(0), 256);
        rocm.fail_initialization = true;
        EXPECT_THROW((void)transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), DeviceId::cuda(0),
            producer_stream, DeviceId::rocm(0), consumer_stream, 256), std::runtime_error);
        EXPECT_EQ(cuda.registrations, cuda.unregistrations);
        EXPECT_EQ(rocm.registrations, rocm.unregistrations);
        EXPECT_EQ(remaining(*authority, DeviceId::cuda(0)), sizeof(CapturedTransferCursor));
        EXPECT_EQ(remaining(*authority, DeviceId::rocm(0)), sizeof(CapturedTransferCursor));
        EXPECT_EQ(remaining(*authority, DeviceId::cpu()), CapturedTransferChannel::memoryFor(256).mapped_host_bytes);
    }

    TEST_F(CapturedTransferChannelBindingTest, RejectsWrongEndpointBoundsNullStreamsAndNativeFailure)
    {
        auto authority = admission(DeviceId::cuda(0), DeviceId::rocm(0), 256);
        EXPECT_THROW((void)transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), DeviceId::cuda(0),
            nullptr, DeviceId::rocm(0), consumer_stream, 256), std::invalid_argument);
        EXPECT_THROW((void)transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), DeviceId::cuda(0),
            producer_stream, DeviceId::cuda(0), consumer_stream, 256), std::invalid_argument);
        auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), DeviceId::cuda(0),
            producer_stream, DeviceId::rocm(0), consumer_stream, 256);
        auto source = transfer.allocateDeviceTransferBuffer(256, DeviceId::cuda(0));
        EXPECT_THROW((void)transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Consumer, {1, 256}, source), std::invalid_argument);
        EXPECT_THROW((void)transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer, {0, 256}, source), std::invalid_argument);
        EXPECT_THROW((void)transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer, {1, 256}, source, 1), std::out_of_range);
        auto binding = transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer, {1, 256}, source);
        EXPECT_THROW(transfer.enqueueCapturedTransfer(binding, nullptr), std::invalid_argument);
        EXPECT_TRUE(cuda.calls.empty());
        cuda.fail_boundary = true;
        EXPECT_THROW(transfer.enqueueCapturedTransfer(binding, producer_stream), std::runtime_error);
        ASSERT_EQ(cuda.calls.size(), 1u);
        EXPECT_EQ(cuda.calls.back().operation, "acquire");
    }

    TEST_F(CapturedTransferChannelBindingTest, TensorBindingsRequireResidentCanonicalStorage)
    {
        auto authority = admission(DeviceId::cuda(0), DeviceId::rocm(0), 64);
        auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), DeviceId::cuda(0),
            producer_stream, DeviceId::rocm(0), consumer_stream, 64);
        auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{16});
        tensor->setBackendForTesting(&cuda);
        EXPECT_THROW((void)transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer, {1, 64}, tensor), std::invalid_argument);
        TransferEngine::prepareDeviceInput(tensor.get(), DeviceId::cuda(0), producer_stream);
        auto binding = transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer, {1, 64}, tensor);
        EXPECT_NO_THROW(transfer.enqueueCapturedTransfer(binding, producer_stream));
        // Binding uses the physical extent, not a convenient padded/capacity
        // assumption inherited from a different tensor or message geometry.
        EXPECT_THROW((void)transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer, {1, 64}, tensor, 4), std::out_of_range);
        EXPECT_THROW((void)transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Consumer, {1, 64}, tensor), std::invalid_argument);
    }

    /** @test Both vendor orders and TP widths retain native broadcast after leader-only transfer. */
    TEST_F(CapturedTransferChannelBindingTest, PipelineDomainsReceiveThenBroadcastExactPhysicalRows)
    {
        for (const bool reverse : {false, true})
        for (const int width : {1, 2, 4, 8})
        for (const size_t elements : {1u, 17u, 64u})
        {
            SCOPED_TRACE(::testing::Message() << "reverse=" << reverse << " width=" << width << " elements=" << elements);
            const auto producer = reverse ? DeviceId::rocm(1) : DeviceId::cuda(1);
            const auto consumer = reverse ? DeviceId::cuda(5) : DeviceId::rocm(5);
            auto authority = admission(producer, consumer, 64 * sizeof(float));
            auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), producer,
                producer_stream, consumer, consumer_stream, 64 * sizeof(float));
            EXPECT_EQ(channel->endpointDevice(CapturedTransferEndpoint::Producer), producer);
            EXPECT_EQ(channel->endpointDevice(CapturedTransferEndpoint::Consumer), consumer);
            EXPECT_THROW(channel->endpointDevice(static_cast<CapturedTransferEndpoint>(99)), std::invalid_argument);
            DomainCollective domain;
            constexpr std::array ordinals{5, 1, 7, 2, 6, 3, 4, 0};
            std::vector<GlobalDeviceAddress> members;
            for (int member = 0; member < width; ++member)
                members.push_back(GlobalDeviceAddress::fromLocalDeviceId(reverse
                    ? DeviceId::cuda(ordinals[member]) : DeviceId::rocm(ordinals[member])));
            domain.setDevices(members);
            domain.setBackend(reverse ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
            auto *destination_backend = reverse ? &cuda : &rocm;
            domain.calls = &destination_backend->calls;
            const auto bank = [&](DeviceId device, float initial) {
                auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{128});
                tensor->setBackendForTesting(resolve(device));
                TransferEngine::prepareDeviceInput(tensor.get(), device, consumer_stream);
                std::fill_n(static_cast<float *>(tensor->gpu_data_ptr()), 128, initial);
                return tensor;
            };
            auto source = bank(producer, 0);
            for (size_t i = 0; i < 128; ++i) static_cast<float *>(source->gpu_data_ptr())[i] = float(1000 + i);
            std::vector<std::shared_ptr<FP32Tensor>> destinations;
            for (int member = 0; member < width; ++member)
                destinations.push_back(bank(members[member].toLocalDeviceId(), -91));
            using Stage = PipelineActivationExchange;
            Stage send(producer, source, {.channel = channel, .message = {27, elements * sizeof(float)}}, transfer);
            llaminar2::testing::MockDeviceContext source_context(producer,
                reverse ? ComputeBackendType::GPU_ROCM : ComputeBackendType::GPU_CUDA);
            send.setGPUStream(producer_stream);
            cuda.calls.clear(); rocm.calls.clear();
            ASSERT_TRUE(send.execute(&source_context));
            for (int member = 0; member < width; ++member)
            {
                const auto device = members[member].toLocalDeviceId();
                Stage receive(device, destinations[member], {.context = width > 1 ? &domain : nullptr,
                    .participant = member, .channel = channel, .message = {27, elements * sizeof(float)},
                    .endpoint = CapturedTransferEndpoint::Consumer}, transfer);
                EXPECT_EQ(receive.type(), ComputeStageType::PIPELINE_ACTIVATION_EXCHANGE);
                EXPECT_TRUE(receive.isGraphCapturable());
                EXPECT_EQ(receive.estimatedMemoryBytes(), elements * sizeof(float));
                receive.setGPUStream(consumer_stream);
                llaminar2::testing::MockDeviceContext destination_context(device,
                    reverse ? ComputeBackendType::GPU_CUDA : ComputeBackendType::GPU_ROCM);
                destination_backend->calls.clear();
                ASSERT_TRUE(receive.execute(&destination_context));
                const auto &calls = destination_backend->calls;
                ASSERT_EQ(calls.size(), (member == 0 ? 3u : 0u) + (width > 1 ? 1u : 0u));
                if (member == 0)
                {
                    EXPECT_EQ(calls[0].operation, "acquire");
                    EXPECT_EQ(calls[1].operation, "copy");
                    EXPECT_EQ(calls[2].operation, "publish");
                }
                if (width > 1) EXPECT_EQ(calls.back().operation, "broadcast");
                for (const auto &call : calls) EXPECT_EQ(call.stream, consumer_stream);
                for (size_t i = 0; i < 128; ++i)
                    EXPECT_EQ(static_cast<const float *>(destinations[member]->gpu_data_ptr())[i], i < elements ? float(1000 + i) : -91);
                EXPECT_EQ(destination_backend->getSyncCount(), 0u);
                EXPECT_EQ(destination_backend->getStreamSyncCount(), 0u);
            }
        }
    }

    /** @test A TP nonleader cannot publish; wrong membership or native transport fails before enqueue. */
    TEST_F(CapturedTransferChannelBindingTest, PipelineDomainsRejectInvalidOwnershipAndMembershipDrift)
    {
        const auto producer = DeviceId::cuda(0), consumer = DeviceId::rocm(0);
        auto authority = admission(producer, consumer, 64);
        auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), producer,
            producer_stream, consumer, consumer_stream, 64);
        auto hidden = std::make_shared<FP32Tensor>(std::vector<size_t>{32});
        hidden->setBackendForTesting(&rocm);
        TransferEngine::prepareDeviceInput(hidden.get(), consumer, consumer_stream);
        DomainCollective group;
        group.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        group.setBackend(CollectiveBackendType::RCCL);
        group.calls = &rocm.calls;
        using Stage = PipelineActivationExchange;
        const Stage::CapturedDomain valid{.context = &group, .channel = channel, .message = {13, 64},
            .endpoint = CapturedTransferEndpoint::Consumer};
        Stage stage(consumer, hidden, valid, transfer);
        const auto reject = [&](auto change) {
            auto invalid = valid; change(invalid);
            EXPECT_THROW((Stage(consumer, hidden, invalid, transfer)), std::invalid_argument);
        };
        reject([](auto &p) { p.channel.reset(); });
        reject([](auto &p) { p.message.bytes = 63; });
        reject([](auto &p) { p.message.key = 0; });
        reject([](auto &p) { p.participant = 1; });
        reject([](auto &p) { p.context = nullptr; p.participant = 1; });
        reject([](auto &p) { p.endpoint = CapturedTransferEndpoint::Producer; });
        reject([](auto &p) { p.endpoint = static_cast<CapturedTransferEndpoint>(42); });
        stage.setGPUStream(consumer_stream);
        llaminar2::testing::MockDeviceContext context(consumer, ComputeBackendType::GPU_ROCM);
        group.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(2)});
        rocm.calls.clear();
        EXPECT_THROW(stage.execute(&context), std::logic_error);
        EXPECT_TRUE(rocm.calls.empty());
        group.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        group.setBackend(CollectiveBackendType::HOST);
        EXPECT_THROW(stage.execute(&context), std::invalid_argument);
        EXPECT_TRUE(rocm.calls.empty());
        group.setBackend(CollectiveBackendType::RCCL);
        group.fail = true;
        EXPECT_FALSE(stage.execute(&context));
        ASSERT_EQ(rocm.calls.size(), 4u);
        EXPECT_EQ(rocm.calls.back().operation, "broadcast");
    }

    /** @test Every graph role encloses local compute, independent of TP width or vendor order. */
    TEST_F(CapturedTransferChannelBindingTest, PipelineGraphDomainsEncloseEveryForwardRole)
    {
        using Edges = PipelineForwardGraphEdges;
        enum class Role { Prefill, Ordinary, AdmittedOrdinary, Condition, Verifier2, Verifier16 };
        MPIContext mpi(0, 1);
        TensorFactory factory(mpi);
        for (const bool reverse : {false, true})
        for (const int width : {1, 2, 4, 8})
        {
            const auto head = reverse ? DeviceId::rocm(1) : DeviceId::cuda(1);
            const auto tail = reverse ? DeviceId::cuda(1) : DeviceId::rocm(1);
            constexpr size_t capacity = 16 * 32 * sizeof(float);
            auto authority = admission(head, tail, capacity, 0, 2);
            auto activation = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), head,
                producer_stream, tail, consumer_stream, capacity);
            auto metadata = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), tail,
                consumer_stream, head, producer_stream, capacity);
            for (size_t domain = 0; domain < 2; ++domain)
            {
                llaminar2::test::MockLocalTPContext context;
                std::vector<GlobalDeviceAddress> members;
                for (int member = 0; member < width; ++member)
                    members.push_back(GlobalDeviceAddress::fromLocalDeviceId(DeviceId{
                        (domain ? tail : head).type, (member + 1) % 8}));
                context.setDevices(members);
                context.setBackend(members.front().toLocalDeviceId().is_cuda()
                    ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
                for (int member = 0; member < width; ++member)
                {
                    const auto device = members[member].toLocalDeviceId();
                    BufferArena arena(::llaminar2::test::cpuArenaFixtureConfig(&factory));
                    ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 16, 32, "FP32", DeviceId::cpu()));
                    for (const auto id : {BufferId::REQUEST_TOKEN_IDS, BufferId::MTP_LOGICAL_SEQUENCE_STATE, BufferId::MTP_VERIFIER_INPUT_TOKENS,
                             BufferId::MTP_VERIFIER_POSITION_IDS, BufferId::MTP_VERIFIER_REQUEST_LENGTHS})
                        ASSERT_TRUE(arena.registerBuffer(id, 1, 32, "INT32", DeviceId::cpu()));
                    ASSERT_TRUE(arena.allocate());
                    for (const auto id : {BufferId::HIDDEN_STATE, BufferId::REQUEST_TOKEN_IDS, BufferId::MTP_LOGICAL_SEQUENCE_STATE,
                             BufferId::MTP_VERIFIER_INPUT_TOKENS, BufferId::MTP_VERIFIER_POSITION_IDS,
                             BufferId::MTP_VERIFIER_REQUEST_LENGTHS})
                    {
                        auto tensor = arena.getSharedTensor(id);
                        tensor->setBackendForTesting(resolve(device));
                        TransferEngine::prepareDeviceInput(tensor.get(), device, producer_stream);
                    }
                    auto *logical = static_cast<int32_t *>(arena.getSharedTensor(BufferId::MTP_LOGICAL_SEQUENCE_STATE)->gpu_data_ptr());
                    Edges::FollowerState follower{.backend = resolve(device), .checkpoint = {
                        .cache = reinterpret_cast<IKVCache *>(logical), .sequence_index = 0,
                        .checkpoint_device = logical + 8, .checkpoint_bytes = 2 * sizeof(int32_t)}};
                    Edges::CapturedDomain port{.stage_index = domain, .stage_count = 2,
                        .context = width > 1 ? &context : nullptr, .participant = member,
                        .activation_in = domain ? activation : nullptr,
                        .activation_out = domain ? nullptr : activation,
                        .metadata_in = domain ? nullptr : metadata,
                        .metadata_out = domain ? metadata : nullptr};
                    Edges edges(device, port, arena, 32, domain ? std::nullopt : std::optional(follower), transfer);
                    for (const auto role : {Role::Prefill, Role::Ordinary, Role::AdmittedOrdinary, Role::Condition, Role::Verifier2, Role::Verifier16})
                    {
                        SCOPED_TRACE(::testing::Message() << "reverse=" << reverse << " width=" << width
                            << " domain=" << domain << " member=" << member << " role=" << int(role));
                        const bool verifier = role == Role::Verifier2 || role == Role::Verifier16;
                        ForwardInput input;
                        input.device = device;
                        input.seq_len = role == Role::Prefill || role == Role::Verifier16 ? 16 : role == Role::Verifier2 ? 2 : 1;
                        input.execution_phase = role == Role::Prefill ? ForwardExecutionPhase::Prefill : ForwardExecutionPhase::Decode;
                        input.execution_role = verifier ? ForwardExecutionRole::GroupedMTPVerifier :
                            role == Role::Condition ? ForwardExecutionRole::MTPCondition : ForwardExecutionRole::MainInference;
                        input.external_hidden_state = domain ? arena.getSharedTensor(BufferId::HIDDEN_STATE).get() : nullptr;
                        input.kv_cache = domain ? nullptr : follower.checkpoint.cache;
                        input.token_ids_device = verifier ? arena.getSharedTensor(BufferId::MTP_VERIFIER_INPUT_TOKENS)->gpu_data_ptr() : logical;
                        input.position_ids_device = verifier ? arena.getSharedTensor(BufferId::MTP_VERIFIER_POSITION_IDS)->gpu_data_ptr() : logical + 1;
                        input.sequence_lengths_device = verifier ? static_cast<const int32_t *>(
                            arena.getSharedTensor(BufferId::MTP_VERIFIER_REQUEST_LENGTHS)->gpu_data_ptr()) : logical + 2;
                        if (role == Role::AdmittedOrdinary)
                            input.token_ids_device = arena.getSharedTensor(BufferId::REQUEST_TOKEN_IDS)->gpu_data_ptr();
                        if (role == Role::Ordinary || role == Role::AdmittedOrdinary)
                            input.device_decode_position = DeviceDecodePositionBinding{resolve(device), logical + 2, logical + 1};
                        if (role == Role::Prefill) input.device_prefill_chunk.emplace();
                        EXPECT_TRUE(edges.encloses(input));
                        ComputeGraph graph;
                        graph.addNode("model", std::make_unique<llaminar2::testing::MockComputeStage>(), device);
                        ASSERT_NO_THROW(edges.append(graph, input));
                        // Native TP capture must rendezvous on a common logical
                        // forward even when only the leader has an egress node.
                        const auto &wave = graph.getNode("model")->graph_capture_wave;
                        ASSERT_TRUE(wave);
                        EXPECT_EQ(wave->participation, GraphCaptureWaveParticipation::Active);
                        EXPECT_EQ(wave->identity, "pipeline_domain_" + std::to_string(domain) +
                            ":role=" + std::to_string(static_cast<int>(input.execution_role)) +
                            ":phase=" + std::to_string(static_cast<int>(input.execution_phase)) +
                            ":state=" + std::to_string(static_cast<int>(input.state_transaction)) +
                            ":rows=" + std::to_string(input.seq_len));
                        EXPECT_EQ(graph.getNode("pipeline_activation_send") != nullptr, !domain && !member);
                        EXPECT_EQ(graph.getNode("pipeline_activation_receive") != nullptr, domain != 0);
                        EXPECT_EQ(graph.getNode("pipeline_verifier_checkpoint") != nullptr, !domain && verifier);
                        EXPECT_EQ(graph.getNode("pipeline_mtp_rows") != nullptr, role != Role::Prefill);
                        if (role != Role::Prefill)
                        {
                            EXPECT_EQ(graph.getRootNodes(), std::vector<std::string>{"pipeline_mtp_rows"});
                            EXPECT_EQ(graph.getNode("pipeline_mtp_rows")->stage->estimatedMemoryBytes(),
                                sizeof(int32_t) * (role == Role::Ordinary || role == Role::AdmittedOrdinary ? 1 : 2 * input.seq_len + 1));
                        }
                        // Exact row binding is checked before any graph mutation.
                        if (role == Role::Condition)
                        {
                            input.token_ids_device = static_cast<int32_t *>(arena.getSharedTensor(BufferId::MTP_VERIFIER_INPUT_TOKENS)->gpu_data_ptr()) + 1;
                            ComputeGraph invalid;
                            invalid.addNode("model", std::make_unique<llaminar2::testing::MockComputeStage>(), device);
                            EXPECT_ANY_THROW(edges.append(invalid, input));
                            EXPECT_EQ(invalid.getRootNodes(), std::vector<std::string>{"model"});
                            EXPECT_EQ(invalid.getLeafNodes(), std::vector<std::string>{"model"});
                        }
                    }
                    if (!domain && member)
                    {
                        auto invalid = port;
                        invalid.participant = 0; // Previously escaped validation because this member sends nothing.
                        EXPECT_THROW((Edges(device, invalid, arena, 32, follower, transfer)), std::invalid_argument);
                    }
                }
            }
        }
    }

    /** @test Native request admission is a NextToken-only owner, independent of mailbox addresses or vendor order. */
    TEST_F(CapturedTransferChannelBindingTest, PipelineGraphDomainsAdmittedDecodeUsesCanonicalTokenOwner)
    {
        using Edges=PipelineForwardGraphEdges;
        MPIContext mpi(0,1);TensorFactory factory(mpi);
        for(bool reverse:{false,true})for(int width:{1,2,4,8}) {
            const auto head=reverse?DeviceId::rocm(1):DeviceId::cuda(1);
            const auto tail=reverse?DeviceId::cuda(1):DeviceId::rocm(1);
            constexpr std::size_t capacity=16*32*sizeof(float);
            auto authority=admission(head,tail,capacity,0,2);
            auto activation=transfer.createCapturedTransferChannel(*authority,DeviceId::cpu(),head,
                producer_stream,tail,consumer_stream,capacity);
            auto metadata=transfer.createCapturedTransferChannel(*authority,DeviceId::cpu(),tail,
                consumer_stream,head,producer_stream,capacity);
            for(int domain=0;domain<2;++domain) {
                llaminar2::test::MockLocalTPContext context;
                std::vector<GlobalDeviceAddress> members;
                for(int member=0;member<width;++member)members.push_back(GlobalDeviceAddress::fromLocalDeviceId(
                    DeviceId{(domain?tail:head).type,(member+1)%8}));
                context.setDevices(members);context.setBackend(members.front().toLocalDeviceId().is_cuda()
                    ?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL);
                for(int member=0;member<width;++member) {
                    SCOPED_TRACE(::testing::Message()<<"reverse="<<reverse<<" width="<<width<<" domain="<<domain<<" member="<<member);
                    const auto device=members[member].toLocalDeviceId();
                    BufferArena arena(::llaminar2::test::cpuArenaFixtureConfig(&factory));
                    ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE,16,32,"FP32",DeviceId::cpu()));
                    constexpr std::array integers{BufferId::REQUEST_TOKEN_IDS,BufferId::REQUEST_POSITION_IDS,
                        BufferId::MTP_LOGICAL_SEQUENCE_STATE,BufferId::MTP_VERIFIER_INPUT_TOKENS,
                        BufferId::MTP_VERIFIER_POSITION_IDS,BufferId::MTP_VERIFIER_REQUEST_LENGTHS};
                    for(auto id:integers)ASSERT_TRUE(arena.registerBuffer(id,1,32,"INT32",DeviceId::cpu()));
                    ASSERT_TRUE(arena.allocate());
                    for(auto id:integers) {
                        auto tensor=arena.getSharedTensor(id);tensor->setBackendForTesting(resolve(device));
                        TransferEngine::prepareDeviceInput(tensor.get(),device,producer_stream);
                    }
                    auto hidden=arena.getSharedTensor(BufferId::HIDDEN_STATE);hidden->setBackendForTesting(resolve(device));
                    TransferEngine::prepareDeviceInput(hidden.get(),device,producer_stream);
                    const auto words=[&](BufferId id){return static_cast<int32_t *>(arena.getSharedTensor(id)->gpu_data_ptr());};
                    auto *logical=words(BufferId::MTP_LOGICAL_SEQUENCE_STATE);
                    Edges::FollowerState follower{.backend=resolve(device),.checkpoint={
                        .cache=reinterpret_cast<IKVCache *>(logical),.sequence_index=0,
                        .checkpoint_device=logical+8,.checkpoint_bytes=2*sizeof(int32_t)}};
                    Edges::CapturedDomain port{.stage_index=std::size_t(domain),.stage_count=2,
                        .context=width>1?&context:nullptr,.participant=member,
                        .activation_in=domain?activation:nullptr,.activation_out=domain?nullptr:activation,
                        .metadata_in=domain?nullptr:metadata,.metadata_out=domain?metadata:nullptr};
                    Edges edges(device,port,arena,32,domain?std::nullopt:std::optional(follower),transfer);
                    ForwardInput input;input.device=device;input.seq_len=1;
                    input.execution_role=ForwardExecutionRole::MainInference;input.execution_phase=ForwardExecutionPhase::Decode;
                    input.external_hidden_state=domain?hidden.get():nullptr;
                    input.kv_cache=domain?nullptr:follower.checkpoint.cache;
                    input.token_ids_device=words(BufferId::REQUEST_TOKEN_IDS);
                    input.position_ids_device=words(BufferId::REQUEST_POSITION_IDS);
                    input.sequence_lengths_device=logical+2;
                    input.device_decode_position=DeviceDecodePositionBinding{resolve(device),logical+2,
                        words(BufferId::REQUEST_POSITION_IDS)};
                    const auto makeGraph=[&] {
                        auto graph=std::make_unique<ComputeGraph>();
                        graph->addNode("model",std::make_unique<llaminar2::testing::MockComputeStage>(),device);
                        return graph;
                    };
                    const auto proveOwner=[&](const ForwardInput &binding,BufferId expected) {
                        auto graph=makeGraph();ASSERT_NO_THROW(edges.append(*graph,binding));
                        auto *node=graph->getNode("pipeline_mtp_rows");ASSERT_NE(node,nullptr);
                        const auto contract=node->stage->bufferContract();
                        const auto &banks=domain?contract.inputs:contract.outputs;
                        ASSERT_EQ(banks.size(),1u);EXPECT_EQ(banks.front().id,expected);
                        EXPECT_EQ(node->stage->estimatedMemoryBytes(),sizeof(int32_t));
                    };
                    ASSERT_NO_FATAL_FAILURE(proveOwner(input,BufferId::REQUEST_TOKEN_IDS));
                    auto mailbox=input;mailbox.token_ids_device=logical+4;
                    ASSERT_NO_FATAL_FAILURE(proveOwner(mailbox,BufferId::MTP_LOGICAL_SEQUENCE_STATE));
                    auto verifier=input;verifier.execution_role=ForwardExecutionRole::GroupedMTPVerifier;verifier.seq_len=2;
                    verifier.device_decode_position.reset();
                    verifier.token_ids_device=words(BufferId::MTP_VERIFIER_INPUT_TOKENS);
                    verifier.position_ids_device=words(BufferId::MTP_VERIFIER_POSITION_IDS);
                    verifier.sequence_lengths_device=words(BufferId::MTP_VERIFIER_REQUEST_LENGTHS);
                    auto complete=makeGraph();ASSERT_NO_THROW(edges.append(*complete,verifier));
                    const auto reject=[&](ForwardInput invalid) {
                        auto graph=makeGraph();auto &backend=device.is_cuda()?cuda:rocm;backend.calls.clear();
                        EXPECT_THROW(edges.append(*graph,invalid),std::exception);
                        EXPECT_EQ(graph->getRootNodes(),std::vector<std::string>{"model"});
                        EXPECT_EQ(graph->getLeafNodes(),std::vector<std::string>{"model"});
                        EXPECT_EQ(graph->getNode("pipeline_mtp_rows"),nullptr);
                        EXPECT_TRUE(backend.calls.empty());
                    };
                    auto invalid=input;invalid.token_ids_device=words(BufferId::REQUEST_TOKEN_IDS)+1;reject(invalid);
                    invalid=input;invalid.token_ids_device=words(BufferId::REQUEST_POSITION_IDS);reject(invalid);
                    invalid=input;invalid.execution_role=ForwardExecutionRole::MTPCondition;
                    invalid.device_decode_position.reset();invalid.position_ids_device=logical+1;reject(invalid);
                    invalid=verifier;invalid.token_ids_device=words(BufferId::REQUEST_TOKEN_IDS);reject(invalid);
                    invalid=verifier;invalid.token_ids_device=words(BufferId::MTP_VERIFIER_INPUT_TOKENS)+1;reject(invalid);
                }
            }
        }
    }

    /** @test Wire keys disambiguate every role, field and supported verifier row count. */
    TEST(PipelineMetadataLayout, CompleteSchemaHasNoRoleOrGeometryAliases)
    {
        using Layout = PipelineMetadataLayout;
        using Kind = Layout::Kind;
        std::set<uint64_t> keys;
        for (const auto kind : {Kind::Condition, Kind::Verifier, Kind::CommittedState, Kind::NextToken})
        {
            for (int rows = kind == Kind::Verifier ? 2 : 1; rows <= (kind == Kind::Verifier ? 256 : 1); ++rows)
            {
                Layout layout(kind, rows);
                const size_t count = kind == Kind::CommittedState ? 4u : kind == Kind::NextToken ? 1u : 3u;
                EXPECT_EQ(layout.fieldCount(), count);
                for (size_t field = 0; field < count; ++field)
                {
                    EXPECT_TRUE(keys.insert(layout.message(field).key).second);
                    EXPECT_EQ(layout.message(field).bytes,
                        sizeof(int32_t) * (kind == Kind::Verifier && field < 2 ? size_t(rows) : 1u));
                }
                EXPECT_THROW(layout.message(count), std::out_of_range);
            }
        }
        EXPECT_THROW((Layout(Kind::Verifier, 1)), std::invalid_argument);
        EXPECT_THROW((Layout(Kind::Condition, 2)), std::invalid_argument);
        EXPECT_THROW((Layout(Kind::CommittedState, 0)), std::invalid_argument);
        EXPECT_THROW((Layout(Kind::NextToken, -1)), std::invalid_argument);
        EXPECT_THROW((Layout(static_cast<Kind>(99), 1)), std::invalid_argument);
    }

    /** @test Native TP fanout follows leader receipt in both vendor orders and widths 1–8. */
    TEST_F(CapturedTransferChannelBindingTest, PipelineMetadataKeepsNativeFanoutAndExactBanks)
    {
        using Layout = PipelineMetadataLayout;
        using Kind = Layout::Kind;
        for (const bool reverse : {false, true})
        for (const int width : {1, 2, 4, 8})
        for (const auto layout : {Layout(Kind::Condition, 1), Layout(Kind::Verifier, 2),
                 Layout(Kind::Verifier, 4), Layout(Kind::Verifier, 16), Layout(Kind::CommittedState, 1),
                 Layout(Kind::NextToken, 1)})
        {
            const auto producer = reverse ? DeviceId::rocm(1) : DeviceId::cuda(1);
            const auto consumer = reverse ? DeviceId::cuda(5) : DeviceId::rocm(5);
            auto authority = admission(producer, consumer, 64);
            auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), producer,
                producer_stream, consumer, consumer_stream, 64);
            MetadataCollective group;
            constexpr std::array ordinals{5, 1, 7, 2, 6, 3, 4, 0};
            std::vector<GlobalDeviceAddress> members;
            for (int member = 0; member < width; ++member)
                members.push_back(GlobalDeviceAddress::fromLocalDeviceId(reverse
                    ? DeviceId::cuda(ordinals[member]) : DeviceId::rocm(ordinals[member])));
            group.setDevices(members);
            group.setBackend(reverse ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
            group.calls = &(reverse ? cuda : rocm).calls;
            for (int member = 0; member < width; ++member)
            {
                const auto device = members[member].toLocalDeviceId();
                auto owner = std::make_shared<INT32Tensor>(std::vector<size_t>{128});
                owner->setBackendForTesting(resolve(device));
                TransferEngine::prepareDeviceInput(owner.get(), device, consumer_stream);
                std::vector<PipelineMetadataBank> banks;
                for (size_t field = 0; field < layout.fieldCount(); ++field)
                    banks.emplace_back(owner, BufferId::MTP_LOGICAL_SEQUENCE_STATE,
                        (field * 32 + 1) * sizeof(int32_t), layout.elements(field));
                PipelineMetadataExchange receive(device, layout,
                    {.context = width > 1 ? &group : nullptr, .participant = member, .inbound = channel},
                    std::move(banks), transfer);
                receive.setGPUStream(consumer_stream);
                llaminar2::testing::MockDeviceContext context(device,
                    reverse ? ComputeBackendType::GPU_CUDA : ComputeBackendType::GPU_ROCM);
                auto &backend = reverse ? cuda : rocm;
                backend.calls.clear(); backend.native_bindings.clear();
                ASSERT_TRUE(receive.execute(&context));
                const auto count = layout.fieldCount();
                ASSERT_EQ(backend.calls.size(), (member == 0 ? 3 * count : 0) + (width > 1 ? count : 0));
                for (size_t field = 0; member == 0 && field < count; ++field)
                {
                    EXPECT_EQ(backend.calls[3 * field].operation, "acquire");
                    EXPECT_EQ(backend.calls[3 * field + 1].operation, "copy");
                    EXPECT_EQ(backend.calls[3 * field + 2].operation, "publish");
                    EXPECT_EQ(backend.calls[3 * field + 1].bytes, layout.message(field).bytes);
                    EXPECT_EQ(backend.native_bindings[field].message.key, layout.message(field).key);
                }
                for (size_t field = 0; width > 1 && field < count; ++field)
                    EXPECT_EQ(backend.calls[(member == 0 ? 3 * count : 0) + field].operation, "metadata_broadcast");
                for (const auto &call : backend.calls) EXPECT_EQ(call.stream, consumer_stream);
                EXPECT_EQ(backend.getSyncCount(), 0u);
                EXPECT_EQ(backend.getStreamSyncCount(), 0u);
                EXPECT_TRUE(receive.isGraphCapturable());
            }
        }
    }

    /** @test Schema/owner/membership errors are rejected before transport mutates state. */
    TEST_F(CapturedTransferChannelBindingTest, PipelineMetadataRejectsPartialForeignAndChangedBindings)
    {
        using Layout = PipelineMetadataLayout;
        const Layout layout(Layout::Kind::Verifier, 16);
        auto authority = admission(DeviceId::cuda(0), DeviceId::rocm(0), 64);
        auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), DeviceId::cuda(0),
            producer_stream, DeviceId::rocm(0), consumer_stream, 64);
        auto owner = std::make_shared<INT32Tensor>(std::vector<size_t>{128});
        owner->setBackendForTesting(&rocm);
        TransferEngine::prepareDeviceInput(owner.get(), DeviceId::rocm(0), consumer_stream);
        constexpr auto id = BufferId::MTP_LOGICAL_SEQUENCE_STATE;
        EXPECT_THROW((PipelineMetadataBank(owner, id, 1, 1)), std::out_of_range);
        EXPECT_THROW((PipelineMetadataBank(owner, id, 0, 0)), std::out_of_range);
        EXPECT_THROW((PipelineMetadataBank(owner, id, 508, 2)), std::out_of_range);
        EXPECT_THROW((PipelineMetadataBank(owner, id, 0, std::numeric_limits<size_t>::max())), std::out_of_range);
        std::vector<PipelineMetadataBank> banks;
        for (size_t field = 0; field < layout.fieldCount(); ++field)
            banks.emplace_back(owner, id, field * 32 * sizeof(int32_t), layout.elements(field));
        MetadataCollective group;
        group.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        group.setBackend(CollectiveBackendType::RCCL);
        group.calls = &rocm.calls;
        const PipelineMetadataExchange::CapturedDomain domain{.context = &group, .inbound = channel};
        auto incomplete = banks; incomplete.pop_back();
        EXPECT_THROW((PipelineMetadataExchange(DeviceId::rocm(0), layout, domain, incomplete, transfer)), std::invalid_argument);
        EXPECT_THROW((PipelineMetadataExchange(DeviceId::rocm(1), layout, domain, banks, transfer)), std::invalid_argument);
        auto wrong_role = domain; wrong_role.inbound.reset(); wrong_role.outbound = channel;
        EXPECT_THROW((PipelineMetadataExchange(DeviceId::rocm(0), layout, wrong_role, banks, transfer)), std::invalid_argument);
        PipelineMetadataExchange stage(DeviceId::rocm(0), layout, domain, banks, transfer);
        stage.setGPUStream(consumer_stream);
        llaminar2::testing::MockDeviceContext context(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);
        group.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(2)});
        rocm.calls.clear();
        EXPECT_THROW(stage.execute(&context), std::logic_error);
        EXPECT_TRUE(rocm.calls.empty());
        group.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        group.fail = true;
        EXPECT_FALSE(stage.execute(&context));
    }
}
