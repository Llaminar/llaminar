/**
 * @file Test__CapturedTransferChannelBinding.cpp
 * @brief Device-free proofs of captured-channel admission, ownership and submission.
 *
 * Independent backend spies make physical aliases and exact native submission
 * order observable without touching accelerators. These tests complement the
 * protocol state-machine sweep; they do not certify GPU memory visibility.
 */
#include "transfer/CapturedTransferChannel.h"
#include "execution/local_execution/orchestrators/PipelineActivationExchange.h"
#include "execution/local_execution/orchestrators/PipelineMetadataExchange.h"
#include "execution/local_execution/orchestrators/PipelineForwardGraphEdges.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "memory/BufferArena.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
#include "../../mocks/MockBackend.h"
#include "../../mocks/MockComputeStage.h"
#include "../../mocks/MockLocalTPContext.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <vector>

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
        { EXPECT_EQ(registered.erase(ptr), 1u); ++unregistrations; return true; }
        /** @brief Freeze timer geometry during setup, with explicit failure injection. */
        bool prepareCapturedTransferChannelKernels(int, int timeout_ms, std::uint64_t *ticks) override
        { ++preparations; EXPECT_EQ(timeout_ms, 30000); *ticks = 42; return !fail_prepare; }
        /** @brief Record only; protocol execution is independently tested in its pure suite. */
        bool enqueueCapturedTransferChannelBoundary(const CapturedTransferChannelDeviceBinding &binding,
            CapturedTransferBoundaryOperation operation, int, void *stream) override
        {
            EXPECT_TRUE(binding.valid());
            native_bindings.push_back(binding);
            calls.push_back({operation == CapturedTransferBoundaryOperation::Acquire ? "acquire" : "publish",
                stream, static_cast<size_t>(binding.message.bytes)});
            return !fail_boundary;
        }
        /** @brief Copy exact bytes, retaining sentinels outside the requested message. */
        bool copyDeviceVisibleRegionByKernelOnStream(void *dst, const void *src, size_t bytes, int, void *stream) override
        { calls.push_back({"copy", stream, bytes}); std::memcpy(dst, src, bytes); return true; }
        /** @brief Import an event into the exact stream, never wait on the host. */
        bool streamWaitEvent(void *stream, void *event, int device) override
        { calls.push_back({"event", stream, 0}); return MockBackend::streamWaitEvent(stream, event, device); }
        /** @brief Exercise rollback after some of the physical owners have materialized. */
        void *createEvent(int device) override
        { return fail_event ? nullptr : MockBackend::createEvent(device); }
        /** @brief Complete setup only; ordinary enqueue must never query this event again. */
        bool queryEvent(void *, int, bool *ready) override
        { ++setup_queries; *ready = true; return true; }

        std::set<void *> registered;
        int registrations = 0, unregistrations = 0, preparations = 0, setup_queries = 0;
        bool fail_prepare = false, fail_event = false, fail_boundary = false;
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
    class CapturedTransferChannelBindingTest : public ::testing::Test
    {
    protected:
        /** @brief Install only mock authorities, with no backend-manager discovery. */
        void SetUp() override { cuda_ = &cuda; rocm_ = &rocm; }
        /** @brief Clear borrowed test pointers after all local owners have retired. */
        void TearDown() override { cuda_ = nullptr; rocm_ = nullptr; }
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

    TEST_F(CapturedTransferChannelBindingTest, PartialSetupRetiresEventsBuffersRegistrationsAndClaims)
    {
        auto authority = admission(DeviceId::cuda(0), DeviceId::rocm(0), 256);
        rocm.fail_event = true;
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
        enum class Role { Prefill, Ordinary, Condition, Verifier2, Verifier16 };
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
                    BufferArena arena({.factory = &factory});
                    ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 16, 32, "FP32", DeviceId::cpu()));
                    for (const auto id : {BufferId::MTP_LOGICAL_SEQUENCE_STATE, BufferId::MTP_VERIFIER_INPUT_TOKENS,
                             BufferId::MTP_VERIFIER_POSITION_IDS, BufferId::MTP_VERIFIER_REQUEST_LENGTHS})
                        ASSERT_TRUE(arena.registerBuffer(id, 1, 32, "INT32", DeviceId::cpu()));
                    ASSERT_TRUE(arena.allocate());
                    for (const auto id : {BufferId::HIDDEN_STATE, BufferId::MTP_LOGICAL_SEQUENCE_STATE,
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
                    for (const auto role : {Role::Prefill, Role::Ordinary, Role::Condition, Role::Verifier2, Role::Verifier16})
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
                        if (role == Role::Ordinary)
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
                                sizeof(int32_t) * (role == Role::Ordinary ? 1 : 2 * input.seq_len + 1));
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
                    EXPECT_EQ(backend.native_bindings[2 * field].message.key, layout.message(field).key);
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
