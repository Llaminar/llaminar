/**
 * @file PooledReplicatedReaderNativeReplay.inc
 * @brief Replay the production reader metadata kernels around a real captured payload receipt.
 *
 * The existing native channel fixture supplies original contexts, streams, PMA,
 * terminal events and capture scopes. Immutable recorded frames are uploaded
 * only between terminal observations; no inference state is mirrored. CPU ops
 * independently check the captured GPU result. Transport selection consumes the
 * same cached directional inventory as production, with no fallback.
 */
#include "kernels/qwen4exp/PooledImmutableReplicatedReaderKernels.h"

namespace {
namespace reader = llaminar2::qwen4exp::replicated_reader;
namespace reader_state = llaminar2::qwen4exp::adaptive_lookup::pooled;

/** @brief One persistent, explicitly admitted diagnostic operand bank. */
struct alignas(64) ReaderReplayBank
{
    reader_state::LookupFrame frame{};
    reader::Header header{}, reply{}, recorded_ack{};
    std::uint64_t header_bytes = sizeof(reader::Header);
    std::uint64_t published_bytes = 0;
    std::int32_t original_rows = 0;
    int healthy = 0;
    int error = 0;
};
static_assert(std::is_trivially_copyable_v<ReaderReplayBank>);

/** @brief Dispatch the actual compiled production kernel, never a test lowering. */
void launchReader(DeviceId device, reader::Action action, reader::NativeArgs args, void* stream)
{
#ifdef HAVE_CUDA
    if (device.is_cuda()) reader::enqueueCUDAPooledReplicatedReader(action, args, stream);
#endif
#ifdef HAVE_ROCM
    if (device.is_rocm()) reader::enqueueROCmPooledReplicatedReader(action, args, stream);
#endif
}

/**
 * @brief Capture once and replay original metadata, actual live payload and ACK proof.
 * @param source Original source endpoint from this rank's immutable inventory.
 * @param destination Original reader; metadata kernels execute on its exact stream.
 */
void replicatedReaderNativeReplay(DeviceId source, DeviceId destination)
{
    const auto mpi = MPIContextFactory::global();
    const auto inventory = mpi->clusterInventory();
    ASSERT_TRUE(inventory);
    const auto& rank = inventory->ranks.at(mpi->rank());
    const auto coverage = localTPPeerAccessCoverage(rank, {source, destination});
    ASSERT_NE(coverage, PeerAccessCoverage::Partial);
    const auto storage = coverage == PeerAccessCoverage::Complete
        ? CapturedTransferPayloadStorage::PeerDevice : CapturedTransferPayloadStorage::MappedHost;
    ::testing::Test::RecordProperty("peer_access_coverage", peerAccessCoverageName(coverage));
    ::testing::Test::RecordProperty("reader_payload_route",
        storage == CapturedTransferPayloadStorage::PeerDevice ? "peer_device" : "mapped_host");
    constexpr reader::Geometry geometry{1, 2, 256, 7};
    constexpr std::size_t capacity = 256 * 7 * sizeof(float);
    constexpr std::size_t bank_bytes = sizeof(ReaderReplayBank);
    constexpr std::size_t count_bytes = sizeof(std::uint64_t);
    auto& transfer = TransferEngine::instance();
    auto& producer = GPUDeviceContextPool::instance().getContext(source);
    auto& consumer = GPUDeviceContextPool::instance().getContext(destination);
    auto* source_backend = getBackendFor(source);
    auto* reader_backend = getBackendFor(destination);
    ASSERT_TRUE(source_backend);
    ASSERT_TRUE(reader_backend);
    void* source_stream = nullptr;
    void* reader_stream = nullptr;
    producer.submitAndWait([&] { source_stream = producer.getOrCreateAuxiliaryStream("pooled_reader_replay"); });
    consumer.submitAndWait([&] { reader_stream = consumer.getOrCreateAuxiliaryStream("pooled_reader_replay"); });
    const auto physical = CapturedTransferChannel::memoryFor(capacity, storage);
    const auto host_bytes = TransferEngine::mappedHostRegionAllocationBytes(bank_bytes) +
        TransferEngine::mappedHostRegionAllocationBytes(capacity + count_bytes) +
        TransferEngine::mappedHostRegionAllocationBytes(capacity + bank_bytes + count_bytes);
    PhysicalMemoryPlanBuilder plan;
    plan.add(planningObservedResource(rank, DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
        physical.mapped_host_bytes + host_bytes);
    plan.add(planningObservedResource(rank, source), PhysicalMemoryOwner::ActivationTransportStaging,
        physical.cursor_bytes_per_device + physical.producer_payload_bytes + capacity + count_bytes);
    plan.add(planningObservedResource(rank, destination), PhysicalMemoryOwner::ActivationTransportStaging,
        physical.cursor_bytes_per_device + capacity + bank_bytes + count_bytes);
    for (const auto device : {source, destination})
        plan.add(planningObservedResource(rank, device), PhysicalMemoryOwner::NativeGraphExecutable,
            (device == source ? 1 : 2) * GPUGraphMemoryContract::reservationBytesPerExecutable(
                device, GPUGraphExecutableClass::BoundedFlatHelper));
    auto memory = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()), rank.rank);
    {
        auto channel = storage == CapturedTransferPayloadStorage::PeerDevice
            ? transfer.createPeerCapturedTransferChannel(*memory, DeviceId::cpu(), source, source_stream,
                destination, reader_stream, capacity, rank)
            : transfer.createCapturedTransferChannel(*memory, DeviceId::cpu(), source, source_stream,
                destination, reader_stream, capacity);
        auto host_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging, host_bytes);
        auto source_claim = memory->claimNewAllocation(source, PhysicalMemoryOwner::ActivationTransportStaging, capacity + count_bytes);
        auto reader_claim = memory->claimNewAllocation(destination, PhysicalMemoryOwner::ActivationTransportStaging, capacity + bank_bytes + count_bytes);
        const std::array devices{source, destination};
        auto host_bank = transfer.allocateMappedHostRegion(bank_bytes, devices);
        auto input = transfer.allocateMappedHostRegion(capacity + count_bytes, devices);
        auto snapshot = transfer.allocateMappedHostRegion(capacity + bank_bytes + count_bytes, devices);
        auto source_payload = transfer.allocateDeviceTransferBuffer(capacity, source);
        auto source_count = transfer.allocateDeviceTransferBuffer(count_bytes, source);
        auto output = transfer.allocateDeviceTransferBuffer(capacity, destination);
        auto received_count = transfer.allocateDeviceTransferBuffer(count_bytes, destination);
        auto operands = transfer.allocateDeviceTransferBuffer(bank_bytes, destination);
        auto* bank = static_cast<ReaderReplayBank*>(operands->mutableDeviceData());
        auto send = transfer.bindDeviceCountedTransfer(
            transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer, {991, capacity}, source_payload), source_count);
        auto receive = transfer.bindDeviceCountedTransfer(
            transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Consumer, {991, capacity}, output), received_count);
        reader::NativeArgs args{destination, geometry, 1, 11,
            DeviceRowRange::deviceCounted(256, &bank->original_rows), &bank->frame, &bank->header,
            &bank->header, &bank->header_bytes, &bank->published_bytes, &bank->healthy, &bank->error,
            CapturedTransferReceiveCompletion::notRequired()};
        consumer.submitAndWait([&] { args.payload_completion = transfer.capturedReceiveCompletion(receive, reader_stream); });
        auto source_graph_claim = memory->reserveNewAllocations(source, PhysicalMemoryOwner::NativeGraphExecutable,
            GPUGraphMemoryContract::reservationBytesPerExecutable(source, GPUGraphExecutableClass::BoundedFlatHelper));
        auto reader_graph_claim = memory->reserveNewAllocations(destination, PhysicalMemoryOwner::NativeGraphExecutable,
            2 * GPUGraphMemoryContract::reservationBytesPerExecutable(destination, GPUGraphExecutableClass::BoundedFlatHelper));
        std::unique_ptr<IGPUGraphCapture> source_graph, reader_graph, confirm_graph;
        Terminal source_done{source_backend, source, nullptr};
        Terminal reader_done{reader_backend, destination, nullptr};
        producer.submitAndWait([&] { source_done.event = source_backend->createEvent(source.ordinal); });
        consumer.submitAndWait([&] { reader_done.event = reader_backend->createEvent(destination.ordinal); });
        ASSERT_TRUE(source_done.event);
        ASSERT_TRUE(reader_done.event);
        const CapturedTransferInvocation original{71, 91};
        transfer.beginCapturedTransferInvocation(channel, original);
        // On every exceptional exit, cancel before joining both original native
        // streams. The graphs and physical owners outlive this cleanup scope.
        std::optional<CancelledChannelRetirement> unwind(std::in_place, transfer, channel, original, producer, consumer,
            source_stream, reader_stream, source_done, reader_done);
        const auto validate_helper = [](const IGPUGraphCapture& graph) {
            std::string error;
            const bool valid = graph.validateExecutableMemoryClass(GPUGraphExecutableClass::BoundedFlatHelper, &error);
            require(valid, error.c_str());
        };
        producer.submitAndWait([&] {
            source_graph = producer.createGraphCapture(source_stream);
            ScopedBackendGraphCapture capture(producer, *source_graph, "pooled reader source");
            require(capture.begin(), "Reader source capture failed");
            transfer.enqueueCapturedTransfer(send, source_stream);
            capture.finish();
            validate_helper(*source_graph);
            require(source_graph->instantiate(), "Reader source instantiate failed");
        });
        consumer.submitAndWait([&] {
            reader_graph = consumer.createGraphCapture(reader_stream);
            ScopedBackendGraphCapture capture(consumer, *reader_graph, "pooled reader metadata");
            require(capture.begin(), "Reader metadata capture failed");
            transfer.enqueueCapturedTransfer(receive, reader_stream);
            launchReader(destination, reader::Action::PrepareResultHeader, args, reader_stream);
            auto accept = args;
            accept.publication = &bank->reply;
            launchReader(destination, reader::Action::AcceptResultHeader, accept, reader_stream);
            launchReader(destination, reader::Action::PrepareResult, args, reader_stream);
            auto ack = accept;
            ack.received_bytes = static_cast<const std::uint64_t*>(received_count->deviceData());
            launchReader(destination, reader::Action::AcknowledgeResult, ack, reader_stream);
            auto confirm = args;
            // ConfirmAck consumes the returned fixed ACK header, not the payload
            // receive receipt already authenticated by AcknowledgeResult. Keep
            // the original malformed operand as a no-launch rejection control.
            EXPECT_THROW(launchReader(destination, reader::Action::ConfirmAck,
                confirm, reader_stream), std::invalid_argument);
            confirm.payload_completion = CapturedTransferReceiveCompletion::notRequired();
            confirm.received = &bank->reply;
            launchReader(destination, reader::Action::ConfirmAck, confirm, reader_stream);
            capture.finish();
            validate_helper(*reader_graph);
            require(reader_graph->instantiate(), "Reader metadata instantiate failed");
            confirm_graph = consumer.createGraphCapture(reader_stream);
            ScopedBackendGraphCapture confirm_capture(consumer, *confirm_graph, "pooled reader recorded ACK");
            require(confirm_capture.begin(), "Reader confirm capture failed");
            confirm.received = &bank->recorded_ack;
            launchReader(destination, reader::Action::ConfirmAck, confirm, reader_stream);
            confirm_capture.finish();
            validate_helper(*confirm_graph);
            require(confirm_graph->instantiate(), "Reader confirm instantiate failed");
        });
        /** Upload a recorded operand bank only after the preceding terminal. */
        const auto upload = [&](const ReaderReplayBank& recorded, std::uint64_t bytes, bool payload = true) {
            std::memcpy(host_bank->mutableHostData(), &recorded, bank_bytes);
            auto* input_bytes = static_cast<unsigned char*>(input->mutableHostData());
            for (std::size_t index = 0; index < capacity; ++index) input_bytes[index] = patternByte(index, recorded.frame.interval.sequence % 2);
            std::memcpy(input_bytes + capacity, &bytes, count_bytes);
            if (payload) producer.submitAndWait([&] {
                transfer.enqueueMappedHostToDevice(*input, 0, *source_payload, 0, capacity, source, source_stream);
                transfer.enqueueMappedHostToDevice(*input, capacity, *source_count, 0, count_bytes, source, source_stream);
            });
            consumer.submitAndWait([&] {
                transfer.enqueueMappedHostToDevice(*host_bank, 0, *operands, 0, bank_bytes, destination, reader_stream);
                require(reader_backend->memset(output->mutableDeviceData(), 0xa5, capacity, destination.ordinal, reader_stream),
                    "Reader output poison failed");
            });
        };
        /** Read diagnostic bytes only after the complete original consumer graph. */
        const auto observe = [&] {
            consumer.submitAndWait([&] {
                transfer.enqueueDeviceToMappedHost(*operands, 0, *snapshot, 0, bank_bytes, destination, reader_stream);
                transfer.enqueueDeviceToMappedHost(*received_count, 0, *snapshot, bank_bytes, count_bytes, destination, reader_stream);
                transfer.enqueueDeviceToMappedHost(*output, 0, *snapshot, bank_bytes + count_bytes, capacity, destination, reader_stream);
                require(reader_backend->recordEvent(reader_done.event, destination.ordinal, reader_stream), "Reader observation event failed");
            });
            reader_done.await(std::chrono::seconds(30));
            ReaderReplayBank result;
            std::memcpy(&result, snapshot->mutableHostData(), bank_bytes);
            return result;
        };
        /** Both original submissions precede either whole-graph terminal wait. */
        const auto run = [&] {
            producer.submitAndWait([&] {
                require(source_graph->launch(), "Reader source replay failed");
                require(source_backend->recordEvent(source_done.event, source.ordinal, source_stream), "Reader source terminal failed");
            });
            consumer.submitAndWait([&] { require(reader_graph->launch(), "Reader metadata replay failed"); });
            auto result = observe();
            source_done.await(std::chrono::seconds(30));
            return result;
        };
        std::uint64_t sequence = 0;
        for (int reset = 0; reset < 2; ++reset) {
            for (const auto count : {256u, 16u, 0u, 64u, 0u, 256u, 16u}) {
                SCOPED_TRACE(::testing::Message() << "reset=" << reset << " count=" << count);
                ReaderReplayBank recorded{};
                recorded.frame.interval = {11, static_cast<std::uint64_t>(reset + 1), ++sequence};
                recorded.frame.phase = reader_state::LookupPhase::Ready;
                recorded.frame.status = reader::Status::Success;
                recorded.frame.original_ids = count;
                recorded.original_rows = static_cast<int>(count);
                const auto bytes = reader::outputBytes(geometry, count);
                upload(recorded, bytes);
                const auto observed = run();
                reader::Header expected_header{}, expected_reply{};
                ASSERT_TRUE(reader::prepare(recorded.frame, geometry, 1, &recorded.error, expected_header));
                ASSERT_TRUE(reader::accept(expected_header, sizeof(expected_header), geometry, 1, 11,
                    count, &recorded.error, expected_reply));
                ASSERT_TRUE(reader::acknowledge(expected_reply, bytes, geometry, &recorded.error, CapturedTransferStatus::Ready));
                ASSERT_TRUE(reader::confirm(expected_reply, sizeof(expected_reply), recorded.frame, geometry, 1, &recorded.error));
                EXPECT_EQ(std::memcmp(&observed.header, &expected_header, sizeof(expected_header)), 0);
                EXPECT_EQ(std::memcmp(&observed.reply, &expected_reply, sizeof(expected_reply)), 0);
                EXPECT_EQ(observed.published_bytes, bytes);
                EXPECT_EQ(observed.error, 0);
                EXPECT_EQ(observed.healthy, 0); // Cold prefill Ok0 is valid; it is not an admission flag.
                std::uint64_t received_bytes = ~std::uint64_t{0};
                const auto* raw = static_cast<const unsigned char*>(snapshot->mutableHostData());
                std::memcpy(&received_bytes, raw + bank_bytes, count_bytes);
                EXPECT_EQ(received_bytes, bytes);
                for (std::size_t index = 0; index < capacity; ++index)
                    ASSERT_EQ(raw[bank_bytes + count_bytes + index], index < bytes
                        ? patternByte(index, sequence % 2) : 0xa5) << index;
            }
        }
        // The same captured pipeline rejects malformed original count/header
        // evidence without poisoning the channel's next legitimate publication.
        for (int fault = 0; fault < 4; ++fault) {
            ReaderReplayBank recorded{};
            recorded.frame.interval = {11, 8, ++sequence};
            recorded.frame.phase = reader_state::LookupPhase::Ready;
            recorded.frame.status = reader::Status::Success;
            recorded.frame.original_ids = 16;
            recorded.original_rows = 16;
            if (fault == 0) recorded.header_bytes = 0;
            if (fault == 1) recorded.original_rows = 32;
            if (fault == 2) recorded.frame.status = reader::Status::Cancelled;
            if (fault == 3) recorded.error = 37;
            upload(recorded, reader::outputBytes(geometry, 16));
            const auto observed = run();
            EXPECT_EQ(observed.error, fault == 3 ? 37 : static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
            EXPECT_EQ(observed.healthy, 0);
            EXPECT_EQ(observed.published_bytes, 0u);
            EXPECT_EQ(observed.reply.status, reader::Status::Cancelled);
        }
        // ConfirmAck must reject absent, late, foreign and unsuccessful evidence
        // independently of the real successful payload that preceded it.
        for (int fault = 0; fault < 5; ++fault) {
            ReaderReplayBank recorded{};
            recorded.frame.interval = {11, 9, ++sequence};
            recorded.frame.phase = reader_state::LookupPhase::Ready;
            recorded.frame.status = reader::Status::Success;
            recorded.frame.original_ids = 16;
            recorded.original_rows = 16;
            ASSERT_TRUE(reader::prepare(recorded.frame, geometry, 1, &recorded.error, recorded.recorded_ack));
            if (fault == 0) recorded.header_bytes = 0;
            if (fault == 1) --recorded.recorded_ack.interval.sequence;
            if (fault == 2) ++recorded.recorded_ack.coordinate;
            if (fault == 3) recorded.recorded_ack.status = reader::Status::Cancelled;
            if (fault == 4) recorded.error = 37; // An original unrelated first error must survive unchanged.
            // The CPU oracle must never seed the GPU's failure: each capture
            // receives the original clean bank (or original sticky fault37).
            int oracle_error = recorded.error;
            const bool accepted = reader::confirm(recorded.recorded_ack, recorded.header_bytes,
                recorded.frame, geometry, 1, &oracle_error);
            EXPECT_FALSE(accepted);
            if (!accepted && oracle_error == 0)
                oracle_error = static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure);
            EXPECT_EQ(recorded.error, fault == 4 ? 37 : 0);
            upload(recorded, 0, false);
            consumer.submitAndWait([&] { require(confirm_graph->launch(), "Recorded ACK replay failed"); });
            const auto observed = observe();
            EXPECT_EQ(observed.error, oracle_error);
            EXPECT_EQ(observed.healthy, 0);
            EXPECT_EQ(observed.published_bytes, 0u);
        }
        // Terminal cancellation is deliberately last. The original channel may
        // never be rearmed after this failure, even for a legitimate empty query.
        ASSERT_TRUE(transfer.cancelCapturedTransferInvocation(channel, original));
        for (int repeat = 0; repeat < 20; ++repeat) {
            ReaderReplayBank recorded{};
            recorded.frame.interval = {11, 10, ++sequence};
            recorded.frame.phase = reader_state::LookupPhase::Ready;
            recorded.frame.status = reader::Status::Success;
            upload(recorded, 0);
            const auto observed = run();
            reader::Header expected_header{};
            int error = 0;
            ASSERT_TRUE(reader::prepare(recorded.frame, geometry, 1, &error, expected_header));
            EXPECT_EQ(std::memcmp(&observed.header, &expected_header, sizeof(expected_header)), 0);
            EXPECT_EQ(observed.header.status, reader::Status::Success);
            EXPECT_EQ(observed.header.original_ids, 0u);
            auto expected = observed.header;
            ASSERT_FALSE(reader::acknowledge(expected, 0, geometry, &error, CapturedTransferStatus::Cancelled));
            const auto* cancelled_bytes = static_cast<const unsigned char*>(snapshot->mutableHostData());
            std::uint64_t cancelled_extent = ~std::uint64_t{0};
            std::memcpy(&cancelled_extent, cancelled_bytes + bank_bytes, count_bytes);
            EXPECT_EQ(cancelled_extent, 0u);
            for (std::size_t index = 0; index < capacity; ++index)
                ASSERT_EQ(cancelled_bytes[bank_bytes + count_bytes + index], 0xa5) << index;
            EXPECT_EQ(observed.reply.status, reader::Status::Cancelled);
            EXPECT_EQ(observed.error, static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
            EXPECT_EQ(observed.published_bytes, 0u);
            EXPECT_EQ(observed.healthy, 0);
        }
        unwind.reset();
        EXPECT_EQ(transfer.finishCapturedTransferInvocation(channel, original), CapturedTransferInvocationRetirement::Cancelled);
    }
    for (const auto device : {DeviceId::cpu(), source, destination}) {
        EXPECT_EQ(memory->claimedBytes(device, PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        EXPECT_EQ(memory->reservedBytes(device, PhysicalMemoryOwner::NativeGraphExecutable), 0u);
    }
}
} // namespace

#ifdef HAVE_CUDA
TEST(PooledReplicatedReaderNative, CUDAOriginalCapturedMetadataReplay)
{ replicatedReaderNativeReplay(DeviceId::cuda(0), DeviceId::cuda(1)); }
#endif
#ifdef HAVE_ROCM
TEST(PooledReplicatedReaderNative, ROCmOriginalCapturedMetadataReplay)
{ replicatedReaderNativeReplay(DeviceId::rocm(0), DeviceId::rocm(1)); }
#endif
