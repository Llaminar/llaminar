/**
 * @file NativeCompletedReaderReplay.inc
 * @brief Original captured CCL byte completion, real reader ACK and recorded wire replay.
 *
 * This component replays declared original frame/data inputs. It does not load a
 * model or certify lookup placement. One prepared source/PMA/communicator owns
 * both endpoint graphs and exact protocol banks. Terminal observation queries
 * those graphs' original streams without creating a second event resource.
 */
#include "kernels/qwen4exp/PooledImmutableReplicatedReaderKernels.h"
#include <algorithm>
#include <span>
#ifdef HAVE_CUDA
#include "collective/backends/NCCLDynamicLoader.h"
#endif
#ifdef HAVE_ROCM
#include "collective/backends/RCCLDynamicLoader.h"
#endif

namespace {
namespace ncr = llaminar2::qwen4exp::replicated_reader;
namespace ncs = llaminar2::qwen4exp::adaptive_lookup::pooled;
/** @brief Recorded component surface; wire tails never impersonate semantic PLE rows. */
enum class NativeCompletedSurface { Wire, ReaderRows };
/** @brief Persistent diagnostic operands; only required metadata/payload crosses the wire. */
struct alignas(64) NativeCompletedBank {
    ncs::LookupFrame frame{};
    ncr::Header outgoing{}, incoming{}, reply{}, acknowledgement{};
    NativeByteSourceSlice slice{};
    std::uint64_t header_bytes = sizeof(ncr::Header), received_bytes = 0, returned_bytes = 0;
    std::uint64_t acknowledgement_bytes = sizeof(ncr::Header), published_bytes = 0;
    std::int32_t original_rows = 0;
    int healthy = 0, error = 0;
    unsigned long long forward_bytes = 0, return_bytes = 0;
    ncclLlaminarByteCompletion forward{}, returned{};
    ncclLlaminarCompletionOperand forward_operand{}, return_operand{};
};
static_assert(std::is_trivially_copyable_v<NativeCompletedBank>);

/** @brief Replay actual original endpoint capture with row-safe ACK or arbitrary live byte tails.
 * @param source Original source endpoint; explicit cohorts retain its declared native coordinate.
 * @param reader Original distinct homogeneous receiver, never a replicated source/provider.
 * @param surface Recorded wire-only or actual row-safe metadata/ACK obligations.
 * @param addressed_cohort Exact optional original communicator population, in native coordinate order.
 * @param deep_progress Retain a large original bank and verify tiny/empty native metadata economy with bounded terminal probes.
 * Nonempty cohorts use the public addressed path; unrelated actual participants
 * retain communicator resources without joining either endpoint graph.
 */
void nativeCompletedPairReplay(DeviceId source, DeviceId reader, NativeCompletedSurface surface,
    std::span<const DeviceId> addressed_cohort = {}, bool deep_progress = false)
{
    const std::size_t capacity = deep_progress ? 256ULL * 1024 * 1024 : 512;
    const std::size_t source_bytes = 2 * capacity, guarded_bytes = capacity + 64;
    const std::size_t source_observed_bytes = deep_progress ? 64 : source_bytes;
    const std::size_t payload_observed_bytes = deep_progress ? 512 + 64 : guarded_bytes;
    constexpr std::uint64_t source_subspan_origin = 64;
    constexpr std::uint64_t table = 17, result_key = 51, return_key = 52;
    constexpr ncr::Geometry geometry{1, 2, 32, 4};
    const bool semantic = surface == NativeCompletedSurface::ReaderRows;
    const std::array devices{source, reader};
    const std::vector original_devices = addressed_cohort.empty()
        ? std::vector<DeviceId>{reader, source}
        : std::vector<DeviceId>(addressed_cohort.begin(), addressed_cohort.end());
    require(original_devices.size() >= 2 && original_devices.size() <= 8,
        "Completed replay needs the exact declared native population");
    std::array<int, 2> native_coordinates{-1, -1};
    std::vector<GlobalDeviceAddress> addresses;
    for (std::size_t coordinate = 0; coordinate < original_devices.size(); ++coordinate) {
        const auto device = original_devices[coordinate];
        require(device.is_cuda() == source.is_cuda() && device.is_rocm() == source.is_rocm(),
            "Completed replay needs one homogeneous original communicator");
        require(std::count(original_devices.begin(), original_devices.end(), device) == 1,
            "Completed replay cannot emulate ranks by repeating a device");
        for (std::size_t i = 0; i < devices.size(); ++i)
            if (devices[i] == device) native_coordinates[i] = static_cast<int>(coordinate);
        addresses.push_back(device.is_cuda() ? GlobalDeviceAddress::cuda(device.ordinal) : GlobalDeviceAddress::rocm(device.ordinal));
    }
    require(native_coordinates[0] >= 0 && native_coordinates[1] >= 0 && source != reader,
        "Completed replay endpoints must be distinct actual original participants");
    auto original = test::PreparedNativeLookupFixture::prepare(original_devices);
    require(!deep_progress || (surface == NativeCompletedSurface::Wire && !addressed_cohort.empty()),
        "Deep native progress replay needs original addressed wire ownership");
    // The retained original native cohort supplies immutable compiled geometry
    // before the one final PMA payload admission; no live value is downloaded.
    auto collective = createLocalTPContext(addresses, {}, source.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
    require(bool(collective), "Completed native pair communicator missing");
    ncclLlaminarP2pProgressShape progress_shape{};
    if (!addressed_cohort.empty()) {
#ifdef HAVE_CUDA
        if (source.is_cuda()) require(nccl_dynamic::nativeP2pProgressGeometry(&progress_shape) == nccl_dynamic::ncclSuccess,
            "Original NCCL progress geometry missing");
#endif
#ifdef HAVE_ROCM
        if (source.is_rocm()) require(rccl_dynamic::nativeP2pProgressGeometry(&progress_shape) == rccl_dynamic::ncclSuccess,
            "Original RCCL progress geometry missing");
#endif
        require(progress_shape.channels && !progress_shape.reserved && progress_shape.bytes ==
            sizeof(ncclLlaminarP2pProgress) + 2 * sizeof(std::uint64_t) * progress_shape.channels,
            "Original native progress geometry is incomplete");
    }
    const auto progress_bytes = static_cast<std::size_t>(progress_shape.bytes);
    const auto collective_bom = CollectiveMemoryEstimator::localTP(1, capacity,
        source.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
    const auto host_size = 4 * sizeof(NativeCompletedBank) + source_observed_bytes + 2 * payload_observed_bytes + 4 * progress_bytes;
    PhysicalMemoryPlanBuilder plan;
    plan.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
        TransferEngine::mappedHostRegionAllocationBytes(host_size));
    for (const auto device : devices) {
        const auto resource = original->resource(device);
        plan.add(resource, PhysicalMemoryOwner::ActivationTransportStaging,
            sizeof(NativeCompletedBank) + 2 * progress_bytes + (device == source ? source_bytes + (semantic ? 0 : guarded_bytes) : guarded_bytes));

        const auto graphs = RetainedParticipantGraphExecutable::generalBOM(original->resource(DeviceId::cpu()), resource, 2);
        for (const auto& row : graphs.resources()) plan.add(row);
    }
    for (const auto device : original_devices)
        plan.add(original->resource(device), PhysicalMemoryOwner::LocalCollective, collective_bom.perDeviceBytes());
    original->admit(plan.build());
    auto memory = original->memory();
    require(collective->reserveCollectiveResources(collective_bom.backend_payload_capacity_bytes,
        collective_bom.fp16_scratch_elements, memory), "Completed native pair collective admission failed");
    ::testing::Test::RecordProperty("qualification_scope", semantic ? "recorded_native_reader_ack" : "recorded_native_wire");
    ::testing::Test::RecordProperty("payload_route", source.is_cuda() ? "nccl" : "rccl");
    ::testing::Test::RecordProperty("payload_modes", "completed_prefix,completed_source_slice");
    ::testing::Test::RecordProperty("native_source_coordinate", std::to_string(native_coordinates[0]));
    ::testing::Test::RecordProperty("native_consumer_coordinate", std::to_string(native_coordinates[1]));
    ::testing::Test::RecordProperty("native_original_population", std::to_string(original_devices.size()));
    ::testing::Test::RecordProperty("native_payload_addressing", addressed_cohort.empty() ? "original_broadcast_control" : "actual_endpoint_only");
    auto& transfer = TransferEngine::instance();
    {
        std::array<std::shared_ptr<IWorkerGPUContext>, 2> workers{original->permit(source).worker(), original->permit(reader).worker()};
        std::array<IBackend*, 2> backends{getBackendFor(source), getBackendFor(reader)};
        std::array<void*, 2> streams{};
        std::array<NativeCompletionAuthority, 2> authorities;
        std::array<PhysicalMemoryAllocationLease, 2> claims;
        std::array<std::shared_ptr<DeviceTransferBuffer>, 2> metadata, progress;
        for (std::size_t i = 0; i < 2; ++i) {
            streams[i] = workers[i]->defaultStream();
            require(reinterpret_cast<std::uintptr_t>(streams[i]) > 2, "Completed pair requires original exact streams");
            claims[i] = memory->claimNewAllocation(devices[i], PhysicalMemoryOwner::ActivationTransportStaging,
                sizeof(NativeCompletedBank) + 2 * progress_bytes + (i == 0 ? source_bytes + (semantic ? 0 : guarded_bytes) : guarded_bytes));
            metadata[i] = transfer.allocateDeviceTransferBuffer(sizeof(NativeCompletedBank), devices[i]);
            if (progress_bytes) progress[i] = transfer.allocateDeviceTransferBuffer(2 * progress_bytes, devices[i]);
            authorities[i] = collective->nativeCompletionAuthority(native_coordinates[i]);
        }
        auto source_data = transfer.allocateDeviceTransferBuffer(source_bytes, source);
        auto output = transfer.allocateDeviceTransferBuffer(guarded_bytes, reader);
        auto returned = semantic ? std::shared_ptr<DeviceTransferBuffer>{} : transfer.allocateDeviceTransferBuffer(guarded_bytes, source);
        auto host_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
            TransferEngine::mappedHostRegionAllocationBytes(host_size));
        auto host = transfer.allocateMappedHostRegion(host_size, original_devices);
        auto* host_base = static_cast<std::byte*>(host->mutableHostData());
        const auto source_offset = 4 * sizeof(NativeCompletedBank);
        const auto output_offset = source_offset + source_observed_bytes;
        const auto returned_offset = output_offset + payload_observed_bytes;
        for (std::size_t byte = 0; byte < source_observed_bytes; ++byte)
            host_base[source_offset + byte] = std::byte((byte * 7 + 3) % 251);
        auto* banks0 = reinterpret_cast<NativeCompletedBank*>(metadata[0]->mutableDeviceData());
        auto* banks1 = reinterpret_cast<NativeCompletedBank*>(metadata[1]->mutableDeviceData());
        const std::array banks{banks0, banks1};
        // This whole-component observation uses the same original exact-stream
        // query as RetainedParticipantGraphExecutable::observeTerminal(). No
        // independent read-ahead/ordering-event program or allowance is needed.
        const auto await = [&] {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            for (std::size_t i = 0; i < 2; ++i) {
                GPUStreamExecutionState state = GPUStreamExecutionState::Pending;
                while (state != GPUStreamExecutionState::Complete) {
                    workers[i]->submitAndWait([&] {
                        state = workers[i]->queryStreamExecutionState(streams[i], "native completed pair terminal");
                    });
                    require(state == GPUStreamExecutionState::Complete || std::chrono::steady_clock::now() < deadline,
                        "Completed pair original terminal timed out");
                    if (state != GPUStreamExecutionState::Complete) std::this_thread::yield();
                }
            }
        };
        workers[0]->submitAndWait([&] {
            if (deep_progress) require(backends[0]->memset(source_data->mutableDeviceData(), 0x3c,
                source_bytes, source.ordinal, streams[0]), "Deep original source initialization failed");
            else transfer.enqueueMappedHostToDevice(*host, source_offset, *source_data, 0, source_bytes, source, streams[0]);
        });
        const auto observe_progress = [&](std::size_t i, std::size_t phase) {
            if (!progress_bytes) return;
            auto* output = reinterpret_cast<ncclLlaminarP2pProgress*>(
                static_cast<std::byte*>(progress[i]->mutableDeviceData()) + phase * progress_bytes);
#ifdef HAVE_CUDA
            if (devices[i].is_cuda()) require(nccl_dynamic::nativeP2pProgressSnapshot(authorities[i].binding(),
                native_coordinates[1-i], output, progress_bytes, streams[i]) == nccl_dynamic::ncclSuccess,
                "Original NCCL terminal progress launch failed");
#endif
#ifdef HAVE_ROCM
            if (devices[i].is_rocm()) require(rccl_dynamic::nativeP2pProgressSnapshot(authorities[i].binding(),
                native_coordinates[1-i], output, progress_bytes, streams[i]) == rccl_dynamic::ncclSuccess,
                "Original RCCL terminal progress launch failed");
#endif
        };
        await();
        std::array<std::vector<std::shared_ptr<RetainedParticipantGraphExecutable>>, 2> owners;
        NativeReplayFailureDrain drain;
        drain.abort_and_join = [&] {
            require(collective->signalNativeCompletionAbort(), "Completed pair could not signal original native cancellation");
            await();
            for (const auto& endpoint : owners)
                for (const auto& owner : endpoint)
                    require(test::RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(owner), "Completed pair physical stream retirement unproved");
        };
        const auto args = [&](std::size_t i, ncr::Action action) {
            auto* bank = banks[i];
            ncr::NativeArgs result{devices[i], geometry, 1, table,
                DeviceRowRange::deviceCounted(32, &bank->original_rows), &bank->frame,
                action == ncr::Action::ConfirmAck ? &bank->acknowledgement : &bank->incoming,
                action == ncr::Action::PrepareResultHeader ? &bank->outgoing : &bank->reply,
                action == ncr::Action::AcknowledgeResult ? &bank->received_bytes :
                    action == ncr::Action::ConfirmAck ? &bank->acknowledgement_bytes : &bank->header_bytes,
                action == ncr::Action::PrepareResult ? &bank->slice.live_bytes : &bank->published_bytes,
                &bank->healthy, &bank->error, CapturedTransferReceiveCompletion::notRequired()};
            if (action == ncr::Action::AcknowledgeResult || action == ncr::Action::ConfirmAck) {
                result.native_payload_completion = authorities[i].receive(&bank->forward);
                result.native_payload_message = result_key;
            }
            // Fixed Header/ACK receives borrow the same original communicator
            // authority as production, so a late native abort vetoes success.
            if (action == ncr::Action::AcceptResultHeader || action == ncr::Action::ConfirmAck)
                result.native_fixed_header = authorities[i].binding();
            return result;
        };
        const auto launch = [&](std::size_t i, ncr::Action action) {
            auto operands = args(i, action);
#ifdef HAVE_CUDA
            if (devices[i].is_cuda()) ncr::enqueueCUDAPooledReplicatedReader(action, operands, streams[i]);
#endif
#ifdef HAVE_ROCM
            if (devices[i].is_rocm()) ncr::enqueueROCmPooledReplicatedReader(action, operands, streams[i]);
#endif
        };
        // Both operations use the retained original communicator and exact endpoint
        // streams. In addressed mode no third participant records or submits work.
        const auto move = [&](std::size_t i, void* payload, const NativeCollectiveBytes& bytes,
                              bool forward, const char* stage) {
            const int producer = native_coordinates[forward ? 0 : 1];
            const int consumer = native_coordinates[forward ? 1 : 0];
            return addressed_cohort.empty()
                ? collective->nativeBroadcastBytesOnStream(payload, payload, bytes, producer,
                    native_coordinates[i], streams[i], stage)
                : collective->nativeTransferBytesOnStream(payload, bytes, NativeByteEdge(producer, consumer),
                    native_coordinates[i], streams[i], stage);
        };
        std::array<std::size_t, 2> mixed_first_use{};
        std::array<NativeCompletedBank, 2> last_observed{};
        for (int mode = 0; mode < 2; ++mode) {
            for (std::size_t i = 0; i < 2; ++i)
                owners[i].push_back(test::RetainedParticipantGraphExecutableFixtureAccess::original(original->permit(devices[i]), streams[i]));
            std::barrier begun(2), recorded(2), ended(2);
            std::array<bool, 2> began{};
            std::array<std::exception_ptr, 2> errors{};
            std::array<bool, 2> recording_failed{};
            const auto capture = [&](std::size_t i) {
                auto& definition = test::RetainedParticipantGraphExecutableFixtureAccess::capture(owners[i].back());
                std::unique_ptr<ScopedBackendGraphCapture> scope;
                try {
                    scope = std::make_unique<ScopedBackendGraphCapture>(*workers[i], definition, "completed native reader pair");
                    require(scope->begin(), "Completed native pair begin failed");
                    began[i] = true;
                } catch (...) { errors[i] = std::current_exception(); }
                begun.arrive_and_wait();
                if (began[0] && began[1]) try {
                    if (semantic && i == 0) launch(i, ncr::Action::PrepareResultHeader);
                    require(move(i, i == 0 ? &banks0->outgoing : &banks1->incoming,
                        NativeCollectiveBytes::fixed(sizeof(ncr::Header)), true, "native_original_result_header"), "Native original header failed");
                    if (semantic) launch(i, i == 0 ? ncr::Action::PrepareResult : ncr::Action::AcceptResultHeader);
                    const auto bytes = mode == 0
                        ? NativeCollectiveBytes::prefix(i == 0 ? &banks0->slice.live_bytes : &banks1->received_bytes,
                            capacity, &banks[i]->forward_operand)
                        : i == 0 ? NativeCollectiveBytes::sourceSlice(&banks0->slice, source_bytes - source_subspan_origin,
                            capacity, &banks0->forward_operand, source_subspan_origin)
                            : NativeCollectiveBytes::receiveSlice(&banks1->received_bytes, capacity, &banks1->forward_operand);
                    // A sliced source keeps its exact admitted subspan pointer and absolute descriptor.
                    auto* source_subspan = static_cast<std::byte*>(source_data->mutableDeviceData()) +
                        (mode == 0 ? 0 : source_subspan_origin);
                    require(move(i, i == 0 ? source_subspan : output->mutableDeviceData(), bytes,
                        true, "native_original_result_payload"), "Native completed payload failed");
                    if (semantic) {
                        if (i == 1) launch(i, ncr::Action::AcknowledgeResult);
                        require(move(i, i == 0 ? &banks0->acknowledgement : &banks1->reply,
                            NativeCollectiveBytes::fixed(sizeof(ncr::Header)), false, "native_original_reader_ack"), "Native original ACK failed");
                        if (i == 0) launch(i, ncr::Action::ConfirmAck);
                    } else {
                        const auto reverse = NativeCollectiveBytes::prefix(i == 0 ? &banks0->returned_bytes : &banks1->received_bytes,
                            capacity, &banks[i]->return_operand);
                        require(move(i, i == 0 ? returned->mutableDeviceData() : output->mutableDeviceData(), reverse,
                            false, "native_original_wire_return"), "Native completed return failed");
                    }
                } catch (...) { errors[i] = std::current_exception(); collective->signalNativeCompletionAbort(); }
                recorded.arrive_and_wait();
                if (scope) try { scope->finish(); } catch (...) {
                    if (!errors[i]) errors[i] = std::current_exception();
                    collective->signalNativeCompletionAbort();
                }
                // Runtime storage may consume the original native source
                // definition. Both recordings must have ended before either
                // participant prepares or instantiates its executable. Failed
                // recordings still reach this cold join, then neither proceeds.
                // These flags remain immutable after the join: errors[] may
                // subsequently receive participant-local instantiation failures.
                recording_failed[i] = bool(errors[i]);
                ended.arrive_and_wait();
                if (!recording_failed[0] && !recording_failed[1]) try {
                    require(definition.prepareRuntimeContextStorage(memory), "Completed pair original runtime storage preparation failed");
                    require(definition.instantiate(), "Completed pair instantiate failed");
                    mixed_first_use[i] += definition.residentMemoryBytes();
                } catch (...) {
                    errors[i] = std::current_exception();
                    collective->signalNativeCompletionAbort();
                }
            };
            auto first = workers[0]->submitAsync([&] { capture(0); });
            auto second = workers[1]->submitAsync([&] { capture(1); });
            first.get(); second.get();
            for (const auto& error : errors) if (error) std::rethrow_exception(error);
            for (std::size_t sample = 0; sample < 20; ++sample) {
                const std::size_t live = sample == 19 ? 0 : semantic ? std::array<std::size_t, 5>{512, 0, 256, 0, 512}[sample % 5]
                    : std::array<std::size_t, 8>{512, 0, 1, 17, 511, 3, 0, 257}[sample % 8];
                const auto origin = mode == 0 ? 0 : live == 0 ? source_bytes : source_bytes - live;
                SCOPED_TRACE(::testing::Message() << "surface=" << static_cast<unsigned>(surface)
                    << " mode=" << mode << " sample=" << sample << " live=" << live << " origin=" << origin);
                const std::uint64_t admission = 1 + sample / 10;
                const std::uint64_t sequence = 1 + sample % 10;
                for (std::size_t i = 0; i < 2; ++i) {
                    NativeCompletedBank input{};
                    input.frame.interval = {table, admission, sequence};
                    input.frame.phase = ncs::LookupPhase::Ready;
                    input.frame.status = qwen4exp::adaptive_lookup::Status::Success;
                    input.frame.original_ids = semantic ? static_cast<std::uint32_t>(live / 16) : 16;
                    input.original_rows = static_cast<std::int32_t>(input.frame.original_ids);
                    if (!semantic)
                        input.outgoing = {input.frame.interval, 1, input.frame.original_ids, 4, 1, 2, qwen4exp::adaptive_lookup::Status::Success};
                    input.slice = {origin, semantic ? 0 : live};
                    auto* bank = banks[i];
                    const auto* interval = i == 0 ? &bank->frame.interval : semantic ? &bank->reply.interval : &bank->incoming.interval;
                    input.forward_operand = {&interval->table, &interval->admission, &interval->sequence,
                        &bank->error, &bank->forward, &bank->forward_bytes, result_key,
                        static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure), 0};
                    input.return_operand = {&interval->table, &interval->admission, &interval->sequence,
                        &bank->error, &bank->returned, &bank->return_bytes, return_key,
                        static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure), 0};
                    std::memcpy(host_base + i * sizeof(input), &input, sizeof(input));
                    workers[i]->submitAndWait([&] {
                        transfer.enqueueMappedHostToDevice(*host, i * sizeof(input), *metadata[i], 0, sizeof(input), devices[i], streams[i]);
                        if (i == 1 || returned)
                            require(backends[i]->memset(i == 0 ? returned->mutableDeviceData() : output->mutableDeviceData(),
                                0xa5, guarded_bytes, devices[i].ordinal, streams[i]), "Completed native guard initialization failed");
                    });
                }
                for (std::size_t i = 0; i < 2; ++i)
                    workers[i]->submitAndWait([&] { observe_progress(i, 0); });
                // Both original endpoint graphs are submitted before either terminal is awaited.
                for (const auto i : sample % 2 ? std::array<std::size_t, 2>{1, 0} : std::array<std::size_t, 2>{0, 1})
                    workers[i]->submitAndWait([&] { require(test::RetainedParticipantGraphExecutableFixtureAccess::submit(owners[i].back(), sample + 1), "Completed native pair replay failed"); });
                for (std::size_t i = 0; i < 2; ++i)
                    workers[i]->submitAndWait([&] { observe_progress(i, 1); });
                await();
                std::array<NativeCompletedBank, 2> observed{};
                std::array<std::uint32_t, 2> aborted{};
                for (std::size_t i = 0; i < 2; ++i) {
                    require(test::RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(owners[i].back()), "Completed pair original launch not terminal");
                    workers[i]->submitAndWait([&] {
                        transfer.enqueueDeviceToMappedHost(*metadata[i], 0, *host,
                            (2 + i) * sizeof(NativeCompletedBank), sizeof(NativeCompletedBank), devices[i], streams[i]);
                        require(backends[i]->deviceToHostOnStream(&aborted[i], const_cast<const std::uint32_t*>(authorities[i].binding().abortFlag), sizeof(aborted[i]), devices[i].ordinal, streams[i]), "Original abort observation failed");
                        if (i == 0) transfer.enqueueDeviceToMappedHost(*source_data, 0, *host, source_offset, source_observed_bytes, source, streams[0]);
                        else transfer.enqueueDeviceToMappedHost(*output, 0, *host, output_offset, payload_observed_bytes, reader, streams[1]);
                        if (progress_bytes) transfer.enqueueDeviceToMappedHost(*progress[i], 0, *host,
                            returned_offset + payload_observed_bytes + 2 * i * progress_bytes,
                            2 * progress_bytes, devices[i], streams[i]);
                        if (i == 0 && returned) transfer.enqueueDeviceToMappedHost(*returned, 0, *host, returned_offset, payload_observed_bytes, source, streams[0]);
                    });
                }
                await();
                std::memcpy(observed.data(), host_base + 2 * sizeof(NativeCompletedBank), sizeof(observed));
                for (std::size_t i = 0; i < 2; ++i) {
                    ASSERT_EQ(observed[i].error, 0);
                    ASSERT_TRUE(ncclLlaminarCompletionMatches(observed[i].forward, table, admission, sequence,
                        live, result_key, authorities[i].binding(), aborted[i], observed[i].error));
                    if (!semantic) ASSERT_TRUE(ncclLlaminarCompletionMatches(observed[i].returned, table, admission, sequence,
                        live, return_key, authorities[i].binding(), aborted[i], observed[i].error));
                }
                if (progress_bytes && !semantic && live <= 1) {
                    // Native SIMPLE counts steps per channel. A fixed Header,
                    // its count metadata, and one live/empty payload need at
                    // most three sends/two returns on any original channel.
                    // Device observations never select or authorize replay.
                    for (std::size_t i = 0; i < 2; ++i) {
                        const auto* before = reinterpret_cast<const std::uint64_t*>(host_base +
                            returned_offset + payload_observed_bytes + 2 * i * progress_bytes);
                        const auto* after = reinterpret_cast<const std::uint64_t*>(
                            reinterpret_cast<const std::byte*>(before) + progress_bytes);
                        EXPECT_GE(after[0] - before[0], i == 0 ? 3u : 2u);
                        EXPECT_GE(after[1] - before[1], i == 0 ? 2u : 3u);
                        std::uint64_t max_sent=0,max_received=0,sum_sent=0,sum_received=0;
                        for (std::size_t channel = 0; channel < progress_shape.channels; ++channel) {
                            const auto sent = after[2 + 2 * channel] - before[2 + 2 * channel];
                            const auto received = after[3 + 2 * channel] - before[3 + 2 * channel];
                            max_sent=std::max(max_sent,sent);max_received=std::max(max_received,received);
                            sum_sent+=sent;sum_received+=received;
                            EXPECT_LE(sent, i == 0 ? 3u : 2u) << "channel=" << channel;
                            EXPECT_LE(received, i == 0 ? 2u : 3u) << "channel=" << channel;
                        }
                        EXPECT_EQ(sum_sent,after[0]-before[0]);EXPECT_EQ(sum_received,after[1]-before[1]);
                        ::testing::Test::RecordProperty("native_progress_"+std::to_string(mode)+"_"+
                            std::to_string(sample)+"_"+std::to_string(i),
                            "capacity="+std::to_string(capacity)+" live="+std::to_string(live)+
                            " channels="+std::to_string(progress_shape.channels)+" sent="+std::to_string(sum_sent)+
                            " received="+std::to_string(sum_received)+" max_sent="+std::to_string(max_sent)+
                            " max_received="+std::to_string(max_received));
                    }
                }
                last_observed = observed;
                ASSERT_EQ(observed[0].forward_bytes, live);
                ASSERT_EQ(observed[1].forward_bytes, 0u);
                if (!semantic) {
                    ASSERT_EQ(observed[0].return_bytes, 0u);
                    ASSERT_EQ(observed[1].return_bytes, live);
                }
                if (semantic) {
                    ASSERT_EQ(observed[1].reply.status, qwen4exp::adaptive_lookup::Status::Success);
                    ASSERT_TRUE(ncr::confirm(observed[0].acknowledgement, sizeof(ncr::Header), observed[0].frame, geometry, 1, &observed[0].error));
                }
                for (std::size_t byte = 0; byte < source_observed_bytes; ++byte)
                    ASSERT_EQ(host_base[source_offset + byte], deep_progress ? std::byte{0x3c} : std::byte((byte * 7 + 3) % 251));
                for (std::size_t byte = 0; byte < payload_observed_bytes; ++byte) {
                    const auto expected = byte < live ? (deep_progress ? std::byte{0x3c} : std::byte(((origin + byte) * 7 + 3) % 251)) : std::byte{0xa5};
                    ASSERT_EQ(host_base[output_offset + byte], expected) << byte;
                    if (returned) ASSERT_EQ(host_base[returned_offset + byte], expected) << byte;
                }
            }
        }
        // The final successful frame is empty. Replay its exact old Completed
        // receipt while the original source is deliberately not submitted. Only
        // the original native signal interrupts the reader; no flag is written
        // through its immutable view and no logical source completion is claimed.
        for (std::size_t i = 0; i < 2; ++i) {
            std::memcpy(host_base + i * sizeof(NativeCompletedBank), &last_observed[i], sizeof(NativeCompletedBank));
            workers[i]->submitAndWait([&] {
                transfer.enqueueMappedHostToDevice(*host, i * sizeof(NativeCompletedBank), *metadata[i],
                    0, sizeof(NativeCompletedBank), devices[i], streams[i]);
            });
        }
        workers[1]->submitAndWait([&] {
            require(test::RetainedParticipantGraphExecutableFixtureAccess::submit(owners[1].back(), 21),
                "Completed pair missing-source cancellation submit failed");
        });
        require(collective->signalNativeCompletionAbort(), "Original native signal-only cancellation failed");
        await();
        for (const auto& endpoint : owners)
            for (const auto& owner : endpoint)
                require(test::RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(owner),
                    "Cancelled native pair lacks original physical stream proof");
        NativeCompletedBank cancelled{};
        std::uint32_t native_abort = 0;
        workers[1]->submitAndWait([&] {
            transfer.enqueueDeviceToMappedHost(*metadata[1], 0, *host, 3 * sizeof(NativeCompletedBank),
                sizeof(NativeCompletedBank), reader, streams[1]);
            require(backends[1]->deviceToHostOnStream(&native_abort,
                const_cast<const std::uint32_t*>(authorities[1].binding().abortFlag), sizeof(native_abort),
                reader.ordinal, streams[1]), "Cancelled native abort observation failed");
        });
        await();
        std::memcpy(&cancelled, host_base + 3 * sizeof(NativeCompletedBank), sizeof(cancelled));
        ASSERT_NE(native_abort, 0u);
        ASSERT_EQ(cancelled.error,
            static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
        // Rejection alone could be satisfied by the old Completed receipt once
        // its external abort/error becomes nonzero. Require a newly native-written
        // terminal disposition and its exact original communicator provenance.
        const auto expect_aborted = [&](const ncclLlaminarByteCompletion& receipt,
                                        std::uint64_t expected_message) {
            EXPECT_EQ(receipt.phase, ncclLlaminarCompletionPhase::Aborted);
            EXPECT_EQ(receipt.bytes, 0u);
            EXPECT_EQ(receipt.nativeAbortFlag, authorities[1].binding().abortFlag);
            EXPECT_EQ(receipt.nativeCommunicator, authorities[1].binding().deviceCommunicator);
            EXPECT_EQ(receipt.message, expected_message);
            EXPECT_EQ(receipt.reserved, 0u);
        };
        expect_aborted(cancelled.forward, result_key);
        if (!semantic) expect_aborted(cancelled.returned, return_key);
        // The raw count receive may retain stale FIFO bits after abort; only the
        // native resolved receipt, checked above, certifies the zero live extent.
        ASSERT_FALSE(ncclLlaminarCompletionMatches(cancelled.forward, table, 2, 10, 0,
            result_key, authorities[1].binding(), native_abort, cancelled.error));
        if (semantic) {
            ASSERT_EQ(cancelled.reply.status, qwen4exp::adaptive_lookup::Status::Cancelled);
            EXPECT_EQ(cancelled.reply.interval.table, 0u);
            EXPECT_EQ(cancelled.reply.interval.admission, 0u);
            EXPECT_EQ(cancelled.reply.interval.sequence, 0u);
        }
        ::testing::Test::RecordProperty("cancelled_empty_replay", "original_signal_absent_source_no_success");
        for (std::size_t i = 0; i < 2; ++i)
            ::testing::Test::RecordProperty("native_first_use_mixed_bytes_" + std::to_string(i), std::to_string(mixed_first_use[i]));
        drain.completed = true;
    }
    collective.reset();
    original->requirePayloadRetired();
    original->retire();
    original->requireExactRetired();
}

/**
 * @brief Replay declared actual homogeneous ranks without fake rank/device aliases.
 * @param cuda Select the declared CUDA or ROCm native hardware population.
 * @param degree Exact required device count; absent hardware fails fixture admission.
 * @param source_coordinate Original native source coordinate, including noncoordinator sources.
 * Each case records two endpoint graphs while every unrelated communicator member
 * stays absent. Prefix/slice forward and reverse receipts prove exact live bytes,
 * poison tails, zero/partial/large-to-small reset and native cancellation provenance.
 */
void nativeCompletedAddressedReplay(bool cuda, int degree, int source_coordinate, bool deep_progress = false)
{
    std::vector<DeviceId> devices;
    for (int ordinal = 0; ordinal < degree; ++ordinal)
        devices.push_back(cuda ? DeviceId::cuda(ordinal) : DeviceId::rocm(ordinal));
    require(source_coordinate >= 0 && source_coordinate < degree, "Invalid declared addressed source coordinate");
    const int consumer_coordinate = (source_coordinate + 1) % degree;
    nativeCompletedPairReplay(devices[source_coordinate], devices[consumer_coordinate],
        NativeCompletedSurface::Wire, devices, deep_progress);
}
} // namespace

#ifdef HAVE_CUDA
TEST(NativeCompletedCollective, CUDA_WireReplay) { nativeCompletedPairReplay(DeviceId::cuda(0), DeviceId::cuda(1), NativeCompletedSurface::Wire); }
TEST(NativeCompletedCollective, CUDA_ReaderAckReplay) { nativeCompletedPairReplay(DeviceId::cuda(0), DeviceId::cuda(1), NativeCompletedSurface::ReaderRows); }
#endif
#ifdef HAVE_ROCM
TEST(NativeCompletedCollective, ROCm_WireReplay) { nativeCompletedPairReplay(DeviceId::rocm(0), DeviceId::rocm(1), NativeCompletedSurface::Wire); }
TEST(NativeCompletedCollective, ROCm_ReaderAckReplay) { nativeCompletedPairReplay(DeviceId::rocm(0), DeviceId::rocm(1), NativeCompletedSurface::ReaderRows); }
#endif

#ifdef HAVE_CUDA
TEST(NativeCompletedCollective, CUDA_Addressed2Source0Replay) { nativeCompletedAddressedReplay(true, 2, 0); }
TEST(NativeCompletedCollective, CUDA_Addressed2Source1Replay) { nativeCompletedAddressedReplay(true, 2, 1); }
TEST(NativeCompletedCollective, CUDA_Addressed4Source0Replay) { nativeCompletedAddressedReplay(true, 4, 0); }
TEST(NativeCompletedCollective, CUDA_Addressed4Source3Replay) { nativeCompletedAddressedReplay(true, 4, 3); }
TEST(NativeCompletedCollective, CUDA_Addressed8Source0Replay) { nativeCompletedAddressedReplay(true, 8, 0); }
TEST(NativeCompletedCollective, CUDA_Addressed8Source7Replay) { nativeCompletedAddressedReplay(true, 8, 7); }
#endif

#ifdef HAVE_ROCM
TEST(NativeCompletedCollective, ROCm_Addressed2Source0Replay) { nativeCompletedAddressedReplay(false, 2, 0); }
TEST(NativeCompletedCollective, ROCm_Addressed2Source1Replay) { nativeCompletedAddressedReplay(false, 2, 1); }
TEST(NativeCompletedCollective, ROCm_Addressed4Source0Replay) { nativeCompletedAddressedReplay(false, 4, 0); }
TEST(NativeCompletedCollective, ROCm_Addressed4Source3Replay) { nativeCompletedAddressedReplay(false, 4, 3); }
TEST(NativeCompletedCollective, ROCm_Addressed8Source0Replay) { nativeCompletedAddressedReplay(false, 8, 0); }
TEST(NativeCompletedCollective, ROCm_Addressed8Source7Replay) { nativeCompletedAddressedReplay(false, 8, 7); }
#endif

#ifdef HAVE_CUDA
TEST(NativeCompletedCollective, CUDA_Addressed2LiveProgressReplay) { nativeCompletedAddressedReplay(true, 2, 0, true); }
#endif
#ifdef HAVE_ROCM
TEST(NativeCompletedCollective, ROCm_Addressed2LiveProgressReplay) { nativeCompletedAddressedReplay(false, 2, 0, true); }
#endif
