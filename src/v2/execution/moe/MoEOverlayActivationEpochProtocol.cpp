/**
 * @file MoEOverlayActivationEpochProtocol.cpp
 * @brief Atomic CPU oracle for node-local device-owned activation epochs.
 */

#include "MoEOverlayActivationEpochProtocol.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** Largest stage count representable by two 16-bit bank visit streams. */
        constexpr std::uint32_t kMaximumStageCount =
            2u * ((1u << kMoEOverlayActivationVisitBits) - 1u);

        /** @brief Atomic view over an ABI scalar without changing its layout. */
        template <typename Value>
        std::atomic_ref<Value> atomicValue(Value &value) noexcept
        {
            return std::atomic_ref<Value>(value);
        }

        /** @brief Acquire-load a scalar through the ABI's atomic ownership edge. */
        template <typename Value>
        Value acquireValue(const Value &value) noexcept
        {
            return atomicValue(const_cast<Value &>(value)).load(
                std::memory_order_acquire);
        }

        /** @brief Store a caller-requested diagnostic only when supplied. */
        bool fail(std::string message, std::string *error)
        {
            if (error)
                *error = std::move(message);
            return false;
        }

        /** @return Whether an endpoint enum names one real single writer. */
        constexpr bool validEndpoint(
            MoEOverlayActivationEndpoint endpoint) noexcept
        {
            return endpoint == MoEOverlayActivationEndpoint::Continuation ||
                   endpoint == MoEOverlayActivationEndpoint::Follower;
        }

        /** @return Whether a status code represents a terminal endpoint fault. */
        constexpr bool fatalStatusCode(
            MoEOverlayActivationStatusCode code) noexcept
        {
            switch (code)
            {
            case MoEOverlayActivationStatusCode::InvalidControl:
            case MoEOverlayActivationStatusCode::InvalidIdentity:
            case MoEOverlayActivationStatusCode::StaleGeneration:
            case MoEOverlayActivationStatusCode::OutOfOrder:
            case MoEOverlayActivationStatusCode::PayloadMismatch:
            case MoEOverlayActivationStatusCode::PeerAborted:
            case MoEOverlayActivationStatusCode::ExplicitAbort:
            case MoEOverlayActivationStatusCode::GenerationOverflow:
                return true;
            case MoEOverlayActivationStatusCode::Idle:
            case MoEOverlayActivationStatusCode::Success:
            case MoEOverlayActivationStatusCode::NotReady:
            case MoEOverlayActivationStatusCode::BufferBusy:
            case MoEOverlayActivationStatusCode::TimedOut:
                return false;
            }
            return false;
        }

        /** @return Previous stage, with -1 representing the initial state. */
        constexpr std::int32_t previousStage(
            std::uint32_t stage_ordinal) noexcept
        {
            return stage_ordinal == 0u
                       ? -1
                       : static_cast<std::int32_t>(stage_ordinal - 1u);
        }

        /** @return Whether one endpoint can accumulate a descriptor exactly. */
        bool canAccumulatePublishedTraffic(
            const MoEOverlayActivationEndpointStatus &status,
            std::uint64_t payload_bytes,
            std::uint64_t live_rows,
            std::uint64_t live_entries) noexcept
        {
            constexpr auto maximum =
                std::numeric_limits<std::uint64_t>::max();
            return status.published_payload_bytes <=
                       maximum - payload_bytes &&
                   status.published_live_rows <= maximum - live_rows &&
                   status.published_live_entries <=
                       maximum - live_entries &&
                   status.published_stage_count != maximum;
        }

        /** @brief Add one already-validated descriptor to endpoint-owned totals. */
        void accumulatePublishedTraffic(
            MoEOverlayActivationEndpointStatus &status,
            std::uint64_t payload_bytes,
            std::uint64_t live_rows,
            std::uint64_t live_entries) noexcept
        {
            status.published_payload_bytes += payload_bytes;
            status.published_live_rows += live_rows;
            status.published_live_entries += live_entries;
            ++status.published_stage_count;
        }
    } // namespace

    bool MoEOverlayActivationEpochConfig::valid() const noexcept
    {
        if (channel_nonce == 0u || topology_fingerprint_low == 0u ||
            topology_fingerprint_high == 0u || workspace_generation == 0u ||
            !source.valid() || !target.valid() || graph_role_mask == 0u ||
            (source.world_rank == target.world_rank &&
             source.participant_id == target.participant_id) ||
            model_layer_indices.empty() ||
            model_layer_indices.size() > kMaximumStageCount)
        {
            return false;
        }

        /*
         * A retained transaction executes each declared graph stage exactly
         * once. Duplicate model-layer indices usually indicate that an MTP
         * sidecar manifest was flattened incorrectly rather than declared as a
         * separate graph role, so setup rejects them explicitly.
         */
        std::int32_t previous_layer = -1;
        for (const std::int32_t layer : model_layer_indices)
        {
            if (layer <= previous_layer)
                return false;
            previous_layer = layer;
        }
        return true;
    }

    std::uint64_t MoEOverlayActivationEpochProtocol::stageManifestDigest(
        std::span<const std::int32_t> model_layer_indices) noexcept
    {
        if (model_layer_indices.empty() ||
            model_layer_indices.size() > kMaximumStageCount)
        {
            return 0u;
        }

        std::uint64_t digest = 14695981039346656037ull;
        const auto mixByte = [&](std::uint8_t byte)
        {
            digest ^= static_cast<std::uint64_t>(byte);
            digest *= 1099511628211ull;
        };
        const auto mix32 = [&](std::uint32_t value)
        {
            for (std::size_t byte = 0; byte < sizeof(value); ++byte)
            {
                mixByte(static_cast<std::uint8_t>(value & 0xffu));
                value >>= 8u;
            }
        };

        mix32(static_cast<std::uint32_t>(model_layer_indices.size()));
        for (const std::int32_t layer : model_layer_indices)
        {
            if (layer < 0)
                return 0u;
            mix32(static_cast<std::uint32_t>(layer));
        }
        return digest == 0u ? 0x9e3779b97f4a7c15ull : digest;
    }

    void MoEOverlayActivationEpochProtocol::initialize(
        MoEOverlayActivationEpochControl &control,
        const MoEOverlayActivationEpochConfig &config)
    {
        if (!config.valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay activation epoch requires a valid node-local channel and ordered stage manifest");
        }

        const bool pristine =
            control.channel.channel_nonce == 0u &&
            control.admission.last_generation == 0u &&
            control.admission.ready_signal == 0u &&
            control.identity.epoch_generation == 0u &&
            control.continuation_status.typedState() ==
                MoEOverlayActivationEndpointState::Idle &&
            control.follower_status.typedState() ==
                MoEOverlayActivationEndpointState::Idle &&
            control.buffers[0].dispatch_signal.value == 0u &&
            control.buffers[0].return_signal.value == 0u &&
            control.buffers[1].dispatch_signal.value == 0u &&
            control.buffers[1].return_signal.value == 0u;
        if (!pristine)
        {
            throw std::logic_error(
                "ExpertOverlay activation epoch control is already initialized or live");
        }

        control = MoEOverlayActivationEpochControl{};
        control.channel.channel_nonce = config.channel_nonce;
        control.channel.topology_fingerprint_low =
            config.topology_fingerprint_low;
        control.channel.topology_fingerprint_high =
            config.topology_fingerprint_high;
        control.channel.workspace_generation = config.workspace_generation;
        control.channel.stage_manifest_digest = stageManifestDigest(
            config.model_layer_indices);
        control.channel.source_world_rank = config.source.world_rank;
        control.channel.target_world_rank = config.target.world_rank;
        control.channel.source_participant_id = config.source.participant_id;
        control.channel.target_participant_id = config.target.participant_id;
        control.channel.source_tier_priority = config.source.tier_priority;
        control.channel.target_tier_priority = config.target.tier_priority;
        control.channel.source_domain_ordinal = config.source.domain_ordinal;
        control.channel.target_domain_ordinal = config.target.domain_ordinal;
        control.channel.stage_count = static_cast<std::uint32_t>(
            config.model_layer_indices.size());
        control.channel.lane_ordinal = config.lane_ordinal;
        control.channel.graph_role_mask = config.graph_role_mask;
    }

    MoEOverlayActivationEpochProtocol::MoEOverlayActivationEpochProtocol(
        MoEOverlayActivationEpochControl &control,
        MoEOverlayActivationEpochConfig config)
        : control_(&control), config_(std::move(config))
    {
        if (!config_.valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay activation epoch binding requires a valid expected configuration");
        }
        const auto &header = control_->channel;
        if (header.magic != kMoEOverlayActivationMagic ||
            header.abi_version != kMoEOverlayActivationABIVersion ||
            header.channel_nonce != config_.channel_nonce ||
            header.topology_fingerprint_low !=
                config_.topology_fingerprint_low ||
            header.topology_fingerprint_high !=
                config_.topology_fingerprint_high ||
            header.workspace_generation != config_.workspace_generation ||
            header.stage_manifest_digest !=
                stageManifestDigest(config_.model_layer_indices) ||
            header.source_world_rank != config_.source.world_rank ||
            header.target_world_rank != config_.target.world_rank ||
            header.source_participant_id != config_.source.participant_id ||
            header.target_participant_id != config_.target.participant_id ||
            header.source_tier_priority != config_.source.tier_priority ||
            header.target_tier_priority != config_.target.tier_priority ||
            header.source_domain_ordinal != config_.source.domain_ordinal ||
            header.target_domain_ordinal != config_.target.domain_ordinal ||
            header.stage_count != config_.model_layer_indices.size() ||
            header.buffer_count != kMoEOverlayActivationBufferCount ||
            header.lane_ordinal != config_.lane_ordinal ||
            header.graph_role_mask != config_.graph_role_mask)
        {
            throw std::logic_error(
                "ExpertOverlay activation epoch mapping disagrees with its exact topology or retained stage manifest");
        }
    }

    std::optional<MoEOverlayActivationEpochIdentity>
    MoEOverlayActivationEpochProtocol::arm(
        const MoEOverlayInferenceTransactionTicket &ticket,
        std::uint64_t epoch_generation,
        std::uint64_t deadline_ns,
        std::string *error)
    {
        if (error)
            error->clear();
        if (admissionState() != MoEOverlayActivationAdmissionState::Idle ||
            acquireValue(control_->admission.ready_signal) != 0u ||
            endpointState(MoEOverlayActivationEndpoint::Continuation) !=
                MoEOverlayActivationEndpointState::Idle ||
            endpointState(MoEOverlayActivationEndpoint::Follower) !=
                MoEOverlayActivationEndpointState::Idle)
        {
            fail("ExpertOverlay activation epoch slot is not quiescent", error);
            return std::nullopt;
        }
        for (std::uint32_t bank = 0u;
             bank < kMoEOverlayActivationBufferCount;
             ++bank)
        {
            if (dispatchTimeline(bank) != 0u || returnTimeline(bank) != 0u)
            {
                fail(
                    "ExpertOverlay activation epoch slot retained a signal outside its completed lease",
                    error);
                return std::nullopt;
            }
        }
        const auto graph_role_value = static_cast<std::uint32_t>(
            ticket.graph_role);
        const bool graph_role_admitted =
            graph_role_value < 32u &&
            (control_->channel.graph_role_mask &
             (std::uint32_t{1} << graph_role_value)) != 0u;
        if (!ticket.valid() ||
            ticket.action != MoEOverlayInferenceTransactionAction::Execute ||
            !graph_role_admitted ||
            ticket.workspace_generation !=
                control_->channel.workspace_generation ||
            ticket.topology_fingerprint_low !=
                control_->channel.topology_fingerprint_low ||
            ticket.topology_fingerprint_high !=
                control_->channel.topology_fingerprint_high ||
            ticket.source_world_rank != control_->channel.source_world_rank ||
            ticket.target_world_rank != control_->channel.target_world_rank)
        {
            fail("ExpertOverlay activation ticket does not authenticate this retained rank-pair graph", error);
            return std::nullopt;
        }
        const std::uint64_t last_generation = acquireValue(
            control_->admission.last_generation);
        if (epoch_generation == 0u || epoch_generation <= last_generation)
        {
            fail("ExpertOverlay activation generation is stale or non-monotonic", error);
            return std::nullopt;
        }
        if (epoch_generation > kMoEOverlayActivationMaxGeneration)
        {
            fail("ExpertOverlay activation generation exceeds the 64-bit timeline ABI", error);
            return std::nullopt;
        }
        if (deadline_ns == 0u)
        {
            fail("ExpertOverlay activation epoch requires a positive watchdog deadline", error);
            return std::nullopt;
        }

        MoEOverlayActivationEpochIdentity identity{
            .request_generation = ticket.request_generation,
            .command_id = ticket.command_id,
            .transaction_ordinal = ticket.transaction_ordinal,
            .logical_step_id = ticket.logical_step_id,
            .workspace_generation = ticket.workspace_generation,
            .placement_epoch = ticket.placement_epoch,
            .topology_fingerprint_low = ticket.topology_fingerprint_low,
            .topology_fingerprint_high = ticket.topology_fingerprint_high,
            .channel_nonce = control_->channel.channel_nonce,
            .epoch_generation = epoch_generation,
            .stage_manifest_digest = control_->channel.stage_manifest_digest,
            .digest = {},
            .source_world_rank = ticket.source_world_rank,
            .target_world_rank = ticket.target_world_rank,
            .source_participant_id = control_->channel.source_participant_id,
            .target_participant_id = control_->channel.target_participant_id,
            .source_tier_priority = control_->channel.source_tier_priority,
            .target_tier_priority = control_->channel.target_tier_priority,
            .source_domain_ordinal = control_->channel.source_domain_ordinal,
            .target_domain_ordinal = control_->channel.target_domain_ordinal,
            .graph_role = static_cast<std::uint32_t>(ticket.graph_role),
            .request_count = static_cast<std::uint32_t>(ticket.request_count),
            .logical_rows_per_request = static_cast<std::uint32_t>(
                ticket.logical_rows_per_request),
            .physical_rows_per_request = static_cast<std::uint32_t>(
                ticket.physical_rows_per_request),
            .draft_depth = ticket.draft_depth,
            .sidecar_depth = ticket.sidecar_depth,
            .stage_count = control_->channel.stage_count,
            .lane_ordinal = control_->channel.lane_ordinal,
            .reserved = {},
        };
        identity.digest = moeOverlayActivationIdentityDigest(identity);
        if (!identity.digest.valid())
        {
            fail("ExpertOverlay activation identity digest is invalid", error);
            return std::nullopt;
        }

        /*
         * Publish every immutable byte and endpoint's Armed identity before the
         * admission release store. Device activation acquires admission.state
         * before it validates either record.
         */
        control_->identity = identity;
        clearStatus(control_->continuation_status);
        clearStatus(control_->follower_status);
        for (MoEOverlayActivationEndpointStatus *endpoint_status :
             {&control_->continuation_status, &control_->follower_status})
        {
            endpoint_status->digest = identity.digest;
            endpoint_status->generation = epoch_generation;
            endpoint_status->code = static_cast<std::uint32_t>(
                MoEOverlayActivationStatusCode::Success);
            endpoint_status->state = static_cast<std::uint32_t>(
                MoEOverlayActivationEndpointState::Armed);
        }

        control_->admission.deadline_ns = deadline_ns;
        control_->admission.observed_timeout_ns = 0u;
        control_->admission.digest = identity.digest;
        control_->admission.code = static_cast<std::uint32_t>(
            MoEOverlayActivationStatusCode::Success);
        atomicValue(control_->admission.last_generation).store(
            epoch_generation, std::memory_order_release);
        atomicValue(control_->admission.state).store(
            static_cast<std::uint32_t>(
                MoEOverlayActivationAdmissionState::Armed),
            std::memory_order_release);
        // This is the sole graph-admission edge. Every identity/status byte is
        // immutable and visible before either endpoint can pass its retained
        // native wait node.
        atomicValue(control_->admission.ready_signal).store(
            kMoEOverlayActivationAdmissionTimeline,
            std::memory_order_release);
        return identity;
    }

    bool MoEOverlayActivationEpochProtocol::activate(
        MoEOverlayActivationEndpoint endpoint,
        const MoEOverlayActivationEpochIdentity &identity,
        std::string *error)
    {
        if (error)
            error->clear();
        if (!validEndpoint(endpoint))
            return fail("ExpertOverlay activation endpoint is invalid", error);
        if (admissionState() != MoEOverlayActivationAdmissionState::Armed)
        {
            return reject(
                endpoint,
                MoEOverlayActivationOperation::Activate,
                MoEOverlayActivationStatusCode::InvalidControl,
                "ExpertOverlay activation endpoint observed an unarmed scheduler slot",
                error);
        }
        if (!matchesActiveIdentity(identity))
        {
            return reject(
                endpoint,
                MoEOverlayActivationOperation::Activate,
                MoEOverlayActivationStatusCode::InvalidIdentity,
                "ExpertOverlay activation endpoint rejected a stale or divergent identity",
                error);
        }
        auto &endpoint_status = status(endpoint);
        if (endpointState(endpoint) != MoEOverlayActivationEndpointState::Armed)
        {
            return reject(
                endpoint,
                MoEOverlayActivationOperation::Activate,
                MoEOverlayActivationStatusCode::OutOfOrder,
                "ExpertOverlay activation endpoint may become active exactly once",
                error);
        }
        publishStatus(
            endpoint,
            MoEOverlayActivationOperation::Activate,
            MoEOverlayActivationStatusCode::Success,
            -1,
            -1,
            0u);
        atomicValue(endpoint_status.state).store(
            static_cast<std::uint32_t>(
                MoEOverlayActivationEndpointState::Active),
            std::memory_order_release);
        return true;
    }

    bool MoEOverlayActivationEpochProtocol::publishDispatch(
        const MoEOverlayActivationEpochIdentity &identity,
        std::uint32_t stage_ordinal,
        std::uint64_t live_rows,
        std::uint64_t live_entries,
        std::uint64_t payload_bytes,
        std::string *error)
    {
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Continuation;
        constexpr auto operation =
            MoEOverlayActivationOperation::PublishDispatch;
        if (!validateActiveEndpoint(endpoint, identity, operation, error))
            return false;
        const std::int32_t layer = modelLayer(stage_ordinal);
        const bool empty_payload =
            live_rows == 0u && live_entries == 0u && payload_bytes == 0u;
        const bool nonempty_payload =
            live_rows != 0u && live_entries >= live_rows &&
            payload_bytes != 0u;
        if (layer < 0 || (!empty_payload && !nonempty_payload))
        {
            return reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                "ExpertOverlay dispatch descriptor has invalid stage or compact-row geometry",
                error);
        }

        auto &endpoint_status = status(endpoint);
        const std::int32_t last_published = acquireValue(
            endpoint_status.last_published_stage);
        if (last_published != previousStage(stage_ordinal))
        {
            return reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::OutOfOrder,
                "ExpertOverlay continuation attempted out-of-order dispatch publication",
                error);
        }
        if (stage_ordinal >= kMoEOverlayActivationBufferCount)
        {
            const std::int32_t required_consumed = static_cast<std::int32_t>(
                stage_ordinal - kMoEOverlayActivationBufferCount);
            if (acquireValue(endpoint_status.last_consumed_stage) <
                required_consumed)
            {
                publishStatus(
                    endpoint,
                    operation,
                    MoEOverlayActivationStatusCode::BufferBusy,
                    static_cast<std::int32_t>(stage_ordinal),
                    layer,
                    0u);
                return fail(
                    "ExpertOverlay dispatch bank still owns an unconsumed return",
                    error);
            }
        }

        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(stage_ordinal);
        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(stage_ordinal));
        if (timeline == 0u)
        {
            return reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::GenerationOverflow,
                "ExpertOverlay dispatch timeline cannot represent this generation/stage",
                error);
        }
        auto &buffer = control_->buffers[bank];
        const std::uint64_t observed = acquireValue(
            buffer.dispatch_signal.value);
        if (observed == kMoEOverlayActivationAbortTimeline ||
            observed >= timeline)
        {
            return reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::StaleGeneration,
                "ExpertOverlay dispatch timeline would regress or overwrite a future publication",
                error);
        }

        if (!canAccumulatePublishedTraffic(
                endpoint_status,
                payload_bytes,
                live_rows,
                live_entries))
        {
            return reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                "ExpertOverlay dispatch traffic evidence overflowed its fixed-width ABI",
                error);
        }

        buffer.dispatch_descriptor = {
            .digest = identity.digest,
            .timeline = timeline,
            .placement_epoch = identity.placement_epoch,
            .live_rows = live_rows,
            .live_entries = live_entries,
            .payload_bytes = payload_bytes,
            .stage_ordinal = stage_ordinal,
            .model_layer_index = layer,
        };
        accumulatePublishedTraffic(
            endpoint_status,
            payload_bytes,
            live_rows,
            live_entries);
        /* Descriptor and packet writes happen-before the release timeline. */
        atomicValue(buffer.dispatch_signal.value).store(
            timeline, std::memory_order_release);
        atomicValue(endpoint_status.last_published_stage).store(
            static_cast<std::int32_t>(stage_ordinal),
            std::memory_order_release);
        publishStatus(
            endpoint,
            operation,
            MoEOverlayActivationStatusCode::Success,
            static_cast<std::int32_t>(stage_ordinal),
            layer,
            timeline);
        return true;
    }

    std::optional<MoEOverlayActivationPayloadDescriptor>
    MoEOverlayActivationEpochProtocol::consumeDispatch(
        const MoEOverlayActivationEpochIdentity &identity,
        std::uint32_t stage_ordinal,
        std::string *error)
    {
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Follower;
        constexpr auto operation =
            MoEOverlayActivationOperation::ConsumeDispatch;
        if (!validateActiveEndpoint(endpoint, identity, operation, error))
            return std::nullopt;
        const std::int32_t layer = modelLayer(stage_ordinal);
        auto &endpoint_status = status(endpoint);
        if (layer < 0 ||
            acquireValue(endpoint_status.last_consumed_stage) !=
                previousStage(stage_ordinal))
        {
            reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::OutOfOrder,
                "ExpertOverlay follower attempted out-of-order dispatch consumption",
                error);
            return std::nullopt;
        }

        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(stage_ordinal);
        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(stage_ordinal));
        auto &buffer = control_->buffers[bank];
        const std::uint64_t observed = acquireValue(
            buffer.dispatch_signal.value);
        if (observed == kMoEOverlayActivationAbortTimeline)
        {
            reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PeerAborted,
                "ExpertOverlay follower observed a continuation abort sentinel",
                error);
            return std::nullopt;
        }
        if (observed < timeline)
        {
            publishStatus(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::NotReady,
                static_cast<std::int32_t>(stage_ordinal),
                layer,
                observed);
            fail("ExpertOverlay dispatch timeline is not ready", error);
            return std::nullopt;
        }
        const MoEOverlayActivationPayloadDescriptor descriptor =
            buffer.dispatch_descriptor;
        if (observed != timeline ||
            !validDescriptor(descriptor, identity, stage_ordinal, timeline) ||
            (descriptor.live_rows == 0u
                 ? descriptor.live_entries != 0u ||
                       descriptor.payload_bytes != 0u
                 : descriptor.live_entries < descriptor.live_rows ||
                       descriptor.payload_bytes == 0u))
        {
            reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                "ExpertOverlay follower rejected mismatched dispatch identity or payload geometry",
                error);
            return std::nullopt;
        }

        atomicValue(endpoint_status.last_consumed_stage).store(
            static_cast<std::int32_t>(stage_ordinal),
            std::memory_order_release);
        publishStatus(
            endpoint,
            operation,
            MoEOverlayActivationStatusCode::Success,
            static_cast<std::int32_t>(stage_ordinal),
            layer,
            timeline);
        return descriptor;
    }

    bool MoEOverlayActivationEpochProtocol::publishReturn(
        const MoEOverlayActivationEpochIdentity &identity,
        std::uint32_t stage_ordinal,
        std::uint64_t payload_bytes,
        std::string *error)
    {
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Follower;
        constexpr auto operation = MoEOverlayActivationOperation::PublishReturn;
        if (!validateActiveEndpoint(endpoint, identity, operation, error))
            return false;
        const std::int32_t layer = modelLayer(stage_ordinal);
        auto &endpoint_status = status(endpoint);
        if (layer < 0 ||
            acquireValue(endpoint_status.last_published_stage) !=
                previousStage(stage_ordinal) ||
            acquireValue(endpoint_status.last_consumed_stage) <
                static_cast<std::int32_t>(stage_ordinal))
        {
            return reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::OutOfOrder,
                "ExpertOverlay follower attempted return publication before exact dispatch consumption",
                error);
        }

        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(stage_ordinal);
        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(stage_ordinal));
        auto &buffer = control_->buffers[bank];
        const std::uint64_t dispatch_observed = acquireValue(
            buffer.dispatch_signal.value);
        const auto dispatch = buffer.dispatch_descriptor;
        if (dispatch_observed != timeline ||
            !validDescriptor(dispatch, identity, stage_ordinal, timeline) ||
            ((payload_bytes == 0u) != (dispatch.live_rows == 0u)))
        {
            return reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                "ExpertOverlay return publication lost its exact dispatch descriptor",
                error);
        }
        const std::uint64_t return_observed = acquireValue(
            buffer.return_signal.value);
        if (return_observed == kMoEOverlayActivationAbortTimeline ||
            return_observed >= timeline)
        {
            return reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::StaleGeneration,
                "ExpertOverlay return timeline would regress or overwrite a future publication",
                error);
        }

        if (!canAccumulatePublishedTraffic(
                endpoint_status,
                payload_bytes,
                dispatch.live_rows,
                /*live_entries=*/0u))
        {
            return reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                "ExpertOverlay return traffic evidence overflowed its fixed-width ABI",
                error);
        }

        buffer.return_descriptor = {
            .digest = identity.digest,
            .timeline = timeline,
            .placement_epoch = identity.placement_epoch,
            .live_rows = dispatch.live_rows,
            .live_entries = 0u,
            .payload_bytes = payload_bytes,
            .stage_ordinal = stage_ordinal,
            .model_layer_index = layer,
        };
        accumulatePublishedTraffic(
            endpoint_status,
            payload_bytes,
            dispatch.live_rows,
            /*live_entries=*/0u);
        atomicValue(buffer.return_signal.value).store(
            timeline, std::memory_order_release);
        atomicValue(endpoint_status.last_published_stage).store(
            static_cast<std::int32_t>(stage_ordinal),
            std::memory_order_release);
        publishStatus(
            endpoint,
            operation,
            MoEOverlayActivationStatusCode::Success,
            static_cast<std::int32_t>(stage_ordinal),
            layer,
            timeline);
        return true;
    }

    std::optional<MoEOverlayActivationPayloadDescriptor>
    MoEOverlayActivationEpochProtocol::consumeReturn(
        const MoEOverlayActivationEpochIdentity &identity,
        std::uint32_t stage_ordinal,
        std::string *error)
    {
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Continuation;
        constexpr auto operation = MoEOverlayActivationOperation::ConsumeReturn;
        if (!validateActiveEndpoint(endpoint, identity, operation, error))
            return std::nullopt;
        const std::int32_t layer = modelLayer(stage_ordinal);
        auto &endpoint_status = status(endpoint);
        if (layer < 0 ||
            acquireValue(endpoint_status.last_consumed_stage) !=
                previousStage(stage_ordinal) ||
            acquireValue(endpoint_status.last_published_stage) <
                static_cast<std::int32_t>(stage_ordinal))
        {
            reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::OutOfOrder,
                "ExpertOverlay continuation attempted out-of-order return consumption",
                error);
            return std::nullopt;
        }

        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(stage_ordinal);
        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(stage_ordinal));
        auto &buffer = control_->buffers[bank];
        const std::uint64_t observed = acquireValue(
            buffer.return_signal.value);
        if (observed == kMoEOverlayActivationAbortTimeline)
        {
            reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PeerAborted,
                "ExpertOverlay continuation observed a follower abort sentinel",
                error);
            return std::nullopt;
        }
        if (observed < timeline)
        {
            publishStatus(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::NotReady,
                static_cast<std::int32_t>(stage_ordinal),
                layer,
                observed);
            fail("ExpertOverlay return timeline is not ready", error);
            return std::nullopt;
        }
        const MoEOverlayActivationPayloadDescriptor descriptor =
            buffer.return_descriptor;
        const auto dispatch = buffer.dispatch_descriptor;
        if (observed != timeline ||
            !validDescriptor(descriptor, identity, stage_ordinal, timeline) ||
            descriptor.live_entries != 0u ||
            descriptor.live_rows != dispatch.live_rows)
        {
            reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                "ExpertOverlay continuation rejected mismatched return identity or row geometry",
                error);
            return std::nullopt;
        }

        atomicValue(endpoint_status.last_consumed_stage).store(
            static_cast<std::int32_t>(stage_ordinal),
            std::memory_order_release);
        publishStatus(
            endpoint,
            operation,
            MoEOverlayActivationStatusCode::Success,
            static_cast<std::int32_t>(stage_ordinal),
            layer,
            timeline);
        return descriptor;
    }

    bool MoEOverlayActivationEpochProtocol::complete(
        MoEOverlayActivationEndpoint endpoint,
        const MoEOverlayActivationEpochIdentity &identity,
        std::string *error)
    {
        constexpr auto operation = MoEOverlayActivationOperation::Complete;
        if (!validEndpoint(endpoint))
            return fail("ExpertOverlay completion endpoint is invalid", error);
        if (!validateActiveEndpoint(endpoint, identity, operation, error))
            return false;
        const std::int32_t final_stage = static_cast<std::int32_t>(
            identity.stage_count - 1u);
        auto &endpoint_status = status(endpoint);
        if (acquireValue(endpoint_status.last_published_stage) != final_stage ||
            acquireValue(endpoint_status.last_consumed_stage) != final_stage)
        {
            return reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::OutOfOrder,
                "ExpertOverlay endpoint cannot complete before every stage round trip",
                error);
        }
        publishStatus(
            endpoint,
            operation,
            MoEOverlayActivationStatusCode::Success,
            final_stage,
            modelLayer(static_cast<std::uint32_t>(final_stage)),
            status(endpoint).observed_timeline);
        atomicValue(endpoint_status.state).store(
            static_cast<std::uint32_t>(
                MoEOverlayActivationEndpointState::Complete),
            std::memory_order_release);
        return true;
    }

    bool MoEOverlayActivationEpochProtocol::abort(
        MoEOverlayActivationEndpoint endpoint,
        const MoEOverlayActivationEpochIdentity &identity,
        MoEOverlayActivationStatusCode code,
        std::string *error)
    {
        if (!validEndpoint(endpoint))
            return fail("ExpertOverlay abort endpoint is invalid", error);
        if (!fatalStatusCode(code))
        {
            return fail(
                "ExpertOverlay explicit abort requires a terminal device status code",
                error);
        }
        if (!matchesActiveIdentity(identity))
        {
            return reject(
                endpoint,
                MoEOverlayActivationOperation::Abort,
                MoEOverlayActivationStatusCode::InvalidIdentity,
                "ExpertOverlay abort identity is stale or divergent",
                error);
        }
        auto &endpoint_status = status(endpoint);
        const auto state = endpointState(endpoint);
        if (state != MoEOverlayActivationEndpointState::Armed &&
            state != MoEOverlayActivationEndpointState::Active)
        {
            return fail(
                "ExpertOverlay endpoint cannot abort from its current lifecycle",
                error);
        }

        publishStatus(
            endpoint,
            MoEOverlayActivationOperation::Abort,
            code,
            -1,
            -1,
            kMoEOverlayActivationAbortTimeline);
        if (endpoint == MoEOverlayActivationEndpoint::Continuation)
        {
            for (auto &buffer : control_->buffers)
            {
                atomicValue(buffer.dispatch_signal.value).store(
                    kMoEOverlayActivationAbortTimeline,
                    std::memory_order_release);
            }
        }
        else
        {
            for (auto &buffer : control_->buffers)
            {
                atomicValue(buffer.return_signal.value).store(
                    kMoEOverlayActivationAbortTimeline,
                    std::memory_order_release);
            }
        }
        atomicValue(endpoint_status.state).store(
            static_cast<std::uint32_t>(
                MoEOverlayActivationEndpointState::Aborted),
            std::memory_order_release);
        return true;
    }

    bool MoEOverlayActivationEpochProtocol::acknowledgePeerAbort(
        MoEOverlayActivationEndpoint endpoint,
        const MoEOverlayActivationEpochIdentity &identity,
        std::string *error)
    {
        if (!validEndpoint(endpoint))
            return fail("ExpertOverlay abort acknowledgement endpoint is invalid", error);
        const auto peer =
            endpoint == MoEOverlayActivationEndpoint::Continuation
                ? MoEOverlayActivationEndpoint::Follower
                : MoEOverlayActivationEndpoint::Continuation;
        if (endpointState(peer) != MoEOverlayActivationEndpointState::Aborted)
        {
            return fail(
                "ExpertOverlay endpoint cannot acknowledge an abort the peer has not published",
                error);
        }
        return abort(
            endpoint,
            identity,
            MoEOverlayActivationStatusCode::PeerAborted,
            error);
    }

    bool MoEOverlayActivationEpochProtocol::markTimedOut(
        std::uint64_t epoch_generation,
        std::uint64_t observed_ns,
        std::string *error)
    {
        if (error)
            error->clear();
        if (admissionState() != MoEOverlayActivationAdmissionState::Armed)
            return fail("ExpertOverlay watchdog observed no armed activation epoch", error);
        if (epoch_generation != control_->identity.epoch_generation)
            return fail("ExpertOverlay watchdog generation is stale or divergent", error);
        if (observed_ns < acquireValue(control_->admission.deadline_ns))
            return fail("ExpertOverlay activation epoch has not reached its watchdog deadline", error);

        control_->admission.observed_timeout_ns = observed_ns;
        atomicValue(control_->admission.code).store(
            static_cast<std::uint32_t>(
                MoEOverlayActivationStatusCode::TimedOut),
            std::memory_order_release);
        atomicValue(control_->admission.state).store(
            static_cast<std::uint32_t>(
                MoEOverlayActivationAdmissionState::Failed),
            std::memory_order_release);
        return true;
    }

    bool MoEOverlayActivationEpochProtocol::reset(
        const MoEOverlayActivationEpochIdentity &identity,
        std::string *error)
    {
        if (error)
            error->clear();
        if (!matchesActiveIdentity(identity))
            return fail("ExpertOverlay reset identity is stale or divergent", error);
        if (endpointState(MoEOverlayActivationEndpoint::Continuation) !=
                MoEOverlayActivationEndpointState::Complete ||
            endpointState(MoEOverlayActivationEndpoint::Follower) !=
                MoEOverlayActivationEndpointState::Complete)
        {
            return fail(
                "ExpertOverlay activation epoch may reset only after both endpoints complete",
                error);
        }

        for (std::uint32_t bank = 0u;
             bank < kMoEOverlayActivationBufferCount;
             ++bank)
        {
            if (bank >= identity.stage_count)
                continue;
            std::uint32_t last_stage = identity.stage_count - 1u;
            if (moeOverlayActivationBufferIndex(last_stage) != bank)
                --last_stage;
            const std::uint64_t expected =
                moeOverlayActivationLeasedTimelineValue(
                    moeOverlayActivationBufferVisit(last_stage));
            if (dispatchTimeline(bank) != expected ||
                returnTimeline(bank) != expected)
            {
                return fail(
                    "ExpertOverlay activation reset observed incomplete final bank timelines",
                    error);
            }
        }

        /*
         * Both exact endpoint terminal events are quiescent now, so no old wait
         * or packet access can remain. Clear the leased graph-node signals and
         * descriptors before admitting another generation, while preserving
         * last_generation as the independent anti-replay authority.
         */
        for (auto &buffer : control_->buffers)
        {
            buffer.dispatch_descriptor = {};
            buffer.return_descriptor = {};
            atomicValue(buffer.dispatch_signal.value).store(
                0u, std::memory_order_release);
            atomicValue(buffer.return_signal.value).store(
                0u, std::memory_order_release);
        }
        // Close admission before clearing identity. A mistakenly resubmitted
        // old graph therefore waits instead of observing a partially reset
        // scheduler record.
        atomicValue(control_->admission.ready_signal).store(
            0u, std::memory_order_release);
        control_->identity = {};
        clearStatus(control_->continuation_status);
        clearStatus(control_->follower_status);
        control_->admission.deadline_ns = 0u;
        control_->admission.observed_timeout_ns = 0u;
        control_->admission.digest = {};
        control_->admission.code = static_cast<std::uint32_t>(
            MoEOverlayActivationStatusCode::Idle);
        atomicValue(control_->admission.state).store(
            static_cast<std::uint32_t>(
                MoEOverlayActivationAdmissionState::Idle),
            std::memory_order_release);
        return true;
    }

    MoEOverlayActivationEpochIdentity
    MoEOverlayActivationEpochProtocol::activeIdentity() const noexcept
    {
        /* Acquire admission before copying the immutable scheduler-owned bytes. */
        (void)admissionState();
        return control_->identity;
    }

    MoEOverlayActivationAdmissionState
    MoEOverlayActivationEpochProtocol::admissionState() const noexcept
    {
        return static_cast<MoEOverlayActivationAdmissionState>(
            acquireValue(control_->admission.state));
    }

    MoEOverlayActivationEndpointState
    MoEOverlayActivationEpochProtocol::endpointState(
        MoEOverlayActivationEndpoint endpoint) const noexcept
    {
        if (!validEndpoint(endpoint))
            return MoEOverlayActivationEndpointState::Aborted;
        return static_cast<MoEOverlayActivationEndpointState>(
            acquireValue(status(endpoint).state));
    }

    MoEOverlayActivationEndpointStatus
    MoEOverlayActivationEpochProtocol::endpointStatus(
        MoEOverlayActivationEndpoint endpoint) const noexcept
    {
        if (!validEndpoint(endpoint))
        {
            MoEOverlayActivationEndpointStatus invalid;
            invalid.state = static_cast<std::uint32_t>(
                MoEOverlayActivationEndpointState::Aborted);
            invalid.code = static_cast<std::uint32_t>(
                MoEOverlayActivationStatusCode::InvalidControl);
            return invalid;
        }
        const auto &source = status(endpoint);
        MoEOverlayActivationEndpointStatus snapshot;
        /*
         * State is the endpoint's release-published ownership edge. Acquire it
         * before copying diagnostics and traffic totals so Complete makes every
         * preceding device/system-memory write visible to this observer.
         */
        snapshot.state = acquireValue(source.state);
        snapshot.digest = source.digest;
        snapshot.generation = source.generation;
        snapshot.observed_timeline = acquireValue(source.observed_timeline);
        snapshot.code = acquireValue(source.code);
        snapshot.operation = acquireValue(source.operation);
        snapshot.last_published_stage = acquireValue(
            source.last_published_stage);
        snapshot.last_consumed_stage = acquireValue(
            source.last_consumed_stage);
        snapshot.last_model_layer = acquireValue(source.last_model_layer);
        snapshot.published_payload_bytes = acquireValue(
            source.published_payload_bytes);
        snapshot.published_live_rows = acquireValue(
            source.published_live_rows);
        snapshot.published_live_entries = acquireValue(
            source.published_live_entries);
        snapshot.published_stage_count = acquireValue(
            source.published_stage_count);
        return snapshot;
    }

    std::optional<MoEOverlayActivationEpochTraffic>
    MoEOverlayActivationEpochProtocol::completedTraffic(
        const MoEOverlayActivationEpochIdentity &identity,
        std::string *error) const
    {
        if (error)
            error->clear();
        if (!matchesActiveIdentity(identity))
        {
            fail(
                "ExpertOverlay traffic snapshot identity is stale or divergent",
                error);
            return std::nullopt;
        }

        /*
         * endpointStatus() acquires state first. Device writers publish every
         * traffic scalar and issue a system fence before their release-store
         * of Complete, so a Complete snapshot owns all preceding totals.
         */
        const auto continuation = endpointStatus(
            MoEOverlayActivationEndpoint::Continuation);
        const auto follower = endpointStatus(
            MoEOverlayActivationEndpoint::Follower);
        const auto complete_for_identity = [&identity](
                                               const auto &status)
        {
            return status.typedState() ==
                       MoEOverlayActivationEndpointState::Complete &&
                   status.typedCode() ==
                       MoEOverlayActivationStatusCode::Success &&
                   status.digest == identity.digest &&
                   status.generation == identity.epoch_generation &&
                   status.published_stage_count == identity.stage_count;
        };
        if (!complete_for_identity(continuation) ||
            !complete_for_identity(follower))
        {
            std::ostringstream diagnostic;
            diagnostic
                << "ExpertOverlay traffic snapshot requires two complete "
                   "endpoint-owned stage totals (expected="
                << identity.stage_count
                << ", continuation={state=" << continuation.state
                << ",code=" << continuation.code
                << ",generation=" << continuation.generation
                << ",published=" << continuation.published_stage_count
                << ",last_published="
                << continuation.last_published_stage
                << ",last_consumed=" << continuation.last_consumed_stage
                << "}, follower={state=" << follower.state
                << ",code=" << follower.code
                << ",generation=" << follower.generation
                << ",published=" << follower.published_stage_count
                << ",last_published=" << follower.last_published_stage
                << ",last_consumed=" << follower.last_consumed_stage
                << "})";
            fail(diagnostic.str(), error);
            return std::nullopt;
        }

        const bool row_symmetric =
            continuation.published_live_rows ==
            follower.published_live_rows;
        const bool entry_geometry_valid =
            follower.published_live_entries == 0u &&
            (continuation.published_live_rows == 0u
                 ? continuation.published_live_entries == 0u
                 : continuation.published_live_entries >=
                       continuation.published_live_rows);
        const bool byte_geometry_valid =
            (continuation.published_payload_bytes == 0u) ==
                (continuation.published_live_rows == 0u) &&
            (follower.published_payload_bytes == 0u) ==
                (follower.published_live_rows == 0u);
        if (!row_symmetric || !entry_geometry_valid ||
            !byte_geometry_valid)
        {
            fail(
                "ExpertOverlay completed traffic totals violate compact dispatch/return geometry",
                error);
            return std::nullopt;
        }

        return MoEOverlayActivationEpochTraffic{
            .dispatch_payload_bytes =
                continuation.published_payload_bytes,
            .return_payload_bytes = follower.published_payload_bytes,
            .dispatch_live_rows = continuation.published_live_rows,
            .return_live_rows = follower.published_live_rows,
            .dispatch_live_entries =
                continuation.published_live_entries,
            .dispatch_stage_count =
                continuation.published_stage_count,
            .return_stage_count = follower.published_stage_count,
        };
    }

    std::uint64_t MoEOverlayActivationEpochProtocol::dispatchTimeline(
        std::uint32_t buffer_index) const noexcept
    {
        if (buffer_index >= kMoEOverlayActivationBufferCount)
            return kMoEOverlayActivationAbortTimeline;
        return acquireValue(
            control_->buffers[buffer_index].dispatch_signal.value);
    }

    std::uint64_t MoEOverlayActivationEpochProtocol::returnTimeline(
        std::uint32_t buffer_index) const noexcept
    {
        if (buffer_index >= kMoEOverlayActivationBufferCount)
            return kMoEOverlayActivationAbortTimeline;
        return acquireValue(
            control_->buffers[buffer_index].return_signal.value);
    }

    bool MoEOverlayActivationEpochProtocol::matchesActiveIdentity(
        const MoEOverlayActivationEpochIdentity &identity) const noexcept
    {
        if (admissionState() != MoEOverlayActivationAdmissionState::Armed ||
            identity != control_->identity || !identity.digest.valid() ||
            identity.digest != moeOverlayActivationIdentityDigest(identity) ||
            identity.channel_nonce != control_->channel.channel_nonce ||
            identity.topology_fingerprint_low !=
                control_->channel.topology_fingerprint_low ||
            identity.topology_fingerprint_high !=
                control_->channel.topology_fingerprint_high ||
            identity.stage_manifest_digest !=
                control_->channel.stage_manifest_digest ||
            identity.source_world_rank !=
                control_->channel.source_world_rank ||
            identity.target_world_rank !=
                control_->channel.target_world_rank ||
            identity.source_participant_id !=
                control_->channel.source_participant_id ||
            identity.target_participant_id !=
                control_->channel.target_participant_id ||
            identity.source_tier_priority !=
                control_->channel.source_tier_priority ||
            identity.target_tier_priority !=
                control_->channel.target_tier_priority ||
            identity.source_domain_ordinal !=
                control_->channel.source_domain_ordinal ||
            identity.target_domain_ordinal !=
                control_->channel.target_domain_ordinal ||
            identity.stage_count != control_->channel.stage_count ||
            identity.lane_ordinal != control_->channel.lane_ordinal ||
            identity.epoch_generation == 0u ||
            identity.epoch_generation > kMoEOverlayActivationMaxGeneration)
        {
            return false;
        }
        return true;
    }

    std::int32_t MoEOverlayActivationEpochProtocol::modelLayer(
        std::uint32_t stage_ordinal) const noexcept
    {
        if (stage_ordinal >= config_.model_layer_indices.size())
            return -1;
        return config_.model_layer_indices[stage_ordinal];
    }

    MoEOverlayActivationEndpointStatus &
    MoEOverlayActivationEpochProtocol::status(
        MoEOverlayActivationEndpoint endpoint) noexcept
    {
        return endpoint == MoEOverlayActivationEndpoint::Follower
                   ? control_->follower_status
                   : control_->continuation_status;
    }

    const MoEOverlayActivationEndpointStatus &
    MoEOverlayActivationEpochProtocol::status(
        MoEOverlayActivationEndpoint endpoint) const noexcept
    {
        return endpoint == MoEOverlayActivationEndpoint::Follower
                   ? control_->follower_status
                   : control_->continuation_status;
    }

    void MoEOverlayActivationEpochProtocol::publishStatus(
        MoEOverlayActivationEndpoint endpoint,
        MoEOverlayActivationOperation operation,
        MoEOverlayActivationStatusCode code,
        std::int32_t stage_ordinal,
        std::int32_t model_layer,
        std::uint64_t timeline) noexcept
    {
        auto &endpoint_status = status(endpoint);
        atomicValue(endpoint_status.observed_timeline).store(
            timeline, std::memory_order_release);
        atomicValue(endpoint_status.last_model_layer).store(
            model_layer, std::memory_order_release);
        /* `stage_ordinal` is diagnostic here; ownership counters update separately. */
        (void)stage_ordinal;
        atomicValue(endpoint_status.operation).store(
            static_cast<std::uint32_t>(operation),
            std::memory_order_release);
        atomicValue(endpoint_status.code).store(
            static_cast<std::uint32_t>(code),
            std::memory_order_release);
    }

    bool MoEOverlayActivationEpochProtocol::reject(
        MoEOverlayActivationEndpoint endpoint,
        MoEOverlayActivationOperation operation,
        MoEOverlayActivationStatusCode code,
        std::string message,
        std::string *error)
    {
        auto &endpoint_status = status(endpoint);
        publishStatus(
            endpoint,
            operation,
            code,
            -1,
            -1,
            kMoEOverlayActivationAbortTimeline);
        if (endpoint == MoEOverlayActivationEndpoint::Continuation)
        {
            for (auto &buffer : control_->buffers)
            {
                atomicValue(buffer.dispatch_signal.value).store(
                    kMoEOverlayActivationAbortTimeline,
                    std::memory_order_release);
            }
        }
        else
        {
            for (auto &buffer : control_->buffers)
            {
                atomicValue(buffer.return_signal.value).store(
                    kMoEOverlayActivationAbortTimeline,
                    std::memory_order_release);
            }
        }
        atomicValue(endpoint_status.state).store(
            static_cast<std::uint32_t>(
                MoEOverlayActivationEndpointState::Aborted),
            std::memory_order_release);
        return fail(std::move(message), error);
    }

    bool MoEOverlayActivationEpochProtocol::validateActiveEndpoint(
        MoEOverlayActivationEndpoint endpoint,
        const MoEOverlayActivationEpochIdentity &identity,
        MoEOverlayActivationOperation operation,
        std::string *error)
    {
        if (!validEndpoint(endpoint))
            return fail("ExpertOverlay activation endpoint is invalid", error);
        if (admissionState() != MoEOverlayActivationAdmissionState::Armed)
        {
            return reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::InvalidControl,
                "ExpertOverlay device operation observed an unarmed or failed scheduler slot",
                error);
        }
        if (!matchesActiveIdentity(identity))
        {
            return reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::InvalidIdentity,
                "ExpertOverlay device operation rejected a stale or divergent identity",
                error);
        }
        if (endpointState(endpoint) != MoEOverlayActivationEndpointState::Active)
        {
            return reject(
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::OutOfOrder,
                "ExpertOverlay device operation requires one active endpoint",
                error);
        }
        return true;
    }

    bool MoEOverlayActivationEpochProtocol::validDescriptor(
        const MoEOverlayActivationPayloadDescriptor &descriptor,
        const MoEOverlayActivationEpochIdentity &identity,
        std::uint32_t stage_ordinal,
        std::uint64_t timeline) const noexcept
    {
        const bool empty_payload =
            descriptor.live_rows == 0u && descriptor.payload_bytes == 0u;
        const bool nonempty_payload =
            descriptor.live_rows != 0u && descriptor.payload_bytes != 0u;
        return descriptor.digest == identity.digest &&
               descriptor.timeline == timeline &&
               descriptor.placement_epoch == identity.placement_epoch &&
               (empty_payload || nonempty_payload) &&
               descriptor.stage_ordinal == stage_ordinal &&
               descriptor.model_layer_index == modelLayer(stage_ordinal);
    }

    void MoEOverlayActivationEpochProtocol::clearStatus(
        MoEOverlayActivationEndpointStatus &endpoint_status) noexcept
    {
        endpoint_status = MoEOverlayActivationEndpointStatus{};
    }
} // namespace llaminar2
