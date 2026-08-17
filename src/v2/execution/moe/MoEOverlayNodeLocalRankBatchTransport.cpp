/**
 * @file MoEOverlayNodeLocalRankBatchTransport.cpp
 * @brief In-place shared-row implementation of the node-local MoE channel.
 */

#include "MoEOverlayNodeLocalRankBatchTransport.h"

#include "collective/CollectiveTimeoutPolicy.h"
#include "interfaces/IMPIContext.h"
#include "interfaces/IMPITopology.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace llaminar2
{
    namespace
    {
        constexpr uint64_t kSharedChannelMagic = 0x5a43574f454d4c4cULL;
        constexpr uint32_t kSharedChannelVersion = 4;
        constexpr size_t kCacheLine = 64;

        enum class PublicationState : uint32_t
        {
            Empty = 0,
            Ready = 1,
            Aborted = 2,
        };

        static_assert(std::atomic<uint32_t>::is_always_lock_free);
        static_assert(std::is_trivially_copyable_v<MoEOverlayRankBatchKey>);

        /** @brief Immutable mapping identity followed by two isolated states. */
        struct alignas(kCacheLine) SharedChannelHeader
        {
            uint64_t magic = 0;
            uint32_t version = 0;
            uint32_t participant_count = 0;
            uint32_t activation_family_count = 0;
            uint32_t reserved0 = 0;
            uint64_t mapping_bytes = 0;
            uint64_t channel_hash = 0;
            uint64_t max_rows_per_participant = 0;
            uint64_t max_entries_per_participant = 0;
            int32_t d_model = 0;
            int32_t top_k = 0;
            std::atomic<uint32_t> initialized{0};
            uint32_t reserved1 = 0;
            /** Source rank has first-touched every return-consumer page. */
            alignas(kCacheLine) std::atomic<uint32_t>
                source_payload_pages_ready{0};
            /** Target rank has first-touched every dispatch-consumer page. */
            alignas(kCacheLine) std::atomic<uint32_t>
                target_payload_pages_ready{0};
        };

        /** @brief One producer/consumer publication state on its own line. */
        struct alignas(kCacheLine) SharedPublication
        {
            std::atomic<uint32_t> state{0};
            uint32_t reserved = 0;
            MoEOverlayRankBatchKey key{};
        };

        /** @brief Scalar metadata for one participant's in-place arrays. */
        struct alignas(kCacheLine) SharedParticipantControl
        {
            int32_t participant_id = -1;
            int32_t dispatch_source_participant = -1;
            int32_t dispatch_target_participant = -1;
            int32_t dispatch_d_model = 0;
            int32_t dispatch_top_k = 0;
            uint32_t reserved0 = 0;
            uint64_t dispatch_residency_epoch = 0;
            uint64_t dispatch_live_rows = 0;
            uint64_t dispatch_live_entries = 0;

            int32_t return_source_participant = -1;
            int32_t return_target_participant = -1;
            int32_t return_d_model = 0;
            uint32_t reserved1 = 0;
            uint64_t return_residency_epoch = 0;
            uint64_t return_live_rows = 0;
        };

        /** @brief Byte offsets of one participant's two shared row families. */
        struct ParticipantLayout
        {
            size_t dispatch_begin = 0;
            size_t row_ids = 0;
            size_t entry_offsets = 0;
            size_t expert_ids = 0;
            size_t route_weights = 0;
            size_t hidden_rows = 0;
            size_t dispatch_end = 0;
            size_t return_begin = 0;
            size_t return_row_ids = 0;
            size_t output_rows = 0;
            size_t return_end = 0;
        };

        size_t alignUp(size_t value, size_t alignment)
        {
            if (alignment == 0 || value >
                                      std::numeric_limits<size_t>::max() -
                                          (alignment - 1u))
            {
                throw std::overflow_error(
                    "MoE shared activation channel alignment overflow");
            }
            return (value + alignment - 1u) & ~(alignment - 1u);
        }

        size_t checkedAdd(size_t lhs, size_t rhs, const char *what)
        {
            if (rhs > std::numeric_limits<size_t>::max() - lhs)
                throw std::overflow_error(
                    std::string("MoE shared activation channel overflow: ") + what);
            return lhs + rhs;
        }

        size_t checkedMultiply(size_t lhs, size_t rhs, const char *what)
        {
            if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
                throw std::overflow_error(
                    std::string("MoE shared activation channel overflow: ") + what);
            return lhs * rhs;
        }

        uint64_t fnv1a(uint64_t hash, const void *data, size_t bytes) noexcept
        {
            const auto *cursor = static_cast<const unsigned char *>(data);
            for (size_t index = 0; index < bytes; ++index)
            {
                hash ^= cursor[index];
                hash *= 1099511628211ULL;
            }
            return hash;
        }

        uint64_t channelHash(
            uint64_t node_namespace,
            const MoEOverlayRankBatchTransportConfig &config) noexcept
        {
            uint64_t hash = 1469598103934665603ULL;
            const auto mix = [&](const auto &value)
            {
                hash = fnv1a(hash, &value, sizeof(value));
            };
            mix(node_namespace);
            mix(config.source_world_rank);
            mix(config.target_world_rank);
            mix(config.tier_index);
            mix(config.domain_ordinal);
            mix(config.max_rows_per_participant);
            mix(config.max_entries_per_participant);
            mix(config.d_model);
            mix(config.top_k);
            mix(config.transaction_topology.workspace_generation);
            mix(config.transaction_topology.topology_fingerprint_low);
            mix(config.transaction_topology.topology_fingerprint_high);
            mix(config.source_endpoint.world_rank);
            mix(config.source_endpoint.participant_id);
            mix(config.source_endpoint.tier_priority);
            mix(config.source_endpoint.domain_ordinal);
            mix(config.target_tier_priority);
            for (const int participant : config.workspace->participantIds())
                mix(participant);
            const size_t family_count =
                config.activation_graph_families.size();
            mix(family_count);
            for (const auto &family : config.activation_graph_families)
            {
                mix(family.graph_role_mask);
                const size_t stage_count = family.model_layer_indices.size();
                mix(stage_count);
                for (const std::int32_t layer : family.model_layer_indices)
                    mix(layer);
            }
            hash = fnv1a(
                hash,
                config.channel_identity.data(),
                config.channel_identity.size());
            return hash == 0 ? 1 : hash;
        }

        std::string sharedMemoryName(
            uint64_t node_namespace,
            uint64_t channel_hash)
        {
            std::ostringstream stream;
            stream << "/llaminar_moe_zc_" << std::hex
                   << node_namespace << '_' << channel_hash;
            return stream.str();
        }

        void cpuRelax() noexcept
        {
#if defined(__x86_64__) || defined(__i386__)
            _mm_pause();
#else
            std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
        }

        bool sameRoundTrip(
            const MoEOverlayRankBatchKey &dispatch,
            const MoEOverlayRankBatchKey &returned) noexcept
        {
            return dispatch.generation_id == returned.generation_id &&
                   dispatch.step_id == returned.step_id &&
                   dispatch.key_namespace == returned.key_namespace &&
                   dispatch.histogram_source == returned.histogram_source &&
                   dispatch.mtp_depth == returned.mtp_depth &&
                   dispatch.layer_idx == returned.layer_idx &&
                   dispatch.tier_idx == returned.tier_idx &&
                   dispatch.domain_ordinal == returned.domain_ordinal &&
                   dispatch.source_world_rank == returned.source_world_rank &&
                   dispatch.target_world_rank == returned.target_world_rank &&
                   dispatch.direction ==
                       MoEOverlayCollectiveDirection::Dispatch &&
                   returned.direction ==
                       MoEOverlayCollectiveDirection::ReturnReduce;
        }

        /** @brief Guard one process-local transport object against concurrent use. */
        class InFlightGuard final
        {
        public:
            explicit InFlightGuard(std::atomic_flag &flag) noexcept
                : flag_(flag),
                  acquired_(!flag_.test_and_set(std::memory_order_acquire))
            {
            }
            ~InFlightGuard()
            {
                if (acquired_)
                    flag_.clear(std::memory_order_release);
            }
            bool acquired() const noexcept { return acquired_; }

        private:
            std::atomic_flag &flag_;
            bool acquired_ = false;
        };

        const char *phaseFor(const MoEOverlayRankBatchKey &key) noexcept
        {
            return key.histogram_source == ExpertHistogramSource::PrefillChunk
                       ? "prefill"
                       : (key.histogram_source ==
                                  ExpertHistogramSource::GroupedVerifier
                              ? "grouped_verifier"
                              : "decode");
        }

        void recordSharedTransaction(
            const MoEOverlayRankBatchKey &key,
            size_t bytes,
            const char *endpoint_role,
            uint64_t wait_ns,
            uint64_t total_ns)
        {
            if (PerfStatsCollector::isDomainEnabled("forward_graph"))
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    key.direction == MoEOverlayCollectiveDirection::Dispatch
                        ? "moe_overlay_rank_batch_dispatch_transactions"
                        : "moe_overlay_rank_batch_return_transactions",
                    1.0,
                    "moe_overlay",
                    "node_local_shared_rows",
                    {{"bytes", std::to_string(bytes)},
                     {"endpoint_role", endpoint_role},
                     {"layer", std::to_string(key.layer_idx)},
                     {"source_world_rank", std::to_string(key.source_world_rank)},
                     {"target_world_rank", std::to_string(key.target_world_rank)},
                     {"transport", "node_local_shared_rows"}});
            }
            if (!PerfStatsCollector::isDomainEnabled("moe_overlay_transport"))
                return;
            const PerfStatsCollector::Tags tags{
                {"bytes", std::to_string(bytes)},
                {"direction", llaminar2::toString(key.direction)},
                {"domain_ordinal", std::to_string(key.domain_ordinal)},
                {"endpoint_role", endpoint_role},
                {"generation", std::to_string(key.generation_id)},
                {"layer", std::to_string(key.layer_idx)},
                {"logical_step", std::to_string(key.step_id)},
                {"source_world_rank", std::to_string(key.source_world_rank)},
                {"target_world_rank", std::to_string(key.target_world_rank)},
                {"tier", std::to_string(key.tier_idx)},
                {"transport", "node_local_shared_rows"}};
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_transport",
                "rank_batch_total",
                total_ns,
                phaseFor(key),
                "node_local_shared_rows",
                tags);
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_transport",
                "rank_batch_wire_wait",
                wait_ns,
                phaseFor(key),
                "node_local_shared_rows",
                tags);
            PerfStatsCollector::addCounter(
                "moe_overlay_transport",
                "rank_batch_zero_copy_publications",
                1.0,
                phaseFor(key),
                "node_local_shared_rows",
                tags);
        }

    } // namespace

    /**
     * @brief Own one endpoint-private grant at a stable CPU or GPU address.
     *
     * Construction is setup-only. GPU storage is allocated through
     * TransferEngine without publishing tensor authority because stage zero
     * overwrites every grant byte before any later stage consumes it. The
     * tensor owner exists solely to provide canonical allocation teardown after
     * every retained executable referencing the address has been destroyed.
     */
    class MoEOverlayNodeLocalRankBatchTransport::DeviceGrantStorage final
    {
    public:
        /** @brief Allocate one grant for an exact lane/family/device identity. */
        DeviceGrantStorage(
            DeviceId device,
            int participant_id,
            size_t graph_family_ordinal)
            : device_(device),
              participant_id_(participant_id),
              graph_family_ordinal_(graph_family_ordinal)
        {
            if (!device_.is_valid() || participant_id_ < 0)
            {
                throw std::invalid_argument(
                    "ExpertOverlay device grant requires a valid endpoint and payload identity");
            }
            if (device_.is_gpu())
            {
                constexpr size_t words =
                    sizeof(MoEOverlayActivationDeviceEpochGrant) /
                    sizeof(std::int32_t);
                static_assert(
                    words * sizeof(std::int32_t) ==
                    sizeof(MoEOverlayActivationDeviceEpochGrant));
                gpu_storage_ =
                    std::make_unique<INT32Tensor>(std::vector<size_t>{words});
                TransferEngine::allocateDeviceStorage(
                    gpu_storage_.get(), device_);
                grant_ = static_cast<MoEOverlayActivationDeviceEpochGrant *>(
                    gpu_storage_->gpu_data_ptr());

            }
            else
            {
                cpu_storage_ =
                    std::make_unique<MoEOverlayActivationDeviceEpochGrant>();
                grant_ = cpu_storage_.get();
            }
            if (!grant_)
            {
                throw std::runtime_error(
                    "ExpertOverlay device grant allocation returned a null address");
            }
        }

        /** @return Whether this owner exactly matches a requested lane. */
        [[nodiscard]] bool matches(
            DeviceId device,
            int participant_id,
            size_t graph_family_ordinal) const noexcept
        {
            return device_ == device && participant_id_ == participant_id &&
                   graph_family_ordinal_ == graph_family_ordinal;
        }

        /** @return Stable endpoint-local address embedded during graph capture. */
        [[nodiscard]] MoEOverlayActivationDeviceEpochGrant *grant() const noexcept
        {
            return grant_;
        }

    private:
        DeviceId device_ = DeviceId::invalid(); ///< Exact allocation device.
        int participant_id_ = -1; ///< Logical packet lane.
        size_t graph_family_ordinal_ = 0u; ///< Main/MTP graph family.
        std::unique_ptr<INT32Tensor> gpu_storage_; ///< Canonical GPU allocation owner.
        std::unique_ptr<MoEOverlayActivationDeviceEpochGrant> cpu_storage_; ///< CPU authority.
        MoEOverlayActivationDeviceEpochGrant *grant_ = nullptr; ///< Stable captured address.
    };

    std::vector<MoEOverlayActivationGraphFamilyManifest>
    makeMoEOverlayActivationGraphFamilyManifests(
        const MoEOverlayInferenceGraphFamilyIdentity &graph_family)
    {
        if (!graph_family.valid())
        {
            throw std::invalid_argument(
                "MoE activation channel requires a valid retained graph family");
        }
        const auto role_bit = [](MoEOverlayInferenceGraphRole role)
        {
            return std::uint32_t{1}
                   << static_cast<std::uint32_t>(role);
        };

        std::vector<MoEOverlayActivationGraphFamilyManifest> manifests;
        manifests.reserve(1u + graph_family.mtp_source_layers.size());
        MoEOverlayActivationGraphFamilyManifest main;
        main.graph_role_mask =
            role_bit(MoEOverlayInferenceGraphRole::MainPrefill) |
            role_bit(MoEOverlayInferenceGraphRole::MainDecode) |
            role_bit(MoEOverlayInferenceGraphRole::MTPGroupedVerifier);
        main.model_layer_indices.reserve(
            static_cast<size_t>(graph_family.main_layer_count));
        for (int layer = 0; layer < graph_family.main_layer_count; ++layer)
            main.model_layer_indices.push_back(layer);
        manifests.push_back(std::move(main));

        for (const int source_layer : graph_family.mtp_source_layers)
        {
            manifests.push_back({
                .graph_role_mask =
                    role_bit(MoEOverlayInferenceGraphRole::MTPDraft),
                .model_layer_indices = {source_layer},
            });
        }
        return manifests;
    }

    class MoEOverlayNodeLocalRankBatchTransport::Mapping final
    {
    public:
        explicit Mapping(const MoEOverlayRankBatchTransportConfig &config)
            : participants_(config.workspace->participantIds()),
              max_rows_(config.max_rows_per_participant),
              max_entries_(config.max_entries_per_participant),
              d_model_(config.d_model),
              top_k_(config.top_k),
              activation_family_count_(
                  config.activation_graph_families.size())
        {
            const auto *const topology = config.mpi_ctx->topology();
            if (!topology ||
                topology->node_shared_memory_namespace() == 0)
            {
                throw std::runtime_error(
                    "Node-local MoE channel requires a fresh physical-topology namespace");
            }
            channel_hash_ = channelHash(
                topology->node_shared_memory_namespace(), config);
            shm_name_ = sharedMemoryName(
                topology->node_shared_memory_namespace(), channel_hash_);
            computeLayout();
            mapOrCreate(config);
        }

        ~Mapping()
        {
            if (base_)
            {
                ::munmap(base_, mapping_bytes_);
                base_ = nullptr;
            }
            if (fd_ >= 0)
            {
                ::close(fd_);
                fd_ = -1;
            }
        }

        Mapping(const Mapping &) = delete;
        Mapping &operator=(const Mapping &) = delete;

        size_t bytes() const noexcept { return mapping_bytes_; }
        void *baseAddress() const noexcept { return base_; }
        const std::string &name() const noexcept { return shm_name_; }
        SharedPublication &dispatchPublication() const noexcept
        {
            return *dispatch_publication_;
        }
        SharedPublication &returnPublication() const noexcept
        {
            return *return_publication_;
        }

        bool waitFor(
            SharedPublication &publication,
            PublicationState expected,
            const char *operation,
            uint64_t *wait_ns,
            std::string *error) const
        {
            const auto begin = std::chrono::steady_clock::now();
            const auto deadline =
                begin + std::chrono::milliseconds(
                            collective_timeout_policy::
                                kDefaultCollectiveTimeoutMs);
            uint64_t spins = 0;
            while (true)
            {
                const auto state = static_cast<PublicationState>(
                    publication.state.load(std::memory_order_acquire));
                if (state == expected)
                {
                    if (wait_ns)
                    {
                        *wait_ns = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - begin)
                                .count());
                    }
                    return true;
                }
                if (state == PublicationState::Aborted)
                {
                    if (error)
                        *error = std::string(operation) +
                                 " observed an aborted shared channel";
                    return false;
                }
                if ((++spins & 0x3fffU) == 0 &&
                    std::chrono::steady_clock::now() >= deadline)
                {
                    if (error)
                    {
                        *error = std::string(operation) +
                                 " timed out after " +
                                 std::to_string(
                                     collective_timeout_policy::
                                         kDefaultCollectiveTimeoutMs) +
                                 "ms";
                    }
                    return false;
                }
                cpuRelax();
            }
        }

        MoEOverlaySparseRows dispatchRows(int participant_id) const
        {
            const size_t index = participantIndex(participant_id);
            const ParticipantLayout &layout = layouts_[index];
            MoEOverlaySparseRows rows;
            rows.target_participant = participant_id;
            rows.d_model = d_model_;
            rows.top_k = top_k_;
            rows.row_capacity = max_rows_;
            rows.entry_capacity = max_entries_;
            rows.row_ids_host = at<int32_t>(layout.row_ids);
            rows.entry_offsets_host = at<int32_t>(layout.entry_offsets);
            rows.expert_ids_host = at<int32_t>(layout.expert_ids);
            rows.route_weights_host = at<float>(layout.route_weights);
            rows.hidden_rows_fp32 = at<float>(layout.hidden_rows);
            return rows;
        }

        MoEOverlayReturnRows returnRows(int participant_id) const
        {
            const size_t index = participantIndex(participant_id);
            const ParticipantLayout &layout = layouts_[index];
            MoEOverlayReturnRows rows;
            rows.source_participant = participant_id;
            rows.d_model = d_model_;
            rows.row_capacity = max_rows_;
            rows.row_ids_host = at<int32_t>(layout.return_row_ids);
            rows.output_rows_fp32 = at<float>(layout.output_rows);
            return rows;
        }

        /** @return Shared activation-only physical-row matrix. */
        float *sharedActivationHiddenRows() const noexcept
        {
            return at<float>(shared_dispatch_hidden_rows_offset_);
        }

        SharedParticipantControl &control(size_t index) const
        {
            return controls_[index];
        }

        MoEOverlayActivationEpochControl &activationControl(
            int participant_id,
            size_t graph_family_ordinal) const
        {
            if (graph_family_ordinal >= activation_family_count_)
            {
                throw std::out_of_range(
                    "MoE shared activation graph-family ordinal is outside immutable topology");
            }
            const size_t participant_index = participantIndex(participant_id);
            return activation_controls_[
                graph_family_ordinal * participants_.size() +
                participant_index];
        }

        size_t activationFamilyCount() const noexcept
        {
            return activation_family_count_;
        }

        size_t offsetOf(const void *address, size_t bytes) const
        {
            if (!base_ || !address)
            {
                throw std::out_of_range(
                    "MoE shared activation alias requires a live mapped address");
            }
            const auto base = reinterpret_cast<std::uintptr_t>(base_);
            const auto value = reinterpret_cast<std::uintptr_t>(address);
            if (value < base)
            {
                throw std::out_of_range(
                    "MoE shared activation alias precedes the mapped region");
            }
            const size_t offset = static_cast<size_t>(value - base);
            if (offset > mapping_bytes_ || bytes > mapping_bytes_ - offset)
            {
                throw std::out_of_range(
                    "MoE shared activation alias range exceeds the mapped region");
            }
            return offset;
        }

        size_t participantIndex(int participant_id) const
        {
            const auto found = std::lower_bound(
                participants_.begin(), participants_.end(), participant_id);
            if (found == participants_.end() || *found != participant_id)
                throw std::out_of_range(
                    "MoE shared channel participant is outside immutable topology");
            return static_cast<size_t>(found - participants_.begin());
        }

        bool dispatchPointersMatch(
            int participant_id,
            const MoEOverlaySparseRows &rows) const
        {
            const auto expected = dispatchRows(participant_id);
            return rows.row_capacity == expected.row_capacity &&
                   rows.entry_capacity == expected.entry_capacity &&
                   rows.row_ids_host == expected.row_ids_host &&
                   rows.entry_offsets_host == expected.entry_offsets_host &&
                   rows.expert_ids_host == expected.expert_ids_host &&
                   rows.route_weights_host == expected.route_weights_host &&
                   rows.hidden_rows_fp32 == expected.hidden_rows_fp32;
        }

        bool returnPointersMatch(
            int participant_id,
            const MoEOverlayReturnRows &rows) const
        {
            const auto expected = returnRows(participant_id);
            return rows.row_capacity == expected.row_capacity &&
                   rows.row_ids_host == expected.row_ids_host &&
                   rows.output_rows_fp32 == expected.output_rows_fp32;
        }

        void unlinkName() noexcept
        {
            if (!shm_name_.empty() &&
                ::shm_unlink(shm_name_.c_str()) != 0 && errno != ENOENT)
            {
                LOG_WARN("MoE shared activation channel could not unlink "
                         << shm_name_ << ": " << std::strerror(errno));
            }
        }

    private:
        /*
         * The transport exposes the exact initialization recipe through its
         * typed mapped-activation capability. Keep nonce derivation private to
         * Mapping while allowing that owning transport to return the recipe.
         */
        friend class MoEOverlayNodeLocalRankBatchTransport;

        template <typename T>
        T *at(size_t offset) const noexcept
        {
            return reinterpret_cast<T *>(
                static_cast<std::byte *>(base_) + offset);
        }

        size_t appendRegion(size_t bytes, size_t alignment = kCacheLine)
        {
            size_t offset = alignUp(mapping_bytes_, alignment);
            mapping_bytes_ = checkedAdd(offset, bytes, "payload region");
            return offset;
        }

        void computeLayout()
        {
            const long raw_page_size = ::sysconf(_SC_PAGESIZE);
            if (raw_page_size <= 0 ||
                (static_cast<size_t>(raw_page_size) &
                 (static_cast<size_t>(raw_page_size) - 1u)) != 0u)
            {
                throw std::runtime_error(
                    "MoE shared activation channel requires a power-of-two system page size");
            }
            page_size_ = static_cast<size_t>(raw_page_size);
            const size_t participant_count = participants_.size();
            mapping_bytes_ = sizeof(SharedChannelHeader);
            dispatch_publication_offset_ = appendRegion(
                sizeof(SharedPublication));
            return_publication_offset_ = appendRegion(
                sizeof(SharedPublication));
            controls_offset_ = appendRegion(
                checkedMultiply(
                    participant_count,
                    sizeof(SharedParticipantControl),
                    "participant controls"));
            activation_controls_offset_ = appendRegion(
                checkedMultiply(
                    checkedMultiply(
                        participant_count,
                        activation_family_count_,
                        "activation lane count"),
                    sizeof(MoEOverlayActivationEpochControl),
                    "activation epoch controls"));

            const size_t row_id_bytes = checkedMultiply(
                max_rows_, sizeof(int32_t), "row ids");
            const size_t entry_offset_bytes = checkedMultiply(
                checkedAdd(max_rows_, 1u, "entry offset count"),
                sizeof(int32_t),
                "entry offsets");
            const size_t expert_bytes = checkedMultiply(
                max_entries_, sizeof(int32_t), "expert ids");
            const size_t weight_bytes = checkedMultiply(
                max_entries_, sizeof(float), "route weights");
            const size_t activation_bytes = checkedMultiply(
                checkedMultiply(max_rows_, static_cast<size_t>(d_model_),
                                "activation elements"),
                sizeof(float),
                "activation bytes");

            /* The physical continuation activation is identical for every
             * participant behind this rank-pair channel. Isolate one matrix on
             * target-owned pages so a captured bulk path publishes it once and
             * lane-local CSR metadata merely selects rows from it. */
            mapping_bytes_ = alignUp(mapping_bytes_, page_size_);
            shared_dispatch_begin_ = mapping_bytes_;
            shared_dispatch_hidden_rows_offset_ =
                appendRegion(activation_bytes);
            mapping_bytes_ = alignUp(mapping_bytes_, page_size_);
            shared_dispatch_end_ = mapping_bytes_;

            layouts_.resize(participant_count);
            for (ParticipantLayout &layout : layouts_)
            {
                /* Dispatch and return occupy disjoint page families so the
                 * endpoint that synchronously reads each direction can own its
                 * NUMA placement. GPU writes are posted; GPU reads otherwise
                 * pay the full remote-socket UPI latency on every cache line. */
                mapping_bytes_ = alignUp(mapping_bytes_, page_size_);
                layout.dispatch_begin = mapping_bytes_;
                layout.row_ids = appendRegion(row_id_bytes);
                layout.entry_offsets = appendRegion(entry_offset_bytes);
                layout.expert_ids = appendRegion(expert_bytes);
                layout.route_weights = appendRegion(weight_bytes);
                layout.hidden_rows = appendRegion(activation_bytes);
                mapping_bytes_ = alignUp(mapping_bytes_, page_size_);
                layout.dispatch_end = mapping_bytes_;
                layout.return_begin = mapping_bytes_;
                layout.return_row_ids = appendRegion(row_id_bytes);
                layout.output_rows = appendRegion(activation_bytes);
                mapping_bytes_ = alignUp(mapping_bytes_, page_size_);
                layout.return_end = mapping_bytes_;
            }
            mapping_bytes_ = alignUp(mapping_bytes_, page_size_);
        }

        /**
         * @brief First-touch one direction on its synchronous consumer socket.
         *
         * CUDA/HIP host registration pins the complete mapping and may fault
         * every sparse payload page from whichever rank registers first. That
         * defeats ordinary producer first-touch and was observed to place the
         * entire CUDA-to-ROCm channel on NUMA node zero. Both ranks therefore
         * touch disjoint page-aligned directions before either constructor may
         * register the mapping with a GPU driver. The rank launcher/topology
         * owns the current memory policy; no backend or socket number is
         * hard-coded here.
         *
         * @param config Immutable rank-pair topology and local-rank authority.
         * @throws std::runtime_error if the peer does not complete placement
         *         within the standard collective timeout.
         */
        void placeDirectionalPayloadPages(
            const MoEOverlayRankBatchTransportConfig &config)
        {
            const bool source =
                config.mpi_ctx->rank() == config.source_world_rank;
            const bool target =
                config.mpi_ctx->rank() == config.target_world_rank;
            if (source == target)
            {
                throw std::logic_error(
                    "MoE shared activation payload placement requires exactly one local endpoint role");
            }

            size_t touched_bytes = 0u;
            if (target)
            {
                if (shared_dispatch_begin_ % page_size_ != 0u ||
                    shared_dispatch_end_ % page_size_ != 0u ||
                    shared_dispatch_begin_ >= shared_dispatch_end_ ||
                    shared_dispatch_end_ > mapping_bytes_)
                {
                    throw std::logic_error(
                        "MoE shared activation physical payload range is not page isolated");
                }
                std::memset(
                    static_cast<std::byte *>(base_) +
                        shared_dispatch_begin_,
                    0,
                    shared_dispatch_end_ - shared_dispatch_begin_);
                touched_bytes = checkedAdd(
                    touched_bytes,
                    shared_dispatch_end_ - shared_dispatch_begin_,
                    "shared dispatch first-touch bytes");
            }
            for (const ParticipantLayout &layout : layouts_)
            {
                const size_t begin =
                    source ? layout.return_begin : layout.dispatch_begin;
                const size_t end =
                    source ? layout.return_end : layout.dispatch_end;
                if (begin % page_size_ != 0u || end % page_size_ != 0u ||
                    begin >= end || end > mapping_bytes_)
                {
                    throw std::logic_error(
                        "MoE shared activation directional payload range is not page isolated");
                }
                std::memset(
                    static_cast<std::byte *>(base_) + begin,
                    0,
                    end - begin);
                touched_bytes = checkedAdd(
                    touched_bytes,
                    end - begin,
                    "directional first-touch bytes");
            }

            auto &local_ready =
                source ? header_->source_payload_pages_ready
                       : header_->target_payload_pages_ready;
            auto &peer_ready =
                source ? header_->target_payload_pages_ready
                       : header_->source_payload_pages_ready;
            local_ready.store(1u, std::memory_order_release);

            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::milliseconds(
                    collective_timeout_policy::kDefaultCollectiveTimeoutMs);
            while (peer_ready.load(std::memory_order_acquire) == 0u)
            {
                if (std::chrono::steady_clock::now() >= deadline)
                {
                    throw std::runtime_error(
                        "MoE shared activation channel timed out waiting for peer directional NUMA placement");
                }
                cpuRelax();
            }

            PerfStatsCollector::addCounter(
                "memory",
                "moe_overlay_directional_first_touch_bytes",
                static_cast<double>(touched_bytes),
                "model_setup",
                "cpu",
                {{"consumer_role", source ? "continuation" : "follower"},
                 {"numa_node",
                  std::to_string(
                      config.mpi_ctx->topology()->placement().numa_node)},
                 {"world_rank", std::to_string(config.mpi_ctx->rank())}});
        }

        /** @brief Derive one exact participant/family channel contract. */
        MoEOverlayActivationEpochConfig activationConfig(
            const MoEOverlayRankBatchTransportConfig &config,
            size_t participant_index,
            size_t graph_family_ordinal) const
        {
            if (participant_index >= participants_.size() ||
                graph_family_ordinal >=
                    config.activation_graph_families.size())
            {
                throw std::out_of_range(
                    "MoE activation channel configuration index is outside immutable topology");
            }
            const size_t lane_index =
                graph_family_ordinal * participants_.size() +
                participant_index;
            if (lane_index >
                static_cast<size_t>(std::numeric_limits<std::uint32_t>::max()))
            {
                throw std::overflow_error(
                    "MoE activation lane ordinal exceeds the shared ABI");
            }

            std::uint64_t nonce = channel_hash_;
            nonce = fnv1a(
                nonce,
                &participants_[participant_index],
                sizeof(participants_[participant_index]));
            nonce = fnv1a(
                nonce,
                &graph_family_ordinal,
                sizeof(graph_family_ordinal));
            if (nonce == 0u)
                nonce = 0x9e3779b97f4a7c15ull;

            const auto &family =
                config.activation_graph_families[graph_family_ordinal];
            return {
                .channel_nonce = nonce,
                .topology_fingerprint_low =
                    config.transaction_topology.topology_fingerprint_low,
                .topology_fingerprint_high =
                    config.transaction_topology.topology_fingerprint_high,
                .workspace_generation =
                    config.transaction_topology.workspace_generation,
                .source = config.source_endpoint,
                .target = {
                    .world_rank = config.target_world_rank,
                    .participant_id = participants_[participant_index],
                    .tier_priority = config.target_tier_priority,
                    .domain_ordinal = config.domain_ordinal,
                },
                .lane_ordinal = static_cast<std::uint32_t>(lane_index),
                .graph_role_mask = family.graph_role_mask,
                .model_layer_indices = family.model_layer_indices,
            };
        }

        /** @brief Create/map and validate the exact POSIX shared-channel ABI. */
        void mapOrCreate(const MoEOverlayRankBatchTransportConfig &config)
        {
            bool creator = false;
            fd_ = ::shm_open(
                shm_name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
            if (fd_ >= 0)
            {
                creator = true;
                if (::ftruncate(fd_, static_cast<off_t>(mapping_bytes_)) != 0)
                {
                    const std::string detail = std::strerror(errno);
                    ::close(fd_);
                    fd_ = -1;
                    ::shm_unlink(shm_name_.c_str());
                    throw std::runtime_error(
                        "MoE shared activation channel ftruncate failed: " +
                        detail);
                }
            }
            else if (errno == EEXIST)
            {
                fd_ = ::shm_open(shm_name_.c_str(), O_RDWR, 0);
            }
            if (fd_ < 0)
            {
                throw std::runtime_error(
                    "MoE shared activation channel shm_open failed for " +
                    shm_name_ + ": " + std::strerror(errno));
            }

            struct stat status{};
            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::milliseconds(
                    collective_timeout_policy::kDefaultCollectiveTimeoutMs);
            while (::fstat(fd_, &status) == 0 &&
                   static_cast<size_t>(status.st_size) < mapping_bytes_)
            {
                if (std::chrono::steady_clock::now() >= deadline)
                    throw std::runtime_error(
                        "MoE shared activation channel timed out waiting for mapping size");
                cpuRelax();
            }
            if (status.st_size < 0 ||
                static_cast<size_t>(status.st_size) != mapping_bytes_)
            {
                throw std::runtime_error(
                    "MoE shared activation channel mapping size disagrees across ranks");
            }

            base_ = ::mmap(
                nullptr,
                mapping_bytes_,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd_,
                0);
            if (base_ == MAP_FAILED)
            {
                base_ = nullptr;
                throw std::runtime_error(
                    "MoE shared activation channel mmap failed: " +
                    std::string(std::strerror(errno)));
            }
            header_ = static_cast<SharedChannelHeader *>(base_);
            dispatch_publication_ =
                at<SharedPublication>(dispatch_publication_offset_);
            return_publication_ =
                at<SharedPublication>(return_publication_offset_);
            controls_ =
                at<SharedParticipantControl>(controls_offset_);
            activation_controls_ =
                at<MoEOverlayActivationEpochControl>(
                    activation_controls_offset_);

            if (creator)
            {
                /*
                 * Initialize metadata only. Payload pages are deliberately
                 * first-touched by their actual producer socket during setup
                 * or first use, avoiding one-sided NUMA placement.
                 */
                std::memset(
                    base_,
                    0,
                    activation_controls_offset_ +
                        participants_.size() * activation_family_count_ *
                            sizeof(MoEOverlayActivationEpochControl));
                header_->magic = kSharedChannelMagic;
                header_->version = kSharedChannelVersion;
                header_->participant_count =
                    static_cast<uint32_t>(participants_.size());
                header_->activation_family_count = static_cast<uint32_t>(
                    activation_family_count_);
                header_->mapping_bytes = mapping_bytes_;
                header_->channel_hash = channel_hash_;
                header_->max_rows_per_participant = max_rows_;
                header_->max_entries_per_participant = max_entries_;
                header_->d_model = d_model_;
                header_->top_k = top_k_;
                for (size_t index = 0; index < participants_.size(); ++index)
                    controls_[index].participant_id = participants_[index];
                for (size_t family = 0;
                     family < activation_family_count_;
                     ++family)
                {
                    for (size_t participant = 0;
                         participant < participants_.size();
                         ++participant)
                    {
                        MoEOverlayActivationEpochProtocol::initialize(
                            activation_controls_[
                                family * participants_.size() + participant],
                            activationConfig(config, participant, family));
                    }
                }
                header_->initialized.store(1, std::memory_order_release);
            }
            else
            {
                while (header_->initialized.load(std::memory_order_acquire) == 0)
                {
                    if (std::chrono::steady_clock::now() >= deadline)
                        throw std::runtime_error(
                            "MoE shared activation channel initialization timed out");
                    cpuRelax();
                }
            }

            if (header_->magic != kSharedChannelMagic ||
                header_->version != kSharedChannelVersion ||
                header_->participant_count != participants_.size() ||
                header_->activation_family_count !=
                    activation_family_count_ ||
                header_->mapping_bytes != mapping_bytes_ ||
                header_->channel_hash != channel_hash_ ||
                header_->max_rows_per_participant != max_rows_ ||
                header_->max_entries_per_participant != max_entries_ ||
                header_->d_model != d_model_ || header_->top_k != top_k_)
            {
                throw std::runtime_error(
                    "MoE shared activation channel ABI/geometry mismatch");
            }
            for (size_t index = 0; index < participants_.size(); ++index)
            {
                if (controls_[index].participant_id != participants_[index])
                    throw std::runtime_error(
                        "MoE shared activation channel participant topology mismatch");
            }
            for (size_t family = 0;
                 family < activation_family_count_;
                 ++family)
            {
                for (size_t participant = 0;
                     participant < participants_.size();
                     ++participant)
                {
                    const MoEOverlayActivationEpochProtocol validator(
                        activation_controls_[
                            family * participants_.size() + participant],
                        activationConfig(config, participant, family));
                    (void)validator;
                }
            }

            /* Complete both disjoint first-touch operations before the owning
             * transport returns and registers the whole mapping with CUDA/HIP.
             * Registration is the point at which drivers are allowed to pin
             * and fault pages, so this shared barrier is part of setup rather
             * than a hot-path synchronization. */
            placeDirectionalPayloadPages(config);

            PerfStatsCollector::addCounter(
                "memory",
                "moe_overlay_node_local_shared_channel_bytes",
                static_cast<double>(mapping_bytes_),
                "model_setup",
                "cpu",
                {{"channel_hash", std::to_string(channel_hash_)},
                 {"activation_families",
                  std::to_string(activation_family_count_)},
                 {"participants", std::to_string(participants_.size())},
                 {"world_rank", std::to_string(config.mpi_ctx->rank())}});
        }

        std::vector<int> participants_;
        size_t max_rows_ = 0;
        size_t max_entries_ = 0;
        int d_model_ = 0;
        int top_k_ = 0;
        size_t activation_family_count_ = 0;
        size_t page_size_ = 0;
        uint64_t channel_hash_ = 0;
        std::string shm_name_;
        int fd_ = -1;
        void *base_ = nullptr;
        size_t mapping_bytes_ = 0;
        size_t dispatch_publication_offset_ = 0;
        size_t return_publication_offset_ = 0;
        size_t controls_offset_ = 0;
        size_t activation_controls_offset_ = 0;
        /** Target-owned page range for the one physical activation payload. */
        size_t shared_dispatch_begin_ = 0;
        /** Byte offset of the shared physical activation matrix. */
        size_t shared_dispatch_hidden_rows_offset_ = 0;
        /** Exclusive end of the shared physical activation page range. */
        size_t shared_dispatch_end_ = 0;
        std::vector<ParticipantLayout> layouts_;
        SharedChannelHeader *header_ = nullptr;
        SharedPublication *dispatch_publication_ = nullptr;
        SharedPublication *return_publication_ = nullptr;
        SharedParticipantControl *controls_ = nullptr;
        MoEOverlayActivationEpochControl *activation_controls_ = nullptr;
    };

    MoEOverlayNodeLocalRankBatchTransport::
        MoEOverlayNodeLocalRankBatchTransport(
            MoEOverlayRankBatchTransportConfig config)
        : config_(std::move(config))
    {
        if (!config_.mpi_ctx || !config_.workspace ||
            config_.source_world_rank < 0 ||
            config_.target_world_rank < 0 ||
            config_.source_world_rank == config_.target_world_rank ||
            config_.source_world_rank >= config_.mpi_ctx->world_size() ||
            config_.target_world_rank >= config_.mpi_ctx->world_size() ||
            (config_.mpi_ctx->rank() != config_.source_world_rank &&
             config_.mpi_ctx->rank() != config_.target_world_rank) ||
            config_.max_rows_per_participant == 0 ||
            config_.max_entries_per_participant == 0 ||
            config_.d_model <= 0 || config_.top_k <= 0 ||
            config_.tier_index < 0 || config_.domain_ordinal < 0 ||
            config_.channel_identity.empty() ||
            config_.transaction_slot_count == 0 ||
            !config_.transaction_topology.valid() ||
            config_.transaction_topology.source_world_rank !=
                config_.source_world_rank ||
            config_.transaction_topology.target_world_rank !=
                config_.target_world_rank ||
            !config_.source_endpoint.valid() ||
            config_.source_endpoint.world_rank !=
                config_.source_world_rank ||
            config_.activation_graph_families.empty() ||
            config_.activation_graph_families.size() >
                static_cast<size_t>(
                    std::numeric_limits<std::uint32_t>::max()) ||
            config_.local_devices.empty())
        {
            throw std::invalid_argument(
                "Node-local MoE rank-batch transport requires a complete rank, topology, and geometry contract");
        }
        if (std::any_of(
                config_.activation_graph_families.begin(),
                config_.activation_graph_families.end(),
                [](const auto &family) { return !family.valid(); }))
        {
            throw std::invalid_argument(
                "Node-local MoE activation channel requires valid retained graph-family manifests");
        }
        const auto *const topology = config_.mpi_ctx->topology();
        if (!topology ||
            !topology->same_node(
                config_.source_world_rank,
                config_.target_world_rank))
        {
            throw std::runtime_error(
                "Node-local MoE shared activation channel cannot bind ranks on different physical nodes");
        }
        mapping_ = std::make_shared<Mapping>(config_);
        try
        {
            mapped_region_ = transfer_engine_.registerExternalMappedHostRegion(
                mapping_->baseAddress(),
                mapping_->bytes(),
                config_.local_devices,
                mapping_);
        }
        catch (...)
        {
            if (config_.mpi_ctx->rank() == config_.source_world_rank)
                mapping_->unlinkName();
            throw;
        }

        /*
         * Materialize every grant before any model graph can request a lane.
         * The Cartesian product is immutable topology, not a runtime cache:
         * migration may make an initially empty participant active, and every
         * main/MTP retained family therefore needs its own serially reusable
         * endpoint state at a capture-stable address.
         */
        const auto &participants = config_.workspace->participantIds();
        device_grants_.reserve(
            config_.local_devices.size() * participants.size() *
            config_.activation_graph_families.size());
        for (const DeviceId device : config_.local_devices)
        {
            for (const int participant_id : participants)
            {
                for (size_t family = 0u;
                     family < config_.activation_graph_families.size();
                     ++family)
                {
                    device_grants_.push_back(
                        std::make_unique<DeviceGrantStorage>(
                            device,
                            participant_id,
                            family));
                }
            }
        }
        dispatch_ledger_.resize(config_.transaction_slot_count);
        return_ledger_.resize(config_.transaction_slot_count);
    }

    MoEOverlayNodeLocalRankBatchTransport::~MoEOverlayNodeLocalRankBatchTransport()
    {
        if (config_.mpi_ctx &&
            config_.mpi_ctx->rank() == config_.source_world_rank && mapping_)
        {
            mapping_->unlinkName();
        }
    }

    int MoEOverlayNodeLocalRankBatchTransport::sourceWorldRank() const noexcept
    {
        return config_.source_world_rank;
    }

    int MoEOverlayNodeLocalRankBatchTransport::targetWorldRank() const noexcept
    {
        return config_.target_world_rank;
    }

    int MoEOverlayNodeLocalRankBatchTransport::localWorldRank() const noexcept
    {
        return config_.mpi_ctx ? config_.mpi_ctx->rank() : -1;
    }

    const std::vector<int> &
    MoEOverlayNodeLocalRankBatchTransport::participantIds() const noexcept
    {
        return config_.workspace->participantIds();
    }

    MoEOverlaySparseRows
    MoEOverlayNodeLocalRankBatchTransport::sharedDispatchRows(
        int participant_id) const
    {
        return mapping_->dispatchRows(participant_id);
    }

    MoEOverlayReturnRows
    MoEOverlayNodeLocalRankBatchTransport::sharedReturnRows(
        int participant_id) const
    {
        return mapping_->returnRows(participant_id);
    }

    size_t MoEOverlayNodeLocalRankBatchTransport::mappedBytes() const noexcept
    {
        return mapping_ ? mapping_->bytes() : 0;
    }

    size_t MoEOverlayNodeLocalRankBatchTransport::
        activationGraphFamilyCount() const noexcept
    {
        return mapping_ ? mapping_->activationFamilyCount() : 0u;
    }

    std::uint32_t MoEOverlayNodeLocalRankBatchTransport::
        activationStageOrdinal(
            size_t graph_family_ordinal,
            std::int32_t model_layer_index) const
    {
        if (graph_family_ordinal >=
            config_.activation_graph_families.size())
        {
            throw std::out_of_range(
                "ExpertOverlay activation graph-family ordinal is outside the channel manifest");
        }
        const auto &layers =
            config_.activation_graph_families[graph_family_ordinal]
                .model_layer_indices;
        const auto found = std::find(
            layers.begin(), layers.end(), model_layer_index);
        if (found == layers.end())
        {
            throw std::out_of_range(
                "ExpertOverlay model layer is absent from the selected activation graph family");
        }
        return static_cast<std::uint32_t>(
            std::distance(layers.begin(), found));
    }

    MoEOverlayActivationEpochControl &
    MoEOverlayNodeLocalRankBatchTransport::activationEpochControl(
        int target_participant_id,
        size_t graph_family_ordinal) const
    {
        if (!mapping_)
        {
            throw std::logic_error(
                "MoE activation epoch control requested from an unmapped transport");
        }
        return mapping_->activationControl(
            target_participant_id, graph_family_ordinal);
    }

    MoEOverlayActivationEpochConfig
    MoEOverlayNodeLocalRankBatchTransport::activationEpochConfig(
        int target_participant_id,
        size_t graph_family_ordinal) const
    {
        const auto found = std::find(
            config_.workspace->participantIds().begin(),
            config_.workspace->participantIds().end(),
            target_participant_id);
        if (found == config_.workspace->participantIds().end())
        {
            throw std::out_of_range(
                "ExpertOverlay activation scheduler requested an unknown participant");
        }
        if (!mapping_)
        {
            throw std::logic_error(
                "ExpertOverlay activation scheduler requested an unmapped channel");
        }
        return mapping_->activationConfig(
            config_,
            static_cast<size_t>(std::distance(
                config_.workspace->participantIds().begin(), found)),
            graph_family_ordinal);
    }

    std::shared_ptr<const MappedHostTransferRegion>
    MoEOverlayNodeLocalRankBatchTransport::mappedTransferRegion() const noexcept
    {
        return mapped_region_;
    }

    size_t MoEOverlayNodeLocalRankBatchTransport::mappedOffset(
        const void *shared_host_address,
        size_t bytes) const
    {
        if (!mapping_)
        {
            throw std::logic_error(
                "MoE shared activation offset requested from an unmapped transport");
        }
        return mapping_->offsetOf(shared_host_address, bytes);
    }

    void *MoEOverlayNodeLocalRankBatchTransport::mappedDeviceAlias(
        DeviceId device,
        const void *shared_host_address,
        size_t bytes) const
    {
        if (!mapped_region_)
        {
            throw std::logic_error(
                "MoE shared activation alias requested before device registration");
        }
        const size_t offset = mappedOffset(shared_host_address, bytes);
        if (!mapped_region_->contains(offset, bytes))
        {
            throw std::out_of_range(
                "MoE shared activation alias range exceeds registered pages");
        }
        return mapped_region_->deviceAlias(device, offset);
    }

    MoEOverlayMappedActivationDeviceLane
    MoEOverlayNodeLocalRankBatchTransport::activationDeviceLane(
        int target_participant_id,
        size_t graph_family_ordinal,
        DeviceId device) const
    {
        if (!device.is_valid() || !mapped_region_ ||
            !mapped_region_->hasDevice(device))
        {
            throw std::invalid_argument(
                "MoE activation device lane requires one planner-declared local endpoint");
        }

        auto &control = activationEpochControl(
            target_participant_id, graph_family_ordinal);
        const MoEOverlaySparseRows dispatch_rows =
            sharedDispatchRows(target_participant_id);
        const MoEOverlayReturnRows return_rows =
            sharedReturnRows(target_participant_id);

        const auto alias = [&](const void *host_address, size_t bytes)
        {
            return mappedDeviceAlias(device, host_address, bytes);
        };
        const auto dispatch_row_bytes =
            dispatch_rows.row_capacity * sizeof(std::int32_t);
        const auto dispatch_offset_bytes =
            (dispatch_rows.row_capacity + 1u) * sizeof(std::int32_t);
        const auto entry_id_bytes =
            dispatch_rows.entry_capacity * sizeof(std::int32_t);
        const auto entry_weight_bytes =
            dispatch_rows.entry_capacity * sizeof(float);
        const auto activation_bytes =
            dispatch_rows.row_capacity *
            static_cast<size_t>(dispatch_rows.d_model) * sizeof(float);
        const auto return_row_bytes =
            return_rows.row_capacity * sizeof(std::int32_t);
        const auto return_activation_bytes =
            return_rows.row_capacity *
            static_cast<size_t>(return_rows.d_model) * sizeof(float);
        float *const shared_activation_rows =
            mapping_->sharedActivationHiddenRows();

        const auto grant_owner = std::find_if(
            device_grants_.begin(),
            device_grants_.end(),
            [&](const auto &candidate)
            {
                return candidate && candidate->matches(
                                        device,
                                        target_participant_id,
                                        graph_family_ordinal);
            });
        if (grant_owner == device_grants_.end())
        {
            throw std::logic_error(
                "MoE activation lane has no endpoint-private device grant");
        }

        MoEOverlayMappedActivationDeviceLane lane{
            .device = device,
            .target_participant_id = target_participant_id,
            .graph_family_ordinal = graph_family_ordinal,
            .mapped_region = mapped_region_,
            .shared_dispatch_hidden_rows_fp32 =
                static_cast<float *>(alias(
                    shared_activation_rows, activation_bytes)),
            .control_host = &control,
            .control_device = static_cast<MoEOverlayActivationEpochControl *>(
                alias(&control, sizeof(control))),
            .grant_device = (*grant_owner)->grant(),
            .dispatch = {
                .row_ids = static_cast<std::int32_t *>(alias(
                    dispatch_rows.row_ids_host, dispatch_row_bytes)),
                .entry_offsets = static_cast<std::int32_t *>(alias(
                    dispatch_rows.entry_offsets_host,
                    dispatch_offset_bytes)),
                .expert_ids = static_cast<std::int32_t *>(alias(
                    dispatch_rows.expert_ids_host, entry_id_bytes)),
                .route_weights = static_cast<float *>(alias(
                    dispatch_rows.route_weights_host,
                    entry_weight_bytes)),
                .hidden_rows_fp32 = static_cast<float *>(alias(
                    dispatch_rows.hidden_rows_fp32, activation_bytes)),
                .row_capacity = dispatch_rows.row_capacity,
                .entry_capacity = dispatch_rows.entry_capacity,
                .d_model = dispatch_rows.d_model,
                .top_k = dispatch_rows.top_k,
            },
            .returned = {
                .row_ids = static_cast<std::int32_t *>(alias(
                    return_rows.row_ids_host, return_row_bytes)),
                .output_rows_fp32 = static_cast<float *>(alias(
                    return_rows.output_rows_fp32,
                    return_activation_bytes)),
                .row_capacity = return_rows.row_capacity,
                .d_model = return_rows.d_model,
            },
            .dispatch_hidden_offset = mappedOffset(
                dispatch_rows.hidden_rows_fp32, activation_bytes),
            .shared_dispatch_hidden_offset = mappedOffset(
                shared_activation_rows, activation_bytes),
            .return_output_offset = mappedOffset(
                return_rows.output_rows_fp32, return_activation_bytes),
            .admission_signal_offset = mappedOffset(
                &control.admission.ready_signal,
                sizeof(control.admission.ready_signal)),
        };

        for (std::uint32_t bank = 0u;
             bank < kMoEOverlayActivationBufferCount;
             ++bank)
        {
            lane.dispatch_signal_offsets[bank] = mappedOffset(
                &control.buffers[bank].dispatch_signal.value,
                sizeof(control.buffers[bank].dispatch_signal.value));
            lane.return_signal_offsets[bank] = mappedOffset(
                &control.buffers[bank].return_signal.value,
                sizeof(control.buffers[bank].return_signal.value));
        }
        if (!lane.valid())
        {
            throw std::logic_error(
                "MoE activation device lane resolved an incomplete mapped graph identity");
        }
        return lane;
    }

    bool MoEOverlayNodeLocalRankBatchTransport::claimTransaction(
        const MoEOverlayRankBatchKey &key,
        std::vector<std::optional<MoEOverlayRankBatchKey>> &ledger,
        std::string *error)
    {
        if (!key.isValid() || ledger.empty())
        {
            if (error)
                *error = "shared rank-batch transaction key or ledger is invalid";
            return false;
        }
        const size_t first = static_cast<size_t>(key.sequence % ledger.size());
        for (size_t probe = 0; probe < ledger.size(); ++probe)
        {
            auto &slot = ledger[(first + probe) % ledger.size()];
            if (!slot)
            {
                slot = key;
                return true;
            }
            if (slot->sequence != key.sequence)
                continue;
            if (*slot == key)
            {
                if (error)
                    *error = "stale shared rank-batch transaction key reuse rejected";
                return false;
            }
            slot = key;
            return true;
        }
        if (error)
            *error = "shared rank-batch ledger has no graph-sequence slot";
        return false;
    }

    MoEOverlayCollectiveResult
    MoEOverlayNodeLocalRankBatchTransport::exchangeDispatch(
        const MoEOverlayRankBatchKey &key,
        std::span<const MoEOverlaySparseRows *const> outbound,
        std::span<MoEOverlaySparseRows *const> inbound)
    {
        MoEOverlayCollectiveResult result;
        InFlightGuard guard(in_flight_);
        if (!guard.acquired())
        {
            result.ok = false;
            result.error_code = 1;
            result.error = "concurrent reuse of one shared rank-batch channel object is forbidden";
            return result;
        }
        if (key.direction != MoEOverlayCollectiveDirection::Dispatch ||
            key.source_world_rank != config_.source_world_rank ||
            key.target_world_rank != config_.target_world_rank ||
            !claimTransaction(key, dispatch_ledger_, &result.error))
        {
            result.ok = false;
            result.error_code = 2;
            if (result.error.empty())
                result.error = "dispatch key does not match shared channel topology";
            return result;
        }

        using Clock = std::chrono::steady_clock;
        const bool timing = PerfStatsCollector::isDomainEnabled(
            "moe_overlay_transport");
        const auto begin = timing ? Clock::now() : Clock::time_point{};
        uint64_t wait_ns = 0;
        size_t logical_bytes = 0;
        auto &publication = mapping_->dispatchPublication();
        const bool source = localWorldRank() == sourceWorldRank();
        if (source)
        {
            if (!inbound.empty() ||
                outbound.size() != participantIds().size() ||
                pending_return_dispatch_key_)
            {
                result.ok = false;
                result.error_code = 3;
                result.error = "shared dispatch source has invalid row roles or an unmatched return";
                return result;
            }
            if (publication.state.load(std::memory_order_acquire) !=
                static_cast<uint32_t>(PublicationState::Empty))
            {
                result.ok = false;
                result.error_code = 3;
                result.error = "shared dispatch producer reached a non-empty slot before writing";
                return result;
            }
            for (size_t index = 0; index < outbound.size(); ++index)
            {
                const auto *rows = outbound[index];
                const int participant = participantIds()[index];
                if (!rows ||
                    !mapping_->dispatchPointersMatch(participant, *rows) ||
                    rows->target_participant != participant ||
                    rows->d_model != config_.d_model ||
                    rows->top_k != config_.top_k ||
                    rows->live_row_count > rows->row_capacity ||
                    rows->live_entry_count > rows->entry_capacity ||
                    ((rows->live_row_count != 0 ||
                      rows->live_entry_count != 0) &&
                     rows->residency_epoch == 0))
                {
                    result.ok = false;
                    result.error_code = 4;
                    result.error = "shared dispatch source row view violates immutable storage or geometry";
                    return result;
                }
                auto &control = mapping_->control(index);
                control.dispatch_source_participant = rows->source_participant;
                control.dispatch_target_participant = rows->target_participant;
                control.dispatch_d_model = rows->d_model;
                control.dispatch_top_k = rows->top_k;
                control.dispatch_residency_epoch = rows->residency_epoch;
                control.dispatch_live_rows = rows->live_row_count;
                control.dispatch_live_entries = rows->live_entry_count;
                logical_bytes = checkedAdd(
                    logical_bytes,
                    compactMoEOverlayDispatchBytes(*rows),
                    "dispatch accounting");
            }
            publication.key = key;
            publication.state.store(
                static_cast<uint32_t>(PublicationState::Ready),
                std::memory_order_release);
            pending_return_dispatch_key_ = key;
        }
        else
        {
            if (!outbound.empty() ||
                inbound.size() != participantIds().size() ||
                !mapping_->waitFor(
                    publication,
                    PublicationState::Ready,
                    "shared rank-batch dispatch",
                    timing ? &wait_ns : nullptr,
                    &result.error))
            {
                result.ok = false;
                result.error_code = 5;
                if (result.error.empty())
                    result.error = "shared dispatch target has invalid row roles";
                return result;
            }
            if (publication.key != key)
            {
                publication.state.store(
                    static_cast<uint32_t>(PublicationState::Aborted),
                    std::memory_order_release);
                result.ok = false;
                result.error_code = 6;
                result.error = "shared dispatch publication key mismatch";
                return result;
            }
            for (size_t index = 0; index < inbound.size(); ++index)
            {
                auto *rows = inbound[index];
                const int participant = participantIds()[index];
                const auto &control = mapping_->control(index);
                if (!rows ||
                    !mapping_->dispatchPointersMatch(participant, *rows) ||
                    control.dispatch_target_participant != participant ||
                    control.dispatch_d_model != config_.d_model ||
                    control.dispatch_top_k != config_.top_k ||
                    control.dispatch_live_rows > rows->row_capacity ||
                    control.dispatch_live_entries > rows->entry_capacity ||
                    ((control.dispatch_live_rows != 0 ||
                      control.dispatch_live_entries != 0) &&
                     control.dispatch_residency_epoch == 0))
                {
                    publication.state.store(
                        static_cast<uint32_t>(PublicationState::Aborted),
                        std::memory_order_release);
                    result.ok = false;
                    result.error_code = 7;
                    result.error = "shared dispatch metadata violates target capacity or topology";
                    return result;
                }
                rows->key = key.key_namespace ==
                                    MoEOverlayCollectiveNamespace::MTP
                                ? makeMTPMoEOverlayCollectiveKey(
                                      key.generation_id,
                                      key.step_id,
                                      key.mtp_depth,
                                      key.layer_idx,
                                      key.tier_idx,
                                      participant,
                                      participant,
                                      key.direction)
                                : makeMoEOverlayCollectiveKey(
                                      key.generation_id,
                                      key.step_id,
                                      key.layer_idx,
                                      key.tier_idx,
                                      participant,
                                      participant,
                                      key.direction);
                rows->key.histogram_source = key.histogram_source;
                rows->residency_epoch = control.dispatch_residency_epoch;
                rows->source_participant =
                    control.dispatch_source_participant;
                rows->target_participant =
                    control.dispatch_target_participant;
                rows->d_model = control.dispatch_d_model;
                rows->top_k = control.dispatch_top_k;
                rows->live_row_count = control.dispatch_live_rows;
                rows->live_entry_count = control.dispatch_live_entries;
                logical_bytes = checkedAdd(
                    logical_bytes,
                    compactMoEOverlayDispatchBytes(*rows),
                    "dispatch accounting");
            }
            publication.state.store(
                static_cast<uint32_t>(PublicationState::Empty),
                std::memory_order_release);
        }
        const uint64_t total_ns = timing
                                      ? static_cast<uint64_t>(
                                            std::chrono::duration_cast<
                                                std::chrono::nanoseconds>(
                                                Clock::now() - begin)
                                                .count())
                                      : 0;
        recordSharedTransaction(
            key,
            logical_bytes,
            source ? "source" : "target",
            wait_ns,
            total_ns);
        result.collective_complete = true;
        return result;
    }

    MoEOverlayCollectiveResult
    MoEOverlayNodeLocalRankBatchTransport::exchangeReturn(
        const MoEOverlayRankBatchKey &key,
        std::span<const MoEOverlayReturnRows *const> outbound,
        std::span<MoEOverlayReturnRows *const> inbound)
    {
        MoEOverlayCollectiveResult result;
        InFlightGuard guard(in_flight_);
        if (!guard.acquired())
        {
            result.ok = false;
            result.error_code = 1;
            result.error = "concurrent reuse of one shared rank-batch channel object is forbidden";
            return result;
        }
        if (key.direction != MoEOverlayCollectiveDirection::ReturnReduce ||
            key.source_world_rank != config_.source_world_rank ||
            key.target_world_rank != config_.target_world_rank ||
            !claimTransaction(key, return_ledger_, &result.error))
        {
            result.ok = false;
            result.error_code = 2;
            if (result.error.empty())
                result.error = "return key does not match shared channel topology";
            return result;
        }

        using Clock = std::chrono::steady_clock;
        const bool timing = PerfStatsCollector::isDomainEnabled(
            "moe_overlay_transport");
        const auto begin = timing ? Clock::now() : Clock::time_point{};
        uint64_t wait_ns = 0;
        size_t logical_bytes = 0;
        auto &publication = mapping_->returnPublication();
        const bool target = localWorldRank() == targetWorldRank();
        if (target)
        {
            if (!inbound.empty() ||
                outbound.size() != participantIds().size() ||
                publication.state.load(std::memory_order_acquire) !=
                    static_cast<uint32_t>(PublicationState::Empty))
            {
                result.ok = false;
                result.error_code = 3;
                result.error = "shared return producer has invalid roles or a non-empty slot";
                return result;
            }
            for (size_t index = 0; index < outbound.size(); ++index)
            {
                const auto *rows = outbound[index];
                const int participant = participantIds()[index];
                if (!rows ||
                    !mapping_->returnPointersMatch(participant, *rows) ||
                    rows->source_participant != participant ||
                    rows->d_model != config_.d_model ||
                    rows->live_row_count > rows->row_capacity ||
                    (rows->live_row_count != 0 &&
                     rows->residency_epoch == 0))
                {
                    result.ok = false;
                    result.error_code = 4;
                    result.error = "shared return source row view violates immutable storage or geometry";
                    return result;
                }
                auto &control = mapping_->control(index);
                control.return_source_participant = rows->source_participant;
                control.return_target_participant = rows->target_participant;
                control.return_d_model = rows->d_model;
                control.return_residency_epoch = rows->residency_epoch;
                control.return_live_rows = rows->live_row_count;
                logical_bytes = checkedAdd(
                    logical_bytes,
                    compactMoEOverlayReturnBytes(*rows),
                    "return accounting");
            }
            publication.key = key;
            publication.state.store(
                static_cast<uint32_t>(PublicationState::Ready),
                std::memory_order_release);
        }
        else
        {
            if (!outbound.empty() ||
                inbound.size() != participantIds().size() ||
                !pending_return_dispatch_key_ ||
                !sameRoundTrip(*pending_return_dispatch_key_, key) ||
                !mapping_->waitFor(
                    publication,
                    PublicationState::Ready,
                    "shared rank-batch return",
                    timing ? &wait_ns : nullptr,
                    &result.error))
            {
                result.ok = false;
                result.error_code = 5;
                if (result.error.empty())
                    result.error = "shared return source has no matching dispatch or row roles";
                return result;
            }
            if (publication.key != key)
            {
                publication.state.store(
                    static_cast<uint32_t>(PublicationState::Aborted),
                    std::memory_order_release);
                result.ok = false;
                result.error_code = 6;
                result.error = "shared return publication key mismatch";
                return result;
            }
            for (size_t index = 0; index < inbound.size(); ++index)
            {
                auto *rows = inbound[index];
                const int participant = participantIds()[index];
                const auto &control = mapping_->control(index);
                if (!rows ||
                    !mapping_->returnPointersMatch(participant, *rows) ||
                    control.return_source_participant != participant ||
                    control.return_d_model != config_.d_model ||
                    control.return_live_rows > rows->row_capacity ||
                    (control.return_live_rows != 0 &&
                     control.return_residency_epoch == 0))
                {
                    publication.state.store(
                        static_cast<uint32_t>(PublicationState::Aborted),
                        std::memory_order_release);
                    result.ok = false;
                    result.error_code = 7;
                    result.error = "shared return metadata violates source capacity or topology";
                    return result;
                }
                rows->key = key.key_namespace ==
                                    MoEOverlayCollectiveNamespace::MTP
                                ? makeMTPMoEOverlayCollectiveKey(
                                      key.generation_id,
                                      key.step_id,
                                      key.mtp_depth,
                                      key.layer_idx,
                                      key.tier_idx,
                                      participant,
                                      participant,
                                      key.direction)
                                : makeMoEOverlayCollectiveKey(
                                      key.generation_id,
                                      key.step_id,
                                      key.layer_idx,
                                      key.tier_idx,
                                      participant,
                                      participant,
                                      key.direction);
                rows->key.histogram_source = key.histogram_source;
                rows->residency_epoch = control.return_residency_epoch;
                rows->source_participant =
                    control.return_source_participant;
                rows->target_participant =
                    control.return_target_participant;
                rows->d_model = control.return_d_model;
                rows->live_row_count = control.return_live_rows;
                logical_bytes = checkedAdd(
                    logical_bytes,
                    compactMoEOverlayReturnBytes(*rows),
                    "return accounting");
            }
            publication.state.store(
                static_cast<uint32_t>(PublicationState::Empty),
                std::memory_order_release);
            pending_return_dispatch_key_.reset();
        }
        const uint64_t total_ns = timing
                                      ? static_cast<uint64_t>(
                                            std::chrono::duration_cast<
                                                std::chrono::nanoseconds>(
                                                Clock::now() - begin)
                                                .count())
                                      : 0;
        recordSharedTransaction(
            key,
            logical_bytes,
            target ? "target" : "source",
            wait_ns,
            total_ns);
        result.collective_complete = true;
        return result;
    }

    MoEOverlayRankBatchTransportKind
    resolveMoEOverlayRankBatchTransportKind(
        const IMPIContext &mpi_ctx,
        int source_world_rank,
        int target_world_rank)
    {
        if (source_world_rank < 0 || target_world_rank < 0 ||
            source_world_rank == target_world_rank ||
            source_world_rank >= mpi_ctx.world_size() ||
            target_world_rank >= mpi_ctx.world_size())
        {
            throw std::invalid_argument(
                "MoE rank-batch transport selection requires two valid world ranks");
        }
        const auto *const topology = mpi_ctx.topology();
        if (!topology)
        {
            throw std::runtime_error(
                "MoE rank-batch transport selection requires authoritative physical topology");
        }
        return topology->same_node(source_world_rank, target_world_rank)
                   ? MoEOverlayRankBatchTransportKind::NodeLocalSharedRows
                   : MoEOverlayRankBatchTransportKind::MPI;
    }

    std::shared_ptr<IMoEOverlayRankBatchTransport>
    createMoEOverlayRankBatchTransport(
        MoEOverlayRankBatchTransportConfig config)
    {
        if (!config.mpi_ctx)
            throw std::invalid_argument(
                "MoE rank-batch transport factory requires an MPI context");
        const auto kind = resolveMoEOverlayRankBatchTransportKind(
            *config.mpi_ctx,
            config.source_world_rank,
            config.target_world_rank);
        if (kind == MoEOverlayRankBatchTransportKind::NodeLocalSharedRows)
        {
            return std::make_shared<MoEOverlayNodeLocalRankBatchTransport>(
                std::move(config));
        }
        return std::make_shared<MoEOverlayMPIRankBatchTransport>(
            MoEOverlayMPIRankBatchTransport::Config{
                .mpi_ctx = std::move(config.mpi_ctx),
                .source_world_rank = config.source_world_rank,
                .target_world_rank = config.target_world_rank,
                .workspace = std::move(config.workspace),
                .transaction_slot_count = config.transaction_slot_count,
                .asynchronous_send_slot_count =
                    config.asynchronous_send_slot_count,
            });
    }

} // namespace llaminar2
