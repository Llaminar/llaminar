/**
 * @file MoEOverlayMPIEconomyEvidenceExchange.cpp
 * @brief Versioned MPI_Iallgather protocol for distributed economy evidence.
 */

#include "MoEOverlayMPIEconomyEvidenceExchange.h"
#include "MoEOverlayEconomyEvidenceCodec.h"

#include "MoEOverlayMPIFatal.h"

#include "collective/CollectiveTimeoutPolicy.h"
#include "interfaces/IMPIContext.h"
#include "utils/FNV1a.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /** Wire discriminator for one complete calibration attempt. */
        constexpr std::uint32_t kAttemptMagic = 0x45434154u; // "ECAT"
        /** Wire discriminator for one pre-staging calibration readiness edge. */
        constexpr std::uint32_t kReadinessMagic = 0x45435244u; // "ECRD"
        /** Wire discriminator for one rank's service matrix. */
        constexpr std::uint32_t kServiceMagic = 0x45435356u; // "ECSV"
        constexpr std::uint16_t kWireVersion = 3;

        /** @brief One rank's complete versioned calibration attempt packet. */
        struct EvidenceCountHeader
        {
            std::uint32_t magic = kAttemptMagic;
            std::uint16_t version = 5;
            std::uint16_t reserved = 0;
            std::int32_t world_rank = -1;
            std::int32_t world_size = 0;
            std::uint64_t calibration_sequence = 0;
            /** Complete typed target words, never representative cohort members. */
            std::int32_t first_target_word = -1;
            std::int32_t second_target_word = -1;
            std::int32_t layer = -1;
            std::uint32_t source = 0;
            std::int32_t real_rows = 0;
            std::int32_t execution_rows = 0;
            std::int32_t transaction_count = 0;
            std::int32_t speculative_depth = 0;
            std::uint64_t schedule_fingerprint = 0;
            std::uint64_t baseline_nanoseconds = 0;
            std::uint64_t concurrent_nanoseconds = 0;
            std::uint32_t exact_overlap = 0;
            std::uint32_t reserved_result = 0;
            /** Count/identity metadata only; samples use exact-live payloads. */
            std::uint64_t payload_bytes = 0;
            std::uint64_t payload_hash = 0;
            std::uint64_t expected_epoch = 0;
            std::uint64_t candidate_epoch = 0;
            std::uint64_t wave_wall_nanoseconds = 0;
            std::array<std::int32_t,2> experts{{-1,-1}};
            std::uint32_t evidence_kind = 0;
            std::uint32_t migration_count = 0;
            std::uint64_t transaction_purpose = 0;
            std::uint64_t packet_hash = 0;
        };

        /** @brief Fixed-layout all-rank inference-terminal readiness packet. */
        struct CalibrationReadinessWire
        {
            std::uint32_t magic = kReadinessMagic;
            std::uint16_t version = kWireVersion;
            std::uint16_t kind = 0;
            std::int32_t world_rank = -1;
            std::int32_t world_size = 0;
            std::uint64_t calibration_sequence = 0;
            std::int32_t first_participant = -1;
            std::int32_t second_participant = -1;
            std::int32_t layer = -1;
            std::uint32_t source = 0;
            std::int32_t real_rows = 0;
            std::int32_t execution_rows = 0;
            std::int32_t transaction_count = 0;
            std::int32_t speculative_depth = 0;
            std::uint64_t schedule_fingerprint = 0;
            std::uint64_t packet_hash = 0;
        };

        /** @brief Header preceding a dense participant/layer service packet. */
        struct ServiceHeaderWire
        {
            std::uint32_t magic = kServiceMagic;
            std::uint16_t version = kWireVersion;
            std::uint16_t reserved = 0;
            std::int32_t world_rank = -1;
            std::int32_t world_size = 0;
            std::uint32_t participant_count = 0;
            std::uint32_t num_layers = 0;
            std::uint32_t active_source_mask = 0;
            std::uint32_t economy_source_mask = 0;
            std::uint64_t production_topology_fingerprint = 0;
            std::uint64_t cell_count = 0;
            std::uint64_t packet_bytes = 0;
            std::uint64_t packet_hash = 0;
        };

        /** @brief Dense optional service row; absence is explicit. */
        struct ServiceCellWire
        {
            std::uint32_t present = 0;
            std::int32_t participant_id = -1;
            std::int32_t layer = -1;
            std::uint32_t reserved = 0;
            std::array<std::uint64_t, kExpertHistogramProductionSourceCount>
                total_nanoseconds{};
            std::array<std::uint64_t, kExpertHistogramProductionSourceCount>
                activation_count{};
            std::array<std::uint64_t, kExpertHistogramProductionSourceCount>
                sample_count{};
            std::array<std::uint32_t, kExpertHistogramProductionSourceCount>
                overflowed{};
        };

        static_assert(std::is_trivially_copyable_v<EvidenceCountHeader>);
        static_assert(
            std::is_trivially_copyable_v<CalibrationReadinessWire>);
        static_assert(std::is_trivially_copyable_v<ServiceHeaderWire>);
        static_assert(std::is_trivially_copyable_v<ServiceCellWire>);

        /** @brief Hash one zero-initialized fixed attempt packet. */
        std::uint64_t evidenceHeaderHash(const EvidenceCountHeader &wire) noexcept
        {
            return fnv1a64(
                &wire,
                offsetof(EvidenceCountHeader, packet_hash));
        }

        /** @brief Hash one fixed readiness packet excluding its hash field. */
        std::uint64_t readinessHash(
            const CalibrationReadinessWire &wire) noexcept
        {
            return fnv1a64(
                &wire,
                offsetof(CalibrationReadinessWire, packet_hash));
        }

        /** @brief Hash a service packet while treating its hash field as zero. */
        std::uint64_t serviceHash(
            const void *packet,
            std::size_t packet_bytes) noexcept
        {
            if (!packet || packet_bytes < sizeof(ServiceHeaderWire))
                return 0;
            constexpr std::size_t hash_offset =
                offsetof(ServiceHeaderWire, packet_hash);
            constexpr std::array<std::uint8_t, sizeof(std::uint64_t)> zeros{};
            const auto *bytes = static_cast<const std::uint8_t *>(packet);
            std::uint64_t hash = fnv1a64(bytes, hash_offset);
            hash = fnv1a64(zeros.data(), zeros.size(), hash);
            return fnv1a64(
                bytes + hash_offset + sizeof(std::uint64_t),
                packet_bytes - hash_offset - sizeof(std::uint64_t),
                hash);
        }

        /** @return Stable low-bit wire encoding of runtime phase availability. */
        std::uint32_t encodeSourceMask(
            const ExpertHistogramProductionSourceMask &sources) noexcept
        {
            std::uint32_t result = 0;
            for (std::size_t phase = 0; phase < sources.size(); ++phase)
            {
                if (sources[phase])
                    result |= std::uint32_t{1} << phase;
            }
            return result;
        }

        /** @return Stable identity of every retained layer's two phase masks. */
        std::uint64_t productionTopologyFingerprint(
            const ExpertHistogramProductionTopology &topology) noexcept
        {
            std::uint64_t hash = kFNV1a64OffsetBasis;
            for (std::size_t layer = 0; layer < topology.layerCount(); ++layer)
            {
                const std::array<std::uint32_t, 2> masks{
                    encodeSourceMask(
                        topology.sources(static_cast<int>(layer))),
                    encodeSourceMask(
                        topology.economySources(static_cast<int>(layer))),
                };
                hash = fnv1a64(masks.data(), sizeof(masks), hash);
            }
            return hash;
        }

        /** @brief Render one stable MPI error diagnostic. */
        std::string mpiError(const char *operation, int mpi_error)
        {
            char buffer[MPI_MAX_ERROR_STRING]{};
            int length = 0;
            const int described =
                MPI_Error_string(mpi_error, buffer, &length);
            std::ostringstream message;
            message << "ExpertOverlay " << operation
                    << " failed with MPI code " << mpi_error;
            if (described == MPI_SUCCESS && length > 0)
                message << ": " << std::string(buffer, buffer + length);
            return message.str();
        }

        /** @brief Assign a caller diagnostic only when requested. */
        void assignError(std::string *error, std::string message)
        {
            if (error)
                *error = std::move(message);
        }
    } // namespace

    struct MoEOverlayMPIEconomyEvidenceExchange::Impl
    {
        enum class Active : std::uint8_t
        {
            Idle,
            MigrationProfile,
            Attempt,
            CalibrationReadiness,
            ServiceReadiness,
            Service,
        };

        Config config;
        MPI_Comm communicator = MPI_COMM_NULL;
        MPI_Request request = MPI_REQUEST_NULL;
        Active active = Active::Idle;
        EvidenceCountHeader local_attempt;
        std::vector<EvidenceCountHeader> gathered_attempts;
        std::shared_ptr<const MoEOverlayMigrationEvidenceContext> evidence_context;
        std::vector<std::byte> local_payload, gathered_payload;
        std::vector<int> payload_counts, payload_offsets;
        bool payload_active = false;
        CalibrationReadinessWire local_calibration_readiness;
        std::vector<CalibrationReadinessWire>
            gathered_calibration_readiness;
        int local_service_readiness = 0;
        int global_service_readiness = 0;
        std::vector<std::uint64_t> local_service;
        std::vector<std::uint64_t> gathered_service;
        std::size_t service_packet_bytes = 0;
        std::size_t service_packet_words = 0;
        std::chrono::steady_clock::time_point started_at{};
        MoEOverlayMPIEconomyEvidenceExchangeStats stats;

        /** @return Retained routed-layer count from the topology authority. */
        [[nodiscard]] int numLayers() const noexcept
        {
            return static_cast<int>(
                config.production_topology.layerCount());
        }

        /** @return Union of graph-reachable phases for wire authentication. */
        [[nodiscard]] ExpertHistogramProductionSourceMask reachableSources()
            const noexcept
        {
            return config.production_topology.activeSources();
        }

        /** @brief Emit one rare control-plane counter. */
        void record(const char *name, double value = 1.0) const
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                name,
                value,
                "maintenance",
                config.perf_device,
                {{"world_rank",
                  std::to_string(config.mpi_context->rank())},
                 {"world_size",
                  std::to_string(config.mpi_context->world_size())},
                 {"blocking", "false"}});
        }

        /** @brief Begin one byte all-gather over preallocated storage. */
        bool beginRaw(
            const void *local,
            void *gathered,
            std::size_t bytes,
            Active kind,
            std::string *error)
        {
            if (active != Active::Idle || request != MPI_REQUEST_NULL)
            {
                ++stats.concurrent_start_rejections;
                assignError(
                    error,
                    "ExpertOverlay economy evidence lane already owns an exchange");
                return false;
            }
            if (!local || !gathered || bytes == 0 ||
                bytes > static_cast<std::size_t>(INT_MAX))
            {
                ++stats.rejected_packets;
                assignError(
                    error,
                    "ExpertOverlay economy evidence packet exceeds the MPI byte ABI");
                return false;
            }
            started_at = std::chrono::steady_clock::now();
            const int mpi_result = MPI_Iallgather(
                local,
                static_cast<int>(bytes),
                MPI_BYTE,
                gathered,
                static_cast<int>(bytes),
                MPI_BYTE,
                communicator,
                &request);
            if (mpi_result != MPI_SUCCESS || request == MPI_REQUEST_NULL)
            {
                request = MPI_REQUEST_NULL;
                ++stats.mpi_failures;
                assignError(error, mpiError("MPI_Iallgather", mpi_result));
                return false;
            }
            active = kind;
            if (error)
                error->clear();
            return true;
        }

        /** @brief Query one active request and enforce the standard timeout. */
        MoEOverlayResidencyWaveProgress pollRaw(
            Active expected,
            std::string *error)
        {
            ++stats.progress_polls;
            if (active != expected || request == MPI_REQUEST_NULL)
            {
                assignError(
                    error,
                    "ExpertOverlay economy evidence polled the wrong exchange kind");
                return MoEOverlayResidencyWaveProgress::Failed;
            }
            int complete = 0;
            const int mpi_result =
                MPI_Test(&request, &complete, MPI_STATUS_IGNORE);
            if (mpi_result != MPI_SUCCESS)
            {
                ++stats.mpi_failures;
                const auto message = mpiError("MPI_Test", mpi_result);
                assignError(error, message);
                abortMoEOverlayMPI(
                    communicator,
                    config.mpi_context->rank(),
                    "economy_evidence",
                    message);
            }
            if (!complete)
            {
                if (std::chrono::steady_clock::now() - started_at >=
                    std::chrono::milliseconds(
                        collective_timeout_policy::
                            kDefaultCollectiveTimeoutMs))
                {
                    ++stats.mpi_failures;
                    std::ostringstream diagnostic;
                    const char *operation =
                        expected == Active::ServiceReadiness
                            ? "MPI_Iallreduce"
                            : payload_active && (expected==Active::Attempt || expected==Active::MigrationProfile)
                                ? "MPI_Iallgatherv" : "MPI_Iallgather";
                    diagnostic
                        << operation
                        << " exceeded the canonical 30-second collective timeout"
                        << " exchange_kind="
                        << static_cast<int>(active)
                        << " expected_kind="
                        << static_cast<int>(expected)
                        << " packet_bytes="
                        << (expected == Active::Service
                                ? service_packet_bytes
                                : expected == Active::ServiceReadiness
                                      ? sizeof(int)
                                      : expected ==
                                                Active::CalibrationReadiness
                                            ? sizeof(
                                                  CalibrationReadinessWire)
                                            : (payload_active ? local_payload.size() : sizeof(EvidenceCountHeader)));
                    assignError(error, diagnostic.str());
                    abortMoEOverlayMPI(
                        communicator,
                        config.mpi_context->rank(),
                        "economy_evidence",
                        diagnostic.str());
                }
                return MoEOverlayResidencyWaveProgress::Pending;
            }
            active = Active::Idle;
            if (error)
                error->clear();
            return MoEOverlayResidencyWaveProgress::Ready;
        }

        /** @brief Validate every header before posting any observation payload.
         * @param wire Rank header. @param rank Actual collective source. */
        void validateEvidenceHeader(const EvidenceCountHeader &wire,int rank) const
        {
            const auto &local=local_attempt;
            if(wire.magic!=kAttemptMagic || wire.version!=5 || wire.reserved || wire.reserved_result ||
               wire.world_rank!=rank || wire.world_size!=config.mpi_context->world_size() ||
               wire.packet_hash!=evidenceHeaderHash(wire) || !wire.packet_hash || wire.exact_overlap>1 ||
               wire.calibration_sequence!=local.calibration_sequence || wire.first_target_word!=local.first_target_word ||
               wire.second_target_word!=local.second_target_word || wire.layer!=local.layer ||
               wire.source!=local.source || wire.real_rows!=local.real_rows || wire.execution_rows!=local.execution_rows ||
               wire.transaction_count!=local.transaction_count || wire.speculative_depth!=local.speculative_depth ||
               wire.schedule_fingerprint!=local.schedule_fingerprint || wire.evidence_kind!=local.evidence_kind ||
               wire.expected_epoch!=local.expected_epoch || wire.candidate_epoch!=local.candidate_epoch ||
               wire.migration_count!=local.migration_count || wire.experts!=local.experts ||
               wire.transaction_purpose!=local.transaction_purpose ||
               !wire.wave_wall_nanoseconds || wire.payload_bytes>static_cast<std::uint64_t>(INT_MAX) ||
               wire.payload_bytes!=MoEOverlayEconomyEvidenceCodec::maximumEncodedBytes(*evidence_context,rank))
                throw std::invalid_argument("Economy evidence count header differs from original local wave identity");
            if(wire.evidence_kind==1 && (wire.baseline_nanoseconds || wire.concurrent_nanoseconds || wire.exact_overlap))
                throw std::invalid_argument("Migration profile contains attempt-only metadata");
        }

        /** @brief Gather exact live counts then payloads using the same request and deadline.
         * @param expected Existing exchange kind. @param error Failure diagnostic.
         * @return Pending, complete payload, or failed state. */
        MoEOverlayResidencyWaveProgress pollEvidence(Active expected,std::string *error)
        {
            const auto progress=pollRaw(expected,error);
            if(progress!=MoEOverlayResidencyWaveProgress::Ready)return progress;
            if(payload_active)return MoEOverlayResidencyWaveProgress::Ready;
            try {
                std::size_t total=0;
                for(std::size_t rank=0;rank<gathered_attempts.size();++rank){
                    const auto &wire=gathered_attempts[rank];validateEvidenceHeader(wire,static_cast<int>(rank));
                    if(wire.payload_bytes>static_cast<std::size_t>(INT_MAX)-total)
                        throw std::invalid_argument("Economy evidence live receive prefix exceeds MPI byte ABI");
                    payload_counts[rank]=static_cast<int>(wire.payload_bytes);payload_offsets[rank]=static_cast<int>(total);
                    total+=wire.payload_bytes;
                }
                gathered_payload.resize(total);
                // Zero counts are real empty observers, never capacity-sized transmissions.
                const int status=MPI_Iallgatherv(local_payload.data(),static_cast<int>(local_payload.size()),MPI_BYTE,
                    gathered_payload.data(),payload_counts.data(),payload_offsets.data(),MPI_BYTE,communicator,&request);
                if(status!=MPI_SUCCESS || request==MPI_REQUEST_NULL){
                    ++stats.mpi_failures;
                    abortMoEOverlayMPI(communicator,config.mpi_context->rank(),"economy_evidence",mpiError("MPI_Iallgatherv",status));
                }
                stats.evidence_payload_bytes_sent+=local_payload.size();
                stats.evidence_payload_bytes_received+=total;
                ++stats.evidence_payload_collectives_started;
                record("mpi_economy_evidence_payload_bytes_sent",static_cast<double>(local_payload.size()));
                record("mpi_economy_evidence_payload_bytes_received",static_cast<double>(total));
                payload_active=true;active=expected;
                // started_at belongs to both collectives; never restart the timeout here.
                return MoEOverlayResidencyWaveProgress::Pending;
            } catch(const std::exception &exception) {
                ++stats.rejected_packets;assignError(error,exception.what());
                // A rank may not abandon the collective sequence while peers enter payload exchange.
                abortMoEOverlayMPI(communicator,config.mpi_context->rank(),"economy_evidence",exception.what());
            }
        }

        /** @brief Bind local original geometry and encode exact observation bytes before posting.
         * @param wire Common profile/attempt header. @param context Original prepared-wave metadata.
         * @param rows Local independent quotes and observations. @param kind Existing exchange kind.
         * @param error Failure diagnostic. @return Whether the count request was posted. */
        bool beginEvidence(EvidenceCountHeader wire,std::shared_ptr<const MoEOverlayMigrationEvidenceContext> context,
            const std::vector<MoEOverlayCompletedMigrationMeasurement> &rows,Active kind,std::string *error)
        {
            if(active!=Active::Idle || request!=MPI_REQUEST_NULL){
                ++stats.concurrent_start_rejections;assignError(error,"Economy evidence lane already owns an exchange");return false;
            }
            try {
                if(!context || rows.size()!=2 || !context->matches(rows))
                    throw std::invalid_argument("Economy evidence requires the original checked prepared-wave context");
                const auto &ids=context->identities();
                for(const auto &id:ids)for(const auto *target:{&id.geometry->targets().first,&id.geometry->targets().second})
                    for(const auto &member:target->members) {
                        const auto *original=config.owner_map.participantForId(member.participant_id);
                        if(!original || !original->world_rank_known || original->world_rank!=member.world_rank ||
                           original->device!=member.device || original->address!=member.address)
                            throw std::invalid_argument("Economy evidence context differs from original participant inventory");
                    }
                // Bound receive counts by original piece incidence before accepting sender lengths.
                local_payload=MoEOverlayEconomyEvidenceCodec::encode(*context,rows,config.mpi_context->rank());
                if(local_payload.size()>static_cast<std::size_t>(INT_MAX))throw std::invalid_argument("Economy evidence live payload exceeds MPI byte ABI");
                wire.expected_epoch=ids.front().expected_epoch;wire.candidate_epoch=ids.front().candidate_epoch;
                wire.migration_count=static_cast<std::uint32_t>(ids.size());
                wire.transaction_purpose=static_cast<std::uint64_t>(context->purpose());
                wire.experts={ids[0].expert,ids[1].expert};wire.wave_wall_nanoseconds=rows.front().wave_wall_nanoseconds;
                for(const auto &row:rows)if(row.wave_wall_nanoseconds!=wire.wave_wall_nanoseconds)
                    throw std::invalid_argument("Economy evidence rows disagree on original wave duration");
                wire.payload_bytes=local_payload.size();wire.payload_hash=fnv1a64(local_payload.data(),local_payload.size());
                wire.packet_hash=evidenceHeaderHash(wire);local_attempt=wire;evidence_context=std::move(context);payload_active=false;
                std::fill(gathered_attempts.begin(),gathered_attempts.end(),EvidenceCountHeader{});
                const bool posted=beginRaw(&local_attempt,gathered_attempts.data(),sizeof(EvidenceCountHeader),kind,error);
                if(posted){stats.evidence_count_bytes_sent+=sizeof(EvidenceCountHeader);
                    record("mpi_economy_evidence_count_bytes_sent",sizeof(EvidenceCountHeader));}
                return posted;
            }catch(const std::exception &exception){++stats.rejected_packets;assignError(error,exception.what());return false;}
        }

        /** @brief Decode one rank against this wave's original metadata only.
         * @param rank Actual collective source. @return Context-shaped partial observations. */
        std::vector<MoEOverlayCompletedMigrationMeasurement> decodeEvidence(std::size_t rank) const
        {
            const auto &wire=gathered_attempts.at(rank);
            const auto bytes=std::span<const std::byte>(gathered_payload).subspan(payload_offsets.at(rank),payload_counts.at(rank));
            if(fnv1a64(bytes.data(),bytes.size())!=wire.payload_hash)
                throw std::invalid_argument("Economy observation payload failed packet integrity");
            return MoEOverlayEconomyEvidenceCodec::decode(*evidence_context,bytes,static_cast<int>(rank),wire.wave_wall_nanoseconds);
        }
    };

    MoEOverlayMPIEconomyEvidenceExchange::
        MoEOverlayMPIEconomyEvidenceExchange(Config config)
        : impl_(std::make_unique<Impl>())
    {
        impl_->config = std::move(config);
        auto &owned = impl_->config;
        if (!owned.mpi_context ||
            owned.owner_map.participants().empty() ||
            !owned.production_topology.valid() ||
            owned.production_topology.layerCount() >
                static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            owned.mpi_context->world_size() < 2 ||
            owned.mpi_context->rank() < 0 ||
            owned.mpi_context->rank() >= owned.mpi_context->world_size() ||
            owned.mpi_context->communicator() == MPI_COMM_NULL)
        {
            throw std::invalid_argument(
                "ExpertOverlay MPI economy evidence requires complete multi-rank topology");
        }
        if (owned.perf_device.empty())
            owned.perf_device = "expert_overlay_distributed";
        for (const auto &participant : owned.owner_map.participants())
        {
            if (!participant.world_rank_known ||
                participant.world_rank < 0 ||
                participant.world_rank >= owned.mpi_context->world_size())
            {
                throw std::invalid_argument(
                    "ExpertOverlay MPI economy evidence requires resolved participant rank ownership");
            }
        }

        int initialized = 0;
        int finalized = 0;
        int thread_support = MPI_THREAD_SINGLE;
        if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized ||
            MPI_Finalized(&finalized) != MPI_SUCCESS || finalized ||
            MPI_Query_thread(&thread_support) != MPI_SUCCESS ||
            thread_support < MPI_THREAD_MULTIPLE)
        {
            throw std::runtime_error(
                "ExpertOverlay MPI economy evidence requires live MPI_THREAD_MULTIPLE support");
        }

        const std::size_t participants =
            owned.owner_map.participants().size();
        const std::size_t num_layers =
            owned.production_topology.layerCount();
        if (participants >
                std::numeric_limits<std::size_t>::max() /
                    num_layers ||
            participants * num_layers >
                (std::numeric_limits<std::size_t>::max() -
                 sizeof(ServiceHeaderWire)) /
                    sizeof(ServiceCellWire))
        {
            throw std::overflow_error(
                "ExpertOverlay MPI service packet geometry overflows host size");
        }
        const std::size_t cells =
            participants * num_layers;
        const std::size_t semantic_packet_bytes =
            sizeof(ServiceHeaderWire) + cells * sizeof(ServiceCellWire);
        impl_->service_packet_words =
            (semantic_packet_bytes + sizeof(std::uint64_t) - 1u) /
            sizeof(std::uint64_t);
        impl_->service_packet_bytes =
            impl_->service_packet_words * sizeof(std::uint64_t);
        if (impl_->service_packet_bytes > static_cast<std::size_t>(INT_MAX))
        {
            throw std::overflow_error(
                "ExpertOverlay MPI service packet exceeds the MPI count ABI");
        }
        impl_->gathered_attempts.resize(
            static_cast<std::size_t>(owned.mpi_context->world_size()));
        impl_->payload_counts.resize(impl_->gathered_attempts.size());
        impl_->payload_offsets.resize(impl_->gathered_attempts.size());
        impl_->gathered_calibration_readiness.resize(
            static_cast<std::size_t>(owned.mpi_context->world_size()));
        impl_->local_service.resize(impl_->service_packet_words);
        if (impl_->service_packet_words >
            std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(owned.mpi_context->world_size()))
        {
            throw std::overflow_error(
                "ExpertOverlay MPI gathered service packet geometry overflows host size");
        }
        impl_->gathered_service.resize(
            impl_->service_packet_words *
            static_cast<std::size_t>(owned.mpi_context->world_size()));

        const int duplicate = MPI_Comm_dup(
            owned.mpi_context->communicator(), &impl_->communicator);
        if (duplicate != MPI_SUCCESS ||
            impl_->communicator == MPI_COMM_NULL)
        {
            throw std::runtime_error(
                mpiError("MPI_Comm_dup", duplicate));
        }
        impl_->record("mpi_economy_evidence_lanes_materialized");
    }

    MoEOverlayMPIEconomyEvidenceExchange::
        ~MoEOverlayMPIEconomyEvidenceExchange()
    {
        if (!impl_)
            return;
        if (impl_->request != MPI_REQUEST_NULL ||
            impl_->active != Impl::Active::Idle)
        {
            LOG_ERROR(
                "[MoEOverlayMPIEconomyEvidenceExchange] Destroyed with an active exchange");
            std::terminate();
        }
        if (impl_->communicator == MPI_COMM_NULL)
            return;
        int finalized = 0;
        if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized ||
            MPI_Comm_free(&impl_->communicator) != MPI_SUCCESS)
        {
            LOG_ERROR(
                "[MoEOverlayMPIEconomyEvidenceExchange] Could not free its private communicator");
            std::terminate();
        }
    }

    bool MoEOverlayMPIEconomyEvidenceExchange::beginMigrationProfile(
        const MoEOverlayMigrationProfileEvidence &local,std::string *error)
    {
        if(!local.valid()){++impl_->stats.rejected_packets;assignError(error,"Invalid original-context migration-profile evidence");return false;}
        EvidenceCountHeader wire{};wire.world_rank=impl_->config.mpi_context->rank();wire.world_size=worldSize();wire.evidence_kind=1;
        wire.calibration_sequence=local.profile_sequence;wire.first_target_word=local.coordinate.source_target.word();
        wire.second_target_word=local.coordinate.destination_target.word();wire.layer=local.coordinate.layer;
        if(!impl_->beginEvidence(wire,local.context,local.local_measurements,Impl::Active::MigrationProfile,error))return false;
        ++impl_->stats.migration_profile_exchanges_started;impl_->record("mpi_economy_migration_profile_exchanges_started");return true;
    }

    MoEOverlayResidencyWaveProgress MoEOverlayMPIEconomyEvidenceExchange::pollMigrationProfile(
        MoEOverlayMigrationProfileResult *result,std::string *error)
    {
        if(!result){assignError(error,"Migration-profile exchange requires output owner");return MoEOverlayResidencyWaveProgress::Failed;}
        const auto progress=impl_->pollEvidence(Impl::Active::MigrationProfile,error);
        if(progress!=MoEOverlayResidencyWaveProgress::Ready)return progress;
        try {
            std::vector<MoEOverlayMigrationProfileEvidence> ranks;ranks.reserve(impl_->gathered_attempts.size());
            for(std::size_t rank=0;rank<impl_->gathered_attempts.size();++rank){const auto &wire=impl_->gathered_attempts[rank];
                ranks.push_back({.profile_sequence=wire.calibration_sequence,
                    .coordinate={MoEOverlayRouteTarget::fromWord(wire.first_target_word),
                        MoEOverlayRouteTarget::fromWord(wire.second_target_word),wire.layer},
                    .local_measurements=impl_->decodeEvidence(rank),.context=impl_->evidence_context});
            }
            *result=MoEOverlayEconomyEvidenceMerger::mergeMigrationProfile(ranks);
        }catch(const std::exception &exception){++impl_->stats.rejected_packets;assignError(error,exception.what());return MoEOverlayResidencyWaveProgress::Failed;}
        ++impl_->stats.migration_profile_exchanges_completed;impl_->record("mpi_economy_migration_profile_exchanges_completed");
        if(error)error->clear();return MoEOverlayResidencyWaveProgress::Ready;
    }

    bool MoEOverlayMPIEconomyEvidenceExchange::beginAttempt(
        const MoEOverlayCalibrationAttemptEvidence &local,std::string *error)
    {
        if(!local.valid()){++impl_->stats.rejected_packets;assignError(error,"Invalid original-context attempt evidence");return false;}
        EvidenceCountHeader wire{};wire.world_rank=impl_->config.mpi_context->rank();wire.world_size=worldSize();wire.evidence_kind=2;
        wire.calibration_sequence=local.calibration_sequence;wire.first_target_word=local.coordinate.source_target.word();
        wire.second_target_word=local.coordinate.destination_target.word();wire.layer=local.coordinate.layer;
        wire.source=static_cast<std::uint32_t>(local.source);wire.real_rows=local.workload.real_rows;
        wire.execution_rows=local.workload.execution_rows;wire.transaction_count=local.workload.transaction_count;
        wire.speculative_depth=local.workload.speculative_depth;wire.schedule_fingerprint=local.workload.schedule_fingerprint;
        wire.baseline_nanoseconds=local.baseline_nanoseconds;wire.concurrent_nanoseconds=local.concurrent_nanoseconds;
        wire.exact_overlap=local.exact_overlap?1u:0u;
        if(!impl_->beginEvidence(wire,local.context,local.local_measurements,Impl::Active::Attempt,error))return false;
        ++impl_->stats.attempt_exchanges_started;impl_->record("mpi_economy_attempt_exchanges_started");return true;
    }

    MoEOverlayResidencyWaveProgress MoEOverlayMPIEconomyEvidenceExchange::pollAttempt(
        MoEOverlayCalibrationAttemptResult *result,std::string *error)
    {
        if(!result){assignError(error,"Attempt exchange requires output owner");return MoEOverlayResidencyWaveProgress::Failed;}
        const auto progress=impl_->pollEvidence(Impl::Active::Attempt,error);
        if(progress!=MoEOverlayResidencyWaveProgress::Ready)return progress;
        try {
            std::vector<MoEOverlayCalibrationAttemptEvidence> ranks;ranks.reserve(impl_->gathered_attempts.size());
            for(std::size_t rank=0;rank<impl_->gathered_attempts.size();++rank){const auto &wire=impl_->gathered_attempts[rank];
                ranks.push_back({.calibration_sequence=wire.calibration_sequence,
                    .coordinate={MoEOverlayRouteTarget::fromWord(wire.first_target_word),
                        MoEOverlayRouteTarget::fromWord(wire.second_target_word),wire.layer},
                    .source=static_cast<ExpertHistogramSource>(wire.source),
                    .workload={.source=static_cast<ExpertHistogramSource>(wire.source),.real_rows=wire.real_rows,
                        .execution_rows=wire.execution_rows,.transaction_count=wire.transaction_count,
                        .speculative_depth=wire.speculative_depth,.schedule_fingerprint=wire.schedule_fingerprint},
                    .baseline_nanoseconds=wire.baseline_nanoseconds,.concurrent_nanoseconds=wire.concurrent_nanoseconds,
                    .exact_overlap=wire.exact_overlap!=0,.local_measurements=impl_->decodeEvidence(rank),.context=impl_->evidence_context});
            }
            *result=MoEOverlayEconomyEvidenceMerger::mergeAttempt(ranks);
        }catch(const std::exception &exception){++impl_->stats.rejected_packets;assignError(error,exception.what());return MoEOverlayResidencyWaveProgress::Failed;}
        ++impl_->stats.attempt_exchanges_completed;impl_->record("mpi_economy_attempt_exchanges_completed");
        if(error)error->clear();return MoEOverlayResidencyWaveProgress::Ready;
    }

    bool MoEOverlayMPIEconomyEvidenceExchange::beginCalibrationReadiness(
        const MoEOverlayCalibrationReadiness &local,
        std::string *error)
    {
        if (!local.valid())
        {
            ++impl_->stats.rejected_packets;
            assignError(
                error,
                "ExpertOverlay MPI economy exchange received invalid calibration readiness");
            return false;
        }

        CalibrationReadinessWire wire{};
        wire.kind = static_cast<std::uint16_t>(local.kind);
        wire.world_rank = impl_->config.mpi_context->rank();
        wire.world_size = impl_->config.mpi_context->world_size();
        wire.calibration_sequence = local.calibration_sequence;
        wire.first_participant = local.coordinate.source_participant;
        wire.second_participant = local.coordinate.destination_participant;
        wire.layer = local.coordinate.layer;
        wire.source = static_cast<std::uint32_t>(local.source);
        wire.real_rows = local.workload.real_rows;
        wire.execution_rows = local.workload.execution_rows;
        wire.transaction_count = local.workload.transaction_count;
        wire.speculative_depth = local.workload.speculative_depth;
        wire.schedule_fingerprint = local.workload.schedule_fingerprint;
        wire.packet_hash = readinessHash(wire);
        impl_->local_calibration_readiness = wire;
        std::fill(
            impl_->gathered_calibration_readiness.begin(),
            impl_->gathered_calibration_readiness.end(),
            CalibrationReadinessWire{});
        if (!impl_->beginRaw(
                &impl_->local_calibration_readiness,
                impl_->gathered_calibration_readiness.data(),
                sizeof(CalibrationReadinessWire),
                Impl::Active::CalibrationReadiness,
                error))
        {
            return false;
        }
        ++impl_->stats.calibration_readiness_exchanges_started;
        impl_->record(
            "mpi_economy_calibration_readiness_exchanges_started");
        LOG_DEBUG(
            "[ExpertOverlay][Economy] Rank "
            << impl_->config.mpi_context->rank()
            << " began baseline readiness sequence="
            << local.calibration_sequence << " source="
            << static_cast<std::uint32_t>(local.source) << " rows="
            << local.workload.real_rows << '/'
            << local.workload.execution_rows << " transactions="
            << local.workload.transaction_count << " fingerprint="
            << local.workload.schedule_fingerprint);
        return true;
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayMPIEconomyEvidenceExchange::pollCalibrationReadiness(
        std::string *error)
    {
        const auto progress = impl_->pollRaw(
            Impl::Active::CalibrationReadiness, error);
        if (progress != MoEOverlayResidencyWaveProgress::Ready)
            return progress;

        bool identity_mismatch = false;
        try
        {
            std::optional<MoEOverlayCalibrationReadiness> expected;
            for (std::size_t rank = 0;
                 rank < impl_->gathered_calibration_readiness.size();
                 ++rank)
            {
                const auto &wire =
                    impl_->gathered_calibration_readiness[rank];
                if (wire.magic != kReadinessMagic ||
                    wire.version != kWireVersion ||
                    wire.world_rank != static_cast<int>(rank) ||
                    wire.world_size != worldSize() || wire.packet_hash == 0 ||
                    wire.packet_hash != readinessHash(wire))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay MPI calibration readiness failed version, rank, or hash authentication");
                }
                const MoEOverlayCalibrationReadiness decoded{
                    .kind = static_cast<
                        MoEOverlayCalibrationReadinessKind>(wire.kind),
                    .calibration_sequence = wire.calibration_sequence,
                    .coordinate = {
                        .source_participant = wire.first_participant,
                        .destination_participant = wire.second_participant,
                        .layer = wire.layer,
                    },
                    .source =
                        static_cast<ExpertHistogramSource>(wire.source),
                    .workload = {
                        .source =
                            static_cast<ExpertHistogramSource>(wire.source),
                        .real_rows = wire.real_rows,
                        .execution_rows = wire.execution_rows,
                        .transaction_count = wire.transaction_count,
                        .speculative_depth = wire.speculative_depth,
                        .schedule_fingerprint = wire.schedule_fingerprint,
                    },
                };
                if (!decoded.valid())
                {
                    throw std::invalid_argument(
                        "ExpertOverlay MPI calibration readiness decoded a malformed baseline identity");
                }
                if (expected && decoded != *expected)
                    identity_mismatch = true;
                expected = decoded;
            }
        }
        catch (const std::exception &exception)
        {
            ++impl_->stats.rejected_packets;
            assignError(error, exception.what());
            return MoEOverlayResidencyWaveProgress::Failed;
        }

        if (identity_mismatch)
        {
            /*
             * Probe ownership is intentionally non-blocking: ranks can finish
             * different production chunks before this private all-gather is
             * polled. Both samples are authentic, but comparing their timing
             * would not be rigorous. Every rank sees the same gathered packet
             * order, so `Deferred` is a deterministic whole-attempt retry and
             * no migration resource has been reserved yet.
             */
            ++impl_->stats.calibration_readiness_retries;
            impl_->record(
                "mpi_economy_calibration_readiness_retries");
            if (error)
                error->clear();
            return MoEOverlayResidencyWaveProgress::Deferred;
        }

        ++impl_->stats.calibration_readiness_exchanges_completed;
        impl_->record(
            "mpi_economy_calibration_readiness_exchanges_completed");
        LOG_DEBUG(
            "[ExpertOverlay][Economy] Rank "
            << impl_->config.mpi_context->rank()
            << " completed baseline readiness");
        if (error)
            error->clear();
        return MoEOverlayResidencyWaveProgress::Ready;
    }

    bool MoEOverlayMPIEconomyEvidenceExchange::beginServiceReadiness(
        MoEOverlayServiceEvidenceReadiness local_readiness,
        std::string *error)
    {
        if (!idle())
        {
            ++impl_->stats.concurrent_start_rejections;
            assignError(
                error,
                "ExpertOverlay MPI economy evidence lane already owns an exchange");
            return false;
        }

        switch (local_readiness)
        {
        case MoEOverlayServiceEvidenceReadiness::Stopping:
        case MoEOverlayServiceEvidenceReadiness::AwaitingEvidence:
        case MoEOverlayServiceEvidenceReadiness::Ready:
            break;
        default:
            ++impl_->stats.rejected_packets;
            assignError(
                error,
                "ExpertOverlay MPI service readiness received an invalid typed disposition");
            return false;
        }

        impl_->local_service_readiness = static_cast<int>(local_readiness);
        impl_->global_service_readiness = 0;
        impl_->started_at = std::chrono::steady_clock::now();
        const int mpi_result = MPI_Iallreduce(
            &impl_->local_service_readiness,
            &impl_->global_service_readiness,
            1,
            MPI_INT,
            MPI_MIN,
            impl_->communicator,
            &impl_->request);
        if (mpi_result != MPI_SUCCESS ||
            impl_->request == MPI_REQUEST_NULL)
        {
            impl_->request = MPI_REQUEST_NULL;
            ++impl_->stats.mpi_failures;
            assignError(
                error,
                mpiError("MPI_Iallreduce", mpi_result));
            return false;
        }
        impl_->active = Impl::Active::ServiceReadiness;
        ++impl_->stats.service_readiness_exchanges_started;
        impl_->record("mpi_economy_service_readiness_exchanges_started");
        if (error)
            error->clear();
        return true;
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayMPIEconomyEvidenceExchange::pollServiceReadiness(
        MoEOverlayServiceEvidenceReadiness *global_readiness,
        std::string *error)
    {
        if (!global_readiness)
        {
            assignError(
                error,
                "ExpertOverlay MPI service readiness requires an output owner");
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        const auto progress = impl_->pollRaw(
            Impl::Active::ServiceReadiness, error);
        if (progress != MoEOverlayResidencyWaveProgress::Ready)
            return progress;

        if (impl_->global_service_readiness <
                static_cast<int>(
                    MoEOverlayServiceEvidenceReadiness::Stopping) ||
            impl_->global_service_readiness >
                static_cast<int>(MoEOverlayServiceEvidenceReadiness::Ready))
        {
            ++impl_->stats.rejected_packets;
            assignError(
                error,
                "ExpertOverlay MPI service readiness returned an invalid typed reduction");
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        *global_readiness = static_cast<
            MoEOverlayServiceEvidenceReadiness>(
            impl_->global_service_readiness);
        ++impl_->stats.service_readiness_exchanges_completed;
        if (*global_readiness ==
            MoEOverlayServiceEvidenceReadiness::AwaitingEvidence)
        {
            ++impl_->stats.service_readiness_incomplete;
            impl_->record("mpi_economy_service_readiness_incomplete");
        }
        else if (*global_readiness ==
                 MoEOverlayServiceEvidenceReadiness::Stopping)
        {
            ++impl_->stats.service_readiness_stops;
            impl_->record("mpi_economy_service_readiness_stops");
        }
        impl_->record("mpi_economy_service_readiness_exchanges_completed");
        if (error)
            error->clear();
        return MoEOverlayResidencyWaveProgress::Ready;
    }

    bool MoEOverlayMPIEconomyEvidenceExchange::beginService(
        const std::vector<MoEOverlayParticipantLayerServiceTotals> &local,
        std::string *error)
    {
        if (!idle())
        {
            ++impl_->stats.concurrent_start_rejections;
            assignError(
                error,
                "ExpertOverlay MPI economy evidence lane already owns an exchange");
            return false;
        }
        std::fill(
            impl_->local_service.begin(),
            impl_->local_service.end(),
            std::uint64_t{0});
        auto *header = reinterpret_cast<ServiceHeaderWire *>(
            impl_->local_service.data());
        *header = {
            .world_rank = worldRank(),
            .world_size = worldSize(),
            .participant_count = static_cast<std::uint32_t>(
                impl_->config.owner_map.participants().size()),
            .num_layers = static_cast<std::uint32_t>(
                impl_->numLayers()),
            .active_source_mask =
                encodeSourceMask(impl_->reachableSources()),
            .economy_source_mask = encodeSourceMask(
                impl_->config.production_topology.economyActiveSources()),
            .production_topology_fingerprint =
                productionTopologyFingerprint(
                    impl_->config.production_topology),
            .cell_count = static_cast<std::uint64_t>(
                impl_->config.owner_map.participants().size()) *
                static_cast<std::uint64_t>(impl_->numLayers()),
            .packet_bytes = impl_->service_packet_bytes,
        };
        auto *cells = reinterpret_cast<ServiceCellWire *>(
            reinterpret_cast<std::uint8_t *>(
                impl_->local_service.data()) +
            sizeof(ServiceHeaderWire));
        std::vector<int> participant_ids;
        participant_ids.reserve(
            impl_->config.owner_map.participants().size());
        for (const auto &participant :
             impl_->config.owner_map.participants())
        {
            participant_ids.push_back(participant.participant_id);
        }
        std::sort(participant_ids.begin(), participant_ids.end());
        for (std::size_t participant = 0;
             participant < participant_ids.size();
             ++participant)
        {
            for (int layer = 0;
                 layer < impl_->numLayers();
                 ++layer)
            {
                auto &cell = cells[
                    participant *
                        static_cast<std::size_t>(impl_->numLayers()) +
                    static_cast<std::size_t>(layer)];
                cell.participant_id = participant_ids[participant];
                cell.layer = layer;
            }
        }
        for (const auto &row : local)
        {
            const auto participant = std::lower_bound(
                participant_ids.begin(), participant_ids.end(),
                row.participant_id);
            const auto *owner =
                impl_->config.owner_map.participantForId(row.participant_id);
            if (!row.valid())
            {
                ++impl_->stats.rejected_packets;
                assignError(
                    error,
                    "ExpertOverlay MPI service exchange received a malformed local row");
                return false;
            }
            if (participant == participant_ids.end() ||
                *participant != row.participant_id || !owner)
            {
                ++impl_->stats.rejected_packets;
                assignError(
                    error,
                    "ExpertOverlay MPI service exchange received an unknown local participant");
                return false;
            }
            if (!owner->world_rank_known ||
                owner->world_rank != worldRank())
            {
                ++impl_->stats.rejected_packets;
                std::ostringstream diagnostic;
                diagnostic
                    << "ExpertOverlay MPI service exchange received participant "
                    << row.participant_id << " on world rank " << worldRank()
                    << " but its authoritative owner rank is "
                    << (owner->world_rank_known ? owner->world_rank : -1);
                assignError(error, diagnostic.str());
                return false;
            }
            if (row.layer < 0 || row.layer >= impl_->numLayers())
            {
                ++impl_->stats.rejected_packets;
                std::ostringstream diagnostic;
                diagnostic
                    << "ExpertOverlay MPI service exchange received out-of-range layer "
                    << row.layer << " for " << impl_->numLayers()
                    << " retained layers";
                assignError(error, diagnostic.str());
                return false;
            }
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                if (!impl_->config.production_topology.reachable(
                        row.layer, phase) &&
                    row.sample_count[phase] != 0)
                {
                    ++impl_->stats.rejected_packets;
                    std::ostringstream diagnostic;
                    diagnostic
                        << "ExpertOverlay MPI service exchange received "
                        << row.sample_count[phase]
                        << " samples for unreachable layer " << row.layer
                        << " phase " << phase;
                    assignError(error, diagnostic.str());
                    return false;
                }
            }
            const std::size_t participant_offset =
                static_cast<std::size_t>(
                    participant - participant_ids.begin());
            auto &cell = cells[
                participant_offset *
                    static_cast<std::size_t>(impl_->numLayers()) +
                static_cast<std::size_t>(row.layer)];
            if (cell.present != 0)
            {
                ++impl_->stats.rejected_packets;
                assignError(
                    error,
                    "ExpertOverlay MPI service exchange received a duplicate local row");
                return false;
            }
            cell.present = 1;
            cell.total_nanoseconds = row.total_nanoseconds;
            cell.activation_count = row.activation_count;
            cell.sample_count = row.sample_count;
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                cell.overflowed[phase] = row.overflowed[phase] ? 1u : 0u;
            }
        }
        header->packet_hash = serviceHash(
            impl_->local_service.data(), impl_->service_packet_bytes);
        std::fill(
            impl_->gathered_service.begin(),
            impl_->gathered_service.end(),
            std::uint64_t{0});
        if (!impl_->beginRaw(
                impl_->local_service.data(),
                impl_->gathered_service.data(),
                impl_->service_packet_bytes,
                Impl::Active::Service,
                error))
        {
            return false;
        }
        ++impl_->stats.service_exchanges_started;
        impl_->record("mpi_economy_service_exchanges_started");
        return true;
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayMPIEconomyEvidenceExchange::pollService(
        std::vector<MoEOverlayParticipantLayerServiceTotals> *result,
        std::string *error)
    {
        if (!result)
        {
            assignError(
                error,
                "ExpertOverlay MPI service exchange requires an output owner");
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        const auto progress = impl_->pollRaw(Impl::Active::Service, error);
        if (progress != MoEOverlayResidencyWaveProgress::Ready)
            return progress;
        try
        {
            std::vector<std::vector<
                MoEOverlayParticipantLayerServiceTotals>> rank_rows(
                static_cast<std::size_t>(worldSize()));
            const std::size_t expected_cells =
                impl_->config.owner_map.participants().size() *
                static_cast<std::size_t>(impl_->numLayers());
            for (int rank = 0; rank < worldSize(); ++rank)
            {
                const auto *begin = reinterpret_cast<const std::uint8_t *>(
                    impl_->gathered_service.data() +
                    static_cast<std::size_t>(rank) *
                        impl_->service_packet_words);
                const auto *header =
                    reinterpret_cast<const ServiceHeaderWire *>(begin);
                if (header->magic != kServiceMagic ||
                    header->version != kWireVersion ||
                    header->reserved != 0 || header->world_rank != rank ||
                    header->world_size != worldSize() ||
                    header->participant_count !=
                        impl_->config.owner_map.participants().size() ||
                    header->num_layers !=
                        static_cast<std::uint32_t>(
                            impl_->numLayers()) ||
                    header->active_source_mask !=
                        encodeSourceMask(impl_->reachableSources()) ||
                    header->economy_source_mask != encodeSourceMask(
                        impl_->config.production_topology
                            .economyActiveSources()) ||
                    header->production_topology_fingerprint !=
                        productionTopologyFingerprint(
                            impl_->config.production_topology) ||
                    header->cell_count != expected_cells ||
                    header->packet_bytes != impl_->service_packet_bytes ||
                    header->packet_hash == 0 ||
                    header->packet_hash != serviceHash(
                        begin, impl_->service_packet_bytes))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay MPI service packet failed version, geometry, rank, or hash authentication");
                }
                const auto *cells = reinterpret_cast<
                    const ServiceCellWire *>(
                    begin + sizeof(ServiceHeaderWire));
                for (std::size_t cell_index = 0;
                     cell_index < expected_cells;
                     ++cell_index)
                {
                    const auto &cell = cells[cell_index];
                    if (cell.present == 0)
                        continue;
                    if (cell.present != 1 || cell.reserved != 0)
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay MPI service packet has an invalid presence marker");
                    }
                    MoEOverlayParticipantLayerServiceTotals row{
                        .participant_id = cell.participant_id,
                        .layer = cell.layer,
                        .total_nanoseconds = cell.total_nanoseconds,
                        .activation_count = cell.activation_count,
                        .sample_count = cell.sample_count,
                    };
                    for (std::size_t phase = 0;
                         phase < kExpertHistogramProductionSourceCount;
                         ++phase)
                    {
                        if (cell.overflowed[phase] > 1)
                        {
                            throw std::invalid_argument(
                                "ExpertOverlay MPI service packet has an invalid overflow marker");
                        }
                        row.overflowed[phase] =
                            cell.overflowed[phase] != 0;
                    }
                    rank_rows[static_cast<std::size_t>(rank)].push_back(
                        std::move(row));
                }
            }
            *result = MoEOverlayEconomyEvidenceMerger::mergeService(
                rank_rows,
                impl_->config.owner_map,
                impl_->config.production_topology);
        }
        catch (const std::exception &exception)
        {
            ++impl_->stats.rejected_packets;
            assignError(error, exception.what());
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        ++impl_->stats.service_exchanges_completed;
        impl_->record("mpi_economy_service_exchanges_completed");
        if (error)
            error->clear();
        return MoEOverlayResidencyWaveProgress::Ready;
    }

    bool MoEOverlayMPIEconomyEvidenceExchange::idle() const noexcept
    {
        return impl_->active == Impl::Active::Idle &&
               impl_->request == MPI_REQUEST_NULL;
    }

    int MoEOverlayMPIEconomyEvidenceExchange::worldRank() const noexcept
    {
        return impl_->config.mpi_context->rank();
    }

    int MoEOverlayMPIEconomyEvidenceExchange::worldSize() const noexcept
    {
        return impl_->config.mpi_context->world_size();
    }

    MoEOverlayMPIEconomyEvidenceExchangeStats
    MoEOverlayMPIEconomyEvidenceExchange::stats() const noexcept
    {
        return impl_->stats;
    }
} // namespace llaminar2
