/**
 * @file ExpertWeightTransfer.h
 * @brief Cross-rank MPI transfer of pre-packed GEMM weights for MoE expert rebalancing.
 *
 * When experts migrate between NUMA sockets (MPI ranks) during rebalancing,
 * this avoids repacking from scratch by transferring the already-packed weights
 * via MPI. Uses a two-phase protocol:
 *   Phase A (blocking):  Exchange sizes so receivers can pre-allocate buffers.
 *   Phase B (non-blocking): Bulk data transfer with MPI_Isend/Irecv + Waitall.
 *
 * Header-only. No dependency on DeviceGraphOrchestrator, MoEExpertComputeStage, or KernelFactory.
 */

#pragma once

#include "MoELayeredExpertOwnership.h"
#include "../../kernels/IPackedWeights.h"
#include "../../kernels/PackedWeightsSerialization.h"
#include "../../memory/NUMAAllocator.h"
#include "../../tensors/AlignedVector.h"
#include "../../utils/Logger.h"
#include "../../utils/MPITags.h"

#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    class ITensorGemm;

    /// One exact layer/expert migration between ownership participants.
    struct ExpertMigration
    {
        int layer_idx;
        int expert_id;
        int src_rank;
        int dst_rank;

        bool operator==(const ExpertMigration &) const = default;
    };

    /**
     * @brief Uninitialized, cache-line-aligned wire storage for one projection.
     *
     * MPI writes every logical byte, so zero-initializing these buffers would
     * consume memory bandwidth and establish the wrong first-touch policy just
     * before the receive overwrites them.
     */
    using ExpertTransferBuffer = AlignedVector<uint8_t>;

    /// Serialized weight blobs for one expert's 3 projections.
    struct ExpertWeightBlobs
    {
        ExpertTransferBuffer gate;
        ExpertTransferBuffer up;
        ExpertTransferBuffer down;

        bool empty() const { return gate.empty() && up.empty() && down.empty(); }
        size_t totalBytes() const { return gate.size() + up.size() + down.size(); }
    };

    /// Key for received weights: [layer_idx][expert_id] → blobs
    using ReceivedWeightsMap = std::unordered_map<int, std::unordered_map<int, ExpertWeightBlobs>>;

    /**
     * @brief Three already-prepared projection objects detached from one expert.
     *
     * Ownership migration moves these objects out of the source GEMM engines in
     * O(1). Replica publication may instead populate them with explicit clones.
     * Neither operation constructs a concatenated wire-format byte vector.
     */
    struct ExpertPackedWeights
    {
        std::unique_ptr<IPackedWeights> gate;
        std::unique_ptr<IPackedWeights> up;
        std::unique_ptr<IPackedWeights> down;

        bool complete() const noexcept
        {
            return gate != nullptr && up != nullptr && down != nullptr;
        }

        size_t totalBytes() const noexcept
        {
            return (gate ? gate->sizeBytes() : 0) +
                   (up ? up->sizeBytes() : 0) +
                   (down ? down->sizeBytes() : 0);
        }
    };

    /**
     * @brief Final shared CPU GEMM engines published for one arriving expert.
     *
     * Multiple cached graph stages for the same layer adopt these exact shared
     * engines. This prevents every graph instance from deserializing another
     * private copy and gives PreparedWeightStore one canonical arrival lifetime.
     */
    struct PreparedExpertEngines
    {
        std::shared_ptr<ITensorGemm> gate;
        std::shared_ptr<ITensorGemm> up;
        std::shared_ptr<ITensorGemm> down;
        size_t packed_bytes = 0;

        bool complete() const noexcept
        {
            return gate != nullptr && up != nullptr && down != nullptr;
        }
    };

    /// Final direct CPU arrivals keyed by exact layer and expert identity.
    using ReceivedPreparedExpertsMap =
        std::unordered_map<int, std::unordered_map<int, PreparedExpertEngines>>;

    /**
     * @brief Backend-explicit arrivals produced by one migration wave.
     *
     * Homogeneous CPU movement fills `prepared`; cross-backend conversion paths
     * fill `serialized`. A single migration must never populate both forms for
     * the same layer/expert, which keeps arrival ownership unambiguous.
     */
    struct ExpertTransferResult
    {
        ReceivedWeightsMap serialized;
        ReceivedPreparedExpertsMap prepared;

        bool empty() const noexcept
        {
            return serialized.empty() && prepared.empty();
        }
    };

    /**
     * @brief Exact phase and payload evidence for one rank's migration wave.
     *
     * Dynamic CPU expert movement is intentionally an amortized operation, but
     * it still has to be economical.  This record separates packed-weight
     * extraction, metadata exchange, destination allocation, payload exchange,
     * and result publication so end-to-end PerfStats can identify the expensive
     * phase without relying on noisy log timestamps.
     */
    struct ExpertTransferEvidence
    {
        size_t manifest_entries = 0;
        size_t outgoing_entries = 0;
        size_t incoming_entries = 0;
        size_t outgoing_bytes = 0;
        size_t incoming_bytes = 0;
        uint64_t payload_prepare_ns = 0;
        uint64_t metadata_exchange_ns = 0;
        uint64_t receive_allocation_ns = 0;
        uint64_t payload_exchange_ns = 0;
        uint64_t result_publication_ns = 0;
        uint64_t total_ns = 0;
    };

    /**
     * @brief Coordinates MPI transfer of pre-packed GEMM weights between ranks.
     *
     * All methods are static. Pure-logic helpers (buildManifest, departingExperts,
     * arrivingMigrations) require no MPI. `transferLayered()` performs the
     * exact cross-rank communication and throws on protocol failure. Raw expert
     * repacking after host release is not a supported recovery path.
     */
    class ExpertWeightTransfer
    {
    public:
        // ── Manifest Building (pure logic, no MPI) ──────────────────────

        /**
         * @brief Build migration manifest from old→new placement.
         * @return Entries for experts that changed rank.
         */
        static inline std::vector<ExpertMigration> buildManifest(
            const MoELayeredExpertOwnership &old_ownership,
            const MoELayeredExpertOwnership &new_ownership)
        {
            std::vector<ExpertMigration> manifest;
            for (const auto &change : new_ownership.changesFrom(old_ownership))
            {
                manifest.push_back({
                    .layer_idx = change.layer_idx,
                    .expert_id = change.expert_id,
                    .src_rank = change.previous_participant,
                    .dst_rank = change.current_participant,
                });
            }
            return manifest;
        }

        /**
         * @brief Get expert IDs this rank is sending (departing experts).
         */
        static inline std::vector<ExpertMigration> departingMigrations(
            const std::vector<ExpertMigration> &manifest,
            int my_rank)
        {
            std::vector<ExpertMigration> result;
            for (const auto &m : manifest)
            {
                if (m.src_rank == my_rank)
                    result.push_back(m);
            }
            return result;
        }

        /**
         * @brief Get expert IDs this rank is receiving (arriving experts).
         */
        static inline std::vector<ExpertMigration> arrivingMigrations(
            const std::vector<ExpertMigration> &manifest,
            int my_rank)
        {
            std::vector<ExpertMigration> result;
            for (const auto &m : manifest)
            {
                if (m.dst_rank == my_rank)
                    result.push_back(m);
            }
            return result;
        }

        // ── MPI Transfer ────────────────────────────────────────────────

        /**
         * @brief Move CPU packed sections directly into final destination engines.
         *
         * This is the homogeneous CPU ownership/replica protocol. The source
         * callback returns native packed objects; metadata is exchanged in one
         * non-blocking wave; MPI then writes every data section directly into
         * its final NUMA-local allocation. The engine factory consumes those
         * allocations without serialization or deserialization copies.
         *
         * @param manifest Exact layer/expert migration records.
         * @param get_weights Destructive detach or explicit clone callback for
         *        locally sourced experts.
         * @param make_engine Factory that consumes one final packed projection.
         * @param my_rank Calling MPI rank.
         * @param target_numa_node Exact NUMA node that owns final receive pages.
         * @param comm Communicator shared by every manifest participant.
         * @param evidence Optional exact phase evidence destination.
         * @return Shared prepared engines for this rank's arrivals.
         * @throws std::runtime_error or std::invalid_argument on any malformed
         *         payload, unsupported packed representation, or MPI failure.
         */
        static inline ReceivedPreparedExpertsMap transferLayeredPreparedCPU(
            const std::vector<ExpertMigration> &manifest,
            std::function<ExpertPackedWeights(int layer_idx, int expert_id)>
                get_weights,
            std::function<std::shared_ptr<ITensorGemm>(
                std::unique_ptr<IPackedWeights>)>
                make_engine,
            int my_rank,
            int target_numa_node,
            MPI_Comm comm,
            ExpertTransferEvidence *evidence = nullptr)
        {
            if (evidence)
                *evidence = ExpertTransferEvidence{};
            if (manifest.empty())
                return {};
            if (!get_weights || !make_engine)
            {
                throw std::invalid_argument(
                    "Direct CPU expert transfer requires source and engine factories");
            }
            if (target_numa_node < 0)
            {
                throw std::invalid_argument(
                    "Direct CPU expert transfer requires one exact destination NUMA node");
            }

            using Clock = std::chrono::steady_clock;
            using Descriptor = packed_weights_serialization::
                PackedWeightsTransferDescriptor;
            using ConstViews = packed_weights_serialization::
                PackedWeightsConstSectionViews;
            using ReceiveTarget = packed_weights_serialization::
                PackedWeightsReceiveTarget;
            constexpr size_t projection_count = 3;
            constexpr size_t section_count = packed_weights_serialization::
                PACKED_WEIGHT_SECTION_COUNT;

            const auto t0 = Clock::now();
            if (evidence)
                evidence->manifest_entries = manifest.size();

            auto require_mpi_success = [](int rc, const char *operation)
            {
                if (rc == MPI_SUCCESS)
                    return;
                throw std::runtime_error(
                    std::string("ExpertWeightTransfer ") + operation +
                    " failed with MPI error " + std::to_string(rc));
            };

            for (const auto &migration : manifest)
            {
                if (migration.layer_idx < 0 || migration.expert_id < 0 ||
                    migration.src_rank < 0 || migration.dst_rank < 0 ||
                    migration.src_rank == migration.dst_rank)
                {
                    throw std::invalid_argument(
                        "Direct CPU expert transfer manifest contains an invalid layered migration");
                }
            }

            struct SendEntry
            {
                int layer = -1;
                int expert_id = -1;
                ExpertPackedWeights weights;
                std::array<Descriptor, projection_count> descriptors{};
                std::array<ConstViews, projection_count> views{};
            };
            std::vector<SendEntry> send_entries;

            auto find_send_entry = [&](int layer, int expert_id) -> SendEntry *
            {
                auto it = std::find_if(
                    send_entries.begin(),
                    send_entries.end(),
                    [&](const SendEntry &entry)
                    {
                        return entry.layer == layer &&
                               entry.expert_id == expert_id;
                    });
                return it == send_entries.end() ? nullptr : &*it;
            };

            for (const auto &migration : manifest)
            {
                if (migration.src_rank != my_rank ||
                    find_send_entry(
                        migration.layer_idx, migration.expert_id) != nullptr)
                {
                    continue;
                }

                ExpertPackedWeights weights = get_weights(
                    migration.layer_idx, migration.expert_id);
                if (!weights.complete())
                {
                    throw std::runtime_error(
                        "Direct CPU expert transfer source returned an incomplete prepared expert");
                }

                send_entries.push_back({
                    .layer = migration.layer_idx,
                    .expert_id = migration.expert_id,
                    .weights = std::move(weights),
                });
                auto &entry = send_entries.back();
                std::array<IPackedWeights *, projection_count> projections{
                    entry.weights.gate.get(),
                    entry.weights.up.get(),
                    entry.weights.down.get(),
                };
                for (size_t projection = 0;
                     projection < projection_count;
                     ++projection)
                {
                    if (!packed_weights_serialization::describeTransfer(
                            *projections[projection],
                            entry.descriptors[projection],
                            entry.views[projection]))
                    {
                        throw std::runtime_error(
                            "Direct CPU expert transfer cannot describe a source projection");
                    }
                    if (entry.descriptors[projection].header.has_native_blocks)
                    {
                        throw std::runtime_error(
                            "Direct CPU expert transfer requires eager interleaved weights, not deferred native blocks");
                    }
                }
                if (evidence)
                {
                    ++evidence->outgoing_entries;
                    evidence->outgoing_bytes += entry.weights.totalBytes();
                }
            }

            const auto payload_prepare_end = Clock::now();
            if (evidence)
            {
                evidence->payload_prepare_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        payload_prepare_end - t0)
                        .count());
            }

            struct ReceiveEntry
            {
                int layer = -1;
                int expert_id = -1;
                int src_rank = -1;
                std::array<Descriptor, projection_count> descriptors{};
                std::array<ReceiveTarget, projection_count> targets{};
            };
            std::vector<ReceiveEntry> receive_entries;
            receive_entries.reserve(static_cast<size_t>(std::count_if(
                manifest.begin(),
                manifest.end(),
                [&](const ExpertMigration &migration)
                {
                    return migration.dst_rank == my_rank;
                })));
            for (const auto &migration : manifest)
            {
                if (migration.dst_rank != my_rank)
                    continue;
                receive_entries.push_back({
                    .layer = migration.layer_idx,
                    .expert_id = migration.expert_id,
                    .src_rank = migration.src_rank,
                });
                if (evidence)
                    ++evidence->incoming_entries;
            }

            std::vector<MPI_Request> metadata_requests;
            metadata_requests.reserve(
                receive_entries.size() + manifest.size());
            for (auto &entry : receive_entries)
            {
                MPI_Request request;
                require_mpi_success(
                    MPI_Irecv(
                        entry.descriptors.data(),
                        static_cast<int>(sizeof(entry.descriptors)),
                        MPI_BYTE,
                        entry.src_rank,
                        mpi_tags::weightTransferSizeTag(
                            entry.layer, entry.expert_id),
                        comm,
                        &request),
                    "direct metadata receive");
                metadata_requests.push_back(request);
            }
            for (const auto &migration : manifest)
            {
                if (migration.src_rank != my_rank)
                    continue;
                auto *entry = find_send_entry(
                    migration.layer_idx, migration.expert_id);
                if (!entry)
                {
                    throw std::logic_error(
                        "Direct CPU expert transfer lost source metadata");
                }
                MPI_Request request;
                require_mpi_success(
                    MPI_Isend(
                        entry->descriptors.data(),
                        static_cast<int>(sizeof(entry->descriptors)),
                        MPI_BYTE,
                        migration.dst_rank,
                        mpi_tags::weightTransferSizeTag(
                            migration.layer_idx, migration.expert_id),
                        comm,
                        &request),
                    "direct metadata send");
                metadata_requests.push_back(request);
            }
            if (!metadata_requests.empty())
            {
                std::vector<MPI_Status> statuses(metadata_requests.size());
                require_mpi_success(
                    MPI_Waitall(
                        static_cast<int>(metadata_requests.size()),
                        metadata_requests.data(),
                        statuses.data()),
                    "direct metadata waitall");
            }

            const auto metadata_exchange_end = Clock::now();
            if (evidence)
            {
                evidence->metadata_exchange_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        metadata_exchange_end - payload_prepare_end)
                        .count());
            }

            std::vector<MPI_Request> payload_requests;
            for (auto &entry : receive_entries)
            {
                for (size_t projection = 0;
                     projection < projection_count;
                     ++projection)
                {
                    if (entry.descriptors[projection].header.has_native_blocks)
                    {
                        throw std::runtime_error(
                            "Direct CPU expert receive rejects deferred native-block weights");
                    }
                    entry.targets[projection] =
                        packed_weights_serialization::allocateTransferTarget(
                            entry.descriptors[projection]);
                    for (size_t section = 0; section < section_count; ++section)
                    {
                        const size_t bytes =
                            entry.targets[projection].sizes[section];
                        if (bytes == 0)
                            continue;
                        if (!NUMAAllocator::instance().
                                bindUntouchedExternalRangeToNode(
                                    entry.targets[projection].data[section],
                                    bytes,
                                    target_numa_node))
                        {
                            throw std::runtime_error(
                                "Direct CPU expert transfer could not bind final receive storage to its rank-local NUMA node");
                        }
                        if (bytes > static_cast<size_t>(INT_MAX))
                        {
                            throw std::length_error(
                                "Direct CPU expert receive section exceeds the MPI int limit");
                        }
                        MPI_Request request;
                        require_mpi_success(
                            MPI_Irecv(
                                entry.targets[projection].data[section],
                                static_cast<int>(bytes),
                                MPI_BYTE,
                                entry.src_rank,
                                mpi_tags::weightTransferDataTag(
                                    entry.layer,
                                    entry.expert_id,
                                    static_cast<int>(projection)),
                                comm,
                                &request),
                            "direct payload receive");
                        payload_requests.push_back(request);
                    }
                }
            }

            const auto receive_allocation_end = Clock::now();
            if (evidence)
            {
                evidence->receive_allocation_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        receive_allocation_end - metadata_exchange_end)
                        .count());
            }

            for (const auto &migration : manifest)
            {
                if (migration.src_rank != my_rank)
                    continue;
                auto *entry = find_send_entry(
                    migration.layer_idx, migration.expert_id);
                if (!entry)
                {
                    throw std::logic_error(
                        "Direct CPU expert transfer lost source payload");
                }
                for (size_t projection = 0;
                     projection < projection_count;
                     ++projection)
                {
                    for (size_t section = 0; section < section_count; ++section)
                    {
                        const size_t bytes =
                            entry->views[projection].sizes[section];
                        if (bytes == 0)
                            continue;
                        if (bytes > static_cast<size_t>(INT_MAX))
                        {
                            throw std::length_error(
                                "Direct CPU expert send section exceeds the MPI int limit");
                        }
                        MPI_Request request;
                        require_mpi_success(
                            MPI_Isend(
                                entry->views[projection].data[section],
                                static_cast<int>(bytes),
                                MPI_BYTE,
                                migration.dst_rank,
                                mpi_tags::weightTransferDataTag(
                                    migration.layer_idx,
                                    migration.expert_id,
                                    static_cast<int>(projection)),
                                comm,
                                &request),
                            "direct payload send");
                        payload_requests.push_back(request);
                    }
                }
            }
            if (!payload_requests.empty())
            {
                std::vector<MPI_Status> statuses(payload_requests.size());
                require_mpi_success(
                    MPI_Waitall(
                        static_cast<int>(payload_requests.size()),
                        payload_requests.data(),
                        statuses.data()),
                    "direct payload waitall");
            }

            const auto payload_exchange_end = Clock::now();
            if (evidence)
            {
                evidence->payload_exchange_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        payload_exchange_end - receive_allocation_end)
                        .count());
            }

            ReceivedPreparedExpertsMap result;
            size_t incoming_bytes = 0;
            for (auto &entry : receive_entries)
            {
                PreparedExpertEngines prepared;
                std::array<std::shared_ptr<ITensorGemm> *, projection_count>
                    engines{&prepared.gate, &prepared.up, &prepared.down};
                for (size_t projection = 0;
                     projection < projection_count;
                     ++projection)
                {
                    for (size_t section = 0; section < section_count; ++section)
                        prepared.packed_bytes +=
                            entry.targets[projection].sizes[section];
                    *engines[projection] = make_engine(
                        std::move(entry.targets[projection].weights));
                    if (!*engines[projection])
                    {
                        throw std::runtime_error(
                            "Direct CPU expert transfer could not construct a destination GEMM engine");
                    }
                }
                incoming_bytes += prepared.packed_bytes;
                auto [it, inserted] = result[entry.layer].emplace(
                    entry.expert_id, std::move(prepared));
                (void)it;
                if (!inserted)
                {
                    throw std::runtime_error(
                        "Direct CPU expert transfer received duplicate layer/expert ownership");
                }
            }

            const auto t1 = Clock::now();
            if (evidence)
            {
                evidence->incoming_bytes = incoming_bytes;
                evidence->result_publication_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        t1 - payload_exchange_end)
                        .count());
                evidence->total_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        t1 - t0)
                        .count());
            }
            return result;
        }

        /**
         * @brief Execute exact layer/expert cross-rank weight transfers.
         *
         * Two-phase protocol:
         *   Phase A: Blocking size exchange (24 bytes per migration).
         *   Phase B: Non-blocking bulk data transfer with MPI_Waitall.
         *
         * @param manifest     Migration entries (from buildManifest).
         * @param get_blobs    Callback: (layer_idx, expert_id) → serialized blobs.
         *                     Called only for experts this rank is SENDING.
         * @param my_rank      This rank's ID.
         * @param comm         MPI communicator.
         * @param evidence     Optional caller-owned phase evidence destination.
         * @return Map of received weights: [layer_idx][expert_id] → blobs.
         * @throws std::runtime_error when the MPI protocol or source payload
         *         fails. There is no repack or replay recovery path.
         */
        static inline ReceivedWeightsMap transferLayered(
            const std::vector<ExpertMigration> &manifest,
            std::function<ExpertWeightBlobs(int layer_idx, int expert_id)> get_blobs,
            int my_rank,
            MPI_Comm comm,
            ExpertTransferEvidence *evidence = nullptr)
        {
            if (evidence)
                *evidence = ExpertTransferEvidence{};
            if (manifest.empty())
                return {};

            using Clock = std::chrono::steady_clock;
            const auto t0 = Clock::now();
            if (evidence)
                evidence->manifest_entries = manifest.size();

            auto require_mpi_success = [](int rc, const char *operation)
            {
                if (rc == MPI_SUCCESS)
                    return;
                throw std::runtime_error(
                    std::string("ExpertWeightTransfer ") + operation +
                    " failed with MPI error " + std::to_string(rc));
            };

            // Collect each distinct local layer/expert payload once. Replica
            // manifests may send the same resident payload to several targets.
            struct BlobKey
            {
                int layer = -1;
                int expert_id = -1;
            };
            std::vector<BlobKey> send_keys;
            std::vector<ExpertWeightBlobs> send_blobs_store;

            auto find_send_index = [&](int layer, int expert_id) -> int
            {
                for (size_t index = 0; index < send_keys.size(); ++index)
                {
                    if (send_keys[index].layer == layer &&
                        send_keys[index].expert_id == expert_id)
                    {
                        return static_cast<int>(index);
                    }
                }
                return -1;
            };

            for (const auto &migration : manifest)
            {
                if (migration.layer_idx < 0 || migration.expert_id < 0 ||
                    migration.src_rank < 0 || migration.dst_rank < 0 ||
                    migration.src_rank == migration.dst_rank)
                {
                    throw std::invalid_argument(
                        "ExpertWeightTransfer manifest contains an invalid layered migration");
                }
                if (migration.src_rank != my_rank ||
                    find_send_index(migration.layer_idx, migration.expert_id) >= 0)
                {
                    continue;
                }

                auto blobs = get_blobs(
                    migration.layer_idx, migration.expert_id);
                if (blobs.empty())
                {
                    throw std::runtime_error(
                        "ExpertWeightTransfer source returned an empty prepared expert payload");
                }
                send_keys.push_back({migration.layer_idx, migration.expert_id});
                if (evidence)
                {
                    ++evidence->outgoing_entries;
                    evidence->outgoing_bytes += blobs.totalBytes();
                }
                send_blobs_store.push_back(std::move(blobs));
            }

            const auto payload_prepare_end = Clock::now();
            if (evidence)
            {
                evidence->payload_prepare_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        payload_prepare_end - t0)
                        .count());
            }

            struct RecvSizeEntry
            {
                int expert_id = -1;
                int layer = -1;
                int src_rank = -1;
                uint64_t sizes[3]{};
            };
            std::vector<RecvSizeEntry> recv_size_entries;

            // Phase A exchanges exactly one size record per layered migration.
            // Every rank walks the same manifest order, keeping reciprocal
            // ownership swap pairs deterministic without a global barrier.
            for (const auto &migration : manifest)
            {
                const int tag = mpi_tags::weightTransferSizeTag(
                    migration.layer_idx, migration.expert_id);
                if (migration.src_rank == my_rank)
                {
                    const int index = find_send_index(
                        migration.layer_idx, migration.expert_id);
                    if (index < 0)
                    {
                        throw std::logic_error(
                            "ExpertWeightTransfer lost a source payload before size exchange");
                    }
                    const auto &blobs = send_blobs_store[static_cast<size_t>(index)];
                    uint64_t sizes[3] = {
                        blobs.gate.size(), blobs.up.size(), blobs.down.size()};
                    require_mpi_success(
                        MPI_Send(
                            sizes, 3, MPI_UINT64_T, migration.dst_rank, tag, comm),
                        "size send");
                }
                else if (migration.dst_rank == my_rank)
                {
                    RecvSizeEntry entry;
                    entry.expert_id = migration.expert_id;
                    entry.layer = migration.layer_idx;
                    entry.src_rank = migration.src_rank;
                    require_mpi_success(
                        MPI_Recv(
                            entry.sizes,
                            3,
                            MPI_UINT64_T,
                            migration.src_rank,
                            tag,
                            comm,
                            MPI_STATUS_IGNORE),
                        "size receive");
                    recv_size_entries.push_back(entry);
                    if (evidence)
                        ++evidence->incoming_entries;
                }
            }

            const auto metadata_exchange_end = Clock::now();
            if (evidence)
            {
                evidence->metadata_exchange_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        metadata_exchange_end - payload_prepare_end)
                        .count());
            }

            std::vector<MPI_Request> requests;
            struct RecvBufEntry
            {
                int expert_id = -1;
                int layer = -1;
                ExpertTransferBuffer gate;
                ExpertTransferBuffer up;
                ExpertTransferBuffer down;
            };
            std::vector<RecvBufEntry> recv_bufs;
            recv_bufs.reserve(recv_size_entries.size());

            for (const auto &entry : recv_size_entries)
            {
                RecvBufEntry buf;
                buf.expert_id = entry.expert_id;
                buf.layer = entry.layer;
                buf.gate.resize_uninitialized(entry.sizes[0]);
                buf.up.resize_uninitialized(entry.sizes[1]);
                buf.down.resize_uninitialized(entry.sizes[2]);

                const ExpertTransferBuffer *projection_buffers[3] = {
                    &buf.gate, &buf.up, &buf.down};
                for (int projection = 0; projection < 3; ++projection)
                {
                    if (entry.sizes[projection] == 0)
                        continue;
                    if (entry.sizes[projection] > static_cast<uint64_t>(INT_MAX))
                    {
                        throw std::length_error(
                            "ExpertWeightTransfer receive payload exceeds the MPI int limit");
                    }
                    MPI_Request request;
                    require_mpi_success(
                        MPI_Irecv(
                            const_cast<uint8_t *>(projection_buffers[projection]->data()),
                            static_cast<int>(entry.sizes[projection]),
                            MPI_BYTE,
                            entry.src_rank,
                            mpi_tags::weightTransferDataTag(
                                entry.layer, entry.expert_id, projection),
                            comm,
                            &request),
                        "payload receive");
                    requests.push_back(request);
                }
                recv_bufs.push_back(std::move(buf));
            }

            const auto receive_allocation_end = Clock::now();
            if (evidence)
            {
                evidence->receive_allocation_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        receive_allocation_end - metadata_exchange_end)
                        .count());
            }

            for (const auto &migration : manifest)
            {
                if (migration.src_rank != my_rank)
                    continue;
                const int index = find_send_index(
                    migration.layer_idx, migration.expert_id);
                if (index < 0)
                {
                    throw std::logic_error(
                        "ExpertWeightTransfer lost a source payload before data exchange");
                }
                const auto &blobs = send_blobs_store[static_cast<size_t>(index)];
                const ExpertTransferBuffer *projection_data[3] = {
                    &blobs.gate, &blobs.up, &blobs.down};
                for (int projection = 0; projection < 3; ++projection)
                {
                    if (projection_data[projection]->empty())
                        continue;
                    if (projection_data[projection]->size() > static_cast<size_t>(INT_MAX))
                    {
                        throw std::length_error(
                            "ExpertWeightTransfer send payload exceeds the MPI int limit");
                    }
                    MPI_Request request;
                    require_mpi_success(
                        MPI_Isend(
                            projection_data[projection]->data(),
                            static_cast<int>(projection_data[projection]->size()),
                            MPI_BYTE,
                            migration.dst_rank,
                            mpi_tags::weightTransferDataTag(
                                migration.layer_idx,
                                migration.expert_id,
                                projection),
                            comm,
                            &request),
                        "payload send");
                    requests.push_back(request);
                }
            }

            if (!requests.empty())
            {
                std::vector<MPI_Status> statuses(requests.size());
                require_mpi_success(
                    MPI_Waitall(
                        static_cast<int>(requests.size()),
                        requests.data(),
                        statuses.data()),
                    "payload waitall");
            }

            const auto payload_exchange_end = Clock::now();
            if (evidence)
            {
                evidence->payload_exchange_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        payload_exchange_end - receive_allocation_end)
                        .count());
            }

            ReceivedWeightsMap result;
            size_t total_bytes = 0;
            for (auto &buf : recv_bufs)
            {
                ExpertWeightBlobs blobs;
                blobs.gate = std::move(buf.gate);
                blobs.up = std::move(buf.up);
                blobs.down = std::move(buf.down);
                total_bytes += blobs.totalBytes();
                result[buf.layer][buf.expert_id] = std::move(blobs);
            }

            const auto t1 = Clock::now();
            if (evidence)
            {
                evidence->incoming_bytes = total_bytes;
                evidence->result_publication_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        t1 - payload_exchange_end)
                        .count());
                evidence->total_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                        .count());
            }
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            LOG_DEBUG("[ExpertWeightTransfer] Transferred " << manifest.size()
                                                             << " exact layer/expert payloads, "
                                                             << (total_bytes / (1024.0 * 1024.0)) << " MB received in "
                                                             << ms << " ms");

            return result;
        }
    };

} // namespace llaminar2
