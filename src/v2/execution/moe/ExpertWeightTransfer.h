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

#include "../../kernels/IPackedWeights.h"
#include "../../kernels/PackedWeightsSerialization.h"
#include "../../utils/Logger.h"
#include "../../utils/MPITags.h"

#include <mpi.h>
#include <chrono>
#include <climits>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace llaminar2 {

/// A single expert migration: expert moves from src_rank to dst_rank.
struct ExpertMigration {
    int expert_id;
    int src_rank;
    int dst_rank;
};

/// Serialized weight blobs for one expert's 3 projections.
struct ExpertWeightBlobs {
    std::vector<uint8_t> gate;
    std::vector<uint8_t> up;
    std::vector<uint8_t> down;

    bool empty() const { return gate.empty() && up.empty() && down.empty(); }
    size_t totalBytes() const { return gate.size() + up.size() + down.size(); }
};

/// Key for received weights: [layer_idx][expert_id] → blobs
using ReceivedWeightsMap = std::unordered_map<int, std::unordered_map<int, ExpertWeightBlobs>>;

/**
 * @brief Coordinates MPI transfer of pre-packed GEMM weights between ranks.
 *
 * All methods are static. Pure-logic helpers (buildManifest, departingExperts,
 * arrivingExperts) require no MPI. transferAllLayers() performs the actual
 * cross-rank communication and returns an empty map on failure so the caller
 * can fall back to repacking.
 */
class ExpertWeightTransfer {
public:
    // ── Manifest Building (pure logic, no MPI) ──────────────────────

    /**
     * @brief Build migration manifest from old→new placement.
     * @return Entries for experts that changed rank.
     */
    static inline std::vector<ExpertMigration> buildManifest(
        const std::vector<int>& old_placement,
        const std::vector<int>& new_placement)
    {
        std::vector<ExpertMigration> manifest;
        const size_t n = std::min(old_placement.size(), new_placement.size());
        for (size_t i = 0; i < n; ++i)
        {
            if (old_placement[i] != new_placement[i])
            {
                manifest.push_back({static_cast<int>(i), old_placement[i], new_placement[i]});
            }
        }
        return manifest;
    }

    /**
     * @brief Get expert IDs this rank is sending (departing experts).
     */
    static inline std::vector<int> departingExperts(
        const std::vector<ExpertMigration>& manifest,
        int my_rank)
    {
        std::vector<int> result;
        for (const auto& m : manifest)
        {
            if (m.src_rank == my_rank)
                result.push_back(m.expert_id);
        }
        return result;
    }

    /**
     * @brief Get expert IDs this rank is receiving (arriving experts).
     */
    static inline std::vector<int> arrivingExperts(
        const std::vector<ExpertMigration>& manifest,
        int my_rank)
    {
        std::vector<int> result;
        for (const auto& m : manifest)
        {
            if (m.dst_rank == my_rank)
                result.push_back(m.expert_id);
        }
        return result;
    }

    // ── MPI Transfer ────────────────────────────────────────────────

    /**
     * @brief Execute cross-rank weight transfer for all layers.
     *
     * Two-phase protocol:
     *   Phase A: Blocking size exchange (24 bytes per migration × layer).
     *   Phase B: Non-blocking bulk data transfer with MPI_Waitall.
     *
     * @param manifest     Migration entries (from buildManifest).
     * @param num_layers   Number of MoE layers.
     * @param get_blobs    Callback: (layer_idx, expert_id) → serialized blobs.
     *                     Called only for experts this rank is SENDING.
     * @param my_rank      This rank's ID.
     * @param comm         MPI communicator.
     * @return Map of received weights: [layer_idx][expert_id] → blobs.
     *         Empty map on failure (caller should fall back to repack).
     */
    static inline ReceivedWeightsMap transferAllLayers(
        const std::vector<ExpertMigration>& manifest,
        int num_layers,
        std::function<ExpertWeightBlobs(int layer_idx, int expert_id)> get_blobs,
        int my_rank,
        MPI_Comm comm)
    {
        if (manifest.empty() || num_layers <= 0)
            return {};

        auto t0 = std::chrono::steady_clock::now();

        try
        {
            // ── Phase A: Blocking size exchange ─────────────────────
            // For each (migration, layer), sender sends uint64_t[3] sizes,
            // receiver receives them. This is fast: 24 bytes per message.

            // Collect all send blobs upfront (sender side) so we can extract sizes.
            // Key: (expert_id, layer) → blobs
            struct BlobKey {
                int expert_id;
                int layer;
            };
            std::vector<BlobKey> send_keys;
            std::vector<ExpertWeightBlobs> send_blobs_store;

            for (const auto& m : manifest)
            {
                if (m.src_rank == my_rank)
                {
                    for (int layer = 0; layer < num_layers; ++layer)
                    {
                        auto blobs = get_blobs(layer, m.expert_id);
                        send_keys.push_back({m.expert_id, layer});
                        send_blobs_store.push_back(std::move(blobs));
                    }
                }
            }

            // Build lookup for send blobs: (expert_id, layer) → index
            auto findSendIdx = [&](int expert_id, int layer) -> int {
                for (size_t i = 0; i < send_keys.size(); ++i)
                {
                    if (send_keys[i].expert_id == expert_id && send_keys[i].layer == layer)
                        return static_cast<int>(i);
                }
                return -1;
            };

            // Size exchange structures for receivers
            struct RecvSizeEntry {
                int expert_id;
                int layer;
                int src_rank;
                uint64_t sizes[3]; // gate, up, down
            };
            std::vector<RecvSizeEntry> recv_size_entries;

            // Phase A: Exchange sizes — ordered by manifest then layer
            for (const auto& m : manifest)
            {
                for (int layer = 0; layer < num_layers; ++layer)
                {
                    int tag = mpi_tags::weightTransferSizeTag(layer, m.expert_id);

                    if (m.src_rank == my_rank)
                    {
                        int idx = findSendIdx(m.expert_id, layer);
                        if (idx < 0)
                        {
                            LOG_ERROR("[ExpertWeightTransfer] Missing send blob for expert "
                                      << m.expert_id << " layer " << layer);
                            return {};
                        }
                        uint64_t sizes[3] = {
                            send_blobs_store[idx].gate.size(),
                            send_blobs_store[idx].up.size(),
                            send_blobs_store[idx].down.size()
                        };
                        int rc = MPI_Send(sizes, 3, MPI_UINT64_T, m.dst_rank, tag, comm);
                        if (rc != MPI_SUCCESS)
                        {
                            LOG_ERROR("[ExpertWeightTransfer] MPI_Send sizes failed: rc=" << rc);
                            return {};
                        }
                    }
                    else if (m.dst_rank == my_rank)
                    {
                        RecvSizeEntry entry;
                        entry.expert_id = m.expert_id;
                        entry.layer = layer;
                        entry.src_rank = m.src_rank;
                        int rc = MPI_Recv(entry.sizes, 3, MPI_UINT64_T,
                                          m.src_rank, tag, comm, MPI_STATUS_IGNORE);
                        if (rc != MPI_SUCCESS)
                        {
                            LOG_ERROR("[ExpertWeightTransfer] MPI_Recv sizes failed: rc=" << rc);
                            return {};
                        }
                        recv_size_entries.push_back(entry);
                    }
                }
            }

            // ── Phase B: Non-blocking data exchange ─────────────────
            std::vector<MPI_Request> requests;

            // Allocate receive buffers
            struct RecvBufEntry {
                int expert_id;
                int layer;
                std::vector<uint8_t> gate;
                std::vector<uint8_t> up;
                std::vector<uint8_t> down;
            };
            std::vector<RecvBufEntry> recv_bufs;
            recv_bufs.reserve(recv_size_entries.size());

            // Post receives
            for (const auto& entry : recv_size_entries)
            {
                RecvBufEntry buf;
                buf.expert_id = entry.expert_id;
                buf.layer = entry.layer;
                buf.gate.resize(entry.sizes[0]);
                buf.up.resize(entry.sizes[1]);
                buf.down.resize(entry.sizes[2]);

                const std::vector<uint8_t>* proj_bufs[3] = {&buf.gate, &buf.up, &buf.down};
                for (int proj = 0; proj < 3; ++proj)
                {
                    if (entry.sizes[proj] == 0) continue;
                    if (entry.sizes[proj] > static_cast<uint64_t>(INT_MAX))
                    {
                        LOG_ERROR("[ExpertWeightTransfer] Expert " << entry.expert_id
                                  << " layer " << entry.layer << " proj " << proj
                                  << " exceeds MPI int limit: " << entry.sizes[proj] << " bytes");
                        return {};
                    }
                    int tag = mpi_tags::weightTransferDataTag(entry.layer, entry.expert_id, proj);
                    MPI_Request req;
                    int rc = MPI_Irecv(
                        const_cast<uint8_t*>(proj_bufs[proj]->data()),
                        static_cast<int>(entry.sizes[proj]),
                        MPI_BYTE, entry.src_rank, tag, comm, &req);
                    if (rc != MPI_SUCCESS)
                    {
                        LOG_ERROR("[ExpertWeightTransfer] MPI_Irecv failed: rc=" << rc);
                        return {};
                    }
                    requests.push_back(req);
                }
                recv_bufs.push_back(std::move(buf));
            }

            // Post sends
            for (const auto& m : manifest)
            {
                if (m.src_rank != my_rank) continue;
                for (int layer = 0; layer < num_layers; ++layer)
                {
                    int idx = findSendIdx(m.expert_id, layer);
                    if (idx < 0) continue;
                    const auto& blobs = send_blobs_store[idx];
                    const std::vector<uint8_t>* proj_data[3] = {&blobs.gate, &blobs.up, &blobs.down};
                    for (int proj = 0; proj < 3; ++proj)
                    {
                        if (proj_data[proj]->empty()) continue;
                        if (proj_data[proj]->size() > static_cast<size_t>(INT_MAX))
                        {
                            LOG_ERROR("[ExpertWeightTransfer] Send blob for expert " << m.expert_id
                                      << " layer " << layer << " proj " << proj
                                      << " exceeds MPI int limit: " << proj_data[proj]->size() << " bytes");
                            return {};
                        }
                        int tag = mpi_tags::weightTransferDataTag(layer, m.expert_id, proj);
                        MPI_Request req;
                        int rc = MPI_Isend(
                            proj_data[proj]->data(),
                            static_cast<int>(proj_data[proj]->size()),
                            MPI_BYTE, m.dst_rank, tag, comm, &req);
                        if (rc != MPI_SUCCESS)
                        {
                            LOG_ERROR("[ExpertWeightTransfer] MPI_Isend failed: rc=" << rc);
                            return {};
                        }
                        requests.push_back(req);
                    }
                }
            }

            // Wait for all non-blocking operations
            if (!requests.empty())
            {
                std::vector<MPI_Status> statuses(requests.size());
                int rc = MPI_Waitall(
                    static_cast<int>(requests.size()),
                    requests.data(), statuses.data());
                if (rc != MPI_SUCCESS)
                {
                    LOG_ERROR("[ExpertWeightTransfer] MPI_Waitall failed: rc=" << rc);
                    return {};
                }
            }

            // ── Build result map ────────────────────────────────────
            ReceivedWeightsMap result;
            size_t total_bytes = 0;
            for (auto& buf : recv_bufs)
            {
                ExpertWeightBlobs blobs;
                blobs.gate = std::move(buf.gate);
                blobs.up   = std::move(buf.up);
                blobs.down = std::move(buf.down);
                total_bytes += blobs.totalBytes();
                result[buf.layer][buf.expert_id] = std::move(blobs);
            }

            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            LOG_INFO("[ExpertWeightTransfer] Transferred " << manifest.size()
                     << " experts × " << num_layers << " layers, "
                     << (total_bytes / (1024.0 * 1024.0)) << " MB received in "
                     << ms << " ms");

            return result;
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("[ExpertWeightTransfer] Exception during transfer: " << e.what());
            return {};
        }
        catch (...)
        {
            LOG_ERROR("[ExpertWeightTransfer] Unknown exception during transfer");
            return {};
        }
    }
};

} // namespace llaminar2
