/**
 * @file MoEOverlaySparseCollective.cpp
 * @brief Compact sparse payload transport for graph-native MoE overlay collectives.
 */

#include "execution/moe/MoEOverlaySparseCollective.h"

#include "interfaces/IMPIContext.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace llaminar2
{
    namespace
    {
        constexpr uint32_t kPacketMagic = 0x32454f4dU; // "MOE2"
        constexpr uint32_t kPacketVersion = 2;
        constexpr uint8_t kPacketKindDispatch = 1;
        constexpr uint8_t kPacketKindReturn = 2;

        template <typename T>
        void appendPod(std::vector<uint8_t> &buffer, const T &value)
        {
            const auto *ptr = reinterpret_cast<const uint8_t *>(&value);
            buffer.insert(buffer.end(), ptr, ptr + sizeof(T));
        }

        template <typename T>
        bool readPod(const uint8_t *data,
                     size_t size,
                     size_t *offset,
                     T *out)
        {
            if (!data || !offset || !out || *offset + sizeof(T) > size)
                return false;
            std::memcpy(out, data + *offset, sizeof(T));
            *offset += sizeof(T);
            return true;
        }

        std::string keyToStableString(const MoEOverlayCollectiveKey &key)
        {
            std::ostringstream ss;
            ss << static_cast<int>(key.key_namespace) << ':'
               << key.generation_id << ':'
               << key.step_id << ':'
               << key.mtp_depth << ':'
               << key.layer_idx << ':'
               << key.tier_idx << ':'
               << key.domain_id << ':'
               << key.participant_id << ':'
               << static_cast<int>(key.direction) << ':'
               << key.sequence;
            return ss.str();
        }

        uint64_t makeCollectiveSequence(MoEOverlayCollectiveNamespace key_namespace,
                                        int mtp_depth,
                                        int layer_idx,
                                        int tier_idx,
                                        int participant_id,
                                        MoEOverlayCollectiveDirection direction)
        {
            const uint64_t namespace_offset =
                key_namespace == MoEOverlayCollectiveNamespace::MTP ? (1ull << 56) : 0ull;
            const uint64_t depth_component =
                key_namespace == MoEOverlayCollectiveNamespace::MTP
                    ? static_cast<uint64_t>(std::max(mtp_depth, 0)) * (1ull << 40)
                    : 0ull;
            const uint64_t direction_offset =
                direction == MoEOverlayCollectiveDirection::Dispatch ? 0ull : 2048ull;
            return namespace_offset +
                   depth_component +
                   static_cast<uint64_t>(std::max(layer_idx, 0)) * 8192ull +
                   direction_offset +
                   static_cast<uint64_t>(std::max(tier_idx, 0)) * 128ull +
                   static_cast<uint64_t>(std::max(participant_id, 0));
        }

        bool validateSparseRows(const MoEOverlaySparseRows &rows,
                                std::string *error)
        {
            if (rows.d_model <= 0 || rows.top_k <= 0)
            {
                if (error)
                    *error = "invalid sparse payload dimensions";
                return false;
            }
            if (!rows.row_ids_host || !rows.entry_offsets_host ||
                !rows.expert_ids_host || !rows.route_weights_host || !rows.hidden_rows_fp32)
            {
                if (error)
                    *error = "sparse payload is missing host buffers";
                return false;
            }
            if (rows.live_row_count > rows.row_capacity || rows.live_entry_count > rows.entry_capacity)
            {
                if (error)
                    *error = "sparse payload live counts exceed capacity";
                return false;
            }
            return true;
        }

        bool validateReturnRows(const MoEOverlayReturnRows &rows,
                                std::string *error)
        {
            if (rows.d_model <= 0)
            {
                if (error)
                    *error = "invalid return payload dimensions";
                return false;
            }
            if (!rows.row_ids_host || !rows.output_rows_fp32)
            {
                if (error)
                    *error = "return payload is missing host buffers";
                return false;
            }
            if (rows.live_row_count > rows.row_capacity)
            {
                if (error)
                    *error = "return payload live row count exceeds capacity";
                return false;
            }
            return true;
        }

        bool ensureSparseInboundCapacity(MoEOverlaySparseRows *inbound,
                                         size_t required_rows,
                                         size_t required_entries,
                                         std::string *error)
        {
            if (!inbound)
            {
                if (error)
                    *error = "null inbound sparse payload";
                return false;
            }

            if (required_rows > inbound->row_capacity || required_entries > inbound->entry_capacity)
            {
                if (error)
                    *error = "inbound sparse payload capacity exceeded";
                return false;
            }
            return true;
        }

        bool ensureReturnInboundCapacity(MoEOverlayReturnRows *inbound,
                                         size_t required_rows,
                                         std::string *error)
        {
            if (!inbound)
            {
                if (error)
                    *error = "null inbound return payload";
                return false;
            }
            if (required_rows > inbound->row_capacity)
            {
                if (error)
                    *error = "inbound return payload capacity exceeded";
                return false;
            }
            return true;
        }

        struct DispatchPayloadCopy
        {
            MoEOverlayCollectiveKey key;
            int32_t source_participant = -1;
            int32_t target_participant = -1;
            int32_t d_model = 0;
            int32_t top_k = 0;
            std::vector<int32_t> row_ids;
            std::vector<int32_t> entry_offsets;
            std::vector<int32_t> expert_ids;
            std::vector<float> route_weights;
            std::vector<float> hidden_rows;
        };

        struct ReturnPayloadCopy
        {
            MoEOverlayCollectiveKey key;
            int32_t source_participant = -1;
            int32_t target_participant = -1;
            int32_t d_model = 0;
            std::vector<int32_t> row_ids;
            std::vector<float> output_rows;
        };

        DispatchPayloadCopy copyDispatchPayload(const MoEOverlaySparseRows &rows)
        {
            DispatchPayloadCopy copy;
            copy.key = rows.key;
            copy.source_participant = rows.source_participant;
            copy.target_participant = rows.target_participant;
            copy.d_model = rows.d_model;
            copy.top_k = rows.top_k;
            copy.row_ids.assign(rows.row_ids_host, rows.row_ids_host + rows.live_row_count);
            copy.entry_offsets.assign(rows.entry_offsets_host,
                                      rows.entry_offsets_host + rows.live_row_count + 1);
            copy.expert_ids.assign(rows.expert_ids_host,
                                   rows.expert_ids_host + rows.live_entry_count);
            copy.route_weights.assign(rows.route_weights_host,
                                      rows.route_weights_host + rows.live_entry_count);
            copy.hidden_rows.assign(rows.hidden_rows_fp32,
                                    rows.hidden_rows_fp32 + rows.live_row_count * static_cast<size_t>(rows.d_model));
            return copy;
        }

        ReturnPayloadCopy copyReturnPayload(const MoEOverlayReturnRows &rows)
        {
            ReturnPayloadCopy copy;
            copy.key = rows.key;
            copy.source_participant = rows.source_participant;
            copy.target_participant = rows.target_participant;
            copy.d_model = rows.d_model;
            copy.row_ids.assign(rows.row_ids_host, rows.row_ids_host + rows.live_row_count);
            copy.output_rows.assign(rows.output_rows_fp32,
                                    rows.output_rows_fp32 + rows.live_row_count * static_cast<size_t>(rows.d_model));
            return copy;
        }

        bool appendDispatchInbound(const DispatchPayloadCopy &payload,
                                   MoEOverlaySparseRows *inbound,
                                   std::string *error)
        {
            const size_t rows = payload.row_ids.size();
            const size_t entries = payload.expert_ids.size();
            const size_t total_rows = inbound->live_row_count + rows;
            const size_t total_entries = inbound->live_entry_count + entries;
            if (!ensureSparseInboundCapacity(inbound, total_rows, total_entries, error))
                return false;

            if (inbound->d_model != payload.d_model || inbound->top_k != payload.top_k)
            {
                if (error)
                    *error = "inbound sparse payload dimension mismatch";
                return false;
            }

            std::copy(payload.row_ids.begin(),
                      payload.row_ids.end(),
                      inbound->row_ids_host + inbound->live_row_count);
            std::copy(payload.expert_ids.begin(),
                      payload.expert_ids.end(),
                      inbound->expert_ids_host + inbound->live_entry_count);
            std::copy(payload.route_weights.begin(),
                      payload.route_weights.end(),
                      inbound->route_weights_host + inbound->live_entry_count);

            const size_t d_model = static_cast<size_t>(inbound->d_model);
            std::copy(payload.hidden_rows.begin(),
                      payload.hidden_rows.end(),
                      inbound->hidden_rows_fp32 + inbound->live_row_count * d_model);

            if (rows == 0)
            {
                if (inbound->live_row_count == 0)
                    inbound->entry_offsets_host[0] = 0;
                return true;
            }

            if (inbound->live_row_count == 0)
                inbound->entry_offsets_host[0] = 0;

            for (size_t row = 0; row < rows; ++row)
            {
                inbound->entry_offsets_host[inbound->live_row_count + row] =
                    static_cast<int32_t>(inbound->live_entry_count + static_cast<size_t>(payload.entry_offsets[row]));
            }
            inbound->entry_offsets_host[inbound->live_row_count + rows] =
                static_cast<int32_t>(inbound->live_entry_count + static_cast<size_t>(payload.entry_offsets[rows]));

            inbound->live_row_count = total_rows;
            inbound->live_entry_count = total_entries;
            return true;
        }

        bool appendReturnInbound(const ReturnPayloadCopy &payload,
                                 MoEOverlayReturnRows *inbound,
                                 std::string *error)
        {
            const size_t rows = payload.row_ids.size();
            const size_t total_rows = inbound->live_row_count + rows;
            if (!ensureReturnInboundCapacity(inbound, total_rows, error))
                return false;

            if (inbound->d_model != payload.d_model)
            {
                if (error)
                    *error = "inbound return payload dimension mismatch";
                return false;
            }

            const size_t d_model = static_cast<size_t>(inbound->d_model);
            std::copy(payload.row_ids.begin(),
                      payload.row_ids.end(),
                      inbound->row_ids_host + inbound->live_row_count);
            std::copy(payload.output_rows.begin(),
                      payload.output_rows.end(),
                      inbound->output_rows_fp32 + inbound->live_row_count * d_model);

            inbound->live_row_count = total_rows;
            return true;
        }

        std::vector<uint8_t> serializeDispatchPacket(const DispatchPayloadCopy &payload)
        {
            std::vector<uint8_t> bytes;
            bytes.reserve(128 +
                          payload.row_ids.size() * sizeof(int32_t) +
                          payload.entry_offsets.size() * sizeof(int32_t) +
                          payload.expert_ids.size() * sizeof(int32_t) +
                          payload.route_weights.size() * sizeof(float) +
                          payload.hidden_rows.size() * sizeof(float));

            appendPod(bytes, kPacketMagic);
            appendPod(bytes, kPacketVersion);
            appendPod(bytes, kPacketKindDispatch);
            appendPod(bytes, static_cast<uint8_t>(payload.key.key_namespace));
            appendPod(bytes, payload.key.generation_id);
            appendPod(bytes, payload.key.step_id);
            appendPod(bytes, payload.key.mtp_depth);
            appendPod(bytes, payload.key.layer_idx);
            appendPod(bytes, payload.key.tier_idx);
            appendPod(bytes, payload.key.domain_id);
            appendPod(bytes, payload.key.participant_id);
            appendPod(bytes, static_cast<uint8_t>(payload.key.direction));
            appendPod(bytes, payload.key.sequence);
            appendPod(bytes, payload.source_participant);
            appendPod(bytes, payload.target_participant);
            appendPod(bytes, payload.d_model);
            appendPod(bytes, payload.top_k);

            const uint32_t row_count = static_cast<uint32_t>(payload.row_ids.size());
            const uint32_t entry_count = static_cast<uint32_t>(payload.expert_ids.size());
            appendPod(bytes, row_count);
            appendPod(bytes, entry_count);

            if (!payload.row_ids.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.row_ids.data()),
                             reinterpret_cast<const uint8_t *>(payload.row_ids.data() + payload.row_ids.size()));
            if (!payload.entry_offsets.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.entry_offsets.data()),
                             reinterpret_cast<const uint8_t *>(payload.entry_offsets.data() + payload.entry_offsets.size()));
            if (!payload.expert_ids.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.expert_ids.data()),
                             reinterpret_cast<const uint8_t *>(payload.expert_ids.data() + payload.expert_ids.size()));
            if (!payload.route_weights.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.route_weights.data()),
                             reinterpret_cast<const uint8_t *>(payload.route_weights.data() + payload.route_weights.size()));
            if (!payload.hidden_rows.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.hidden_rows.data()),
                             reinterpret_cast<const uint8_t *>(payload.hidden_rows.data() + payload.hidden_rows.size()));
            return bytes;
        }

        bool deserializeDispatchPacket(const uint8_t *data,
                                       size_t size,
                                       DispatchPayloadCopy *out,
                                       std::string *error)
        {
            if (!data || !out)
            {
                if (error)
                    *error = "invalid dispatch packet buffer";
                return false;
            }

            size_t offset = 0;
            uint32_t magic = 0;
            uint32_t version = 0;
            uint8_t kind = 0;
            if (!readPod(data, size, &offset, &magic) ||
                !readPod(data, size, &offset, &version) ||
                !readPod(data, size, &offset, &kind))
            {
                if (error)
                    *error = "dispatch packet header is truncated";
                return false;
            }
            if (magic != kPacketMagic || version != kPacketVersion || kind != kPacketKindDispatch)
            {
                if (error)
                    *error = "dispatch packet header is invalid";
                return false;
            }

            uint8_t key_namespace = 0;
            uint8_t direction = 0;
            uint32_t row_count = 0;
            uint32_t entry_count = 0;
            if (!readPod(data, size, &offset, &key_namespace) ||
                !readPod(data, size, &offset, &out->key.generation_id) ||
                !readPod(data, size, &offset, &out->key.step_id) ||
                !readPod(data, size, &offset, &out->key.mtp_depth) ||
                !readPod(data, size, &offset, &out->key.layer_idx) ||
                !readPod(data, size, &offset, &out->key.tier_idx) ||
                !readPod(data, size, &offset, &out->key.domain_id) ||
                !readPod(data, size, &offset, &out->key.participant_id) ||
                !readPod(data, size, &offset, &direction) ||
                !readPod(data, size, &offset, &out->key.sequence) ||
                !readPod(data, size, &offset, &out->source_participant) ||
                !readPod(data, size, &offset, &out->target_participant) ||
                !readPod(data, size, &offset, &out->d_model) ||
                !readPod(data, size, &offset, &out->top_k) ||
                !readPod(data, size, &offset, &row_count) ||
                !readPod(data, size, &offset, &entry_count))
            {
                if (error)
                    *error = "dispatch packet metadata is truncated";
                return false;
            }

            out->key.key_namespace = static_cast<MoEOverlayCollectiveNamespace>(key_namespace);
            out->key.direction = static_cast<MoEOverlayCollectiveDirection>(direction);
            if (out->key.direction != MoEOverlayCollectiveDirection::Dispatch)
            {
                if (error)
                    *error = "dispatch packet has wrong collective direction";
                return false;
            }
            if (!out->key.isValid())
            {
                if (error)
                    *error = "dispatch packet has invalid collective key";
                return false;
            }

            out->row_ids.resize(row_count);
            out->entry_offsets.resize(static_cast<size_t>(row_count) + 1u);
            out->expert_ids.resize(entry_count);
            out->route_weights.resize(entry_count);
            out->hidden_rows.resize(static_cast<size_t>(row_count) * static_cast<size_t>(out->d_model));

            const auto copyInto = [&](void *dst, size_t bytes, const char *name) -> bool
            {
                if (offset + bytes > size)
                {
                    if (error)
                    {
                        std::ostringstream ss;
                        ss << "dispatch packet is truncated while reading " << name;
                        *error = ss.str();
                    }
                    return false;
                }
                std::memcpy(dst, data + offset, bytes);
                offset += bytes;
                return true;
            };

            if (!copyInto(out->row_ids.data(), out->row_ids.size() * sizeof(int32_t), "row ids") ||
                !copyInto(out->entry_offsets.data(), out->entry_offsets.size() * sizeof(int32_t), "entry offsets") ||
                !copyInto(out->expert_ids.data(), out->expert_ids.size() * sizeof(int32_t), "expert ids") ||
                !copyInto(out->route_weights.data(), out->route_weights.size() * sizeof(float), "route weights") ||
                !copyInto(out->hidden_rows.data(), out->hidden_rows.size() * sizeof(float), "hidden rows"))
            {
                return false;
            }

            return true;
        }

        std::vector<uint8_t> serializeReturnPacket(const ReturnPayloadCopy &payload)
        {
            std::vector<uint8_t> bytes;
            bytes.reserve(96 +
                          payload.row_ids.size() * sizeof(int32_t) +
                          payload.output_rows.size() * sizeof(float));

            appendPod(bytes, kPacketMagic);
            appendPod(bytes, kPacketVersion);
            appendPod(bytes, kPacketKindReturn);
            appendPod(bytes, static_cast<uint8_t>(payload.key.key_namespace));
            appendPod(bytes, payload.key.generation_id);
            appendPod(bytes, payload.key.step_id);
            appendPod(bytes, payload.key.mtp_depth);
            appendPod(bytes, payload.key.layer_idx);
            appendPod(bytes, payload.key.tier_idx);
            appendPod(bytes, payload.key.domain_id);
            appendPod(bytes, payload.key.participant_id);
            appendPod(bytes, static_cast<uint8_t>(payload.key.direction));
            appendPod(bytes, payload.key.sequence);
            appendPod(bytes, payload.source_participant);
            appendPod(bytes, payload.target_participant);
            appendPod(bytes, payload.d_model);

            const uint32_t row_count = static_cast<uint32_t>(payload.row_ids.size());
            appendPod(bytes, row_count);

            if (!payload.row_ids.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.row_ids.data()),
                             reinterpret_cast<const uint8_t *>(payload.row_ids.data() + payload.row_ids.size()));
            if (!payload.output_rows.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.output_rows.data()),
                             reinterpret_cast<const uint8_t *>(payload.output_rows.data() + payload.output_rows.size()));
            return bytes;
        }

        bool deserializeReturnPacket(const uint8_t *data,
                                     size_t size,
                                     ReturnPayloadCopy *out,
                                     std::string *error)
        {
            if (!data || !out)
            {
                if (error)
                    *error = "invalid return packet buffer";
                return false;
            }

            size_t offset = 0;
            uint32_t magic = 0;
            uint32_t version = 0;
            uint8_t kind = 0;
            if (!readPod(data, size, &offset, &magic) ||
                !readPod(data, size, &offset, &version) ||
                !readPod(data, size, &offset, &kind))
            {
                if (error)
                    *error = "return packet header is truncated";
                return false;
            }
            if (magic != kPacketMagic || version != kPacketVersion || kind != kPacketKindReturn)
            {
                if (error)
                    *error = "return packet header is invalid";
                return false;
            }

            uint8_t key_namespace = 0;
            uint8_t direction = 0;
            uint32_t row_count = 0;
            if (!readPod(data, size, &offset, &key_namespace) ||
                !readPod(data, size, &offset, &out->key.generation_id) ||
                !readPod(data, size, &offset, &out->key.step_id) ||
                !readPod(data, size, &offset, &out->key.mtp_depth) ||
                !readPod(data, size, &offset, &out->key.layer_idx) ||
                !readPod(data, size, &offset, &out->key.tier_idx) ||
                !readPod(data, size, &offset, &out->key.domain_id) ||
                !readPod(data, size, &offset, &out->key.participant_id) ||
                !readPod(data, size, &offset, &direction) ||
                !readPod(data, size, &offset, &out->key.sequence) ||
                !readPod(data, size, &offset, &out->source_participant) ||
                !readPod(data, size, &offset, &out->target_participant) ||
                !readPod(data, size, &offset, &out->d_model) ||
                !readPod(data, size, &offset, &row_count))
            {
                if (error)
                    *error = "return packet metadata is truncated";
                return false;
            }

            out->key.key_namespace = static_cast<MoEOverlayCollectiveNamespace>(key_namespace);
            out->key.direction = static_cast<MoEOverlayCollectiveDirection>(direction);
            if (out->key.direction != MoEOverlayCollectiveDirection::ReturnReduce)
            {
                if (error)
                    *error = "return packet has wrong collective direction";
                return false;
            }
            if (!out->key.isValid())
            {
                if (error)
                    *error = "return packet has invalid collective key";
                return false;
            }

            out->row_ids.resize(row_count);
            out->output_rows.resize(static_cast<size_t>(row_count) * static_cast<size_t>(out->d_model));

            const auto copyInto = [&](void *dst, size_t bytes, const char *name) -> bool
            {
                if (offset + bytes > size)
                {
                    if (error)
                    {
                        std::ostringstream ss;
                        ss << "return packet is truncated while reading " << name;
                        *error = ss.str();
                    }
                    return false;
                }
                std::memcpy(dst, data + offset, bytes);
                offset += bytes;
                return true;
            };

            if (!copyInto(out->row_ids.data(), out->row_ids.size() * sizeof(int32_t), "row ids") ||
                !copyInto(out->output_rows.data(), out->output_rows.size() * sizeof(float), "output rows"))
            {
                return false;
            }

            return true;
        }

        std::vector<uint8_t> gatherAllPayloads(const IMPIContext &mpi_ctx,
                                               const std::vector<uint8_t> &send_payload,
                                               std::vector<int> *recv_counts,
                                               std::vector<int> *displs)
        {
            const int world_size = mpi_ctx.world_size();
            std::vector<int32_t> counts32(static_cast<size_t>(world_size), 0);
            int32_t send_count = static_cast<int32_t>(send_payload.size());
            mpi_ctx.allgather_bytes(&send_count, counts32.data(), sizeof(int32_t));

            recv_counts->assign(static_cast<size_t>(world_size), 0);
            displs->assign(static_cast<size_t>(world_size), 0);
            int total = 0;
            for (int rank = 0; rank < world_size; ++rank)
            {
                (*recv_counts)[static_cast<size_t>(rank)] = counts32[static_cast<size_t>(rank)];
                (*displs)[static_cast<size_t>(rank)] = total;
                total += counts32[static_cast<size_t>(rank)];
            }

            std::vector<uint8_t> gathered(static_cast<size_t>(std::max(0, total)), 0);
            mpi_ctx.allgatherv_bytes(send_payload.empty() ? nullptr : send_payload.data(),
                                     send_count,
                                     gathered.empty() ? nullptr : gathered.data(),
                                     recv_counts->data(),
                                     displs->data());
            return gathered;
        }

    } // namespace

    const char *toString(MoEOverlayCollectiveDirection direction)
    {
        switch (direction)
        {
        case MoEOverlayCollectiveDirection::Dispatch:
            return "Dispatch";
        case MoEOverlayCollectiveDirection::ReturnReduce:
            return "ReturnReduce";
        }
        return "Unknown";
    }

    const char *toString(MoEOverlayCollectiveNamespace key_namespace)
    {
        switch (key_namespace)
        {
        case MoEOverlayCollectiveNamespace::Main:
            return "Main";
        case MoEOverlayCollectiveNamespace::MTP:
            return "MTP";
        }
        return "Unknown";
    }

    bool MoEOverlayCollectiveKey::isValid() const
    {
        if (layer_idx < 0 || tier_idx < 0 || domain_id < 0)
            return false;

        switch (key_namespace)
        {
        case MoEOverlayCollectiveNamespace::Main:
            return mtp_depth < 0;
        case MoEOverlayCollectiveNamespace::MTP:
            return mtp_depth >= 0;
        }
        return false;
    }

    std::string MoEOverlayCollectiveKey::toString() const
    {
        return keyToStableString(*this);
    }

    bool operator==(const MoEOverlayCollectiveKey &lhs, const MoEOverlayCollectiveKey &rhs)
    {
        return lhs.generation_id == rhs.generation_id &&
               lhs.step_id == rhs.step_id &&
               lhs.key_namespace == rhs.key_namespace &&
               lhs.mtp_depth == rhs.mtp_depth &&
               lhs.layer_idx == rhs.layer_idx &&
               lhs.tier_idx == rhs.tier_idx &&
               lhs.domain_id == rhs.domain_id &&
               lhs.participant_id == rhs.participant_id &&
               lhs.direction == rhs.direction &&
               lhs.sequence == rhs.sequence;
    }

    bool operator!=(const MoEOverlayCollectiveKey &lhs, const MoEOverlayCollectiveKey &rhs)
    {
        return !(lhs == rhs);
    }

    bool operator<(const MoEOverlayCollectiveKey &lhs, const MoEOverlayCollectiveKey &rhs)
    {
        return std::tie(lhs.generation_id,
                        lhs.step_id,
                        lhs.key_namespace,
                        lhs.mtp_depth,
                        lhs.layer_idx,
                        lhs.tier_idx,
                        lhs.domain_id,
                        lhs.participant_id,
                        lhs.direction,
                        lhs.sequence) <
               std::tie(rhs.generation_id,
                        rhs.step_id,
                        rhs.key_namespace,
                        rhs.mtp_depth,
                        rhs.layer_idx,
                        rhs.tier_idx,
                        rhs.domain_id,
                        rhs.participant_id,
                        rhs.direction,
                        rhs.sequence);
    }

    MoEOverlayCollectiveKey makeMoEOverlayCollectiveKey(
        uint64_t generation_id,
        uint64_t step_id,
        int layer_idx,
        int tier_idx,
        int domain_id,
        int participant_id,
        MoEOverlayCollectiveDirection direction)
    {
        MoEOverlayCollectiveKey key;
        key.generation_id = generation_id;
        key.step_id = step_id;
        key.key_namespace = MoEOverlayCollectiveNamespace::Main;
        key.mtp_depth = -1;
        key.layer_idx = layer_idx;
        key.tier_idx = tier_idx;
        key.domain_id = domain_id;
        key.participant_id = participant_id;
        key.direction = direction;
        key.sequence = makeCollectiveSequence(key.key_namespace,
                                              key.mtp_depth,
                                              key.layer_idx,
                                              key.tier_idx,
                                              key.participant_id,
                                              key.direction);
        return key;
    }

    MoEOverlayCollectiveKey makeMTPMoEOverlayCollectiveKey(
        uint64_t generation_id,
        uint64_t decode_step_id,
        int mtp_depth,
        int layer_idx,
        int tier_idx,
        int domain_id,
        int participant_id,
        MoEOverlayCollectiveDirection direction)
    {
        MoEOverlayCollectiveKey key;
        key.generation_id = generation_id;
        key.step_id = decode_step_id;
        key.key_namespace = MoEOverlayCollectiveNamespace::MTP;
        key.mtp_depth = mtp_depth;
        key.layer_idx = layer_idx;
        key.tier_idx = tier_idx;
        key.domain_id = domain_id;
        key.participant_id = participant_id;
        key.direction = direction;
        key.sequence = makeCollectiveSequence(key.key_namespace,
                                              key.mtp_depth,
                                              key.layer_idx,
                                              key.tier_idx,
                                              key.participant_id,
                                              key.direction);
        return key;
    }

    size_t denseMoEOverlayDispatchBytes(int seq_len, int top_k, int d_model)
    {
        if (seq_len <= 0 || top_k <= 0 || d_model <= 0)
            return 0;

        const size_t rows = static_cast<size_t>(seq_len);
        const size_t hidden_row_bytes = static_cast<size_t>(d_model) * sizeof(float);
        const size_t routing_row_bytes = static_cast<size_t>(top_k) * 2u * sizeof(float);
        return rows * (hidden_row_bytes + routing_row_bytes);
    }

    size_t denseMoEOverlayReturnBytes(int seq_len, int d_model)
    {
        if (seq_len <= 0 || d_model <= 0)
            return 0;

        return static_cast<size_t>(seq_len) * static_cast<size_t>(d_model) * sizeof(float);
    }

    size_t compactMoEOverlayDispatchBytes(const MoEOverlaySparseRows &rows)
    {
        if (rows.d_model <= 0)
            return 0;

        return rows.live_row_count * sizeof(int32_t) +
               (rows.live_row_count + 1u) * sizeof(int32_t) +
               rows.live_entry_count * sizeof(int32_t) +
               rows.live_entry_count * sizeof(float) +
               rows.live_row_count * static_cast<size_t>(rows.d_model) * sizeof(float);
    }

    size_t compactMoEOverlayReturnBytes(const MoEOverlayReturnRows &rows)
    {
        if (rows.d_model <= 0)
            return 0;

        return rows.live_row_count * sizeof(int32_t) +
               rows.live_row_count * static_cast<size_t>(rows.d_model) * sizeof(float);
    }

    MoEOverlaySparseTransferCounters measureMoEOverlaySparseTransferCounters(
        int seq_len,
        int top_k,
        int d_model,
        const MoEOverlaySparseRows *dispatch_rows,
        const MoEOverlayReturnRows *return_rows)
    {
        MoEOverlaySparseTransferCounters counters;
        counters.dense_dispatch_bytes = denseMoEOverlayDispatchBytes(seq_len, top_k, d_model);
        counters.dense_return_bytes = denseMoEOverlayReturnBytes(seq_len, d_model);

        if (dispatch_rows)
        {
            counters.compact_dispatch_bytes = compactMoEOverlayDispatchBytes(*dispatch_rows);
            counters.compact_row_count = dispatch_rows->live_row_count;
            counters.compact_entry_count = dispatch_rows->live_entry_count;
        }
        if (return_rows)
        {
            counters.compact_return_bytes = compactMoEOverlayReturnBytes(*return_rows);
            counters.compact_row_count = std::max(counters.compact_row_count, return_rows->live_row_count);
        }

        return counters;
    }

    void MoEOverlayCollectiveWorkspace::ensureCapacity(size_t max_rows,
                                                       size_t max_entries,
                                                       int d_model,
                                                       int top_k,
                                                       DeviceId device)
    {
        max_rows_ = std::max(max_rows_, max_rows);
        max_entries_ = std::max(max_entries_, max_entries);
        d_model_ = std::max(d_model_, d_model);
        top_k_ = std::max(top_k_, top_k);
        device_ = device;

        for (auto &[_, buffers] : buffers_by_layer_tier_)
        {
            ensureSparseStorage(buffers.dispatch_receive);
            ensureSparseStorage(buffers.local_expert_input);
            ensureReturnStorage(buffers.local_expert_output);
            ensureReturnStorage(buffers.return_receive);
        }
    }

    void MoEOverlayCollectiveWorkspace::resetForStep(uint64_t generation_id,
                                                     uint64_t step_id)
    {
        generation_id_ = generation_id;
        step_id_ = step_id;

        for (auto &[_, buffers] : buffers_by_layer_tier_)
        {
            if (!buffers.dispatch_receive.entry_offsets_host.empty())
                buffers.dispatch_receive.entry_offsets_host[0] = 0;
            if (!buffers.local_expert_input.entry_offsets_host.empty())
                buffers.local_expert_input.entry_offsets_host[0] = 0;
        }
    }

    MoEOverlayCollectiveWorkspace::LayerTierBuffers &
    MoEOverlayCollectiveWorkspace::buffersFor(int layer_idx, int tier_idx)
    {
        auto &buffers = buffers_by_layer_tier_[{layer_idx, tier_idx}];
        ensureSparseStorage(buffers.dispatch_receive);
        ensureSparseStorage(buffers.local_expert_input);
        ensureReturnStorage(buffers.local_expert_output);
        ensureReturnStorage(buffers.return_receive);
        return buffers;
    }

    void MoEOverlayCollectiveWorkspace::ensureSparseStorage(SparseStorage &storage)
    {
        if (max_rows_ == 0 || max_entries_ == 0 || d_model_ <= 0 || top_k_ <= 0)
            return;

        if (storage.row_ids_host.size() < max_rows_)
            storage.row_ids_host.resize(max_rows_);
        if (storage.entry_offsets_host.size() < max_rows_ + 1u)
            storage.entry_offsets_host.resize(max_rows_ + 1u, 0);
        if (storage.expert_ids_host.size() < max_entries_)
            storage.expert_ids_host.resize(max_entries_);
        if (storage.route_weights_host.size() < max_entries_)
            storage.route_weights_host.resize(max_entries_);
        const size_t hidden_count = max_rows_ * static_cast<size_t>(d_model_);
        if (storage.hidden_rows_fp32.size() < hidden_count)
            storage.hidden_rows_fp32.resize(hidden_count);
    }

    void MoEOverlayCollectiveWorkspace::ensureReturnStorage(ReturnStorage &storage)
    {
        if (max_rows_ == 0 || d_model_ <= 0)
            return;

        if (storage.row_ids_host.size() < max_rows_)
            storage.row_ids_host.resize(max_rows_);
        const size_t output_count = max_rows_ * static_cast<size_t>(d_model_);
        if (storage.output_rows_fp32.size() < output_count)
            storage.output_rows_fp32.resize(output_count);
    }

    MoEOverlaySparseRows MoEOverlayCollectiveWorkspace::dispatchReceive(int layer_idx, int tier_idx)
    {
        auto &storage = buffersFor(layer_idx, tier_idx).dispatch_receive;
        MoEOverlaySparseRows view;
        view.d_model = d_model_;
        view.top_k = top_k_;
        view.row_capacity = storage.row_ids_host.size();
        view.entry_capacity = storage.expert_ids_host.size();
        view.row_ids_host = storage.row_ids_host.data();
        view.entry_offsets_host = storage.entry_offsets_host.data();
        view.expert_ids_host = storage.expert_ids_host.data();
        view.route_weights_host = storage.route_weights_host.data();
        view.hidden_rows_fp32 = storage.hidden_rows_fp32.data();
        view.live_row_count = 0;
        view.live_entry_count = 0;
        if (view.entry_offsets_host && view.row_capacity > 0)
            view.entry_offsets_host[0] = 0;
        return view;
    }

    MoEOverlaySparseRows MoEOverlayCollectiveWorkspace::localExpertInput(int layer_idx, int tier_idx)
    {
        auto &storage = buffersFor(layer_idx, tier_idx).local_expert_input;
        MoEOverlaySparseRows view;
        view.d_model = d_model_;
        view.top_k = top_k_;
        view.row_capacity = storage.row_ids_host.size();
        view.entry_capacity = storage.expert_ids_host.size();
        view.row_ids_host = storage.row_ids_host.data();
        view.entry_offsets_host = storage.entry_offsets_host.data();
        view.expert_ids_host = storage.expert_ids_host.data();
        view.route_weights_host = storage.route_weights_host.data();
        view.hidden_rows_fp32 = storage.hidden_rows_fp32.data();
        view.live_row_count = 0;
        view.live_entry_count = 0;
        if (view.entry_offsets_host && view.row_capacity > 0)
            view.entry_offsets_host[0] = 0;
        return view;
    }

    MoEOverlayReturnRows MoEOverlayCollectiveWorkspace::localExpertOutput(int layer_idx, int tier_idx)
    {
        auto &storage = buffersFor(layer_idx, tier_idx).local_expert_output;
        MoEOverlayReturnRows view;
        view.d_model = d_model_;
        view.row_capacity = storage.row_ids_host.size();
        view.row_ids_host = storage.row_ids_host.data();
        view.output_rows_fp32 = storage.output_rows_fp32.data();
        view.live_row_count = 0;
        return view;
    }

    MoEOverlayReturnRows MoEOverlayCollectiveWorkspace::returnReceive(int layer_idx, int tier_idx)
    {
        auto &storage = buffersFor(layer_idx, tier_idx).return_receive;
        MoEOverlayReturnRows view;
        view.d_model = d_model_;
        view.row_capacity = storage.row_ids_host.size();
        view.row_ids_host = storage.row_ids_host.data();
        view.output_rows_fp32 = storage.output_rows_fp32.data();
        view.live_row_count = 0;
        return view;
    }

    struct MoEOverlayLocalSparseCollectiveContext::Slot
    {
        explicit Slot(int participant_count)
            : seen(static_cast<size_t>(participant_count), false),
              dispatch_payloads(static_cast<size_t>(participant_count)),
              return_payloads(static_cast<size_t>(participant_count))
        {
        }

        std::mutex mutex;
        bool active = false;
        MoEOverlayCollectiveKey key;
        int seen_count = 0;
        std::vector<bool> seen;
        std::vector<DispatchPayloadCopy> dispatch_payloads;
        std::vector<ReturnPayloadCopy> return_payloads;

        void reset()
        {
            active = false;
            key = {};
            seen_count = 0;
            std::fill(seen.begin(), seen.end(), false);
            for (auto &payload : dispatch_payloads)
                payload = {};
            for (auto &payload : return_payloads)
                payload = {};
        }
    };

    MoEOverlayLocalSparseCollectiveContext::MoEOverlayLocalSparseCollectiveContext(Config config)
        : participant_count_(config.participant_count)
    {
        if (participant_count_ <= 0)
            throw std::invalid_argument("local sparse collective requires participant_count > 0");
        if (config.slot_count == 0)
            throw std::invalid_argument("local sparse collective requires slot_count > 0");

        slots_.reserve(config.slot_count);
        for (size_t slot = 0; slot < config.slot_count; ++slot)
            slots_.push_back(std::make_unique<Slot>(participant_count_));
    }

    MoEOverlayLocalSparseCollectiveContext::~MoEOverlayLocalSparseCollectiveContext() = default;

    MoEOverlayCollectiveResult MoEOverlayLocalSparseCollectiveContext::dispatch(const MoEOverlayCollectiveKey &key,
                                                                                const MoEOverlaySparseRows &outbound,
                                                                                MoEOverlaySparseRows *inbound,
                                                                                IDeviceContext *)
    {
        return publishDispatch(key, outbound, inbound);
    }

    MoEOverlayCollectiveResult MoEOverlayLocalSparseCollectiveContext::returnReduce(const MoEOverlayCollectiveKey &key,
                                                                                    const MoEOverlayReturnRows &outbound,
                                                                                    MoEOverlayReturnRows *inbound,
                                                                                    IDeviceContext *)
    {
        return publishReturn(key, outbound, inbound);
    }

    void MoEOverlayLocalSparseCollectiveContext::abort(const MoEOverlayCollectiveKey &key,
                                                       int reason_code)
    {
        aborted_keys_[keyToStableString(key)] = reason_code == 0 ? 1 : reason_code;
    }

    MoEOverlayCollectiveResult MoEOverlayLocalSparseCollectiveContext::publishDispatch(const MoEOverlayCollectiveKey &key,
                                                                                       const MoEOverlaySparseRows &outbound,
                                                                                       MoEOverlaySparseRows *inbound)
    {
        MoEOverlayCollectiveResult result;
        if (key.direction != MoEOverlayCollectiveDirection::Dispatch)
        {
            result.ok = false;
            result.error_code = 1;
            result.error = "dispatch called with non-dispatch key";
            return result;
        }
        if (!key.isValid())
        {
            result.ok = false;
            result.error_code = 2;
            result.error = "invalid dispatch key";
            return result;
        }

        std::string validation_error;
        if (!validateSparseRows(outbound, &validation_error))
        {
            result.ok = false;
            result.error_code = 3;
            result.error = validation_error;
            return result;
        }

        const auto key_string = keyToStableString(key);
        if (const auto aborted = aborted_keys_.find(key_string); aborted != aborted_keys_.end())
        {
            result.ok = false;
            result.error_code = aborted->second;
            result.error = "dispatch key was aborted";
            return result;
        }
        if (completed_keys_.count(key_string) > 0)
        {
            result.ok = false;
            result.error_code = 4;
            result.error = "stale dispatch key reuse rejected";
            return result;
        }

        if (outbound.source_participant < 0 || outbound.source_participant >= participant_count_)
        {
            result.ok = false;
            result.error_code = 5;
            result.error = "dispatch source participant out of range";
            return result;
        }

        auto &slot = *slots_[static_cast<size_t>(key.sequence) % slots_.size()];
        std::lock_guard<std::mutex> guard(slot.mutex);
        if (!slot.active)
        {
            slot.active = true;
            slot.key = key;
            slot.seen_count = 0;
        }
        else if (slot.key != key)
        {
            result.ok = false;
            result.error_code = 6;
            result.error = "dispatch slot occupied by another key";
            return result;
        }

        const size_t participant = static_cast<size_t>(outbound.source_participant);
        if (slot.seen[participant])
        {
            result.ok = false;
            result.error_code = 7;
            result.error = "duplicate dispatch publish for participant";
            return result;
        }

        slot.seen[participant] = true;
        ++slot.seen_count;
        slot.dispatch_payloads[participant] = copyDispatchPayload(outbound);

        if (inbound)
        {
            inbound->key = key;
            inbound->source_participant = outbound.source_participant;
            inbound->target_participant = outbound.source_participant;
            inbound->d_model = outbound.d_model;
            inbound->top_k = outbound.top_k;
            inbound->live_row_count = 0;
            inbound->live_entry_count = 0;
            if (inbound->entry_offsets_host && inbound->row_capacity > 0)
                inbound->entry_offsets_host[0] = 0;
        }

        if (slot.seen_count != participant_count_)
            return result;

        result.collective_complete = true;
        const int target = outbound.source_participant;
        if (inbound)
        {
            for (const auto &payload : slot.dispatch_payloads)
            {
                if (payload.target_participant != target)
                    continue;
                std::string append_error;
                if (!appendDispatchInbound(payload, inbound, &append_error))
                {
                    result.ok = false;
                    result.error_code = 8;
                    result.error = append_error;
                    slot.reset();
                    return result;
                }
            }
        }

        completed_keys_.insert(key_string);
        slot.reset();
        return result;
    }

    MoEOverlayCollectiveResult MoEOverlayLocalSparseCollectiveContext::publishReturn(const MoEOverlayCollectiveKey &key,
                                                                                     const MoEOverlayReturnRows &outbound,
                                                                                     MoEOverlayReturnRows *inbound)
    {
        MoEOverlayCollectiveResult result;
        if (key.direction != MoEOverlayCollectiveDirection::ReturnReduce)
        {
            result.ok = false;
            result.error_code = 1;
            result.error = "returnReduce called with non-return key";
            return result;
        }
        if (!key.isValid())
        {
            result.ok = false;
            result.error_code = 2;
            result.error = "invalid return key";
            return result;
        }

        std::string validation_error;
        if (!validateReturnRows(outbound, &validation_error))
        {
            result.ok = false;
            result.error_code = 3;
            result.error = validation_error;
            return result;
        }

        const auto key_string = keyToStableString(key);
        if (const auto aborted = aborted_keys_.find(key_string); aborted != aborted_keys_.end())
        {
            result.ok = false;
            result.error_code = aborted->second;
            result.error = "return key was aborted";
            return result;
        }
        if (completed_keys_.count(key_string) > 0)
        {
            result.ok = false;
            result.error_code = 4;
            result.error = "stale return key reuse rejected";
            return result;
        }

        if (outbound.source_participant < 0 || outbound.source_participant >= participant_count_)
        {
            result.ok = false;
            result.error_code = 5;
            result.error = "return source participant out of range";
            return result;
        }

        auto &slot = *slots_[static_cast<size_t>(key.sequence) % slots_.size()];
        std::lock_guard<std::mutex> guard(slot.mutex);
        if (!slot.active)
        {
            slot.active = true;
            slot.key = key;
            slot.seen_count = 0;
        }
        else if (slot.key != key)
        {
            result.ok = false;
            result.error_code = 6;
            result.error = "return slot occupied by another key";
            return result;
        }

        const size_t participant = static_cast<size_t>(outbound.source_participant);
        if (slot.seen[participant])
        {
            result.ok = false;
            result.error_code = 7;
            result.error = "duplicate return publish for participant";
            return result;
        }

        slot.seen[participant] = true;
        ++slot.seen_count;
        slot.return_payloads[participant] = copyReturnPayload(outbound);

        if (inbound)
        {
            inbound->key = key;
            inbound->source_participant = outbound.source_participant;
            inbound->target_participant = outbound.source_participant;
            inbound->d_model = outbound.d_model;
            inbound->live_row_count = 0;
        }

        if (slot.seen_count != participant_count_)
            return result;

        result.collective_complete = true;
        const int target = outbound.source_participant;
        if (inbound)
        {
            for (const auto &payload : slot.return_payloads)
            {
                if (payload.target_participant != target)
                    continue;
                std::string append_error;
                if (!appendReturnInbound(payload, inbound, &append_error))
                {
                    result.ok = false;
                    result.error_code = 8;
                    result.error = append_error;
                    slot.reset();
                    return result;
                }
            }
        }

        completed_keys_.insert(key_string);
        slot.reset();
        return result;
    }

    MoEOverlayMPISparseCollectiveContext::MoEOverlayMPISparseCollectiveContext(Config config)
        : config_(std::move(config))
    {
    }

    MoEOverlayMPISparseCollectiveContext::~MoEOverlayMPISparseCollectiveContext() = default;

    int MoEOverlayMPISparseCollectiveContext::localParticipantId() const
    {
        if (config_.local_participant_id >= 0)
            return config_.local_participant_id;
        if (!config_.mpi_ctx)
            return 0;
        return config_.mpi_ctx->rank();
    }

    MoEOverlayCollectiveResult MoEOverlayMPISparseCollectiveContext::dispatch(const MoEOverlayCollectiveKey &key,
                                                                              const MoEOverlaySparseRows &outbound,
                                                                              MoEOverlaySparseRows *inbound,
                                                                              IDeviceContext *)
    {
        return dispatchHostStaged(key, outbound, inbound);
    }

    MoEOverlayCollectiveResult MoEOverlayMPISparseCollectiveContext::returnReduce(const MoEOverlayCollectiveKey &key,
                                                                                  const MoEOverlayReturnRows &outbound,
                                                                                  MoEOverlayReturnRows *inbound,
                                                                                  IDeviceContext *)
    {
        return returnHostStaged(key, outbound, inbound);
    }

    void MoEOverlayMPISparseCollectiveContext::abort(const MoEOverlayCollectiveKey &key,
                                                     int reason_code)
    {
        aborted_keys_[keyToStableString(key)] = reason_code == 0 ? 1 : reason_code;
    }

    MoEOverlayCollectiveResult MoEOverlayMPISparseCollectiveContext::dispatchHostStaged(const MoEOverlayCollectiveKey &key,
                                                                                        const MoEOverlaySparseRows &outbound,
                                                                                        MoEOverlaySparseRows *inbound)
    {
        MoEOverlayCollectiveResult result;
        if (!config_.mpi_ctx)
        {
            result.ok = false;
            result.error_code = 1;
            result.error = "MPI sparse collective has no MPI context";
            return result;
        }
        if (key.direction != MoEOverlayCollectiveDirection::Dispatch)
        {
            result.ok = false;
            result.error_code = 2;
            result.error = "dispatch called with non-dispatch key";
            return result;
        }

        const auto key_string = keyToStableString(key);
        if (const auto aborted = aborted_keys_.find(key_string); aborted != aborted_keys_.end())
        {
            result.ok = false;
            result.error_code = aborted->second;
            result.error = "dispatch key was aborted";
            return result;
        }
        if (completed_keys_.count(key_string) > 0)
        {
            result.ok = false;
            result.error_code = 3;
            result.error = "stale dispatch key reuse rejected";
            return result;
        }

        std::string validation_error;
        if (!validateSparseRows(outbound, &validation_error))
        {
            result.ok = false;
            result.error_code = 4;
            result.error = validation_error;
            return result;
        }

        DispatchPayloadCopy outbound_copy = copyDispatchPayload(outbound);
        std::vector<uint8_t> send_packet = serializeDispatchPacket(outbound_copy);
        std::vector<int> recv_counts;
        std::vector<int> displs;
        std::vector<uint8_t> recv_payloads = gatherAllPayloads(*config_.mpi_ctx, send_packet, &recv_counts, &displs);

        if (inbound)
        {
            inbound->key = key;
            inbound->source_participant = localParticipantId();
            inbound->target_participant = localParticipantId();
            inbound->d_model = outbound.d_model;
            inbound->top_k = outbound.top_k;
            inbound->live_row_count = 0;
            inbound->live_entry_count = 0;
            if (inbound->entry_offsets_host && inbound->row_capacity > 0)
                inbound->entry_offsets_host[0] = 0;
        }

        const int target = localParticipantId();
        for (int rank = 0; rank < config_.mpi_ctx->world_size(); ++rank)
        {
            const int count = recv_counts[static_cast<size_t>(rank)];
            const int disp = displs[static_cast<size_t>(rank)];
            if (count <= 0)
                continue;

            DispatchPayloadCopy packet;
            std::string parse_error;
            if (!deserializeDispatchPacket(recv_payloads.data() + disp,
                                           static_cast<size_t>(count),
                                           &packet,
                                           &parse_error))
            {
                result.ok = false;
                result.error_code = 5;
                result.error = parse_error;
                return result;
            }
            if (packet.key != key)
            {
                result.ok = false;
                result.error_code = 6;
                result.error = "dispatch key mismatch across MPI ranks";
                return result;
            }

            if (inbound && packet.target_participant == target)
            {
                std::string append_error;
                if (!appendDispatchInbound(packet, inbound, &append_error))
                {
                    result.ok = false;
                    result.error_code = 7;
                    result.error = append_error;
                    return result;
                }
            }
        }

        completed_keys_.insert(key_string);
        result.collective_complete = true;
        return result;
    }

    MoEOverlayCollectiveResult MoEOverlayMPISparseCollectiveContext::returnHostStaged(const MoEOverlayCollectiveKey &key,
                                                                                      const MoEOverlayReturnRows &outbound,
                                                                                      MoEOverlayReturnRows *inbound)
    {
        MoEOverlayCollectiveResult result;
        if (!config_.mpi_ctx)
        {
            result.ok = false;
            result.error_code = 1;
            result.error = "MPI sparse collective has no MPI context";
            return result;
        }
        if (key.direction != MoEOverlayCollectiveDirection::ReturnReduce)
        {
            result.ok = false;
            result.error_code = 2;
            result.error = "returnReduce called with non-return key";
            return result;
        }

        const auto key_string = keyToStableString(key);
        if (const auto aborted = aborted_keys_.find(key_string); aborted != aborted_keys_.end())
        {
            result.ok = false;
            result.error_code = aborted->second;
            result.error = "return key was aborted";
            return result;
        }
        if (completed_keys_.count(key_string) > 0)
        {
            result.ok = false;
            result.error_code = 3;
            result.error = "stale return key reuse rejected";
            return result;
        }

        std::string validation_error;
        if (!validateReturnRows(outbound, &validation_error))
        {
            result.ok = false;
            result.error_code = 4;
            result.error = validation_error;
            return result;
        }

        ReturnPayloadCopy outbound_copy = copyReturnPayload(outbound);
        std::vector<uint8_t> send_packet = serializeReturnPacket(outbound_copy);
        std::vector<int> recv_counts;
        std::vector<int> displs;
        std::vector<uint8_t> recv_payloads = gatherAllPayloads(*config_.mpi_ctx, send_packet, &recv_counts, &displs);

        if (inbound)
        {
            inbound->key = key;
            inbound->source_participant = localParticipantId();
            inbound->target_participant = localParticipantId();
            inbound->d_model = outbound.d_model;
            inbound->live_row_count = 0;
        }

        const int target = localParticipantId();
        for (int rank = 0; rank < config_.mpi_ctx->world_size(); ++rank)
        {
            const int count = recv_counts[static_cast<size_t>(rank)];
            const int disp = displs[static_cast<size_t>(rank)];
            if (count <= 0)
                continue;

            ReturnPayloadCopy packet;
            std::string parse_error;
            if (!deserializeReturnPacket(recv_payloads.data() + disp,
                                         static_cast<size_t>(count),
                                         &packet,
                                         &parse_error))
            {
                result.ok = false;
                result.error_code = 5;
                result.error = parse_error;
                return result;
            }
            if (packet.key != key)
            {
                result.ok = false;
                result.error_code = 6;
                result.error = "return key mismatch across MPI ranks";
                return result;
            }

            if (inbound && packet.target_participant == target)
            {
                std::string append_error;
                if (!appendReturnInbound(packet, inbound, &append_error))
                {
                    result.ok = false;
                    result.error_code = 7;
                    result.error = append_error;
                    return result;
                }
            }
        }

        completed_keys_.insert(key_string);
        result.collective_complete = true;
        return result;
    }

} // namespace llaminar2
