/**
 * @file PersistentStateMemoryEstimator.h
 * @brief Exact cache and recurrent-state memory planning for hybrid models.
 *
 * The estimator mirrors production cache construction: full-attention layers
 * reserve sequence-length-scaled KV storage, GDN layers reserve fixed live
 * state banks, and MTP reserves shifted KV plus concurrently retained rollback
 * payloads. No generic reserve is added.
 */

#pragma once

#include "backends/DeviceId.h"

#include <cstddef>
#include <string>

namespace llaminar2
{
    struct ModelMemoryProfile;

    /**
     * @brief Persistent cache/state bytes that coexist with graph workspaces.
     */
    struct PersistentStateEstimate
    {
        size_t kv_cache_bytes = 0;
        size_t live_recurrent_state_bytes = 0;
        size_t checkpoint_state_bytes = 0;
        size_t sequence_metadata_bytes = 0;
        /** Exact GPU serialization bytes for the main cache's GDN state. */
        size_t prefix_hybrid_device_state_bytes = 0;
        int main_full_attention_layers = 0;
        int main_gdn_layers = 0;
        int mtp_full_attention_layers = 0;
        int mtp_gdn_layers = 0;

        size_t stateBytes() const noexcept
        {
            return live_recurrent_state_bytes +
                   checkpoint_state_bytes +
                   sequence_metadata_bytes;
        }
    };

    class PersistentStateMemoryEstimator
    {
    public:
        /**
         * @brief Reproduce cache-owned persistent allocations for one device.
         *
         * @param profile Compact GGUF architecture and tensor inventory.
         * @param device Target CPU, CUDA, or ROCm device.
         * @param batch_size Maximum simultaneously live request count.
         * @param max_seq_len Full KV horizon, independent of graph row buckets.
         * @param local_kv_heads Participant-local full-attention KV heads.
         * @param local_query_heads Exact participant-local query heads used by
         *        recurrent/GDN state sharding.
         * @param total_shards Tensor-parallel participant count.
         * @param first_layer First main-model layer owned by this participant.
         * @param last_layer Last main-model layer owned by this participant.
         * @param kv_precision Runtime KV storage format.
         * @param mtp_enabled Whether shifted MTP caches/checkpoints are resident.
         */
        static PersistentStateEstimate estimate(
            const ModelMemoryProfile &profile,
            DeviceId device,
            int batch_size,
            int max_seq_len,
            int local_kv_heads,
            int local_query_heads,
            int total_shards,
            int first_layer,
            int last_layer,
            const std::string &kv_precision,
            bool mtp_enabled);
    };
}
