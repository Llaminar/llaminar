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
        /** Complete persistent bytes owned by committed and shifted caches. */
        size_t kv_cache_bytes = 0;
        /** Persistent bytes owned only by the committed main-model cache. */
        size_t main_kv_cache_bytes = 0;
        /** Persistent bytes owned only by the shifted MTP sidecar cache. */
        size_t mtp_kv_cache_bytes = 0;
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
         * @param main_local_kv_heads Participant-local main-attention KV heads.
         * @param mtp_local_kv_heads Participant-local shifted-sidecar KV heads;
         *        this may equal the full model width when the predictor is
         *        replicated beside a sharded main cache.
         * @param local_query_head_start First globally numbered query head
         *        owned by this participant.
         * @param local_query_heads Exact participant-local query heads used by
         *        recurrent/GDN state sharding.
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
            int main_local_kv_heads,
            int mtp_local_kv_heads,
            int local_query_head_start,
            int local_query_heads,
            int first_layer,
            int last_layer,
            const std::string &kv_precision,
            bool mtp_enabled);
    };
}
