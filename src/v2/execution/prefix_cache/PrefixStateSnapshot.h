#pragma once

#include "backends/DeviceId.h"
#include "execution/prefix_cache/PrefixStorageBackend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{

    enum class PrefixStateProvenance
    {
        Unknown,
        PayloadCheckpoint,
        LogicalCheckpoint,
        DecodeEquivalent,
        VerifierPrefillRows,
        VerifierPrefillRowsDecodeEquivalent,
        SidecarDraftOnly,
    };

    const char *toString(PrefixStateProvenance provenance);
    bool isDecodeEquivalent(PrefixStateProvenance provenance);

    /**
     * @brief Opaque device-owned checkpoint of one KV cache sequence.
     *
     * The checkpoint contains backend-private canonical ring metadata, never
     * KV payload bytes and never a host mirror. @ref cache_depth identifies
     * the primary cache with -1 and shifted MTP cache N with N. The allocation
     * is retained through shared ownership so an asynchronous capture cannot
     * outlive or race reuse of its pool slot.
     */
    struct DeviceKVSequenceStateCheckpoint
    {
        int cache_depth = -1;
        int sequence_index = -1;
        int metadata_layer_count = 0;
        size_t bytes = 0;
        DeviceId device = DeviceId::invalid();

        /**
         * Canonical SequenceMetadata claim retained with the pooled storage.
         *
         * This member precedes the allocation owner deliberately: reverse
         * destruction frees VRAM before releasing its admitted byte claim.
         */
        std::shared_ptr<void> physical_memory_lease;
        std::shared_ptr<void> storage;
        std::shared_ptr<void> ready_event;

        void *data() const
        {
            return storage.get();
        }

        bool valid() const
        {
            return cache_depth >= -1 &&
                   sequence_index >= 0 &&
                   metadata_layer_count > 0 &&
                   bytes > 0 &&
                   device.is_gpu() &&
                   storage != nullptr &&
                   ready_event != nullptr;
        }
    };

    struct PrefixLookupResult
    {
        bool supported = false;
        bool cache_enabled = false;
        int cached_tokens = 0;
        int block_size = 0;
        uint64_t fingerprint_key = 0;
        uint64_t placement_epoch = 0;
        bool requires_terminal_hidden = true;
        bool requires_terminal_logits = true;
        bool has_terminal_hidden = false;
        bool has_terminal_logits = false;
        bool restore_model_runtime_state = true;
        bool restore_hybrid_state_for_suffix_prefill = false;
        std::string bypass_reason;
        std::vector<PrefixBlockHandle> blocks;

        bool hit() const { return supported && cached_tokens > 0 && !blocks.empty(); }
        PrefixLookupResult clampedTo(int token_count) const;
    };

    /**
     * @brief Scheduler-owned identity for one live rollback checkpoint.
     *
     * A GPU checkpoint archives device-owned KV and recurrent state
     * asynchronously.  It must not rediscover the logical request cursor from
     * a host mirror because that mirror may legitimately lag a graph-captured
     * publication.  The request scheduler already owns the exact transaction
     * position, so it supplies that immutable control-plane fact alongside the
     * sequence being archived.
     *
     * The fields intentionally have invalid defaults so callers cannot obtain
     * a meaningful checkpoint by default construction.  Concrete runners must
     * reject a request unless @ref valid returns true.
     */
    struct PrefixCheckpointCaptureRequest
    {
        int sequence_index = -1;
        int logical_cached_tokens = -1;

        /**
         * @brief Whether the request names a valid sequence cursor.
         */
        bool valid() const
        {
            return sequence_index >= 0 && logical_cached_tokens >= 0;
        }
    };

    struct PrefixStateSnapshot
    {
        PrefixStateSnapshot() = default;
        ~PrefixStateSnapshot() = default;
        PrefixStateSnapshot(const PrefixStateSnapshot &) = default;
        PrefixStateSnapshot &operator=(const PrefixStateSnapshot &) = default;
        PrefixStateSnapshot(PrefixStateSnapshot &&other) noexcept
        {
            swap(other);
        }
        PrefixStateSnapshot &operator=(PrefixStateSnapshot &&other) noexcept
        {
            if (this != &other)
            {
                PrefixStateSnapshot old;
                swap(old);
                swap(other);
            }
            return *this;
        }

        bool valid = false;
        bool logical_checkpoint = false;
        PrefixStateProvenance provenance = PrefixStateProvenance::Unknown;
        int cached_tokens = 0;
        std::vector<int> mtp_cached_tokens;
        std::vector<DeviceKVSequenceStateCheckpoint>
            device_sequence_state_checkpoints;
        std::vector<PrefixBlockHandle> blocks;
        std::vector<PrefixBlockHandle> mtp_blocks;
        std::vector<PrefixStateSnapshot> participant_snapshots;
        /**
         * @brief Optional GPU event proving asynchronous snapshot payloads are ready.
         *
         * Logical MTP checkpoints can export recurrent/hybrid state into
         * device-resident payload buffers without synchronizing the host.  The
         * event lets later restores wait on the producing stream before they
         * import from that payload.  The producing runner also keeps a pending
         * copy of the same event so the next live-state mutation cannot
         * overwrite the source state while the snapshot copy is still queued.
         */
        std::shared_ptr<void> ready_event;
        void *ready_producer_stream = nullptr;
        DeviceId ready_device = DeviceId::invalid();
        bool ready_event_valid = false;

        bool decodeEquivalent() const
        {
            return isDecodeEquivalent(provenance);
        }

        void swap(PrefixStateSnapshot &other) noexcept
        {
            using std::swap;
            swap(valid, other.valid);
            swap(logical_checkpoint, other.logical_checkpoint);
            swap(provenance, other.provenance);
            swap(cached_tokens, other.cached_tokens);
            mtp_cached_tokens.swap(other.mtp_cached_tokens);
            device_sequence_state_checkpoints.swap(
                other.device_sequence_state_checkpoints);
            blocks.swap(other.blocks);
            mtp_blocks.swap(other.mtp_blocks);
            participant_snapshots.swap(other.participant_snapshots);
            ready_event.swap(other.ready_event);
            swap(ready_producer_stream, other.ready_producer_stream);
            swap(ready_device, other.ready_device);
            swap(ready_event_valid, other.ready_event_valid);
        }
    };

    inline void swap(PrefixStateSnapshot &lhs, PrefixStateSnapshot &rhs) noexcept
    {
        lhs.swap(rhs);
    }

} // namespace llaminar2
