/**
 * @file PrefixStorageBackend.h
 * @brief Typed ownership handles for RAM, device-hot, and disk prefix payloads.
 *
 * Prefix snapshots are serialized storage objects, not mirrors of live
 * inference state. CPU snapshots may use ordinary vectors. GPU snapshots use
 * pinned host allocations for the RAM capacity tier so harvest and restore can
 * use true asynchronous DMA on explicit streams. Optional device-hot copies
 * remain accelerators for frequently reused entries and never replace RAM as
 * the primary capacity tier.
 */

#pragma once

#include "execution/prefix_cache/PrefixCacheKey.h"
#include "execution/prefix_cache/PrefixPayloadLayout.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

namespace llaminar2
{
    class TensorBase;

    /**
     * @brief Shared completion state for an asynchronously serialized RAM block.
     *
     * A GPU prefix harvest first packs live state into persistent device
     * staging, then submits D2H copies into the block's pinned RAM sections.
     * The host bytes are not readable merely because the API calls returned.
     * This object ties those bytes to the producer event and provides the two
     * legal consumption boundaries:
     *
     * - GPU restore queues a stream wait and immediately continues with H2D.
     * - Host consumers such as disk serialization wait for this one event.
     *
     * The state is shared by every copy of PrefixBlockHandle and by each pinned
     * allocation deleter. Consequently, an evicted block cannot free DMA
     * destinations while the transfer that fills them is still in flight.
     */
    class PrefixPayloadReadiness
    {
    public:
        /**
         * @brief Prepare an event-backed ownership edge before queuing copies.
         *
         * Preparation must happen before the first asynchronous copy targets
         * this handle. That ordering makes event-allocation failure recoverable:
         * no pinned or device allocation can already be in use when this method
         * returns false.
         *
         * @param ready_event Unrecorded backend event with shared destruction ownership.
         * @param producer_device GPU that owns the event.
         * @param producer_stream Explicit stream that will own every queued copy.
         * @return true when the completion contract is prepared.
         */
        bool prepare(
            std::shared_ptr<void> ready_event,
            DeviceId producer_device,
            void *producer_stream);

        /**
         * @brief Publish a successfully recorded prepared event.
         *
         * The caller records `event()` after the final queued copy and then
         * invokes this method. Returning false after asynchronous work has been
         * submitted is an unrecoverable ownership-contract violation: callers
         * must fail hard rather than synchronize or release storage.
         *
         * @return true exactly once after a valid preparation.
         */
        bool publishRecorded();

        /** @return Opaque prepared backend event, or nullptr before preparation. */
        void *event() const { return ready_event_.get(); }

        /**
         * @brief Queue a dependency on a GPU consumer stream.
         *
         * This does not block the host. A same-stream consumer needs no backend
         * call because stream order already proves readiness.
         */
        bool waitOnStream(void *consumer_stream) const;

        /**
         * @brief Wait on the producer event at a true host-consumption boundary.
         *
         * Repeated calls are cheap after the first successful wait. This method
         * is also used by pinned allocation deleters before returning storage to
         * the backend.
         */
        bool waitOnHost() const;

        bool published() const { return published_; }
        bool prepared() const { return ready_event_ != nullptr; }
        DeviceId producerDevice() const { return producer_device_; }
        void *producerStream() const { return producer_stream_; }

    private:
        std::shared_ptr<void> ready_event_;
        DeviceId producer_device_ = DeviceId::invalid();
        void *producer_stream_ = nullptr;
        bool published_ = false;
        mutable std::atomic<bool> host_wait_complete_{false};
    };

    enum class PrefixStorageTier
    {
        DeviceHot,
        Ram,
        Disk,
    };

    struct PrefixBlockHandle
    {
        PrefixCacheKey key;
        PrefixStorageTier tier = PrefixStorageTier::Ram;
        PrefixPayloadLayout layout;
        void *kv_payload = nullptr;
        void *hybrid_payload = nullptr;
        void *mtp_payload = nullptr;
        void *terminal_hidden = nullptr;
        void *terminal_logits = nullptr;
        size_t total_bytes = 0;
        bool has_hybrid_state = false;
        bool has_terminal_hidden = false;
        bool has_terminal_logits = false;
        bool has_model_runtime_state = false;

        std::shared_ptr<std::vector<uint8_t>> kv_storage;
        std::shared_ptr<std::vector<uint8_t>> hybrid_storage;
        std::shared_ptr<std::vector<uint8_t>> mtp_storage;
        std::shared_ptr<std::vector<uint8_t>> terminal_hidden_storage;
        std::shared_ptr<std::vector<uint8_t>> terminal_logits_storage;
        std::shared_ptr<std::vector<uint8_t>> model_runtime_state_storage;

        /**
         * Backend-pinned RAM owners used by GPU prefix archives.
         *
         * The raw payload members above always point into either these owners
         * or the corresponding vector owner. Keeping ownership separate from
         * addressing lets disk code and orchestrator code consume one uniform
         * byte contract without guessing whether a pointer is pageable.
         */
        std::shared_ptr<void> pinned_kv_storage;
        std::shared_ptr<void> pinned_hybrid_storage;
        std::shared_ptr<void> pinned_mtp_storage;
        std::shared_ptr<void> pinned_terminal_hidden_storage;
        std::shared_ptr<void> pinned_terminal_logits_storage;

        /**
         * Completion state shared with every pinned owner above.
         *
         * This member is declared after the pinned owners deliberately. During
         * ordinary handle destruction its reference is released first, while
         * each pinned owner's deleter retains the state long enough to wait
         * before freeing the corresponding DMA destination.
         */
        std::shared_ptr<PrefixPayloadReadiness> payload_readiness;

        /**
         * Event permanently paired with a transient live-checkpoint pool slot.
         *
         * Durable RAM/device-hot prefix records use payload_readiness above.
         * Live MTP transaction checkpoints instead reuse one preallocated event
         * per reusable storage slot, eliminating event creation from decode.
         * Copies of this handle retain the event alongside the slot's storage.
         */
        std::shared_ptr<void> live_checkpoint_ready_event;

        std::shared_ptr<TensorBase> device_kv_storage;
        std::shared_ptr<TensorBase> device_hybrid_storage;
        std::shared_ptr<TensorBase> device_mtp_storage;
        std::shared_ptr<TensorBase> device_terminal_hidden_storage;
        std::shared_ptr<TensorBase> device_terminal_logits_storage;

        /**
         * Pure-device owners used by the optional device-hot prefix tier.
         *
         * These allocations contain the already serialized byte layout. They
         * intentionally have no TensorBase host mirror: hot-tier harvest forks
         * the device staging bytes directly into these allocations, and hot
         * restore imports them directly into live GPU state. The older typed
         * tensor members above remain for live checkpoint objects whose graph
         * contracts require TensorBase identity.
         */
        std::shared_ptr<void> device_kv_allocation;
        std::shared_ptr<void> device_hybrid_allocation;
        std::shared_ptr<void> device_mtp_allocation;
        std::shared_ptr<void> device_terminal_hidden_allocation;
        std::shared_ptr<void> device_terminal_logits_allocation;

        bool valid() const { return key.valid() && total_bytes > 0; }
        size_t kvKBytes() const { return static_cast<size_t>(layout.fa_layers) * layout.bytes_per_fa_layer_k; }
        size_t kvVBytes() const { return static_cast<size_t>(layout.fa_layers) * layout.bytes_per_fa_layer_v; }
        size_t kvBytes() const { return layout.faKVBytes(); }
        size_t hybridBytes() const
        {
            return layout.includes_hybrid_state ? layout.hybrid_state_bytes : 0;
        }
        size_t terminalHiddenBytes() const
        {
            return layout.includes_terminal_hidden ? layout.terminal_hidden_bytes : 0;
        }
        size_t terminalLogitsBytes() const
        {
            return layout.includes_terminal_logits ? layout.terminal_logits_bytes : 0;
        }
        size_t mtpKBytes() const;
        size_t mtpVBytes() const;
        uint8_t *kvKData();
        uint8_t *kvVData();
        const uint8_t *kvKData() const;
        const uint8_t *kvVData() const;
        uint8_t *mtpKData();
        uint8_t *mtpVData();
        const uint8_t *mtpKData() const;
        const uint8_t *mtpVData() const;

        /** @brief Return the base of serialized full-attention K bytes in VRAM. */
        uint8_t *deviceKVKData();
        const uint8_t *deviceKVKData() const;

        /** @brief Return the base of serialized full-attention V bytes in VRAM. */
        uint8_t *deviceKVVData();
        const uint8_t *deviceKVVData() const;

        /** @brief Return the base of serialized shifted-MTP K bytes in VRAM. */
        uint8_t *deviceMTPKData();
        const uint8_t *deviceMTPKData() const;

        /** @brief Return the base of serialized shifted-MTP V bytes in VRAM. */
        uint8_t *deviceMTPVData();
        const uint8_t *deviceMTPVData() const;

        /** @brief Return the serialized recurrent-state payload in VRAM. */
        void *deviceHybridData();
        const void *deviceHybridData() const;

        /** @brief Return the archived terminal-hidden row in VRAM. */
        void *deviceTerminalHiddenData();
        const void *deviceTerminalHiddenData() const;

        /** @brief Return the archived terminal-logits row in VRAM. */
        void *deviceTerminalLogitsData();
        const void *deviceTerminalLogitsData() const;

        /** @brief Queue archive readiness on a GPU restore stream. */
        bool waitForPayloadOnStream(void *consumer_stream) const
        {
            return !payload_readiness ||
                   payload_readiness->waitOnStream(consumer_stream);
        }

        /** @brief Wait before reading serialized bytes on the host. */
        bool waitForPayloadOnHost() const
        {
            return !payload_readiness ||
                   payload_readiness->waitOnHost();
        }
    };

    class IPrefixStorageBackend
    {
    public:
        virtual ~IPrefixStorageBackend() = default;
        virtual bool canStore(size_t bytes) const = 0;
        virtual PrefixBlockHandle allocate(const PrefixCacheKey &key,
                                           const PrefixPayloadLayout &layout) = 0;
        virtual bool release(const PrefixBlockHandle &handle) = 0;
    };

} // namespace llaminar2
