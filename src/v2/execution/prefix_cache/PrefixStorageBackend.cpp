/**
 * @file PrefixStorageBackend.cpp
 * @brief Prefix payload addressing and asynchronous readiness implementation.
 */

#include "execution/prefix_cache/PrefixStorageBackend.h"

#include "backends/BackendManager.h"
#include "tensors/TensorClasses.h"

namespace llaminar2
{
    bool PrefixPayloadReadiness::prepare(
        std::shared_ptr<void> ready_event,
        DeviceId producer_device,
        void *producer_stream)
    {
        if (!ready_event ||
            !producer_device.is_gpu() ||
            !producer_stream ||
            ready_event_ ||
            published_)
        {
            return false;
        }

        ready_event_ = std::move(ready_event);
        producer_device_ = producer_device;
        producer_stream_ = producer_stream;
        host_wait_complete_.store(false, std::memory_order_release);
        return true;
    }

    bool PrefixPayloadReadiness::publishRecorded()
    {
        if (!ready_event_ ||
            !producer_device_.is_gpu() ||
            !producer_stream_ ||
            published_)
        {
            return false;
        }

        published_ = true;
        return true;
    }

    bool PrefixPayloadReadiness::waitOnStream(void *consumer_stream) const
    {
        if (!published_)
            return true;
        if (!consumer_stream || !ready_event_ || !producer_device_.is_gpu())
            return false;
        if (consumer_stream == producer_stream_)
            return true;

        IBackend *backend = getBackendFor(producer_device_);
        return backend &&
               backend->streamWaitEvent(
                   consumer_stream,
                   ready_event_.get(),
                   producer_device_.gpu_ordinal());
    }

    bool PrefixPayloadReadiness::waitOnHost() const
    {
        if (!published_ ||
            host_wait_complete_.load(std::memory_order_acquire))
        {
            return true;
        }
        if (!ready_event_ || !producer_device_.is_gpu())
            return false;

        IBackend *backend = getBackendFor(producer_device_);
        if (!backend ||
            !backend->waitForEvent(
                ready_event_.get(),
                producer_device_.gpu_ordinal()))
        {
            return false;
        }
        host_wait_complete_.store(true, std::memory_order_release);
        return true;
    }

    size_t PrefixBlockHandle::mtpKBytes() const
    {
        if (layout.mtp_layers > 0 && layout.bytes_per_mtp_layer_k > 0)
        {
            return static_cast<size_t>(layout.mtp_layers) * layout.bytes_per_mtp_layer_k;
        }
        return layout.includes_mtp_state ? layout.mtpKVBytes() / 2 : 0;
    }

    size_t PrefixBlockHandle::mtpVBytes() const
    {
        const size_t total = layout.includes_mtp_state ? layout.mtpKVBytes() : 0;
        const size_t k_bytes = mtpKBytes();
        return total > k_bytes ? total - k_bytes : 0;
    }

    uint8_t *PrefixBlockHandle::kvKData()
    {
        return static_cast<uint8_t *>(kv_payload);
    }

    uint8_t *PrefixBlockHandle::kvVData()
    {
        auto *base = static_cast<uint8_t *>(kv_payload);
        if (!base)
        {
            return nullptr;
        }
        return base + kvKBytes();
    }

    const uint8_t *PrefixBlockHandle::kvKData() const
    {
        return static_cast<const uint8_t *>(kv_payload);
    }

    const uint8_t *PrefixBlockHandle::kvVData() const
    {
        const auto *base = static_cast<const uint8_t *>(kv_payload);
        if (!base)
        {
            return nullptr;
        }
        return base + kvKBytes();
    }

    uint8_t *PrefixBlockHandle::mtpKData()
    {
        return static_cast<uint8_t *>(mtp_payload);
    }

    uint8_t *PrefixBlockHandle::mtpVData()
    {
        auto *base = static_cast<uint8_t *>(mtp_payload);
        if (!base)
        {
            return nullptr;
        }
        return base + mtpKBytes();
    }

    const uint8_t *PrefixBlockHandle::mtpKData() const
    {
        return static_cast<const uint8_t *>(mtp_payload);
    }

    const uint8_t *PrefixBlockHandle::mtpVData() const
    {
        const auto *base = static_cast<const uint8_t *>(mtp_payload);
        if (!base)
        {
            return nullptr;
        }
        return base + mtpKBytes();
    }

    uint8_t *PrefixBlockHandle::deviceKVKData()
    {
        if (device_kv_allocation)
            return static_cast<uint8_t *>(device_kv_allocation.get());
        return device_kv_storage
                   ? static_cast<uint8_t *>(device_kv_storage->gpu_data_ptr())
                   : nullptr;
    }

    const uint8_t *PrefixBlockHandle::deviceKVKData() const
    {
        if (device_kv_allocation)
            return static_cast<const uint8_t *>(device_kv_allocation.get());
        return device_kv_storage
                   ? static_cast<const uint8_t *>(device_kv_storage->gpu_data_ptr())
                   : nullptr;
    }

    uint8_t *PrefixBlockHandle::deviceKVVData()
    {
        auto *base = deviceKVKData();
        return base ? base + kvKBytes() : nullptr;
    }

    const uint8_t *PrefixBlockHandle::deviceKVVData() const
    {
        const auto *base = deviceKVKData();
        return base ? base + kvKBytes() : nullptr;
    }

    uint8_t *PrefixBlockHandle::deviceMTPKData()
    {
        if (device_mtp_allocation)
            return static_cast<uint8_t *>(device_mtp_allocation.get());
        return device_mtp_storage
                   ? static_cast<uint8_t *>(device_mtp_storage->gpu_data_ptr())
                   : nullptr;
    }

    const uint8_t *PrefixBlockHandle::deviceMTPKData() const
    {
        if (device_mtp_allocation)
            return static_cast<const uint8_t *>(device_mtp_allocation.get());
        return device_mtp_storage
                   ? static_cast<const uint8_t *>(device_mtp_storage->gpu_data_ptr())
                   : nullptr;
    }

    uint8_t *PrefixBlockHandle::deviceMTPVData()
    {
        auto *base = deviceMTPKData();
        return base ? base + mtpKBytes() : nullptr;
    }

    const uint8_t *PrefixBlockHandle::deviceMTPVData() const
    {
        const auto *base = deviceMTPKData();
        return base ? base + mtpKBytes() : nullptr;
    }

    void *PrefixBlockHandle::deviceHybridData()
    {
        if (device_hybrid_allocation)
            return device_hybrid_allocation.get();
        return device_hybrid_storage
                   ? device_hybrid_storage->gpu_data_ptr()
                   : nullptr;
    }

    const void *PrefixBlockHandle::deviceHybridData() const
    {
        if (device_hybrid_allocation)
            return device_hybrid_allocation.get();
        return device_hybrid_storage
                   ? device_hybrid_storage->gpu_data_ptr()
                   : nullptr;
    }

    void *PrefixBlockHandle::deviceTerminalHiddenData()
    {
        if (device_terminal_hidden_allocation)
            return device_terminal_hidden_allocation.get();
        return device_terminal_hidden_storage
                   ? device_terminal_hidden_storage->gpu_data_ptr()
                   : nullptr;
    }

    const void *PrefixBlockHandle::deviceTerminalHiddenData() const
    {
        if (device_terminal_hidden_allocation)
            return device_terminal_hidden_allocation.get();
        return device_terminal_hidden_storage
                   ? device_terminal_hidden_storage->gpu_data_ptr()
                   : nullptr;
    }

    void *PrefixBlockHandle::deviceTerminalLogitsData()
    {
        if (device_terminal_logits_allocation)
            return device_terminal_logits_allocation.get();
        return device_terminal_logits_storage
                   ? device_terminal_logits_storage->gpu_data_ptr()
                   : nullptr;
    }

    const void *PrefixBlockHandle::deviceTerminalLogitsData() const
    {
        if (device_terminal_logits_allocation)
            return device_terminal_logits_allocation.get();
        return device_terminal_logits_storage
                   ? device_terminal_logits_storage->gpu_data_ptr()
                   : nullptr;
    }

} // namespace llaminar2
