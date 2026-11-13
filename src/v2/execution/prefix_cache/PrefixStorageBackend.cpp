/**
 * @file PrefixStorageBackend.cpp
 * @brief Prefix payload addressing and asynchronous readiness implementation.
 *
 * Runtime extensions share owned byte views across RAM and device-hot tiers.
 * Retired pinned ranges query their exact published producer event; prepared
 * but unpublished edges cannot authorize reuse of an asynchronous destination.
 */

#include "execution/prefix_cache/PrefixStorageBackend.h"

#include "backends/BackendManager.h"
#include "tensors/TensorClasses.h"

#include <limits>
#include <stdexcept>
#include <cerrno>
#include <sys/random.h>

namespace llaminar2
{
    PrefixPayloadIdentity PrefixPayloadIdentity::fresh()
    {
        // Native entropy supplies only sixteen metadata bytes. This must never
        // inspect cache contents or share the model's sampling RNG state.
        std::array<uint64_t, 2> words{};
        auto *cursor = reinterpret_cast<uint8_t *>(words.data());
        size_t remaining = sizeof(words);
        while (remaining > 0u)
        {
            const auto count = ::getrandom(cursor, remaining, 0);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("prefix payload identity generation failed");
            cursor += count;
            remaining -= static_cast<size_t>(count);
        }
        PrefixPayloadIdentity identity{words[0], words[1]};
        if (!identity.valid()) throw std::runtime_error("prefix payload identity is empty");
        return identity;
    }

    PrefixPayloadLayout prefixReadLayout(PrefixPayloadLayout layout, PrefixPayloadReadSet read_set)
    {
        if (read_set == PrefixPayloadReadSet::SequenceRows)
        {
            layout.includes_hybrid_state = false;
            layout.main_auxiliary.includes_endpoint = false;
            layout.mtp_auxiliary.includes_endpoint = false;
            layout.includes_terminal_hidden = false;
            layout.includes_terminal_logits = false;
        }
        return layout;
    }
    PrefixPayloadAllocationPlan::PrefixPayloadAllocationPlan(std::array<size_t, 7> sections)
        : sections_(sections)
    {
        for (const size_t bytes : sections_)
        {
            if (bytes > std::numeric_limits<size_t>::max() - total_bytes_)
                throw std::overflow_error("prefix section allocation BOM overflow");
            total_bytes_ += bytes;
        }
    }

    PrefixPayloadAllocationPlan PrefixPayloadAllocationPlan::contiguous(size_t bytes)
    { return PrefixPayloadAllocationPlan({bytes, 0u, 0u, 0u, 0u, 0u, 0u}); }

    PrefixPayloadAllocationPlan PrefixPayloadAllocationPlan::archive(
        const PrefixPayloadLayout &layout, size_t runtime_bytes)
    {
        const size_t limit = std::numeric_limits<size_t>::max();
        if (layout.fa_layers < 0 || layout.mtp_layers < 0 ||
            layout.bytes_per_fa_layer_k > limit - layout.bytes_per_fa_layer_v ||
            layout.bytes_per_mtp_layer_k > limit - layout.bytes_per_mtp_layer_v)
            throw std::invalid_argument("invalid prefix section allocation geometry");
        const size_t fa_stride = layout.bytes_per_fa_layer_k + layout.bytes_per_fa_layer_v;
        const size_t mtp_stride = layout.bytes_per_mtp_layer_k + layout.bytes_per_mtp_layer_v;
        if ((layout.fa_layers > 0 && fa_stride > limit / static_cast<size_t>(layout.fa_layers)) ||
            (layout.mtp_layers > 0 && mtp_stride > limit / static_cast<size_t>(layout.mtp_layers)))
            throw std::overflow_error("prefix section layer allocation overflow");
        return PrefixPayloadAllocationPlan({static_cast<size_t>(layout.fa_layers) * fa_stride,
            layout.includes_hybrid_state ? layout.hybrid_state_bytes : 0u,
            layout.includes_mtp_state ? layout.mtpKVBytes() : 0u,
            layout.includes_terminal_hidden ? layout.terminal_hidden_bytes : 0u,
            layout.includes_terminal_logits ? layout.terminal_logits_bytes : 0u,
            layout.modelAuxiliaryBytes(), runtime_bytes});
    }

    PrefixPayloadReadLease PrefixPayloadReadLease::wholeArchive(PrefixBlockHandle source)
    { return PrefixPayloadReadLease(std::move(source)); }

    PrefixPayloadReadLease PrefixPayloadReadLease::sequenceRows(PrefixBlockHandle source)
    {
        // These sections are not read by a nonterminal row import. Drop their
        // physical owners as well as addresses; a rich archive remains intact
        // in the cache until ordinary eviction retires that independent owner.
        source.hybrid_storage.reset();
        source.terminal_hidden_storage.reset();
        source.terminal_logits_storage.reset();
        source.model_runtime_state_storage.reset();
        source.pinned_hybrid_storage.reset();
        source.pinned_terminal_hidden_storage.reset();
        source.pinned_terminal_logits_storage.reset();
        source.device_hybrid_storage.reset();
        source.device_terminal_hidden_storage.reset();
        source.device_terminal_logits_storage.reset();
        source.device_hybrid_allocation.reset();
        source.device_terminal_hidden_allocation.reset();
        source.device_terminal_logits_allocation.reset();
        source.ram_section_memory_leases[static_cast<size_t>(PrefixPayloadSection::RecurrentState)].reset();
        source.ram_section_memory_leases[static_cast<size_t>(PrefixPayloadSection::TerminalHidden)].reset();
        source.ram_section_memory_leases[static_cast<size_t>(PrefixPayloadSection::TerminalLogits)].reset();
        source.ram_runtime_state_memory_lease.reset();
        source.hybrid_payload = source.terminal_hidden = source.terminal_logits = nullptr;
        source.has_hybrid_state = source.has_terminal_hidden = source.has_terminal_logits = false;
        source.has_model_runtime_state = false;
        source.layout.main_auxiliary.includes_endpoint = false;
        source.layout.mtp_auxiliary.includes_endpoint = false;
        if (source.layout.modelAuxiliaryBytes() == 0)
        {
            // A PLE-only endpoint contributes no row-block data. Its read
            // lease must not prolong an unused physical owner or claim.
            source.model_auxiliary_storage.reset();
            source.pinned_model_auxiliary_storage.reset();
            source.device_model_auxiliary_allocation.reset();
            source.model_auxiliary_payload = nullptr;
            source.ram_section_memory_leases[static_cast<size_t>(PrefixPayloadSection::ModelAuxiliary)].reset();
        }
        source.layout.includes_hybrid_state = source.layout.includes_terminal_hidden =
            source.layout.includes_terminal_logits = false;
        source.total_bytes = PrefixPayloadAllocationPlan::archive(source.layout).totalBytes();
        return PrefixPayloadReadLease(std::move(source));
    }

    PrefixRuntimeStateStorage::PrefixRuntimeStateStorage(
        std::shared_ptr<std::vector<uint8_t>> storage)
    {
        if (!storage || storage->empty())
            throw std::invalid_argument("Prefix runtime state requires owned, nonempty bytes");
        size_ = storage->size();
        storage_ = std::shared_ptr<void>(storage, storage->data());
    }

    PrefixRuntimeStateStorage::PrefixRuntimeStateStorage(
        std::shared_ptr<void> storage, size_t bytes)
        : storage_(std::move(storage)), size_(bytes)
    {
        if (!storage_ || size_ == 0u)
            throw std::invalid_argument("Prefix runtime state requires an owned, nonempty range");
    }

    namespace
    {
        /** @brief Mathematical compact subsection; physical allocation ownership stays shared. */
        enum class AuxiliaryPart { MainBlocks, MTPBlocks, MainEndpoint, MTPEndpoint };
        /** @brief Resolve only the live subsection, keeping row-only reads independent of endpoint sizes. */
        std::pair<size_t, size_t> auxiliaryRange(const PrefixPayloadLayout &layout, AuxiliaryPart part, int layer)
        {
            (void)layout.modelAuxiliaryBytes();
            const auto main_blocks = layout.main_auxiliary.blockBytes(layout.fa_layers);
            const auto mtp_blocks = layout.mtp_auxiliary.blockBytes(layout.mtp_layers);
            if (part == AuxiliaryPart::MainBlocks || part == AuxiliaryPart::MTPBlocks)
            {
                const bool main = part == AuxiliaryPart::MainBlocks;
                const int layers = main ? layout.fa_layers : layout.mtp_layers;
                const auto stride = main ? layout.main_auxiliary.bytes_per_attention_layer
                                         : layout.mtp_auxiliary.bytes_per_attention_layer;
                if (layer < 0 || layer >= layers) throw std::out_of_range("prefix auxiliary FA ordinal is not owned");
                return {(main ? 0 : main_blocks) + static_cast<size_t>(layer) * stride, stride};
            }
            if (part == AuxiliaryPart::MainEndpoint)
                return {main_blocks + mtp_blocks, layout.main_auxiliary.endpointBytes()};
            return {main_blocks + mtp_blocks + layout.main_auxiliary.endpointBytes(), layout.mtp_auxiliary.endpointBytes()};
        }
        /** @brief Project an exact live range; absent storage/empty sections remain null views. */
        const uint8_t *auxiliarySubview(const void *base, const PrefixPayloadLayout &layout, AuxiliaryPart part, int layer = 0)
        {
            const auto [offset, bytes] = auxiliaryRange(layout, part, layer);
            return base && bytes ? static_cast<const uint8_t *>(base) + offset : nullptr;
        }
    }

    const uint8_t *PrefixBlockHandle::mainAuxiliaryBlockData(int local_fa) const
    { return auxiliarySubview(model_auxiliary_payload, layout, AuxiliaryPart::MainBlocks, local_fa); }
    uint8_t *PrefixBlockHandle::mainAuxiliaryBlockData(int local_fa)
    { return const_cast<uint8_t *>(static_cast<const PrefixBlockHandle &>(*this).mainAuxiliaryBlockData(local_fa)); }

    const uint8_t *PrefixBlockHandle::deviceMainAuxiliaryBlockData(int local_fa) const
    { return auxiliarySubview(deviceModelAuxiliaryData(), layout, AuxiliaryPart::MainBlocks, local_fa); }
    uint8_t *PrefixBlockHandle::deviceMainAuxiliaryBlockData(int local_fa)
    { return const_cast<uint8_t *>(static_cast<const PrefixBlockHandle &>(*this).deviceMainAuxiliaryBlockData(local_fa)); }

    const uint8_t *PrefixBlockHandle::mtpAuxiliaryBlockData(int local_fa) const
    { return auxiliarySubview(model_auxiliary_payload, layout, AuxiliaryPart::MTPBlocks, local_fa); }
    uint8_t *PrefixBlockHandle::mtpAuxiliaryBlockData(int local_fa)
    { return const_cast<uint8_t *>(static_cast<const PrefixBlockHandle &>(*this).mtpAuxiliaryBlockData(local_fa)); }

    const uint8_t *PrefixBlockHandle::deviceMTPAuxiliaryBlockData(int local_fa) const
    { return auxiliarySubview(deviceModelAuxiliaryData(), layout, AuxiliaryPart::MTPBlocks, local_fa); }
    uint8_t *PrefixBlockHandle::deviceMTPAuxiliaryBlockData(int local_fa)
    { return const_cast<uint8_t *>(static_cast<const PrefixBlockHandle &>(*this).deviceMTPAuxiliaryBlockData(local_fa)); }

    const uint8_t *PrefixBlockHandle::mainAuxiliaryEndpointData() const
    { return auxiliarySubview(model_auxiliary_payload, layout, AuxiliaryPart::MainEndpoint); }
    uint8_t *PrefixBlockHandle::mainAuxiliaryEndpointData()
    { return const_cast<uint8_t *>(static_cast<const PrefixBlockHandle &>(*this).mainAuxiliaryEndpointData()); }

    const uint8_t *PrefixBlockHandle::deviceMainAuxiliaryEndpointData() const
    { return auxiliarySubview(deviceModelAuxiliaryData(), layout, AuxiliaryPart::MainEndpoint); }
    uint8_t *PrefixBlockHandle::deviceMainAuxiliaryEndpointData()
    { return const_cast<uint8_t *>(static_cast<const PrefixBlockHandle &>(*this).deviceMainAuxiliaryEndpointData()); }

    const uint8_t *PrefixBlockHandle::mtpAuxiliaryEndpointData() const
    { return auxiliarySubview(model_auxiliary_payload, layout, AuxiliaryPart::MTPEndpoint); }
    uint8_t *PrefixBlockHandle::mtpAuxiliaryEndpointData()
    { return const_cast<uint8_t *>(static_cast<const PrefixBlockHandle &>(*this).mtpAuxiliaryEndpointData()); }

    const uint8_t *PrefixBlockHandle::deviceMTPAuxiliaryEndpointData() const
    { return auxiliarySubview(deviceModelAuxiliaryData(), layout, AuxiliaryPart::MTPEndpoint); }
    uint8_t *PrefixBlockHandle::deviceMTPAuxiliaryEndpointData()
    { return const_cast<uint8_t *>(static_cast<const PrefixBlockHandle &>(*this).deviceMTPAuxiliaryEndpointData()); }

    bool PrefixPayloadReadiness::prepare(
        std::shared_ptr<void> ready_event,
        DeviceId producer_device,
        void *producer_stream, IBackend &producer_backend)
    {
        if (producer_backend.backendDeviceType() != producer_device.type ||
            producer_device.ordinal < 0 ||
            producer_device.ordinal >= producer_backend.deviceCount() || !ready_event ||
            !producer_device.is_gpu() ||
            !producer_stream ||
            ready_event_ ||
            published_)
        {
            return false;
        }

        producer_backend_ = &producer_backend;
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
            return !prepared();
        if (!consumer_stream || !ready_event_ || !producer_device_.is_gpu())
            return false;
        if (consumer_stream == producer_stream_)
            return true;

        IBackend *backend = producer_backend_;
        return backend &&
               backend->streamWaitEvent(
                   consumer_stream,
                   ready_event_.get(),
                   producer_device_.gpu_ordinal());
    }

    bool PrefixPayloadReadiness::waitOnHost() const
    {
        if (!published_) return !prepared();
        if (host_wait_complete_.load(std::memory_order_acquire)) return true;
        if (!ready_event_ || !producer_device_.is_gpu())
            return false;

        IBackend *backend = producer_backend_;
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

    bool PrefixPayloadReadiness::queryComplete(bool *ready) const
    {
        if (!ready)
            return false;
        *ready = false;
        if (!published_)
        {
            // Disk hydration has no GPU producer and needs no event. Once
            // preparation starts, an absent publication is an invalid edge,
            // never proof that a retired DMA destination can be overwritten.
            *ready = !ready_event_;
            return *ready;
        }
        if (host_wait_complete_.load(std::memory_order_acquire))
        {
            *ready = true;
            return true;
        }
        if (!ready_event_ || !producer_device_.is_gpu())
            return false;
        IBackend *backend = producer_backend_;
        if (!backend || !backend->queryEvent(
                ready_event_.get(), producer_device_.gpu_ordinal(), ready))
            return false;
        if (*ready)
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
