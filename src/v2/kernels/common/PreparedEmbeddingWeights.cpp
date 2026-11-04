/**
 * @file PreparedEmbeddingWeights.cpp
 * @brief Implementation of PreparedEmbeddingWeights lifecycle
 *
 * Uses IBackend abstraction for GPU memory management — no raw CUDA/HIP calls.
 * An adopted final CPU tensor keeps its own storage and admission lifetime;
 * moving or destroying a prepared alias never backend-frees its borrowed pointer.
 */

#include "PreparedEmbeddingWeights.h"
#include "../../backends/BackendManager.h"
#include "../../planning/PhysicalMemoryAuthority.h"
#include "../../utils/Logger.h"
#include "../../tensors/Tensors.h"

#include <limits>
#include <stdexcept>

namespace llaminar2
{

    PreparedEmbeddingWeights::PreparedEmbeddingWeights() = default;

    void PreparedEmbeddingWeights::bindPhysicalMemoryLease(
        PhysicalMemoryAllocationLease lease)
    {
        if (!lease.valid() || lease.bytes() != layout.byte_size)
        {
            throw std::invalid_argument(
                "Prepared embedding requires a valid exact-size physical-memory lease");
        }
        if (physical_memory_lease_ || final_cpu_tensor_)
        {
            throw std::logic_error(
                "Prepared embedding backing already owns its physical-memory lease");
        }
        physical_memory_lease_ =
            std::make_unique<PhysicalMemoryAllocationLease>(
                std::move(lease));
    }

    void PreparedEmbeddingWeights::bindFinalCPUStorage(std::shared_ptr<const TensorBase> tensor)
    {
        if (device_data || physical_memory_lease_ || final_cpu_tensor_)
            throw std::logic_error("Prepared embedding backing is already bound");
        if (!tensor || !device_id.is_cpu() || tensor->home_device() != device_id ||
            tensor->current_device().has_value() || !tensor->isHostAuthoritative() ||
            tensor->is_view() || tensor->is_mmap_data() || tensor->isMapped() ||
            tensor->transferStorageOwner() != tensor.get() || !tensor->raw_data() ||
            tensor->shape().size() != 2u || !tensor->rows() ||
            tensor->cols() != static_cast<size_t>(d_model) || tensor->rows() != vocab_size ||
            !NativeEmbeddingLayout::supports(tensor->native_type()) ||
            layout != NativeEmbeddingLayout::forSource(tensor->native_type(), vocab_size, d_model) ||
            tensor->size_bytes() != layout.byte_size ||
            vocab_offset > total_vocab || vocab_size > total_vocab - vocab_offset)
            throw std::invalid_argument("Final CPU embedding requires matching owning native storage and vocabulary");
        final_cpu_tensor_ = std::move(tensor);
        // All embedding consumers treat this pointer as immutable native bytes.
        // Its tensor owns deallocation, so retirement must never backend-free it.
        device_data = const_cast<void *>(final_cpu_tensor_->raw_data());
    }

    void PreparedEmbeddingWeights::releaseDeviceData() noexcept
    {
        if (final_cpu_tensor_)
        {
            device_data = nullptr;
            layout = {};
            iq_tables = {};
            final_cpu_tensor_.reset();
            return;
        }
        if (!device_data)
        {
            physical_memory_lease_.reset();
            return;
        }

        IBackend *backend = getBackendFor(device_id);
        if (backend)
        {
            backend->free(device_data, device_id.ordinal);
        }
        else
        {
            LOG_WARN("[PreparedEmbeddingWeights] No backend for device " << device_id
                                                                         << " — leaking " << layout.byte_size << " bytes");
        }

        device_data = nullptr;
        layout = {};
        /* Release the ledger claim only after the backend allocation is gone. */
        physical_memory_lease_.reset();
    }

    PreparedEmbeddingWeights::~PreparedEmbeddingWeights()
    {
        releaseDeviceData();
    }

    PreparedEmbeddingWeights::PreparedEmbeddingWeights(PreparedEmbeddingWeights &&other) noexcept
        : device_data(other.device_data),
          layout(other.layout),
          iq_tables(other.iq_tables),
          vocab_size(other.vocab_size),
          vocab_offset(other.vocab_offset),
          total_vocab(other.total_vocab),
          d_model(other.d_model),
          device_id(other.device_id),
          physical_memory_lease_(
              std::move(other.physical_memory_lease_)),
          final_cpu_tensor_(std::move(other.final_cpu_tensor_))
    {
        other.device_data = nullptr;
        other.layout = {};
        other.iq_tables = {};
    }

    PreparedEmbeddingWeights &PreparedEmbeddingWeights::operator=(PreparedEmbeddingWeights &&other) noexcept
    {
        if (this != &other)
        {
            releaseDeviceData();
            device_data = other.device_data;
            layout = other.layout;
            iq_tables = other.iq_tables;
            vocab_size = other.vocab_size;
            vocab_offset = other.vocab_offset;
            total_vocab = other.total_vocab;
            d_model = other.d_model;
            device_id = other.device_id;
            physical_memory_lease_ =
                std::move(other.physical_memory_lease_);
            final_cpu_tensor_ = std::move(other.final_cpu_tensor_);
            other.device_data = nullptr;
            other.layout = {};
            other.iq_tables = {};
        }
        return *this;
    }

} // namespace llaminar2
