/**
 * @file PreparedEmbeddingWeights.cpp
 * @brief Implementation of PreparedEmbeddingWeights lifecycle
 *
 * Uses IBackend abstraction for GPU memory management — no raw CUDA/HIP calls.
 */

#include "PreparedEmbeddingWeights.h"
#include "../../backends/BackendManager.h"
#include "../../planning/PhysicalMemoryAuthority.h"
#include "../../utils/Logger.h"

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
        if (physical_memory_lease_)
        {
            throw std::logic_error(
                "Prepared embedding physical-memory lease is already bound");
        }
        physical_memory_lease_ =
            std::make_unique<PhysicalMemoryAllocationLease>(
                std::move(lease));
    }

    void PreparedEmbeddingWeights::releaseDeviceData() noexcept
    {
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
              std::move(other.physical_memory_lease_))
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
            other.device_data = nullptr;
            other.layout = {};
            other.iq_tables = {};
        }
        return *this;
    }

} // namespace llaminar2
