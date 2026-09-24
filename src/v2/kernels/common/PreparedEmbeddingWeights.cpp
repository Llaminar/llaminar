/**
 * @file PreparedEmbeddingWeights.cpp
 * @brief Implementation of PreparedEmbeddingWeights lifecycle
 *
 * Uses IBackend abstraction for GPU memory management — no raw CUDA/HIP calls.
 */

#include "PreparedEmbeddingWeights.h"
#include "EmbedQ8Block.h"
#include "../../backends/BackendManager.h"
#include "../../planning/PhysicalMemoryAuthority.h"
#include "../../utils/Logger.h"

#include <limits>
#include <stdexcept>

namespace llaminar2
{

    PreparedEmbeddingWeights::PreparedEmbeddingWeights() = default;

    size_t PreparedEmbeddingWeights::allocationBytes(
        size_t vocab_size,
        int d_model)
    {
        if (d_model <= 0)
        {
            throw std::invalid_argument(
                "Prepared embedding allocation requires a positive model width");
        }
        const size_t width = static_cast<size_t>(d_model);
        const size_t blocks_per_row = (width + 31u) / 32u;
        if (blocks_per_row != 0u &&
            vocab_size > std::numeric_limits<size_t>::max() /
                             blocks_per_row)
        {
            throw std::overflow_error(
                "Prepared embedding block count overflows size_t");
        }
        const size_t blocks = vocab_size * blocks_per_row;
        if (blocks > std::numeric_limits<size_t>::max() /
                         sizeof(EmbedQ8Block))
        {
            throw std::overflow_error(
                "Prepared embedding byte size overflows size_t");
        }
        return blocks * sizeof(EmbedQ8Block);
    }

    void PreparedEmbeddingWeights::bindPhysicalMemoryLease(
        PhysicalMemoryAllocationLease lease)
    {
        if (!lease.valid() || lease.bytes() != byte_size)
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
                                                                         << " — leaking " << byte_size << " bytes");
        }

        device_data = nullptr;
        byte_size = 0;
        /* Release the ledger claim only after the backend allocation is gone. */
        physical_memory_lease_.reset();
    }

    PreparedEmbeddingWeights::~PreparedEmbeddingWeights()
    {
        releaseDeviceData();
    }

    PreparedEmbeddingWeights::PreparedEmbeddingWeights(PreparedEmbeddingWeights &&other) noexcept
        : device_data(other.device_data),
          byte_size(other.byte_size),
          blocks_per_row(other.blocks_per_row),
          vocab_size(other.vocab_size),
          vocab_offset(other.vocab_offset),
          total_vocab(other.total_vocab),
          d_model(other.d_model),
          device_id(other.device_id),
          physical_memory_lease_(
              std::move(other.physical_memory_lease_))
    {
        other.device_data = nullptr;
        other.byte_size = 0;
    }

    PreparedEmbeddingWeights &PreparedEmbeddingWeights::operator=(PreparedEmbeddingWeights &&other) noexcept
    {
        if (this != &other)
        {
            releaseDeviceData();
            device_data = other.device_data;
            byte_size = other.byte_size;
            blocks_per_row = other.blocks_per_row;
            vocab_size = other.vocab_size;
            vocab_offset = other.vocab_offset;
            total_vocab = other.total_vocab;
            d_model = other.d_model;
            device_id = other.device_id;
            physical_memory_lease_ =
                std::move(other.physical_memory_lease_);
            other.device_data = nullptr;
            other.byte_size = 0;
        }
        return *this;
    }

} // namespace llaminar2
