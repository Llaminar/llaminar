#pragma once

#include <cstddef>
#include <string>

namespace llaminar2
{

    /**
     * @brief Interface for publishing weight load progress to a cross-rank aggregator.
     *
     * This decouples WeightLoadProgress (MPI-free) from the MPI-based aggregator
     * implementation, avoiding <mpi.h> leaking into non-MPI compilation units.
     */
    class IProgressPublisher
    {
    public:
        virtual ~IProgressPublisher() = default;

        /// Publish a new device registration. Returns local slot index.
        virtual int publishDevice(const std::string &label, size_t total_bytes) = 0;

        /// Publish progress update (local memory write, zero overhead).
        virtual void publishProgress(int local_device_idx, size_t bytes_loaded) = 0;

        /// Publish device finished.
        virtual void publishFinished(int local_device_idx) = 0;
    };

} // namespace llaminar2
