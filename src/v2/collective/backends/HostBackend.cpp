/**
 * @file HostBackend.cpp
 * @brief CPU-based fallback collective backend implementation
 *
 * This backend provides heterogeneous GPU collective operations by
 * staging data through host memory.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "HostBackend.h"
#include "../../utils/Logger.h"
#include "../../backends/GPUDeviceContextPool.h"
#include <cstring>
#include <cstdlib>
#include <algorithm>

// Forward declare the copy functions that are implemented in runtime-specific files
namespace llaminar2
{
    namespace host_backend_detail
    {

#ifdef HAVE_CUDA
        bool cudaCopyToHost(void *host_dst, const void *device_src, int device_ordinal, size_t bytes, void *stream);
        bool cudaCopyFromHost(void *device_dst, const void *host_src, int device_ordinal, size_t bytes, void *stream);
        bool cudaHostRegisterBuffer(void *ptr, size_t size);
        void cudaHostUnregisterBuffer(void *ptr);
#endif

#ifdef HAVE_ROCM
        bool hipCopyToHost(void *host_dst, const void *device_src, int device_ordinal, size_t bytes, void *stream);
        bool hipCopyFromHost(void *device_dst, const void *host_src, int device_ordinal, size_t bytes, void *stream);
        bool hipHostRegisterBuffer(void *ptr, size_t size);
        void hipHostUnregisterBuffer(void *ptr);
#endif

    } // namespace host_backend_detail
} // namespace llaminar2

namespace llaminar2
{

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    HostBackend::HostBackend()
        : initialized_(false)
    {
        LOG_DEBUG("HostBackend: Created");
    }

    HostBackend::~HostBackend()
    {
        if (initialized_)
        {
            shutdown();
        }

        // Free staging buffer if allocated
        if (staging_buffer_)
        {
#ifdef HAVE_CUDA
            if (staging_registered_cuda_)
                host_backend_detail::cudaHostUnregisterBuffer(staging_buffer_);
#endif
#ifdef HAVE_ROCM
            if (staging_registered_rocm_)
                host_backend_detail::hipHostUnregisterBuffer(staging_buffer_);
#endif
            std::free(staging_buffer_);
            staging_buffer_ = nullptr;
            staging_buffer_size_ = 0;
            staging_registered_cuda_ = false;
            staging_registered_rocm_ = false;
        }
    }

    // =========================================================================
    // Capability Queries
    // =========================================================================

    bool HostBackend::supportsDirectTransfer(DeviceId src, DeviceId dst) const
    {
        // Direct transfer (no staging) only possible between CPU buffers
        // GPU transfers require copying through host memory
        return src.type == DeviceType::CPU && dst.type == DeviceType::CPU;
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool HostBackend::initialize(const DeviceGroup &group)
    {
        if (initialized_)
        {
            LOG_WARN("HostBackend::initialize: Already initialized, shutting down first");
            shutdown();
        }

        group_ = group;

        // Scan for CUDA and ROCm devices in the group
        has_cuda_ = false;
        has_rocm_ = false;

        for (const auto &device : group_.devices)
        {
            if (device.type == DeviceType::CUDA && !has_cuda_)
            {
                has_cuda_ = true;
                cuda_device_ = device;
            }
            else if (device.type == DeviceType::ROCm && !has_rocm_)
            {
                has_rocm_ = true;
                rocm_device_ = device;
            }
        }

        LOG_INFO("HostBackend: Initialized for group '" << group_.name
                                                        << "' with " << group_.size() << " device(s)"
                                                        << " (CUDA: " << (has_cuda_ ? "yes" : "no")
                                                        << ", ROCm: " << (has_rocm_ ? "yes" : "no") << ")");

        initialized_ = true;
        return true;
    }

    bool HostBackend::isInitialized() const
    {
        return initialized_;
    }

    void HostBackend::shutdown()
    {
        if (initialized_)
        {
            LOG_DEBUG("HostBackend: Shutting down");

            initialized_ = false;
            group_ = DeviceGroup{};
            has_cuda_ = false;
            has_rocm_ = false;
        }
    }

    // =========================================================================
    // Collective Operations
    // =========================================================================

    bool HostBackend::allreduce(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (!initialized_)
        {
            last_error_ = "HostBackend::allreduce: Backend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        LOG_DEBUG("HostBackend::allreduce: count=" << count
                                                   << " dtype=" << static_cast<int>(dtype)
                                                   << " op=" << toString(op));

        // For single-device group: no-op (data already "reduced")
        if (group_.size() <= 1)
        {
            LOG_DEBUG("HostBackend::allreduce: Single device, no-op");
            return true;
        }

        // For multi-device groups: stage through host and reduce
        size_t bytes = count * elementSize(dtype);

        // Ensure we have a staging buffer
        if (!ensureStagingBuffer(bytes))
        {
            last_error_ = "HostBackend::allreduce: Failed to allocate staging buffer";
            LOG_ERROR(last_error_);
            return false;
        }

        LOG_DEBUG("HostBackend::allreduce: Multi-device reduction for " << group_.size() << " devices");

        // Allocate a second staging buffer for reduction accumulator
        std::vector<uint8_t> accumulator(bytes, 0);
        bool first_device = true;

        // Phase 1: Copy each device's data to host and reduce
        for (const auto &device : group_.devices)
        {
            // Copy device buffer to staging
            if (!copyToHost(staging_buffer_, buffer, device, bytes))
            {
                last_error_ = "HostBackend::allreduce: Failed to copy from device " +
                              std::to_string(device.ordinal);
                LOG_ERROR(last_error_);
                return false;
            }

            if (first_device)
            {
                // First device: just copy to accumulator
                std::memcpy(accumulator.data(), staging_buffer_, bytes);
                first_device = false;
            }
            else
            {
                // Subsequent devices: reduce into accumulator
                reduceOnHost(accumulator.data(), staging_buffer_, count, dtype, op);
            }
        }

        // Phase 2: Broadcast reduced result back to all devices
        std::memcpy(staging_buffer_, accumulator.data(), bytes);

        for (const auto &device : group_.devices)
        {
            if (!copyFromHost(buffer, staging_buffer_, device, bytes))
            {
                last_error_ = "HostBackend::allreduce: Failed to copy to device " +
                              std::to_string(device.ordinal);
                LOG_ERROR(last_error_);
                return false;
            }
        }

        LOG_DEBUG("HostBackend::allreduce: Completed successfully");
        return true;
    }

    bool HostBackend::allreduceMulti(
        const std::vector<void *> &buffers,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (!initialized_)
        {
            last_error_ = "HostBackend::allreduceMulti: Backend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (buffers.size() != group_.size())
        {
            last_error_ = "HostBackend::allreduceMulti: buffer count " +
                          std::to_string(buffers.size()) + " != group size " +
                          std::to_string(group_.size());
            LOG_ERROR(last_error_);
            return false;
        }

        LOG_DEBUG("HostBackend::allreduceMulti: count=" << count
                                                        << " dtype=" << static_cast<int>(dtype)
                                                        << " op=" << toString(op)
                                                        << " devices=" << group_.size());

        // Single-device group: nothing to reduce
        if (group_.size() <= 1)
        {
            return true;
        }

        size_t bytes = count * elementSize(dtype);

        // Ensure staging buffer is sized for one device's payload
        if (!ensureStagingBuffer(bytes))
        {
            last_error_ = "HostBackend::allreduceMulti: Failed to allocate staging buffer";
            LOG_ERROR(last_error_);
            return false;
        }

        // Reduction accumulator on host
        std::vector<uint8_t> accumulator(bytes, 0);
        bool first = true;

        // Phase 1: D2H from each device and reduce into accumulator
        for (size_t i = 0; i < group_.size(); ++i)
        {
            const auto &device = group_.devices[i];
            const void *src = buffers[i];

            if (src == nullptr)
            {
                last_error_ = "HostBackend::allreduceMulti: buffer[" +
                              std::to_string(i) + "] is null";
                LOG_ERROR(last_error_);
                return false;
            }

            if (!copyToHost(staging_buffer_, src, device, bytes))
            {
                last_error_ = "HostBackend::allreduceMulti: D2H copy failed from " +
                              device.toString();
                LOG_ERROR(last_error_);
                return false;
            }

            if (first)
            {
                std::memcpy(accumulator.data(), staging_buffer_, bytes);
                first = false;
            }
            else
            {
                reduceOnHost(accumulator.data(), staging_buffer_, count, dtype, op);
            }
        }

        // Phase 2: H2D reduced result back to each device
        std::memcpy(staging_buffer_, accumulator.data(), bytes);
        for (size_t i = 0; i < group_.size(); ++i)
        {
            const auto &device = group_.devices[i];
            if (!copyFromHost(buffers[i], staging_buffer_, device, bytes))
            {
                last_error_ = "HostBackend::allreduceMulti: H2D copy failed to " +
                              device.toString();
                LOG_ERROR(last_error_);
                return false;
            }
        }

        LOG_DEBUG("HostBackend::allreduceMulti: Completed successfully");
        return true;
    }

    bool HostBackend::allgather(
        const void *send_buf,
        void *recv_buf,
        size_t send_count,
        CollectiveDataType dtype)
    {
        if (!initialized_)
        {
            last_error_ = "HostBackend::allgather: Backend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        LOG_DEBUG("HostBackend::allgather: send_count=" << send_count
                                                        << " dtype=" << static_cast<int>(dtype));

        // For single-device: copy send to recv
        if (group_.size() <= 1)
        {
            size_t bytes = send_count * elementSize(dtype);
            LOG_DEBUG("HostBackend::allgather: Single device, copying " << bytes << " bytes");
            std::memcpy(recv_buf, send_buf, bytes);
            return true;
        }

        // For multi-device groups: gather from each device
        size_t element_bytes = elementSize(dtype);
        size_t slice_bytes = send_count * element_bytes;
        size_t total_bytes = slice_bytes * group_.size();

        // Ensure staging buffer
        if (!ensureStagingBuffer(slice_bytes))
        {
            last_error_ = "HostBackend::allgather: Failed to allocate staging buffer";
            LOG_ERROR(last_error_);
            return false;
        }

        LOG_DEBUG("HostBackend::allgather: Gathering from " << group_.size() << " devices");

        // Phase 1: Gather each device's slice to host buffer
        std::vector<uint8_t> gathered(total_bytes);

        for (size_t i = 0; i < group_.size(); ++i)
        {
            const auto &device = group_.devices[i];
            uint8_t *dst = gathered.data() + i * slice_bytes;

            // Copy this device's slice to gathered buffer
            if (!copyToHost(staging_buffer_, send_buf, device, slice_bytes))
            {
                last_error_ = "HostBackend::allgather: Failed to gather from device " +
                              std::to_string(device.ordinal);
                LOG_ERROR(last_error_);
                return false;
            }
            std::memcpy(dst, staging_buffer_, slice_bytes);
        }

        // Phase 2: Broadcast full gathered buffer to all devices
        for (const auto &device : group_.devices)
        {
            // For large buffers, we may need to do this in chunks
            // For now, copy the full buffer directly
            if (device.type == DeviceType::CPU)
            {
                std::memcpy(recv_buf, gathered.data(), total_bytes);
            }
            else
            {
                // Copy in chunks via staging buffer
                size_t offset = 0;
                while (offset < total_bytes)
                {
                    size_t chunk = std::min(staging_buffer_size_, total_bytes - offset);
                    std::memcpy(staging_buffer_, gathered.data() + offset, chunk);
                    if (!copyFromHost(static_cast<uint8_t *>(recv_buf) + offset,
                                      staging_buffer_, device, chunk))
                    {
                        last_error_ = "HostBackend::allgather: Failed to broadcast to device";
                        LOG_ERROR(last_error_);
                        return false;
                    }
                    offset += chunk;
                }
            }
        }

        LOG_DEBUG("HostBackend::allgather: Completed successfully");
        return true;
    }

    bool HostBackend::allgatherv(
        const void *send_buf,
        size_t send_count,
        void *recv_buf,
        const std::vector<int> &recv_counts,
        const std::vector<int> &displacements,
        CollectiveDataType dtype)
    {
        if (!initialized_)
        {
            last_error_ = "HostBackend::allgatherv: Backend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        size_t element_bytes = elementSize(dtype);

        LOG_DEBUG("HostBackend::allgatherv: send_count=" << send_count
                                                         << " dtype=" << static_cast<int>(dtype)
                                                         << " num_ranks=" << recv_counts.size());

        // For single-device group: copy send to appropriate offset in recv
        if (group_.size() <= 1)
        {
            size_t send_bytes = send_count * element_bytes;
            // Single device = rank 0, use displacement[0]
            int offset = displacements.empty() ? 0 : displacements[0];
            size_t offset_bytes = static_cast<size_t>(offset) * element_bytes;

            LOG_DEBUG("HostBackend::allgatherv: Single device, copying " << send_bytes
                                                                         << " bytes to offset " << offset_bytes);

            uint8_t *dst = static_cast<uint8_t *>(recv_buf) + offset_bytes;
            std::memcpy(dst, send_buf, send_bytes);
            return true;
        }

        // For multi-device groups: gather from each device at variable offsets
        size_t send_bytes = send_count * element_bytes;

        // Ensure staging buffer for largest receive
        int max_recv = 0;
        for (int count : recv_counts)
        {
            max_recv = std::max(max_recv, count);
        }
        size_t max_recv_bytes = static_cast<size_t>(max_recv) * element_bytes;

        if (!ensureStagingBuffer(max_recv_bytes))
        {
            last_error_ = "HostBackend::allgatherv: Failed to allocate staging buffer";
            LOG_ERROR(last_error_);
            return false;
        }

        LOG_DEBUG("HostBackend::allgatherv: Gathering from " << group_.size() << " devices");

        // Phase 1: Gather each device's slice to recv buffer at variable offsets
        for (size_t i = 0; i < group_.size(); ++i)
        {
            const auto &device = group_.devices[i];
            int recv_count_i = (i < recv_counts.size()) ? recv_counts[i] : 0;
            int displacement_i = (i < displacements.size()) ? displacements[i] : 0;

            size_t recv_bytes_i = static_cast<size_t>(recv_count_i) * element_bytes;
            size_t offset_bytes = static_cast<size_t>(displacement_i) * element_bytes;
            uint8_t *dst = static_cast<uint8_t *>(recv_buf) + offset_bytes;

            // Copy this device's slice to recv buffer
            if (!copyToHost(staging_buffer_, send_buf, device, recv_bytes_i))
            {
                last_error_ = "HostBackend::allgatherv: Failed to gather from device " +
                              std::to_string(device.ordinal);
                LOG_ERROR(last_error_);
                return false;
            }
            std::memcpy(dst, staging_buffer_, recv_bytes_i);
        }

        // Phase 2: Copy gathered buffer back to all devices
        // Calculate total gathered size
        size_t total_bytes = 0;
        for (size_t i = 0; i < recv_counts.size(); ++i)
        {
            size_t end_offset = static_cast<size_t>(displacements[i]) * element_bytes +
                                static_cast<size_t>(recv_counts[i]) * element_bytes;
            total_bytes = std::max(total_bytes, end_offset);
        }

        for (const auto &device : group_.devices)
        {
            if (device.type == DeviceType::CPU)
            {
                // Already in host memory, no copy needed
                continue;
            }
            else
            {
                // Copy in chunks via staging buffer
                size_t offset = 0;
                while (offset < total_bytes)
                {
                    size_t chunk = std::min(staging_buffer_size_, total_bytes - offset);
                    std::memcpy(staging_buffer_, static_cast<const uint8_t *>(recv_buf) + offset, chunk);
                    if (!copyFromHost(static_cast<uint8_t *>(recv_buf) + offset,
                                      staging_buffer_, device, chunk))
                    {
                        last_error_ = "HostBackend::allgatherv: Failed to broadcast to device";
                        LOG_ERROR(last_error_);
                        return false;
                    }
                    offset += chunk;
                }
            }
        }

        LOG_DEBUG("HostBackend::allgatherv: Completed successfully");
        return true;
    }

    bool HostBackend::reduceScatter(
        const void *send_buf,
        void *recv_buf,
        size_t recv_count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (!initialized_)
        {
            last_error_ = "HostBackend::reduceScatter: Backend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        LOG_DEBUG("HostBackend::reduceScatter: recv_count=" << recv_count
                                                            << " dtype=" << static_cast<int>(dtype)
                                                            << " op=" << toString(op));

        // For single-device: copy send to recv
        if (group_.size() <= 1)
        {
            size_t bytes = recv_count * elementSize(dtype);
            LOG_DEBUG("HostBackend::reduceScatter: Single device, copying " << bytes << " bytes");
            std::memcpy(recv_buf, send_buf, bytes);
            return true;
        }

        // For multi-device groups: reduce then scatter
        size_t element_bytes = elementSize(dtype);
        size_t slice_bytes = recv_count * element_bytes;
        size_t total_elements = recv_count * group_.size();
        size_t total_bytes = slice_bytes * group_.size();

        // Ensure staging buffer
        if (!ensureStagingBuffer(total_bytes))
        {
            last_error_ = "HostBackend::reduceScatter: Failed to allocate staging buffer";
            LOG_ERROR(last_error_);
            return false;
        }

        LOG_DEBUG("HostBackend::reduceScatter: ReduceScatter across " << group_.size() << " devices");

        // Phase 1: Gather all data and reduce
        std::vector<uint8_t> accumulator(total_bytes, 0);
        bool first_device = true;

        for (const auto &device : group_.devices)
        {
            // Copy device buffer to staging
            if (!copyToHost(staging_buffer_, send_buf, device, total_bytes))
            {
                last_error_ = "HostBackend::reduceScatter: Failed to copy from device";
                LOG_ERROR(last_error_);
                return false;
            }

            if (first_device)
            {
                std::memcpy(accumulator.data(), staging_buffer_, total_bytes);
                first_device = false;
            }
            else
            {
                reduceOnHost(accumulator.data(), staging_buffer_, total_elements, dtype, op);
            }
        }

        // Phase 2: Scatter appropriate slice to each device
        for (size_t i = 0; i < group_.size(); ++i)
        {
            const auto &device = group_.devices[i];
            const uint8_t *slice_src = accumulator.data() + i * slice_bytes;

            std::memcpy(staging_buffer_, slice_src, slice_bytes);
            if (!copyFromHost(recv_buf, staging_buffer_, device, slice_bytes))
            {
                last_error_ = "HostBackend::reduceScatter: Failed to scatter to device";
                LOG_ERROR(last_error_);
                return false;
            }
        }

        LOG_DEBUG("HostBackend::reduceScatter: Completed successfully");
        return true;
    }

    bool HostBackend::broadcast(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        int root_rank)
    {
        if (!initialized_)
        {
            last_error_ = "HostBackend::broadcast: Backend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        LOG_DEBUG("HostBackend::broadcast: count=" << count
                                                   << " dtype=" << static_cast<int>(dtype)
                                                   << " root_rank=" << root_rank);

        // For single-device group: no-op
        if (group_.size() <= 1)
        {
            LOG_DEBUG("HostBackend::broadcast: Single device, no-op");
            return true;
        }

        // Validate root rank
        if (root_rank < 0 || static_cast<size_t>(root_rank) >= group_.size())
        {
            last_error_ = "HostBackend::broadcast: Invalid root_rank " +
                          std::to_string(root_rank);
            LOG_ERROR(last_error_);
            return false;
        }

        size_t bytes = count * elementSize(dtype);

        // Ensure staging buffer
        if (!ensureStagingBuffer(bytes))
        {
            last_error_ = "HostBackend::broadcast: Failed to allocate staging buffer";
            LOG_ERROR(last_error_);
            return false;
        }

        LOG_DEBUG("HostBackend::broadcast: Broadcasting from rank " << root_rank
                                                                    << " to " << group_.size() << " devices");

        // Phase 1: Copy from root device to staging
        const auto &root_device = group_.devices[root_rank];
        if (!copyToHost(staging_buffer_, buffer, root_device, bytes))
        {
            last_error_ = "HostBackend::broadcast: Failed to copy from root device";
            LOG_ERROR(last_error_);
            return false;
        }

        // Phase 2: Copy to all other devices
        for (size_t i = 0; i < group_.size(); ++i)
        {
            if (static_cast<int>(i) == root_rank)
                continue; // Skip root

            const auto &device = group_.devices[i];
            if (!copyFromHost(buffer, staging_buffer_, device, bytes))
            {
                last_error_ = "HostBackend::broadcast: Failed to copy to device " +
                              std::to_string(device.ordinal);
                LOG_ERROR(last_error_);
                return false;
            }
        }

        LOG_DEBUG("HostBackend::broadcast: Completed successfully");
        return true;
    }

    // =========================================================================
    // Device-to-Device Copy Operations
    // =========================================================================

    bool HostBackend::copy(
        void *dst_ptr, DeviceId dst_device,
        const void *src_ptr, DeviceId src_device,
        size_t bytes)
    {
        if (bytes == 0)
            return true;
        if (!dst_ptr || !src_ptr)
            return false;

        // CPU ↔ CPU: direct memcpy
        if (src_device.is_cpu() && dst_device.is_cpu())
        {
            std::memcpy(dst_ptr, src_ptr, bytes);
            return true;
        }

        // GPU → CPU
        if (!src_device.is_cpu() && dst_device.is_cpu())
        {
            return copyToHost(dst_ptr, src_ptr, src_device, bytes);
        }

        // CPU → GPU
        if (src_device.is_cpu() && !dst_device.is_cpu())
        {
            return copyFromHost(dst_ptr, src_ptr, dst_device, bytes);
        }

        // GPU → GPU (cross-vendor or same-vendor): stage through host
        if (!ensureStagingBuffer(bytes))
        {
            LOG_ERROR("HostBackend::copy: Failed to ensure staging buffer for "
                      << bytes << " bytes");
            return false;
        }

        // D2H from source GPU to staging buffer
        if (!copyToHost(staging_buffer_, src_ptr, src_device, bytes))
        {
            LOG_ERROR("HostBackend::copy: D2H failed from " << src_device.toString());
            return false;
        }

        // H2D from staging buffer to destination GPU
        if (!copyFromHost(dst_ptr, staging_buffer_, dst_device, bytes))
        {
            LOG_ERROR("HostBackend::copy: H2D failed to " << dst_device.toString());
            return false;
        }

        return true;
    }

    bool HostBackend::copyAsync(
        void *dst_ptr, DeviceId dst_device,
        const void *src_ptr, DeviceId src_device,
        size_t bytes, void *stream)
    {
        (void)stream; // No streams on CPU
        return copy(dst_ptr, dst_device, src_ptr, src_device, bytes);
    }

    bool HostBackend::supportsCopy(DeviceId src_device, DeviceId dst_device) const
    {
        // HostBackend can copy between any device pair via host staging
        return true;
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    size_t HostBackend::elementSize(CollectiveDataType dtype) const
    {
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
            return 4;
        case CollectiveDataType::FLOAT16:
        case CollectiveDataType::BFLOAT16:
            return 2;
        case CollectiveDataType::INT32:
            return 4;
        case CollectiveDataType::INT8:
            return 1;
        default:
            LOG_WARN("HostBackend::elementSize: Unknown dtype, assuming 4 bytes");
            return 4;
        }
    }

    bool HostBackend::ensureStagingBuffer(size_t required_bytes)
    {
        if (staging_buffer_ && staging_buffer_size_ >= required_bytes)
        {
            return true;
        }

        // Free old buffer if exists
        if (staging_buffer_)
        {
#ifdef HAVE_CUDA
            if (staging_registered_cuda_)
            {
                host_backend_detail::cudaHostUnregisterBuffer(staging_buffer_);
                staging_registered_cuda_ = false;
            }
#endif
#ifdef HAVE_ROCM
            if (staging_registered_rocm_)
            {
                host_backend_detail::hipHostUnregisterBuffer(staging_buffer_);
                staging_registered_rocm_ = false;
            }
#endif
            std::free(staging_buffer_);
            staging_buffer_ = nullptr;
            staging_buffer_size_ = 0;
        }

        // Allocate page-aligned buffer (needed for host registration)
        size_t alloc_size = std::max(required_bytes, size_t(64 * 1024 * 1024)); // Min 64MB
        void *ptr = nullptr;
        if (posix_memalign(&ptr, 4096, alloc_size) != 0)
        {
            LOG_ERROR("HostBackend: Failed to allocate staging buffer of " << alloc_size << " bytes");
            return false;
        }

        staging_buffer_ = ptr;
        staging_buffer_size_ = alloc_size;

        // Register with both runtimes for efficient DMA
#ifdef HAVE_CUDA
        if (has_cuda_)
        {
            if (!host_backend_detail::cudaHostRegisterBuffer(staging_buffer_, alloc_size))
            {
                LOG_WARN("HostBackend: cudaHostRegister failed");
            }
            else
            {
                staging_registered_cuda_ = true;
            }
        }
#endif
#ifdef HAVE_ROCM
        if (has_rocm_)
        {
            if (!host_backend_detail::hipHostRegisterBuffer(staging_buffer_, alloc_size))
            {
                LOG_WARN("HostBackend: hipHostRegister failed");
            }
            else
            {
                staging_registered_rocm_ = true;
            }
        }
#endif

        LOG_DEBUG("HostBackend: Allocated staging buffer of " << alloc_size << " bytes");
        return true;
    }

    bool HostBackend::copyToHost(void *host_dst, const void *device_src,
                                 DeviceId device, size_t bytes)
    {
        if (device.type == DeviceType::CPU)
        {
            std::memcpy(host_dst, device_src, bytes);
            return true;
        }

        void *stream = GPUDeviceContextPool::instance().getContext(device).defaultStream();

#ifdef HAVE_CUDA
        if (device.type == DeviceType::CUDA)
        {
            return host_backend_detail::cudaCopyToHost(host_dst, device_src, device.ordinal, bytes, stream);
        }
#endif

#ifdef HAVE_ROCM
        if (device.type == DeviceType::ROCm)
        {
            return host_backend_detail::hipCopyToHost(host_dst, device_src, device.ordinal, bytes, stream);
        }
#endif

        LOG_ERROR("HostBackend::copyToHost: Unsupported device type");
        return false;
    }

    bool HostBackend::copyFromHost(void *device_dst, const void *host_src,
                                   DeviceId device, size_t bytes)
    {
        if (device.type == DeviceType::CPU)
        {
            std::memcpy(device_dst, host_src, bytes);
            return true;
        }

        void *stream = GPUDeviceContextPool::instance().getContext(device).defaultStream();

#ifdef HAVE_CUDA
        if (device.type == DeviceType::CUDA)
        {
            return host_backend_detail::cudaCopyFromHost(device_dst, host_src, device.ordinal, bytes, stream);
        }
#endif

#ifdef HAVE_ROCM
        if (device.type == DeviceType::ROCm)
        {
            return host_backend_detail::hipCopyFromHost(device_dst, host_src, device.ordinal, bytes, stream);
        }
#endif

        LOG_ERROR("HostBackend::copyFromHost: Unsupported device type");
        return false;
    }

    void HostBackend::reduceOnHost(void *dst, const void *src, size_t count,
                                   CollectiveDataType dtype, CollectiveOp op)
    {
        // FP32 reduction (most common case)
        if (dtype == CollectiveDataType::FLOAT32)
        {
            float *dst_f = static_cast<float *>(dst);
            const float *src_f = static_cast<const float *>(src);

            switch (op)
            {
            case CollectiveOp::ALLREDUCE_SUM:
#pragma omp parallel for simd
                for (size_t i = 0; i < count; ++i)
                {
                    dst_f[i] += src_f[i];
                }
                break;

            case CollectiveOp::ALLREDUCE_MAX:
#pragma omp parallel for simd
                for (size_t i = 0; i < count; ++i)
                {
                    dst_f[i] = std::max(dst_f[i], src_f[i]);
                }
                break;

            case CollectiveOp::ALLREDUCE_MIN:
#pragma omp parallel for simd
                for (size_t i = 0; i < count; ++i)
                {
                    dst_f[i] = std::min(dst_f[i], src_f[i]);
                }
                break;

            default:
                LOG_WARN("HostBackend::reduceOnHost: Unsupported op, defaulting to SUM");
#pragma omp parallel for simd
                for (size_t i = 0; i < count; ++i)
                {
                    dst_f[i] += src_f[i];
                }
            }
            return;
        }

        // INT32 reduction
        if (dtype == CollectiveDataType::INT32)
        {
            int32_t *dst_i = static_cast<int32_t *>(dst);
            const int32_t *src_i = static_cast<const int32_t *>(src);

            switch (op)
            {
            case CollectiveOp::ALLREDUCE_SUM:
#pragma omp parallel for simd
                for (size_t i = 0; i < count; ++i)
                {
                    dst_i[i] += src_i[i];
                }
                break;

            case CollectiveOp::ALLREDUCE_MAX:
#pragma omp parallel for simd
                for (size_t i = 0; i < count; ++i)
                {
                    dst_i[i] = std::max(dst_i[i], src_i[i]);
                }
                break;

            case CollectiveOp::ALLREDUCE_MIN:
#pragma omp parallel for simd
                for (size_t i = 0; i < count; ++i)
                {
                    dst_i[i] = std::min(dst_i[i], src_i[i]);
                }
                break;

            default:
                LOG_WARN("HostBackend::reduceOnHost: Unsupported op for INT32");
            }
            return;
        }

        // FP16 reduction (convert to float for computation)
        if (dtype == CollectiveDataType::FLOAT16)
        {
            const uint16_t *src_h = static_cast<const uint16_t *>(src);
            uint16_t *dst_h = static_cast<uint16_t *>(dst);

            // Helper lambdas for FP16 conversion (IEEE 754 half-precision)
            auto fp16_to_fp32 = [](uint16_t h) -> float
            {
                uint32_t sign = (h >> 15) & 0x1;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;

                if (exp == 0)
                {
                    if (mant == 0)
                        return sign ? -0.0f : 0.0f;
                    // Denormalized
                    while (!(mant & 0x400))
                    {
                        mant <<= 1;
                        exp--;
                    }
                    exp++;
                    mant &= ~0x400;
                }
                else if (exp == 31)
                {
                    uint32_t result = (sign << 31) | 0x7F800000 | (mant << 13);
                    return *reinterpret_cast<float *>(&result);
                }

                uint32_t result = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                return *reinterpret_cast<float *>(&result);
            };

            auto fp32_to_fp16 = [](float f) -> uint16_t
            {
                uint32_t x = *reinterpret_cast<uint32_t *>(&f);
                uint32_t sign = (x >> 31) & 0x1;
                int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
                uint32_t mant = (x >> 13) & 0x3FF;

                if (exp <= 0)
                    return static_cast<uint16_t>(sign << 15);
                if (exp >= 31)
                    return static_cast<uint16_t>((sign << 15) | 0x7C00);

                return static_cast<uint16_t>((sign << 15) | (exp << 10) | mant);
            };

            switch (op)
            {
            case CollectiveOp::ALLREDUCE_SUM:
#pragma omp parallel for
                for (size_t i = 0; i < count; ++i)
                {
                    float sum = fp16_to_fp32(dst_h[i]) + fp16_to_fp32(src_h[i]);
                    dst_h[i] = fp32_to_fp16(sum);
                }
                break;

            case CollectiveOp::ALLREDUCE_MAX:
#pragma omp parallel for
                for (size_t i = 0; i < count; ++i)
                {
                    float a = fp16_to_fp32(dst_h[i]);
                    float b = fp16_to_fp32(src_h[i]);
                    dst_h[i] = fp32_to_fp16(std::max(a, b));
                }
                break;

            case CollectiveOp::ALLREDUCE_MIN:
#pragma omp parallel for
                for (size_t i = 0; i < count; ++i)
                {
                    float a = fp16_to_fp32(dst_h[i]);
                    float b = fp16_to_fp32(src_h[i]);
                    dst_h[i] = fp32_to_fp16(std::min(a, b));
                }
                break;

            default:
                LOG_WARN("HostBackend::reduceOnHost: Unsupported op for FP16");
            }
            return;
        }

        // BF16 reduction (convert to float for computation)
        if (dtype == CollectiveDataType::BFLOAT16)
        {
            const uint16_t *src_bf = static_cast<const uint16_t *>(src);
            uint16_t *dst_bf = static_cast<uint16_t *>(dst);

            // BF16 is just the upper 16 bits of FP32
            auto bf16_to_fp32 = [](uint16_t bf) -> float
            {
                uint32_t x = static_cast<uint32_t>(bf) << 16;
                return *reinterpret_cast<float *>(&x);
            };

            auto fp32_to_bf16 = [](float f) -> uint16_t
            {
                uint32_t x = *reinterpret_cast<uint32_t *>(&f);
                return static_cast<uint16_t>(x >> 16);
            };

            switch (op)
            {
            case CollectiveOp::ALLREDUCE_SUM:
#pragma omp parallel for
                for (size_t i = 0; i < count; ++i)
                {
                    float sum = bf16_to_fp32(dst_bf[i]) + bf16_to_fp32(src_bf[i]);
                    dst_bf[i] = fp32_to_bf16(sum);
                }
                break;

            case CollectiveOp::ALLREDUCE_MAX:
#pragma omp parallel for
                for (size_t i = 0; i < count; ++i)
                {
                    float a = bf16_to_fp32(dst_bf[i]);
                    float b = bf16_to_fp32(src_bf[i]);
                    dst_bf[i] = fp32_to_bf16(std::max(a, b));
                }
                break;

            case CollectiveOp::ALLREDUCE_MIN:
#pragma omp parallel for
                for (size_t i = 0; i < count; ++i)
                {
                    float a = bf16_to_fp32(dst_bf[i]);
                    float b = bf16_to_fp32(src_bf[i]);
                    dst_bf[i] = fp32_to_bf16(std::min(a, b));
                }
                break;

            default:
                LOG_WARN("HostBackend::reduceOnHost: Unsupported op for BF16");
            }
            return;
        }

        // INT8 reduction (with saturation)
        if (dtype == CollectiveDataType::INT8)
        {
            const int8_t *src_i8 = static_cast<const int8_t *>(src);
            int8_t *dst_i8 = static_cast<int8_t *>(dst);

            switch (op)
            {
            case CollectiveOp::ALLREDUCE_SUM:
#pragma omp parallel for simd
                for (size_t i = 0; i < count; ++i)
                {
                    int32_t sum = static_cast<int32_t>(dst_i8[i]) + static_cast<int32_t>(src_i8[i]);
                    dst_i8[i] = static_cast<int8_t>(std::max(-128, std::min(127, sum)));
                }
                break;

            case CollectiveOp::ALLREDUCE_MAX:
#pragma omp parallel for simd
                for (size_t i = 0; i < count; ++i)
                {
                    dst_i8[i] = std::max(dst_i8[i], src_i8[i]);
                }
                break;

            case CollectiveOp::ALLREDUCE_MIN:
#pragma omp parallel for simd
                for (size_t i = 0; i < count; ++i)
                {
                    dst_i8[i] = std::min(dst_i8[i], src_i8[i]);
                }
                break;

            default:
                LOG_WARN("HostBackend::reduceOnHost: Unsupported op for INT8");
            }
            return;
        }

        LOG_WARN("HostBackend::reduceOnHost: Dtype not implemented, no reduction performed");
    }

} // namespace llaminar2
