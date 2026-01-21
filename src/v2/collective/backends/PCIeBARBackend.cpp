/**
 * @file PCIeBARBackend.cpp
 * @brief Direct CUDA↔ROCm collective backend via PCIe BAR mapping
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "PCIeBARBackend.h"
#include "../../utils/Logger.h"
#include "../../kernels/cuda/ops/CUDAVectorAddKernels.h"

#include <algorithm>

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#include <cuda_runtime.h>
#endif

namespace llaminar2
{

    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    PCIeBARBackend::PCIeBARBackend(std::unique_ptr<DirectP2PEngine> p2p_engine)
        : p2p_engine_(std::move(p2p_engine))
    {
        if (!p2p_engine_)
        {
            p2p_engine_ = std::make_unique<DirectP2PEngine>();
        }
    }

    PCIeBARBackend::~PCIeBARBackend()
    {
        shutdown();
    }

    // =========================================================================
    // Capability Queries
    // =========================================================================

    bool PCIeBARBackend::supportsDirectTransfer(DeviceId src, DeviceId dst) const
    {
        if (!p2p_engine_ || !p2p_engine_->isPCIeBarActive())
        {
            return false;
        }

        // True if one is CUDA and one is ROCm
        return (src.is_cuda() && dst.is_rocm()) || (src.is_rocm() && dst.is_cuda());
    }

    bool PCIeBARBackend::isAvailable() const
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        // Check if we can probe for BAR P2P capability
        auto caps = DirectP2PEngine::probeCapabilities();
        return caps.canDoPCIeBarP2P();
#else
        return false;
#endif
    }

    bool PCIeBARBackend::isPCIeBarActive() const
    {
        return p2p_engine_ && p2p_engine_->isPCIeBarActive();
    }

    // =========================================================================
    // BAR Region Allocator
    // =========================================================================

    std::optional<std::pair<void *, size_t>> PCIeBARBackend::allocateInBarRegion(size_t size)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        // Check BAR is active
        if (!p2p_engine_ || !p2p_engine_->isPCIeBarActive())
        {
            LOG_ERROR("allocateInBarRegion: PCIe BAR not initialized");
            return std::nullopt;
        }

        // Initialize BAR info if not done yet
        if (bar_host_ptr_ == nullptr)
        {
            bar_host_ptr_ = p2p_engine_->getBarHostPtr();
            bar_total_mapped_size_ = p2p_engine_->getBarMappedSize();
        }

        if (bar_host_ptr_ == nullptr || bar_total_mapped_size_ == 0)
        {
            LOG_ERROR("allocateInBarRegion: BAR not properly mapped");
            return std::nullopt;
        }

        // Align size to BAR_ALLOC_ALIGNMENT (256 bytes)
        size_t aligned_size = (size + BAR_ALLOC_ALIGNMENT - 1) & ~(BAR_ALLOC_ALIGNMENT - 1);

        // Check if we have enough space
        if (bar_alloc_offset_ + aligned_size > bar_total_mapped_size_)
        {
            LOG_ERROR("allocateInBarRegion: Not enough BAR space. Requested: " << aligned_size
                                                                               << ", available: " << (bar_total_mapped_size_ - bar_alloc_offset_));
            return std::nullopt;
        }

        // Bump allocate
        size_t offset = bar_alloc_offset_;
        void *ptr = static_cast<char *>(bar_host_ptr_) + offset;
        bar_alloc_offset_ += aligned_size;

        // Track allocation
        BarAllocation alloc{ptr, offset, aligned_size};
        bar_allocations_.push_back(alloc);

        LOG_DEBUG("allocateInBarRegion: Allocated " << aligned_size << " bytes at offset " << offset
                                                    << " (total used: " << bar_alloc_offset_ << "/" << bar_total_mapped_size_ << ")");

        return std::make_pair(ptr, offset);
#else
        (void)size;
        return std::nullopt;
#endif
    }

    void PCIeBARBackend::freeBarBuffer(void *ptr)
    {
        if (!ptr)
            return;

        // Find and remove from tracking (no actual deallocation - bump allocator)
        auto it = std::remove_if(bar_allocations_.begin(), bar_allocations_.end(),
                                 [ptr](const BarAllocation &alloc)
                                 { return alloc.ptr == ptr; });

        if (it != bar_allocations_.end())
        {
            LOG_DEBUG("freeBarBuffer: Removed allocation at " << ptr);
            bar_allocations_.erase(it, bar_allocations_.end());
        }
    }

    // =========================================================================
    // Buffer Registration
    // =========================================================================

    bool PCIeBARBackend::registerBuffer(const std::string &collective_id,
                                        DeviceId device,
                                        void *buffer,
                                        size_t size)
    {
        if (!buffer || size == 0)
        {
            LOG_ERROR("registerBuffer: Invalid buffer or size");
            return false;
        }

        auto &collective = registered_collectives_[collective_id];

        if (device.is_cuda())
        {
            // CUDA buffer: store directly, bar_offset not applicable
            collective.cuda_buffer = RegisteredBuffer(device, buffer, size, 0, true);
            collective.cuda_registered = true;
            LOG_DEBUG("registerBuffer: Registered CUDA buffer for " << collective_id
                                                                    << " (ptr=" << buffer << ", size=" << size << ")");
            return true;
        }
        else if (device.is_rocm())
        {
            // ROCm buffer: must find its BAR offset
            // Look through our allocations to find this buffer
            for (const auto &alloc : bar_allocations_)
            {
                if (alloc.ptr == buffer)
                {
                    collective.rocm_buffer = RegisteredBuffer(device, buffer, size, alloc.offset, false);
                    collective.rocm_registered = true;
                    LOG_DEBUG("registerBuffer: Registered ROCm buffer for " << collective_id
                                                                            << " (ptr=" << buffer << ", offset=" << alloc.offset << ", size=" << size << ")");
                    return true;
                }
            }

            // Buffer not found in BAR allocations
            LOG_ERROR("registerBuffer: ROCm buffer " << buffer << " was not allocated via allocateInBarRegion()");
            return false;
        }

        LOG_ERROR("registerBuffer: Unsupported device type");
        return false;
    }

    void PCIeBARBackend::unregisterBuffer(const std::string &collective_id, DeviceId device)
    {
        auto it = registered_collectives_.find(collective_id);
        if (it == registered_collectives_.end())
        {
            return;
        }

        if (device.is_cuda())
        {
            it->second.cuda_registered = false;
            it->second.cuda_buffer = RegisteredBuffer();
        }
        else if (device.is_rocm())
        {
            it->second.rocm_registered = false;
            it->second.rocm_buffer = RegisteredBuffer();
        }

        // Remove entry if both are unregistered
        if (!it->second.cuda_registered && !it->second.rocm_registered)
        {
            registered_collectives_.erase(it);
        }
    }

    std::optional<RegisteredBuffer> PCIeBARBackend::getBuffer(const std::string &collective_id,
                                                              DeviceId device) const
    {
        auto it = registered_collectives_.find(collective_id);
        if (it == registered_collectives_.end())
        {
            return std::nullopt;
        }

        if (device.is_cuda() && it->second.cuda_registered)
        {
            return it->second.cuda_buffer;
        }
        else if (device.is_rocm() && it->second.rocm_registered)
        {
            return it->second.rocm_buffer;
        }

        return std::nullopt;
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool PCIeBARBackend::initialize(const DeviceGroup &group)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (initialized_)
        {
            return true;
        }

        // Find CUDA and ROCm devices in the group
        for (const auto &dev : group.devices)
        {
            if (dev.is_cuda() && !has_cuda_)
            {
                cuda_device_ = dev;
                has_cuda_ = true;
            }
            else if (dev.is_rocm() && !has_rocm_)
            {
                rocm_device_ = dev;
                has_rocm_ = true;
            }
        }

        if (!has_cuda_ || !has_rocm_)
        {
            LOG_ERROR("PCIeBARBackend requires both CUDA and ROCm devices in group");
            return false;
        }

        // Initialize P2P engine if not already done
        if (!p2p_engine_->isPCIeBarActive())
        {
            // Request 1GB BAR mapping for large transfers
            constexpr size_t map_size = 1024 * 1024 * 1024;
            if (!p2p_engine_->initializePCIeBar(cuda_device_, rocm_device_, 0, map_size))
            {
                LOG_ERROR("Failed to initialize PCIe BAR P2P between CUDA:" << cuda_device_.ordinal
                                                                            << " and ROCm:" << rocm_device_.ordinal);
                return false;
            }

            // Benchmark to get measured bandwidth
            auto result = p2p_engine_->benchmarkPCIeBar(64 * 1024 * 1024, 3);
            if (result.success)
            {
                measured_bandwidth_gbps_ = (result.read_gbps + result.write_gbps) / 2.0;
                LOG_INFO("PCIe BAR P2P initialized: " << measured_bandwidth_gbps_ << " GB/s average");
            }
        }

        // Initialize BAR allocator state
        bar_host_ptr_ = p2p_engine_->getBarHostPtr();
        bar_total_mapped_size_ = p2p_engine_->getBarMappedSize();
        bar_alloc_offset_ = 0; // Start allocations at beginning of BAR

        initialized_ = true;
        LOG_INFO("PCIeBARBackend initialized for group: " << group.name);
        return true;
#else
        LOG_ERROR("PCIeBARBackend requires HAVE_CUDA and HAVE_ROCM");
        return false;
#endif
    }

    void PCIeBARBackend::shutdown()
    {
        freeTempBuffer();

        // Clear BAR allocator state
        bar_alloc_offset_ = 0;
        bar_total_mapped_size_ = 0;
        bar_host_ptr_ = nullptr;
        bar_allocations_.clear();

        // Clear registered buffers
        registered_collectives_.clear();

        // p2p_engine_ destructor handles BAR cleanup
        initialized_ = false;
        has_cuda_ = false;
        has_rocm_ = false;
    }

    // =========================================================================
    // Collective Operations
    // =========================================================================

    bool PCIeBARBackend::allreduce(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (!initialized_)
        {
            LOG_ERROR("PCIeBARBackend not initialized");
            return false;
        }

        if (op != CollectiveOp::ALLREDUCE_SUM)
        {
            LOG_ERROR("PCIeBARBackend only supports ALLREDUCE_SUM currently");
            return false;
        }

        size_t bytes = count * datatypeSize(dtype);

        // Ensure we have a temp buffer on CUDA side
        if (!ensureTempBuffer(bytes))
        {
            return false;
        }

        // The buffer is assumed to be on CUDA device (common case for tensor parallelism)
        // Algorithm:
        // 1. Read ROCm's data into CUDA temp buffer via BAR
        // 2. Add CUDA buffer + temp buffer -> CUDA buffer
        // 3. Write CUDA buffer to ROCm via BAR

        // Step 1: Read ROCm data via BAR
        if (!transferROCmtoCUDA(0, cuda_temp_buffer_, bytes))
        {
            LOG_ERROR("Failed to read ROCm data via PCIe BAR");
            return false;
        }

        // Step 2: Reduce on CUDA
        if (!reduceOnCUDA(buffer, buffer, cuda_temp_buffer_, count, dtype, op))
        {
            LOG_ERROR("Failed to perform reduction on CUDA");
            return false;
        }

        // Step 3: Write result back to ROCm
        if (!transferCUDAtoROCm(buffer, 0, bytes))
        {
            LOG_ERROR("Failed to write result to ROCm via PCIe BAR");
            return false;
        }

        return synchronize();
#else
        return false;
#endif
    }

    bool PCIeBARBackend::allgather(
        const void *send_buf,
        void *recv_buf,
        size_t send_count,
        CollectiveDataType dtype)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (!initialized_)
        {
            LOG_ERROR("PCIeBARBackend not initialized");
            return false;
        }

        size_t bytes = send_count * datatypeSize(dtype);

        // For 2-device AllGather:
        // recv_buf layout: [CUDA_data | ROCm_data]
        // Each device has send_count elements, output has 2 * send_count

        // Assuming recv_buf is on CUDA device:
        // 1. Copy local CUDA data to first half of recv_buf
        // 2. Read ROCm data via BAR to second half
        // 3. Write full recv_buf back to ROCm

        // Step 1: Local copy (CUDA data already in place if send_buf == recv_buf slice)
        if (send_buf != recv_buf)
        {
            cudaMemcpy(recv_buf, send_buf, bytes, cudaMemcpyDeviceToDevice);
        }

        // Step 2: Read ROCm data to second half
        void *rocm_section = static_cast<char *>(recv_buf) + bytes;
        if (!transferROCmtoCUDA(0, rocm_section, bytes))
        {
            LOG_ERROR("AllGather: failed to read ROCm data");
            return false;
        }

        // Step 3: Write full buffer to ROCm
        if (!transferCUDAtoROCm(recv_buf, 0, 2 * bytes))
        {
            LOG_ERROR("AllGather: failed to write to ROCm");
            return false;
        }

        return synchronize();
#else
        return false;
#endif
    }

    bool PCIeBARBackend::allgatherv(
        const void *send_buf,
        size_t send_count,
        void *recv_buf,
        const std::vector<int> &recv_counts,
        const std::vector<int> &displacements,
        CollectiveDataType dtype)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (!initialized_)
        {
            LOG_ERROR("PCIeBARBackend::allgatherv not initialized");
            return false;
        }

        // For 2-device variable AllGather (CUDA + ROCm):
        // recv_buf layout: [device0_data @ disp[0] | device1_data @ disp[1]]

        // If counts are equal, fall back to regular allgather
        if (recv_counts.size() == 2 && recv_counts[0] == recv_counts[1])
        {
            return allgather(send_buf, recv_buf, send_count, dtype);
        }

        size_t elem_size = datatypeSize(dtype);

        // Assuming device 0 is CUDA, device 1 is ROCm
        // Step 1: Copy local CUDA data to its position
        size_t cuda_bytes = recv_counts[0] * elem_size;
        size_t cuda_offset = displacements[0] * elem_size;
        char *cuda_dst = static_cast<char *>(recv_buf) + cuda_offset;
        if (send_buf != cuda_dst)
        {
            cudaMemcpy(cuda_dst, send_buf, cuda_bytes, cudaMemcpyDeviceToDevice);
        }

        // Step 2: Read ROCm data to its position via BAR
        size_t rocm_bytes = recv_counts[1] * elem_size;
        size_t rocm_offset = displacements[1] * elem_size;
        char *rocm_dst = static_cast<char *>(recv_buf) + rocm_offset;
        if (!transferROCmtoCUDA(0, rocm_dst, rocm_bytes))
        {
            LOG_ERROR("AllGatherV: failed to read ROCm data");
            return false;
        }

        // Step 3: Calculate total size and write full buffer to ROCm
        size_t total_bytes = 0;
        for (size_t i = 0; i < recv_counts.size(); ++i)
        {
            size_t end = (displacements[i] + recv_counts[i]) * elem_size;
            total_bytes = std::max(total_bytes, end);
        }

        if (!transferCUDAtoROCm(recv_buf, 0, total_bytes))
        {
            LOG_ERROR("AllGatherV: failed to write to ROCm");
            return false;
        }

        return synchronize();
#else
        (void)send_buf;
        (void)send_count;
        (void)recv_buf;
        (void)recv_counts;
        (void)displacements;
        (void)dtype;
        LOG_ERROR("PCIeBARBackend::allgatherv requires HAVE_CUDA and HAVE_ROCM");
        return false;
#endif
    }

    bool PCIeBARBackend::reduceScatter(
        const void *send_buf,
        void *recv_buf,
        size_t recv_count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        // TODO: Implement reduce-scatter for PCIe BAR
        LOG_ERROR("PCIeBARBackend::reduceScatter not yet implemented");
        return false;
    }

    bool PCIeBARBackend::broadcast(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        int root)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (!initialized_)
        {
            LOG_ERROR("PCIeBARBackend not initialized");
            return false;
        }

        size_t bytes = count * datatypeSize(dtype);

        if (root == 0)
        {
            // CUDA is root - write to ROCm
            if (!transferCUDAtoROCm(buffer, 0, bytes))
            {
                return false;
            }
        }
        else
        {
            // ROCm is root - read to CUDA
            if (!transferROCmtoCUDA(0, buffer, bytes))
            {
                return false;
            }
        }

        return synchronize();
#else
        return false;
#endif
    }

    // =========================================================================
    // Registered Buffer AllReduce
    // =========================================================================

    bool PCIeBARBackend::allreduceRegistered(
        const std::string &collective_id,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (!initialized_)
        {
            LOG_ERROR("PCIeBARBackend not initialized");
            return false;
        }

        if (op != CollectiveOp::ALLREDUCE_SUM)
        {
            LOG_ERROR("PCIeBARBackend only supports ALLREDUCE_SUM currently");
            return false;
        }

        // Look up registered buffers
        auto it = registered_collectives_.find(collective_id);
        if (it == registered_collectives_.end())
        {
            LOG_ERROR("allreduceRegistered: No buffers registered for collective_id=" << collective_id);
            return false;
        }

        const auto &collective = it->second;
        if (!collective.cuda_registered)
        {
            LOG_ERROR("allreduceRegistered: CUDA buffer not registered for " << collective_id);
            return false;
        }
        if (!collective.rocm_registered)
        {
            LOG_ERROR("allreduceRegistered: ROCm buffer not registered for " << collective_id);
            return false;
        }

        size_t bytes = count * datatypeSize(dtype);

        // Validate size against registered buffers
        if (bytes > collective.cuda_buffer.size || bytes > collective.rocm_buffer.size)
        {
            LOG_ERROR("allreduceRegistered: Requested size " << bytes
                                                             << " exceeds registered buffer sizes (CUDA: " << collective.cuda_buffer.size
                                                             << ", ROCm: " << collective.rocm_buffer.size << ")");
            return false;
        }

        // Ensure we have a temp buffer on CUDA side
        if (!ensureTempBuffer(bytes))
        {
            LOG_ERROR("allreduceRegistered: Failed to allocate temp buffer");
            return false;
        }

        void *cuda_buffer = collective.cuda_buffer.ptr;
        size_t rocm_bar_offset = collective.rocm_buffer.bar_offset;

        LOG_DEBUG("allreduceRegistered: " << collective_id
                                          << " (CUDA ptr=" << cuda_buffer << ", ROCm offset=" << rocm_bar_offset
                                          << ", bytes=" << bytes << ")");

        // Step 1: Read ROCm data via BAR using registered offset
        if (!transferROCmtoCUDA(rocm_bar_offset, cuda_temp_buffer_, bytes))
        {
            LOG_ERROR("allreduceRegistered: Failed to read ROCm data via PCIe BAR");
            return false;
        }

        // Step 2: Reduce on CUDA (cuda_buffer += cuda_temp_buffer_)
        if (!reduceOnCUDA(cuda_buffer, cuda_buffer, cuda_temp_buffer_, count, dtype, op))
        {
            LOG_ERROR("allreduceRegistered: Failed to perform reduction on CUDA");
            return false;
        }

        // Step 3: Write result back to ROCm at registered offset
        if (!transferCUDAtoROCm(cuda_buffer, rocm_bar_offset, bytes))
        {
            LOG_ERROR("allreduceRegistered: Failed to write result to ROCm via PCIe BAR");
            return false;
        }

        return synchronize();
#else
        (void)collective_id;
        (void)count;
        (void)dtype;
        (void)op;
        return false;
#endif
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    bool PCIeBARBackend::synchronize()
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        // Synchronize CUDA device
        int prev_device;
        cudaGetDevice(&prev_device);
        cudaSetDevice(cuda_device_.ordinal);
        cudaDeviceSynchronize();
        cudaSetDevice(prev_device);

        // Note: ROCm synchronization would be similar with hipSetDevice/hipDeviceSynchronize
        // but we're focusing on CUDA-initiated transfers
        return true;
#else
        return false;
#endif
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    bool PCIeBARBackend::ensureTempBuffer(size_t bytes)
    {
#if defined(HAVE_CUDA)
        if (cuda_temp_buffer_ && cuda_temp_buffer_size_ >= bytes)
        {
            return true;
        }

        freeTempBuffer();

        // Allocate using CUDA runtime
        int prev_device;
        cudaGetDevice(&prev_device);
        cudaSetDevice(cuda_device_.ordinal);

        if (cudaMalloc(&cuda_temp_buffer_, bytes) == cudaSuccess)
        {
            cuda_temp_buffer_size_ = bytes;
            cudaSetDevice(prev_device);
            return true;
        }

        cudaSetDevice(prev_device);
        return false;
#else
        return false;
#endif
    }

    void PCIeBARBackend::freeTempBuffer()
    {
#if defined(HAVE_CUDA)
        if (cuda_temp_buffer_)
        {
            int prev_device;
            cudaGetDevice(&prev_device);
            cudaSetDevice(cuda_device_.ordinal);
            cudaFree(cuda_temp_buffer_);
            cudaSetDevice(prev_device);
            cuda_temp_buffer_ = nullptr;
            cuda_temp_buffer_size_ = 0;
        }
#endif
    }

    bool PCIeBARBackend::transferCUDAtoROCm(const void *cuda_src, size_t offset, size_t bytes)
    {
        if (!p2p_engine_ || !p2p_engine_->isPCIeBarActive())
        {
            return false;
        }

        auto result = p2p_engine_->transferViaPCIeBar(
            const_cast<void *>(cuda_src), offset, bytes, DirectP2PEngine::Direction::ToAMD);
        return result.success;
    }

    bool PCIeBARBackend::transferROCmtoCUDA(size_t offset, void *cuda_dst, size_t bytes)
    {
        if (!p2p_engine_ || !p2p_engine_->isPCIeBarActive())
        {
            return false;
        }

        auto result = p2p_engine_->transferViaPCIeBar(
            cuda_dst, offset, bytes, DirectP2PEngine::Direction::ToNVIDIA);
        return result.success;
    }

    bool PCIeBARBackend::reduceOnCUDA(
        void *output,
        const void *input1,
        const void *input2,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
#if defined(HAVE_CUDA)
        // Use CUDA kernel for GPU-side reduction
        if (dtype != CollectiveDataType::FLOAT32 || op != CollectiveOp::ALLREDUCE_SUM)
        {
            LOG_ERROR("PCIeBARBackend reduction only supports FP32 SUM currently");
            return false;
        }

        // Since input1 == output (in-place), we need: output += input2
        // Use the vectorized CUDA kernel for high performance
        bool success = cuda::launchVectorAddInplace_f32(
            static_cast<float *>(output),
            static_cast<const float *>(input2),
            count,
            nullptr // default stream
        );

        if (!success)
        {
            LOG_ERROR("CUDA vector add kernel failed");
            return false;
        }

        return true;
#else
        return false;
#endif
    }

    size_t PCIeBARBackend::datatypeSize(CollectiveDataType dtype) const
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
            return 4;
        }
    }

} // namespace llaminar2
