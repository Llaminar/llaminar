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
#include "../../backends/BackendManager.h"

#include <algorithm>

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#include <cuda_runtime.h>
#include <cuda.h> // For cuCtxSetCurrent, cuDevicePrimaryCtxRetain
#endif

namespace llaminar2
{

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
    // Static instance pointer for cross-thread CUDA event wait proxying
    PCIeBARBackend *PCIeBARBackend::s_instance_ = nullptr;
#endif

    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    PCIeBARBackend::PCIeBARBackend(std::unique_ptr<DirectP2PEngine> p2p_engine)
    {
        // IMPORTANT: Always use the shared singleton instance
        // The DirectP2PEngine manages CUDA IOMEMORY registrations that cannot
        // be reliably re-initialized within a process. Using the singleton
        // ensures that:
        // 1. BAR resources are initialized once and shared across all backends
        // 2. Tests don't conflict by creating/destroying separate engines
        // 3. The engine lives for the entire process lifetime
        //
        // If a custom engine was passed in (for testing), we ignore it and
        // use the singleton anyway to prevent resource conflicts.
        if (p2p_engine)
        {
            LOG_WARN("[PCIeBARBackend] Ignoring custom p2p_engine - using shared singleton");
        }
        p2p_engine_ = DirectP2PEngine::getSharedInstance();
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
                // Cache in the engine for subsequent backends using the singleton
                p2p_engine_->setCachedBandwidthGBps(measured_bandwidth_gbps_);
                LOG_INFO("PCIe BAR P2P initialized: " << measured_bandwidth_gbps_ << " GB/s average");
            }
        }
        else
        {
            // Engine already active (singleton), retrieve cached bandwidth
            measured_bandwidth_gbps_ = p2p_engine_->getCachedBandwidthGBps();
            LOG_DEBUG("[PCIeBARBackend] Using cached bandwidth from singleton: "
                      << measured_bandwidth_gbps_ << " GB/s");
        }

        // Initialize BAR allocator state
        bar_host_ptr_ = p2p_engine_->getBarHostPtr();
        bar_total_mapped_size_ = p2p_engine_->getBarMappedSize();
        bar_alloc_offset_ = 0; // Start allocations at beginning of BAR

        // Start the CUDA worker thread BEFORE warmup or any GPU operations.
        // This thread handles CUDA event waits from HIP threads to avoid
        // "context is destroyed" errors from HIP runtime contamination.
        if (!startCUDAWorker())
        {
            LOG_ERROR("Failed to start CUDA worker thread");
            return false;
        }

        // Warmup the CUDA kernel by launching it once on the worker thread.
        // This forces CUDA to JIT-compile/load the PTX code BEFORE HIP initializes.
        {
            LOG_DEBUG("[PCIeBARBackend] Warming up CUDA VectorAdd kernel...");

            // Allocate small temp buffers for warmup
            void *warmup_buf1 = nullptr;
            void *warmup_buf2 = nullptr;

            cudaError_t set_err = cudaSetDevice(cuda_device_.ordinal);
            if (set_err != cudaSuccess)
            {
                LOG_WARN("[PCIeBARBackend] Could not set CUDA device for warmup: " << cudaGetErrorString(set_err));
            }

            cudaError_t alloc_err1 = cudaMalloc(&warmup_buf1, 64);
            cudaError_t alloc_err2 = cudaMalloc(&warmup_buf2, 64);

            if (alloc_err1 == cudaSuccess && alloc_err2 == cudaSuccess)
            {
                // Initialize to zero
                cudaMemset(warmup_buf1, 0, 64);
                cudaMemset(warmup_buf2, 0, 64);
                cudaDeviceSynchronize();

                // Launch kernel once via worker thread to force JIT compilation
                float *out = static_cast<float *>(warmup_buf1);
                const float *in2 = static_cast<const float *>(warmup_buf2);

                auto warmup_future = submitCUDAWork([this, out, in2]() -> bool
                                                    {
                    cudaError_t err = cudaSetDevice(cuda_device_.ordinal);
                    if (err != cudaSuccess) return false;

                    // Launch kernel with small count
                    bool success = cuda::launchVectorAddInplace_f32(out, in2, 16, 0);
                    if (!success)
                    {
                        LOG_ERROR("[PCIeBARBackend] CUDA kernel warmup FAILED - kernel may not work");
                        return false;
                    }

                    err = cudaDeviceSynchronize();
                    return err == cudaSuccess; });

                if (warmup_future.get())
                {
                    LOG_INFO("[PCIeBARBackend] CUDA VectorAdd kernel warmup successful");
                }
                else
                {
                    LOG_WARN("[PCIeBARBackend] CUDA kernel warmup failed - may have issues later");
                }

                cudaFree(warmup_buf1);
                cudaFree(warmup_buf2);
            }
            else
            {
                if (warmup_buf1)
                    cudaFree(warmup_buf1);
                if (warmup_buf2)
                    cudaFree(warmup_buf2);
                LOG_WARN("[PCIeBARBackend] Could not allocate warmup buffers");
            }
        }

        // Create persistent stream for reduction operations
        {
            cudaStream_t stream;
            cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (err == cudaSuccess)
            {
                cuda_reduction_stream_ = stream;
                LOG_DEBUG("[PCIeBARBackend] Created persistent reduction stream");
            }
            else
            {
                LOG_WARN("[PCIeBARBackend] Failed to create reduction stream: " << cudaGetErrorString(err)
                                                                                << " - will use default stream");
                cuda_reduction_stream_ = nullptr;
            }
        }

        // Create additional streams for pipelined transfers
        {
            cudaStream_t read_stream, write_stream;
            cudaError_t err1 = cudaStreamCreateWithFlags(&read_stream, cudaStreamNonBlocking);
            cudaError_t err2 = cudaStreamCreateWithFlags(&write_stream, cudaStreamNonBlocking);

            if (err1 == cudaSuccess && err2 == cudaSuccess)
            {
                cuda_read_stream_ = read_stream;
                cuda_write_stream_ = write_stream;
                LOG_DEBUG("[PCIeBARBackend] Created pipelined transfer streams");
            }
            else
            {
                LOG_WARN("[PCIeBARBackend] Failed to create pipeline streams - pipelined allreduce disabled");
                if (err1 == cudaSuccess)
                    cudaStreamDestroy(read_stream);
                if (err2 == cudaSuccess)
                    cudaStreamDestroy(write_stream);
                cuda_read_stream_ = nullptr;
                cuda_write_stream_ = nullptr;
            }

            // Create synchronization events (two per type for ping-pong buffers)
            cudaEvent_t read_events[2], compute_events[2];
            cudaError_t err3 = cudaEventCreateWithFlags(&read_events[0], cudaEventDisableTiming);
            cudaError_t err4 = cudaEventCreateWithFlags(&read_events[1], cudaEventDisableTiming);
            cudaError_t err5 = cudaEventCreateWithFlags(&compute_events[0], cudaEventDisableTiming);
            cudaError_t err6 = cudaEventCreateWithFlags(&compute_events[1], cudaEventDisableTiming);

            if (err3 == cudaSuccess && err4 == cudaSuccess &&
                err5 == cudaSuccess && err6 == cudaSuccess)
            {
                cuda_read_complete_event_[0] = read_events[0];
                cuda_read_complete_event_[1] = read_events[1];
                cuda_compute_complete_event_[0] = compute_events[0];
                cuda_compute_complete_event_[1] = compute_events[1];
                LOG_DEBUG("[PCIeBARBackend] Created pipeline synchronization events (2 per type)");
            }
            else
            {
                if (err3 == cudaSuccess)
                    cudaEventDestroy(read_events[0]);
                if (err4 == cudaSuccess)
                    cudaEventDestroy(read_events[1]);
                if (err5 == cudaSuccess)
                    cudaEventDestroy(compute_events[0]);
                if (err6 == cudaSuccess)
                    cudaEventDestroy(compute_events[1]);
                cuda_read_complete_event_[0] = nullptr;
                cuda_read_complete_event_[1] = nullptr;
                cuda_compute_complete_event_[0] = nullptr;
                cuda_compute_complete_event_[1] = nullptr;
            }
        }

        // Set global instance for cross-thread event wait proxying
        s_instance_ = this;

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
        // Clear global instance first
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (s_instance_ == this)
        {
            s_instance_ = nullptr;
        }

        // Stop CUDA worker thread first (before freeing resources it might use)
        stopCUDAWorker();

        // Destroy reduction stream
        if (cuda_reduction_stream_)
        {
            cudaStreamDestroy(static_cast<cudaStream_t>(cuda_reduction_stream_));
            cuda_reduction_stream_ = nullptr;
        }

        // Destroy pipeline streams
        if (cuda_read_stream_)
        {
            cudaStreamDestroy(static_cast<cudaStream_t>(cuda_read_stream_));
            cuda_read_stream_ = nullptr;
        }
        if (cuda_write_stream_)
        {
            cudaStreamDestroy(static_cast<cudaStream_t>(cuda_write_stream_));
            cuda_write_stream_ = nullptr;
        }

        // Destroy pipeline events (two per type)
        for (int i = 0; i < 2; ++i)
        {
            if (cuda_read_complete_event_[i])
            {
                cudaEventDestroy(static_cast<cudaEvent_t>(cuda_read_complete_event_[i]));
                cuda_read_complete_event_[i] = nullptr;
            }
            if (cuda_compute_complete_event_[i])
            {
                cudaEventDestroy(static_cast<cudaEvent_t>(cuda_compute_complete_event_[i]));
                cuda_compute_complete_event_[i] = nullptr;
            }
        }

        // Free second temp buffer
        if (cuda_temp_buffer2_)
        {
            IBackend *cuda_backend = getCUDABackend();
            if (cuda_backend)
            {
                cuda_backend->free(cuda_temp_buffer2_, cuda_device_.ordinal);
            }
            cuda_temp_buffer2_ = nullptr;
        }
#endif

        freeTempBuffer();

        // Clear BAR allocator state
        bar_alloc_offset_ = 0;
        bar_total_mapped_size_ = 0;
        bar_host_ptr_ = nullptr;
        bar_allocations_.clear();

        // Clear registered buffers
        registered_collectives_.clear();

        // NOTE: p2p_engine_ is a shared_ptr to the process-wide singleton.
        // We don't cleanup the engine here - it persists for the process lifetime
        // to avoid CUDA IOMEMORY re-registration issues. The shared_ptr will
        // decrement its reference count but the no-op deleter ensures the
        // singleton is never actually destroyed.
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

        // Use pipelined allreduce for large buffers if pipeline streams are available
        if (bytes >= PIPELINE_THRESHOLD && cuda_read_stream_ && cuda_write_stream_)
        {
            LOG_DEBUG("[PCIeBARBackend] Using pipelined allreduce for " << bytes << " bytes");
            return allreducePipelined(buffer, count, dtype, op);
        }

        // Sequential allreduce for small buffers (kernel launch overhead dominates)

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

        void *cuda_buffer = collective.cuda_buffer.ptr;
        size_t rocm_bar_offset = collective.rocm_buffer.bar_offset;

        // Use pipelined allreduce for large buffers if pipeline streams are available
        if (bytes >= PIPELINE_THRESHOLD && cuda_read_stream_ && cuda_write_stream_)
        {
            LOG_DEBUG("[allreduceRegistered] Using pipelined path for " << bytes << " bytes");
            return allreducePipelinedWithOffset(cuda_buffer, rocm_bar_offset, count, dtype, op);
        }

        // Sequential allreduce for small buffers

        // Ensure we have a temp buffer on CUDA side
        if (!ensureTempBuffer(bytes))
        {
            LOG_ERROR("allreduceRegistered: Failed to allocate temp buffer");
            return false;
        }

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
        // Synchronize CUDA device using the backend abstraction
        IBackend *cuda_backend = getCUDABackend();
        if (!cuda_backend)
        {
            LOG_ERROR("[PCIeBARBackend::synchronize] CUDA backend not available");
            return false;
        }

        // Use streamSynchronize for lighter-weight sync (only default stream)
        if (!cuda_backend->streamSynchronize(cuda_device_.ordinal))
        {
            LOG_ERROR("[PCIeBARBackend::synchronize] Stream synchronize failed");
            return false;
        }

        // Note: ROCm synchronization would be similar with ROCm backend
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

        LOG_DEBUG("[PCIeBARBackend::ensureTempBuffer] Need " << bytes << " bytes, have "
                                                             << cuda_temp_buffer_size_ << " bytes - GROWING buffer (never shrinks)");

        // GROW-ONLY: Allocate new larger buffer, free old one
        // This is called during hot-path if pre-reservation was insufficient
        IBackend *cuda_backend = getCUDABackend();
        if (!cuda_backend)
        {
            LOG_ERROR("[PCIeBARBackend::ensureTempBuffer] CUDA backend not available");
            return false;
        }

        // Allocate new buffer before freeing old one
        void *new_buffer = cuda_backend->allocate(bytes, cuda_device_.ordinal);
        if (!new_buffer)
        {
            LOG_ERROR("[PCIeBARBackend::ensureTempBuffer] Failed to allocate " << bytes << " bytes");
            return false;
        }

        // Free old buffer if it exists
        if (cuda_temp_buffer_)
        {
            cuda_backend->free(cuda_temp_buffer_, cuda_device_.ordinal);
        }

        cuda_temp_buffer_ = new_buffer;
        cuda_temp_buffer_size_ = bytes;
        LOG_DEBUG("[PCIeBARBackend::ensureTempBuffer] Buffer grown to " << bytes << " bytes at ptr="
                                                                        << std::hex << cuda_temp_buffer_ << std::dec);
        return true;
#else
        return false;
#endif
    }

    bool PCIeBARBackend::reserveTempBufferBytes(size_t bytes)
    {
        LOG_INFO("[PCIeBARBackend] Pre-reserving temp buffer: " << bytes << " bytes");
        return ensureTempBuffer(bytes);
    }

    void PCIeBARBackend::freeTempBuffer()
    {
#if defined(HAVE_CUDA)
        if (cuda_temp_buffer_)
        {
            LOG_TRACE("[PCIeBARBackend::freeTempBuffer] Freeing temp buffer ptr="
                      << std::hex << cuda_temp_buffer_ << std::dec
                      << " size=" << cuda_temp_buffer_size_);
            // Free using the backend abstraction
            IBackend *cuda_backend = getCUDABackend();
            if (cuda_backend)
            {
                cuda_backend->free(cuda_temp_buffer_, cuda_device_.ordinal);
            }
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
        // Only SUM is supported currently
        if (op != CollectiveOp::ALLREDUCE_SUM)
        {
            LOG_ERROR("PCIeBARBackend reduction only supports SUM currently");
            return false;
        }

        // Set device
        cudaError_t err = cudaSetDevice(cuda_device_.ordinal);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[reduceOnCUDA] cudaSetDevice failed: " << cudaGetErrorString(err));
            return false;
        }

        // Use persistent stream (or default if creation failed)
        cudaStream_t stream = cuda_reduction_stream_
                                  ? static_cast<cudaStream_t>(cuda_reduction_stream_)
                                  : nullptr;

        // output = output + input2 (in-place add)
        // Note: input1 == output for allreduce case
        bool success = false;
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
            success = cuda::launchVectorAddInplace_f32(
                static_cast<float *>(output),
                static_cast<const float *>(input2),
                count,
                stream);
            break;

        case CollectiveDataType::FLOAT16:
            success = cuda::launchVectorAddInplace_f16(
                output,
                input2,
                count,
                stream);
            break;

        case CollectiveDataType::BFLOAT16:
            success = cuda::launchVectorAddInplace_bf16(
                output,
                input2,
                count,
                stream);
            break;

        case CollectiveDataType::INT8:
            success = cuda::launchVectorAddInplace_i8(
                static_cast<int8_t *>(output),
                static_cast<const int8_t *>(input2),
                count,
                stream);
            break;

        case CollectiveDataType::INT32:
            success = cuda::launchVectorAddInplace_i32(
                static_cast<int32_t *>(output),
                static_cast<const int32_t *>(input2),
                count,
                stream);
            break;

        default:
            LOG_ERROR("[reduceOnCUDA] Unsupported dtype: " << static_cast<int>(dtype));
            return false;
        }

        if (!success)
        {
            LOG_ERROR("[reduceOnCUDA] Kernel launch failed");
            return false;
        }

        // Synchronize the stream to ensure reduction completes before next transfer
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[reduceOnCUDA] Stream sync failed: " << cudaGetErrorString(err));
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

    // =========================================================================
    // CUDA Worker Thread Implementation
    // =========================================================================
    //
    // When running heterogeneous CUDA+ROCm LOCAL TP, the ROCm executor thread
    // may need to wait for CUDA events. Direct cudaEventSynchronize() fails
    // with "context is destroyed" from a HIP-contaminated thread.
    // This worker thread provides a clean CUDA context for event waits.
    // =========================================================================

    bool PCIeBARBackend::startCUDAWorker()
    {
        if (cuda_worker_running_.load())
        {
            return true; // Already running
        }

        cuda_worker_stop_.store(false);

        try
        {
            cuda_worker_thread_ = std::thread(&PCIeBARBackend::cudaWorkerLoop, this);

            // Wait for worker to signal it's ready
            std::unique_lock<std::mutex> lock(cuda_work_mutex_);
            cuda_work_cv_.wait(lock, [this]()
                               { return cuda_worker_running_.load(); });

            LOG_TRACE("[PCIeBARBackend] CUDA worker thread started");
            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[PCIeBARBackend] Failed to start CUDA worker thread: " << e.what());
            return false;
        }
    }

    void PCIeBARBackend::stopCUDAWorker()
    {
        if (!cuda_worker_running_.load())
        {
            return; // Not running
        }

        // Signal worker to stop
        {
            std::lock_guard<std::mutex> lock(cuda_work_mutex_);
            cuda_worker_stop_.store(true);
        }
        cuda_work_cv_.notify_one();

        // Wait for worker to finish
        if (cuda_worker_thread_.joinable())
        {
            cuda_worker_thread_.join();
        }

        cuda_worker_running_.store(false);
        LOG_TRACE("[PCIeBARBackend] CUDA worker thread stopped");
    }

    std::future<bool> PCIeBARBackend::submitCUDAWork(std::function<bool()> work)
    {
        std::packaged_task<bool()> task(std::move(work));
        std::future<bool> future = task.get_future();

        {
            std::lock_guard<std::mutex> lock(cuda_work_mutex_);
            cuda_work_queue_.push(std::move(task));
        }
        cuda_work_cv_.notify_one();

        return future;
    }

    void PCIeBARBackend::cudaWorkerLoop()
    {
        // CRITICAL: This thread must initialize CUDA before any HIP contamination.
        // Use driver API to retain primary context, ensuring it won't be destroyed.

        // First, set the runtime API device
        cudaError_t err = cudaSetDevice(cuda_device_.ordinal);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[PCIeBARBackend] CUDA worker failed to set device " << cuda_device_.ordinal
                                                                           << ": " << cudaGetErrorString(err));
            cuda_worker_running_.store(true); // Signal ready (will fail operations)
            cuda_work_cv_.notify_all();
            return;
        }

        // Also set the driver API context for robustness
        CUdevice cu_device;
        CUresult cu_err = cuDeviceGet(&cu_device, cuda_device_.ordinal);
        if (cu_err == CUDA_SUCCESS)
        {
            CUcontext ctx;
            cu_err = cuDevicePrimaryCtxRetain(&ctx, cu_device);
            if (cu_err == CUDA_SUCCESS)
            {
                cu_err = cuCtxSetCurrent(ctx);
                if (cu_err == CUDA_SUCCESS)
                {
                    LOG_TRACE("[PCIeBARBackend] CUDA worker thread retained primary context");
                }
                else
                {
                    LOG_WARN("[PCIeBARBackend] cuCtxSetCurrent failed: " << cu_err);
                }
            }
            else
            {
                LOG_WARN("[PCIeBARBackend] cuDevicePrimaryCtxRetain failed: " << cu_err);
            }
        }
        else
        {
            LOG_WARN("[PCIeBARBackend] cuDeviceGet failed: " << cu_err);
        }

        // Create a dedicated stream for worker operations (non-blocking)
        cudaStream_t stream;
        err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (err == cudaSuccess)
        {
            cuda_worker_stream_ = stream;
        }
        else
        {
            LOG_WARN("[PCIeBARBackend] Failed to create worker stream: " << cudaGetErrorString(err));
            cuda_worker_stream_ = nullptr;
        }

        // Signal ready
        cuda_worker_running_.store(true);
        cuda_work_cv_.notify_all();

        // Work loop
        while (true)
        {
            std::packaged_task<bool()> task;

            {
                std::unique_lock<std::mutex> lock(cuda_work_mutex_);
                cuda_work_cv_.wait(lock, [this]()
                                   { return cuda_worker_stop_.load() || !cuda_work_queue_.empty(); });

                if (cuda_worker_stop_.load() && cuda_work_queue_.empty())
                {
                    break;
                }

                if (!cuda_work_queue_.empty())
                {
                    task = std::move(cuda_work_queue_.front());
                    cuda_work_queue_.pop();
                }
            }

            if (task.valid())
            {
                // Execute the task (context is already set on this thread)
                task();
            }
        }

        // Cleanup stream
        if (cuda_worker_stream_)
        {
            cudaStreamDestroy(static_cast<cudaStream_t>(cuda_worker_stream_));
            cuda_worker_stream_ = nullptr;
        }
    }

    // =========================================================================
    // CUDA Event Wait Implementation (via Worker Thread)
    // =========================================================================
    //
    // Routes CUDA event waits through the dedicated worker thread to avoid
    // HIP runtime contamination issues.
    // =========================================================================

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

    bool PCIeBARBackend::waitForCUDAEvent(void *event, int device_id)
    {
        if (!cuda_worker_running_.load())
        {
            LOG_ERROR("[PCIeBARBackend::waitForCUDAEvent] CUDA worker thread not running");
            return false;
        }

        if (!event)
        {
            // No event to wait for - consider this success
            return true;
        }

        cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
        int captured_device_id = device_id;

        // Submit the event wait to the worker thread
        auto future = submitCUDAWork([cuda_event, captured_device_id, this]() -> bool
                                     {
            // Use CUDA Driver API for more robust context handling.
            // The runtime API's cudaSetDevice may not properly restore context
            // after HIP has contaminated the process.
            
            // First, try to get/set the primary context via driver API
            CUdevice cu_device;
            CUresult cu_err = cuDeviceGet(&cu_device, captured_device_id);
            if (cu_err != CUDA_SUCCESS)
            {
                LOG_ERROR("[waitForCUDAEvent/worker] cuDeviceGet(" << captured_device_id 
                          << ") failed: " << cu_err);
                return false;
            }

            // Retain and set the primary context explicitly
            CUcontext ctx;
            cu_err = cuDevicePrimaryCtxRetain(&ctx, cu_device);
            if (cu_err != CUDA_SUCCESS)
            {
                LOG_ERROR("[waitForCUDAEvent/worker] cuDevicePrimaryCtxRetain failed: " << cu_err);
                return false;
            }

            cu_err = cuCtxSetCurrent(ctx);
            if (cu_err != CUDA_SUCCESS)
            {
                LOG_ERROR("[waitForCUDAEvent/worker] cuCtxSetCurrent failed: " << cu_err);
                cuDevicePrimaryCtxRelease(cu_device);
                return false;
            }

            // Use driver API for event synchronization instead of runtime API.
            // This is more robust when HIP has contaminated the process.
            CUevent cu_event = static_cast<CUevent>(cuda_event);
            
            // Check event status
            cu_err = cuEventQuery(cu_event);
            if (cu_err != CUDA_SUCCESS && cu_err != CUDA_ERROR_NOT_READY)
            {
                // If the event is somehow invalid, try a full device sync instead
                LOG_WARN("[waitForCUDAEvent/worker] cuEventQuery failed: " << cu_err
                          << ", falling back to full device sync");
                cu_err = cuCtxSynchronize();
                if (cu_err != CUDA_SUCCESS)
                {
                    LOG_ERROR("[waitForCUDAEvent/worker] cuCtxSynchronize failed: " << cu_err);
                    cuDevicePrimaryCtxRelease(cu_device);
                    return false;
                }
                return true;
            }

            // Wait for event to complete
            cu_err = cuEventSynchronize(cu_event);
            if (cu_err != CUDA_SUCCESS)
            {
                LOG_WARN("[waitForCUDAEvent/worker] cuEventSynchronize failed: " << cu_err
                          << ", falling back to full device sync");
                cu_err = cuCtxSynchronize();
                if (cu_err != CUDA_SUCCESS)
                {
                    LOG_ERROR("[waitForCUDAEvent/worker] cuCtxSynchronize fallback failed: " << cu_err);
                    cuDevicePrimaryCtxRelease(cu_device);
                    return false;
                }
            }
            
            return true; });

        // Wait for worker to complete
        return future.get();
    }

    // =========================================================================
    // Pipelined AllReduce Implementation
    // =========================================================================

    bool PCIeBARBackend::allreducePipelined(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        // For unregistered allreduce, assume ROCm buffer is at BAR offset 0
        return allreducePipelinedWithOffset(buffer, 0, count, dtype, op);
    }

    bool PCIeBARBackend::allreducePipelinedWithOffset(
        void *cuda_buffer,
        size_t rocm_bar_offset,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        (void)op; // Only SUM is supported

        size_t element_size = datatypeSize(dtype);
        size_t total_bytes = count * element_size;

        // Calculate chunk parameters
        // Align chunk size to element boundaries
        size_t chunk_bytes = std::min(PIPELINE_CHUNK_SIZE, total_bytes);
        size_t elements_per_chunk = chunk_bytes / element_size;
        chunk_bytes = elements_per_chunk * element_size; // Aligned to elements

        size_t num_chunks = (total_bytes + chunk_bytes - 1) / chunk_bytes;

        LOG_DEBUG("[allreducePipelined] total=" << total_bytes << " bytes, chunk="
                                                << chunk_bytes << " bytes, num_chunks=" << num_chunks);

        // Ensure we have double buffers for pipelining
        if (!ensureTempBuffer(chunk_bytes))
        {
            LOG_ERROR("[allreducePipelined] Failed to allocate temp buffer 1");
            return false;
        }

        // Allocate second temp buffer if needed
        if (!cuda_temp_buffer2_ || cuda_temp_buffer_size_ < chunk_bytes)
        {
            if (cuda_temp_buffer2_)
            {
                IBackend *cuda_backend = getCUDABackend();
                if (cuda_backend)
                {
                    cuda_backend->free(cuda_temp_buffer2_, cuda_device_.ordinal);
                }
            }

            IBackend *cuda_backend = getCUDABackend();
            if (!cuda_backend)
            {
                LOG_ERROR("[allreducePipelined] No CUDA backend available");
                return false;
            }

            cuda_temp_buffer2_ = cuda_backend->allocate(chunk_bytes, cuda_device_.ordinal);
            if (!cuda_temp_buffer2_)
            {
                LOG_ERROR("[allreducePipelined] Failed to allocate temp buffer 2");
                return false;
            }
        }

        cudaStream_t read_stream = static_cast<cudaStream_t>(cuda_read_stream_);
        cudaStream_t compute_stream = static_cast<cudaStream_t>(cuda_reduction_stream_);
        cudaStream_t write_stream = static_cast<cudaStream_t>(cuda_write_stream_);

        // Events per ping-pong buffer (index by buffer_id = chunk % 2)
        cudaEvent_t read_events[2] = {
            static_cast<cudaEvent_t>(cuda_read_complete_event_[0]),
            static_cast<cudaEvent_t>(cuda_read_complete_event_[1])};
        cudaEvent_t compute_events[2] = {
            static_cast<cudaEvent_t>(cuda_compute_complete_event_[0]),
            static_cast<cudaEvent_t>(cuda_compute_complete_event_[1])};

        // Get CUDA-accessible BAR pointer for async transfers
        void *cuda_bar_ptr = p2p_engine_->getCudaBarPointer();
        if (!cuda_bar_ptr)
        {
            LOG_ERROR("[allreducePipelined] No CUDA BAR pointer available");
            return false;
        }

        // Ping-pong between temp buffers
        void *temp_buffers[2] = {cuda_temp_buffer_, cuda_temp_buffer2_};

        // Pipeline stages:
        // Stage 0: Read chunk 0
        // Stage 1: Read chunk 1 | Compute chunk 0
        // Stage 2: Read chunk 2 | Compute chunk 1 | Write chunk 0
        // ...
        // Final-2: Compute[N-1] | Write[N-2]
        // Final-1: Write[N-1]

        LOG_DEBUG("[allreducePipelined] total=" << total_bytes << " bytes, chunk="
                                                << chunk_bytes << " bytes, num_chunks=" << num_chunks
                                                << ", rocm_bar_offset=" << rocm_bar_offset);

        for (size_t stage = 0; stage < num_chunks + 2; ++stage)
        {
            // Read stage: stage < num_chunks
            if (stage < num_chunks)
            {
                size_t chunk_offset = stage * chunk_bytes;
                size_t this_chunk_bytes = std::min(chunk_bytes, total_bytes - chunk_offset);
                size_t buffer_id = stage % 2;
                void *temp_buf = temp_buffers[buffer_id];

                void *bar_src = static_cast<char *>(cuda_bar_ptr) + rocm_bar_offset + chunk_offset;

                // Async read from BAR to temp buffer
                cudaMemcpyAsync(temp_buf, bar_src, this_chunk_bytes,
                                cudaMemcpyDeviceToDevice, read_stream);
                // Record event for THIS buffer
                cudaEventRecord(read_events[buffer_id], read_stream);
            }

            // Compute stage: stage >= 1 && stage < num_chunks + 1
            if (stage >= 1 && stage < num_chunks + 1)
            {
                size_t compute_chunk = stage - 1;
                size_t chunk_offset = compute_chunk * chunk_bytes;
                size_t this_chunk_bytes = std::min(chunk_bytes, total_bytes - chunk_offset);
                size_t this_chunk_elements = this_chunk_bytes / element_size;
                size_t buffer_id = compute_chunk % 2;
                void *temp_buf = temp_buffers[buffer_id];
                char *output_ptr = static_cast<char *>(cuda_buffer) + chunk_offset;

                // Wait for THIS buffer's read to complete before computing
                cudaStreamWaitEvent(compute_stream, read_events[buffer_id], 0);

                // Async reduction kernel
                reduceOnCUDAAsync(output_ptr, temp_buf, this_chunk_elements, dtype, compute_stream);
                // Record event for THIS buffer's compute
                cudaEventRecord(compute_events[buffer_id], compute_stream);
            }

            // Write stage: stage >= 2
            if (stage >= 2)
            {
                size_t write_chunk = stage - 2;
                size_t chunk_offset = write_chunk * chunk_bytes;
                size_t this_chunk_bytes = std::min(chunk_bytes, total_bytes - chunk_offset);
                size_t buffer_id = write_chunk % 2;
                char *output_ptr = static_cast<char *>(cuda_buffer) + chunk_offset;

                // Wait for THIS buffer's compute to complete before writing
                cudaStreamWaitEvent(write_stream, compute_events[buffer_id], 0);

                void *bar_dst = static_cast<char *>(cuda_bar_ptr) + rocm_bar_offset + chunk_offset;

                // Async write to BAR
                cudaMemcpyAsync(bar_dst, output_ptr, this_chunk_bytes, cudaMemcpyDeviceToDevice, write_stream);
            }
        }

        // Synchronize all streams
        cudaStreamSynchronize(read_stream);
        cudaStreamSynchronize(compute_stream);
        cudaStreamSynchronize(write_stream);

        return true;
    }

    bool PCIeBARBackend::reduceOnCUDAAsync(
        void *output,
        const void *input,
        size_t count,
        CollectiveDataType dtype,
        void *stream_ptr)
    {
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
            return cuda::launchVectorAddInplace_f32(
                static_cast<float *>(output),
                static_cast<const float *>(input),
                count, stream);

        case CollectiveDataType::FLOAT16:
            return cuda::launchVectorAddInplace_f16(output, input, count, stream);

        case CollectiveDataType::BFLOAT16:
            return cuda::launchVectorAddInplace_bf16(output, input, count, stream);

        case CollectiveDataType::INT8:
            return cuda::launchVectorAddInplace_i8(
                static_cast<int8_t *>(output),
                static_cast<const int8_t *>(input),
                count, stream);

        case CollectiveDataType::INT32:
            return cuda::launchVectorAddInplace_i32(
                static_cast<int32_t *>(output),
                static_cast<const int32_t *>(input),
                count, stream);

        default:
            LOG_ERROR("[reduceOnCUDAAsync] Unsupported dtype: " << static_cast<int>(dtype));
            return false;
        }
    }

    bool PCIeBARBackend::transferROCmtoCUDAAsync(size_t bar_offset, void *cuda_dst, size_t bytes, void *stream_ptr)
    {
        if (!p2p_engine_ || !p2p_engine_->isPCIeBarActive())
        {
            return false;
        }

        void *cuda_bar_ptr = p2p_engine_->getCudaBarPointer();
        if (!cuda_bar_ptr)
        {
            return false;
        }

        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
        cudaError_t err = cudaMemcpyAsync(cuda_dst,
                                          static_cast<char *>(cuda_bar_ptr) + bar_offset,
                                          bytes, cudaMemcpyDeviceToDevice, stream);
        return err == cudaSuccess;
    }

    bool PCIeBARBackend::transferCUDAtoROCmAsync(const void *cuda_src, size_t bar_offset, size_t bytes, void *stream_ptr)
    {
        if (!p2p_engine_ || !p2p_engine_->isPCIeBarActive())
        {
            return false;
        }

        void *cuda_bar_ptr = p2p_engine_->getCudaBarPointer();
        if (!cuda_bar_ptr)
        {
            return false;
        }

        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
        cudaError_t err = cudaMemcpyAsync(static_cast<char *>(cuda_bar_ptr) + bar_offset,
                                          cuda_src, bytes, cudaMemcpyDeviceToDevice, stream);
        return err == cudaSuccess;
    }

#endif // HAVE_CUDA && HAVE_ROCM

} // namespace llaminar2
