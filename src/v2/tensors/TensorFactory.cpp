/**
 * @file TensorFactory.cpp
 * @brief Factory implementation for creating tensors with NUMA-aware allocation
 * @author David Sanftenberg
 */

#include "TensorFactory.h"
#include "BlockStructures.h"
#include "TensorClasses.h" // For FP32Tensor::createMapped()
#include "../utils/Logger.h"
#include "../backends/p2p/DirectP2P.h" // For BAR-backed tensor support
#include <stdexcept>
#include <sstream>
#include <cstring> // For std::memcpy in numaMemcpy
#include <omp.h>   // For parallel numaMemcpy

#include <numa.h>
#include <numaif.h>
#include <sched.h> // For sched_getcpu() in NUMA node detection

#if defined(HAVE_ROCM)
#include <hip/hip_runtime.h>
#endif

namespace llaminar2
{
    TensorFactory::TensorFactory(const MPIContext &mpi_ctx)
        : mpi_rank_(mpi_ctx.rank()), numa_node_(-1)
    {
        // Determine NUMA node for this MPI rank
        numa_node_ = getNumaNodeForRank(mpi_rank_);

        // Bind current thread to NUMA node if available
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }
    }

    std::unique_ptr<FP32Tensor> TensorFactory::createFP32(const std::vector<size_t> &shape, DeviceId device)
    {
        // Ensure we're on the correct NUMA node
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        // For GPU devices, use mapped memory if enabled
        // Mapped memory enables zero-copy host access (avoids slow memcpy syncs)
        if (use_mapped_memory_for_gpu_ && device.is_gpu())
        {
            LOG_TRACE("[TensorFactory::createFP32] Using mapped memory for GPU tensor");
            return FP32Tensor::createMapped(shape, device);
        }

        return std::make_unique<FP32Tensor>(shape, device);
    }

    std::unique_ptr<FP16Tensor> TensorFactory::createFP16(const std::vector<size_t> &shape)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<FP16Tensor>(shape);
    }

    std::unique_ptr<FP16Tensor> TensorFactory::createFP16(const std::vector<size_t> &shape,
                                                          const std::vector<uint16_t> &fp16_data)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<FP16Tensor>(shape, fp16_data);
    }

    std::unique_ptr<BF16Tensor> TensorFactory::createBF16(const std::vector<size_t> &shape)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<BF16Tensor>(shape);
    }

    std::unique_ptr<BF16Tensor> TensorFactory::createBF16(const std::vector<size_t> &shape,
                                                          const std::vector<uint16_t> &bf16_data)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<BF16Tensor>(shape, bf16_data);
    }

    std::unique_ptr<INT32Tensor> TensorFactory::createINT32(const std::vector<size_t> &shape)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<INT32Tensor>(shape);
    }

    std::unique_ptr<Q8_1Tensor> TensorFactory::createQ8_1(const std::vector<size_t> &shape, DeviceId device)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        // Use the mutable activation buffer constructor (no raw_data, allocates internally)
        // This creates a Q8_1Tensor that supports mutable_data() and quantize_from_cache()
        return std::make_unique<Q8_1Tensor>(shape, device);
    }

    std::unique_ptr<Q8_1Tensor> TensorFactory::createQ8_1(const std::vector<size_t> &shape,
                                                          const std::vector<uint8_t> &raw_data)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<Q8_1Tensor>(shape, raw_data);
    }

    std::unique_ptr<Q16_1Tensor> TensorFactory::createQ16_1(const std::vector<size_t> &shape,
                                                            DeviceId device)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        // Use the mutable activation buffer constructor
        // Q16_1Tensor(shape, device) creates an empty tensor for residual accumulation
        return std::make_unique<Q16_1Tensor>(shape, device);
    }

    std::unique_ptr<Q16_1Tensor> TensorFactory::createQ16_1(const std::vector<size_t> &shape,
                                                            Q16BlockSize block_size,
                                                            DeviceId device)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        // Use the mutable activation buffer constructor with custom block size
        return std::make_unique<Q16_1Tensor>(shape, block_size, device);
    }

    std::unique_ptr<TensorBase> TensorFactory::createActivation(const std::vector<size_t> &shape,
                                                                ActivationPrecision precision,
                                                                DeviceId device)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        std::unique_ptr<TensorBase> tensor;

        switch (precision)
        {
        case ActivationPrecision::FP32:
            // FP32 createFP32 already accepts device
            return createFP32(shape, device);

        case ActivationPrecision::BF16:
            tensor = createBF16(shape);
            break;

        case ActivationPrecision::FP16:
            tensor = createFP16(shape);
            break;

        case ActivationPrecision::Q8_1:
            tensor = createQ8_1(shape, device);
            break;

        case ActivationPrecision::Q16_1:
            // Q16_1: High-precision quantized format for residual stream
            return createQ16_1(shape, device);

        case ActivationPrecision::HybridQ16:
            // For HybridQ16 mode, createActivation returns Q16_1 for residual buffers
            // Buffer allocation logic in DeviceGraphOrchestrator handles specific buffer types
            return createQ16_1(shape, device);

        default:
            LOG_ERROR("TensorFactory::createActivation: unknown precision, defaulting to FP32");
            return createFP32(shape, device);
        }

        // Note: Device is set through constructor for tensor types that support it
        // (FP32, Q8_1, Q16_1). Other types (BF16, FP16) are CPU-only for now.

        return tensor;
    }

    std::unique_ptr<TensorBase> TensorFactory::createActivation(const std::vector<size_t> &shape,
                                                                ActivationPrecision precision,
                                                                int head_dim,
                                                                DeviceId device)
    {
        // For Q16_1 and HybridQ16, use optimal block size based on head_dim
        if (precision == ActivationPrecision::Q16_1 ||
            precision == ActivationPrecision::HybridQ16)
        {
            Q16BlockSize block_size = optimal_q16_block_size(head_dim);
            LOG_DEBUG("TensorFactory::createActivation: Q16 with head_dim=" << head_dim
                                                                            << " -> block_size=" << static_cast<int>(block_size));
            return createQ16_1(shape, block_size, device);
        }

        // All other precisions: delegate to base overload
        return createActivation(shape, precision, device);
    }

    std::unique_ptr<TensorBase> TensorFactory::createQuantized(TensorType type,
                                                               const std::vector<size_t> &shape,
                                                               const std::vector<uint8_t> &raw_data)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        switch (type)
        {
        case TensorType::IQ4_NL:
            return std::make_unique<IQ4_NLTensor>(shape, raw_data);
        case TensorType::Q8_0:
            return std::make_unique<Q8_0Tensor>(shape, raw_data);
        case TensorType::Q4_0:
            return std::make_unique<Q4_0Tensor>(shape, raw_data);
        case TensorType::Q4_1:
            return std::make_unique<Q4_1Tensor>(shape, raw_data);
        case TensorType::Q5_0:
            return std::make_unique<Q5_0Tensor>(shape, raw_data);
        case TensorType::Q5_1:
            return std::make_unique<Q5_1Tensor>(shape, raw_data);
        case TensorType::Q6_K:
            return std::make_unique<Q6_KTensor>(shape, raw_data);
        case TensorType::Q2_K:
            return std::make_unique<Q2_KTensor>(shape, raw_data);
        case TensorType::Q5_K:
            return std::make_unique<Q5_KTensor>(shape, raw_data);
        case TensorType::Q3_K:
            return std::make_unique<Q3_KTensor>(shape, raw_data);
        case TensorType::Q4_K:
            return std::make_unique<Q4_KTensor>(shape, raw_data);
        case TensorType::Q8_K:
            return std::make_unique<Q8_KTensor>(shape, raw_data);
        case TensorType::IQ4_XS:
            return std::make_unique<IQ4_XSTensor>(shape, raw_data);
        case TensorType::IQ2_XXS:
            return std::make_unique<IQ2_XXSTensor>(shape, raw_data);
        case TensorType::IQ2_XS:
            return std::make_unique<IQ2_XSTensor>(shape, raw_data);
        case TensorType::IQ3_XXS:
            return std::make_unique<IQ3_XXSTensor>(shape, raw_data);
        case TensorType::IQ2_S:
            return std::make_unique<IQ2_STensor>(shape, raw_data);
        case TensorType::IQ3_S:
            return std::make_unique<IQ3_STensor>(shape, raw_data);
        case TensorType::IQ1_S:
            return std::make_unique<IQ1_STensor>(shape, raw_data);
        case TensorType::IQ1_M:
            return std::make_unique<IQ1_MTensor>(shape, raw_data);
        default:
            LOG_ERROR("TensorFactory::createQuantized: unsupported type " << static_cast<int>(type));
            std::ostringstream oss;
            oss << "TensorFactory::createQuantized: unsupported type " << static_cast<int>(type);
            throw std::runtime_error(oss.str());
        }
    }

    std::unique_ptr<TensorBase> TensorFactory::createQuantizedZeroCopy(
        TensorType type,
        const std::vector<size_t> &shape,
        const uint8_t *mmap_data,
        size_t byte_size,
        std::shared_ptr<void> mmap_lifetime_owner)
    {
        // No NUMA binding needed — data lives in mmap'd memory, not heap-allocated

        switch (type)
        {
        case TensorType::IQ4_NL:
            return std::make_unique<IQ4_NLTensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::Q8_0:
            return std::make_unique<Q8_0Tensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::Q4_0:
            return std::make_unique<Q4_0Tensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::Q4_1:
            return std::make_unique<Q4_1Tensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::Q5_0:
            return std::make_unique<Q5_0Tensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::Q5_1:
            return std::make_unique<Q5_1Tensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::Q6_K:
            return std::make_unique<Q6_KTensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::Q2_K:
            return std::make_unique<Q2_KTensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::Q5_K:
            return std::make_unique<Q5_KTensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::Q3_K:
            return std::make_unique<Q3_KTensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::Q4_K:
            return std::make_unique<Q4_KTensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::Q8_K:
            return std::make_unique<Q8_KTensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::IQ4_XS:
            return std::make_unique<IQ4_XSTensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::IQ2_XXS:
            return std::make_unique<IQ2_XXSTensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::IQ2_XS:
            return std::make_unique<IQ2_XSTensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::IQ3_XXS:
            return std::make_unique<IQ3_XXSTensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::IQ2_S:
            return std::make_unique<IQ2_STensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::IQ3_S:
            return std::make_unique<IQ3_STensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::IQ1_S:
            return std::make_unique<IQ1_STensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        case TensorType::IQ1_M:
            return std::make_unique<IQ1_MTensor>(shape, mmap_data, byte_size, std::move(mmap_lifetime_owner));
        default:
            LOG_ERROR("TensorFactory::createQuantizedZeroCopy: unsupported type " << static_cast<int>(type));
            std::ostringstream oss;
            oss << "TensorFactory::createQuantizedZeroCopy: unsupported type " << static_cast<int>(type);
            throw std::runtime_error(oss.str());
        }
    }

    void TensorFactory::ensureNumaBinding()
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }
    }

    void TensorFactory::numaMemcpy(void *dst, const void *src, size_t bytes)
    {
        // For small buffers, single-threaded memcpy is faster (avoids OMP overhead)
        constexpr size_t PARALLEL_THRESHOLD = 128 * 1024; // 128 KB
        if (bytes < PARALLEL_THRESHOLD)
        {
            std::memcpy(dst, src, bytes);
            return;
        }

        // Parallel memcpy ensures destination pages are first-touched by
        // multiple threads on the correct NUMA node (set via membind).
        // Each thread copies a contiguous chunk, triggering page faults
        // that place physical pages on the bound NUMA node.
        auto *d = static_cast<char *>(dst);
        const auto *s = static_cast<const char *>(src);

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            size_t chunk = bytes / nthreads;
            size_t start = tid * chunk;
            size_t end = (tid == nthreads - 1) ? bytes : start + chunk;
            std::memcpy(d + start, s + start, end - start);
        }
    }

    bool TensorFactory::isNumaAvailable()
    {
        if (numa_available() < 0)
        {
            return false;
        }

        // Check if system has multiple NUMA nodes
        int max_node = numa_max_node();
        return max_node > 0;
    }

    void TensorFactory::bindToNumaNode()
    {
        if (numa_node_ >= 0 && numa_available() >= 0)
        {
            // Bind current thread to NUMA node
            struct bitmask *mask = numa_allocate_nodemask();
            numa_bitmask_clearall(mask);
            numa_bitmask_setbit(mask, numa_node_);

            // Set memory binding policy for allocations
            numa_set_membind(mask);

            // Also bind CPU affinity (optional, helps with first-touch)
            // numa_run_on_node_mask(mask);

            numa_free_nodemask(mask);
        }
    }

    int TensorFactory::getNumaNodeForRank(int rank)
    {
        if (numa_available() < 0)
        {
            return -1;
        }

        int max_node = numa_max_node();
        if (max_node <= 0)
        {
            return -1; // Only one NUMA node
        }

        // Detect actual NUMA node from the CPU this thread is running on.
        // This correctly handles cases where process binding doesn't follow
        // rank order, e.g. `-d cpu:1` with a single rank bound to socket 1
        // (rank=0 but NUMA node=1).  The old round-robin mapping
        // (`rank % (max_node+1)`) would incorrectly return node 0.
        int cpu = sched_getcpu();
        if (cpu >= 0)
        {
            int node = numa_node_of_cpu(cpu);
            if (node >= 0 && node <= max_node)
            {
                LOG_DEBUG("[TensorFactory] Detected NUMA node " << node
                                                                << " for rank " << rank << " (cpu " << cpu << ")");
                return node;
            }
        }

        // Fallback: round-robin mapping (only reached if sched_getcpu fails)
        int fallback = rank % (max_node + 1);
        LOG_WARN("[TensorFactory] sched_getcpu() failed, using round-robin NUMA mapping: "
                 << "rank " << rank << " -> node " << fallback);
        return fallback;
    }

    // =========================================================================
    // BAR-Backed Tensor Support
    // =========================================================================

    void TensorFactory::setDirectP2P(std::shared_ptr<DirectP2PEngine> p2p)
    {
        direct_p2p_ = std::move(p2p);
        if (direct_p2p_)
        {
            LOG_DEBUG("TensorFactory: DirectP2PEngine context set for BAR-backed tensor allocation");
        }
        else
        {
            LOG_DEBUG("TensorFactory: DirectP2PEngine context cleared");
        }
    }

    bool TensorFactory::canCreateBARBacked() const
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        // Check if DirectP2P is set and has active BAR mapping
        if (!direct_p2p_)
        {
            return false;
        }
        return direct_p2p_->isPCIeBarActive();
#else
        // BAR-backed tensors require both CUDA and ROCm backends
        return false;
#endif
    }

    std::unique_ptr<FP32Tensor> TensorFactory::createFP32BARBacked(
        const std::vector<size_t> &shape,
        DeviceId rocm_device,
        DeviceId cuda_device)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        // Validate inputs
        if (!rocm_device.is_rocm())
        {
            LOG_ERROR("TensorFactory::createFP32BARBacked: rocm_device must be a ROCm device, got "
                      << rocm_device.to_string());
            return nullptr;
        }

        if (!cuda_device.is_cuda())
        {
            LOG_ERROR("TensorFactory::createFP32BARBacked: cuda_device must be a CUDA device, got "
                      << cuda_device.to_string());
            return nullptr;
        }

        if (!direct_p2p_)
        {
            LOG_ERROR("TensorFactory::createFP32BARBacked: DirectP2PEngine not set. "
                      "Call setDirectP2P() first.");
            throw std::runtime_error("TensorFactory::createFP32BARBacked: DirectP2PEngine not configured");
        }

        if (!direct_p2p_->isPCIeBarActive())
        {
            LOG_ERROR("TensorFactory::createFP32BARBacked: PCIe BAR not active. "
                      "Call DirectP2PEngine::initializePCIeBar() first.");
            throw std::runtime_error("TensorFactory::createFP32BARBacked: PCIe BAR not active");
        }

        // Calculate required size
        size_t element_count = 1;
        for (size_t dim : shape)
        {
            element_count *= dim;
        }
        size_t byte_size = element_count * sizeof(float);

        // Allocate from BAR region using bump allocator
        // This ensures each tensor gets a unique offset in the BAR region
        auto alloc = direct_p2p_->allocateInBar(byte_size, 256); // 256-byte alignment for GPU
        if (!alloc)
        {
            LOG_ERROR("TensorFactory::createFP32BARBacked: Failed to allocate " << byte_size
                                                                                << " bytes from BAR region (may be out of space)");
            throw std::runtime_error("TensorFactory::createFP32BARBacked: BAR allocation failed");
        }

        void *bar_host_ptr = alloc->first;  // ROCm BAR mmap pointer at allocated offset
        void *cuda_bar_ptr = alloc->second; // CUDA device pointer at allocated offset

        // KEY INSIGHT: The BAR mmap address is AMD GPU VRAM mapped to CPU virtual address space.
        // We discovered through testing (TestZeroCopyBidirectional) that hipMemcpy(D2D)
        // accepts this mmap CPU address directly as a "device" pointer - the ROCm driver
        // recognizes it as device memory without needing hipHostRegister.
        //
        // DO NOT use hipHostRegister here - it fails with "invalid argument" because
        // the BAR mmap is not normal host memory. Instead, just use the mmap address
        // directly as the ROCm pointer.
        void *hip_device_ptr = bar_host_ptr; // BAR mmap works directly with hipMemcpy D2D

        LOG_DEBUG("TensorFactory::createFP32BARBacked: Using BAR mmap directly for HIP: "
                  << "bar_host_ptr=" << bar_host_ptr << " (works with hipMemcpy D2D)");

        // Create BAR-backed FP32 tensor
        // Note: FP32Tensor::createBARBacked() expects a PCIeBARBackend* for full integration,
        // but for the TensorFactory API, we use DirectP2PEngine directly.
        // We'll create the tensor with BAR state populated via initBARBackedDirect.
        auto tensor = std::make_unique<FP32Tensor>(shape, rocm_device);

        // Configure the tensor for BAR-backed operation using direct pointers
        // - rocm_ptr: HIP device pointer (from hipHostGetDevicePointer)
        // - cuda_ptr: CUDA device pointer (from cuMemHostGetDevicePointer)
        tensor->initBARBackedDirect(
            hip_device_ptr, // rocm_ptr - HIP device pointer for ROCm kernels
            cuda_bar_ptr,   // cuda_ptr - CUDA reads/writes via PCIe
            rocm_device,    // host device (owns the BAR memory)
            cuda_device,    // accessor device (reads via PCIe)
            byte_size       // allocation size
        );

        LOG_DEBUG("TensorFactory::createFP32BARBacked: Created BAR-backed tensor "
                  << shape[0] << "x" << (shape.size() > 1 ? shape[1] : 1)
                  << " (" << byte_size << " bytes)"
                  << " rocm_ptr=" << hip_device_ptr
                  << " cuda_ptr=" << cuda_bar_ptr);

        return tensor;
#else
        LOG_ERROR("TensorFactory::createFP32BARBacked: Requires both HAVE_CUDA and HAVE_ROCM");
        throw std::runtime_error("TensorFactory::createFP32BARBacked: Requires HAVE_CUDA and HAVE_ROCM");
#endif
    }

} // namespace llaminar2
