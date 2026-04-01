/**
 * @file BARBackedTensor.h
 * @brief BAR-backed tensor design and implementation for zero-copy PCIeBAR allreduce
 *
 * ============================================================================
 * DESIGN DOCUMENT: BAR-Backed Tensors for Heterogeneous Tensor Parallelism
 * ============================================================================
 *
 * ## 1. Motivation
 *
 * In PCIeBAR heterogeneous tensor parallelism (NVIDIA ↔ AMD GPUs), the
 * allreduce pattern typically requires:
 *
 *   1. CUDA kernel writes output to CUDA device memory
 *   2. Copy CUDA device memory → system memory (staging)
 *   3. Copy system memory → AMD VRAM via BAR
 *   4. AMD accumulates, writes back
 *   5. Copy AMD VRAM via BAR → system memory
 *   6. Copy system memory → CUDA device memory
 *
 * With BAR-backed tensors, CUDA kernels can write **directly** to the BAR:
 *
 *   1. CUDA kernel writes output directly to BAR-mapped buffer
 *      (writes go directly to AMD VRAM via PCIe posted writes ~2.65 GB/s)
 *   2. AMD reads from its VRAM (local access, fast)
 *   3. AMD accumulates, writes to its local buffer
 *   4. CUDA reads result via BAR (~2.67 GB/s with rBAR)
 *
 * This eliminates 2 copy operations (staging through system memory).
 *
 * ## 2. Memory Model Comparison
 *
 * ### 2.1 Existing Mapped Memory (cudaHostAllocMapped)
 *
 * ```
 * ┌──────────────────────────────────────────────────────────────────────┐
 * │                        System (Host) Memory                          │
 * │   ┌──────────────────────────────────────────────────────────────┐   │
 * │   │  cudaHostAlloc(cudaHostAllocMapped) buffer                   │   │
 * │   │  • Physically in DRAM                                        │   │
 * │   │  • CPU can read/write directly                               │   │
 * │   │  • GPU accesses via PCIe (slower for GPU, uncached)          │   │
 * │   └──────────────────────────────────────────────────────────────┘   │
 * │                    ↑                               ↑                  │
 * │                    │ PCIe                          │ PCIe             │
 * │                    │                               │                  │
 * │        ┌───────────┴───────────┐       ┌──────────┴──────────┐       │
 * │        │ CUDA GPU             │       │ AMD GPU (no access) │       │
 * │        │ cudaHostGetDevicePtr │       │                     │       │
 * │        └──────────────────────┘       └─────────────────────┘       │
 * └──────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Characteristics:**
 * - Buffer lives in system DRAM
 * - CUDA accesses via PCIe (uncached, ~12-16 GB/s to/from CUDA)
 * - AMD GPU has NO direct access (can't register DRAM as IoMemory)
 * - Good for: single-GPU scenarios, CPU-GPU sharing
 *
 * ### 2.2 BAR-Backed Memory (New)
 *
 * ```
 * ┌──────────────────────────────────────────────────────────────────────┐
 * │                         AMD GPU VRAM                                 │
 * │   ┌──────────────────────────────────────────────────────────────┐   │
 * │   │  BAR-mapped region (e.g., first 1GB of VRAM)                 │   │
 * │   │  • Physically in AMD VRAM                                    │   │
 * │   │  • mmap'd into host address space via BAR0                   │   │
 * │   │  • Registered with CUDA via cuMemHostRegister(IOMEMORY)      │   │
 * │   └──────────────────────────────────────────────────────────────┘   │
 * │                    ↑                               ↑                  │
 * │                    │ PCIe BAR                      │ Local VRAM       │
 * │                    │ (~2.65 GB/s)                  │ (~900 GB/s)      │
 * │        ┌───────────┴───────────┐       ┌──────────┴──────────┐       │
 * │        │ CUDA GPU              │       │ AMD GPU             │       │
 * │        │ cuMemHostGetDevicePtr │       │ Direct local access │       │
 * │        │ (writes to AMD VRAM!) │       │                     │       │
 * │        └──────────────────────┘       └─────────────────────┘       │
 * └──────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Characteristics:**
 * - Buffer lives in **AMD VRAM** (not system DRAM)
 * - CUDA writes go directly to AMD VRAM via PCIe posted writes (~2.65 GB/s)
 * - AMD accesses locally (full VRAM bandwidth ~900 GB/s)
 * - Host access via mmap (O_SYNC, uncached, ~1-2 GB/s)
 *
 * ## 3. API Design
 *
 * ### 3.1 Factory Method
 *
 * ```cpp
 * // Create a BAR-backed FP32 tensor
 * auto tensor = FP32Tensor::createBARBacked(
 *     {seq_len, hidden_dim},   // shape
 *     cuda_device,              // CUDA device that will write to this tensor
 *     rocm_device,              // ROCm device whose BAR hosts this tensor
 *     pcie_bar_backend          // PCIeBARBackend that manages the BAR
 * );
 *
 * // Tensor can be used like any other:
 * void* cuda_ptr = tensor->gpu_data_ptr();  // CUDA-accessible BAR pointer
 * void* rocm_ptr = tensor->rocm_data_ptr(); // ROCm-accessible local pointer
 * ```
 *
 * ### 3.2 Coherence Model
 *
 * BAR-backed tensors have a **write-through** model:
 *
 * | Operation | Behavior |
 * |-----------|----------|
 * | `ensureOnDevice(cuda)` | No-op (CUDA pointer always valid) |
 * | `ensureOnDevice(rocm)` | No-op (ROCm pointer is local VRAM access) |
 * | `transitionTo(MAPPED)` | Sets coherence state, NO copy needed |
 * | `ensureOnHost()` | Synchronize (event or full sync), then access via mmap |
 * | `data()` | Returns mmap'd host pointer (slow, uncached access) |
 *
 * **Key insight:** Unlike normal tensors, `transitionTo(MAPPED)` does NOT
 * invalidate other views. Both CUDA and ROCm see the same physical memory.
 *
 * ### 3.3 When to Use BAR-Backed Tensors
 *
 * **DO use for:**
 * - Allreduce output buffers (write once by CUDA, read by AMD)
 * - Allgather receive buffers (CUDA writes, AMD reads local portion)
 * - Small-to-medium buffers where DMA setup overhead > transfer time
 *   (rough threshold: <64KB per buffer)
 *
 * **DON'T use for:**
 * - Weight tensors (need GPU cache locality, reread many times)
 * - Large activation tensors (uncached writes hurt bandwidth-bound kernels)
 * - Single-GPU scenarios (no benefit, adds complexity)
 *
 * ### 3.4 Performance Considerations
 *
 * **Writing to BAR (CUDA → AMD VRAM):**
 * - PCIe posted writes, ~2.65 GB/s on PCIe 3.0 x16
 * - **Uncached writes** - each store goes directly to PCIe, no coalescing
 * - Memory-bound kernels may be slower than writing to local VRAM first
 * - Best for: small buffers, infrequent writes, allreduce patterns
 *
 * **Reading from BAR (CUDA ← AMD VRAM):**
 * - ~2.67 GB/s with Resizable BAR (symmetric with writes)
 * - ~0.8 GB/s WITHOUT rBAR (PCIe completion packets are slow)
 * - Check `DirectP2PCapability::pcie_bar_iomemory_supported`
 *
 * **Host access:**
 * - mmap with O_SYNC - uncached, ~1-2 GB/s
 * - Avoid frequent `data()` calls in hot paths
 * - Use only for debugging/serialization
 *
 * ## 4. Implementation Details
 *
 * ### 4.1 State Variables (added to TensorBase or derived class)
 *
 * ```cpp
 * // BAR-backed memory state
 * bool is_bar_backed_ = false;           // True if using BAR-backed allocation
 * size_t bar_offset_ = 0;                // Offset within BAR region
 * size_t bar_size_ = 0;                  // Allocated size in BAR
 * void* bar_rocm_ptr_ = nullptr;         // ROCm-accessible pointer (mmap'd BAR)
 * void* bar_cuda_device_ptr_ = nullptr;  // CUDA-accessible pointer (from cuMemHostGetDevicePointer)
 * DeviceId bar_host_device_;             // ROCm device whose BAR this is
 * DeviceId bar_accessor_device_;         // CUDA device with registered access
 * PCIeBARBackend* bar_backend_ = nullptr; // Backend that manages the BAR
 * ```
 *
 * ### 4.2 Allocation Flow
 *
 * ```
 * createBARBacked(shape, cuda_device, rocm_device, backend)
 *     │
 *     ├──> Calculate size: count * sizeof(float)
 *     │
 *     ├──> backend->allocateInBarRegion(size)
 *     │        │
 *     │        └──> Returns (rocm_ptr, bar_offset)
 *     │
 *     ├──> bar_cuda_device_ptr_ = engine->getCudaBarPointer() + offset
 *     │
 *     ├──> Set state:
 *     │      is_bar_backed_ = true
 *     │      bar_rocm_ptr_ = rocm_ptr
 *     │      bar_offset_ = offset
 *     │      gpu_data_ptr_ = bar_cuda_device_ptr_  // For CUDA kernel dispatch
 *     │      gpu_device_ = cuda_device
 *     │
 *     └──> Return tensor
 * ```
 *
 * ### 4.3 Integration with Graph Execution
 *
 * Stages that output to BAR-backed tensors work normally:
 *
 * 1. `StageCoherence::ensureInputsOnDevice()` - inputs not affected
 * 2. Stage executes, writes to `tensor->gpu_data_ptr()` (which is BAR pointer)
 * 3. `StageCoherence::markOutputsDirty()` - marks tensor dirty
 *
 * The magic is that `gpu_data_ptr()` returns the BAR-mapped CUDA pointer,
 * so CUDA kernels transparently write to AMD VRAM.
 *
 * ## 5. Safety and Error Handling
 *
 * ### 5.1 Capability Checks
 *
 * ```cpp
 * auto caps = DirectP2PEngine::probeCapabilities();
 * if (!caps.pcie_bar_accessible || !caps.pcie_bar_iomemory_supported) {
 *     // Fall back to host-staged allreduce
 * }
 * ```
 *
 * ### 5.2 Lifetime Management
 *
 * - BAR allocations must be freed before PCIeBARBackend::shutdown()
 * - Tensor destructor calls backend->freeBarBuffer()
 * - Backend tracks allocations via bump allocator
 *
 * ### 5.3 Thread Safety
 *
 * - CUDA operations on BAR memory must use the CUDA worker thread
 *   (see PCIeBARBackend::submitCUDAWork)
 * - ROCm operations can run on any thread (local VRAM access)
 * - Host access (data()) requires synchronization first
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "TensorClasses.h"
#include "../collective/backends/PCIeBARBackend.h"
#include "../backends/p2p/DirectP2P.h"
#include <optional>

namespace llaminar2
{

    /**
     * @brief Allocation info for a BAR-backed buffer
     *
     * Tracks the BAR region allocation for a tensor, including both
     * the CUDA-accessible and ROCm-accessible pointers.
     */
    struct BARAllocationInfo
    {
        /// Offset within the BAR-mapped region (for bump allocator tracking)
        size_t offset = 0;

        /// Size of the allocation in bytes
        size_t size = 0;

        /// ROCm-accessible pointer (direct mmap of BAR, points to AMD VRAM)
        /// This is the "host" pointer from PCIeBARBackend::allocateInBarRegion()
        void *rocm_ptr = nullptr;

        /// CUDA-accessible pointer (from cuMemHostGetDevicePointer after IOMEMORY registration)
        /// CUDA kernels use this to write directly to AMD VRAM
        void *cuda_device_ptr = nullptr;

        /// The ROCm device whose VRAM hosts this buffer
        DeviceId host_device;

        /// The CUDA device that has registered access to this BAR
        DeviceId accessor_device;

        /// Backend managing this allocation (for deallocation)
        PCIeBARBackend *backend = nullptr;

        /// True if this is a valid allocation
        bool isValid() const
        {
            return rocm_ptr != nullptr && cuda_device_ptr != nullptr && backend != nullptr;
        }
    };

    /**
     * @brief Configuration for creating a BAR-backed tensor
     */
    struct BARBackedTensorConfig
    {
        /// Shape of the tensor
        std::vector<size_t> shape;

        /// CUDA device that will write to this tensor
        DeviceId cuda_device;

        /// ROCm device whose BAR will host this tensor
        DeviceId rocm_device;

        /// PCIeBARBackend managing the BAR (must outlive the tensor)
        PCIeBARBackend *backend = nullptr;

        /// Optional: pre-allocated BAR region (if nullptr, will allocate)
        void *pre_allocated_rocm_ptr = nullptr;
        size_t pre_allocated_offset = 0;
    };

    //=========================================================================
    // BAR-Backed Tensor Extension Methods (added to FP32Tensor)
    //=========================================================================

    /**
     * @brief Mixin class providing BAR-backed tensor functionality
     *
     * This class provides the BAR-specific state and methods that can be
     * composed with existing tensor classes. It's designed to be inherited
     * alongside the regular tensor inheritance hierarchy.
     *
     * Usage pattern:
     * - FP32Tensor::createBARBacked() creates a tensor that has BAR state
     * - The tensor's gpu_data_ptr() returns the CUDA-accessible BAR pointer
     * - rocm_data_ptr() returns the ROCm-accessible local pointer
     *
     * Implementation note:
     * Rather than creating a full BARBackedFP32Tensor subclass, we add
     * BAR state directly to TensorBase and activate it via factory methods.
     * This keeps the tensor type system simple while enabling BAR functionality.
     */
    class IBARBackedTensor
    {
    public:
        virtual ~IBARBackedTensor() = default;

        /**
         * @brief Check if this tensor uses BAR-backed memory
         * @return true if tensor data resides in AMD VRAM accessed via BAR
         */
        virtual bool isBARBacked() const = 0;

        /**
         * @brief Get the ROCm-accessible pointer to tensor data
         *
         * This pointer is directly accessible by ROCm kernels since
         * it points to local AMD VRAM. For non-BAR-backed tensors,
         * returns nullptr.
         *
         * @return void* ROCm device pointer, or nullptr if not BAR-backed
         */
        virtual void *rocm_data_ptr() = 0;
        virtual const void *rocm_data_ptr() const = 0;

        /**
         * @brief Get the BAR allocation info for this tensor
         *
         * @return BARAllocationInfo with offset, pointers, and backend
         */
        virtual const BARAllocationInfo &barAllocationInfo() const = 0;

        /**
         * @brief Get the CUDA device that has write access to this BAR
         *
         * @return DeviceId of the CUDA accessor device
         */
        virtual DeviceId barAccessorDevice() const = 0;

        /**
         * @brief Get the ROCm device whose VRAM hosts this buffer
         *
         * @return DeviceId of the ROCm host device
         */
        virtual DeviceId barHostDevice() const = 0;
    };

    //=========================================================================
    // TensorBase BAR Extension (protected state added to TensorBase)
    //=========================================================================
    //
    // The following state variables would be added to TensorBase's protected
    // section. For this prototype, we document them here and implement via
    // a derived class approach.
    //
    // FUTURE: These could be added directly to TensorBase:
    //
    // protected:
    //     // BAR-backed memory state (active when is_bar_backed_ == true)
    //     bool is_bar_backed_ = false;
    //     BARAllocationInfo bar_info_;
    //

    //=========================================================================
    // Utility Functions
    //=========================================================================

    /**
     * @brief Check if PCIeBAR-backed tensors are available
     *
     * Queries the DirectP2PEngine capabilities to determine if BAR-backed
     * tensors can be created on this system.
     *
     * @return true if system supports BAR-backed tensors
     */
    inline bool isPCIeBarTensorSupported()
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        auto caps = DirectP2PEngine::probeCapabilities();
        return caps.pcie_bar_accessible && caps.pcie_bar_iomemory_supported;
#else
        return false;
#endif
    }

    /**
     * @brief Calculate recommended maximum size for BAR-backed tensors
     *
     * BAR-backed tensors have performance trade-offs. For large tensors,
     * the uncached write penalty may outweigh the copy elimination benefit.
     *
     * @param bar_bandwidth_gbps Measured BAR write bandwidth (typically 2.65 for PCIe 3.0)
     * @param dma_bandwidth_gbps DMA copy bandwidth (typically 12-16 GB/s for CUDA)
     * @param dma_setup_overhead_us DMA setup overhead in microseconds
     * @return Recommended maximum tensor size in bytes
     *
     * Rule of thumb: Use BAR-backed for tensors where DMA overhead dominates.
     * Typically this is tensors < 64KB.
     */
    inline size_t recommendedMaxBARTensorSize(
        double bar_bandwidth_gbps = 2.65,
        double dma_bandwidth_gbps = 14.0,
        double dma_setup_overhead_us = 5.0)
    {
        // Break-even point: time_bar_write = time_dma_copy + time_dma_setup
        // size / bar_bw = size / dma_bw + overhead
        // size * (1/bar_bw - 1/dma_bw) = overhead
        // size = overhead / (1/bar_bw - 1/dma_bw)

        double bar_time_per_byte = 1.0 / (bar_bandwidth_gbps * 1e9); // seconds/byte
        double dma_time_per_byte = 1.0 / (dma_bandwidth_gbps * 1e9); // seconds/byte
        double overhead_seconds = dma_setup_overhead_us * 1e-6;

        if (bar_time_per_byte <= dma_time_per_byte)
        {
            // BAR is faster than DMA (unlikely but possible with high-end systems)
            return SIZE_MAX; // No size limit
        }

        double size = overhead_seconds / (bar_time_per_byte - dma_time_per_byte);
        return static_cast<size_t>(size);
    }

} // namespace llaminar2
