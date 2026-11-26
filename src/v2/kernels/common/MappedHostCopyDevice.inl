/**
 * @file MappedHostCopyDevice.inl
 * @brief Byte-exact CUDA/HIP copies that do not enter a copy-engine queue.
 *
 * The caller supplies setup-stable device-visible addresses, including the
 * exact mapped alias for host pages. A small grid saturates the host link while
 * leaving most compute resources available to inference. Completion is the
 * caller's exact stream event; these kernels contain no command protocol,
 * polling loop, allocation, or mutable host-side state.
 */

/**
 * @brief Copy aligned vectors and their disjoint, possibly empty byte tail.
 * @param destination Device-visible destination, aligned to sixteen bytes.
 * @param source Immutable device-visible source, aligned to sixteen bytes.
 * @param bytes Exact live extent, possibly zero, not required to be vector-aligned.
 * @param copy_block Block coordinate within this endpoint, not another batch message.
 * @param copy_blocks Positive immutable payload block count for this endpoint.
 *
 * The tail is assigned to distinct lanes after the last complete vector. An
 * odd byte count must not force the entire expert through scalar byte loads.
 * Stream completion provides publication; no per-thread system fence is needed.
 */
__device__ __forceinline__ void mappedHostCopyVectorBody(
    uint4 *__restrict__ destination,
    const uint4 *__restrict__ source,
    std::size_t bytes, unsigned copy_block, unsigned copy_blocks)
{
    const std::size_t lane =
        static_cast<std::size_t>(copy_block) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(copy_blocks) * blockDim.x;
    for (std::size_t index = lane; index < bytes / sizeof(uint4); index += stride)
        destination[index] = source[index];
    for (std::size_t index = bytes / sizeof(uint4) * sizeof(uint4) + lane;
         index < bytes; index += stride)
        reinterpret_cast<std::uint8_t *>(destination)[index] =
            reinterpret_cast<const std::uint8_t *>(source)[index];
}

/**
 * @brief Copy an arbitrarily aligned pair without reading outside its bounds.
 * @param destination Device-visible destination with no alignment requirement.
 * @param source Stable device-visible source with no alignment requirement.
 * @param bytes Exact live extent of both regions; zero performs no memory access.
 * @param copy_block Block coordinate within this endpoint.
 * @param copy_blocks Positive endpoint-local stride in blocks.
 */
__device__ __forceinline__ void mappedHostCopyByteBody(
    std::uint8_t *__restrict__ destination,
    const std::uint8_t *__restrict__ source,
    std::size_t bytes, unsigned copy_block, unsigned copy_blocks)
{
    const std::size_t stride =
        static_cast<std::size_t>(copy_blocks) * blockDim.x;
    for (std::size_t index =
             static_cast<std::size_t>(copy_block) * blockDim.x + threadIdx.x;
         index < bytes; index += stride)
        destination[index] = source[index];
}

/** @brief Preserve the original whole-grid copy coordinates for existing callers.
 * @param destination Aligned exact destination.
 * @param source Aligned immutable source.
 * @param bytes Exact live extent. */
__device__ __forceinline__ void mappedHostCopyVectorBody(
    uint4 *destination, const uint4 *source, std::size_t bytes)
{ mappedHostCopyVectorBody(destination, source, bytes, blockIdx.x, gridDim.x); }

/** @brief Preserve the original whole-grid byte copy coordinates for existing callers.
 * @param destination Exact unaligned destination.
 * @param source Immutable unaligned source.
 * @param bytes Exact live extent. */
__device__ __forceinline__ void mappedHostCopyByteBody(
    std::uint8_t *destination, const std::uint8_t *source, std::size_t bytes)
{ mappedHostCopyByteBody(destination, source, bytes, blockIdx.x, gridDim.x); }

/** @brief Fixed-extent aligned lowering; shared body also serves device-counted lanes.
 * @param destination Sixteen-byte-aligned destination.
 * @param source Sixteen-byte-aligned source.
 * @param bytes Exact capture-time extent. */
__global__ void mappedHostCopyVectorKernel(
    uint4 *__restrict__ destination, const uint4 *__restrict__ source, std::size_t bytes)
{ mappedHostCopyVectorBody(destination, source, bytes); }

/** @brief Fixed-extent unaligned lowering with identical byte/tail semantics.
 * @param destination Arbitrarily aligned destination.
 * @param source Arbitrarily aligned source.
 * @param bytes Exact capture-time extent. */
__global__ void mappedHostCopyByteKernel(
    std::uint8_t *__restrict__ destination, const std::uint8_t *__restrict__ source, std::size_t bytes)
{ mappedHostCopyByteBody(destination, source, bytes); }

/** @brief Device-facing descriptor for one fused single-block mapped copy + system release. */
struct MappedHostCopyBatchPublishLaunch
{
    static constexpr std::uint32_t kMaxSegments = 6u;
    void *dst[kMaxSegments] = {};
    const void *src[kMaxSegments] = {};
    std::size_t bytes[kMaxSegments] = {};
    std::uint32_t segment_count = 0u;
    std::uint64_t *publication_signal = nullptr;
    std::uint64_t publication_value = 0u;
};

/**
 * @brief Copy up to six bounded device regions to mapped host aliases and
 *        release-publish one 64-bit timeline signal in a single block.
 *
 * Replaces up to six serial graph nodes (small header/route/hidden copies plus
 * publishSystemReleaseValue64) for decode-sized tickets on both CUDA and ROCm.
 */
__global__ void mappedHostCopyBatchPublishKernel(MappedHostCopyBatchPublishLaunch launch)
{
    if (blockIdx.x != 0u)
        return;
    const std::uint32_t count = launch.segment_count <= MappedHostCopyBatchPublishLaunch::kMaxSegments
        ? launch.segment_count
        : MappedHostCopyBatchPublishLaunch::kMaxSegments;
    for (std::uint32_t i = 0u; i < count; ++i)
    {
        void *const dst = launch.dst[i];
        const void *const src = launch.src[i];
        const std::size_t bytes = launch.bytes[i];
        if (!dst || !src || bytes == 0u)
            continue;
        const bool vector_aligned =
            (reinterpret_cast<std::uintptr_t>(dst) % alignof(uint4) == 0u) &&
            (reinterpret_cast<std::uintptr_t>(src) % alignof(uint4) == 0u);
        if (vector_aligned && bytes >= sizeof(uint4))
        {
            mappedHostCopyVectorBody(
                static_cast<uint4 *>(dst),
                static_cast<const uint4 *>(src),
                bytes, 0u, 1u);
        }
        else
        {
            mappedHostCopyByteBody(
                static_cast<std::uint8_t *>(dst),
                static_cast<const std::uint8_t *>(src),
                bytes, 0u, 1u);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0u && launch.publication_signal)
    {
        // mappedSystemRelease64 emits a system-scope release store
        // (st.release.sys on CUDA, __ATOMIC_RELEASE/__HIP_MEMORY_SCOPE_SYSTEM on HIP),
        // which already orders all CTA writes synchronized by __syncthreads()
        // without issuing a second redundant membar.sys stall.
        mappedSystemRelease64(launch.publication_signal, launch.publication_value);
    }
}
