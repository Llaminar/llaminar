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
 * @param bytes Exact positive extent, not required to be vector-aligned.
 *
 * The tail is assigned to distinct lanes after the last complete vector. An
 * odd byte count must not force the entire expert through scalar byte loads.
 * Stream completion provides publication; no per-thread system fence is needed.
 */
__global__ void mappedHostCopyVectorKernel(
    uint4 *__restrict__ destination,
    const uint4 *__restrict__ source,
    std::size_t bytes)
{
    const std::size_t lane =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
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
 * @param bytes Exact positive extent of both regions.
 */
__global__ void mappedHostCopyByteKernel(
    std::uint8_t *__restrict__ destination,
    const std::uint8_t *__restrict__ source,
    std::size_t bytes)
{
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < bytes; index += stride)
        destination[index] = source[index];
}
