/**
 * @file HIPGraphArgumentGeometryKernel.hip
 * @brief Device-only heavy-argument kernel and original native symbol projection.
 *
 * This HIP translation unit intentionally imports only the packet ABI and HIP.
 * Every lane reads original kernarg words directly; no full packet/private array
 * copy is created. The ordinary C++ host fixture owns all planning/MPI state.
 */
#include "HIPGraphArgumentGeometryABI.h"
#include <hip/hip_runtime.h>
namespace llaminar2::hip_argument_geometry_test {
/**
 * @brief Consume every original argument word with one deterministic output per lane.
 * @param packet Native by-value packet; no pointer to caller stack storage survives setup.
 */
__global__ void argumentGeometry4096(Packet packet)
{
    std::uint32_t sum = packet.salt;
    for (unsigned word = threadIdx.x; word < 1020; word += Threads)
        sum += packet.words[word];
    packet.output[packet.slot * Threads + threadIdx.x] = sum;
}
/** @copydoc hipGraphArgumentGeometryKernel */
void *hipGraphArgumentGeometryKernel()
{
    return reinterpret_cast<void *>(argumentGeometry4096);
}
} // namespace llaminar2::hip_argument_geometry_test
