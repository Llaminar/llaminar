/**
 * @file ExpertTierDeviceFormatPredicates.cu
 * @brief Exercise production format predicates with runtime CUDA kernel arguments.
 *
 * Host constexpr evaluation cannot certify calls made with device-owned layout
 * fields. This probe writes only predicate results to test-owned observation
 * storage; it never reads or modifies transfer payloads.
 */
#include "execution/moe/ExpertTierWeightDeviceLayout.h"
#include <cuda_runtime.h>

namespace llaminar2::test
{
    namespace
    {
        /**
         * @brief Evaluate the three production dispatch predicates on device.
         * @param layout Runtime layout copied as a native kernel argument.
         * @param result Three test-owned output bytes in catalog/preserve/Q8 order.
         */
        __global__ void expertTierDeviceFormatPredicates(
            ExpertTierWeightDeviceLayout layout, std::uint8_t *result)
        {
            result[0] = layout.gpuFormatMatchesSource();
            result[1] = expertTierPreservesNativeSingleScale(layout.cpu_encoding, layout.cpu_codebook_id);
            result[2] = cpu::native_quantised::isSignedInt8SingleScaleSource(layout.cpu_codebook_id);
        }
    }

    /**
     * @brief Submit one predicate observation on the fixture's explicit stream.
     * @param layout Runtime candidate tuple, valid or deliberately malformed.
     * @param result Three device output bytes owned by the fixture.
     * @param stream Exact non-default observation stream.
     * @return Native launch status; invalid pointers or streams are rejected.
     */
    cudaError_t launchExpertTierDeviceFormatPredicates(
        ExpertTierWeightDeviceLayout layout, std::uint8_t *result, cudaStream_t stream)
    {
        if (!result || !stream) return cudaErrorInvalidValue;
        expertTierDeviceFormatPredicates<<<1, 1, 0, stream>>>(layout, result);
        return cudaGetLastError();
    }
}
