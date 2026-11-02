/** @file GPUFlatGraphStructure.cpp
 * @brief Select compiled adapter object geometry without native creation or provider queries.
 */
#include "GPUFlatGraphStructure.h"
namespace llaminar2 {
namespace detail {
#ifdef HAVE_CUDA
std::size_t cudaGraphCaptureObjectBytes()noexcept;
#endif
#ifdef HAVE_ROCM
std::size_t hipGraphCaptureObjectBytes()noexcept;
#endif
}
std::size_t nativeGraphCaptureObjectBytes(DeviceId device){
    if(!device.is_valid())throw std::invalid_argument("Native graph object geometry requires a valid GPU identity");
#ifdef HAVE_CUDA
    if(device.is_cuda())return detail::cudaGraphCaptureObjectBytes();
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm())return detail::hipGraphCaptureObjectBytes();
#endif
    throw std::invalid_argument("Native graph object geometry requires an exact compiled GPU backend");
}
}
