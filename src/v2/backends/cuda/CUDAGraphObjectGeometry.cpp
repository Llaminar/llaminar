/** @file CUDAGraphObjectGeometry.cpp
 * @brief Exact compiled CUDAGraphCapture object extent; this TU creates no SDK resource.
 *
 * Keep CUDA/HIP headers in separate TUs: their global vector types are distinct
 * SDK ABIs and must never be included into the same host translation unit.
 */
#include "CUDAGraphCapture.h"
#include <cstddef>
namespace llaminar2::detail {
/** @return Original compiled adapter object bytes, without a worker/provider query. */
std::size_t cudaGraphCaptureObjectBytes()noexcept{return sizeof(CUDAGraphCapture);}
}
