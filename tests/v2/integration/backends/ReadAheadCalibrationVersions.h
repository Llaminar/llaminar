/** @file ReadAheadCalibrationVersions.h
 * @brief Plain host ABI isolates CUDA/HIP version headers in separate translation units.
 * Native version queries execute only inside the explicitly selected diagnostic worker.
 */
#pragma once
#include <utility>
namespace llaminar2::test {
#ifdef HAVE_CUDA
/** @return Actual CUDA runtime/driver API versions, or throw on incomplete native identity. */
std::pair<int,int> readAheadCalibrationCUDAVersions();
#endif
#ifdef HAVE_ROCM
/** @return Actual HIP runtime/driver API versions, or throw on incomplete native identity. */
std::pair<int,int> readAheadCalibrationROCmVersions();
#endif
}
