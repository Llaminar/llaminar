/** @file ReadAheadCalibrationCUDAVersions.cpp
 * @brief Query selected CUDA version identity without contaminating the mixed-backend host ABI.
 */
#include "ReadAheadCalibrationVersions.h"
#include <cuda_runtime_api.h>
#include <stdexcept>
namespace llaminar2::test {
std::pair<int,int> readAheadCalibrationCUDAVersions(){int runtime=0,driver=0;if(cudaRuntimeGetVersion(&runtime)!=cudaSuccess || cudaDriverGetVersion(&driver)!=cudaSuccess || runtime<=0 || driver<=0)throw std::runtime_error("Selected CUDA version identity unavailable");return {runtime,driver};}
}
