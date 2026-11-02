/** @file ReadAheadCalibrationROCmVersions.cpp
 * @brief Query selected HIP version identity separately from CUDA's incompatible SDK vector ABI.
 */
#include "ReadAheadCalibrationVersions.h"
#include <hip/hip_runtime_api.h>
#include <stdexcept>
namespace llaminar2::test {
std::pair<int,int> readAheadCalibrationROCmVersions(){int runtime=0,driver=0;if(hipRuntimeGetVersion(&runtime)!=hipSuccess || hipDriverGetVersion(&driver)!=hipSuccess || runtime<=0 || driver<=0)throw std::runtime_error("Selected HIP version identity unavailable");return {runtime,driver};}
}
