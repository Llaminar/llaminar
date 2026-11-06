/** @file LocalHotRecordedResponse.cu
 * @brief CUDA exact-stream lowering of the immutable L1 replay response producer.
 */
#include <cuda_runtime.h>
#include "LocalHotRecordedResponse.cuh"
#include <stdexcept>
namespace llaminar2::test::local_hot_recorded {
void launchCUDA(Action action,const Args& args,void* stream)
{
    if(!stream || reinterpret_cast<std::uintptr_t>(stream)<=2)throw std::invalid_argument("Recorded L1 response requires exact CUDA stream");
    recordedInput<<<1,128,0,static_cast<cudaStream_t>(stream)>>>(action,args);
    if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("Recorded L1 response CUDA enqueue failed");
}
}
