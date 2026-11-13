/** @file LocalHotRecordedResponse.hip
 * @brief HIP exact-stream lowering of the same immutable L1 replay response producer.
 */
#include <hip/hip_runtime.h>
#include "LocalHotRecordedResponse.cuh"
#include <stdexcept>
namespace llaminar2::test::local_hot_recorded {
void launchROCm(Action action,const Args& args,void* stream)
{
    if(!stream || reinterpret_cast<std::uintptr_t>(stream)<=2)throw std::invalid_argument("Recorded L1 response requires exact HIP stream");
    recordedInput<<<1,128,0,static_cast<hipStream_t>(stream)>>>(action,args);
    if(hipGetLastError()!=hipSuccess)throw std::runtime_error("Recorded L1 response HIP enqueue failed");
}
}
