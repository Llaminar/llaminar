/**
 * @file SelectedDescriptorObserver.hip
 * @brief Read selected local role descriptors after the real retained producer.
 */
#include <hip/hip_runtime.h>
#include "kernels/common/MoEDecodeDescriptorPublication.h"
#include "kernels/rocm/moe/ROCmNativeRuntimeBankView.h"

namespace {
using namespace llaminar2;
using Runtime=rocm::native_runtime_abi::DeviceMoELayerRuntimeView;

/** @brief Copy only published local rows. Every inactive or remote slot writes zero. */
__global__ void observeSelectedDescriptors(const Runtime* runtime,MoEDecodeDescriptorPublication publication,
    int top_k,unsigned char* output)
{
    const unsigned slot=blockIdx.x;
    const int expert=runtime->topk_expert_ids[slot];
    const DeviceNativeQuantisedMatrixDesc* const roles[]{publication.gate,publication.up,publication.down};
    for(unsigned i=threadIdx.x;i<3*sizeof(DeviceNativeQuantisedMatrixDesc);i+=blockDim.x) {
        unsigned char value=0;
        if(expert>=0&&expert<publication.expert_count)
            value=reinterpret_cast<const unsigned char*>(roles[i/sizeof(DeviceNativeQuantisedMatrixDesc)]+expert)[i%sizeof(DeviceNativeQuantisedMatrixDesc)];
        output[slot*3*sizeof(DeviceNativeQuantisedMatrixDesc)+i]=value;
    }
}
}

/** @brief Submit the exact fixture reader on its producer-ordered graph stream. */
extern "C" bool launchSelectedDescriptorObserver(const void* runtime,
    llaminar2::MoEDecodeDescriptorPublication publication,int top_k,void* output,void* stream)
{
    if(!runtime||!publication.valid()||top_k<=0||!output||reinterpret_cast<std::uintptr_t>(stream)<=2)return false;
    observeSelectedDescriptors<<<top_k,64,0,static_cast<hipStream_t>(stream)>>>(
        static_cast<const Runtime*>(runtime),publication,top_k,static_cast<unsigned char*>(output));
    return hipGetLastError()==hipSuccess;
}
