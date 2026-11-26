/**
 * @file ROCmResidentMoEFFNShard1.hip
 * @brief Resident MoE FFN shard 1 (decoder groups 9..15).
 */
#include "ROCmNativeQuantisedMoEDeviceHelpers.hip.inc"
#include "ROCmResidentMoEFFNBody.hip.inc"

LLAMINAR_DEFINE_ROCM_IQ_GRID_INIT(resident_ffn_1)

namespace llaminar2::rocm {
template<unsigned Groups>
static const void* residentMoEFFNFunctionForWidthS1(ResidentMoEFFNSchedulerWidth width)
{
    constexpr auto mask=ResidentMoEFFNDecoderProgram::familyMask<Groups>();
    switch(width)
    {
        case ResidentMoEFFNSchedulerWidth::UInt32: return reinterpret_cast<const void*>(&rocmResidentMoEFFNKernel<std::uint32_t,mask>);
        case ResidentMoEFFNSchedulerWidth::UInt64: return reinterpret_cast<const void*>(&rocmResidentMoEFFNKernel<std::uint64_t,mask>);
    }
    throw std::invalid_argument("Resident MoE FFN scheduler width is invalid");
}
const void* residentMoEFFNKernelFunctionShard1(ResidentMoEFFNDecoderProgram program, ResidentMoEFFNSchedulerWidth width)
{
    switch(program.groups())
    {
        case 9:  return residentMoEFFNFunctionForWidthS1<9>(width);
        case 10: return residentMoEFFNFunctionForWidthS1<10>(width);
        case 11: return residentMoEFFNFunctionForWidthS1<11>(width);
        case 12: return residentMoEFFNFunctionForWidthS1<12>(width);
        case 13: return residentMoEFFNFunctionForWidthS1<13>(width);
        case 14: return residentMoEFFNFunctionForWidthS1<14>(width);
        case 15: return residentMoEFFNFunctionForWidthS1<15>(width);
        default: throw std::invalid_argument("Resident MoE FFN source envelope is invalid");
    }
}
} // namespace llaminar2::rocm
