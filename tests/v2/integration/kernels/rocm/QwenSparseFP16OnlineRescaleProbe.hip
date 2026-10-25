/**
 * @file QwenSparseFP16OnlineRescaleProbe.hip
 * @brief Test-owned evaluator witness using the production helper and HIP flags.
 *
 * Linked only into the existing ROCm native test/perf executables. The serving
 * core exports no experiment entry and no runtime execution policy is added.
 */
#include "QwenSparseFP16OnlineRescaleProbe.h"
#include "kernels/rocm/attention/ROCmQwenSparseOnlineRescale.cuh"
#include <stdexcept>
namespace llaminar2::qwen4exp::online_rescale_trial
{
namespace
{
/**
 * @brief Compare scale programs through one shared compiled arithmetic suffix.
 * @details Adjacent threads own old/new sides of the same immutable record.
 * The old probe duplicated denominator expressions, allowing two equivalent
 * multiplies with two NaN operands to choose different payloads after register
 * operand commutation. One unbranched suffix isolates the scales under test;
 * every side still publishes and compares all original FP32 words strictly.
 */
__global__ void onlineRescaleEvaluatorWitness(const Input *input,Output *output,int count)
{
    const int thread=static_cast<int>(threadIdx.x);
    const int index=thread/2;
    const int side=thread&1;
    if (index>=count) return;
    const auto value=input[index];
    const float next=fmaxf(value.maximum,value.tile_maximum);
    rocm_sparse_detail::SparseHalfOnlineScales scales;
    if (side==1)
        scales=rocm_sparse_detail::sparseHalfOnlineScales(
            value.maximum,value.tile_maximum,value.denominator,next);
    else
        scales={value.denominator == 0.0f ? 0.0f : expf(value.maximum-next),
                expf(value.tile_maximum-next)};

    // Both sides execute these same source operations and compiled operand
    // ordering. No FMA/multiply is removed, duplicated or made noinline.
    const float total=fmaf(value.total,scales.previous,value.weighted*scales.tile);
    const float denominator=fmaf(value.denominator,scales.previous,value.tile_denominator*scales.tile);

    auto& result=output[index];
    // Dynamic indexing into an actual typed array keeps a single store suffix.
    // Adjacent threads own different SideResult objects; no struct-as-float
    // pointer arithmetic, overlapping stores or old/new store branch exists.
    result.sides[side]={scales.previous,scales.tile,total,denominator};
    if (side==0)
    {
        result.next_maximum=next;
        result.zero_exponential=expf(value.zero);
    }
}
}
const void *evaluatorFunction() noexcept
{
    return reinterpret_cast<const void *>(onlineRescaleEvaluatorWitness);
}
void evaluate(const Input *input,Output *output,int count,hipStream_t stream)
{
    if (!stream || stream==hipStreamLegacy || stream==hipStreamPerThread)
        throw std::invalid_argument("Online-rescale witness requires an exact nondefault stream");
    if (!input || !output || count<=0 || count>256)
        throw std::invalid_argument("Online-rescale witness requires1..256 persistent records");
    const auto begin=reinterpret_cast<std::uintptr_t>(input);
    const auto end=reinterpret_cast<std::uintptr_t>(output);
    const auto inputBytes=std::size_t(count)*sizeof(Input),outputBytes=std::size_t(count)*sizeof(Output);
    if (begin%alignof(Input) || end%alignof(Output) ||
        inputBytes>UINTPTR_MAX-begin || outputBytes>UINTPTR_MAX-end ||
        !(begin+inputBytes<=end || end+outputBytes<=begin))
        throw std::invalid_argument("Online-rescale witness has overlapping or invalid record extents");
    onlineRescaleEvaluatorWitness<<<dim3(1),dim3(512),0,stream>>>(input,output,count);
    const auto status=hipGetLastError();
    if (status!=hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
}
