/**
 * @file ROCmMoEGroupedWmma.hip
 * @brief Exact wave64 WMMA leaves borrowing the production MoE group directory.
 *
 * Each workgroup decodes up to two live K32 blocks for sixteen grouped rows and 32/64
 * columns. Integer WMMA shares those operands; canonical contribution helpers
 * and the original source-format partition loop own every FP32 rounding point.
 * Gate/up and down differ only in immutable descriptor and input-row bindings.
 * No kernel builds routing state, allocates memory or publishes weighted routes.
 */
#include <hip/hip_runtime.h>
#include "ROCmMoEGroupedWmma.h"
#include "ROCmNativeQuantisedDecodeEquivalentMath.h"
#include "kernels/common/MoEGroupedDirectory.h"
#include "kernels/common/NativeQuantisedGroupedDecodePolicy.h"
#include "kernels/rocm/common/ROCmSignedDot4.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "ROCmNativeQuantisedGemmDecode.hip.inc"
#include <stdexcept>

namespace llaminar2::rocm
{
namespace
{
thread_local MoEGroupedPrefillEngine recording_engine = MoEGroupedPrefillEngine::Automatic;
thread_local MoEGroupedWmmaRowFloor recording_floor = MoEGroupedWmmaRowFloor::EveryExpert;

/** @brief Authenticate the compiled native instruction and maximum tile resources.
 * @param device Admitted visible HIP ordinal. @return Whether WMMA can execute. */
bool nativeGroupedDevice(int device)
{
    const auto &h = rocmHardwareMetadata().at(device);
    return (h.isa_target == "gfx1100" || h.isa_target.starts_with("gfx1100:")) &&
        h.maximum_threads_per_block >= 256 && h.lds_per_block >= 8192;
}
}

ScopedMoEGroupedPrefillEngine::ScopedMoEGroupedPrefillEngine(
    MoEGroupedPrefillEngine engine, MoEGroupedWmmaRowFloor floor)
    : previous_(recording_engine), previous_floor_(recording_floor)
{
    if (!validMoEGroupedWmmaRowFloor(floor) ||
        ((engine == MoEGroupedPrefillEngine::Automatic || engine == MoEGroupedPrefillEngine::OrderedDot4) &&
         floor != MoEGroupedWmmaRowFloor::EveryExpert))
        throw std::invalid_argument("grouped WMMA row floor requires an explicit native candidate");
    switch (engine)
    {
    case MoEGroupedPrefillEngine::Automatic:
    case MoEGroupedPrefillEngine::OrderedDot4:
    case MoEGroupedPrefillEngine::WmmaM16N32:
    case MoEGroupedPrefillEngine::WmmaM16N64:
        recording_engine = engine; recording_floor = floor; return;
    }
    throw std::invalid_argument("unknown grouped MoE dot engine");
}
ScopedMoEGroupedPrefillEngine::~ScopedMoEGroupedPrefillEngine()
{ recording_engine = previous_; recording_floor = previous_floor_; }

MoEGroupedPrefillPolicy selectedMoEGroupedPrefillPolicy(int device)
{
    if (recording_engine == MoEGroupedPrefillEngine::Automatic ||
        recording_engine == MoEGroupedPrefillEngine::OrderedDot4)
        return {MoEGroupedPrefillEngine::OrderedDot4,MoEGroupedWmmaRowFloor::EveryExpert};
    if (!nativeGroupedDevice(device))
        throw std::invalid_argument("grouped MoE WMMA requires actual gfx1100 and native tile resources");
    return {recording_engine,recording_floor};
}

namespace grouped_wmma_detail
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__)
using I32x4 = int __attribute__((ext_vector_type(4)));

/** @brief Compute an unweighted grouped projection tile with exact serial arithmetic.
 * @tparam FMT Actual physical descriptor decoder; IQ4 ids remain distinct.
 * @tparam BN Number of output columns, with one wave64 per sixteen columns.
 * @tparam TwoRoles Whether grid.z selects gate/up instead of one down descriptor.
 * @param v Frozen bindings and the existing device live-route admission. */
template<NativeQuantisedGemmFormat FMT, int BN, bool TwoRoles>
__global__ __launch_bounds__(BN * 4) void projection(MoEGroupedWmmaArguments v)
{
#if defined(__gfx1100__)
    static_assert(__builtin_amdgcn_is_invocable(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64));
    static_assert(BN == 32 || BN == 64);
    using Traits = NATIVE_QUANTISEDGemmTraits<FMT>;
    constexpr int BM = 16, KB = 2, Threads = BN * 4;
    constexpr int Sums = FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? 4 :
        Traits::is_dual_scale_asym ? 2 : Traits::is_asymmetric ? 1 : 0;
    if (!v.admission.allows(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled)) return;
    const int directory_index = blockIdx.y;
    if (directory_index >= v.directory_capacity) return;
    const uint32_t tile = v.directory[directory_index];
    if (tile == UINT32_MAX) return;
    const int expert = tile & moe_grouped_directory::kExpertMask;
    const int first_row = tile >> moe_grouped_directory::kExpertBits;
    if (expert >= v.experts) __builtin_trap();
    const int count = v.group_counts[expert], offset = v.group_offsets[expert];
    if (count <= first_row || offset < 0 ||
        static_cast<int64_t>(offset) + count > v.admission.activeSlots()) __builtin_trap();
    const int live = min(BM, count - first_row), first_slot = offset + first_row;
    const auto desc = TwoRoles && blockIdx.z ? v.second_descriptors[expert] : v.descriptors[expert];
    // Each physical-format launch has exactly its own descriptor writers.
    // In particular, an IQ4_NL descriptor never borrows IQ4_XS auxiliary data.
    if (desc.codebook_id != FMT) return;
    if (!desc.payload || !desc.scales || desc.n != v.n || desc.k != v.k ||
        desc.blocks_per_row != static_cast<uint32_t>(v.k / 32) ||
        (Traits::needs_aux_compute && !desc.mins) ||
        (Traits::is_dual_scale_asym && !desc.emins)) __builtin_trap();
    const auto producer = v.dot_admission.select(desc,count);
    if (producer == MoEGroupedDotProducer::Invalid) __builtin_trap();
    if (producer != MoEGroupedDotProducer::Wmma) return;
    const int blocks = v.k / 32, partitions = v.partitions.forDescriptor(desc);
    if (partitions <= 0 || partitions > blocks ||
        partitions > NativeQuantisedGroupedDecodePolicy::maximum_k_partitions) __builtin_trap();
    const int span = (blocks + partitions - 1) / partitions;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
    const int n0 = blockIdx.x * BN;
    __shared__ int source_rows[BM];
    __shared__ I32x4 a[KB][BM][2], b[KB][BN][2];
    __shared__ uint16_t primary[KB][BN];
    __shared__ uint16_t secondary[KB][Traits::needs_aux_compute ? BN : 1];
    __shared__ uint32_t minima[KB][Traits::is_dual_scale_asym ? BN : 1];
    __shared__ uint16_t qh[KB][FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? BN : 1];
    __shared__ int sums[KB][Sums ? Sums : 1][BM];
    __shared__ float scales[KB][BM];
    if (tid < live)
    {
        const int slot = first_slot + tid;
        const int index = v.input == MoEGroupedWmmaInput::GroupedRows ? slot : v.group_token_indices[slot];
        const int row = v.input == MoEGroupedWmmaInput::OriginalRouteSlots ? index / v.top_k : index;
        const int bound = v.input == MoEGroupedWmmaInput::GroupedRows ? v.admission.activeSlots() : v.rows;
        if (index < 0 || row < 0 || row >= bound) __builtin_trap();
        source_rows[tid] = row;
    }
    __syncthreads();
    float total[4] = {};
    // Keep empty trailing partitions: their +0 folds are part of grouped
    // serial arithmetic, including the signed-zero result of underflow.
    for (int p = 0; p < partitions; ++p)
    {
        float partial[4] = {};
        const int end = min((p + 1) * span, blocks);
        for (int first = p * span; first < end; first += KB)
        {
            // A short serial partition must not decode or publish an unused
            // second stage. Both producers share this live bound, while the
            // consumer and ordered partition fold retain their original work.
            const int stages = min(KB,end-first);
            for (int item = tid; item < stages * BN; item += Threads)
            {
                const int stage = item / BN, col = item % BN, kb = first + stage;
                const size_t linear = static_cast<size_t>(kb) * v.n + n0 + col;
                uint32_t raw[Traits::payload_dwords] = {};
                uint16_t scale = 0, aux = 0; uint32_t emin = 0;
                if (kb < end && n0 + col < v.n)
                {
                    load_payload<Traits::payload_bytes>(desc.payload + linear * Traits::payload_bytes, raw);
                    scale = static_cast<const uint16_t *>(desc.scales)[linear];
                    if constexpr (Traits::needs_aux_compute) aux = static_cast<const uint16_t *>(desc.mins)[linear];
                    if constexpr (Traits::is_dual_scale_asym) emin = static_cast<const uint32_t *>(desc.emins)[linear];
                }
                int decoded[8]; decode_block<FMT>(raw, decoded);
                I32x4 lo, hi;
                #pragma unroll
                for (int g = 0; g < 4; ++g) { lo[g] = decoded[g]; hi[g] = decoded[g+4]; }
                b[stage][col ^ stage][(col >> 2) & 1] = lo;
                b[stage][col ^ stage][((col >> 2) & 1) ^ 1] = hi;
                primary[stage][col] = scale;
                if constexpr (Traits::needs_aux_compute) secondary[stage][col] = aux;
                if constexpr (Traits::is_dual_scale_asym) minima[stage][col] = emin;
                if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M) qh[stage][col] = static_cast<uint16_t>(raw[1]);
            }
            for (int item = tid; item < stages * BM * 2; item += Threads)
            {
                // Adjacent half-block owners remain in one 32-lane subgroup
                // for exact activation sums, including a single live stage.
                const int stage = item / (BM * 2), row = item % (BM * 2) / 2, half = item % 2;
                const int kb = first + stage;
                I32x4 values = {};
                if (row < live && kb < end)
                    values = reinterpret_cast<const I32x4 *>(v.activations +
                        static_cast<size_t>(source_rows[row]) * v.k + kb * 32)[half];
                a[stage][row ^ stage][half ^ ((row >> 2) & 1)] = values;
                if constexpr (Sums == 1)
                {
                    int sum = 0;
                    #pragma unroll
                    for (int g = 0; g < 4; ++g) sum = rocm_integer_contract::signedDot4(values[g],0x01010101,sum);
                    sum += __shfl_xor(sum,1,32);
                    if (half == 0) sums[stage][0][row] = sum;
                }
                else if constexpr (Sums > 1)
                {
                    #pragma unroll
                    for (int group = 0; group < Sums / 2; ++group)
                    {
                        int sum = 0;
                        #pragma unroll
                        for (int g = 0; g < 8 / Sums; ++g)
                            sum = rocm_integer_contract::signedDot4(values[group * (8 / Sums) + g],0x01010101,sum);
                        sums[stage][half * (Sums / 2) + group][row] = sum;
                    }
                }
                if (half == 0) scales[stage][row] = row < live && kb < end ?
                    v.activation_scales[static_cast<size_t>(source_rows[row]) * blocks + kb] : 1.0f;
            }
            __syncthreads();
            #pragma unroll 1
            for (int stage = 0; stage < KB && first + stage < end; ++stage)
            {
                const int ar = lane % 16, col = wave * 16 + lane % 16;
                const I32x4 al = a[stage][ar ^ stage][(ar >> 2) & 1];
                const I32x4 ah = a[stage][ar ^ stage][((ar >> 2) & 1) ^ 1];
                const I32x4 bl = b[stage][col ^ stage][(col >> 2) & 1];
                const I32x4 bh = b[stage][col ^ stage][((col >> 2) & 1) ^ 1];
                const I32x4 zero = {};
                const auto lo = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true,al,true,bl,zero,false);
                const auto hi = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true,ah,true,bh,Traits::is_dual_scale ? zero : lo,false);
                const uint16_t scale = primary[stage][col];
                const uint16_t aux = Traits::needs_aux_compute ? secondary[stage][col] : uint16_t{0};
                const auto ws = [&] {
                    if constexpr (Traits::is_dual_scale)
                        return rocm_native_quantised_load_dual_weight_scales<FMT>(scale,aux);
                    else return ROCmNativeQuantisedDualWeightScales{
                        rocm_native_quantised_load_single_weight_scale<FMT>(scale,aux),0.0f};
                }();
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    const int row = e * 4 + lane / 16;
                    const float sa = scales[stage][row];
                    float &acc = partial[e];
                    if constexpr (Traits::is_dual_scale_asym)
                    {
                        const uint32_t pair = minima[stage][col];
                        const uint16_t ml = pair, mh = pair >> 16;
                        acc = rocm_native_quantised_commit_corrected_dual_scale_block(acc,lo[e],hi[e],
                            ws.low,ws.high,sums[stage][0][row],sums[stage][1][row],
                            __half2float(*reinterpret_cast<const __half *>(&ml)),
                            __half2float(*reinterpret_cast<const __half *>(&mh)),sa);
                    }
                    else if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M)
                    {
                        const uint16_t bits = qh[stage][col];
                        acc = rocm_native_quantised_commit_iq1_m_block(acc,lo[e],hi[e],
                            sums[stage][0][row],sums[stage][1][row],sums[stage][2][row],sums[stage][3][row],
                            bits & 0x0008 ? -0.125f : 0.125f,bits & 0x0080 ? -0.125f : 0.125f,
                            bits & 0x0800 ? -0.125f : 0.125f,bits & 0x8000 ? -0.125f : 0.125f,ws.low,ws.high,sa);
                    }
                    else if constexpr (Traits::is_dual_scale)
                        acc = rocm_native_quantised_commit_dual_scale_block(acc,lo[e],hi[e],ws.low,ws.high,sa);
                    else if constexpr (Traits::is_asymmetric)
                        acc = rocm_native_quantised_commit_corrected_single_scale_block(acc,hi[e],ws.low,
                            sums[stage][0][row],__half2float(*reinterpret_cast<const __half *>(&aux)),sa);
                    else acc = rocm_native_quantised_commit_single_scale_block(acc,hi[e],ws.low,sa);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int e = 0; e < 4; ++e)
            total[e] = partitions == 1 ? partial[e] : __fadd_rn(total[e],partial[e]);
    }
    float *out = TwoRoles && blockIdx.z ? v.second_output : v.output;
    #pragma unroll
    for (int e = 0; e < 4; ++e)
    {
        const int row = e * 4 + lane / 16, col = n0 + wave * 16 + lane % 16;
        if (row < live && col < v.n) out[static_cast<size_t>(first_slot + row) * v.n + col] = total[e];
    }
#endif
}
#endif
}

bool submitMoEGroupedWmma(MoEGroupedPrefillEngine engine, uint32_t codebooks,
    const MoEGroupedWmmaArguments &args, void *stream)
{
    const auto native_stream = static_cast<hipStream_t>(stream);
    if (!native_stream || native_stream == hipStreamLegacy || native_stream == hipStreamPerThread ||
        !args.activations || !args.activation_scales || !args.group_counts || !args.group_offsets ||
        !args.directory || !args.descriptors || !args.output || args.directory_capacity <= 0 ||
        args.rows <= 0 || args.top_k <= 0 || args.n <= 0 || args.k <= 0 || args.k % 32 ||
        args.experts <= 0 || args.device < 0 || !nativeGroupedDevice(args.device) ||
        (args.input != MoEGroupedWmmaInput::GroupedRows && !args.group_token_indices) ||
        ((args.second_descriptors == nullptr) != (args.second_output == nullptr)) ||
        !args.dot_admission.binds(args.second_descriptors ? MoEPrefillProjectionRole::GateUp
                                                        : MoEPrefillProjectionRole::Down,args.n,args.k)) return false;
    int stream_device = -1;
    if (hipStreamGetDevice(native_stream,&stream_device) != hipSuccess || stream_device != args.device) return false;
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__)
    uint32_t remaining = codebooks;
    auto format = [&]<NativeQuantisedGemmFormat FMT>() {
        constexpr uint32_t bit = uint32_t{1} << FMT;
        if (!(remaining & bit)) return true;
        remaining &= ~bit;
        auto tile = [&]<int BN>() {
            const dim3 grid((args.n + BN - 1) / BN,args.directory_capacity,args.second_descriptors ? 2 : 1);
            if (args.second_descriptors)
                grouped_wmma_detail::projection<FMT,BN,true><<<grid,BN*4,0,native_stream>>>(args);
            else grouped_wmma_detail::projection<FMT,BN,false><<<grid,BN*4,0,native_stream>>>(args);
            return hipGetLastError() == hipSuccess;
        };
        switch (engine)
        {
        case MoEGroupedPrefillEngine::WmmaM16N32: return tile.template operator()<32>();
        case MoEGroupedPrefillEngine::WmmaM16N64: return tile.template operator()<64>();
        default: return false;
        }
    };
#define MOE_WMMA_FORMAT(FMT) if (!format.template operator()<FMT>()) return false
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_Q4_0);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_IQ4_NL);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_IQ4_XS_SOURCE);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_Q4_1);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_Q5_0);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_Q5_1);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_Q6_K);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_Q3_K);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_Q2_K);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_IQ3_S);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_IQ3_XXS);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_IQ2_S);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_IQ2_XS);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_IQ2_XXS);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_IQ1_S);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_IQ1_M);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_Q8_0);
    MOE_WMMA_FORMAT(NATIVE_QUANTISED_GEMM_EXPANDED_INT8_MIN);
#undef MOE_WMMA_FORMAT
    return codebooks && !remaining;
#else
    return false;
#endif
}

bool initializeMoEGroupedWmmaGrids(int device, const void *iq3s, const void *iq3xxs,
    const void *iq2s, const void *iq2xs, const void *iq2xxs, const void *iq1s)
{
    return initialize_native_quantised_gemm_grids(device,iq3s,iq3xxs,iq2s,iq2xs,iq2xxs,iq1s);
}
}
