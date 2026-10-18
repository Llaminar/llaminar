/**
 * @file CUDATensorPrefillKernels.cu
 * @brief Safeguarded native FP16 tensor prefill with complete native code objects.
 *
 * Each warp owns sixteen queries and their FP32 persistent numerator. Q/K uses
 * ascending native half MMA operands with FP32 accumulation. Temporary bounded
 * P/V MMA is half precision with exact binary scaling; maxima, denominators,
 * recurrence and compact-summary publication remain FP32. K/V double buffering,
 * XOR operand swizzles and register-owned probability fragments adapt the
 * Apache-2.0 NInfer 3090 structure pinned in vendor/ninfer_attention/NOTICE.
 *
 * This numerical TU has no cross-TU device symbols. The host graph controller
 * borrows function handles; complete code-object compilation keeps the retained
 * numerator in registers instead of RDC scratch. Fatal spill checking and every
 * shipped architecture apply without exceptions. Arena, stream, graph and event
 * authorities are unchanged; no host inference state or runtime allocation is
 * introduced by this boundary.
 */
#include "CUDATensorPrefillKernels.h"
#include "CUDATensorAttentionPrimitives.cuh"
#include "CUDATensorContextPartition.cuh"
#include <cuda_fp16.h>
#include <cfloat>
#include <stdexcept>

namespace {
/**
 * @brief Map one admitted live logical KV row to its device-owned ring row.
 * @param logical_row Live logical row, bounded by physical ring capacity.
 * @param ring_origin Normalized physical origin; zero for linear storage.
 * @param ring_capacity Positive physical ring extent, or zero for linear KV.
 * @return Physical operand row after at most one wrap, without host state.
 */
__device__ __forceinline__ unsigned fa2_physical_kv_row(
        int logical_row, int ring_origin, int ring_capacity)
    {
        if (ring_capacity <= 0)
            return static_cast<unsigned>(logical_row);
        const unsigned row = static_cast<unsigned>(ring_origin) +
                             static_cast<unsigned>(logical_row);
        const unsigned capacity = static_cast<unsigned>(ring_capacity);
        return row >= capacity ? row - capacity : row;
    }
namespace tensor_prefill_attention
{
constexpr auto kPartitionPolicy = llaminar2::cuda::tensor_attention_policy::SplitPolicy::Four;

using namespace llaminar2::cuda::tensor_prefill;
static_assert(kSharedBytes == 98368);

/** @brief Pack two FP32 probabilities into the fork's A-fragment word.
 * @param first Lower adjacent score column.
 * @param second Upper adjacent score column.
 * @return Round-to-nearest-even FP16 pair, with no intermediate shared plane.
 */
__device__ __forceinline__ unsigned pack(float first, float second)
{
    const half2 pair = __floats2half2_rn(first, second);
    return reinterpret_cast<const unsigned &>(pair);
}

/** @brief Exact binary representation scale for one tile's probabilities.
 * @param maximum Largest active scaled FP32 score in this sixteen-key row tile.
 * @param scale_l2 Base-two conversion for the already scaled FP32 score.
 * @param running_maximum Scaled FP32 running maximum before subtraction.
 * @return A nonnegative exponent capped at 126, including masked-row zero.
 * Integer truncation keeps the tile maximum at or below approximately one.
 * Bounding before conversion also makes fully masked/extreme rows safe.
 */
__device__ __forceinline__ int probability_shift(float maximum, float scale_l2, float running_maximum)
{
    if (!(maximum > -FLT_MAX)) return 0;
    const float log_probability = __fmul_rn(__fsub_rn(maximum, running_maximum), scale_l2);
    return static_cast<int>(fminf(126.0f, fmaxf(0.0f, -log_probability)));
}

/** @brief Copy both native FP16 operand planes into one next-stage slot.
 * @param keys Native K allocation.
 * @param values Native V allocation.
 * @param stage Exclusive double-buffer slot.
 * @param heads Native visible KV heads.
 * @param head Selected native KV head.
 * @param begin First logical key of this tile.
 * @param live Live rows; the remaining shared rows are zero-filled.
 * @param origin Device-published ring origin.
 * @param capacity Device-published ring capacity, or zero for linear KV.
 * Publication occurs only at the caller's cp.wait/barrier frontier, matching
 * the fork's one-barrier steady-state pipeline.
 */
__device__ __forceinline__ void issue(const half *keys, const half *values,
    half *stage, unsigned *range, int heads, int head, int begin, int live, int origin, int capacity)
{
    unsigned largest = 0;
#pragma unroll
    for (int i = 0; i < kKeys * kWidth / 8 / kThreads; ++i)
    {
        const int chunk = threadIdx.x + i * kThreads;
        const int row = chunk / 32;
        const int dimension = (chunk & 31) * 8;
        const size_t native = (static_cast<size_t>(fa2_physical_kv_row(begin + min(row, max(live-1,0)), origin, capacity)) * heads + head) * kWidth + dimension;
        const int swizzled = row * kWidth + llaminar2::cuda::attention_primitives::key_column(row, dimension);
        llaminar2::cuda::attention_primitives::copy128ZeroFilled(stage + swizzled,
            keys + native, row < live ? 16 : 0);
        // The native format has no range plane. The producer derives eight
        // warp maxima from the same vector load it publishes, without a second
        // cache read or an extra barrier in the double-buffered sweep.
        uint4 value = make_uint4(0,0,0,0);
        if (row < live) value = *reinterpret_cast<const uint4 *>(values + native);
        *reinterpret_cast<uint4 *>(stage + kKeys * kWidth + swizzled) = value;
        const unsigned maximum_pair = max(max(value.x & 0x7fff7fffu,value.y & 0x7fff7fffu),
                                          max(value.z & 0x7fff7fffu,value.w & 0x7fff7fffu));
        // Packed lexicographic max cannot reduce the low halves: reduce each
        // half independently before the full-warp integer maximum.
        largest=max(largest,max(max(max(value.x&0x7fffu,value.y&0x7fffu),max(value.z&0x7fffu,value.w&0x7fffu)), maximum_pair>>16));
    }
    largest=__reduce_max_sync(0xffffffff,largest);
    if ((threadIdx.x & 31)==0) range[threadIdx.x / 32]=largest;
    llaminar2::cuda::attention_primitives::commitCopies();
}

/** @brief Retain the fork's complete warp-owned attention sweep over native KV.
 * @tparam QueryHeads Participant query head count.
 * @tparam KVHeads Visible native KV head count.
 * @tparam GQA Replicated mapping, or zero for participant-local KV sharding.
 * @tparam HalfAcc Select bounded FP16 P/V accumulation; persistent sums stay FP32.
 * @param queries FP32 queries, converted directly to FP16 operands.
 * @param keys Native FP16 key cache.
 * @param values Native FP16 value cache.
 * @param output Normalized direct output if exactly one partition is live.
 * @param query_rows Captured query extent, including a partial final tile.
 * @param kv_length Static visible length if device parameters are absent.
 * @param position_offset Static query origin if device parameters are absent.
 * @param params Canonical device-owned live/ring geometry.
 * @param scale QK normalization scalar.
 * @param head_start Global query offset for replicated KV.
 * @param partial FP32 canonical-partition numerators.
 * @param maxima FP32 canonical-partition maxima.
 * @param denominators FP32 canonical-partition denominators.
 * @param capacity Admitted partition count.
 * @param direct_limit Captured direct/context controller threshold.
 * @param explicit_partition Diagnostic partition, or -1 for ordinary replay.
 */
template <int QueryHeads, int KVHeads, int GQA, bool HalfAcc>
__device__ __forceinline__ void run(const float *__restrict__ queries, const void *__restrict__ keys,
    const void *__restrict__ values, float *__restrict__ output, int query_rows, int kv_length, int position_offset,
    const llaminar2::attention::AttentionDeviceParams *__restrict__ params, float scale,
    int head_start, float *__restrict__ partial, float *__restrict__ maxima, float *__restrict__ denominators,
    int capacity, int direct_limit, int explicit_partition)
{
    const int origin = params ? params->ring_row_origin : 0;
    const int ring_capacity = params ? params->ring_row_capacity : 0;
    if (params) { kv_length = params->kv_len; position_offset = params->position_offset; }
    const int canonical_partitions = min(capacity, max(1, (kv_length + 255) / 256));
    const auto partition_plan = llaminar2::cuda::tensor_attention_policy::plan<kPartitionPolicy>(kv_length, capacity);
    const int partitions = partition_plan.count;
    const int partition_keys = partition_plan.key_span;
    if (direct_limit > 1 && canonical_partitions <= direct_limit) return;
    const bool direct = direct_limit == 1 && canonical_partitions == 1;
    const int first = explicit_partition >= 0 ? explicit_partition : static_cast<int>(blockIdx.x) / QueryHeads;
    const int step = explicit_partition >= 0 ? capacity : static_cast<int>(gridDim.x) / QueryHeads;
    if (first >= partitions || step <= 0) return;
    const int head = blockIdx.x % QueryHeads;
    const int kv_head = GQA > 0 ? (head_start + head) / GQA : head / (QueryHeads / KVHeads);
    const int query_start = (gridDim.y - 1 - blockIdx.y) * kRows;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int row0 = warp * 16 + gid;
    const int row1 = row0 + 8;
    extern __shared__ char smem[];
    half *q = reinterpret_cast<half *>(smem);
    half *stages = q + kRows * kWidth;
    unsigned *ranges = reinterpret_cast<unsigned *>(smem + kOperandBytes);
    // The first K/V copy overlaps Q preparation, just as in the fork's prologue.
    const int first_live = min(kKeys, max(0, min(kv_length, query_start + kRows + position_offset) - first * partition_keys));
    if (first_live)
        issue(static_cast<const half *>(keys), static_cast<const half *>(values), stages, ranges,
            KVHeads, kv_head, first * partition_keys, first_live, origin, ring_capacity);
    for (int i = threadIdx.x; i < kRows * kWidth; i += kThreads)
    {
        const int row = i / kWidth;
        const int dimension = i % kWidth;
        // A bucket need not be a multiple of the 128-row native tile. Every
        // lane still participates in barriers/MMA; padding owns no global Q
        // read and no global result/summary slot.
        q[row * kWidth + llaminar2::cuda::attention_primitives::key_column(row, dimension)] = __float2half_rn(
            query_start + row < query_rows
                ? queries[((query_start + row) * QueryHeads + head) * kWidth + dimension]
                : 0.0f);
    }
    for (int partition = first; partition < partitions; partition += step)
    {
        float acc[32][4] = {};
        float running_m0 = -FLT_MAX, running_m1 = -FLT_MAX;
        float running_l0 = 0.0f, running_l1 = 0.0f;
        const int begin = partition * partition_keys;
        const int end = min(min(begin + partition_keys, kv_length), query_start + kRows + position_offset);
        const int tiles = max(0, (end - begin + kKeys - 1) / kKeys);
        if (partition != first && tiles)
            issue(static_cast<const half *>(keys), static_cast<const half *>(values), stages, ranges,
                KVHeads, kv_head, begin, min(kKeys, end-begin), origin, ring_capacity);
        for (int tile = 0; tile < tiles; ++tile)
        {
            llaminar2::cuda::attention_primitives::waitAllCopies();
            __syncthreads();
            if (tile + 1 < tiles)
                issue(static_cast<const half *>(keys), static_cast<const half *>(values),
                    stages + ((tile+1)&1) * kStageHalves, ranges + ((tile+1)&1) * kWarps, KVHeads, kv_head,
                    begin + (tile+1) * kKeys, min(kKeys, end - begin - (tile+1) * kKeys), origin, ring_capacity);
            const int key_start = begin + tile * kKeys;
            // Entire inactive warps skip arithmetic, but still participate in copy publication.
            if (key_start > query_start + warp * 16 + 15 + position_offset) continue;
            const half *k = stages + (tile&1) * kStageHalves;
            const half *v = k + kKeys * kWidth;
            const unsigned *range=ranges+(tile&1)*kWarps;
            unsigned largest=0;
#pragma unroll
            for(int w=0;w<kWarps;++w)largest=max(largest,range[w]);
            const int shift=largest>0x6b80u ? static_cast<int>(largest>>10)-26+
                static_cast<int>((largest&0x3ffu)>0x380u) : 0;
            const half down_scalar=__ushort_as_half(static_cast<unsigned short>((15-shift)<<10));
            const half2 down=__halves2half2(down_scalar,down_scalar);
            const float up=__int_as_float((127+shift)<<23);
            float score[kKeys/8][4] = {};
#pragma unroll 1
            for (int dimension = 0; dimension < kWidth; dimension += 16)
            {
                unsigned a0,a1,a2,a3;
                const int qr = warp * 16 + (lane & 15);
                llaminar2::cuda::attention_primitives::load_four(q + qr * kWidth + llaminar2::cuda::attention_primitives::key_column(qr, dimension + (lane >> 4) * 8), a0,a1,a2,a3);
#pragma unroll
                for (int n = 0; n < kKeys/16; ++n)
                {
                    const int kr = n * 16 + (lane & 7) + (lane >> 4) * 8;
                    const int kd = dimension + ((lane >> 3)&1) * 8;
                    unsigned b0,b1,b2,b3;
                    llaminar2::cuda::attention_primitives::load_four(k + kr * kWidth + llaminar2::cuda::attention_primitives::key_column(kr,kd), b0,b1,b2,b3);
                    llaminar2::cuda::attention_primitives::advance(a0,a1,a2,a3,b0,b1,score[2*n][0],score[2*n][1],score[2*n][2],score[2*n][3]);
                    llaminar2::cuda::attention_primitives::advance(a0,a1,a2,a3,b2,b3,score[2*n+1][0],score[2*n+1][1],score[2*n+1][2],score[2*n+1][3]);
                }
            }
            float maximum0 = -FLT_MAX, maximum1 = -FLT_MAX;
#pragma unroll
            for (int n = 0; n < kKeys/8; ++n)
            {
                const int key = key_start + n * 8 + 2 * lid;
#pragma unroll
                for (int d = 0; d < 4; ++d)
                {
                    const int absolute_key = key + (d&1);
                    const int query = query_start + (d < 2 ? row0 : row1) + position_offset;
                    score[n][d] = absolute_key < end && absolute_key <= query ? __fmul_rn(score[n][d],scale) : -FLT_MAX;
                }
                maximum0 = fmaxf(maximum0, fmaxf(score[n][0],score[n][1]));
                maximum1 = fmaxf(maximum1, fmaxf(score[n][2],score[n][3]));
            }
#pragma unroll
            for (int delta = 1; delta <= 2; delta <<= 1)
            {
                maximum0 = fmaxf(maximum0,__shfl_xor_sync(0xffffffff,maximum0,delta));
                maximum1 = fmaxf(maximum1,__shfl_xor_sync(0xffffffff,maximum1,delta));
            }
            const float next0 = fmaxf(running_m0,maximum0), next1 = fmaxf(running_m1,maximum1);
            const float scale_l2 = 1.4426950408889634074f;
            const float alpha0 = maximum0 > -FLT_MAX ? __expf(__fsub_rn(running_m0,next0)) : 1.0f;
            const float alpha1 = maximum1 > -FLT_MAX ? __expf(__fsub_rn(running_m1,next1)) : 1.0f;
            // Denominator arithmetic continues to use original P. Only
            // the bounded half tensor operand is scaled; no extra exponential
            // or persistent numerator/denominator state is introduced.
            const int probability_shift0 = probability_shift(maximum0, scale_l2, next0);
            const int probability_shift1 = probability_shift(maximum1, scale_l2, next1);
            const float probability_up0 = __int_as_float((127 + probability_shift0) << 23);
            const float probability_up1 = __int_as_float((127 + probability_shift1) << 23);
            unsigned pa[kKeys/16][4];
            float sum0=0.0f,sum1=0.0f;
#pragma unroll
            for (int n = 0; n < kKeys/8; ++n)
            {
                const float p0 = score[n][0] > -FLT_MAX ? __expf(__fsub_rn(score[n][0],next0)) : 0.0f;
                const float p1 = score[n][1] > -FLT_MAX ? __expf(__fsub_rn(score[n][1],next0)) : 0.0f;
                const float p2 = score[n][2] > -FLT_MAX ? __expf(__fsub_rn(score[n][2],next1)) : 0.0f;
                const float p3 = score[n][3] > -FLT_MAX ? __expf(__fsub_rn(score[n][3],next1)) : 0.0f;
                score[n][0]=p0; score[n][1]=p1; score[n][2]=p2; score[n][3]=p3;
                pa[n>>1][(n&1)*2] = pack(__fmul_rn(p0,probability_up0),__fmul_rn(p1,probability_up0));
                pa[n>>1][(n&1)*2+1] = pack(__fmul_rn(p2,probability_up1),__fmul_rn(p3,probability_up1));
            }
            // Each owner sees keys 0..15 in ascending order. This
            // matches the canonical per-tile denominator without scalar P/V.
#pragma unroll
            for (int group=0; group<kKeys/8; ++group)
            {
#pragma unroll 1
                for (int source_lane=0; source_lane<4; ++source_lane)
                {
#pragma unroll
                    for (int column=0; column<2; ++column)
                    {
                        sum0 += __shfl_sync(0xffffffff,score[group][column],source_lane,4);
                        sum1 += __shfl_sync(0xffffffff,score[group][column+2],source_lane,4);
                    }
                }
            }
            running_l0 = fmaf(running_l0,alpha0,sum0); running_l1 = fmaf(running_l1,alpha1,sum1);
            running_m0 = next0; running_m1 = next1;
#pragma unroll
            for (int n = 0; n < 32; ++n)
            {
                unsigned b[kKeys/16][2];
#pragma unroll
                for (int j = 0; j < kKeys/16; ++j)
                {
                    const int row = j * 16 + (lane & 15);
                    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(v + row * kWidth + llaminar2::cuda::attention_primitives::key_column(row,n*8)));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(b[j][0]),"=r"(b[j][1]) : "r"(address));
                }
                unsigned h0=0,h1=0;
                float value[4]={};
#pragma unroll
                for (int j=0; j<kKeys/16; ++j)
                {
                    if (shift)
                    {
                        reinterpret_cast<half2 &>(b[j][0])=__hmul2(reinterpret_cast<const half2 &>(b[j][0]),down);
                        reinterpret_cast<half2 &>(b[j][1])=__hmul2(reinterpret_cast<const half2 &>(b[j][1]),down);
                    }
                    if constexpr (HalfAcc)
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                            : "+r"(h0),"+r"(h1) : "r"(pa[j][0]),"r"(pa[j][1]),"r"(pa[j][2]),"r"(pa[j][3]),"r"(b[j][0]),"r"(b[j][1]));
                    else llaminar2::cuda::attention_primitives::advance(pa[j][0],pa[j][1],pa[j][2],pa[j][3],b[j][0],b[j][1],value[0],value[1],value[2],value[3]);
                }
                if constexpr (HalfAcc)
                {
                    const float2 a=__half22float2(reinterpret_cast<const half2 &>(h0)), b=__half22float2(reinterpret_cast<const half2 &>(h1));
                    value[0]=a.x;value[1]=a.y;value[2]=b.x;value[3]=b.y;
                }
#pragma unroll
                for (int d=0;d<4;++d)
                {
                    const float inverse_power = __int_as_float(
                        (127 + shift - (d<2?probability_shift0:probability_shift1)) << 23);
                    acc[n][d]=fmaf(acc[n][d],d<2?alpha0:alpha1,
                        __fmul_rn(value[d],inverse_power));
                }
            }
        }
#pragma unroll
        for(int r=0;r<2;++r)
        {
            const int query=query_start+(r?row1:row0);
            if (query >= query_rows) continue;
            const float m=r?running_m1:running_m0,l=r?running_l1:running_l0;
            const size_t index=(static_cast<size_t>(query)*QueryHeads+head)*capacity+partition;
            float *target=direct ? output+(query*QueryHeads+head)*kWidth : partial+index*kWidth;
            const float factor=direct?(l>0.0f?1.0f/l:0.0f):1.0f;
#pragma unroll
            for(int n=0;n<32;++n)
            {
                target[n*8+lid*2]=acc[n][r*2]*factor;
                target[n*8+lid*2+1]=acc[n][r*2+1]*factor;
            }
            if(!direct&&lid==0) { maxima[index]=m;denominators[index]=l; }
        }
        // The partition boundary also retires the two reusable staging slots.
        __syncthreads();
    }
}
} // namespace tensor_prefill_attention
/**
 * @brief Execute the exact captured native phase for one head geometry.
 * @tparam QueryHeads Complete local query-head extent.
 * @tparam KVHeads Complete visible native KV-head extent.
 * @tparam GQA Global replicated mapping, or zero for local KV shards.
 * Parameters use the documented canonical phase ABI. Live length, position
 * and ring geometry are read exclusively from device-owned attention params.
 */
template<int QueryHeads,int KVHeads,int GQA>
__global__ __launch_bounds__(llaminar2::cuda::tensor_prefill::kThreads,1)
void tensor_prefill_kernel(
        const float *__restrict__ Q,
        const void *__restrict__ K,
        const void *__restrict__ V,
        float *__restrict__ O,
        int batch_size,
        int seq_len,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float softmax_scale,
        bool causal,
        int window_size,
        int position_offset,
        const llaminar2::attention::AttentionDeviceParams *__restrict__ device_params,
        const float *__restrict__ mask,
        int tile_q,
        int tile_kv,
        int qkv_pad,
        int scores_pad,
        int head_start,
        int gqa_n_rep,
        float *__restrict__ O_partial,
        float *__restrict__ m_partial,
        float *__restrict__ l_partial,
        int context_partition_size,
        int max_context_partitions,
        int device_direct_partition_limit,
        int explicit_context_partition)
{
    tensor_prefill_attention::run<QueryHeads,KVHeads,GQA,true>(
        Q,K,V,O,seq_len,kv_len,position_offset,device_params,softmax_scale,head_start,
        O_partial,m_partial,l_partial,max_context_partitions,
        device_direct_partition_limit,explicit_context_partition);
}
} // namespace

namespace llaminar2::cuda::tensor_prefill {
Kernel kernel(HeadGeometry geometry) {
    switch (geometry) {
      case HeadGeometry::Local8x1: return tensor_prefill_kernel<8,1,0>;
      case HeadGeometry::Local12x2: return tensor_prefill_kernel<12,2,0>;
      case HeadGeometry::Local16x2: return tensor_prefill_kernel<16,2,0>;
      case HeadGeometry::Local24x4: return tensor_prefill_kernel<24,4,0>;
      case HeadGeometry::Replicated8x2: return tensor_prefill_kernel<8,2,8>;
      case HeadGeometry::Replicated12x4: return tensor_prefill_kernel<12,4,6>;
    }
    throw std::invalid_argument("Invalid CUDA tensor-prefill head geometry");
}
} // namespace llaminar2::cuda::tensor_prefill
