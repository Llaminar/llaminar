/**
 * @file ROCmPersistentHCReadBody.hip.inc
 * @brief Original HC phases on a typed256/1024-thread admitted resident grid.
 *
 * Every global phase edge is visible below. Work is indexed independently of
 * physical residency: four norm tasks, original K32 publications, fixed split-K
 * down tiles plus independent injection, original ordered reducer/publication,
 * then original two-term up/read. Shared memory is reused only after the last
 * reader retires. The caller supplies its admitted grid/join authority; no
 * global allocation or fixed processor count is introduced by this phase body.
 * Include after the original native-M1 arithmetic definitions.
 */
#pragma once
#include "ROCmPersistentHCRead.h"
#include "ROCmResidentGridBarrier.cuh"
#include "kernels/qwen4exp/GPUBranchRMSNormBody.cuh"
#include "kernels/common/OriginalSingleQ8Block32.cuh"
#include "kernels/common/GPUHCShortKBlockBody.cuh"
#include "ROCmSmallNFP32ProjectionBody.hip.inc"
namespace llaminar2::rocm {
/** @brief Phase-disjoint CTA storage for one exact physical issue geometry.
 * @tparam Threads Original standalone256 or composed1024 physical threads.
 */
template<unsigned Threads> union PersistentHCReadSharedFor {
 float reduction[256]; ///< Original logical256 norm/injection tree.
 float down[PersistentHCReadGeometry<Threads>::down_window][PersistentHCReadGeometry<Threads>::down_columns]; ///< Complete original K32 contributions.
 struct {float primary[16][32],residual[16][32];} up; ///< Separate original two-term folds.
};
/** @brief Existing standalone storage name preserves its exact8KiB contract. */
using PersistentHCReadShared=PersistentHCReadSharedFor<256>;
static_assert(sizeof(PersistentHCReadShared)==8192);
static_assert(sizeof(PersistentHCReadSharedFor<1024>)==4096);
/**
 * @brief One fixed-partition down tile with original per-K32 rounded contributions.
 * @param a Frozen original source and bank binding.
 * @param shared Phase-exclusive LDS sized by the declared physical geometry.
 * @param tile Output64-column tile, with exact original N stride.
 * @param partition Original arithmetic partition; independent of resident grid.
 * @tparam Threads Explicit physical issue geometry; arithmetic partitions do not change.
 *
 * Physical column cohorts create independent complete signed dot8 chains.
 * Cohort0 alone folds actual K32 contributions ascending. The accumulator survives
 * bounded windows, so P1/long K introduce no regrouping or padded zero additions.
 */
template<unsigned Threads>
__device__ __forceinline__ void persistentHCDownTask(const PersistentHCReadArgs& a,
 PersistentHCReadSharedFor<Threads>& shared,unsigned tile,unsigned partition) {
 using Geometry=PersistentHCReadGeometry<Threads>;
 constexpr unsigned Columns=Geometry::down_columns,Contributors=Geometry::down_contributors,
  Window=Geometry::down_window;
 const unsigned group=threadIdx.x/Columns,column=threadIdx.x%Columns;
 const int n=static_cast<int>(tile*Columns+column),N=a.down_weights.n;
 const int blocks=a.down_weights.k/32,per=(blocks+a.partitions-1)/a.partitions;
 const int first=static_cast<int>(partition)*per,last=min(first+per,blocks);
 float sum=0.0f;
 for(int begin=first;begin<last;begin+=Window) {
  const int count=min(static_cast<int>(Window),last-begin);
  for(int offset=static_cast<int>(group);offset<count;offset+=Contributors)if(n<N) {
   const int b=begin+offset;
   const auto linear=static_cast<std::size_t>(b)*N+n;
   const auto* weight=reinterpret_cast<const int32_t*>(a.down_weights.payload+linear*32);
   const auto* input=reinterpret_cast<const int32_t*>(a.input_codes+b*32);
   int32_t dot=0;
#pragma unroll
   for(int word=0;word<8;++word)
    dot=::llaminar2::rocm_integer_contract::signedDot4(input[word],weight[word],dot);
   const float scale=rocm_native_quantised_load_single_weight_scale<NATIVE_QUANTISED_Q8_0>(
    static_cast<const uint16_t*>(a.down_weights.scales)[linear],uint16_t{0});
   shared.down[offset][column]=::llaminar2::device_native_quantised_contract::singleScaleBlock(
    dot,scale,a.input_scales[b]);
  }
  __syncthreads();
  if(group==0&&n<N)for(int offset=0;offset<count;++offset)
   sum=rocm_native_quantised_accumulate_rn(sum,shared.down[offset][column]);
  __syncthreads();
 }
 if(group==0&&n<N) {
  // Retain the original explicit final multiply even though block scales own
  // the per-K32 scale and the original terminal row multiplier is exactly1.
  if(a.partitions==1)a.raw_down[n]=sum*1.0f;
  else a.down_partials[static_cast<std::size_t>(partition)*N+n]=sum*1.0f;
 }
}
/**
 * @brief Original short-K logical tasks mapped onto a declared physical CTA.
 * @param a Exact original two-term operands/public outputs.
 * @param shared Phase-exclusive original contribution planes.
 * @param tile Eight mixed channels and four branch-major logit segments.
 * @tparam Threads Physical visit stride; never a numerical grouping choice.
 *
 * The logical K32x32 tasks need up to512 workers. Two256-thread rounds or one
 *1024-thread visit retain the same contributions; every physical lane joins.
 * The32 original column owners fold primary/residual independently and add once.
 */
template<unsigned Threads>
__device__ __forceinline__ void persistentHCUpReadTask(const PersistentHCReadArgs& a,
 PersistentHCReadSharedFor<Threads>& shared,unsigned tile) {
 const unsigned H=a.norm.hidden,groups=a.up.weights.blocks_per_row;
 const bool live=a.up.rows.activeRows()!=0;
 for(unsigned logical=threadIdx.x;logical<groups*32;logical+=Threads) {
  const unsigned group=logical/32,lane=logical%32,channel=tile*8+lane%8;
  const std::uint64_t column=static_cast<std::uint64_t>(lane/8)*H+channel;
  float p=0.0f,q=0.0f;
  if(live&&channel<H)hc_short_k_native::contribution(a.up,group,column,p,q);
  shared.up.primary[group][lane]=p;shared.up.residual[group][lane]=q;
 }
 __syncthreads();
 if(threadIdx.x<32&&live) {
  const unsigned lane=threadIdx.x,channel=tile*8+lane%8;
  if(channel<H) {
   float p=0.0f,q=0.0f;
   for(unsigned group=0;group<groups;++group) {
    p=device_fp32_contract::add(p,shared.up.primary[group][lane]);
    q=device_fp32_contract::add(q,shared.up.residual[group][lane]);
   }
   const float value=device_fp32_contract::add(p,q);
   a.up.output[static_cast<std::uint64_t>(lane/8)*H+channel]=value;
   shared.up.primary[0][lane]=value;
  }
 }
 __syncthreads();
 if(threadIdx.x<8&&live) {
  const unsigned channel=tile*8+threadIdx.x;
  if(channel<H) {
   float mixed=0.0f;
#pragma unroll
   for(unsigned branch=0;branch<4;++branch) {
    const float gate=1.0f/(1.0f+expf(-shared.up.primary[0][branch*8+threadIdx.x]));
    mixed=fmaf(gate,a.norm.output[static_cast<std::uint64_t>(branch)*H+channel],mixed);
   }
   a.mixed[channel]=mixed*0.25f;
  }
 }
 // Every reader must retire before the CTA reuses these planes for its next tile.
 __syncthreads();
}
/**
 * @brief Complete participant-local HC prefix, ready for a following resident phase.
 * @tparam PublishNextPhase True publishes mixed/injection to a following grid phase;
 * false uses the standalone kernel completion edge without a redundant final fence.
 * @param a Original immutable declaration and existing physical bank pointers.
 * @param shared Caller-owned phase-disjoint8KiB(256) or4KiB(1024) CTA storage.
 * @param grid Caller-owned admitted grid; every physical CTA participates in each edge.
 * @param write Explicit original write view, or typed existing-input boundary.
 * @tparam Threads Physical256/1024 issue geometry; logical norm/injection stay256.
 * @tparam Grid Existing cooperative or instance-owned resident join authority.
 * @tparam Write PersistentHCNoWrite or the explicit original PersistentHCWriteInput.
 * @details Four internal edges retire norm, input quantization, down, and down
 * publication; PublishNextPhase adds the fifth mixed/injection-ready edge. The
 * caller's cold owner authenticates all count/source/storage/write identities.
 */
template<bool PublishNextPhase=true,unsigned Threads=256,class Grid,class Write=PersistentHCNoWrite>
__device__ __forceinline__ void persistentHCReadPhase(const PersistentHCReadArgs& a,
 PersistentHCReadSharedFor<Threads>& shared,Grid& grid,Write write={}) {
 static_assert(std::is_same_v<Write,PersistentHCNoWrite>||std::is_same_v<Write,PersistentHCWriteInput>,
  "HC prefix requires its typed existing-input or original residual-write boundary");
 using Geometry=PersistentHCReadGeometry<Threads>;
 const unsigned lane=threadIdx.x,H=a.norm.hidden,R=a.down_weights.n,K=a.down_weights.k;
 const bool logicalMember=lane<256;
 int normRows=a.norm.extent.liveRows?*a.norm.extent.liveRows:1;
 if(normRows<0||normRows>1)__builtin_trap();
 if(normRows)for(unsigned task=blockIdx.x;task<4;task+=gridDim.x) {
  if constexpr(std::is_same_v<Write,PersistentHCNoWrite>)
   qwen4exp::branch_norm_native::task(a.norm.input,a.norm.gain,a.norm.output,H,a.norm.epsilon,
    task,shared.reduction,lane,logicalMember);
  else
   qwen4exp::branch_norm_native::writeTask(write.residual,write.block,write.gateLogits,write.output,
    a.norm.gain,a.norm.output,H,a.norm.epsilon,task,shared.reduction,lane,logicalMember);
  __syncthreads();
 }
 grid.sync(); // All normalized values, including retained count0 values, are now readable.
 for(unsigned block=blockIdx.x*Geometry::quant_cohorts+lane/32;block<K/32;block+=gridDim.x*Geometry::quant_cohorts) {
  const unsigned index=block*32+lane%32;
  original_single_q8::publish(a.norm.output[index],true,a.input_codes+block*32,
   a.input_scales+block,nullptr,lane%32);
 }
 grid.sync(); // Every down tile sees the same complete original SingleQ8 row.
 const unsigned columnTiles=(R+Geometry::down_columns-1)/Geometry::down_columns,downTasks=columnTiles*a.partitions;
 const unsigned injectionTasks=a.injection.weights?4:0;
 for(unsigned task=blockIdx.x;task<downTasks+injectionTasks;task+=gridDim.x) {
  if(task<downTasks)persistentHCDownTask<Threads>(a,shared,task%columnTiles,task/columnTiles);
  else if(a.injection.rows.activeRows()) {
   const unsigned output=task-downTasks;
   float value;
   if(a.injection.load_pairs==8)value=floating_projection_native::row<8>(a.norm.output,a.injection.weights+static_cast<std::size_t>(output)*K,K,shared.reduction,lane,logicalMember);
   else if(a.injection.load_pairs==4)value=floating_projection_native::row<4>(a.norm.output,a.injection.weights+static_cast<std::size_t>(output)*K,K,shared.reduction,lane,logicalMember);
   else value=floating_projection_native::row<1>(a.norm.output,a.injection.weights+static_cast<std::size_t>(output)*K,K,shared.reduction,lane,logicalMember);
   if(lane==0)a.injection.output[output]=value;
  }
  __syncthreads();
 }
 grid.sync(); // Down no longer reads QUANT_A/scales; reducer may reuse their prefix.
 const bool publish=a.publication.rows.activeRows()!=0;
 for(unsigned column=blockIdx.x*Threads+lane;column<R;column+=gridDim.x*Threads) {
  const float raw=a.partitions==1?a.raw_down[column]:nativeM1FoldColumnValue(a.down_partials,column,R,a.partitions);
  a.raw_down[column]=raw;
  if(publish)hc_down_publication::publish(raw,a.publication,column);
 }
 grid.sync(); // Two-term banks and public activated row are complete before up.
 for(unsigned tile=blockIdx.x;tile<(H+7)/8;tile+=gridDim.x)persistentHCUpReadTask<Threads>(a,shared,tile);
 if constexpr(PublishNextPhase)grid.sync();
}
/** @brief Standalone complete HC operation; all public outputs remain in their original owners. */
#if defined(LLAMINAR_DEFINE_PERSISTENT_HC_READ_KERNEL)
__global__ __launch_bounds__(256) void persistentHCReadKernel(PersistentHCReadArgs args,std::uint32_t* barrier_state) {
 ResidentGridBarrierFor<4> grid{barrier_state};
 __shared__ PersistentHCReadShared shared;
 persistentHCReadPhase<false>(args,shared,grid);
}
#endif
} // namespace llaminar2::rocm
