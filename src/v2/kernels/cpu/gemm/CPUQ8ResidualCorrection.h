/**
 * @file CPUQ8ResidualCorrection.h
 * @brief Native prepared-weight two-term Q8 publication and SIMD projection.
 *
 * Existing VNNI microkernels retain their canonical block contributions and
 * independent register accumulators; a typed store policy folds each real
 * row's two final terms once. Two real rows share immutable weight loads.
 * Canonical native source codebooks keep their original packed decode, minima
 * and subscale arithmetic; only the independently quantized activation changes.
 * Output/scratch holes are untouched. The CPU authority validates the complete
 * live prefix before starting its workshare. The operation owns no hot heap
 * allocation; standalone OpenMP team runtime overhead is shared with SingleQ8.
 * HC two-term input publication retains the original activated FP32 value,
 * backend nonlinear arithmetic and prepared projection/workspace authority.
 * HC publication owners span the original256 row-local channels, retaining
 * bounded locality while exposing shallow rows without another team launch.
 * Both publication policies enter one shared non-inlined projection workshare,
 * preserving its code generation independently of nonlinear publication code.
 */
#pragma once
#include "CPUNativeVNNIGemv.h"
#include "kernels/common/Q8ResidualCorrection.h"
#include <span>
#include <type_traits>
#include "kernels/cpu/qwen4exp/CPUQwen4ExpSIMD.h"
namespace llaminar2::cpu::native_vnni {
/** @brief Form one quantized residual against the first block's stored scale.
 * @param input Thirty-two original FP32 values.
 * @param primary Already published unchanged native primary Q8 block.
 * @param residual Exact secondary Q8 destination.
 * @param policy Existing prepared source's activation/reduction contract.
 */
inline void quantizeResidualBlock(const float*input,const Q8_1Block&primary,Q8_1Block&residual,
 CPUProjectionNumericalPolicy policy) {
 alignas(64)float values[32];const auto scale=_mm256_set1_ps(nativeVNNIFP16ScaleToFP32(primary.d));
 for(int i=0;i<32;i+=8) {
  const auto bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(primary.qs+i));
  const auto decoded=persistRoundedAVX2(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)),scale));
  _mm256_store_ps(values+i,_mm256_sub_ps(_mm256_loadu_ps(input+i),decoded));
 }
 quantizeActivationBlock(values,residual,32,policy);
}
/** @brief Validate and borrow actual immutable native source storage.
 * @param packed Exact prepared engine export; no repack or duplicate source exists.
 * @param m Positive physical row capacity.
 * @param n Exact output width.
 * @param k Exact reduction width.
 * @throws std::invalid_argument For an unsupported source tuple or incomplete shape.
 */
inline void validateCPUQ8ResidualSource(const CPUNativeVNNIPackedWeights&packed,int m,int n,int k) {
 validateQ8ResidualCorrectionGeometry(m,n,k);
 const auto &format=requireQ8ResidualSourceFormat(
  NativeVnniSourceIdentity{packed.codebook_id,packed.is_superblock,true},m,n,k);
 if(packed.N!=n||packed.K!=k||packed.blocks_per_row!=k/32||!packed.hasInterleavedData()||
    packed.is_asymmetric!=format.is_asymmetric||packed.payload_bytes!=format.payload_bytes||
    packed.encoding!=preparedEncodingForSource(format)||
    packed.data_stride!=preparedDataStride(packed.encoding)||
    packed.interleaved_block_stride!=preparedInterleavedBlockStride(packed.encoding,packed.is_asymmetric)||
    packed.numerical_policy!=CPUProjectionNumericalPolicy::BackendNative)
  throw std::invalid_argument("Two-term CPU quantized projection requires its exact canonical prepared source tuple and geometry");
}
/**
 * @brief Evaluate two real rows as four independent original-source dot chains.
 * @param packed Immutable canonical nibble, expanded, Q6 or multi-scale source.
 * @param primary0 First real row's unchanged native activation blocks.
 * @param residual0 First real row's independently quantized residual blocks.
 * @param primary1 Second row's blocks, or null for an odd final row.
 * @param residual1 Second row's residual blocks, paired with primary1.
 * @param destination0 First row's valid output interval.
 * @param destination1 Second row's valid output interval, or null.
 * @param chunk Original64-column packed-source chunk.
 * @param valid Actual output columns within that final chunk.
 * @param avx512 Actual runtime ISA selected before the workshare.
 *
 * These are existing grouped microkernels, not serial M1 inference replay.
 * Each chain visits the complete K range in its original order, including
 * source-specific minima and IQ subscales. A bounded stack tile receives only
 * four64-column fragments; the two final terms fold once without an MxN mirror.
 * Q8_0 retains its existing fused-store fast path outside this helper.
 */
inline void cpuNativeResidualChunk(const CPUNativeVNNIPackedWeights &packed,
 const Q8_1Block *primary0,const Q8_1Block *residual0,
 const Q8_1Block *primary1,const Q8_1Block *residual1,
 float *destination0,float *destination1,int chunk,int valid,bool avx512)
{
 alignas(64) float term0[64],term1[64],term2[64],term3[64];
 const int blocks=packed.blocks_per_row;
 const bool nibble=packed.usesNibbleLUT();
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
 if(avx512)
 {
  const auto lut=nibble?build_decode_lut(packed.codebook_id):_mm512_setzero_si512();
  if(primary1)
  {
   if(nibble)gemm_4row_native_1z_chunk(packed,primary0,residual0,primary1,residual1,
     term0,term1,term2,term3,chunk,0,blocks,lut,false);
   else gemm_4row_non_nibble_2z_chunk(packed,primary0,residual0,primary1,residual1,
     term0,term1,term2,term3,chunk,0,blocks,false);
  }
  else if(nibble)gemm_2row_native_chunk(packed,primary0,residual0,term0,term1,chunk,0,blocks,lut,false);
  else gemm_2row_non_nibble_chunk(packed,primary0,residual0,term0,term1,chunk,0,blocks,false);
 }
 else
#endif
 {
  const auto lut=nibble?build_decode_lut_avx2_for_codebook(packed.codebook_id):_mm256_setzero_si256();
  if(nibble)gemm_2row_native_chunk_avx2(packed,primary0,residual0,term0,term1,chunk,0,blocks,lut,false);
  else gemm_2row_non_nibble_chunk_avx2(packed,primary0,residual0,term0,term1,chunk,0,blocks,false);
  if(primary1)
  {
   if(nibble)gemm_2row_native_chunk_avx2(packed,primary1,residual1,term2,term3,chunk,0,blocks,lut,false);
   else gemm_2row_non_nibble_chunk_avx2(packed,primary1,residual1,term2,term3,chunk,0,blocks,false);
  }
 }
 for(int column=0;column<64;column+=8)
 {
  _mm256_store_ps(term0+column,_mm256_add_ps(_mm256_load_ps(term0+column),_mm256_load_ps(term1+column)));
  if(primary1)_mm256_store_ps(term2+column,_mm256_add_ps(_mm256_load_ps(term2+column),_mm256_load_ps(term3+column)));
 }
 std::copy_n(term0,valid,destination0);
 if(primary1)std::copy_n(term2,valid,destination1);
}
/**
 * @brief Run one shared compiled projection workshare after either publication policy.
 * @param packed Admitted immutable prepared source, shared by all team members.
 * @param output Admitted live-row FP32 destination.
 * @param live Already validated authoritative live row count.
 * @param primary Complete published primary blocks; read-only after the barrier.
 * @param residual Complete published residual blocks; read-only after the barrier.
 * @param avx512 Existing runtime projection ISA choice, resolved before publication.
 * @pre Every member of the existing team calls after the same finite-publication
 * check and the publication workshare's implicit barrier. This helper creates
 * no team, allocates no storage and retains the original static workshare.
 * @details Keep dispatch and task-local tail storage outside the publication
 * template so HC and already-activated inputs execute the same compiled loop.
 * GCC noipa also prevents cloning/constant-propagation variants; other supported
 * compilers retain an explicit noinline boundary whose actual callee identity
 * must be authenticated by native ISA evidence before performance acceptance.
 */
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noipa))
#else
__attribute__((noinline))
#endif
inline void cpuQ8ResidualProjectionWorkshare(const CPUNativeVNNIPackedWeights&packed,
 float*output,int live,std::span<const Q8_1Block>primary,std::span<const Q8_1Block>residual,bool avx512) {
 const int n=packed.N,bpr=packed.blocks_per_row;
 const std::int64_t tiles=(static_cast<std::int64_t>(live)+1)/2,chunks=(static_cast<std::int64_t>(n)+63)/64;
#pragma omp for schedule(static)
  for(std::int64_t task=0;task<tiles*chunks;++task) {
   const auto row=task%tiles*2,chunk=task/tiles;
   const int valid=std::min<std::int64_t>(64,n-chunk*64);
   // Preserve the original Q8_0 kernel and its register-store fold exactly.
   // All other canonical encodings retain their own source arithmetic above.
   if(packed.codebook_id!=native_vnni_formats::Q8_0.codebook_id||packed.is_superblock||
      packed.is_asymmetric||!packed.usesExpandedInt8())
   {
    const bool second=row+1<live;
    cpuNativeResidualChunk(packed,primary.data()+row*bpr,residual.data()+row*bpr,
      second?primary.data()+(row+1)*bpr:nullptr,second?residual.data()+(row+1)*bpr:nullptr,
      output+row*n+chunk*64,second?output+(row+1)*n+chunk*64:nullptr,chunk,valid,avx512);
    continue;
   }
   alignas(64) float tail0[64],tail1[64];
   float *destination0=valid==64?output+row*n+chunk*64:tail0;
   float *destination1=row+1<live?(valid==64?output+(row+1)*n+chunk*64:tail1):nullptr;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
   if(avx512&&row+1<live) {
    gemm_4row_non_nibble_2z_chunk<true>(packed,primary.data()+row*bpr,residual.data()+row*bpr,
      primary.data()+(row+1)*bpr,residual.data()+(row+1)*bpr,
      destination0,nullptr,destination1,nullptr,chunk,0,bpr,false);
    if(valid<64){std::copy_n(tail0,valid,output+row*n+chunk*64);std::copy_n(tail1,valid,output+(row+1)*n+chunk*64);}
    continue;
   }
#endif
   for(auto current=row;current<std::min<std::int64_t>(row+2,live);++current) {
    float*destination=current==row?destination0:destination1;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    if(avx512)gemm_2row_non_nibble_chunk<true>(packed,primary.data()+current*bpr,residual.data()+current*bpr,
      destination,nullptr,chunk,0,bpr,false);
    else
#endif
      gemm_2row_non_nibble_chunk_avx2<true>(packed,primary.data()+current*bpr,residual.data()+current*bpr,
        destination,nullptr,chunk,0,bpr,false);
    if(valid<64)std::copy_n(destination,valid,output+current*n+chunk*64);
   }
  }
}
/** @brief Already-activated source retains the original publisher. */
struct ActivatedQ8Publication {};
/** @brief HC publication additionally authors the existing FP32 activation output. */
struct HCDownQ8Publication { float* activated; };
/** @brief Execute the explicit paired-Q8 operation over one borrowed prepared source.
 * @param packed Actual prepared native source with fixed arithmetic policy.
 * @param input Distinct host-valid FP32 rows.
 * @param output Host-owned destination; only live rows are authored.
 * @param rows Immutable capacity and optional CPU-owned authoritative count.
 * @param primary Participant-owned first Q8 block bank.
 * @param residual Independently owned second Q8 block bank.
 * @param publication Original activated input or typed HC activation destination.
 * @tparam Publication Compile-time semantic input operation, never a runtime skip.
 */
template<class Publication>
inline void cpuQ8ResidualCorrectedProjectionImpl(const CPUNativeVNNIPackedWeights&packed,const float*input,
 float*output,DeviceRowRange rows,std::span<Q8_1Block>primary,std::span<Q8_1Block>residual,Publication publication) {
 const int physical=rows.physicalRows(),n=packed.N,k=packed.K,bpr=packed.blocks_per_row;
 validateCPUQ8ResidualSource(packed,physical,n,k);
 const auto needed=cpuProjectionQ8Blocks(physical,k);
 if(!input||!output||input==output||primary.size()<needed||residual.size()<needed||primary.data()==residual.data())
  throw std::invalid_argument("Two-term CPU Q8 requires distinct prepared operands and complete admitted primary/residual banks");
 const std::array writes{
  Q8ResidualStorageSpan{output,static_cast<std::size_t>(physical)*n*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{primary.data(),needed*sizeof(Q8_1Block),alignof(Q8_1Block)},
  Q8ResidualStorageSpan{residual.data(),needed*sizeof(Q8_1Block),alignof(Q8_1Block)}};
 const std::array reads{
  Q8ResidualStorageSpan{input,static_cast<std::size_t>(physical)*k*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{packed.interleavedBase(),interleavedWorkspaceSize(packed),1},
  Q8ResidualStorageSpan{rows.countOwner(),rows.countOwner()?sizeof(std::int32_t):0,alignof(std::int32_t)}};
 validateQ8ResidualStorage(writes,reads);
 if constexpr(std::is_same_v<Publication,HCDownQ8Publication>) {
  const auto bytes=static_cast<std::size_t>(physical)*k*sizeof(float);
  const std::array activationWrites{Q8ResidualStorageSpan{publication.activated,bytes,alignof(float)}};
  const std::array activationReads{
   Q8ResidualStorageSpan{input,publication.activated==input?0:bytes,alignof(float)},
   reads[1],reads[2],writes[0],writes[1],writes[2]};
  if(!publication.activated)throw std::invalid_argument("HC CPU requires its retained activation output");
  validateQ8ResidualStorage(activationWrites,activationReads);
 }
 const int live=rows.activeRowsFor(rows.countOwner()?*rows.countOwner():rows.countCapacity());
 if(live<0)throw std::invalid_argument("Two-term CPU Q8 live count exceeds retained capacity");
 if(live==0)return;
 const bool avx512=activeISALevel()>=ISALevel::AVX512;
 const int publicationWidth=[] {
  if constexpr(std::is_same_v<Publication,HCDownQ8Publication>)return qwen4exp::cpu_detail::vectorWidth();
  else return 1;
 }();
 // HC retains the original row-local256-channel activation grain. Each task
 // publishes its own eight-or-fewer complete Q8 blocks, so M1/K320 exposes
 // two independent owners instead of serializing all SiLU work on one row.
 // Wider tasks also amortize ISA dispatch without ten adjacent block writers
 // contending on the compact36-byte Q8 bank cache lines. Ordinary already-
 // activated publication keeps one task per row and its original block loop.
 constexpr bool hcPublication=std::is_same_v<Publication,HCDownQ8Publication>;
 const std::int64_t publicationTilesPerRow=hcPublication?(static_cast<std::int64_t>(bpr)+7)/8:1;
 const auto finitePublication=[&]() {
  for(std::size_t block=0;block<static_cast<std::size_t>(live)*bpr;++block)
   if((primary[block].d&0x7c00u)==0x7c00u)return false;
  return true;
 };
 auto work=[&]() {
#pragma omp for schedule(static)
  for(std::int64_t task=0;task<static_cast<std::int64_t>(live)*publicationTilesPerRow;++task) {
   const auto row=task/publicationTilesPerRow;
   const int firstBlock=hcPublication?static_cast<int>(task%publicationTilesPerRow)*8:0;
   const int lastBlock=hcPublication?std::min(firstBlock+8,bpr):bpr;
   if constexpr(std::is_same_v<Publication,HCDownQ8Publication>) {
    //256 and every admitted K are multiples of32, hence each SIMD vector has
    // the same original row/channel boundary. Exact input/activated alias is
    // safe: another tile never reads or writes any channel owned by this tile.
    const auto base=static_cast<std::size_t>(row)*k;
    qwen4exp::cpu_detail::residualDownChannels(input+base,publication.activated+base,
      firstBlock*32,lastBlock*32,publicationWidth);
   }
   for(int block=firstBlock;block<lastBlock;++block) {
   const auto offset=static_cast<std::size_t>(row)*bpr+block;
   const auto*source=input+static_cast<std::size_t>(row)*k+block*32;
   if constexpr(std::is_same_v<Publication,HCDownQ8Publication>) {
    source=publication.activated+static_cast<std::size_t>(row)*k+block*32;
   }
   bool finite=true;for(int element=0;element<32;++element)finite=finite&&std::isfinite(source[element]);
   if(!finite){primary[offset].d=0x7c00u;continue;}
   quantizeActivationBlock(source,primary[offset],32,packed.numerical_policy);
   if((primary[offset].d&0x7c00u)!=0x7c00u)
    quantizeResidualBlock(source,primary[offset],residual[offset],packed.numerical_policy);
   }
  }
  // The first workshare's implicit barrier authenticates every active primary
  // scale. All workers observe the same bank and skip output on any invalid
  // block, without a second flag authority or a throw inside this workshare.
  if(finitePublication()) {
   cpuQ8ResidualProjectionWorkshare(packed,output,live,primary,residual,avx512);
  }
 };
 OMP_WORKSHARE_REGION(work);
 if(!finitePublication())throw std::invalid_argument("Two-term CPU Q8 activation is nonfinite or its native primary half scale overflows");
}
/** @brief Preserve the original already-activated CPU entrypoint and source program.
 * @param packed Exact retained prepared source.
 * @param input Original activated FP32 rows.
 * @param output Distinct retained FP32 projection output.
 * @param rows Original physical/live prefix authority.
 * @param primary Admitted primary Q8 block bank.
 * @param residual Admitted residual Q8 block bank.
 */
inline void cpuQ8ResidualCorrectedProjection(const CPUNativeVNNIPackedWeights& packed,const float* input,
 float* output,DeviceRowRange rows,std::span<Q8_1Block> primary,std::span<Q8_1Block> residual) {
 cpuQ8ResidualCorrectedProjectionImpl(packed,input,output,rows,primary,residual,ActivatedQ8Publication{});
}
/** @brief Author original HC activation and both banks before the same projection workshare.
 * @param packed Exact retained prepared source.
 * @param input Original raw down-projection values.
 * @param activated Retained original /4 SiLU result; exact input alias is permitted.
 * @param output Distinct retained FP32 projection output.
 * @param rows Original physical/live prefix authority.
 * @param primary Admitted primary Q8 block bank.
 * @param residual Admitted residual Q8 block bank.
 */
inline void cpuHCResidualDownQ8Projection(const CPUNativeVNNIPackedWeights& packed,const float* input,float* activated,
 float* output,DeviceRowRange rows,std::span<Q8_1Block> primary,std::span<Q8_1Block> residual) {
 cpuQ8ResidualCorrectedProjectionImpl(packed,input,output,rows,primary,residual,HCDownQ8Publication{activated});
}

}
