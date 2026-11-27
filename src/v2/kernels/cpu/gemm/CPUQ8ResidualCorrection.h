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
#include "CPUNativeQuantisedGemv.h"
#include "kernels/common/Q8ResidualCorrection.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
#if defined(__aarch64__) && defined(__ARM_NEON)
#include "CPUNativeNEONResidualCorrection.h"
#endif
#include <span>
#include <type_traits>
#include <atomic>
#include <limits>
#include "kernels/cpu/qwen4exp/CPUQwen4ExpSIMD.h"
namespace llaminar2::cpu::native_quantised {
#if defined(__aarch64__) && defined(__ARM_NEON)
/** @brief Native signed-byte instruction identity, resolved before publication. */
using CPUQ8ResidualISA = multi_scale::NEONByteDot;
#else
/** @brief Existing x86 choice: true is AVX512, false is AVX2. */
using CPUQ8ResidualISA = bool;
#endif
/** @brief Form one quantized residual against the first block's stored scale.
 * @param input Thirty-two original FP32 values.
 * @param primary Already published unchanged native primary Q8 block.
 * @param residual Exact secondary Q8 destination.
 * @param policy Existing prepared source's activation/reduction contract.
 */
inline void quantizeResidualBlock(const float*input,const Q8_1Block&primary,Q8_1Block&residual,
 CPUProjectionNumericalPolicy policy) {
#if defined(__aarch64__) && defined(__ARM_NEON)
 quantizeResidualBlockNEON(input,primary,residual,policy);
#else
 alignas(64)float values[32];
 const float scale_f = nativeQuantisedFP16ScaleToFP32(primary.d);
#if defined(__AVX512F__) && defined(__AVX512BW__)
 if (activeISALevel() >= ISALevel::AVX512) {
  const __m512 scale = _mm512_set1_ps(scale_f);
  for (int i = 0; i < 32; i += 16) {
   const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(primary.qs + i));
   __m512 decoded = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes)), scale);
   asm volatile("" : "+v"(decoded));
   _mm512_store_ps(values + i, _mm512_sub_ps(_mm512_loadu_ps(input + i), decoded));
  }
  quantizeActivationBlock(values, residual, 32, policy);
  return;
 }
#endif
 const auto scale=_mm256_set1_ps(scale_f);
 for(int i=0;i<32;i+=8) {
  const auto bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(primary.qs+i));
  const auto decoded=persistRoundedAVX2(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)),scale));
  _mm256_store_ps(values+i,_mm256_sub_ps(_mm256_loadu_ps(input+i),decoded));
 }
 quantizeActivationBlock(values,residual,32,policy);
#endif
}
/** @brief Validate and borrow actual immutable native source storage.
 * @param packed Exact prepared engine export; no repack or duplicate source exists.
 * @param m Positive physical row capacity.
 * @param n Exact output width.
 * @param k Exact reduction width.
 * @throws std::invalid_argument For an unsupported source tuple or incomplete shape.
 */
inline void validateCPUQ8ResidualSource(const CPUNativeQuantisedPackedWeights&packed,int m,int n,int k) {
 validateQ8ResidualCorrectionGeometry(m,n,k);
 const auto &format=requireQ8ResidualSourceFormat(
  NativeQuantisedSourceIdentity{packed.codebook_id,packed.is_superblock,true},m,n,k);
 if(packed.N!=n||packed.K!=k||packed.blocks_per_row!=k/32||!packed.hasInterleavedData()||
    packed.is_asymmetric!=format.is_asymmetric||
    (packed.payload_bytes!=format.payload_bytes&&!((packed.usesExpandedInt8()||packed.usesUnsignedInt8()||packed.usesNibbleLUT())&&packed.payload_bytes==preparedDataStride(packed.encoding)/64))||
    packed.encoding!=preparedEncodingForSource(format)||
    std::int64_t(packed.data_stride)!=preparedDataStride(packed.encoding)||
    std::int64_t(packed.interleaved_block_stride)!=preparedInterleavedBlockStride(packed.encoding,packed.is_asymmetric)||
    packed.numerical_policy!=CPUProjectionNumericalPolicy::BackendNative)
  throw std::invalid_argument("Two-term CPU quantized projection requires its exact canonical prepared source tuple and geometry");
#if defined(__aarch64__) && defined(__ARM_NEON)
 // Reject unsealed/deferred storage and unimplemented tuples before any team.
 CPUNativeNEONProjectionTile<1>::requireExecutable(packed.readView());
#endif
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
inline void cpuNativeResidualChunk(const CPUNativeQuantisedPackedWeights &packed,
 const Q8_1Block *primary0,const Q8_1Block *residual0,
 const Q8_1Block *primary1,const Q8_1Block *residual1,
 float *destination0,float *destination1,int chunk,int valid,CPUQ8ResidualISA avx512)
{
#if defined(__aarch64__) && defined(__ARM_NEON)
#if LLAMINAR_COMPILED_WITH_NEON_I8MM
 if(avx512==multi_scale::NEONByteDot::I8MM)
  return nativeNEONResidualChunk<multi_scale::NEONByteDot::I8MM>(packed.readView(),primary0,residual0,primary1,residual1,
    destination0,destination1,chunk,valid);
#endif
#if defined(__ARM_FEATURE_DOTPROD)
 if(avx512==multi_scale::NEONByteDot::DotProd)
  return nativeNEONResidualChunk<multi_scale::NEONByteDot::DotProd>(packed.readView(),primary0,residual0,primary1,residual1,
    destination0,destination1,chunk,valid);
#endif
 if(avx512!=multi_scale::NEONByteDot::ASIMD)
  throw std::invalid_argument("Native ARM residual projection instruction was not compiled");
 return nativeNEONResidualChunk<multi_scale::NEONByteDot::ASIMD>(packed.readView(),primary0,residual0,primary1,residual1,
   destination0,destination1,chunk,valid);
#else
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
 // The final sum owns the caller's row. Mask the last store so a short
 // output never writes into the following row or the caller's guard bytes.
 const auto publish = [valid](const float *primary, const float *residual, float *destination) {
  for (int column = 0; column < valid; column += 8)
   storeNativeQuantisedLiveAVX2(destination, valid, column,
    _mm256_add_ps(_mm256_load_ps(primary + column), _mm256_load_ps(residual + column)));
 };
 publish(term0, term1, destination0);
 if (primary1) publish(term2, term3, destination1);
#endif
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
 * @details Keep dispatch and bounded output stores outside the publication
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
inline void cpuQ8ResidualProjectionWorkshare(const CPUNativeQuantisedPackedWeights&packed,
 float*output,int live,std::span<const Q8_1Block>primary,std::span<const Q8_1Block>residual,CPUQ8ResidualISA avx512) {
 const int n=packed.N,bpr=packed.blocks_per_row;
 const std::int64_t tiles=(static_cast<std::int64_t>(live)+1)/2,chunks=(static_cast<std::int64_t>(n)+63)/64;
 // For single-row M=1 projections (e.g. HC up with bpr=10 = 360B/bank), copy
 // the freshly published Q8 banks once into thread-private stack storage in
 // staggered order so all workers read from private L1d instead of contending
 // on writer-private Modified cache lines across the non-inclusive L3 mesh.
 constexpr int kMaxLocalBlocks = 64;
 alignas(64) Q8_1Block local_primary[kMaxLocalBlocks];
 alignas(64) Q8_1Block local_residual[kMaxLocalBlocks];
 const Q8_1Block *primary_base = primary.data();
 const Q8_1Block *residual_base = residual.data();
 if (live == 1 && bpr > 0 && bpr <= kMaxLocalBlocks && omp_get_num_threads() > 1) {
  const int tid = omp_get_thread_num();
  const int nt = omp_get_num_threads();
  const int start = (tid * bpr) / nt;
  for (int step = 0; step < bpr; ++step) {
   int idx = start + step;
   if (idx >= bpr) idx -= bpr;
   local_primary[idx] = primary_base[idx];
   local_residual[idx] = residual_base[idx];
  }
  primary_base = local_primary;
  residual_base = local_residual;
 }
#if !defined(__aarch64__)
 // For narrow single-row Q8_0 projections (e.g. QFN hc_down N=320 -> 5 chunks on 28 cores),
 // split each 64-column chunk into its four independent 16-column ZMM sub-chunks so 20+
 // cores saturate all DDR4 memory channels with bitwise-identical K-reduction order.
 const bool is_q8_0_direct =
     packed.codebook_id == native_quantised_formats::Q8_0.codebook_id &&
     !packed.is_superblock && !packed.is_asymmetric &&
     (packed.usesExpandedInt8() || packed.usesUnsignedInt8());
 if (live == 1 && is_q8_0_direct && chunks * 2 <= omp_get_num_threads())
 {
  const std::int64_t subchunks = chunks * 4;
#pragma omp for schedule(static)
  for (std::int64_t task = 0; task < subchunks; ++task)
  {
   const int chunk = static_cast<int>(task / 4);
   const int z = static_cast<int>(task % 4);
   if (chunk * 64 + z * 16 >= n)
    continue;
   float *destination = output + chunk * 64;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
   if (avx512)
    gemm_2row_q8_subchunk_avx512<true>(packed, primary_base, residual_base,
     destination, nullptr, chunk, z, 0, bpr, false);
   else
#endif
    gemm_2row_q8_subchunk_avx2<true>(packed, primary_base, residual_base,
     destination, nullptr, chunk, z, 0, bpr, false);
  }
  return;
 }
#endif
#if !defined(__aarch64__) && defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
 if (avx512 && live >= 6 && is_q8_0_direct && packed.usesUnsignedInt8())
 {
  const std::int64_t tiles3 = (static_cast<std::int64_t>(live) + 2) / 3;
#pragma omp for schedule(static)
  for (std::int64_t task = 0; task < tiles3 * chunks; ++task)
  {
   const auto row = (task % tiles3) * 3;
   const auto chunk = static_cast<int>(task / tiles3);
   float *destination0 = output + row * n + chunk * 64;
   if (row + 2 < live)
   {
    gemm_6row_unsigned_int8_2z_sum_pairs(
        packed,
        primary_base + row * bpr, residual_base + row * bpr,
        primary_base + (row + 1) * bpr, residual_base + (row + 1) * bpr,
        primary_base + (row + 2) * bpr, residual_base + (row + 2) * bpr,
        destination0, destination0 + n, destination0 + 2 * n,
        chunk, 0, bpr);
   }
   else if (row + 1 < live)
   {
    gemm_4row_non_nibble_2z_chunk<true>(
        packed,
        primary_base + row * bpr, residual_base + row * bpr,
        primary_base + (row + 1) * bpr, residual_base + (row + 1) * bpr,
        destination0, nullptr, destination0 + n, nullptr,
        chunk, 0, bpr, false);
   }
   else
   {
    gemm_2row_non_nibble_chunk<true>(
        packed, primary_base + row * bpr, residual_base + row * bpr,
        destination0, nullptr, chunk, 0, bpr, false, 64, -1);
   }
  }
  return;
 }
#endif
#pragma omp for schedule(static)
  for(std::int64_t task=0;task<tiles*chunks;++task) {
   const auto row=task%tiles*2,chunk=task/tiles;
   const int valid=std::min<std::int64_t>(64,n-chunk*64);
#if defined(__aarch64__) && defined(__ARM_NEON)
   const bool second=row+1<live;
   cpuNativeResidualChunk(packed,primary_base+row*bpr,residual_base+row*bpr,
     second?primary_base+(row+1)*bpr:nullptr,second?residual_base+(row+1)*bpr:nullptr,
     output+row*n+chunk*64,second?output+(row+1)*n+chunk*64:nullptr,chunk,valid,avx512);
#else
   // Preserve the original Q8_0 kernel and its register-store fold exactly.
   // All other canonical encodings retain their own source arithmetic above.
   if(packed.codebook_id!=native_quantised_formats::Q8_0.codebook_id||packed.is_superblock||
      packed.is_asymmetric||!(packed.usesExpandedInt8()||packed.usesUnsignedInt8()))
   {
    const bool second=row+1<live;
    cpuNativeResidualChunk(packed,primary_base+row*bpr,residual_base+row*bpr,
      second?primary_base+(row+1)*bpr:nullptr,second?residual_base+(row+1)*bpr:nullptr,
      output+row*n+chunk*64,second?output+(row+1)*n+chunk*64:nullptr,chunk,valid,avx512);
    continue;
   }
   float *destination0=output+row*n+chunk*64;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
   if(avx512&&row+1<live) {
    gemm_4row_non_nibble_2z_chunk<true>(packed,primary_base+row*bpr,residual_base+row*bpr,
      primary_base+(row+1)*bpr,residual_base+(row+1)*bpr,
      destination0,nullptr,destination0+n,nullptr,chunk,0,bpr,false);
    continue;
   }
#endif
   for(auto current=row;current<std::min<std::int64_t>(row+2,live);++current) {
    float*destination=destination0+(current-row)*n;
    const int next_c = (live == 1 && chunk + 1 < chunks) ? static_cast<int>(chunk + 1) : -1;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    if(avx512)gemm_2row_non_nibble_chunk<true>(packed,primary_base+current*bpr,residual_base+current*bpr,
      destination,nullptr,chunk,0,bpr,false,64,next_c);
    else
#endif
      gemm_2row_non_nibble_chunk_avx2<true>(packed,primary_base+current*bpr,residual_base+current*bpr,
        destination,nullptr,chunk,0,bpr,false);
   }
#endif
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
inline void cpuQ8ResidualCorrectedProjectionImpl(const CPUNativeQuantisedPackedWeights&packed,const float*input,
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
  Q8ResidualStorageSpan{rows.countOwner(),rows.countOwner()?sizeof(std::int32_t):0,alignof(std::int32_t)},
  Q8ResidualStorageSpan{&packed,sizeof(packed),alignof(CPUNativeQuantisedPackedWeights)}};
 validateQ8ResidualStorage(writes,reads);
 if constexpr(std::is_same_v<Publication,HCDownQ8Publication>) {
  const auto bytes=static_cast<std::size_t>(physical)*k*sizeof(float);
  const std::array activationWrites{Q8ResidualStorageSpan{publication.activated,bytes,alignof(float)}};
  const std::array activationReads{
   Q8ResidualStorageSpan{input,publication.activated==input?0:bytes,alignof(float)},
   reads[1],reads[2],reads[3],writes[0],writes[1],writes[2]};
  if(!publication.activated)throw std::invalid_argument("HC CPU requires its retained activation output");
  validateQ8ResidualStorage(activationWrites,activationReads);
 }
 const int live=rows.activeRowsFor(rows.countOwner()?*rows.countOwner():rows.countCapacity());
 if(live<0)throw std::invalid_argument("Two-term CPU Q8 live count exceeds retained capacity");
 if(!llaminar::v2::beginWorkshare(live!=0))return;
#if defined(__aarch64__) && defined(__ARM_NEON)
 const auto avx512=neon_execution::instruction(ISAPath::AUTO);
#else
 const bool avx512=activeISALevel()>=ISALevel::AVX512;
#endif
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
 // For single-row M=1 HC decode (e.g. bpr=10, K=320), split across 2-block
 // (72-byte) pairs so 5 workers quantize in parallel with minimal cache-line
 // sharing; for single-row wide-K ActivatedQ8Publication (e.g. hc_down bpr=320,
 // K=10240), split into 16-block chunks so 20 workers quantize in parallel
 // instead of serializing all 10,240 floats on one core.
 const int pubBlocksPerTask = hcPublication
     ? ((live == 1 && bpr > 2) ? 2 : 8)
     : ((live == 1 && bpr >= 16) ? 16 : bpr);
 const std::int64_t publicationTilesPerRow =
     (pubBlocksPerTask > 0 && pubBlocksPerTask < bpr)
         ? (static_cast<std::int64_t>(bpr) + pubBlocksPerTask - 1) / pubBlocksPerTask
         : 1;
 // Shared nonfinite flag published inside the quantization workshare; the
 // workshare's implicit barrier guarantees every thread observes the complete
 // validity decision in O(1) without rescanning all live*bpr blocks.
 std::atomic<bool> failed{false};
 auto work=[&]() {
#pragma omp for schedule(static)
  for(std::int64_t task=0;task<static_cast<std::int64_t>(live)*publicationTilesPerRow;++task) {
   const auto row=task/publicationTilesPerRow;
   const int firstBlock=publicationTilesPerRow>1?static_cast<int>(task%publicationTilesPerRow)*pubBlocksPerTask:0;
   const int lastBlock=publicationTilesPerRow>1?std::min(firstBlock+pubBlocksPerTask,bpr):bpr;
   if constexpr(std::is_same_v<Publication,HCDownQ8Publication>) {
    //256 and every admitted K are multiples of32, hence each SIMD vector has
    // the same original row/channel boundary. Exact input/activated alias is
    // safe: another tile never reads or writes any channel owned by this tile.
    const auto base=static_cast<std::size_t>(row)*k;
    qwen4exp::cpu_detail::residualDownChannels(input+base,publication.activated+base,
      firstBlock*32,lastBlock*32,publicationWidth);
   }
   bool tile_failed = false;
   for(int block=firstBlock;block<lastBlock;++block) {
   const auto offset=static_cast<std::size_t>(row)*bpr+block;
   const auto*source=input+static_cast<std::size_t>(row)*k+block*32;
   if constexpr(std::is_same_v<Publication,HCDownQ8Publication>) {
    source=publication.activated+static_cast<std::size_t>(row)*k+block*32;
   }
   bool finite=true;
#if defined(__AVX512F__)
   if(avx512) {
    const __m512 abs_mask=_mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    const __m512 inf_v=_mm512_set1_ps(std::numeric_limits<float>::infinity());
    const __mmask16 m0=_mm512_cmp_ps_mask(_mm512_and_ps(_mm512_loadu_ps(source),abs_mask),inf_v,_CMP_LT_OQ);
    const __mmask16 m1=_mm512_cmp_ps_mask(_mm512_and_ps(_mm512_loadu_ps(source+16),abs_mask),inf_v,_CMP_LT_OQ);
    finite=((m0&m1)==0xffffu);
   } else
#endif
   {
    for(int element=0;element<32;++element)finite=finite&&std::isfinite(source[element]);
   }
   if(!finite){primary[offset].d=0x7c00u;tile_failed=true;continue;}
   quantizeActivationBlock(source,primary[offset],32,packed.numerical_policy);
   if((primary[offset].d&0x7c00u)!=0x7c00u)
    quantizeResidualBlock(source,primary[offset],residual[offset],packed.numerical_policy);
   else
    tile_failed=true;
   }
   if(tile_failed)failed.store(true,std::memory_order_relaxed);
  }
  // The first workshare's implicit barrier authenticates every active primary
  // scale. All workers observe the same status without a throw inside this
  // workshare. When called inside an enclosing lexical workshare, `failed` is
  // stack-local per thread, so scan the shared primary half-scales instead.
  bool any_failed = false;
  if (llaminar::v2::inLexicalWorkshare()) {
   const int total_blocks = live * bpr;
   for (int i = 0; i < total_blocks; ++i) {
    if ((primary[i].d & 0x7c00u) == 0x7c00u) {
     any_failed = true;
     break;
    }
   }
   if (any_failed) failed.store(true, std::memory_order_relaxed);
  } else {
   any_failed = failed.load(std::memory_order_relaxed);
  }
  if(!any_failed) {
   cpuQ8ResidualProjectionWorkshare(packed,output,live,primary,residual,avx512);
  } else {
   (void)llaminar::v2::beginWorkshare(false);
  }
 };
 OMP_WORKSHARE_REGION(work);
 if(failed.load(std::memory_order_relaxed))throw std::invalid_argument("Two-term CPU Q8 activation is nonfinite or its native primary half scale overflows");
}
/** @brief Preserve the original already-activated CPU entrypoint and source program.
 * @param packed Exact retained prepared source.
 * @param input Original activated FP32 rows.
 * @param output Distinct retained FP32 projection output.
 * @param rows Original physical/live prefix authority.
 * @param primary Admitted primary Q8 block bank.
 * @param residual Admitted residual Q8 block bank.
 */
inline void cpuQ8ResidualCorrectedProjection(const CPUNativeQuantisedPackedWeights& packed,const float* input,
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
inline void cpuHCResidualDownQ8Projection(const CPUNativeQuantisedPackedWeights& packed,const float* input,float* activated,
 float* output,DeviceRowRange rows,std::span<Q8_1Block> primary,std::span<Q8_1Block> residual) {
 cpuQ8ResidualCorrectedProjectionImpl(packed,input,output,rows,primary,residual,HCDownQ8Publication{activated});
}

}
