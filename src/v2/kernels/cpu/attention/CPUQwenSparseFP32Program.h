/**
 * @file CPUQwenSparseFP32Program.h
 * @brief Native QSA FP32 dot programs with their original fixed arithmetic trees.
 *
 * The QSA eight-lane fold is not the generic attention helper's pairwise fold.
 * NEON keeps two four-lane chains and explicitly spells QSA's eight-lane tree;
 * AVX2 retains eight chains and AVX512 sixteen. These synchronous arithmetic
 * leaves own no operands, allocation, decoded head or persistent coefficient
 * state. The enclosing invocation authenticates geometry, original storage and
 * the selected native ISA before any worksharing/publication. No Scalar path is
 * advertised or selected to disguise an absent native implementation.
 */
#pragma once
#include "utils/CPUFeatures.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#if LLAMINAR_COMPILED_WITH_NEON
#include "tensors/NEONFP32.h"
#elif defined(__AVX2__) || defined(__AVX512F__)
#include "tensors/X86FP32.h"
#endif

namespace llaminar2::cpu::qsa_native
{
/** @brief Admit one real native program before collective execution.
 * @param isa Original execution-selected ISA, not an inferred compiler default.
 * @throws std::invalid_argument For Scalar, foreign or uncompiled execution.
 */
inline void requireNativeProgram(ISALevel isa)
{
    if(isa==ISALevel::Scalar||!isNativeCPUExecutionISAPair(compiledCPUISALevel(),isa))
        throw std::invalid_argument("CPU Qwen sparse arithmetic requires an admitted native ISA");
}
/** @brief Reject an impossible post-admission program without crossing an OpenMP exception edge. */
[[noreturn]] inline void rejectProgram()
{
    std::fputs("CPU Qwen sparse arithmetic lost its admitted native program\n",stderr);
    std::abort();
}
#if LLAMINAR_COMPILED_WITH_NEON
/** @brief Original adjacent eight-coordinate registers; never decoded cache storage. */
struct Fragment8{float32x4_t low,high;};
/** @brief Load eight original FP32 coordinates under the enclosing borrow.
 * @param source Complete original eight-coordinate fragment.
 * @return The original words in native registers, without conversion or rebinding.
 */
[[gnu::always_inline]] inline Fragment8 load8(const float*source)noexcept
{return {vld1q_f32(source),vld1q_f32(source+4)};}
/** @brief Retain the QSA eight-chain final fold, not a generic dot reduction.
 * @param sum Eight completed original-coordinate FMA chains.
 * @return ((0+4)+(2+6))+((1+5)+(3+7)), with separate binary32 additions.
 */
[[gnu::always_inline]] inline float fold8(Fragment8 sum)noexcept
{
    const auto combined=neon_fp32::add(sum.low,sum.high);
    auto pair=vadd_f32(vget_low_f32(combined),vget_high_f32(combined));
    asm volatile("" : "+w"(pair));
    return vget_lane_f32(vpadd_f32(pair,pair),0);
}
/** @brief Apply intentional coordinatewise FMA with no hidden arithmetic tree change.
 * @param accumulator Original eight chains; @param a Original left fragment.
 * @param b Original right fragment; @return Updated native FMA chains.
 */
[[gnu::always_inline]] inline Fragment8 fma8(Fragment8 accumulator,Fragment8 a,Fragment8 b)noexcept
{return {vfmaq_f32(accumulator.low,a.low,b.low),vfmaq_f32(accumulator.high,a.high,b.high)};}
/** @return Eight original positive-zero FMA chains. */
[[gnu::always_inline]] inline Fragment8 zero8()noexcept
{return {vdupq_n_f32(0),vdupq_n_f32(0)};}
#endif
#if defined(__AVX2__)
/** @brief Original eight-coordinate x86 register fragment; never decoded head storage. */
using Fragment8=__m256;
/** @brief Load eight original FP32 coordinates under the enclosing borrow.
 * @param source Complete original eight-coordinate fragment; @return Original native words.
 */
[[gnu::always_inline]] inline Fragment8 load8(const float*source)noexcept{return _mm256_loadu_ps(source);}
/** @brief Update original eight native FMA chains without changing their reduction.
 * @param accumulator Original chains; @param a Left fragment; @param b Right fragment.
 * @return Coordinatewise intentional native FMAs.
 */
[[gnu::always_inline]] inline Fragment8 fma8(Fragment8 accumulator,Fragment8 a,Fragment8 b)noexcept{return _mm256_fmadd_ps(a,b,accumulator);}
/** @return Eight original positive-zero FMA chains. */
[[gnu::always_inline]] inline Fragment8 zero8()noexcept{return _mm256_setzero_ps();}
/** @brief Retain the existing QSA eight-lane x86 reduction without a lane scratch array.
 * @param sum Eight completed original-coordinate chains.
 * @return Existing fixed QSA butterfly, not a partition-dependent reduction.
 */
[[gnu::always_inline]] inline float fold8(__m256 sum)noexcept
{
    const auto combined=_mm_add_ps(_mm256_castps256_ps128(sum),_mm256_extractf128_ps(sum,1));
    const auto pairs=_mm_add_ps(combined,_mm_movehl_ps(combined,combined));
    return _mm_cvtss_f32(_mm_add_ss(pairs,_mm_shuffle_ps(pairs,pairs,_MM_SHUFFLE(1,1,1,1))));
}
#endif
#if defined(__AVX512F__)
/** @brief Retain the original sixteen-chain QSA butterfly, without FP32 lane storage.
 * @param sum Sixteen completed original FMA chains; @return Existing QSA scalar dot.
 */
[[gnu::always_inline]] inline float fold16(__m512 sum)noexcept
{
    return fold8(x86_fp32::persistRounded(_mm256_add_ps(_mm512_castps512_ps256(sum),_mm512_extractf32x8_ps(sum,1))));
}
#endif

/** @brief Dot original FP32 indexer/rotation vectors under the exact admitted native program.
 * @tparam Dimensions Original QSA width128 or256, never a padded/cache allocation size.
 * @param a Complete original left row; @param b Complete original right row.
 * @param isa Previously admitted native ISA; @return Fixed per-program dot result.
 * @details The caller validates complete synchronous input extents before the
 * collective; no exception, decoded bank or whole-operation scalar replay runs
 * inside this leaf. Grouping and worker ownership never enter its reduction.
 */
template<std::size_t Dimensions>
[[gnu::always_inline]] inline float dot(const float*a,const float*b,ISALevel isa)noexcept
{
    static_assert(Dimensions==128||Dimensions==256);
    switch(isa)
    {
#if LLAMINAR_COMPILED_WITH_NEON
    case ISALevel::NEON:case ISALevel::NEONDotProd:
    {
        auto sum=zero8();
        for(std::size_t coordinate=0;coordinate<Dimensions;coordinate+=8)
            sum=fma8(sum,load8(a+coordinate),load8(b+coordinate));
        return fold8(sum);
    }
#endif
#if defined(__AVX2__)
    case ISALevel::AVX2:
    {
        auto sum=_mm256_setzero_ps();
        for(std::size_t coordinate=0;coordinate<Dimensions;coordinate+=8)
            sum=_mm256_fmadd_ps(_mm256_loadu_ps(a+coordinate),_mm256_loadu_ps(b+coordinate),sum);
        return fold8(sum);
    }
#endif
#if defined(__AVX512F__)
    case ISALevel::AVX512:
    {
        auto sum=_mm512_setzero_ps();
        for(std::size_t coordinate=0;coordinate<Dimensions;coordinate+=16)
            sum=_mm512_fmadd_ps(_mm512_loadu_ps(a+coordinate),_mm512_loadu_ps(b+coordinate),sum);
        return fold16(sum);
    }
#endif
    default:rejectProgram();
    }
}
} // namespace llaminar2::cpu::qsa_native
