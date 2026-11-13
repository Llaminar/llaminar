/**
 * @file CPUQwenSparseIndexerProgram.h
 * @brief Native fixed-tree indexer RMS/cast/partial-RoPE over original FP32 projections.
 *
 * AArch64 prepares four independent coordinates per ASIMD register; x86 uses
 * its eight-coordinate AVX register. This changes only physical register tiles,
 * never the original128-square butterfly or per-coordinate arithmetic. Keeping
 * one native register live per operand limits half/BF16 cast pressure; compiler
 * resource gates still have to prove the actual complete specialization.
 * Rounded input/square rows are not materialized. Normalized128 and pooled128
 * are produced reference cast-boundary operands, completed before output stores
 * for original alias semantics, not decoded KV heads or weight mirrors. Native
 * RoPE pairs retain at most Width-1 scalar fringe pairs and no bulk-copy callback.
 * Each head binds one typed original table/frequency borrow before publication;
 * rotary loops never reload the parameter object's source/geometry through an
 * output-aliasable reference or construct a coefficient-plane shadow.
 */
#pragma once
#include "CPUQwenSparseRepresentation.h"
#include "CPUQwenSparseNativeCodec.h"
#include "kernels/qwen4exp/QwenSparseMath.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <utility>
namespace llaminar2::cpu::qsa_native::indexer
{
/** @brief Reject corruption inside its original owner without collective unwinding.
 * @param stage Exact original arithmetic stage, preserving its fatal diagnostic.
 */
[[noreturn]] inline void rejectNumeric(const char*stage)noexcept
{std::fputs("ERROR: CPU QSA invalid consumed numeric data in ",stderr);std::fputs(stage,stderr);std::fputc('\n',stderr);std::abort();}
/** @brief Retain a scalar binary32 edge without memory or a changed FP mode.
 * @param value Completed scalar result; @return Exact original rounded word.
 */
[[gnu::always_inline]] inline float roundedScalar(float value)noexcept
{
#if LLAMINAR_COMPILED_WITH_NEON
    asm volatile("" : "+w"(value));
#else
    asm volatile("" : "+x"(value));
#endif
    return value;
}
/** @brief Native register geometry for preparation, independent of score-chain policy. */
struct Program final
{
#if LLAMINAR_COMPILED_WITH_NEON
    using Value=float32x4_t;static constexpr unsigned Width=4;
#else
    using Value=Fragment8;static constexpr unsigned Width=8;
#endif
    /** @brief Read exactly one original native register.
     * @param input Original Width-coordinate extent; @return Original words.
     */
    [[gnu::always_inline]] static Value load(const float*input)noexcept
    {
#if LLAMINAR_COMPILED_WITH_NEON
        return vld1q_f32(input);
#else
        return load8(input);
#endif
    }
    /** @brief Schedule a right subtree only after its original left summary completes.
     * @param original Unchanged original root, not a copied operand.
     * @param completed Original already-rounded left subtree.
     * @return Identical root with an opaque compiler register dependency.
     * @details Without this edge independent raw loads can move above their
     * half casts and inflate the live set. This emits no memory fence or instruction,
     * changes no arithmetic/tree edge, and leaves hardware scheduling intact.
     */
    [[gnu::always_inline]] static const float* follow(const float*original,Value completed)noexcept
    {
#if LLAMINAR_COMPILED_WITH_NEON
        asm volatile("" : "+r"(original) : "w"(completed));
#else
        asm volatile("" : "+r"(original) : "x"(completed));
#endif
        return original;
    }
    /** @brief Publish one exclusive native register.
     * @param output Original exclusive extent; @param value Original register.
     */
    [[gnu::always_inline]] static void store(float*output,Value value)noexcept
    {
#if LLAMINAR_COMPILED_WITH_NEON
        vst1q_f32(output,value);
#else
        store8(output,value);
#endif
    }
    /** @brief Retain an identity edge so a bounded native copy is not replaced by a memcpy call.
     * @param value Original words; @return Identical opaque register words.
     */
    [[gnu::always_inline]] static Value retain(Value value)noexcept
    {
#if LLAMINAR_COMPILED_WITH_NEON
        return neon_fp32::persistRounded(value);
#else
        return x86_fp32::persistRounded(value);
#endif
    }
    /** @brief Broadcast one original scalar.
     * @param value Original word; @return Width identical words.
     */
    [[gnu::always_inline]] static Value broadcast(float value)noexcept
    {
#if LLAMINAR_COMPILED_WITH_NEON
        return vdupq_n_f32(value);
#else
        return broadcast8(value);
#endif
    }
    /** @brief Retain a separately rounded native product.
     * @param a Original multiplicands; @param b Original multipliers; @return Rounded products.
     */
    [[gnu::always_inline]] static Value multiply(Value a,Value b)noexcept
    {
#if LLAMINAR_COMPILED_WITH_NEON
        return neon_fp32::multiply(a,b);
#else
        return multiply8(a,b);
#endif
    }
    /** @brief Retain a separately rounded native sum.
     * @param a Original addends; @param b Original addends; @return Rounded sums.
     */
    [[gnu::always_inline]] static Value add(Value a,Value b)noexcept
    {
#if LLAMINAR_COMPILED_WITH_NEON
        return neon_fp32::add(a,b);
#else
        return add8(a,b);
#endif
    }
    /** @brief Execute the full original representation program in one register.
     * @tparam Rep Original representation; @param value Original words.
     * @return Exact full-IEEE cast-boundary containers.
     */
    template<qwen4exp::QwenSparseScalar Rep>[[gnu::always_inline]] static Value round(Value value)noexcept
    {
#if LLAMINAR_COMPILED_WITH_NEON
        return representation4<Rep>(value);
#else
        return roundRepresentation8<Rep>(value);
#endif
    }
    /** @brief Check every consumed register lane.
     * @param value Original lanes; @return True only for finite words.
     */
    [[gnu::always_inline]] static bool finite(Value value)noexcept
    {
#if LLAMINAR_COMPILED_WITH_NEON
        return vminvq_u32(vcleq_f32(vabsq_f32(value),vdupq_n_f32(std::numeric_limits<float>::max())))==~std::uint32_t(0);
#else
        return finite8(value);
#endif
    }
    /** @brief Negate each original normalized word without multiplying/reassociating it.
     * @param value Original finite lanes; @return Exact sign-flipped words.
     */
    [[gnu::always_inline]] static Value negate(Value value)noexcept
    {
#if LLAMINAR_COMPILED_WITH_NEON
        return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(value),vdupq_n_u32(0x80000000u)));
#else
        return _mm256_xor_ps(value,_mm256_set1_ps(-0.0f));
#endif
    }
    /** @brief Complete the last original butterfly stages for this physical register.
     * @param value Original last-stage partials; @return Original128-square scalar summary.
     */
    [[gnu::always_inline]] static float fold(Value value)noexcept
    {
#if LLAMINAR_COMPILED_WITH_NEON
        auto pairs=vadd_f32(vget_low_f32(value),vget_high_f32(value));asm volatile("" : "+w"(pairs));
        return vget_lane_f32(vpadd_f32(pairs,pairs),0);
#else
        return fold8(value);
#endif
    }
    /** @brief Insert one produced analytic factor without constructing a coefficient row.
     * @tparam Lane Original lane; @param target Produced register; @param value Produced factor.
     */
    template<unsigned Lane>[[gnu::always_inline]] static void insert(Value&target,float value)noexcept
    {
        static_assert(Lane<Width);
#if LLAMINAR_COMPILED_WITH_NEON
        target=vsetq_lane_f32(value,target,Lane);
#else
        target=_mm256_blend_ps(target,_mm256_set1_ps(value),1u<<Lane);
#endif
    }
};
/** @brief Retain the exact128-square tree with original-coordinate register leaves.
 * @tparam Rep Original representation; @tparam Count Remaining native groups.
 * @tparam Stride Original group-index step; @tparam First Original group index.
 * @param input Original complete128-coordinate head; @return Original last-stage partials.
 */
template<qwen4exp::QwenSparseScalar Rep,unsigned Count=128/Program::Width,unsigned Stride=1,unsigned First=0>
[[gnu::always_inline]] inline Program::Value squares(const float*input)noexcept
{
    if constexpr(Count==1){const auto value=Program::round<Rep>(Program::load(input+First*Program::Width));return Program::multiply(value,value);}
    else
    {
        const auto left=squares<Rep,Count/2,Stride*2,First>(input);
        const auto right=squares<Rep,Count/2,Stride*2,First+Stride>(Program::follow(input,left));return Program::add(left,right);
    }
}
/** @brief Produced native factor registers, never persistent coefficient storage. */
struct Factors{Program::Value cosine,sine;};
/** @brief Two position-bound roots in the original admitted tables, not coefficient copies.
 * @details The enclosing synchronous CPU entry owns range/lifetime admission.
 * This view copies roots only; it does not establish capacity or a storage lease.
 */
struct TableFactors final
{
    const float*const cosine; ///< Original cosine plane at this admitted position.
    const float*const sine; ///< Original sine plane at this admitted position.
    /** @brief Read one original fringe pair without re-reading mutable policy fields.
     * @param pair Original pair index; @param co Original cosine; @param si Original sine.
     */
    [[gnu::always_inline]] void scalar(int pair,float&co,float&si)const noexcept
    {co=cosine[pair];si=sine[pair];}
};
/** @brief Borrow the setup-owned frequencies of the admitted64-wide analytic-theta program.
 * @details The original CPU program always receives these invariants. Its
 * consumed factor is cosf/sinf(float(position)*original_frequency), never a
 * repeated power evaluation or a cached/generated coefficient plane.
 */
struct AnalyticFactors final
{
    const int position; ///< Absolute original position.
    const float*const frequencies; ///< Original inline policy array; never copied.
    /** @brief Derive one pair using the original FP32 angle and library argument reduction.
     * @param pair Original pair index; @param co Produced cosine; @param si Produced sine.
     */
    [[gnu::always_inline]] void scalar(int pair,float&co,float&si)const noexcept
    {const float angle=roundedScalar(float(position)*frequencies[pair]);qwen4exp::detail::hostSeparateSinCos(angle,co,si);}
};
/** @brief Derive one analytic pair directly into its native register lane.
 * @tparam Lane Original lane; @tparam Source Original bound source type.
 * @param factors Produced registers; @param source Immutable scalar policy/original roots.
 * @param pair First original pair in the fragment.
 */
template<unsigned Lane,class Source>[[gnu::always_inline]] inline void analyticFactor(Factors&factors,const Source&source,int pair)noexcept
{
    float co,si;source.scalar(pair+int(Lane),co,si);
    Program::insert<Lane>(factors.cosine,co);Program::insert<Lane>(factors.sine,si);
}
/** @brief Read one native fragment from exactly one already-bound source policy.
 * @tparam Rep Original representation; @tparam Source Original factor source.
 * @tparam Lanes Consecutive original lanes; @param source Original bound roots/policy.
 * @param pair First original pair; @param indices Compile-time lane token.
 * @return Exact rounded original factors, without per-fragment source dispatch.
 */
template<qwen4exp::QwenSparseScalar Rep,class Source,unsigned...Lanes>
[[gnu::always_inline]] inline Factors factors(const Source&source,int pair,std::integer_sequence<unsigned,Lanes...>indices)noexcept
{
    static_assert(std::is_same_v<Source,TableFactors>||std::is_same_v<Source,AnalyticFactors>);
    (void)indices;Factors result;
    if constexpr(std::is_same_v<Source,TableFactors>)result={Program::load(source.cosine+pair),Program::load(source.sine+pair)};
    else{result={Program::broadcast(0),Program::broadcast(0)};(analyticFactor<Lanes>(result,source,pair),...);}
    result.cosine=Program::round<Rep>(result.cosine);result.sine=Program::round<Rep>(result.sine);
    if(!Program::finite(result.cosine)||!Program::finite(result.sine))rejectNumeric("indexer rotary representation");return result;
}
/** @brief Consume the produced normalization stage under one closed rotary source.
 * @tparam Rep Original representation; @tparam Source Original factor-source type.
 * @param normalized Complete produced128-coordinate stage.
 * @param admittedHalf Original table pair count; analytic admission requires32.
 * @param source Bound scalar policy/original roots, copied by value rather than aliasing outputs.
 * @param output Exclusive original128-coordinate destination.
 * @details The policy dispatch is once per head, not once per fragment. Keeping
 * source/geometry in local values removes p-field alias reloads and permits
 * position-bound table pointers to stay live without hot scalar stack spills.
 */
template<qwen4exp::QwenSparseScalar Rep,class Source>
inline void rotate(const std::array<float,128>&normalized,int admittedHalf,Source source,float*output)noexcept
{
    // Analytic64 is a real admission invariant, not a runtime-general shape.
    // Sealing it here removes unreachable analytic fringes and lets its fixed
    // nonrotary suffix use frame offsets instead of spilled pointer induction.
    const int half=[&]
    {
        if constexpr(std::is_same_v<Source,AnalyticFactors>)
        {
            if(admittedHalf!=32){std::fputs("ERROR: CPU QSA indexer lost its admitted64-wide analytic geometry\n",stderr);std::abort();}
            return 32;
        }
        else return admittedHalf;
    }();
    int pair=0;
    for(;pair+int(Program::Width)<=half;pair+=int(Program::Width))
    {
        const auto coefficient=factors<Rep>(source,pair,std::make_integer_sequence<unsigned,Program::Width>{});
        const auto left=Program::load(normalized.data()+pair),right=Program::load(normalized.data()+pair+half);
        const auto first=Program::round<Rep>(Program::add(Program::round<Rep>(Program::multiply(left,coefficient.cosine)),Program::round<Rep>(Program::multiply(Program::negate(right),coefficient.sine))));
        const auto second=Program::round<Rep>(Program::add(Program::round<Rep>(Program::multiply(right,coefficient.cosine)),Program::round<Rep>(Program::multiply(left,coefficient.sine))));
        if(!Program::finite(first)||!Program::finite(second))rejectNumeric("indexer rotary representation");Program::store(output+pair,first);Program::store(output+pair+half,second);
    }
    for(;pair<half;++pair)
    {
        float co,si;source.scalar(pair,co,si);
        co=qwen4exp::detail::roundRepresentation(co,Rep);si=qwen4exp::detail::roundRepresentation(si,Rep);
        const auto cast=[](float value){return qwen4exp::detail::roundRepresentation(value,Rep);};
        const auto first=cast(roundedScalar(cast(roundedScalar(normalized[pair]*co))+cast(roundedScalar(-normalized[pair+half]*si))));
        const auto second=cast(roundedScalar(cast(roundedScalar(normalized[pair+half]*co))+cast(roundedScalar(normalized[pair]*si))));
        if(!std::isfinite(first)||!std::isfinite(second))rejectNumeric("indexer rotary representation");output[pair]=first;output[pair+half]=second;
    }
    unsigned c=2*half;for(;c+Program::Width<=128;c+=Program::Width)Program::store(output+c,Program::retain(Program::load(normalized.data()+c)));
    for(;c<128;++c)output[c]=roundedScalar(normalized[c]);
}
/** @brief Execute original128-coordinate normalization and cast-boundary split-half RoPE.
 * @tparam Rep Admitted representation, never a replacement execution mode.
 * @param input Original projection or produced pool; @param weight Original imported weights.
 * @param position Absolute original position; @param p Original admitted policy.
 * @param output Exclusive128-coordinate output, including complete same-head reuse.
 * @details The produced normalization stage completes before any output store;
 * all original FP32/cast/coordinate orders remain independent of physical Width.
 */
template<qwen4exp::QwenSparseScalar Rep>
inline void prepare(const float*input,const float*weight,int position,const qwen4exp::QwenSparseIndexer&p,float*output)noexcept
{
    const auto summary=Program::fold(squares<Rep>(input));
    const auto mean=roundedScalar(summary/128.0f),variance=roundedScalar(mean+p.norm_epsilon);
    const float inverse=1.0f/std::sqrt(variance);if(!std::isfinite(inverse)||inverse<=0)rejectNumeric("indexer RMS summary");
    const auto scale=Program::broadcast(inverse);std::array<float,128>normalized;
    for(unsigned c=0;c<128;c+=Program::Width)
    {
        const auto value=Program::round<Rep>(Program::multiply(Program::multiply(Program::round<Rep>(Program::load(input+c)),scale),Program::load(weight+c)));
        if(!Program::finite(value))rejectNumeric("indexer normalized representation");Program::store(normalized.data()+c,value);
    }
    const int dimension=p.rotary_dim,half=dimension/2;
    switch(p.rotary_source)
    {
    case qwen4exp::QwenSparseRotarySource::TablePlanes:
    {
        const auto offset=std::size_t(position)*std::size_t(half);
        rotate<Rep>(normalized,half,TableFactors{.cosine=p.cosine+offset,.sine=p.sine+offset},output);return;
    }
    case qwen4exp::QwenSparseRotarySource::AnalyticTheta:
        rotate<Rep>(normalized,half,AnalyticFactors{.position=position,.frequencies=p.inverse_frequencies.data()},output);return;
    case qwen4exp::QwenSparseRotarySource::CanonicalEngineTheta:
        break; // This is a core-key source, rejected by indexer admission.
    }
    // Cold admission rejected unknown policies; never substitute another source.
    std::fputs("ERROR: CPU QSA indexer lost its admitted rotary source\n",stderr);std::abort();
}
/** @brief Produce one rounded four-member original pool under the same preparation owner.
 * @tparam Rep Original representation; @param raw Original four raw128 roots, including accepted tail.
 * @param weight Original key RMS weights; @param position Original block-start position.
 * @param p Original policy; @param output Exclusive prepared pool.
 * @details Each operand casts before increasing-order additions; quarter multiplication
 * and mean cast precede normalization. The produced pool completes before output stores.
 * One outlined call owns the whole128-coordinate pool, not one scalar/row replay.
 * This bounds its root-register live set independently of the enclosing OpenMP
 * request lambda; copying its four immutable roots does not copy any operand.
 */
template<qwen4exp::QwenSparseScalar Rep>
[[gnu::noinline]] void preparePool(const std::array<const float*,4>&raw,const float*weight,int position,const qwen4exp::QwenSparseIndexer&p,float*output)noexcept
{
    std::array<float,128>pooled;const auto quarter=Program::broadcast(0.25f);
    const auto*const first=raw[0];const auto*const second=raw[1];
    const auto*const third=raw[2];const auto*const fourth=raw[3];
    for(unsigned coordinate=0;coordinate<128;coordinate+=Program::Width)
    {
        auto sum=Program::round<Rep>(Program::load(first+coordinate));
        sum=Program::add(sum,Program::round<Rep>(Program::load(second+coordinate)));
        sum=Program::add(sum,Program::round<Rep>(Program::load(third+coordinate)));
        sum=Program::add(sum,Program::round<Rep>(Program::load(fourth+coordinate)));
        Program::store(pooled.data()+coordinate,Program::round<Rep>(Program::multiply(sum,quarter)));
    }
    prepare<Rep>(pooled.data(),weight,position,p,output);
}
} // namespace llaminar2::cpu::qsa_native::indexer
