/**
 * @file CPUQwenSparseNativeCodec.h
 * @brief Original-storage native QSA codec fragments, without decoded cache heads.
 *
 * A checked complete bank supplies original physical rows, codebooks and anchors.
 * Each native reader decodes eight coordinates into registers and immediately
 * consumes them in the caller's declared dot/value program. QSA deliberately
 * rounds decoding before probability FMA; weight-times-scale reassociation from
 * other attention programs is not this numerical contract. Original source and
 * fitted TQ radii retain different roles, with no canonical-global substitution.
 * Bindings copy pointer roots only and never allocate, repack, establish capacity
 * or extend owner leases. Row borrows require a stable enclosing decoder.
 */
#pragma once
#include "CPUQwenSparseFP32Program.h"
#include "kernels/qwen4exp/QwenSparseNativeBankContract.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#if LLAMINAR_COMPILED_WITH_NEON
#include "NEONTQAttentionPrimitives.h"
#include "tensors/NEONFP16.h"
#elif !defined(__AVX2__)
#error "Native QSA codecs require ARM NEON or the production AVX2/AVX512 ISA"
#endif
namespace llaminar2::cpu::qsa_native
{
using qwen4exp::native_bank::Binding;
/** @brief Empty compile-time metadata for codecs that do not use a TQ4 register table. */
struct NoTable{};
/** @brief Retain independent rounded binary32 products after inlining.
 * @param a Original first eight lanes; @param b Original second eight lanes.
 * @return Eight separately rounded products, never fused into a consumer FMA.
 */
[[gnu::always_inline]] inline Fragment8 multiply8(Fragment8 a,Fragment8 b)noexcept
{
#if LLAMINAR_COMPILED_WITH_NEON
    return {neon_fp32::multiply(a.low,b.low),neon_fp32::multiply(a.high,b.high)};
#else
    return x86_fp32::persistRounded(_mm256_mul_ps(a,b));
#endif
}
/** @brief Retain independent rounded binary32 sums after inlining.
 * @param a Original first eight lanes; @param b Original second eight lanes.
 * @return Eight separately rounded sums.
 */
[[gnu::always_inline]] inline Fragment8 add8(Fragment8 a,Fragment8 b)noexcept
{
#if LLAMINAR_COMPILED_WITH_NEON
    return {neon_fp32::add(a.low,b.low),neon_fp32::add(a.high,b.high)};
#else
    return x86_fp32::persistRounded(_mm256_add_ps(a,b));
#endif
}
/** @brief Broadcast one original scalar metadata value without changing its arithmetic.
 * @param value Original scalar; @return Eight identical original words.
 */
[[gnu::always_inline]] inline Fragment8 broadcast8(float value)noexcept
{
#if LLAMINAR_COMPILED_WITH_NEON
    return {vdupq_n_f32(value),vdupq_n_f32(value)};
#else
    return _mm256_set1_ps(value);
#endif
}
/** @brief Fail numeric corruption in every build without throwing across worksharing. */
[[noreturn]] inline void rejectCodec()noexcept
{std::fputs("ERROR: QSA original native codec has invalid coordinates or nonfinite numeric data\n",stderr);std::abort();}
/** @brief Check every original decoded lane before it enters max/FMA/publication.
 * @param value Original decoded eight-lane fragment; @return True only for finite lanes.
 */
[[gnu::always_inline]] inline bool finite8(Fragment8 value)noexcept
{
#if LLAMINAR_COMPILED_WITH_NEON
    const auto limit=vdupq_n_f32(std::numeric_limits<float>::max());
    return vminvq_u32(vandq_u32(vcleq_f32(vabsq_f32(value.low),limit),vcleq_f32(vabsq_f32(value.high),limit)))==~std::uint32_t(0);
#else
    const auto magnitude=_mm256_andnot_ps(_mm256_set1_ps(-0.0f),value);
    return _mm256_movemask_ps(_mm256_cmp_ps(magnitude,_mm256_set1_ps(std::numeric_limits<float>::max()),_CMP_LE_OQ))==255;
#endif
}
/** @brief Publish one already-admitted eight-coordinate FP32 output fragment.
 * @param output Original exclusive output; @param value Original computed native lanes.
 */
[[gnu::always_inline]] inline void store8(float*output,Fragment8 value)noexcept
{
#if LLAMINAR_COMPILED_WITH_NEON
    vst1q_f32(output,value.low);vst1q_f32(output+4,value.high);
#else
    _mm256_storeu_ps(output,value);
#endif
}

/** @brief Native codec specialization bound only to a checked original physical bank.
 * @tparam Encoding Existing admitted codec, selected once outside the coordinate loop.
 */
template<TensorType Encoding>class Decoder final
{
    static constexpr bool rotated=Encoding==TensorType::TQ4||Encoding==TensorType::TQ8;
#if LLAMINAR_COMPILED_WITH_NEON
    using Table=std::conditional_t<Encoding==TensorType::TQ4,neon_tq_attention::TQ4Lookup,NoTable>;
#else
    using Table=NoTable;
#endif
    /** @brief Original row-local radial metadata, never a decoded coefficient array. */
    struct Radius{Fragment8 scale;bool zero;};
public:
    /** @brief Whether this exact admitted codec publishes a rotated coordinate basis. */
    static constexpr bool rotatedBasis=rotated;
    static_assert(Encoding==TensorType::FP32||Encoding==TensorType::FP16||Encoding==TensorType::BF16||
        Encoding==TensorType::Q8_1||Encoding==TensorType::Q16_1||Encoding==TensorType::AQ8||rotated);
    /** @brief Bind specialization/head identity under already checked bank geometry.
     * @param binding Original complete-bank proof, copied as immutable pointer roots only.
     * @param head Original local KV head, not a global GQA head.
     * @return Stable native decoder; a wrong internal specialization is always fatal.
     */
    [[nodiscard]] static Decoder bind(const Binding&binding,int head)noexcept
    {
        if(binding.original().codec!=Encoding||head<0||head>=binding.heads())rejectCodec();
        return Decoder(binding,head);
    }
    Decoder(const Decoder&)=delete;Decoder&operator=(const Decoder&)=delete;
    Decoder(Decoder&&)=delete;Decoder&operator=(Decoder&&)=delete;
    /** @brief One exact original physical head, retaining only its root and radial metadata. */
    class Row final
    {
    public:
        /** @brief Decode one native eight-coordinate original fragment without staging a head.
         * @param coordinate Original aligned coordinate0..248, independent of stride/padding.
         * @return Finite decoded register fragment in the original declared basis.
         * @details Numeric corruption is fatal before this fragment can be stored;
         * no repaired-zero value or exception can escape through a collective.
         */
        [[gnu::always_inline]] Fragment8 read(int coordinate)const noexcept
        {
            if(coordinate<0||coordinate>248||coordinate%8)rejectCodec();
            Fragment8 result;
            if constexpr(Encoding==TensorType::FP32)result=load8(reinterpret_cast<const float*>(bytes_)+coordinate);
            else if constexpr(Encoding==TensorType::FP16)
            {
                const auto*input=reinterpret_cast<const std::uint16_t*>(bytes_)+coordinate;
#if LLAMINAR_COMPILED_WITH_NEON
                result={neon_fp16::widen(vld1_u16(input)),neon_fp16::widen(vld1_u16(input+4))};
#else
                result=_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input)));
#endif
            }
            else if constexpr(Encoding==TensorType::BF16)
            {
                const auto*input=reinterpret_cast<const std::uint16_t*>(bytes_)+coordinate;
#if LLAMINAR_COMPILED_WITH_NEON
                result={vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(input),16)),vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(input+4),16))};
#else
                result=_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input))),16));
#endif
            }
            else if constexpr(Encoding==TensorType::Q8_1)
            {
                const auto&block=reinterpret_cast<const Q8_1Block*>(bytes_)[coordinate/32];
                result=multiply8(broadcast8(qwen4exp::detail::decodeHalf(block.d)),signedBytes(block.qs+coordinate%32));
            }
            else if constexpr(Encoding==TensorType::Q16_1)
            {
                const auto size=owner_.binding_.original().q16_block_size;
                const auto*block=bytes_+std::size_t(coordinate/size)*(8+2*size);
                const auto*input=reinterpret_cast<const std::int16_t*>(block+8)+coordinate%size;
#if LLAMINAR_COMPILED_WITH_NEON
                const auto codes=vld1q_s16(input);const Fragment8 widened={vcvtq_f32_s32(vmovl_s16(vget_low_s16(codes))),vcvtq_f32_s32(vmovl_s16(vget_high_s16(codes)))};
#else
                const auto widened=_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input))));
#endif
                result=multiply8(broadcast8(word(block)),widened);
            }
            else if constexpr(Encoding==TensorType::AQ8)
            {
                const auto codes=signedBytes(reinterpret_cast<const std::int8_t*>(bytes_+4+coordinate));
#if LLAMINAR_COMPILED_WITH_NEON
                const Fragment8 magnitude={vabsq_f32(codes.low),vabsq_f32(codes.high)};
#else
                const auto magnitude=_mm256_andnot_ps(_mm256_set1_ps(-0.0f),codes);
#endif
                result=add8(multiply8(multiply8(broadcast8(word(bytes_)),codes),magnitude),
                    load8(owner_.binding_.original().anchor+owner_.head_*256+coordinate));
            }
            else
            {
                if(radius_.zero)return zero8();
                Fragment8 values;
#if LLAMINAR_COMPILED_WITH_NEON
                if constexpr(Encoding==TensorType::TQ4)
                {const auto decoded=owner_.table_.decode(bytes_+8+(coordinate/8)*3,bytes_[104+coordinate/8]);values={decoded.lo,decoded.hi};}
                else
                {const auto book=std::span<const float,256>(owner_.binding_.original().centroids,256);
                 values={neon_tq_attention::lookupTQ8(book,bytes_+8+coordinate),neon_tq_attention::lookupTQ8(book,bytes_+12+coordinate)};}
#else
                __m256i indices;
                if constexpr(Encoding==TensorType::TQ8)indices=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(bytes_+8+coordinate)));
                else
                {
                    const auto*packed=bytes_+8+(coordinate/8)*3;const unsigned low=unsigned(packed[0])|(unsigned(packed[1])<<8)|(unsigned(packed[2])<<16);
                    const unsigned high=bytes_[104+coordinate/8];
                    indices=_mm256_setr_epi32((low&7)|((high&1)<<3),((low>>3)&7)|(((high>>1)&1)<<3),
                        ((low>>6)&7)|(((high>>2)&1)<<3),((low>>9)&7)|(((high>>3)&1)<<3),
                        ((low>>12)&7)|(((high>>4)&1)<<3),((low>>15)&7)|(((high>>5)&1)<<3),
                        ((low>>18)&7)|(((high>>6)&1)<<3),((low>>21)&7)|(((high>>7)&1)<<3));
                }
                values=_mm256_i32gather_ps(owner_.binding_.original().centroids,indices,sizeof(float));
#endif
                result=multiply8(values,radius_.scale);
            }
            if(!finite8(result))rejectCodec();return result;
        }
    private:
        friend class Decoder;
        /** @brief Seal original per-row metadata, with no payload conversion or shadow.
         * @param owner Stable original bank/codec table authority; @param bytes Exact physical head root.
         */
        Row(const Decoder&owner,const unsigned char*bytes)noexcept:owner_(owner),bytes_(bytes),radius_(radius(owner,bytes)){}
        /** @brief Bind only the selected original radial role; zero-source rows do not read fitted metadata.
         * @param owner Stable key/value policy; @param bytes Original head header.
         * @return Row-local register scale, or empty metadata for a nonrotated codec.
         */
        static std::conditional_t<rotated,Radius,NoTable> radius(const Decoder&owner,const unsigned char*bytes)noexcept
        {
            if constexpr(rotated)
            {
                const float source=word(bytes);if(!std::isfinite(source)||source<0)rejectCodec();
                if(source<1e-30f)return Radius{zero8(),true};
                const auto selected=owner.binding_.role()==qwen4exp::detail::BankRole::Key?source:word(bytes+4);
                if(!std::isfinite(selected)||selected<0)rejectCodec();
                return Radius{multiply8(broadcast8(selected),broadcast8(0.0625f)),false};
            }
            else return NoTable{};
        }
        /** @brief Load one original FP32 metadata word without typed-alias/alignment inference.
         * @param pointer Original metadata bytes; @return Exact original FP32 word.
         */
        static float word(const unsigned char*pointer)noexcept
        {float value;std::memcpy(&value,pointer,sizeof(value));return value;}
        /** @brief Widen eight original signed byte codes into native FP32 registers.
         * @param pointer Original complete eight-code fragment; @return Exact integer-valued lanes.
         */
        [[gnu::always_inline]] static Fragment8 signedBytes(const std::int8_t*pointer)noexcept
        {
#if LLAMINAR_COMPILED_WITH_NEON
            const auto codes=vmovl_s8(vld1_s8(pointer));
            return {vcvtq_f32_s32(vmovl_s16(vget_low_s16(codes))),vcvtq_f32_s32(vmovl_s16(vget_high_s16(codes)))};
#else
            return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pointer))));
#endif
        }
        const Decoder&owner_;const unsigned char*bytes_;
        [[no_unique_address]] std::conditional_t<rotated,Radius,NoTable>radius_;
    };
    /** @brief Resolve a canonical physical row before entering its coordinate loop.
     * @param physical Original mapped physical row; @return Stable synchronous original-head borrow.
     */
    [[nodiscard]] Row row(int physical)const&noexcept{return Row(*this,binding_.head(physical,head_).data());}
    Row row(int)const&&=delete;
private:
    /** @brief Retain pointer roots and the exact admitted original TQ4 register table only.
     * @param binding Original complete-bank metadata; @param head Original local KV head.
     */
    Decoder(const Binding&binding,int head)noexcept:binding_(binding),head_(head),table_(table(binding)){}
    /** @brief Load a TQ4 table only for its exact original encoding.
     * @param binding Original codebook authority; @return Native original table or no table.
     */
    static Table table(const Binding&binding)noexcept
    {
#if LLAMINAR_COMPILED_WITH_NEON
        if constexpr(Encoding==TensorType::TQ4)return Table(std::span<const float,16>(binding.original().centroids,16));
        else { (void)binding;return {}; }
#else
        (void)binding;return {};
#endif
    }
    Binding binding_;int head_;[[no_unique_address]]Table table_;
};
} // namespace llaminar2::cpu::qsa_native
