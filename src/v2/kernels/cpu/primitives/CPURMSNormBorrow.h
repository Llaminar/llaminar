/**
 * @file CPURMSNormBorrow.h
 * @brief Complete original floating/compressed RMSNorm bank borrows before worksharing.
 *
 * Geometry, address arithmetic, exact-alias policy and immutable gamma/metadata
 * protection are established before workers can write. These synchronous views
 * neither invent capacity nor extend arena leases; PhysicalMemoryAuthority and
 * the caller's tensors remain the storage authorities. No decoded bank, live
 * allocation ledger, persistent execution shadow or numerical state is owned.
 */
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include "../../../tensors/BlockStructures.h"

namespace llaminar2::primitives::rmsnorm_operands
{
/** @brief Compare nonempty byte spans without forming overflowing end pointers.
 * @param a First bank; @param a_bytes Its physical bytes.
 * @param b Second bank; @param b_bytes Its physical bytes.
 * @return Whether the declared byte envelopes intersect.
 */
inline bool overlaps(const void *a,std::size_t a_bytes,const void *b,std::size_t b_bytes) noexcept
{
    if(!a_bytes||!b_bytes)return false;
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    return x<=y?y-x<a_bytes:x-y<b_bytes;
}
/** @brief Validate declared original bank arithmetic, without asserting allocated capacity.
 * @param bank Original synchronous borrow; @param units Original physical words.
 * @param unit_bytes Exact physical word width.
 * @throws std::invalid_argument For missing data or unrepresentable/wrapping extents.
 */
inline void requireBank(const void *bank,std::size_t units,std::size_t unit_bytes)
{
    if(!unit_bytes||units>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/unit_bytes)
        throw std::invalid_argument("CPU RMS original bank extent exceeds the pointer domain");
    const auto bytes=units*unit_bytes;
    if(bytes&&(!bank||reinterpret_cast<std::uintptr_t>(bank)>std::numeric_limits<std::uintptr_t>::max()-bytes))
        throw std::invalid_argument("CPU RMS requires a complete nonwrapping original bank");
}

/** @brief Checked complete original floating bank and immutable normalization policy.
 * @tparam Word Exact original/result floating physical word (FP32 or encoded16-bit).
 */
template<class Word>
class FloatingBorrow final
{
    static_assert(std::is_same_v<Word,float>||std::is_same_v<Word,std::uint16_t>);
public:
    /** @brief Bind the complete invocation before any worker can mutate data.
     * @param source Original complete row-major bank; @param gamma Immutable FP32 weights or identity.
     * @param output Original writable bank, permitting only exact source aliases.
     * @param rows Original row count; @param columns Original logical width.
     * @param epsilon Finite nonnegative original normalization policy.
     * @return A synchronous immutable addressing/policy borrow.
     * @throws std::invalid_argument For invalid geometry, policy or aliases.
     */
    static FloatingBorrow bind(const Word *source,const float *gamma,Word *output,
        std::size_t rows,std::size_t columns,float epsilon)
    {
        if(!std::isfinite(epsilon)||epsilon<0.0f||rows>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())||
           columns>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(Word)||
           (columns&&rows>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(Word)/columns))
            throw std::invalid_argument("CPU RMS requires representable original geometry and a finite nonnegative epsilon");
        const auto words=rows*columns;
        requireBank(source,words,sizeof(Word));requireBank(output,words,sizeof(Word));
        if(gamma)requireBank(gamma,columns,sizeof(float));
        const auto bytes=words*sizeof(Word);
        if(source!=output&&overlaps(source,bytes,output,bytes))
            throw std::invalid_argument("CPU RMS supports exact source aliases, not partial original-bank overlap");
        if(gamma&&overlaps(output,bytes,gamma,columns*sizeof(float)))
            throw std::invalid_argument("CPU RMS output aliases immutable original gamma");
        return FloatingBorrow(source,gamma,output,rows,columns,epsilon);
    }
    /** @return Whether the complete original invocation has no arithmetic work. */
    [[nodiscard]] bool empty()const noexcept{return rows_==0||columns_==0;}
    /** @return Original row count, independent of worker assignment. */
    [[nodiscard]] std::size_t rows()const noexcept{return rows_;}
    /** @return Original logical width, independent of physical padding. */
    [[nodiscard]] std::size_t columns()const noexcept{return columns_;}
    /** @return Original finite epsilon, unchanged by dtype or worker count. */
    [[nodiscard]] float epsilon()const noexcept{return epsilon_;}
    /** @return Original immutable weights, or an explicitly empty identity span. */
    [[nodiscard]] std::span<const float> gamma()const noexcept
    {return gamma_?std::span<const float>(gamma_,columns_):std::span<const float>{};}
    /** @brief Borrow one original logical input row.
     * @param row Original row coordinate; @return Complete original row.
     * @throws std::out_of_range For an invalid original row coordinate.
     */
    [[nodiscard]] std::span<const Word> sourceRow(std::size_t row)const
    {
        if(row>=rows_)throw std::out_of_range("CPU RMS input row outside original bank");
        return {columns_?source_+row*columns_:source_,columns_};
    }
    /** @brief Borrow one original logical output row.
     * @param row Original row coordinate; @return Complete original output row.
     * @throws std::out_of_range For an invalid original row coordinate.
     */
    [[nodiscard]] std::span<Word> outputRow(std::size_t row)const
    {
        if(row>=rows_)throw std::out_of_range("CPU RMS output row outside original bank");
        return {columns_?output_+row*columns_:output_,columns_};
    }
    /** @brief Retain immutable invocation metadata throughout publication.
     * @param metadata Original live policy/descriptor; @param bytes Its complete physical extent.
     * @throws std::invalid_argument For any metadata/output alias.
     */
    void requireMetadataReadOnly(const void *metadata,std::size_t bytes)const
    {
        if(overlaps(output_,rows_*columns_*sizeof(Word),metadata,bytes))
            throw std::invalid_argument("CPU RMS output aliases retained invocation metadata");
    }
private:
    /** @brief Retain only already-checked original addressing and policy.
     * @param source Original input; @param gamma Original immutable weights.
     * @param output Original output; @param rows Original rows; @param columns Original width.
     * @param epsilon Original finite policy.
     */
    FloatingBorrow(const Word *source,const float *gamma,Word *output,std::size_t rows,std::size_t columns,float epsilon)
        :source_(source),gamma_(gamma),output_(output),rows_(rows),columns_(columns),epsilon_(epsilon){}
    const Word *source_;
    const float *gamma_;
    Word *output_;
    std::size_t rows_,columns_;
    float epsilon_;
};
/** @brief Checked original Q16-to-FP32 expansion; compressed storage is never an output alias.
 * @details The physical72-byte source block and128-byte publication fragment
 * have different geometries. Complete-bank protection precedes worksharing,
 * including overlaps with future rows, immutable gamma and retained options.
 * This synchronous borrow neither owns storage nor proves its allocated capacity.
 */
class Q16ExpansionBorrow final
{
public:
    /** @brief Bind complete original compressed/input and floating/output banks.
     * @param source Original Q16 blocks; @param gamma Complete immutable FP32 gamma.
     * @param output Disjoint writable FP32 bank; @param rows Original logical M.
     * @param columns Original K, which must be divisible by32; @param epsilon Original finite policy.
     * @return Checked original coordinates and policy, without decoded storage.
     * @throws std::invalid_argument For malformed extents, pointers, policy or complete-bank aliases.
     */
    static Q16ExpansionBorrow bind(const Q16_1Block *source,const float *gamma,float *output,
                                   std::size_t rows,std::size_t columns,float epsilon)
    {
        constexpr auto bound=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
        if(columns%32||rows>bound||columns>bound/sizeof(float)||
           (columns&&rows>bound/sizeof(float)/columns)||!std::isfinite(epsilon)||epsilon<0.0f)
            throw std::invalid_argument("Q16 RMS requires complete32-value rows, representable geometry and finite nonnegative epsilon");
        const auto blocks=columns/32;
        if(blocks&&rows>bound/sizeof(Q16_1Block)/blocks)
            throw std::invalid_argument("Q16 RMS original compressed bank exceeds the pointer domain");
        requireBank(source,rows*blocks,sizeof(Q16_1Block));
        requireBank(output,rows*columns,sizeof(float));
        if(rows&&columns)requireBank(gamma,columns,sizeof(float));
        const auto bytes=rows*columns*sizeof(float);
        if(overlaps(output,bytes,source,rows*blocks*sizeof(Q16_1Block))||
           overlaps(output,bytes,gamma,columns*sizeof(float)))
            throw std::invalid_argument("Q16 RMS expansion aliases original compressed storage or immutable gamma");
        return Q16ExpansionBorrow(source,gamma,output,rows,columns,epsilon);
    }
    /** @return Whether the complete invocation contains no arithmetic rows/columns. */
    [[nodiscard]] bool empty()const noexcept{return !rows_||!columns_;}
    /** @return Original row count, not worker-local ownership. */
    [[nodiscard]] std::size_t rows()const noexcept{return rows_;}
    /** @return Original logical FP32 width, not physical block count. */
    [[nodiscard]] std::size_t columns()const noexcept{return columns_;}
    /** @return Original finite nonnegative epsilon. */
    [[nodiscard]] float epsilon()const noexcept{return epsilon_;}
    /** @return Complete immutable original weights; empty work requires no pointer. */
    [[nodiscard]] std::span<const float> gamma()const noexcept
    {return empty()?std::span<const float>{}:std::span<const float>(gamma_,columns_);}
    /** @brief Borrow a complete original compressed row without decoding.
     * @param row Original row coordinate; @return Original physical blocks.
     * @throws std::out_of_range For an invalid row coordinate.
     */
    [[nodiscard]] std::span<const Q16_1Block> sourceRow(std::size_t row)const
    {
        if(row>=rows_)throw std::out_of_range("Q16 RMS source row outside original bank");
        const auto blocks=columns_/32;
        return {blocks?source_+row*blocks:source_,blocks};
    }
    /** @brief Borrow one disjoint original floating output row.
     * @param row Original row coordinate; @return Writable logical output.
     * @throws std::out_of_range For an invalid row coordinate.
     */
    [[nodiscard]] std::span<float> outputRow(std::size_t row)const
    {
        if(row>=rows_)throw std::out_of_range("Q16 RMS output row outside original bank");
        return {columns_?output_+row*columns_:output_,columns_};
    }
    /** @brief Protect complete retained invocation metadata before publication.
     * @param metadata Original live descriptor/options; @param bytes Complete metadata extent.
     * @throws std::invalid_argument For any writable-bank intersection.
     */
    void requireMetadataReadOnly(const void *metadata,std::size_t bytes)const
    {
        if(overlaps(output_,rows_*columns_*sizeof(float),metadata,bytes))
            throw std::invalid_argument("Q16 RMS output aliases retained invocation metadata");
    }
private:
    /** @brief Retain only validated original addressing and policy.
     * @param source Original blocks; @param gamma Original gamma; @param output Disjoint result.
     * @param rows Original M; @param columns Original K; @param epsilon Original finite policy.
     */
    Q16ExpansionBorrow(const Q16_1Block *source,const float *gamma,float *output,
                       std::size_t rows,std::size_t columns,float epsilon)
        :source_(source),gamma_(gamma),output_(output),rows_(rows),columns_(columns),epsilon_(epsilon){}
    const Q16_1Block *source_;
    const float *gamma_;
    float *output_;
    std::size_t rows_,columns_;
    float epsilon_;
};
} // namespace llaminar2::primitives::rmsnorm_operands
