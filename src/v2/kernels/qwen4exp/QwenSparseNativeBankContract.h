/**
 * @file QwenSparseNativeBankContract.h
 * @brief Checked original QSA physical-bank and codec-table byte geometry.
 *
 * Static validation and CPU native readers consume the same geometry authority:
 * a logical256-coordinate head is never inferred from stride/padding or an ISA.
 * Binding validates representable addresses and metadata, not allocation capacity
 * or a lease. PhysicalMemoryAuthority/cache owners still admit and retain every
 * original allocation through synchronous consumption or recorded GPU lifetime.
 * These immutable descriptors copy pointer roots only; no payload, decoded head,
 * fitted coefficient bank or parallel live memory ledger is created.
 */
#pragma once
#include "QwenSparseMath.h"
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace llaminar2::qwen4exp::native_bank
{
/** @brief Representable original byte interval; construction proves no physical capacity. */
class Range final
{
public:
    /** @brief Check a complete declared interval before deriving any subpointer.
     * @param pointer Original allocation root; @param bytes Declared consumed extent.
     * @param alignment Required physical element alignment, not SIMD alignment.
     * @return Synchronous non-owning address-domain proof.
     * @throws std::invalid_argument For null, wrapping or misaligned storage.
     */
    [[nodiscard]] static Range bind(const void*pointer,std::size_t bytes,std::size_t alignment)
    {
        const auto address=reinterpret_cast<std::uintptr_t>(pointer);
        if(!pointer||!alignment||address%alignment||bytes>std::numeric_limits<std::uintptr_t>::max()-address)
            throw std::invalid_argument("Qwen sparse attention: missing, wrapping or misaligned original storage");
        return Range(address,bytes);
    }
    /** @brief Test declared-byte intersection without forming out-of-object pointers.
     * @param other Original interval from the same checked domain.
     * @return True exactly when nonempty original byte ranges intersect.
     */
    [[nodiscard]] bool overlaps(Range other)const noexcept
    {return bytes_&&other.bytes_&&(address_<=other.address_?other.address_-address_<bytes_:address_-other.address_<other.bytes_);}
    /** @return Original declared byte extent, never a reservation or capacity estimate. */
    [[nodiscard]] std::size_t bytes()const noexcept{return bytes_;}
private:
    /** @brief Retain only a previously checked root and extent.
     * @param address Representable aligned root; @param bytes Checked declared extent.
     */
    Range(std::uintptr_t address,std::size_t bytes)noexcept:address_(address),bytes_(bytes){}
    std::uintptr_t address_;std::size_t bytes_;
};

/** @brief Single layout/role authority for original native heads and immutable tables. */
class Binding final
{
public:
    /** @brief Authenticate one complete physical bank before worksharing or graph construction.
     * @param bank Original setup-owned pointer roots and declared physical strides.
     * @param role Closed key/value policy; AQ8 is an anchored-key encoding only.
     * @param heads Original local KV head count, bounded by the canonical two heads.
     * @param tokens Original physical row capacity, not a live or selected count.
     * @return Immutable synchronous descriptor, with no ownership/lease extension.
     * @throws std::invalid_argument For unsupported codec/role/layout, malformed
     * geometry, absent metadata, unaligned/wrapping addresses or size overflow.
     */
    [[nodiscard]] static Binding bind(const QwenSparseNativeBank&bank,detail::BankRole role,int heads,int tokens)
    {
        if((role!=detail::BankRole::Key&&role!=detail::BankRole::Value)||heads<1||heads>2||tokens<1)
            throw std::invalid_argument("Qwen sparse attention: invalid original native bank role or capacity");
        std::size_t bytes=0,alignment=4;
        switch(bank.codec)
        {
        case TensorType::FP32:bytes=1024;break;
        case TensorType::FP16:case TensorType::BF16:bytes=512;alignment=2;break;
        case TensorType::Q8_1:bytes=8*36;break;
        case TensorType::Q16_1:
            if(bank.q16_block_size!=32&&bank.q16_block_size!=64&&bank.q16_block_size!=128)
                throw std::invalid_argument("Qwen sparse attention: unsupported native Q16 block size");
            bytes=std::size_t(256/bank.q16_block_size)*(8+2*bank.q16_block_size);break;
        case TensorType::AQ8:
            if(role!=detail::BankRole::Key||!bank.anchor)
                throw std::invalid_argument("Qwen sparse attention: AQ8 requires an immutable key anchor");
            bytes=260;(void)Range::bind(bank.anchor,std::size_t(heads)*256*sizeof(float),alignof(float));break;
        case TensorType::TQ4:case TensorType::TQ8:
            (void)Range::bind(bank.centroids,(bank.codec==TensorType::TQ4?16:256)*sizeof(float),alignof(float));
            for(int head=0;head<heads;++head)(void)Range::bind(bank.rotations[head],256*256*sizeof(float),alignof(float));
            bytes=bank.codec==TensorType::TQ4?136:264;break;
        default:throw std::invalid_argument("Qwen sparse attention: unsupported native cache codec");
        }
        const auto lastToken=multiply(std::size_t(tokens-1),bank.token_stride_bytes);
        const auto lastHead=multiply(std::size_t(heads-1),bank.head_stride_bytes);
        const bool tokenMajor=bank.layout==QwenSparseNativeLayout::TokenMajor&&bank.head_stride_bytes>=bytes&&
            bank.token_stride_bytes>=add(lastHead,bytes);
        const bool headMajor=bank.layout==QwenSparseNativeLayout::HeadMajor&&bank.token_stride_bytes>=bytes&&
            bank.head_stride_bytes>=add(lastToken,bytes);
        if((!tokenMajor&&!headMajor)||bank.token_stride_bytes%alignment||bank.head_stride_bytes%alignment)
            throw std::invalid_argument("Qwen sparse attention: invalid native bank geometry");
        const auto total=add(add(lastToken,lastHead),bytes);
        const auto range=Range::bind(bank.data,total,alignment);
        return Binding(bank,role,heads,tokens,bytes,range);
    }
    /** @brief Resolve one original selected head; no codec conversion or allocation occurs.
     * @param physical Already resolved canonical physical row.
     * @param head Original participant-local KV head.
     * @return Exact original physical head bytes, excluding stride/padding.
     * @details Dynamic corruption is fatal without throwing across an OpenMP
     * collective. The public caller validates every consumed mapping first.
     */
    [[nodiscard]] std::span<const unsigned char> head(int physical,int head)const noexcept
    {
        if(physical<0||physical>=tokens_||head<0||head>=heads_)rejectCoordinates();
        return {static_cast<const unsigned char*>(bank_.data)+std::size_t(physical)*bank_.token_stride_bytes+
            std::size_t(head)*bank_.head_stride_bytes,head_bytes_};
    }
    /** @brief Exclude a prospective publication from all original read-only operands.
     * @param output Complete checked publication range, including inactive capacity.
     * @throws std::invalid_argument Before publication if any original bank/table overlaps.
     */
    void exclude(Range output)const
    {
        if(range_.overlaps(output))throw std::invalid_argument("Qwen sparse attention: output overlaps original native bank");
        if(bank_.codec==TensorType::AQ8&&Range::bind(bank_.anchor,std::size_t(heads_)*256*sizeof(float),alignof(float)).overlaps(output))
            throw std::invalid_argument("Qwen sparse attention: output overlaps original AQ8 anchor");
        if(bank_.codec==TensorType::TQ4||bank_.codec==TensorType::TQ8)
        {
            if(Range::bind(bank_.centroids,(bank_.codec==TensorType::TQ4?16:256)*sizeof(float),alignof(float)).overlaps(output))
                throw std::invalid_argument("Qwen sparse attention: output overlaps original TQ codebook");
            for(int head=0;head<heads_;++head)if(Range::bind(bank_.rotations[head],256*256*sizeof(float),alignof(float)).overlaps(output))
                throw std::invalid_argument("Qwen sparse attention: output overlaps original TQ rotation");
        }
    }
    /** @return Unmodified original bank roots/codec, never decoded or independently owned state. */
    [[nodiscard]] const QwenSparseNativeBank& original()const noexcept{return bank_;}
    /** @return Original key/value radial program sealed by binding. */
    [[nodiscard]] detail::BankRole role()const noexcept{return role_;}
    /** @return Original admitted participant-local KV head count. */
    [[nodiscard]] int heads()const noexcept{return heads_;}
    /** @return Complete original consumed byte range, not the allocation's capacity. */
    [[nodiscard]] Range range()const noexcept{return range_;}
private:
    /** @brief Fail fatally in every build without propagating an exception through a collective.
     * @details The ordinary assertion interface throws (and is absent in Release);
     * this post-admission coordinate invariant must instead remain always fatal.
     */
    [[noreturn]] static void rejectCoordinates()noexcept
    {
        std::fputs("ERROR: QSA native reader lost its admitted original coordinates\n",stderr);std::abort();
    }
    /** @brief Retain authenticated roots only; source owners keep all allocation/lease authority.
     * @param bank Original immutable metadata; @param role Admitted radial program.
     * @param heads Valid local heads; @param tokens Physical row capacity.
     * @param bytes Exact head bytes; @param range Checked full consumed byte interval.
     */
    Binding(const QwenSparseNativeBank&bank,detail::BankRole role,int heads,int tokens,std::size_t bytes,Range range)noexcept:
        bank_(bank),role_(role),heads_(heads),tokens_(tokens),head_bytes_(bytes),range_(range){}
    /** @brief Check declared geometry multiplication without reserving or estimating memory.
     * @param a Original factor; @param b Original stride; @return Exact product.
     * @throws std::invalid_argument For unrepresentable original geometry.
     */
    static std::size_t multiply(std::size_t a,std::size_t b)
    {
        if(a&&b>std::numeric_limits<std::size_t>::max()/a)throw std::invalid_argument("Qwen sparse attention: native bank stride overflow");
        return a*b;
    }
    /** @brief Check declared extent addition without forming a wrapping pointer.
     * @param a Original offset; @param b Original extent; @return Exact sum.
     * @throws std::invalid_argument For unrepresentable original geometry.
     */
    static std::size_t add(std::size_t a,std::size_t b)
    {
        if(b>std::numeric_limits<std::size_t>::max()-a)throw std::invalid_argument("Qwen sparse attention: native bank extent overflow");
        return a+b;
    }
    QwenSparseNativeBank bank_;detail::BankRole role_;int heads_,tokens_;std::size_t head_bytes_;Range range_;
};
} // namespace llaminar2::qwen4exp::native_bank
