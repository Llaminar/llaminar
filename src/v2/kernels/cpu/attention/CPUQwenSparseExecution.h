/**
 * @file CPUQwenSparseExecution.h
 * @brief Checked CPU QSA invocation with native original-storage score/value execution.
 *
 * Admission seals CPU-owned replay, original pointer roots, complete read/write
 * exclusions and every consumed selected-ring address before worksharing. Each
 * row/head then streams original codec fragments through the declared eight- or
 * sixteen-chain score program. The rotated-value path retains only eight weighted
 * coordinates at a time and applies the original ordered transpose-FMA program;
 * no decoded cache head or full weighted-value mirror exists. Produced query
 * rotation and softmax logits are distinct algorithm operands, not cache shadows.
 * PhysicalMemoryAuthority/cache owners retain capacity and leases; these checked
 * synchronous descriptors neither allocate nor establish independent accounting.
 */
#pragma once
#include "CPUQwenSparseNativeCodec.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <type_traits>
namespace llaminar2::qwen4exp::cpu_sparse
{
namespace native=::llaminar2::cpu::qsa_native;
/** @brief Authenticate the original replay interval before any caller loads or forms member addresses.
 * @param replay Original untrusted CPU replay root, with owner-retained storage/lease.
 * @throws std::invalid_argument For null, wrapping or misaligned original bytes.
 * @details This is an address-domain proof, not an allocation/capacity claim.
 */
inline void validateReplayRoot(const QwenSparseReplay*replay)
{(void)native_bank::Range::bind(replay,sizeof(QwenSparseReplay),alignof(QwenSparseReplay));}
/** @brief Validate the original CPU-owned causal replay before any publication.
 * @param replay Original transaction state; @param rows Admitted candidate rows.
 * @param tokens Original physical/logical token limit.
 * @throws std::invalid_argument For noncausal, overflowing or unrepresentable state.
 */
inline void validateReplay(const QwenSparseReplay&replay,int rows,int tokens)
{
    if(replay.prefix_tokens<0||replay.live_rows<0||replay.live_rows>rows||
        std::int64_t(replay.prefix_tokens)+replay.live_rows>tokens||replay.position_origin<0||
        std::int64_t(replay.position_origin)+replay.prefix_tokens+replay.live_rows>std::numeric_limits<int>::max())
        throw std::invalid_argument("CPU Qwen sparse replay: invalid causal prefix or live row count");
}
/** @brief Select one original native decoder outside its coordinate/selected-row loops.
 * @tparam Function Original operation accepting a stable decoder.
 * @param binding Complete original bank; @param head Original local KV head.
 * @param function Synchronous native consumer; @return Original consumer result.
 */
template<class Function>decltype(auto) withDecoder(const native_bank::Binding&binding,int head,Function&&function)noexcept
{
#define QSA_CODEC_CASE(codec) case TensorType::codec:{const auto decoder=native::Decoder<TensorType::codec>::bind(binding,head);return function(decoder);}
    switch(binding.original().codec)
    {
    QSA_CODEC_CASE(FP32);QSA_CODEC_CASE(FP16);QSA_CODEC_CASE(BF16);QSA_CODEC_CASE(Q8_1);
    QSA_CODEC_CASE(Q16_1);QSA_CODEC_CASE(AQ8);QSA_CODEC_CASE(TQ4);QSA_CODEC_CASE(TQ8);
    default:native::rejectCodec();
    }
#undef QSA_CODEC_CASE
}
#if defined(__AVX512F__)
/** @brief Join adjacent original register fragments into the declared sixteen-lane program.
 * @param low Original coordinates0..7; @param high Original coordinates8..15.
 * @return Sixteen original coordinates, without FP32 staging storage.
 */
[[gnu::always_inline]] inline __m512 join16(native::Fragment8 low,native::Fragment8 high)noexcept
{return _mm512_insertf32x8(_mm512_castps256_ps512(low),high,1);}
#endif
/** @brief Score an original selected row with the exact declared native chain tree.
 * @tparam Row Stable original physical-head borrow; @param row Original codec row.
 * @param query Original/produced query basis; @param isa Cold-admitted execution ISA.
 * @return Original finite dot, with no decoded head or whole-operation scalar replay.
 */
template<class Row>[[gnu::always_inline]] inline float score(const Row&row,const float*query,ISALevel isa)noexcept
{
#if defined(__AVX512F__)
    if(isa==ISALevel::AVX512)
    {
        auto total=_mm512_setzero_ps();
        for(int c=0;c<256;c+=16)total=_mm512_fmadd_ps(_mm512_loadu_ps(query+c),join16(row.read(c),row.read(c+8)),total);
        return native::fold16(total);
    }
#else
    (void)isa;
#endif
    auto total=native::zero8();
    for(int c=0;c<256;c+=8)total=native::fma8(total,native::load8(query+c),row.read(c));
    return native::fold8(total);
}
/** @brief Extract one already-computed weighted coordinate without a FP32 lane array.
 * @tparam Lane Original coordinate within a native eight-coordinate fragment.
 * @param values Original computed lanes; @return The exact original FP32 word.
 */
template<unsigned Lane>[[gnu::always_inline]] inline float lane(native::Fragment8 values)noexcept
{
    static_assert(Lane<8);
#if LLAMINAR_COMPILED_WITH_NEON
    if constexpr(Lane<4)return vgetq_lane_f32(values.low,Lane);
    else return vgetq_lane_f32(values.high,Lane-4);
#else
    const auto part=Lane<4?_mm256_castps256_ps128(values):_mm256_extractf128_ps(values,1);
    return _mm_cvtss_f32(_mm_shuffle_ps(part,part,_MM_SHUFFLE(Lane%4,Lane%4,Lane%4,Lane%4)));
#endif
}
/** @brief Apply one original transpose coordinate in the fixed increasing-coordinate FMA order.
 * @tparam Lane Original lane0..7; @param weighted Completed eight weighted coordinates.
 * @param matrix Original admitted row-major rotation; @param base Original coordinate base.
 * @param output Exclusive spatial-basis result; @param isa Admitted native program.
 */
template<unsigned Lane>[[gnu::always_inline]] inline void transposeCoordinate(native::Fragment8 weighted,const float*matrix,int base,float*output,ISALevel isa)noexcept
{
    const auto value=lane<Lane>(weighted);const auto*coefficients=matrix+std::size_t(base+Lane)*256;
#if defined(__AVX512F__)
    if(isa==ISALevel::AVX512)
    {
        const auto scale=_mm512_set1_ps(value);
        for(int c=0;c<256;c+=16)
        {
            const auto lo=native::load8(coefficients+c),hi=native::load8(coefficients+c+8);
            if(!native::finite8(lo)||!native::finite8(hi))native::rejectCodec();
            const auto result=_mm512_fmadd_ps(join16(lo,hi),scale,_mm512_loadu_ps(output+c));
            _mm512_storeu_ps(output+c,result);
        }
        return;
    }
#else
    (void)isa;
#endif
    const auto scale=native::broadcast8(value);
    for(int c=0;c<256;c+=8)
    {
        const auto coefficients8=native::load8(coefficients+c);if(!native::finite8(coefficients8))native::rejectCodec();
        native::store8(output+c,native::fma8(native::load8(output+c),coefficients8,scale));
    }
}
/** @brief Sequence the exact original eight adjacent transpose coordinates, not an arithmetic reduction.
 * @tparam Lanes Increasing compile-time coordinate indices; @param weighted Original weighted fragment.
 * @param matrix Original rotation; @param base Original coordinate base.
 * @param output Exclusive result; @param isa Admitted native program; @param indices Ordered index token.
 */
template<unsigned...Lanes>[[gnu::always_inline]] inline void transposeFragment(native::Fragment8 weighted,const float*matrix,int base,float*output,ISALevel isa,std::integer_sequence<unsigned,Lanes...>indices)noexcept
{
    (void)indices;(transposeCoordinate<Lanes>(weighted,matrix,base,output,isa),...);
}

/** @brief One complete checked CPU invocation; only this class authors row/head publications. */
class Invocation final
{
public:
    /** @brief Seal original replay, addresses, exclusions and native program before collective execution.
     * @param p Existing arena/cache-owned public kernel bindings.
     * @return Synchronous invocation descriptor, never a capacity/lease or host device-state shadow.
     * @throws std::invalid_argument Before stores for unsupported policy/ISA, malformed
     * replay/ring/selected indices, wrapping metadata or conflicting publications.
     */
    [[nodiscard]] static Invocation bind(const QwenSparseAttention&p)
    {
        validateQwenSparseAttentionArithmetic(p.arithmetic,DeviceType::CPU);validateQwenSparseAttention(p);
        const auto isa=activeISALevel();native::requireNativeProgram(isa);
        // Replay is an original untrusted root: its interval must be admitted
        // before even the first CPU-owned word is copied or interpreted.
        validateReplayRoot(p.replay);
        const auto replay=*p.replay;validateReplay(replay,p.row_capacity,p.token_capacity);
        if(p.kv.key_policy.encoding!=attention::AttentionKeyCacheEncoding::PostRotary)
            throw std::invalid_argument("CPU Qwen sparse attention requires canonical post-RoPE cache publication");
        const auto keys=native_bank::Binding::bind(p.kv.keys,detail::BankRole::Key,p.kv_heads,p.token_capacity);
        const auto values=native_bank::Binding::bind(p.kv.values,detail::BankRole::Value,p.kv_heads,p.token_capacity);
        const auto rowHeads=std::size_t(p.row_capacity)*p.query_heads;
        const auto output=range(p.output,rowHeads*256);exclude(p,keys,values,output,true);
        if(p.observations)
        {
            const auto observations=range(p.observations,rowHeads);exclude(p,keys,values,observations,false);
            if(observations.overlaps(output))throw std::invalid_argument("CPU QSA output and observations overlap");
        }
        for(int row=0;row<replay.live_rows;++row)
        {
            int origin=0,capacity=0;
            if(!detail::resolveMapping(p.kv,replay,row,p.token_capacity,origin,capacity))
                throw std::invalid_argument("CPU Qwen sparse attention: stale or invalid canonical ring mapping");
            const auto count=p.selected_count[row];if(count<=0||count>QwenSparseSelected)
                throw std::invalid_argument("CPU Qwen sparse attention: invalid selected live count");
            int previous=-1;for(int i=0;i<count;++i)
            {
                const auto token=p.selected[std::size_t(row)*QwenSparseSelected+i];
                if(token<=previous||token>=replay.prefix_tokens+row+1)
                    throw std::invalid_argument("CPU Qwen sparse attention: noncausal, repeated, or unsorted selected index");
                previous=token;
            }
        }
        return Invocation(p,replay,isa,keys,values);
    }
    /** @return Exact original live row/head work, including the zero-work collective case. */
    [[nodiscard]] int rowHeads()const noexcept{return replay_.live_rows*p_.query_heads;}
    /** @brief Execute one admitted independent row/head through the native original-storage program.
     * @param rowHead Original invocation-local row/head index, never an inactive capacity row.
     * @details The caller gives each index one worksharing owner and completes
     * its retained team before returning. Numeric corruption remains fatal;
     * earlier retired rows are not rolled back or repaired to a second mode.
     */
    void execute(int rowHead)const noexcept
    {
        if(rowHead<0||rowHead>=rowHeads())native::rejectCodec();
        const int row=rowHead/p_.query_heads,head=rowHead%p_.query_heads;
        const int kvHead=(p_.query_head_start+head)/12-p_.kv_head_start;
        const auto count=p_.selected_count[row];if(count<=0||count>QwenSparseSelected)native::rejectCodec();
        const auto*selected=p_.selected+std::size_t(row)*QwenSparseSelected;
        int origin=0,capacity=0;if(!detail::resolveMapping(p_.kv,replay_,row,p_.token_capacity,origin,capacity))native::rejectCodec();
        const float*query=p_.query+std::size_t(rowHead)*256;float rotatedQuery[256];
        if(detail::isRotated(p_.kv.keys))
        {
            for(int c=0;c<256;++c)
            {rotatedQuery[c]=native::dot<256>(p_.kv.keys.rotations[kvHead]+c*256,query,isa_);if(!std::isfinite(rotatedQuery[c]))native::rejectCodec();}
            query=rotatedQuery;
        }
        float logits[QwenSparseSelected];float maximum=-std::numeric_limits<float>::infinity();
        withDecoder(keys_,kvHead,[&](const auto&decoder)
        {
            for(int i=0;i<count;++i)
            {
                const auto nativeRow=decoder.row(detail::physicalRow(selected[i],origin,capacity));
                logits[i]=score(nativeRow,query,isa_)*0.0625f;if(!std::isfinite(logits[i]))native::rejectCodec();
                maximum=std::max(maximum,logits[i]);
            }
        });
        float denominator=0;for(int i=0;i<count;++i){logits[i]=std::exp(logits[i]-maximum);denominator+=logits[i];}
        if(!std::isfinite(denominator)||denominator<=0)native::rejectCodec();
        for(int i=0;i<count;++i)logits[i]/=denominator;
        auto*output=p_.output+std::size_t(rowHead)*256;std::fill_n(output,256,0.0f);
        withDecoder(values_,kvHead,[&](const auto&decoder)
        {
            if constexpr(std::remove_cvref_t<decltype(decoder)>::rotatedBasis)
            {
                for(int base=0;base<256;base+=8)
                {
                    auto weighted=native::zero8();
                    for(int i=0;i<count;++i)
                    {const auto source=decoder.row(detail::physicalRow(selected[i],origin,capacity));weighted=native::fma8(weighted,source.read(base),native::broadcast8(logits[i]));}
                    if(!native::finite8(weighted))native::rejectCodec();
                    transposeFragment(weighted,p_.kv.values.rotations[kvHead],base,output,isa_,std::make_integer_sequence<unsigned,8>{});
                }
            }
            else for(int i=0;i<count;++i)
            {
                const auto source=decoder.row(detail::physicalRow(selected[i],origin,capacity));
#if defined(__AVX512F__)
                if(isa_==ISALevel::AVX512)
                {
                    const auto probability=_mm512_set1_ps(logits[i]);
                    for(int c=0;c<256;c+=16)_mm512_storeu_ps(output+c,_mm512_fmadd_ps(probability,join16(source.read(c),source.read(c+8)),_mm512_loadu_ps(output+c)));
                    continue;
                }
#endif
                const auto probability=native::broadcast8(logits[i]);
                for(int c=0;c<256;c+=8)native::store8(output+c,native::fma8(native::load8(output+c),probability,source.read(c)));
            }
        });
        for(int c=0;c<256;++c)
        {
            if(p_.gate)
            {const auto gate=p_.gate[std::size_t(rowHead)*256+c];if(!std::isfinite(gate))native::rejectCodec();output[c]*=1.0f/(1.0f+std::exp(-gate));}
            if(!std::isfinite(output[c]))native::rejectCodec();
        }
        if(p_.observations)p_.observations[rowHead]={std::uint64_t(count)*256,std::uint64_t(count)*256,std::uint32_t(count)};
    }
private:
    /** @brief Retain sealed original roots/replay only, with no new allocation/lease authority.
     * @param p Original metadata; @param replay CPU-owned immutable transaction snapshot.
     * @param isa Selected native program; @param keys Original key binding; @param values Original value binding.
     */
    Invocation(const QwenSparseAttention&p,QwenSparseReplay replay,ISALevel isa,native_bank::Binding keys,native_bank::Binding values)noexcept:
        p_(p),replay_(replay),isa_(isa),keys_(keys),values_(values){}
    /** @brief Form a checked complete original element interval, without claiming capacity.
     * @tparam T Original element type; @param pointer Original root; @param count Declared original elements.
     * @return Checked byte interval; @throws std::invalid_argument For element-byte overflow/wrapping/alignment.
     */
    template<class T>static native_bank::Range range(const T*pointer,std::size_t count)
    {
        if(count>std::numeric_limits<std::size_t>::max()/sizeof(T))throw std::invalid_argument("CPU QSA original element extent overflow");
        return native_bank::Range::bind(pointer,count*sizeof(T),alignof(T));
    }
    /** @brief Exclude each publication from all consumed original read-only banks and metadata.
     * @param p Original caller roots; @param keys Original key authority; @param values Original value authority.
     * @param write Complete proposed publication; @param exactQueryAlias Whether full same-head query reuse is admitted.
     * @throws std::invalid_argument Before any publication for conflicting roots.
     */
    static void exclude(const QwenSparseAttention&p,const native_bank::Binding&keys,const native_bank::Binding&values,native_bank::Range write,bool exactQueryAlias)
    {
        keys.exclude(write);values.exclude(write);const auto rows=std::size_t(p.row_capacity);
        if(write.overlaps(range(p.query,rows*p.query_heads*256))&&(!exactQueryAlias||p.output!=p.query))
            throw std::invalid_argument("CPU QSA publication overlaps another original query interval");
        if(write.overlaps(range(p.replay,1))||write.overlaps(range(&p,1))||write.overlaps(range(p.selected,rows*QwenSparseSelected))||
            write.overlaps(range(p.selected_count,rows))||(p.gate&&write.overlaps(range(p.gate,rows*p.query_heads*256)))||
            (p.kv.addressing==QwenSparseKVAddressing::RowParameters&&write.overlaps(range(p.kv.row_parameters,rows))))
            throw std::invalid_argument("CPU QSA publication overlaps original immutable operands or metadata");
    }
    QwenSparseAttention p_;QwenSparseReplay replay_;ISALevel isa_;native_bank::Binding keys_,values_;
};
} // namespace llaminar2::qwen4exp::cpu_sparse
