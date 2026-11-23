/**
 * @file ROCmNativeLayoutPreload.hip.inc
 * @brief Fork-style expert/role preloads adapted to canonical K32-major columns.
 * Four experts at the same K position are preloaded before consumption; no
 * consecutive-K batching changes the source structure or FP32 commit order.
 * Adapted from Castagna Veloce hc-persist.cu at
 * 3483d462715d61f21ed5877836be35a724446359; see ROCmCastagnaT1.LICENSE.
 */
#include "ROCmNativeLayoutDownGeometry.h"
#include "kernels/rocm/common/ROCmWaveAddress.h"
#include "kernels/common/NativeQuantisedDescriptorValidation.h"
#include <cstddef>
namespace llaminar2::rocm::native_layout_ffn {
/**
 * @brief Validate invariant numeric source fields once at original LDS publication.
 * @details Readiness and original source-policy partitions remain separate. A
 * numeric rejection returns the original positive-zero partial; it must not
 * turn a selected route into an unready route or bypass its weight multiply.
 * This classifier borrows the same descriptor and uses the canonical traits;
 * it owns no source, validity cache, allocation, or arithmetic policy.
 */
struct PhaseProjectionValidation {
    /** @brief One numeric-validity byte, distinct from original source readiness. */
    enum class State : uint8_t { Invalid, Valid };
    /**
     * @brief Evaluate the original role's descriptor-invariant zero predicate.
     * @tparam FMT Original physical decoder; canonical mixed IQ4 compatibility remains exact.
     * @param descriptor Original immutable source descriptor.
     * @param N Original output extent. @param K Original complete reduction extent.
     * @return True precisely when the former numeric-lane invariant guard accepts.
     * @pre The publication leader has already authenticated original source-policy P.
     */
    template<NativeQuantisedFormat FMT>
    __device__ __forceinline__ static bool accepts(const DeviceNativeQuantisedMatrixDesc& descriptor,unsigned N,unsigned K)
    {
        return NativeQuantisedDescriptorValidation::accepts(descriptor,N,K,
            static_cast<unsigned>(FMT));
    }
    /**
     * @brief Classify the same physical codebook selected by phasePartial.
     * @param descriptor Original immutable descriptor. @param N Original output extent.
     * @param K Original complete reduction extent.
     * @return Numeric validity only; unsupported physical/policy state remains fatal at its original authority.
     * @details Actual IQ4_NL stays physical4 and needs no secondary coefficient;
     * physical24 retains its original coefficient. No source-policy id is normalized here.
     */
    __device__ __forceinline__ static State classify(const DeviceNativeQuantisedMatrixDesc& descriptor,unsigned N,unsigned K)
    {
#define NUMERIC_CASE(F) case F:return accepts<F>(descriptor,N,K)?State::Valid:State::Invalid;
        switch(descriptor.codebook_id)
        {
        NUMERIC_CASE(NATIVE_QUANTISED_Q4_0) NUMERIC_CASE(NATIVE_QUANTISED_IQ4_NL)
        NUMERIC_CASE(NATIVE_QUANTISED_IQ4_XS_SOURCE) NUMERIC_CASE(NATIVE_QUANTISED_MXFP4)
        NUMERIC_CASE(NATIVE_QUANTISED_Q4_1)
        NUMERIC_CASE(NATIVE_QUANTISED_Q5_0) NUMERIC_CASE(NATIVE_QUANTISED_Q5_1)
        NUMERIC_CASE(NATIVE_QUANTISED_Q6_K) NUMERIC_CASE(NATIVE_QUANTISED_Q3_K) NUMERIC_CASE(NATIVE_QUANTISED_Q2_K)
        NUMERIC_CASE(NATIVE_QUANTISED_IQ3_S) NUMERIC_CASE(NATIVE_QUANTISED_IQ3_XXS)
        NUMERIC_CASE(NATIVE_QUANTISED_IQ2_S) NUMERIC_CASE(NATIVE_QUANTISED_IQ2_XS) NUMERIC_CASE(NATIVE_QUANTISED_IQ2_XXS)
        NUMERIC_CASE(NATIVE_QUANTISED_IQ1_S) NUMERIC_CASE(NATIVE_QUANTISED_IQ1_M)
        NUMERIC_CASE(NATIVE_QUANTISED_Q8_0) NUMERIC_CASE(NATIVE_QUANTISED_EXPANDED_INT8_MIN)
        default:return State::Invalid;
        }
#undef NUMERIC_CASE
    }
};
static_assert(sizeof(PhaseProjectionValidation::State)==1);
/**
 * @brief Phase-local immutable source metadata, reused by gate/up teams and down batches.
 * @details Original descriptors remain owned by the prepared table. These
 * borrowed planes eliminate descriptor chasing inside a K32 loop; they neither
 * re-encode weights nor outlive the original CTA task. Each wave reads its own
 * role record at a time. Descriptor dimensions are checked by the original
 * publication leader and are not copied into the numeric-worker record.
 * Gate/up expert/output tasks and independent down
 * packet/output tasks are wave-aligned; no wave mixes different task metadata.
 */
struct PhaseProjection {
    const uint8_t* payload; ///< Original canonical K32-major bytes.
    const uint16_t* scales; ///< Original primary metadata.
    const uint16_t* mins; ///< Original optional secondary metadata.
    const uint32_t* emins; ///< Original optional dual-minimum metadata.
    unsigned parts; ///< Exact original source-policy partition count.
    unsigned partition_blocks; ///< Original ceil(K32/parts), resolved once by the batch leader.
    uint8_t codebook; ///< Original physical decoder, independent of source-policy identity.
    bool ready; ///< Original selected and payload-ready state.
    PhaseProjectionValidation::State numeric_validity; ///< Separate original invariant-zero decision, not a readiness replacement.
    /// Packed metadata layout of the planes (nativeQuantisedMetadataLayout): zero
    /// for per-block planes; superblock-coded roles hold codes in `scales` and
    /// per-superblock d/dmin words in `mins` (no emins). Occupies former padding.
    uint8_t metadata_layout;
    /** @return The role's typed metadata planes. */
    __device__ __forceinline__ ::llaminar2::rocm_native_metadata::MetadataPlanes meta() const
    { return ::llaminar2::rocm_native_metadata::planesFromLayout(scales,mins,emins,metadata_layout); }
};
static_assert(std::is_standard_layout_v<PhaseProjection> &&
    std::is_trivially_copyable_v<PhaseProjection> && sizeof(PhaseProjection)==48&&offsetof(PhaseProjection,codebook)==40&&
    offsetof(PhaseProjection,ready)==41&&offsetof(PhaseProjection,numeric_validity)==42&&
    offsetof(PhaseProjection,metadata_layout)==43,
    "Numeric-worker metadata retains only original planes, layout, partitions and publication validity in48bytes");

/**
 * @brief Recover immutable phase metadata from the first active lane of its physical wave.
 * @tparam T Scalar or pointer field type, preserved without truncation.
 * @param value This lane's field from the wave-owned batch descriptor.
 * @return Identical original field, now explicitly wave-uniform to the compiler.
 * @pre Gate/up expert/output tasks and down packet/output tasks align to whole
 * waves; every active lane reads the same role record. Never broadcast a column,
 * partition, K index or arithmetic operand through this helper.
 * @details No descriptor copy or extra LDS record is created. Each field remains
 * at its original consumption site and carries no cross-wave publication claim.
 */
template<class T>
__device__ __forceinline__ T waveMetadataValue(T value)
{
    if constexpr(std::is_pointer_v<T>)
    {
        return reinterpret_cast<T>(waveUniformAddress(reinterpret_cast<uintptr_t>(value)));
    }
    else return static_cast<T>(__builtin_amdgcn_readfirstlane(static_cast<unsigned>(value)));
}
/**
 * @brief Recover one role's metadata planes and representation, wave-uniformly.
 * @tparam FMT Physical decoder selected by the caller.
 * @tparam Binding PhaseProjection reference (LDS or generic provenance).
 * @param binding Role record shared by the whole wave.
 * @return Planes for blockMetadata(): per-block planes keep exactly the planes
 *         this decoder consumes; a codable decoder also borrows the secondary
 *         slot, which holds the superblock plane of a coded role.
 */
template<NativeQuantisedFormat FMT,class Binding>
__device__ __forceinline__ ::llaminar2::rocm_native_metadata::MetadataPlanes phaseRoleMetadata(const Binding& binding)
{
    using Traits=NATIVE_QUANTISEDTraits<FMT>;
    constexpr bool codable=::llaminar2::nativeQuantisedSuperblockCodable(static_cast<uint8_t>(FMT));
    const uint16_t* mins=nullptr;
    const uint32_t* emins=nullptr;
    uint8_t layout=0;
    if constexpr(codable)layout=waveMetadataValue(binding.metadata_layout);
    if constexpr(Traits::requires_secondary_metadata||codable)mins=waveMetadataValue(binding.mins);
    if constexpr(Traits::needs_emin_array)emins=waveMetadataValue(binding.emins);
    return ::llaminar2::rocm_native_metadata::planesFromLayout(waveMetadataValue(binding.scales),mins,emins,layout);
}
/** @brief Native32-bit LDS pointee; the host pass only parses this device helper. */
#if defined(__HIP_DEVICE_COMPILE__)
template<class T> using CurrentFormatLDSValue=T __attribute__((opencl_local));
#else
template<class T> using CurrentFormatLDSValue=T;
#endif
/**
 * @brief Re-form a proven LDS address only inside its selected physical-format arm.
 * @tparam T Original LDS pointee, retaining const qualification.
 * @param pointer Existing local-space batch record or output base.
 * @return The same32-bit local-space address through a register-only compiler boundary.
 * @pre Both callers explicitly recover LDS provenance from their original shared arrays.
 * @details Local pointer size is checked before bit_cast; no generic64-bit
 * address is truncated. The empty constraint adds no memory-ordering claim
 * and never broadcasts across32-lane batches. Actual IR/resources must prove
 * whether the selected-arm lifetime improves generated code.
 */
template<class T>
__device__ __forceinline__ CurrentFormatLDSValue<T>* currentFormatAddress(CurrentFormatLDSValue<T>* pointer)
{
#if defined(__HIP_DEVICE_COMPILE__)
    static_assert(sizeof(pointer)==sizeof(uint32_t));
    uint32_t address=__builtin_bit_cast(uint32_t,pointer);
    asm volatile("" : "+v"(address));
    return __builtin_bit_cast(CurrentFormatLDSValue<T>*,address);
#else
    return pointer;
#endif
}

/** @brief Empty extended-metadata arm for formats without a separate emins plane. */
template<bool Extended> struct BlockExtendedBits {};
/** @brief Exact original Q2_K extended metadata word, present only when consumed. */
template<> struct BlockExtendedBits<true> {uint32_t extended;};
/**
 * @brief One complete current-K operand, including original unconverted metadata bits.
 * @tparam FMT Exact physical format; only its required metadata planes are loaded.
 * @details Primary/secondary half words occupy one register, like the source's
 * packed scale/min word. Formats without a secondary plane keep its upper half
 * zero. The separate extended word exists only for the canonical emins format.
 */
template<NativeQuantisedFormat FMT> struct BlockOperand : BlockExtendedBits<NATIVE_QUANTISEDTraits<FMT>::needs_emin_array> {
    alignas(4) uint8_t packed[NATIVE_QUANTISEDTraits<FMT>::payload_bytes];
    uint32_t scale_pair; ///< Original primary bits in low16, required secondary bits in high16.
};
static_assert(sizeof(BlockOperand<NATIVE_QUANTISED_Q5_1>)==24 &&
    sizeof(BlockOperand<NATIVE_QUANTISED_Q8_0>)==36 &&
    sizeof(BlockOperand<NATIVE_QUANTISED_EXPANDED_INT8_MIN>)==36 &&
    sizeof(BlockOperand<NATIVE_QUANTISED_Q2_K>)==16,
    "Complete operands contain only aligned original payload and required metadata words");
/**
 * @brief Default original generic weight load; explicit full-entry readers may retain AS1 provenance.
 * @details This reader changes no byte layout, source policy or arithmetic. Existing
 * standalone callers keep the exact original scalar-width load expressions.
 */
struct NativeLayoutWeightReader {
    /** @brief Borrow one original typed scalar. @tparam T Exact original type.
     * @param pointer Admitted immutable weight field. @return Original loaded bits. */
    template<class T> __device__ __forceinline__ static T load(const T* pointer) {return *pointer;}
};
/**
 * @brief Load exactly one original canonical payload into its existing operand.
 * @tparam FMT Original physical decoder; odd payload strides retain bounded byte loads.
 * @param operand Existing current-K operand, retaining its original metadata fields.
 * @param payload Original prepared payload plane. @param linear Original K32-major index.
 * @details Shared by the ordinary complete-block helper and EB4's payload-first
 * pass. The original word/byte load expressions and source extent are unchanged.
 */
template<NativeQuantisedFormat FMT,class Reader=NativeLayoutWeightReader>
__device__ __forceinline__ void preloadBlockPayload(BlockOperand<FMT>& operand,
    const uint8_t* payload,size_t linear)
{
    const auto* source=payload+linear*NATIVE_QUANTISEDTraits<FMT>::payload_bytes;
    constexpr unsigned bytes=NATIVE_QUANTISEDTraits<FMT>::payload_bytes;
    if constexpr(bytes%4==0)
    {
        auto* destination=reinterpret_cast<uint32_t*>(operand.packed);
        const auto* words=reinterpret_cast<const uint32_t*>(source);
        #pragma unroll
        for(unsigned word=0;word<bytes/4;++word)destination[word]=Reader::template load<uint32_t>(words+word);
    }
    else
    {
        #pragma unroll
        for(unsigned byte=0;byte<bytes;++byte)operand.packed[byte]=Reader::template load<uint8_t>(source+byte);
    }
}
/**
 * @brief Load only this original format's required metadata into its existing operand.
 * @tparam FMT Original physical decoder; no scale conversion occurs.
 * @param operand Existing current-K operand, retaining its original payload bytes.
 * @param meta Role metadata planes and representation (per-block or superblock-coded).
 * @param address Block-major address (linear index plus superblock index).
 */
template<NativeQuantisedFormat FMT,class Reader=NativeLayoutWeightReader,int Representation=-1>
__device__ __forceinline__ void preloadBlockMetadata(BlockOperand<FMT>& operand,
    const ::llaminar2::rocm_native_metadata::MetadataPlanes& meta,::llaminar2::rocm_native_metadata::BlockAddress address)
{
    using Traits=NATIVE_QUANTISEDTraits<FMT>;
    // Representation: 0 per-block only, 1 coded only (both selected once per
    // role by withRoleRepresentation), -1 uniform runtime branch.
    if constexpr(::llaminar2::nativeQuantisedSuperblockCodable(static_cast<uint8_t>(FMT)) && Representation!=0)
    {
        // Superblock-coded roles reconstruct the identical staged words; the
        // branch is uniform for a whole role and never diverges within a wave.
        if(Representation==1||meta.coded())
        {
            const auto words=::llaminar2::rocm_native_metadata::blockMetadataAt<static_cast<uint8_t>(FMT)>(meta,address);
            uint32_t secondary=0;
            if constexpr(Traits::requires_secondary_metadata)secondary=words.secondary;
            operand.scale_pair=uint32_t(words.primary)|(secondary<<16);
            if constexpr(Traits::needs_emin_array)operand.extended=words.emin;
            return;
        }
    }
    const auto* scales=meta.primary;
    const auto* mins=static_cast<const uint16_t*>(meta.secondary);
    const auto* emins=meta.emins;
    const size_t linear=address.linear;
    // Normalize to the staged primary word: a one-byte E8M0 plane is indexed
    // by bytes and widened to its BF16 word, a halfword plane is loaded as is.
    uint32_t primary;
    if constexpr(::llaminar2::nativeQuantisedScalePlaneBytes(static_cast<uint8_t>(FMT))==1)
        primary=::llaminar2::mxfp4::powerOfTwoBF16Word(
            Reader::template load<uint8_t>(reinterpret_cast<const uint8_t*>(scales)+linear));
    else
        primary=Reader::template load<uint16_t>(scales+linear);
    uint32_t secondary=0;
    if constexpr(Traits::requires_secondary_metadata)secondary=Reader::template load<uint16_t>(mins+linear);
    operand.scale_pair=primary|(secondary<<16);
    if constexpr(Traits::needs_emin_array)operand.extended=Reader::template load<uint32_t>(emins+linear);
}
/**
 * @brief Preload one complete canonical block without converting its metadata.
 * @tparam FMT Original independently selected physical decoder.
 * @param payload Original payload plane; odd strides retain bounded byte loads.
 * @param meta Role metadata planes and representation (per-block or superblock-coded).
 * @param address Block-major address (linear index plus superblock index).
 * @return Current payload and exactly the format's required metadata bits.
 * @details Ordinary callers keep their original payload-then-metadata ordering;
 * both stages reuse the same helpers as EB4, without a second decoder/load policy.
 */
template<NativeQuantisedFormat FMT,class Reader=NativeLayoutWeightReader,int Representation=-1>
__device__ __forceinline__ BlockOperand<FMT> preloadBlock(const uint8_t* payload,
    const ::llaminar2::rocm_native_metadata::MetadataPlanes& meta,::llaminar2::rocm_native_metadata::BlockAddress address)
{
    BlockOperand<FMT> operand;
    preloadBlockPayload<FMT,Reader>(operand,payload,address.linear);
    preloadBlockMetadata<FMT,Reader,Representation>(operand,meta,address);
    return operand;
}

/**
 * @brief Run @p body with the role's metadata representation fixed at compile time.
 * @tparam FMT Physical decoder; non-codable decoders only instantiate the per-block body.
 * @param meta Role planes (wave-uniform representation).
 * @param body Generic lambda taking `template<int Representation>`.
 * @details One uniform branch per role/partition keeps each K loop carrying
 * only its own representation's registers, so per-block roles retain the
 * original loop code and register budget.
 */
template<NativeQuantisedFormat FMT,class Body>
__device__ __forceinline__ decltype(auto) withRoleRepresentation(const ::llaminar2::rocm_native_metadata::MetadataPlanes& meta,Body&& body)
{
    if constexpr(::llaminar2::nativeQuantisedSuperblockCodable(static_cast<uint8_t>(FMT)))
        if(meta.coded())return body.template operator()<1>();
    return body.template operator()<0>();
}
/**
 * @brief Keep a defined current-iteration payload and metadata in registers until consumption.
 * @tparam FMT Canonical physical decoder determining the exact payload extent.
 * @param operand Fresh current-K payload and metadata, zero for an absent expert.
 * @details A volatile register-only operand boundary prevents LLVM from
 * replacing an unused zero arm with the preceding iteration's payload. It
 * emits no instruction or memory fence. Every word is required on both arms,
 * so the loop carries only the four numerical accumulators, not old payloads.
 * Bounded copies preserve alignment safety for odd-stride physical formats.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ void retainCurrentBlock(BlockOperand<FMT>& operand)
{
    constexpr unsigned bytes=NATIVE_QUANTISEDTraits<FMT>::payload_bytes;
    #pragma unroll
    for(unsigned offset=0;offset<bytes;offset+=4)
    {
        constexpr unsigned word_bytes=4;
        const unsigned remaining=bytes-offset;
        const unsigned count=remaining<word_bytes?remaining:word_bytes;
        uint32_t word=0;
        // Both copies have compile-time extents after unrolling. No source or
        // local payload access extends beyond this format's admitted bytes.
        __builtin_memcpy(&word,operand.packed+offset,count);
        asm volatile("" : "+v"(word));
        __builtin_memcpy(operand.packed+offset,&word,count);
    }
    asm volatile("" : "+v"(operand.scale_pair));
    if constexpr(NATIVE_QUANTISEDTraits<FMT>::needs_emin_array)
        asm volatile("" : "+v"(operand.extended));
}
/**
 * @brief Native roles select original activation issue without changing arithmetic.
 * @details This internal compile-time choice changes only when immutable LDS
 * bytes enter registers. It owns no storage, source policy, arithmetic or host mode.
 */
enum class ActivationIssue { GateUpBlockPreload, DownWordsPreload };
/**
 * @brief Original K32 activation bytes held only in lane-owned registers.
 * @details No activation repack is created. Words are the original eight signed
 * byte packets, followed by the original FP32 scale and optional exact code sum.
 * Arrays have fixed constant-index consumers, including canonical IQ1_M and
 * dual-asymmetric corrections. Production resource admission must prove scalar
 * replacement; this record does not authorize private or shared allocations.
 */
struct ActivationBlockOperand {
    int32_t words[8]; ///< Original groups0..7, never decoded or reordered.
    float scale; ///< Original FP32 activation scale, copied bit-for-bit.
    int32_t sum; ///< Original signed-code sum when required, otherwise unused zero.
};
/**
 * @brief Copy the same eight original activation words into lane-owned registers.
 * @param words Current consumer's fixed groups0..7 destination.
 * @param code Immutable original CTA-local32-byte activation block,4-byte aligned.
 * @details This is the common input-load body for GU's full block and down's
 * words-only operand. It creates no repack, cross-expert cache or allocation.
 * Each caller retains its own register-only issue boundary and metadata timing.
 */
__device__ __forceinline__ void preloadActivationWords(int32_t (&words)[8],const int8_t* code)
{
    const auto* input=(const CurrentFormatLDSValue<int32_t>*)code;
    #pragma unroll
    for(unsigned group=0;group<8;++group)words[group]=input[group];
}
/**
 * @brief Preload original activation words and required metadata before any dot.
 * @tparam FMT Original physical codebook, determining whether the sum is consumed.
 * @param code Immutable CTA-local original32-byte activation block.
 * @param scales Original CTA-local FP32 scale plane. @param sums Original sum plane.
 * @param index Original K32 metadata index. @return Exact original activation values.
 * @pre The original role readiness, source policy and interval guards already
 * accept this lane. Code is4-byte aligned; index is inside the admitted source.
 * @details The register-only boundary retains all eight input words and required
 * metadata before decoding/SDOT. It is not a memory fence or a new publication.
 * Each source is loaded once and paired Q4 reuses the same returned values.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ ActivationBlockOperand preloadActivationBlock(
    const int8_t* code,const float* scales,const int32_t* sums,uint32_t index)
{
    ActivationBlockOperand result;
    preloadActivationWords(result.words,code);
    result.scale=((const CurrentFormatLDSValue<float>*)scales)[index];
    result.sum=0;
    if constexpr(NATIVE_QUANTISEDTraits<FMT>::is_asymmetric)
        result.sum=((const CurrentFormatLDSValue<int32_t>*)sums)[index];
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr(NATIVE_QUANTISEDTraits<FMT>::is_asymmetric)
        asm volatile("" : "+v"(result.words[0]),"+v"(result.words[1]),
            "+v"(result.words[2]),"+v"(result.words[3]),"+v"(result.words[4]),
            "+v"(result.words[5]),"+v"(result.words[6]),"+v"(result.words[7]),
            "+v"(result.scale),"+v"(result.sum));
    else
        asm volatile("" : "+v"(result.words[0]),"+v"(result.words[1]),
            "+v"(result.words[2]),"+v"(result.words[3]),"+v"(result.words[4]),
            "+v"(result.words[5]),"+v"(result.words[6]),"+v"(result.words[7]),
            "+v"(result.scale));
#endif
    return result;
}
/**
 * @brief Consume original groups0..7 with the existing ordered integer chains.
 * @tparam FMT Original physical decoder.
 * @param operand Original current-K payload. @param code Original32-code activation,
 * or the same byte packets already held in the preload operand.
 * @param acc_lo Receives original low-chain dot. @param acc_hi Receives original high-chain dot.
 * @details Shared by ordinary and paired consumers; the group mappings and
 * previous-dot dependencies are unchanged, and no FP32 commit occurs here.
 * @tparam Grid Original table authority by default, or an explicitly published exact lookup view.
 * @param grid Lookup bytes retained until this complete canonical consumer retires.
 */
template<NativeQuantisedFormat FMT,class Grid=NativeQuantisedGridLookup>
__device__ __forceinline__ void consumePackedIntegerDots(const BlockOperand<FMT>& operand,
    const int8_t* code,int32_t& acc_lo,int32_t& acc_hi,Grid grid={})
{
    using Traits=NATIVE_QUANTISEDTraits<FMT>;
    const auto* activation=reinterpret_cast<const int32_t*>(code);
    acc_lo=0;acc_hi=0;int32_t previous_dot=0;
    // Each raw packed weight crosses its register-only boundary after the
    // preceding SDOT. Activation words are already register-held; the original
    // independent low/high chains and every FP32 boundary remain unchanged.
    #pragma unroll
    for(unsigned group=0;group<8;++group)
    {
        const int32_t weight=moe_decode_native_quantised_group<FMT,true>(operand.packed,group,previous_dot,grid);
        const int32_t input=activation[group];
        if constexpr(Traits::is_dual_scale)
        {
            if(group<4)
                previous_dot=acc_lo=::llaminar2::rocm_integer_contract::signedDot4(input,weight,acc_lo);
            else
                previous_dot=acc_hi=::llaminar2::rocm_integer_contract::signedDot4(input,weight,acc_hi);
        }
        else previous_dot=acc_lo=::llaminar2::rocm_integer_contract::signedDot4(input,weight,acc_lo);
    }
}
/**
 * @brief Reuse the preloaded original activation block across paired Q4 chains.
 * @tparam FMT Existing physical Q4 decoder, unchanged by the caller predicate.
 * @tparam OrderedDecode Whether immutable bit extraction is kept behind the previous dot.
 * This scheduling dependency bounds standalone live operands; it is not an arithmetic node.
 * @param gate Original current-K gate payload. @param up Original up payload.
 * @param activation Original groups0..7 already preloaded into lane-owned registers.
 * @param gate_dot Receives the unchanged gate integer chain.
 * @param up_dot Receives the unchanged up integer chain.
 * @details Each group appends gate then up exactly as the former two four-word
 * packets did. Only immutable input issue moves earlier; each decoder retains
 * every original integer boundary. A full-entry caller can let independent
 * immutable bit extraction overlap while the same gate/up SDOT chains remain ordered.
 */
template<NativeQuantisedFormat FMT,bool OrderedDecode=true>
__device__ __forceinline__ void consumePairedQ4IntegerDots(const BlockOperand<FMT>& gate,
    const BlockOperand<FMT>& up,const int32_t* activation,int32_t& gate_dot,int32_t& up_dot)
{
    static_assert(FMT==NATIVE_QUANTISED_Q4_1,
        "The existing paired-role predicate admits only canonical Q4_1");
    gate_dot=0;up_dot=0;
    #pragma unroll
    for(unsigned group=0;group<8;++group)
    {
        const int32_t input=activation[group];
        const int32_t gate_weight=moe_decode_native_quantised_group<FMT,OrderedDecode>(gate.packed,group,gate_dot);
        gate_dot=::llaminar2::rocm_integer_contract::signedDot4(input,gate_weight,gate_dot);
        const int32_t up_weight=moe_decode_native_quantised_group<FMT,OrderedDecode>(up.packed,group,up_dot);
        up_dot=::llaminar2::rocm_integer_contract::signedDot4(input,up_weight,up_dot);
    }
}
/**
 * @brief Load one original activation metadata scalar only after the last dot.
 * @tparam T Exact float scale or int32 sum type.
 * @param base Original CTA-local per-expert metadata plane. @param index Original K32 block.
 * @param last_dot Completed original integer chain; unchanged by this dependency.
 * @return Original metadata bytes from one scalar source-level read.
 * @pre Base comes from this kernel's original shared arena, never global memory.
 * The original admitted K gives index<K/32<=INT_MAX/32, fitting uint32_t.
 * @details The existing block index crosses the last-dot dependency before
 * forming the AS3 address. No combined metadata address needs to stay live
 * through decoding. This adds no memory ordering or cross-lane broadcast.
 */
template<class T>
__device__ __forceinline__ T activationMetadataAfterDot(const T* base,uint32_t index,int32_t last_dot)
{
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(index) : "v"(last_dot));
#endif
    const auto* pointer=(const CurrentFormatLDSValue<T>*)base+index;
#if defined(__HIP_DEVICE_COMPILE__)
    static_assert(sizeof(pointer)==sizeof(uint32_t));
#endif
    return *pointer;
}
/**
 * @brief Delegate completed integer chains to the sole canonical FP32 commit.
 * @tparam FMT Original physical decoder.
 * @param operand Original payload/metadata bits. @param code Original activation codes.
 * @param acc_lo Original low-chain result. @param acc_hi Original high-chain result.
 * @param scale Original activation scale. @param sum Original required signed sum.
 * @param accumulator Original ascending-K32 FP32 state, updated once.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ void commitPackedIntegerDots(const BlockOperand<FMT>& operand,
    const int8_t* code,int32_t acc_lo,int32_t acc_hi,float scale,int32_t sum,float& accumulator)
{
    using Traits=NATIVE_QUANTISEDTraits<FMT>;
    const auto* activation=reinterpret_cast<const int32_t*>(code);
    uint32_t extended=0;
    if constexpr(Traits::needs_emin_array)extended=operand.extended;
    native_quantised_small_m_commit_integer_dots_bits<FMT>(accumulator,activation,acc_lo,acc_hi,
        operand.packed,static_cast<uint16_t>(operand.scale_pair),static_cast<uint16_t>(operand.scale_pair>>16),
        extended,scale,sum);
}
/**
 * @brief Consume original activation words and load metadata after the final dot.
 * @tparam FMT Original physical decoder.
 * @param operand Complete original current-K weight payload/metadata.
 * @param code Original activation bytes in LDS or their exact register-held words.
 * @param scales Original CTA-local scale plane. @param sums Original signed-sum plane.
 * @param index Original K32 metadata index. @param accumulator Original increasing-K32 sum.
 * @details Down words-preload delegates to this single body. IQ1_M and
 * dual-asymmetric corrections receive the same original register-held words
 * through the canonical commit; no correction is duplicated.
 * @tparam Grid Original table authority by default, or an explicitly published exact lookup view.
 * @param grid Lookup bytes retained until this complete canonical consumer retires.
 */
template<NativeQuantisedFormat FMT,class Grid=NativeQuantisedGridLookup>
__device__ __forceinline__ void consumePackedBlockLateMetadata(const BlockOperand<FMT>& operand,
    const int8_t* code,const float* scales,const int32_t* sums,uint32_t index,float& accumulator,Grid grid={})
{
    int32_t acc_lo,acc_hi;
    consumePackedIntegerDots<FMT>(operand,code,acc_lo,acc_hi,grid);
    const int32_t last_dot=NATIVE_QUANTISEDTraits<FMT>::is_dual_scale?acc_hi:acc_lo;
    const float scale=activationMetadataAfterDot(scales,index,last_dot);
    int32_t sum=0;
    if constexpr(NATIVE_QUANTISEDTraits<FMT>::is_asymmetric)
        sum=activationMetadataAfterDot(sums,index,last_dot);
    commitPackedIntegerDots<FMT>(operand,code,acc_lo,acc_hi,scale,sum,accumulator);
}
/**
 * @brief Consume an original block through the one canonical integer/FP32 body.
 * @tparam FMT Original physical decoder. @tparam Issue Internal native activation issue contract.
 * @param operand Original current-K payload. @param code Original32-code activation.
 * @param scales Original CTA-local scale plane. @param sums Original signed-sum plane.
 * @param index Original K32 metadata index. @param accumulator Original ascending-K32 FP32 state.
 * @details Down preloads only the current expert's eight activation words, then
 * loads its original scale/sum after dots. GU retains its existing full preload.
 * Canonical IQ1_M/dual-asymmetric corrections borrow the same words. EB4's
 * caller still loads all four weights first; no four-expert activation array
 * exists. Full shipping allocation must prove register scalar replacement and
 * that future consumers do not extend these current-expert word lifetimes.
 * @tparam Grid Original table authority by default, or an explicitly published exact lookup view.
 * @param grid Lookup bytes retained until this complete canonical consumer retires.
 */
template<NativeQuantisedFormat FMT,ActivationIssue Issue=ActivationIssue::DownWordsPreload,class Grid=NativeQuantisedGridLookup>
__device__ __forceinline__ void consumePackedBlock(const BlockOperand<FMT>& operand,
    const int8_t* code,const float* scales,const int32_t* sums,uint32_t index,float& accumulator,Grid grid={})
{
    if constexpr(Issue==ActivationIssue::GateUpBlockPreload)
    {
        const auto activation=preloadActivationBlock<FMT>(code,scales,sums,index);
        const auto* original_bytes=reinterpret_cast<const int8_t*>(activation.words);
        int32_t acc_lo,acc_hi;
        consumePackedIntegerDots<FMT>(operand,original_bytes,acc_lo,acc_hi,grid);
        commitPackedIntegerDots<FMT>(operand,original_bytes,acc_lo,acc_hi,
            activation.scale,activation.sum,accumulator);
    }
    else
    {
        static_assert(Issue==ActivationIssue::DownWordsPreload);
        int32_t words[8];
        preloadActivationWords(words,code);
#if defined(__HIP_DEVICE_COMPILE__)
        // All current-expert words precede its first dot. Unlike GU, metadata
        // remains late; no scale/sum is retained through this decoder.
        asm volatile("" : "+v"(words[0]),"+v"(words[1]),"+v"(words[2]),"+v"(words[3]),
            "+v"(words[4]),"+v"(words[5]),"+v"(words[6]),"+v"(words[7]));
#endif
        consumePackedBlockLateMetadata<FMT>(operand,
            reinterpret_cast<const int8_t*>(words),scales,sums,index,accumulator,grid);
    }
}

/**
 * @brief Traverse one original role partition without following its descriptor in the loop.
 * @tparam Reader Original generic issue by default; full entry may prove global-only weight storage.
 * @tparam FMT Original physical decoder, independent of source partition policy.
 * @tparam Issue Internal native activation issue contract; down preloads words only.
 * @tparam Binding Existing metadata reference preserving native LDS address space.
 * @param binding Phase-local original metadata, published by this batch's leader.
 * @param n Original output column. @param code Original activation row.
 * @param scale Original block scales. @param sums Original exact signed-code sums.
 * @param N Original output width. @param K Original reduction width.
 * @param part Exact original partition; empty intervals return original positive zero.
 * @return Original ascending-K32 partial, or positive zero for the leader-authenticated invariant rejection.
 * @pre Caller selects the record's exact physical decoder and original N/K;
 * the original publication leader authors numeric_validity before the same LDS join.
 * @tparam Grid Original table authority by default, or an explicitly published exact lookup view.
 * @param grid Lookup bytes retained until this complete canonical consumer retires.
 */
template<NativeQuantisedFormat FMT,ActivationIssue Issue=ActivationIssue::DownWordsPreload,
    class Reader=NativeLayoutWeightReader,bool SplitRepresentation=false,class Binding,class Grid=NativeQuantisedGridLookup>
__device__ __forceinline__ float phaseRolePartial(const Binding& binding,unsigned n,
    const int8_t* code,const float* scale,const int32_t* sums,unsigned N,unsigned K,unsigned part,Grid grid={})
{
    if constexpr ((FMT == NATIVE_QUANTISED_IQ2_S || FMT == NATIVE_QUANTISED_IQ3_S ||
                   FMT == NATIVE_QUANTISED_IQ3_XXS) && requires { grid.template useDeviceTable<FMT>(); }) {
        // Select the address space once for the whole K partition. Both paths
        // consume the same encoded weights with the same ordered integer dots.
        if (grid.template useDeviceTable<FMT>())
            return phaseRolePartial<FMT,Issue,Reader,SplitRepresentation>(binding,n,code,scale,sums,N,K,part,
                NativeQuantisedGridLookup{});
    }
    using Traits=NATIVE_QUANTISEDTraits<FMT>;
    if(waveMetadataValue(binding.numeric_validity)!=PhaseProjectionValidation::State::Valid)return 0.f;
    const unsigned first=part*waveMetadataValue(binding.partition_blocks),last=min(K/32,first+waveMetadataValue(binding.partition_blocks));
    const auto* payload=waveMetadataValue(binding.payload);
    const auto meta=phaseRoleMetadata<FMT>(binding);
    // SplitRepresentation selects the representation once per partition and
    // instantiates two loops (each carrying only its own registers); otherwise
    // one loop takes a uniform per-role branch. Each kernel family uses the
    // form that keeps it within its measured register budget.
    const auto partition=[&]<int Representation>() {
        float accumulator=0.f;
        #pragma unroll 1
        for(unsigned block=first;block<last;++block)
        {
            const auto operand=preloadBlock<FMT,Reader,Representation>(payload,meta,
                ::llaminar2::rocm_native_metadata::blockAddress(meta,block,n,N));
            consumePackedBlock<FMT,Issue>(operand,
                code+size_t(block)*32,scale,sums,block,accumulator,grid);
        }
        return accumulator;
    };
    if constexpr(SplitRepresentation)
        return withRoleRepresentation<FMT>(meta,partition);
    else
        return partition.template operator()<-1>();
}
/**
 * @brief Reuse one immutable activation block across compatible independent role partials.
 * @tparam FMT Actual physical decoder, never inferred from the other role's source policy.
 * @tparam Reader Existing original generic reader or the admitted global-only weight reader.
 * @tparam Binding Original phase record reference, retaining native LDS provenance.
 * @param gate Original gate record. @param up Original independently validated up record.
 * @param n Original output column. @param code Immutable original activation codes.
 * @param scale Original activation block scales. @param sums Original exact code sums.
 * @param N Original complete output width. @param K Original complete reduction width.
 * @param part Exact original partition coordinate, common only after the compatibility checks.
 * @param gate_acc Receives this original gate partial only when compatibility succeeds.
 * @param up_acc Receives this original up partial only when compatibility succeeds.
 * @return True after both partials are complete. False leaves outputs unchanged;
 * the caller must execute the existing independent-role consumers for that valid
 * incompatible pair, including their original zero and unsupported-format rules.
 * @details Compatibility is a reuse condition, not a new capability or source
 * policy. Both original publication leaders have already performed their own
 * fatal policy/readiness/numeric classification before the existing LDS join.
 * Only equal ready/valid physical formats and equal P/partition widths share
 * activation reads. Source scopes name one current weight operand per role;
 * actual compiler live ranges remain a mandatory resource/ISA gate. Each role
 * retains its own ascending K32 commits; no integer decoder,
 * FP commit, partition fold, public store or publication join is duplicated here.
 * @pre All active peers borrow the same immutable role pair. Codes/scales/sums
 * remain read-only until both roles finish, as in the existing GU phase.
 * gate_acc and up_acc are the distinct original role accumulators.
 */
template<NativeQuantisedFormat FMT,class Reader=NativeLayoutWeightReader,class Binding>
__device__ __forceinline__ bool pairedIndependentRolePartial(const Binding& gate,const Binding& up,
    unsigned n,const int8_t* code,const float* scale,const int32_t* sums,unsigned N,unsigned K,unsigned part,
    float& gate_acc,float& up_acc)
{
    if(n>=N||!waveMetadataValue(gate.ready)||!waveMetadataValue(up.ready)||
        waveMetadataValue(gate.numeric_validity)!=PhaseProjectionValidation::State::Valid||
        waveMetadataValue(up.numeric_validity)!=PhaseProjectionValidation::State::Valid||
        waveMetadataValue(gate.codebook)!=FMT||waveMetadataValue(up.codebook)!=FMT)return false;
    const unsigned parts=waveMetadataValue(gate.parts),width=waveMetadataValue(gate.partition_blocks);
    if(part>=parts||parts!=waveMetadataValue(up.parts)||width!=waveMetadataValue(up.partition_blocks))return false;
    const auto* gp=waveMetadataValue(gate.payload);
    const auto* upayload=waveMetadataValue(up.payload);
    const auto gate_meta=phaseRoleMetadata<FMT>(gate),up_meta=phaseRoleMetadata<FMT>(up);
    const unsigned first=part*width,last=min(K/32,first+width);
    gate_acc=0.f;up_acc=0.f;
    withRoleRepresentation<FMT>(gate_meta,[&]<int GateRepresentation>() {
    withRoleRepresentation<FMT>(up_meta,[&]<int UpRepresentation>() {
#pragma unroll 1
    for(unsigned block=first;block<last;++block) {
        ActivationBlockOperand activation;
        {
            const auto operand=preloadBlock<FMT,Reader,GateRepresentation>(gp,gate_meta,
                ::llaminar2::rocm_native_metadata::blockAddress(gate_meta,block,n,N));
            activation=preloadActivationBlock<FMT>(code+std::size_t(block)*32,scale,sums,block);
            const auto* original=reinterpret_cast<const int8_t*>(activation.words);
            int32_t low,high;
            consumePackedIntegerDots<FMT>(operand,original,low,high);
            commitPackedIntegerDots<FMT>(operand,original,low,high,activation.scale,activation.sum,gate_acc);
        }
        {
            const auto operand=preloadBlock<FMT,Reader,UpRepresentation>(upayload,up_meta,
                ::llaminar2::rocm_native_metadata::blockAddress(up_meta,block,n,N));
            const auto* original=reinterpret_cast<const int8_t*>(activation.words);
            int32_t low,high;
            consumePackedIntegerDots<FMT>(operand,original,low,high);
            commitPackedIntegerDots<FMT>(operand,original,low,high,activation.scale,activation.sum,up_acc);
        }
    }
    });
    });
    return true;
}
/**
 * @brief Q4 strategy: load gate/up together using each role's scalar original planes.
 * @tparam FMT Original paired Q4 physical decoder selected by the caller.
 * @tparam Binding Existing metadata reference type, retaining its native LDS address space.
 * @param gate Existing published gate metadata. @param up Existing published up metadata.
 * @param n Lane-owned original output column. @param code Original activation row.
 * @param scale Original activation scales. @param sums Original signed code sums.
 * @param N Original output width. @param K Original reduction width.
 * @param part Lane-owned original partition.
 * @param gate_acc Receives original gate ascending-K32 partial.
 * @param up_acc Receives original up ascending-K32 partial.
 * @pre One active wave belongs wholly to the same original gate/up expert and output tile.
 * References borrow the existing shared records without copying either one.
 * The caller retains its original paired predicate; no new shape rejection or
 * role reordering is introduced here.
 */
template<NativeQuantisedFormat FMT,class Binding>
__device__ __forceinline__ void pairedRolePartial(const Binding& gate,const Binding& up,
    unsigned n,const int8_t* code,const float* scale,const int32_t* sums,unsigned N,unsigned K,unsigned part,
    float& gate_acc,float& up_acc)
{
    using Traits=NATIVE_QUANTISEDTraits<FMT>;
    const auto* gate_payload=waveMetadataValue(gate.payload);
    const auto* up_payload=waveMetadataValue(up.payload);
    const auto gate_meta=phaseRoleMetadata<FMT>(gate),up_meta=phaseRoleMetadata<FMT>(up);
    gate_acc=0.f;up_acc=0.f;
    // Publication computed this exact original ceil(K32/P) once, including
    // P1 and empty tails. Equal-P pairing changes neither role's interval.
    const unsigned blocks=K/32,width=waveMetadataValue(gate.partition_blocks);
    const unsigned first=part*width,last=min(blocks,first+width);
    withRoleRepresentation<FMT>(gate_meta,[&]<int GateRepresentation>() {
    withRoleRepresentation<FMT>(up_meta,[&]<int UpRepresentation>() {
    #pragma unroll 1
    for(unsigned block=first;block<last;++block)
    {
        const auto g=preloadBlock<FMT,NativeLayoutWeightReader,GateRepresentation>(gate_payload,gate_meta,
                ::llaminar2::rocm_native_metadata::blockAddress(gate_meta,block,n,N)),
            u=preloadBlock<FMT,NativeLayoutWeightReader,UpRepresentation>(up_payload,up_meta,
                ::llaminar2::rocm_native_metadata::blockAddress(up_meta,block,n,N));
        const auto activation=preloadActivationBlock<FMT>(code+size_t(block)*32,scale,sums,block);
        int32_t g_lo,u_lo;
        consumePairedQ4IntegerDots<FMT>(g,u,activation.words,g_lo,u_lo);
        // Gate then up retain the original FP32 commit order and share exactly
        // the same immutable activation words, scale and required code sum.
        const auto* original_bytes=reinterpret_cast<const int8_t*>(activation.words);
        commitPackedIntegerDots<FMT>(g,original_bytes,g_lo,0,activation.scale,activation.sum,gate_acc);
        commitPackedIntegerDots<FMT>(u,original_bytes,u_lo,0,activation.scale,activation.sum,up_acc);
    }
    });
    });
}
/**
 * @brief Return canonical EB4 partials in scalar registers instead of an LDS plane.
 * @details Four experts retain exact current-K payload-first preloads and their
 * original independent accumulators. The caller retires this format before any
 * other codebook. Eight-bit formats use the canonical single-role helper directly;
 * absent tails never read operands or add a numerical commit.
 * @tparam FMT Original physical decoder; the existing role mask owns its inclusion.
 * @param bindings Four original immutable phase records. @param n Original output column.
 * @param code Original packet activation prefix. @param scale Original block scales.
 * @param sums Original signed code sums. @param N Original output width.
 * @param K Original reduction width. @param part Exact original partition.
 * @return Identical completed per-expert partials, with no added FP32 operations.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ float4 expertBatchPartial(const PhaseProjection* bindings,unsigned n,const int8_t* code,
    const float* scale,const int32_t* sums,unsigned N,unsigned K,unsigned part)
{
    constexpr unsigned EB=4;
    static_assert(!NATIVE_QUANTISEDTraits<FMT>::is_8bit,
        "Eight-bit source formats retire each expert before the next operand");
    // The caller selected this physical format from the same immutable packet.
    // Keep LDS field addresses inside this arm; no prior-format result exists.
    const auto* current_bindings=currentFormatAddress(
        (CurrentFormatLDSValue<const PhaseProjection>*)bindings);
    float accumulator[EB]{0.f,0.f,0.f,0.f};
    const unsigned blocks=K/32;
    unsigned first[EB],width[EB];
    unsigned union_first=blocks,union_last=0;
    #pragma unroll
    for(unsigned expert=0;expert<EB;++expert)
    {
        const auto& binding=current_bindings[expert];
        const bool participating=waveMetadataValue(binding.ready)&&part<waveMetadataValue(binding.parts)&&n<N&&waveMetadataValue(binding.codebook)==FMT;
        width[expert]=waveMetadataValue(binding.partition_blocks);
        first[expert]=nativeLayoutIntervalFirst(blocks,width[expert],part,participating);
        const unsigned end=nativeLayoutIntervalEnd(blocks,first[expert],width[expert]);
        if(first[expert]<end)
        {
            union_first=min(union_first,first[expert]);
            union_last=max(union_last,end);
        }
    }
    // Visit only the exact envelope of this lane's eligible original intervals.
    // Existing per-expert owns guards retain gaps and ascending K32 commits;
    // an empty union executes no source reads or numerical work.
    #pragma unroll 1
    for(unsigned block=union_first;block<union_last;++block)
    {
        BlockOperand<FMT> operand[EB]{};bool owns[EB];
        // Mirror the source EB4 load-all/consume-all structure without holding
        // four descriptor pointer triples across K. Issue every owned payload
        // before metadata packing can impose a per-expert completion dependency.
        #pragma unroll
        for(unsigned expert=0;expert<EB;++expert)
        {
            const auto& binding=current_bindings[expert];
            // Width is wave-uniform; the blocks sentinel rejects inactive
            // lanes without retaining four lane-dependent end bounds.
            owns[expert]=nativeLayoutIntervalContains(block,first[expert],width[expert]);
            if(owns[expert])
                preloadBlockPayload<FMT>(operand[expert],waveMetadataValue(binding.payload),size_t(block)*N+n);
        }
        #pragma unroll
        for(unsigned expert=0;expert<EB;++expert)
        {
            if(owns[expert])
            {
                const auto& binding=current_bindings[expert];
                const auto meta=phaseRoleMetadata<FMT>(binding);
                preloadBlockMetadata<FMT>(operand[expert],meta,::llaminar2::rocm_native_metadata::blockAddress(meta,block,n,N));
            }
        }
        // Require all four complete current-K operands before consuming any expert. The
        // absent-expert zeros are register-lifetime witnesses only: they never
        // cause source reads, decoding, or an extra floating-point commit.
        #pragma unroll
        for(unsigned expert=0;expert<EB;++expert)retainCurrentBlock(operand[expert]);
        #pragma unroll
        for(unsigned expert=0;expert<EB;++expert)
            if(owns[expert])
                consumePackedBlock<FMT>(operand[expert],code+size_t(expert)*K+size_t(block)*32,
                    scale+size_t(expert)*blocks,sums+size_t(expert)*blocks,block,accumulator[expert]);
    }
    // Nonparticipants retain their initialized positive zero: their blocks
    // sentinel rejects every cursor before any load or numerical commit. Return
    // the original accumulators directly without rescanning immutable metadata.
    return float4{accumulator[0],accumulator[1],accumulator[2],accumulator[3]};
}
}
