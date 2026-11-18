/**
 * @file ROCmGroupedFFNGenericNumerics.hip.inc
 * @brief Ordered NativeQuantised K32 leaves with one decoded payload per group of users.
 *
 * The format switch selects physical packed bytes. Captured serial partition
 * policy is selected independently from source provenance for the same role.
 * Included under the existing grouped HIP translation unit, whose codebook
 * tables and rounding helpers own all format-specific arithmetic.
 */
namespace generic {

/** @brief Original quantized activation and optional exact full-block integer sum. */
struct ProjectionRow {
    const int8_t* codes{};
    const float* scales{};
    const int16_t* sum32{};
};

/** @brief Independent ordered accumulators sharing the same immutable weight decode. */
template<int MaxUsers> struct ProjectionUsers {
    ProjectionRow rows[MaxUsers]{};
    int live{};
};

/** @brief Publish the exact original projection result of each admitted user. */
template<int MaxUsers> struct ProjectionResult {
    float value[MaxUsers]{};
};

static constexpr bool validFormats(const NativeQuantisedExecutionFormatEnvelope& formats)
{
    return formats.valid() && !(formats.execution_codebook_mask & ~ResidentMoEFFNDecoderProgram::all_decoders);
}

/**
 * @brief Compute exact local split-K partials and fold in original ascending order.
 * @tparam PartitionWorkers Number of existing wave subgroups contributing each output column.
 * @details A decoded K32 group is reused across all live users of the same
 * expert. The single-partition policy returns the leaf directly as in serial M1.
 * For Q2_K and IQ1_M the shared original accumulator retains its segmented sums.
 */
template<NativeQuantisedFormatPrefill Format,int MaxUsers,int PartitionWorkers,bool OneBlockPerPartition>
__device__ __forceinline__ ProjectionResult<MaxUsers> orderedProjectionFormat(
    const DeviceNativeQuantisedMatrixDesc& desc,int column,int width,
    const ProjectionUsers<MaxUsers>& users,int parts,int lane)
{
    static_assert(PartitionWorkers==1 || PartitionWorkers==4);
    using Traits=PrefillNATIVE_QUANTISEDTraits<Format>;
    const int blocks=width/32;
    if(!prefill_native_quantised_desc_shape_ok<Format>(desc,desc.n,width))
        __builtin_trap();
    const int worker=lane/(64/PartitionWorkers);
    // Only the original resolved policy selects the one-block program.
    // Arithmetic still starts each partition at +0, then appends its result.
    const int stride=OneBlockPerPartition?1:(blocks+parts-1)/parts;
    ProjectionResult<MaxUsers> total{};
#pragma unroll 1
    for(int group=0;group<parts;group+=PartitionWorkers) {
        const int partition=group+worker;
        ProjectionResult<MaxUsers> partial{};
        if(partition<parts) {
            const int begin=partition*stride;
            const int end=OneBlockPerPartition?begin+1:min(begin+stride,blocks);
            for(int b=begin;b<end;++b) {
                const size_t index=size_t(b)*desc.n+column;
                const auto* payload=desc.payload+index*Traits::payload_bytes;
                int32_t decoded[8];
                moe_prefill_decode_groups<Format>(payload,decoded);
#pragma unroll
                for(int user=0;user<MaxUsers;++user) {
                    if(user>=users.live)continue;
                    const auto& row=users.rows[user];
                    const auto* codes=reinterpret_cast<const int32_t*>(row.codes+b*32);
                    const auto scale=row.scales[b];
                    if constexpr(Traits::is_asymmetric && !Traits::is_dual_scale) {
                        if(row.sum32) {
                            partial.value[user]=moe_prefill_accumulate_decoded_block<Format,true>(
                                desc,index,payload,decoded,codes,scale,partial.value[user],row.sum32[b]);
                            continue;
                        }
                    }
                    partial.value[user]=moe_prefill_accumulate_decoded_block<Format>(
                        desc,index,payload,decoded,codes,scale,partial.value[user]);
                }
            }
        }
#pragma unroll
        for(int user=0;user<MaxUsers;++user) {
            if(user>=users.live)continue;
            if constexpr(PartitionWorkers==1) {
                if(parts==1)total.value[user]=partial.value[user];
                else total.value[user]=rocm_native_quantised_accumulate_rn(total.value[user],partial.value[user]);
            } else {
                const int columnLane=lane%16;
#pragma unroll
                for(int peer=0;peer<PartitionWorkers;++peer) {
                    if(group+peer<parts) {
                        const auto value=__shfl(partial.value[user],columnLane+16*peer,64);
                        if(parts==1)total.value[user]=value;
                        else total.value[user]=rocm_native_quantised_accumulate_rn(total.value[user],value);
                    }
                }
            }
        }
    }
    return total;
}

/**
 * @brief Resolve the physical decoder independently for each projection role.
 * @param parts Original source policy resolved before capture by the existing serial owner.
 * @param formats Capture-stable execution and source policy masks for this role.
 */
template<int MaxUsers,int PartitionWorkers,unsigned Groups=15,bool OneBlockPerPartition=false>
__device__ __forceinline__ ProjectionResult<MaxUsers> orderedProjection(
    const DeviceNativeQuantisedMatrixDesc& desc,int column,int columns,int width,
    const ProjectionUsers<MaxUsers>& users,int parts,int lane,
    const NativeQuantisedExecutionFormatEnvelope& formats)
{
    if(desc.n!=columns || column<0 || column>=columns || users.live<1 || users.live>MaxUsers ||
        parts<1 || parts>width/32 || (OneBlockPerPartition && parts!=width/32) ||
        !(formats.execution_codebook_mask & nativeQuantisedCodebookMaskBit(desc.codebook_id)) ||
        !(formats.policy_codebook_mask & nativeQuantisedCodebookMaskBit(canonicalDeviceVnniCodebookId(desc.arithmeticPolicyCodebookId()))))
        __builtin_trap();
#define LLAMINAR_ORDERED_FORMAT(ID) case ID: \
    if constexpr(ResidentMoEFFNDecoderProgram::familyMask<Groups>() & nativeQuantisedCodebookMaskBit(ID)) \
        return orderedProjectionFormat<ID,MaxUsers,PartitionWorkers,OneBlockPerPartition>(desc,column,width,users,parts,lane); \
    else __builtin_trap()
    switch(static_cast<NativeQuantisedFormatPrefill>(desc.codebook_id)) {
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_Q4_0);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_IQ4_NL);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_IQ4_XS_SOURCE);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_Q4_1);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_Q5_0);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_Q5_1);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_Q6_K);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_Q3_K);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_Q2_K);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_IQ3_S);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_IQ3_XXS);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_IQ2_S);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_IQ2_XS);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_IQ2_XXS);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_IQ1_S);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_IQ1_M);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_Q8_0);
        LLAMINAR_ORDERED_FORMAT(NATIVE_QUANTISED_PF_EXPANDED_INT8_MIN);
        default: __builtin_trap();
    }
#undef LLAMINAR_ORDERED_FORMAT
}
} // namespace generic
