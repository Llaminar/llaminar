/**
 * @file ROCmGroupedFFNFormatSelection.h
 * @brief Cold admission for counted verifier leaves using the existing native decoder registry.
 * Physical masks follow expert movements. Original source policies separately
 * preserve the partition/fold order observed by serial decode.
 */
#pragma once

#include "ROCmGroupedFFNFrontier.h"
#include "ROCmResidentMoEFFN.h"
#include "kernels/common/NativeQuantisedDescriptorValidation.h"
#include <optional>

namespace llaminar2::rocm::grouped_ffn {

/**
 * @brief Select a measured verifier body only for original sources accepted by its numeric leaves.
 * @param columns Participant-local routed columns.
 * @param gateUp Reachable gate and up execution formats and arithmetic policies.
 * @param down Reachable down execution formats and arithmetic policies.
 * @param original Original gate, up and down GGUF identities sealed by the fragment owner.
 * @return The eligible decoder, or an empty option for a different graph layout.
 */
inline std::optional<RoutedDecoder> measuredDecoder(int columns,
    NativeQuantisedExecutionFormatEnvelope gateUp,NativeQuantisedExecutionFormatEnvelope down,
    const std::array<ExpertWeightFormat,3>& original)
{
    if(columns!=routedWidth && columns!=routedColumns(RoutedGeometry::F160))return std::nullopt;
    for(auto decoder:{RoutedDecoder::Q4K_Q51,RoutedDecoder::Q4K_Q80,RoutedDecoder::Q5K_Q80}) {
        if(columns!=routedColumns(RoutedGeometry::F160) && decoder!=RoutedDecoder::Q4K_Q51)continue;
        const auto gateMask=nativeQuantisedCodebookMaskBit(routedGateCodebook(decoder));
        const auto downMask=nativeQuantisedCodebookMaskBit(routedDownCodebook(decoder));
        if(gateUp.execution_codebook_mask!=gateMask || gateUp.policy_codebook_mask!=gateMask ||
           down.execution_codebook_mask!=downMask || down.policy_codebook_mask!=downMask)continue;
        // Q5_1 and Q5_K share a decoder. Only the superblock pair satisfies
        // the measured gate's source contract. The Q8 down leaf takes source 19.
        const NativeQuantisedSourceIdentity q5k{native_quantised_formats::Q5_K.codebook_id,true,true};
        if(decoder==RoutedDecoder::Q5K_Q80 &&
           (original[0].native_quantised!=q5k || original[1].native_quantised!=q5k))continue;
        const auto& source=original[2].native_quantised;
        if(decoder!=RoutedDecoder::Q4K_Q51 &&
           (!source.present || source.codebook_id!=native_quantised_formats::Q8_0.codebook_id))continue;
        return decoder;
    }
    return std::nullopt;
}

/**
 * @brief Restrict the measured affine-sum layout to its original authenticated GGUF sources.
 * @details Q4_1 and Q4_K share execution codebook 5. They remain distinct
 * source layouts and may have different cost even when the native decoder matches.
 * @param geometry Original participant-local routed width.
 * @param decoder Exact two-leaf specialization.
 * @param routed Source declarations for gate, up and down, independent of current residency.
 * @param shared Immutable original shared gate, up and down descriptors.
 * @return Whether the source, shape and decoder match a native measured sum program.
 */
inline bool hasMeasuredSumLayout(RoutedGeometry geometry,RoutedDecoder decoder,
    const std::array<ExpertWeightFormat,3>& routed,const std::array<DeviceNativeQuantisedMatrixDesc,3>& shared)
{
    if(geometry!=RoutedGeometry::F160 || !supportsExactK32Sums(decoder))return false;
    for(const auto& matrix:shared)
        if(!matrix.source_identity_present || matrix.source_codebook_id!=19 || matrix.source_is_superblock)
            return false;
    for(std::size_t i=0;i<routed.size();++i) {
        const auto& source=routed[i].native_quantised;
        if(!routed[i].isNativeQuantised() || !source.present || source.is_superblock!=(i<2) ||
           source.codebook_id!=(i<2?routedGateCodebook(decoder):routedDownCodebook(decoder)))return false;
    }
    return true;
}

/**
 * @brief Seal the routed table declarations and the three independent immutable shared projections.
 * @param routedWidth Participant-local expert intermediate columns.
 * @param sharedWidth Participant-local shared intermediate columns.
 * @param gateUp All reachable physical gate/up descriptors and their original reduction policies.
 * @param down All reachable physical down descriptors and their original reduction policies.
 * @param routedSources Exact original gate, up and down arithmetic identities from the bound fragment family.
 * @param shared Original prepared shared gate, up and down descriptors.
 * @return Smallest native decoder family together with each role's original envelope.
 * @throws std::invalid_argument For an incomplete projection, width or source identity.
 */
inline SourceFormatDeclaration declareSourceFormats(int routedWidth,int sharedWidth,
    NativeQuantisedExecutionFormatEnvelope gateUp,NativeQuantisedExecutionFormatEnvelope down,
    const std::array<NativeQuantisedSourceIdentity,3>& routedSources,
    const std::array<DeviceNativeQuantisedMatrixDesc,3>& shared)
{
    if(routedWidth<=0 || routedWidth%32 || sharedWidth<=0 || sharedWidth%32)
        throw std::invalid_argument("Grouped FFN requires complete K32 intermediate widths");
    if(!gateUp.valid() || !down.valid())
        throw std::invalid_argument("Grouped FFN requires both routed source envelopes");
    // Validate the declared physical union separately from the exact per-role source.
    // The compact-only union and the CPU-migratable union can differ for the same GGUF.
    std::array<NativeQuantisedExecutionFormatEnvelope,2> compact{},movable{};
    for(std::size_t role=0;role<routedSources.size();++role) {
        const auto source=routedSources[role];
        const auto decoder=canonicalDeviceVnniCodebookId(source);
        const unsigned projection=role==2?1:0;
        if(!source.present || !addNativeQuantisedExecutionFormat(compact[projection],decoder,source,false) ||
           !addNativeQuantisedExecutionFormat(movable[projection],decoder,source,true))
            throw std::invalid_argument("Grouped FFN requires an authenticated original source for each routed projection");
    }
    const std::array declared{gateUp,down};
    for(std::size_t i=0;i<declared.size();++i) {
        const auto mask=declared[i].execution_codebook_mask;
        if(declared[i].policy_codebook_mask!=compact[i].policy_codebook_mask ||
           (mask&compact[i].execution_codebook_mask)!=compact[i].execution_codebook_mask ||
           (mask&~movable[i].execution_codebook_mask))
            throw std::invalid_argument("Grouped FFN routed envelope disagrees with its original projections");
    }
    WeightFormats formats{gateUp,down,{}};
    for(std::size_t role=0;role<shared.size();++role) {
        const auto& matrix=shared[role];
        const bool isDown=role==2;
        if(matrix.n!=(isDown?hidden:sharedWidth) || matrix.k!=(isDown?sharedWidth:hidden) ||
           !NativeQuantisedDescriptorValidation::publishable(matrix) || !matrix.source_identity_present ||
           !addNativeQuantisedExecutionFormat(formats.shared[role],matrix.codebook_id,
               {matrix.source_codebook_id,bool(matrix.source_is_superblock),true},false))
            throw std::invalid_argument("Grouped FFN shared projection has invalid planes, shape or source policy");
    }
    return {routedWidth,sharedWidth,formats,ResidentMoEFFNDecoderProgram::fromExecutionMask(formats.executionMask()).groups(),routedSources};
}

} // namespace llaminar2::rocm::grouped_ffn
