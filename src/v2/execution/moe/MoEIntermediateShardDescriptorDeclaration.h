/**
 * @file MoEIntermediateShardDescriptorDeclaration.h
 * @brief Closed source-format authority for capture-stable local-F descriptor tables.
 *
 * An empty original GPU target owns source geometry and arithmetic policy but
 * no prepared matrix. This declaration borrows its SAME immutable family and
 * uses the canonical native-format policy to admit future runtime arrivals.
 * It contains no engine, descriptor pointer, live expert selection or epoch.
 */
#pragma once
#include "DeviceMoEIntermediateShardBinding.h"
#include "ExpertWeightFormat.h"
#include "backends/DeviceId.h"
#include "kernels/common/DeviceNativeQuantisedMatrixDesc.h"
#include "kernels/common/NativeQuantisedDescriptorValidation.h"
#include "kernels/common/DeviceMoEFloatingMatrixDesc.h"
#include <array>
#include <memory>

namespace llaminar2
{
    class MoEOverlayFragmentFamily;

    /** @brief Immutable original-family proof, never permission for a static/whole blank table. */
    class MoEIntermediateShardDescriptorDeclaration final
    {
    public:
        /** @brief Absent original-family authority, used by existing ordinary table APIs. */
        MoEIntermediateShardDescriptorDeclaration() noexcept = default;
        /**
         * @brief Authenticate original source formats without inspecting or preparing an expert.
         * @param family SAME closed source/preparation/member family retained by the runtime.
         * @return Exact source-preserving precision and reachable native decoder envelopes.
         * @throws std::invalid_argument For absent/foreign/non-GPU or mixed arithmetic families.
         */
        static MoEIntermediateShardDescriptorDeclaration fromOriginalFamily(
            std::shared_ptr<const MoEOverlayFragmentFamily> family);
        /** @return Whether canonical construction admitted this source declaration. */
        bool valid() const noexcept { return bool(family_); }
        /** @return The exact original family whose ownership this declaration retains. */
        const MoEOverlayFragmentFamily* originalFamily() const noexcept { return family_.get(); }
        /** @return Exact original member/source geometry shared with runtime placement validation. */
        const DeviceMoEIntermediateShardBinding& originalBinding() const noexcept { return original_; }
        /** @return Fixed source-preserving floating precision, or the native quantized family. */
        DeviceMoEWeightFormat weightFormat() const noexcept { return format_; }
        /** @return Gate/up physical decoders and source arithmetic reachable after movement. */
        NativeQuantisedExecutionFormatEnvelope nativeGateUpEnvelope() const noexcept { return gate_up_; }
        /** @return Down physical decoders and source arithmetic reachable after movement. */
        NativeQuantisedExecutionFormatEnvelope nativeDownEnvelope() const noexcept { return down_; }
        /** @return Exact original semantic gate/up/down source formats, independent of residency. */
        const std::array<ExpertWeightFormat,3>& sourceFormats() const noexcept { return sources_; }
        /** @param device Exact original physical backend/device. @param experts Original E extent.
         * @param model_columns Original H. @param local_intermediate This member's original F interval width.
         * @return Whether cold upload matches this source's exact device and local geometry. */
        bool validFor(DeviceId device,int experts,int model_columns,int local_intermediate) const noexcept;
        /** @param other Another original source declaration or the absent ordinary-table arm.
         * @return Complete source/family/member and immutable policy equality for table-cache identity. */
        bool sameIdentity(const MoEIntermediateShardDescriptorDeclaration& other) const noexcept;
        /** @param gate Actual prepared gate. @param up Actual prepared up.
         * @return Whether both nonblank projections retain their original source identities and geometry. */
        bool matchesNativeGateUp(const DeviceNativeQuantisedMatrixDesc& gate,
            const DeviceNativeQuantisedMatrixDesc& up) const noexcept;
        /** @param down Actual prepared down projection.
         * @return Whether this nonblank projection retains its original source identity and geometry. */
        bool matchesNativeDown(const DeviceNativeQuantisedMatrixDesc& down) const noexcept;
        /** @brief Seal ready gate/up publication against original source and complete physical planes.
         * @param gate SAME prepared gate. @param up SAME prepared up.
         * @return Whether the existing bank owner may expose this complete native role pair. */
        bool matchesPublishedNativeGateUp(const DeviceNativeQuantisedMatrixDesc& gate,
            const DeviceNativeQuantisedMatrixDesc& up) const noexcept
        {
            return matchesNativeGateUp(gate,up) &&
                NativeQuantisedDescriptorValidation::publishable(gate) &&
                NativeQuantisedDescriptorValidation::publishable(up);
        }
        /** @brief Seal ready down publication without changing direct-descriptor source checks.
         * @param down SAME prepared down. @return Whether original source and physical geometry are complete. */
        bool matchesPublishedNativeDown(const DeviceNativeQuantisedMatrixDesc& down) const noexcept
        {
            return matchesNativeDown(down) && NativeQuantisedDescriptorValidation::publishable(down);
        }

        /** @param gate Actual prepared gate. @param up Actual prepared up.
         * @return Whether both nonblank projections retain the declared floating family's exact shape. */
        bool matchesFloatingGateUp(const DeviceMoEFloatingMatrixDesc& gate,
            const DeviceMoEFloatingMatrixDesc& up) const noexcept;
        /** @param down Actual prepared down projection.
         * @return Whether this nonblank projection retains the declared floating family's exact shape. */
        bool matchesFloatingDown(const DeviceMoEFloatingMatrixDesc& down) const noexcept;
    private:
        /** @param role Semantic gate/up/down index, checked before indexing. @param descriptor Actual prepared matrix.
         * @return Whether the real native role retains source identity/shape without widening policy. */
        bool matchesNative(std::size_t role,const DeviceNativeQuantisedMatrixDesc& descriptor) const noexcept;
        /** @param role Semantic gate/up/down index, checked before indexing. @param descriptor Actual prepared matrix.
         * @return Whether the real floating role retains original geometry within the declared precision family. */
        bool matchesFloating(std::size_t role,const DeviceMoEFloatingMatrixDesc& descriptor) const noexcept;
        std::shared_ptr<const MoEOverlayFragmentFamily> family_;
        DeviceMoEIntermediateShardBinding original_;
        DeviceId device_=DeviceId::invalid();
        int experts_=0;
        std::array<ExpertWeightFormat,3> sources_{};
        NativeQuantisedExecutionFormatEnvelope gate_up_,down_;
        DeviceMoEWeightFormat format_=DeviceMoEWeightFormat::NativeQuantised;
    };

inline bool MoEIntermediateShardDescriptorDeclaration::validFor(DeviceId device,int experts,int h,int f) const noexcept
{
    return valid() && device_==device && experts_==experts && h>0 && f>0 &&
        original_.modelColumns()==static_cast<std::uint32_t>(h) &&
        original_.localIntermediateColumns()==static_cast<std::uint32_t>(f);
}
inline bool MoEIntermediateShardDescriptorDeclaration::sameIdentity(
    const MoEIntermediateShardDescriptorDeclaration& other) const noexcept
{
    return family_==other.family_ && !family_.owner_before(other.family_) && !other.family_.owner_before(family_) &&
        original_.sameIdentity(other.original_) && device_==other.device_ && experts_==other.experts_ &&
        sources_==other.sources_ && format_==other.format_;
}
inline bool MoEIntermediateShardDescriptorDeclaration::matchesNative(std::size_t role,
    const DeviceNativeQuantisedMatrixDesc& descriptor) const noexcept
{
    if(!valid() || format_!=DeviceMoEWeightFormat::NativeQuantised || role>=sources_.size() || !descriptor.valid())return false;
    const auto& source=sources_[role].native_quantised;
    if(descriptor.source_identity_present!=1u || descriptor.source_codebook_id!=source.codebook_id ||
       descriptor.source_is_superblock!=static_cast<std::uint8_t>(source.is_superblock) ||
       descriptor.n!=int(role==2?original_.modelColumns():original_.localIntermediateColumns()) ||
       descriptor.k!=int(role==2?original_.localIntermediateColumns():original_.modelColumns()))return false;
    const auto envelope=role==2?down_:gate_up_;
    if(!(envelope.execution_codebook_mask&nativeQuantisedCodebookMaskBit(descriptor.codebook_id)))return false;
    NativeQuantisedExecutionFormatEnvelope checked;
    return addNativeQuantisedExecutionFormat(checked,descriptor.codebook_id,source,true);
}
inline bool MoEIntermediateShardDescriptorDeclaration::matchesNativeGateUp(
    const DeviceNativeQuantisedMatrixDesc& gate,const DeviceNativeQuantisedMatrixDesc& up) const noexcept
{ return matchesNative(0,gate) && matchesNative(1,up); }
inline bool MoEIntermediateShardDescriptorDeclaration::matchesNativeDown(const DeviceNativeQuantisedMatrixDesc& down) const noexcept
{ return matchesNative(2,down); }
inline bool MoEIntermediateShardDescriptorDeclaration::matchesFloating(std::size_t role,
    const DeviceMoEFloatingMatrixDesc& descriptor) const noexcept
{
    return valid() && deviceMoEWeightFormatIsFloating(format_) && role<sources_.size() && descriptor.valid() &&
        descriptor.n==int(role==2?original_.modelColumns():original_.localIntermediateColumns()) &&
        descriptor.k==int(role==2?original_.localIntermediateColumns():original_.modelColumns());
}
inline bool MoEIntermediateShardDescriptorDeclaration::matchesFloatingGateUp(
    const DeviceMoEFloatingMatrixDesc& gate,const DeviceMoEFloatingMatrixDesc& up) const noexcept
{ return matchesFloating(0,gate) && matchesFloating(1,up); }
inline bool MoEIntermediateShardDescriptorDeclaration::matchesFloatingDown(const DeviceMoEFloatingMatrixDesc& down) const noexcept
{ return matchesFloating(2,down); }

}
