/**
 * @file MoEOverlayFixedDownProjectionBank.h
 * @brief Model-lifetime down slices that never participate in expert movement.
 *
 * A projection-distributed participant computes its own output columns for
 * every routed expert, including experts whose gate/up owner is elsewhere.
 * This immutable bank retains those prepared engines independently of the
 * placement RCU. It has no epoch, resident mask, arrival, or retirement API.
 * Captured graph bindings retain it for their full lifetime; ordinary model
 * teardown releases it after graph readers have retired.
 */
#pragma once

#include "MoEExpertProjectionOwnership.h"
#include "MoEExpertOwnerMap.h"
#include "kernels/common/DeviceNativeVNNIMatrixDesc.h"

#include <memory>
#include <variant>
#include <vector>

namespace llaminar2
{
    class ExpertGemmRegistry;
    class ITensorGemm;

    /** @brief Authenticated complete expert inventory for one fixed down interval. */
    class MoEOverlayFixedDownProjectionBank final
    {
    public:
        /** @brief Prepared native views in global expert-ID order, with source provenance. */
        struct NativeDescriptorTable
        {
            std::vector<DeviceNativeVNNIMatrixDesc> experts;
        };

        /** @brief One exact floating precision shared by every fixed down slice. */
        struct FloatingDescriptorTable
        {
            DeviceMoEWeightFormat format;
            std::vector<DeviceMoEFloatingMatrixDesc> experts;
        };

        /** @brief A complete validated table, never parallel optional descriptor families. */
        using DescriptorTable = std::variant<NativeDescriptorTable, FloatingDescriptorTable>;

        /**
         * @brief Bind exact prepared slices, without loading, packing, or allocating weights.
         * @param registry Model-owned preparation authority.
         * @param participant Exact domain/device/rank/slice coordinate.
         * @param layer Model layer named by every lookup.
         * @param ownership Source-authenticated full-K output partition.
         * @return Immutable complete bank; even a zero-gate/up owner needs every down slice.
         * @throws std::invalid_argument For inconsistent or nonpartitioned geometry.
         * @throws std::runtime_error If any fixed slice is missing.
         */
        static std::shared_ptr<const MoEOverlayFixedDownProjectionBank> resolve(
            const ExpertGemmRegistry &registry, const MoEExpertOwnerParticipant &participant,
            int layer, const MoEExpertProjectionOwnership &ownership);

        /** @return Immutable source arithmetic and local output interval. */
        [[nodiscard]] const MoEExpertProjectionOwnership &ownership() const noexcept { return ownership_; }
        /** @return Stable logical participant whose graph owns these slices. */
        [[nodiscard]] int participantId() const noexcept { return participant_.participant_id; }
        /** @return The exact physical endpoint, never inferred from its slice index. */
        [[nodiscard]] DeviceId device() const noexcept { return participant_.device; }
        /** @return Model layer whose fixed projections are retained. */
        [[nodiscard]] int layer() const noexcept { return layer_; }
        /** @return All fixed slices in global expert-ID order, independent of residency. */
        [[nodiscard]] const std::vector<std::shared_ptr<ITensorGemm>> &engines() const noexcept { return engines_; }
        /**
         * @brief Export setup-only views for the existing persistent kernel table interface.
         * @return Every down slice, with exact local N and unchanged full K.
         * @throws std::runtime_error If any prepared engine has invalid geometry,
         *         missing/contradictory native provenance, or a different arithmetic family.
         *
         * Registry identity authenticates which slice was requested; this boundary
         * independently validates what the engine will actually execute. Export
         * performs no device allocation, copy, publication or weight conversion.
         * The consumer must retain this bank while a graph embeds these pointers.
         * The original unsliced arithmetic geometry remains in ownership().
         */
        [[nodiscard]] DescriptorTable exportDescriptorTable() const;
        /** @return Exact endpoint, source geometry, and engine identity equivalence. */
        [[nodiscard]] bool sameIdentity(const MoEOverlayFixedDownProjectionBank &other) const noexcept;

    private:
        /** @brief Seal a complete registry-validated bank; only resolve can construct it. */
        MoEOverlayFixedDownProjectionBank(MoEExpertOwnerParticipant participant, int layer,
            MoEExpertProjectionOwnership ownership, std::vector<std::shared_ptr<ITensorGemm>> engines);

        MoEExpertOwnerParticipant participant_;
        int layer_;
        MoEExpertProjectionOwnership ownership_;
        std::vector<std::shared_ptr<ITensorGemm>> engines_;
    };
} // namespace llaminar2
