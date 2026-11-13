/**
 * @file PlanningAdmissionPreparation.h
 * @brief Per-search immutable source/shape compilation, independent of physical admission.
 *
 * This owner retains only original metadata-derived declarations. It contains
 * no capacity, quota, certificate, allocation or mutable inference state. Every
 * physical candidate still binds its actual inventory and enters canonical PMA.
 */
#pragma once
#include "planning/PlanningModelMetadata.h"
#include "execution/moe/MoEOverlayInferenceTransaction.h"
#include "execution/moe/MoEOverlayPhysicalResidencyFabric.h"
#include "loaders/GPUVramPreflight.h"
#include <memory>

namespace llaminar2
{
    class ModelLoader;
    /** @brief Device-free compilation counts, not timing or hardware availability claims. */
    struct PlanningAdmissionPreparationCounts
    {
        size_t upload_geometry_builds = 0, mtp_validations = 0, graph_family_builds = 0;
        size_t full_manifest_builds = 0, interval_manifest_builds = 0;
    };
    /** @brief One exact loader/model/policy lifetime reused throughout exhaustive candidate selection. */
    class PlanningAdmissionPreparation final
    {
    public:
        /** @brief Retain original inputs and resolve upload geometry once before candidate enumeration. */
        PlanningAdmissionPreparation(const ModelLoader &loader, const PlanningModelMetadata &model,
            GPUWeightLoadMemoryPolicy weight_load);
        /** @brief Retire pure declarations; no physical resource is owned. */
        ~PlanningAdmissionPreparation();
        /** @brief Reject another source/model/upload policy before reusing any declaration. */
        void requireOriginal(const ModelLoader &loader, const PlanningModelMetadata &model,
            GPUWeightLoadMemoryPolicy weight_load) const;
        /** @return Original exact upload-ring geometry; never a capacity decision. */
        const GPUWeightLoadMemoryGeometry &weightLoadGeometry() const noexcept;
        /** @brief Authenticate the canonical MTP source manifest once when the request retains it. */
        void requireMTP();
        /** @return Canonical family compiled once for the complete original shape/policy identity. */
        const MoEOverlayInferenceGraphFamilyIdentity &graphFamily(MoEOverlayMTPGraphFamilyPolicy mtp,
            std::uint64_t generation, int graph_rows, int decode_rows, int requests, int draft_depth);
        /** @return Original exact layer interval, sliced from one canonical complete-family manifest. */
        const std::vector<MoEOverlayLayerWeightManifest> &manifest(int first_layer, int layers, int complete_layers);
        /** @return Repeatable pure-work evidence for device-free scaling regressions. */
        PlanningAdmissionPreparationCounts counts() const noexcept;
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}
