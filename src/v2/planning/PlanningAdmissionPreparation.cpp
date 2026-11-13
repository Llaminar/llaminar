/**
 * @file PlanningAdmissionPreparation.cpp
 * @brief Exact source/shape memoization with no physical identity or live ledger.
 *
 * Keys retain every argument of the canonical source builders. Complete model
 * manifests are built before interval slicing; no absent role or sidecar is
 * invented. References remain stable for this single original search lifetime.
 */
#include "planning/PlanningAdmissionPreparation.h"
#include "loaders/ModelLoader.h"
#include "execution/mtp/MTPWeightManifest.h"
#include <map>
#include <tuple>
#include <stdexcept>
namespace llaminar2
{
    /** @brief Original source borrows and pure declarations, never hardware/capacity state. */
    struct PlanningAdmissionPreparation::Impl
    {
        const ModelLoader &loader;
        const PlanningModelMetadata &model;
        GPUWeightLoadMemoryPolicy policy;
        GPUWeightLoadMemoryGeometry upload;
        PlanningAdmissionPreparationCounts counts;
        using FamilyKey = std::tuple<MoEOverlayMTPGraphFamilyPolicy,std::uint64_t,int,int,int,int>;
        std::map<FamilyKey,MoEOverlayInferenceGraphFamilyIdentity> families;
        std::map<int,std::vector<MoEOverlayLayerWeightManifest>> full;
        std::map<std::tuple<int,int,int>,std::vector<MoEOverlayLayerWeightManifest>> intervals;
        /** @brief Resolve the one source-dependent upload declaration. */
        Impl(const ModelLoader &loader_, const PlanningModelMetadata &model_, GPUWeightLoadMemoryPolicy policy_)
            : loader(loader_), model(model_), policy(policy_), upload(resolveGPUWeightLoadMemoryGeometry(
                  maximumGGUFTensorPayloadBytes(loader_.getModel()), policy_))
        { counts.upload_geometry_builds = 1; }
    };
    PlanningAdmissionPreparation::PlanningAdmissionPreparation(const ModelLoader &loader,
        const PlanningModelMetadata &model, GPUWeightLoadMemoryPolicy policy)
        : impl_(std::make_unique<Impl>(loader,model,policy)) {}
    PlanningAdmissionPreparation::~PlanningAdmissionPreparation() = default;
    void PlanningAdmissionPreparation::requireOriginal(const ModelLoader &loader,
        const PlanningModelMetadata &model, GPUWeightLoadMemoryPolicy policy) const
    {
        if (&loader != &impl_->loader || &model != &impl_->model ||
            policy.staging_stream_count != impl_->policy.staging_stream_count ||
            policy.staging_budget_bytes != impl_->policy.staging_budget_bytes)
            throw std::invalid_argument("Admission preparation belongs to another original source or upload policy");
    }
    const GPUWeightLoadMemoryGeometry &PlanningAdmissionPreparation::weightLoadGeometry() const noexcept { return impl_->upload; }
    PlanningAdmissionPreparationCounts PlanningAdmissionPreparation::counts() const noexcept { return impl_->counts; }
    void PlanningAdmissionPreparation::requireMTP()
    {
        if (impl_->counts.mtp_validations) return;
        (void)requireMTPWeightManifest(impl_->loader,impl_->loader.architecture(),static_cast<int>(impl_->loader.blockCount()));
        ++impl_->counts.mtp_validations;
    }
    const MoEOverlayInferenceGraphFamilyIdentity &PlanningAdmissionPreparation::graphFamily(
        MoEOverlayMTPGraphFamilyPolicy mtp, std::uint64_t generation, int graph_rows,
        int decode_rows, int requests, int draft_depth)
    {
        const Impl::FamilyKey key{mtp,generation,graph_rows,decode_rows,requests,draft_depth};
        auto found=impl_->families.find(key);
        if (found==impl_->families.end())
        {
            auto value=resolveMoEOverlayInferenceGraphFamilyIdentity(impl_->loader,impl_->loader.architecture(),
                static_cast<int>(impl_->loader.blockCount()),mtp,generation,graph_rows,decode_rows,requests,draft_depth);
            found=impl_->families.emplace(key,std::move(value)).first;
            ++impl_->counts.graph_family_builds;
        }
        return found->second;
    }
    const std::vector<MoEOverlayLayerWeightManifest> &PlanningAdmissionPreparation::manifest(
        int first, int layers, int complete)
    {
        if (first<0 || layers<=0 || complete<=0 || first>complete || layers>complete-first)
            throw std::invalid_argument("Admission source manifest interval is outside its complete graph family");
        const auto key=std::tuple{first,layers,complete};
        auto found=impl_->intervals.find(key);
        if (found!=impl_->intervals.end()) return found->second;
        auto source=impl_->full.find(complete);
        if (source==impl_->full.end())
        {
            auto value=buildMoEOverlayLayerWeightManifestFromLoader(impl_->loader,complete,
                impl_->model.memoryProfile().expert_count);
            if (value.size()!=static_cast<size_t>(complete))
                throw std::logic_error("Canonical complete expert manifest changed its layer geometry");
            source=impl_->full.emplace(complete,std::move(value)).first;
            ++impl_->counts.full_manifest_builds;
        }
        std::vector<MoEOverlayLayerWeightManifest> interval(source->second.begin()+first,source->second.begin()+first+layers);
        found=impl_->intervals.emplace(key,std::move(interval)).first;
        ++impl_->counts.interval_manifest_builds;
        return found->second;
    }
}
