/**
 * @file Test__MoEIntermediateShardFFNFinalization.cpp
 * @brief Original-source native replay of local routed/shared FFN completion.
 *
 * One real GGUF/ModelContext/WeightManager owns every participant's immutable
 * routed and shared source. Production TP assignments select shared N/N/K
 * slices; the closed original fragment cohort selects routed F slices. Each
 * captured candidate executes router, local routed projections/fold, local
 * shared FFN, shared gate/add and exactly one CanonicalRankOrder native FFN
 * sum. A separately captured M1 graph is the diagnostic row oracle for the
 * SAME decode-equivalent policy, never a production row replay mechanism.
 * Floating-router aliases require the original Q8 hidden publication before
 * preparation, including the one-row MTP bridge, without changing router math.
 *
 * This model-free numerical fixture does not certify full-model KL, live-grid
 * economy, real-source model admission or cross-tier expert movement. The
 * serving guard remains closed until those independent gates pass. Optional
 * independent gate/up sources exercise actual admitted IQ4 role asymmetry
 * through the same source loader, prepared owners and full replay captures.
 */
#include <charconv>
#include "tensors/TensorSlice.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/compute_stages/IGraphOutputDemand.h"
#include "kernels/common/FullNativeHCFFNOutputDemand.h"
#include "fixtures/moe/OriginalFragmentFabric.h"
#include "execution/moe/MoEOverlayHostAuthorityDeviceBankPublisher.h"
#include "execution/moe/MoEOverlayDeviceControllerRuntimeBinding.h"
#include <gtest/gtest.h>
#include "../../performance/kernels/native_quantised_dispatch/NativeQuantisedGeometryTimingOrder.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/NativeContextPreparation.h"
#include "backends/BackendManager.h"
#ifdef HAVE_ROCM
#include "backends/rocm/HIPGraphCapture.h"
#include "collective/OriginalNativeGraphCaptureGroup.h"
#include "kernels/rocm/gemm/ROCmResidentMoEFFN.h"
#include "kernels/rocm/gemm/ROCmGateUpLastProducer.h"
#include "kernels/rocm/gemm/ROCmDownLastProducer.h"
#include "kernels/rocm/gemm/ROCmNativeLayoutRouterFFN.h"
#include "kernels/rocm/gemm/ROCmFullNativeHCFFN.h"
#include "kernels/rocm/gemm/retained_hc_ffn/ROCmRetainedM1HCFFNDevice.h"
#include "planning/TensorParallelSourceDeclaration.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpHyperconnectionBuilder.h"
#include "utils/GpuPreparedGemmHarness.h"
#endif
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <utility>
#include <syncstream>
#include "collective/LocalTPContext.h"
#include "config/TensorParallelConfig.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoEIntermediateShardPipeline.h"
#include "execution/compute_stages/stages/MoERoutingStage.h"
#include "execution/compute_stages/stages/MoEOverlayEpochBoundaryStage.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "execution/moe/DeviceMoEIntermediateShardBinding.h"
#include "execution/moe/MoEIntermediateShardRuntimeBinding.h"
#include "execution/moe/MoEIntermediateShardStaticGraphAdmission.h"
#include "execution/moe/MoEGroupedVerifierHistogramBoundarySet.h"
#include "execution/moe/MoEOverlayHostAuthorityBankRecipe.h"
#include "execution/moe/MoEOverlayOriginalGPUCohortContexts.h"
#include "execution/moe/MoERuntimeTable.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "execution/moe/ExpertPreparedMemoryGeometry.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/moe/MoEOverlayPreparedGraphBank.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "execution/compute_stages/ComputeStageUtils.h"
#include "execution/compute_stages/stages/TPAllreduceStage.h"
#include "kernels/cuda/gemm/CUDAQuantisedGemmWorkspaceContract.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmWorkspaceContract.h"
#include "kernels/common/OriginalM1HiddenQ8Publication.h"
#include "kernels/common/MoEIntermediateShardRowExecution.h"
#include "kernels/KernelFactory.h"
#include "HiddenQ8NativeGraphIdentity.h"
#include "loaders/ModelContext.h"
#include "loaders/BoundedGPUExpertSource.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/WeightManager.h"
#include "loaders/gpu_pipeline/WeightVRAMPool.h"
#include "models/qwen35moe/Qwen35MoESchema.h"
#include "planning/CollectiveMemoryEstimator.h"
#include "planning/PlanningObservedResource.h"
#include "planning/PlanningGPUExpertMeasurement.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include "backends/RetainedParticipantGraphExecutable.h"
#include "execution/moe/MoEOverlayCPUServiceMeasurement.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "fixtures/backends/OriginalRankLocalNativeReplay.h"
#include "utils/PlanningGGUFFixture.h"
#include "../../utils/ScopedGPUCompletionEvent.h"
#include "utils/QuantizedVerifierFormats.h"
#include <algorithm>
#include <array>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <map>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <functional>
#if defined(LLAMINAR_CASTAGNA_KERNEL_AB)
#include "CastagnaHcPersistExport.h"
#include "CastagnaNativeWeightExport.h"
#include "CastagnaExactCandidateOracle.h"
#include "utils/BoundedCenteredQ4KSource.h"
#include <cmath>
#endif

namespace llaminar2::test
{
namespace
{
constexpr int kH = 256, kRoutedF = 1024, kSharedF = 256, kExperts = 8, kTopK = 2, kMaxRows = 129;
/** @brief Original source router recipe; shared by unchanged staged and candidate graphs. */
enum class OriginalFFNRouterSource { TieControl,Dense };
/** @brief Optional exact source/topology assertion; never overrides production slicing. */
enum class OriginalFFNGeometryContract { SourceDerived,QFNTensorSharded,QFNSingletonShardShape };
/** @brief Immutable original-source dimensions passed through admission, preparation and actual worker capture. */
struct OriginalFFNFixtureGeometry {
    int hidden=kH; ///< Model/source H; original default fixture remains unchanged.
    int routed=kRoutedF; ///< Original source routed intermediate width before any production TP cut.
    int shared=kSharedF; ///< Original independent shared source width before TP.
    int experts=kExperts; ///< Original source population and actual router logit width.
    int top_k=kTopK; ///< Original metadata-selected route extent, never a prepared override.
    OriginalFFNRouterSource router_source=OriginalFFNRouterSource::TieControl; ///< Original raw source bytes, not a routing override.
    OriginalFFNGeometryContract contract=OriginalFFNGeometryContract::SourceDerived; ///< Assert an exact admitted source geometry.
};
constexpr std::array kRoutedRoles{WeightRole::MoEExpertGate, WeightRole::MoEExpertUp, WeightRole::MoEExpertDown};
constexpr std::array kRoutedNames{"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"};
constexpr std::array kSharedNames{"blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight", "blk.0.ffn_down_shexp.weight",
    "blk.0.ffn_gate_inp_shexp.weight", "blk.0.ffn_gate_inp.weight"};
constexpr std::array kSharedRoles{WeightRole::SharedExpertGate, WeightRole::SharedExpertUp, WeightRole::SharedExpertDown,
    WeightRole::SharedExpertInputGate, WeightRole::MoERouter};

/** @brief Fail on the original operation, preserving exact native diagnostic context. */
void require(bool valid, const std::string &operation)
{ if (!valid) throw std::runtime_error("Original local FFN finalization: " + operation); }

/** @brief Explicit diagnostic observation; canonical timing never binds HTS storage. */
enum class CastagnaPhaseObservation { Disabled, PerCTA };

/** @brief Keep source-only recipe checks independent from native FFN arithmetic. */
enum class OriginalSourceQualification { FFNFinalization, HostAuthorityRecipes, PreparedService, ResidentFFN, ResidentFFNEconomy, GateUpLastProducer, GateUpPairedK32, DownLastProducer, OriginalM1HiddenQ8, NativeLayoutRouterFFN, NativeLayoutRouterFFNParentReplay, NativeLayoutRouterFFNEconomy,CastagnaFullRegionAB,CastagnaFullRegionABEconomy,CastagnaFullRegionABReal,CastagnaFullRegionABRealEconomy,CastagnaFullRegionABBounded,CastagnaFullRegionABBoundedEconomy, FullNativeHCFFNFinal, FullNativeHCFFNMiddle, GroupedQ8DownReplay };
/** @brief Explicit test-only full production graph lowering; default preserves every R57 arm. */
enum class FullProductionHCProbe { Disabled, FinalRead, MiddleRead };
/** @brief Cold observer/read demand; every mode uses the actual production full entry. */
enum class FullProductionHCObservation { ObservedAll, DemandedOnly, HCUpOnly, OutsideHCUpReader, RetainedDemandedOnly };
/** @brief Expected immutable optional policy in this fixture's exact final graph.
 * @param observation Existing observer/filter or actual outside-reader selection.
 * @return Declared role bits; mandatory internal operands are outside this mask.
 */
constexpr FullNativeHCFFNOutputDemand fullProductionExpectedDemand(FullProductionHCObservation observation) {
    if(observation==FullProductionHCObservation::ObservedAll)return FullNativeHCFFNOutputDemand::all();
    if(observation==FullProductionHCObservation::DemandedOnly||
        observation==FullProductionHCObservation::RetainedDemandedOnly)return FullNativeHCFFNOutputDemand::none();
    return FullNativeHCFFNOutputDemand::none().with(FullNativeHCFFNOutputRole::HCUp);
}
/** @brief Exact model read-before-write storage reuse across original HC/FFN phases. */
enum class FullProductionHCStorageReuse { Distinct, ModelReadBeforeWrite };

/** @brief Original source factories; shared quantization is independent of routed weights. */
struct SourceFormat
{
    std::string name;
    TensorType type;
    ExpertWeightFormat identity;
    QuantizedVerifierWeightCreator create;
    const QuantizedVerifierFormatCase *shared_quantized = nullptr; ///< Absent denotes the explicit original FP32 shared control.
    const QuantizedVerifierFormatCase *up_quantized = nullptr; ///< Optional actual source up tensor, not a descriptor relabel.
    const QuantizedVerifierFormatCase *down_quantized = nullptr; ///< Optional actual source down tensor with independent native encoding.
    std::uint8_t expected_native_layout_groups = 0; ///< Optional exact compiled family asserted from original source envelopes.
    const QuantizedVerifierFormatCase *shared_up_quantized = nullptr; ///< Optional genuine shared-up encoding.
    const QuantizedVerifierFormatCase *shared_down_quantized = nullptr; ///< Optional original shared-down encoding independent of gate/up.
};

/**
 * @param qualification Distinguish representative arithmetic replay from all-format source publication.
 * @return Original registered sources; native execution component gates retain their separate all-format sweep.
 */
std::vector<SourceFormat> sourceFormats(OriginalSourceQualification qualification)
{
    std::vector<SourceFormat> result;
    const bool all_formats = qualification != OriginalSourceQualification::FFNFinalization;
    for (const auto &source : quantizedVerifierFormats())
        // The exhaustive runtime matrix also contains prepared-only Q8_1.
        // Its rejected GGUF boundary has an explicit device-free regression;
        // every encodable original source remains in this positive source sweep.
        if (source.tensor_type != TensorType::Q8_1 &&
            (all_formats || source.tensor_type == TensorType::Q4_0 || source.tensor_type == TensorType::Q4_K))
            result.push_back({source.label, source.tensor_type,
                ExpertWeightFormat::nativeQuantised({source.source_codebook_id, source.source_is_superblock, true}), source.create});
    if(qualification==OriginalSourceQualification::ResidentFFN)
    {
        // Every native source format exercises both routed and shared decode;
        // the extra mixed case matches QFN's independent shared Q8 source.
        const auto &formats=quantizedVerifierFormats();
        for(auto &entry:result)
        {
            const auto match=std::find_if(formats.begin(),formats.end(),[&](const auto &f){return f.tensor_type==entry.type;});
            require(match!=formats.end(),"resident original shared source format");
            entry.shared_quantized=&*match;
        }
        const auto shared=std::find_if(formats.begin(),formats.end(),[](const auto &f){return f.tensor_type==TensorType::Q8_0;});
        const auto routed=std::find_if(result.begin(),result.end(),[](const auto &f){return f.type==TensorType::Q4_K;});
        require(shared!=formats.end() && routed!=result.end(),"resident original mixed source formats");
        auto mixed=*routed;mixed.name+="_SharedQ8_0";mixed.shared_quantized=&*shared;result.push_back(std::move(mixed));
        return result;
    }
    result.push_back({"FP32", TensorType::FP32, ExpertWeightFormat::floating(TensorType::FP32),
        [](const auto &shape, uint32_t seed) { return TestTensorFactory::createFP32Random(shape, -.125f, .125f, seed); }});
    if (all_formats)
    {
        result.push_back({"FP16", TensorType::FP16, ExpertWeightFormat::floating(TensorType::FP16),
            [](const auto &shape, uint32_t seed) { return TestTensorFactory::createFP16Random(shape, -.125f, .125f, seed); }});
        result.push_back({"BF16", TensorType::BF16, ExpertWeightFormat::floating(TensorType::BF16),
            [](const auto &shape, uint32_t seed) { return TestTensorFactory::createBF16Random(shape, -.125f, .125f, seed); }});
    }
    // The actual QFN shared branch uses Q8_0. Keep the existing floating shared
    // controls and exercise native shared preparation/slicing in the same graph.
    const auto &formats = quantizedVerifierFormats();
    const auto shared = std::find_if(formats.begin(), formats.end(), [](const auto &entry) {
        return entry.tensor_type == TensorType::Q8_0;
    });
    const auto routed = std::find_if(result.begin(), result.end(), [](const auto &entry) {
        return entry.type == TensorType::Q4_K;
    });
    require(shared != formats.end() && routed != result.end(), "canonical routed/shared source formats");
    auto quantized_shared = *routed;
    quantized_shared.name += "_SharedQ8_0";
    quantized_shared.shared_quantized = &*shared;
    result.push_back(std::move(quantized_shared));
    return result;
}

/**
 * @return All original same-format sources plus both production-admitted IQ4 mixed directions.
 * @details The canonical gate/up compatibility rule admits equal physical IDs
 * or IQ4_NL/IQ4_XS cross pairs. Other mixed IDs are deliberately not fabricated.
 */
std::vector<SourceFormat> pairedGateUpSourceFormats()
{
    auto result=sourceFormats(OriginalSourceQualification::ResidentFFN);
    const auto& formats=quantizedVerifierFormats();
    for(const auto gate_type:std::array{TensorType::IQ4_NL,TensorType::IQ4_XS})
    {
        const auto up_type=gate_type==TensorType::IQ4_NL?TensorType::IQ4_XS:TensorType::IQ4_NL;
        const auto gate=std::find_if(result.begin(),result.end(),[&](const auto& f){return f.type==gate_type;});
        const auto up=std::find_if(formats.begin(),formats.end(),[&](const auto& f){return f.tensor_type==up_type;});
        require(gate!=result.end()&&up!=formats.end(),"paired original mixed IQ4 source factories");
        auto mixed=*gate;mixed.name+="_Up"+std::string(up->label);mixed.up_quantized=&*up;
        result.push_back(std::move(mixed));
    }
    return result;
}

/**
 * @brief Extend only NativeLayout coverage with an original-source full-union family.
 * @return Existing sources plus IQ3_XXS gate/up, Q6_K down and IQ2_S shared.
 * @details Runtime-mutable IQ3_XXS admits its authentic INT8 promotion as well as
 * its current grid decoder. Together with Q6 and shared IQ2, this requires all
 * four compiled groups without inventing a descriptor or changing source bytes.
 */
std::vector<SourceFormat> nativeLayoutSourceFormats()
{
    auto result=pairedGateUpSourceFormats();
    const auto& formats=quantizedVerifierFormats();
    const auto gate=std::find_if(result.begin(),result.end(),[](const auto& f){return f.type==TensorType::IQ3_XXS;});
    const auto down=std::find_if(formats.begin(),formats.end(),[](const auto& f){return f.tensor_type==TensorType::Q6_K;});
    const auto shared=std::find_if(formats.begin(),formats.end(),[](const auto& f){return f.tensor_type==TensorType::IQ2_S;});
    require(gate!=result.end()&&down!=formats.end()&&shared!=formats.end(),"full-union original source factories");
    auto full_union=*gate;
    full_union.name="IQ3_XXS_DownQ6_K_SharedIQ2_S_FullUnion";
    full_union.down_quantized=&*down;
    full_union.shared_quantized=&*shared;
    full_union.expected_native_layout_groups=15;
    result.push_back(std::move(full_union));
    return result;
}

/** @brief Publish valid original payloads, including the actual shared matrices and both gates. */
void writeOriginalSources(PlanningGGUFFixture &file, const SourceFormat &format,OriginalFFNFixtureGeometry shape={})
{
    const int kH=shape.hidden,kRoutedF=shape.routed,kSharedF=shape.shared,kExperts=shape.experts;
    for (std::size_t role = 0; role < kRoutedNames.size(); ++role)
    {
        const int n = role == 2 ? kH : kRoutedF, k = role == 2 ? kRoutedF : kH;
        const auto* original_format=role==1?format.up_quantized:role==2?format.down_quantized:nullptr;
        const auto source = original_format
            ?original_format->create({std::size_t(kExperts*n),std::size_t(k)},6701+role)
            :format.create({std::size_t(kExperts*n),std::size_t(k)},6701+role);
        file.writePayload(kRoutedNames[role], 0, {static_cast<const uint8_t *>(source->raw_data()), source->size_bytes()});
    }
    for (std::size_t role = 0; role < kSharedNames.size(); ++role)
    {
        if (role < 3 && format.shared_quantized)
        {
            const std::size_t n = role == 2 ? kH : kSharedF, k = role == 2 ? kSharedF : kH;
            const std::array formats{format.shared_quantized,
                format.shared_up_quantized ? format.shared_up_quantized : format.shared_quantized,
                format.shared_down_quantized ? format.shared_down_quantized : format.shared_quantized};
            const auto* original_format = formats[role];
            const auto source = original_format->create({n, k}, 7103 + role);
            file.writePayload(kSharedNames[role], 0,
                {static_cast<const uint8_t *>(source->raw_data()), source->size_bytes()});
            continue;
        }
        const std::size_t count = role < 3 ? kSharedF * kH : role == 3 ? kH : kExperts * kH;
        std::vector<float> source(count);
        for (std::size_t i = 0; i < count; ++i)
            source[i] = float(int((i * 7 + role * 3) % 19) - 9) / 128.f;
        if(role==4&&shape.router_source==OriginalFFNRouterSource::Dense)
        {
            const auto router=TestTensorFactory::createFP32Random({size_t(kExperts),size_t(kH)},-.125f,.125f,8127);
            std::memcpy(source.data(),router->raw_data(),source.size()*sizeof(float));
        }
        else if (role == 4)
        {
            std::fill(source.begin(), source.end(), 0.f);
            for (int expert = 0; expert < kExperts; ++expert)
                source[std::size_t(expert) * kH] = expert == 0 ? 2.f : expert == 1 ? 1.f : -1.f;
        }
        file.writePayload(kSharedNames[role], 0, {reinterpret_cast<const uint8_t *>(source.data()), source.size() * sizeof(float)});
    }
}

/** @return Shared/source plan using the original LocalTP assignment rather than a new slicing path. */
WeightPlan sharedPlan(ModelContextId model, const std::vector<DeviceId> &devices, DeviceId owner)
{
    WeightPlan result(InferenceStrategy{.mode = devices.size() > 1 ? WeightInferenceMode::LocalTP : WeightInferenceMode::SingleDevice,
        .model_id = model, .tp_degree = int(devices.size()), .devices = devices});
    for (std::size_t member = 0; member < devices.size(); ++member)
        if (devices[member] == owner)
        for (std::size_t role = 0; role < kSharedNames.size(); ++role)
        {
            WeightRequirement request;
            request.canonical_name = kSharedNames[role]; request.role = kSharedRoles[role]; request.layer = 0;
            request.target_device = devices[member]; request.lookup_device = devices[member];
            request.tp_domain = 0; request.tp_rank_or_device_index = int(member);
            request.bypass_tensor_parallel = role >= 3;
            request.residency_category = role < 4 ? WeightResidencyCategory::SharedExpert : WeightResidencyCategory::RootNonExpert;
            result.add(std::move(request));
        }
    return result;
}

/** @return Dependency ledger derived from the exact production arena/weight contracts. */
std::vector<GraphCaptureDependencyLedger::StagePlan> capturePlan(ComputeGraph &graph,
    const std::map<BufferId, TensorBase *> &buffers)
{
    std::vector<GraphCaptureDependencyLedger::StagePlan> plans;
    std::map<BufferId, std::size_t> writers;
    for (const auto &name : graph.getExecutionOrder())
    {
        auto *stage = graph.getNode(name)->stage.get();
        GraphCaptureDependencyLedger::StagePlan plan{.stage_identity = stage, .stage_name = name};
        const auto contract = stage->bufferContract();
        for (const auto &input : contract.allArenaReads())
        {
            auto *tensor = buffers.at(input.id);
            if (const auto writer = writers.find(input.id); writer != writers.end())
                plan.internal_inputs.push_back({tensor, writer->second});
            else plan.external_inputs.push_back(tensor);
        }
        for (auto *weight : contract.weight_tensors) plan.external_inputs.push_back(dynamic_cast<TensorBase *>(weight));
        for (const auto &output : contract.allWrites())
        { plan.outputs.push_back(buffers.at(output.id)); writers[output.id] = plans.size(); }
        plans.push_back(std::move(plan));
    }
    return plans;
}

#include "MoEIntermediateShardFFNFinalizationContract.inc"

/** @brief Cold-only adversary selecting one forbidden router-weight output overlap. */
enum class NativeRouterAliasProbe { None, RouterOutput, FFNOutput };

/**
 * @brief Borrow genuine output metadata while reporting a forbidden device address.
 *
 * This view never owns, rebinds or writes GPU storage. Actual backend admission
 * must reject it before capture; the original output and router remain alive.
 */
class NativeRouterAliasTensor final : public TensorSlice
{
public:
    /** @param output Original FP32 output owner. @param address Original router weight address. */
    NativeRouterAliasTensor(std::shared_ptr<TensorBase> output, void* address)
        : TensorSlice(std::move(output), SliceMetadata{}), address_(address) {}
    /** @return Deliberately overlapping address for cold owner validation only. */
    void* gpu_data_ptr() override { return address_; }
    /** @return The same deliberately overlapping address for const validation. */
    const void* gpu_data_ptr() const override { return address_; }
private:
    void* address_;
};

/** @brief Select the fixture-only collective witness or production payload scope. */
enum class OriginalFFNCollectiveWitness {DiagnosticSideband,ProductionPayloadOnly};
/** @brief Late model lowering preserves the original staged router until capture. */
enum class EpochGraphLowering { FixtureNativeLayout,ModelRetained };

/** @brief Persistent storage and one preparation-only captured production FFN graph. */
struct Lane
{
    std::shared_ptr<FP32Tensor> hidden, indices, weights, routes, routed, shared, combined, gate_scratch, up_scratch, receipt;
    std::shared_ptr<INT32Tensor> active, sideband;
    std::shared_ptr<MoERoutedPipelineKernelOwner> owner;
    std::unique_ptr<IMoEKernel> hidden_q8_shared_kernel; ///< Original private backend instance exposed only to this focused fixture.
    std::shared_ptr<DeviceMoERuntimeTable> runtime;
    std::unique_ptr<DeviceWorkspaceManager> workspace;
    std::shared_ptr<NativeRouterAliasTensor> alias_probe; ///< Cold rejection view, retained beyond graph destruction.
    BufferArena output_arena; ///< Canonical original tensor registration, retained through graph retirement.
    ComputeGraph graph;
    DeviceGraphExecutor output_demand_executor; ///< Actual cold selector; never executes this fixture's native graph.
    FullProductionHCObservation output_observation=FullProductionHCObservation::ObservedAll;
    const TensorBase* hc_up_observation=nullptr; ///< Original tensor identity used by singleton filtering.
    bool outside_hc_up_reader=false; ///< Existing production ResidualRead adds one real consumer kernel.
    std::unique_ptr<IGPUGraphCapture> capture;
    std::unique_ptr<IGPUGraphCapture> serial_prefill_scratch; ///< Retained test-only alternate participant, ordered before decode.
    std::vector<GraphCaptureDependencyLedger::StagePlan> plan;
    std::size_t hidden_q8_workspace_bytes=0; ///< Diagnostic original-versus-required BOM receipt.
    int rows = 0;
    IWorkerGPUContext *worker = nullptr;
    void *stream = nullptr;
    ILocalTPContext *tp = nullptr;
    bool expected_cold_rejection = false; ///< Only alias probes intentionally unwind before native recording.
    int exceptions = std::uncaught_exceptions();
    /** @brief Join terminal fixture work before any native graph or tensor member retires. */
    ~Lane()
    {
        if (worker && stream)
        {
            if (std::uncaught_exceptions() > exceptions && tp && !expected_cold_rejection) tp->requestAbort();
            (void)worker->synchronizeStreamChecked(stream);
        }
    }
};

/** @brief Preserve the original counted fixture by default; new M1 controls/candidates both use actual fully-active rows. */
enum class OriginalM1HiddenQ8FixturePolicy { CountedOriginal,FullActiveOriginal,FullActiveReuse };
/** @brief Explicit ownership regression; timing and existing unrelated lanes stay unchanged. */
enum class CompletionSerialFamilyReplay { Disabled,PrefillScratchBeforeDecode };
/** @brief One exact original M1 named requirement set and its canonical PMA physical layout. */
struct OriginalM1HiddenQ8WorkspaceQuote final {
    WorkspaceRequirements requirements;
    SerialWorkspaceFamilyPlan physical;
};
/**
 * @brief Compose the production workspace ABI from original source/participant geometry before weights allocate.
 * @param device Exact original backend participant.
 * @param participants Original collective degree and shared N/N/K cut count.
 * @param routedColumns Original source ownership's local F extent, never a capacity estimate.
 * @param sharedFormat Actual immutable shared gate/up/down source format.
 * @param computeUnits Original inventory SM/CU count, without a runtime probe.
 * @return Same named contracts and physical plan consumed by both retained M1 lanes.
 */
std::shared_ptr<const OriginalM1HiddenQ8WorkspaceQuote> quoteOriginalM1HiddenQ8Workspace(DeviceId device,
    int participants,int routedColumns,const QuantizedVerifierFormatCase& sharedFormat,int computeUnits) {
    require(device.is_gpu()&&participants>0&&kSharedF%participants==0&&routedColumns>0&&computeUnits>0,
        "original hidden Q8 exact workspace source/participant geometry");
    const int sharedColumns=kSharedF/participants;
    auto requirements=device.is_cuda()?MoEWorkspaceBuffers::cudaRouting(1,kH,kExperts)
        :MoEWorkspaceBuffers::rocmRouting(1,kH,kExperts);
    requirements.merge(device.is_cuda()?MoEWorkspaceBuffers::cudaMoE(1,kH,routedColumns,kExperts,kTopK)
        :MoEWorkspaceBuffers::rocmMoE(1,kH,routedColumns,kExperts,kTopK));
    WorkspaceRequirements shared;
    const std::array columns{sharedColumns,sharedColumns};
    if(device.is_rocm()) {
        shared.merge(rocm::quantized_gemm_workspace::projectionRequirements(1,sharedColumns,kH));
        shared.merge(rocm::quantized_gemm_workspace::projectionRequirements(1,kH,sharedColumns));
        rocm::quantized_gemm_workspace::appendFusedProjectionRequirements(shared,1,columns);
    } else {
#ifdef HAVE_CUDA
        const auto hardware=cuda::NativeQuantisedWorkspaceHardware::checked(device.ordinal,computeUnits);
        const cuda::quantized_gemm_workspace::NativeCodebooks native{sharedFormat.device_execution_codebook_id,
            canonicalDeviceVnniCodebookId(sharedFormat.source_codebook_id)};
        shared.merge(cuda::quantized_gemm_workspace::projectionRequirements(1,sharedColumns,kH,hardware,native));
        shared.merge(cuda::quantized_gemm_workspace::projectionRequirements(1,kH,sharedColumns,hardware,native));
        cuda::quantized_gemm_workspace::appendFusedProjectionRequirements(shared,1,columns,kH);
#else
        throw std::invalid_argument("Original hidden Q8 workspace requires its compiled CUDA backend");
#endif
    }
    addCudaConcurrentDecodeGemvSideStreamWorkspace(shared,device,1,2);
    shared.merge(device.is_cuda()?MoEWorkspaceBuffers::cudaMoE(1,kH,sharedColumns,1,1)
        :MoEWorkspaceBuffers::rocmMoE(1,kH,sharedColumns,1,1));
    requirements.merge(shared);
    requirements.merge(TPAllreduceStage::canonicalRankWorkspaceRequirements(kH,participants));
    auto physical=DeviceWorkspaceManager::planSerialFamily({requirements});
    require(physical.valid(),"canonical original hidden Q8 workspace physical plan");
    return std::make_shared<const OriginalM1HiddenQ8WorkspaceQuote>(std::move(requirements),std::move(physical));
}
/**
 * @brief Require the actual original stage union to match the pre-allocation quote exactly.
 * @param actual Original production stage declarations after prepared weights are installed.
 * @param quoted Same original source/geometry quote admitted by PhysicalMemoryAuthority.
 * @details Every field participates: an accidental new buffer or changed lifetime must fail before allocation.
 */
void requireOriginalM1HiddenQ8Workspace(const WorkspaceRequirements& actual,const WorkspaceRequirements& quoted) {
    require(actual.buffers.size()==quoted.buffers.size()&&!actual.cpu_grouped_moe_storage&&!actual.cpu_router_q8_storage,
        "original hidden Q8 quote retains exactly the actual native stage names");
    for(const auto& descriptor:actual.buffers) {
        const auto* original=quoted.find(descriptor.name);
        require(original&&original->size_bytes==descriptor.size_bytes&&original->alignment==descriptor.alignment&&
            original->required==descriptor.required&&original->regime==descriptor.regime&&
            original->content_lifetime==descriptor.content_lifetime,
            "original hidden Q8 pre-allocation quote differs from actual stage: "+descriptor.name);
    }
}
/**
 * @brief Borrow one genuinely published migration epoch for both replay programs.
 *
 * The runtime and bank retain their original model/cohort and completed physical
 * arrival. This is never an alternative source constructor or an epoch relabel.
 */
/** @brief Test prerequisite: retain real initial publication or the original later-arrival requirement. */
enum class PublishedFFNEpochAdmission { LaterPublished, InitialPublished };
struct PublishedFFNEpoch final {
    std::shared_ptr<DeviceMoERuntimeTable> runtime;
    std::shared_ptr<DeviceMoEOverlayEpochArena> arena;
    MoEOverlayParticipantBankLease bank;
    std::vector<int> promoted_experts; ///< Actual CPU-arrival IDs whose down descriptors are physical23.
    std::vector<int> canonical_experts; ///< Retained original GPU residents required in the same observed EB4 packet.
    /** @return Exact completed source epoch, or zero for an empty borrow. */
    uint64_t epoch() const noexcept { return bank ? bank->epoch : 0; }
};

/**
 * @brief Declare, prepare and capture one actual local FFN graph without eager inference.
 * @param collective_witness Preserve diagnostic sideband coverage unless economy explicitly selects the production payload only.
 * @param observation Explicit actual executor cold observation policy, sealed before native preparation.
 * @param published Optional exact later-epoch runtime/bank borrow. Ordinary callers
 * keep the original setup publication; promoted replay uses its event-published
 * candidate unchanged and brackets execution with the canonical epoch stages.
 * @param epoch_admission Explicit test prerequisite, never a production epoch override.
 * The default still requires a later real arrival; initial replay must retain epoch1 source/runtime publication.
 */
std::unique_ptr<Lane> makeLane(int rows, bool grouped, DeviceId device, int member, ILocalTPContext &tp,
    IWorkerGPUContext &worker, IDeviceContext &context, void *stream, const MoEOverlayParticipantLayerBank &source,
    const FrozenModelWeightSet &shared_weights, const TensorParallelConfig &shared_partition, PreparedWeightStore &store,
    MoEIntermediateShardFFNExecution execution=MoEIntermediateShardFFNExecution::Staged,
    MoEGateUpPublicationPolicy gate_up_publication=MoEGateUpPublicationPolicy::Staged,
    OriginalM1HiddenQ8FixturePolicy hidden_q8=OriginalM1HiddenQ8FixturePolicy::CountedOriginal,
    std::shared_ptr<PhysicalMemoryAuthority> memory={},
    std::shared_ptr<const OriginalM1HiddenQ8WorkspaceQuote> hidden_q8_workspace={},
    MoEDownPublicationPolicy down_publication=MoEDownPublicationPolicy::Staged,
    CompletionSerialFamilyReplay completion_family=CompletionSerialFamilyReplay::Disabled,
    OriginalFFNFixtureGeometry shape={},bool native_layout_router=false,
    NativeRouterAliasProbe alias_probe=NativeRouterAliasProbe::None,
    const PublishedFFNEpoch* published=nullptr,
    OriginalFFNCollectiveWitness collective_witness=OriginalFFNCollectiveWitness::DiagnosticSideband,
    const std::function<void(Lane&,std::map<BufferId,TensorBase*>&)>& prefix={},
    FullProductionHCObservation observation=FullProductionHCObservation::ObservedAll,
    PublishedFFNEpochAdmission epoch_admission=PublishedFFNEpochAdmission::LaterPublished,
    EpochGraphLowering epoch_graph=EpochGraphLowering::FixtureNativeLayout)
{
    const int kH=shape.hidden,kExperts=shape.experts,kTopK=shape.top_k;
    auto lane = std::make_unique<Lane>(); lane->rows = rows; lane->output_observation=observation;
    lane->expected_cold_rejection=alias_probe!=NativeRouterAliasProbe::None;
    lane->worker = &worker; lane->stream = stream; lane->tp = &tp;
    const auto fp = [&](std::vector<std::size_t> shape) {
        auto value = TestTensorFactory::createFP32(shape);
        std::fill_n(value->mutable_data(), value->numel(), 0.f);
        require(value->ensureOnDevice(device, stream), "admit finalizer tensor"); return value;
    };
    const std::size_t storage_rows=std::size_t(rows)+(hidden_q8==OriginalM1HiddenQ8FixturePolicy::CountedOriginal?0:1);
    lane->hidden = fp({storage_rows, kH}); lane->indices = fp({storage_rows, kTopK});
    lane->weights = fp({storage_rows, kTopK}); lane->routes = fp({storage_rows, kTopK, kH});
    lane->routed = fp({storage_rows, kH}); lane->shared = fp({storage_rows, kH});
    lane->combined = fp({storage_rows, kH}); lane->receipt = fp({2});
    const auto* shared_gate_source=shared_weights.optionalLayerForDevice(0,"ffn_gate_shexp.weight",device);
    require(shared_gate_source&&shared_gate_source->tensor,"original shared gate source owns local F");
    const int shared_f = static_cast<int>(shared_gate_source->tensor->rows());
    require(shared_f>0&&shared_f%32==0,"original prepared shared TP slice is a positive Q8 group");
    lane->gate_scratch = fp({storage_rows, std::size_t(shared_f)});
    lane->up_scratch = fp({storage_rows, std::size_t(shared_f)});
    lane->active = TestTensorFactory::createINT32({1}); lane->active->mutable_typed_data()[0] = rows;
    require(lane->active->ensureOnDevice(device, stream), "admit original device row owner");
    if(collective_witness==OriginalFFNCollectiveWitness::DiagnosticSideband)
    {
        lane->sideband = TestTensorFactory::createINT32({1});
        lane->sideband->mutable_typed_data()[0] = member + 1;
        require(lane->sideband->ensureOnDevice(device, stream), "admit original final-collective sideband");
    }
    if(hidden_q8!=OriginalM1HiddenQ8FixturePolicy::CountedOriginal&&(rows!=1||grouped))
        throw std::invalid_argument("Original hidden Q8 fixture admits only fully-active ordinary M1");
    const auto *count = hidden_q8==OriginalM1HiddenQ8FixturePolicy::CountedOriginal
        ?static_cast<const int32_t *>(lane->active->gpu_data_ptr()):nullptr;
    lane->owner = std::make_shared<MoERoutedPipelineKernelOwner>();
    if(hidden_q8==OriginalM1HiddenQ8FixturePolicy::FullActiveReuse)
        require(lane->owner->router_q8_publication->requireOriginalM1Consumer(lane->hidden.get(),kH),"declare original M1 router/shared source");
    if(published) {
        const bool model_retained=epoch_graph==EpochGraphLowering::ModelRetained;
        require(rows==1&&!grouped&&(native_layout_router||model_retained)==
            (execution==MoEIntermediateShardFFNExecution::ResidentCandidate)&&
            (!model_retained||(!native_layout_router&&bool(prefix)&&alias_probe==NativeRouterAliasProbe::None)),
            "promoted replay requires its exact original staged, native-layout or late model-lowered M1 identity");
        require(published->runtime && published->bank &&
            (epoch_admission==PublishedFFNEpochAdmission::InitialPublished?published->epoch()==1:published->epoch()>1) &&
            &published->bank->layerForModelLayer(0)==&source &&
            published->runtime->deviceId()==device && published->runtime->usesOverlayEpochTicket() &&
            published->runtime->hostLayerState(0).active_epoch==published->epoch(),
            "promoted FFN borrows the exact published candidate bank and device runtime");
        lane->runtime=published->runtime;
    } else {
        lane->runtime = std::make_shared<DeviceMoERuntimeTable>(DeviceMoERuntimeTable::Config{
            .device_id = device, .num_layers = 1, .num_experts = kExperts, .top_k = kTopK,
            .mirror_to_device = true, .prefill_token_capacity = rows, .fragment_families = {source.fragment_family}});
        const auto &original_owners = *source.fragment_family->preparation()->jointOwnerMap();
        publishInitialMoEIntermediateShardRuntimeLayer(*lane->runtime, original_owners, source, 0, stream);
        require(lane->runtime->hasInitialLayerRuntimeState(0), "publish original fragment runtime");
        const auto initial_bank = lane->runtime->hostLayerState(0).active_bank;
        const auto initial_epoch = lane->runtime->hostLayerState(0).active_epoch;
        // Each retained row family can encounter the same original cold source.
        // Re-enter the production publisher before and after sealing; neither call
        // may rewind its epoch or reconstruct descriptor storage behind a graph.
        publishInitialMoEIntermediateShardRuntimeLayer(*lane->runtime, original_owners, source, 0, stream);
        lane->runtime->sealAndPublishCompleteInitialRuntimeState(stream);
        publishInitialMoEIntermediateShardRuntimeLayer(*lane->runtime, original_owners, source, 0, stream);
        require(lane->runtime->hostLayerState(0).active_bank == initial_bank &&
            lane->runtime->hostLayerState(0).active_epoch == initial_epoch, "retain original cold publication across graph declarations");
        EXPECT_THROW(publishInitialMoEIntermediateShardRuntimeLayer(
            *lane->runtime, original_owners, source, 0, nullptr), std::invalid_argument);
    }
    const auto *router_weight = shared_weights.optionalLayerForDevice(0, "ffn_gate_inp.weight", device);
    require(router_weight && router_weight->identity.role == WeightRole::MoERouter, "original router source binding");
    MoERoutingStage::Params router;
    router.device_id = device; router.layer_idx = 0; router.input = lane->hidden.get(); router.gate_weights = router_weight->tensor;
    router.seq_len = rows; router.d_model = kH; router.num_experts = kExperts; router.top_k = kTopK;
    router.output_indices = lane->indices.get(); router.output_weights = lane->weights.get();
    router.moe_runtime_table = lane->runtime.get(); router.routed_pipeline_kernel_owner = lane->owner;
    router.routed_row_execution_policy = RoutedExpertRowExecutionPolicy::IntermediateShardLocal;
    router.collect_device_runtime_histogram = !grouped; router.active_row_count_device = count;
    router.force_decode_equivalent_verifier_prefill = grouped;
    router.force_grouped_verifier_prefill_for_decode = grouped && rows == 1;
    router.grouped_verifier_histogram_role =
        selectMoEGroupedVerifierHistogramRole(grouped, /*collects_history=*/false);
    if(alias_probe!=NativeRouterAliasProbe::None) {
        require(native_layout_router,"alias adversary requires actual native-layout owner admission");
        auto output=alias_probe==NativeRouterAliasProbe::RouterOutput?lane->indices:lane->shared;
        lane->alias_probe=std::make_shared<NativeRouterAliasTensor>(output,router_weight->tensor->gpu_data_ptr());
        if(alias_probe==NativeRouterAliasProbe::RouterOutput)router.output_indices=lane->alias_probe.get();
    }
    lane->graph.addNode("router", std::make_unique<MoERoutingStage>(router), device);
    MoEIntermediateShardPipelineParams local;
    local.family = source.fragment_family; local.runtime = lane->runtime.get(); local.kernel_owner = lane->owner;
    local.rows = rows; local.top_k = kTopK;
    local.row_mode = grouped ? MoEIntermediateShardRowMode::Grouped : MoEIntermediateShardRowMode::Decode;
    local.gate_up_publication=gate_up_publication;
    local.down_publication=down_publication;
    local.phase = grouped ? MoEOverlayServicePhaseHint::GroupedVerifier : MoEOverlayServicePhaseHint::Decode;
    local.live_rows = count?DeviceRowRange::deviceCounted(rows,count):DeviceRowRange::fullyActive(rows); local.prefix = "routed/"; local.router_node = "router";
    local.hidden = {lane->hidden.get(), BufferId::NORMALIZED}; local.routing_indices = {alias_probe==NativeRouterAliasProbe::RouterOutput?static_cast<ITensor*>(lane->alias_probe.get()):lane->indices.get(), BufferId::MOE_EXPERT_INDICES};
    local.routing_weights = {lane->weights.get(), BufferId::MOE_EXPERT_WEIGHTS};
    local.route_partials = {lane->routes.get(), BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS};
    local.output = {lane->routed.get(), BufferId::MOE_COMBINED_OUTPUT};
    const auto routed = appendMoEIntermediateShardPipeline(lane->graph, local, source);
    SharedExpertFFNStage::Params shared;
    shared.device_id = device; shared.input = lane->hidden.get();
    shared.output = alias_probe==NativeRouterAliasProbe::FFNOutput?static_cast<TensorBase*>(lane->alias_probe.get()):lane->shared.get();
    shared.seq_len = rows; shared.d_model = kH; shared.intermediate = shared_f;
    shared.gate_scratch = lane->gate_scratch.get(); shared.up_scratch = lane->up_scratch.get();
    shared.active_row_count_device = count;
    if(hidden_q8!=OriginalM1HiddenQ8FixturePolicy::CountedOriginal)shared.phase=MoEOverlayServicePhaseHint::Decode;
    // GPU shared verification uses the production grouped-table contract.
    // The similarly named dense-GEMM diagnostic flag rejects GPU capture.
    shared.force_decode_equivalent_verifier_prefill = false;
    shared.force_grouped_verifier_prefill_for_decode = grouped;
    if (grouped||hidden_q8==OriginalM1HiddenQ8FixturePolicy::FullActiveReuse)
        shared.required_router_q8_publication = lane->owner->router_q8_publication;
    shared.prepared_store = &store; shared.output_role = SharedExpertOutputRole::ParticipantLocalPartial; shared.phase = local.phase;
    shared.input_buffer_id = BufferId::NORMALIZED; shared.output_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
    shared.gate_scratch_buffer_id = BufferId::MOE_GATE_SCRATCH; shared.up_scratch_buffer_id = BufferId::MOE_UP_SCRATCH;
    const auto binding = [&](const char *name) -> const WeightBinding & {
        const auto *result = shared_weights.optionalLayerForDevice(0, name, device);
        require(result != nullptr, std::string("original shared source: ") + name); return *result;
    };
    const auto &gate = binding("ffn_gate_shexp.weight"), &up = binding("ffn_up_shexp.weight"), &down = binding("ffn_down_shexp.weight");
    shared.gate_w = gate.tensor; shared.up_w = up.tensor; shared.down_w = down.tensor;
    shared.prepared_ref_gate = store.preparedRefForBinding(gate.binding_id, device);
    shared.prepared_ref_up = store.preparedRefForBinding(up.binding_id, device);
    shared.prepared_ref_down = store.preparedRefForBinding(down.binding_id, device);
    if (grouped)
    {
        // Mirror the model's cold consumer declaration. Merely sharing the
        // publication pointer lets a Q8 router mask a missing FP16/BF16 producer.
        DeviceMoEExpertDescriptor descriptor;
        require(shared.prepared_ref_gate && shared.prepared_ref_up && shared.prepared_ref_down &&
            exportDeviceMoEExpertWeightDescriptors(store.gemmKernel(*shared.prepared_ref_gate),
                store.gemmKernel(*shared.prepared_ref_up), store.gemmKernel(*shared.prepared_ref_down),
                kH, shared_f, descriptor), "original shared projection descriptors");
        if (!deviceMoEWeightFormatIsFloating(descriptor.weight_format))
            require(lane->owner->router_q8_publication->requireConsumerRows(rows, kH),
                "cold shared consumer requires the exact router Q8 row extent");
    }
    if(published&&shape.experts==512) {
        require(shared.prepared_ref_gate&&shared.prepared_ref_up&&shared.prepared_ref_down,
            "native epoch missing original prepared shared refs gate/up/down: "+
            std::to_string(bool(shared.prepared_ref_gate))+std::to_string(bool(shared.prepared_ref_up))+
            std::to_string(bool(shared.prepared_ref_down)));
        require(store.modelId()==source.fragment_family->model(),"native epoch original shared model identity mismatch");
        require(shared.d_model==routed.model_columns&&shared.seq_len==local.rows&&shared.active_row_count_device==local.live_rows->countOwner(),
            "native epoch original shared geometry/row authority mismatch");
    }
    auto shared_stage=std::make_unique<SharedExpertFFNStage>(shared);
    if(hidden_q8!=OriginalM1HiddenQ8FixturePolicy::CountedOriginal) {
        lane->hidden_q8_shared_kernel=llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
        shared_stage->setMoEKernelForTesting(lane->hidden_q8_shared_kernel.get());
    }
    lane->graph.addNode("shared",std::move(shared_stage),device);
    // Production retains this edge because routed and shared work reuse the
    // original scratch. It also declares the exact local-region sink.
    lane->graph.addDependency("shared",routed.producer);
    MoEIntermediateShardFFNFinalization finish;
    finish.shared_ffn_node = "shared"; finish.shared_weights = &shared_weights;
    finish.shared_partition = tp.degree() > 1 ? &shared_partition : nullptr;
    finish.local_combined = {lane->combined.get(), BufferId::FFN_OUTPUT};
    finish.prefix = "m" + std::to_string(rows) + (grouped ? "_verifier/" : "_decode/");
    finish.final_sum.device_id = device; finish.final_sum.tp_ctx = &tp; finish.final_sum.tensor = lane->combined.get();
    finish.final_sum.count = std::size_t(rows) * kH; finish.final_sum.precision = "fp32";
    finish.final_sum.arithmetic_policy = TPAllreduceArithmeticPolicy::CanonicalRankOrder;
    finish.final_sum.tensor_buffer_id = BufferId::FFN_OUTPUT; finish.final_sum.sideband_device_index = member;
    // The original finalizer validates one row authority for the complete
    // compute declaration. A fully-active M1 has no counted receipt here.
    finish.final_sum.live_rows=NativeCollectiveRows(*local.live_rows,kH);
    if(hidden_q8==OriginalM1HiddenQ8FixturePolicy::CountedOriginal)
        finish.final_sum.live_rows=NativeCollectiveRows(*local.live_rows,kH).withPayloadReceipt(
            static_cast<unsigned long long*>(lane->receipt->gpu_data_ptr()));
    if(lane->sideband)
        finish.final_sum.sidebands.push_back({.kind = LocalTPCollectiveSidebandKind::AllreduceSum,
            .recv_buffer = lane->sideband->gpu_data_ptr(), .element_count = 1, .dtype = CollectiveDataType::INT32,
            .name = "original_ffn_fixture_sideband"});
    if(execution==MoEIntermediateShardFFNExecution::Staged)
    {
        proveIntermediateShardFFNFinalizationDeclaration(lane->graph,routed,finish);
        require(lane->graph.size()==6,"exactly one router, two routed stages, shared FFN, gate-add and final sum");
    }
    else
    {
        finish.execution=execution;
        const auto output=finalizeMoEIntermediateShardFFN(lane->graph,routed,finish);
        require(lane->graph.size()==3,"resident graph keeps router, one local program and explicit final sum");
        require(lane->graph.getNode(output.producer)->stage->isCollectiveStage() &&
            lane->graph.getNode(output.producer)->dependencies==std::vector<std::string>{"shared"},
            "resident final collective remains explicit and consumes the original local sink");
        require(!lane->graph.getNode("shared")->stage->isCollectiveStage(),"resident local program contains no collective");
    }
    if(native_layout_router) {
        require(execution==MoEIntermediateShardFFNExecution::ResidentCandidate,"native-layout router FFN requires original resident owner");
        lowerNativeLayoutRouterFFN(lane->graph,"shared");
        require(lane->graph.size()==2,"native-layout router FFN retains local program and original final collective");
    }
    if(published) {
        const auto& arena=published->arena;
        require(arena && arena.get()==lane->runtime->overlayEpochArena() &&
            published->epoch()==lane->runtime->hostLayerState(0).active_epoch,
            "promoted FFN epoch arena preserves its actual publication");
        MoEOverlayEpochBoundaryStage::Params acquire;
        acquire.device_id=device; acquire.arena=arena;
        acquire.operation=MoEOverlayEpochBoundaryStage::Operation::Acquire;
        acquire.stage_name="promoted_ffn_epoch_acquire";
        auto release=acquire;
        release.operation=MoEOverlayEpochBoundaryStage::Operation::Release;
        release.stage_name="promoted_ffn_epoch_release";
        lane->graph.addNode("promoted_epoch_acquire",std::make_unique<MoEOverlayEpochBoundaryStage>(acquire),device);
        lane->graph.addDependency(native_layout_router?"shared":"router","promoted_epoch_acquire");
        lane->graph.addNode("promoted_epoch_release",std::make_unique<MoEOverlayEpochBoundaryStage>(release),device);
        lane->graph.addDependency("promoted_epoch_release",finish.prefix+"ffn_sum");
    }
    if(hidden_q8!=OriginalM1HiddenQ8FixturePolicy::CountedOriginal) {
        // Only the test observation rebinds the already-authored final sum,
        // before workspace preparation or capture. Compute/router/shared rows
        // retain their original FullyActive identity. The original device live1
        // scalar authenticates exactly the same H payload and its byte receipt.
        auto* node=lane->graph.getNode(finish.prefix+"ffn_sum");
        auto* sum=node?dynamic_cast<TPAllreduceStage*>(node->stage.get()):nullptr;
        require(sum,"original hidden Q8 final native collective owner");
        auto observation=sum->params();
        observation.live_rows=NativeCollectiveRows(DeviceRowRange::deviceCounted(rows,
            static_cast<const int32_t*>(lane->active->gpu_data_ptr())),kH).withPayloadReceipt(
                static_cast<unsigned long long*>(lane->receipt->gpu_data_ptr()));
        sum->setParams(observation); // Validates and commits through the original stage API.
    }
    // Optional full-region diagnostic composes original HC nodes before the
    // existing cold quote/leases/capture. Empty preserves every old fixture.
    std::map<BufferId,TensorBase*> prefix_arenas;
    if(prefix) {
        prefix(*lane,prefix_arenas);
        // Newly appended original HC stages have not passed through the model
        // executor. Attach its exact fixture stream before workspace consumers
        // materialize kernels or cold preparation validates their execution.
        for(const auto& name:lane->graph.getExecutionOrder()) {
            auto* stage=lane->graph.getNode(name)->stage.get();
            stage->setGPUStream(stream);
            require(stage->requireGPUStream()==stream,"full-region stage retains exact preparation stream: "+name);
        }
    }
    std::map<BufferId, TensorBase *> arenas{{BufferId::NORMALIZED, lane->hidden.get()},
        {BufferId::MOE_EXPERT_INDICES, lane->indices.get()}, {BufferId::MOE_EXPERT_WEIGHTS, lane->weights.get()},
        {BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS, lane->routes.get()}, {BufferId::MOE_COMBINED_OUTPUT, lane->routed.get()},
        {BufferId::MOE_SHARED_EXPERT_OUTPUT, lane->shared.get()}, {BufferId::FFN_OUTPUT, lane->combined.get()},
        {BufferId::MOE_GATE_SCRATCH, lane->gate_scratch.get()}, {BufferId::MOE_UP_SCRATCH, lane->up_scratch.get()}};
    for(const auto& [id,tensor]:prefix_arenas) {
        require(!arenas.contains(id),"full-region prefix has distinct original arena identities");
        arenas.emplace(id,tensor);
    }
    for(const auto& [id,tensor]:arenas)
        require(lane->output_arena.registerExternalBuffer(id,tensor),"register exact original cold demand tensor");
    lane->output_demand_executor.setArena(&lane->output_arena);
    if(observation==FullProductionHCObservation::ObservedAll||observation==FullProductionHCObservation::HCUpOnly) {
        lane->output_demand_executor.setSnapshotCallback([](const auto&,const auto&) {
            require(false,"cold demand selection must not execute a snapshot callback");
        });
        if(observation==FullProductionHCObservation::HCUpOnly) {
            require(lane->hc_up_observation,"singleton observation retains the original HC-up tensor");
            lane->output_demand_executor.setSnapshotStageFilter([tensor=lane->hc_up_observation](const auto&,const StageDumpInfo& info) {
                return std::any_of(info.outputs.begin(),info.outputs.end(),[&](const auto& output){return output.tensor==tensor;});
            });
        }
    }
    // Same production executor selector as ordinary capture, before requirement
    // collection, storage/native preparation, signatures or a captured command.
    lane->output_demand_executor.sealGraphOutputDemands(lane->graph);
    WorkspaceRequirements requirements;
    for (const auto &name : lane->graph.getExecutionOrder())
        if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(lane->graph.getNode(name)->stage.get())) {
            if(prefix) {
                if(auto* gemm=dynamic_cast<GEMMStage*>(lane->graph.getNode(name)->stage.get())) {
                    const auto& params=gemm->getParams();
                    requirements.merge(consumer->getWorkspaceRequirements(params.m,params.n,params.k));
                    continue;
                }
            }
            requirements.merge(consumer->getWorkspaceRequirements(rows));
        }
    if(hidden_q8!=OriginalM1HiddenQ8FixturePolicy::CountedOriginal) {
        require(memory&&hidden_q8_workspace,"original M1 lanes require their admitted exact workspace quote");
        requireOriginalM1HiddenQ8Workspace(requirements,hidden_q8_workspace->requirements);
    }
    std::optional<SerialWorkspaceFamilyPlan> completion_plan;
    WorkspaceRequirements completion_prefill;
    if(completion_family==CompletionSerialFamilyReplay::PrefillScratchBeforeDecode)
    {
        require(device.is_rocm()&&rows==1&&!grouped&&!hidden_q8_workspace&&!memory&&
            (gate_up_publication==MoEGateUpPublicationPolicy::LastProducer||
             down_publication==MoEDownPublicationPolicy::LastProducerRouteFold),
            "completion family regression requires its original ROCm M1 completion candidate");
        const int local_f=int(DeviceMoEIntermediateShardBinding::fromOriginalFamily(*source.fragment_family).localIntermediateColumns());
        completion_prefill=MoEWorkspaceBuffers::rocmMoE(kMaxRows,kH,std::max(local_f,shared_f),kExperts,kTopK);
        completion_plan=DeviceWorkspaceManager::planSerialFamily({requirements,completion_prefill});
        require(completion_plan->valid(),"original completion serial-family canonical physical plan");
        for(const char* counter:{MoEWorkspaceBuffers::GATEUP_TILE_COMPLETION,MoEWorkspaceBuffers::DOWN_TILE_COMPLETION})
            if(requirements.find(counter))
            {
                const auto* kept=completion_plan->find(counter);
                require(kept&&kept->descriptor.content_lifetime==WorkspaceContentLifetime::SerialGraphFamily,
                    "completion epoch must remain live across every serial graph participant");
                for(const auto& other:completion_plan->placements)if(other.descriptor.name!=counter)
                    require(!(kept->offset<other.offset+other.descriptor.size_bytes&&
                        other.offset<kept->offset+kept->descriptor.size_bytes),
                        "initialize-once completion epoch aliases a serial participant");
            }
    }
    lane->hidden_q8_workspace_bytes=hidden_q8_workspace?hidden_q8_workspace->physical.totalPhysicalBytes()
        :completion_plan?completion_plan->totalPhysicalBytes():requirements.total_bytes_with_alignment();
    lane->workspace = memory
        ?std::make_unique<DeviceWorkspaceManager>(device,lane->hidden_q8_workspace_bytes,memory)
        :std::make_unique<DeviceWorkspaceManager>(device,lane->hidden_q8_workspace_bytes);
    require(hidden_q8_workspace?lane->workspace->allocateSerialFamily(hidden_q8_workspace->physical)
        :completion_plan?lane->workspace->allocateSerialFamily(*completion_plan)
        :lane->workspace->allocate(requirements),"admit exact named production stage workspace");
    const auto hidden_q8_claim=memory?memory->claimedBytes(device,
        PhysicalMemoryOwner::ExecutionWorkspace,PhysicalMemoryMaterializationKind::NewAllocation):0;
    if(gate_up_publication==MoEGateUpPublicationPolicy::LastProducer)
    {
        // Leave a queued nonzero producer on the setup stream. The cold
        // metadata initializer must order behind it and prove zero completion;
        // the first actual consumer is an imported parent on another stream.
        auto* backend=getBackendFor(device);
        require(backend&&backend->memset(lane->workspace->getBuffer(MoEWorkspaceBuffers::GATEUP_TILE_COMPLETION),
            0xff,lane->workspace->getBufferSize(MoEWorkspaceBuffers::GATEUP_TILE_COMPLETION),device.ordinal,stream),
            "queue dirty original counter storage before cold publication");
    }
    if(down_publication==MoEDownPublicationPolicy::LastProducerRouteFold)
    {
        auto* backend=getBackendFor(device);
        require(backend&&backend->memset(lane->workspace->getBuffer(MoEWorkspaceBuffers::DOWN_TILE_COMPLETION),
            0xff,lane->workspace->getBufferSize(MoEWorkspaceBuffers::DOWN_TILE_COMPLETION),device.ordinal,stream),
            "queue dirty two-level down counters before exact cold metadata publication");
    }
    for (const auto &name : lane->graph.getExecutionOrder())
    {
        auto *stage = lane->graph.getNode(name)->stage.get();
        if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(stage)) consumer->bindWorkspace(lane->workspace.get());
        const auto declaration_identity=stage->graphCaptureVariantSignature();
        require(stage->prepareGraphLaunch(&context, stream), "cold stage preparation: " + name);
        require(stage->prepareGraphLaunch(&context, stream), "idempotent stage preparation: " + name);
        if(execution==MoEIntermediateShardFFNExecution::ResidentCandidate && name=="shared")
            require(stage->graphCaptureVariantSignature()==declaration_identity,"materialization preserves resident declarative identity");
    }
    if(memory) {
        const auto after=memory->claimedBytes(device,
            PhysicalMemoryOwner::ExecutionWorkspace,PhysicalMemoryMaterializationKind::NewAllocation);
        require(after==hidden_q8_claim,"cold Q8 publication/read leases add no physical workspace claim");
        std::cout<<"ORIGINAL_M1_HIDDEN_Q8_PMA,"<<device.toString()<<','<<static_cast<int>(hidden_q8)<<','
            <<lane->hidden_q8_workspace_bytes<<','<<hidden_q8_claim<<','<<after<<'\n';
    }
    if(completion_plan)
    {
        // These six canonical prefill arithmetic planes are disposable. No
        // descriptor/cache/source table or counter is cleared by this retained
        // alternate participant. Its large writes reproduce the intervening
        // prefill scratch lifetime that a flat M1-only fixture never exercises.
        auto* backend=getBackendFor(device);require(backend,"completion prefill scratch backend");
        lane->serial_prefill_scratch=worker.createGraphCapture(stream);
        require(bool(lane->serial_prefill_scratch),"completion alternate participant capture owner");
        ScopedBackendGraphCapture recording(worker,*lane->serial_prefill_scratch,"original serial prefill scratch participant");
        require(recording.begin(),"completion prefill scratch capture begin");
        for(const char* name:{MoEWorkspaceBuffers::PREFILL_A_INT8,MoEWorkspaceBuffers::PREFILL_A_SCALES,
            MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8,MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES,
            MoEWorkspaceBuffers::PREFILL_GATE,MoEWorkspaceBuffers::PREFILL_UP})
        {
            const auto* descriptor=completion_prefill.find(name);
            require(descriptor&&descriptor->size_bytes,"original prefill arithmetic plane declaration");
            require(backend->memset(lane->workspace->getBuffer(name),0xa5,descriptor->size_bytes,device.ordinal,stream),
                "capture original alternate prefill scratch extent");
        }
        recording.finish();require(lane->serial_prefill_scratch->instantiate(),"completion alternate participant instantiate");
        require(lane->serial_prefill_scratch->nodeCount()>0,"completion alternate participant must write native scratch");
    }
    lane->plan = capturePlan(lane->graph, arenas);
    return lane;
}

/** @brief Record only prepared production operations after the setup source reader has retired. */
void captureLane(Lane &lane, bool grouped, DeviceId device, int member, IWorkerGPUContext &worker,
    IDeviceContext &context, void *stream, std::barrier<> &rendezvous)
{
    lane.capture = worker.createGraphCapture(stream);
    require(lane.capture != nullptr, "construct native finalizer capture");
    // Shared prepared engines are model-owned. Rebind this lane's admitted
    // workspace before recording its fixed pointers; no model math executes.
    for (const auto &name : lane.graph.getExecutionOrder())
    {
        auto *stage = lane.graph.getNode(name)->stage.get();
        if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(stage)) consumer->bindWorkspace(lane.workspace.get());
        require(stage->prepareGraphLaunch(&context, stream), "idempotent exact-lane capture preparation: " + name);
    }
    rendezvous.arrive_and_wait();
    GraphCaptureDependencyLedger ledger(device, stream, lane.plan, "original_local_ffn",
        GraphCaptureDependencyLedger::ExternalInputAuthority::BindDeclaredAddressesOnly);
    ScopedBackendGraphCapture recording(worker, *lane.capture, "original_local_ffn", &ledger);
    require(recording.begin(), "begin cold production finalizer graph");
    for (const auto &name : lane.graph.getExecutionOrder())
    {
        auto *stage = lane.graph.getNode(name)->stage.get();
        ScopedGraphCaptureStage scope(stage);
        try { require(stage->execute(&context), "capture stage returned false"); }
        catch (const std::exception &error) {
            throw std::runtime_error("Original finalizer stage=" + name + " device=" + device.toString() +
                " member=" + std::to_string(member) + " rows=" + std::to_string(lane.rows) +
                " grouped=" + std::to_string(grouped) + ": " + error.what());
        }
        scope.complete();
    }
    recording.finish(); require(lane.capture->instantiate(), "instantiate actual local FFN graph");
    rendezvous.arrive_and_wait();
}

/**
 * @brief Run the production grouped graph against a diagnostic serial graph on one worker.
 * @param shape Original source geometry for both retained captures.
 * @param capacities Exact immutable grouped row capacities to qualify.
 */
void replayParticipant(DeviceId device, int member, ILocalTPContext &tp, std::barrier<> &rendezvous,
    MoEOverlayParticipantResidencyRegistry &banks, int original_member,
    const FrozenModelWeightSet &shared, const TensorParallelConfig &shared_partition, PreparedWeightStore &store,
    OriginalFFNFixtureGeometry shape,std::span<const int> capacities)
{
    const int hidden=shape.hidden;
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&] {
        auto *local_tp = dynamic_cast<LocalTPContext *>(&tp);
        require(local_tp != nullptr, "original native LocalTP implementation");
        local_tp->setCurrentDeviceIndex(member);
        auto *stream = worker.getOrCreateAuxiliaryStream("original_intermediate_ffn_finalizer");
        auto context = IDeviceContext::create(device, 1);
        auto *backend = getBackendFor(device);
        const auto build = [&](int rows, bool grouped) {
            auto frame = banks.initialJointLayer(original_member, 0);
            require(frame != nullptr, "original joint layer");
            auto setup = frame->acquireInitialConstruction();
            require(bool(setup), "original setup-only source reader");
            auto lane = makeLane(rows, grouped, device, member, tp, worker, *context, stream,
                setup->layerForModelLayer(0), shared, shared_partition, store,
                MoEIntermediateShardFFNExecution::Staged,MoEGateUpPublicationPolicy::Staged,
                OriginalM1HiddenQ8FixturePolicy::CountedOriginal,{}, {},MoEDownPublicationPolicy::Staged,
                CompletionSerialFamilyReplay::Disabled,shape);
            // Runtime and stages retain only the original closed family. No
            // initial epoch reader survives into recording or request replay.
            setup = {};
            captureLane(*lane, grouped, device, member, worker, *context, stream, rendezvous);
            return lane;
        };
        auto serial = build(1, false);
        for (const int rows : capacities)
        {
            auto candidate = build(rows, true);
            std::vector<float> input(std::size_t(rows) * hidden), expected(input.size()), actual(input.size());
            for (int row = 0; row < rows; ++row)
            {
                for (int column = 0; column < hidden; ++column)
                    input[std::size_t(row) * hidden + column] = column == 0 ? 1.f + float(row % 7) / 16.f
                        : float((row * 5 + column * 3) % 17 - 8) / 128.f;
                const int live = 1, sideband = member + 1;
                require(backend->hostToDeviceOnStream(serial->sideband->gpu_data_ptr(), &sideband, sizeof(sideband),
                    device.ordinal, stream), "reset original M1 collective sideband");
                require(backend->hostToDeviceOnStream(serial->hidden->gpu_data_ptr(), input.data() + std::size_t(row) * hidden,
                    hidden * sizeof(float), device.ordinal, stream), "publish diagnostic serial input");
                require(backend->hostToDeviceOnStream(serial->active->gpu_data_ptr(), &live, sizeof(live),
                    device.ordinal, stream), "publish diagnostic serial count");
                rendezvous.arrive_and_wait();
                require(serial->capture->launch(), "launch real M1 FFN oracle graph");
                require(backend->deviceToHostOnStream(expected.data() + std::size_t(row) * hidden,
                    serial->combined->gpu_data_ptr(), hidden * sizeof(float), device.ordinal, stream), "observe terminal M1 FFN oracle");
                require(worker.synchronizeStreamChecked(stream), "join diagnostic M1 observation");
                rendezvous.arrive_and_wait();
            }
            require(std::any_of(expected.begin(), expected.end(), [](float value) { return value != 0.f; }), "nonzero complete FFN witness");
            const auto capture_identity = [&] {
                std::vector<uint64_t> signatures;
                for (const auto &name : candidate->graph.getExecutionOrder())
                    signatures.push_back(candidate->graph.getNode(name)->stage->graphCaptureVariantSignature());
                return signatures;
            };
            const auto original_identity = capture_identity();
            for (int replay = 0; replay < 20; ++replay)
            {
                const std::array counts{rows, 0, rows > 1 ? rows - 1 : 0, std::min(1, rows), rows / 2, rows};
                const int live = counts[replay % counts.size()];
                if (replay == 7 || replay == 13)
                    for (const auto &name : candidate->graph.getExecutionOrder())
                        candidate->graph.getNode(name)->stage->resetSessionStatePreservingCapturedReplay();
                require(capture_identity() == original_identity, "request reset preserves complete capture identity");
                for (auto *tensor : {candidate->routes.get(), candidate->routed.get(), candidate->shared.get(), candidate->combined.get(),
                        candidate->gate_scratch.get(), candidate->up_scratch.get()})
                    require(backend->memset(tensor->gpu_data_ptr(), 0xa5, tensor->size_bytes(), device.ordinal, stream), "dirty prior activation bytes");
                require(backend->hostToDeviceOnStream(candidate->hidden->gpu_data_ptr(), input.data(), input.size() * sizeof(float),
                    device.ordinal, stream), "publish recorded candidate input");
                require(backend->hostToDeviceOnStream(candidate->active->gpu_data_ptr(), &live, sizeof(live),
                    device.ordinal, stream), "publish sole device-owned live count");
                require(backend->memset(candidate->receipt->gpu_data_ptr(), 0, candidate->receipt->size_bytes(),
                    device.ordinal, stream), "reset passive native outgoing-byte receipt");
                const int sideband = member + 1;
                require(backend->hostToDeviceOnStream(candidate->sideband->gpu_data_ptr(), &sideband, sizeof(sideband),
                    device.ordinal, stream), "publish original final-sum sideband");
                rendezvous.arrive_and_wait();
                require(candidate->capture->launch(), "replay actual shared+routed native FFN graph");
                unsigned long long sent = 0; int sideband_sum = 0;
                require(backend->deviceToHostOnStream(actual.data(), candidate->combined->gpu_data_ptr(), actual.size() * sizeof(float),
                    device.ordinal, stream), "observe terminal FFN result");
                require(backend->deviceToHostOnStream(&sent, candidate->receipt->gpu_data_ptr(), sizeof(sent), device.ordinal, stream), "observe native useful-byte receipt");
                require(backend->deviceToHostOnStream(&sideband_sum, candidate->sideband->gpu_data_ptr(), sizeof(sideband_sum),
                    device.ordinal, stream), "observe final-sum sideband publication");
                require(worker.synchronizeStreamChecked(stream), "terminal complete FFN observation join");
                require(sideband_sum == tp.degree() * (tp.degree() + 1) / 2,
                    "same final FFN collective must preserve original sideband publication even for empty rows");
                const auto expected_sent = static_cast<unsigned long long>(live) * hidden * sizeof(float) * (tp.degree() - 1);
                require(sent == expected_sent, "one final native FFN sum must send exactly live H rows, including empty prefixes");
                for (std::size_t element = 0; element < actual.size(); ++element)
                {
                    const float reference = element < std::size_t(live) * hidden ? expected[element] : 0.f;
                    if (!std::isfinite(actual[element]) || std::bit_cast<uint32_t>(actual[element]) != std::bit_cast<uint32_t>(reference))
                        throw std::runtime_error("Original final FFN bytes differ: member=" + std::to_string(member) +
                            " rows=" + std::to_string(rows) + " live=" + std::to_string(live) +
                            " replay=" + std::to_string(replay) + " element=" + std::to_string(element) +
                            " actual=" + std::to_string(actual[element]) + " serial=" + std::to_string(reference));
                }
                rendezvous.arrive_and_wait();
            }
        }
    });
}

#include "MoEIntermediateShardResidentFFNReplay.inc"
#include "RealRetainedTPSourceWindow.inc"
#if defined(HAVE_ROCM)
#include "FullNativeHCFFNReplay.inc"
#include "FullNativeHCFFNEpochReplay.inc"
#endif
#include "MoEPromoted23NativeLayoutReplay.inc"
#if defined(LLAMINAR_CASTAGNA_KERNEL_AB)
#include "CastagnaFullRegionABReplay.inc"
#endif

/**
 * @brief Materialize authentic sources once and retain every original native participant together.
 * @param devices Canonical original participant devices.
 * @param singleton_auto_identity Exercise the original singleton automatic identity.
 * @param qualification Explicit numerical or diagnostic qualification path.
 * @param selected_source Optional original source-format declaration.
 * @param hidden_q8_observation Explicit hidden-Q8 replay observation policy.
 * @param local_program Original participant-local FFN program declaration.
 * @param shape Original admitted source geometry.
 * @param phase_observation Explicit external HTS diagnostic; never canonical timing.
 */
void proveOriginalFinalization(const std::vector<DeviceId> &devices, bool singleton_auto_identity = false,
    OriginalSourceQualification qualification = OriginalSourceQualification::FFNFinalization,
    const SourceFormat* selected_source = nullptr,
    LocalFFNProgram local_program=LocalFFNProgram::Original,OriginalFFNFixtureGeometry shape={},
    CastagnaPhaseObservation phase_observation=CastagnaPhaseObservation::Disabled,
    FullProductionHCProbe full_hc=FullProductionHCProbe::Disabled,
    FullProductionHCStorageReuse hc_storage_reuse=FullProductionHCStorageReuse::Distinct,
    bool full_hc_timing=false,
    FullProductionHCObservation full_hc_observation=FullProductionHCObservation::ObservedAll,
    RealRetainedTPSourceWindow* original_window=nullptr)
{
    const int kH=shape.hidden,kRoutedF=shape.routed,kSharedF=shape.shared,kExperts=shape.experts,kTopK=shape.top_k;
    require(kH>0&&kH%32==0&&kRoutedF>0&&kRoutedF%32==0&&kSharedF>0&&kSharedF%32==0&&
        kExperts>0&&kTopK>0&&kTopK<=kExperts,"positive aligned original source fixture geometry");
    require((qualification == OriginalSourceQualification::PreparedService ||
        qualification == OriginalSourceQualification::ResidentFFN ||
        qualification == OriginalSourceQualification::ResidentFFNEconomy ||
        qualification == OriginalSourceQualification::GateUpLastProducer ||
        qualification == OriginalSourceQualification::GateUpPairedK32 ||
        qualification == OriginalSourceQualification::DownLastProducer ||
        qualification == OriginalSourceQualification::OriginalM1HiddenQ8 ||
        qualification == OriginalSourceQualification::NativeLayoutRouterFFN ||
        qualification == OriginalSourceQualification::NativeLayoutRouterFFNParentReplay ||
        qualification == OriginalSourceQualification::NativeLayoutRouterFFNEconomy ||
        qualification == OriginalSourceQualification::CastagnaFullRegionAB ||
        qualification == OriginalSourceQualification::CastagnaFullRegionABEconomy ||
        qualification == OriginalSourceQualification::CastagnaFullRegionABReal ||
        qualification == OriginalSourceQualification::CastagnaFullRegionABRealEconomy ||
        qualification == OriginalSourceQualification::CastagnaFullRegionABBounded ||
        qualification == OriginalSourceQualification::CastagnaFullRegionABBoundedEconomy ||
        qualification == OriginalSourceQualification::FullNativeHCFFNFinal ||
        qualification == OriginalSourceQualification::FullNativeHCFFNMiddle ||
        qualification == OriginalSourceQualification::GroupedQ8DownReplay) == (selected_source != nullptr),
        "prepared calibration requires one original model source per process");
    require(!devices.empty(), "selected original participant family");
    const auto mpi = MPIContextFactory::global();
    const auto inventory = mpi->clusterInventory();
    require(inventory != nullptr, "original discovered rank inventory");
    const auto &rank = inventory->ranks.at(mpi->rank());
    std::vector<GlobalDeviceAddress> addresses;
    for (const auto device : devices) addresses.push_back(GlobalDeviceAddress::fromLocalDeviceId(device));
    const auto collective_backend = singleton_auto_identity ? CollectiveBackendType::AUTO
        : devices.front().is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
    const auto formats = selected_source ? std::vector<SourceFormat>{*selected_source} : sourceFormats(qualification);
    for (const auto &format : formats)
    {
        SCOPED_TRACE(format.name);
        // The component captures this complete one-layer model. Its source
        // metadata, placement, family banks and publication authority therefore
        // share one actual layer extent, without trimming resolved metadata.
        PlanningGGUFFixture file(true, false, PlanningGGUFFixture::sourceType(format.type), kRoutedF,
            std::nullopt, 8, PlanningGGUFFixture::Geometry::Directory, 1, kExperts,
            format.shared_quantized ? PlanningGGUFFixture::sourceType(format.shared_quantized->tensor_type)
                                    : GGUFTensorType::F32,
            format.up_quantized||format.down_quantized?std::optional<std::array<GGUFTensorType,3>>{std::array{
                PlanningGGUFFixture::sourceType(format.type),
                PlanningGGUFFixture::sourceType(format.up_quantized?format.up_quantized->tensor_type:format.type),
                PlanningGGUFFixture::sourceType(format.down_quantized?format.down_quantized->tensor_type:format.type)}}:std::nullopt,
            PlanningGGUFFixture::MoEDimensions{uint32_t(kH),uint32_t(kSharedF),uint32_t(kTopK)},
            format.shared_up_quantized?std::optional<GGUFTensorType>{
                PlanningGGUFFixture::sourceType(format.shared_up_quantized->tensor_type)}:std::nullopt,
            format.shared_down_quantized?std::optional<GGUFTensorType>{
                PlanningGGUFFixture::sourceType(format.shared_down_quantized->tensor_type)}:std::nullopt);
        if(original_window)original_window->copyTo(file,shape,format);
        else
#if defined(LLAMINAR_CASTAGNA_KERNEL_AB)
        if(qualification==OriginalSourceQualification::CastagnaFullRegionAB||
           qualification==OriginalSourceQualification::CastagnaFullRegionABEconomy||
           qualification==OriginalSourceQualification::CastagnaFullRegionABReal||
           qualification==OriginalSourceQualification::CastagnaFullRegionABRealEconomy||
           qualification==OriginalSourceQualification::CastagnaFullRegionABBounded||
           qualification==OriginalSourceQualification::CastagnaFullRegionABBoundedEconomy)
            writeCastagnaABOriginalWindows(file,format,shape,
                qualification==OriginalSourceQualification::CastagnaFullRegionABReal||
                qualification==OriginalSourceQualification::CastagnaFullRegionABRealEconomy?realCastagnaABInput():
                qualification==OriginalSourceQualification::CastagnaFullRegionABBounded||
                qualification==OriginalSourceQualification::CastagnaFullRegionABBoundedEconomy?
                    CastagnaABInput{.source=CastagnaABSource::BoundedCentered}:CastagnaABInput{});
        else
#endif
            writeOriginalSources(file,format,shape);
        // Birth scope precedes every native/model owner so all source engines,
        // local collectives and calibration loans retire before physical reset.
        auto tp = createLocalTPContext(addresses, {}, collective_backend);
        require(tp != nullptr, "original ordered native communicator");
        // The actual ModelContext owns the source and prepared store used by
        // every graph. No test-registered prepared handle or fabricated binding
        // substitutes for production original-source materialization.
        auto model_context = ModelContext::create(file.path(), mpi);
        require(model_context != nullptr, "genuine model context");
        auto &manager = *model_context->concreteWeightManager();
        auto &loader = model_context->concreteLoader();
        manager.setWeightShardingConfig(Qwen35MoESchemaFactory{}.getWeightShardingConfig());
        const ModelContextId model{reinterpret_cast<uint64_t>(model_context.get())};
        auto store = std::make_shared<PreparedWeightStore>(model);
        manager.setPreparedWeightStore(store);
        // Dense/shared FFN cuts use the shared source's F axis. Routed experts
        // own their independent, source-authenticated fragment cohort below.
        manager.setTensorParallelConfig(std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::fromLocalTPContext(*tp, 8, 2, kSharedF, 320)));
        auto placement = std::make_shared<MoERoutedExpertPlacementPlan>();
        placement->enabled = true; placement->topology = RoutedExpertPlacementTopology::SingleDomain;
        placement->continuation_domain = "original_ffn"; placement->shared_expert_domain = "original_ffn";
        placement->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        RoutedExpertDomain domain;
        domain.name = "original_ffn"; domain.scope = ExecutionDomainScope::RANK_LOCAL; domain.owner_rank = mpi->rank();
        domain.backend = collective_backend; domain.routed_compute_policy = RoutedExpertComputePolicy::TensorSharded;
        domain.participants = addresses;
        placement->domains.push_back(std::move(domain));
        placement->routed_tiers.push_back({.name = "original_resident", .domain = "original_ffn", .priority = 0});
        placement->placements.push_back({.layer = 0, .routed_expert_tier = std::vector<int>(kExperts, 0)});
        // Freeze the actual normalized authority before any source preparation;
        // singleton no-peer proof must not accept Unresolved as an alias.
        placement->authority_execution=resolveMoEOverlayAuthorityExecutionKind(*placement);
        const auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(placement,
            {.current_world_rank = mpi->rank()});
        const auto preparation = std::make_shared<const MoEExpertOverlayPreparationPlan>(
            MoEExpertOverlayPreparationPlan::build(*runtime_plan, loader));
        const auto parents = manager.materialize(preparation->sourceWeightPlan(loader, model));
        const auto routed = parents.projectedExpertMatrices();
        std::vector<FrozenModelWeightSet> shared;
        shared.reserve(devices.size());
        for (const auto device : devices) shared.push_back(manager.materialize(sharedPlan(model, devices, device)));
        // A genuine singleton has an identity sum, not a HOST/native wire
        // transaction. Preserve the common quote fields for larger cohorts.
        const auto collective = devices.size() > 1
            ? CollectiveMemoryEstimator::localTP(kMaxRows, kH, collective_backend)
            : decltype(CollectiveMemoryEstimator::localTP(kMaxRows, kH, collective_backend)){};
        const auto service_metadata=PlanningModelMetadata::fromLoader(loader);
        const auto& source_profile=service_metadata.memoryProfile();
        require(source_profile.d_model==kH&&source_profile.expert_count==kExperts&&
            source_profile.expert_used_count==kTopK&&source_profile.expert_feed_forward_length==kRoutedF,
            "original source metadata owns H/F/expert/topK component geometry");
        PhysicalMemoryPlanBuilder bom;
        for (const auto device : devices)
        {
            std::size_t routed_bytes = 0, maximum_source = 0;
            WeightVRAMPool primary_plan;
            for (const auto &binding : routed.bindings())
                if (binding.residency.home_device == device)
                {
                    const auto projection = binding.projection_ownership->projection(binding.identity.role);
                    const auto* role_format=binding.identity.role==WeightRole::MoEExpertUp?format.up_quantized:
                        binding.identity.role==WeightRole::MoEExpertDown?format.down_quantized:nullptr;
                    const auto identity=role_format
                        ?ExpertWeightFormat::nativeQuantised({role_format->source_codebook_id,
                            role_format->source_is_superblock,true}):format.identity;
                    routed_bytes += resolveExpertPreparedProjectionMemoryGeometry(projection.rows, projection.columns,
                        identity, GPUPreparedWeightPoolLayout::SourceNative).gpu_live_bytes;
                }
            for (const auto &local_source : shared)
            for (const auto &binding : local_source.bindings())
                if (binding.residency.home_device == device)
                {
                    // Quote the loader's separated native planes and their
                    // alignment, rather than the compact original GGUF bytes.
                    const auto *bounded = boundedGPUWeightSource(*binding.tensor);
                    const auto *native = gpuWeightSourceFormat(*binding.tensor);
                    const auto source_bytes = bounded ? bounded->sourceBytes() : binding.tensor->size_bytes();
                    if (native)
                        primary_plan.planWeight(binding.identity.canonical_name, binding.tensor->rows(), binding.tensor->cols(),
                            native->payload_bytes, canonicalDeviceVnniFormat(*native).is_asymmetric,
                            native->has_emins, source_bytes);
                    else
                    {
                        const auto type = binding.tensor->native_type();
                        require(type == TensorType::FP32 || type == TensorType::FP16 || type == TensorType::BF16,
                            "original shared GPU source requires native preparation or floating storage");
                        primary_plan.planRawWeight(binding.identity.canonical_name, binding.tensor->rows(), binding.tensor->cols(),
                            binding.tensor->size_bytes());
                    }
                }
            const auto primary_bytes = alignGPUWeightLoadAllocation(primary_plan.totalPlannedBytes());
            for (const auto &binding : parents.bindings())
                if (binding.residency.home_device == device)
                {
                    // Metadata-only expert parents intentionally retain zero
                    // physical bytes. The upload ring consumes their original
                    // selected source extent, including coalesced expert rows.
                    const auto *source = dynamic_cast<const BoundedGPUExpertSource *>(binding.tensor);
                    require(source != nullptr && source->sourceBytes() != 0,
                        "original bounded expert source for staging admission");
                    require(source->size_bytes() == 0, "expert source must retain no whole-source payload");
                    maximum_source = std::max(maximum_source, source->sourceBytes());
                }
            // Upload slots hold original source bytes, not the larger prepared
            // planes. The same pool plan retains that distinct staging extent.
            maximum_source = std::max(maximum_source, primary_plan.maximumPlannedStagingBytes());
            const auto staging = resolveGPUWeightLoadMemoryGeometry(maximum_source, configuredGPUWeightLoadMemoryPolicy());
            const auto gpu = planningObservedResource(rank, device), host = planningObservedResource(rank, DeviceId::cpu());
            bom.add(gpu, PhysicalMemoryOwner::RoutedExpertWeights, routed_bytes)
                .add(gpu, PhysicalMemoryOwner::PrimaryModelWeights, primary_bytes)
                .add(gpu, PhysicalMemoryOwner::WeightLoadStaging, staging.staging_bytes)
                .add(host, PhysicalMemoryOwner::WeightLoadStaging, staging.host_staging_bytes)
                .add(gpu, PhysicalMemoryOwner::LocalCollective, collective.perDeviceBytes());


        }
        const auto certificate=std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(bom.build());
        auto authority=std::make_shared<PhysicalMemoryAuthority>(certificate,mpi->rank());
        manager.installPhysicalMemoryAuthority(authority);
        for (const auto device : devices)
        {
            require(manager.prepareMoEExpertOverlayWeights(*runtime_plan, device, &routed, nullptr,
                PreparedWeightAdmission::AllocateCompleteSet, preparation), "prepare original routed F slices");
            const auto member = std::find(devices.begin(), devices.end(), device) - devices.begin();
            require(manager.prepareWeightsForDevice(shared[member], device), "prepare original participant-local shared F slices and gates");
            if (format.shared_up_quantized || format.shared_down_quantized) {
                // Read the loader's three original, potentially differently encoded
                // sources. Down transposes the gate/up axes and uses the local TP slice.
                require(format.shared_quantized, "asymmetric shared matrices have an original gate encoding");
                constexpr std::array names{"ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
                const std::array expected{format.shared_quantized,
                    format.shared_up_quantized ? format.shared_up_quantized : format.shared_quantized,
                    format.shared_down_quantized ? format.shared_down_quantized : format.shared_quantized};
                std::array<DeviceNativeQuantisedMatrixDesc, names.size()> actual{};
                for (std::size_t role = 0; role < names.size(); ++role) {
                    const auto* binding = shared[member].optionalLayerForDevice(0, names[role], device);
                    require(binding && binding->tensor, "shared role has an original frozen tensor binding");
                    const auto ref = store->preparedRefForBinding(binding->binding_id, device);
                    require(ref.has_value(), "shared role retains its original prepared reference");
                    const auto* engine = store->gemmKernel(*ref);
                    auto& desc = actual[role];
                    require(engine && engine->weights_converted() && engine->inspectPreparedNativeQuantisedMatrixDesc(desc) &&
                        desc.valid() && desc.n == binding->tensor->rows() && desc.k == binding->tensor->cols() &&
                        desc.codebook_id == expected[role]->device_execution_codebook_id && desc.source_identity_present &&
                        desc.source_codebook_id == expected[role]->source_codebook_id &&
                        bool(desc.source_is_superblock) == expected[role]->source_is_superblock,
                        "shared role preserves the local dimensions and exact source encoding");
                    for (std::size_t earlier = 0; earlier < role; ++earlier)
                        require(actual[earlier].payload != desc.payload && actual[earlier].scales != desc.scales,
                            "independently sourced shared weights retain distinct physical planes");
                }
            }
        }
        if (devices.size() > 1)
            require(tp->reserveCollectiveResources(collective.backend_payload_capacity_bytes,
                collective.fp16_scratch_elements, authority), "admit original native collective BOM");
        std::vector<int> original_members;
        for (const auto device : devices)
        {
            const auto request = preparation->requestFor(device, 0, 0, ExpertGemmRegistry::WeightRole::GATE);
            require(request && request->joint_source, "original per-device cohort source");
            original_members.push_back(request->joint_source->participant().participant_id);
        }
        if(shape.contract==OriginalFFNGeometryContract::QFNTensorSharded) {
            require((devices.size()==1||devices.size()==2||devices.size()==4||devices.size()==8)&&kH==2560&&kRoutedF==640&&kSharedF==640&&kExperts==512&&kTopK==10,
                "exact QFN TensorSharded original source geometry");
            std::vector<int> local_widths;
            for(const auto device:devices) {
                const auto request=preparation->requestFor(device,0,0,ExpertGemmRegistry::WeightRole::GATE);
                require(request&&request->joint_source,"exact TP original joint source");
                local_widths.push_back(request->joint_source->ownership().projection(WeightRole::MoEExpertGate).rows);
            }
            std::sort(local_widths.begin(),local_widths.end());
            const auto expected=devices.size()==1?std::vector<int>{640}:devices.size()==2?std::vector<int>{256,384}:
                devices.size()==4?std::vector<int>{128,128,128,256}:std::vector<int>{64,64,64,64,96,96,96,96};
            require(local_widths==expected,
                "production source partitioner must retain exact positive QFN TP1/2/4/8 routed widths");
            std::vector<int> shared_widths;
            for(const auto device:devices)shared_widths.push_back(manager.tensorParallelConfig()->forDevice(device).d_ff_count);
            std::sort(shared_widths.begin(),shared_widths.end());
            const auto expected_shared=devices.size()==1?std::vector<int>{640}:devices.size()==2?std::vector<int>{320,320}:
                devices.size()==4?std::vector<int>{160,160,160,160}:std::vector<int>{64,64,64,64,96,96,96,96};
            require(shared_widths==expected_shared,"original independent QFN shared source TP1/2/4/8 widths");
        }
        MoEOverlayParticipantResidencyRegistry banks({.local_participant_ids = original_members,
            .num_layers = 1, .num_experts = kExperts, .initial_epoch = 1,
            .projection_preparation = preparation, .original_model = model});
        std::string error;
        require(banks.bindOriginalPreparedContext(model_context, preparation, &error), "bind genuine original prepared model: " + error);
        for (const auto member : original_members)
            require(banks.registerInitialJointBank(manager.expertGemmRegistry(), model, member, &error), "register original family: " + error);
        if(format.up_quantized&&(qualification==OriginalSourceQualification::FullNativeHCFFNFinal||
                                 qualification==OriginalSourceQualification::FullNativeHCFFNMiddle)) {
            // Inspect the original retained fragment engines before capture;
            // never author or substitute a runtime/device descriptor. Checking
            // every expert covers whichever slots the real router later selects.
            const auto& formats=quantizedVerifierFormats();
            const auto gate=std::find_if(formats.begin(),formats.end(),[&](const auto& f){return f.tensor_type==format.type;});
            require(gate!=formats.end(),"asymmetric routed gate has its original source format");
            const std::array expected{&*gate,format.up_quantized};
            for(const auto member:original_members) {
                const auto frame=banks.initialJointLayer(member,0);
                require(bool(frame),"asymmetric routed pair retains its original layer owner");
                const auto setup=frame->acquireInitialConstruction();
                require(bool(setup),"asymmetric routed pair has its original cold construction lease");
                const auto& source=setup->layerForModelLayer(0);
                require(source.fragments.size()==std::size_t(kExperts),"asymmetric routed source covers every real expert");
                for(const auto& payload:source.fragments) {
                    std::array<DeviceNativeQuantisedMatrixDesc,2> actual;
                    require(payload.gate()&&payload.up()&&
                        payload.gate()->exportNativeQuantisedMatrixDesc(actual[0])&&
                        payload.up()->exportNativeQuantisedMatrixDesc(actual[1]),
                        "asymmetric routed pair exports the original prepared fragment engines");
                    for(unsigned role=0;role<2;++role)
                        require(actual[role].valid()&&actual[role].n==kRoutedF&&actual[role].k==kH&&
                            actual[role].codebook_id==expected[role]->device_execution_codebook_id&&
                            actual[role].source_identity_present&&
                            actual[role].source_codebook_id==expected[role]->source_codebook_id&&
                            bool(actual[role].source_is_superblock)==expected[role]->source_is_superblock&&
                            (actual[role].codebook_id!=kNativeQuantisedIQ4XSSourceScaleCodebook||actual[role].mins),
                            "asymmetric routed role retains its own physical/source identity and actual coefficient plane");
                    require(actual[0].payload!=actual[1].payload&&actual[0].scales!=actual[1].scales,
                        "asymmetric routed roles retain distinct original prepared allocations");
                }
            }
        }
        for (std::size_t member = 0; member < devices.size(); ++member)
            (void)requireOriginalStaticGPUIntermediateShardGraph(*placement, *preparation->jointOwnerMap(),
                *tp, devices[member], int(member), mpi->rank(), 0, 1, kExperts);


        std::barrier rendezvous(devices.size());
        std::vector<std::thread> threads(devices.size());
        std::vector<std::exception_ptr> errors(devices.size());
        for (std::size_t member = 0; member < devices.size(); ++member)
            threads[member] = std::thread([&, member] {
                try
                {
#if defined(HAVE_ROCM)
                    if(qualification==OriginalSourceQualification::FullNativeHCFFNFinal||
                       qualification==OriginalSourceQualification::FullNativeHCFFNMiddle)
                        replayFullNativeHCFFN(devices[member],member,*tp,rendezvous,banks,original_members[member],
                            shared[member],*manager.tensorParallelConfig(),*store,shape,
                            qualification==OriginalSourceQualification::FullNativeHCFFNMiddle?
                                FullProductionHCProbe::MiddleRead:FullProductionHCProbe::FinalRead,hc_storage_reuse,
                            format.expected_native_layout_groups,full_hc_timing,full_hc_observation,original_window);
                    else
#endif
#if defined(LLAMINAR_CASTAGNA_KERNEL_AB)
                    if(qualification==OriginalSourceQualification::CastagnaFullRegionAB||
                       qualification==OriginalSourceQualification::CastagnaFullRegionABEconomy||
                       qualification==OriginalSourceQualification::CastagnaFullRegionABReal||
                       qualification==OriginalSourceQualification::CastagnaFullRegionABRealEconomy||
                       qualification==OriginalSourceQualification::CastagnaFullRegionABBounded||
                       qualification==OriginalSourceQualification::CastagnaFullRegionABBoundedEconomy)
                        replayCastagnaFullRegionAB(devices[member],member,*tp,rendezvous,banks,
                            original_members[member],shared[member],*manager.tensorParallelConfig(),*store,
                            loader,shape,qualification==OriginalSourceQualification::CastagnaFullRegionABEconomy||
                            qualification==OriginalSourceQualification::CastagnaFullRegionABRealEconomy||
                            qualification==OriginalSourceQualification::CastagnaFullRegionABBoundedEconomy,
                            qualification==OriginalSourceQualification::CastagnaFullRegionABReal||
                            qualification==OriginalSourceQualification::CastagnaFullRegionABRealEconomy?realCastagnaABInput():
                            qualification==OriginalSourceQualification::CastagnaFullRegionABBounded||
                            qualification==OriginalSourceQualification::CastagnaFullRegionABBoundedEconomy?
                                CastagnaABInput{.source=CastagnaABSource::BoundedCentered}:CastagnaABInput{},phase_observation,full_hc,full_hc_observation);
                    else
#endif
                    if(qualification==OriginalSourceQualification::NativeLayoutRouterFFN||qualification==OriginalSourceQualification::NativeLayoutRouterFFNParentReplay||qualification==OriginalSourceQualification::NativeLayoutRouterFFNEconomy)
                        replayResidentParticipant(devices[member],member,*tp,rendezvous,banks,original_members[member],
                            shared[member],*manager.tensorParallelConfig(),*store,
                            qualification==OriginalSourceQualification::NativeLayoutRouterFFNEconomy?ResidentFFNObservation::PairedEvents:
                            qualification==OriginalSourceQualification::NativeLayoutRouterFFNParentReplay?ResidentFFNObservation::RepeatedParent:ResidentFFNObservation::Replay,
                            format.name,LocalFFNReplayCandidate::Resident,LocalFFNProgram::Original,shape,true,nullptr,
                            format.expected_native_layout_groups);
                    else if(qualification==OriginalSourceQualification::ResidentFFN || qualification==OriginalSourceQualification::ResidentFFNEconomy ||
                       qualification==OriginalSourceQualification::GateUpLastProducer || qualification==OriginalSourceQualification::GateUpPairedK32 ||
                       qualification==OriginalSourceQualification::DownLastProducer)
                        replayResidentParticipant(devices[member],member,*tp,rendezvous,banks,original_members[member],
                            shared[member],*manager.tensorParallelConfig(),*store,
                            qualification==OriginalSourceQualification::ResidentFFNEconomy?ResidentFFNObservation::PairedEvents:ResidentFFNObservation::Replay,
                            format.name,qualification==OriginalSourceQualification::GateUpPairedK32
                                ? LocalFFNReplayCandidate::GateUpPairedK32:
                              qualification==OriginalSourceQualification::GateUpLastProducer
                                ? LocalFFNReplayCandidate::GateUpLastProducer:
                              qualification==OriginalSourceQualification::DownLastProducer
                                ? LocalFFNReplayCandidate::DownLastProducer:LocalFFNReplayCandidate::Resident,local_program,shape);
                    else {
                        constexpr std::array default_counts{1,3,kMaxRows};
                        constexpr std::array q8_down_counts{3,4};
                        const std::span<const int> capacities=qualification==OriginalSourceQualification::GroupedQ8DownReplay?
                            std::span<const int>{q8_down_counts}:std::span<const int>{default_counts};
                        replayParticipant(devices[member],member,*tp,rendezvous,banks,original_members[member],
                            shared[member],*manager.tensorParallelConfig(),*store,shape,capacities);
                    }
                }
                catch (...)
                { errors[member] = std::current_exception(); tp->requestAbort(); rendezvous.arrive_and_drop(); }
            });
        for (auto &thread : threads) thread.join();
        for (const auto &failure : errors) if (failure) std::rethrow_exception(failure);
    }
}

/** @brief Original source-format component; no raw weight preparation or alternate layout is involved. */
SourceFormat nativeLayoutRouterFFNSource() {
 const auto sources=sourceFormats(OriginalSourceQualification::ResidentFFN);
 const auto gate=std::find_if(sources.begin(),sources.end(),[](const auto& s){return s.name=="Q4_K_SharedQ8_0";});
 const auto& formats=quantizedVerifierFormats();
 const auto down=std::find_if(formats.begin(),formats.end(),[](const auto& f){return f.tensor_type==TensorType::Q5_1;});
 require(gate!=sources.end()&&down!=formats.end(),"native-layout original model source combination");
 auto result=*gate;result.name="Q4_K_DownQ5_1_SharedQ8_0";result.down_quantized=&*down;return result;
}
}
} // namespace llaminar2::test

#if defined(HAVE_ROCM) && defined(LLAMINAR_CASTAGNA_KERNEL_AB)
namespace llaminar2::test {
/** @brief Run the existing source/capture owner with an explicit full production lowering. */
void replayFullProductionHCFFN(FullProductionHCProbe mode,bool real,bool timing=false,
    FullProductionHCObservation observation=FullProductionHCObservation::ObservedAll) {
    const bool retained=observation==FullProductionHCObservation::RetainedDemandedOnly;
    const char* requested_pci=retained?std::getenv("LLAMINAR_TEST_RETAINED_HC_FFN_PCI"):nullptr;
    const char* gguf=retained?std::getenv("LLAMINAR_TEST_CASTAGNA_GGUF"):nullptr;
    if(retained)require(gguf&&*gguf,"Set LLAMINAR_TEST_CASTAGNA_GGUF to the first local QFN UD-Q4_K_XL shard");
    const std::array<std::string_view,1> pci{requested_pci?requested_pci:""};
    originalRankLocalReplay(DeviceType::ROCm,1,[&](const auto& devices) {
        const auto input = realCastagnaABInput();
        auto source = nativeLayoutRouterFFNSource();
        if (real) {
            ModelLoader model;
            model.setUseMmap(true);
            require(model.loadModel(input.path), "read original source metadata for the Castagna comparison");
            const auto gate = model.nativeWeightSourceGeometry(input.name("ffn_gate_exps.weight")).original_type;
            const auto down = model.nativeWeightSourceGeometry(input.name("ffn_down_exps.weight")).original_type;
            source = RealRetainedTPSourceWindow::sourceFormat(gate, down);
        }
        const auto routed = qfn_castagna_reference::equalWeightSourceShard(640, 128, input.degree, input.participant);
        const auto shared = qfn_castagna_reference::equalWeightSourceShard(640, 64, input.degree, input.participant);
        const OriginalFFNFixtureGeometry shape{.hidden=2560,.routed=static_cast<int>(routed.count),
            .shared=static_cast<int>(shared.count),.experts=512,.top_k=10,.router_source=OriginalFFNRouterSource::Dense};
        // Inventory-only receipt: original discovery owns rank/device placement.
        // No HIP query, visibility remap or native work is added to observation.
        const auto mpi=MPIContextFactory::global();
        require(mpi&&mpi->clusterInventory()&&devices.size()==1,"real production comparator original singleton inventory");
        const auto& rank=mpi->clusterInventory()->getRank(mpi->rank());
        require(rank.rank==mpi->rank(),"real production comparator original rank identity");
        const auto gpu=std::find_if(rank.gpus.begin(),rank.gpus.end(),[&](const auto& value) {
            return value.type==DeviceType::ROCm&&value.local_device_id==devices.front().ordinal;
        });
        require(gpu!=rank.gpus.end()&&rankOwnsGPUDevice(rank,*gpu)&&!gpu->pci_bus_address.empty()&&
            std::string_view(gpu->native_isa).substr(0,gpu->native_isa.find(':'))=="gfx906",
            "real production comparator exact original owned native ISA");
        std::osyncstream(std::cout)<<"FULL_PRODUCTION_CASTAGNA_DEVICE,rank="<<mpi->rank()
            <<",native_ordinal="<<gpu->local_device_id<<",pci="<<gpu->pci_bus_address<<",native_isa="<<gpu->native_isa
            <<",H="<<shape.hidden<<",F="<<shape.routed<<",S="<<shape.shared<<",E="<<shape.experts<<",topK="<<shape.top_k
            <<",full_production_mode="<<unsigned(mode)<<",source="<<(real?"real_gguf":"bounded_original")
            <<",timing="<<timing<<'\n';
        const auto qualification=real?(timing?OriginalSourceQualification::CastagnaFullRegionABRealEconomy:
            OriginalSourceQualification::CastagnaFullRegionABReal):OriginalSourceQualification::CastagnaFullRegionABBounded;
        proveOriginalFinalization(devices,false,qualification,&source,
            LocalFFNProgram::Original,shape,CastagnaPhaseObservation::Disabled,mode,FullProductionHCStorageReuse::Distinct,false,observation);
    },"gfx906",pci.front().empty()?std::span<const std::string_view>{}:std::span<const std::string_view>{pci});
}
/** @test Opt-in closed production final-read capture versus the unchanged Castagna export. */
TEST(CastagnaFullRegionAB,DISABLED_ROCmRetainedClosedRealFinalEvents) {
    replayFullProductionHCFFN(FullProductionHCProbe::FinalRead,true,true,FullProductionHCObservation::RetainedDemandedOnly);
}
} // namespace llaminar2::test
#endif

#if defined(HAVE_ROCM)
namespace llaminar2::test {
/**
 * @brief Canonical model-free full-entry replay with original native source preparation.
 * @param qualification Original final or middle-read boundary including declared injection.
 * @param storage_reuse Explicit original model tensor/BufferId temporal alias policy.
 * @param shape Original source geometry; all roles are prepared from these actual extents.
 */
/** @brief Existing canonical model-free source geometry, shared by all observation modes. */
constexpr OriginalFFNFixtureGeometry kFullNativeHCFFNFixtureGeometry{.hidden=2560,.routed=256,.shared=192,.experts=512,.top_k=10,
    .router_source=OriginalFFNRouterSource::Dense};
void replayMandatoryFullNativeHCFFN(OriginalSourceQualification qualification,
    FullProductionHCStorageReuse storage_reuse=FullProductionHCStorageReuse::Distinct,
    OriginalFFNFixtureGeometry shape=kFullNativeHCFFNFixtureGeometry,
    FullProductionHCObservation observation=FullProductionHCObservation::ObservedAll) {
    const auto source=nativeLayoutRouterFFNSource();
    // Site-local tuning may reserve one physical card without changing rank
    // inventory, device visibility, or the default model-free test topology.
    const char* requested_pci=observation==FullProductionHCObservation::RetainedDemandedOnly?
        std::getenv("LLAMINAR_TEST_RETAINED_HC_FFN_PCI"):nullptr;
    const std::array<std::string_view,1> requested{requested_pci?requested_pci:""};
    originalRankLocalReplay(DeviceType::ROCm,1,[&](const auto& devices) {
        if(observation==FullProductionHCObservation::RetainedDemandedOnly) {
            const auto mpi=MPIContextFactory::global();
            const auto& rank=mpi->clusterInventory()->getRank(mpi->rank());
            require(devices.size()==1,"retained replay uses the original singleton rank-local coordinate");
            const auto gpu=std::find_if(rank.gpus.begin(),rank.gpus.end(),[&](const auto& entry) {
                return entry.type==DeviceType::ROCm&&entry.local_device_id==devices.front().ordinal;
            });
            require(gpu!=rank.gpus.end()&&rankOwnsGPUDevice(rank,*gpu),
                "retained replay's selected coordinate belongs to the original MPI rank");
            std::osyncstream(std::cout)<<"FULL_PRODUCTION_RETAINED_DEVICE,rank="<<mpi->rank()
                <<",pci="<<gpu->pci_bus_address<<",isa="<<gpu->native_isa<<'\n';
        }
        proveOriginalFinalization(devices,false,qualification,&source,
            LocalFFNProgram::Original,shape,CastagnaPhaseObservation::Disabled,
            FullProductionHCProbe::Disabled,storage_reuse,false,observation);
    },"gfx906",requested[0].empty()?std::span<const std::string_view>{}:std::span<const std::string_view>{requested});
}
/** @test The opt-in native writer preserves original routes, histograms and outputs on a closed retained capture. */
TEST(FullNativeHCFFN,ROCmRetainedClosedReplay) {
    replayMandatoryFullNativeHCFFN(OriginalSourceQualification::FullNativeHCFFNFinal,
        FullProductionHCStorageReuse::Distinct,kFullNativeHCFFNFixtureGeometry,FullProductionHCObservation::RetainedDemandedOnly);
}
/** @test Both native Q8-down source families keep grouped byte equality through count/reset replay. */
TEST(FullNativeHCFFN,ROCmGroupedQ8DownByteReplay)
{
    const auto qualification=OriginalSourceQualification::GroupedQ8DownReplay;
    const auto sources=sourceFormats(qualification);
    const auto& reference_formats=quantizedVerifierFormats();
    const auto q8=std::find_if(reference_formats.begin(),reference_formats.end(),[](const auto& candidate) {
        return candidate.tensor_type==TensorType::Q8_0;
    });
    require(q8!=reference_formats.end(),"original native Q8_0 grouped source");
    constexpr OriginalFFNFixtureGeometry shape{.hidden=2560,.routed=384,.shared=320,.experts=16,.top_k=10,
        .router_source=OriginalFFNRouterSource::Dense};
    const char* requested_pci=std::getenv("LLAMINAR_TEST_RETAINED_HC_FFN_PCI");
    const std::array<std::string_view,1> requested{requested_pci?requested_pci:""};
    originalRankLocalReplay(DeviceType::ROCm,1,[&](const auto& devices) {
        for(const auto gate:{TensorType::Q4_K,TensorType::Q5_K}) {
            const auto source=std::find_if(sources.begin(),sources.end(),[&](const auto& value){return value.type==gate;});
            require(source!=sources.end(),"original native grouped gate format");
            auto format=*source;
            format.shared_quantized=&*q8;
            format.down_quantized=&*q8;
            proveOriginalFinalization(devices,false,qualification,&format,LocalFFNProgram::Original,shape);
            std::osyncstream(std::cout)<<"QFN_Q8_DOWN_GROUPED_REPLAY,gate="<<format.name
                <<",down=Q8_0,grouped_capacities=3/4,replays_per_capacity=20,byte_oracle=original_serial"<<'\n';
        }
    },"gfx906",requested[0].empty()?std::span<const std::string_view>{}:std::span<const std::string_view>{requested});
}
} // namespace llaminar2::test
#endif

#if defined(HAVE_ROCM)
namespace llaminar2::test {

/** @return Exact user-selected original layer and retained TP source participant. */
RealRetainedTPSourceWindow realProductionTPInputFromEnv()
{
    const char* path=std::getenv("LLAMINAR_TEST_CASTAGNA_GGUF");
    require(path&&*path,"real TP window requires explicit original source");
    const auto coordinate=[](const char* name) {
        const char* value=std::getenv(name);
        require(value&&*value,std::string("missing original TP window coordinate ")+name);
        const std::string_view input(value);
        int number=-1;
        const auto [end,error]=std::from_chars(input.data(),input.data()+input.size(),number);
        require(error==std::errc{}&&end==input.data()+input.size()&&number>=0,
            std::string("invalid original TP window coordinate ")+name);
        return number;
    };
    // Original ordered source roster, deliberately separate from the local
    // replay owner. Neither fork cuts nor the local replay device determine F or S.
    const char* roster=std::getenv("LLAMINAR_TEST_QFN_SOURCE_ROCM_ORDINALS");
    require(roster&&*roster,"original source requires its resolved physical roster");
    std::vector<DeviceId> members;
    for(std::string_view input(roster);!input.empty();) {
        const auto separator=input.find(',');
        const auto value=input.substr(0,separator);
        int ordinal=-1;
        const auto [end,error]=std::from_chars(value.data(),value.data()+value.size(),ordinal);
        require(error==std::errc{}&&end==value.data()+value.size()&&ordinal>=0,
            "invalid resolved original source ROCm ordinal");
        members.push_back(DeviceId::rocm(ordinal));
        if(separator==std::string_view::npos) break;
        input.remove_prefix(separator+1);
        require(!input.empty(),"missing final original source ROCm ordinal");
    }
    return RealRetainedTPSourceWindow(path,coordinate("LLAMINAR_TEST_QFN_SOURCE_LAYER"),
        coordinate("LLAMINAR_TEST_QFN_SOURCE_PARTICIPANT"),members);
}

/** @test Exact original QFN TP source windows through both actual graph owners. */
TEST(FullNativeHCFFN, DISABLED_ROCmRetainedProductionTPWindow)
{
    const char* pci=std::getenv("LLAMINAR_TEST_RETAINED_HC_FFN_PCI");
    require(pci&&*pci,"real TP window requires explicit PCI");
    auto window=realProductionTPInputFromEnv();
    auto format=window.format();
    const char* mode=std::getenv("LLAMINAR_TEST_QFN_WINDOW_EVENTS");
    require(!mode||std::string_view(mode)=="1","original window event mode must be explicit");
    const bool timing=mode!=nullptr;
    if(timing) {
        const auto mask=nativeQuantisedCodebookMaskBit(canonicalDeviceVnniCodebookId(format.identity.native_quantised))|
            nativeQuantisedCodebookMaskBit(format.down_quantized->device_execution_codebook_id)|
            nativeQuantisedCodebookMaskBit(format.shared_quantized->device_execution_codebook_id);
        format.expected_native_layout_groups=rocm::ResidentMoEFFNDecoderProgram::fromExecutionMask(mask).groups();
    }
    const auto shape=window.geometry();
    const std::array<std::string_view,1> address{pci};
    originalRankLocalReplay(DeviceType::ROCm,1,[&](const auto& devices) {
        proveOriginalFinalization(devices,false,OriginalSourceQualification::FullNativeHCFFNFinal,
            &format,LocalFFNProgram::Original,shape,CastagnaPhaseObservation::Disabled,
            FullProductionHCProbe::FinalRead,FullProductionHCStorageReuse::Distinct,timing,
            FullProductionHCObservation::RetainedDemandedOnly,&window);
    },"gfx906",address);
}

/** @test Original native fragments from a real TP4 window move and are consumed after epoch publication. */
TEST(FullNativeHCFFN, DISABLED_ROCmRetainedProductionWindowEpoch)
{
    try {
        auto window=realProductionTPInputFromEnv();
        const char* pci=std::getenv("LLAMINAR_TEST_RETAINED_HC_FFN_MOVEMENT_PCIS");
        require(pci&&*pci,"real window movement requires three explicit PCI owners");
        std::string_view list(pci);
        std::array<std::string_view,3> owners;
        for(std::size_t index=0;index<owners.size();++index) {
            const auto separator=list.find(',');
            require((separator==std::string_view::npos)==(index+1==owners.size()),
                "real window movement requires exactly three original PCI identities");
            owners[index]=list.substr(0,separator);
            require(!owners[index].empty(),"real window movement PCI is nonempty");
            if(separator!=std::string_view::npos)list.remove_prefix(separator+1);
        }
        promoted23_replay::run(promoted23_replay::NumericalProgram::FullHCFFNEpoch,owners,
            promoted23_replay::EpochSource::GPUNative5_7,&window);
        if(::testing::Test::HasFailure())MPI_Abort(MPI_COMM_WORLD,2);
    } catch(const std::exception& error){ADD_FAILURE()<<error.what();MPI_Abort(MPI_COMM_WORLD,2);throw;}
}

/** @test Layer-zero raw model weights exercise four genuine TP participants through an actual arrival. */
TEST(FullNativeHCFFN, DISABLED_ROCmRetainedProductionTP4Epoch)
{
    try {
        const char* scope=std::getenv("LLAMINAR_TEST_RETAINED_HC_FFN_MOVEMENT_PCIS");
        require(scope&&*scope,"real TP4 epoch requires exact singleton, four-rank target, and singleton PCI groups");
        auto source=realProductionTPInputFromEnv();
        const auto pci=originalRetainedClosedMovementPCIs(scope);
        promoted23_replay::run(promoted23_replay::NumericalProgram::FullHCFFNEpoch,pci,
            promoted23_replay::EpochSource::GPUNativeRealTP4,&source);
        if(::testing::Test::HasFailure())MPI_Abort(MPI_COMM_WORLD,2);
    } catch(const std::exception& error){ADD_FAILURE()<<error.what();MPI_Abort(MPI_COMM_WORLD,2);throw;}
}
}
#endif

#if defined(HAVE_ROCM)
#include "FullNativeHCFFNFamilyReplay.inc"
#endif
