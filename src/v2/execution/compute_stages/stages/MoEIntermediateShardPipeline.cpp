/**
 * @file MoEIntermediateShardPipeline.cpp
 * @brief Captured source-authenticated local-F math and ordered local routed folding.
 *
 * Setup snapshots only original descriptor metadata. Every invocation resolves
 * descriptors through the acquired runtime bank, including after migration or
 * a zero-local-work replay. The graph exposes two local nodes and no transport.
 * The reduction order is local router slots first; the later explicit FFN sum
 * therefore has its own numerical policy, distinct from whole-expert routing.
 * Declaration identity precedes physical arena allocation. Launch preparation
 * seals the actual addresses; retained identity checks reject later rebinding
 * without changing the graph variant merely because initial storage appeared.
 */
#include "kernels/common/NativeLayoutBarrierLayout.h"
#include "MoEIntermediateShardPipeline.h"
#include "kernels/rocm/gemm/retained_hc_ffn/ROCmRetainedM1HCFFNWorkspace.h"
#include "MoERoutingStage.h"
#include "kernels/KernelFactory.h"
#include "qwen4exp/Qwen4ExpHyperconnectionBuilder.h"
#include "kernels/common/FullNativeHCFFNBarrierLayout.h"
#include "kernels/common/FullNativeHCFFNOutputDemand.h"
#include "FullNativeHCFFNOutputDemandResolver.h"
#include "execution/compute_stages/IGraphOutputDemand.h"
#include "MoEExpertComputeStage.h"
#include "loaders/PreparedWeightStore.h"
#include "config/TensorParallelConfig.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "execution/moe/MoEIntermediateShardDescriptorDeclaration.h"
#include "execution/moe/MoEGroupedVerifierHistogramBoundarySet.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/KernelFactory.h"
#include "kernels/common/ResidentMoEFFN.h"
#ifdef HAVE_ROCM
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#include "kernels/rocm/gemm/ROCmGroupedFFNFormatSelection.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "utils/DebugEnv.h"
#endif
#ifdef HAVE_ROCM
#include "kernels/rocm/gemm/ROCmResidentMoEFFN.h"
#include "kernels/rocm/gemm/ROCmFullNativeHCFFN.h"
#endif
#include "tensors/TensorClasses.h"
#include "tensors/GpuTensorView.h"
#include "utils/FNV1a.h"
#include "utils/PerfStatsCollector.h"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llaminar2
{
namespace
{
    /** @brief Two sequential local phases sharing the original router's kernel/workspace. */
    enum class Phase { Compute, Fold };
    /** @brief Distinct original route authorities; cold service cannot enter through a serving flag. */
    enum class RouteAuthority {OriginalRouter,OriginalRouterWithRebalance,PreparedService};

    /** @brief Frozen graph binding; source engines and placement readers are never retained. */
    class Binding final
    {
    public:
        /** @brief Validate original family, runtime geometry and every external tensor before graph mutation. */
        Binding(MoEIntermediateShardPipelineParams declaration, const MoEOverlayParticipantLayerBank &source,
            RouteAuthority original_route=RouteAuthority::OriginalRouter)
            : params(std::move(declaration)),route_authority(original_route)
        {
            if (!params.family || !params.runtime || !params.kernel_owner || params.rows <= 0 ||
                params.top_k <= 0 || params.rows > std::numeric_limits<int>::max() / params.top_k ||
                params.prefix.empty() || (route_authority!=RouteAuthority::PreparedService && params.router_node.empty()) ||
                params.row_policy != RoutedExpertRowExecutionPolicy::IntermediateShardLocal ||
                !validMoEGroupedPlanDemand(params.demand) ||
                (params.row_mode != MoEIntermediateShardRowMode::Decode && params.row_mode != MoEIntermediateShardRowMode::Grouped) ||
                (params.phase != MoEOverlayServicePhaseHint::Decode && params.phase != MoEOverlayServicePhaseHint::Prefill &&
                 params.phase != MoEOverlayServicePhaseHint::GroupedVerifier) ||
                (params.row_mode == MoEIntermediateShardRowMode::Decode &&
                 (params.rows != 1 || params.demand != MoEGroupedPlanDemand::None || params.phase != MoEOverlayServicePhaseHint::Decode)) ||
                (params.demand == MoEGroupedPlanDemand::OrdinaryPrefill && params.phase != MoEOverlayServicePhaseHint::Prefill) ||
                (params.demand == MoEGroupedPlanDemand::DeferredAcceptedRows && params.phase != MoEOverlayServicePhaseHint::GroupedVerifier))
                throw std::invalid_argument("Intermediate-shard graph requires explicit family, row, phase and demand policy");
            descriptor_declaration=MoEIntermediateShardDescriptorDeclaration::fromOriginalFamily(params.family);
            original = descriptor_declaration.originalBinding();
            const auto &cohort = *params.family->cohort();
            const auto &geometry = cohort.ownership(params.family->participantIndex()).geometry();
            layer = cohort.source().layer_idx;
            device_id = cohort.participants()[params.family->participantIndex()].address.toLocalDeviceId();
            if (params.runtime->deviceId() != device_id || params.runtime->fragmentFamily(layer) != params.family.get() ||
                source.fragment_family != params.family || !source.valid(geometry.experts))
                throw std::logic_error("Intermediate-shard graph changed its original runtime/source family");
            const auto &runtime = params.runtime->hostLayerState(layer);
            if (!runtime.intermediate_shard.sameIdentity(original) || runtime.expert_count != std::uint32_t(geometry.experts) ||
                runtime.participant_id != params.family->participantIndex() ||
                runtime.participant_count != cohort.participants().size() || runtime.top_k != std::uint32_t(params.top_k) ||
                params.top_k > geometry.experts || runtime.prefill_token_capacity < std::uint32_t(params.rows) ||
                runtime.prefill_route_capacity < std::uint32_t(params.rows * params.top_k))
                throw std::logic_error("Intermediate-shard graph runtime capacity or immutable geometry disagrees");
            if (params.live_rows)
            {
                const auto rows = params.live_rows->captureIdentity();
                if (rows.count_capacity != params.rows || rows.first_physical_row != 0 ||
                    rows.physical_rows != params.rows || rows.repeats_per_row != 1)
                    throw std::invalid_argument("Intermediate-shard graph requires the router's exact live row prefix");
            }
            const auto h_bytes = std::size_t(params.rows) * original.modelColumns() * sizeof(float);
            const auto route_bytes = std::size_t(params.rows) * params.top_k * sizeof(float);
            requireBuffer(params.hidden, h_bytes);
            if (!dynamic_cast<TensorBase *>(params.hidden.tensor))
                throw std::invalid_argument("Intermediate-shard hidden input requires the production TensorBase interface");
            requireBuffer(params.routing_indices, route_bytes);
            requireBuffer(params.routing_weights, route_bytes);
            requireBuffer(params.route_partials, h_bytes * params.top_k);
            requireBuffer(params.output, h_bytes);
            const auto tensors = bindings();
            for (std::size_t i = 0; i < tensors.size(); ++i)
                for (std::size_t j = 0; j < i; ++j)
                    if (tensors[i].tensor == tensors[j].tensor || tensors[i].id == tensors[j].id ||
                        (tensors[i].tensor->gpu_data_ptr() && tensors[i].tensor->gpu_data_ptr() == tensors[j].tensor->gpu_data_ptr()))
                        throw std::invalid_argument("Intermediate-shard graph requires distinct arena identities and storage");

            // Export original prepared descriptors while the caller's exact
            // construction borrow is alive. These metadata values carry no
            // ownership, and runtime-source tables never dereference an old
            // source pointer to choose a live expert after a bank replacement.
            format=descriptor_declaration.weightFormat();
            if(params.gate_up_publication!=MoEGateUpPublicationPolicy::Staged&&
                (params.gate_up_publication!=MoEGateUpPublicationPolicy::LastProducer||!device_id.is_rocm()||
                 params.row_mode!=MoEIntermediateShardRowMode::Decode||params.rows!=1||
                 format!=DeviceMoEWeightFormat::NativeQuantised))
                throw std::invalid_argument("Last-producer publication requires explicit original ROCm native M1 decode");
            if(params.down_publication!=MoEDownPublicationPolicy::Staged&&
                (params.down_publication!=MoEDownPublicationPolicy::LastProducerRouteFold||!device_id.is_rocm()||
                 params.row_mode!=MoEIntermediateShardRowMode::Decode||params.rows!=1||
                 format!=DeviceMoEWeightFormat::NativeQuantised))
                throw std::invalid_argument("Down/route completion requires explicit original ROCm native M1 decode");
            descriptors.resize(geometry.experts);
            for (int expert : source.fragment_expert_ids)
            {
                const auto &payload = source.fragments[expert];
                auto &descriptor = descriptors[expert];
                if (!params.family->accepts(payload, expert) ||
                    !exportDeviceMoEFragmentWeightDescriptors(payload, descriptor) || descriptor.weight_format != format ||
                    (deviceMoEWeightFormatIsFloating(format)
                        ? (!descriptor_declaration.matchesFloatingGateUp(descriptor.floating_gate,descriptor.floating_up) ||
                           !descriptor_declaration.matchesFloatingDown(descriptor.floating_down))
                        : (!descriptor_declaration.matchesNativeGateUp(descriptor.gate,descriptor.up) ||
                           !descriptor_declaration.matchesNativeDown(descriptor.down))))
                    throw std::invalid_argument("Intermediate-shard graph source has a foreign fragment/precision contract");
            }
            // Empty initial residency still declares every decoder/precision
            // reachable from the original source; no fake matrix is exported.
            auto mix = [&](std::uint64_t value) { signature = (signature ^ value) * kFNV1a64Prime; };
            mix(reinterpret_cast<std::uintptr_t>(params.family.get()));
            mix(reinterpret_cast<std::uintptr_t>(params.runtime));
            mix(reinterpret_cast<std::uintptr_t>(params.runtime->deviceLayerState(layer)));
            mix(reinterpret_cast<std::uintptr_t>(params.kernel_owner.get()));
            mix(static_cast<uint64_t>(route_authority));
            mix(params.rows); mix(params.top_k); mix(original.modelColumns()); mix(original.localIntermediateColumns());
            mix(original.firstIntermediateColumn()); mix(static_cast<unsigned>(params.row_mode));
            mix(static_cast<unsigned>(params.row_policy)); mix(static_cast<unsigned>(params.demand));
            mix(static_cast<unsigned>(params.phase)); mix(static_cast<unsigned>(format));
            mix(static_cast<unsigned>(params.gate_up_publication));
            mix(static_cast<unsigned>(params.down_publication));
            for(const auto& source:descriptor_declaration.sourceFormats()) {
                mix(static_cast<unsigned>(source.kind));mix(source.native_quantised.codebook_id);
                mix(source.native_quantised.is_superblock);mix(source.native_quantised.present);
            }
            const auto gate_up_envelope=descriptor_declaration.nativeGateUpEnvelope();
            const auto down_envelope=descriptor_declaration.nativeDownEnvelope();
            mix(gate_up_envelope.execution_codebook_mask);mix(gate_up_envelope.policy_codebook_mask);
            mix(down_envelope.execution_codebook_mask);mix(down_envelope.policy_codebook_mask);
            for (const auto &tensor : tensors)
            {
                mix(reinterpret_cast<std::uintptr_t>(tensor.tensor)); mix(static_cast<unsigned>(tensor.id));
            }
            if (params.live_rows)
            {
                const auto rows = params.live_rows->captureIdentity();
                mix(reinterpret_cast<std::uintptr_t>(rows.count)); mix(rows.count_capacity); mix(rows.first_physical_row);
                mix(rows.physical_rows); mix(rows.repeats_per_row); mix(rows.schema_version);
            }
            // Hash semantic descriptor fields rather than struct padding.
            // Runtime still owns live membership; this is setup-table identity.
            for (const auto &descriptor : descriptors)
            {
                for (const auto &matrix : {descriptor.gate, descriptor.up, descriptor.down})
                {
                    mix(reinterpret_cast<std::uintptr_t>(matrix.payload));
                    mix(reinterpret_cast<std::uintptr_t>(matrix.scales));
                    mix(reinterpret_cast<std::uintptr_t>(matrix.mins));
                    mix(reinterpret_cast<std::uintptr_t>(matrix.emins));
                    mix(matrix.n); mix(matrix.k); mix(matrix.blocks_per_row); mix(matrix.codebook_id);
                    mix(matrix.allocation_payload_bytes_per_block); mix(matrix.allocation_has_mins);
                    mix(matrix.allocation_has_emins); mix(matrix.source_codebook_id);
                    mix(matrix.source_is_superblock); mix(matrix.source_identity_present);
                }
                for (const auto &matrix : {descriptor.floating_gate, descriptor.floating_up, descriptor.floating_down})
                { mix(reinterpret_cast<std::uintptr_t>(matrix.data)); mix(matrix.n); mix(matrix.k); }
                mix(descriptor.logical_expert_id); mix(static_cast<unsigned>(descriptor.weight_format));
                mix(static_cast<unsigned>(descriptor.floating_allocation_format));
                mix(static_cast<unsigned>(descriptor.projection_set));
            }
            if (!signature) signature = 1;
        }

        /** @return Fixed participant endpoint, never inferred from a local/global member number. */
        DeviceId device() const noexcept { return device_id; }
        /** @return Exact borrowed external tensor identities in stable order. */
        std::array<MoEIntermediateShardTensorBinding, 5> bindings() const
        { return {params.hidden, params.routing_indices, params.routing_weights, params.route_partials, params.output}; }

        /** @brief Materialize original runtime tables and seal only consumed tensor addresses.
         * @param workspace Existing admitted workspace. @param stream Exact cold stream.
         * @param output_demand Complete local fusion's sealed demand; standalone phases require all outputs.
         */
        void prepare(DeviceWorkspaceManager *workspace, void *stream,
            FullNativeHCFFNOutputDemand output_demand=FullNativeHCFFNOutputDemand::all())
        {
            if (!workspace || !stream || isGraphCaptureActive())
                throw std::logic_error("Intermediate-shard preparation requires admitted workspace outside capture");
            const auto addresses = materializedTensorAddresses(output_demand);
            if (prepared)
            {
                requireTensorAddresses();
                if (prepared->output_demand!=output_demand || prepared->workspace != workspace || prepared->workspace_id != workspace->id() || prepared->stream != stream ||
                    !params.kernel_owner->kernel)
                    throw std::logic_error("Intermediate-shard prepared stream/workspace identity changed");
                return;
            }
            auto &kernel = params.kernel_owner->kernel;
            if (!kernel) kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device());
            if (!kernel) throw std::runtime_error("Intermediate-shard graph has no production GPU kernel");
            kernel->setGPUStream(stream);
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
            if (!consumer) throw std::logic_error("Intermediate-shard kernel lost its admitted workspace interface");
            consumer->bindWorkspace(workspace);
            const auto n = descriptors.size();
            std::vector<DeviceNativeQuantisedMatrixDesc> gates(n), ups(n), downs(n);
            std::vector<DeviceMoEFloatingMatrixDesc> fgates(n), fups(n), fdowns(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                gates[i] = descriptors[i].gate; ups[i] = descriptors[i].up; downs[i] = descriptors[i].down;
                fgates[i] = descriptors[i].floating_gate; fups[i] = descriptors[i].floating_up; fdowns[i] = descriptors[i].floating_down;
            }
            const auto source = MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable;
            const int h = original.modelColumns(), f = original.localIntermediateColumns();
            const int gate = deviceMoEWeightFormatIsFloating(format)
                ? kernel->uploadGroupedExpertFloatingGateUpDescriptorTables(fgates.data(), fups.data(), format, int(n), h, f, source, &descriptor_declaration)
                : kernel->uploadGroupedExpertGateUpDescriptorTables(gates.data(), ups.data(), int(n), h, f, source, &descriptor_declaration);
            const int down = deviceMoEWeightFormatIsFloating(format)
                ? kernel->uploadGroupedExpertFloatingDownDescriptorTable(fdowns.data(), format, int(n), h, f, source, &descriptor_declaration)
                : kernel->uploadGroupedExpertDownDescriptorTable(downs.data(), int(n), h, f, source, &descriptor_declaration);
            if (gate < 0 || down < 0 || (params.row_mode == MoEIntermediateShardRowMode::Decode &&
                !kernel->prepareGroupedRuntimeDecodeLaunchState(gate, down, params.top_k, h, f, source)))
                throw std::runtime_error("Intermediate-shard preparation rejected the original descriptor family");
            if (params.row_mode == MoEIntermediateShardRowMode::Decode &&
                route_authority == RouteAuthority::OriginalRouter)
                kernel->preparePairedRouterDecodeDescriptors(
                    params.runtime->deviceLayerState(layer), gate, down, params.top_k);
            const auto gate_up_signature=params.row_mode==MoEIntermediateShardRowMode::Decode
                ? kernel->prepareGateUpPublication(gate,params.top_k,params.gate_up_publication):0;
            const auto down_signature=params.row_mode==MoEIntermediateShardRowMode::Decode
                ? kernel->prepareDownPublication(down,params.top_k,params.down_publication):0;
            prepared = Prepared{workspace, workspace->id(), stream, gate, down, addresses,
                params.runtime->deviceLayerState(layer),gate_up_signature,down_signature,output_demand};
        }

        /** @brief Authenticate the physical seal when querying an already prepared graph identity. */
        void validatePreparedStorage() const
        {
            if (!prepared) return;
            if (prepared->workspace->id() != prepared->workspace_id)
                throw std::logic_error("Intermediate-shard prepared workspace generation changed");
            if(params.runtime->deviceLayerState(layer)!=prepared->runtime)
                throw std::logic_error("Intermediate-shard original runtime storage changed after preparation");
            if(params.row_mode==MoEIntermediateShardRowMode::Decode&&
                (!params.kernel_owner->kernel||params.kernel_owner->kernel->gateUpPublicationSignature(
                    prepared->gate,params.top_k)!=prepared->gate_up_signature))
                throw std::logic_error("Intermediate-shard gate/up publication program/storage changed after preparation");
            if(params.row_mode==MoEIntermediateShardRowMode::Decode&&
                (!params.kernel_owner->kernel||params.kernel_owner->kernel->downPublicationSignature(
                    prepared->down,params.top_k)!=prepared->down_signature))
                throw std::logic_error("Intermediate-shard down/route program/storage changed after preparation");
            requireTensorAddresses();
        }

        /** @brief Retire descriptor state only after the executor retires its native graphs. */
        void invalidate()
        {
            if (!prepared) return;
            params.kernel_owner->kernel->resetDynamicState();
            params.kernel_owner->kernel->clearGPUStreamBinding();
            prepared.reset();
        }

        /** @brief Enqueue actual production fragment math or its deterministic local slot fold. */
        bool execute(Phase phase, void *stream)
        {
            if (!prepared || prepared->output_demand!=FullNativeHCFFNOutputDemand::all() ||
                !stream || prepared->stream != stream || !params.kernel_owner->kernel)
                throw std::logic_error("Intermediate-shard execution lacks its exact prepared stream/kernel");
            requireTensorAddresses();
            auto &kernel = *params.kernel_owner->kernel;
            kernel.setGPUStream(stream);
            const int h = original.modelColumns(), f = original.localIntermediateColumns();
            // The candidate computed routed bytes under the compute leaf's
            // exact original stream. This unchanged terminal stage remains the
            // sole coherence/publication authority and dependency seen by shared
            // FFN and the explicit collective; no consumer can observe an early
            // value, and no native clearing/fold node is submitted here.
            if(phase==Phase::Fold&&params.down_publication==MoEDownPublicationPolicy::LastProducerRouteFold)
                return true;
            if (phase == Phase::Fold)
                return requireOperation(kernel.reduceCanonicalRouteContributions(params.route_partials.tensor, params.output.tensor,
                    params.rows, params.top_k, h), "ordered local router-slot fold");
            auto *runtime = params.runtime->deviceLayerState(layer);
            if (params.row_mode == MoEIntermediateShardRowMode::Decode)
            {
                // PreparedService's closed cold token authenticates the one
                // fixed top-k input already published in its private runtime.
                // It needs no duplicate selector or grouped-plan launch.
                // The authenticated router already published placement-filtered
                // device top-k state and demand. Consume that publication
                // directly: an explicit-packet selector would count it twice.
                // The legacy API retains the later reducer's tensor identity;
                // canonical_route_contributions is its sole write/coherence
                // target, so the reducer output is neither read nor published.
                return requireOperation(kernel.groupedExpertDecodeFromRuntime(runtime,
                    dynamic_cast<TensorBase *>(params.hidden.tensor), prepared->gate, prepared->down,
                    params.top_k, params.output.tensor, h, f,
                    MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable, params.route_partials.tensor),
                    "runtime-published local fragment decode");
            }
            requireOperation(kernel.publishCompleteGroupedPrefillPlanFromRouter(runtime, params.routing_indices.tensor,
                params.routing_weights.tensor, params.rows, params.rows, int(descriptors.size()), params.top_k,
                prepared->gate, prepared->down, true, params.demand, DeviceMoEProjectionSet::IntermediateShard),
                "complete local fragment grouped-plan publication");
            return requireOperation(kernel.executeGroupedPrefillProjectionFromPublishedRuntimePlan(runtime, params.runtime->hostLayerState(layer),
                params.hidden.tensor, nullptr, prepared->gate, prepared->down, params.rows, h, f,
                int(descriptors.size()), params.top_k, MoEPrefillProjectionExecution::intermediateShard(h), params.route_partials.tensor),
                "runtime-published local fragment grouped projection");
        }

#ifdef HAVE_ROCM
        /** @brief Use the same original runtime route publisher, then consume its exact grouped frontier. */
        bool executeFrontier(const rocm::grouped_ffn::LocalPublication& publication,void* stream)
        {
            if(!prepared || prepared->stream!=stream || !params.kernel_owner->kernel ||
                params.row_mode!=MoEIntermediateShardRowMode::Grouped ||
                params.phase!=MoEOverlayServicePhaseHint::GroupedVerifier || !params.live_rows ||
                publication.rows.captureIdentity()!=params.live_rows->captureIdentity())
                throw std::logic_error("Grouped FFN frontier lost the original verifier publication binding");
            requireTensorAddresses();
            auto* kernel=dynamic_cast<ROCmMoEKernel*>(params.kernel_owner->kernel.get());
            if(!kernel)throw std::logic_error("Grouped FFN frontier requires the original ROCm runtime owner");
            kernel->setGPUStream(stream);
            auto* runtime=params.runtime->deviceLayerState(layer);
            if(qfnGroupedFFNDirectBankExperiment()) {
                auto direct=publication;
                direct.directRouter={params.routing_indices.tensor,params.routing_weights.tensor,params.demand};
                return requireOperation(kernel->executeGroupedFFNFrontierFromPublishedRuntimePlan(runtime,
                    params.runtime->hostLayerState(layer),params.hidden.tensor,prepared->gate,prepared->down,
                    params.route_partials.tensor,direct),"two-node grouped local FFN with acquired bank");
            }
            requireOperation(kernel->publishCompleteGroupedPrefillPlanFromRouter(runtime,params.routing_indices.tensor,
                params.routing_weights.tensor,params.rows,params.rows,int(descriptors.size()),params.top_k,
                prepared->gate,prepared->down,true,params.demand,DeviceMoEProjectionSet::IntermediateShard),
                "complete local fragment grouped-plan publication");
            return requireOperation(kernel->executeGroupedFFNFrontierFromPublishedRuntimePlan(runtime,
                params.runtime->hostLayerState(layer),params.hidden.tensor,prepared->gate,prepared->down,
                params.route_partials.tensor,publication),"two-node grouped local FFN");
        }
#endif

        /** @brief Capture-stable descriptor and workspace publication, containing no live placement. */
        struct Prepared
        {
            DeviceWorkspaceManager *workspace;
            std::uint64_t workspace_id;
            void *stream;
            int gate;
            int down;
            std::array<const void *, 5> tensor_addresses;
            const DeviceMoELayerRuntime *runtime;
            std::uint64_t gate_up_signature; ///< Separately sealed prepared native arguments; declaration policy stays stable.
            std::uint64_t down_signature; ///< Exact two-level completion lease/source seal.
            FullNativeHCFFNOutputDemand output_demand; ///< Materialization contract of this exact prepared caller.
        };
        const MoEIntermediateShardPipelineParams params;
        const RouteAuthority route_authority;
        DeviceMoEIntermediateShardBinding original;
        int layer = -1;
        DeviceId device_id = DeviceId::invalid();
        DeviceMoEWeightFormat format = DeviceMoEWeightFormat::NativeQuantised;
        MoEIntermediateShardDescriptorDeclaration descriptor_declaration;
        std::vector<DeviceMoEExpertDescriptor> descriptors;
        std::optional<Prepared> prepared;
        std::uint64_t signature = kFNV1a64OffsetBasis;
    private:
        /** @brief Preserve exact immutable phase/geometry when a backend rejects an operation. */
        bool requireOperation(bool success, const char *operation) const
        {
            if (!success)
                throw std::runtime_error(std::string("Intermediate-shard ") + operation + " failed on " + device().toString() +
                    " layer=" + std::to_string(layer) + " global_member=" + std::to_string(original.target().participantId()) +
                    " cohort_member=" + std::to_string(params.family->participantIndex()) +
                    " rows=" + std::to_string(params.rows) + " local_F=" + std::to_string(original.localIntermediateColumns()) +
                    " top_k=" + std::to_string(params.top_k) + " phase=" + std::to_string(static_cast<unsigned>(params.phase)) +
                    " row_mode=" + (params.row_mode == MoEIntermediateShardRowMode::Decode ? "decode" : "grouped"));
            return true;
        }
        /** @brief Resolve only the original tensor values consumed by this prepared operation.
         * @param demand Immutable fusion output demand. @return Exact addresses; omitted roles are explicitly null.
         */
        std::array<const void *, 5> materializedTensorAddresses(FullNativeHCFFNOutputDemand demand) const
        {
            const auto tensors = bindings();
            std::array<const void *, 5> addresses{};
            for (std::size_t i = 0; i < tensors.size(); ++i)
            {
                if((i==3&&!demand.contains(FullNativeHCFFNOutputRole::RouteContributions))||
                   (i==4&&!demand.contains(FullNativeHCFFNOutputRole::Routed)))continue;
                addresses[i] = tensors[i].tensor->gpu_data_ptr();
                if (!addresses[i])
                    throw std::logic_error("Intermediate-shard graph requires materialized arena buffer " +
                        std::string(bufferIdName(tensors[i].id)));
                for (std::size_t j = 0; j < i; ++j)
                    if (addresses[i] == addresses[j])
                        throw std::logic_error("Intermediate-shard graph arena buffers alias after materialization");
            }
            return addresses;
        }

        /** @brief Reject absent or rebound physical storage before recording, replay or enqueue. */
        void requireTensorAddresses() const
        {
            if (!prepared || prepared->tensor_addresses != materializedTensorAddresses(prepared->output_demand))
                throw std::logic_error("Intermediate-shard graph arena addresses changed after launch preparation");
        }

        /** @brief Validate required live geometry, without creating another capacity ledger. */
        static void requireBuffer(MoEIntermediateShardTensorBinding binding, std::size_t bytes)
        {
            if (!binding.tensor || binding.tensor->native_type() != TensorType::FP32 || binding.tensor->size_bytes() < bytes)
                throw std::invalid_argument("Intermediate-shard graph requires sufficient FP32 arena binding " + std::string(bufferIdName(binding.id)));
        }
    };

    /** @brief Thin production compute leaf, with no private scratch or collective authority. */
    class Stage final : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        /** @brief Freeze one phase of the source-authenticated local transaction. */
        Stage(std::shared_ptr<Binding> binding, Phase phase, std::string name)
            : IComputeStage(binding->device()), binding_(std::move(binding)), phase_(phase), name_(std::move(name)) {}
        /** @return Exact original cold binding, for lossless final FFN lowering only. */
        const std::shared_ptr<Binding> &binding() const noexcept { return binding_; }
        /** @return The immutable local phase authenticated by the finalizer. */
        Phase phase() const noexcept { return phase_; }
        /** @return Existing local MoE phase classification; never a collective. */
        ComputeStageType type() const override { return ComputeStageType::MOE_PROJECTION_PHASE; }
        /** @return Unique model/layer/local-phase graph identity. */
        std::string name() const override { return name_; }
        /** @return Frozen full declaration identity, independent of live rows or placement epoch. */
        std::uint64_t graphCaptureVariantSignature() const override
        {
            binding_->validatePreparedStorage();
            const auto value = (binding_->signature ^ std::uint64_t(phase_)) * kFNV1a64Prime;
            return value ? value : 1;
        }
        /** @return Only the original GPU backend can execute this participant. */
        bool supportsBackend(ComputeBackendType backend) const override
        { return (binding_->device().is_cuda() && backend == ComputeBackendType::GPU_CUDA) ||
                 (binding_->device().is_rocm() && backend == ComputeBackendType::GPU_ROCM); }
        /** @return Whether persistent tables and the arena workspace have been published. */
        bool isGraphCapturable() const override { return workspace_ && binding_->prepared.has_value(); }
        /** @return Setup creates metadata without an eager forward. */
        bool supportsGraphCaptureAfterLaunchPreparation() const override { return true; }
        /** @return Router invalid slots exclude inactive physical rows without a count readback. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override { return true; }
        /** @return Cold exact buckets require preparation only. */
        bool supportsLazyPrefillGraphCapturePreflight() const override { return true; }
        /** @return Metadata materialization is a pre-capture operation. */
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override { return GraphLaunchPreparationPolicy::CaptureOnly; }
        /** @brief Prepare all tables on the executor's exact stream and workspace. */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override
        {
            if (!ctx || ctx->deviceId() != binding_->device() || !stream) return false;
            binding_->prepare(workspace_, stream);
            setGPUStream(stream);
            return true;
        }
        /** @brief Retire shared dynamic state after native executable invalidation. */
        void invalidateKernelDynamicState() override { binding_->invalidate(); }
        /** @brief Enqueue real local kernels and publish the exact producer stream. */
        bool execute(IDeviceContext *ctx) override
        {
            if (!ctx || ctx->deviceId() != binding_->device() || !workspace_) return false;
            if (!binding_->prepared || binding_->prepared->workspace != workspace_ ||
                binding_->prepared->workspace_id != workspace_->id())
                throw std::logic_error("Intermediate-shard stage workspace changed after capture preparation");
            const auto execution = gpuExecution();
            for (const auto &input : inputs()) execution.requirePreparedInput(input.tensor);
            execution.requirePreparedOutput(output().tensor);
            if(phase_==Phase::Compute&&binding_->params.down_publication==MoEDownPublicationPolicy::LastProducerRouteFold)
                execution.requirePreparedOutput(binding_->params.output.tensor);
            if (!binding_->execute(phase_, execution.nativeStream())) return false;
            if (phase_ == Phase::Fold)
                PerfStatsCollector::addCounter("moe_intermediate_shard",
                    binding_->params.down_publication==MoEDownPublicationPolicy::Staged?"local_route_fold_enqueued":"local_route_fold_published",1.0,
                    "graph_capture", binding_->device().toString(),
                    {{"publication", binding_->params.down_publication==MoEDownPublicationPolicy::Staged?"local_partial":"last_producer_route_fold"}, {"layer", std::to_string(binding_->layer)},
                     {"physical_rows", std::to_string(binding_->params.rows)},
                     {"local_f", std::to_string(binding_->original.localIntermediateColumns())}});
            execution.publish(output().tensor);
            return true;
        }
        /** @return Exact arena dependency roles for this phase. */
        StageBufferContract bufferContract() const override
        {
            auto contract = StageBufferContract::build();
            for (const auto &input : inputs()) contract.addInput(input.id);
            // Declare the physical early write at Compute so arena liveness
            // cannot recycle an input interval before the terminal Fold stage
            // publishes routed coherence. Both stages keep their original DAG.
            if(phase_==Phase::Compute&&binding_->params.down_publication==MoEDownPublicationPolicy::LastProducerRouteFold)
                contract.addOutput(binding_->params.output.id);
            return contract.addOutput(output().id);
        }
        /** @return External tensor shape/type declarations; no storage is allocated here. */
        StageBufferRequirements getBufferRequirements() const override
        {
            StageBufferRequirements requirements;
            std::size_t index = 0;
            for (const auto &input : inputs()) requirements.addInput("fragment_input_" + std::to_string(index++),
                input.tensor->shape(), toBufferTensorType(input.tensor->native_type()));
            requirements.addOutput("fragment_local_partial", output().tensor->shape(), BufferTensorType::FP32);
            if(phase_==Phase::Compute&&binding_->params.down_publication==MoEDownPublicationPolicy::LastProducerRouteFold)
                requirements.addOutput("fragment_early_routed",binding_->params.output.tensor->shape(),BufferTensorType::FP32);
            return requirements;
        }
        /** @return Existing backend MoE workspace with actual local F and physical rows. */
        WorkspaceRequirements getWorkspaceRequirements(int m, int = 0, int = 0) const override
        {
            const auto &p = binding_->params;
            const int rows = std::max(m, p.rows), h = binding_->original.modelColumns(), f = binding_->original.localIntermediateColumns();
            return binding_->device().is_cuda()
                ? MoEWorkspaceBuffers::cudaMoE(rows, h, f, int(binding_->descriptors.size()), p.top_k)
                : MoEWorkspaceBuffers::rocmMoE(rows, h, f, int(binding_->descriptors.size()), p.top_k,p.gate_up_publication,p.down_publication);
        }
        /** @brief Borrow the executor's admitted workspace. */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override { workspace_ = workspace; }
        /** @brief Release this stage's borrow; shared metadata retires through graph invalidation. */
        void unbindWorkspace() override { workspace_ = nullptr; }
        /** @return Whether the executor has supplied its workspace. */
        bool hasWorkspace() const override { return workspace_ != nullptr; }
        /** @return The exact executor-owned workspace. */
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }
        /** @return Inputs are already prepared; only this phase's output is published. */
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }
        /** @return Explicit local-partial diagnostics, without complete-expert timing/publication claims. */
        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo info;
            info.addScalarInt("intermediate_shard_local_partial", 1);
            info.addScalarInt("first_intermediate_column", binding_->original.firstIntermediateColumn());
            info.addScalarInt("local_intermediate_columns", binding_->original.localIntermediateColumns());
            info.addOutput(phase_ == Phase::Compute ? "local_route_partials" : "local_routed_partial", output().tensor,
                binding_->params.rows * (phase_ == Phase::Compute ? binding_->params.top_k : 1), binding_->original.modelColumns());
            return info;
        }
    private:
        /** @return Immutable input list, with no per-execution allocation. */
        std::span<const MoEIntermediateShardTensorBinding> inputs() const
        { return phase_ == Phase::Compute ? std::span<const MoEIntermediateShardTensorBinding>(compute_inputs_)
                                         : std::span<const MoEIntermediateShardTensorBinding>(fold_inputs_); }
        /** @return Sole publication owned by this local phase. */
        MoEIntermediateShardTensorBinding output() const
        { return phase_ == Phase::Compute ? binding_->params.route_partials : binding_->params.output; }
        const std::shared_ptr<Binding> binding_;
        const Phase phase_;
        const std::string name_;
        const std::array<MoEIntermediateShardTensorBinding, 3> compute_inputs_{
            binding_->params.hidden, binding_->params.routing_indices, binding_->params.routing_weights};
        const std::array<MoEIntermediateShardTensorBinding, 1> fold_inputs_{binding_->params.route_partials};
        DeviceWorkspaceManager *workspace_ = nullptr;
    };
#ifdef HAVE_ROCM
#include "MoEIntermediateShardGroupedFFNStage.inc"
#endif
#include "MoEIntermediateShardResidentStage.inc"

    /** @brief Append only after the selected original route authority has fully validated. */
    MoEIntermediateShardLocalPartial appendLocalPhases(ComputeGraph& graph,const std::shared_ptr<Binding>& binding)
    {
        const auto& p=binding->params;
        const std::array names{p.prefix + "local_fragment_compute", p.prefix + "local_route_fold"};
        for (const auto &name : names)
            if (graph.getNode(name)) throw std::logic_error("Intermediate-shard node identity already exists");
        auto compute = std::make_unique<Stage>(binding, Phase::Compute, names[0]);
        auto fold = std::make_unique<Stage>(binding, Phase::Fold, names[1]);
        graph.addNode(names[0], std::move(compute), binding->device());
        if(binding->route_authority!=RouteAuthority::PreparedService)
            graph.addDependency(names[0], p.router_node);
        graph.addNode(names[1], std::move(fold), binding->device());
        graph.addDependency(names[1], names[0]);
        return {names[1], p.output, p.family, p.live_rows, p.rows, int(binding->original.modelColumns()), p.phase};
    }

}

MoEIntermediateShardLocalPartial appendMoEIntermediateShardPipeline(ComputeGraph &graph,
    MoEIntermediateShardPipelineParams params, const MoEOverlayParticipantLayerBank &source)
{
    // Validate every declaration before mutating topology. No device interaction
    // occurs here, and the caller's source borrow ends at this function boundary.
    const auto *original_router = graph.getNode(params.router_node);
    const auto *original_router_stage = original_router
        ? dynamic_cast<const MoERoutingStage *>(original_router->stage.get()) : nullptr;
    const auto authority = original_router_stage && original_router_stage->declaration().device_rebalance_route_apply
        ? RouteAuthority::OriginalRouterWithRebalance : RouteAuthority::OriginalRouter;
    auto binding = std::make_shared<Binding>(std::move(params), source, authority);
    const auto &p = binding->params;
    const auto *router = graph.getNode(p.router_node);
    const auto *router_stage = router ? dynamic_cast<const MoERoutingStage *>(router->stage.get()) : nullptr;
    if (!router_stage || router_stage->layerIndex() != binding->layer || router_stage->device() != binding->device())
        throw std::invalid_argument("Intermediate-shard pipeline requires its original graph-owned router");
    const auto &r = router_stage->declaration();
    // Decode can contain several verifier/request rows. Its arithmetic role
    // must not silently become ordinary prefill merely because M is larger
    // than one: those router policies may use different activation precision.
    // The original router owns that policy, and the consumer authenticates it
    // before any stage is appended or any capture can retain the wrong math.
    const bool grouped_decode_arithmetic =
        p.row_mode == MoEIntermediateShardRowMode::Grouped &&
        p.phase != MoEOverlayServicePhaseHint::Prefill;
    if (r.force_decode_equivalent_verifier_prefill != grouped_decode_arithmetic)
        throw std::invalid_argument("Intermediate-shard router arithmetic disagrees with its explicit execution phase");
    if (r.routed_pipeline_kernel_owner != p.kernel_owner || r.moe_runtime_table != p.runtime ||
        r.routed_row_execution_policy != p.row_policy || r.seq_len != p.rows || r.top_k != p.top_k ||
        r.d_model != int(binding->original.modelColumns()) || r.num_experts != int(binding->descriptors.size()) ||
        r.input != p.hidden.tensor || r.input_buffer_id != p.hidden.id ||
        r.output_indices != p.routing_indices.tensor || r.output_indices_buffer_id != p.routing_indices.id ||
        r.output_weights != p.routing_weights.tensor || r.output_weights_buffer_id != p.routing_weights.id ||
        r.active_row_count_device != (p.live_rows ? p.live_rows->countOwner() : nullptr) ||
        r.decode_route_publication != MoEDecodeRoutePublicationPolicy::DeviceRuntimeTable || r.owned_rows ||
        (p.row_mode == MoEIntermediateShardRowMode::Decode &&
         (r.force_grouped_verifier_prefill_for_decode || r.force_decode_equivalent_verifier_prefill)) ||
        (p.row_mode == MoEIntermediateShardRowMode::Grouped && p.rows == 1 &&
         !r.force_grouped_verifier_prefill_for_decode && !r.force_decode_equivalent_verifier_prefill))
        throw std::invalid_argument("Intermediate-shard router changed its original owner, arena, live rows or execution policy");
    return appendLocalPhases(graph,binding);
}

bool MoEIntermediateShardPreparedServiceRoute::authenticates(
    const MoEIntermediateShardPipelineParams& p,const MoEOverlayParticipantLayerBank& source) const
{
    if(!family_ || !runtime_ || p.family!=family_ || p.runtime!=runtime_ || source.fragment_family!=family_ ||
       p.top_k!=1 || p.rows!=rows_ || p.phase!=phase_ || p.demand!=MoEGroupedPlanDemand::None ||
       p.live_rows || !p.router_node.empty() || p.routing_indices.tensor!=indices_ || p.routing_weights.tensor!=weights_ ||
       runtime_->usesOverlayEpochTicket() || !runtime_->isMirroredToDevice())return false;
    const auto layer=family_->cohort()->source().layer_idx;
    const auto& state=runtime_->hostLayerState(layer);
    return epoch_ && state.active_epoch==epoch_ && state.topk_expert_ids[0]==expert_ &&
        state.topk_weights[0]==1.f && runtime_->fragmentFamily(layer)==family_.get() &&
        expert_>=0 && size_t(expert_)<source.fragments.size() && source.resident_mask.at(size_t(expert_)) &&
        family_->accepts(source.fragments[size_t(expert_)],expert_);
}

MoEIntermediateShardLocalPartial appendPreparedMoEIntermediateShardPipeline(ComputeGraph& graph,
    MoEIntermediateShardPipelineParams params,const MoEOverlayParticipantLayerBank& source,
    const MoEIntermediateShardPreparedServiceRoute& route)
{
    if(!route.authenticates(params,source))
        throw std::invalid_argument("Prepared intermediate-shard service lost its original source/epoch/fixed route provenance");
    auto binding=std::make_shared<Binding>(std::move(params),source,RouteAuthority::PreparedService);
    return appendLocalPhases(graph,binding);
}

MoEIntermediateShardFFNOutput finalizeMoEIntermediateShardFFN(ComputeGraph &graph,
    const MoEIntermediateShardLocalPartial &routed, MoEIntermediateShardFFNFinalization finalization)
{
    if(finalization.execution!=MoEIntermediateShardFFNExecution::Staged &&
       finalization.execution!=MoEIntermediateShardFFNExecution::ResidentCandidate)
        throw std::invalid_argument("Unknown intermediate-shard FFN execution policy");
    const auto *routed_node = graph.getNode(routed.producer);
    const auto *fold = routed_node ? dynamic_cast<const Stage *>(routed_node->stage.get()) : nullptr;
    if (!fold || fold->phase() != Phase::Fold)
        throw std::invalid_argument("Final FFN requires this graph's original intermediate-shard local fold");
    const auto binding = fold->binding();
    const auto &p = binding->params;
    const auto same_rows = [](const std::optional<DeviceRowRange> &a, const std::optional<DeviceRowRange> &b) {
        return a.has_value() == b.has_value() && (!a || a->captureIdentity() == b->captureIdentity());
    };
    if (routed.output.tensor != p.output.tensor || routed.output.id != p.output.id || routed.family != p.family ||
        routed.rows != p.rows || routed.model_columns != int(binding->original.modelColumns()) ||
        routed.phase != p.phase || !same_rows(routed.live_rows, p.live_rows))
        throw std::logic_error("Final FFN cannot replace the original local-partial publication contract");
    if (finalization.prefix.empty() || finalization.shared_ffn_node.empty())
        throw std::invalid_argument("Final FFN requires explicit local producer and graph identities");
    const auto *shared_node = graph.getNode(finalization.shared_ffn_node);
    const auto *shared = shared_node ? dynamic_cast<const SharedExpertFFNStage *>(shared_node->stage.get()) : nullptr;
    if (!shared || shared->device() != binding->device())
        throw std::invalid_argument("Final FFN requires a real shared-FFN producer on the original participant");
    const auto &s = shared->declaration();
    const auto *count = p.live_rows ? p.live_rows->countOwner() : nullptr;
    const bool decode_equivalent = p.row_mode == MoEIntermediateShardRowMode::Grouped && p.phase != MoEOverlayServicePhaseHint::Prefill;
    // Finalizing a complete verifier FFN must also close its history lifecycle.
    // Local projection stages own no history boundary, so the original router
    // must declare StaticNoPublication or the admitted accepted-row publisher.
    const auto *router_node = graph.getNode(p.router_node);
    const auto *router = router_node ? dynamic_cast<const MoERoutingStage *>(router_node->stage.get()) : nullptr;
    if (!router || router->groupedVerifierHistogramRole() !=
            selectMoEGroupedVerifierHistogramRole(decode_equivalent,
                p.demand == MoEGroupedPlanDemand::DeferredAcceptedRows))
        throw std::invalid_argument("Final FFN router has no matching explicit verifier-history boundary");
    if (s.output_role != SharedExpertOutputRole::ParticipantLocalPartial || s.phase != p.phase ||
        s.input != p.hidden.tensor || s.input_buffer_id != p.hidden.id || s.seq_len != p.rows ||
        s.d_model != routed.model_columns || s.intermediate <= 0 || !s.output ||
        s.active_row_count_device != count || s.force_decode_equivalent_verifier_prefill ||
        s.force_grouped_verifier_prefill_for_decode != decode_equivalent || s.disable_grouped_decode_shortcut ||
        (decode_equivalent && s.required_router_q8_publication != p.kernel_owner->router_q8_publication) ||
        (s.required_router_q8_publication && s.required_router_q8_publication != p.kernel_owner->router_q8_publication) ||
        !s.gate_w || !s.up_w || !s.down_w || !s.prepared_store ||
        !s.prepared_ref_gate || !s.prepared_ref_up || !s.prepared_ref_down ||
        s.prepared_store->modelId() != p.family->model())
        throw std::invalid_argument("Final FFN shared producer changed its original source, numerical phase, rows or local-partial role");

    // Shared F is its own original source axis. Authenticate committed Store
    // rectangles rather than deriving it from the unrelated routed F cuts or
    // rebuilding a participant/source map on the host.
    const auto &cohort = *p.family->cohort();
    const auto member = p.family->participantIndex();
    if (!finalization.shared_weights || finalization.shared_weights->strategy().model_id != p.family->model() ||
        finalization.shared_weights->strategy().tp_degree != int(cohort.participants().size()) ||
        finalization.shared_weights->strategy().devices.size() != cohort.participants().size())
        throw std::invalid_argument("Final FFN requires its original frozen shared plan and native TP degree");
    finalization.shared_weights->validateForGraph();
    const auto *gate_input = finalization.shared_weights->optionalLayerForDevice(
        binding->layer, "ffn_gate_inp_shexp.weight", binding->device());
    if (!gate_input || !gate_input->immutable || !gate_input->tensor_owner ||
        gate_input->tensor_owner.get() != gate_input->tensor || gate_input->identity.model_id != p.family->model() ||
        gate_input->identity.role != WeightRole::SharedExpertInputGate ||
        gate_input->identity.tp_rank_or_device_index != int(member) ||
        gate_input->slice.source_rows != std::size_t(routed.model_columns) || gate_input->slice.source_cols != 1 ||
        gate_input->slice.row_start || gate_input->slice.col_start ||
        gate_input->slice.row_count != std::size_t(routed.model_columns) || gate_input->slice.col_count != 1 ||
        gate_input->slice.expert_start || gate_input->slice.expert_count || !gate_input->slice.expert_ids.empty())
        throw std::invalid_argument("Final FFN requires the original replicated shared input-gate binding");
    auto *shared_gate = gate_input->tensor;
    const auto scope = "Final FFN layer=" + std::to_string(binding->layer) +
        " device=" + binding->device().toString() + " member=" + std::to_string(member) +
        " original_participant=" + std::to_string(cohort.participants()[member].participant_id);
    const auto source_binding = [&](const PreparedWeightRef &ref, TensorBase *tensor, WeightRole role) {
        const auto source_scope = scope + " role=" + toString(role) + " binding=" + std::to_string(ref.binding_id);
        const auto original = s.prepared_store->binding(ref);
        if (!original)
            throw std::invalid_argument(source_scope + " has no original prepared-store binding");
        if (!original->immutable || ref.model_id != p.family->model() || ref.device != binding->device() ||
            original->identity.model_id != p.family->model() || original->identity.layer != binding->layer ||
            original->identity.role != role || original->identity.tp_rank_or_device_index != int(member))
            throw std::invalid_argument(source_scope + " changed immutable original identity: immutable=" +
                std::to_string(original->immutable) + " source_model=" + std::to_string(original->identity.model_id.value) +
                " source_layer=" + std::to_string(original->identity.layer) + " source_role=" + toString(original->identity.role) +
                " source_member=" + std::to_string(original->identity.tp_rank_or_device_index));
        if (original->tensor != tensor || !original->tensor_owner || original->tensor_owner.get() != tensor)
            throw std::invalid_argument(source_scope + " changed original tensor ownership: tensor_match=" +
                std::to_string(original->tensor == tensor) + " owner_present=" + std::to_string(bool(original->tensor_owner)) +
                " owner_match=" + std::to_string(original->tensor_owner.get() == tensor));
        const auto &cut = original->slice;
        if (!cut.source_rows || !cut.source_cols || !cut.row_count || !cut.col_count ||
            cut.source_rows > std::size_t(std::numeric_limits<int>::max()) ||
            cut.source_cols > std::size_t(std::numeric_limits<int>::max()) ||
            cut.expert_start || cut.expert_count || !cut.expert_ids.empty() || tensor->shape().size() != 2)
            throw std::invalid_argument(source_scope + " has incomplete original matrix/cut metadata: source_N=" +
                std::to_string(cut.source_rows) + " source_K=" + std::to_string(cut.source_cols) +
                " first_N=" + std::to_string(cut.row_start) + " local_N=" + std::to_string(cut.row_count) +
                " first_K=" + std::to_string(cut.col_start) + " local_K=" + std::to_string(cut.col_count));
        const auto &declared = finalization.shared_weights->bindings();
        const auto source = std::find_if(declared.begin(), declared.end(), [&](const auto &entry) {
            return entry.binding_id == ref.binding_id && entry.identity.sameSourceDefinition(original->identity) &&
                entry.slice == original->slice && entry.tensor_owner == original->tensor_owner;
        });
        if (source == declared.end())
            throw std::invalid_argument(source_scope + " prepared source is absent or changed in its original participant-local frozen set");
        return *original;
    };
    const auto gate = source_binding(*s.prepared_ref_gate, s.gate_w, WeightRole::SharedExpertGate);
    const auto up = source_binding(*s.prepared_ref_up, s.up_w, WeightRole::SharedExpertUp);
    const auto down = source_binding(*s.prepared_ref_down, s.down_w, WeightRole::SharedExpertDown);
    // GPU preparation retains the original WeightBinding slice/identity and
    // actual tensor owner. native_source is a CPU final-consumer field and is
    // deliberately neither required nor fabricated for GPU shared operands.
    const auto &g = gate.slice, &u = up.slice, &d = down.slice;
    const auto h = std::size_t(routed.model_columns), local_f = std::size_t(s.intermediate);
    const auto shared_f = g.source_rows;
    if (g.source_cols != h || u.source_rows != shared_f || u.source_cols != h ||
        d.source_rows != h || d.source_cols != shared_f ||
        g.row_start != u.row_start || g.row_count != local_f || u.row_count != local_f ||
        g.col_start || u.col_start || g.col_count != h || u.col_count != h ||
        d.row_start || d.row_count != h || d.col_start != g.row_start || d.col_count != local_f ||
        !shared_f || local_f > shared_f || g.row_start > shared_f || local_f > shared_f - g.row_start)
        throw std::invalid_argument(scope + " shared triplet source/cuts disagree: H=" + std::to_string(h) +
            " source_F=" + std::to_string(shared_f) + " declared_local_F=" + std::to_string(local_f) +
            " gate=[" + std::to_string(g.row_start) + "," + std::to_string(g.row_count) +
            "] up=[" + std::to_string(u.row_start) + "," + std::to_string(u.row_count) +
            "] down_K=[" + std::to_string(d.col_start) + "," + std::to_string(d.col_count) + "]");
    // Production freezes sources per participant. Their original retained
    // dense/shared assignment proves complete F ownership without inventing
    // peer bindings or reconstructing another sharding formula. These are the
    // SAME d_ff_start/count fields consumed by WeightManager's FFN slicer.
    const auto *partition = finalization.shared_partition;
    const bool original_singleton = cohort.participants().size() == 1 &&
        finalization.shared_weights->strategy().mode == WeightInferenceMode::SingleDevice &&
        member == 0 && g.row_start == 0 && local_f == shared_f;
    if (!partition && !original_singleton)
        throw std::invalid_argument(scope + " requires its original shared TP partition; only an original SingleDevice full source is an identity");
    if (partition && (!partition->validate() || partition->worldSize() != int(cohort.participants().size()) ||
        partition->totalDFF() != int(shared_f)))
        throw std::invalid_argument(scope + " has no valid original shared TP partition matching source_F=" + std::to_string(shared_f));
    if (partition)
    {
        const auto &local_shared = partition->forRank(int(member));
        if (local_shared.d_ff_start != int(g.row_start) || local_shared.d_ff_count != int(local_f))
            throw std::invalid_argument(scope + " prepared shared cut differs from original TP assignment: first_F=" +
                std::to_string(g.row_start) + " local_F=" + std::to_string(local_f) + " assignment=[" +
                std::to_string(local_shared.d_ff_start) + "," + std::to_string(local_shared.d_ff_count) + "]");
    }
    std::size_t shared_cursor = 0;
    for (std::size_t index = 0; index < cohort.participants().size(); ++index)
    {
        const auto device = cohort.participants()[index].address.toLocalDeviceId();
        if (finalization.shared_weights->strategy().devices[index] != device)
            throw std::invalid_argument("Final FFN shared plan changed original native participant order");
        if (partition)
        {
            const auto &original = partition->forRank(int(index));
            if (original.device != device || original.d_ff_start != int(shared_cursor) || original.d_ff_count <= 0 ||
                shared_cursor > shared_f || std::size_t(original.d_ff_count) > shared_f - shared_cursor)
                throw std::invalid_argument("Final FFN original shared assignment must cover F exactly once in native member order");
            shared_cursor += std::size_t(original.d_ff_count);
        }
        else
            shared_cursor += local_f; // Original full SingleDevice cut, already authenticated above.
    }
    if (shared_cursor != shared_f)
        throw std::invalid_argument("Final FFN original shared assignment omits source F columns");
    DeviceMoEExpertDescriptor prepared{};
    if (!exportDeviceMoEExpertWeightDescriptors(s.prepared_store->gemmKernel(*s.prepared_ref_gate),
            s.prepared_store->gemmKernel(*s.prepared_ref_up), s.prepared_store->gemmKernel(*s.prepared_ref_down),
            routed.model_columns, s.intermediate, prepared))
        throw std::invalid_argument(scope + " prepared GEMM descriptor triple cannot export original H=" +
            std::to_string(h) + " local_F=" + std::to_string(local_f));

    auto *combined = dynamic_cast<TensorBase *>(finalization.local_combined.tensor);
    const auto elements = std::size_t(p.rows) * h;
    if (!shared_gate || shared_gate->native_type() != TensorType::FP32 ||
        shared_gate->numel() != h || !combined || combined->native_type() != TensorType::FP32 ||
        combined->numel() < elements || s.output->native_type() != TensorType::FP32 || s.output->numel() < elements)
        throw std::invalid_argument("Final FFN requires the prepared FP32 H gate and admitted FP32 shared/combined banks");
    const std::array arenas{p.hidden, p.routing_indices, p.routing_weights, p.route_partials, p.output,
        MoEIntermediateShardTensorBinding{s.output, s.output_buffer_id}, finalization.local_combined};
    for (std::size_t i = 0; i < arenas.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (arenas[i].tensor == arenas[j].tensor || arenas[i].id == arenas[j].id ||
                (arenas[i].tensor->gpu_data_ptr() && arenas[i].tensor->gpu_data_ptr() == arenas[j].tensor->gpu_data_ptr()))
                throw std::invalid_argument("Final FFN requires distinct original routed, shared and combined arena roles");

    for (const auto &arena : arenas)
        if (shared_gate == arena.tensor || (shared_gate->gpu_data_ptr() &&
            shared_gate->gpu_data_ptr() == arena.tensor->gpu_data_ptr()))
            throw std::invalid_argument("Final FFN gate weight cannot alias mutable activation storage");
    auto sum = std::move(finalization.final_sum);
    auto *tp = dynamic_cast<ILocalTPContext *>(sum.tp_ctx);
    if (!tp || sum.device_id != binding->device() || tp->degree() != int(cohort.participants().size()) ||
        sum.sideband_device_index != int(member) ||
        tp->devices().size() != cohort.participants().size())
        throw std::logic_error("Final FFN sum requires the original native cohort and local coordinate");
    for (std::size_t index = 0; index < cohort.participants().size(); ++index)
        if (tp->devices()[index] != cohort.participants()[index].address)
            throw std::logic_error("Final FFN sum changed the original participant reduction order");
    const bool singleton_identity = tp->degree() == 1 && tp->backend() == CollectiveBackendType::HOST;
    if ((!singleton_identity && binding->device().is_cuda() && tp->backend() != CollectiveBackendType::NCCL) ||
        (!singleton_identity && binding->device().is_rocm() && tp->backend() != CollectiveBackendType::RCCL) ||
        sum.arithmetic_policy != TPAllreduceArithmeticPolicy::CanonicalRankOrder || sum.precision != "fp32" ||
        sum.tensor != combined || sum.count != elements || sum.tensor_buffer_id != finalization.local_combined.id || sum.request_rows)
        throw std::invalid_argument("Final FFN requires one native CanonicalRankOrder FP32 whole-prefix sum; request-row layout is unsupported");
    const auto expected_rows = p.live_rows.value_or(DeviceRowRange::fullyActive(p.rows));
    if (sum.live_rows && (sum.live_rows->rows().captureIdentity() != expected_rows.captureIdentity() ||
        sum.live_rows->elementsPerRow() != h))
        throw std::invalid_argument("Final FFN collective cannot substitute another live-row authority or row width");
    // Preserve an original native payload receipt as well as all sidebands.
    // Only construct the missing row descriptor from the sole routed owner.
    if (p.live_rows && !sum.live_rows) sum.live_rows = NativeCollectiveRows(*p.live_rows, h);
    const auto gate_name = finalization.prefix + "local_shared_gate_add", sum_name = finalization.prefix + "ffn_sum";
    if ((!sum.stage_name.empty() && sum.stage_name != sum_name) || graph.getNode(gate_name) || graph.getNode(sum_name))
        throw std::logic_error("Final FFN node identity already exists or differs from the declared sum");
    sum.stage_name = sum_name;
    SharedExpertGateStage::Params epilogue;
    epilogue.device_id = binding->device(); epilogue.input = s.input; epilogue.gate_inp = shared_gate;
    epilogue.shared_output = s.output; epilogue.routed_residual = dynamic_cast<TensorBase *>(p.output.tensor);
    epilogue.combined_output = combined; epilogue.seq_len = p.rows; epilogue.d_model = routed.model_columns;
    epilogue.active_row_count_device = count; epilogue.output_role = SharedExpertOutputRole::ParticipantLocalPartial;
    epilogue.phase = p.phase; epilogue.input_buffer_id = p.hidden.id; epilogue.output_buffer_id = s.output_buffer_id;
    epilogue.residual_buffer_id = p.output.id; epilogue.combined_output_buffer_id = finalization.local_combined.id;
    // Both constructors validate before any topology mutation. The original
    // sum owns its existing sideband metadata/workspace publication unchanged.
    auto gate_stage = std::make_unique<SharedExpertGateStage>(epilogue);
    auto sum_stage = std::make_unique<TPAllreduceStage>(std::move(sum));
    if(finalization.execution==MoEIntermediateShardFFNExecution::ResidentCandidate)
    {
        const auto compute_name=p.prefix+"local_fragment_compute";
        const auto *compute_node=graph.getNode(compute_name);
        const auto *compute=compute_node ? dynamic_cast<const Stage *>(compute_node->stage.get()) : nullptr;
        if(!compute || compute->binding()!=binding || compute->phase()!=Phase::Compute)
            throw std::logic_error("Resident FFN must consume this graph's original routed compute and fold");
        auto resident=std::make_unique<ResidentStage>(binding,s,prepared,shared_gate,
            finalization.local_combined,finalization.shared_ffn_node,unsigned(g.row_start));
        // The original shared branch follows the routed fold because they
        // reuse scratch. Keep that sink's public graph identity and preserve
        // every external input; only the explicit final sum consumes its new
        // combined publication. The generic graph API rejects hidden interior
        // consumers or any communication/capture boundary before mutation.
        graph.replaceClosedLocalRegion({compute_name,routed.producer,finalization.shared_ffn_node},
            finalization.shared_ffn_node,std::move(resident));
        graph.addNode(sum_name,std::move(sum_stage),binding->device());
        graph.addDependency(sum_name,finalization.shared_ffn_node);
        return {sum_name,finalization.local_combined,p.family,p.live_rows,p.rows,routed.model_columns,p.phase};
    }
#ifdef HAVE_ROCM
    if(auto program=groupedFFNFrontierProgram(*binding,s,prepared))
    {
        const auto compute_name=p.prefix+"local_fragment_compute";
        const auto* node=graph.getNode(compute_name);
        const auto* compute=node?dynamic_cast<const Stage*>(node->stage.get()):nullptr;
        if(!compute || compute->binding()!=binding || compute->phase()!=Phase::Compute)
            throw std::logic_error("Grouped FFN frontier lost the original local compute producer");
        if(frontier::usesExactK32Sums(*program)) {
            const auto* routerNode=graph.getNode(p.router_node);
            const auto* router=routerNode?dynamic_cast<const MoERoutingStage*>(routerNode->stage.get()):nullptr;
            auto& owner=*p.kernel_owner;
            if(!router || router->declaration().routed_pipeline_kernel_owner!=p.kernel_owner ||
               router->declaration().input!=p.hidden.tensor ||
               router->declaration().active_row_count_device!=p.live_rows->countOwner() ||
               !router->declaration().gate_weights || router->declaration().gate_weights->native_type()!=TensorType::FP16)
                throw std::logic_error("Exact K32 sums need the original shared counted FP16 router owner");
            if(!owner.kernel)owner.kernel=llaminar::v2::kernels::KernelFactory::createMoEKernel(binding->device());
            if(!owner.kernel || !owner.kernel->bindRouterQ8HiddenPublication(owner.router_q8_publication,
                    MoERouterQ8PublicationAccess::ProducerAndConsumerWithExactK32Sums))
                throw std::logic_error("Exact K32 sum publication failed cold binding");
        }
        auto frontier=std::make_unique<GroupedFFNFrontierStage>(binding,s,prepared,std::move(*program),shared_gate,
            finalization.local_combined,finalization.shared_ffn_node);
        graph.replaceClosedLocalRegion({compute_name,routed.producer,finalization.shared_ffn_node},
            finalization.shared_ffn_node,std::move(frontier));
        graph.addNode(sum_name,std::move(sum_stage),binding->device());
        graph.addDependency(sum_name,finalization.shared_ffn_node);
        return {sum_name,finalization.local_combined,p.family,p.live_rows,p.rows,routed.model_columns,p.phase};
    }
#endif
    // The shared grouped kernel borrows only immutable original router Q8
    // addresses, never its mutable grouping state. Make its producer edge
    // explicit before the existing shared stage can be scheduled.
    if (s.required_router_q8_publication && std::find(shared_node->dependencies.begin(),
            shared_node->dependencies.end(), p.router_node) == shared_node->dependencies.end())
        graph.addDependency(finalization.shared_ffn_node, p.router_node);
    graph.addNode(gate_name, std::move(gate_stage), binding->device());
    graph.addDependency(gate_name, routed.producer);
    graph.addDependency(gate_name, finalization.shared_ffn_node);
    graph.addNode(sum_name, std::move(sum_stage), binding->device());
    graph.addDependency(sum_name, gate_name);
    return {sum_name, finalization.local_combined, p.family, p.live_rows, p.rows, routed.model_columns, p.phase};
}

/** @brief Borrow the original router under the caller's closed M1 graph role. */
static void lowerNativeLayoutRouterFFNForRole(ComputeGraph& graph,const std::string& resident_node,
    ResidentFFNGraphRole role)
{
    auto* node=graph.getNode(resident_node);
    const auto* resident=node?dynamic_cast<const ResidentStage*>(node->stage.get()):nullptr;
    if(!resident)throw std::invalid_argument("Native-layout router FFN requires its original resident stage");
    const auto router_name=resident->routerName();
    const auto* router_node=graph.getNode(router_name);
    const auto* router=router_node?dynamic_cast<const MoERoutingStage*>(router_node->stage.get()):nullptr;
    if(!router)throw std::invalid_argument("Native-layout router FFN requires its original graph router");
    auto replacement=resident->withNativeRouter(router->declaration(),role);
    graph.replaceClosedLocalRegion({router_name,resident_node},resident_node,std::move(replacement));
}

/** @copydoc lowerNativeLayoutRouterFFN */
void lowerNativeLayoutRouterFFN(ComputeGraph& graph,const std::string& resident_node)
{
    lowerNativeLayoutRouterFFNForRole(graph,resident_node,ResidentFFNGraphRole::LocalRegion);
}

/** @brief Close a matched original HC/FFN region while preserving declared outside edges. */
static void lowerFullNativeHCFFNRegion(ComputeGraph& graph,const std::string& resident_node,
    const std::string& previous_write_node,const std::string& hc_read_node,bool retain_original_injection,
    ResidentFFNGraphRole role=ResidentFFNGraphRole::LocalRegion)
{
    auto* node=graph.getNode(resident_node);
    const auto* resident=node?dynamic_cast<const ResidentStage*>(node->stage.get()):nullptr;
    if(!resident)throw std::invalid_argument("Full native HC/FFN requires its original resident FFN sink");
    auto region=Qwen4ExpHyperconnectionBuilder::describeFullNativeReadRegion(graph,previous_write_node,hc_read_node);
    if(retain_original_injection)region.injection.clear();
    const auto* read=dynamic_cast<const Qwen4ExpPrimitiveStage*>(graph.getNode(region.read)->stage.get());
    const auto& operation=std::get<Qwen4ExpPrimitiveStage::ResidualRead>(read->getParams().operation);
    auto prefix=Qwen4ExpHyperconnectionBuilder::retainFullNativeReadPhase(graph,region);
    const auto implementation=retain_original_injection?rocm::FullNativeHCFFNImplementation::RetainedGFX906M1:rocm::FullNativeHCFFNImplementation::Canonical;
    auto replacement=resident->withFullNativePrefix(std::move(prefix),operation.output,implementation,role);
    std::vector<std::string> members{region.previous_write,region.branch_norm,region.down,region.up,region.read,resident_node};
    if(!region.injection.empty())members.push_back(region.injection);
    for(std::size_t i=0;i<members.size();++i)for(std::size_t j=0;j<i;++j)
        if(members[i]==members[j])throw std::invalid_argument("Full native HC region contains duplicate original nodes");
    // Preserve the exact existing outside readers of every original public
    // producer. In particular the independent injection remains exported to
    // the later paired write, never redirected through a private stash.
    std::vector<LocalRegionExport> exports;
    for(const auto& producer:members) {
        if(producer==resident_node)continue;
        LocalRegionExport declaration{.producer=producer,.consumers={}};
        for(const auto& name:graph.getExecutionOrder()) {
            if(std::find(members.begin(),members.end(),name)!=members.end())continue;
            const auto& dependencies=graph.getNode(name)->dependencies;
            if(std::find(dependencies.begin(),dependencies.end(),producer)!=dependencies.end())declaration.consumers.push_back(name);
        }
        if(!declaration.consumers.empty())exports.push_back(std::move(declaration));
    }
    graph.replaceClosedLocalRegion(members,resident_node,std::move(replacement),exports);
}

/** @copydoc lowerFullNativeHCFFN */
void lowerFullNativeHCFFN(ComputeGraph& graph,const std::string& resident_node,
    const std::string& previous_write_node,const std::string& hc_read_node)
{
    lowerFullNativeHCFFNRegion(graph,resident_node,previous_write_node,hc_read_node,false);
}

/** @brief Lower one exact retained local region with its original invocation role. */
static void lowerRetainedM1HCFFNForRole(ComputeGraph& graph,const std::string& previous_write_node,
    ResidentFFNGraphRole role)
{
    using HC=Qwen4ExpPrimitiveStage;
    const auto* boundary=graph.getNode(previous_write_node);
    const auto* write=boundary?dynamic_cast<const HC*>(boundary->stage.get()):nullptr;
    if(!write||!std::holds_alternative<HC::ResidualWrite>(write->getParams().operation)||!boundary->device.is_rocm())
        throw std::invalid_argument("Retained M1 HC/FFN requires the original ROCm attention write boundary");
    const auto order=graph.getExecutionOrder();
    const ComputeNode* norm=nullptr;
    const ComputeNode* read=nullptr;
    const ComputeNode* sink=nullptr;
    const auto depends=[](const ComputeNode& node,const std::string& producer) {
        return std::find(node.dependencies.begin(),node.dependencies.end(),producer)!=node.dependencies.end();
    };
    for(const auto& name:order) {
        const auto* node=graph.getNode(name);
        const auto* primitive=dynamic_cast<const HC*>(node->stage.get());
        if(!primitive||node->device!=boundary->device)continue;
        if(depends(*node,previous_write_node)&&std::holds_alternative<HC::BranchRMSNorm>(primitive->getParams().operation)) {
            if(norm)throw std::invalid_argument("Retained M1 HC/FFN has multiple original normalization edges");
            norm=node;
        }
        if(norm&&depends(*node,norm->name)&&std::holds_alternative<HC::ResidualRead>(primitive->getParams().operation)) {
            if(read)throw std::invalid_argument("Retained M1 HC/FFN has multiple original residual-read edges");
            read=node;
        }
    }
    if(!read)throw std::invalid_argument("Retained M1 HC/FFN requires its original staged HC read");
    for(const auto& name:order) {
        const auto* node=graph.getNode(name);
        const auto* resident=dynamic_cast<const ResidentStage*>(node->stage.get());
        if(!resident||node->device!=boundary->device)continue;
        const auto* router=graph.getNode(resident->routerName());
        if(!router||(!depends(*node,read->name)&&!depends(*router,read->name)))continue;
        if(sink)throw std::invalid_argument("Retained M1 HC/FFN has multiple original local FFN sinks");
        sink=node;
    }
    if(!sink)throw std::invalid_argument("Retained M1 HC/FFN requires its original resident local FFN sink");
    const auto sink_name=sink->name,read_name=read->name;
    lowerNativeLayoutRouterFFNForRole(graph,sink_name,role);
    lowerFullNativeHCFFNRegion(graph,sink_name,previous_write_node,read_name,true,role);
}

/** @copydoc lowerRetainedM1HCFFN */
void lowerRetainedM1HCFFN(ComputeGraph& graph,const std::string& previous_write_node)
{
    lowerRetainedM1HCFFNForRole(graph,previous_write_node,ResidentFFNGraphRole::LocalRegion);
}

/** @copydoc lowerRetainedM1HCFFNMainInference */
void lowerRetainedM1HCFFNMainInference(ComputeGraph& graph,const std::string& previous_write_node)
{
    lowerRetainedM1HCFFNForRole(graph,previous_write_node,ResidentFFNGraphRole::MainInference);
}

}
