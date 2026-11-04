/**
 * @file MoEOverlayCPUServiceMeasurement.cpp
 * @brief Measures prepared CPU/GPU expert FFNs without changing serving state.
 *
 * Preparation owns bounded inputs and the production backend workspace.
 * Every physical payload is claimed before construction and released after
 * the last stage has retired. Different phases/classes reuse that capacity
 * serially. Only the measured service rows survive setup; no graph, weight
 * copy, inference counter, or alternate placement lifecycle is retained.
 */
#include "MoEOverlayCPUServiceMeasurement.h"
#include "planning/PlanningGPUExpertMeasurement.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include <map>
#include "backends/RetainedParticipantGraphExecutable.h"

#include "MoEOverlayPreparedWeightSource.h"
#include "CPUExpertShardSourceSelection.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
#include "utils/Logger.h"
#include "collective/GlobalTPContext.h"
#include "memory/CPUJointExpertRouteStorageLayout.h"
#include "execution/compute_stages/stages/MoECPUExpertShardRouteStage.h"
#include "kernels/cpu/moe/CPUCanonicalRouteReducer.h"
#include <array>
#include <exception>
#include <cmath>

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        using llaminar::v2::kernels::KernelFactory;
        // Small real-kernel observations, not full-model synthetic inference.
        constexpr int kPrefillRows = MoEOverlayCPUServiceMeasurement::kMaximumRows;
        constexpr int kVerifierRows = 4;

        /** @return Production phase and bounded representative batch geometry. */
        std::pair<MoEOverlayServicePhaseHint, int> phaseGeometry(ExpertHistogramSource source)
        {
            switch (source)
            {
            case ExpertHistogramSource::DecodeToken: return {MoEOverlayServicePhaseHint::Decode, 1};
            case ExpertHistogramSource::PrefillChunk: return {MoEOverlayServicePhaseHint::Prefill, kPrefillRows};
            case ExpertHistogramSource::GroupedVerifier: return {MoEOverlayServicePhaseHint::GroupedVerifier, kVerifierRows};
            default: throw std::invalid_argument("CPU service measurement requires a priced production phase");
            }
        }

        /** @return The exact CPU workspace constructor geometry used by a probe. */
        CPUGroupedMoESerialWorkspace::Config workspaceConfig(
            MoEOverlayCPUServiceMeasurement::Geometry geometry, int rows,
            DeviceId device, std::shared_ptr<PhysicalMemoryAuthority> memory = {})
        {
            return {.row_capacity = static_cast<size_t>(rows),
                    .d_model = geometry.d_model, .expert_intermediate = geometry.intermediate,
                    .num_experts = 1, .routing_top_k = 1,
                    .debug_name = "overlay_prepared_service", .device_id = device,
                    .physical_memory_authority = std::move(memory)};
        }

        /** @brief The calibration's final tensor roots, separately owned from grouped scratch. */
        struct ServiceStorageLayout
        {
            CPUStorageRoot<float> input, indices, weights, output;
            CPUGroupedMoESerialWorkspace::StorageLayout grouped;
            /** @return Checked complete physical calibration bank. */
            size_t physicalBytes() const
            {
                size_t total = grouped.physicalBytes();
                for (size_t bytes : {input.physicalBytes(), indices.physicalBytes(), weights.physicalBytes(), output.physicalBytes()})
                {
                    if (bytes > std::numeric_limits<size_t>::max() - total)
                        throw std::overflow_error("CPU prepared-service root extent overflows");
                    total += bytes;
                }
                return total;
            }
        };

        /** @return One allocation-free declaration used by admission and real calibration roots. */
        ServiceStorageLayout serviceStorageLayout(
            MoEOverlayCPUServiceMeasurement::Geometry geometry, int rows, DeviceId device)
        {
            const auto root = [&](size_t count) { return CPUStorageRoot<float>(
                count, PhysicalMemoryOwner::ExecutionWorkspace, device); };
            const auto grouped = CPUGroupedMoESerialWorkspace::storageLayout(workspaceConfig(geometry, rows, device));
            return {root(grouped.batch.capacity()), root(static_cast<size_t>(rows)),
                root(static_cast<size_t>(rows)), root(grouped.batch.capacity()), grouped};
        }

        /** @brief Reuse canonical route extent/placement under the probe's existing PMA owner.
         * @tparam T Native root element.
         * @param source Existing graph-independent route declaration.
         * @return Same physical geometry with setup execution ownership, never a second ledger.
         */
        template<class T> CPUStorageRoot<T> setupRoot(const CPUStorageRoot<T>& source)
        {
            return {source.capacity(), PhysicalMemoryOwner::ExecutionWorkspace,
                source.allocator(), source.placement(), source.initialization()};
        }

        /** @brief Exact serial probe roots; the same declaration feeds pricing and construction. */
        struct JointServiceStorageLayout
        {
            CPUJointExpertRouteStorageLayout route;
            CPUGroupedMoESerialWorkspace::Config grouped_config;
            ServiceStorageLayout service;
            CPUStorageRoot<float> packed, combined, release;
            CPUStorageRoot<RootedFloatRecordRange> ranges;
            CPUStorageRoot<int32_t> maps;
            CPUStorageRoot<float> required;
            /** @return Complete physical roots without invocation workspace, checked for overflow. */
            size_t physicalBytes() const
            {
                size_t total=service.physicalBytes();
                for(auto bytes:{packed.physicalBytes(),combined.physicalBytes(),release.physicalBytes(),
                    ranges.physicalBytes(),maps.physicalBytes(),required.physicalBytes()}) {
                    if(bytes>SIZE_MAX-total)throw std::overflow_error("CPU joint prepared-compute roots overflow");
                    total+=bytes;
                }
                return total;
            }
        };

        /** @brief Declare one original member's exact input-ready/output-ready probe.
         * @param cohort Same original cuts/member/native source declaration.
         * @param member Original domain ordinal.
         * @param phase Existing representative phase family.
         * @return Same roots for admission and setup; no graph or payload is constructed.
         */
        JointServiceStorageLayout jointServiceStorageLayout(
            const std::shared_ptr<const MoEExpertShardCohort>& cohort,size_t member,
            ExpertHistogramSource phase)
        {
            const auto rows=phaseGeometry(phase).second;
            CPUJointExpertRouteStorageLayout route(cohort,member,rows,1);
            const auto& geometry=cohort->ownership(member).geometry();
            auto config=workspaceConfig({geometry.model_columns,route.intermediate()},rows,
                cohort->participants()[member].address.toLocalDeviceId());
            config.num_experts=geometry.experts;
            const auto base=serviceStorageLayout({geometry.model_columns,route.intermediate()},rows,config.device_id);
            ServiceStorageLayout service{base.input,base.indices,base.weights,base.output,
                CPUGroupedMoESerialWorkspace::storageLayout(config)};
            const auto placement=route.packed().placement();
            const auto floatRoot=[&](size_t count){return CPUStorageRoot<float>(count,
                PhysicalMemoryOwner::ExecutionWorkspace,route.packed().allocator(),placement);};
            const auto rangeRoot=route.ranges()?setupRoot(*route.ranges()):
                CPUStorageRoot<RootedFloatRecordRange>(0,PhysicalMemoryOwner::ExecutionWorkspace,
                    route.packed().allocator(),placement);
            const auto mapRoot=route.maps()?setupRoot(*route.maps()):CPUStorageRoot<int32_t>(
                0,PhysicalMemoryOwner::ExecutionWorkspace,route.packed().allocator(),placement);
            return {route,config,std::move(service),setupRoot(route.packed()),
                floatRoot(route.isRoot()?static_cast<size_t>(rows)*geometry.model_columns:0),
                floatRoot(1),rangeRoot,mapRoot,route.required()?setupRoot(*route.required()):floatRoot(0)};
        }

        /** @brief Compare immutable original shared owners, rejecting no-op aliases.
         * @tparam T Original source/preparation type.
         * @param a First retained owner.
         * @param b Second retained owner.
         * @return Exact object and control-block identity.
         */
        template<class T> bool sameSourceOwner(const std::shared_ptr<const T>& a,
            const std::shared_ptr<const T>& b)
        {return a && a.get()==b.get() && !a.owner_before(b) && !b.owner_before(a);}

        /** @return Exact physical calibration geometry; CPU ordinal does not change rounding. */
        size_t payloadBytes(MoEOverlayCPUServiceMeasurement::Geometry geometry, int rows)
        {
            return serviceStorageLayout(geometry, rows, DeviceId::cpu()).physicalBytes();
        }

        /**
         * @brief Reject geometry mismatches before constructing any scratch or stage.
         * @param engine Exact prepared source retained by the bank lease.
         * @param device CPU physical owner.
         * @param n Expected full projection output width.
         * @param k Expected full projection input width.
         */
        void requireProjection(const std::shared_ptr<ITensorGemm> &engine, DeviceId device, int n, int k)
        {
            MoEOverlayPreparedWeightSource source;
            std::string error;
            if (!resolveMoEOverlayPreparedWeightSource(engine, device, source, &error))
                throw std::invalid_argument("CPU prepared-service source: " + error);
            const auto actual_n = source.cpu_packed ? source.cpu_packed->N : source.floating.n;
            const auto actual_k = source.cpu_packed ? source.cpu_packed->K : source.floating.k;
            if (actual_n != n || actual_k != k)
                throw std::invalid_argument("CPU prepared-service projection differs from admitted geometry");
        }

        /**
         * @brief Run one complete FFN with exclusively owned invocation storage.
         * @return Sum of physical wall-clock nanoseconds across measured samples.
         *
         * The first invocation touches code/weights and starts the existing
         * OpenMP team outside the timed samples. Inputs are nonzero bounded FP32;
         * the sole selected expert is privately indexed at zero, without
         * modifying its global bank identity or any routing histogram.
         */
        uint64_t measureExpert(const MoEOverlayPreparedExpertPayload &source,
                              DeviceId device, int layer, ExpertHistogramSource source_phase,
                              MoEOverlayCPUServiceMeasurement::Geometry geometry,
                              const std::shared_ptr<PhysicalMemoryAuthority> &memory)
        {
            const auto [phase, rows] = phaseGeometry(source_phase);
            requireProjection(source.gate(), device, geometry.intermediate, geometry.d_model);
            requireProjection(source.up(), device, geometry.intermediate, geometry.d_model);
            requireProjection(source.down(), device, geometry.d_model, geometry.intermediate);
            const auto layout = serviceStorageLayout(geometry, rows, device);
            FP32Tensor input({static_cast<size_t>(rows), static_cast<size_t>(geometry.d_model)}, layout.input.allocate(memory), device);
            FP32Tensor indices({static_cast<size_t>(rows), 1}, layout.indices.allocate(memory), device);
            FP32Tensor weights({static_cast<size_t>(rows), 1}, layout.weights.allocate(memory), device);
            FP32Tensor output({static_cast<size_t>(rows), static_cast<size_t>(geometry.d_model)}, layout.output.allocate(memory), device);
            for (int row = 0; row < rows; ++row)
            {
                indices.mutable_typed_data()[row] = 0.0f;
                weights.mutable_typed_data()[row] = 1.0f;
                for (int col = 0; col < geometry.d_model; ++col)
                    input.mutable_typed_data()[static_cast<size_t>(row) * geometry.d_model + col] =
                        0.0001f * static_cast<float>((col % 31 * 13 + row * 7) % 31 - 15);
            }
            const MoEOverlayPreparedExpertPayload engines{
                KernelFactory::createExpertServiceExecutionView(source.gate(), device),
                KernelFactory::createExpertServiceExecutionView(source.up(), device),
                KernelFactory::createExpertServiceExecutionView(source.down(), device)};
            auto workspace = std::make_shared<CPUGroupedMoESerialWorkspace>(workspaceConfig(geometry, rows, device, memory));
            MoEExpertComputeStage::Params p;
            p.device_id = device;
            p.input = &input;
            p.output = &output;
            p.output_registered_in_arena = false;
            p.routing_indices = &indices;
            p.routing_weights = &weights;
            p.seq_len = rows;
            p.d_model = geometry.d_model;
            p.expert_intermediate = geometry.intermediate;
            p.num_experts = 1;
            p.top_k = 1;
            p.layer_idx = layer;
            p.expert_mask = {true};
            p.prepared_gate_gemm = {engines.gate().get()};
            p.prepared_up_gemm = {engines.up().get()};
            p.prepared_down_gemm = {engines.down().get()};
            p.expert_weight_resolution_policy = MoEExpertWeightResolutionPolicy::PreparedRegistryOnly;
            p.cpu_router_q8_input_publication = CPURouterQ8InputPublicationPolicy::PublishTransportedRows;
            p.cpu_grouped_serial_workspace = std::move(workspace);
            p.service_phase = phase;
            p.force_decode_equivalent_verifier_prefill = phase == MoEOverlayServicePhaseHint::GroupedVerifier;
            std::unique_ptr<DeviceWorkspaceManager> invocation_workspace;
            MoEExpertComputeStage stage(std::move(p));
            const auto requirements = stage.getWorkspaceRequirements(rows);
            // This is a separate physical allocation from the grouped tensor
            // bank above. Its own authority lease retires with the manager.
            invocation_workspace = std::make_unique<DeviceWorkspaceManager>(device,
                requirements.total_bytes_with_alignment(), memory);
            if (!invocation_workspace->allocate(requirements))
                throw std::runtime_error("CPU prepared-service invocation workspace admission failed");
            stage.bindWorkspace(invocation_workspace.get());
            auto context = IDeviceContext::create(device);
            if (!stage.execute(context.get()))
                throw std::runtime_error("CPU prepared-service warmup failed");
            uint64_t total = 0;
            for (int sample = 0; sample < MoEOverlayCPUServiceMeasurement::kMeasuredSamples; ++sample)
            {
                const auto start = std::chrono::steady_clock::now();
                if (!stage.execute(context.get()))
                    throw std::runtime_error("CPU prepared-service FFN failed");
                const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start).count();
                if (nanoseconds <= 0 || static_cast<uint64_t>(nanoseconds) > UINT64_MAX - total)
                    throw std::runtime_error("CPU prepared-service timing is not positive and representable");
                total += static_cast<uint64_t>(nanoseconds);
            }
            return total;
        }
    }

    size_t MoEOverlayCPUServiceMeasurement::GPUStorageLayout::hostBytes() const
    {
        size_t result = 0;
        for (const auto *root : {&input, &indices, &weights, &output})
        {
            if (root->physicalBytes() > std::numeric_limits<size_t>::max() - result)
                throw std::overflow_error("Prepared GPU service host roots overflow");
            result += root->physicalBytes();
        }
        return result;
    }

    size_t MoEOverlayCPUServiceMeasurement::GPUStorageLayout::deviceBytes() const
    {
        size_t result = 0;
        for (const auto *root : {&input, &indices, &weights, &output})
        {
            if (root->capacity() > (std::numeric_limits<size_t>::max() - result) / sizeof(float))
                throw std::overflow_error("Prepared GPU service tensor copies overflow");
            result += root->capacity() * sizeof(float);
        }
        return result;
    }

    MoEOverlayCPUServiceMeasurement::GPUStorageLayout
    MoEOverlayCPUServiceMeasurement::gpuStorageLayout(Geometry geometry, int rows)
    {
        if (geometry.d_model <= 0 || geometry.intermediate <= 0 || rows <= 0 || rows > kMaximumRows ||
            size_t(geometry.d_model) > std::numeric_limits<size_t>::max() / size_t(rows))
            throw std::invalid_argument("Prepared GPU service requires valid original phase geometry");
        const auto root = [](size_t count) {
            return CPUStorageRoot<float>(count, PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu());
        };
        const auto payload = size_t(rows) * size_t(geometry.d_model);
        return {root(payload), root(size_t(rows)), root(size_t(rows)), root(payload)};
    }

    MoEOverlayCPUServiceMeasurement::GPUAllocation
    MoEOverlayCPUServiceMeasurement::gpuAllocationBytes(Geometry geometry, size_t invocation)
    {
        const auto layout = gpuStorageLayout(geometry, kMaximumRows);
        const auto payload = layout.deviceBytes();
        if (invocation > std::numeric_limits<size_t>::max() - payload)
            throw std::overflow_error("Prepared GPU service workspace overflows");
        return {layout.hostBytes(), payload + invocation};
    }

    size_t MoEOverlayCPUServiceMeasurement::allocationBytes(Geometry geometry, size_t invocation_workspace_bytes)
    {
        const int rows = std::max(kPrefillRows, kVerifierRows);
        const size_t tensors = payloadBytes(geometry, rows);
        const size_t invocation = invocation_workspace_bytes;
        if (invocation > std::numeric_limits<size_t>::max() - tensors)
            throw std::overflow_error("CPU prepared-service total workspace overflow");
        return tensors + invocation;
    }

    int MoEOverlayCPUServiceMeasurement::rowsForPhase(ExpertHistogramSource source)
    {
        return phaseGeometry(source).second;
    }

    uint64_t MoEOverlayCPUServiceMeasurement::measurePrepared(
        const MoEOverlayPreparedExpertPayload &source, DeviceId device, int layer,
        ExpertHistogramSource phase, Geometry geometry,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        if (!device.is_cpu() || layer < 0 || !memory || !source.complete())
            throw std::invalid_argument("CPU prepared-service observation requires a complete admitted CPU expert");
        (void)payloadBytes(geometry, kMaximumRows);
        return measureExpert(source, device, layer, phase, geometry, memory);
    }

    size_t MoEOverlayCPUServiceMeasurement::cpuJointAllocationBytes(
        const std::shared_ptr<const MoEExpertShardCohort>& cohort,size_t member,
        ExpertHistogramSource phase,size_t invocation_workspace_bytes)
    {
        const auto bytes=jointServiceStorageLayout(cohort,member,phase).physicalBytes();
        if(invocation_workspace_bytes>SIZE_MAX-bytes)
            throw std::overflow_error("CPU joint prepared-compute invocation extent overflows");
        return bytes+invocation_workspace_bytes;
    }

    std::optional<MoEOverlayCPUServiceMeasurement::PreparedComputeObservation>
    MoEOverlayCPUServiceMeasurement::measurePreparedCPUJoint(
        const MoEOverlayParticipantResidencyRegistry& registry,uint64_t epoch,
        int participant,int layer,int expert,ExpertHistogramSource source_phase,
        const std::shared_ptr<IGlobalTPContext>& domain,
        const std::shared_ptr<PhysicalMemoryAuthority>& memory,std::span<float> diagnostic_output)
    {
        // This producer borrows authentic source/bank/native owners. The generic
        // FFN stage below is only a mathematical consumer of their known private
        // service views; it never mints a whole-expert payload or serving permit.
        auto* native=dynamic_cast<GlobalTPContext*>(domain.get());
        if(!native || !native->authenticatesOriginalCPUExpertBorrow(domain))
            throw std::invalid_argument("CPU joint prepared-compute requires original admitted native domain owner");
        // After authenticating the original native owner, every rank-local
        // rejection must wake its original peers instead of leaving them in a
        // later collective. No borrowed wrong domain is ever entered.
        struct AbortOnFailure {
            IGlobalTPContext& domain; int exceptions=std::uncaught_exceptions();
            /** @brief Signal the existing native abort when failed setup unwinds. */
            ~AbortOnFailure() {if(std::uncaught_exceptions()>exceptions)domain.requestAbort();}
        } abort_on_failure{*domain};
        if(!memory || epoch!=registry.initialEpoch() || expert<0)
            throw std::invalid_argument("CPU joint prepared-compute requires original admitted initial source");
        std::string error;
        auto source_use=registry.acquireOriginalPreparedSource(&error);
        if(!source_use)throw std::invalid_argument("CPU joint prepared-compute original source: "+error);
        const auto cold=registry.initialCPUJointLayer(participant,layer);
        const auto frame=cold?cold->acquireEpoch(epoch):std::nullopt;
        if(!frame || !frame->bankLease() || !frame->sourceSelection())
            throw std::invalid_argument("CPU joint prepared-compute initial source/bank is absent");
        const auto& source=*frame->sourceSelection();
        const auto& cohort=source.cohort();
        const auto member=source.participantIndex();
        const auto* placement=registry.initialOwnerMap().placementFor(layer,expert);
        const auto& bank=frame->bankLease()->layerForModelLayer(layer);
        if(!placement || !sameSourceOwner(placement->shard_cohort,cohort) ||
            !source.contains(expert) || member!=static_cast<size_t>(domain->myIndex()) ||
            expert>=static_cast<int>(bank.cpu_fragments.size()) ||
            expert>=static_cast<int>(bank.resident_mask.size()) || !bank.resident_mask[expert])
            throw std::invalid_argument("CPU joint prepared-compute changed original target/member/expert");
        const auto& payload=bank.cpu_fragments[expert];
        if(!payload.valid() || payload.model()!=registry.originalModel() ||
            payload.expertId()!=expert || payload.participantIndex()!=member ||
            !sameSourceOwner(payload.cohort(),cohort) ||
            !sameSourceOwner(payload.preparation(),source_use->preparation()))
            throw std::invalid_argument("CPU joint prepared-compute has foreign prepared fragment provenance");
        const auto [phase,rows]=phaseGeometry(source_phase);
        const auto layout=jointServiceStorageLayout(cohort,member,source_phase);
        const auto& geometry=cohort->ownership(member).geometry();
        const auto device=source.participant().device;
        const int h=geometry.model_columns,fi=layout.route.intermediate(),e=geometry.experts;
        if(!diagnostic_output.empty() && (!layout.route.isRoot() ||
            diagnostic_output.size()!=static_cast<size_t>(rows)*h))
            throw std::invalid_argument("CPU joint prepared-compute diagnostic output requires exact root extent");
        requireProjection(payload.gate(),device,fi,h);
        requireProjection(payload.up(),device,fi,h);
        requireProjection(payload.down(),device,h,fi);

        FP32Tensor input({static_cast<size_t>(rows),static_cast<size_t>(h)},layout.service.input.allocate(memory),device);
        FP32Tensor indices({static_cast<size_t>(rows),1},layout.service.indices.allocate(memory),device);
        FP32Tensor weights({static_cast<size_t>(rows),1},layout.service.weights.allocate(memory),device);
        FP32Tensor output({static_cast<size_t>(rows),static_cast<size_t>(h)},layout.service.output.allocate(memory),device);
        FP32Tensor packed({layout.packed.capacity()},layout.packed.allocate(memory),device);
        FP32Tensor combined({layout.combined.capacity()},layout.combined.allocate(memory),device);
        FP32Tensor release({1},layout.release.allocate(memory),device);
        auto ranges=layout.ranges.allocate(memory);
        auto maps=layout.maps.allocate(memory);
        auto required=layout.required.allocate(memory);
        for(int row=0;row<rows;++row) {
            indices.mutable_typed_data()[row]=static_cast<float>(expert);
            weights.mutable_typed_data()[row]=1.0f;
            for(int col=0;col<h;++col)input.mutable_typed_data()[static_cast<size_t>(row)*h+col]=
                0.0001f*static_cast<float>((col%31*13+row*7)%31-15);
        }
        std::array<std::shared_ptr<ITensorGemm>,3> engines{
            KernelFactory::createExpertServiceExecutionView(payload.gate(),device),
            KernelFactory::createExpertServiceExecutionView(payload.up(),device),
            KernelFactory::createExpertServiceExecutionView(payload.down(),device)};
        auto config=layout.grouped_config;
        config.physical_memory_authority=memory;
        auto workspace=std::make_shared<CPUGroupedMoESerialWorkspace>(std::move(config));
        MoEExpertComputeStage::Params p;
        p.device_id=device;p.input=&input;p.output=&output;p.output_registered_in_arena=false;
        p.routing_indices=&indices;p.routing_weights=&weights;p.seq_len=rows;p.d_model=h;
        p.expert_intermediate=fi;p.num_experts=e;p.top_k=1;p.layer_idx=layer;
        p.expert_mask.assign(e,false);p.expert_mask[expert]=true;
        p.prepared_gate_gemm.assign(e,nullptr);p.prepared_up_gemm.assign(e,nullptr);p.prepared_down_gemm.assign(e,nullptr);
        p.prepared_gate_gemm[expert]=engines[0].get();p.prepared_up_gemm[expert]=engines[1].get();p.prepared_down_gemm[expert]=engines[2].get();
        p.expert_weight_resolution_policy=MoEExpertWeightResolutionPolicy::PreparedRegistryOnly;
        p.cpu_router_q8_input_publication=CPURouterQ8InputPublicationPolicy::PublishTransportedRows;
        p.cpu_grouped_serial_workspace=std::move(workspace);p.service_phase=phase;
        p.force_decode_equivalent_verifier_prefill=phase==MoEOverlayServicePhaseHint::GroupedVerifier;
        p.canonical_route_contributions=&packed;
        p.canonical_route_arithmetic=MoECanonicalRouteArithmeticPolicy::UnweightedExpertRowThenOrderedFMA;
        p.canonical_route_layout=MoECanonicalRoutePublicationLayout::PackedIndexedRouteRows;
        MoEExpertComputeStage compute(std::move(p));
        const auto requirements=compute.getWorkspaceRequirements(rows);
        DeviceWorkspaceManager invocation(device,requirements.total_bytes_with_alignment(),memory);
        if(!invocation.allocate(requirements))throw std::runtime_error("CPU joint prepared-compute invocation admission failed");
        compute.bindWorkspace(&invocation);
        MoECPUExpertShardRouteStage gather({.original_source=frame->sourceSelection(),.tp=domain.get(),
            .packed_records=&packed,.route_indices=&indices,.route_weights=&weights,
            .combined_routes=layout.route.isRoot()?&combined:nullptr,
            .source_ranges=std::span<RootedFloatRecordRange>(ranges.data(),ranges.size()),
            .participant_route_maps=std::span<int32_t>(maps.data(),maps.size()),
            .required_route_weights=std::span<float>(required.data(),required.size()),
            .rows=rows,.top_k=1,.columns=h,.root_participant=0,
            .stage_name="overlay_prepared_cpu_target_compute",.original_tp_owner=domain,
            .standalone_execution_phase=source_phase});
        auto context=IDeviceContext::create(device);
        // Metadata-only job agreement is outside the timed boundary. A caller
        // cannot accidentally combine different experts/phases with matching row
        // shapes. Fixed eight-member storage is the canonical cohort maximum;
        // the collective carries exactly one live job record per actual member.
        struct Job { uint64_t epoch; int layer,expert,phase,tier,rows,h,f,e; };
        const Job job{epoch,layer,expert,static_cast<int>(source_phase),placement->tier_idx,
            rows,h,geometry.intermediate_columns,e};
        std::array<Job,8> jobs{};
        if(domain->degree()<2 || domain->degree()>static_cast<int>(jobs.size()) ||
            !domain->allgatherBytes(&job,jobs.data(),sizeof(job)))
            throw std::runtime_error("CPU joint prepared-compute job agreement failed");
        for(int i=0;i<domain->degree();++i) {
            const auto& j=jobs[i];
            if(j.epoch!=job.epoch || j.layer!=job.layer || j.expert!=job.expert || j.phase!=job.phase ||
                j.tier!=job.tier || j.rows!=job.rows || j.h!=job.h || j.f!=job.f || j.e!=job.e)
                throw std::invalid_argument("CPU joint prepared-compute members selected different jobs");
        }
        const auto execute=[&] {
            if(!compute.execute(context.get()) || !gather.execute(context.get()))
                throw std::runtime_error("CPU joint prepared-compute FFN/gather failed");
            if(layout.route.isRoot() && !cpu::moe::reduceCanonicalRouteRowsOrderedFMA({
                .route_rows=combined.typed_data(),.route_weights=weights.typed_data(),
                .output=output.mutable_typed_data(),.rows=rows,.top_k=1,.columns=h}))
                throw std::runtime_error("CPU joint prepared-compute canonical fold failed");
        };
        execute(); // Actual FFN/collective/fold warmup, outside all measured intervals.
        uint64_t total=0;
        for(int sample=0;sample<kMeasuredSamples;++sample) {
            domain->barrier(); // Input-ready rendezvous is explicitly OUTSIDE root wall.
            release.mutable_typed_data()[0]=layout.route.isRoot()?static_cast<float>(sample+1):0.0f;
            const auto start=layout.route.isRoot()?std::chrono::steady_clock::now():
                std::chrono::steady_clock::time_point{};
            if(!domain->broadcastFloatElements(&release,1,0,"overlay_prepared_cpu_target_start") ||
                release.typed_data()[0]!=static_cast<float>(sample+1))
                throw std::runtime_error("CPU joint prepared-compute start release failed");
            execute();
            if(layout.route.isRoot()) {
                const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
                if(ns<=0 || static_cast<uint64_t>(ns)>UINT64_MAX-total)
                    throw std::runtime_error("CPU joint prepared-compute root timing overflows");
                total+=static_cast<uint64_t>(ns);
            }
        }
        if(!layout.route.isRoot())return std::nullopt;
        for(size_t i=0;i<output.numel();++i)
            if(!std::isfinite(output.typed_data()[i]))
                throw std::runtime_error("CPU joint prepared-compute canonical output is nonfinite");
        if(!diagnostic_output.empty())
            std::copy_n(output.typed_data(),output.numel(),diagnostic_output.data());
        return PreparedComputeObservation{.target=MoEOverlayRouteTarget::cpuCohort(placement->tier_idx),
            .layer=layer,.expert=expert,.phase=source_phase,.logical_rows_per_invocation=static_cast<uint64_t>(rows),
            .measured_invocations=kMeasuredSamples,.total_root_wall_nanoseconds=total};
    }

    std::vector<MoEOverlayParticipantLayerServiceTotals> MoEOverlayCPUServiceMeasurement::measure(
        const MoEOverlayParticipantResidencyRegistry &registry, uint64_t epoch,
        const MoEOverlayEconomyCalibrationLayerCatalog &catalog,
        const ExpertHistogramProductionTopology &topology, Geometry geometry,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory,
        const ModelMemoryProfile *model_profile, const RankInventory *rank_inventory)
    {
        if (!memory) throw std::invalid_argument("Prepared-service measurement requires physical admission");
        if (epoch != registry.initialEpoch())
            throw std::logic_error("CPU prepared-service measurement may only lease the published initial epoch");
        (void)payloadBytes(geometry, kMaximumRows); // Validate before indexing the immutable banks.
        const auto started = std::chrono::steady_clock::now();
        const auto ids = registry.localParticipantIds();
        std::vector<MoEOverlayParticipantLayerServiceTotals> live;
        std::vector<MoEOverlayParticipantLayerServiceTotals> result;
        for (int id : ids)
        {
            std::vector<MoEOverlayParticipantLayerServiceTotals> rows;
            const auto endpoint = registry.endpoint(id);
            if (!endpoint || endpoint->firstModelLayer() != catalog.firstModelLayer() ||
                static_cast<std::size_t>(endpoint->numLayers()) != catalog.layerCount())
                throw std::logic_error("CPU prepared-service registry belongs to another model stage");
            if (!endpoint->trySnapshotServiceMeasurements(&rows))
                throw std::logic_error("CPU prepared-service setup raced a service writer");
            live.insert(live.end(), rows.begin(), rows.end());
            for (size_t layer = 0; layer < catalog.layerCount(); ++layer)
                result.push_back({.participant_id = id, .layer = catalog.firstModelLayer() + static_cast<int>(layer)});
        }
        std::map<DeviceId,std::shared_ptr<RetainedParticipantGraphExecutable>> gpu_calibrations;
        size_t observations = 0;
        for (const auto &gap : catalog.serviceEvidenceGaps(live, ids, topology))
        {
            const auto endpoint = registry.endpoint(gap.participant_id);
            const auto bank = endpoint->acquire(epoch);
            if (!bank) throw std::logic_error("CPU prepared-service lost its published bank lease");
            int selected_layer = -1;
            int selected_expert = -1;
            for (int layer : gap.eligible_layers)
            {
                const auto &entries = bank->layerForModelLayer(layer);
                for (size_t expert = 0; expert < entries.experts.size(); ++expert)
                    if (entries.resident_mask.at(expert) && entries.experts[expert].complete())
                    {
                        selected_layer = layer;
                        selected_expert = static_cast<int>(expert);
                        break;
                    }
                if (selected_expert >= 0) break;
            }
            if (selected_expert < 0)
                throw std::logic_error("Prepared service unsupported: priced class has no complete resident source; participant=" +
                    std::to_string(gap.participant_id) + " device=" + endpoint->device().toString() +
                    " representative_layer=" + std::to_string(gap.representative_layer) +
                    " phase=" + std::to_string(expertHistogramProductionSourceIndex(gap.source)));
            uint64_t duration = 0;
            if (bank->device.is_cpu())
                duration = measurePrepared(bank->layerForModelLayer(selected_layer).experts[selected_expert],
                    bank->device, selected_layer, gap.source, geometry, memory);
            else
            {
                if (!model_profile || !rank_inventory || !bank->device.is_gpu() || rank_inventory->rank != memory->worldRank())
                    throw std::invalid_argument("Prepared GPU service requires original rank/model workspace metadata");
                const auto physical=std::find_if(rank_inventory->gpus.begin(),rank_inventory->gpus.end(),[&](const auto& gpu){
                    return gpu.type==bank->device.type && gpu.local_device_id==bank->device.ordinal;
                });
                if(physical==rank_inventory->gpus.end() || physical->compute_units<=0)
                    throw std::invalid_argument("Prepared GPU service lost its original compute geometry");
                const auto invocation = WorkspaceMemoryEstimator::estimatePreparedExpertService(*model_profile,
                    {.device = bank->device, .device_compute_units = physical->compute_units,
                     .resident_graph_rows = kMaximumRows, .first_layer = catalog.firstModelLayer(),
                     .last_layer = catalog.firstModelLayer() + int(catalog.layerCount()) - 1,
                     .apportioned_routed_experts = true});
                auto& calibration=gpu_calibrations[bank->device];
                if(!calibration)calibration=PlanningGPUExpertMeasurement::beginPreparedCalibration(bank->device,memory);
                duration = PlanningGPUExpertMeasurement::measurePrepared(
                    bank->layerForModelLayer(selected_layer).experts[selected_expert], bank->device,
                    selected_layer, gap.source, geometry.d_model, geometry.intermediate, memory, invocation, calibration);
            }
            const auto participant_index = static_cast<size_t>(std::lower_bound(ids.begin(), ids.end(), gap.participant_id) - ids.begin());
            auto &row = result.at(participant_index * catalog.layerCount() +
                static_cast<std::size_t>(selected_layer - catalog.firstModelLayer()));
            const auto phase = expertHistogramProductionSourceIndex(gap.source);
            row.total_nanoseconds[phase] = duration;
            row.activation_count[phase] = static_cast<uint64_t>(phaseGeometry(gap.source).second) * kMeasuredSamples;
            row.sample_count[phase] = kMeasuredSamples;
            ++observations;
            LOG_DEBUG("[ExpertOverlay] prepared service participant=" << gap.participant_id
                << " layer=" << selected_layer << " expert=" << selected_expert
                << " phase=" << phase << " rows=" << phaseGeometry(gap.source).second
                << " samples=" << kMeasuredSamples << " nanoseconds=" << duration);
        }
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        LOG_INFO("[ExpertOverlay] prepared service observations=" << observations
            << " elapsed_ms=" << elapsed_ms);
        return result;
    }
}
