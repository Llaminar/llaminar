/**
 * @file MTPNativeFixtureAdmission.inc
 * @brief Planner-admitted native MTP components with original owned FP32 sources.
 *
 * This model-free container never manufactures prepared handles. The real
 * WeightManager consumes the retained frozen sources through its GPU pipeline,
 * and the same PMA owns its prepared-weight pool, graph state and workspaces.
 * This fixture tests graph construction and publication; it does not certify
 * the separate raw non-GEMM upload allocation-accounting boundary.
 */
#include "backends/HardwareInventory.h"
#include "loaders/GPUHostLoadPreflight.h"
#include "planning/MemoryPlanner.h"
#include "execution/local_execution/graph/SchemaFactoryRegistry.h"
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
namespace {
/** @brief Own source tensors and their actual model/Store/PMA across native graph retirement. */
class NativeMTPFixtureAdmission final {
public:
    /**
     * @brief Declare the complete original source and native state before materialization.
     * @param graph Actual Qwen35 graph/schema and authored retained MTP policy.
     * @param mpi Original single-process fixture rank, never an emulated peer.
     * @param source Original complete component bindings, including predictor norms.
     * @param source_owner Real composite owner retaining every source tensor.
     * @param batch Authored concurrent request capacity.
     */
    NativeMTPFixtureAdmission(std::shared_ptr<IGraphBuilder> graph,std::shared_ptr<IMPIContext> mpi,
        const FrozenModelWeightSet &source,std::shared_ptr<void> source_owner,int batch)
        : graph_(std::move(graph)),mpi_(std::move(mpi)),source_owner_(std::move(source_owner))
    {
        const auto &c=graph_->config();const auto device=c.default_device;
        if(!device.is_gpu() || !source_owner_ || batch<=0)
            throw std::invalid_argument("Native MTP fixture requires original GPU/source/request ownership");
        const auto bindings=makeModelWeightBindings(source,device);
        int original_layers=c.n_layers;
        for(const auto &depth:bindings.mtp.depths)original_layers=std::max(original_layers,depth.source_layer_index+1);
        model_=ModelContext::createForTesting("native-mtp-owned-tensor-source",mpi_,original_layers,true);
        manager_=model_->concreteWeightManager();
        manager_->setWeightShardingConfig(SchemaFactoryRegistry::getWeightShardingConfig(graph_->architectureName()));
        manager_->setModelDimensions(c.n_heads,c.n_kv_heads,c.head_dim);
        auto strategy=source.strategy();
        // Same identity projection as the production inference factory.
        strategy.model_id=ModelContextId{reinterpret_cast<uint64_t>(model_.get())};
        strategy.devices={device};
        if(!manager_->preparedWeightStore()->bindModelIdIfUnset(strategy.model_id))
            throw std::logic_error("Native MTP Store changed its original model identity");
        ModelWeightSetBuilder publication(strategy);
        ModelMemoryProfile profile;
        profile.architecture=graph_->architectureName();profile.n_layers=c.n_layers+bindings.mtp.depths.size();
        profile.mtp_layer_count=bindings.mtp.depths.size();profile.d_model=c.d_model;profile.d_ff=c.d_ff;
        profile.n_heads=c.n_heads;profile.n_kv_heads=c.n_kv_heads;profile.head_dim=c.head_dim;
        profile.vocab_size=c.vocab_size;profile.max_seq_len=c.max_seq_len;
        profile.gdn_conv_kernel_size=c.gdn.conv_kernel_size;profile.gdn_state_size=c.gdn.state_size;
        profile.gdn_inner_size=c.gdn.inner_size;profile.gdn_group_count=c.gdn.group_count;
        profile.gdn_time_step_rank=c.gdn.time_step_rank;profile.full_attention_interval=c.gdn.full_attention_interval;
        size_t maximum_source=0;
        for(const auto &original:source.bindings()) {
            if(!original.tensor || original.tensor->native_type()!=TensorType::FP32 || !original.tensor->raw_data())
                throw std::invalid_argument("Native MTP fixture changed its original FP32 source");
            auto binding=original;
            binding.identity.model_id=strategy.model_id;
            binding.tensor_owner=std::shared_ptr<TensorBase>(source_owner_,binding.tensor);
            binding.residency.resident_device=device;
            publication.addBinding(binding);
            int physical_layer=binding.identity.layer;
            for(size_t depth=0;depth<bindings.mtp.depths.size();++depth)
                if(physical_layer==bindings.mtp.depths[depth].source_layer_index)physical_layer=c.n_layers+depth;
            const auto &shape=binding.tensor->shape();
            profile.tensors.push_back({.name=binding.identity.canonical_name,.native_bytes=binding.tensor->size_bytes(),
                .quant_type="F32",.elements=binding.tensor->numel(),.K=shape.size()==2?shape.back():0,
                .layer_index=physical_layer});
            profile.total_native_bytes+=binding.tensor->size_bytes();
            maximum_source=std::max(maximum_source,binding.tensor->size_bytes());
        }
        frozen_=std::make_unique<FrozenModelWeightSet>(strategy,publication.freezeBindings());
        const auto *hw=DeviceManager::instance().hardware();
        if(!hw)throw std::logic_error("Native MTP fixture lost original hardware publication");
        const auto &devices=device.is_cuda()?hw->cuda_devices:hw->rocm_devices;
        const auto found=std::find_if(devices.begin(),devices.end(),[&](const auto &entry){return entry.device_id==device.ordinal;});
        if(found==devices.end())throw std::invalid_argument("Native MTP endpoint is absent from original hardware publication");
        const auto cpu=hw->cpuDevice();
        DevicePlanConfig plan;
        plan.world_rank=0;plan.device=device;plan.device_total_bytes=found->total_memory_bytes;
        plan.device_free_bytes=found->free_memory_bytes;plan.device_compute_units=found->compute_units;
        plan.associated_host_memory=PhysicalMemoryResource{0,DeviceId::cpu(),cpu.total_memory_bytes,cpu.free_memory_bytes};
        plan.cpu_execution=hw->cpu_execution;plan.batch_size=batch;
        plan.max_seq_len=plan.activation_seq_len=c.max_seq_len;
        plan.first_layer=c.pp_layer_offset;plan.last_layer=c.pp_layer_offset+c.n_layers-1;
        plan.local_kv_heads=c.n_kv_heads;plan.kv_precision=kvCachePrecisionToString(c.kv_cache_precision);
        plan.mtp_enabled=retainsMTPGraphCapacity(c.mtp);
        plan.mtp_target_query_rows=resolveMTPRetainedTargetQueryRows(c.mtp);
        plan.generation_request_capacity=std::max(batch,c.mtp.max_request_batch);
        const auto family=resolveMTPVerifierGraphFamily(mtpVerifierExecutionRole(declaredDeviceGenerationExecutionPolicy(
            device.type,DeviceGenerationExecutionDomain::NativeDomain)));
        plan.captured_serving_graphs=resolveCapturedServingGraphMemoryInventory(prefillGraphBucketSizes(c.max_seq_len),
            c.mtp,family,MTPGenerationParentRole::Controller,DeviceGenerationExecutionDomain::NativeDomain);
        const auto staging=resolveGPUWeightLoadMemoryGeometry(maximum_source,configuredGPUWeightLoadMemoryPolicy());
        plan.weight_load_staging={staging.staging_bytes,staging.host_staging_bytes};
        const auto declaration=MemoryPlanner::plan(profile,{plan});
        memory_=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(declaration.physicalPlan()),0);
        manager_->installPhysicalMemoryAuthority(memory_);
    }
    /** @return Same model/graph/rank/authority; no isolated allocation mode is selected. */
    DeviceGraphOrchestrator::Dependencies dependencies() const {
        DeviceGraphOrchestrator::Dependencies deps;
        deps.model_ctx=model_;deps.graph_builder=graph_;deps.mpi_ctx=mpi_;
        deps.weight_manager=manager_;deps.physical_memory_authority=memory_;return deps;
    }
    /** @brief Prepare actual immutable sources through the production GPU pipeline once. */
    void prepare() {
        if(!manager_->prepareWeightsForDevice(*frozen_,graph_->config().default_device))
            throw std::runtime_error("Native MTP original source preparation failed");
        graph_->setPreparedWeightStore(manager_->preparedWeightStore().get());
    }
    /** @return Actual original frozen view retained by the source owner. */
    std::unique_ptr<FrozenModelWeightSet> takeWeights(){return std::move(frozen_);}
private:
    std::shared_ptr<IGraphBuilder> graph_;
    std::shared_ptr<IMPIContext> mpi_;
    std::shared_ptr<void> source_owner_;
    std::shared_ptr<ModelContext> model_;
    std::shared_ptr<WeightManager> manager_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    std::unique_ptr<FrozenModelWeightSet> frozen_;
};
}
#endif
