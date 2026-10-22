/**
 * @file OriginalOutputSliceNativeFixture.cpp
 * @brief Captured production native Q/K/V original-output-slice byte proof.
 *
 * Actual native sources are prepared once through the existing GPU loader and
 * Factory/store publication. The same unopted Q engine and original full K/V
 * engines provide the byte oracle. Independent prepared K/V row views carry
 * store-authenticated original-N policy while their storage/launch N stays local.
 * This is a synthetic component correctness gate, not model/economy evidence.
 */
#include "OriginalOutputSliceNativeFixture.h"
#include <gtest/gtest.h>
#include "GpuPreparedGemmHarness.h"
#include "QuantizedVerifierFormats.h"
#include "ScopedGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/compute_stages/ComputeStageUtils.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "transfer/TransferEngine.h"
#include "utils/GemmContext.h"
#include <array>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
extern "C" int cudaNativeVNNIGemvTuned_getSerialPartitionN();
#endif
#ifdef HAVE_ROCM
extern "C" int rocmGemv_native_vnni_get_serial_partition_n();
#endif

namespace llaminar2::test::original_output_slice_native
{
/** @brief Fail without publishing partial proof. @param condition Required invariant. @param message Precise failed edge. */
inline void require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error("Original-output slice native: " + message);
}

/** @param device Exact backend. @return Existing host arithmetic-scope observation, never device state. */
inline int currentPolicy(DeviceId device)
{
#ifdef HAVE_CUDA
    if (device.is_cuda()) return cudaNativeVNNIGemvTuned_getSerialPartitionN();
#endif
#ifdef HAVE_ROCM
    if (device.is_rocm()) return rocmGemv_native_vnni_get_serial_partition_n();
#endif
    throw std::invalid_argument("Original-output slice requires a compiled GPU backend");
}

/** @param original Original rows. @param first Row origin. @param local Physical rows. @param k Complete K. @return Actual fixture interval. */
inline WeightSliceSpec interval(int original, int first, int local, int k)
{
    return {.source_rows=size_t(original), .source_cols=size_t(k),
        .row_start=size_t(first), .row_count=size_t(local), .col_count=size_t(k),
        .inner_is_presliced=true};
}

/** @brief Retain every native/host owner until the exact stream is retired. */
struct State
{
    static constexpr int q_n=128, original_n=512, local_n=256, k=2560, max_m=17;
    DeviceId device;
    const int row_capacity; ///< Immutable physical rows for this independently captured fixture.
    std::string source_label; ///< Canonical synthetic source alias, not a model/GGUF identity.
    IBackend *backend;
    ScopedGPUStream stream;
    std::vector<std::shared_ptr<TensorBase>> sources;
    std::vector<GpuPreparedGemm> prepared;
    std::unique_ptr<FP32Tensor> input;
    std::array<std::unique_ptr<FP32Tensor>,6> output;
    WorkspaceRequirements requirements;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::unique_ptr<DeviceWorkspaceManager> workspace;
    std::vector<std::unique_ptr<IGPUGraphCapture>> graphs;
    std::vector<float> input_host;
    std::array<std::vector<float>,6> actual;
    std::array<bool,6> observed_nonzero{};
    std::optional<OriginalOutputSlice> k_fact, v_fact;

    /**
     * @brief Establish one exact row-capacity/stream owner before setup can enqueue.
     * @param endpoint Exact GPU.
     * @param rows Physical capacity; the small suite retains its original17-row default.
     * A separate bulk state keeps small verifier poison/readback/workspace bounded.
     */
    explicit State(DeviceId endpoint,int rows=max_m)
        : device(endpoint), row_capacity(rows), backend(getBackendFor(endpoint)), stream(endpoint)
    {
        require(backend != nullptr,"backend missing");
        require(row_capacity>0,"row capacity must be positive");
    }
    /** @brief Join before graph, temporary-copy, workspace, prepared and source owners retire. */
    ~State()
    {
        (void)backend->synchronizeStream(stream.get(),device.ordinal);
        for (auto &entry : prepared)
            if (entry.kernel)
            {
                if (auto *consumer=dynamic_cast<IWorkspaceConsumer *>(entry.kernel)) consumer->unbindWorkspace();
                entry.kernel->clearGPUStreamBinding();
            }
    }

    /** @param local Select physical source views. @return Actual public production descriptors. */
    std::vector<ITensorGemm::TensorProjectionDesc> projections(bool local) const
    {
        if (!local)
            return {{prepared[0].kernel,output[0].get(),q_n,nullptr,"Q"},
                    {prepared[1].kernel,output[1].get(),original_n,nullptr,"K"},
                    {prepared[2].kernel,output[2].get(),original_n,nullptr,"V"}};
        return {{prepared[0].kernel,output[3].get(),q_n,nullptr,"Q"},
                {prepared[3].kernel,output[4].get(),local_n,nullptr,"K",k_fact},
                {prepared[4].kernel,output[5].get(),local_n,nullptr,"V",v_fact}};
    }

    /** @brief Prepare once and bind exact declared scratch before capture. @param format Canonical original source. @param seed Deterministic synthetic data seed. */
    void prepare(const QuantizedVerifierFormatCase &format, unsigned seed)
    {
        source_label=format.label;
        sources.emplace_back(format.create({q_n,k},seed));
        sources.emplace_back(format.create({original_n,k},seed+1));
        sources.emplace_back(format.create({original_n,k},seed+2));
        sources.push_back(sources[1]->create_view({local_n,k},0));
        sources.push_back(sources[2]->create_view({local_n,k},size_t(local_n)*k));
        std::vector<TensorBase *> raw;
        for (const auto &source:sources) raw.push_back(source.get());
        const std::array<GpuPreparedGemmSourceBinding,5> bindings{{
            {sources[0],interval(q_n,0,q_n,k)},
            {sources[1],interval(original_n,0,original_n,k)},
            {sources[2],interval(original_n,0,original_n,k)},
            {sources[3],interval(original_n,0,local_n,k)},
            {sources[4],interval(original_n,local_n,local_n,k)}}};
        prepared=makeGpuPreparedGemmBatch(raw,device,"test.original_output_slice",ModelContextId{950086},bindings);
        for (size_t index=0;index<prepared.size();++index)
        {
            DeviceNativeVNNIMatrixDesc actual_descriptor{};
            require(prepared[index].kernel->exportNativeVNNIMatrixDesc(actual_descriptor) &&
                actual_descriptor.valid(),"actual native prepared descriptor missing");
            require(prepared[index].kernel->get_n()==actual_descriptor.n &&
                prepared[index].kernel->get_k()==actual_descriptor.k,
                "public GEMM dimensions differ from actual prepared native descriptor");
            require(actual_descriptor.n==int(sources[index]->rows()) && actual_descriptor.k==k &&
                actual_descriptor.codebook_id==format.device_execution_codebook_id &&
                actual_descriptor.source_identity_present &&
                actual_descriptor.source_codebook_id==format.source_codebook_id &&
                bool(actual_descriptor.source_is_superblock)==format.source_is_superblock,
                "actual prepared/source family or physical extent differs");
        }
        k_fact=prepared[3].store->originalOutputSlice(prepared[3].ref,*sources[3]);
        v_fact=prepared[4].store->originalOutputSlice(prepared[4].ref,*sources[4]);
        require(k_fact->policyColumns()==original_n && v_fact->policyColumns()==original_n &&
            k_fact->firstColumn()==0 && v_fact->firstColumn()==local_n &&
            k_fact->physicalColumns()==local_n && v_fact->physicalColumns()==local_n &&
            k_fact->reductionColumns()==k && v_fact->reductionColumns()==k,"actual source interval differs");
        require(k_fact->sourceIdentity().codebook_id==format.source_codebook_id &&
            v_fact->sourceIdentity().is_superblock==format.source_is_superblock,"source family differs");
        const int outer_policy=currentPolicy(device);
        {
            auto old=prepared[3].kernel->beginOutputPartitionEquivalenceScope(local_n,7);
            require(currentPolicy(device)==7,"old policy scope missing");
            {
                auto typed=prepared[3].kernel->beginOriginalOutputSliceScope(*k_fact);
                require(currentPolicy(device)==original_n,"original policy scope missing");
            }
            require(currentPolicy(device)==7,"typed scope failed to restore enclosing policy");
        }
        require(currentPolicy(device)==outer_policy,"nested scope leaked");
        input=TestTensorFactory::createFP32Zeros({size_t(row_capacity),k});
        input_host.resize(size_t(row_capacity)*k);
        for (size_t index=0;index<output.size();++index)
        {
            const int n=index%3==0 ? q_n : index<3 ? original_n : local_n;
            output[index]=TestTensorFactory::createFP32Zeros({size_t(row_capacity)+1,size_t(n)});
            actual[index].resize((size_t(row_capacity)+1)*n);
            require(output[index]->allocateOnDevice(device,stream.get()),"output allocation failed");
            TransferEngine::prepareDeviceInput(output[index].get(),device,stream.get());
        }
        TransferEngine::prepareDeviceInput(input.get(),device,stream.get());
        const int incoming=currentPolicy(device);
        const std::vector<int> planned_rows=row_capacity==max_m
            ? std::vector<int>{1,3,max_m} : std::vector<int>{row_capacity};
        for (int rows:planned_rows)
        {
            // Inspect declarations before the serial-family merge: the full
            // oracle bank must not hide a local policyN-as-storage regression.
            // Existing N-linear banks share the same source K tree and row
            // envelope here; N512/N256 are also aligned counter-tile widths.
            // Halving physical N must therefore halve these actual declarations.
            for (size_t local_index:{size_t{3},size_t{4}})
            {
                const size_t original_index=local_index-2;
                auto *full_consumer=dynamic_cast<IWorkspaceConsumer *>(prepared[original_index].kernel);
                auto *local_consumer=dynamic_cast<IWorkspaceConsumer *>(prepared[local_index].kernel);
                require(full_consumer && local_consumer,"workspace interfaces missing");
                const auto full_requirements=full_consumer->getWorkspaceRequirements(rows,original_n,k);
                const auto &fact=local_index==3 ? *k_fact : *v_fact;
                auto scope=prepared[local_index].kernel->beginOriginalOutputSliceScope(fact);
                const auto local_requirements=local_consumer->getWorkspaceRequirements(rows,local_n,k);
                std::vector<const char *> physical_banks{
                    GemmWorkspaceBuffers::ACC_INT32,GemmWorkspaceBuffers::TEMP_C_FP32};
                if (device.is_cuda()) physical_banks.push_back(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
                else
                {
                    physical_banks.push_back(GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL);
                    physical_banks.push_back(GemmWorkspaceBuffers::ROCM_SELFREDUCE_COUNTERS);
                }
                for (const auto *name:physical_banks)
                {
                    const auto *whole=full_requirements.find(name);
                    const auto *part=local_requirements.find(name);
                    require(whole && part && part->size_bytes && whole->size_bytes==2*part->size_bytes,
                        std::string("local scratch is not physical-width sized: ")+name);
                }
            }
            for (const bool local:{false,true})
            {
                const auto bundle=projections(local);
                WorkspaceRequirements bundle_requirements;
                std::vector<int> widths;
                for (const auto &projection:bundle)
                {
                    projection.requireOriginalOutputSlice(k);
                    auto scope=projection.beginOriginalOutputSliceScope();
                    auto *consumer=dynamic_cast<IWorkspaceConsumer *>(projection.kernel);
                    require(consumer != nullptr,"missing workspace interface");
                    bundle_requirements.merge(consumer->getWorkspaceRequirements(rows,projection.n,k));
                    widths.push_back(projection.n);
                }
                dynamic_cast<IWorkspaceConsumer *>(prepared[0].kernel)->appendFusedProjectionWorkspaceRequirements(
                    bundle_requirements,rows,widths,k);
                addCudaConcurrentDecodeGemvSideStreamWorkspace(bundle_requirements,device,rows,3);
                // Each stage appends its own fused banks once; the serial
                // family then merges complete stage declarations by name/max.
                requirements.merge(bundle_requirements);
            }
        }
        require(currentPolicy(device)==incoming,"workspace scope leaked to Q or caller");
        const size_t bytes=requirements.total_bytes_with_alignment();
        PhysicalMemoryPlanBuilder planner;
        planner.add({.world_rank=0,.device=device,.total_bytes=bytes,.admission_available_bytes=bytes},
            PhysicalMemoryOwner::ExecutionWorkspace,bytes);
        memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(planner.build()),0);
        workspace=std::make_unique<DeviceWorkspaceManager>(device,bytes,memory);
        require(workspace->allocate(requirements),"declared workspace allocation failed");
        for (auto &entry:prepared)
        {
            entry.kernel->setGPUStream(stream.get());
            dynamic_cast<IWorkspaceConsumer *>(entry.kernel)->bindWorkspace(workspace.get());
            require(entry.kernel->prepareFusedProjectionGraphCapture(3),"capture preparation failed");
        }
        require(backend->synchronizeStream(stream.get(),device.ordinal),"setup retirement failed");
    }

    /** @brief Poison all mutable request scratch and all physical output/guard bytes. */
    void poison()
    {
        for (const auto &buffer:requirements.buffers)
            if (buffer.size_bytes && buffer.content_lifetime==WorkspaceContentLifetime::ParticipantExecution)
            {
                void *address=workspace->getBuffer(buffer.name);
                if (!address && !buffer.required) continue;
                require(address != nullptr,"required poison bank missing: "+buffer.name);
                require(workspace->getBufferSize(buffer.name)>=buffer.size_bytes,
                    "actual scratch capacity is below its declaration: "+buffer.name);
                require(backend->memset(address,0xa5,buffer.size_bytes,device.ordinal,stream.get()),
                    "scratch poison failed: "+buffer.name+" bytes="+std::to_string(buffer.size_bytes));
            }
        for (size_t index=0;index<output.size();++index)
            require(backend->memset(output[index]->gpu_data_ptr(),0xff,actual[index].size()*sizeof(float),
                device.ordinal,stream.get()),"output poison failed");
    }

    /** @brief Update resident input with cancellation/signed-zero histories. @param history Retained replay index. */
    void changeInput(int history)
    {
        for (size_t index=0;index<input_host.size();++index)
        {
            const int centered=int((index*13+size_t(history)*7)%43)-21;
            float value=float(centered)*0.03125f;
            if (history%5==0) value=((index/32)%2 ? -1.f : 1.f)*float(1+index%3)*0.25f;
            if (history%7==0 && index%17!=0) value=index%2 ? -0.f : 0.f;
            input_host[index]=value;
        }
        require(backend->hostToDevice(input->gpu_data_ptr(),input_host.data(),input_host.size()*sizeof(float),
            device.ordinal,stream.get()),"resident input publication failed");
    }

    /** @brief Record fresh production graph; scopes end before replay. @param context Exact worker. @param rows Active M. @param grouped Public verifier API. @param local Source-sliced arm. @return State-owned executable. */
    IGPUGraphCapture &capture(IWorkerGPUContext &context,int rows,bool grouped,bool local)
    {
        require(rows>0 && rows<=row_capacity,"capture rows exceed immutable capacity");
        graphs.push_back(context.createGraphCapture(stream.get()));
        require(graphs.back()!=nullptr,"native capture owner missing");
        auto &graph=*graphs.back();
        require(!graph.hasExecutable() && graph.nodeCount()==0,"capture owner is not fresh");
        const int incoming=currentPolicy(device);
        const auto bundle=projections(local);
        // Setup copies publish external events. Join their exact stream before
        // capture and declare the same dependency edge the production graph uses.
        TransferEngine::requireDeviceInput(input.get(),device,stream.get());
        int stage_identity=0;
        GraphCaptureDependencyLedger::StagePlan plan{
            .stage_identity=&stage_identity,
            .stage_name="native_original_output_slice",
            .external_inputs={input.get()},
            .internal_inputs={},
            .outputs={bundle[0].output,bundle[1].output,bundle[2].output}};
        GraphCaptureDependencyLedger ledger(device,stream.get(),{std::move(plan)},
            "native original-output slice");
        {
            ScopedBackendGraphCapture recording(context,graph,"native original-output slice",&ledger);
            require(recording.begin(),"capture begin failed");
            {
                ScopedGraphCaptureStage stage(&stage_identity);
                const bool ok=grouped
                    ? prepared[0].kernel->multiply_fused_verifier_rows_decode_equivalent(input.get(),bundle,rows,k,nullptr,workspace.get())
                    : prepared[0].kernel->multiply_fused_tensor(input.get(),bundle,rows,k,nullptr,workspace.get());
                require(ok,"public fused recording failed");
                stage.complete();
            }
            recording.finish();
        }
        require(currentPolicy(device)==incoming,"recording scope leaked");
        require(graph.instantiate(),"native instantiate failed");
        std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
        require(graph.inspectKernelNodes(nodes,&error),"actual node inventory failed: "+error);
        require(!nodes.empty(),"empty compute graph");
        for (const auto &node:nodes)
            require(node.valid() && node.name_resolved,"incomplete actual function/grid/resource evidence");
        if (!grouped && rows==512)
        {
            size_t prefill_producers=0;
            for (const auto &node:nodes)
            {
                const bool producer=device.is_cuda()
                    ? node.name.find("nativeVnniTC_BK64")!=std::string::npos ||
                      node.name.find("nativeVnniTC_BK256")!=std::string::npos
                    : node.name.find("native_vnni_gemm_kernel")!=std::string::npos;
                require(node.name.find("gemv_native_vnni")==std::string::npos,
                    "ordinary bulk graph contains a native GEMV/verifier producer");
                if (!producer) continue;
                ++prefill_producers;
                // The existing CUDA launchers put M tiles on grid.x; HIP's
                // cooperative launcher puts them on grid.y. Narrow Q can
                // legitimately have a single column tile on the other axis.
                require((device.is_cuda() ? node.grid_x : node.grid_y)>1,
                    "bulk producer lacks a multi-tile row grid");
                std::fprintf(stderr,
                    "ORIGINAL_SLICE_BULK_KERNEL device=%s source=%s arm=%s M=%d "
                    "function=%llu grid=%u/%u/%u block=%u/%u/%u registers=%u "
                    "dynamic_shared=%zu static_shared=%zu local_bytes=%zu active_blocks=%u symbol=%s\n",
                    device.toString().c_str(),source_label.c_str(),local ? "slice" : "full",rows,
                    static_cast<unsigned long long>(node.function_identity),
                    node.grid_x,node.grid_y,node.grid_z,node.block_x,node.block_y,node.block_z,
                    node.registers_per_thread,node.dynamic_shared_memory_bytes,node.static_shared_memory_bytes,
                    node.local_memory_bytes_per_thread,node.max_active_blocks_per_sm,node.name.c_str());
            }
            // One real bulk producer for each Q/K/V member; reducers and shared
            // activation quantization remain separately visible in this graph.
            require(prefill_producers==3,"ordinary bulk graph must contain three native prefill producers");
            std::fprintf(stderr,
                "ORIGINAL_SLICE_BULK_GRAPH device=%s source=%s arm=%s M=%d K=%d "
                "physical_N=%d/%d/%d arithmetic_N=%d/%d/%d producers=%zu nodes=%zu\n",
                device.toString().c_str(),source_label.c_str(),local ? "slice" : "full",rows,k,
                q_n,local ? local_n : original_n,local ? local_n : original_n,
                q_n,original_n,original_n,prefill_producers,nodes.size());
            std::fflush(stderr);
        }
        return graph;
    }

    /** @brief Read physical output and check guard bytes. @param local Sliced arm. @param rows Active M. */
    void read(bool local,int rows)
    {
        const size_t first=local ? 3 : 0;
        for (size_t index=first;index<first+3;++index)
            require(backend->deviceToHost(actual[index].data(),output[index]->gpu_data_ptr(),
                actual[index].size()*sizeof(float),device.ordinal,stream.get()),"terminal output copy failed");
        require(backend->synchronizeStream(stream.get(),device.ordinal),"terminal retirement failed");
        for (size_t index=first;index<first+3;++index)
        {
            const size_t live=size_t(rows)*output[index]->cols();
            for (size_t value=0;value<live;++value)
            {
                require(std::isfinite(actual[index][value]),"nonfinite/unwritten live result");
                observed_nonzero[index]=observed_nonzero[index] || actual[index][value]!=0.f;
            }
            const auto *tail=reinterpret_cast<const unsigned char *>(actual[index].data()+live);
            const size_t tail_bytes=(actual[index].size()-live)*sizeof(float);
            for (size_t byte=0;byte<tail_bytes;++byte) require(tail[byte]==0xff,"write exceeded physical live rows");
        }
    }

};

/** @brief Prove captured ordinary/grouped original-source slices. @param device Exact GPU endpoint. */
void prove(DeviceId device)
{
    auto &context=GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&] {
        ScopedGemmContext attention_context(GemmContext::ATTN);
        require(quantizedVerifierFormats().size()==native_vnni_formats::kAllSourceFormats.size(),
            "canonical source inventory incomplete");
        unsigned seed=950101;
        // Existing complete strengthened creators avoid degenerate IQ3_S and
        // overflowing IQ4_XS witnesses; format/source/decoder inventory is unchanged.
        for (const auto &format:quantizedMoEVerifierFormats())
        {
            try
            {
            {
            State state(device);
            state.prepare(format,seed);seed+=11;
            const int incoming=currentPolicy(device);
            // Invalid source/engine and old opposite-direction policy must fail
            // before a native node can be admitted to the fresh recording.
            state.graphs.push_back(context.createGraphCapture(state.stream.get()));
            require(state.graphs.back()!=nullptr,"negative capture owner missing");
            auto &negative=*state.graphs.back();
            {
                ScopedBackendGraphCapture recording(context,negative,"original-output slice rejected before enqueue");
                require(recording.begin(),"negative capture begin failed");
                auto bad=state.projections(true);bad[2].original_output_slice=state.k_fact;
                bool rejected=false;
                try { (void)state.prepared[0].kernel->multiply_fused_tensor(state.input.get(),bad,1,State::k,nullptr,state.workspace.get()); }
                catch (const std::invalid_argument &) { rejected=true; }
                require(rejected,"foreign same-shape engine was accepted");
                rejected=false;
                try { auto old=state.prepared[3].kernel->beginOutputPartitionEquivalenceScope(State::local_n,State::original_n); }
                catch (const std::invalid_argument &) { rejected=true; }
                require(rejected,"old two-int direction was relaxed");
                recording.finish();
            }
            require(negative.nodeCount()==0,"rejected fact admitted work");
            for (int rows:{1,3,State::max_m})
                for (const bool grouped:{false,true})
                {
                    state.changeInput(1);
                    require(state.backend->synchronizeStream(state.stream.get(),device.ordinal),"pre-capture input join failed");
                    auto &full=state.capture(context,rows,grouped,false);
                    auto &local=state.capture(context,rows,grouped,true);
                    for (int history=0;history<20;++history)
                    {
                        state.changeInput(history);
                        state.poison();require(full.launch(),"full-source graph replay failed");state.read(false,rows);
                        state.poison();require(local.launch(),"original-slice graph replay failed");state.read(true,rows);
                        require(std::memcmp(state.actual[0].data(),state.actual[3].data(),size_t(rows)*State::q_n*sizeof(float))==0,
                            std::string(format.label)+" Q sibling bytes changed");
                        for (int row=0;row<rows;++row)
                        {
                            require(std::memcmp(state.actual[1].data()+size_t(row)*State::original_n,
                                state.actual[4].data()+size_t(row)*State::local_n,State::local_n*sizeof(float))==0,
                                std::string(format.label)+" K original-prefix bytes changed");
                            require(std::memcmp(state.actual[2].data()+size_t(row)*State::original_n+State::local_n,
                                state.actual[5].data()+size_t(row)*State::local_n,State::local_n*sizeof(float))==0,
                                std::string(format.label)+" V original-offset bytes changed");
                        }
                        require(currentPolicy(device)==incoming,"replay changed host scope");
                    }
                }
            for (bool nonzero:state.observed_nonzero) require(nonzero,std::string(format.label)+" degenerate all-zero projection witness");
            }
            // Retire the small graph/input/source/workspace owner before the
            // separate bulk admission; the two fixtures never coexist.
            // Independently bounded bulk storage leaves the proven small-row
            // suite unchanged. Prepare five sources once for this geometry,
            // outside capture/replays; no per-history packing or override.
            State bulk(device,512);
            bulk.prepare(format,seed-11);
            bulk.changeInput(1);
            require(bulk.backend->synchronizeStream(bulk.stream.get(),device.ordinal),"bulk pre-capture input join failed");
            auto &bulk_full=bulk.capture(context,512,false,false);
            auto &bulk_local=bulk.capture(context,512,false,true);
            const int bulk_policy=currentPolicy(device);
            for (int history:{1,7,0})
            {
                bulk.changeInput(history);
                bulk.poison();require(bulk_full.launch(),"bulk full-source replay failed");bulk.read(false,512);
                bulk.poison();require(bulk_local.launch(),"bulk original-slice replay failed");bulk.read(true,512);
                require(std::memcmp(bulk.actual[0].data(),bulk.actual[3].data(),size_t(512)*State::q_n*sizeof(float))==0,
                    std::string(format.label)+" bulk Q sibling bytes changed");
                for (int row=0;row<512;++row)
                {
                    require(std::memcmp(bulk.actual[1].data()+size_t(row)*State::original_n,
                        bulk.actual[4].data()+size_t(row)*State::local_n,State::local_n*sizeof(float))==0,
                        std::string(format.label)+" bulk K original-prefix bytes changed");
                    require(std::memcmp(bulk.actual[2].data()+size_t(row)*State::original_n+State::local_n,
                        bulk.actual[5].data()+size_t(row)*State::local_n,State::local_n*sizeof(float))==0,
                        std::string(format.label)+" bulk V original-offset bytes changed");
                }
                require(currentPolicy(device)==bulk_policy,"bulk replay changed host scope");
            }
            for (bool nonzero:bulk.observed_nonzero) require(nonzero,std::string(format.label)+" degenerate bulk projection witness");
            }
            catch (const std::exception &error)
            {
                throw std::runtime_error(std::string(format.label)+": "+error.what());
            }
        }
    });
}
}
