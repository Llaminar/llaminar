/**
 * @file Test__OriginalOutputSlice.cpp
 * @brief Device-free store authentication and real CPU fused source-slice proofs.
 *
 * Native source tensors and real prepared engines use the production store
 * publication boundary. No GPU, model, dispatch override, reference download,
 * payload digest, or caller-authored engine identity participates in the proof.
 */
#include <gtest/gtest.h>
#include "execution/diagnostics/QueryKVGeometryObservation.h"
#include "utils/DebugEnv.h"
#include <cstdlib>
#include <set>
#include "loaders/PreparedWeightStore.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "execution/compute_stages/stages/FusedQKVGEMMStage.h"
#include "tensors/Tensors.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "../../utils/CPUProjectionTestWorkspace.h"
#include "../../utils/PreparedWeightTestHarness.h"
#include "../../utils/QuantizedVerifierFormats.h"
#include <array>
#include <cstring>
#include <functional>
#include <memory>
#include <omp.h>

namespace
{
using namespace llaminar2;
using namespace llaminar2::cpu::native_vnni;
using Factory = llaminar::v2::kernels::KernelFactory;

/** @brief Retain source, actual packed engine, handle and store through every invocation. */
struct Matrix
{
    std::shared_ptr<TensorBase> source;
    std::shared_ptr<CPUNativeVNNIGemmKernel> engine;
    PreparedWeightStore store{ModelContextId{9917}};
    WeightBinding binding;
    PreparedWeightRef ref;

    /** @brief Publish real native execution with the requested immutable source interval. */
    Matrix(std::shared_ptr<TensorBase> tensor, WeightSliceSpec slice)
        : source(std::move(tensor)), engine(std::make_shared<CPUNativeVNNIGemmKernel>(source.get()))
    {
        binding = test::makePreparedWeightTestBinding(source.get(), DeviceId::cpu(),
            "original.output.weight", ModelContextId{9917});
        binding.tensor_owner = source;
        binding.slice = slice;
        publish();
    }
    /** @return Actual owned publication wrappers for deliberate metadata adversaries. */
    std::shared_ptr<Factory::PreparedGemmHandle> makeHandle() const
    {
        auto handle = std::make_shared<Factory::PreparedGemmHandle>();
        handle->tensor = source.get(); handle->device_id = DeviceId::cpu();
        handle->kind = Factory::GemmPreparationKind::CPU_PACKED;
        handle->prepared_weights = std::make_shared<Factory::PreparedGemmWeights>();
        handle->prepared_weights->kind = handle->kind;
        handle->prepared_weights->kernel = engine.get();
        handle->prepared_weights->owned_kernel = engine;
        return handle;
    }
    /** @brief Republish metadata only for deliberate factory rejection witnesses. */
    void publish()
    {
        ref = store.registerPreparedGemmHandle(binding, PreparedWeightKind::CpuPackedGemm,
            DeviceId::cpu(), makeHandle());
    }
    /** @return The actual store's immutable source interval. */
    OriginalOutputSlice slice() const { return store.originalOutputSlice(ref, *source); }
};

/** @return Complete original matrix row interval; K is never partitioned. */
WeightSliceSpec interval(int original, int first, int physical, int k)
{
    return {.source_rows=static_cast<size_t>(original), .source_cols=static_cast<size_t>(k),
        .row_start=static_cast<size_t>(first), .row_count=static_cast<size_t>(physical),
        .col_count=static_cast<size_t>(k), .inner_is_presliced=true};
}

/** @brief Keep the functional Unit sweep small and restore its ambient worker policy. */
struct Workers
{
    int count = omp_get_max_threads(), dynamic = omp_get_dynamic();
    explicit Workers(int threads = 2) { omp_set_dynamic(0); omp_set_num_threads(threads); }
    ~Workers() { omp_set_num_threads(count); omp_set_dynamic(dynamic); }
};

/** @return Owned FP32 rows; all consumed bytes receive finite changing values. */
std::shared_ptr<FP32Tensor> fp32(int m, int n)
{
    auto result = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m),static_cast<size_t>(n)});
    for (size_t i=0; i<result->numel(); ++i) result->mutable_data()[i]=(static_cast<int>(i%19)-9)*0.03125f;
    return result;
}
}

/** @test Source offsets are generic, and only the actual frozen published binding may mint a fact. */
TEST(OriginalOutputSlice, ActualBindingAndGenericOrigins)
{
    std::shared_ptr<TensorBase> source(test::TestTensorFactory::createQ8_0Random({512,256}, 17));
    for (int tp=1;tp<=8;++tp)
        for (int rank=0;rank<tp;++rank)
        {
            const int first = 512*rank/tp, last=512*(rank+1)/tp;
            auto local=source->create_view({static_cast<size_t>(last-first),256}, first*256);
            Matrix matrix(local, interval(512,first,last-first,256));
            const auto fact=matrix.slice();
            EXPECT_EQ(fact.policyColumns(),512); EXPECT_EQ(fact.firstColumn(),first);
            EXPECT_EQ(fact.physicalColumns(),last-first); EXPECT_EQ(fact.reductionColumns(),256);
            EXPECT_EQ(fact.preparedKind(),matrix.ref.kind);
            EXPECT_EQ(fact.retainedEngine(),matrix.engine.get());
            EXPECT_NO_THROW(fact.require(matrix.engine.get(),last-first,256));
            EXPECT_THROW(fact.require(nullptr,last-first,256),std::invalid_argument);
            EXPECT_THROW(fact.require(matrix.engine.get(),last-first+1,256),std::invalid_argument);
            EXPECT_THROW(fact.require(matrix.engine.get(),last-first,288),std::invalid_argument);
            EXPECT_THROW(matrix.store.originalOutputSlice(matrix.ref,*source),std::invalid_argument);
            auto foreign=matrix.ref; foreign.device=DeviceId::cuda(0);
            EXPECT_THROW(matrix.store.originalOutputSlice(foreign,*local),std::invalid_argument);
        }
}

/** @test Missing original coordinates, foreign arithmetic and K/expert slices fail before recording. */
TEST(OriginalOutputSlice, InvalidSourceGeometryAndLegacyDirection)
{
    auto source=test::TestTensorFactory::createQ8_0Random({256,256},27);
    Matrix matrix(std::move(source),interval(512,256,256,256));
    const auto original=matrix.binding;
    const std::array<std::function<void(WeightBinding&)>,9> mutations{{
        [](auto &b){b.immutable=false;}, [](auto &b){b.slice.source_rows=0;},
        [](auto &b){b.slice.row_start=257;}, [](auto &b){b.slice.row_count=255;},
        [](auto &b){b.slice.col_start=32;}, [](auto &b){b.slice.col_count=224;},
        [](auto &b){b.slice.source_cols=512;}, [](auto &b){b.slice.expert_count=1;},
        [](auto &b){b.slice.expert_ids={0};}
    }};
    for(const auto &mutate:mutations)
    {
        matrix.binding=original; mutate(matrix.binding); matrix.publish();
        EXPECT_THROW(matrix.slice(),std::invalid_argument);
    }
    matrix.binding=original;
    auto original_engine=matrix.engine;
    auto foreign_source=test::TestTensorFactory::createQ4_0Random({256,256},29);
    matrix.engine=std::make_shared<CPUNativeVNNIGemmKernel>(foreign_source.get());
    matrix.publish();
    EXPECT_THROW(matrix.slice(),std::invalid_argument);
    matrix.engine=original_engine; matrix.publish();
    const auto fact=matrix.slice();
    EXPECT_THROW(matrix.engine->beginOutputPartitionEquivalenceScope(256,512),std::invalid_argument);
    {
        auto outer=matrix.engine->beginOutputPartitionEquivalenceScope(256,7);
        EXPECT_EQ(cpuNativeVNNISerialEquivalentPolicyN(256),7);
        {
            auto scope=matrix.engine->beginOriginalOutputSliceScope(fact);
            EXPECT_EQ(cpuNativeVNNISerialEquivalentPolicyN(256),512);
            EXPECT_THROW(fact.require(matrix.engine.get(),256,288),std::invalid_argument);
        }
        EXPECT_EQ(cpuNativeVNNISerialEquivalentPolicyN(256),7);
    }
    EXPECT_EQ(cpuNativeVNNISerialEquivalentPolicyN(256),256);
}

/** @test All registered native sources retain Q and exact original K/V bytes in ordinary/grouped bundles. */
TEST(OriginalOutputSlice, AllSourcesFusedQUnchangedOrdinaryAndGrouped)
{
    ASSERT_EQ(test::quantizedVerifierFormats().size(),native_vnni_formats::kAllSourceFormats.size());
    for (int threads : {1, 2, 3})
    for(const auto &format:test::quantizedVerifierFormats())
    {
        Workers workers(threads);
        SCOPED_TRACE(threads);
        SCOPED_TRACE(format.label);
        std::shared_ptr<TensorBase> qsource(format.create({128,512},31));
        std::shared_ptr<TensorBase> ksource(format.create({512,512},37));
        std::shared_ptr<TensorBase> vsource(format.create({512,512},41));
        Matrix q(qsource,interval(128,0,128,512));
        Matrix k(ksource,interval(512,0,512,512));
        Matrix v(vsource,interval(512,0,512,512));
        Matrix ks(ksource->create_view({256,512},0),interval(512,0,256,512));
        Matrix vs(vsource->create_view({256,512},256*512),interval(512,256,256,512));
        const auto kfact=ks.slice(), vfact=vs.slice();
        WorkspaceRequirements requirements=test::cpuProjectionTestRequirements(3,
            {q.engine.get(),k.engine.get(),v.engine.get()});
        for(const auto *matrix:{&ks,&vs})
        {
            auto scope=matrix->engine->beginOriginalOutputSliceScope(matrix->slice());
            requirements.merge(matrix->engine->getWorkspaceRequirements(3));
        }
        test::CPUProjectionTestWorkspace workspace(requirements);
        for(int rows:{1,3}) for(bool grouped:{false,true})
        {
            auto input=fp32(rows,512), fullq=fp32(rows,128), fullk=fp32(rows,512), fullv=fp32(rows,512);
            auto localq=fp32(rows,128), localk=fp32(rows,256), localv=fp32(rows,256);
            std::vector<ITensorGemm::TensorProjectionDesc> full{
                {q.engine.get(),fullq.get(),128},{k.engine.get(),fullk.get(),512},{v.engine.get(),fullv.get(),512}};
            std::vector<ITensorGemm::TensorProjectionDesc> local{
                {q.engine.get(),localq.get(),128},
                {ks.engine.get(),localk.get(),256,nullptr,"K",kfact},
                {vs.engine.get(),localv.get(),256,nullptr,"V",vfact}};
            if (grouped || rows == 1)
            {
                const std::array<FusedVerifierRowsDesc,3> descriptors{{
                    {.packed=q.engine->exportCPUNativeVNNIPackedWeights(), .N=128, .ldc=128},
                    {.packed=ks.engine->exportCPUNativeVNNIPackedWeights(), .N=256, .ldc=256, .policy_n=kfact.policyColumns()},
                    {.packed=vs.engine->exportCPUNativeVNNIPackedWeights(), .N=256, .ldc=256, .policy_n=vfact.policyColumns()}}};
                const auto plan=planNativeVNNIFusedRows(descriptors,rows,2);
                EXPECT_EQ(plan.projections[0].policy_n,128);
                EXPECT_EQ(plan.projections[1].policy_n,512);
                EXPECT_EQ(plan.projections[2].policy_n,512);
                EXPECT_EQ(plan.projections[1].n_chunks,4);
                EXPECT_EQ(plan.projections[1].k_tiles,
                    serialTileConfigForPackedMatrix(*k.engine->exportCPUNativeVNNIPackedWeights(),512,512,2).k_tiles);
            }
            const auto invoke=[&](const auto &projections){return grouped
                ? q.engine->multiply_fused_verifier_rows_decode_equivalent(input.get(),projections,rows,512,nullptr,workspace.get())
                : q.engine->multiply_fused_tensor(input.get(),projections,rows,512,nullptr,workspace.get());};
            ASSERT_TRUE(invoke(full)); ASSERT_TRUE(invoke(local));
            EXPECT_EQ(std::memcmp(fullq->data(),localq->data(),rows*128*sizeof(float)),0);
            for(int row=0;row<rows;++row)
            {
                EXPECT_EQ(std::memcmp(fullk->data()+row*512,localk->data()+row*256,256*sizeof(float)),0);
                EXPECT_EQ(std::memcmp(fullv->data()+row*512+256,localv->data()+row*256,256*sizeof(float)),0);
            }
            EXPECT_EQ(cpuNativeVNNISerialEquivalentPolicyN(128),128);
            EXPECT_EQ(cpuNativeVNNISerialEquivalentPolicyN(256),256);
        }
    }
}

/** @test The real stage authenticates refs, hashes source identity and keeps aliased unopted Q unchanged. */
TEST(OriginalOutputSlice, RealStageRejectsForeignBindingAndKeepsQ)
{
    Workers workers;
    Matrix matrix(test::TestTensorFactory::createQ8_0Random({256,512},51),interval(512,256,256,512));
    const auto fact=matrix.slice();
    auto input=fp32(3,512), q=fp32(3,256), k=fp32(3,256), v=fp32(3,256);
    FusedQKVGEMMStage::Params params{
        .device_id=DeviceId::cpu(), .input=input.get(), .m=3, .k=512,
        .wq=matrix.source.get(), .output_q=q.get(), .n_q=256,
        .wk=matrix.source.get(), .output_k=k.get(), .n_k=256,
        .wv=matrix.source.get(), .output_v=v.get(), .n_v=256,
        .prepared_ref_q=matrix.ref, .prepared_ref_k=matrix.ref, .prepared_ref_v=matrix.ref,
        .prepared_store=&matrix.store};
    CPUDeviceContext context(DeviceId::cpu(),2);
    FusedQKVGEMMStage baseline(params);
    { test::CPUStageTestWorkspace workspace(baseline); ASSERT_TRUE(baseline.execute(&context)); }
    std::vector<float> expected(q->data(),q->data()+q->numel());
    params.original_output_slice_k=fact; params.original_output_slice_v=fact;
    FusedQKVGEMMStage candidate(params);
    EXPECT_EQ(baseline.graphCaptureVariantSignature(),0u);
    EXPECT_NE(candidate.graphCaptureVariantSignature(),0u);
    { test::CPUStageTestWorkspace workspace(candidate); ASSERT_TRUE(candidate.execute(&context)); }
    EXPECT_EQ(std::memcmp(q->data(),expected.data(),expected.size()*sizeof(float)),0);
    EXPECT_EQ(cpuNativeVNNISerialEquivalentPolicyN(256),256);
    params.prepared_ref_k->binding_id++;
    FusedQKVGEMMStage foreign(params);
    EXPECT_THROW(foreign.getWorkspaceRequirements(3,256,512),std::invalid_argument);
    params.prepared_ref_k=matrix.ref; params.k=256;
    FusedQKVGEMMStage foreign_k(params);
    EXPECT_THROW(foreign_k.getWorkspaceRequirements(3,256,256),std::invalid_argument);
}

/** @test Missing/foreign owner snapshots cannot authenticate even matching shapes and codebooks. */
TEST(OriginalOutputSlice, RejectsMissingOrMismatchedPublicationOwners)
{
    Matrix matrix(test::TestTensorFactory::createQ8_0Random({256,512},61),interval(512,256,256,512));
    std::shared_ptr<TensorBase> other(test::TestTensorFactory::createQ8_0Random({256,512},63));
    auto other_engine=std::make_shared<CPUNativeVNNIGemmKernel>(other.get());
    const auto publish=[&](auto handle)
    {
        matrix.ref=matrix.store.registerPreparedGemmHandle(matrix.binding,
            PreparedWeightKind::CpuPackedGemm,DeviceId::cpu(),std::move(handle));
    };
    matrix.binding.tensor_owner.reset(); matrix.publish();
    EXPECT_THROW(matrix.slice(),std::invalid_argument);
    matrix.binding.tensor_owner=other; matrix.publish();
    EXPECT_THROW(matrix.slice(),std::invalid_argument);
    matrix.binding.tensor_owner=matrix.source;
    auto missing=matrix.makeHandle(); missing->prepared_weights->owned_kernel.reset(); publish(missing);
    EXPECT_THROW(matrix.slice(),std::invalid_argument);
    auto mismatched=matrix.makeHandle(); mismatched->prepared_weights->owned_kernel=other_engine; publish(mismatched);
    EXPECT_THROW(matrix.slice(),std::invalid_argument);
    matrix.publish();
    EXPECT_NO_THROW(matrix.slice());
    auto moved_from=matrix.slice();
    const auto retained=std::move(moved_from);
    EXPECT_THROW(moved_from.require(nullptr,256,512),std::invalid_argument);
    EXPECT_THROW(matrix.engine->beginOriginalOutputSliceScope(moved_from),std::invalid_argument);
    EXPECT_NO_THROW(retained.require(matrix.engine.get(),256,512));
    auto foreign_kind=matrix.ref; foreign_kind.kind=PreparedWeightKind::CudaInt8PackedGemm;
    EXPECT_THROW(matrix.store.originalOutputSlice(foreign_kind,*matrix.source),std::invalid_argument);
    // These are observable ownership/format guards. Factory-to-store publication
    // remains the trust boundary for payload affinity; no payload digest is used.
}

/** @test A cached KV-only stage retains exact owners; fresh historical stages cannot rebind. */
TEST(OriginalOutputSlice, ReplacementAndClearRetainOnlyResolvedPublication)
{
    Workers workers;
    Matrix matrix(test::TestTensorFactory::createQ8_0Random({256,512},67),interval(512,256,256,512));
    auto old_handle=matrix.makeHandle();
    matrix.ref=matrix.store.registerPreparedGemmHandle(matrix.binding,
        PreparedWeightKind::CpuPackedGemm,DeviceId::cpu(),old_handle);
    std::optional<OriginalOutputSlice> fact=matrix.slice();
    const auto old_identity=fact->recordingIdentity();
    auto *old_engine=matrix.engine.get();
    std::weak_ptr<ITensorGemm> weak_engine=matrix.engine;
    std::weak_ptr<TensorBase> weak_source=matrix.source;
    auto input=fp32(3,512), k=fp32(3,256), v=fp32(3,256);
    FusedQKVGEMMStage::Params params{
        .device_id=DeviceId::cpu(), .projection_set=AttentionProjectionSet::KeyValueOnly,
        .input=input.get(), .m=3, .k=512,
        .wk=matrix.source.get(), .output_k=k.get(), .n_k=256,
        .wv=matrix.source.get(), .output_v=v.get(), .n_v=256,
        .prepared_ref_k=matrix.ref, .prepared_ref_v=matrix.ref,
        .prepared_store=&matrix.store,
        .original_output_slice_k=*fact, .original_output_slice_v=*fact};
    CPUDeviceContext context(DeviceId::cpu(),2);
    auto stage=std::make_unique<FusedQKVGEMMStage>(params);
    { test::CPUStageTestWorkspace workspace(*stage); ASSERT_TRUE(stage->execute(&context)); }
    const auto stage_identity=stage->graphCaptureVariantSignature();
    const std::vector<float> expected(k->data(),k->data()+k->numel());

    // Publish a distinct source/engine under the very same reference. Owner
    // snapshots must not follow either mutable wrapper or the store index.
    matrix.source=test::TestTensorFactory::createQ8_0Random({256,512},71);
    matrix.engine=std::make_shared<CPUNativeVNNIGemmKernel>(matrix.source.get());
    matrix.binding.tensor=matrix.source.get(); matrix.binding.tensor_owner=matrix.source;
    matrix.publish();
    const auto replacement=matrix.slice();
    EXPECT_NE(replacement.recordingIdentity(),old_identity);
    EXPECT_NE(replacement.retainedEngine(),old_engine);
    old_handle->prepared_weights->owned_kernel=matrix.engine;
    old_handle->prepared_weights->kernel=matrix.engine.get();
    old_handle->tensor=matrix.source.get();
    EXPECT_EQ(fact->retainedEngine(),old_engine);
    EXPECT_EQ(fact->recordingIdentity(),old_identity);
    EXPECT_NO_THROW(fact->require(old_engine,256,512));
    {
        FusedQKVGEMMStage fresh(params);
        EXPECT_THROW(fresh.getWorkspaceRequirements(3,256,512),std::invalid_argument);
    }

    matrix.store.releaseAllPreparedState();
    matrix.source.reset(); matrix.engine.reset(); matrix.binding.tensor_owner.reset();
    matrix.binding.tensor=nullptr; old_handle.reset();
    EXPECT_FALSE(weak_engine.expired()); EXPECT_FALSE(weak_source.expired());
    EXPECT_EQ(stage->graphCaptureVariantSignature(),stage_identity);
    std::string validation_error;
    EXPECT_TRUE(stage->validatePreparedWeights(&validation_error)) << validation_error;
    {
        FusedQKVGEMMStage cold_after_clear(params);
        EXPECT_FALSE(cold_after_clear.validatePreparedWeights(&validation_error));
        EXPECT_THROW(cold_after_clear.getWorkspaceRequirements(3,256,512),std::invalid_argument);
    }
    { test::CPUStageTestWorkspace workspace(*stage); ASSERT_TRUE(stage->execute(&context)); }
    EXPECT_EQ(std::memcmp(k->data(),expected.data(),expected.size()*sizeof(float)),0);
    EXPECT_EQ(std::memcmp(v->data(),expected.data(),expected.size()*sizeof(float)),0);
    // Drop all external facts; the resolved stage alone still owns both exact
    // objects. This does not extend the lifetime of an unopted Q projection.
    fact.reset(); params.original_output_slice_k.reset(); params.original_output_slice_v.reset();
    EXPECT_FALSE(weak_engine.expired()); EXPECT_FALSE(weak_source.expired());
    stage.reset();
    EXPECT_TRUE(weak_engine.expired()); EXPECT_TRUE(weak_source.expired());
}

namespace
{
/** @brief Restore process collection policy after the cold-stage observation witness. */
struct GeometryCollection
{
    const std::array<const char *,2> keys{"LLAMINAR_PERF_STATS_SUMMARY","LLAMINAR_PERF_STATS_FILTER"};
    std::array<std::optional<std::string>,2> old;
    GeometryCollection()
    {
        for(size_t i=0;i<keys.size();++i)if(const char *v=std::getenv(keys[i]))old[i]=v;
        setenv(keys[0],"1",1);setenv(keys[1],"gpu_graph_inventory",1);
        mutableDebugEnv().reload();PerfStatsCollector::reloadConfigurationFromEnvironment();PerfStatsCollector::reset();
    }
    ~GeometryCollection()
    {
        for(size_t i=0;i<keys.size();++i)if(old[i])setenv(keys[i],old[i]->c_str(),1);else unsetenv(keys[i]);
        mutableDebugEnv().reload();PerfStatsCollector::reloadConfigurationFromEnvironment();PerfStatsCollector::reset();
    }
};
}
/** @test Actual cold stage publishes retained per-role facts once; it cannot advertise per-instance replay. */
TEST(OriginalOutputSlice, ColdStageGeometryEvidenceUsesActualOwnedFacts)
{
    Workers workers;GeometryCollection collection;
    Matrix matrix(test::TestTensorFactory::createQ8_0Random({256,512},94),interval(512,256,256,512));
    const auto fact=matrix.slice();auto input=fp32(3,512),q=fp32(3,256),k=fp32(3,256),v=fp32(3,256);
    FusedQKVGEMMStage stage({.device_id=DeviceId::cpu(),.input=input.get(),.m=3,.k=512,
        .wq=matrix.source.get(),.output_q=q.get(),.n_q=256,
        .wk=matrix.source.get(),.output_k=k.get(),.n_k=256,
        .wv=matrix.source.get(),.output_v=v.get(),.n_v=256,
        .prepared_ref_q=matrix.ref,.prepared_ref_k=matrix.ref,.prepared_ref_v=matrix.ref,.prepared_store=&matrix.store,
        .original_output_slice_k=fact,.original_output_slice_v=fact});
    (void)stage.getWorkspaceRequirements(3,256,512);
    (void)stage.getWorkspaceRequirements(3,256,512);
    const auto records=PerfStatsCollector::snapshot({"gpu_graph_inventory.query_kv_geometry"});
    ASSERT_EQ(records.size(),2u);
    std::set<std::string> roles;
    for(const auto &record:records)
    {
        roles.insert(record.tags.at("role"));
        EXPECT_EQ(record.tags.at("binding_id"),std::to_string(fact.bindingId()));
        EXPECT_EQ(record.tags.at("physical_n"),"256");EXPECT_EQ(record.tags.at("policy_n"),"512");
        EXPECT_EQ(record.tags.at("first_column"),"256");
        EXPECT_EQ(record.tags.at("source_name"),matrix.binding.identity.canonical_name);
        EXPECT_EQ(record.tags.at("stage_variant_identity"),std::to_string(stage.graphCaptureVariantSignature()));
        EXPECT_EQ(record.tags.at("per_instance_replay_proven"),"false");
    }
    EXPECT_EQ(roles,(std::set<std::string>{"K","V"}));
}
