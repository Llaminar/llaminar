/**
 * @file Test__CPUExpertShardExecution.cpp
 * @brief Actual prepared CPU expert fragments and ordered pre-weighting partial reduction.
 *
 * Source-native cuts retain their original bytes and logical expert identities.
 * This model-free component replay does not admit a serving topology or claim
 * NUMA placement/throughput; original rank/socket integration remains separate.
 */
#include "kernels/cpu/moe/CPUCanonicalRouteReducer.h"
#include "execution/moe/MoEExpertShardCohort.h"
#include "execution/moe/MoEOverlayPreparedWeightSource.h"
#include "execution/moe/ExpertPreparedMemoryGeometry.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/KernelFactory.h"
#include "backends/BackendManager.h"
#include "tensors/TensorFactory.h"
#include "utils/CPUFeatures.h"
#include "utils/QuantizedVerifierFormats.h"
#include <gtest/gtest.h>
#include "kernels/cpu/moe/CPUCanonicalShardRouteRecords.h"
#include "execution/moe/CanonicalMoERouteRecord.h"
#include <array>
#include <cmath>
#include <cstring>
#include <vector>
#include <omp.h>
namespace llaminar2::test {
namespace {
using cpu::moe::CanonicalRouteDownPartialRows;
/** @brief Publish actual dispatch scope without claiming multi-rank placement or model qualification. */
void recordReplayScope() {
    ::testing::Test::RecordProperty("runtime_isa",isaLevelName(activeISALevel()));
    ::testing::Test::RecordProperty("numa_execution_qualified","false");
    ::testing::Test::RecordProperty("model_accuracy_qualified","false");
}
/** @brief Preserve the caller's actual OpenMP team configuration between replays. */
struct TeamScope {
    int previous=omp_get_max_threads();
    explicit TeamScope(int threads){omp_set_num_threads(threads);}
    ~TeamScope(){omp_set_num_threads(previous);}
};
}
/** @test Participant order precedes route weighting; zero/partial work never touches inactive tails. */
TEST(CPUExpertShardExecution, OrderedPartialRowsPrecedeTopKAndReset) {
    recordReplayScope();
    for(int threads:{1,2,3}) {
        TeamScope team(threads);
        constexpr int capacity=33,topK=4,columns=513;
        std::vector<float> left(capacity*topK*columns),right(left.size()),scratch(left.size(),-71),output(capacity*columns,-91);
        std::vector<float> weights(capacity*topK);
        for(size_t i=0;i<left.size();++i){left[i]=float(int(i%17)-8)*0.125f;right[i]=float(int(i%11)-5)*0.25f;}
        for(size_t i=0;i<weights.size();++i)weights[i]=i%3==0?0.0f:float(int(i%7)-3)*0.125f;
        std::array<CanonicalRouteDownPartialRows,2> parts{{{left},{right}}};
        for(int rows:{33,3,1,0,17,1}) {
            SCOPED_TRACE(rows);std::fill(scratch.begin(),scratch.end(),-71);std::fill(output.begin(),output.end(),-91);
            cpu::moe::CanonicalRouteFoldPlan used;
            ASSERT_TRUE(cpu::moe::reduceCanonicalRouteDownPartialsOrderedFMA({parts,scratch,weights.data(),output.data(),rows,topK,columns},&used));
            if(rows==capacity && threads>1)EXPECT_EQ(used.executor,cpu::moe::CanonicalRouteFoldExecutor::FullOpenMPTeam);
            if(rows==capacity && threads==3) {
                std::array<bool,3> accepted{};
#pragma omp parallel num_threads(3)
                {
                    accepted[size_t(omp_get_thread_num())]=cpu::moe::reduceCanonicalRouteDownPartialsOrderedFMA(
                        {parts,scratch,weights.data(),output.data(),rows,topK,columns});
                }
                for(bool value:accepted)ASSERT_TRUE(value);
            }
            for(int row=0;row<rows;++row)for(int column=0;column<columns;++column) {
                float expected=0;
                for(int route=0;route<topK;++route) {
                    const auto slot=row*topK+route;const auto at=slot*columns+column;
                    if(weights[slot]!=0)expected=std::fma(weights[slot],left[at]+right[at],expected);
                }
                EXPECT_EQ(std::memcmp(&expected,&output[row*columns+column],sizeof(float)),0);
            }
            for(size_t i=size_t(rows)*columns;i<output.size();++i)EXPECT_EQ(output[i],-91);
            for(size_t i=size_t(rows)*topK*columns;i<scratch.size();++i)EXPECT_EQ(scratch[i],-71);
        }
        const auto before=scratch,prior=output;
        auto invalid=parts;invalid[1].values=invalid[1].values.first(1);
        EXPECT_FALSE(cpu::moe::reduceCanonicalRouteDownPartialsOrderedFMA({invalid,scratch,weights.data(),output.data(),8,topK,columns}));
        EXPECT_EQ(scratch,before);EXPECT_EQ(output,prior);
        invalid=parts;invalid[1]=invalid[0];
        EXPECT_FALSE(cpu::moe::reduceCanonicalRouteDownPartialsOrderedFMA({invalid,scratch,weights.data(),output.data(),8,topK,columns}));
        EXPECT_EQ(scratch,before);EXPECT_EQ(output,prior);
    }
}
/** @test A local top-k fold followed by participant addition is a different, forbidden arithmetic tree. */
TEST(CPUExpertShardExecution, RejectsFoldBeforePartialReductionOracle) {
    recordReplayScope();
    const std::array<float,2> a{1e20f,1.0f},b{-1e20f,1.0f},weights{1,1};
    std::array<float,2> scratch{};float output=0;
    const std::array<CanonicalRouteDownPartialRows,2> parts{{{a},{b}}};
    ASSERT_TRUE(cpu::moe::reduceCanonicalRouteDownPartialsOrderedFMA({parts,scratch,weights.data(),&output,1,2,1}));
    EXPECT_EQ(output,2.0f);
    float localA=0,localB=0;
    ASSERT_TRUE(cpu::moe::reduceCanonicalRouteRowsOrderedFMA({a.data(),weights.data(),nullptr,&localA,1,2,1}));
    ASSERT_TRUE(cpu::moe::reduceCanonicalRouteRowsOrderedFMA({b.data(),weights.data(),nullptr,&localB,1,2,1}));
    const float wrong=localA+localB;EXPECT_NE(output,wrong);
}
} // namespace llaminar2::test
namespace llaminar2::test {
namespace {
using llaminar::v2::kernels::KernelFactory;
constexpr int H=256,F=512,E=512,K=2;
/** @brief Test-owned original fragment engine lifetimes; no whole-expert residency capability. */
struct PreparedFragmentEngines {
    std::shared_ptr<ITensorGemm> gate,up,down;
};
using Engines=std::array<PreparedFragmentEngines,2>;
/** @brief Copy complete original native source blocks into the exact compact N/K rectangle. */
std::shared_ptr<TensorBase> nativeFragment(const TensorBase& source,
    const MoEExpertProjectionOwnership::Projection& projection) {
    if(source.rows()!=size_t(projection.source_rows) || source.cols()!=size_t(projection.source_columns))
        throw std::invalid_argument("Recorded source does not match the original expert projection");
    const auto sourceRow=source.size_bytes()/source.rows();
    const auto first=sourceRow*size_t(projection.first_column)/source.cols();
    const auto bytes=sourceRow*size_t(projection.columns)/source.cols();
    if(bytes*source.cols()!=sourceRow*size_t(projection.columns) || first*source.cols()!=sourceRow*size_t(projection.first_column))
        throw std::invalid_argument("Native expert slice cannot split a source block");
    AlignedVector<uint8_t> compact(size_t(projection.rows)*bytes);
    const auto* input=static_cast<const uint8_t*>(source.raw_data());
    for(int row=0;row<projection.rows;++row)
        std::memcpy(compact.data()+size_t(row)*bytes,input+(size_t(projection.first_row)+row)*sourceRow+first,bytes);
    MPIContext mpi(0,1);
    TensorFactory factory(mpi);
    const std::vector<size_t> shape{size_t(projection.rows),size_t(projection.columns)};
    if(source.native_type()==TensorType::FP32) {
        AlignedVector<float> values(compact.size()/sizeof(float));std::memcpy(values.data(),compact.data(),compact.size());
        return factory.createFP32Owned(shape,std::move(values));
    }
    if(source.native_type()==TensorType::FP16 || source.native_type()==TensorType::BF16) {
        AlignedVector<uint16_t> values(compact.size()/sizeof(uint16_t));std::memcpy(values.data(),compact.data(),compact.size());
        if(source.native_type()==TensorType::FP16)return factory.createFP16Owned(shape,std::move(values));
        return factory.createBF16Owned(shape,std::move(values));
    }
    if(source.native_type()==TensorType::Q8_1)
        return std::make_shared<Q8_1Tensor>(shape,std::move(compact),DeviceId::cpu());
    return factory.createQuantizedOwned(source.native_type(),shape,std::move(compact));
}
/** @brief Run the unchanged real prepared CPU FFN and publish unweighted original-route rows. */
std::vector<float> preparedRouteRows(const Engines& engines,int intermediate,
    std::span<const float> inputRows,int rows,int hidden=H,
    MoECanonicalRoutePublicationLayout layout=MoECanonicalRoutePublicationLayout::DenseOriginalRouteSlots) {
    const int H=hidden;
    if(inputRows.size()!=size_t(rows)*H)throw std::invalid_argument("Incomplete original hidden rows");
    FP32Tensor input(std::vector<size_t>{size_t(rows),H});
    FP32Tensor indices(std::vector<size_t>{size_t(rows),K});
    FP32Tensor weights(std::vector<size_t>{size_t(rows),K});
    FP32Tensor output(std::vector<size_t>{size_t(rows),H});
    FP32Tensor routes(layout==MoECanonicalRoutePublicationLayout::PackedIndexedRouteRows
        ?std::vector<size_t>{size_t(rows)*K*(H+1)+1}:std::vector<size_t>{size_t(rows)*K,H});
    std::copy(inputRows.begin(),inputRows.end(),input.mutable_data());
    for(int row=0;row<rows;++row)for(int route=0;route<K;++route) {
        indices.mutable_data()[row*K+route]=route?511:64;
        weights.mutable_data()[row*K+route]=route?0.75f:0.25f;
    }
    MoEExpertComputeStage::Params p;
    p.device_id=DeviceId::cpu();p.seq_len=rows;p.d_model=H;p.expert_intermediate=intermediate;
    p.num_experts=E;p.top_k=K;p.layer_idx=3;p.input=&input;p.routing_indices=&indices;p.routing_weights=&weights;
    p.output=&output;p.output_registered_in_arena=false;p.canonical_route_contributions=&routes;
    p.canonical_route_arithmetic=MoECanonicalRouteArithmeticPolicy::UnweightedExpertRowThenOrderedFMA;
    p.canonical_route_layout=layout;
    p.service_phase=rows==1?MoEOverlayServicePhaseHint::Decode:
        rows==3?MoEOverlayServicePhaseHint::GroupedVerifier:MoEOverlayServicePhaseHint::Prefill;
    p.force_decode_equivalent_verifier_prefill=rows==3;
    p.expert_mask.assign(E,false);p.prepared_gate_gemm.assign(E,nullptr);p.prepared_up_gemm.assign(E,nullptr);p.prepared_down_gemm.assign(E,nullptr);
    for(size_t i=0;i<2;++i) {
        const auto expert=i?511:64;p.expert_mask[expert]=true;
        p.prepared_gate_gemm[expert]=engines[i].gate.get();p.prepared_up_gemm[expert]=engines[i].up.get();p.prepared_down_gemm[expert]=engines[i].down.get();
    }
    p.expert_weight_resolution_policy=MoEExpertWeightResolutionPolicy::PreparedRegistryOnly;
    p.cpu_router_q8_input_publication=CPURouterQ8InputPublicationPolicy::PublishTransportedRows;
    MoEExpertComputeStage declaration(p);
    auto requirements=declaration.getWorkspaceRequirements(rows);
    // This invocation supplies its existing serial owner, so the workspace
    // manager must not materialize a second graph-local grouped backing.
    requirements.cpu_grouped_moe_storage.reset();
    auto serialConfig=CPUGroupedMoESerialWorkspace::Config{
        .row_capacity=size_t(rows),.d_model=H,.expert_intermediate=intermediate,.num_experts=E,.routing_top_k=K,
        .debug_name="cpu_expert_shard_replay",.device_id=DeviceId::cpu()};
    auto serialLayout=std::make_shared<const CPUGroupedMoESerialWorkspace::StorageLayout>(
        CPUGroupedMoESerialWorkspace::storageLayout(serialConfig));
    const auto bytes=requirements.total_bytes_with_alignment();
    const auto total=bytes+serialLayout->physicalBytes();
    PhysicalMemoryPlanBuilder plan;
    plan.add({0,DeviceId::cpu(),total,total},PhysicalMemoryOwner::ExecutionWorkspace,bytes);
    plan.add({0,DeviceId::cpu(),total,total},PhysicalMemoryOwner::ExecutionWorkspace,serialLayout->physicalBytes());
    auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
    std::vector<float> observed;
    {
    serialConfig.physical_memory_authority=memory;serialConfig.declared_storage=serialLayout;
    p.cpu_grouped_serial_workspace=std::make_shared<CPUGroupedMoESerialWorkspace>(serialConfig);
    MoEExpertComputeStage stage(std::move(p));
    DeviceWorkspaceManager workspace(DeviceId::cpu(),bytes,memory);
    if(!workspace.allocate(requirements))throw std::runtime_error("Original CPU shard scratch allocation failed");
    stage.bindWorkspace(&workspace);
    struct Unbind {MoEExpertComputeStage& stage;~Unbind(){stage.unbindWorkspace();}} unbind{stage};
    auto context=IDeviceContext::create(DeviceId::cpu());
    if(!stage.execute(context.get()))
        throw std::runtime_error("Original prepared CPU expert shard execution failed");
    observed.assign(routes.data(),routes.data()+routes.numel());
    }
    if(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace,
        PhysicalMemoryMaterializationKind::NewAllocation)!=0)
        throw std::logic_error("Original CPU expert scratch did not retire");
    return observed;
}
/** @brief Exercise complete native sources without inventing CPU device ordinal1. */
void preparedNativeMatrix(const std::array<QuantizedVerifierWeightCreator,3>& create,
    const std::array<ExpertWeightFormat,3>& format,int hidden_columns=H,int intermediate=F,
    size_t recordedFragmentBytes=0,bool packedCollectiveReplay=false) {
    const int H=hidden_columns,F=intermediate;
    MoEOverlayLayerWeightManifest manifest{3,{{{ExpertTierWeightProjection::Gate,F,H,format[0]},
        {ExpertTierWeightProjection::Up,F,H,format[1]},{ExpertTierWeightProjection::Down,H,F,format[2]}}}};
    const auto cohort=MoEExpertShardCohort::checked(manifest,E,
        {{4,0,GlobalDeviceAddress::cpu(0)},{9,1,GlobalDeviceAddress::cpu(1)}},
        MoEExpertShardCohort::balancedIntervals(manifest,2));
    ASSERT_EQ(cohort.ownership(0).geometry().experts,E);ASSERT_EQ(cohort.ownership(1).geometry().experts,E);
    std::vector<int> selected(448);for(size_t i=0;i<selected.size();++i)selected[i]=int(i)+64;
    for(size_t participant=0;participant<2;++participant) {
        const auto slice=cohort.sourceSlice(participant,WeightRole::MoEExpertDown,selected);
        ASSERT_EQ(slice.expert_count,448u);ASSERT_EQ(slice.expert_ids,selected);
        ASSERT_EQ(cohort.ownership(participant).geometry().experts,512);
    }
    ASSERT_EQ(cohort.participants()[0].address.numa_node,0);ASSERT_EQ(cohort.participants()[1].address.numa_node,1);
    ASSERT_EQ(cohort.participants()[0].address.device_ordinal,0);ASSERT_EQ(cohort.participants()[1].address.device_ordinal,0);
    std::array<Engines,2> shards;
    std::array<std::array<size_t,2>,2> observedBytes{};
    std::array<std::array<std::array<std::shared_ptr<ITensorGemm>,3>,2>,2> handles;
    for(size_t expert=0;expert<2;++expert)for(size_t projection=0;projection<3;++projection) {
        auto source=create[projection](projection==2?std::vector<size_t>{H,F}:std::vector<size_t>{F,H},uint32_t(71+expert*11+projection));
        std::array<std::shared_ptr<ITensorGemm>,2> prepared;
        const auto role=projection==0?WeightRole::MoEExpertGate:projection==1?WeightRole::MoEExpertUp:WeightRole::MoEExpertDown;
        const std::vector<uint8_t> original(static_cast<const uint8_t*>(source->raw_data()),static_cast<const uint8_t*>(source->raw_data())+source->size_bytes());
        for(size_t participant=0;participant<2;++participant) {
            const auto slice=cohort.sourceSlice(participant,role,{expert?511:64});
            const auto rectangle=cohort.ownership(participant).projection(role);
            EXPECT_EQ(slice.row_start,size_t(rectangle.first_column));EXPECT_EQ(slice.col_start,size_t(rectangle.first_row));
            const auto load=cohort.sourceLoadSlice(participant,role,{expert?511:64});
            EXPECT_TRUE(slice.inner_is_presliced);EXPECT_FALSE(load.inner_is_presliced);
            EXPECT_EQ(load.row_start,slice.row_start);EXPECT_EQ(load.col_start,slice.col_start);
            EXPECT_EQ(load.row_count,slice.row_count);EXPECT_EQ(load.col_count,slice.col_count);
            auto compact=nativeFragment(*source,rectangle);
            EXPECT_EQ(compact->native_type(),source->native_type());
            prepared[participant]=KernelFactory::prepareExpertGemmLocal(compact,DeviceId::cpu());
            ASSERT_NE(prepared[participant],nullptr);
            MoEOverlayPreparedWeightSource physical;
            std::string diagnostic;
            ASSERT_TRUE(resolveMoEOverlayPreparedWeightSource(prepared[participant],DeviceId::cpu(),physical,&diagnostic))<<diagnostic;
            EXPECT_EQ(physical.format,format[projection]);
            if(physical.cpu_packed) {
                const auto& packed=*physical.cpu_packed;
                const auto actual=packed.native_interleaved.allocationBytes()+packed.payload.allocationBytes()+packed.int8_flat.allocationBytes();
                const auto expected=resolveExpertPreparedProjectionMemoryGeometry(rectangle.rows,rectangle.columns,format[projection]).cpu_bytes;
                EXPECT_EQ(actual,expected);
                observedBytes[participant][expert]+=actual;
            }
            EXPECT_EQ(std::memcmp(source->raw_data(),original.data(),original.size()),0);
        }
        for(size_t participant=0;participant<2;++participant)
            handles[participant][expert][projection]=std::move(prepared[participant]);
    }
    for(size_t participant=0;participant<2;++participant)for(size_t expert=0;expert<2;++expert)
        shards[participant][expert]={handles[participant][expert][0],handles[participant][expert][1],handles[participant][expert][2]};
    if(recordedFragmentBytes)
        for(const auto& participant:observedBytes)for(size_t bytes:participant)EXPECT_EQ(bytes,recordedFragmentBytes);
    std::vector<float> hidden(8*H);
    for(size_t i=0;i<hidden.size();++i)hidden[i]=std::sin(float(i)*0.013f)*0.1f;
    for(int rows:{8,3,1,3}) {
        SCOPED_TRACE(rows);
        std::array<std::vector<float>,2> actual;
        for(size_t participant=0;participant<2;++participant) {
            actual[participant]=preparedRouteRows(shards[participant],F/2,std::span(hidden).first(size_t(rows)*H),rows,H);
            for(int row=0;row<rows;++row) {
                const auto serial=preparedRouteRows(shards[participant],F/2,std::span(hidden).subspan(size_t(row)*H,H),1,H);
                ASSERT_EQ(serial.size(),size_t(K*H));
                EXPECT_EQ(std::memcmp(actual[participant].data()+size_t(row)*K*H,serial.data(),serial.size()*sizeof(float)),0)
                    << "participant=" << participant << " row=" << row;
            }
        }
        std::array<CanonicalRouteDownPartialRows,2> parts{{{actual[0]},{actual[1]}}};
        std::vector<float> scratch(size_t(rows)*K*H,-77),output(size_t(rows)*H,-88),weights(size_t(rows)*K);
        for(size_t slot=0;slot<weights.size();++slot)weights[slot]=slot%2?0.75f:0.25f;
        ASSERT_TRUE(cpu::moe::reduceCanonicalRouteDownPartialsOrderedFMA({parts,scratch,weights.data(),output.data(),rows,K,H}));
        if(packedCollectiveReplay) {
            std::array<std::vector<float>,2> packed;
            std::vector<float> receive;
            std::array<RootedFloatRecordRange,2> ranges;
            // Deliberately append participant1 before participant0. The receive
            // permutation is irrelevant to the source-cohort arithmetic order.
            for(int participant:{1,0}) {
                packed[participant]=preparedRouteRows(shards[participant],F/2,
                    std::span(hidden).first(size_t(rows)*H),rows,H,
                    MoECanonicalRoutePublicationLayout::PackedIndexedRouteRows);
                const auto count=canonical_moe_route_record::readRecordCount(packed[participant].data(),packed[participant].size());
                ASSERT_EQ(count,size_t(rows)*K);
                const auto first=receive.size()/size_t(H+1);
                receive.insert(receive.end(),packed[participant].begin(),packed[participant].begin()+count*size_t(H+1));
                ranges[participant]={participant,first,count};
            }
            std::vector<float> indices(size_t(rows)*K),required(weights.size()),combined(scratch.size(),-991);
            std::vector<int32_t> maps(size_t(rows)*K*2);
            for(size_t slot=0;slot<indices.size();++slot)indices[slot]=slot%2?511:64;
            const std::array<int,2> selected{64,511};
            ASSERT_TRUE(cpu::moe::combineCanonicalShardRouteRecords({receive,ranges,indices,weights,selected,
                maps,required,combined,E,rows,K,H}));
            EXPECT_EQ(std::memcmp(combined.data(),scratch.data(),scratch.size()*sizeof(float)),0);
        }
        for(int row=0;row<rows;++row)for(int column=0;column<H;++column) {
            float expected=0;
            for(int route=0;route<K;++route) {
                const auto at=(size_t(row)*K+route)*H+column;
                const float combined=actual[0][at]+actual[1][at];
                EXPECT_EQ(std::memcmp(&scratch[at],&combined,sizeof(float)),0);
                expected=std::fma(weights[size_t(row)*K+route],combined,expected);
            }
            EXPECT_EQ(std::memcmp(&output[size_t(row)*H+column],&expected,sizeof(float)),0);
        }
    }
}
}
} // namespace llaminar2::test

namespace llaminar2::test {
/** @test Original native codebook bytes execute through real CPU FFNs, with original512 axis and selected448 identities intact. */
TEST(CPUExpertShardExecution, PreparedNativeSourceAllFormatsAndSerialRows) {
    recordReplayScope();
    if(!hasCPUBackend())initCPUBackend(-1);
    TeamScope team(1);
    for(const auto& format:quantizedMoEVerifierFormats()) {
        SCOPED_TRACE(format.label);
        const auto identity=ExpertWeightFormat::nativeQuantised({format.source_codebook_id,format.source_is_superblock,true});
        preparedNativeMatrix({format.create,format.create,format.create},{identity,identity,identity});
    }
    for(const auto type:{TensorType::FP32,TensorType::FP16,TensorType::BF16}) {
        SCOPED_TRACE(static_cast<int>(type));
        const QuantizedVerifierWeightCreator create=[type](const auto& shape,uint32_t seed)->std::unique_ptr<TensorBase> {
            if(type==TensorType::FP32)return TestTensorFactory::createFP32Random(shape,-0.02f,0.02f,seed);
            if(type==TensorType::FP16)return TestTensorFactory::createFP16Random(shape,-0.02f,0.02f,seed);
            return TestTensorFactory::createBF16Random(shape,-0.02f,0.02f,seed);
        };
        const auto identity=ExpertWeightFormat::floating(type);
        preparedNativeMatrix({create,create,create},{identity,identity,identity});
    }
}
} // namespace llaminar2::test

namespace llaminar2::test {
namespace {
/** @brief Execute one observed original source class, keeping its own exact allocation expectation. */
void qfnOriginalClass(TensorType gateType,TensorType downType,size_t fragmentBytes,bool packed=false) {
    if(!hasCPUBackend())initCPUBackend(-1);
    TeamScope team(1);
    const auto& formats=quantizedMoEVerifierFormats();
    const auto find=[&](TensorType type)->const QuantizedVerifierFormatCase& {
        const auto item=std::find_if(formats.begin(),formats.end(),[&](const auto& value){return value.tensor_type==type;});
        if(item==formats.end())throw std::logic_error("Original QFN source format missing from canonical registry");
        return *item;
    };
    const auto& gate=find(gateType);const auto& down=find(downType);
    const auto gateFormat=ExpertWeightFormat::nativeQuantised({gate.source_codebook_id,gate.source_is_superblock,true});
    const auto downFormat=ExpertWeightFormat::nativeQuantised({down.source_codebook_id,down.source_is_superblock,true});
    preparedNativeMatrix({gate.create,gate.create,down.create},{gateFormat,gateFormat,downFormat},2560,640,fragmentBytes,packed);
}
}
/** @test The43-layer original Q4_K/Q5_1 class executes native320+320 fragments. */
TEST(CPUExpertShardExecution, QFNOriginalMixedFormatsAndNative320Cuts) {
    recordReplayScope();qfnOriginalClass(TensorType::Q4_K,TensorType::Q5_1,2105344);
}
/** @test The4-layer original Q4_K/Q8_0 class retains its distinct physical representation. */
TEST(CPUExpertShardExecution, QFNOriginalQ4KQ8_0Native320Cuts) {
    recordReplayScope();qfnOriginalClass(TensorType::Q4_K,TensorType::Q8_0,2052096);
}
/** @test The1-layer original Q5_K/Q8_0 class cannot borrow the common class byte quote. */
TEST(CPUExpertShardExecution, QFNOriginalQ5KQ8_0Native320Cuts) {
    recordReplayScope();qfnOriginalClass(TensorType::Q5_K,TensorType::Q8_0,2871296);
}
} // namespace llaminar2::test

namespace llaminar2::test {
/** @test Actual prepared native fragments publish compact records, then fixed cohort order survives arrival permutation. */
TEST(CPUExpertShardExecution, PreparedPackedSourceRecordsAllFormatsAndPhases) {
    recordReplayScope();
    if(!hasCPUBackend())initCPUBackend(-1);
    TeamScope team(1);
    for(const auto& format:quantizedMoEVerifierFormats()) {
        SCOPED_TRACE(format.label);
        const auto identity=ExpertWeightFormat::nativeQuantised({format.source_codebook_id,format.source_is_superblock,true});
        preparedNativeMatrix({format.create,format.create,format.create},{identity,identity,identity},H,F,0,true);
    }
    for(const auto type:{TensorType::FP32,TensorType::FP16,TensorType::BF16}) {
        const QuantizedVerifierWeightCreator create=[type](const auto& shape,uint32_t seed)->std::unique_ptr<TensorBase> {
            if(type==TensorType::FP32)return TestTensorFactory::createFP32Random(shape,-0.02f,0.02f,seed);
            if(type==TensorType::FP16)return TestTensorFactory::createFP16Random(shape,-0.02f,0.02f,seed);
            return TestTensorFactory::createBF16Random(shape,-0.02f,0.02f,seed);
        };
        const auto identity=ExpertWeightFormat::floating(type);
        preparedNativeMatrix({create,create,create},{identity,identity,identity},H,F,0,true);
    }
}
TEST(CPUExpertShardExecution, PreparedPackedQFNMixedClasses) {
    recordReplayScope();
    qfnOriginalClass(TensorType::Q4_K,TensorType::Q5_1,2105344,true);
    qfnOriginalClass(TensorType::Q4_K,TensorType::Q8_0,2052096,true);
    qfnOriginalClass(TensorType::Q5_K,TensorType::Q8_0,2871296,true);
}
} // namespace llaminar2::test

#include "CPUFullExpertReferenceContract.inc"
