/**
 * @file Test__Qwen4ExpHCQueryRowsDevice.cpp
 * @brief Actual native-prepared HC sites preserve original query rows in captured-parameter identity.
 *
 * Native setup/control publications only: these cases inspect actual HC GEMM
 * Params/signatures, not claimed projection capture/replay or arithmetic. All
 * original source and query owners retire through their exact setup stream.
 */
#include "../../../utils/Qwen4ExpHCQueryFixture.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/TensorFactory.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "../../../mocks/MockMPIContext.h"
#include <gtest/gtest.h>
#include <cstring>
using namespace llaminar2;
using namespace llaminar2::test;
namespace
{
/** @brief Reject incomplete real setup instead of fabricating a native count. */
void checked(bool result,const char *message){if(!result)throw std::runtime_error(message);}
/** @brief Original native ingress/count storage; copied setup contract, never a production row resolver. */
class NativeQuery final
{
public:
    /**
     * @brief Prepare complete original native chunk fields on one exact setup stream.
     * @param backend Actual backend supplying the original pointers and setup ordering.
     * @param device Same physical participant used by source preparation and the stage.
     * @param stream Non-default native producer stream.
     * @param rows Maximum physical token query count.
     */
    NativeQuery(IBackend &backend, DeviceId device, void *stream, int rows)
        : words_({static_cast<std::size_t>(4 * rows + 8)}), backend_(backend), device_(device), stream_(stream)
    {
        std::fill_n(static_cast<int32_t *>(words_.raw_mutable_data()), words_.numel(), 0);
        try
        {
            TransferEngine::prepareDeviceInput(&words_, device, stream);
            TransferEngine::requireDeviceInput(&words_, device, stream);
        }
        catch (...) { retire(); throw; }
    }
    /** @brief Join only diagnostic setup/publications before native owners unwind. */
    ~NativeQuery() { retire(); }
    /**
     * @brief Describe the actual original chunk owner without a host count mirror.
     * @param rows Exact immutable bucket width, bounded by this storage's admitted capacity.
     * @param alternate Select a distinct original count word for identity-negative control.
     * @return Complete forward input accepted by the production native-chunk authenticator.
     */
    ForwardInput input(int rows, bool alternate = false)
    {
        auto *base = static_cast<int32_t *>(words_.gpu_data_ptr());
        DevicePrefillChunkGraphBinding binding{
            .backend=&backend_, .request_token_ids_device=base,
            .request_position_ids_device=base+rows, .request_total_rows_device=base+4*rows,
            .cached_tokens_device=base+4*rows+1, .chunk_token_ids_device=base+2*rows,
            .chunk_position_ids_device=base+3*rows,
            .chunk_real_rows_device=base+4*rows+(alternate ? 5 : 2),
            .chunk_row_stride_device=base+4*rows+3,
            .request_row_capacity=rows, .bucket_seq_len=rows, .pad_token_id=0};
        // The test owns this declaration. Its recording identity uses every
        // embedded native field; mutable count contents are deliberately absent.
        uint64_t identity=0x51464e47444e0002ULL;
        const auto mix=[&](uint64_t value) { identity ^= value+0x9e3779b97f4a7c15ULL+(identity<<6)+(identity>>2); };
        mix(reinterpret_cast<uintptr_t>(binding.backend));
        for (const auto *pointer : {binding.request_token_ids_device, binding.request_position_ids_device,
             binding.request_total_rows_device, binding.cached_tokens_device,
             static_cast<const int32_t *>(binding.chunk_token_ids_device),
             static_cast<const int32_t *>(binding.chunk_position_ids_device),
             static_cast<const int32_t *>(binding.chunk_real_rows_device),
             static_cast<const int32_t *>(binding.chunk_row_stride_device)})
            mix(reinterpret_cast<uintptr_t>(pointer));
        mix(rows); mix(binding.pad_token_id); mix(static_cast<uint64_t>(device_.type)); mix(device_.ordinal);
        binding.capture_identity=identity ? identity : 1;
        ForwardInput result;
        result.token_ids_device=binding.chunk_token_ids_device;
        result.position_ids_device=binding.chunk_position_ids_device;
        result.execution_role=ForwardExecutionRole::MainInference;
        result.execution_phase=ForwardExecutionPhase::Prefill;
        result.seq_len=rows;
        result.bucket_seq_len=rows;
        result.batch_size=1;
        result.device=device_;
        result.sequence_lengths_device=binding.chunk_real_rows_device;
        result.device_prefill_chunk=binding;
        return result;
    }
    /**
     * @brief Publish a diagnostic count on its original exact setup stream.
     * @param input Descriptor whose original native scalar is changed.
     * @param live Count value, never substituted into graph construction.
     */
    void publish(const ForwardInput &input, int32_t live)
    {
        checked(backend_.hostToDeviceOnStream(const_cast<int32_t *>(input.sequence_lengths_device),
            &live, sizeof(live), device_.ordinal, stream_), "HC fixture count publication failed");
        // A setup-only metadata test owns this terminal join and host input
        // lifetime. Production graph construction never downloads/reads live.
        checked(backend_.synchronizeStream(stream_, device_.ordinal), "HC fixture publication retirement failed");
    }
private:
    /** @brief A failed setup retirement cannot permit native storage destruction. */
    void retire() noexcept
    {
        if (!backend_.synchronizeStream(stream_, device_.ordinal)) std::terminate();
    }
    INT32Tensor words_; ///< One real native allocation for original ingress and independent query words.
    IBackend &backend_; ///< Infrastructure authority, outlives the worker task.
    DeviceId device_; ///< Exact original producer/consumer participant.
    void *stream_; ///< Exact non-default setup stream.
};
/** @brief Real canonical request roots, admitted before native setup and retained through every graph inspection. */
class NativeRequestQuery final
{
public:
    /** @brief Allocate original request roles only, with exact host/device PMA claims and an original geometry record. */
    NativeRequestQuery(IBackend &backend,DeviceId device,void *stream,int rows)
        :backend_(backend),device_(device),stream_(stream),factory_(mpi_),arena_(ArenaConfig{.factory=&factory_})
    {
        const auto bytes=static_cast<std::size_t>(rows)*2*sizeof(std::int32_t)+2*sizeof(std::int32_t);
        const auto resource=[](DeviceId owner){return PhysicalMemoryResource{.world_rank=0,.device=owner,
            .total_bytes=1u<<24,.admission_available_bytes=1u<<24};};
        PhysicalMemoryPlanBuilder plan;
        plan.add(resource(DeviceId::cpu()),PhysicalMemoryOwner::ActivationArena,bytes);
        plan.add(resource(device),PhysicalMemoryOwner::ActivationArena,bytes);
        memory_=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        host_=memory_->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ActivationArena,bytes);
        gpu_=memory_->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,bytes);
        try
        {
            checked(arena_.registerBuffer(BufferId::REQUEST_TOKEN_IDS,1,rows,"INT32",device),"Register original request tokens");
            checked(arena_.registerBuffer(BufferId::REQUEST_POSITION_IDS,1,rows,"INT32",device),"Register original request positions");
            checked(arena_.registerBuffer(BufferId::REQUEST_BATCH_GEOMETRY,1,2,"INT32",device),"Register original request geometry");
            checked(arena_.allocate(),"Allocate original request roots");
            for(const auto role:{BufferId::REQUEST_TOKEN_IDS,BufferId::REQUEST_POSITION_IDS,BufferId::REQUEST_BATCH_GEOMETRY})
            {
                auto *tensor=arena_.getTensor(role);std::memset(tensor->raw_mutable_data(),0,tensor->size_bytes());
                TransferEngine::prepareDeviceInput(tensor,device,stream);TransferEngine::requireDeviceInput(tensor,device,stream);
            }
        }
        catch(...){retire();throw;}
    }
    /** @brief Fence only fixture setup before native roots and their original claims retire. */
    ~NativeRequestQuery(){retire();}
    /** @return Exact original request0 producer, with no caller-authored raw count permission. */
    ForwardInput input(int rows)
    {
        const auto binding=DeviceRequestQueryBinding::fromArena(arena_,device_);
        ForwardInput input;input.device=device_;input.batch_size=1;input.seq_len=rows;
        input.execution_role=ForwardExecutionRole::MainInference;input.execution_phase=ForwardExecutionPhase::Prefill;
        input.token_ids_device=binding.tokens();input.position_ids_device=binding.positions();
        input.sequence_lengths_device=binding.lengths();input.device_request_query=binding;
        requireDeviceRequestQueryOwnsInput(input);return input;
    }
    /** @brief Change only replay data behind the original count, with exact terminal setup lifetime. */
    void publish(const ForwardInput &input,std::int32_t live)
    {
        checked(backend_.hostToDeviceOnStream(const_cast<std::int32_t *>(input.sequence_lengths_device),&live,
            sizeof(live),device_.ordinal,stream_),"Publish original request query count");
        checked(backend_.synchronizeStream(stream_,device_.ordinal),"Retire original request count publication");
    }
private:
    /** @brief Failure cannot release original request storage before asynchronous setup completes. */
    void retire() noexcept {if(!backend_.synchronizeStream(stream_,device_.ordinal))std::terminate();}
    IBackend &backend_;
    DeviceId device_;
    void *stream_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    PhysicalMemoryAllocationLease host_,gpu_;
    test::MockMPIContext mpi_;
    TensorFactory factory_;
    BufferArena arena_;
};
/** @brief Inspect actual workspace and non-chunked ingress HC sites; no projection is executed. */
void proveWorkspaceAndRequestHC(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if(!backend || backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"GPU unavailable";
    [[maybe_unused]] auto context=IDeviceContext::create(device,1);
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&]{
        const auto stream=ExplicitGPUStream(worker.defaultStream()).get();
        HCQuerySource source(device,stream);
        for(const int rows:{1,512})
        {
            HCQueryActivations activation(rows);NativeRequestQuery request(*backend,device,stream,rows);
            NativeQuery foreign(*backend,device,stream,rows);
            const auto original=request.input(rows);
            for(const auto site:hcQuerySites)
            {
                auto builder=source.builder(source.config);
                ComputeGraph runtime;const auto prefix=builder->append(runtime,original,activation,site);
                EXPECT_FALSE(runtime.isWorkspaceDeclaration());
                const auto &up=hcQueryStage(runtime,prefix+"read_up");
                const auto &range=up.getParams().verifier_row_range;ASSERT_TRUE(range);
                EXPECT_EQ(range->countOwner(),original.sequence_lengths_device);EXPECT_EQ(range->physicalRows(),rows);
                const auto identity=up.graphCaptureVariantSignature();
                for(const int live:{rows,0,std::min(9,rows),0,rows})
                {
                    request.publish(original,live);EXPECT_EQ(range->activeRowsFor(live),live);
                    EXPECT_EQ(up.graphCaptureVariantSignature(),identity);
                }
                // Initial/max/bucket manifests have the same permanent banks
                // but no admitted query/chunk. Only the explicit purpose permits
                // their capacity declaration, and no executor may capture it.
                auto declaration=original;declaration.device_request_query.reset();declaration.sequence_lengths_device=nullptr;
                declaration.graph_submission_intent=ForwardGraphSubmissionIntent::WorkspaceDeclaration;
                ComputeGraph geometry;const auto declared=builder->append(geometry,declaration,activation,site);
                ASSERT_TRUE(geometry.isWorkspaceDeclaration());
                const auto &declaredRows=hcQueryStage(geometry,declared+"read_up").getParams().verifier_row_range;
                ASSERT_TRUE(declaredRows);EXPECT_EQ(declaredRows->physicalRows(),rows);EXPECT_EQ(declaredRows->countOwner(),nullptr);
                DeviceGraphExecutor executor;
                EXPECT_THROW(executor.captureRetainedGraphFragment(geometry,nullptr,nullptr),std::invalid_argument);
                EXPECT_THROW(executor.execute(geometry,nullptr),std::invalid_argument);
                for(int mutation=0;mutation<7;++mutation)
                {
                    auto invalid=original;
                    if(mutation==0)invalid.device_request_query.reset();
                    if(mutation==1)invalid.sequence_lengths_device=foreign.input(rows).sequence_lengths_device;
                    if(mutation==2)invalid.token_ids_device=foreign.input(rows).token_ids_device;
                    if(mutation==3)invalid.position_ids_device=foreign.input(rows).position_ids_device;
                    if(mutation==4)invalid.execution_role=ForwardExecutionRole::GroupedMTPVerifier;
                    if(mutation==5)invalid.seq_len=rows+1;
                    if(mutation==6)invalid.graph_submission_intent=static_cast<ForwardGraphSubmissionIntent>(255);
                    ComputeGraph rejected;EXPECT_THROW(builder->append(rejected,invalid,activation,site),std::exception);
                    EXPECT_EQ(rejected.size(),0u);
                }
            }
        }
    }).get();
}
/** @brief Inspect all actual main HC sites on one original native backend.
 * @param device Explicit CUDA/ROCm participant, prepared only under its real worker context.
 * @throws std::exception For incomplete source/count setup or invalid actual graph construction.
 */
void proveHC(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if(!backend || backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"GPU unavailable";
    [[maybe_unused]] auto context=IDeviceContext::create(device,1);
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        const auto stream=ExplicitGPUStream(worker.defaultStream()).get();
        HCQuerySource source(device,stream);
        for(const int rows:{1,15,512,1024,2048})
        {
            HCQueryActivations activation(rows);NativeQuery query(*backend,device,stream,rows);
            auto input=query.input(rows);
            for(const auto site:hcQuerySites)
            {
                for(const bool corrected:{false,true})
                {
                    auto config=source.config;
                    config.qfn_hc_up_activation=corrected?QuantizedActivationOperandPrecision::TwoTermQ8:
                        QuantizedActivationOperandPrecision::SingleQ8;
                    auto builder=source.builder(config);ComputeGraph graph;
                    const auto prefix=builder->append(graph,input,activation,site);
                    const auto &up=hcQueryStage(graph,prefix+"read_up");const auto &params=up.getParams();
                    EXPECT_EQ(params.B,source.upBinding(site).tensor);
                    ASSERT_TRUE(params.prepared_ref);
                    EXPECT_EQ(params.prepared_ref->binding_id,source.upBinding(site).prepared->binding_id);
                    EXPECT_EQ(params.quantized_operand_precision,config.qfn_hc_up_activation);
                    EXPECT_FALSE(params.force_decode_equivalent_verifier_prefill);
                    EXPECT_EQ(params.m,rows);EXPECT_EQ(params.n,10240);EXPECT_EQ(params.k,320);
                    const auto &down=hcQueryStage(graph,prefix+"read_down").getParams();
                    EXPECT_EQ(down.quantized_operand_precision,QuantizedActivationOperandPrecision::SingleQ8);
                    EXPECT_FALSE(down.force_decode_equivalent_verifier_prefill);EXPECT_FALSE(down.verifier_row_range);
                    if(site==HCQuerySite::Attention || site==HCQuerySite::FeedForward)
                    {
                        const auto &inject=hcQueryStage(graph,prefix+"write_projection").getParams();
                        EXPECT_EQ(inject.quantized_operand_precision,QuantizedActivationOperandPrecision::SingleQ8);
                        EXPECT_FALSE(inject.force_decode_equivalent_verifier_prefill);EXPECT_FALSE(inject.verifier_row_range);
                    }
                    if(!corrected){EXPECT_FALSE(params.verifier_row_range);continue;}
                    ASSERT_TRUE(params.verifier_row_range);
                    EXPECT_EQ(params.verifier_row_range->countOwner(),input.sequence_lengths_device);
                    EXPECT_EQ(params.verifier_row_range->physicalRows(),rows);
                    const auto signature=up.graphCaptureVariantSignature();ASSERT_NE(signature,0u);
                    for(const int live:{rows,0,std::min(9,rows),rows/2,0,rows})
                    {
                        query.publish(input,live);
                        EXPECT_EQ(params.verifier_row_range->activeRowsFor(live),live);
                        EXPECT_EQ(up.graphCaptureVariantSignature(),signature);
                    }
                    ComputeGraph changed_owner;auto alternate=query.input(rows,true);
                    const auto alternate_prefix=builder->append(changed_owner,alternate,activation,site);
                    EXPECT_NE(hcQueryStage(changed_owner,alternate_prefix+"read_up").graphCaptureVariantSignature(),signature);
                    if(rows>1)
                    {
                        query.publish(input,0);
                        auto narrower=input;narrower.seq_len=rows-1;narrower.bucket_seq_len=rows-1;
                        narrower.device_prefill_chunk->bucket_seq_len=rows-1;
                        narrower.device_prefill_chunk->capture_identity^=0x4843524f5753ULL;
                        if(!narrower.device_prefill_chunk->capture_identity)narrower.device_prefill_chunk->capture_identity=1;
                        ComputeGraph changed_capacity;const auto narrower_prefix=builder->append(changed_capacity,narrower,activation,site);
                        const auto &different=hcQueryStage(changed_capacity,narrower_prefix+"read_up");
                        EXPECT_EQ(different.getParams().verifier_row_range->countOwner(),input.sequence_lengths_device);
                        EXPECT_EQ(different.getParams().m,rows-1);EXPECT_NE(different.graphCaptureVariantSignature(),signature);
                    }
                    auto scalar=input;scalar.device_prefill_chunk.reset();scalar.seq_len=1;scalar.bucket_seq_len=1;
                    scalar.execution_phase=ForwardExecutionPhase::Decode;
                    ComputeGraph serial;const auto serial_prefix=builder->append(serial,scalar,activation,site);
                    ASSERT_TRUE(hcQueryStage(serial,serial_prefix+"read_up").getParams().verifier_row_range);
                    EXPECT_EQ(hcQueryStage(serial,serial_prefix+"read_up").getParams().verifier_row_range->countOwner(),nullptr);
                    EXPECT_NE(hcQueryStage(serial,serial_prefix+"read_up").graphCaptureVariantSignature(),signature);

                    auto grouped_config=config;grouped_config.grouped_mtp_verifier=true;
                    grouped_config.mtp.enabled=true;grouped_config.mtp.graph_capacity_draft_tokens=15;
                    grouped_config.compute_all_position_logits=true;
                    auto grouped_builder=source.builder(grouped_config);
                    auto grouped=input;grouped.device_prefill_chunk.reset();grouped.execution_phase=ForwardExecutionPhase::Decode;
                    grouped.execution_role=ForwardExecutionRole::GroupedMTPVerifier;
                    ComputeGraph verifier;const auto verifier_prefix=grouped_builder->append(verifier,grouped,activation,site);
                    const auto &verified=hcQueryStage(verifier,verifier_prefix+"read_up");
                    ASSERT_TRUE(verified.getParams().verifier_row_range);
                    EXPECT_EQ(verified.getParams().verifier_row_range->countOwner(),input.sequence_lengths_device);
                    const auto verifier_signature=verified.graphCaptureVariantSignature();
                    for(const int live:{rows,0,std::min(1,rows),0})
                    {
                        query.publish(grouped,live);
                        EXPECT_EQ(verified.getParams().verifier_row_range->activeRowsFor(live),live);
                        EXPECT_EQ(verified.graphCaptureVariantSignature(),verifier_signature);
                    }
                    auto unowned_config=grouped_config;unowned_config.grouped_mtp_verifier=false;
                    ASSERT_TRUE(unowned_config.usesMTPGroupedDecodeEquivalentRows());
                    auto unowned=source.builder(unowned_config);ComputeGraph invalid_group;
                    EXPECT_THROW(unowned->append(invalid_group,grouped,activation,site),std::exception);
                    EXPECT_EQ(invalid_group.size(),0u);
                    for(int mutation=0;mutation!=6;++mutation)
                    {
                        auto invalid=input;
                        if(mutation==0)invalid.device_prefill_chunk.reset();
                        if(mutation==1)invalid.sequence_lengths_device=alternate.sequence_lengths_device;
                        if(mutation==2)invalid.execution_role=ForwardExecutionRole::GroupedMTPVerifier;
                        if(mutation==3)invalid.execution_phase=ForwardExecutionPhase::Decode;
                        if(mutation==4)invalid.batch_size=2;
                        if(mutation==5)invalid.execution_role=static_cast<ForwardExecutionRole>(255);
                        ComputeGraph rejected;EXPECT_THROW(builder->append(rejected,invalid,activation,site),std::exception);
                        EXPECT_EQ(rejected.size(),0u);
                    }
                }
            }
        }
    }).get();
}
}
#ifdef HAVE_CUDA
TEST(Qwen4ExpHCQueryRowsDevice,CUDAActualSitesAndCaptureIdentity){proveHC(DeviceId::cuda(0));}
/** @test Actual CUDA-prepared HC rejects executable geometry declarations and retains original request0 counts. */
TEST(Qwen4ExpHCQueryRowsDevice,CUDAWorkspaceDeclarationAndOriginalRequest){proveWorkspaceAndRequestHC(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(Qwen4ExpHCQueryRowsDevice,ROCmActualSitesAndCaptureIdentity){proveHC(DeviceId::rocm(0));}
/** @test Actual ROCm-prepared HC rejects executable geometry declarations and retains original request0 counts. */
TEST(Qwen4ExpHCQueryRowsDevice,ROCmWorkspaceDeclarationAndOriginalRequest){proveWorkspaceAndRequestHC(DeviceId::rocm(0));}
#endif
