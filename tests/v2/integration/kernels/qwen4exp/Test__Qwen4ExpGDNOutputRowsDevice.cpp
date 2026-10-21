/**
 * @file Test__Qwen4ExpGDNOutputRowsDevice.cpp
 * @brief Actual prepared Stage6 parameters preserve native query-count capture identity.
 *
 * These integration cases prepare original Q8_0 source bytes on the real device
 * and call the same Stage6 composer as buildGDNAttentionGraph. They inspect the
 * actual GEMMStage and its capture signature while publishing changing original
 * query counts. They do not execute/capture the projection: arithmetic, inactive
 * payload access and retained native replay remain separate required gates.
 */
#include "../../../utils/Qwen4ExpGDNOutputStageFixture.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/device/DeviceContext.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <stdexcept>
using namespace llaminar2;
using namespace llaminar2::test;
namespace
{
/** @brief Throw on diagnostic setup failure rather than admitting an invented native owner. */
void checked(bool result, const char *message)
{
    if (!result) throw std::runtime_error(message);
}

/** @brief Real native ingress/control storage retained through all Stage6 metadata observations. */
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
            &live, sizeof(live), device_.ordinal, stream_), "Stage6 fixture count publication failed");
        // A setup-only metadata test owns this terminal join and host input
        // lifetime. Production graph construction never downloads/reads live.
        checked(backend_.synchronizeStream(stream_, device_.ordinal), "Stage6 fixture publication retirement failed");
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

/** @return The actual Stage6 constructed by the production composer, never copied parameter arithmetic. */
const GEMMStage &stageFor(const ComputeGraph &graph)
{
    const auto *node=graph.getNode("layer0_gdn_out_proj");
    if (!node) throw std::runtime_error("Stage6 composer omitted the output projection");
    const auto *stage=dynamic_cast<const GEMMStage *>(node->stage.get());
    if (!stage) throw std::runtime_error("Stage6 composer selected another stage type");
    return *stage;
}

/**
 * @brief Prove actual Stage6 source, original count and signature under ordinary/grouped/serial roles.
 * @param device Explicit native backend, prepared only inside its actual worker context.
 */
void proveRows(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if (!backend || backend->deviceCount()<=device.ordinal) GTEST_SKIP()<<"GPU unavailable";
    [[maybe_unused]] auto context=IDeviceContext::create(device,1);
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        const auto stream=ExplicitGPUStream(worker.defaultStream()).get();
        Qwen4ExpGDNOutputStageFixture source(device,stream);
        auto builder=source.builder(source.config);
        for (const int rows : {1,15,512,1024,2048})
        {
            Qwen4ExpGDNOutputActivationFixture activation(rows);
            NativeQuery query(*backend,device,stream,rows);
            auto input=query.input(rows);
            ASSERT_TRUE(devicePrefillChunkOwnsInput(input));
            ComputeGraph graph;
            activation.addProducer(graph,rows,device);
            builder->append(graph,input,activation.buffers,source.prepared.binding);
            const auto &stage=stageFor(graph);
            const auto &params=stage.getParams();
            ASSERT_TRUE(params.verifier_row_range);
            EXPECT_EQ(params.verifier_row_range->countOwner(), input.sequence_lengths_device);
            EXPECT_EQ(params.verifier_row_range->physicalRows(), rows);
            EXPECT_EQ(params.B, source.weight.get());
            EXPECT_EQ(params.prepared_store, source.prepared.store.get());
            ASSERT_TRUE(params.prepared_ref);
            EXPECT_EQ(params.prepared_ref->binding_id, source.prepared.ref.binding_id);
            EXPECT_EQ(params.quantized_operand_precision, QuantizedActivationOperandPrecision::TwoTermQ8);
            EXPECT_EQ(params.m,rows); EXPECT_EQ(params.n,2560); EXPECT_EQ(params.k,6144);
            const auto signature=stage.graphCaptureVariantSignature();
            ASSERT_NE(signature,0u);
            for (const int live : {0,std::min(9,rows),rows/2,rows,0,std::min(1,rows),rows})
            {
                query.publish(input,live);
                EXPECT_EQ(params.verifier_row_range->activeRowsFor(live),live);
                EXPECT_EQ(stage.graphCaptureVariantSignature(),signature);
            }
            EXPECT_EQ(params.verifier_row_range->activeRowsFor(rows+1),-1);
            EXPECT_EQ(params.verifier_row_range->activeRowsFor(-1),-1);

            ComputeGraph foreign_count;
            activation.addProducer(foreign_count,rows,device);
            auto alternate=query.input(rows,true);
            builder->append(foreign_count,alternate,activation.buffers,source.prepared.binding);
            EXPECT_NE(stageFor(foreign_count).graphCaptureVariantSignature(),signature);
            EXPECT_EQ(stageFor(foreign_count).getParams().verifier_row_range->countOwner(),alternate.sequence_lengths_device);

            if (rows > 1)
            {
                query.publish(input,0);
                auto narrower=input;
                narrower.seq_len=rows-1;
                narrower.bucket_seq_len=rows-1;
                narrower.device_prefill_chunk->bucket_seq_len=rows-1;
                // Same original pointers, new immutable declaration geometry.
                narrower.device_prefill_chunk->capture_identity ^= 0x724f5753ULL;
                if (!narrower.device_prefill_chunk->capture_identity)
                    narrower.device_prefill_chunk->capture_identity=1;
                ComputeGraph changed_capacity;
                activation.addProducer(changed_capacity,rows-1,device);
                builder->append(changed_capacity,narrower,activation.buffers,source.prepared.binding);
                const auto &changed=stageFor(changed_capacity);
                EXPECT_EQ(changed.getParams().verifier_row_range->countOwner(),input.sequence_lengths_device);
                EXPECT_EQ(changed.getParams().verifier_row_range->physicalRows(),rows-1);
                EXPECT_NE(changed.graphCaptureVariantSignature(),signature);
            }

            // Semantic serial Decode1 is fully active, even when its KV-length
            // pointer is nonnull. It must not reinterpret that word as query rows.
            auto scalar=input;
            scalar.device_prefill_chunk.reset();
            scalar.execution_phase=ForwardExecutionPhase::Decode;
            scalar.execution_role=ForwardExecutionRole::MainInference;
            scalar.seq_len=1; scalar.bucket_seq_len=1;
            ComputeGraph serial;
            activation.addProducer(serial,1,device);
            builder->append(serial,scalar,activation.buffers,source.prepared.binding);
            ASSERT_TRUE(stageFor(serial).getParams().verifier_row_range);
            EXPECT_EQ(stageFor(serial).getParams().verifier_row_range->countOwner(),nullptr);
            EXPECT_NE(stageFor(serial).graphCaptureVariantSignature(),signature);

            // A grouped verifier is a distinct semantic owner, including M1.
            // It keeps its original count while serial decode above ignores the
            // cumulative/native word rather than interpreting it as query rows.
            auto grouped_config=source.config;
            grouped_config.mtp.enabled=true;
            grouped_config.mtp.graph_capacity_draft_tokens=15;
            grouped_config.compute_all_position_logits=true;
            grouped_config.grouped_mtp_verifier=true;
            auto grouped_builder=source.builder(grouped_config);
            auto grouped=input;
            grouped.device_prefill_chunk.reset();
            grouped.execution_phase=ForwardExecutionPhase::Decode;
            grouped.execution_role=ForwardExecutionRole::GroupedMTPVerifier;
            ComputeGraph verifier;
            activation.addProducer(verifier,rows,device);
            grouped_builder->append(verifier,grouped,activation.buffers,source.prepared.binding);
            const auto &grouped_stage=stageFor(verifier);
            ASSERT_TRUE(grouped_stage.getParams().verifier_row_range);
            EXPECT_EQ(grouped_stage.getParams().verifier_row_range->countOwner(),input.sequence_lengths_device);
            const auto grouped_signature=grouped_stage.graphCaptureVariantSignature();
            for (const int live : {rows,0,std::min(1,rows),0})
            {
                query.publish(grouped,live);
                EXPECT_EQ(grouped_stage.getParams().verifier_row_range->activeRowsFor(live),live);
                EXPECT_EQ(grouped_stage.graphCaptureVariantSignature(),grouped_signature);
            }

            for (int mutation=0;mutation!=7;++mutation)
            {
                auto invalid=input;
                if (mutation==0) invalid.device_prefill_chunk.reset();
                if (mutation==1) invalid.sequence_lengths_device=alternate.sequence_lengths_device;
                if (mutation==2) invalid.execution_role=ForwardExecutionRole::GroupedMTPVerifier;
                if (mutation==3) invalid.execution_phase=ForwardExecutionPhase::Decode;
                if (mutation==4) invalid.batch_size=2;
                if (mutation==5) invalid.execution_role=static_cast<ForwardExecutionRole>(255);
                if (mutation==6) invalid.execution_phase=static_cast<ForwardExecutionPhase>(255);
                ComputeGraph rejected;
                activation.addProducer(rejected,rows,device);
                EXPECT_THROW(builder->append(rejected,invalid,activation.buffers,source.prepared.binding),std::exception);
                EXPECT_EQ(rejected.size(),1u);
                EXPECT_EQ(rejected.getNode("layer0_gdn_out_proj"),nullptr);
            }
            auto unowned_config=grouped_config;
            unowned_config.grouped_mtp_verifier=false;
            ASSERT_TRUE(unowned_config.usesMTPGroupedDecodeEquivalentRows());
            auto unowned_builder=source.builder(unowned_config);
            for (int mutation=0;mutation!=3;++mutation)
            {
                auto invalid=grouped;
                if (mutation==1) invalid.sequence_lengths_device=nullptr;
                ComputeGraph rejected;
                activation.addProducer(rejected,rows,device);
                auto &selected=mutation==0 ? *builder : mutation==1 ? *grouped_builder : *unowned_builder;
                EXPECT_THROW(selected.append(rejected,invalid,activation.buffers,source.prepared.binding),std::exception);
                EXPECT_EQ(rejected.size(),1u);
                EXPECT_EQ(rejected.getNode("layer0_gdn_out_proj"),nullptr);
            }
            EXPECT_TRUE(source.sourceUnchanged());
        }
    }).get();
}
}
#ifdef HAVE_CUDA
TEST(Qwen4ExpGDNOutputRowsDevice,CUDAParametersAndCaptureIdentity) { proveRows(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
TEST(Qwen4ExpGDNOutputRowsDevice,ROCmParametersAndCaptureIdentity) { proveRows(DeviceId::rocm(0)); }
#endif
