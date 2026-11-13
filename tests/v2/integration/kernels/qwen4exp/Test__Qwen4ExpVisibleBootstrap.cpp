/**
 * @file Test__Qwen4ExpVisibleBootstrap.cpp
 * @brief Native captured QFN shifted-prefill packing across bootstrap, continuation and reset.
 *
 * The real compute stage records one fused transaction on each backend. New
 * request controls are supplied only at test setup boundaries. Replays must
 * retain addresses and graph topology while preserving the prior terminal row,
 * inserting the visible zero-hidden first-token row, and clearing inactive
 * diagnostic capacity. Host downloads are terminal test observations only.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/compute_stages/stages/HiddenStateRowsSelectStage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <vector>

namespace llaminar2::test {
namespace {
/** @brief Independent expected first-row ownership for a known canonical lifecycle transition. */
enum class FirstRow { ZeroSeed, PriorTerminal, NoRows };

/** @brief One explicit lifecycle observation; expected rows do not call the production planner. */
struct Segment {
    int base;
    int rows;
    int shifted;
    int expected_append;
    FirstRow first;
};

/** @brief Retire test-only asynchronous observations before their host buffers leave scope on any assertion. */
struct ObservationRetirement {
    IBackend* backend;
    DeviceId device;
    void* stream;
    ~ObservationRetirement(){(void)backend->synchronizeStream(stream,device.ordinal);}
};

/** @brief Exercise the real retained transaction on the requested participant, including the full 4H width. */
void proveVisibleBootstrap(DeviceId device)
{
    auto* backend=getBackendFor(device);
    if(!backend || backend->deviceCount()<=device.ordinal) GTEST_SKIP()<<"Requested backend unavailable";
    if(!hasCPUBackend()) initCPUBackend(-1);
    auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&] {
        void* stream=worker.defaultStream();
        ASSERT_NE(stream,nullptr);
        constexpr int capacity=17;
        // A resident first row consumes no duplicate shifted row. The final
        // two segments reset to a one-token prompt, then continue from it.
        constexpr std::array segments{
            Segment{0,9,0,9,FirstRow::ZeroSeed},
            Segment{9,2,9,2,FirstRow::PriorTerminal},
            Segment{11,1,12,0,FirstRow::NoRows},
            Segment{12,17,12,17,FirstRow::PriorTerminal},
            Segment{0,1,0,1,FirstRow::ZeroSeed},
            Segment{1,1,1,1,FirstRow::PriorTerminal},
        };
        for(const int width:{32,37,10240})
        {
            SCOPED_TRACE(width);
            FP32Tensor hidden({capacity,static_cast<std::size_t>(width)});
            FP32Tensor packed({capacity,static_cast<std::size_t>(width)});
            FP32Tensor archive({1,static_cast<std::size_t>(width)});
            INT32Tensor tokens({capacity}),positions({capacity}),shifted_tokens({capacity}),
                shifted_positions({capacity}),append({1}),length({1}),stride({1}),main_count({1}),shifted_count({1});
            std::fill_n(hidden.mutable_data(),hidden.numel(),-71.f);
            std::fill_n(packed.mutable_data(),packed.numel(),-83.f);
            std::fill_n(archive.mutable_data(),archive.numel(),-97.f);
            std::array<TensorBase*,12> owners{&hidden,&packed,&archive,&tokens,&positions,&shifted_tokens,
                &shifted_positions,&append,&length,&stride,&main_count,&shifted_count};
            for(auto* tensor:owners) TransferEngine::prepareDeviceInput(tensor,device,stream);
            std::array<const void*,12> addresses{};
            for(std::size_t i=0;i<owners.size();++i) {
                addresses[i]=owners[i]->gpu_data_ptr();
                ASSERT_NE(addresses[i],nullptr);
            }
            const auto int_address=[](INT32Tensor& tensor){return static_cast<int32_t*>(tensor.gpu_data_ptr());};
            HiddenStateRowsSelectStage::Params params;
            params.device_id=device; params.input=&hidden; params.output=&packed;
            params.seq_len=capacity; params.d_model=width; params.selected_row_count=capacity;
            params.device_row_index_source=HiddenStateRowsSelectStage::DeviceRowIndexSource::ShiftedPrefillTransaction;
            params.request_sequence_lengths_device=int_address(length);
            params.request_row_stride_source=HiddenStateRowsSelectStage::RequestRowStrideSource::ExternalDeviceScalar;
            params.request_row_stride_device=int_address(stride);
            params.input_token_ids_device=int_address(tokens); params.input_position_ids_device=int_address(positions);
            params.shifted_token_ids_output_device=int_address(shifted_tokens);
            params.shifted_position_ids_output_device=int_address(shifted_positions);
            params.shifted_append_lengths_output_device=int_address(append);
            params.shifted_cache_origin=MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero;
            params.terminal_hidden_archive=&archive;
            params.main_cached_tokens_by_request={int_address(main_count)};
            params.shifted_cached_tokens_by_request={int_address(shifted_count)};
            params.request_count=1; params.terminal_hidden_archive_buffer_id=BufferId::PREFIX_TERMINAL_HIDDEN;
            HiddenStateRowsSelectStage stage(std::move(params));
            stage.setGPUStream(stream);
            auto graph=worker.createGraphCapture(stream);
            ASSERT_TRUE(graph);
            {
                ScopedBackendGraphCapture recording(worker,*graph,"QFN visible shifted-prefill bootstrap");
                ASSERT_TRUE(recording.begin());
                ASSERT_TRUE(stage.execute(nullptr));
                recording.finish();
            }
            ASSERT_EQ(graph->nodeCount(),1u);
            ASSERT_TRUE(graph->instantiate());
            const auto identity=stage.graphCaptureVariantSignature();
            std::vector<float> prior(width,-97.f),input(hidden.numel()),result(packed.numel()),terminal(width);
            std::array<int32_t,capacity> token_input{},position_input{},token_output{},position_output{};
            ObservationRetirement retirement{backend,device,stream};
            int phase=0;
            for(const auto& segment:segments)
            {
                SCOPED_TRACE(phase);
                for(int row=0;row<capacity;++row) {
                    token_input[row]=37+segment.base+row; position_input[row]=segment.base+row;
                    for(int column=0;column<width;++column)
                        input[std::size_t(row)*width+column]=float(phase*1000+row*10)+float(column)*0.125f;
                }
                const int32_t main=segment.base+segment.rows,live=segment.rows,shifted=segment.shifted;
                ASSERT_TRUE(backend->hostToDeviceOnStream(hidden.gpu_data_ptr(),input.data(),hidden.size_bytes(),device.ordinal,stream));
                ASSERT_TRUE(backend->hostToDeviceOnStream(tokens.gpu_data_ptr(),token_input.data(),tokens.size_bytes(),device.ordinal,stream));
                ASSERT_TRUE(backend->hostToDeviceOnStream(positions.gpu_data_ptr(),position_input.data(),positions.size_bytes(),device.ordinal,stream));
                ASSERT_TRUE(backend->hostToDeviceOnStream(length.gpu_data_ptr(),&live,sizeof(live),device.ordinal,stream));
                ASSERT_TRUE(backend->hostToDeviceOnStream(stride.gpu_data_ptr(),&live,sizeof(live),device.ordinal,stream));
                ASSERT_TRUE(backend->hostToDeviceOnStream(main_count.gpu_data_ptr(),&main,sizeof(main),device.ordinal,stream));
                ASSERT_TRUE(backend->hostToDeviceOnStream(shifted_count.gpu_data_ptr(),&shifted,sizeof(shifted),device.ordinal,stream));
                ASSERT_TRUE(graph->launchOnStream(stream));
                int32_t appended=-1;
                ASSERT_TRUE(backend->deviceToHostOnStream(result.data(),packed.gpu_data_ptr(),packed.size_bytes(),device.ordinal,stream));
                ASSERT_TRUE(backend->deviceToHostOnStream(terminal.data(),archive.gpu_data_ptr(),archive.size_bytes(),device.ordinal,stream));
                ASSERT_TRUE(backend->deviceToHostOnStream(token_output.data(),shifted_tokens.gpu_data_ptr(),shifted_tokens.size_bytes(),device.ordinal,stream));
                ASSERT_TRUE(backend->deviceToHostOnStream(position_output.data(),shifted_positions.gpu_data_ptr(),shifted_positions.size_bytes(),device.ordinal,stream));
                ASSERT_TRUE(backend->deviceToHostOnStream(&appended,append.gpu_data_ptr(),sizeof(appended),device.ordinal,stream));
                // This is the complete terminal observation of one test replay;
                // the retained stage itself has no host boundary or allocation.
                ASSERT_TRUE(backend->synchronizeStream(stream,device.ordinal));
                EXPECT_EQ(appended,segment.expected_append);
                for(int row=0;row<capacity;++row) {
                    EXPECT_EQ(token_output[row],row<segment.expected_append?token_input[row]:0);
                    EXPECT_EQ(position_output[row],row<segment.expected_append?position_input[row]:0);
                    for(int column=0;column<width;++column) {
                        float expected=0.f;
                        if(row<segment.expected_append) {
                            if(row>0) expected=input[std::size_t(row-1)*width+column];
                            else if(segment.first==FirstRow::PriorTerminal) expected=prior[column];
                        }
                        EXPECT_EQ(result[std::size_t(row)*width+column],expected);
                    }
                }
                for(int column=0;column<width;++column)
                    EXPECT_EQ(terminal[column],input[std::size_t(segment.rows-1)*width+column]);
                prior=terminal;
                for(std::size_t i=0;i<owners.size();++i) EXPECT_EQ(owners[i]->gpu_data_ptr(),addresses[i]);
                EXPECT_EQ(stage.graphCaptureVariantSignature(),identity);
                EXPECT_EQ(graph->nodeCount(),1u);
                ++phase;
            }
        }
    });
}
} // namespace

/** @brief CUDA executes the actual QFN bootstrap transaction on a retained native graph. */
TEST(Qwen4ExpVisibleBootstrapNative, CUDA){proveVisibleBootstrap(DeviceId::cuda(0));}
/** @brief ROCm executes the same lifecycle and exact payload oracle on a retained native graph. */
TEST(Qwen4ExpVisibleBootstrapNative, ROCm){proveVisibleBootstrap(DeviceId::rocm(0));}
} // namespace llaminar2::test
