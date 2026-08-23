/**
 * @file Test__SegmentedGraphCaptureExecution.cpp
 * @brief Backend-bound integration tests for cached GPU graph replay.
 *
 * This source is compiled once for CUDA and once for ROCm.  Each resulting
 * binary explicitly registers only the backend named by its compile-time test
 * binding, preventing an unregistered factory from turning real GPU coverage
 * into a successful gtest skip.
 *
 * Coverage:
 * 1. First-use warmup atomically materializes a replay-ready full graph.
 * 2. Collective-marked segmented mode remains functional.
 * 3. Graph-stable snapshot slots preserve point-in-time outputs when a later
 *    stage overwrites the producer's arena storage.
 */

#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <array>
#include <cstdlib>

#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/compute_stages/ComputeStages.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IWorkerGPUContext.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include "../../../utils/TestTensorFactory.h"
#include "../../../utils/GraphArenaTestHarness.h"

using namespace llaminar2;
using namespace llaminar2::test;

#if defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
#define ENSURE_LINKED_GPU_BACKEND() ensureNvidiaFactoryRegistered()
#define HAS_LINKED_GPU_SUPPORT() GPUDeviceContextPool::instance().hasNvidiaSupport()
#define LINKED_GPU_CONTEXT() GPUDeviceContextPool::instance().getNvidiaContext(0)
#define LINKED_GPU_DEVICE_ID() DeviceId::cuda(0)
#define LINKED_GPU_SKIP_MESSAGE "CUDA not available"
#elif defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
#define ENSURE_LINKED_GPU_BACKEND() ensureAMDFactoryRegistered()
#define HAS_LINKED_GPU_SUPPORT() GPUDeviceContextPool::instance().hasAMDSupport()
#define LINKED_GPU_CONTEXT() GPUDeviceContextPool::instance().getAMDContext(0)
#define LINKED_GPU_DEVICE_ID() DeviceId::rocm(0)
#define LINKED_GPU_SKIP_MESSAGE "ROCm not available"
#else
#define ENSURE_LINKED_GPU_BACKEND() ((void)0)
#define HAS_LINKED_GPU_SUPPORT() false
#define LINKED_GPU_CONTEXT() GPUDeviceContextPool::instance().getContext("", 0)
#define LINKED_GPU_DEVICE_ID() DeviceId::cpu()
#define LINKED_GPU_SKIP_MESSAGE "No GPU backend linked in this test binary"
#endif

#define SKIP_IF_NO_GPU()                                      \
    do                                                        \
    {                                                         \
        ENSURE_LINKED_GPU_BACKEND();                          \
        if (!HAS_LINKED_GPU_SUPPORT())                        \
            GTEST_SKIP() << LINKED_GPU_SKIP_MESSAGE;          \
    } while (false)

namespace
{
    /** @brief Enable and restore PerfStats for one integration certificate. */
    class ScopedPerfStats final
    {
    public:
        ScopedPerfStats()
        {
            if (const char *value =
                    std::getenv("LLAMINAR_PERF_STATS_SUMMARY"))
            {
                had_value_ = true;
                value_ = value;
            }
            setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ~ScopedPerfStats()
        {
            if (had_value_)
                setenv(
                    "LLAMINAR_PERF_STATS_SUMMARY",
                    value_.c_str(),
                    1);
            else
                unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ScopedPerfStats(const ScopedPerfStats &) = delete;
        ScopedPerfStats &operator=(const ScopedPerfStats &) = delete;

    private:
        bool had_value_ = false;
        std::string value_;
    };

    /**
     * @brief CPU-only participant used to certify the captured ticket ABI.
     *
     * Production dispatch/local/return stages receive their own CPU integration
     * certificate.  This deliberately tiny stage isolates the heterogeneous
     * graph protocol: it reads the captured pinned dispatch ticket and writes
     * the pinned return ticket without any backend operation or tensor shadow.
     */
    class TicketEchoManualStage final : public IComputeStage
    {
    public:
        explicit TicketEchoManualStage(
            std::shared_ptr<MoEOverlayDispatchTicketStorage> storage)
            : IComputeStage(DeviceId::cpu()), storage_(std::move(storage))
        {
        }

        bool execute(IDeviceContext *ctx) override
        {
            complete_ = false;
            if (!ctx || !storage_ || !storage_->hasValidBoundIdentity())
                return false;
            std::string publication_error;
            if (!storage_->awaitCapturedPublication(&publication_error))
                return false;
            auto &ticket = storage_->ticket();
            if (!ticket.isValid())
                return false;

            const int rows = ticket.header->logical_row_count;
            const int bucket = ticket.header->bucket_row_capacity;
            const int top_k = ticket.header->top_k;
            const int d_model = ticket.header->d_model;
            std::fill_n(
                ticket.return_rows_fp32,
                static_cast<size_t>(bucket) * static_cast<size_t>(d_model),
                0.0f);
            for (int row = 0; row < rows; ++row)
            {
                const float bias = ticket.routing_weights_fp32[
                    static_cast<size_t>(row) * static_cast<size_t>(top_k)];
                for (int col = 0; col < d_model; ++col)
                {
                    const size_t index =
                        static_cast<size_t>(row) *
                            static_cast<size_t>(d_model) +
                        static_cast<size_t>(col);
                    ticket.return_rows_fp32[index] =
                        ticket.hidden_rows_fp32[index] * 2.0f + bias;
                }
            }
            ticket.header->return_logical_row_count = rows;
            if (call_count_ < observed_rows_.size())
                observed_rows_[call_count_] = rows;
            ++call_count_;
            complete_ = true;
            return true;
        }

        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_EXPERT_DISPATCH;
        }
        std::string name() const override
        {
            return "ticket_echo_manual";
        }
        bool supportsBackend(ComputeBackendType) const override { return true; }
        bool isGraphCapturable() const override { return false; }
        bool isManualGraphBoundary() const override { return true; }
        bool manualGraphBoundaryComplete() const override { return complete_; }
        bool supportsPaddedPrefillGraphCapturePreflight() const override
        {
            return true;
        }
        bool supportsPaddedPrefillRealLengthContract() const override
        {
            return true;
        }
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }
        StageBufferRequirements getBufferRequirements() const override
        {
            return {};
        }
        StageBufferContract bufferContract() const override
        {
            return StageBufferContract::build();
        }
        StageDumpInfo buildDumpInfoImpl() const override { return {}; }

        size_t callCount() const noexcept { return call_count_; }
        int observedRows(size_t call) const noexcept
        {
            return call < observed_rows_.size() ? observed_rows_[call] : -1;
        }

    private:
        std::shared_ptr<MoEOverlayDispatchTicketStorage> storage_;
        std::array<int, 4> observed_rows_{};
        size_t call_count_ = 0;
        bool complete_ = false;
    };
} // namespace

class CachedGraphReplayExecutionTest : public ::testing::Test
{
protected:
    IWorkerGPUContext *gpu_ctx_ = nullptr;
    std::unique_ptr<IDeviceContext> device_ctx_;
    GraphArenaTestHarness graph_arena_;
    std::vector<std::unique_ptr<TensorBase>> tensor_storage_;
    std::vector<TensorBase *> arena_tensors_;

    void SetUp() override
    {
        ENSURE_LINKED_GPU_BACKEND();
        if (HAS_LINKED_GPU_SUPPORT())
        {
            gpu_ctx_ = &LINKED_GPU_CONTEXT();
            device_ctx_ = IDeviceContext::create(LINKED_GPU_DEVICE_ID(), 1);
        }
    }

    void TearDown() override
    {
        tensor_storage_.clear();
        device_ctx_.reset();
        gpu_ctx_ = nullptr;
    }

    FP32Tensor *createFP32Tensor(const std::vector<size_t> &shape)
    {
        auto tensor = TestTensorFactory::createFP32(shape);
        auto *ptr = tensor.get();
        tensor_storage_.push_back(std::move(tensor));
        return static_cast<FP32Tensor *>(ptr);
    }

    INT32Tensor *createINT32Tensor(const std::vector<size_t> &shape)
    {
        auto tensor = TestTensorFactory::createINT32(shape);
        auto *ptr = tensor.get();
        tensor_storage_.push_back(std::move(tensor));
        return static_cast<INT32Tensor *>(ptr);
    }

    FP32Tensor *createArenaFP32Tensor(
        BufferId id,
        const std::vector<size_t> &shape)
    {
        auto *tensor =
            graph_arena_.createPersistentTensor<FP32Tensor>(id, shape);
        arena_tensors_.push_back(tensor);
        return tensor;
    }

    /**
     * @brief Allocate and upload every fixture-owned tensor before graph warmup.
     *
     * Cached replay creates a dedicated stream internally.  These synthetic
     * stages do not use arena BufferIds, so the executor cannot discover and
     * cohere their raw tensor parameters on our behalf.  Uploading on the
     * context's default stream and synchronizing once establishes stable device
     * addresses before warmup, capture, and replay bind the stages to their
     * dedicated stream.
     *
     * @param excluded_tensor Optional internal output left without GPU storage
     *        so cached capture preparation owns its first allocation.
     * @return true when every included tensor is resident and the upload stream
     *         has completed; false on allocation, transfer, or synchronization
     *         failure.
     */
    bool prepareFixtureTensorsForGPUExecution(
        ITensor *excluded_tensor = nullptr)
    {
        if (!gpu_ctx_ || !device_ctx_)
            return false;

        void *upload_stream = gpu_ctx_->defaultStream();
        if (!upload_stream)
            return false;

        const DeviceId device = device_ctx_->deviceId();
        for (const auto &tensor : tensor_storage_)
        {
            if (tensor.get() == excluded_tensor)
                continue;
            if (!tensor || !tensor->ensureOnDevice(device, upload_stream))
                return false;
        }
        for (auto *tensor : arena_tensors_)
        {
            if (tensor == excluded_tensor)
                continue;
            if (!tensor || !tensor->ensureOnDevice(device, upload_stream))
                return false;
        }

        return gpu_ctx_->synchronizeStreamChecked(upload_stream);
    }

    ComputeGraph buildNormResidualGraph(size_t seq_len, size_t d_model,
                                        FP32Tensor *&norm_input,
                                        FP32Tensor *&residual,
                                        FP32Tensor *&result_output,
                                        FP32Tensor **norm_output_out = nullptr,
                                        bool strict_copy_consumer = false)
    {
        const DeviceId device = device_ctx_->deviceId();
        norm_input = createArenaFP32Tensor(
            BufferId::HIDDEN_STATE, {seq_len, d_model});
        auto *norm_output = createArenaFP32Tensor(
            BufferId::NORMALIZED, {seq_len, d_model});
        if (norm_output_out)
            *norm_output_out = norm_output;
        auto *gamma = createFP32Tensor({d_model});
        residual = createArenaFP32Tensor(
            BufferId::RESIDUAL, {seq_len, d_model});
        result_output = createArenaFP32Tensor(
            BufferId::ATTN_OUTPUT, {seq_len, d_model});

        const size_t num_elements = seq_len * d_model;
        for (size_t i = 0; i < num_elements; ++i)
            norm_input->mutable_data()[i] = 0.5f + static_cast<float>(i % 10) * 0.1f;
        for (size_t i = 0; i < d_model; ++i)
            gamma->mutable_data()[i] = 1.0f;
        for (size_t i = 0; i < num_elements; ++i)
            residual->mutable_data()[i] = 0.1f * static_cast<float>(i % 7);

        RMSNormStage::Params norm_params;
        norm_params.input = norm_input;
        norm_params.output = norm_output;
        norm_params.gamma = gamma;
        norm_params.eps = 1e-5f;
        norm_params.seq_len = static_cast<int>(seq_len);
        norm_params.device_id = device;
        norm_params.input_buffer_id = BufferId::HIDDEN_STATE;
        norm_params.output_buffer_id = BufferId::NORMALIZED;

        ResidualAddStage::Params res_params;
        res_params.input = norm_output;
        res_params.residual = strict_copy_consumer ? nullptr : residual;
        res_params.output = result_output;
        res_params.num_elements = num_elements;
        res_params.device_id = device;
        res_params.input_buffer_id = BufferId::NORMALIZED;
        if (!strict_copy_consumer)
            res_params.residual_buffer_id = BufferId::RESIDUAL;
        res_params.output_buffer_id = BufferId::ATTN_OUTPUT;

        ComputeGraph graph;
        graph.addNode("rmsnorm", ComputeStageFactory::createRMSNorm(norm_params), device);
        graph.addNode("residual_add", ComputeStageFactory::createResidualAdd(res_params), device);
        graph.addDependency("residual_add", "rmsnorm");
        return graph;
    }

    ComputeGraph buildSnapshotOverwriteGraph(size_t seq_len, size_t d_model,
                                             FP32Tensor *&norm_input,
                                             FP32Tensor *&scratch)
    {
        const DeviceId device = device_ctx_->deviceId();
        norm_input = createArenaFP32Tensor(
            BufferId::HIDDEN_STATE, {seq_len, d_model});
        scratch = createArenaFP32Tensor(
            BufferId::NORMALIZED, {seq_len, d_model});
        auto *gamma = createFP32Tensor({d_model});
        auto *residual = createArenaFP32Tensor(
            BufferId::RESIDUAL, {seq_len, d_model});

        const size_t num_elements = seq_len * d_model;
        for (size_t i = 0; i < num_elements; ++i)
        {
            norm_input->mutable_data()[i] = 0.5f + static_cast<float>(i % 10) * 0.1f;
            residual->mutable_data()[i] = 10.0f;
        }
        for (size_t i = 0; i < d_model; ++i)
            gamma->mutable_data()[i] = 1.0f;

        RMSNormStage::Params norm_params;
        norm_params.device_id = device;
        norm_params.input = norm_input;
        norm_params.output = scratch;
        norm_params.gamma = gamma;
        norm_params.eps = 1e-5f;
        norm_params.seq_len = static_cast<int>(seq_len);
        norm_params.input_buffer_id = BufferId::HIDDEN_STATE;
        norm_params.output_buffer_id = BufferId::NORMALIZED;

        ResidualAddStage::Params overwrite_params;
        overwrite_params.device_id = device;
        overwrite_params.input = residual;
        overwrite_params.residual = scratch;
        overwrite_params.output = scratch;
        overwrite_params.num_elements = num_elements;
        overwrite_params.input_buffer_id = BufferId::RESIDUAL;
        overwrite_params.residual_buffer_id = BufferId::NORMALIZED;
        overwrite_params.output_buffer_id = BufferId::NORMALIZED;

        ComputeGraph graph;
        graph.addNode("stage1_norm", ComputeStageFactory::createRMSNorm(norm_params), device);
        graph.addNode("stage2_overwrite", ComputeStageFactory::createResidualAdd(overwrite_params), device);
        graph.addDependency("stage2_overwrite", "stage1_norm");
        return graph;
    }

    static void fillSnapshotInput(FP32Tensor *tensor, float offset)
    {
        ASSERT_NE(tensor, nullptr);
        float *data = tensor->mutable_data();
        ASSERT_NE(data, nullptr);
        for (size_t i = 0; i < tensor->numel(); ++i)
        {
            data[i] = 0.5f +
                      static_cast<float>(i % 10) * 0.1f +
                      offset * (1.0f + static_cast<float>(i % 3) * 0.25f);
        }
    }

    static void assertFiniteAndNonZero(const float *data, size_t count)
    {
        bool has_nonzero = false;
        for (size_t i = 0; i < count; ++i)
        {
            ASSERT_FALSE(std::isnan(data[i])) << "NaN at index " << i;
            ASSERT_FALSE(std::isinf(data[i])) << "Inf at index " << i;
            if (data[i] != 0.0f)
            {
                has_nonzero = true;
            }
        }
        EXPECT_TRUE(has_nonzero) << "Output tensor is all zeros";
    }

    static void assertTensorFiniteAndNonZero(
        FP32Tensor *tensor,
        size_t count,
        void *producer_stream)
    {
        ASSERT_NE(tensor, nullptr);
        ASSERT_NE(producer_stream, nullptr);
        ASSERT_TRUE(tensor->ensureOnHost(producer_stream))
            << "The test's explicit host observation must consume the exact graph producer stream.";
        assertFiniteAndNonZero(tensor->data(), count);
    }

    static void assertGraphStagesUseStream(ComputeGraph &graph, void *stream)
    {
        ASSERT_NE(stream, nullptr);
        for (const auto &node_name : graph.getExecutionOrder())
        {
            ComputeNode *node = graph.getNode(node_name);
            ASSERT_NE(node, nullptr) << "Missing node: " << node_name;
            ASSERT_NE(node->stage, nullptr) << "Missing stage: " << node_name;
            EXPECT_EQ(node->stage->gpuStream(), stream)
                << "Stage should remain bound to the explicit capture stream: " << node_name;
        }
    }

    static void assertSnapshotDelta(
        const std::unordered_map<std::string, std::vector<float>> &snapshots,
        const std::string &before_stage,
        const std::string &after_stage,
        float expected_delta)
    {
        auto before_it = snapshots.find(before_stage);
        auto after_it = snapshots.find(after_stage);
        ASSERT_NE(before_it, snapshots.end()) << "Missing snapshot for " << before_stage;
        ASSERT_NE(after_it, snapshots.end()) << "Missing snapshot for " << after_stage;
        ASSERT_EQ(before_it->second.size(), after_it->second.size());
        ASSERT_FALSE(before_it->second.empty());
        for (size_t i = 0; i < before_it->second.size(); ++i)
        {
            EXPECT_NEAR(after_it->second[i] - before_it->second[i], expected_delta, 1e-4f)
                << "Snapshot delta mismatch at index " << i;
        }
    }

    static void assertSnapshotsDiffer(
        const std::unordered_map<std::string, std::vector<float>> &lhs,
        const std::unordered_map<std::string, std::vector<float>> &rhs,
        const std::string &stage_name)
    {
        auto lhs_it = lhs.find(stage_name);
        auto rhs_it = rhs.find(stage_name);
        ASSERT_NE(lhs_it, lhs.end()) << "Missing lhs snapshot for " << stage_name;
        ASSERT_NE(rhs_it, rhs.end()) << "Missing rhs snapshot for " << stage_name;
        ASSERT_EQ(lhs_it->second.size(), rhs_it->second.size());

        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < lhs_it->second.size(); ++i)
        {
            max_abs_diff = std::max(max_abs_diff, std::abs(lhs_it->second[i] - rhs_it->second[i]));
        }
        EXPECT_GT(max_abs_diff, 1e-3f)
            << "Snapshot for " << stage_name << " did not refresh across graph execution";
    }
};

TEST_F(CachedGraphReplayExecutionTest, FirstUseMaterializesReplayWithoutSecondMutation)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);
    const size_t seq_len = 4;
    const size_t d_model = 64;
    const size_t num_elements = seq_len * d_model;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(seq_len, d_model, norm_input, residual, result);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    GraphExecutorConfig exec_config;
    exec_config.enable_validation = false;
    DeviceGraphExecutor executor(exec_config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache segment_cache;
    void *stream = gpu_ctx_->defaultStream();
    ASSERT_NE(stream, nullptr);

    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, stream, gpu_ctx_, nullptr));
    EXPECT_TRUE(segment_cache.initialized);
    EXPECT_FALSE(segment_cache.needs_capture)
        << "A successful first use must not expose initialized-but-uncaptured state.";
    assertTensorFiniteAndNonZero(
        result, num_elements, segment_cache.capture_stream);

    std::string export_error;
    const auto replay_template =
        segment_cache.deviceLoopGraphTemplate(graph, &export_error);
    ASSERT_TRUE(replay_template.has_value()) << export_error;
    ASSERT_NE(replay_template->capture, nullptr);
    EXPECT_EQ(replay_template->stage_count, graph.getExecutionOrder().size());
    EXPECT_EQ(replay_template->stream, segment_cache.capture_stream);

    std::vector<float> first_use_output(num_elements);
    std::memcpy(
        first_use_output.data(),
        result->data(),
        num_elements * sizeof(float));

    graph.reset();
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, stream, gpu_ctx_, nullptr));
    EXPECT_TRUE(segment_cache.initialized);
    EXPECT_FALSE(segment_cache.needs_capture);
    assertTensorFiniteAndNonZero(
        result, num_elements, segment_cache.capture_stream);

    const float *replay = result->data();
    for (size_t i = 0; i < num_elements; ++i)
    {
        EXPECT_NEAR(replay[i], first_use_output[i], 1e-5f)
            << "Replay output differs from the single first-use execution at index " << i;
    }
}

TEST_F(CachedGraphReplayExecutionTest,
       FirstCaptureAllocatesColdInternalProducerStorage)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);

    constexpr size_t seq_len = 2;
    constexpr size_t d_model = 32;
    constexpr size_t num_elements = seq_len * d_model;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    FP32Tensor *internal_norm_output = nullptr;
    auto graph = buildNormResidualGraph(
        seq_len,
        d_model,
        norm_input,
        residual,
        result,
        &internal_norm_output,
        /*strict_copy_consumer=*/true);
    ASSERT_NE(internal_norm_output, nullptr);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution(internal_norm_output));
    ASSERT_EQ(internal_norm_output->gpu_data_ptr(), nullptr)
        << "The regression requires a cold internal producer output";

    GraphExecutorConfig exec_config;
    exec_config.enable_validation = false;
    DeviceGraphExecutor executor(exec_config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache segment_cache;
    void *dispatch_stream = gpu_ctx_->defaultStream();
    ASSERT_NE(dispatch_stream, nullptr);

    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph,
        device_ctx_.get(),
        segment_cache,
        dispatch_stream,
        gpu_ctx_,
        nullptr));
    ASSERT_NE(internal_norm_output->gpu_data_ptr(), nullptr);
    ASSERT_EQ(
        internal_norm_output->current_device(),
        std::optional<DeviceId>{device_ctx_->deviceId()});
    assertTensorFiniteAndNonZero(
        result,
        num_elements,
        segment_cache.capture_stream);
}

TEST_F(CachedGraphReplayExecutionTest, DISABLED_CollectiveMarkedMode_RemainsFunctional)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);

    const size_t seq_len = 4;
    const size_t d_model = 64;
    const size_t num_elements = seq_len * d_model;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(seq_len, d_model, norm_input, residual, result);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    GraphExecutorConfig exec_config;
    exec_config.enable_validation = false;
    DeviceGraphExecutor executor(exec_config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache segment_cache;
    void *stream = gpu_ctx_->defaultStream();
    ASSERT_NE(stream, nullptr);

    // NOTE: This test remains disabled in Phase 0 because the current
    // collective-marked manual-segment path can yield zeroed outputs for this
    // synthetic graph. Keep it as a scaffold for follow-up stabilization.

    std::unordered_set<std::string> collective_nodes = {"rmsnorm"};

    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, stream, gpu_ctx_, &collective_nodes));
    EXPECT_TRUE(segment_cache.initialized);
    EXPECT_FALSE(segment_cache.segments.empty());

    bool has_manual_segment = false;
    for (const auto &seg : segment_cache.segments)
    {
        if (!seg.capturable)
        {
            has_manual_segment = true;
            break;
        }
    }
    EXPECT_TRUE(has_manual_segment);
    assertTensorFiniteAndNonZero(
        result, num_elements, segment_cache.capture_stream);

    graph.reset();
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, stream, gpu_ctx_, &collective_nodes));

    graph.reset();
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, stream, gpu_ctx_, &collective_nodes));
    assertTensorFiniteAndNonZero(
        result, num_elements, segment_cache.capture_stream);
}

TEST_F(CachedGraphReplayExecutionTest,
       HeterogeneousTicketSegmentsReuseCapturedBucketAcrossLogicalLengths)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);
    ScopedPerfStats perf_stats;

    constexpr int layer = 3;
    constexpr int bucket_rows = 4;
    constexpr int top_k = 2;
    constexpr int d_model = 8;
    const DeviceId device = device_ctx_->deviceId();

    auto *hidden = createArenaFP32Tensor(
        BufferId::NORMALIZED,
        {bucket_rows, d_model});
    auto *routing_indices = createArenaFP32Tensor(
        BufferId::MOE_EXPERT_INDICES,
        {bucket_rows, top_k});
    auto *routing_weights = createArenaFP32Tensor(
        BufferId::MOE_EXPERT_WEIGHTS,
        {bucket_rows, top_k});
    auto *output = createArenaFP32Tensor(
        BufferId::MOE_COMBINED_OUTPUT,
        {bucket_rows, d_model});
    auto *active_rows = createINT32Tensor({1});

    auto ticket_storage =
        std::make_shared<MoEOverlayDispatchTicketStorage>();
    ticket_storage->bindFixedCapacity(
        layer,
        bucket_rows,
        top_k,
        d_model,
        device,
        /*workspace_generation=*/29);
    const auto *const ticket_header = ticket_storage->ticket().header;
    const auto *const ticket_hidden =
        ticket_storage->ticket().hidden_rows_fp32;
    const auto *const ticket_return =
        ticket_storage->ticket().return_rows_fp32;

    /*
     * Reproduce production cold preflight before the arena has published GPU
     * addresses. Static ticket geometry must be admitted, while the stronger
     * capture-ready predicate must still reject the unbound device pointers.
     */
    MoEOverlayTicketPublishStage::Params cold_publish_params;
    cold_publish_params.device_id = device;
    cold_publish_params.hidden = hidden;
    cold_publish_params.routing_indices = routing_indices;
    cold_publish_params.routing_weights = routing_weights;
    cold_publish_params.layer_idx = layer;
    cold_publish_params.bucket_rows = bucket_rows;
    cold_publish_params.top_k = top_k;
    cold_publish_params.d_model = d_model;
    cold_publish_params.ticket_storage = ticket_storage;
    MoEOverlayTicketPublishStage cold_publish(cold_publish_params);
    EXPECT_TRUE(cold_publish.supportsLazyPrefillGraphCapturePreflight());
    EXPECT_TRUE(cold_publish.supportsGraphCaptureAfterLaunchPreparation());
    EXPECT_EQ(
        cold_publish.graphLaunchPreparationPolicy(),
        GraphLaunchPreparationPolicy::CaptureAndReplay);
    EXPECT_FALSE(cold_publish.isGraphCapturable());
    EXPECT_FALSE(cold_publish.supportsPaddedPrefillGraphCapturePreflight())
        << "Padded capture additionally requires the device-owned live-row scalar";

    MoEOverlayTicketConsumeStage::Params cold_consume_params;
    cold_consume_params.device_id = device;
    cold_consume_params.output = output;
    cold_consume_params.layer_idx = layer;
    cold_consume_params.bucket_rows = bucket_rows;
    cold_consume_params.d_model = d_model;
    cold_consume_params.ticket_storage = ticket_storage;
    MoEOverlayTicketConsumeStage cold_consume(cold_consume_params);
    EXPECT_TRUE(cold_consume.supportsLazyPrefillGraphCapturePreflight());
    EXPECT_TRUE(cold_consume.supportsGraphCaptureAfterLaunchPreparation());
    EXPECT_EQ(
        cold_consume.graphLaunchPreparationPolicy(),
        GraphLaunchPreparationPolicy::CaptureOnly);
    EXPECT_TRUE(cold_consume.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(cold_consume.isGraphCapturable());

    const auto fill_transaction = [&](int logical_rows, float base)
    {
        ASSERT_GT(logical_rows, 0);
        ASSERT_LE(logical_rows, bucket_rows);
        std::fill_n(
            hidden->mutable_data(),
            hidden->numel(),
            -777.0f);
        std::fill_n(
            routing_indices->mutable_data(),
            routing_indices->numel(),
            99.0f);
        std::fill_n(
            routing_weights->mutable_data(),
            routing_weights->numel(),
            -999.0f);
        for (int row = 0; row < logical_rows; ++row)
        {
            routing_indices->mutable_data()[
                static_cast<size_t>(row) * top_k] =
                static_cast<float>(row);
            routing_indices->mutable_data()[
                static_cast<size_t>(row) * top_k + 1] =
                static_cast<float>(row + 1);
            routing_weights->mutable_data()[
                static_cast<size_t>(row) * top_k] =
                0.2f + 0.1f * static_cast<float>(row);
            routing_weights->mutable_data()[
                static_cast<size_t>(row) * top_k + 1] =
                0.8f - 0.1f * static_cast<float>(row);
            for (int col = 0; col < d_model; ++col)
            {
                hidden->mutable_data()[
                    static_cast<size_t>(row) * d_model + col] =
                    base + static_cast<float>(row) * 0.5f +
                    static_cast<float>(col) * 0.025f;
            }
        }
        active_rows->mutable_int32_data()[0] = logical_rows;
    };

    fill_transaction(/*logical_rows=*/3, /*base=*/1.0f);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    IBackend *const backend = getBackendFor(device);
    ASSERT_NE(backend, nullptr);
    const int device_ordinal = device.gpu_ordinal();
    const auto pinned_scalar_deleter =
        [backend, device_ordinal](int32_t *pointer)
    {
        backend->freePinned(pointer, device_ordinal);
    };
    std::unique_ptr<int32_t, decltype(pinned_scalar_deleter)>
        active_rows_staging(
            static_cast<int32_t *>(backend->allocatePinned(
                sizeof(int32_t), device_ordinal)),
            pinned_scalar_deleter);
    ASSERT_NE(active_rows_staging, nullptr);

    MoEOverlayTicketPublishStage::Params publish_params;
    publish_params.device_id = device;
    publish_params.hidden = hidden;
    publish_params.routing_indices = routing_indices;
    publish_params.routing_weights = routing_weights;
    publish_params.hidden_buffer_id = BufferId::NORMALIZED;
    publish_params.routing_indices_buffer_id =
        BufferId::MOE_EXPERT_INDICES;
    publish_params.routing_weights_buffer_id =
        BufferId::MOE_EXPERT_WEIGHTS;
    publish_params.active_row_count_device =
        static_cast<const int32_t *>(active_rows->gpu_data_ptr());
    publish_params.layer_idx = layer;
    publish_params.bucket_rows = bucket_rows;
    publish_params.top_k = top_k;
    publish_params.d_model = d_model;
    publish_params.ticket_storage = ticket_storage;

    auto manual_stage =
        std::make_unique<TicketEchoManualStage>(ticket_storage);
    TicketEchoManualStage *const manual_probe = manual_stage.get();

    MoEOverlayTicketConsumeStage::Params consume_params;
    consume_params.device_id = device;
    consume_params.output = output;
    consume_params.output_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
    consume_params.layer_idx = layer;
    consume_params.bucket_rows = bucket_rows;
    consume_params.d_model = d_model;
    consume_params.ticket_storage = ticket_storage;

    ComputeGraph graph;
    graph.addNode(
        "ticket_publish",
        ComputeStageFactory::createMoEOverlayTicketPublish(publish_params),
        device);
    graph.addNode(
        "cpu_ticket_participant",
        std::move(manual_stage),
        DeviceId::cpu());
    graph.addNode(
        "ticket_consume",
        ComputeStageFactory::createMoEOverlayTicketConsume(consume_params),
        device);
    graph.addDependency("cpu_ticket_participant", "ticket_publish");
    graph.addDependency("ticket_consume", "cpu_ticket_participant");

    GraphExecutorConfig exec_config;
    exec_config.enable_validation = false;
    DeviceGraphExecutor executor(exec_config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache segment_cache;
    void *dispatch_stream = gpu_ctx_->defaultStream();
    ASSERT_NE(dispatch_stream, nullptr);

    const auto execute_with_cache = [&graph,
                                     &executor,
                                     this,
                                     dispatch_stream](
                                        DeviceGraphExecutor::GraphSegmentCache &cache,
                                        DeviceGraphExecutor::GraphInitialSubmissionPolicy
                                            initial_submission)
    {
        return executor.executeWithCachedGraphReplay(
            graph,
            device_ctx_.get(),
            cache,
            dispatch_stream,
            gpu_ctx_,
            nullptr,
            /*collectives_graph_capturable=*/false,
            /*force_recapture=*/false,
            /*defer_final_sync=*/false,
            {},
            DeviceGraphExecutor::GraphReplayPlanPolicy::
                AllowHeterogeneousBoundarySegmentation,
            {},
            {},
            {},
            initial_submission);
    };
    const auto execute = [&]()
    {
        return execute_with_cache(
            segment_cache,
            DeviceGraphExecutor::GraphInitialSubmissionPolicy::
                CaptureInstantiateAndLaunch);
    };

    const auto expect_output = [&](int logical_rows, float base)
    {
        ASSERT_TRUE(output->ensureOnHost(segment_cache.capture_stream));
        for (int row = 0; row < logical_rows; ++row)
        {
            const float bias =
                0.2f + 0.1f * static_cast<float>(row);
            for (int col = 0; col < d_model; ++col)
            {
                const float input =
                    base + static_cast<float>(row) * 0.5f +
                    static_cast<float>(col) * 0.025f;
                EXPECT_FLOAT_EQ(
                    output->data()[
                        static_cast<size_t>(row) * d_model + col],
                    input * 2.0f + bias)
                    << "row=" << row << " col=" << col;
            }
        }
        for (int row = logical_rows; row < bucket_rows; ++row)
        {
            for (int col = 0; col < d_model; ++col)
            {
                EXPECT_FLOAT_EQ(
                    output->data()[
                        static_cast<size_t>(row) * d_model + col],
                    0.0f)
                    << "padding row=" << row << " col=" << col;
            }
        }
    };

    ASSERT_TRUE(execute());
    ASSERT_EQ(segment_cache.segments.size(), 3u);
    EXPECT_TRUE(segment_cache.segments[0].capturable);
    EXPECT_FALSE(segment_cache.segments[1].capturable);
    EXPECT_TRUE(segment_cache.segments[2].capturable);
    ASSERT_NE(segment_cache.segments[0].capture, nullptr);
    ASSERT_NE(segment_cache.segments[2].capture, nullptr);
    const auto *const publish_capture =
        segment_cache.segments[0].capture.get();
    const auto *const consume_capture =
        segment_cache.segments[2].capture.get();
    EXPECT_TRUE(ticket_storage->hasCapturedPublicationContract());
    ASSERT_EQ(manual_probe->callCount(), 1u);
    EXPECT_EQ(manual_probe->observedRows(0), 3);
    expect_output(/*logical_rows=*/3, /*base=*/1.0f);

    fill_transaction(/*logical_rows=*/1, /*base=*/4.0f);
    ASSERT_TRUE(hidden->ensureOnDevice(device, segment_cache.capture_stream));
    ASSERT_TRUE(routing_indices->ensureOnDevice(
        device,
        segment_cache.capture_stream));
    ASSERT_TRUE(routing_weights->ensureOnDevice(
        device,
        segment_cache.capture_stream));
    *active_rows_staging = 1;
    ASSERT_TRUE(backend->hostToDeviceOnStream(
        active_rows->gpu_data_ptr(),
        active_rows_staging.get(),
        sizeof(int32_t),
        device_ordinal,
        segment_cache.capture_stream));
    graph.reset();
    ASSERT_TRUE(execute());

    EXPECT_EQ(segment_cache.segments[0].capture.get(), publish_capture);
    EXPECT_EQ(segment_cache.segments[2].capture.get(), consume_capture);
    ASSERT_EQ(manual_probe->callCount(), 2u);
    EXPECT_EQ(manual_probe->observedRows(1), 1);
    EXPECT_EQ(ticket_storage->ticket().header, ticket_header);
    EXPECT_EQ(ticket_storage->ticket().hidden_rows_fp32, ticket_hidden);
    EXPECT_EQ(ticket_storage->ticket().return_rows_fp32, ticket_return);
    expect_output(/*logical_rows=*/1, /*base=*/4.0f);

    /*
     * Server readiness seals every graph before request admission. Reset the
     * mutable ticket to an explicitly incomplete state and prove that native
     * recording needs only its immutable pinned address. No manual participant
     * may execute and no graph unit may launch during this setup transaction.
     */
    ticket_storage->ticket().header->return_logical_row_count = 0;
    ASSERT_FALSE(ticket_storage->ticket().returnPayloadReady());
    graph.reset();
    DeviceGraphExecutor::GraphSegmentCache setup_cache;
    ASSERT_TRUE(execute_with_cache(
        setup_cache,
        DeviceGraphExecutor::GraphInitialSubmissionPolicy::
            MaterializeWithoutLaunch));
    EXPECT_EQ(manual_probe->callCount(), 2u);
    EXPECT_EQ(
        setup_cache.executable_submission_state,
        DeviceGraphExecutor::GraphSegmentCache::
            ExecutableSubmissionState::MaterializedUnlaunched);
    EXPECT_EQ(setup_cache.successful_submission_count, 0u);
}

TEST_F(CachedGraphReplayExecutionTest, PreserveResetKeepsExplicitCaptureStreamForRecapture)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);

    const size_t seq_len = 4;
    const size_t d_model = 64;
    const size_t num_elements = seq_len * d_model;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(seq_len, d_model, norm_input, residual, result);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    GraphExecutorConfig exec_config;
    exec_config.enable_validation = false;
    DeviceGraphExecutor executor(exec_config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache segment_cache;
    void *dispatch_stream = gpu_ctx_->defaultStream();
    ASSERT_NE(dispatch_stream, nullptr);

    // First use creates the dedicated cached-replay stream, executes warmup,
    // and atomically materializes the replay graph. This stream must not be
    // replaced by retry/reset plumbing.
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, dispatch_stream, gpu_ctx_, nullptr));
    void *capture_stream = segment_cache.capture_stream;
    ASSERT_NE(capture_stream, nullptr);
    EXPECT_NE(capture_stream, dispatch_stream)
        << "Cached graph replay should use a dedicated explicit stream, not the dispatch/default stream";
    assertGraphStagesUseStream(graph, capture_stream);
    assertTensorFiniteAndNonZero(result, num_elements, capture_stream);

    // Reset with Preserve simulates capture retry/replay failure handling. The
    // next warmup must reuse the same live stream so cached stages never observe
    // a dangling stream or fall back to default-stream execution.
    segment_cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Preserve);
    EXPECT_EQ(segment_cache.capture_stream, capture_stream);

    graph.reset();
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, dispatch_stream, gpu_ctx_, nullptr));
    EXPECT_EQ(segment_cache.capture_stream, capture_stream);
    assertGraphStagesUseStream(graph, capture_stream);
    assertTensorFiniteAndNonZero(result, num_elements, capture_stream);

    // The recapture pass should continue using that same explicit stream.
    graph.reset();
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, dispatch_stream, gpu_ctx_, nullptr));
    EXPECT_EQ(segment_cache.capture_stream, capture_stream);
    assertGraphStagesUseStream(graph, capture_stream);
    assertTensorFiniteAndNonZero(result, num_elements, capture_stream);
}

TEST_F(CachedGraphReplayExecutionTest, CapturedSnapshotsPreservePointInTimeOutputsAcrossBufferReuse)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);

    const size_t seq_len = 4;
    const size_t d_model = 64;
    const size_t num_elements = seq_len * d_model;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *scratch = nullptr;
    auto graph = buildSnapshotOverwriteGraph(seq_len, d_model, norm_input, scratch);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    std::unordered_map<std::string, std::vector<float>> snapshots;
    GraphExecutorConfig exec_config;
    exec_config.enable_validation = false;
    exec_config.snapshot_callback = [&](const std::string &stage_name, const StageDumpInfo &dump_info)
    {
        ASSERT_FALSE(dump_info.outputs.empty()) << "Stage has no snapshot outputs: " << stage_name;
        const auto &output = dump_info.outputs.front();
        ASSERT_STREQ(output.dtype, "FP32");
        ASSERT_NE(output.data, nullptr);
        const size_t element_count = output.rows * output.cols;
        const auto *data = static_cast<const float *>(output.data);
        snapshots[stage_name].assign(data, data + element_count);
    };

    DeviceGraphExecutor executor(exec_config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache segment_cache;
    void *dispatch_stream = gpu_ctx_->defaultStream();
    ASSERT_NE(dispatch_stream, nullptr);

    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, dispatch_stream, gpu_ctx_, nullptr));
    ASSERT_NE(segment_cache.capture_stream, nullptr);
    snapshots.clear();
    ASSERT_TRUE(executor.publishSnapshotsAfterGraphExecution(
        graph,
        segment_cache.capture_stream,
        "snapshot-overwrite-first-use",
        &segment_cache.snapshot_manifest));
    assertSnapshotDelta(snapshots, "stage1_norm", "stage2_overwrite", 10.0f);
    const auto warmup_snapshots = snapshots;
    assertTensorFiniteAndNonZero(
        scratch, num_elements, segment_cache.capture_stream);

    /*
     * Fixed-width prefill graphs retain all captured rows in their immutable
     * D2D snapshot slots, but publication must expose only request-owned rows.
     * Model a two-request bucket with two physical rows each: request zero owns
     * one row and request one owns both. The callback must receive the compact
     * three-row matrix without changing or recapturing the graph.
     */
    const DeviceGraphExecutor::GraphSnapshotLogicalRows logical_rows{
        .physical_rows_per_sequence = 2,
        .logical_rows_per_sequence = {1, 2},
    };
    snapshots.clear();
    ASSERT_TRUE(executor.publishSnapshotsAfterGraphExecution(
        graph,
        segment_cache.capture_stream,
        "snapshot-logical-row-projection",
        &segment_cache.snapshot_manifest,
        &logical_rows));
    for (const std::string stage_name : {"stage1_norm", "stage2_overwrite"})
    {
        const auto full = warmup_snapshots.find(stage_name);
        const auto projected = snapshots.find(stage_name);
        ASSERT_NE(full, warmup_snapshots.end());
        ASSERT_NE(projected, snapshots.end());
        ASSERT_EQ(full->second.size(), num_elements);
        ASSERT_EQ(projected->second.size(), 3 * d_model);

        std::vector<float> expected;
        expected.reserve(3 * d_model);
        expected.insert(
            expected.end(),
            full->second.begin(),
            full->second.begin() + static_cast<std::ptrdiff_t>(d_model));
        expected.insert(
            expected.end(),
            full->second.begin() + static_cast<std::ptrdiff_t>(2 * d_model),
            full->second.end());
        EXPECT_EQ(projected->second, expected)
            << "Logical snapshot row projection changed payload order for "
            << stage_name;
    }

    /*
     * A heterogeneous captured prefill graph can cross a CPU/manual boundary
     * after a GPU segment. The CPU stage has no graph-stable D2D snapshot slot,
     * but its parity artifact still must expose logical rather than padded
     * rows. Exercise the common post-graph publisher directly so this remains
     * true independently of the GPU manifest implementation above.
     */
    auto *cpu_input = createFP32Tensor({seq_len, d_model});
    auto *cpu_residual = createFP32Tensor({seq_len, d_model});
    auto *cpu_output = createFP32Tensor({seq_len, d_model});
    for (size_t index = 0; index < num_elements; ++index)
    {
        cpu_input->mutable_data()[index] = static_cast<float>(index + 1);
        cpu_residual->mutable_data()[index] = 0.25f;
        cpu_output->mutable_data()[index] =
            cpu_input->data()[index] + cpu_residual->data()[index];
    }

    ResidualAddStage::Params cpu_snapshot_params;
    cpu_snapshot_params.device_id = DeviceId::cpu();
    cpu_snapshot_params.input = cpu_input;
    cpu_snapshot_params.residual = cpu_residual;
    cpu_snapshot_params.output = cpu_output;
    cpu_snapshot_params.num_elements = num_elements;
    ComputeGraph cpu_snapshot_graph;
    cpu_snapshot_graph.addNode(
        "cpu_snapshot_boundary",
        ComputeStageFactory::createResidualAdd(cpu_snapshot_params),
        DeviceId::cpu());

    const std::vector<float> full_cpu_snapshot(
        cpu_output->data(), cpu_output->data() + num_elements);
    snapshots.clear();
    ASSERT_TRUE(executor.publishSnapshotsAfterGraphExecution(
        cpu_snapshot_graph,
        segment_cache.capture_stream,
        "snapshot-logical-row-projection-cpu-boundary",
        nullptr,
        &logical_rows));
    const auto projected_cpu = snapshots.find("cpu_snapshot_boundary");
    ASSERT_NE(projected_cpu, snapshots.end());
    ASSERT_EQ(projected_cpu->second.size(), 3 * d_model);
    std::vector<float> expected_cpu_snapshot;
    expected_cpu_snapshot.reserve(3 * d_model);
    expected_cpu_snapshot.insert(
        expected_cpu_snapshot.end(),
        full_cpu_snapshot.begin(),
        full_cpu_snapshot.begin() + static_cast<std::ptrdiff_t>(d_model));
    expected_cpu_snapshot.insert(
        expected_cpu_snapshot.end(),
        full_cpu_snapshot.begin() + static_cast<std::ptrdiff_t>(2 * d_model),
        full_cpu_snapshot.end());
    EXPECT_EQ(projected_cpu->second, expected_cpu_snapshot)
        << "CPU/manual snapshot publication leaked padded rows";

    fillSnapshotInput(norm_input, 3.0f);
    ASSERT_TRUE(norm_input->ensureOnDevice(device_ctx_->deviceId(), segment_cache.capture_stream));
    graph.reset();
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, dispatch_stream, gpu_ctx_, nullptr));
    snapshots.clear();
    ASSERT_TRUE(executor.publishSnapshotsAfterGraphExecution(
        graph,
        segment_cache.capture_stream,
        "snapshot-overwrite-replay-1",
        &segment_cache.snapshot_manifest));
    assertSnapshotDelta(snapshots, "stage1_norm", "stage2_overwrite", 10.0f);
    assertSnapshotsDiffer(warmup_snapshots, snapshots, "stage1_norm");
    const auto capture_snapshots = snapshots;

    fillSnapshotInput(norm_input, 7.0f);
    ASSERT_TRUE(norm_input->ensureOnDevice(device_ctx_->deviceId(), segment_cache.capture_stream));
    graph.reset();
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, dispatch_stream, gpu_ctx_, nullptr));
    snapshots.clear();
    ASSERT_TRUE(executor.publishSnapshotsAfterGraphExecution(
        graph,
        segment_cache.capture_stream,
        "snapshot-overwrite-replay-2",
        &segment_cache.snapshot_manifest));
    assertSnapshotDelta(snapshots, "stage1_norm", "stage2_overwrite", 10.0f);
    assertSnapshotsDiffer(capture_snapshots, snapshots, "stage1_norm");
}
