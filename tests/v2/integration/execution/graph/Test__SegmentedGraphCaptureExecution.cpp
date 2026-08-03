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
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cmath>
#include <cstring>
#include <algorithm>

#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/compute_stages/ComputeStages.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "tensors/Tensors.h"

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
     * @return true when every tensor is resident and the upload stream has
     *         completed; false on allocation, transfer, or synchronization
     *         failure.
     */
    bool prepareFixtureTensorsForGPUExecution()
    {
        if (!gpu_ctx_ || !device_ctx_)
            return false;

        void *upload_stream = gpu_ctx_->defaultStream();
        if (!upload_stream)
            return false;

        const DeviceId device = device_ctx_->deviceId();
        for (const auto &tensor : tensor_storage_)
        {
            if (!tensor || !tensor->ensureOnDevice(device, upload_stream))
                return false;
        }
        for (auto *tensor : arena_tensors_)
        {
            if (!tensor || !tensor->ensureOnDevice(device, upload_stream))
                return false;
        }

        return gpu_ctx_->synchronizeStreamChecked(upload_stream);
    }

    ComputeGraph buildNormResidualGraph(size_t seq_len, size_t d_model,
                                        FP32Tensor *&norm_input,
                                        FP32Tensor *&residual,
                                        FP32Tensor *&result_output)
    {
        const DeviceId device = device_ctx_->deviceId();
        norm_input = createArenaFP32Tensor(
            BufferId::HIDDEN_STATE, {seq_len, d_model});
        auto *norm_output = createArenaFP32Tensor(
            BufferId::NORMALIZED, {seq_len, d_model});
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
        res_params.residual = residual;
        res_params.output = result_output;
        res_params.num_elements = num_elements;
        res_params.device_id = device;
        res_params.input_buffer_id = BufferId::NORMALIZED;
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
