/**
 * @file Test__PrefillGraphCacheExecutionCommon.h
 * @brief Shared GPU integration test body for bucketed prefill graph capture.
 *
 * Exercises the production ForwardExecutionEngine prefill path with a small
 * graph containing a real GPU residual-add kernel. Padded-bucket cases append a
 * HiddenStateRowSelectStage so replay-param updates are exercised across real
 * lengths. The graph is intentionally tiny so the test isolates cache lifecycle
 * behavior while still using DeviceGraphExecutor, backend streams, and HIP/CUDA
 * graph capture/replay.
 * Backend-specific wrapper files provide the registration/support/device hooks.
 */

#pragma once

#include <gtest/gtest.h>

#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/compute_stages/stages/HiddenStateRowSelectStage.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/engine/ForwardExecutionEngine.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace llaminar2;

namespace
{
    constexpr int kExactBucketSeqLen = 64;
    constexpr int kLargeBucketSeqLen = 128;
    constexpr int kHiddenDim = 64;
    constexpr int kKVProbeHeadDim = 32;
    constexpr int kKVProbeHeads = 1;
    constexpr int kKVProbeDim = kKVProbeHeads * kKVProbeHeadDim;
    constexpr int kPadTokenId = 0;

    /**
     * @brief Scoped environment override that reloads debugEnv() immediately.
     *
     * DebugEnv caches environment values, so tests that toggle graph-capture
     * gates must reload after setting variables and again after restoring them.
     */
    class ScopedDebugEnv
    {
    public:
        explicit ScopedDebugEnv(std::initializer_list<std::pair<const char *, const char *>> values)
        {
            for (const auto &[name, value] : values)
            {
                Entry entry;
                entry.name = name;
                if (const char *old_value = std::getenv(name))
                {
                    entry.had_value = true;
                    entry.old_value = old_value;
                }
                entries_.push_back(entry);
                ::setenv(name, value, 1);
            }
            mutableDebugEnv().reload();
        }

        ~ScopedDebugEnv()
        {
            for (const auto &entry : entries_)
            {
                if (entry.had_value)
                    ::setenv(entry.name.c_str(), entry.old_value.c_str(), 1);
                else
                    ::unsetenv(entry.name.c_str());
            }
            mutableDebugEnv().reload();
        }

        ScopedDebugEnv(const ScopedDebugEnv &) = delete;
        ScopedDebugEnv &operator=(const ScopedDebugEnv &) = delete;

    private:
        struct Entry
        {
            std::string name;
            bool had_value = false;
            std::string old_value;
        };

        std::vector<Entry> entries_;
    };

    /**
     * @brief Capturable one-kernel GPU stage used by the prefill cache test.
     *
     * The stage opts out of executor-managed coherence because this test does
     * not use a BufferArena. Instead, it performs the minimal tensor uploads and
     * output state transitions needed for graph capture. The actual work is the
     * backend residual-add kernel, so HIP/CUDA graph capture records real GPU
     * nodes rather than a mock callback.
     */
    class GPUResidualAddProbeStage final : public IComputeStage
    {
    public:
        GPUResidualAddProbeStage(
            std::string name,
            DeviceId device,
            FP32Tensor *input,
            FP32Tensor *residual,
            FP32Tensor *output,
            int rows,
            int cols)
            : IComputeStage(device), name_(std::move(name)), input_(input), residual_(residual),
              output_(output), rows_(rows), cols_(cols)
        {
        }

        bool execute(IDeviceContext *ctx) override
        {
            if (!ctx || !ctx->isGPU() || ctx->deviceId() != device())
            {
                LOG_ERROR("[GPUResidualAddProbeStage] Invalid GPU context");
                return false;
            }
            if (!input_ || !residual_ || !output_)
            {
                LOG_ERROR("[GPUResidualAddProbeStage] Missing tensor pointer");
                return false;
            }

            // Warmup uploads happen before capture. During capture, these calls
            // only verify the already-resident device buffers and avoid syncs.
            if (!input_->ensureOnDevice(device(), gpuStream()) ||
                !residual_->ensureOnDevice(device(), gpuStream()) ||
                !output_->allocateOnDevice(device(), gpuStream()))
            {
                LOG_ERROR("[GPUResidualAddProbeStage] Failed to prepare GPU tensors");
                return false;
            }

            ITensorResidualAdd *kernel = nullptr;
            try
            {
                kernel = llaminar::v2::kernels::KernelFactory::getOrCreateResidualAdd(input_, device());
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[GPUResidualAddProbeStage] Kernel creation failed: " << e.what());
                return false;
            }
            if (!kernel)
            {
                LOG_ERROR("[GPUResidualAddProbeStage] KernelFactory returned null residual-add kernel");
                return false;
            }

            kernel->setGPUStream(gpuStream());
            const size_t num_elements = static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
            const bool ok = kernel->apply_tensor(
                input_, residual_, output_, num_elements, nullptr, device().toKernelDeviceIndex());
            if (!ok)
                return false;

            ++execute_count_;
            output_->transitionToWithEvent(
                TensorCoherenceState::DEVICE_AUTHORITATIVE, device(), gpuStream());
            return true;
        }

        ComputeStageType type() const override { return ComputeStageType::ADD_RESIDUAL; }
        std::string name() const override { return name_; }

        bool supportsBackend(ComputeBackendType backend) const override
        {
            return backend == ComputeBackendType::GPU_CUDA ||
                   backend == ComputeBackendType::GPU_ROCM;
        }

        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
        bool isGraphCapturable() const override { return true; }
        bool needsOnGraphReplayed() const override { return true; }

        void onGraphReplayed() override
        {
            ++replay_callback_count_;
            if (output_)
            {
                output_->transitionToWithEvent(
                    TensorCoherenceState::DEVICE_AUTHORITATIVE, device(), gpuStream());
            }
        }

        size_t estimatedFlops() const override
        {
            return static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
        }

        size_t estimatedMemoryBytes() const override
        {
            return 3 * static_cast<size_t>(rows_) * static_cast<size_t>(cols_) * sizeof(float);
        }

        int executeCount() const { return execute_count_; }
        int replayCallbackCount() const { return replay_callback_count_; }

    private:
        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo info;
            info.addInput("input", input_, rows_, cols_);
            info.addInput("residual", residual_, rows_, cols_);
            info.addOutput("output", output_, rows_, cols_);
            return info;
        }

        std::string name_;
        FP32Tensor *input_ = nullptr;
        FP32Tensor *residual_ = nullptr;
        FP32Tensor *output_ = nullptr;
        int rows_ = 0;
        int cols_ = 0;
        int execute_count_ = 0;
        int replay_callback_count_ = 0;
    };

    /**
     * @brief Row-select probe that keeps real kernels but disables arena coherence.
     *
     * The shared graph-cache fixture does not allocate a BufferArena; tensors are
     * managed directly by the synthetic stages. This subclass preserves the
     * production HiddenStateRowSelectStage replay-param behavior while matching
     * the fixture's manual-coherence contract.
     */
    class GPUHiddenStateRowSelectProbeStage final : public HiddenStateRowSelectStage
    {
    public:
        explicit GPUHiddenStateRowSelectProbeStage(HiddenStateRowSelectStage::Params params)
            : HiddenStateRowSelectStage(std::move(params)) {}

        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
    };

    /**
     * @brief KV-cache append probe that keeps production replay-param behavior.
     *
     * The synthetic graph owns tensors directly rather than through BufferArena,
     * so this subclass disables executor coherence while still using the real
     * backend KV cache append kernels and host-side replay callback contract.
     */
    class GPUKVCacheAppendProbeStage final : public KVCacheAppendStage
    {
    public:
        explicit GPUKVCacheAppendProbeStage(KVCacheAppendStage::Params params)
            : KVCacheAppendStage(std::move(params)) {}

        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
    };

    /**
     * @brief Minimal ForwardExecutionEngine host that builds one GPU graph.
     */
    class PrefillGraphCacheTestHost final : public IForwardExecutionHost
    {
    public:
        PrefillGraphCacheTestHost(DeviceId device, IDeviceContext *ctx)
            : device_(device), ctx_(ctx)
        {
        }

        GraphBuildResult buildForwardGraph(const ForwardInput &input) override
        {
            ++build_calls;
            last_build_seq_len = input.seq_len;
            last_build_bucket_seq_len = input.bucket_seq_len;
            last_build_real_seq_len = input.real_seq_len;
            output_tensor_ = nullptr;
            stage_ = nullptr;
            row_select_stage_ = nullptr;
            kv_append_stage_ = nullptr;

            if (input.device != device_ || input.seq_len <= 0)
                return GraphBuildResult("invalid input for GPU prefill graph cache test");

            if (use_kv_append_probe_ && !kv_cache_)
            {
                try
                {
                    llaminar::v2::kernels::KVCacheConfig config;
                    config.precision = ActivationPrecision::FP32;
                    config.device = device_;
                    config.num_layers = 1;
                    config.batch_size = 1;
                    config.max_seq_len = 512;
                    config.n_kv_heads = kKVProbeHeads;
                    config.head_dim = kKVProbeHeadDim;
                    kv_cache_ = llaminar::v2::kernels::KernelFactory::createKVCache(config);
                }
                catch (const std::exception &e)
                {
                    return GraphBuildResult(std::string("failed to create GPU KV cache probe: ") + e.what());
                }
                if (!kv_cache_)
                    return GraphBuildResult("failed to create GPU KV cache probe");
            }

            auto input_tensor = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(input.seq_len), static_cast<size_t>(kHiddenDim)},
                DeviceId::cpu());
            auto residual_tensor = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(input.seq_len), static_cast<size_t>(kHiddenDim)},
                DeviceId::cpu());
            auto output_tensor = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(input.seq_len), static_cast<size_t>(kHiddenDim)},
                DeviceId::cpu());

            const size_t count = static_cast<size_t>(input.seq_len) * static_cast<size_t>(kHiddenDim);
            for (size_t i = 0; i < count; ++i)
            {
                input_tensor->mutable_data()[i] = 1.0f + static_cast<float>(i % 17) * 0.125f;
                residual_tensor->mutable_data()[i] = 0.25f + static_cast<float>(i % 13) * 0.0625f;
            }

            FP32Tensor *input_ptr = input_tensor.get();
            FP32Tensor *residual_ptr = residual_tensor.get();
            FP32Tensor *residual_output_ptr = output_tensor.get();
            tensors_.push_back(std::move(input_tensor));
            tensors_.push_back(std::move(residual_tensor));
            tensors_.push_back(std::move(output_tensor));

            auto stage = std::make_unique<GPUResidualAddProbeStage>(
                "gpu_residual_add_probe",
                device_,
                input_ptr,
                residual_ptr,
                residual_output_ptr,
                input.seq_len,
                kHiddenDim);
            stage_ = stage.get();

            ComputeGraph graph;
            graph.addNode("gpu_residual_add_probe", std::move(stage), device_);

            if (use_kv_append_probe_)
            {
                auto k_tensor = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(input.seq_len), static_cast<size_t>(kKVProbeDim)},
                    DeviceId::cpu());
                auto v_tensor = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(input.seq_len), static_cast<size_t>(kKVProbeDim)},
                    DeviceId::cpu());

                const int real_seq_len = input.real_seq_len > 0 ? input.real_seq_len : input.seq_len;
                for (int row = 0; row < input.seq_len; ++row)
                {
                    const bool hostile_pad = row >= real_seq_len;
                    for (int col = 0; col < kKVProbeDim; ++col)
                    {
                        const size_t idx = static_cast<size_t>(row) * kKVProbeDim + col;
                        k_tensor->mutable_data()[idx] = hostile_pad
                                                            ? 11.0f + static_cast<float>(col) * 0.125f
                                                            : 0.01f * static_cast<float>((row + col) % 19 + 1);
                        v_tensor->mutable_data()[idx] = hostile_pad
                                                            ? 29.0f + static_cast<float>(row - real_seq_len) * 3.0f
                                                            : 0.02f * static_cast<float>((row * 3 + col) % 23 + 1);
                    }
                }

                FP32Tensor *k_ptr = k_tensor.get();
                FP32Tensor *v_ptr = v_tensor.get();
                tensors_.push_back(std::move(k_tensor));
                tensors_.push_back(std::move(v_tensor));

                KVCacheAppendStage::Params kv_params;
                kv_params.K = k_ptr;
                kv_params.V = v_ptr;
                kv_params.kv_cache = kv_cache_.get();
                kv_params.layer_idx = 0;
                kv_params.seq_idx = 0;
                kv_params.num_tokens = input.seq_len;
                kv_params.seq_len = input.seq_len;
                kv_params.batch_size = 1;
                kv_params.head_dim = kKVProbeHeadDim;
                kv_params.device_id = device_;

                auto kv_stage = std::make_unique<GPUKVCacheAppendProbeStage>(kv_params);
                kv_append_stage_ = kv_stage.get();
                graph.addNode("kv_append_probe", std::move(kv_stage), device_);
                graph.addDependency("kv_append_probe", "gpu_residual_add_probe");
            }

            if (use_row_select_probe_)
            {
                auto selected_row_tensor = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{1, static_cast<size_t>(kHiddenDim)},
                    DeviceId::cpu());
                output_tensor_ = selected_row_tensor.get();

                const int initial_real_seq_len = input.real_seq_len > 0 ? input.real_seq_len : input.seq_len;
                HiddenStateRowSelectStage::Params row_params;
                row_params.input = residual_output_ptr;
                row_params.output = output_tensor_;
                row_params.seq_len = input.seq_len;
                row_params.d_model = kHiddenDim;
                row_params.selected_row_idx = initial_real_seq_len - 1;
                row_params.device_id = device_;

                auto row_select_stage = std::make_unique<GPUHiddenStateRowSelectProbeStage>(row_params);
                row_select_stage_ = row_select_stage.get();
                tensors_.push_back(std::move(selected_row_tensor));
                graph.addNode("hidden_state_row_select", std::move(row_select_stage), device_);
                graph.addDependency("hidden_state_row_select", "gpu_residual_add_probe");
            }
            else
            {
                output_tensor_ = residual_output_ptr;
            }

            ForwardOutput output;
            output.logits = output_tensor_;
            output.hidden = output_tensor_;
            return GraphBuildResult(std::move(graph), output);
        }

        IDeviceContext *getDeviceContext(DeviceId device) override
        {
            ++get_context_calls;
            return device == device_ ? ctx_ : nullptr;
        }

        std::unordered_map<DeviceId, IDeviceContext *> getPipelineDeviceContexts() override
        {
            return {{device_, ctx_}};
        }

        bool ensureDeviceWorkspaceAllocated(const ComputeGraph &) override
        {
            ++ensure_workspace_calls;
            return true;
        }

        void syncLogitsAtBoundary(IDeviceContext *ctx) override
        {
            ++sync_logits_calls;
            if (ctx)
                ctx->synchronize();
        }

        TensorBase *logitsTensor() override { return output_tensor_; }

        DeviceGraphExecutor::DecodeCapturePolicy buildDecodeCapturePolicy(
            bool, IDeviceContext *, int) const override
        {
            return {};
        }

        PPCopyInfo resolvePPCopyInfo(const ForwardInput &) const override { return {}; }

        /// @brief Enable the row-select replay-param consumer for padded bucket tests.
        void setUseRowSelectProbe(bool enabled) { use_row_select_probe_ = enabled; }

        /// @brief Enable the real GPU KV append replay-param consumer.
        void setUseKVAppendProbe(bool enabled) { use_kv_append_probe_ = enabled; }

        /// @brief Return the tensor exposed as logits/hidden by the synthetic graph.
        FP32Tensor *outputTensor() const { return output_tensor_; }

        /// @brief Return the residual probe stage built for the cached graph.
        GPUResidualAddProbeStage *stage() const { return stage_; }

        /// @brief Return the optional row-select stage built for padded bucket tests.
        HiddenStateRowSelectStage *rowSelectStage() const { return row_select_stage_; }

        /// @brief Return the optional KV append stage built for padded bucket tests.
        KVCacheAppendStage *kvAppendStage() const { return kv_append_stage_; }

        /// @brief Return the logical cached-token count for the probe KV cache.
        int kvCachedTokensForTesting() const
        {
            return kv_cache_ ? kv_cache_->get_cached_tokens(0, 0) : 0;
        }

        int build_calls = 0;
        int get_context_calls = 0;
        int ensure_workspace_calls = 0;
        int sync_logits_calls = 0;
        int last_build_seq_len = 0;
        int last_build_bucket_seq_len = 0;
        int last_build_real_seq_len = 0;

    private:
        DeviceId device_;
        IDeviceContext *ctx_ = nullptr;
        bool use_row_select_probe_ = false; ///< Whether to append HiddenStateRowSelectStage after residual add.
        bool use_kv_append_probe_ = false;  ///< Whether to append real GPU KVCacheAppendStage after residual add.
        std::vector<std::unique_ptr<FP32Tensor>> tensors_;
        std::unique_ptr<IKVCache> kv_cache_;
        FP32Tensor *output_tensor_ = nullptr;
        GPUResidualAddProbeStage *stage_ = nullptr;
        HiddenStateRowSelectStage *row_select_stage_ = nullptr;
        KVCacheAppendStage *kv_append_stage_ = nullptr;
    };

    ForwardGraphSignature bucketedPrefillSignature(DeviceId device, int seq_len)
    {
        ForwardGraphSignature signature;
        signature.seq_len = seq_len;
        signature.batch_size = 1;
        signature.device = device;
        signature.decode = false;
        signature.standard_path = true;
        signature.pp_stage_enabled = false;
        signature.pp_first_layer = -1;
        signature.pp_last_layer = -1;
        signature.pp_has_embedding = false;
        signature.pp_has_lm_head = false;
        signature.is_bucketed_prefill = true;
        signature.bucket_seq_len = seq_len;
        return signature;
    }

    PrefillGraphCacheKey prefillGraphKey(DeviceId device, int seq_len)
    {
        PrefillGraphCacheKey key;
        key.seq_len = seq_len;
        key.device_id = device;
        return key;
    }

    std::vector<int> makeSequentialInts(int count, int base)
    {
        std::vector<int> values(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i)
            values[static_cast<size_t>(i)] = base + i;
        return values;
    }

    /// @brief Build absolute position IDs for a raw server-style execute() input.
    std::vector<int> makeSequentialPositions(int count, int offset)
    {
        return makeSequentialInts(count, offset);
    }

    /// @brief Return the deterministic residual-add result for a flattened bucket index.
    float expectedProbeValueAtIndex(size_t index)
    {
        const float input = 1.0f + static_cast<float>(index % 17) * 0.125f;
        const float residual = 0.25f + static_cast<float>(index % 13) * 0.0625f;
        return input + residual;
    }

    /// @brief Verify the full bucket residual-add output for exact-bucket tests.
    void expectProbeOutputMatches(PrefillGraphCacheTestHost &host, int seq_len)
    {
        auto *output = host.outputTensor();
        ASSERT_NE(output, nullptr);
        const float *data = output->data();
        ASSERT_NE(data, nullptr);

        const size_t count = static_cast<size_t>(seq_len) * static_cast<size_t>(kHiddenDim);
        for (size_t i = 0; i < std::min<size_t>(count, 256); ++i)
        {
            EXPECT_NEAR(data[i], expectedProbeValueAtIndex(i), 1e-5f) << "Mismatch at output index " << i;
        }
    }

    /// @brief Verify that row-select copied the last real bucket row into the one-row output.
    void expectSelectedProbeOutputMatches(PrefillGraphCacheTestHost &host, int real_seq_len)
    {
        ASSERT_GT(real_seq_len, 0);
        auto *output = host.outputTensor();
        ASSERT_NE(output, nullptr);
        const float *data = output->data();
        ASSERT_NE(data, nullptr);

        const size_t source_offset = static_cast<size_t>(real_seq_len - 1) * static_cast<size_t>(kHiddenDim);
        for (size_t col = 0; col < static_cast<size_t>(kHiddenDim); ++col)
        {
            EXPECT_NEAR(data[col], expectedProbeValueAtIndex(source_offset + col), 1e-5f)
                << "Mismatch at selected-row column " << col << " for real_seq_len=" << real_seq_len;
        }
    }

    class PrefillGraphCacheExecutionTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ensurePrefillGraphCacheBackendRegistered();
            if (!hasPrefillGraphCacheBackendSupport())
                GTEST_SKIP() << prefillGraphCacheBackendSkipMessage();

            device_ = prefillGraphCacheBackendDeviceId();
            device_ctx_ = IDeviceContext::create(device_, 1);
            ASSERT_NE(device_ctx_, nullptr);

            GraphExecutorConfig executor_config;
            executor_config.enable_validation = false;
            executor_ = std::make_unique<DeviceGraphExecutor>(executor_config);

            ForwardExecutionEngine::Config engine_config;
            engine_config.cache_config.enabled = true;
            engine_config.has_unified_pp = false;
            engine_ = std::make_unique<ForwardExecutionEngine>(std::move(engine_config), *executor_);
            host_ = std::make_unique<PrefillGraphCacheTestHost>(device_, device_ctx_.get());
        }

        void TearDown() override
        {
            host_.reset();
            engine_.reset();
            executor_.reset();
            device_ctx_.reset();
        }

        DeviceId device_ = DeviceId::cpu();
        std::unique_ptr<IDeviceContext> device_ctx_;
        std::unique_ptr<DeviceGraphExecutor> executor_;
        std::unique_ptr<ForwardExecutionEngine> engine_;
        std::unique_ptr<PrefillGraphCacheTestHost> host_;
    };

    TEST_F(PrefillGraphCacheExecutionTest, ExactBucketWarmupCaptureReplayLifecycle)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
            {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
        });

        auto tokens = makeSequentialInts(kExactBucketSeqLen, 1000);
        ForwardInput base_input;
        base_input.token_ids = tokens.data();
        base_input.batch_size = 1;
        base_input.seq_len = kExactBucketSeqLen;
        base_input.device = device_;

        const auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            base_input,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/false);
        ASSERT_TRUE(plan) << plan.error;
        ASSERT_FALSE(plan.padding_required);
        ASSERT_EQ(plan.chunk.bucket_seq_len, kExactBucketSeqLen);

        ForwardOutput output;
        const auto signature = bucketedPrefillSignature(device_, kExactBucketSeqLen);
        const auto key = prefillGraphKey(device_, kExactBucketSeqLen);

        ASSERT_TRUE(engine_->runPrefillChunk(base_input, plan, output, *host_));
        EXPECT_EQ(host_->build_calls, 1);
        EXPECT_EQ(host_->last_build_seq_len, kExactBucketSeqLen);
        EXPECT_EQ(host_->last_build_real_seq_len, kExactBucketSeqLen);
        EXPECT_EQ(host_->last_build_bucket_seq_len, kExactBucketSeqLen);
        expectProbeOutputMatches(*host_, kExactBucketSeqLen);

        auto after_build = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_build.has_value());
        EXPECT_TRUE(after_build->forward_cache_valid);
        EXPECT_FALSE(after_build->prefill_cache_initialized)
            << "Current engine builds the reusable forward graph on the first request; prefill cache warmup starts on the first cache hit.";

        ASSERT_TRUE(engine_->runPrefillChunk(base_input, plan, output, *host_));
        EXPECT_EQ(host_->build_calls, 1) << "Same exact bucket should reuse the cached forward graph";
        expectProbeOutputMatches(*host_, kExactBucketSeqLen);

        auto after_warmup = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_warmup.has_value());
        ASSERT_TRUE(after_warmup->prefill_cache_initialized);
        EXPECT_EQ(after_warmup->phase, PrefillGraphPhase::Warmup);
        EXPECT_EQ(after_warmup->warmup_count, 1u);
        EXPECT_EQ(after_warmup->capture_count, 0u);
        EXPECT_EQ(after_warmup->replay_count, 0);

        ASSERT_TRUE(engine_->runPrefillChunk(base_input, plan, output, *host_));
        EXPECT_EQ(host_->build_calls, 1);
        expectProbeOutputMatches(*host_, kExactBucketSeqLen);

        auto after_capture = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_capture.has_value());
        EXPECT_EQ(after_capture->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_capture->warmup_count, 1u);
        EXPECT_EQ(after_capture->capture_count, 1u);
        EXPECT_EQ(after_capture->replay_count, 1)
            << "Capture path launches the newly instantiated graph once so this request produces output.";
        EXPECT_GT(after_capture->node_count, 0u)
            << "The probe stage must record real GPU graph nodes, not an empty/mock capture.";

        ASSERT_TRUE(engine_->runPrefillChunk(base_input, plan, output, *host_));
        EXPECT_EQ(host_->build_calls, 1);
        expectProbeOutputMatches(*host_, kExactBucketSeqLen);

        auto after_replay = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_replay.has_value());
        EXPECT_EQ(after_replay->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_replay->warmup_count, 1u);
        EXPECT_EQ(after_replay->capture_count, 1u);
        EXPECT_EQ(after_replay->replay_count, 2);
        EXPECT_EQ(after_replay->eviction_count, 0u);

        ASSERT_NE(host_->stage(), nullptr);
        EXPECT_GE(host_->stage()->executeCount(), 3)
            << "Normal build, warmup, and capture recording execute the stage directly.";
        EXPECT_GE(host_->stage()->replayCallbackCount(), 2)
            << "Capture launch and Ready replay both run post-graph callbacks.";
    }

    TEST_F(PrefillGraphCacheExecutionTest, PaddedSafeBucketReplaysAcrossDifferentRealLengths)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
            {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
        });

        host_->setUseRowSelectProbe(true);
        host_->setUseKVAppendProbe(true);

        auto tokens61 = makeSequentialInts(kExactBucketSeqLen - 3, 3000);
        ForwardInput input61;
        input61.token_ids = tokens61.data();
        input61.batch_size = 1;
        input61.seq_len = kExactBucketSeqLen - 3;
        input61.token_offset = 128;
        input61.device = device_;

        auto tokens63 = makeSequentialInts(kExactBucketSeqLen - 1, 4000);
        ForwardInput input63;
        input63.token_ids = tokens63.data();
        input63.batch_size = 1;
        input63.seq_len = kExactBucketSeqLen - 1;
        input63.token_offset = 512;
        input63.device = device_;

        const auto plan61 = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            input61,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/true);
        ASSERT_TRUE(plan61) << plan61.error;
        ASSERT_TRUE(plan61.padding_required);
        ASSERT_EQ(plan61.chunk.bucket_seq_len, kExactBucketSeqLen);

        const auto plan63 = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            input63,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/true);
        ASSERT_TRUE(plan63) << plan63.error;
        ASSERT_TRUE(plan63.padding_required);
        ASSERT_EQ(plan63.chunk.bucket_seq_len, kExactBucketSeqLen);

        ForwardOutput output;
        const auto signature = bucketedPrefillSignature(device_, kExactBucketSeqLen);
        const auto key = prefillGraphKey(device_, kExactBucketSeqLen);

        ASSERT_TRUE(engine_->runPrefillChunk(input61, plan61, output, *host_));
        EXPECT_EQ(host_->build_calls, 1);
        EXPECT_EQ(host_->last_build_seq_len, kExactBucketSeqLen);
        EXPECT_EQ(host_->last_build_real_seq_len, kExactBucketSeqLen - 3);
        EXPECT_EQ(host_->last_build_bucket_seq_len, kExactBucketSeqLen);
        ASSERT_NE(host_->rowSelectStage(), nullptr);
        ASSERT_NE(host_->kvAppendStage(), nullptr);
        EXPECT_EQ(host_->rowSelectStage()->selectedRowForTesting(), kExactBucketSeqLen - 4);
        EXPECT_EQ(host_->kvCachedTokensForTesting(), kExactBucketSeqLen - 3)
            << "First padded cache miss must append only real prompt tokens.";
        expectSelectedProbeOutputMatches(*host_, kExactBucketSeqLen - 3);

        auto after_build = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_build.has_value());
        EXPECT_TRUE(after_build->forward_cache_valid);
        EXPECT_FALSE(after_build->prefill_cache_initialized);

        ASSERT_TRUE(engine_->runPrefillChunk(input63, plan63, output, *host_));
        EXPECT_EQ(host_->build_calls, 1) << "Same padded bucket should reuse the cached forward graph";
        EXPECT_EQ(host_->rowSelectStage()->selectedRowForTesting(), kExactBucketSeqLen - 2);
        EXPECT_EQ(host_->kvCachedTokensForTesting(), (kExactBucketSeqLen - 3) + (kExactBucketSeqLen - 1))
            << "Warmup execution must append by the second request's real length, not the bucket length.";
        expectSelectedProbeOutputMatches(*host_, kExactBucketSeqLen - 1);

        auto after_warmup = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_warmup.has_value());
        ASSERT_TRUE(after_warmup->prefill_cache_initialized);
        EXPECT_EQ(after_warmup->phase, PrefillGraphPhase::Warmup);
        EXPECT_EQ(after_warmup->warmup_count, 1u);
        EXPECT_EQ(after_warmup->capture_count, 0u);

        ASSERT_TRUE(engine_->runPrefillChunk(input61, plan61, output, *host_));
        EXPECT_EQ(host_->build_calls, 1);
        EXPECT_EQ(host_->rowSelectStage()->selectedRowForTesting(), kExactBucketSeqLen - 4);
        EXPECT_EQ(host_->kvCachedTokensForTesting(), (kExactBucketSeqLen - 3) + (kExactBucketSeqLen - 1) + (kExactBucketSeqLen - 3))
            << "Capture launch callback must advance KV metadata by real tokens only.";
        expectSelectedProbeOutputMatches(*host_, kExactBucketSeqLen - 3);

        auto after_capture = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_capture.has_value());
        EXPECT_EQ(after_capture->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_capture->warmup_count, 1u);
        EXPECT_EQ(after_capture->capture_count, 1u);
        EXPECT_EQ(after_capture->replay_count, 1);
        EXPECT_GT(after_capture->node_count, 0u);

        ASSERT_TRUE(engine_->runPrefillChunk(input63, plan63, output, *host_));
        EXPECT_EQ(host_->build_calls, 1);
        // The monolithic cache test proves the engine delivers updated real-length
        // metadata before Ready replay. The dedicated row-select graph tests verify
        // the backend-level selected-row device output for no-recapture replays.
        EXPECT_EQ(host_->rowSelectStage()->selectedRowForTesting(), kExactBucketSeqLen - 2);
        EXPECT_EQ(host_->kvCachedTokensForTesting(),
                  (kExactBucketSeqLen - 3) + (kExactBucketSeqLen - 1) + (kExactBucketSeqLen - 3) + (kExactBucketSeqLen - 1))
            << "Ready replay must advance KV metadata by the latest real length, not the bucket length.";

        auto after_replay = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_replay.has_value());
        EXPECT_EQ(after_replay->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_replay->warmup_count, 1u);
        EXPECT_EQ(after_replay->capture_count, 1u)
            << "Changing real_seq_len inside one bucket must update replay params, not recapture.";
        EXPECT_EQ(after_replay->replay_count, 2);
        EXPECT_EQ(after_replay->eviction_count, 0u);
    }

    TEST_F(PrefillGraphCacheExecutionTest, ServerStyleRawExecuteReusesPaddedBucketAcrossRealLengths)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
            {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
        });

        host_->setUseRowSelectProbe(true);
        host_->setUseKVAppendProbe(true);

        auto tokens61 = makeSequentialInts(kExactBucketSeqLen - 3, 5000);
        auto positions61 = makeSequentialPositions(kExactBucketSeqLen - 3, 128);
        ForwardInput input61;
        input61.token_ids = tokens61.data();
        input61.position_ids = positions61.data();
        input61.batch_size = 1;
        input61.seq_len = kExactBucketSeqLen - 3;
        input61.position_offset = 128;
        input61.device = device_;

        auto tokens63 = makeSequentialInts(kExactBucketSeqLen - 1, 6000);
        auto positions63 = makeSequentialPositions(kExactBucketSeqLen - 1, 512);
        ForwardInput input63;
        input63.token_ids = tokens63.data();
        input63.position_ids = positions63.data();
        input63.batch_size = 1;
        input63.seq_len = kExactBucketSeqLen - 1;
        input63.position_offset = 512;
        input63.device = device_;

        ForwardOutput output;
        const auto signature = bucketedPrefillSignature(device_, kExactBucketSeqLen);
        const auto key = prefillGraphKey(device_, kExactBucketSeqLen);

        ASSERT_TRUE(engine_->execute(input61, output, *host_));
        EXPECT_EQ(host_->build_calls, 1);
        EXPECT_EQ(host_->last_build_seq_len, kExactBucketSeqLen);
        EXPECT_EQ(host_->last_build_real_seq_len, kExactBucketSeqLen - 3);
        EXPECT_EQ(host_->last_build_bucket_seq_len, kExactBucketSeqLen);
        ASSERT_NE(host_->rowSelectStage(), nullptr);
        ASSERT_NE(host_->kvAppendStage(), nullptr);
        EXPECT_EQ(host_->rowSelectStage()->selectedRowForTesting(), kExactBucketSeqLen - 4);
        EXPECT_EQ(host_->kvCachedTokensForTesting(), kExactBucketSeqLen - 3)
            << "Raw execute cache miss must append only real prompt tokens.";
        expectSelectedProbeOutputMatches(*host_, kExactBucketSeqLen - 3);

        auto after_build = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_build.has_value());
        EXPECT_TRUE(after_build->forward_cache_valid);
        EXPECT_FALSE(after_build->prefill_cache_initialized);

        ASSERT_TRUE(engine_->execute(input63, output, *host_));
        EXPECT_EQ(host_->build_calls, 1)
            << "Server-style prompts in one bucket must reuse the cached forward graph.";
        EXPECT_EQ(host_->rowSelectStage()->selectedRowForTesting(), kExactBucketSeqLen - 2);
        EXPECT_EQ(host_->kvCachedTokensForTesting(), (kExactBucketSeqLen - 3) + (kExactBucketSeqLen - 1))
            << "Warmup execution must append by the second request's real length, not the bucket length.";
        expectSelectedProbeOutputMatches(*host_, kExactBucketSeqLen - 1);

        auto after_warmup = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_warmup.has_value());
        ASSERT_TRUE(after_warmup->prefill_cache_initialized);
        EXPECT_EQ(after_warmup->phase, PrefillGraphPhase::Warmup);
        EXPECT_EQ(after_warmup->warmup_count, 1u);
        EXPECT_EQ(after_warmup->capture_count, 0u);

        ASSERT_TRUE(engine_->execute(input61, output, *host_));
        EXPECT_EQ(host_->build_calls, 1);
        EXPECT_EQ(host_->rowSelectStage()->selectedRowForTesting(), kExactBucketSeqLen - 4);
        EXPECT_EQ(host_->kvCachedTokensForTesting(), (kExactBucketSeqLen - 3) + (kExactBucketSeqLen - 1) + (kExactBucketSeqLen - 3))
            << "Capture launch callback must advance KV metadata by real tokens only.";
        expectSelectedProbeOutputMatches(*host_, kExactBucketSeqLen - 3);

        auto after_capture = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_capture.has_value());
        EXPECT_EQ(after_capture->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_capture->warmup_count, 1u);
        EXPECT_EQ(after_capture->capture_count, 1u);
        EXPECT_EQ(after_capture->replay_count, 1);
        EXPECT_GT(after_capture->node_count, 0u);

        ASSERT_TRUE(engine_->execute(input63, output, *host_));
        EXPECT_EQ(host_->build_calls, 1);
        EXPECT_EQ(host_->rowSelectStage()->selectedRowForTesting(), kExactBucketSeqLen - 2);
        EXPECT_EQ(host_->kvCachedTokensForTesting(),
                  (kExactBucketSeqLen - 3) + (kExactBucketSeqLen - 1) + (kExactBucketSeqLen - 3) + (kExactBucketSeqLen - 1))
            << "Ready replay must advance KV metadata by the latest real length, not the bucket length.";

        auto after_replay = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_replay.has_value());
        EXPECT_EQ(after_replay->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_replay->warmup_count, 1u);
        EXPECT_EQ(after_replay->capture_count, 1u)
            << "Changing raw real_seq_len inside one bucket must update replay params, not recapture.";
        EXPECT_EQ(after_replay->replay_count, 2);
        EXPECT_EQ(after_replay->eviction_count, 0u);
    }

    TEST_F(PrefillGraphCacheExecutionTest, CrossBucketEvictionRecapturesEligibleBucket)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64,128"},
            {"LLAMINAR_PREFILL_GRAPH_MAX_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
            {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
        });

        auto tokens64 = makeSequentialInts(kExactBucketSeqLen, 7000);
        ForwardInput input64;
        input64.token_ids = tokens64.data();
        input64.batch_size = 1;
        input64.seq_len = kExactBucketSeqLen;
        input64.device = device_;

        auto tokens128 = makeSequentialInts(kLargeBucketSeqLen, 8000);
        ForwardInput input128;
        input128.token_ids = tokens128.data();
        input128.batch_size = 1;
        input128.seq_len = kLargeBucketSeqLen;
        input128.device = device_;

        const auto plan64 = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            input64,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/false);
        ASSERT_TRUE(plan64) << plan64.error;
        ASSERT_EQ(plan64.chunk.bucket_seq_len, kExactBucketSeqLen);

        const auto plan128 = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            input128,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/false);
        ASSERT_TRUE(plan128) << plan128.error;
        ASSERT_EQ(plan128.chunk.bucket_seq_len, kLargeBucketSeqLen);

        ForwardOutput output;
        const auto signature64 = bucketedPrefillSignature(device_, kExactBucketSeqLen);
        const auto key64 = prefillGraphKey(device_, kExactBucketSeqLen);
        const auto signature128 = bucketedPrefillSignature(device_, kLargeBucketSeqLen);
        const auto key128 = prefillGraphKey(device_, kLargeBucketSeqLen);

        ASSERT_TRUE(engine_->runPrefillChunk(input64, plan64, output, *host_));
        ASSERT_TRUE(engine_->runPrefillChunk(input64, plan64, output, *host_));
        ASSERT_TRUE(engine_->runPrefillChunk(input64, plan64, output, *host_));
        EXPECT_EQ(host_->build_calls, 1);

        auto after_64_ready = engine_->prefillGraphCacheSnapshot(signature64, key64);
        ASSERT_TRUE(after_64_ready.has_value());
        EXPECT_EQ(after_64_ready->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_64_ready->warmup_count, 1u);
        EXPECT_EQ(after_64_ready->capture_count, 1u);
        EXPECT_EQ(after_64_ready->eviction_count, 0u);

        ASSERT_TRUE(engine_->runPrefillChunk(input128, plan128, output, *host_));
        EXPECT_EQ(host_->build_calls, 2)
            << "The larger bucket should build once, then evict the older reusable bucket.";
        EXPECT_FALSE(engine_->prefillGraphCacheSnapshot(signature64, key64).has_value())
            << "A max-buckets=1 cap must evict the old top-level bucketed forward graph.";

        auto after_128_build = engine_->prefillGraphCacheSnapshot(signature128, key128);
        ASSERT_TRUE(after_128_build.has_value());
        EXPECT_TRUE(after_128_build->forward_cache_valid);
        EXPECT_FALSE(after_128_build->prefill_cache_initialized);
        EXPECT_EQ(after_128_build->eviction_count, 1u);

        ASSERT_TRUE(engine_->runPrefillChunk(input64, plan64, output, *host_));
        EXPECT_EQ(host_->build_calls, 3)
            << "Requesting the evicted bucket must rebuild its forward graph.";

        auto after_64_rebuild = engine_->prefillGraphCacheSnapshot(signature64, key64);
        ASSERT_TRUE(after_64_rebuild.has_value());
        EXPECT_TRUE(after_64_rebuild->forward_cache_valid);
        EXPECT_FALSE(after_64_rebuild->prefill_cache_initialized)
            << "The first request after eviction should be an explicit rebuild, not a hidden replay.";
        EXPECT_EQ(after_64_rebuild->eviction_count, 2u)
            << "Rebuilding bucket64 under cap=1 should evict bucket128 at the top-level cache.";

        ASSERT_TRUE(engine_->runPrefillChunk(input64, plan64, output, *host_));
        auto after_64_rewarm = engine_->prefillGraphCacheSnapshot(signature64, key64);
        ASSERT_TRUE(after_64_rewarm.has_value());
        ASSERT_TRUE(after_64_rewarm->prefill_cache_initialized);
        EXPECT_EQ(after_64_rewarm->phase, PrefillGraphPhase::Warmup)
            << "The rebuilt bucket must explicitly re-enter warmup.";
        EXPECT_EQ(after_64_rewarm->warmup_count, 1u);
        EXPECT_EQ(after_64_rewarm->capture_count, 0u);
        EXPECT_EQ(after_64_rewarm->eviction_count, 2u);

        ASSERT_TRUE(engine_->runPrefillChunk(input64, plan64, output, *host_));
        auto after_64_recapture = engine_->prefillGraphCacheSnapshot(signature64, key64);
        ASSERT_TRUE(after_64_recapture.has_value());
        EXPECT_EQ(after_64_recapture->phase, PrefillGraphPhase::Ready)
            << "The rebuilt bucket must capture again instead of staying on normal prefill.";
        EXPECT_EQ(after_64_recapture->warmup_count, 1u);
        EXPECT_EQ(after_64_recapture->capture_count, 1u);
        EXPECT_EQ(after_64_recapture->replay_count, 1);
        EXPECT_EQ(after_64_recapture->eviction_count, 2u);
        EXPECT_GT(after_64_recapture->node_count, 0u);
    }

    TEST_F(PrefillGraphCacheExecutionTest, NonExactBucketRejectedBeforeGraphBuild)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
        });

        auto tokens = makeSequentialInts(kExactBucketSeqLen - 1, 2000);
        ForwardInput base_input;
        base_input.token_ids = tokens.data();
        base_input.batch_size = 1;
        base_input.seq_len = kExactBucketSeqLen - 1;
        base_input.device = device_;

        const auto padded_plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            base_input,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/false);
        ASSERT_FALSE(padded_plan);
        EXPECT_TRUE(padded_plan.padding_required);
        EXPECT_NE(padded_plan.error.find("requires caller opt-in"), std::string::npos);

        ForwardOutput output;
        EXPECT_FALSE(engine_->runPrefillChunk(base_input, padded_plan, output, *host_));
        EXPECT_EQ(host_->build_calls, 0)
            << "Non-exact bucketed prefill without padded opt-in must reject before graph build.";

        const auto signature = bucketedPrefillSignature(device_, kExactBucketSeqLen);
        const auto key = prefillGraphKey(device_, kExactBucketSeqLen);
        EXPECT_FALSE(engine_->prefillGraphCacheSnapshot(signature, key).has_value());
    }

} // namespace