/**
 * @file Test__AttentionComputeStage.cpp
 * @brief Unit tests for AttentionComputeStage compute stage
 * @author David Sanftenberg
 *
 * Tests the new AttentionComputeStage which uses KernelFactory for
 * type-safe attention kernel dispatch. This is part of Phase 9 of
 * the multi-device architecture refactoring.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <memory>

#include "backends/DeviceId.h"
#include "execution/compute_stages/ComputeStages.h"
#include "v2/tensors/Tensors.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/utils/MPIContext.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "kernels/rocm/attention/ROCmFlashAttentionKernelT.h"
#endif

using namespace llaminar2;

namespace
{
    constexpr float TOLERANCE = 1e-4f;

#ifdef HAVE_ROCM
    bool hasROCm()
    {
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        return (err == hipSuccess && count > 0);
    }
#endif

    class Test__AttentionComputeStage : public ::testing::Test
    {
    protected:
        MPIContext mpi_ctx_{0, 1, MPI_COMM_WORLD};
        DeviceId device_id_ = DeviceId::cpu();

        void SetUp() override {}
        void TearDown() override {}
    };

    /**
     * @brief Minimal KV cache stub for capture-signature unit tests.
     *
     * AttentionComputeStage::graphCaptureVariantSignature() only needs cache
     * precision, layer count, and the host cached-token count.  A lightweight
     * fake keeps this regression in the unit suite without requiring CUDA/ROCm
     * devices or allocating real KV storage.
     */
    class FakeCaptureKVCache final : public IKVCache
    {
    public:
        int cached_tokens = 0;
        ActivationPrecision k_precision_value = ActivationPrecision::FP16;
        ActivationPrecision v_precision_value = ActivationPrecision::FP16;

        ActivationPrecision k_precision() const override { return k_precision_value; }
        ActivationPrecision v_precision() const override { return v_precision_value; }
        int get_cached_tokens(int, int = 0) const override { return cached_tokens; }
        int max_seq_len() const override { return 4096; }
        int n_layers() const override { return 1; }

        bool get_kv(int, int, ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr) override
        {
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = cached_tokens;
            return true;
        }

        bool get_kv(int, int, const ITensor **out_k, const ITensor **out_v,
                    int *out_kv_len = nullptr) const override
        {
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = cached_tokens;
            return true;
        }

        bool append(int, int, const ITensor *, const ITensor *, int) override
        {
            return false;
        }

        void clear() override { cached_tokens = 0; }
        void clear_sequence(int, int) override { cached_tokens = 0; }
        void clear_layer(int) override { cached_tokens = 0; }
    };

    /**
     * @brief Test that AttentionComputeStage can be constructed with valid params
     */
    TEST_F(Test__AttentionComputeStage, Construction)
    {
        TensorFactory factory(mpi_ctx_);

        // Minimal dimensions
        int seq_len = 4;
        int n_heads = 2;
        int n_kv_heads = 2;
        int head_dim = 8;
        size_t hidden_size = n_heads * head_dim;

        // Create tensors
        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        // Construct stage
        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = true;
        params.mpi_ctx = &mpi_ctx_;
        params.device_id = device_id_;

        auto stage = std::make_unique<AttentionComputeStage>(params);
        ASSERT_NE(stage, nullptr);
        EXPECT_EQ(stage->type(), ComputeStageType::ATTENTION);
    }

    /**
     * @brief Test supportsBackend for CPU backends
     */
    TEST_F(Test__AttentionComputeStage, SupportsBackend_CPU)
    {
        TensorFactory factory(mpi_ctx_);

        int seq_len = 4;
        int n_heads = 2;
        int n_kv_heads = 2;
        int head_dim = 8;
        size_t hidden_size = n_heads * head_dim;

        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;

        auto stage = std::make_unique<AttentionComputeStage>(params);

        // Should support CPU backends
        EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::CPU));
        EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::CPU));

        // Unknown backends should not be supported
        EXPECT_FALSE(stage->supportsBackend(static_cast<ComputeBackendType>(999)));
    }

    /**
     * @brief Multirow GPU verifier capture signatures bucket KV launch regimes.
     *
     * Qwen3.6 MTP verifier attention reads row-local KV length from dynamic
     * device params before every launch.  Exact token positions are therefore
     * not capture variants.  CUDA and ROCm should recapture only when the
     * launch regime bucket can change, avoiding warmup-only verifier graphs on
     * adjacent decode steps.
     */
    TEST_F(Test__AttentionComputeStage, MultirowGpuCaptureSignatureBucketsCachedTokens)
    {
        FakeCaptureKVCache kv_cache;
        kv_cache.cached_tokens = 597;

        auto make_params = [&](DeviceId device)
        {
            AttentionComputeStage::Params params;
            params.device_id = device;
            params.kv_cache = &kv_cache;
            params.layer_idx = 0;
            params.batch_size = 1;
            params.seq_len = 2;
            params.kv_len = kv_cache.cached_tokens;
            params.n_heads = 16;
            params.n_kv_heads = 2;
            params.head_dim = 256;
            params.causal = true;
            params.auto_detect_mode = true;
            params.apply_rope_to_k = true;
            params.rope_theta = 10000000.0f;
            params.partial_rotary_factor = 0.25f;
            return params;
        };

        for (DeviceId device : {DeviceId::cuda(0), DeviceId::rocm(0)})
        {
            AttentionComputeStage stage_before(make_params(device));
            const uint64_t before = stage_before.graphCaptureVariantSignature();
            EXPECT_NE(before, 0u) << device.toString();

            kv_cache.cached_tokens = 598;
            AttentionComputeStage stage_after(make_params(device));
            const uint64_t after = stage_after.graphCaptureVariantSignature();
            EXPECT_NE(after, 0u) << device.toString();
            EXPECT_EQ(before, after)
                << "captured multirow verifier attention should not recapture "
                   "for adjacent token positions in the same launch bucket on "
                << device.toString();

            kv_cache.cached_tokens = 597;
        }
    }

    /**
     * @brief Multirow verifier attention still recaptures at launch bucket boundaries.
     *
     * The capture key intentionally avoids per-token churn, but it must retain
     * real topology changes.  These cached-token choices straddle the first
     * CUDA split bucket and ROCm requested-split-cap bucket respectively.
     */
    TEST_F(Test__AttentionComputeStage, MultirowGpuCaptureSignatureChangesAtLaunchBucketBoundary)
    {
        FakeCaptureKVCache kv_cache;

        auto make_params = [&](DeviceId device)
        {
            AttentionComputeStage::Params params;
            params.device_id = device;
            params.kv_cache = &kv_cache;
            params.layer_idx = 0;
            params.batch_size = 1;
            params.seq_len = 2;
            params.kv_len = kv_cache.cached_tokens;
            params.n_heads = 16;
            params.n_kv_heads = 2;
            params.head_dim = 256;
            params.causal = true;
            params.auto_detect_mode = true;
            params.apply_rope_to_k = true;
            params.rope_theta = 10000000.0f;
            params.partial_rotary_factor = 0.25f;
            return params;
        };

        kv_cache.cached_tokens = 29;
        AttentionComputeStage cuda_before(make_params(DeviceId::cuda(0)));
        const uint64_t cuda_sig_before = cuda_before.graphCaptureVariantSignature();
        ASSERT_NE(cuda_sig_before, 0u);
        kv_cache.cached_tokens = 30;
        AttentionComputeStage cuda_after(make_params(DeviceId::cuda(0)));
        EXPECT_NE(cuda_sig_before, cuda_after.graphCaptureVariantSignature())
            << "CUDA small-M verifier attention must recapture when row split bucket changes.";

        kv_cache.cached_tokens = 62;
        AttentionComputeStage rocm_before(make_params(DeviceId::rocm(0)));
        const uint64_t rocm_sig_before = rocm_before.graphCaptureVariantSignature();
        ASSERT_NE(rocm_sig_before, 0u);
        kv_cache.cached_tokens = 63;
        AttentionComputeStage rocm_after(make_params(DeviceId::rocm(0)));
        EXPECT_NE(rocm_sig_before, rocm_after.graphCaptureVariantSignature())
            << "ROCm small-M verifier attention must recapture when requested split cap changes.";
    }

    /**
     * @brief Long-context CUDA verifier signatures stop changing after split cap.
     *
     * CUDA small-M verifier attention caps the flash-decode split count by SM
     * occupancy and MAX_NUM_SPLITS.  The capture signature must mirror that
     * real launch split count rather than raw KV length; otherwise long-context
     * MTP decode recaptures every sixteen tokens even though the kernel grid is
     * unchanged.
     */
    TEST_F(Test__AttentionComputeStage, CUDAMultirowCaptureSignatureIgnoresFalseLongContextSplitBoundary)
    {
        FakeCaptureKVCache kv_cache;

        auto make_params = [&]()
        {
            AttentionComputeStage::Params params;
            params.device_id = DeviceId::cuda(0);
            params.kv_cache = &kv_cache;
            params.layer_idx = 0;
            params.batch_size = 1;
            params.seq_len = 2;
            params.kv_len = kv_cache.cached_tokens;
            params.n_heads = 16;
            params.n_kv_heads = 2;
            params.head_dim = 256;
            params.causal = true;
            params.auto_detect_mode = true;
            params.apply_rope_to_k = true;
            params.rope_theta = 10000000.0f;
            params.partial_rotary_factor = 0.25f;
            return params;
        };

        kv_cache.cached_tokens = 621; // post-append top row kv_len=623.
        AttentionComputeStage before(make_params());
        const uint64_t before_sig = before.graphCaptureVariantSignature();
        ASSERT_NE(before_sig, 0u);

        kv_cache.cached_tokens = 622; // crosses raw kv_len/16, but not real split cap.
        AttentionComputeStage after(make_params());
        EXPECT_EQ(before_sig, after.graphCaptureVariantSignature())
            << "CUDA capture signature should remain stable after the real "
               "small-M decode split count is capped";
    }

    /**
     * @brief One-row decode does not recapture on every token position.
     *
     * The multirow verifier needs exact KV-length keys because its RoPE-on-read
     * conversion records a variable-size operation.  Normal one-row decode is
     * deliberately more stable: CUDA keeps no attention capture variant, while
     * ROCm keys only by split-K launch bucket.  This catches accidental
     * per-token recapture in the hot decode path.
     */
    TEST_F(Test__AttentionComputeStage, SingleRowGpuCaptureSignatureAvoidsExactTokenCount)
    {
        FakeCaptureKVCache kv_cache;
        kv_cache.cached_tokens = 597;

        auto make_params = [&](DeviceId device)
        {
            AttentionComputeStage::Params params;
            params.device_id = device;
            params.kv_cache = &kv_cache;
            params.layer_idx = 0;
            params.batch_size = 1;
            params.seq_len = 1;
            params.kv_len = kv_cache.cached_tokens;
            params.n_heads = 16;
            params.n_kv_heads = 2;
            params.head_dim = 256;
            params.causal = true;
            params.auto_detect_mode = true;
            params.apply_rope_to_k = true;
            params.rope_theta = 10000000.0f;
            params.partial_rotary_factor = 0.25f;
            return params;
        };

        AttentionComputeStage cuda_before(make_params(DeviceId::cuda(0)));
        EXPECT_EQ(cuda_before.graphCaptureVariantSignature(), 0u);

        AttentionComputeStage rocm_before(make_params(DeviceId::rocm(0)));
        const uint64_t rocm_sig_before = rocm_before.graphCaptureVariantSignature();
        EXPECT_NE(rocm_sig_before, 0u);

        kv_cache.cached_tokens = 598;

        AttentionComputeStage cuda_after(make_params(DeviceId::cuda(0)));
        EXPECT_EQ(cuda_after.graphCaptureVariantSignature(), 0u);

        AttentionComputeStage rocm_after(make_params(DeviceId::rocm(0)));
        const uint64_t rocm_sig_after = rocm_after.graphCaptureVariantSignature();
        EXPECT_EQ(rocm_sig_before, rocm_sig_after)
            << "ROCm one-row decode should remain bucketed by launch topology "
               "rather than recapturing for every exact token count";
    }

    /**
     * @brief Test getDumpInfo returns valid scalar info
     */
    TEST_F(Test__AttentionComputeStage, GetDumpInfo)
    {
        TensorFactory factory(mpi_ctx_);

        int seq_len = 8;
        int n_heads = 4;
        int n_kv_heads = 2;
        int head_dim = 16;
        size_t hidden_size = n_heads * head_dim;

        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = true;
        params.window_size = -1;
        params.device_id = device_id_;

        auto stage = std::make_unique<AttentionComputeStage>(params);
        StageDumpInfo info = stage->getDumpInfo();

        // Check scalars captured key dimensions
        EXPECT_FALSE(info.scalars.empty());

        // Find specific scalars
        bool found_seq_len = false;
        bool found_n_heads = false;
        for (const auto &s : info.scalars)
        {
            if (std::string(s.name) == "seq_len")
            {
                found_seq_len = true;
                EXPECT_EQ(static_cast<int>(s.value), seq_len);
            }
            if (std::string(s.name) == "n_heads")
            {
                found_n_heads = true;
                EXPECT_EQ(static_cast<int>(s.value), n_heads);
            }
        }
        EXPECT_TRUE(found_seq_len);
        EXPECT_TRUE(found_n_heads);
    }

    /**
     * @brief Test estimatedFlops calculation
     */
    TEST_F(Test__AttentionComputeStage, EstimatedFlops)
    {
        TensorFactory factory(mpi_ctx_);

        int seq_len = 8;
        int kv_len = 16;
        int n_heads = 4;
        int n_kv_heads = 2;
        int head_dim = 16;
        size_t hidden_size = n_heads * head_dim;

        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = kv_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;

        auto stage = std::make_unique<AttentionComputeStage>(params);
        size_t flops = stage->estimatedFlops();

        // FLOPs should be:
        // Q@K^T: 2 * batch * n_heads * seq_len * kv_len * head_dim = 2*1*4*8*16*16 = 16384
        // softmax: ~4 * batch * n_heads * seq_len * kv_len = 4*1*4*8*16 = 2048
        // scores@V: 2 * batch * n_heads * seq_len * kv_len * head_dim = 16384
        // Total: 16384 + 2048 + 16384 = 34816
        size_t expected_qk = 2ULL * 1 * n_heads * seq_len * kv_len * head_dim;
        size_t expected_softmax = 4ULL * 1 * n_heads * seq_len * kv_len;
        size_t expected_sv = expected_qk;
        size_t expected_total = expected_qk + expected_softmax + expected_sv;

        EXPECT_EQ(flops, expected_total);
    }

    /**
     * @brief Test basic execute with FP32 tensors
     *
     * This test verifies that AttentionComputeStage::execute() can successfully
     * dispatch to the underlying attention kernel via KernelFactory.
     */
    TEST_F(Test__AttentionComputeStage, Execute_FP32_Basic)
    {
        TensorFactory factory(mpi_ctx_);

        // Small attention problem
        int seq_len = 2;
        int n_heads = 2;
        int n_kv_heads = 2;
        int head_dim = 4;
        size_t hidden_size = n_heads * head_dim;

        // Create tensors
        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        // Initialize with simple values
        float *Q_data = Q->mutable_data();
        float *K_data = K->mutable_data();
        float *V_data = V->mutable_data();
        float *out_data = output->mutable_data();

        size_t total_q = seq_len * hidden_size;
        size_t total_kv = seq_len * n_kv_heads * head_dim;

        // Initialize Q, K, V with small values to avoid numerical issues
        for (size_t i = 0; i < total_q; ++i)
        {
            Q_data[i] = 0.1f * (i % 4);
        }
        for (size_t i = 0; i < total_kv; ++i)
        {
            K_data[i] = 0.1f * ((i + 1) % 4);
            V_data[i] = 0.1f * ((i + 2) % 4);
        }
        for (size_t i = 0; i < total_q; ++i)
        {
            out_data[i] = -999.0f; // Sentinel value
        }

        // Create workspace for attention scores
        auto workspace_scores = factory.createFP32(
            {static_cast<size_t>(n_heads * seq_len), static_cast<size_t>(seq_len)},
            device_id_);

        // Configure stage params
        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = true;
        params.window_size = -1;
        params.workspace_scores = workspace_scores.get();
        params.mpi_ctx = &mpi_ctx_;
        params.device_id = device_id_;

        // Create and execute stage
        auto stage = std::make_unique<AttentionComputeStage>(params);

        // Execute with no device context (CPU execution)
        bool success = stage->execute(nullptr);
        EXPECT_TRUE(success) << "AttentionComputeStage::execute() should succeed";

        // Verify output was modified (not sentinel values)
        bool output_modified = false;
        for (size_t i = 0; i < total_q; ++i)
        {
            if (std::abs(out_data[i] - (-999.0f)) > 1e-6f)
            {
                output_modified = true;
                break;
            }
        }
        EXPECT_TRUE(output_modified) << "Output should be modified after execution";

        // Verify output is finite
        for (size_t i = 0; i < total_q; ++i)
        {
            EXPECT_TRUE(std::isfinite(out_data[i])) << "Output[" << i << "] = " << out_data[i] << " should be finite";
        }
    }

    /**
     * @brief Test execute with GQA (n_kv_heads < n_heads)
     */
    TEST_F(Test__AttentionComputeStage, Execute_FP32_GQA)
    {
        TensorFactory factory(mpi_ctx_);

        // GQA configuration: 4 query heads, 2 KV heads (ratio 2:1)
        int seq_len = 4;
        int n_heads = 4;
        int n_kv_heads = 2;
        int head_dim = 8;
        size_t hidden_size = n_heads * head_dim;

        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        // Initialize with random-ish values
        float *Q_data = Q->mutable_data();
        float *K_data = K->mutable_data();
        float *V_data = V->mutable_data();

        for (size_t i = 0; i < Q->numel(); ++i)
            Q_data[i] = 0.05f * ((i * 7) % 11 - 5);
        for (size_t i = 0; i < K->numel(); ++i)
            K_data[i] = 0.05f * ((i * 13) % 11 - 5);
        for (size_t i = 0; i < V->numel(); ++i)
            V_data[i] = 0.05f * ((i * 17) % 11 - 5);

        auto workspace_scores = factory.createFP32(
            {static_cast<size_t>(n_heads * seq_len), static_cast<size_t>(seq_len)},
            device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = true;
        params.workspace_scores = workspace_scores.get();
        params.mpi_ctx = &mpi_ctx_;
        params.device_id = device_id_;

        auto stage = std::make_unique<AttentionComputeStage>(params);
        bool success = stage->execute(nullptr);
        EXPECT_TRUE(success) << "GQA attention should succeed";

        // Verify output is valid
        float *out_data = output->mutable_data();
        for (size_t i = 0; i < output->numel(); ++i)
        {
            EXPECT_TRUE(std::isfinite(out_data[i])) << "GQA output[" << i << "] should be finite";
        }
    }

    TEST_F(Test__AttentionComputeStage, ContinuationCausalMaskMatchesFullPrefillForQwen35GQA)
    {
        TensorFactory factory(mpi_ctx_);

        constexpr int total_len = 9;
        constexpr int prefix_len = 4;
        constexpr int suffix_len = total_len - prefix_len;
        constexpr int n_heads = 8;
        constexpr int n_kv_heads = 2;
        constexpr int head_dim = 256;
        constexpr size_t q_cols = static_cast<size_t>(n_heads * head_dim);
        constexpr size_t kv_cols = static_cast<size_t>(n_kv_heads * head_dim);

        auto Q_full = factory.createFP32({total_len, q_cols}, device_id_);
        auto K_full = factory.createFP32({total_len, kv_cols}, device_id_);
        auto V_full = factory.createFP32({total_len, kv_cols}, device_id_);
        auto O_full = factory.createFP32({total_len, q_cols}, device_id_);
        auto Q_suffix = factory.createFP32({suffix_len, q_cols}, device_id_);
        auto O_suffix = factory.createFP32({suffix_len, q_cols}, device_id_);

        float *q_full = Q_full->mutable_data();
        float *k_full = K_full->mutable_data();
        float *v_full = V_full->mutable_data();
        for (size_t i = 0; i < Q_full->numel(); ++i)
            q_full[i] = std::sin(static_cast<float>(i % 251) * 0.013f) * 0.05f;
        for (size_t i = 0; i < K_full->numel(); ++i)
            k_full[i] = std::cos(static_cast<float>(i % 257) * 0.011f) * 0.05f;
        for (size_t i = 0; i < V_full->numel(); ++i)
            v_full[i] = std::sin(static_cast<float>(i % 263) * 0.017f) * 0.05f;

        std::copy(q_full + static_cast<size_t>(prefix_len) * q_cols,
                  q_full + static_cast<size_t>(total_len) * q_cols,
                  Q_suffix->mutable_data());

        AttentionComputeStage::Params full_params;
        full_params.Q = Q_full.get();
        full_params.K = K_full.get();
        full_params.V = V_full.get();
        full_params.output = O_full.get();
        full_params.batch_size = 1;
        full_params.seq_len = total_len;
        full_params.kv_len = total_len;
        full_params.n_heads = n_heads;
        full_params.n_kv_heads = n_kv_heads;
        full_params.head_dim = head_dim;
        full_params.causal = true;
        full_params.device_id = device_id_;
        full_params.mpi_ctx = &mpi_ctx_;

        AttentionComputeStage full_stage(full_params);
        ASSERT_TRUE(full_stage.execute(nullptr));

        AttentionComputeStage::Params suffix_params = full_params;
        suffix_params.Q = Q_suffix.get();
        suffix_params.output = O_suffix.get();
        suffix_params.seq_len = suffix_len;
        suffix_params.kv_len = total_len;
        suffix_params.position_offset = prefix_len;

        AttentionComputeStage suffix_stage(suffix_params);
        ASSERT_TRUE(suffix_stage.execute(nullptr));

        const float *full_out = O_full->data() + static_cast<size_t>(prefix_len) * q_cols;
        const float *suffix_out = O_suffix->data();
        float max_diff = 0.0f;
        for (size_t i = 0; i < static_cast<size_t>(suffix_len) * q_cols; ++i)
        {
            max_diff = std::max(max_diff, std::abs(full_out[i] - suffix_out[i]));
        }

        EXPECT_LT(max_diff, 2e-5f)
            << "Continuation causal position offset must match one-shot causal attention "
               "for suffix rows";
    }

    /**
     * @brief Test factory method createAttentionCompute
     */
    TEST_F(Test__AttentionComputeStage, FactoryMethod)
    {
        TensorFactory factory(mpi_ctx_);

        int seq_len = 4;
        int n_heads = 2;
        int n_kv_heads = 2;
        int head_dim = 8;
        size_t hidden_size = n_heads * head_dim;

        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;

        // Use factory method
        auto stage = ComputeStageFactory::createAttentionCompute(params);
        ASSERT_NE(stage, nullptr);
        EXPECT_EQ(stage->type(), ComputeStageType::ATTENTION);
    }

#ifdef HAVE_ROCM
    TEST_F(Test__AttentionComputeStage, ROCmWorkspaceRequirementsUseStageAttentionShape)
    {
        if (!hasROCm())
        {
            GTEST_SKIP() << "ROCm not available";
        }

        TensorFactory factory(mpi_ctx_);

        // Regression for Qwen3.5 MoE long-context decode: the graph-level
        // workspace hints can describe the model-local shard as 8 heads, while
        // the concrete attention stage executes a 16-head local Q projection.
        // Split decode needs partial buffers sized by the stage shape.  If this
        // delegates the allocator hint through unchanged, attn_partial_output is
        // half-sized and overwrites PARTIAL_M/PARTIAL_L during 8-way reduction.
        constexpr int seq_len = 1;
        constexpr int kv_len = 513;
        constexpr int stage_heads = 16;
        constexpr int hinted_heads = 8;
        constexpr int n_kv_heads = 2;
        constexpr int head_dim = 256;
        constexpr int default_decode_splits = 8;

        auto Q = factory.createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(stage_heads * head_dim)},
            DeviceId::cpu());
        auto K = factory.createFP32(
            {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)},
            DeviceId::cpu());
        auto V = factory.createFP32(
            {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)},
            DeviceId::cpu());
        auto output = factory.createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(stage_heads * head_dim)},
            DeviceId::cpu());

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = kv_len;
        params.n_heads = stage_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = false;
        params.device_id = DeviceId::rocm(0);
        params.mpi_ctx = &mpi_ctx_;

        AttentionComputeStage stage(params);
        WorkspaceRequirements reqs = stage.getWorkspaceRequirements(
            /*m=*/seq_len,
            /*n=*/hinted_heads,
            /*k=*/head_dim);

        const auto *partial_output = reqs.find(rocm::AttentionWorkspaceBuffers::PARTIAL_OUTPUT);
        const auto *partial_m = reqs.find(rocm::AttentionWorkspaceBuffers::PARTIAL_M);
        const auto *partial_l = reqs.find(rocm::AttentionWorkspaceBuffers::PARTIAL_L);
        ASSERT_NE(partial_output, nullptr);
        ASSERT_NE(partial_m, nullptr);
        ASSERT_NE(partial_l, nullptr);

        const size_t expected_partial_output = static_cast<size_t>(seq_len) *
                                               static_cast<size_t>(stage_heads) *
                                               static_cast<size_t>(default_decode_splits) *
                                               static_cast<size_t>(head_dim) *
                                               sizeof(float);
        const size_t expected_partial_meta = static_cast<size_t>(seq_len) *
                                             static_cast<size_t>(stage_heads) *
                                             static_cast<size_t>(default_decode_splits) *
                                             sizeof(float);
        const size_t old_underallocated_partial_output = static_cast<size_t>(seq_len) *
                                                         static_cast<size_t>(hinted_heads) *
                                                         static_cast<size_t>(default_decode_splits) *
                                                         static_cast<size_t>(head_dim) *
                                                         sizeof(float);

        EXPECT_GE(partial_output->size_bytes, expected_partial_output);
        EXPECT_GE(partial_m->size_bytes, expected_partial_meta);
        EXPECT_GE(partial_l->size_bytes, expected_partial_meta);
        EXPECT_GT(partial_output->size_bytes, old_underallocated_partial_output)
            << "Attention workspace must be sized from the stage's n_heads, "
               "not a smaller graph-level model hint";
    }
#endif

} // namespace
