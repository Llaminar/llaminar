/**
 * @file Test__FusedQKVGEMMStage.cpp
 * @brief Unit tests for FusedQKVGEMMStage compute stage
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests the FusedQKVGEMMStage which performs fused Q/K/V projections
 * with shared activation quantization.
 *
 * Key test focus:
 * - Bias is correctly applied when set (regression test for QKV bias bug)
 * - Bias is not applied when nullptr
 * - Output with bias minus output without bias equals bias vector
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <random>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "execution/compute_stages/ComputeStages.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/Tensors.h"
#include "tensors/IQQuantTables.h"
#include "tensors/FP16Utils.h"
#include "utils/Logger.h"
#include "../../../../utils/PreparedWeightTestHarness.h"

namespace llaminar2
{
    namespace
    {
        // Helper: Generate random FP32 data
        void fill_random(float *data, size_t count, float bound = 1.0f, unsigned seed = 42)
        {
            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> dist(-bound, bound);
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = dist(gen);
            }
        }

        // Helper: Fill with constant value
        void fill_constant(float *data, size_t count, float value)
        {
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = value;
            }
        }

        // Helper: Create mock IQ4_NL weight tensor for testing
        std::unique_ptr<TensorBase> create_mock_weights(int rows, int cols, unsigned seed = 123)
        {
            std::vector<size_t> shape = {static_cast<size_t>(rows), static_cast<size_t>(cols)};
            size_t blocks_per_row = (cols + 31) / 32;
            size_t total_blocks = rows * blocks_per_row;
            std::vector<uint8_t> raw_data(total_blocks * 18); // 18 bytes per IQ4_NL block

            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> scale_dist(0.1f, 2.0f);
            std::uniform_int_distribution<uint8_t> index_dist(0, 15);

            for (size_t b = 0; b < total_blocks; ++b)
            {
                IQ4_NLBlock *block = reinterpret_cast<IQ4_NLBlock *>(raw_data.data() + b * 18);
                block->d = fp32_to_fp16(scale_dist(rng));
                for (int i = 0; i < 16; ++i)
                {
                    uint8_t low = index_dist(rng);
                    uint8_t high = index_dist(rng);
                    block->qs[i] = (high << 4) | low;
                }
            }

            return std::make_unique<IQ4_NLTensor>(shape, raw_data);
        }

        // Helper: Compute max absolute difference between two arrays
        float max_abs_diff(const float *a, const float *b, size_t count)
        {
            float max_diff = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                float diff = std::abs(a[i] - b[i]);
                if (diff > max_diff)
                    max_diff = diff;
            }
            return max_diff;
        }

        // Helper: Check if two arrays are approximately equal
        bool arrays_approx_equal(const float *a, const float *b, size_t count, float tol = 1e-5f)
        {
            for (size_t i = 0; i < count; ++i)
            {
                if (std::abs(a[i] - b[i]) > tol)
                    return false;
            }
            return true;
        }

        class RecordingWorkspaceGemm final : public ITensorGemm, public IWorkspaceConsumer
        {
        public:
            explicit RecordingWorkspaceGemm(std::string name)
                : name_(std::move(name))
            {
            }

            bool supports_device(int) const override { return true; }

            bool multiply_tensor(
                const TensorBase *,
                TensorBase *,
                int,
                int,
                int,
                bool,
                float,
                float,
                const TensorBase *,
                const IMPIContext *,
                int,
                DeviceWorkspaceManager *,
                int) override
            {
                return false;
            }

            WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override
            {
                observed_m.push_back(m);
                observed_n.push_back(n);
                observed_k.push_back(k);

                WorkspaceRequirements reqs;
                reqs.buffers.push_back({
                    name_ + "_n" + std::to_string(n),
                    static_cast<size_t>(std::max(1, n)),
                    1,
                    true});
                return reqs;
            }

            void bindWorkspace(DeviceWorkspaceManager *workspace) override { workspace_ = workspace; }
            bool hasWorkspace() const override { return workspace_ != nullptr; }
            DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

            mutable std::vector<int> observed_m;
            mutable std::vector<int> observed_n;
            mutable std::vector<int> observed_k;

        private:
            std::string name_;
            DeviceWorkspaceManager *workspace_ = nullptr;
        };

        PreparedWeightRef registerRecordingGemm(
            PreparedWeightStore &store,
            std::shared_ptr<RecordingWorkspaceGemm> kernel,
            const std::string &canonical_name,
            ModelContextId model_id)
        {
            auto binding = test::makePreparedWeightTestBinding(
                nullptr,
                DeviceId::cpu(),
                canonical_name,
                model_id);

            auto prepared_weights =
                std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmWeights>();
            prepared_weights->kind =
                llaminar::v2::kernels::KernelFactory::GemmPreparationKind::CPU_PACKED;
            prepared_weights->kernel = kernel.get();
            prepared_weights->owned_kernel = kernel;

            auto handle =
                std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle>();
            handle->device_id = DeviceId::cpu();
            handle->kind = llaminar::v2::kernels::KernelFactory::GemmPreparationKind::CPU_PACKED;
            handle->prepared_weights = std::move(prepared_weights);

            return store.registerPreparedGemmHandle(
                binding,
                PreparedWeightKind::CpuPackedGemm,
                DeviceId::cpu(),
                std::move(handle));
        }
    }

    // =============================================================================
    // Test Fixture
    // =============================================================================

    class Test__FusedQKVGEMMStage : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Default dimensions (small for fast tests)
            m_ = 4;    // sequence length
            k_ = 64;   // d_model (input features)
            n_q_ = 64; // Q output dimension
            n_k_ = 64; // K output dimension (could be different for GQA)
            n_v_ = 64; // V output dimension

            // Create CPU device context
            ctx_ = std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 4);
            ASSERT_NE(ctx_, nullptr);

            // Create random input activations
            input_ = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(k_)}, DeviceId::cpu());
            fill_random(input_->mutable_data(), m_ * k_, 1.0f, 42);

            // Create weight tensors
            wq_ = create_mock_weights(n_q_, k_, 100);
            wk_ = create_mock_weights(n_k_, k_, 200);
            wv_ = create_mock_weights(n_v_, k_, 300);
            prepared_qkv_.emplace(test::makePreparedQKVFixture(
                wq_.get(), wk_.get(), wv_.get(), DeviceId::cpu(), 0));

            // Create output tensors
            output_q_ = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(n_q_)}, DeviceId::cpu());
            output_k_ = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(n_k_)}, DeviceId::cpu());
            output_v_ = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(n_v_)}, DeviceId::cpu());

            // Create bias vectors (filled with known values for easy verification)
            bias_q_data_.resize(n_q_);
            bias_k_data_.resize(n_k_);
            bias_v_data_.resize(n_v_);

            // Use distinct constant values for easy checking
            fill_constant(bias_q_data_.data(), n_q_, 1.5f);
            fill_constant(bias_k_data_.data(), n_k_, 2.5f);
            fill_constant(bias_v_data_.data(), n_v_, 3.5f);

            // Wrap in FP32Tensor for API
            bias_q_ = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(n_q_)}, DeviceId::cpu());
            bias_k_ = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(n_k_)}, DeviceId::cpu());
            bias_v_ = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(n_v_)}, DeviceId::cpu());
            std::copy(bias_q_data_.begin(), bias_q_data_.end(), bias_q_->mutable_data());
            std::copy(bias_k_data_.begin(), bias_k_data_.end(), bias_k_->mutable_data());
            std::copy(bias_v_data_.begin(), bias_v_data_.end(), bias_v_->mutable_data());
        }

        void attachPreparedRefs(FusedQKVGEMMStage::Params &params)
        {
            params.prepared_ref_q = prepared_qkv_->q_ref;
            params.prepared_ref_k = prepared_qkv_->k_ref;
            params.prepared_ref_v = prepared_qkv_->v_ref;
            params.prepared_store = prepared_qkv_->store.get();
        }

        // Dimensions
        int m_, k_, n_q_, n_k_, n_v_;

        // Device context
        std::unique_ptr<CPUDeviceContext> ctx_;

        // Tensors
        std::unique_ptr<FP32Tensor> input_;
        std::unique_ptr<TensorBase> wq_, wk_, wv_;
        std::optional<test::PreparedQKVFixture> prepared_qkv_;
        std::unique_ptr<FP32Tensor> output_q_, output_k_, output_v_;

        // Bias tensors (FP32Tensor for API compatibility)
        std::vector<float> bias_q_data_, bias_k_data_, bias_v_data_;
        std::unique_ptr<FP32Tensor> bias_q_, bias_k_, bias_v_;
    };

    // =============================================================================
    // Basic Functionality Tests
    // =============================================================================

    TEST_F(Test__FusedQKVGEMMStage, ExecuteWithoutBias)
    {
        // Test basic execution without bias
        FusedQKVGEMMStage::Params params{
            .input = input_.get(),
            .m = m_,
            .k = k_,
            .wq = wq_.get(),
            .output_q = output_q_.get(),
            .n_q = n_q_,
            .bias_q = nullptr,
            .wk = wk_.get(),
            .output_k = output_k_.get(),
            .n_k = n_k_,
            .bias_k = nullptr,
            .wv = wv_.get(),
            .output_v = output_v_.get(),
            .n_v = n_v_,
            .bias_v = nullptr};

        attachPreparedRefs(params);

        FusedQKVGEMMStage stage(params);

        // Execute
        ASSERT_TRUE(stage.execute(ctx_.get()));

        // Verify outputs are non-zero (GEMM was actually computed)
        bool q_nonzero = false, k_nonzero = false, v_nonzero = false;
        for (int i = 0; i < m_ * n_q_; ++i)
        {
            if (std::abs(output_q_->data()[i]) > 1e-10f)
            {
                q_nonzero = true;
                break;
            }
        }
        for (int i = 0; i < m_ * n_k_; ++i)
        {
            if (std::abs(output_k_->data()[i]) > 1e-10f)
            {
                k_nonzero = true;
                break;
            }
        }
        for (int i = 0; i < m_ * n_v_; ++i)
        {
            if (std::abs(output_v_->data()[i]) > 1e-10f)
            {
                v_nonzero = true;
                break;
            }
        }

        EXPECT_TRUE(q_nonzero) << "Q output should have non-zero values";
        EXPECT_TRUE(k_nonzero) << "K output should have non-zero values";
        EXPECT_TRUE(v_nonzero) << "V output should have non-zero values";
    }

    // =============================================================================
    // Bias Tests (Regression tests for QKV bias bug)
    // =============================================================================

    TEST_F(Test__FusedQKVGEMMStage, BiasIsAppliedWhenSet)
    {
        // This is the key regression test for the QKV bias bug.
        // We run the same GEMM with and without bias, and verify the difference.

        // --- Step 1: Run without bias ---
        std::vector<float> output_q_no_bias(m_ * n_q_);
        std::vector<float> output_k_no_bias(m_ * n_k_);
        std::vector<float> output_v_no_bias(m_ * n_v_);

        {
            // Zero output buffers
            std::fill(output_q_->mutable_data(), output_q_->mutable_data() + m_ * n_q_, 0.0f);
            std::fill(output_k_->mutable_data(), output_k_->mutable_data() + m_ * n_k_, 0.0f);
            std::fill(output_v_->mutable_data(), output_v_->mutable_data() + m_ * n_v_, 0.0f);

            FusedQKVGEMMStage::Params params{
                .input = input_.get(),
                .m = m_,
                .k = k_,
                .wq = wq_.get(),
                .output_q = output_q_.get(),
                .n_q = n_q_,
                .bias_q = nullptr,
                .wk = wk_.get(),
                .output_k = output_k_.get(),
                .n_k = n_k_,
                .bias_k = nullptr,
                .wv = wv_.get(),
                .output_v = output_v_.get(),
                .n_v = n_v_,
                .bias_v = nullptr};

            attachPreparedRefs(params);

            FusedQKVGEMMStage stage(params);
            ASSERT_TRUE(stage.execute(ctx_.get()));

            // Save results
            std::copy(output_q_->data(), output_q_->data() + m_ * n_q_, output_q_no_bias.begin());
            std::copy(output_k_->data(), output_k_->data() + m_ * n_k_, output_k_no_bias.begin());
            std::copy(output_v_->data(), output_v_->data() + m_ * n_v_, output_v_no_bias.begin());
        }

        // --- Step 2: Run WITH bias ---
        std::vector<float> output_q_with_bias(m_ * n_q_);
        std::vector<float> output_k_with_bias(m_ * n_k_);
        std::vector<float> output_v_with_bias(m_ * n_v_);

        {
            // Zero output buffers
            std::fill(output_q_->mutable_data(), output_q_->mutable_data() + m_ * n_q_, 0.0f);
            std::fill(output_k_->mutable_data(), output_k_->mutable_data() + m_ * n_k_, 0.0f);
            std::fill(output_v_->mutable_data(), output_v_->mutable_data() + m_ * n_v_, 0.0f);

            FusedQKVGEMMStage::Params params{
                .input = input_.get(),
                .m = m_,
                .k = k_,
                .wq = wq_.get(),
                .output_q = output_q_.get(),
                .n_q = n_q_,
                .bias_q = bias_q_.get(),
                .wk = wk_.get(),
                .output_k = output_k_.get(),
                .n_k = n_k_,
                .bias_k = bias_k_.get(),
                .wv = wv_.get(),
                .output_v = output_v_.get(),
                .n_v = n_v_,
                .bias_v = bias_v_.get()};

            attachPreparedRefs(params);

            FusedQKVGEMMStage stage(params);
            ASSERT_TRUE(stage.execute(ctx_.get()));

            // Save results
            std::copy(output_q_->data(), output_q_->data() + m_ * n_q_, output_q_with_bias.begin());
            std::copy(output_k_->data(), output_k_->data() + m_ * n_k_, output_k_with_bias.begin());
            std::copy(output_v_->data(), output_v_->data() + m_ * n_v_, output_v_with_bias.begin());
        }

        // --- Step 3: Verify that output_with_bias - output_no_bias == bias ---
        // The bias is broadcast across all M rows, so each row should differ by the bias vector

        float tol = 1e-4f; // Tolerance for floating point comparison

        // Check Q projection bias
        for (int row = 0; row < m_; ++row)
        {
            for (int col = 0; col < n_q_; ++col)
            {
                int idx = row * n_q_ + col;
                float diff = output_q_with_bias[idx] - output_q_no_bias[idx];
                float expected = bias_q_data_[col];
                EXPECT_NEAR(diff, expected, tol)
                    << "Q bias mismatch at row=" << row << " col=" << col
                    << " diff=" << diff << " expected=" << expected;
            }
        }

        // Check K projection bias
        for (int row = 0; row < m_; ++row)
        {
            for (int col = 0; col < n_k_; ++col)
            {
                int idx = row * n_k_ + col;
                float diff = output_k_with_bias[idx] - output_k_no_bias[idx];
                float expected = bias_k_data_[col];
                EXPECT_NEAR(diff, expected, tol)
                    << "K bias mismatch at row=" << row << " col=" << col
                    << " diff=" << diff << " expected=" << expected;
            }
        }

        // Check V projection bias
        for (int row = 0; row < m_; ++row)
        {
            for (int col = 0; col < n_v_; ++col)
            {
                int idx = row * n_v_ + col;
                float diff = output_v_with_bias[idx] - output_v_no_bias[idx];
                float expected = bias_v_data_[col];
                EXPECT_NEAR(diff, expected, tol)
                    << "V bias mismatch at row=" << row << " col=" << col
                    << " diff=" << diff << " expected=" << expected;
            }
        }
    }

    TEST_F(Test__FusedQKVGEMMStage, PartialBiasOnly_Q)
    {
        // Test that only Q bias is applied when K and V biases are nullptr
        std::vector<float> output_q_no_bias(m_ * n_q_);
        std::vector<float> output_k_no_bias(m_ * n_k_);
        std::vector<float> output_v_no_bias(m_ * n_v_);

        // Run without any bias first
        {
            std::fill(output_q_->mutable_data(), output_q_->mutable_data() + m_ * n_q_, 0.0f);
            std::fill(output_k_->mutable_data(), output_k_->mutable_data() + m_ * n_k_, 0.0f);
            std::fill(output_v_->mutable_data(), output_v_->mutable_data() + m_ * n_v_, 0.0f);

            FusedQKVGEMMStage::Params params{
                .input = input_.get(),
                .m = m_,
                .k = k_,
                .wq = wq_.get(),
                .output_q = output_q_.get(),
                .n_q = n_q_,
                .bias_q = nullptr,
                .wk = wk_.get(),
                .output_k = output_k_.get(),
                .n_k = n_k_,
                .bias_k = nullptr,
                .wv = wv_.get(),
                .output_v = output_v_.get(),
                .n_v = n_v_,
                .bias_v = nullptr};

            attachPreparedRefs(params);

            FusedQKVGEMMStage stage(params);
            ASSERT_TRUE(stage.execute(ctx_.get()));

            std::copy(output_q_->data(), output_q_->data() + m_ * n_q_, output_q_no_bias.begin());
            std::copy(output_k_->data(), output_k_->data() + m_ * n_k_, output_k_no_bias.begin());
            std::copy(output_v_->data(), output_v_->data() + m_ * n_v_, output_v_no_bias.begin());
        }

        // Run with only Q bias
        std::vector<float> output_q_with_bias(m_ * n_q_);
        std::vector<float> output_k_partial(m_ * n_k_);
        std::vector<float> output_v_partial(m_ * n_v_);

        {
            std::fill(output_q_->mutable_data(), output_q_->mutable_data() + m_ * n_q_, 0.0f);
            std::fill(output_k_->mutable_data(), output_k_->mutable_data() + m_ * n_k_, 0.0f);
            std::fill(output_v_->mutable_data(), output_v_->mutable_data() + m_ * n_v_, 0.0f);

            FusedQKVGEMMStage::Params params{
                .input = input_.get(),
                .m = m_,
                .k = k_,
                .wq = wq_.get(),
                .output_q = output_q_.get(),
                .n_q = n_q_,
                .bias_q = bias_q_.get(), // Only Q has bias
                .wk = wk_.get(),
                .output_k = output_k_.get(),
                .n_k = n_k_,
                .bias_k = nullptr,
                .wv = wv_.get(),
                .output_v = output_v_.get(),
                .n_v = n_v_,
                .bias_v = nullptr};

            attachPreparedRefs(params);

            FusedQKVGEMMStage stage(params);
            ASSERT_TRUE(stage.execute(ctx_.get()));

            std::copy(output_q_->data(), output_q_->data() + m_ * n_q_, output_q_with_bias.begin());
            std::copy(output_k_->data(), output_k_->data() + m_ * n_k_, output_k_partial.begin());
            std::copy(output_v_->data(), output_v_->data() + m_ * n_v_, output_v_partial.begin());
        }

        float tol = 1e-4f;

        // Q should have bias applied
        for (int row = 0; row < m_; ++row)
        {
            for (int col = 0; col < n_q_; ++col)
            {
                int idx = row * n_q_ + col;
                float diff = output_q_with_bias[idx] - output_q_no_bias[idx];
                EXPECT_NEAR(diff, bias_q_data_[col], tol)
                    << "Q bias should be applied at row=" << row << " col=" << col;
            }
        }

        // K and V should be unchanged (no bias)
        EXPECT_TRUE(arrays_approx_equal(output_k_partial.data(), output_k_no_bias.data(), m_ * n_k_, tol))
            << "K output should be unchanged when K bias is nullptr";
        EXPECT_TRUE(arrays_approx_equal(output_v_partial.data(), output_v_no_bias.data(), m_ * n_v_, tol))
            << "V output should be unchanged when V bias is nullptr";
    }

    // =============================================================================
    // Stage Metadata Tests
    // =============================================================================

    TEST_F(Test__FusedQKVGEMMStage, StageType)
    {
        FusedQKVGEMMStage::Params params{
            .input = input_.get(),
            .m = m_,
            .k = k_,
            .wq = wq_.get(),
            .output_q = output_q_.get(),
            .n_q = n_q_,
            .bias_q = nullptr,
            .wk = wk_.get(),
            .output_k = output_k_.get(),
            .n_k = n_k_,
            .bias_k = nullptr,
            .wv = wv_.get(),
            .output_v = output_v_.get(),
            .n_v = n_v_,
            .bias_v = nullptr};

        FusedQKVGEMMStage stage(params);

        EXPECT_EQ(stage.type(), ComputeStageType::GEMM_FUSED_QKV);
    }

    TEST_F(Test__FusedQKVGEMMStage, EstimatedFlops)
    {
        FusedQKVGEMMStage::Params params{
            .input = input_.get(),
            .m = m_,
            .k = k_,
            .wq = wq_.get(),
            .output_q = output_q_.get(),
            .n_q = n_q_,
            .bias_q = nullptr,
            .wk = wk_.get(),
            .output_k = output_k_.get(),
            .n_k = n_k_,
            .bias_k = nullptr,
            .wv = wv_.get(),
            .output_v = output_v_.get(),
            .n_v = n_v_,
            .bias_v = nullptr};

        FusedQKVGEMMStage stage(params);

        // Expected: 2 * M * N * K for each of Q, K, V
        size_t expected_flops =
            2 * static_cast<size_t>(m_) * n_q_ * k_ +
            2 * static_cast<size_t>(m_) * n_k_ * k_ +
            2 * static_cast<size_t>(m_) * n_v_ * k_;

        EXPECT_EQ(stage.estimatedFlops(), expected_flops);
    }

    TEST_F(Test__FusedQKVGEMMStage, WorkspaceRequirementsUsePerProjectionN)
    {
        const ModelContextId model_id{9912};
        PreparedWeightStore store(model_id);
        auto q = std::make_shared<RecordingWorkspaceGemm>("q");
        auto k = std::make_shared<RecordingWorkspaceGemm>("k");
        auto v = std::make_shared<RecordingWorkspaceGemm>("v");

        auto q_ref = registerRecordingGemm(store, q, "blk.0.attn_q.weight", model_id);
        auto k_ref = registerRecordingGemm(store, k, "blk.0.attn_k.weight", model_id);
        auto v_ref = registerRecordingGemm(store, v, "blk.0.attn_v.weight", model_id);

        FusedQKVGEMMStage::Params params{
            .m = 3,
            .k = 128,
            .n_q = 96,
            .n_k = 16,
            .n_v = 24,
            .prepared_ref_q = q_ref,
            .prepared_ref_k = k_ref,
            .prepared_ref_v = v_ref,
            .prepared_store = &store};

        FusedQKVGEMMStage stage(params);
        const auto reqs = stage.getWorkspaceRequirements(/*m=*/3, /*n=*/999, /*k=*/128);
        (void)reqs;

        EXPECT_EQ(q->observed_n, std::vector<int>({96}));
        EXPECT_EQ(k->observed_n, std::vector<int>({16}));
        EXPECT_EQ(v->observed_n, std::vector<int>({24}));
        EXPECT_EQ(q->observed_m, std::vector<int>({3}));
        EXPECT_EQ(k->observed_k, std::vector<int>({128}));
    }

    TEST_F(Test__FusedQKVGEMMStage, SupportsBackend)
    {
        FusedQKVGEMMStage::Params params{
            .input = input_.get(),
            .m = m_,
            .k = k_,
            .wq = wq_.get(),
            .output_q = output_q_.get(),
            .n_q = n_q_,
            .bias_q = nullptr,
            .wk = wk_.get(),
            .output_k = output_k_.get(),
            .n_k = n_k_,
            .bias_k = nullptr,
            .wv = wv_.get(),
            .output_v = output_v_.get(),
            .n_v = n_v_,
            .bias_v = nullptr};

        FusedQKVGEMMStage stage(params);

        // Should support common backends
        EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
        EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
        EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::GPU_CUDA));
        EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::GPU_ROCM));
    }

    // =============================================================================
    // Error Handling Tests
    // =============================================================================

    TEST_F(Test__FusedQKVGEMMStage, NullContextFails)
    {
        FusedQKVGEMMStage::Params params{
            .input = input_.get(),
            .m = m_,
            .k = k_,
            .wq = wq_.get(),
            .output_q = output_q_.get(),
            .n_q = n_q_,
            .bias_q = nullptr,
            .wk = wk_.get(),
            .output_k = output_k_.get(),
            .n_k = n_k_,
            .bias_k = nullptr,
            .wv = wv_.get(),
            .output_v = output_v_.get(),
            .n_v = n_v_,
            .bias_v = nullptr};

        FusedQKVGEMMStage stage(params);

        // Should fail with null context
        EXPECT_FALSE(stage.execute(nullptr));
    }

    TEST_F(Test__FusedQKVGEMMStage, NullInputFails)
    {
        FusedQKVGEMMStage::Params params{
            .input = nullptr, // Null input
            .m = m_,
            .k = k_,
            .wq = wq_.get(),
            .output_q = output_q_.get(),
            .n_q = n_q_,
            .bias_q = nullptr,
            .wk = wk_.get(),
            .output_k = output_k_.get(),
            .n_k = n_k_,
            .bias_k = nullptr,
            .wv = wv_.get(),
            .output_v = output_v_.get(),
            .n_v = n_v_,
            .bias_v = nullptr};

        FusedQKVGEMMStage stage(params);

        EXPECT_FALSE(stage.execute(ctx_.get()));
    }

    TEST_F(Test__FusedQKVGEMMStage, NullWeightFails)
    {
        FusedQKVGEMMStage::Params params{
            .input = input_.get(),
            .m = m_,
            .k = k_,
            .wq = nullptr, // Null weight
            .output_q = output_q_.get(),
            .n_q = n_q_,
            .bias_q = nullptr,
            .wk = wk_.get(),
            .output_k = output_k_.get(),
            .n_k = n_k_,
            .bias_k = nullptr,
            .wv = wv_.get(),
            .output_v = output_v_.get(),
            .n_v = n_v_,
            .bias_v = nullptr};

        FusedQKVGEMMStage stage(params);

        EXPECT_FALSE(stage.execute(ctx_.get()));
    }

    TEST_F(Test__FusedQKVGEMMStage, InvalidDimensionsFails)
    {
        FusedQKVGEMMStage::Params params{
            .input = input_.get(),
            .m = 0, // Invalid dimension
            .k = k_,
            .wq = wq_.get(),
            .output_q = output_q_.get(),
            .n_q = n_q_,
            .bias_q = nullptr,
            .wk = wk_.get(),
            .output_k = output_k_.get(),
            .n_k = n_k_,
            .bias_k = nullptr,
            .wv = wv_.get(),
            .output_v = output_v_.get(),
            .n_v = n_v_,
            .bias_v = nullptr};

        FusedQKVGEMMStage stage(params);

        EXPECT_FALSE(stage.execute(ctx_.get()));
    }

} // namespace llaminar2
