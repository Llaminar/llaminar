/**
 * @file Test__GPUSamplingKernels.cpp
 * @brief Integration tests for GPU sampling kernels (argmaxF32, topKF32)
 *
 * **Purpose**: Validates the GPU-side sampling primitives (argmax and top-k)
 * implemented in both CUDA and ROCm backends. These tests mirror the
 * CPU sampler unit tests in Test__Sampler.cpp, adapted for the GPU kernel API.
 *
 * **Tests Cover**:
 * - Argmax: greedy selection on GPU (standard, uniform, peaked, negative, extreme logits)
 * - Top-K: correctness, ordering, boundary conditions, large vocabularies
 * - Numerical stability: very large/small logits, mixed extremes
 * - Edge cases: single element, uniform distribution, all-negative, all-same
 * - Real-world: Qwen2 vocab size (151936), realistic logit distributions
 * - Device-resident MTP request-batch input composition and first publication
 *
 * **GPU API Model**:
 * - allocate() → hostToDevice() → argmaxF32/topKF32 → free()
 * - argmaxF32/topKF32 perform internal sync + D2H for results (no explicit sync needed)
 *
 * **Backend Selection**:
 * Parameterized over {"CUDA", "ROCm"} — tests skip gracefully if the
 * requested backend is not available.
 *
 * @note Requires CUDA and/or ROCm devices to run. Tests skip gracefully
 *       if the required hardware is not available.
 *
 * @author GitHub Copilot
 * @date June 2026
 */

#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDAGraphCapture.h"
#endif
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/mtp/MTPRejectionSampler.h"
#include "execution/mtp/MTPVerifierOutcomeGraph.h"
#include "execution/compute_stages/stages/MTPDraftTokenPublicationStage.h"
#include "execution/compute_stages/stages/MTPStochasticTargetDistributionStage.h"
#include "execution/compute_stages/stages/MTPVerifierPreparationStage.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IKVCache.h"
#include "kernels/KernelFactory.h"
#include "kernels/common/SamplingMath.h"
#include "transfer/TransferEngine.h"
#include "utils/Sampler.h"
#include "utils/TestTensorFactory.h"

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <set>
#include <map>
#include <random>
#include <stdexcept>
#include <limits>
#include <array>
#include <cstring>

using namespace llaminar2;

namespace
{

    // =========================================================================
    // Test Fixture — parameterized over backend name
    // =========================================================================

    class GPUSamplingTest : public ::testing::TestWithParam<std::string>
    {
    protected:
        void SetUp() override
        {
            const auto &backend_name = GetParam();

            if (backend_name == "CUDA")
            {
                backend_ = getCUDABackend();
                if (!backend_)
                    GTEST_SKIP() << "CUDA backend not available";
            }
            else if (backend_name == "ROCm")
            {
                backend_ = getROCmBackend();
                if (!backend_)
                    GTEST_SKIP() << "ROCm backend not available";
            }
            else
            {
                FAIL() << "Unknown backend: " << backend_name;
            }

            device_id_ = 0;
            ASSERT_TRUE(
                backend_->prepareLogitPenaltyWorkspace(
                    /*vocab_size=*/151936,
                    device_id_))
                << "Failed to prepare persistent logit-penalty workspace on "
                << backend_name;
            stream_ = backend_->createStream(device_id_);
            ASSERT_NE(stream_, nullptr)
                << "Failed to create the fixture's explicit GPU stream on "
                << backend_name;

            // Standard logits (5 tokens) — token 2 has highest logit (3.0)
            standard_logits_ = {1.0f, 2.0f, 3.0f, 0.5f, 1.5f};

            // Uniform logits (all same value)
            uniform_logits_ = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

            // Single peak logits (one clearly dominant token)
            peaked_logits_ = {0.1f, 0.2f, 10.0f, 0.1f, 0.2f};
        }

        void TearDown() override
        {
            // Release the argmax partial-reduction scratch allocated on first use.
            if (backend_)
            {
                if (argmax_partial_vals_)
                    backend_->free(argmax_partial_vals_, device_id_);
                if (argmax_partial_idxs_)
                    backend_->free(argmax_partial_idxs_, device_id_);
                if (stream_)
                    backend_->destroyStream(stream_, device_id_);
            }
            argmax_partial_vals_ = nullptr;
            argmax_partial_idxs_ = nullptr;
            argmax_partial_capacity_ = 0;
            stream_ = nullptr;
            backend_ = nullptr;
        }

        /**
         * @brief Enqueues a fixture-owned H2D transfer on an explicit stream.
         *
         * Tests which do not create a narrower graph-capture stream use the
         * fixture stream. Tests with a dedicated producer stream pass it via
         * the overload below so transfer ordering remains local and visible.
         */
        bool copyHostToDevice(
            void *dst,
            const void *src,
            size_t bytes,
            int device_id)
        {
            return backend_->hostToDevice(
                dst, src, bytes, device_id, stream_);
        }

        /**
         * @brief Enqueues H2D transfer on the caller's exact producer stream.
         */
        bool copyHostToDevice(
            void *dst,
            const void *src,
            size_t bytes,
            int device_id,
            void *stream)
        {
            return backend_->hostToDevice(
                dst, src, bytes, device_id, stream);
        }

        /**
         * @brief Copies device results through the fixture's ordered stream.
         */
        bool copyDeviceToHost(
            void *dst,
            const void *src,
            size_t bytes,
            int device_id)
        {
            return backend_->deviceToHost(
                dst, src, bytes, device_id, stream_);
        }

        /**
         * @brief Copies device results through the exact consumer stream.
         */
        bool copyDeviceToHost(
            void *dst,
            const void *src,
            size_t bytes,
            int device_id,
            void *stream)
        {
            return backend_->deviceToHost(
                dst, src, bytes, device_id, stream);
        }

        // ------------------------------------------------------------------
        // Helper: upload host logits to GPU, return device pointer
        // ------------------------------------------------------------------
        void *uploadLogits(const std::vector<float> &logits)
        {
            size_t bytes = logits.size() * sizeof(float);
            void *d_ptr = backend_->allocate(bytes, device_id_);
            EXPECT_NE(d_ptr, nullptr) << "Device allocation failed";
            if (!d_ptr)
                return nullptr;

            bool ok = copyHostToDevice(d_ptr, logits.data(), bytes, device_id_);
            EXPECT_TRUE(ok) << "H2D transfer failed";
            if (!ok)
            {
                backend_->free(d_ptr, device_id_);
                return nullptr;
            }
            return d_ptr;
        }

        // ------------------------------------------------------------------
        // Helper: free device memory
        // ------------------------------------------------------------------
        void freeDevice(void *d_ptr)
        {
            if (d_ptr)
                backend_->free(d_ptr, device_id_);
        }

        // ------------------------------------------------------------------
        // Helper: argmax with mandatory device scratch (multi-block reduction).
        //
        // Production callers always supply arena-owned partial-reduction scratch,
        // so the CUDA backend has no single-block fallback. These tests mirror
        // that contract: a persistent scratch pair is allocated lazily on first
        // use and reused across calls, then freed in TearDown.
        // ------------------------------------------------------------------
        bool argmaxF32(
            void *d_ptr,
            int n,
            int device_id,
            float *out_value,
            int *out_index)
        {
            return argmaxF32(
                d_ptr, n, device_id, out_value, out_index, stream_);
        }

        bool argmaxF32(void *d_ptr, int n, int device_id,
                       float *out_value, int *out_index,
                       void *stream)
        {
            if (!argmax_partial_vals_)
            {
                argmax_partial_capacity_ = 1024;
                argmax_partial_vals_ =
                    backend_->allocate(argmax_partial_capacity_ * sizeof(float), device_id_);
                argmax_partial_idxs_ =
                    backend_->allocate(argmax_partial_capacity_ * sizeof(int), device_id_);
            }
            return backend_->argmaxF32(d_ptr, n, device_id, out_value, out_index,
                                       stream, argmax_partial_vals_,
                                       argmax_partial_idxs_, argmax_partial_capacity_);
        }

        bool argmaxF32BatchedRows(void *d_ptr, int rows, int cols, int device_id,
                                  float *out_values, int *out_indices)
        {
            if (!argmax_partial_vals_)
            {
                argmax_partial_capacity_ = 1024;
                argmax_partial_vals_ =
                    backend_->allocate(argmax_partial_capacity_ * sizeof(float), device_id_);
                argmax_partial_idxs_ =
                    backend_->allocate(argmax_partial_capacity_ * sizeof(int), device_id_);
            }
            return backend_->argmaxF32BatchedRows(d_ptr,
                                                  rows,
                                                  cols,
                                                  device_id,
                                                  out_values,
                                                  out_indices,
                                                  stream_,
                                                  argmax_partial_vals_,
                                                  argmax_partial_idxs_,
                                                  argmax_partial_capacity_);
        }

        bool topKF32(
            const void *data_device,
            int n,
            int k,
            int device_id,
            float *out_values,
            int *out_indices)
        {
            return topKF32(
                data_device,
                n,
                k,
                device_id,
                out_values,
                out_indices,
                stream_);
        }

        bool topKF32(
            const void *data_device,
            int n,
            int k,
            int device_id,
            float *out_values,
            int *out_indices,
            void *stream)
        {
            return backend_->topKF32(
                data_device,
                n,
                k,
                device_id,
                out_values,
                out_indices,
                stream);
        }

        IBackend *backend_ = nullptr;
        int device_id_ = 0;
        void *stream_ = nullptr;

        // Persistent argmax partial-reduction scratch (allocated on first use).
        void *argmax_partial_vals_ = nullptr;
        void *argmax_partial_idxs_ = nullptr;
        int argmax_partial_capacity_ = 0;

        std::vector<float> standard_logits_;
        std::vector<float> uniform_logits_;
        std::vector<float> peaked_logits_;
    };

    // =========================================================================
    // Instantiate for both backends
    // =========================================================================
    INSTANTIATE_TEST_SUITE_P(
        GPU,
        GPUSamplingTest,
        ::testing::Values("CUDA", "ROCm"),
        [](const ::testing::TestParamInfo<std::string> &info)
        {
            return info.param; // "CUDA" or "ROCm"
        });

    /**
     * @brief Prove captured stochastic summaries read carry policy from device.
     *
     * The same graph executable is replayed twice without changing any node
     * parameters.  Only the persistent generation-controller row changes.  A
     * zero carry bit admits two physical rows for a two-token response budget;
     * a one carry bit admits the already-emitted condition row plus two new rows.
     * Byte-for-byte comparison with SamplingMath proves both CUDA and ROCm read
     * the controller at replay time instead of capturing a stale host scalar.
     */
    TEST_P(
        GPUSamplingTest,
        DeviceGenerationSummaryReadsCarryAndBudgetAtGraphReplay)
    {
        using namespace sampling_math;

        constexpr int row_count = 3;
        constexpr int output_capacity = row_count + 1;
        const std::array<int32_t, row_count> verify_tokens = {101, 102, 103};
        const std::array<int32_t, output_capacity> draft_tokens = {
            55, 101, 102, 103};
        const std::array<int32_t, kSpeculativeBatchMaxStopTokens> stop_tokens = {
            -1, -1, -1, -1, -1, -1, -1, -1};
        const int32_t bonus_token = 104;

        auto make_control = [](int leading_count)
        {
            std::array<int, kDeviceGenerationControlCount> control{};
            EXPECT_TRUE(initialize_device_generation_control(
                /*max_new_tokens=*/8,
                /*response_capacity=*/8,
                DeviceGenerationDepthPolicy::fixed(row_count),
                control.data()));
            control[kDeviceGenerationControlTransactionCommitBudget] = 2;
            control[kDeviceGenerationControlNextLeadingCommittedOutputCount] =
                leading_count;
            return control;
        };
        auto make_expected = [&](int leading_count)
        {
            std::pair<
                std::array<int32_t, output_capacity>,
                std::array<int, kSpeculativeBatchMetaCount>> expected;
            expected.first.fill(-1);
            expected.second.fill(0);
            summarize_speculative_verify_batch_at_commit_boundary(
                draft_tokens[0],
                verify_tokens.data(),
                /*row_accepted=*/nullptr,
                row_count,
                stop_tokens.data(),
                static_cast<int>(stop_tokens.size()),
                bonus_token,
                /*has_bonus_ready_token=*/1,
                /*max_state_commit_rows=*/2,
                expected.first.data(),
                output_capacity,
                expected.second.data(),
                draft_tokens.data(),
                leading_count);
            return expected;
        };

        const auto control_without_carry = make_control(0);
        const auto control_with_carry = make_control(1);
        const auto expected_without_carry = make_expected(0);
        const auto expected_with_carry = make_expected(1);
        ASSERT_NE(expected_without_carry.first, expected_with_carry.first);

        void *d_verify_tokens = backend_->allocate(
            sizeof(verify_tokens), device_id_);
        void *d_draft_tokens = backend_->allocate(
            sizeof(draft_tokens), device_id_);
        void *d_stop_tokens = backend_->allocate(
            sizeof(stop_tokens), device_id_);
        void *d_bonus_token = backend_->allocate(
            sizeof(bonus_token), device_id_);
        void *d_control = backend_->allocate(
            sizeof(control_without_carry), device_id_);
        void *d_output = backend_->allocate(
            sizeof(int32_t) * output_capacity, device_id_);
        void *d_meta = backend_->allocate(
            sizeof(int) * kSpeculativeBatchMetaCount, device_id_);
        const std::array<void *, 7> allocations = {
            d_verify_tokens,
            d_draft_tokens,
            d_stop_tokens,
            d_bonus_token,
            d_control,
            d_output,
            d_meta};
        for (void *allocation : allocations)
            ASSERT_NE(allocation, nullptr);

        std::array<int32_t, output_capacity> actual_without_carry{};
        std::array<int, kSpeculativeBatchMetaCount> meta_without_carry{};
        std::array<int32_t, output_capacity> actual_with_carry{};
        std::array<int, kSpeculativeBatchMetaCount> meta_with_carry{};

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_verify_tokens, verify_tokens.data(), sizeof(verify_tokens),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens, draft_tokens.data(), sizeof(draft_tokens),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_stop_tokens, stop_tokens.data(), sizeof(stop_tokens),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_bonus_token, &bonus_token, sizeof(bonus_token),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_control, control_without_carry.data(),
                    sizeof(control_without_carry), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(
                    backend_->enqueueSummarizeSpeculativeVerifyBatchDeviceGenerationControls(
                        d_verify_tokens,
                        /*verify_accepted_device=*/nullptr,
                        d_draft_tokens,
                        row_count,
                        d_draft_tokens,
                        d_stop_tokens,
                        d_bonus_token,
                        /*has_bonus_token=*/true,
                        d_control,
                        device_id_,
                        /*stream=*/nullptr,
                        output_capacity,
                        d_output,
                        d_meta));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSummarizeSpeculativeVerifyBatchDeviceGenerationControls(
                        d_verify_tokens,
                        /*verify_accepted_device=*/nullptr,
                        d_draft_tokens,
                        row_count,
                        d_draft_tokens,
                        d_stop_tokens,
                        d_bonus_token,
                        /*has_bonus_token=*/true,
                        d_control,
                        device_id_,
                        stream,
                        output_capacity,
                        d_output,
                        d_meta));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(copyDeviceToHost(
                    actual_without_carry.data(), d_output,
                    sizeof(actual_without_carry), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    meta_without_carry.data(), d_meta,
                    sizeof(meta_without_carry), device_id_, stream));

                ASSERT_TRUE(copyHostToDevice(
                    d_control, control_with_carry.data(),
                    sizeof(control_with_carry), device_id_, stream));
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(copyDeviceToHost(
                    actual_with_carry.data(), d_output,
                    sizeof(actual_with_carry), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    meta_with_carry.data(), d_meta,
                    sizeof(meta_with_carry), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx =
                GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        EXPECT_EQ(actual_without_carry, expected_without_carry.first);
        EXPECT_EQ(meta_without_carry, expected_without_carry.second);
        EXPECT_EQ(actual_with_carry, expected_with_carry.first);
        EXPECT_EQ(meta_with_carry, expected_with_carry.second);

        for (void *allocation : allocations)
            backend_->free(allocation, device_id_);
    }

    /**
     * @brief Prove captured resident response assembly matches serial decode.
     *
     * Three request rows execute inside one captured graph.  The first two run
     * two speculative transactions; their second transaction carries the
     * previously emitted condition token in compact row zero, which is the
     * lifecycle edge that a host-side response loop historically handled.  The
     * third request encounters a stop token in its first transaction while
     * response budget remains.  Its later graph transactions deliberately
     * present stale metadata and must remain inert.  Comparing the complete
     * response and control rows byte-for-byte with the shared CPU transition
     * proves CUDA and ROCm preserve carry, early-stop, and absorbing-terminal
     * semantics without a transaction-boundary D2H.
     */
    TEST_P(
        GPUSamplingTest,
        DeviceGenerationControllerCarriesTransactionsByteExactlyAndCaptures)
    {
        using namespace sampling_math;

        constexpr int request_count = 3;
        constexpr int max_new_tokens = 5;
        constexpr int output_token_stride = 4;
        constexpr int response_token_stride = 8;
        constexpr int meta_stride = kSpeculativeBatchMetaCount;
        constexpr int control_stride = kDeviceGenerationControlCount;
        constexpr int verifier_row_capacity = 4;

        using MetaRow = std::array<int, meta_stride>;
        auto make_meta = [](
                             int output_count,
                             int leading_count,
                             int verifier_state_count,
                             int accepted_prefix,
                             int consumed_rows,
                             bool all_accepted,
                             bool stopped = false)
        {
            MetaRow meta{};
            meta[kSpecBatchMetaOk] = 1;
            meta[kSpecBatchMetaOutputCount] = output_count;
            meta[kSpecBatchMetaAcceptedSpeculativePrefix] = accepted_prefix;
            meta[kSpecBatchMetaTargetVerifierStateCommitCount] =
                verifier_state_count;
            meta[kSpecBatchMetaStoppedOnOutput] = stopped ? 1 : 0;
            meta[kSpecBatchMetaAllSpeculativeAccepted] = all_accepted ? 1 : 0;
            meta[kSpecBatchMetaConsumedVerifierRows] = consumed_rows;
            meta[kSpecBatchMetaSampledTerminal] = all_accepted ? 1 : 0;
            meta[kSpecBatchMetaLeadingCommittedOutputCount] = leading_count;
            return meta;
        };

        const std::array<int32_t, request_count * output_token_stride>
            first_tokens = {
                10, 11, 12, -1,
                20, 21, -1, -1,
                30, 31, -1, -1};
        const std::array<int32_t, request_count * output_token_stride>
            second_tokens = {
                12, 13, 14, -1,
                21, 22, 23, 24,
                -901, -902, -903, -904};
        const std::array<int, request_count> first_base_cached_tokens = {
            100, 200, 300};
        const std::array<int, request_count> second_base_cached_tokens = {
            102, 201, 302};
        const std::array<int, request_count> terminal_replay_base_cached_tokens = {
            105, 205, 304};
        std::array<int, request_count * meta_stride> first_meta{};
        std::array<int, request_count * meta_stride> second_meta{};
        const std::array<MetaRow, request_count> first_meta_rows = {
            make_meta(3, 0, 2, 2, 2, true),
            make_meta(2, 0, 1, 0, 1, false),
            make_meta(2, 0, 2, 1, 1, false, true)};
        const std::array<MetaRow, request_count> second_meta_rows = {
            make_meta(3, 1, 2, 1, 2, false),
            make_meta(4, 1, 3, 2, 3, false),
            MetaRow{}};
        for (int request = 0; request < request_count; ++request)
        {
            std::copy(
                first_meta_rows[request].begin(),
                first_meta_rows[request].end(),
                first_meta.begin() + request * meta_stride);
            std::copy(
                second_meta_rows[request].begin(),
                second_meta_rows[request].end(),
                second_meta.begin() + request * meta_stride);
        }

        std::array<int32_t, request_count * response_token_stride>
            expected_response{};
        expected_response.fill(-1);
        std::array<int, request_count * control_stride> expected_control{};
        std::array<int, request_count> expected_restore_rows{};
        std::array<int, request_count> expected_target_cached_tokens{};
        std::array<int, request_count> expected_accepted_state_counts{};
        std::array<int, request_count> expected_publication_ok{};
        std::array<int32_t, request_count> expected_next_condition_tokens{};
        std::array<int, request_count> expected_all_drafts_accepted{};
        std::array<int, request_count> expected_stopped{};
        for (int request = 0; request < request_count; ++request)
        {
            int *control =
                expected_control.data() + request * control_stride;
            int32_t *response =
                expected_response.data() + request * response_token_stride;
            ASSERT_TRUE(initialize_device_generation_control(
                max_new_tokens,
                response_token_stride,
                DeviceGenerationDepthPolicy::fixed(
                    verifier_row_capacity - 1),
                control));
            ASSERT_EQ(
                prepare_device_generation_transaction_budget(
                    verifier_row_capacity,
                    verifier_row_capacity,
                    control),
                verifier_row_capacity);
            ASSERT_TRUE(
                commit_device_generation_and_derive_speculative_publication_metadata(
                first_tokens.data(),
                output_token_stride,
                first_meta.data(),
                meta_stride,
                request,
                verifier_row_capacity,
                first_base_cached_tokens[request],
                response,
                response_token_stride,
                control,
                expected_restore_rows.data() + request,
                expected_target_cached_tokens.data() + request,
                expected_accepted_state_counts.data() + request,
                expected_publication_ok.data() + request,
                expected_next_condition_tokens.data() + request,
                expected_all_drafts_accepted.data() + request,
                expected_stopped.data() + request));
            ASSERT_EQ(
                prepare_device_generation_transaction_budget(
                    verifier_row_capacity,
                    verifier_row_capacity,
                    control),
                request == 0 ? 2 : request == 1 ? 3 : 0);
            ASSERT_TRUE(
                commit_device_generation_and_derive_speculative_publication_metadata(
                second_tokens.data(),
                output_token_stride,
                second_meta.data(),
                meta_stride,
                request,
                verifier_row_capacity,
                second_base_cached_tokens[request],
                response,
                response_token_stride,
                control,
                expected_restore_rows.data() + request,
                expected_target_cached_tokens.data() + request,
                expected_accepted_state_counts.data() + request,
                expected_publication_ok.data() + request,
                expected_next_condition_tokens.data() + request,
                expected_all_drafts_accepted.data() + request,
                expected_stopped.data() + request));
            ASSERT_EQ(
                prepare_device_generation_transaction_budget(
                    verifier_row_capacity,
                    verifier_row_capacity,
                    control),
                0);
            ASSERT_TRUE(
                commit_device_generation_and_derive_speculative_publication_metadata(
                first_tokens.data(),
                output_token_stride,
                first_meta.data(),
                meta_stride,
                request,
                verifier_row_capacity,
                terminal_replay_base_cached_tokens[request],
                response,
                response_token_stride,
                control,
                expected_restore_rows.data() + request,
                expected_target_cached_tokens.data() + request,
                expected_accepted_state_counts.data() + request,
                expected_publication_ok.data() + request,
                expected_next_condition_tokens.data() + request,
                expected_all_drafts_accepted.data() + request,
                expected_stopped.data() + request));
        }

        void *d_first_tokens = backend_->allocate(
            first_tokens.size() * sizeof(int32_t), device_id_);
        void *d_second_tokens = backend_->allocate(
            second_tokens.size() * sizeof(int32_t), device_id_);
        void *d_first_meta = backend_->allocate(
            first_meta.size() * sizeof(int), device_id_);
        void *d_second_meta = backend_->allocate(
            second_meta.size() * sizeof(int), device_id_);
        void *d_first_base = backend_->allocate(
            first_base_cached_tokens.size() * sizeof(int), device_id_);
        void *d_second_base = backend_->allocate(
            second_base_cached_tokens.size() * sizeof(int), device_id_);
        void *d_terminal_replay_base = backend_->allocate(
            terminal_replay_base_cached_tokens.size() * sizeof(int),
            device_id_);
        void *d_response = backend_->allocate(
            expected_response.size() * sizeof(int32_t), device_id_);
        void *d_control = backend_->allocate(
            expected_control.size() * sizeof(int), device_id_);
        void *d_restore_rows = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_target_cached_tokens = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_accepted_state_counts = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_publication_ok = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_next_condition_tokens = backend_->allocate(request_count * sizeof(int32_t), device_id_);
        void *d_next_verifier_condition_tokens =
            backend_->allocate(request_count * sizeof(int32_t), device_id_);
        void *d_all_drafts_accepted = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_stopped = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_next_sidecar_condition_tokens =
            backend_->allocate(request_count * sizeof(int32_t), device_id_);
        void *d_next_sidecar_position_ids =
            backend_->allocate(request_count * sizeof(int32_t), device_id_);
        void *d_shifted_target_cached_tokens =
            backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_shifted_accepted_state_counts =
            backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_shifted_ok =
            backend_->allocate(request_count * sizeof(int), device_id_);

        auto cleanup = [&]()
        {
            void *allocations[] = {
                d_first_tokens,
                d_second_tokens,
                d_first_meta,
                d_second_meta,
                d_first_base,
                d_second_base,
                d_terminal_replay_base,
                d_response,
                d_control,
                d_restore_rows,
                d_target_cached_tokens,
                d_accepted_state_counts,
                d_publication_ok,
                d_next_condition_tokens,
                d_next_verifier_condition_tokens,
                d_all_drafts_accepted,
                d_stopped,
                d_next_sidecar_condition_tokens,
                d_next_sidecar_position_ids,
                d_shifted_target_cached_tokens,
                d_shifted_accepted_state_counts,
                d_shifted_ok};
            for (void *allocation : allocations)
            {
                if (allocation)
                    backend_->free(allocation, device_id_);
            }
        };

        ASSERT_NE(d_first_tokens, nullptr);
        ASSERT_NE(d_second_tokens, nullptr);
        ASSERT_NE(d_first_meta, nullptr);
        ASSERT_NE(d_second_meta, nullptr);
        ASSERT_NE(d_first_base, nullptr);
        ASSERT_NE(d_second_base, nullptr);
        ASSERT_NE(d_terminal_replay_base, nullptr);
        ASSERT_NE(d_response, nullptr);
        ASSERT_NE(d_control, nullptr);
        ASSERT_NE(d_restore_rows, nullptr);
        ASSERT_NE(d_target_cached_tokens, nullptr);
        ASSERT_NE(d_accepted_state_counts, nullptr);
        ASSERT_NE(d_publication_ok, nullptr);
        ASSERT_NE(d_next_condition_tokens, nullptr);
        ASSERT_NE(d_next_verifier_condition_tokens, nullptr);
        ASSERT_NE(d_all_drafts_accepted, nullptr);
        ASSERT_NE(d_stopped, nullptr);
        ASSERT_NE(d_next_sidecar_condition_tokens, nullptr);
        ASSERT_NE(d_next_sidecar_position_ids, nullptr);
        ASSERT_NE(d_shifted_target_cached_tokens, nullptr);
        ASSERT_NE(d_shifted_accepted_state_counts, nullptr);
        ASSERT_NE(d_shifted_ok, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                std::array<int32_t, request_count * response_token_stride>
                    initial_response{};
                initial_response.fill(-1);
                ASSERT_TRUE(copyHostToDevice(
                    d_first_tokens, first_tokens.data(),
                    first_tokens.size() * sizeof(int32_t), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_second_tokens, second_tokens.data(),
                    second_tokens.size() * sizeof(int32_t), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_first_meta, first_meta.data(),
                    first_meta.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_second_meta, second_meta.data(),
                    second_meta.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_first_base, first_base_cached_tokens.data(),
                    first_base_cached_tokens.size() * sizeof(int),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_second_base, second_base_cached_tokens.data(),
                    second_base_cached_tokens.size() * sizeof(int),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_terminal_replay_base,
                    terminal_replay_base_cached_tokens.data(),
                    terminal_replay_base_cached_tokens.size() * sizeof(int),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_response, initial_response.data(),
                    initial_response.size() * sizeof(int32_t), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueInitializeDeviceGeneration(
                    request_count, max_new_tokens,
                    DeviceGenerationDepthPolicy::fixed(
                        verifier_row_capacity - 1),
                    DeviceGenerationLeadingRowDisposition::PendingResponse,
                    response_token_stride,
                    d_response, control_stride, d_control, device_id_, nullptr));
                EXPECT_FALSE(
                    backend_->enqueuePrepareDeviceGenerationTransactionBudget(
                        d_control, control_stride, request_count,
                        verifier_row_capacity,
                        /*maintenance_rows_remaining_device=*/nullptr,
                        /*maintenance_due_device=*/nullptr,
                        /*decode_boundary_advanced_device=*/nullptr,
                        device_id_, nullptr));
                EXPECT_FALSE(
                    backend_->enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
                        d_first_tokens, output_token_stride, d_first_meta,
                        meta_stride, d_first_base, request_count,
                        verifier_row_capacity, d_response,
                        response_token_stride, d_control, control_stride,
                        device_id_, nullptr, d_restore_rows,
                        d_target_cached_tokens, d_accepted_state_counts,
                        d_publication_ok, d_next_condition_tokens,
                        d_all_drafts_accepted, d_stopped,
                        d_next_sidecar_condition_tokens,
                        d_next_sidecar_position_ids,
                        d_next_verifier_condition_tokens));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueInitializeDeviceGeneration(
                    request_count, max_new_tokens,
                    DeviceGenerationDepthPolicy::fixed(
                        verifier_row_capacity - 1),
                    DeviceGenerationLeadingRowDisposition::PendingResponse,
                    response_token_stride,
                    d_response, control_stride, d_control, device_id_, stream));
                ASSERT_TRUE(
                    backend_->enqueuePrepareDeviceGenerationTransactionBudget(
                        d_control, control_stride, request_count,
                        verifier_row_capacity,
                        /*maintenance_rows_remaining_device=*/nullptr,
                        /*maintenance_due_device=*/nullptr,
                        /*decode_boundary_advanced_device=*/nullptr,
                        device_id_, stream));
                ASSERT_TRUE(
                    backend_->enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
                        d_first_tokens, output_token_stride, d_first_meta,
                        meta_stride, d_first_base, request_count,
                        verifier_row_capacity, d_response,
                        response_token_stride, d_control, control_stride,
                        device_id_, stream, d_restore_rows,
                        d_target_cached_tokens, d_accepted_state_counts,
                        d_publication_ok, d_next_condition_tokens,
                        d_all_drafts_accepted, d_stopped,
                        d_next_sidecar_condition_tokens,
                        d_next_sidecar_position_ids,
                        d_next_verifier_condition_tokens));
                ASSERT_TRUE(
                    backend_->enqueuePrepareDeviceGenerationTransactionBudget(
                        d_control, control_stride, request_count,
                        verifier_row_capacity,
                        /*maintenance_rows_remaining_device=*/nullptr,
                        /*maintenance_due_device=*/nullptr,
                        /*decode_boundary_advanced_device=*/nullptr,
                        device_id_, stream));
                ASSERT_TRUE(
                    backend_->enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
                        d_second_tokens, output_token_stride, d_second_meta,
                        meta_stride, d_second_base, request_count,
                        verifier_row_capacity, d_response,
                        response_token_stride, d_control, control_stride,
                        device_id_, stream, d_restore_rows,
                        d_target_cached_tokens, d_accepted_state_counts,
                        d_publication_ok, d_next_condition_tokens,
                        d_all_drafts_accepted, d_stopped,
                        d_next_sidecar_condition_tokens,
                        d_next_sidecar_position_ids,
                        d_next_verifier_condition_tokens));
                /*
                 * A statically captured resident loop may reach one more body
                 * after its controller becomes terminal.  Feed stale compact
                 * rows deliberately: the absorbing controller must suppress
                 * publication without inspecting or appending those rows.
                 */
                ASSERT_TRUE(
                    backend_->enqueuePrepareDeviceGenerationTransactionBudget(
                        d_control, control_stride, request_count,
                        verifier_row_capacity,
                        /*maintenance_rows_remaining_device=*/nullptr,
                        /*maintenance_due_device=*/nullptr,
                        /*decode_boundary_advanced_device=*/nullptr,
                        device_id_, stream));
                ASSERT_TRUE(
                    backend_->enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
                        d_first_tokens, output_token_stride, d_first_meta,
                        meta_stride, d_terminal_replay_base, request_count,
                        verifier_row_capacity, d_response,
                        response_token_stride, d_control, control_stride,
                        device_id_, stream, d_restore_rows,
                        d_target_cached_tokens, d_accepted_state_counts,
                        d_publication_ok, d_next_condition_tokens,
                        d_all_drafts_accepted, d_stopped,
                        d_next_sidecar_condition_tokens,
                        d_next_sidecar_position_ids,
                        d_next_verifier_condition_tokens));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                // Replaying twice proves initialization resets all controller
                // authority without requiring the host to clear state.
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx =
                GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::array<int32_t, request_count * response_token_stride>
            actual_response{};
        std::array<int, request_count * control_stride> actual_control{};
        std::array<int, request_count> actual_restore_rows{};
        std::array<int, request_count> actual_target_cached_tokens{};
        std::array<int, request_count> actual_accepted_state_counts{};
        std::array<int, request_count> actual_publication_ok{};
        std::array<int32_t, request_count> actual_next_condition_tokens{};
        std::array<int32_t, request_count>
            actual_next_verifier_condition_tokens{};
        std::array<int, request_count> actual_all_drafts_accepted{};
        std::array<int, request_count> actual_stopped{};
        std::array<int32_t, request_count>
            actual_next_sidecar_condition_tokens{};
        std::array<int32_t, request_count> actual_next_sidecar_position_ids{};
        ASSERT_TRUE(copyDeviceToHost(
            actual_response.data(), d_response,
            actual_response.size() * sizeof(int32_t), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            actual_control.data(), d_control,
            actual_control.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            actual_restore_rows.data(), d_restore_rows,
            actual_restore_rows.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            actual_target_cached_tokens.data(), d_target_cached_tokens,
            actual_target_cached_tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            actual_accepted_state_counts.data(), d_accepted_state_counts,
            actual_accepted_state_counts.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            actual_publication_ok.data(), d_publication_ok,
            actual_publication_ok.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            actual_next_condition_tokens.data(), d_next_condition_tokens,
            actual_next_condition_tokens.size() * sizeof(int32_t), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            actual_next_verifier_condition_tokens.data(),
            d_next_verifier_condition_tokens,
            actual_next_verifier_condition_tokens.size() * sizeof(int32_t),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            actual_all_drafts_accepted.data(), d_all_drafts_accepted,
            actual_all_drafts_accepted.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            actual_stopped.data(), d_stopped,
            actual_stopped.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            actual_next_sidecar_condition_tokens.data(),
            d_next_sidecar_condition_tokens,
            actual_next_sidecar_condition_tokens.size() * sizeof(int32_t),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            actual_next_sidecar_position_ids.data(),
            d_next_sidecar_position_ids,
            actual_next_sidecar_position_ids.size() * sizeof(int32_t),
            device_id_));

        EXPECT_EQ(actual_response, expected_response);
        EXPECT_EQ(actual_control, expected_control);
        EXPECT_EQ(actual_restore_rows, expected_restore_rows);
        EXPECT_EQ(actual_target_cached_tokens, expected_target_cached_tokens);
        EXPECT_EQ(actual_accepted_state_counts, expected_accepted_state_counts);
        EXPECT_EQ(actual_publication_ok, expected_publication_ok);
        EXPECT_EQ(actual_next_condition_tokens, expected_next_condition_tokens);
        EXPECT_EQ(
            actual_next_verifier_condition_tokens,
            expected_next_condition_tokens)
            << "The canonical verifier-condition bank must match the sidecar "
               "condition derived by the same fused publication kernel.";
        EXPECT_EQ(actual_all_drafts_accepted, expected_all_drafts_accepted);
        EXPECT_EQ(actual_stopped, expected_stopped);
        EXPECT_EQ(
            actual_next_sidecar_condition_tokens,
            expected_next_condition_tokens);
        EXPECT_TRUE(std::equal(
            actual_next_sidecar_position_ids.begin(),
            actual_next_sidecar_position_ids.end(),
            expected_target_cached_tokens.begin(),
            expected_target_cached_tokens.end()));
        for (int request = 0; request < request_count; ++request)
        {
            SCOPED_TRACE(::testing::Message() << "request=" << request);
            const int *control =
                actual_control.data() + request * control_stride;
            EXPECT_EQ(control[kDeviceGenerationControlOk], 1);
            EXPECT_EQ(control[kDeviceGenerationControlRequestComplete], 1);
            EXPECT_EQ(
                control[kDeviceGenerationControlErrorCode],
                static_cast<int>(DeviceGenerationError::None));

            if (request < 2)
            {
                EXPECT_EQ(
                    control[kDeviceGenerationControlResponseTokenCount], 5);
                EXPECT_EQ(
                    control[kDeviceGenerationControlRemainingTokenCount], 0);
                EXPECT_EQ(
                    control[kDeviceGenerationControlTransactionCount], 2);
                EXPECT_EQ(
                    control[
                        kDeviceGenerationControlPublishedStateCommitCount],
                    4);
                EXPECT_EQ(control[kDeviceGenerationControlModelStopped], 0);
            }
            else
            {
                EXPECT_EQ(
                    control[kDeviceGenerationControlResponseTokenCount], 2);
                EXPECT_EQ(
                    control[kDeviceGenerationControlRemainingTokenCount], 3);
                EXPECT_EQ(
                    control[kDeviceGenerationControlTransactionCount], 1);
                EXPECT_EQ(
                    control[
                        kDeviceGenerationControlPublishedStateCommitCount],
                    2);
                EXPECT_EQ(control[kDeviceGenerationControlModelStopped], 1);
            }
        }

        cleanup();
    }

    /**
     * @brief Prove a maintenance-clipped emitted condition remains a carry row.
     *
     * This is the backend regression for a CUDA2 LLEP failure that emitted one
     * correction token twice. Transaction one emits a rejected correction;
     * transaction two enters with that carried row zero and is clipped to one
     * newly publishable state row by the resident maintenance budget. It still
     * emits the next condition token, so transaction three must skip that token
     * when it returns as row zero. The expected response is asserted directly,
     * rather than generated through the shared host helper, so a common
     * host/device accounting defect cannot bless itself as the oracle.
     *
     * The complete three-transaction sequence is captured and replayed twice
     * on CUDA and ROCm. This proves initialization, budget admission, fused
     * response/state publication, and carry ownership all remain resident and
     * graph-replayable at the exact maintenance boundary.
     */
    TEST_P(
        GPUSamplingTest,
        DeviceGenerationMaintenanceClippingCarriesConditionByteExactlyAndCaptures)
    {
        using namespace sampling_math;

        constexpr int request_count = 1;
        constexpr int verifier_rows = 4;
        constexpr int token_stride = verifier_rows;
        constexpr int meta_stride = kSpeculativeBatchMetaCount;
        constexpr int response_stride = 8;
        constexpr int control_stride = kDeviceGenerationControlCount;

        using TokenRow = std::array<int32_t, token_stride>;
        using MetaRow = std::array<int, meta_stride>;
        auto make_meta = [](
                             int output_count,
                             int leading_count,
                             int verifier_state_count,
                             int accepted_prefix,
                             int consumed_rows,
                             bool all_accepted,
                             bool boundary_clipped)
        {
            MetaRow meta{};
            meta[kSpecBatchMetaOk] = 1;
            meta[kSpecBatchMetaOutputCount] = output_count;
            meta[kSpecBatchMetaAcceptedSpeculativePrefix] = accepted_prefix;
            meta[kSpecBatchMetaTargetVerifierStateCommitCount] =
                verifier_state_count;
            meta[kSpecBatchMetaAllSpeculativeAccepted] =
                all_accepted ? 1 : 0;
            meta[kSpecBatchMetaConsumedVerifierRows] = consumed_rows;
            meta[kSpecBatchMetaSampledTerminal] = all_accepted ? 1 : 0;
            meta[kSpecBatchMetaCommitBoundaryClipped] =
                boundary_clipped ? 1 : 0;
            meta[kSpecBatchMetaLeadingCommittedOutputCount] = leading_count;
            return meta;
        };

        const TokenRow rejection_tokens = {10, 20, -1, -1};
        const TokenRow clipped_tokens = {20, 30, -1, -1};
        const TokenRow continuation_tokens = {30, 40, -1, -1};
        const MetaRow rejection_meta = make_meta(
            /*output_count=*/2,
            /*leading_count=*/0,
            /*verifier_state_count=*/1,
            /*accepted_prefix=*/0,
            /*consumed_rows=*/1,
            /*all_accepted=*/false,
            /*boundary_clipped=*/false);
        const MetaRow clipped_meta = make_meta(
            /*output_count=*/2,
            /*leading_count=*/1,
            /*verifier_state_count=*/2,
            /*accepted_prefix=*/1,
            /*consumed_rows=*/1,
            /*all_accepted=*/false,
            /*boundary_clipped=*/true);
        const MetaRow continuation_meta = make_meta(
            /*output_count=*/2,
            /*leading_count=*/1,
            /*verifier_state_count=*/2,
            /*accepted_prefix=*/1,
            /*consumed_rows=*/1,
            /*all_accepted=*/true,
            /*boundary_clipped=*/false);
        const int rejection_base = 100;
        const int clipped_base = 101;
        const int continuation_base = 102;
        const uint32_t maintenance_rows = 1;
        const uint32_t maintenance_due = 0;
        const uint32_t boundary_advanced = 0;

        void *d_rejection_tokens = backend_->allocate(
            sizeof(rejection_tokens), device_id_);
        void *d_clipped_tokens = backend_->allocate(
            sizeof(clipped_tokens), device_id_);
        void *d_continuation_tokens = backend_->allocate(
            sizeof(continuation_tokens), device_id_);
        void *d_rejection_meta = backend_->allocate(
            sizeof(rejection_meta), device_id_);
        void *d_clipped_meta = backend_->allocate(
            sizeof(clipped_meta), device_id_);
        void *d_continuation_meta = backend_->allocate(
            sizeof(continuation_meta), device_id_);
        void *d_rejection_base = backend_->allocate(sizeof(int), device_id_);
        void *d_clipped_base = backend_->allocate(sizeof(int), device_id_);
        void *d_continuation_base = backend_->allocate(sizeof(int), device_id_);
        void *d_maintenance_rows = backend_->allocate(
            sizeof(uint32_t), device_id_);
        void *d_maintenance_due = backend_->allocate(
            sizeof(uint32_t), device_id_);
        void *d_boundary_advanced = backend_->allocate(
            sizeof(uint32_t), device_id_);
        void *d_response = backend_->allocate(
            response_stride * sizeof(int32_t), device_id_);
        void *d_control = backend_->allocate(
            control_stride * sizeof(int), device_id_);
        void *d_restore_row = backend_->allocate(sizeof(int), device_id_);
        void *d_target_cached_tokens = backend_->allocate(sizeof(int), device_id_);
        void *d_accepted_state_count = backend_->allocate(sizeof(int), device_id_);
        void *d_publication_ok = backend_->allocate(sizeof(int), device_id_);
        void *d_next_condition = backend_->allocate(sizeof(int32_t), device_id_);
        void *d_all_accepted = backend_->allocate(sizeof(int), device_id_);
        void *d_stopped = backend_->allocate(sizeof(int), device_id_);
        void *d_next_sidecar_condition = backend_->allocate(
            sizeof(int32_t), device_id_);
        void *d_next_sidecar_position = backend_->allocate(
            sizeof(int32_t), device_id_);
        void *d_next_verifier_condition = backend_->allocate(
            sizeof(int32_t), device_id_);

        const std::array<void *, 24> allocations = {
            d_rejection_tokens,
            d_clipped_tokens,
            d_continuation_tokens,
            d_rejection_meta,
            d_clipped_meta,
            d_continuation_meta,
            d_rejection_base,
            d_clipped_base,
            d_continuation_base,
            d_maintenance_rows,
            d_maintenance_due,
            d_boundary_advanced,
            d_response,
            d_control,
            d_restore_row,
            d_target_cached_tokens,
            d_accepted_state_count,
            d_publication_ok,
            d_next_condition,
            d_all_accepted,
            d_stopped,
            d_next_sidecar_condition,
            d_next_sidecar_position,
            d_next_verifier_condition};
        for (void *allocation : allocations)
            ASSERT_NE(allocation, nullptr);

        auto enqueue_commit = [&](
                                  const void *tokens,
                                  void *meta,
                                  const void *base,
                                  void *stream)
        {
            return backend_
                ->enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
                    tokens,
                    token_stride,
                    meta,
                    meta_stride,
                    base,
                    request_count,
                    verifier_rows,
                    d_response,
                    response_stride,
                    d_control,
                    control_stride,
                    device_id_,
                    stream,
                    d_restore_row,
                    d_target_cached_tokens,
                    d_accepted_state_count,
                    d_publication_ok,
                    d_next_condition,
                    d_all_accepted,
                    d_stopped,
                    d_next_sidecar_condition,
                    d_next_sidecar_position,
                    d_next_verifier_condition);
        };

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *const stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                const std::array<std::pair<void *, const void *>, 12> uploads = {{
                    {d_rejection_tokens, rejection_tokens.data()},
                    {d_clipped_tokens, clipped_tokens.data()},
                    {d_continuation_tokens, continuation_tokens.data()},
                    {d_rejection_meta, rejection_meta.data()},
                    {d_clipped_meta, clipped_meta.data()},
                    {d_continuation_meta, continuation_meta.data()},
                    {d_rejection_base, &rejection_base},
                    {d_clipped_base, &clipped_base},
                    {d_continuation_base, &continuation_base},
                    {d_maintenance_rows, &maintenance_rows},
                    {d_maintenance_due, &maintenance_due},
                    {d_boundary_advanced, &boundary_advanced},
                }};
                const std::array<size_t, uploads.size()> upload_bytes = {
                    sizeof(rejection_tokens),
                    sizeof(clipped_tokens),
                    sizeof(continuation_tokens),
                    sizeof(rejection_meta),
                    sizeof(clipped_meta),
                    sizeof(continuation_meta),
                    sizeof(rejection_base),
                    sizeof(clipped_base),
                    sizeof(continuation_base),
                    sizeof(maintenance_rows),
                    sizeof(maintenance_due),
                    sizeof(boundary_advanced)};
                for (size_t i = 0; i < uploads.size(); ++i)
                {
                    ASSERT_TRUE(copyHostToDevice(
                        uploads[i].first,
                        uploads[i].second,
                        upload_bytes[i],
                        device_id_,
                        stream));
                }
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueInitializeDeviceGeneration(
                    request_count,
                    /*max_new_tokens=*/6,
                    DeviceGenerationDepthPolicy::fixed(verifier_rows - 1),
                    DeviceGenerationLeadingRowDisposition::PendingResponse,
                    response_stride,
                    d_response,
                    control_stride,
                    d_control,
                    device_id_,
                    stream));
                ASSERT_TRUE(
                    backend_->enqueuePrepareDeviceGenerationTransactionBudget(
                        d_control,
                        control_stride,
                        request_count,
                        verifier_rows,
                        /*maintenance_rows_remaining_device=*/nullptr,
                        /*maintenance_due_device=*/nullptr,
                        /*decode_boundary_advanced_device=*/nullptr,
                        device_id_,
                        stream));
                ASSERT_TRUE(enqueue_commit(
                    d_rejection_tokens,
                    d_rejection_meta,
                    d_rejection_base,
                    stream));
                ASSERT_TRUE(
                    backend_->enqueuePrepareDeviceGenerationTransactionBudget(
                        d_control,
                        control_stride,
                        request_count,
                        verifier_rows,
                        d_maintenance_rows,
                        d_maintenance_due,
                        d_boundary_advanced,
                        device_id_,
                        stream));
                ASSERT_TRUE(enqueue_commit(
                    d_clipped_tokens,
                    d_clipped_meta,
                    d_clipped_base,
                    stream));
                ASSERT_TRUE(
                    backend_->enqueuePrepareDeviceGenerationTransactionBudget(
                        d_control,
                        control_stride,
                        request_count,
                        verifier_rows,
                        /*maintenance_rows_remaining_device=*/nullptr,
                        /*maintenance_due_device=*/nullptr,
                        /*decode_boundary_advanced_device=*/nullptr,
                        device_id_,
                        stream));
                ASSERT_TRUE(enqueue_commit(
                    d_continuation_tokens,
                    d_continuation_meta,
                    d_continuation_base,
                    stream));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx =
                GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx =
                GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::array<int32_t, response_stride> actual_response{};
        std::array<int, control_stride> actual_control{};
        ASSERT_TRUE(copyDeviceToHost(
            actual_response.data(),
            d_response,
            sizeof(actual_response),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            actual_control.data(),
            d_control,
            sizeof(actual_control),
            device_id_));

        const std::array<int32_t, 4> expected_response = {10, 20, 30, 40};
        EXPECT_TRUE(std::equal(
            expected_response.begin(),
            expected_response.end(),
            actual_response.begin()));
        EXPECT_EQ(
            actual_control[kDeviceGenerationControlResponseTokenCount],
            4);
        EXPECT_EQ(
            actual_control[kDeviceGenerationControlNextLeadingCommittedOutputCount],
            0);
        EXPECT_EQ(
            actual_control[kDeviceGenerationControlPublishedStateCommitCount],
            4);
        EXPECT_EQ(actual_control[kDeviceGenerationControlOk], 1);
        EXPECT_EQ(
            actual_control[kDeviceGenerationControlErrorCode],
            static_cast<int>(DeviceGenerationError::None));

        for (void *allocation : allocations)
            backend_->free(allocation, device_id_);
    }

    /**
     * @brief Prove verifier admission consumes only an ordinary speculative boundary.
     *
     * Conditional CUDA maintenance skips its expensive body when a commit does
     * not make maintenance due. The next verifier admission must still retire
     * that commit's acknowledgement before another outcome advances the shared
     * clock. The same captured preparation graph is replayed against an ordinary
     * boundary and a due boundary on both GPU backends: the ordinary edge is
     * acknowledged exactly once, while the due edge fails closed and remains
     * available to its required maintenance transaction.
     */
    TEST_P(
        GPUSamplingTest,
        DeviceGenerationAdmissionAcknowledgesOnlyOrdinaryMaintenanceBoundary)
    {
        using namespace sampling_math;

        constexpr int request_count = 2;
        constexpr int max_new_tokens = 8;
        constexpr int response_token_stride = 8;
        constexpr int control_stride = kDeviceGenerationControlCount;
        constexpr int verifier_row_capacity = 4;

        const size_t response_bytes =
            static_cast<size_t>(request_count) * response_token_stride *
            sizeof(int32_t);
        const size_t control_bytes =
            static_cast<size_t>(request_count) * control_stride * sizeof(int);
        void *d_response = backend_->allocate(response_bytes, device_id_);
        void *d_control = backend_->allocate(control_bytes, device_id_);
        void *d_rows_remaining =
            backend_->allocate(sizeof(uint32_t), device_id_);
        void *d_maintenance_due =
            backend_->allocate(sizeof(uint32_t), device_id_);
        void *d_boundary_advanced =
            backend_->allocate(sizeof(uint32_t), device_id_);
        const std::array<void *, 5> allocations = {
            d_response,
            d_control,
            d_rows_remaining,
            d_maintenance_due,
            d_boundary_advanced};
        for (void *allocation : allocations)
            ASSERT_NE(allocation, nullptr);

        auto cleanup = [&]()
        {
            for (void *allocation : allocations)
                backend_->free(allocation, device_id_);
        };

        auto run = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *const stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                std::array<int32_t,
                           request_count * response_token_stride>
                    response{};

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueuePrepareDeviceGenerationTransactionBudget(
                        d_control,
                        control_stride,
                        request_count,
                        verifier_row_capacity,
                        d_rows_remaining,
                        d_maintenance_due,
                        d_boundary_advanced,
                        device_id_,
                        stream));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                auto initialize_controller = [&]()
                {
                    ASSERT_TRUE(copyHostToDevice(
                        d_response,
                        response.data(),
                        response_bytes,
                        device_id_,
                        stream));
                    ASSERT_TRUE(backend_->enqueueInitializeDeviceGeneration(
                        request_count,
                        max_new_tokens,
                        DeviceGenerationDepthPolicy::fixed(
                            verifier_row_capacity - 1),
                        DeviceGenerationLeadingRowDisposition::PendingResponse,
                        response_token_stride,
                        d_response,
                        control_stride,
                        d_control,
                        device_id_,
                        stream));
                };

                uint32_t rows_remaining = 3u;
                uint32_t maintenance_due = 0u;
                uint32_t boundary_advanced = 1u;
                initialize_controller();
                ASSERT_TRUE(copyHostToDevice(
                    d_rows_remaining,
                    &rows_remaining,
                    sizeof(rows_remaining),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_maintenance_due,
                    &maintenance_due,
                    sizeof(maintenance_due),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_boundary_advanced,
                    &boundary_advanced,
                    sizeof(boundary_advanced),
                    device_id_,
                    stream));
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                std::array<int, request_count * control_stride> control{};
                ASSERT_TRUE(copyDeviceToHost(
                    control.data(), d_control, control_bytes, device_id_));
                ASSERT_TRUE(copyDeviceToHost(
                    &boundary_advanced,
                    d_boundary_advanced,
                    sizeof(boundary_advanced),
                    device_id_));
                ASSERT_TRUE(copyDeviceToHost(
                    &maintenance_due,
                    d_maintenance_due,
                    sizeof(maintenance_due),
                    device_id_));
                EXPECT_EQ(boundary_advanced, 0u);
                EXPECT_EQ(maintenance_due, 0u);
                for (int request = 0; request < request_count; ++request)
                {
                    const int *row =
                        control.data() + request * control_stride;
                    EXPECT_EQ(row[kDeviceGenerationControlOk], 1);
                    EXPECT_EQ(
                        row[kDeviceGenerationControlTransactionCommitBudget],
                        3);
                }

                rows_remaining = 0u;
                maintenance_due = 1u;
                boundary_advanced = 1u;
                initialize_controller();
                ASSERT_TRUE(copyHostToDevice(
                    d_rows_remaining,
                    &rows_remaining,
                    sizeof(rows_remaining),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_maintenance_due,
                    &maintenance_due,
                    sizeof(maintenance_due),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_boundary_advanced,
                    &boundary_advanced,
                    sizeof(boundary_advanced),
                    device_id_,
                    stream));
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                ASSERT_TRUE(copyDeviceToHost(
                    control.data(), d_control, control_bytes, device_id_));
                ASSERT_TRUE(copyDeviceToHost(
                    &boundary_advanced,
                    d_boundary_advanced,
                    sizeof(boundary_advanced),
                    device_id_));
                ASSERT_TRUE(copyDeviceToHost(
                    &maintenance_due,
                    d_maintenance_due,
                    sizeof(maintenance_due),
                    device_id_));
                EXPECT_EQ(boundary_advanced, 1u);
                EXPECT_EQ(maintenance_due, 1u);
                for (int request = 0; request < request_count; ++request)
                {
                    const int *row =
                        control.data() + request * control_stride;
                    EXPECT_EQ(row[kDeviceGenerationControlOk], 0);
                    EXPECT_EQ(
                        row[kDeviceGenerationControlErrorCode],
                        static_cast<int>(
                            DeviceGenerationError::InvalidController));
                    EXPECT_EQ(
                        row[kDeviceGenerationControlTransactionCommitBudget],
                        0);
                }
            });
        };

        if (GetParam() == "CUDA")
        {
            run(GPUDeviceContextPool::instance().getNvidiaContext(device_id_));
        }
        else
        {
            run(GPUDeviceContextPool::instance().getAMDContext(device_id_));
        }

        cleanup();
    }

    /**
     * @brief Regress dynamic-depth budget composition through the real GPU kernels.
     *
     * Production may configure a maximum MTP depth of fifteen while selecting
     * depth four for the current transaction.  The five-row active verifier
     * graph, not that larger policy ceiling, owns the response-controller
     * budget.  This test composes initialization, active-geometry budget
     * publication, fused serial-equivalent reduction, and response/state commit
     * in one captured CUDA/ROCm graph.  The chosen rows reproduce a rejection
     * after one accepted draft, including the compact metadata shape that first
     * exposed the stale max-depth budget in the server E2E lane.
     */
    TEST_P(
        GPUSamplingTest,
        DynamicDepthActiveGeometryCommitsFusedOutcomeByteExactly)
    {
        using namespace sampling_math;

        constexpr int comparison_rows = 4;
        constexpr int active_verifier_rows = comparison_rows + 1;
        constexpr int configured_max_comparison_rows = 15;
        constexpr int top_k = 1;
        constexpr int output_token_stride =
            kSpeculativeBatchMaxOutputTokens;
        constexpr int response_token_stride = 64;
        constexpr int meta_stride = kSpeculativeBatchMetaCount;
        constexpr int control_stride = kDeviceGenerationControlCount;
        constexpr uint64_t seed = 0xA57E5EED1234ull;

        DeviceGenerationDepthPolicy depth_policy;
        depth_policy.mode = DeviceGenerationDepthPolicyMode::Dynamic;
        depth_policy.initial_depth = comparison_rows;
        depth_policy.minimum_depth = 1;
        depth_policy.maximum_depth = configured_max_comparison_rows;
        ASSERT_TRUE(depth_policy.valid());

        static_assert(active_verifier_rows < configured_max_comparison_rows);
        const std::array<int32_t, active_verifier_rows> target_ids = {
            101, 102, 103, 104, 105};
        const std::array<float, active_verifier_rows> target_probs = {
            1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
        const std::array<int32_t, active_verifier_rows> verifier_input = {
            10, 101, 999, 1000, 1001};
        std::array<int32_t, kSpeculativeBatchMaxStopTokens> stop_tokens{};
        stop_tokens.fill(-1);
        const int base_position = 23;
        const std::array<int, 1> base_cached_tokens = {base_position};
        const size_t output_bytes =
            static_cast<size_t>(output_token_stride) * sizeof(int32_t);
        const size_t meta_bytes =
            static_cast<size_t>(meta_stride) * sizeof(int);
        const size_t response_bytes =
            static_cast<size_t>(response_token_stride) * sizeof(int32_t);
        const size_t control_bytes =
            static_cast<size_t>(control_stride) * sizeof(int);

        void *d_target_ids = backend_->allocate(sizeof(target_ids), device_id_);
        void *d_target_probs = backend_->allocate(sizeof(target_probs), device_id_);
        void *d_verifier_input = backend_->allocate(sizeof(verifier_input), device_id_);
        void *d_stop_tokens = backend_->allocate(sizeof(stop_tokens), device_id_);
        void *d_base_position = backend_->allocate(sizeof(base_position), device_id_);
        void *d_base_cached_tokens = backend_->allocate(sizeof(base_cached_tokens), device_id_);
        void *d_sampled = backend_->allocate(output_bytes, device_id_);
        void *d_output = backend_->allocate(output_bytes, device_id_);
        void *d_meta = backend_->allocate(meta_bytes, device_id_);
        void *d_response = backend_->allocate(response_bytes, device_id_);
        void *d_control = backend_->allocate(control_bytes, device_id_);
        void *d_restore_row = backend_->allocate(sizeof(int), device_id_);
        void *d_target_cached_tokens = backend_->allocate(sizeof(int), device_id_);
        void *d_accepted_state_count = backend_->allocate(sizeof(int), device_id_);
        void *d_publication_ok = backend_->allocate(sizeof(int), device_id_);
        void *d_next_condition_token = backend_->allocate(sizeof(int32_t), device_id_);
        void *d_next_verifier_condition_token =
            backend_->allocate(sizeof(int32_t), device_id_);
        void *d_all_drafts_accepted = backend_->allocate(sizeof(int), device_id_);
        void *d_stopped = backend_->allocate(sizeof(int), device_id_);
        void *d_next_sidecar_condition_token =
            backend_->allocate(sizeof(int32_t), device_id_);
        void *d_next_sidecar_position_id =
            backend_->allocate(sizeof(int32_t), device_id_);
        const std::array<void *, 21> allocations = {
            d_target_ids,
            d_target_probs,
            d_verifier_input,
            d_stop_tokens,
            d_base_position,
            d_base_cached_tokens,
            d_sampled,
            d_output,
            d_meta,
            d_response,
            d_control,
            d_restore_row,
            d_target_cached_tokens,
            d_accepted_state_count,
            d_publication_ok,
            d_next_condition_token,
            d_next_verifier_condition_token,
            d_all_drafts_accepted,
            d_stopped,
            d_next_sidecar_condition_token,
            d_next_sidecar_position_id};
        for (void *allocation : allocations)
            ASSERT_NE(allocation, nullptr);

        auto cleanup = [&]()
        {
            for (void *allocation : allocations)
                backend_->free(allocation, device_id_);
        };

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *const stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                std::array<int32_t, response_token_stride> initial_response{};
                initial_response.fill(-1);
                ASSERT_TRUE(copyHostToDevice(
                    d_target_ids, target_ids.data(), sizeof(target_ids),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_target_probs, target_probs.data(), sizeof(target_probs),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_verifier_input, verifier_input.data(),
                    sizeof(verifier_input), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_stop_tokens, stop_tokens.data(), sizeof(stop_tokens),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_base_position, &base_position, sizeof(base_position),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_base_cached_tokens, base_cached_tokens.data(),
                    sizeof(base_cached_tokens), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_response, initial_response.data(), response_bytes,
                    device_id_, stream));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueInitializeDeviceGeneration(
                    /*request_count=*/1,
                    /*max_new_tokens=*/response_token_stride,
                    depth_policy,
                    DeviceGenerationLeadingRowDisposition::PendingResponse,
                    response_token_stride,
                    d_response,
                    control_stride,
                    d_control,
                    device_id_,
                    stream));
                ASSERT_TRUE(
                    backend_->enqueuePrepareDeviceGenerationTransactionBudget(
                        d_control,
                        control_stride,
                        /*request_count=*/1,
                        active_verifier_rows,
                        /*maintenance_rows_remaining_device=*/nullptr,
                        /*maintenance_due_device=*/nullptr,
                        /*decode_boundary_advanced_device=*/nullptr,
                        device_id_,
                        stream));
                ASSERT_TRUE(
                    backend_->enqueueSampleAndSummarizeSerialEquivalentSpeculativeBatchDeviceGenerationControls(
                        d_target_ids,
                        d_target_probs,
                        /*target_row_stride=*/top_k,
                        top_k,
                        comparison_rows,
                        seed,
                        d_base_position,
                        /*threshold_position_offset=*/1,
                        d_verifier_input,
                        d_stop_tokens,
                        d_control,
                        device_id_,
                        stream,
                        output_token_stride,
                        d_sampled,
                        d_output,
                        d_meta,
                        /*first_transaction_diagnostic=*/nullptr));
                ASSERT_TRUE(
                    backend_->enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
                        d_output,
                        output_token_stride,
                        d_meta,
                        meta_stride,
                        d_base_cached_tokens,
                        /*request_count=*/1,
                        active_verifier_rows,
                        d_response,
                        response_token_stride,
                        d_control,
                        control_stride,
                        device_id_,
                        stream,
                        d_restore_row,
                        d_target_cached_tokens,
                        d_accepted_state_count,
                        d_publication_ok,
                        d_next_condition_token,
                        d_all_drafts_accepted,
                        d_stopped,
                        d_next_sidecar_condition_token,
                        d_next_sidecar_position_id,
                        d_next_verifier_condition_token));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx =
                GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx =
                GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::array<int32_t, output_token_stride> actual_output{};
        std::array<int, meta_stride> actual_meta{};
        std::array<int32_t, response_token_stride> actual_response{};
        std::array<int, control_stride> actual_control{};
        int publication_ok = 0;
        int next_condition_token = -1;
        int next_verifier_condition_token = -1;
        ASSERT_TRUE(copyDeviceToHost(
            actual_output.data(), d_output, sizeof(actual_output), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            actual_meta.data(), d_meta, sizeof(actual_meta), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            actual_response.data(), d_response, sizeof(actual_response),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            actual_control.data(), d_control, sizeof(actual_control),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &publication_ok, d_publication_ok, sizeof(publication_ok),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &next_condition_token, d_next_condition_token,
            sizeof(next_condition_token), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &next_verifier_condition_token,
            d_next_verifier_condition_token,
            sizeof(next_verifier_condition_token),
            device_id_));

        EXPECT_EQ(actual_meta[kSpecBatchMetaOk], 1);
        EXPECT_EQ(actual_meta[kSpecBatchMetaOutputCount], 3);
        EXPECT_EQ(
            actual_meta[kSpecBatchMetaAcceptedSpeculativePrefix], 1);
        EXPECT_EQ(
            actual_meta[kSpecBatchMetaTargetVerifierStateCommitCount], 2);
        EXPECT_EQ(actual_meta[kSpecBatchMetaReadyToken], -1);
        EXPECT_EQ(actual_output[0], 10);
        EXPECT_EQ(actual_output[1], 101);
        EXPECT_EQ(actual_output[2], 102);
        EXPECT_EQ(actual_response[0], 10);
        EXPECT_EQ(actual_response[1], 101);
        EXPECT_EQ(actual_response[2], 102);
        EXPECT_EQ(publication_ok, 1);
        EXPECT_EQ(next_condition_token, 102);
        EXPECT_EQ(next_verifier_condition_token, next_condition_token);
        EXPECT_EQ(actual_control[kDeviceGenerationControlOk], 1);
        EXPECT_EQ(
            actual_control[kDeviceGenerationControlTransactionCount], 1);
        EXPECT_EQ(
            actual_control[kDeviceGenerationControlResponseTokenCount], 3);
        EXPECT_EQ(
            actual_control[
                kDeviceGenerationControlPublishedStateCommitCount],
            2);
        EXPECT_EQ(
            actual_control[kDeviceGenerationControlErrorCode],
            static_cast<int>(DeviceGenerationError::None));

        cleanup();
    }

    /**
     * @brief Reject a stale device depth without exposing recycled arena bytes.
     *
     * A depth-fifteen-capable request can legitimately replay a graph whose
     * current dynamic selector is smaller, but the selector must fit the graph's
     * declared comparison-row capacity.  This regression deliberately presents
     * a stale fixed depth of fifteen to a five-row fused verifier launch.  CUDA
     * and ROCm must both poison the controller with `InvalidDepthSelector`,
     * invalidate retained diagnostics, and overwrite every compact output byte
     * with the shared canonical invalid value.
     *
     * Seeding all destinations with non-zero sentinels is important: merely
     * checking `MetaOk == 0` would miss the original defect, where the kernel
     * returned before writing metadata and downstream diagnostics interpreted
     * unrelated activation bytes as a compact outcome.
     */
    TEST_P(
        GPUSamplingTest,
        InvalidFusedVerifierDepthPublishesDeterministicFatalOutcome)
    {
        using namespace sampling_math;

        constexpr int declared_comparison_rows = 5;
        constexpr int stale_controller_depth = 15;
        constexpr int sample_rows = declared_comparison_rows + 1;
        constexpr int top_k = 1;
        constexpr int output_capacity = kSpeculativeBatchMaxOutputTokens;
        constexpr uint64_t seed = 0x5A1EDEADBEEFull;

        std::array<int32_t, sample_rows> target_ids{};
        std::array<float, sample_rows> target_probs{};
        std::array<int32_t, sample_rows> verifier_input{};
        std::array<int32_t, kSpeculativeBatchMaxStopTokens> stop_tokens{};
        std::array<int, kDeviceGenerationControlCount> generation_control{};
        std::array<int32_t, output_capacity> sampled_target_tokens{};
        std::array<int32_t, output_capacity> output_tokens{};
        std::array<int, kSpeculativeBatchMetaCount> output_meta{};
        MTPFirstTransactionDiagnosticRecord diagnostic{};
        const int threshold_position = 19;

        target_ids.fill(17);
        target_probs.fill(1.0F);
        verifier_input.fill(17);
        stop_tokens.fill(-1);
        sampled_target_tokens.fill(0x13579BDF);
        output_tokens.fill(0x2468ACE);
        output_meta.fill(0x10203040);
        diagnostic.valid = 1;
        ASSERT_TRUE(initialize_device_generation_control(
            /*max_new_tokens=*/output_capacity,
            /*response_capacity=*/output_capacity,
            DeviceGenerationDepthPolicy::fixed(stale_controller_depth),
            generation_control.data()));
        generation_control[
            kDeviceGenerationControlTransactionCommitBudget] = sample_rows;

        void *d_target_ids = backend_->allocate(sizeof(target_ids), device_id_);
        void *d_target_probs = backend_->allocate(sizeof(target_probs), device_id_);
        void *d_verifier_input = backend_->allocate(sizeof(verifier_input), device_id_);
        void *d_stop_tokens = backend_->allocate(sizeof(stop_tokens), device_id_);
        void *d_generation_control = backend_->allocate(
            sizeof(generation_control), device_id_);
        void *d_threshold_position = backend_->allocate(
            sizeof(threshold_position), device_id_);
        void *d_sampled_target_tokens = backend_->allocate(
            sizeof(sampled_target_tokens), device_id_);
        void *d_output_tokens = backend_->allocate(
            sizeof(output_tokens), device_id_);
        void *d_output_meta = backend_->allocate(sizeof(output_meta), device_id_);
        void *d_diagnostic = backend_->allocate(sizeof(diagnostic), device_id_);
        const std::array<void *, 10> allocations = {
            d_target_ids,
            d_target_probs,
            d_verifier_input,
            d_stop_tokens,
            d_generation_control,
            d_threshold_position,
            d_sampled_target_tokens,
            d_output_tokens,
            d_output_meta,
            d_diagnostic};
        for (void *allocation : allocations)
            ASSERT_NE(allocation, nullptr);

        auto cleanup = [&]()
        {
            for (void *allocation : allocations)
                backend_->free(allocation, device_id_);
        };

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *const stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_target_ids, target_ids.data(), sizeof(target_ids),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_target_probs, target_probs.data(), sizeof(target_probs),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_verifier_input, verifier_input.data(),
                    sizeof(verifier_input), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_stop_tokens, stop_tokens.data(), sizeof(stop_tokens),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_generation_control, generation_control.data(),
                    sizeof(generation_control), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_threshold_position, &threshold_position,
                    sizeof(threshold_position), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_sampled_target_tokens, sampled_target_tokens.data(),
                    sizeof(sampled_target_tokens), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_output_tokens, output_tokens.data(),
                    sizeof(output_tokens), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_output_meta, output_meta.data(), sizeof(output_meta),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_diagnostic, &diagnostic, sizeof(diagnostic),
                    device_id_, stream));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSampleAndSummarizeSerialEquivalentSpeculativeBatchDeviceGenerationControls(
                        d_target_ids,
                        d_target_probs,
                        /*target_row_stride=*/top_k,
                        top_k,
                        declared_comparison_rows,
                        seed,
                        d_threshold_position,
                        /*threshold_position_offset=*/1,
                        d_verifier_input,
                        d_stop_tokens,
                        d_generation_control,
                        device_id_,
                        stream,
                        output_capacity,
                        d_sampled_target_tokens,
                        d_output_tokens,
                        d_output_meta,
                        d_diagnostic));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            run_capture(
                GPUDeviceContextPool::instance().getNvidiaContext(device_id_));
        }
        else
        {
            run_capture(
                GPUDeviceContextPool::instance().getAMDContext(device_id_));
        }

        ASSERT_TRUE(copyDeviceToHost(
            sampled_target_tokens.data(), d_sampled_target_tokens,
            sizeof(sampled_target_tokens), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            output_tokens.data(), d_output_tokens, sizeof(output_tokens),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            output_meta.data(), d_output_meta, sizeof(output_meta), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            generation_control.data(), d_generation_control,
            sizeof(generation_control), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &diagnostic, d_diagnostic, sizeof(diagnostic), device_id_));

        EXPECT_TRUE(std::all_of(
            sampled_target_tokens.begin(),
            sampled_target_tokens.end(),
            [](int32_t token) { return token == -1; }));
        EXPECT_TRUE(std::all_of(
            output_tokens.begin(),
            output_tokens.end(),
            [](int32_t token) { return token == -1; }));
        EXPECT_TRUE(std::all_of(
            output_meta.begin(),
            output_meta.end(),
            [](int value) { return value == 0; }));
        EXPECT_EQ(generation_control[kDeviceGenerationControlOk], 0);
        EXPECT_EQ(
            generation_control[kDeviceGenerationControlRequestComplete], 1);
        EXPECT_EQ(
            generation_control[
                kDeviceGenerationControlTransactionCommitBudget],
            0);
        EXPECT_EQ(
            generation_control[kDeviceGenerationControlErrorCode],
            static_cast<int>(DeviceGenerationError::InvalidDepthSelector));
        EXPECT_EQ(diagnostic.valid, 0u);

        cleanup();
    }

    /**
     * @brief Prove CUDA repeats the production generation transaction on device.
     *
     * The captured body uses the same budget and fused response/publication
     * kernels as stochastic MTP.  It appends exactly one token per iteration;
     * the CUDA conditional graph must therefore execute the body exactly five
     * times, stop from the resident request-complete bit, and expose no host
     * polling boundary between iterations.
     */
    TEST_P(
        GPUSamplingTest,
        DeviceControlledGraphLoopRunsUntilGenerationControllerIsTerminal)
    {
        using namespace sampling_math;

        if (GetParam() != "CUDA")
            GTEST_SKIP() << "CUDA conditional graph coverage is backend-specific";

        constexpr int request_count = 1;
        constexpr int max_new_tokens = 5;
        constexpr int output_token_stride = 1;
        constexpr int response_token_stride = 8;
        constexpr int meta_stride = kSpeculativeBatchMetaCount;
        constexpr int control_stride = kDeviceGenerationControlCount;
        constexpr int verifier_row_capacity = 2;
        constexpr int32_t emitted_token = 77;

        std::array<int32_t, output_token_stride> compact_tokens = {
            emitted_token};
        std::array<int, meta_stride> compact_meta{};
        compact_meta[kSpecBatchMetaOk] = 1;
        compact_meta[kSpecBatchMetaOutputCount] = 1;
        compact_meta[kSpecBatchMetaAcceptedSpeculativePrefix] = 0;
        compact_meta[kSpecBatchMetaTargetVerifierStateCommitCount] = 1;
        compact_meta[kSpecBatchMetaReadyToken] = emitted_token;
        compact_meta[kSpecBatchMetaRejectedVerifiedToken] = emitted_token;
        compact_meta[kSpecBatchMetaStoppedOnOutput] = 0;
        compact_meta[kSpecBatchMetaAllSpeculativeAccepted] = 0;
        compact_meta[kSpecBatchMetaConsumedVerifierRows] = 1;
        compact_meta[kSpecBatchMetaSampledTerminal] = 0;
        compact_meta[kSpecBatchMetaCommitBoundaryClipped] = 0;
        compact_meta[kSpecBatchMetaLeadingCommittedOutputCount] = 0;
        const std::array<int, request_count> base_cached_tokens = {100};

        void *d_compact_tokens = backend_->allocate(
            compact_tokens.size() * sizeof(int32_t), device_id_);
        void *d_compact_meta = backend_->allocate(
            compact_meta.size() * sizeof(int), device_id_);
        void *d_base_cached_tokens = backend_->allocate(
            base_cached_tokens.size() * sizeof(int), device_id_);
        void *d_response = backend_->allocate(
            response_token_stride * sizeof(int32_t), device_id_);
        void *d_control = backend_->allocate(
            control_stride * sizeof(int), device_id_);
        void *d_restore_row = backend_->allocate(sizeof(int), device_id_);
        void *d_target_cached_tokens = backend_->allocate(sizeof(int), device_id_);
        void *d_accepted_state_count = backend_->allocate(sizeof(int), device_id_);
        void *d_publication_ok = backend_->allocate(sizeof(int), device_id_);
        void *d_next_condition_token = backend_->allocate(sizeof(int32_t), device_id_);
        void *d_all_drafts_accepted = backend_->allocate(sizeof(int), device_id_);
        void *d_stopped = backend_->allocate(sizeof(int), device_id_);
        void *d_next_sidecar_condition_token =
            backend_->allocate(sizeof(int32_t), device_id_);
        void *d_next_sidecar_position_id =
            backend_->allocate(sizeof(int32_t), device_id_);
        void *d_next_verifier_condition_token =
            backend_->allocate(sizeof(int32_t), device_id_);

        const std::array<void *, 15> allocations = {
            d_compact_tokens,
            d_compact_meta,
            d_base_cached_tokens,
            d_response,
            d_control,
            d_restore_row,
            d_target_cached_tokens,
            d_accepted_state_count,
            d_publication_ok,
            d_next_condition_token,
            d_all_drafts_accepted,
            d_stopped,
            d_next_sidecar_condition_token,
            d_next_sidecar_position_id,
            d_next_verifier_condition_token};
        for (void *allocation : allocations)
            ASSERT_NE(allocation, nullptr);

        std::array<int32_t, response_token_stride> actual_response{};
        std::array<int, control_stride> actual_control{};
        int32_t actual_next_sidecar_condition_token = -1;
        int32_t actual_next_sidecar_position_id = -1;
        int32_t actual_next_verifier_condition_token = -1;
        auto &cuda_context =
            GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
        cuda_context.submitAndWait([&]()
        {
            void *stream = cuda_context.defaultStream();
            void *prepare_stream = cuda_context.createStream();
            void *commit_stream = cuda_context.createStream();
            ASSERT_NE(stream, nullptr);
            ASSERT_NE(prepare_stream, nullptr);
            ASSERT_NE(commit_stream, nullptr);
            ASSERT_TRUE(copyHostToDevice(
                d_compact_tokens,
                compact_tokens.data(),
                compact_tokens.size() * sizeof(int32_t),
                device_id_,
                stream));
            ASSERT_TRUE(copyHostToDevice(
                d_compact_meta,
                compact_meta.data(),
                compact_meta.size() * sizeof(int),
                device_id_,
                stream));
            ASSERT_TRUE(copyHostToDevice(
                d_base_cached_tokens,
                base_cached_tokens.data(),
                base_cached_tokens.size() * sizeof(int),
                device_id_,
                stream));
            ASSERT_TRUE(backend_->enqueueInitializeDeviceGeneration(
                request_count,
                max_new_tokens,
                DeviceGenerationDepthPolicy::fixed(1),
                DeviceGenerationLeadingRowDisposition::PendingResponse,
                response_token_stride,
                d_response,
                control_stride,
                d_control,
                device_id_,
                stream));

            auto prepare = cuda_context.createGraphCapture(prepare_stream);
            ASSERT_NE(prepare, nullptr);
            ASSERT_TRUE(prepare->beginCapture());
            ASSERT_TRUE(
                backend_->enqueuePrepareDeviceGenerationTransactionBudget(
                    d_control,
                    control_stride,
                    request_count,
                    verifier_row_capacity,
                    /*maintenance_rows_remaining_device=*/nullptr,
                    /*maintenance_due_device=*/nullptr,
                    /*decode_boundary_advanced_device=*/nullptr,
                    device_id_,
                    prepare_stream));
            ASSERT_TRUE(prepare->endCapture());
            ASSERT_GT(prepare->nodeCount(), 0u);

            auto commit = cuda_context.createGraphCapture(commit_stream);
            ASSERT_NE(commit, nullptr);
            ASSERT_TRUE(commit->beginCapture());
            ASSERT_TRUE(
                backend_->enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
                    d_compact_tokens,
                    output_token_stride,
                    d_compact_meta,
                    meta_stride,
                    d_base_cached_tokens,
                    request_count,
                    verifier_row_capacity,
                    d_response,
                    response_token_stride,
                    d_control,
                    control_stride,
                    device_id_,
                    commit_stream,
                    d_restore_row,
                    d_target_cached_tokens,
                    d_accepted_state_count,
                    d_publication_ok,
                    d_next_condition_token,
                    d_all_drafts_accepted,
                    d_stopped,
                    d_next_sidecar_condition_token,
                    d_next_sidecar_position_id,
                    d_next_verifier_condition_token));
            ASSERT_TRUE(commit->endCapture());
            ASSERT_GT(commit->nodeCount(), 0u);

            auto generation = cuda_context.createGraphCapture(stream);
            ASSERT_NE(generation, nullptr);
            ASSERT_TRUE(generation->supportsDeviceControlledWhileLoop());
            const std::array<DeviceControlledLoopFragment, 2>
                transaction_fragments = {{
                    {.name = "transaction budget", .capture = prepare.get()},
                    {.name = "transaction commit", .capture = commit.get()},
                }};
            ASSERT_TRUE(generation->buildDeviceControlledWhileLoop(
                transaction_fragments,
                DeviceControlledLoopPredicate{
                    .control_rows_device =
                        static_cast<const int *>(d_control),
                    .control_stride = control_stride,
                    .request_count = request_count,
                    .healthy_index = kDeviceGenerationControlOk,
                    .complete_index =
                        kDeviceGenerationControlRequestComplete}));
            ASSERT_GE(generation->nodeCount(), 2U)
                << "The parent must contain an entry predicate before its WHILE node";
            ASSERT_TRUE(generation->instantiate());
            ASSERT_TRUE(generation->launch());
            ASSERT_TRUE(generation->launch())
                << "A terminal controller must make a later parent launch execute zero body iterations";

            ASSERT_TRUE(copyDeviceToHost(
                actual_response.data(),
                d_response,
                actual_response.size() * sizeof(int32_t),
                device_id_,
                stream));
            ASSERT_TRUE(copyDeviceToHost(
                actual_control.data(),
                d_control,
                actual_control.size() * sizeof(int),
                device_id_,
                stream));
            ASSERT_TRUE(copyDeviceToHost(
                &actual_next_sidecar_condition_token,
                d_next_sidecar_condition_token,
                sizeof(actual_next_sidecar_condition_token),
                device_id_,
                stream));
            ASSERT_TRUE(copyDeviceToHost(
                &actual_next_sidecar_position_id,
                d_next_sidecar_position_id,
                sizeof(actual_next_sidecar_position_id),
                device_id_,
                stream));
            ASSERT_TRUE(copyDeviceToHost(
                &actual_next_verifier_condition_token,
                d_next_verifier_condition_token,
                sizeof(actual_next_verifier_condition_token),
                device_id_,
                stream));
            ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

            generation.reset();
            commit.reset();
            prepare.reset();
            cuda_context.destroyStream(commit_stream);
            cuda_context.destroyStream(prepare_stream);
        });

        for (int token_index = 0;
             token_index < max_new_tokens;
             ++token_index)
        {
            EXPECT_EQ(actual_response[static_cast<size_t>(token_index)], emitted_token);
        }
        EXPECT_EQ(actual_control[kDeviceGenerationControlOk], 1);
        EXPECT_EQ(
            actual_control[kDeviceGenerationControlResponseTokenCount],
            max_new_tokens);
        EXPECT_EQ(actual_control[kDeviceGenerationControlRemainingTokenCount], 0);
        EXPECT_EQ(actual_control[kDeviceGenerationControlRequestComplete], 1);
        EXPECT_EQ(
            actual_control[kDeviceGenerationControlTransactionCount],
            max_new_tokens);
        EXPECT_EQ(
            actual_control[kDeviceGenerationControlRejectedTransactionCount],
            max_new_tokens);
        EXPECT_EQ(
            actual_control[kDeviceGenerationControlConsumedVerifierRowCount],
            max_new_tokens);
        EXPECT_EQ(
            actual_control[kDeviceGenerationControlErrorCode],
            static_cast<int>(DeviceGenerationError::None));
        EXPECT_EQ(actual_next_sidecar_condition_token, emitted_token);
        EXPECT_EQ(
            actual_next_verifier_condition_token,
            actual_next_sidecar_condition_token);
        EXPECT_EQ(
            actual_next_sidecar_position_id,
            base_cached_tokens.front() +
                compact_meta[kSpecBatchMetaTargetVerifierStateCommitCount]);

        for (void *allocation : allocations)
            backend_->free(allocation, device_id_);
    }

    /**
     * @brief Prove a device word conditionally admits one complete graph fragment.
     *
     * The same instantiated parent is replayed twice. With the predicate word
     * clear, the mandatory transaction commits but its conditional tail leaves
     * the trace untouched. After resetting only device contents and setting the
     * predicate word, the identical executable must run the tail exactly once.
     * This is the focused contract used to omit non-due Dynamic maintenance and
     * its collective without host scheduling or graph recapture. Running the
     * same contract through WHILE and SWITCH/WHILE also proves fixed and
     * dynamic depth share one fragment-execution policy.
     */
    TEST_P(
        GPUSamplingTest,
        DeviceControlledGraphLoopConditionallyExecutesDeviceOwnedFragment)
    {
        if (GetParam() != "CUDA")
            GTEST_SKIP() << "CUDA conditional graph coverage is backend-specific";

        constexpr int healthy_index = 0;
        constexpr int complete_index = 1;
        constexpr int selector_index = 2;
        constexpr int error_index = 3;
        constexpr int control_stride = 4;
        constexpr int trace_sentinel = -1;
        constexpr int trace_publication = 7331;
        const std::array<int, control_stride> initial_control = {1, 0, 0, 0};
        constexpr int complete = 1;
        constexpr uint32_t condition_clear = 0u;
        constexpr uint32_t condition_set = 1u;

        void *d_control = backend_->allocate(
            control_stride * sizeof(int), device_id_);
        void *d_complete = backend_->allocate(sizeof(int), device_id_);
        void *d_condition = backend_->allocate(sizeof(uint32_t), device_id_);
        void *d_trace = backend_->allocate(sizeof(int), device_id_);
        void *d_trace_publication = backend_->allocate(sizeof(int), device_id_);
        const std::array<void *, 5> allocations = {
            d_control,
            d_complete,
            d_condition,
            d_trace,
            d_trace_publication};
        for (void *allocation : allocations)
            ASSERT_NE(allocation, nullptr);

        auto &cuda_context =
            GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
        cuda_context.submitAndWait([&]()
        {
            void *const parent_stream = cuda_context.createStream();
            void *const transaction_stream = cuda_context.createStream();
            void *const conditional_stream = cuda_context.createStream();
            ASSERT_NE(parent_stream, nullptr);
            ASSERT_NE(transaction_stream, nullptr);
            ASSERT_NE(conditional_stream, nullptr);

            ASSERT_TRUE(copyHostToDevice(
                d_complete,
                &complete,
                sizeof(complete),
                device_id_,
                parent_stream));
            ASSERT_TRUE(copyHostToDevice(
                d_trace_publication,
                &trace_publication,
                sizeof(trace_publication),
                device_id_,
                parent_stream));

            auto transaction =
                cuda_context.createGraphCapture(transaction_stream);
            ASSERT_NE(transaction, nullptr);
            ASSERT_TRUE(transaction->beginCapture());
            ASSERT_TRUE(backend_->deviceCopyAsync(
                static_cast<int *>(d_control) + complete_index,
                d_complete,
                sizeof(int),
                device_id_,
                transaction_stream));
            ASSERT_TRUE(transaction->endCapture());

            auto conditional =
                cuda_context.createGraphCapture(conditional_stream);
            ASSERT_NE(conditional, nullptr);
            ASSERT_TRUE(conditional->beginCapture());
            ASSERT_TRUE(backend_->deviceCopyAsync(
                d_trace,
                d_trace_publication,
                sizeof(int),
                device_id_,
                conditional_stream));
            ASSERT_TRUE(conditional->endCapture());

            auto parent = cuda_context.createGraphCapture(parent_stream);
            ASSERT_NE(parent, nullptr);
            const std::array<DeviceControlledLoopFragment, 2> fragments = {{
                {
                    .name = "terminal transaction",
                    .capture = transaction.get(),
                },
                {
                    .name = "conditional maintenance",
                    .capture = conditional.get(),
                    .execution = DeviceControlledLoopFragmentExecution::
                        IfDeviceWordNonZero,
                    .condition_word_device =
                        static_cast<const uint32_t *>(d_condition),
                },
            }};
            ASSERT_TRUE(parent->buildDeviceControlledWhileLoop(
                fragments,
                DeviceControlledLoopPredicate{
                    .control_rows_device =
                        static_cast<const int *>(d_control),
                    .control_stride = control_stride,
                    .request_count = 1,
                    .healthy_index = healthy_index,
                    .complete_index = complete_index,
                }));
            ASSERT_TRUE(parent->instantiate());

            auto run_case = [&](uint32_t condition, int expected_trace)
            {
                int actual_trace = 0;
                ASSERT_TRUE(copyHostToDevice(
                    d_control,
                    initial_control.data(),
                    sizeof(initial_control),
                    device_id_,
                    parent_stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_trace,
                    &trace_sentinel,
                    sizeof(trace_sentinel),
                    device_id_,
                    parent_stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_condition,
                    &condition,
                    sizeof(condition),
                    device_id_,
                    parent_stream));
                ASSERT_TRUE(parent->launch());
                ASSERT_TRUE(copyDeviceToHost(
                    &actual_trace,
                    d_trace,
                    sizeof(actual_trace),
                    device_id_,
                    parent_stream));
                ASSERT_TRUE(backend_->synchronizeStream(
                    parent_stream,
                    device_id_));
                EXPECT_EQ(actual_trace, expected_trace);
            };

            run_case(condition_clear, trace_sentinel);
            run_case(condition_set, trace_publication);

            const std::array<DeviceControlledLoopBranch, 1> branches = {{
                {
                    .ordered_fragments = fragments,
                },
            }};
            ASSERT_TRUE(parent->supportsDeviceControlledSwitchWhileLoop());
            ASSERT_TRUE(parent->buildDeviceControlledSwitchWhileLoop(
                branches,
                DeviceControlledLoopPredicate{
                    .control_rows_device =
                        static_cast<const int *>(d_control),
                    .control_stride = control_stride,
                    .request_count = 1,
                    .healthy_index = healthy_index,
                    .complete_index = complete_index,
                },
                DeviceControlledLoopSwitch{
                    .control_rows_device = static_cast<int *>(d_control),
                    .control_stride = control_stride,
                    .request_count = 1,
                    .healthy_index = healthy_index,
                    .complete_index = complete_index,
                    .selector_index = selector_index,
                    .error_index = error_index,
                    .minimum_selector = 0,
                    .maximum_selector = 0,
                    .invalid_selector_error = 9101,
                }));
            ASSERT_TRUE(parent->instantiate());

            run_case(condition_clear, trace_sentinel);
            run_case(condition_set, trace_publication);

            parent.reset();
            conditional.reset();
            transaction.reset();
            cuda_context.destroyStream(conditional_stream);
            cuda_context.destroyStream(transaction_stream);
            cuda_context.destroyStream(parent_stream);
        });

        for (void *allocation : allocations)
            backend_->free(allocation, device_id_);
    }

    /**
     * @brief Prove a captured two-stream transaction remains native-WHILE compatible.
     *
     * CUDA stream capture represents each cross-stream handoff as an event-record
     * and event-wait operations while the streams are being captured. CUDA must
     * export those internal handoffs as ordinary graph dependencies rather than
     * event nodes, because conditional bodies reject event record/wait nodes. The
     * exported fragment and its parent must preserve the exact sequence:
     *
     * `primary write -> auxiliary write/completion -> primary terminal write`.
     *
     * Native node inspection proves that the source fragment is already
     * conditional-compatible. A root event wait would instead represent an
     * undeclared dependency outside the fragment and remains a hard error.
     */
    TEST_P(
        GPUSamplingTest,
        DeviceControlledGraphLoopExportsInternalEventHandoffsAsDependencies)
    {
        if (GetParam() != "CUDA")
            GTEST_SKIP() << "CUDA conditional graph coverage is backend-specific";

        constexpr int healthy_index = 0;
        constexpr int complete_index = 1;
        constexpr int control_stride = 2;
        constexpr int trace_word_count = 3;

        void *d_control = backend_->allocate(
            control_stride * sizeof(int), device_id_);
        void *d_trace = backend_->allocate(
            trace_word_count * sizeof(int), device_id_);
        void *d_first = backend_->allocate(sizeof(int), device_id_);
        void *d_second = backend_->allocate(sizeof(int), device_id_);
        void *d_third = backend_->allocate(sizeof(int), device_id_);
        void *d_complete = backend_->allocate(sizeof(int), device_id_);
        const std::array<void *, 6> allocations = {
            d_control,
            d_trace,
            d_first,
            d_second,
            d_third,
            d_complete};
        for (void *allocation : allocations)
            ASSERT_NE(allocation, nullptr);

        const std::array<int, control_stride> initial_control = {1, 0};
        const std::array<int, trace_word_count> empty_trace = {-1, -1, -1};
        constexpr int first = 101;
        constexpr int second = 202;
        constexpr int third = 303;
        constexpr int complete = 1;
        std::array<int, control_stride> actual_control{};
        std::array<int, trace_word_count> actual_trace{};

        auto &cuda_context =
            GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
        cuda_context.submitAndWait([&]()
        {
            void *const primary_stream = cuda_context.createStream();
            void *const transfer_stream = cuda_context.createStream();
            void *const compute_ready = cuda_context.createEvent();
            void *const transfer_done = cuda_context.createEvent();
            ASSERT_NE(primary_stream, nullptr);
            ASSERT_NE(transfer_stream, nullptr);
            ASSERT_NE(compute_ready, nullptr);
            ASSERT_NE(transfer_done, nullptr);

            ASSERT_TRUE(copyHostToDevice(
                d_control,
                initial_control.data(),
                sizeof(initial_control),
                device_id_,
                primary_stream));
            ASSERT_TRUE(copyHostToDevice(
                d_trace,
                empty_trace.data(),
                sizeof(empty_trace),
                device_id_,
                primary_stream));
            ASSERT_TRUE(copyHostToDevice(
                d_first, &first, sizeof(first), device_id_, primary_stream));
            ASSERT_TRUE(copyHostToDevice(
                d_second, &second, sizeof(second), device_id_, primary_stream));
            ASSERT_TRUE(copyHostToDevice(
                d_third, &third, sizeof(third), device_id_, primary_stream));
            ASSERT_TRUE(copyHostToDevice(
                d_complete,
                &complete,
                sizeof(complete),
                device_id_,
                primary_stream));
            ASSERT_TRUE(backend_->synchronizeStream(primary_stream, device_id_));

            auto fragment = cuda_context.createGraphCapture(primary_stream);
            ASSERT_NE(fragment, nullptr);
            ASSERT_TRUE(fragment->beginCapture());
            ASSERT_TRUE(backend_->deviceCopyAsync(
                static_cast<int *>(d_trace),
                d_first,
                sizeof(first),
                device_id_,
                primary_stream));
            ASSERT_TRUE(cuda_context.recordEventChecked(
                compute_ready, primary_stream));
            ASSERT_TRUE(cuda_context.waitEventChecked(
                compute_ready, transfer_stream));
            ASSERT_TRUE(backend_->deviceCopyAsync(
                static_cast<int *>(d_trace) + 1,
                d_second,
                sizeof(second),
                device_id_,
                transfer_stream));
            ASSERT_TRUE(backend_->deviceCopyAsync(
                static_cast<int *>(d_control) + complete_index,
                d_complete,
                sizeof(complete),
                device_id_,
                transfer_stream));
            ASSERT_TRUE(cuda_context.recordEventChecked(
                transfer_done, transfer_stream));
            ASSERT_TRUE(cuda_context.waitEventChecked(
                transfer_done, primary_stream));
            ASSERT_TRUE(backend_->deviceCopyAsync(
                static_cast<int *>(d_trace) + 2,
                d_third,
                sizeof(third),
                device_id_,
                primary_stream));
            ASSERT_TRUE(fragment->endCapture());
#ifdef HAVE_CUDA
            auto *const native_fragment =
                dynamic_cast<CUDAGraphCapture *>(fragment.get());
            ASSERT_NE(native_fragment, nullptr);
            size_t source_node_count = 0;
            ASSERT_EQ(
                cudaGraphGetNodes(
                    native_fragment->graph(), nullptr, &source_node_count),
                cudaSuccess);
            std::vector<cudaGraphNode_t> source_nodes(source_node_count);
            ASSERT_EQ(
                cudaGraphGetNodes(
                    native_fragment->graph(),
                    source_nodes.data(),
                    &source_node_count),
                cudaSuccess);
            size_t source_event_node_count = 0;
            for (cudaGraphNode_t source_node : source_nodes)
            {
                cudaGraphNodeType source_type = cudaGraphNodeTypeCount;
                ASSERT_EQ(
                    cudaGraphNodeGetType(source_node, &source_type),
                    cudaSuccess);
                source_event_node_count +=
                    source_type == cudaGraphNodeTypeWaitEvent ||
                            source_type == cudaGraphNodeTypeEventRecord
                        ? 1u
                        : 0u;
            }
            ASSERT_EQ(source_event_node_count, 0u)
                << "Internal stream handoffs must export as graph dependencies";
#endif

            auto parent = cuda_context.createGraphCapture(primary_stream);
            ASSERT_NE(parent, nullptr);
            const std::array<DeviceControlledLoopFragment, 1> fragments = {{
                {.name = "two-stream event transaction", .capture = fragment.get()},
            }};
            ASSERT_TRUE(parent->buildDeviceControlledWhileLoop(
                fragments,
                DeviceControlledLoopPredicate{
                    .control_rows_device = static_cast<const int *>(d_control),
                    .control_stride = control_stride,
                    .request_count = 1,
                    .healthy_index = healthy_index,
                    .complete_index = complete_index}));
            ASSERT_TRUE(parent->instantiate());
            ASSERT_TRUE(parent->launch());
            ASSERT_TRUE(copyDeviceToHost(
                actual_trace.data(),
                d_trace,
                sizeof(actual_trace),
                device_id_,
                primary_stream));
            ASSERT_TRUE(copyDeviceToHost(
                actual_control.data(),
                d_control,
                sizeof(actual_control),
                device_id_,
                primary_stream));
            ASSERT_TRUE(backend_->synchronizeStream(primary_stream, device_id_));

            parent.reset();
            fragment.reset();
            cuda_context.destroyEvent(transfer_done);
            cuda_context.destroyEvent(compute_ready);
            cuda_context.destroyStream(transfer_stream);
            cuda_context.destroyStream(primary_stream);
        });

        EXPECT_EQ(actual_trace, (std::array<int, trace_word_count>{
                                    first,
                                    second,
                                    third}));
        EXPECT_EQ(actual_control[healthy_index], 1);
        EXPECT_EQ(actual_control[complete_index], 1);

        for (void *allocation : allocations)
            backend_->free(allocation, device_id_);
    }

    /**
     * @brief Prove CUDA selects complete WHILE transactions from device state.
     *
     * Branch one writes the first trace word and publishes selector two through
     * a captured D2D copy. Branch two writes the second trace word and publishes
     * request completion. One parent launch must therefore execute both bodies
     * in order without a host selector read. A malformed selector is then
     * required to invalidate the controller and execute neither branch.
     */
    TEST_P(
        GPUSamplingTest,
        DeviceControlledSwitchWhileSelectsTransactionsAndFailsInvalidDepth)
    {
        if (GetParam() != "CUDA")
            GTEST_SKIP() << "CUDA SWITCH/WHILE coverage is backend-specific";

        constexpr int healthy_index = 0;
        constexpr int complete_index = 1;
        constexpr int selector_index = 2;
        constexpr int error_index = 3;
        constexpr int control_stride = 4;
        constexpr int invalid_selector_error = 7719;

        void *d_control = backend_->allocate(
            control_stride * sizeof(int), device_id_);
        void *d_trace = backend_->allocate(2 * sizeof(int), device_id_);
        void *d_trace_one = backend_->allocate(sizeof(int), device_id_);
        void *d_trace_two = backend_->allocate(sizeof(int), device_id_);
        void *d_selector_two = backend_->allocate(sizeof(int), device_id_);
        void *d_complete = backend_->allocate(sizeof(int), device_id_);
        const std::array<void *, 6> allocations = {
            d_control,
            d_trace,
            d_trace_one,
            d_trace_two,
            d_selector_two,
            d_complete};
        for (void *allocation : allocations)
            ASSERT_NE(allocation, nullptr);

        const int trace_one = 101;
        const int trace_two = 202;
        const int selector_two = 2;
        const int complete = 1;
        const std::array<int, control_stride> initial_control = {
            1,
            0,
            1,
            0};
        const std::array<int, 2> empty_trace = {-1, -1};

        std::array<int, control_stride> actual_control{};
        std::array<int, 2> actual_trace{};
        auto &cuda_context =
            GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
        cuda_context.submitAndWait([&]()
        {
            void *const parent_stream = cuda_context.defaultStream();
            void *const branch_one_stream = cuda_context.createStream();
            void *const branch_two_stream = cuda_context.createStream();
            ASSERT_NE(parent_stream, nullptr);
            ASSERT_NE(branch_one_stream, nullptr);
            ASSERT_NE(branch_two_stream, nullptr);

            ASSERT_TRUE(copyHostToDevice(
                d_control,
                initial_control.data(),
                sizeof(initial_control),
                device_id_,
                parent_stream));
            ASSERT_TRUE(copyHostToDevice(
                d_trace,
                empty_trace.data(),
                sizeof(empty_trace),
                device_id_,
                parent_stream));
            ASSERT_TRUE(copyHostToDevice(
                d_trace_one,
                &trace_one,
                sizeof(trace_one),
                device_id_,
                parent_stream));
            ASSERT_TRUE(copyHostToDevice(
                d_trace_two,
                &trace_two,
                sizeof(trace_two),
                device_id_,
                parent_stream));
            ASSERT_TRUE(copyHostToDevice(
                d_selector_two,
                &selector_two,
                sizeof(selector_two),
                device_id_,
                parent_stream));
            ASSERT_TRUE(copyHostToDevice(
                d_complete,
                &complete,
                sizeof(complete),
                device_id_,
                parent_stream));
            ASSERT_TRUE(backend_->synchronizeStream(parent_stream, device_id_));

            auto branch_one =
                cuda_context.createGraphCapture(branch_one_stream);
            ASSERT_NE(branch_one, nullptr);
            ASSERT_TRUE(branch_one->beginCapture());
            ASSERT_TRUE(backend_->deviceCopyAsync(
                d_trace,
                d_trace_one,
                sizeof(int),
                device_id_,
                branch_one_stream));
            ASSERT_TRUE(backend_->deviceCopyAsync(
                static_cast<int *>(d_control) + selector_index,
                d_selector_two,
                sizeof(int),
                device_id_,
                branch_one_stream));
            ASSERT_TRUE(branch_one->endCapture());
            ASSERT_GT(branch_one->nodeCount(), 0u);

            auto branch_two =
                cuda_context.createGraphCapture(branch_two_stream);
            ASSERT_NE(branch_two, nullptr);
            ASSERT_TRUE(branch_two->beginCapture());
            ASSERT_TRUE(backend_->deviceCopyAsync(
                static_cast<int *>(d_trace) + 1,
                d_trace_two,
                sizeof(int),
                device_id_,
                branch_two_stream));
            ASSERT_TRUE(backend_->deviceCopyAsync(
                static_cast<int *>(d_control) + complete_index,
                d_complete,
                sizeof(int),
                device_id_,
                branch_two_stream));
            ASSERT_TRUE(branch_two->endCapture());
            ASSERT_GT(branch_two->nodeCount(), 0u);

            const std::array<DeviceControlledLoopFragment, 1>
                branch_one_fragments = {{
                    {.name = "selector one", .capture = branch_one.get()},
                }};
            const std::array<DeviceControlledLoopFragment, 1>
                branch_two_fragments = {{
                    {.name = "selector two", .capture = branch_two.get()},
                }};
            const std::array<DeviceControlledLoopBranch, 3> branches = {
                DeviceControlledLoopBranch{},
                DeviceControlledLoopBranch{
                    .ordered_fragments = branch_one_fragments},
                DeviceControlledLoopBranch{
                    .ordered_fragments = branch_two_fragments}};

            auto parent = cuda_context.createGraphCapture(parent_stream);
            ASSERT_NE(parent, nullptr);
            ASSERT_TRUE(parent->supportsDeviceControlledSwitchWhileLoop());
            ASSERT_TRUE(parent->buildDeviceControlledSwitchWhileLoop(
                branches,
                DeviceControlledLoopPredicate{
                    .control_rows_device =
                        static_cast<const int *>(d_control),
                    .control_stride = control_stride,
                    .request_count = 1,
                    .healthy_index = healthy_index,
                    .complete_index = complete_index},
                DeviceControlledLoopSwitch{
                    .control_rows_device = static_cast<int *>(d_control),
                    .control_stride = control_stride,
                    .request_count = 1,
                    .healthy_index = healthy_index,
                    .complete_index = complete_index,
                    .selector_index = selector_index,
                    .error_index = error_index,
                    .minimum_selector = 1,
                    .maximum_selector = 2,
                    .invalid_selector_error = invalid_selector_error}));
            ASSERT_TRUE(parent->instantiate());
            ASSERT_TRUE(parent->launch());
            ASSERT_TRUE(parent->launch())
                << "A terminal controller must execute zero later iterations";
            ASSERT_TRUE(copyDeviceToHost(
                actual_trace.data(),
                d_trace,
                sizeof(actual_trace),
                device_id_,
                parent_stream));
            ASSERT_TRUE(copyDeviceToHost(
                actual_control.data(),
                d_control,
                sizeof(actual_control),
                device_id_,
                parent_stream));
            ASSERT_TRUE(backend_->synchronizeStream(parent_stream, device_id_));

            EXPECT_EQ(actual_trace, (std::array<int, 2>{101, 202}));
            EXPECT_EQ(actual_control[healthy_index], 1);
            EXPECT_EQ(actual_control[complete_index], 1);
            EXPECT_EQ(actual_control[selector_index], 2);
            EXPECT_EQ(actual_control[error_index], 0);

            const std::array<int, control_stride> invalid_control = {
                1,
                0,
                3,
                0};
            ASSERT_TRUE(copyHostToDevice(
                d_control,
                invalid_control.data(),
                sizeof(invalid_control),
                device_id_,
                parent_stream));
            ASSERT_TRUE(copyHostToDevice(
                d_trace,
                empty_trace.data(),
                sizeof(empty_trace),
                device_id_,
                parent_stream));
            ASSERT_TRUE(parent->launch());
            ASSERT_TRUE(copyDeviceToHost(
                actual_trace.data(),
                d_trace,
                sizeof(actual_trace),
                device_id_,
                parent_stream));
            ASSERT_TRUE(copyDeviceToHost(
                actual_control.data(),
                d_control,
                sizeof(actual_control),
                device_id_,
                parent_stream));
            ASSERT_TRUE(backend_->synchronizeStream(parent_stream, device_id_));

            EXPECT_EQ(actual_trace, empty_trace);
            EXPECT_EQ(actual_control[healthy_index], 0);
            EXPECT_EQ(actual_control[complete_index], 1);
            EXPECT_EQ(actual_control[error_index], invalid_selector_error);

            parent.reset();
            branch_two.reset();
            branch_one.reset();
            cuda_context.destroyStream(branch_two_stream);
            cuda_context.destroyStream(branch_one_stream);
        });

        for (void *allocation : allocations)
            backend_->free(allocation, device_id_);
    }

    /**
     * @brief Prove chained MTP sidecars gather request-major columns on device.
     *
     * The production proposal matrix stores all depths for request zero, then all
     * depths for request one. A chained grouped sidecar therefore consumes one
     * strided column. This regression exercises that exact layout and verifies
     * that the same launch derives absolute positions from resident base rows.
     */
    TEST_P(GPUSamplingTest, MTPBatchedSidecarInputCompositionUsesStridedDeviceRows)
    {
        constexpr int request_count = 4;
        constexpr int draft_depth = 3;
        const std::array<int32_t, request_count * draft_depth> proposal_matrix = {
            101, 102, 103,
            201, 202, 203,
            301, 302, 303,
            401, 402, 403};
        const std::array<int32_t, request_count> base_positions = {7, 11, 19, 23};

        void *d_proposals = backend_->allocate(
            proposal_matrix.size() * sizeof(int32_t), device_id_);
        void *d_base_positions = backend_->allocate(
            base_positions.size() * sizeof(int32_t), device_id_);
        void *d_conditions = backend_->allocate(
            request_count * sizeof(int32_t), device_id_);
        void *d_positions = backend_->allocate(
            request_count * sizeof(int32_t), device_id_);
        ASSERT_NE(d_proposals, nullptr);
        ASSERT_NE(d_base_positions, nullptr);
        ASSERT_NE(d_conditions, nullptr);
        ASSERT_NE(d_positions, nullptr);

        auto run = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_proposals,
                    proposal_matrix.data(),
                    proposal_matrix.size() * sizeof(int32_t),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_base_positions,
                    base_positions.data(),
                    base_positions.size() * sizeof(int32_t),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->enqueuePrepareMTPBatchedSidecarInputs(
                    static_cast<const int32_t *>(d_proposals) + 1,
                    draft_depth,
                    d_base_positions,
                    /*position_offset=*/1,
                    request_count,
                    device_id_,
                    stream,
                    d_conditions,
                    d_positions));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };
        if (GetParam() == "CUDA")
            run(GPUDeviceContextPool::instance().getNvidiaContext(device_id_));
        else
            run(GPUDeviceContextPool::instance().getAMDContext(device_id_));

        std::array<int32_t, request_count> conditions{};
        std::array<int32_t, request_count> positions{};
        ASSERT_TRUE(copyDeviceToHost(
            conditions.data(), d_conditions, sizeof(conditions), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            positions.data(), d_positions, sizeof(positions), device_id_));
        EXPECT_EQ(conditions,
                  (std::array<int32_t, request_count>{102, 202, 302, 402}));
        EXPECT_EQ(positions,
                  (std::array<int32_t, request_count>{8, 12, 20, 24}));

        backend_->free(d_positions, device_id_);
        backend_->free(d_conditions, device_id_);
        backend_->free(d_base_positions, device_id_);
        backend_->free(d_proposals, device_id_);
    }

    /**
     * @brief Prove grouped verifier positions follow mutable device KV counts.
     *
     * Accepted-state publication intentionally does not update a host position
     * mirror on GPU. The next grouped verifier must expand each request's
     * canonical device KV count into a contiguous absolute-position row before
     * captured replay. This test captures that production primitive once, then
     * changes only the device-owned base positions. Byte-exact output on every
     * replay catches stale host scalars, a captured initial value, incorrect
     * request-major indexing, and a missing long-context offset.
     */
    TEST_P(
        GPUSamplingTest,
        MTPGroupedVerifierPositionsTrackDeviceKVCountsAcrossGraphReplays)
    {
        constexpr int request_count = 4;
        constexpr int padded_seq_len = 5;
        constexpr int total_rows = request_count * padded_seq_len;
        const std::array<std::array<int32_t, request_count>, 3> live_positions = {{
            {{7, 11, 19, 23}},
            {{4095, 8191, 12287, 16383}},
            {{2383, 2387, 2391, 2395}},
        }};

        void *d_live_positions = backend_->allocate(
            request_count * sizeof(int32_t), device_id_);
        void *d_verifier_positions = backend_->allocate(
            total_rows * sizeof(int32_t), device_id_);
        ASSERT_NE(d_live_positions, nullptr);
        ASSERT_NE(d_verifier_positions, nullptr);

        std::array<std::array<int32_t, total_rows>, live_positions.size()>
            observed{};

        auto run = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueuePrepareMTPVerifierPositionIds(
                    d_live_positions,
                    request_count,
                    padded_seq_len,
                    device_id_,
                    stream,
                    d_verifier_positions));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                for (size_t replay = 0; replay < live_positions.size(); ++replay)
                {
                    ASSERT_TRUE(copyHostToDevice(
                        d_live_positions,
                        live_positions[replay].data(),
                        request_count * sizeof(int32_t),
                        device_id_,
                        stream));
                    ASSERT_TRUE(capture->launch());
                    ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                    ASSERT_TRUE(copyDeviceToHost(
                        observed[replay].data(),
                        d_verifier_positions,
                        total_rows * sizeof(int32_t),
                        device_id_));
                }
            });
        };
        if (GetParam() == "CUDA")
            run(GPUDeviceContextPool::instance().getNvidiaContext(device_id_));
        else
            run(GPUDeviceContextPool::instance().getAMDContext(device_id_));

        for (size_t replay = 0; replay < live_positions.size(); ++replay)
        {
            for (int request = 0; request < request_count; ++request)
            {
                for (int token = 0; token < padded_seq_len; ++token)
                {
                    const int flat_row =
                        request * padded_seq_len + token;
                    EXPECT_EQ(
                        observed[replay][static_cast<size_t>(flat_row)],
                        live_positions[replay][static_cast<size_t>(request)] +
                            token)
                        << "replay=" << replay
                        << " request=" << request
                        << " token=" << token;
                }
            }
        }

        backend_->free(d_verifier_positions, device_id_);
        backend_->free(d_live_positions, device_id_);
    }

    /**
     * @brief Prove one captured geometry launch publishes exact ragged widths.
     *
     * The production grouped verifier already publishes its compact physical
     * logit rows before replay. Short-conv and GDN must derive their recurrent
     * state masks from those same device rows, rather than from a stale prompt
     * length or a participant-external host vector. This test changes both live
     * KV positions and ragged row geometry between graph replays while retaining
     * every captured pointer and launch parameter.
     */
    TEST_P(
        GPUSamplingTest,
        MTPGroupedVerifierGeometryDerivesRaggedLengthsAcrossGraphReplays)
    {
        constexpr int request_count = 4;
        constexpr int padded_seq_len = 5;
        constexpr int total_rows = request_count * padded_seq_len;
        constexpr int valid_row_count = 12;
        const std::array<std::array<int32_t, request_count>, 3> live_positions = {{
            {{7, 11, 19, 23}},
            {{4095, 8191, 12287, 16383}},
            {{2383, 2387, 2391, 2395}},
        }};
        const std::array<std::array<int32_t, request_count>, 3> expected_lengths = {{
            {{5, 2, 4, 1}},
            {{1, 5, 3, 3}},
            {{4, 4, 2, 2}},
        }};

        std::array<std::array<int32_t, valid_row_count>, 3> valid_rows{};
        for (size_t replay = 0; replay < valid_rows.size(); ++replay)
        {
            int cursor = 0;
            for (int request = 0; request < request_count; ++request)
            {
                for (int token = 0;
                     token < expected_lengths[replay][static_cast<size_t>(request)];
                     ++token)
                {
                    valid_rows[replay][static_cast<size_t>(cursor++)] =
                        request * padded_seq_len + token;
                }
            }
            ASSERT_EQ(cursor, valid_row_count);
        }

        void *d_live_positions = backend_->allocate(
            request_count * sizeof(int32_t), device_id_);
        void *d_valid_rows = backend_->allocate(
            valid_row_count * sizeof(int32_t), device_id_);
        void *d_verifier_positions = backend_->allocate(
            total_rows * sizeof(int32_t), device_id_);
        void *d_request_lengths = backend_->allocate(
            request_count * sizeof(int32_t), device_id_);
        ASSERT_NE(d_live_positions, nullptr);
        ASSERT_NE(d_valid_rows, nullptr);
        ASSERT_NE(d_verifier_positions, nullptr);
        ASSERT_NE(d_request_lengths, nullptr);

        std::array<std::array<int32_t, total_rows>, live_positions.size()>
            observed_positions{};
        std::array<std::array<int32_t, request_count>, live_positions.size()>
            observed_lengths{};

        auto run = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueuePrepareMTPVerifierGeometry(
                    d_live_positions,
                    d_valid_rows,
                    valid_row_count,
                    /*generation_control_device=*/nullptr,
                    /*generation_control_stride=*/0,
                    request_count,
                    padded_seq_len,
                    device_id_,
                    stream,
                    d_verifier_positions,
                    d_request_lengths));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                for (size_t replay = 0; replay < live_positions.size(); ++replay)
                {
                    ASSERT_TRUE(copyHostToDevice(
                        d_live_positions,
                        live_positions[replay].data(),
                        request_count * sizeof(int32_t),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_valid_rows,
                        valid_rows[replay].data(),
                        valid_row_count * sizeof(int32_t),
                        device_id_,
                        stream));
                    ASSERT_TRUE(capture->launch());
                    ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                    ASSERT_TRUE(copyDeviceToHost(
                        observed_positions[replay].data(),
                        d_verifier_positions,
                        total_rows * sizeof(int32_t),
                        device_id_));
                    ASSERT_TRUE(copyDeviceToHost(
                        observed_lengths[replay].data(),
                        d_request_lengths,
                        request_count * sizeof(int32_t),
                        device_id_));
                }
            });
        };
        if (GetParam() == "CUDA")
            run(GPUDeviceContextPool::instance().getNvidiaContext(device_id_));
        else
            run(GPUDeviceContextPool::instance().getAMDContext(device_id_));

        for (size_t replay = 0; replay < live_positions.size(); ++replay)
        {
            EXPECT_EQ(observed_lengths[replay], expected_lengths[replay]);
            for (int request = 0; request < request_count; ++request)
            {
                for (int token = 0; token < padded_seq_len; ++token)
                {
                    const int flat_row = request * padded_seq_len + token;
                    EXPECT_EQ(
                        observed_positions[replay][static_cast<size_t>(flat_row)],
                        live_positions[replay][static_cast<size_t>(request)] + token)
                        << "replay=" << replay
                        << " request=" << request
                        << " token=" << token;
                }
            }
        }

        backend_->free(d_request_lengths, device_id_);
        backend_->free(d_verifier_positions, device_id_);
        backend_->free(d_valid_rows, device_id_);
        backend_->free(d_live_positions, device_id_);
    }

    /**
     * @brief Prove captured verifier widths follow the resident depth controller.
     *
     * A maximum-capacity verifier graph must serve every logical MTP depth.
     * This regression captures one sixteen-row geometry kernel, changes only
     * controller rows between replays, and checks exact request-major strides on
     * both GPU backends. The final replay corrupts one depth/width pair and
     * proves the kernel poisons that request instead of consulting the static
     * physical row count.
     */
    TEST_P(
        GPUSamplingTest,
        MTPGroupedVerifierGeometryUsesDeviceDepthAcrossMaximumGraphReplays)
    {
        using namespace sampling_math;

        constexpr int request_count = 4;
        constexpr int padded_seq_len = 16;
        constexpr int total_rows = request_count * padded_seq_len;
        constexpr int control_stride = kDeviceGenerationControlCount;
        const std::array<int32_t, request_count> live_positions = {
            31, 4095, 8191, 16383};
        const std::array<std::array<int, request_count>, 2> replay_depths = {{
            {{1, 2, 4, 8}},
            {{15, 8, 4, 1}},
        }};

        std::array<int, request_count * control_stride> controls{};
        std::array<int32_t, total_rows> observed_positions{};
        std::array<int32_t, request_count> observed_lengths{};

        void *d_live_positions = backend_->allocate(
            sizeof(live_positions), device_id_);
        void *d_controls = backend_->allocate(sizeof(controls), device_id_);
        void *d_verifier_positions = backend_->allocate(
            sizeof(observed_positions), device_id_);
        void *d_request_lengths = backend_->allocate(
            sizeof(observed_lengths), device_id_);
        ASSERT_NE(d_live_positions, nullptr);
        ASSERT_NE(d_controls, nullptr);
        ASSERT_NE(d_verifier_positions, nullptr);
        ASSERT_NE(d_request_lengths, nullptr);

        auto initialize_controls = [&](const auto &depths)
        {
            controls.fill(0);
            for (int request = 0; request < request_count; ++request)
            {
                int *const row =
                    controls.data() + request * control_stride;
                ASSERT_TRUE(initialize_device_generation_control(
                    /*max_new_tokens=*/32,
                    /*response_capacity=*/32,
                    DeviceGenerationDepthPolicy::fixed(
                        depths[static_cast<size_t>(request)]),
                    row));
            }
        };

        auto run = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_live_positions,
                    live_positions.data(),
                    sizeof(live_positions),
                    device_id_,
                    stream));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueuePrepareMTPVerifierGeometry(
                    d_live_positions,
                    /*valid_graph_rows_device=*/nullptr,
                    /*valid_graph_row_count=*/total_rows,
                    d_controls,
                    control_stride,
                    request_count,
                    padded_seq_len,
                    device_id_,
                    stream,
                    d_verifier_positions,
                    d_request_lengths));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                for (const auto &depths : replay_depths)
                {
                    initialize_controls(depths);
                    ASSERT_TRUE(copyHostToDevice(
                        d_controls,
                        controls.data(),
                        sizeof(controls),
                        device_id_,
                        stream));
                    ASSERT_TRUE(capture->launch());
                    ASSERT_TRUE(copyDeviceToHost(
                        observed_positions.data(),
                        d_verifier_positions,
                        sizeof(observed_positions),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        observed_lengths.data(),
                        d_request_lengths,
                        sizeof(observed_lengths),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        controls.data(),
                        d_controls,
                        sizeof(controls),
                        device_id_,
                        stream));
                    ASSERT_TRUE(
                        backend_->synchronizeStream(stream, device_id_));

                    for (int request = 0; request < request_count; ++request)
                    {
                        EXPECT_EQ(
                            observed_lengths[static_cast<size_t>(request)],
                            depths[static_cast<size_t>(request)] + 1);
                        EXPECT_EQ(
                            controls[static_cast<size_t>(request) *
                                         control_stride +
                                     kDeviceGenerationControlOk],
                            1);
                        for (int token = 0;
                             token < padded_seq_len;
                             ++token)
                        {
                            const int flat =
                                request * padded_seq_len + token;
                            EXPECT_EQ(
                                observed_positions[
                                    static_cast<size_t>(flat)],
                                live_positions[
                                    static_cast<size_t>(request)] + token);
                        }
                    }
                }

                initialize_controls(replay_depths.front());
                controls[kDeviceGenerationControlActiveVerifierRowCount] = 7;
                ASSERT_TRUE(copyHostToDevice(
                    d_controls,
                    controls.data(),
                    sizeof(controls),
                    device_id_,
                    stream));
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(copyDeviceToHost(
                    observed_lengths.data(),
                    d_request_lengths,
                    sizeof(observed_lengths),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyDeviceToHost(
                    controls.data(),
                    d_controls,
                    sizeof(controls),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_EQ(observed_lengths[0], 0);
                EXPECT_EQ(
                    controls[kDeviceGenerationControlOk],
                    0);
                EXPECT_EQ(
                    controls[kDeviceGenerationControlRequestComplete],
                    1);
                EXPECT_EQ(
                    controls[kDeviceGenerationControlErrorCode],
                    static_cast<int>(
                        DeviceGenerationError::InvalidDepthSelector));
            });
        };

        if (GetParam() == "CUDA")
            run(GPUDeviceContextPool::instance().getNvidiaContext(device_id_));
        else
            run(GPUDeviceContextPool::instance().getAMDContext(device_id_));

        backend_->free(d_request_lengths, device_id_);
        backend_->free(d_verifier_positions, device_id_);
        backend_->free(d_controls, device_id_);
        backend_->free(d_live_positions, device_id_);
    }

    /**
     * @brief Prove one controller-owned verifier prelude serves every M=2..16.
     *
     * This is intentionally an integration test rather than a mocked stage
     * test.  It binds a production FP16 ring cache, captures the exact typed
     * preparation stage at its maximum physical width, and then changes only
     * token bytes, controller words, and canonical cache metadata behind stable
     * device addresses. The unchanged graph must publish the active token
     * prefix, zero every inactive suffix byte, expand absolute positions, write
     * the logical request width, publish the same serial-visible commit budget
     * for both reducer modes, and capture both the scalar base count and the
     * opaque all-layer KV checkpoint. The request penalty policy is published
     * separately and must remain untouched by every replay.
     *
     * Covering every supported speculative width prevents endpoint-only tests
     * from leaving holes at intermediate depths. A malformed depth/row pair is
     * also replayed through the same executable and must poison the controller.
     * CUDA and ROCm run the same production-stage body so the lifecycle contract
     * cannot drift by backend.
     */
    TEST_P(
        GPUSamplingTest,
        CapturedControllerOwnedMTPVerifierPreparationIsMTotal)
    {
        constexpr int max_verifier_width = 16;
        constexpr int initial_cached_tokens = 3;
        constexpr int num_layers = 1;
        constexpr int n_kv_heads = 2;
        constexpr int head_dim = 16;
        constexpr int max_seq_len = 64;
        constexpr float presence_penalty = 0.25F;
        constexpr float frequency_penalty = 0.125F;

        const DeviceId execution_device =
            GetParam() == "CUDA"
                ? DeviceId::cuda(device_id_)
                : DeviceId::rocm(device_id_);
        auto execution_context = IDeviceContext::create(execution_device);
        ASSERT_NE(execution_context, nullptr);

        llaminar::v2::kernels::KVCacheConfig cache_config;
        cache_config.precision = ActivationPrecision::FP16;
        cache_config.device = execution_device;
        cache_config.num_layers = num_layers;
        cache_config.batch_size = 1;
        cache_config.max_seq_len = max_seq_len;
        cache_config.n_kv_heads = n_kv_heads;
        cache_config.head_dim = head_dim;
        auto cache =
            llaminar::v2::kernels::KernelFactory::createKVCache(cache_config);
        ASSERT_NE(cache, nullptr);

        auto *workspace_consumer =
            dynamic_cast<IWorkspaceConsumer *>(cache.get());
        ASSERT_NE(workspace_consumer, nullptr)
            << "GPU KV cache must declare its complete persistent workspace";
        const WorkspaceRequirements cache_requirements =
            workspace_consumer->getWorkspaceRequirements(
                max_seq_len,
                /*n=*/1,
                head_dim);
        DeviceWorkspaceManager cache_workspace(
            execution_device,
            cache_requirements.total_bytes_with_alignment() + 4096U);
        ASSERT_TRUE(cache_workspace.allocate(cache_requirements));
        workspace_consumer->bindWorkspace(&cache_workspace);

        const size_t checkpoint_bytes =
            cache->deviceSequenceStateCheckpointBytes();
        ASSERT_GT(checkpoint_bytes, 0U)
            << "GPU MTP requires a complete device-resident KV checkpoint";
        ASSERT_NE(cache->deviceSequenceCachedTokenCountPtr(0), nullptr);

        auto initial_k = test::TestTensorFactory::createFP16Random(
            {initial_cached_tokens,
             static_cast<size_t>(n_kv_heads),
             static_cast<size_t>(head_dim)},
            -1.0F,
            1.0F,
            20260801U);
        auto initial_v = test::TestTensorFactory::createFP16Random(
            {initial_cached_tokens,
             static_cast<size_t>(n_kv_heads),
             static_cast<size_t>(head_dim)},
            -1.0F,
            1.0F,
            20260802U);
        auto continuation_k = test::TestTensorFactory::createFP16Random(
            {1U,
             static_cast<size_t>(n_kv_heads),
             static_cast<size_t>(head_dim)},
            -1.0F,
            1.0F,
            20260803U);
        auto continuation_v = test::TestTensorFactory::createFP16Random(
            {1U,
             static_cast<size_t>(n_kv_heads),
             static_cast<size_t>(head_dim)},
            -1.0F,
            1.0F,
            20260804U);

        void *d_first_token = backend_->allocate(sizeof(int32_t), device_id_);
        void *d_draft_tokens = backend_->allocate(
            (max_verifier_width - 1) * sizeof(int32_t), device_id_);
        void *d_verifier_tokens = backend_->allocate(
            max_verifier_width * sizeof(int32_t), device_id_);
        void *d_position_ids = backend_->allocate(
            max_verifier_width * sizeof(int32_t), device_id_);
        void *d_request_length = backend_->allocate(
            sizeof(int32_t), device_id_);
        void *d_base_snapshot = backend_->allocate(
            sizeof(int32_t), device_id_);
        void *d_checkpoint = backend_->allocate(
            checkpoint_bytes, device_id_);
        void *d_penalty_policy = backend_->allocate(
            sizeof(MTPGreedyPenaltyPolicy), device_id_);
        void *d_generation_control = backend_->allocate(
            sampling_math::kDeviceGenerationControlCount * sizeof(int),
            device_id_);
        const std::array<void *, 9> allocations = {
            d_first_token,
            d_draft_tokens,
            d_verifier_tokens,
            d_position_ids,
            d_request_length,
            d_base_snapshot,
            d_checkpoint,
            d_penalty_policy,
            d_generation_control,
        };
        for (void *allocation : allocations)
            ASSERT_NE(allocation, nullptr);

        std::array<int32_t, max_verifier_width - 1> draft_tokens{};
        std::array<int32_t, max_verifier_width> observed_tokens{};
        std::array<int32_t, max_verifier_width> observed_positions{};
        std::array<int, sampling_math::kDeviceGenerationControlCount>
            generation_control{};
        std::vector<uint8_t> checkpoint_before(checkpoint_bytes);
        std::vector<uint8_t> checkpoint_after(checkpoint_bytes);

        auto run = [&](IWorkerGPUContext &worker_context)
        {
            worker_context.submitAndWait([&]()
            {
                void *const stream = worker_context.defaultStream();
                ASSERT_NE(stream, nullptr);

                auto &transfer = TransferEngine::instance();
                ASSERT_TRUE(
                    transfer.uploadFull(
                        initial_k.get(), execution_device, stream)
                        .success);
                ASSERT_TRUE(
                    transfer.uploadFull(
                        initial_v.get(), execution_device, stream)
                        .success);
                ASSERT_TRUE(
                    transfer.uploadFull(
                        continuation_k.get(), execution_device, stream)
                        .success);
                ASSERT_TRUE(
                    transfer.uploadFull(
                        continuation_v.get(), execution_device, stream)
                        .success);
                ASSERT_TRUE(cache->appendWithStream(
                    0,
                    0,
                    initial_k.get(),
                    initial_v.get(),
                    initial_cached_tokens,
                    stream));
                ASSERT_TRUE(
                    backend_->enqueueConfigureMTPGreedyPenaltyPolicyDevice(
                        d_penalty_policy,
                        presence_penalty,
                        frequency_penalty,
                        /*first_token_already_in_history=*/true,
                        device_id_,
                        stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                const MTPVerifierPreparationStage::TokenRowBinding token_row{
                    .first_token_device =
                        static_cast<const int32_t *>(d_first_token),
                    .draft_tokens_device =
                        static_cast<const int32_t *>(d_draft_tokens),
                    .destination_device =
                        static_cast<int32_t *>(d_verifier_tokens),
                    .draft_token_count = max_verifier_width - 1,
                };
                const MTPVerifierPreparationStage::MainKVCheckpointBinding
                    checkpoint_binding{
                        .cache = cache.get(),
                        .sequence_index = 0,
                        .checkpoint_device = d_checkpoint,
                        .checkpoint_bytes = checkpoint_bytes,
                    };
                const std::array token_rows{token_row};
                const std::array checkpoint_bindings{checkpoint_binding};

                MTPVerifierPreparationStage::Params stage_params;
                stage_params.device_id = execution_device;
                stage_params.backend = backend_;
                stage_params.token_rows = token_rows;
                stage_params.request_count = 1;
                stage_params.padded_seq_len = max_verifier_width;
                stage_params.base_cached_tokens_device =
                    cache->deviceSequenceCachedTokenCountPtr(0);
                stage_params.valid_graph_rows_device = nullptr;
                stage_params.valid_graph_row_count = max_verifier_width;
                stage_params.generation_control_device =
                    static_cast<int *>(d_generation_control);
                stage_params.generation_control_stride =
                    sampling_math::kDeviceGenerationControlCount;
                stage_params.position_ids_device =
                    static_cast<int32_t *>(d_position_ids);
                stage_params.request_lengths_device =
                    static_cast<int32_t *>(d_request_length);
                stage_params.base_cached_tokens_snapshot_device =
                    static_cast<int32_t *>(d_base_snapshot);
                stage_params.main_kv_checkpoints = checkpoint_bindings;
                MTPVerifierPreparationStage stage(stage_params);
                EXPECT_THROW(
                    stage.execute(execution_context.get()),
                    std::logic_error)
                    << "captured verifier preparation must reject a null stream";
                stage.setGPUStream(stream);

                /*
                 * Logical depth is replay data in controller-owned mode. Prove
                 * the stage's graph identity keeps only the physical bucket and
                 * stable pointer topology, rather than accidentally recapturing
                 * for a new draft count or setup-time logical row count.
                 */
                const MTPVerifierPreparationStage::TokenRowBinding
                    narrow_token_row{
                        .first_token_device = token_row.first_token_device,
                        .draft_tokens_device = token_row.draft_tokens_device,
                        .destination_device = token_row.destination_device,
                        .draft_token_count = 1,
                    };
                const std::array narrow_token_rows{narrow_token_row};
                auto narrow_stage_params = stage_params;
                narrow_stage_params.token_rows = narrow_token_rows;
                narrow_stage_params.valid_graph_row_count = 2;
                EXPECT_TRUE(stage.hasSameCaptureIdentity(narrow_stage_params));

                auto capture = worker_context.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                const bool stage_enqueued =
                    stage.execute(execution_context.get());
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(stage_enqueued);
                ASSERT_GE(capture->nodeCount(), 2U)
                    << "checkpoint and fused controlled-row preparation were not captured";

                std::vector<GPUGraphKernelNodeInfo> captured_kernels;
                std::string inventory_error;
                ASSERT_TRUE(capture->inspectKernelNodes(
                    captured_kernels,
                    &inventory_error))
                    << inventory_error;
                ASSERT_FALSE(captured_kernels.empty())
                    << "the production verifier-preparation capture must expose its kernel nodes";
                for (const GPUGraphKernelNodeInfo &kernel : captured_kernels)
                {
                    EXPECT_TRUE(kernel.valid())
                        << "invalid captured launch at " << kernel.graph_path;
                    EXPECT_TRUE(kernel.name_resolved)
                        << "backend failed to name captured launch at "
                        << kernel.graph_path;
                    EXPECT_FALSE(kernel.name.empty());
                    EXPECT_GT(kernel.registers_per_thread, 0U);
                    EXPECT_GT(kernel.max_threads_per_block, 0U);
                    EXPECT_GT(kernel.max_active_blocks_per_sm, 0U);
                }
                ASSERT_TRUE(capture->instantiate());

                auto initialize_control = [&](int width)
                {
                    ASSERT_GE(width, 2);
                    ASSERT_LE(width, max_verifier_width);
                    ASSERT_TRUE(
                        sampling_math::initialize_device_generation_control(
                            /*max_new_tokens=*/max_seq_len,
                            /*response_capacity=*/max_seq_len,
                            sampling_math::DeviceGenerationDepthPolicy::fixed(
                                width - 1),
                            generation_control.data()));
                };

                auto upload_transaction = [&](int32_t first_token)
                {
                    ASSERT_TRUE(copyHostToDevice(
                        d_first_token,
                        &first_token,
                        sizeof(first_token),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_draft_tokens,
                        draft_tokens.data(),
                        draft_tokens.size() * sizeof(int32_t),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_generation_control,
                        generation_control.data(),
                        sizeof(generation_control),
                        device_id_,
                        stream));
                };

                auto observe_and_validate =
                    [&](int width,
                        int expected_base,
                        int32_t expected_first_token,
                        std::vector<uint8_t> &observed_checkpoint)
                {
                    int32_t observed_length = -1;
                    int32_t observed_base = -1;
                    MTPGreedyPenaltyPolicy observed_policy{};
                    ASSERT_TRUE(copyDeviceToHost(
                        observed_tokens.data(),
                        d_verifier_tokens,
                        sizeof(observed_tokens),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        observed_positions.data(),
                        d_position_ids,
                        sizeof(observed_positions),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        &observed_length,
                        d_request_length,
                        sizeof(observed_length),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        &observed_base,
                        d_base_snapshot,
                        sizeof(observed_base),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        &observed_policy,
                        d_penalty_policy,
                        sizeof(observed_policy),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        observed_checkpoint.data(),
                        d_checkpoint,
                        checkpoint_bytes,
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        generation_control.data(),
                        d_generation_control,
                        sizeof(generation_control),
                        device_id_,
                        stream));
                    ASSERT_TRUE(
                        backend_->synchronizeStream(stream, device_id_));

                    EXPECT_EQ(observed_tokens[0], expected_first_token);
                    for (int slot = 1; slot < width; ++slot)
                    {
                        EXPECT_EQ(
                            observed_tokens[static_cast<size_t>(slot)],
                            draft_tokens[static_cast<size_t>(slot - 1)]);
                    }
                    for (int slot = width;
                         slot < max_verifier_width;
                         ++slot)
                    {
                        EXPECT_EQ(
                            observed_tokens[static_cast<size_t>(slot)],
                            0)
                            << "inactive verifier token suffix was not zeroed";
                    }
                    for (int row = 0; row < max_verifier_width; ++row)
                    {
                        EXPECT_EQ(
                            observed_positions[static_cast<size_t>(row)],
                            expected_base + row);
                    }
                    EXPECT_EQ(observed_length, width);
                    EXPECT_EQ(observed_base, expected_base);
                    EXPECT_EQ(
                        generation_control[
                            sampling_math::kDeviceGenerationControlOk],
                        1);
                    EXPECT_EQ(
                        generation_control[
                            sampling_math::
                                kDeviceGenerationControlRequestComplete],
                        0);
                    EXPECT_EQ(
                        generation_control[
                            sampling_math::
                                kDeviceGenerationControlTransactionCommitBudget],
                        width)
                        << "The reusable physical graph must derive its commit "
                           "budget from the live logical width.";
                    EXPECT_FLOAT_EQ(
                        observed_policy.presence_penalty,
                        presence_penalty);
                    EXPECT_FLOAT_EQ(
                        observed_policy.frequency_penalty,
                        frequency_penalty);
                    EXPECT_EQ(
                        observed_policy.first_token_already_in_history,
                        1);
                    EXPECT_EQ(observed_policy.enabled, 1);
                };

                int expected_base = initial_cached_tokens;
                for (int width = 2; width <= max_verifier_width; ++width)
                {
                    SCOPED_TRACE(::testing::Message()
                                 << GetParam() << " verifier_width=" << width);
                    int32_t first_token = 1000 + width;
                    for (int slot = 0; slot < max_verifier_width - 1; ++slot)
                    {
                        draft_tokens[static_cast<size_t>(slot)] =
                            2000 + width * 100 + slot;
                    }
                    initialize_control(width);
                    upload_transaction(first_token);
                    ASSERT_TRUE(capture->launch());
                    observe_and_validate(
                        width,
                        expected_base,
                        first_token,
                        checkpoint_before);

                    ASSERT_TRUE(cache->appendWithStream(
                        0,
                        0,
                        continuation_k.get(),
                        continuation_v.get(),
                        1,
                        stream));
                    ++expected_base;
                    first_token = 3000 + width;
                    for (int slot = 0; slot < max_verifier_width - 1; ++slot)
                    {
                        draft_tokens[static_cast<size_t>(slot)] =
                            4000 + width * 100 + slot;
                    }
                    initialize_control(width);
                    upload_transaction(first_token);
                    ASSERT_TRUE(capture->launch());
                    observe_and_validate(
                        width,
                        expected_base,
                        first_token,
                        checkpoint_after);
                    EXPECT_NE(checkpoint_after, checkpoint_before)
                        << "replayed checkpoint ignored the advanced live cache";
                }

                initialize_control(/*width=*/2);
                generation_control[
                    sampling_math::
                        kDeviceGenerationControlActiveVerifierRowCount] = 7;
                upload_transaction(/*first_token=*/9999);
                ASSERT_TRUE(capture->launch());

                int32_t invalid_length = -1;
                ASSERT_TRUE(copyDeviceToHost(
                    &invalid_length,
                    d_request_length,
                    sizeof(invalid_length),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyDeviceToHost(
                    generation_control.data(),
                    d_generation_control,
                    sizeof(generation_control),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                EXPECT_EQ(invalid_length, 0);
                EXPECT_EQ(
                    generation_control[
                        sampling_math::kDeviceGenerationControlOk],
                    0);
                EXPECT_EQ(
                    generation_control[
                        sampling_math::
                            kDeviceGenerationControlRequestComplete],
                    1);
                EXPECT_EQ(
                    generation_control[
                        sampling_math::
                            kDeviceGenerationControlTransactionCommitBudget],
                    0);
                EXPECT_EQ(
                    generation_control[
                        sampling_math::kDeviceGenerationControlErrorCode],
                    static_cast<int>(
                        sampling_math::DeviceGenerationError::
                            InvalidDepthSelector));
            });
        };

        if (GetParam() == "CUDA")
        {
            run(GPUDeviceContextPool::instance().getNvidiaContext(device_id_));
        }
        else
        {
            run(GPUDeviceContextPool::instance().getAMDContext(device_id_));
        }

        for (void *allocation : allocations)
            backend_->free(allocation, device_id_);
        workspace_consumer->unbindWorkspace();
        cache.reset();
    }

    /**
     * @brief Prove captured scalar MTP sidecars follow mutable device KV state.
     *
     * A scalar GPU MTP transaction has two production input shapes:
     *
     * - the first sidecar consumes the main-model target sample at the current
     *   live KV position, and
     * - each chained sidecar consumes a draft sample at that same live position
     *   plus its compile-time depth offset.
     *
     * Prefix-cache restore and graph replay can change the live KV count without
     * changing any captured pointer. This regression captures both compositions
     * once, mutates only the device-resident base-position word between replays,
     * and proves that both CUDA and ROCm read the new value on every launch. A
     * host-captured scalar or stale host mirror would leave the second replay at
     * the first position and fail byte equality.
     *
     * All allocations and immutable token uploads happen before capture. The hot
     * path consists only of one H2D test-state mutation, graph launch, stream
     * synchronization for test observation, and the final D2H result read.
     */
    TEST_P(
        GPUSamplingTest,
        MTPDeviceOwnedSidecarsTrackMutableLivePositionAcrossGraphReplays)
    {
        constexpr int request_count = 1;
        constexpr int32_t target_token = 101;
        constexpr int32_t draft_token = 202;
        const std::array<int32_t, 3> live_positions = {7, 4095, 8193};

        void *d_target_token =
            backend_->allocate(sizeof(target_token), device_id_);
        void *d_draft_token =
            backend_->allocate(sizeof(draft_token), device_id_);
        void *d_live_position =
            backend_->allocate(sizeof(int32_t), device_id_);
        void *d_target_condition =
            backend_->allocate(sizeof(int32_t), device_id_);
        void *d_target_position =
            backend_->allocate(sizeof(int32_t), device_id_);
        void *d_draft_condition =
            backend_->allocate(sizeof(int32_t), device_id_);
        void *d_draft_position =
            backend_->allocate(sizeof(int32_t), device_id_);
        ASSERT_NE(d_target_token, nullptr);
        ASSERT_NE(d_draft_token, nullptr);
        ASSERT_NE(d_live_position, nullptr);
        ASSERT_NE(d_target_condition, nullptr);
        ASSERT_NE(d_target_position, nullptr);
        ASSERT_NE(d_draft_condition, nullptr);
        ASSERT_NE(d_draft_position, nullptr);

        std::array<int32_t, live_positions.size()> observed_target_tokens{};
        std::array<int32_t, live_positions.size()> observed_target_positions{};
        std::array<int32_t, live_positions.size()> observed_draft_tokens{};
        std::array<int32_t, live_positions.size()> observed_draft_positions{};

        auto run = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_target_token,
                    &target_token,
                    sizeof(target_token),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_token,
                    &draft_token,
                    sizeof(draft_token),
                    device_id_,
                    stream));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueuePrepareMTPBatchedSidecarInputs(
                    d_target_token,
                    /*condition_token_stride=*/1,
                    d_live_position,
                    /*position_offset=*/0,
                    request_count,
                    device_id_,
                    stream,
                    d_target_condition,
                    d_target_position));
                ASSERT_TRUE(backend_->enqueuePrepareMTPBatchedSidecarInputs(
                    d_draft_token,
                    /*condition_token_stride=*/1,
                    d_live_position,
                    /*position_offset=*/1,
                    request_count,
                    device_id_,
                    stream,
                    d_draft_condition,
                    d_draft_position));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                for (size_t replay = 0; replay < live_positions.size(); ++replay)
                {
                    ASSERT_TRUE(copyHostToDevice(
                        d_live_position,
                        &live_positions[replay],
                        sizeof(live_positions[replay]),
                        device_id_,
                        stream));
                    ASSERT_TRUE(capture->launch());
                    ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                    ASSERT_TRUE(copyDeviceToHost(
                        &observed_target_tokens[replay],
                        d_target_condition,
                        sizeof(int32_t),
                        device_id_));
                    ASSERT_TRUE(copyDeviceToHost(
                        &observed_target_positions[replay],
                        d_target_position,
                        sizeof(int32_t),
                        device_id_));
                    ASSERT_TRUE(copyDeviceToHost(
                        &observed_draft_tokens[replay],
                        d_draft_condition,
                        sizeof(int32_t),
                        device_id_));
                    ASSERT_TRUE(copyDeviceToHost(
                        &observed_draft_positions[replay],
                        d_draft_position,
                        sizeof(int32_t),
                        device_id_));
                }
            });
        };
        if (GetParam() == "CUDA")
            run(GPUDeviceContextPool::instance().getNvidiaContext(device_id_));
        else
            run(GPUDeviceContextPool::instance().getAMDContext(device_id_));

        for (size_t replay = 0; replay < live_positions.size(); ++replay)
        {
            EXPECT_EQ(observed_target_tokens[replay], target_token);
            EXPECT_EQ(observed_target_positions[replay], live_positions[replay]);
            EXPECT_EQ(observed_draft_tokens[replay], draft_token);
            EXPECT_EQ(observed_draft_positions[replay],
                      live_positions[replay] + 1);
        }

        backend_->free(d_draft_position, device_id_);
        backend_->free(d_draft_condition, device_id_);
        backend_->free(d_target_position, device_id_);
        backend_->free(d_target_condition, device_id_);
        backend_->free(d_live_position, device_id_);
        backend_->free(d_draft_token, device_id_);
        backend_->free(d_target_token, device_id_);
    }

    /**
     * @brief Prove shifted-MTP suffix tokens and positions stay device-owned.
     *
     * The LocalTP publication path consumes compact verifier metadata without
     * reading the accepted-state count or the verifier-base position on the
     * host. This regression drives the exact fused production primitive and
     * covers three semantic boundaries in one request batch:
     *
     * - a partially accepted suffix copies only publishable output tokens,
     * - rows beyond the accepted suffix use the request's live first token, and
     * - invalid metadata uses the caller-provided filler token.
     *
     * Every request also starts from a distinct resident base position. Byte
     * equality therefore proves that request indexing and the position offset
     * are fused correctly on both CUDA and ROCm.
     */
    TEST_P(GPUSamplingTest,
           MTPShiftedKVPreparationFusesResidentTokensAndPositionsByteExactly)
    {
        constexpr int request_count = 3;
        constexpr int meta_stride = sampling_math::kSpeculativeBatchMetaCount;
        constexpr int output_stride =
            sampling_math::kSpeculativeBatchMaxOutputTokens;
        constexpr int row_count = 4;
        constexpr int first_output_token_index = 1;
        constexpr int position_offset = 2;
        constexpr int32_t invalid_metadata_filler = -77;

        std::array<int, request_count * meta_stride> meta{};
        auto set_request_meta = [&](int request,
                                    int ok,
                                    int output_count,
                                    int accepted_state_count)
        {
            int *request_meta = meta.data() + request * meta_stride;
            request_meta[sampling_math::kSpecBatchMetaOk] = ok;
            request_meta[sampling_math::kSpecBatchMetaOutputCount] = output_count;
            request_meta[
                sampling_math::kSpecBatchMetaTargetVerifierStateCommitCount] =
                accepted_state_count;
        };
        set_request_meta(/*request=*/0, /*ok=*/1,
                         /*output_count=*/5, /*accepted_state_count=*/4);
        set_request_meta(/*request=*/1, /*ok=*/1,
                         /*output_count=*/4, /*accepted_state_count=*/2);
        set_request_meta(/*request=*/2, /*ok=*/0,
                         /*output_count=*/5, /*accepted_state_count=*/4);

        const auto output_tokens = []
        {
            std::array<int32_t, request_count * output_stride> tokens{};
            for (int request = 0; request < request_count; ++request)
            {
                for (int token = 0; token < 5; ++token)
                {
                    tokens[static_cast<size_t>(request * output_stride + token)] =
                        (request + 1) * 100 + token + 1;
                }
            }
            return tokens;
        }();
        const std::array<int32_t, request_count> base_positions = {17, 53, 89};

        void *d_meta = backend_->allocate(sizeof(meta), device_id_);
        void *d_output_tokens =
            backend_->allocate(sizeof(output_tokens), device_id_);
        void *d_base_positions =
            backend_->allocate(sizeof(base_positions), device_id_);
        void *d_prepared_tokens = backend_->allocate(
            request_count * row_count * sizeof(int32_t), device_id_);
        void *d_prepared_positions = backend_->allocate(
            request_count * row_count * sizeof(int32_t), device_id_);
        ASSERT_NE(d_meta, nullptr);
        ASSERT_NE(d_output_tokens, nullptr);
        ASSERT_NE(d_base_positions, nullptr);
        ASSERT_NE(d_prepared_tokens, nullptr);
        ASSERT_NE(d_prepared_positions, nullptr);

        auto run = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_meta, meta.data(), sizeof(meta), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_output_tokens,
                    output_tokens.data(),
                    sizeof(output_tokens),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_base_positions,
                    base_positions.data(),
                    sizeof(base_positions),
                    device_id_,
                    stream));

                for (int request = 0; request < request_count; ++request)
                {
                    ASSERT_TRUE(
                        backend_->enqueuePrepareSpeculativeShiftedKVTokens(
                            d_meta,
                            meta_stride,
                            d_output_tokens,
                            output_stride,
                            request,
                            first_output_token_index,
                            row_count,
                            invalid_metadata_filler,
                            device_id_,
                            stream,
                            static_cast<int32_t *>(d_prepared_tokens) +
                                request * row_count,
                            d_base_positions,
                            position_offset,
                            static_cast<int32_t *>(d_prepared_positions) +
                                request * row_count));
                }
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };
        if (GetParam() == "CUDA")
            run(GPUDeviceContextPool::instance().getNvidiaContext(device_id_));
        else
            run(GPUDeviceContextPool::instance().getAMDContext(device_id_));

        std::array<int32_t, request_count * row_count> prepared_tokens{};
        std::array<int32_t, request_count * row_count> prepared_positions{};
        ASSERT_TRUE(copyDeviceToHost(
            prepared_tokens.data(),
            d_prepared_tokens,
            sizeof(prepared_tokens),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            prepared_positions.data(),
            d_prepared_positions,
            sizeof(prepared_positions),
            device_id_));

        const std::array<int32_t, request_count * row_count> expected_tokens = {
            102, 103, 104, 101,
            202, 201, 201, 201,
            -77, -77, -77, -77};
        const std::array<int32_t, request_count * row_count> expected_positions = {
            19, 20, 21, 22,
            55, 56, 57, 58,
            91, 92, 93, 94};
        EXPECT_EQ(prepared_tokens, expected_tokens);
        EXPECT_EQ(prepared_positions, expected_positions);

        backend_->free(d_prepared_positions, device_id_);
        backend_->free(d_prepared_tokens, device_id_);
        backend_->free(d_base_positions, device_id_);
        backend_->free(d_output_tokens, device_id_);
        backend_->free(d_meta, device_id_);
    }

    /**
     * @brief Prove terminal prefill sampling publishes a complete device mailbox.
     *
     * Sample identities and prompt positions begin in GPU-owned rows. The kernel
     * must initialize every mutable transaction predicate without adopting a host
     * mirror or synchronizing the device. The input position row deliberately
     * aliases the target-position output, matching production metadata ownership.
     */
    TEST_P(GPUSamplingTest, MTPInitialLogicalStatePublicationSeedsEveryDeviceField)
    {
        constexpr int request_count = 4;
        const std::array<int32_t, request_count> sampled_tokens = {31, 41, 59, 26};
        const std::array<int32_t, request_count> prompt_lengths = {7, 11, 19, 23};

        std::array<void *, 8> device_rows{};
        for (void *&row : device_rows)
        {
            row = backend_->allocate(request_count * sizeof(int32_t), device_id_);
            ASSERT_NE(row, nullptr);
        }
        void *d_samples = device_rows[0];
        void *d_base = device_rows[1];
        void *d_target = device_rows[2];
        void *d_accepted = device_rows[3];
        void *d_next = device_rows[4];
        void *d_all_accepted = device_rows[5];
        void *d_stopped = device_rows[6];
        void *d_ok = device_rows[7];

        auto run = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_samples,
                    sampled_tokens.data(),
                    sampled_tokens.size() * sizeof(int32_t),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_target,
                    prompt_lengths.data(),
                    prompt_lengths.size() * sizeof(int32_t),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->enqueueInitializeMTPDeviceLogicalState(
                    d_samples,
                    d_target,
                    request_count,
                    device_id_,
                    stream,
                    d_base,
                    d_target,
                    d_accepted,
                    d_next,
                    d_all_accepted,
                    d_stopped,
                    d_ok));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };
        if (GetParam() == "CUDA")
            run(GPUDeviceContextPool::instance().getNvidiaContext(device_id_));
        else
            run(GPUDeviceContextPool::instance().getAMDContext(device_id_));

        auto read = [&](void *device_row)
        {
            std::array<int32_t, request_count> host{};
            EXPECT_TRUE(copyDeviceToHost(
                host.data(), device_row, sizeof(host), device_id_));
            return host;
        };
        EXPECT_EQ(read(d_base), prompt_lengths);
        EXPECT_EQ(read(d_target), prompt_lengths);
        EXPECT_EQ(read(d_accepted),
                  (std::array<int32_t, request_count>{0, 0, 0, 0}));
        EXPECT_EQ(read(d_next), sampled_tokens);
        EXPECT_EQ(read(d_all_accepted),
                  (std::array<int32_t, request_count>{0, 0, 0, 0}));
        EXPECT_EQ(read(d_stopped),
                  (std::array<int32_t, request_count>{0, 0, 0, 0}));
        EXPECT_EQ(read(d_ok),
                  (std::array<int32_t, request_count>{1, 1, 1, 1}));

        for (auto it = device_rows.rbegin(); it != device_rows.rend(); ++it)
            backend_->free(*it, device_id_);
    }

    /**
     * @brief Prove graph replay samples and publishes from one resident position row.
     *
     * Request admission uploads immutable prompt positions before graph capture.
     * The captured production primitives then derive one position-keyed stochastic
     * draw per request, write sampled tokens to device slots, and initialize the
     * logical-state mailbox in place. Replaying the same graph after changing only
     * the device position row proves that neither backend captured host thresholds
     * or launch-time position scalars. Every token is checked against the shared
     * serial sampling math, while every mailbox field is checked exactly.
     */
    TEST_P(GPUSamplingTest, MTPPrefillResidentPositionSamplingPublishesInitialMailboxExactly)
    {
        constexpr int request_count = 2;
        constexpr int top_k = 4;
        const std::array<int32_t, request_count * top_k> token_ids = {
            10, 11, 12, 13,
            20, 21, 22, 23};
        const std::array<float, request_count * top_k> probabilities = {
            0.10f, 0.20f, 0.30f, 0.40f,
            0.40f, 0.30f, 0.20f, 0.10f};
        const std::array<uint64_t, request_count> seeds = {
            0x123456789ABCDEF0ull,
            0x0FEDCBA987654321ull};
        const std::array<std::array<int32_t, request_count>, 2> position_rows = {{
            {{3, 2}},
            {{11, 19}},
        }};

        void *d_token_ids = backend_->allocate(sizeof(token_ids), device_id_);
        void *d_probabilities =
            backend_->allocate(sizeof(probabilities), device_id_);
        std::array<void *, 8> mailbox_rows{};
        for (void *&row : mailbox_rows)
            row = backend_->allocate(request_count * sizeof(int32_t), device_id_);

        ASSERT_NE(d_token_ids, nullptr);
        ASSERT_NE(d_probabilities, nullptr);
        for (void *row : mailbox_rows)
            ASSERT_NE(row, nullptr);

        void *d_samples = mailbox_rows[0];
        void *d_base = mailbox_rows[1];
        void *d_target = mailbox_rows[2];
        void *d_accepted = mailbox_rows[3];
        void *d_next = mailbox_rows[4];
        void *d_all_accepted = mailbox_rows[5];
        void *d_stopped = mailbox_rows[6];
        void *d_ok = mailbox_rows[7];

        std::array<std::array<int32_t, request_count>, 2> sampled_results{};
        std::array<std::array<int32_t, request_count>, 2> base_results{};
        std::array<std::array<int32_t, request_count>, 2> target_results{};
        std::array<std::array<int32_t, request_count>, 2> accepted_results{};
        std::array<std::array<int32_t, request_count>, 2> next_results{};
        std::array<std::array<int32_t, request_count>, 2> all_accepted_results{};
        std::array<std::array<int32_t, request_count>, 2> stopped_results{};
        std::array<std::array<int32_t, request_count>, 2> ok_results{};

        auto run = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_token_ids,
                    token_ids.data(),
                    sizeof(token_ids),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_probabilities,
                    probabilities.data(),
                    sizeof(probabilities),
                    device_id_,
                    stream));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                for (int request = 0; request < request_count; ++request)
                {
                    ASSERT_TRUE(backend_->enqueueSampleDistributionF32Device(
                        static_cast<int32_t *>(d_token_ids) + request * top_k,
                        static_cast<float *>(d_probabilities) + request * top_k,
                        top_k,
                        /*threshold=*/0.0f,
                        device_id_,
                        stream,
                        static_cast<int32_t *>(d_samples) + request,
                        /*out_probability_device=*/nullptr,
                        seeds[static_cast<size_t>(request)],
                        static_cast<int32_t *>(d_target) + request,
                        /*threshold_position_offset=*/0));
                }
                ASSERT_TRUE(backend_->enqueueInitializeMTPDeviceLogicalState(
                    d_samples,
                    d_target,
                    request_count,
                    device_id_,
                    stream,
                    d_base,
                    d_target,
                    d_accepted,
                    d_next,
                    d_all_accepted,
                    d_stopped,
                    d_ok));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                for (size_t replay = 0; replay < position_rows.size(); ++replay)
                {
                    ASSERT_TRUE(copyHostToDevice(
                        d_target,
                        position_rows[replay].data(),
                        sizeof(position_rows[replay]),
                        device_id_,
                        stream));
                    ASSERT_TRUE(capture->launch());
                    ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                    ASSERT_TRUE(copyDeviceToHost(
                        sampled_results[replay].data(),
                        d_samples,
                        sizeof(sampled_results[replay]),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        base_results[replay].data(),
                        d_base,
                        sizeof(base_results[replay]),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        target_results[replay].data(),
                        d_target,
                        sizeof(target_results[replay]),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        accepted_results[replay].data(),
                        d_accepted,
                        sizeof(accepted_results[replay]),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        next_results[replay].data(),
                        d_next,
                        sizeof(next_results[replay]),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        all_accepted_results[replay].data(),
                        d_all_accepted,
                        sizeof(all_accepted_results[replay]),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        stopped_results[replay].data(),
                        d_stopped,
                        sizeof(stopped_results[replay]),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        ok_results[replay].data(),
                        d_ok,
                        sizeof(ok_results[replay]),
                        device_id_,
                        stream));
                }
            });
        };
        if (GetParam() == "CUDA")
            run(GPUDeviceContextPool::instance().getNvidiaContext(device_id_));
        else
            run(GPUDeviceContextPool::instance().getAMDContext(device_id_));

        const std::array<int32_t, request_count> zeros = {0, 0};
        const std::array<int32_t, request_count> ones = {1, 1};
        for (size_t replay = 0; replay < position_rows.size(); ++replay)
        {
            std::array<int32_t, request_count> expected_tokens{};
            for (int request = 0; request < request_count; ++request)
            {
                const float threshold =
                    sampling_math::mtp_spec_threshold_from_seed(
                        seeds[static_cast<size_t>(request)],
                        position_rows[replay][static_cast<size_t>(request)],
                        0 /* MTPSpecStochasticDrawPurpose::Sample */);
                expected_tokens[static_cast<size_t>(request)] =
                    sampling_math::sample_distribution_with_threshold(
                        token_ids.data() + request * top_k,
                        probabilities.data() + request * top_k,
                        top_k,
                        threshold);
            }

            EXPECT_EQ(sampled_results[replay], expected_tokens);
            EXPECT_EQ(base_results[replay], position_rows[replay]);
            EXPECT_EQ(target_results[replay], position_rows[replay]);
            EXPECT_EQ(accepted_results[replay], zeros);
            EXPECT_EQ(next_results[replay], expected_tokens);
            EXPECT_EQ(all_accepted_results[replay], zeros);
            EXPECT_EQ(stopped_results[replay], zeros);
            EXPECT_EQ(ok_results[replay], ones);
        }

        for (auto it = mailbox_rows.rbegin(); it != mailbox_rows.rend(); ++it)
            backend_->free(*it, device_id_);
        backend_->free(d_probabilities, device_id_);
        backend_->free(d_token_ids, device_id_);
    }

    // =========================================================================
    //  ARGMAX TESTS — mirrors Greedy Sampling from Test__Sampler.cpp
    // =========================================================================

    TEST_P(GPUSamplingTest, Argmax_StandardLogits)
    {
        // Should select token with highest logit (index 2, value 3.0)
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok) << "argmaxF32 not supported on " << GetParam();

        EXPECT_EQ(out_index, 2) << "Argmax should select index 2 (logit 3.0)";
        EXPECT_FLOAT_EQ(out_value, 3.0f) << "Argmax value should be 3.0";

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_UniformLogits)
    {
        // With uniform logits, should select first occurrence (index 0)
        void *d_ptr = uploadLogits(uniform_logits_);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(uniform_logits_.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 0) << "Argmax of uniform should select first occurrence";
        EXPECT_FLOAT_EQ(out_value, 2.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_SingleToken)
    {
        std::vector<float> single = {5.0f};
        void *d_ptr = uploadLogits(single);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, 1, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 0);
        EXPECT_FLOAT_EQ(out_value, 5.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_Deterministic)
    {
        // Argmax should always return the same result
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        int first_index = -1;
        float first_value = 0.0f;
        bool ok = argmaxF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                      device_id_, &first_value, &first_index);
        ASSERT_TRUE(ok);

        for (int i = 0; i < 10; ++i)
        {
            float out_value = 0.0f;
            int out_index = -1;
            ok = argmaxF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                     device_id_, &out_value, &out_index);
            ASSERT_TRUE(ok);
            EXPECT_EQ(out_index, first_index) << "Iteration " << i;
            EXPECT_FLOAT_EQ(out_value, first_value) << "Iteration " << i;
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_PeakedLogits)
    {
        // Peaked distribution: token 2 has logit 10.0, rest are 0.1-0.2
        void *d_ptr = uploadLogits(peaked_logits_);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(peaked_logits_.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 2) << "Argmax should select peaked token";
        EXPECT_FLOAT_EQ(out_value, 10.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_NegativeLogits)
    {
        // All negative logits: max is -1.0 at index 2
        std::vector<float> negative = {-5.0f, -2.0f, -1.0f, -10.0f};
        void *d_ptr = uploadLogits(negative);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(negative.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 2) << "Argmax should select least-negative value";
        EXPECT_FLOAT_EQ(out_value, -1.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_ExtremeLogits)
    {
        // Extreme difference: token 2 = 100.0, rest = -1000.0
        std::vector<float> extreme = {-1000.0f, -1000.0f, 100.0f, -1000.0f};
        void *d_ptr = uploadLogits(extreme);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(extreme.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 2);
        EXPECT_FLOAT_EQ(out_value, 100.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_AllZeros)
    {
        std::vector<float> zeros = {0.0f, 0.0f, 0.0f, 0.0f};
        void *d_ptr = uploadLogits(zeros);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(zeros.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 0) << "All zeros: argmax should select first element";
        EXPECT_FLOAT_EQ(out_value, 0.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_AllSameValue)
    {
        // All same non-zero: should pick index 0
        void *d_ptr = uploadLogits(uniform_logits_);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(uniform_logits_.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 0) << "Uniform: argmax should select first occurrence";
        EXPECT_FLOAT_EQ(out_value, 2.0f);

        freeDevice(d_ptr);
    }

    // =========================================================================
    // ARGMAX NUMERICAL STABILITY TESTS
    // =========================================================================

    TEST_P(GPUSamplingTest, Argmax_VeryLargeLogits)
    {
        // Very large values — should not overflow or produce wrong result
        std::vector<float> large = {500.0f, 501.0f, 502.0f, 500.5f};
        void *d_ptr = uploadLogits(large);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(large.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 2);
        EXPECT_FLOAT_EQ(out_value, 502.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_VerySmallLogits)
    {
        // Very small (negative) values
        std::vector<float> small = {-500.0f, -501.0f, -499.0f, -500.5f};
        void *d_ptr = uploadLogits(small);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(small.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 2) << "Argmax should pick -499.0 (highest)";
        EXPECT_FLOAT_EQ(out_value, -499.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_MixedExtremeLogits)
    {
        // Mix of very large and very small
        std::vector<float> mixed = {-1000.0f, 1000.0f, -1000.0f, -1000.0f};
        void *d_ptr = uploadLogits(mixed);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(mixed.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 1);
        EXPECT_FLOAT_EQ(out_value, 1000.0f);

        freeDevice(d_ptr);
    }

    // =========================================================================
    // ARGMAX LARGE VOCABULARY TESTS
    // =========================================================================

    TEST_P(GPUSamplingTest, Argmax_LargeVocab_50K)
    {
        // 50K vocabulary with a peak at a specific index
        const int n = 50000;
        std::vector<float> logits(n, 0.0f);
        logits[12345] = 10.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 12345);
        EXPECT_FLOAT_EQ(out_value, 10.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_Qwen2VocabSize)
    {
        // Qwen2.5 vocab_size = 151936
        const int n = 151936;
        std::vector<float> logits(n, 0.0f);
        logits[256] = 15.0f;    // Top prediction
        logits[8159] = 14.0f;   // Second
        logits[100160] = 13.5f; // Third

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 256) << "Argmax should pick global max";
        EXPECT_FLOAT_EQ(out_value, 15.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_LargeVocabTieBreaksToLowestTokenId)
    {
        // Regression for CUDA skip-gather greedy decode: equal winning logits
        // must match std::max_element semantics and select the first/lower id,
        // even when the tied candidates land in different reduction lanes.
        const int n = 248320;
        std::vector<float> logits(n, -8.0f);
        logits[248046] = 17.0f;
        logits[248068] = 17.0f;
        logits[1024] = 16.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 248046);
        EXPECT_FLOAT_EQ(out_value, 17.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_BatchedRowsQwen36VocabMatchesSerialRows)
    {
        // Phase 13.8 MTP verifier rows sample a compact [M, vocab] logits
        // tensor. Keep the Qwen3.6-sized batched-row argmax honest against the
        // serial row path so a sampler bug cannot masquerade as state drift.
        constexpr int rows = 3;
        constexpr int cols = 248320;
        const int expected[rows] = {271, 33075, 248068};

        std::vector<float> logits(
            static_cast<size_t>(rows) * static_cast<size_t>(cols),
            -9.0f);
        for (int row = 0; row < rows; ++row)
        {
            const size_t base = static_cast<size_t>(row) * static_cast<size_t>(cols);
            logits[base + static_cast<size_t>(expected[row])] =
                25.0f + static_cast<float>(row);
            logits[base + static_cast<size_t>(expected[row] + 1)] =
                24.0f + static_cast<float>(row);
        }
        logits[static_cast<size_t>(2) * static_cast<size_t>(cols) + 248100] =
            logits[static_cast<size_t>(2) * static_cast<size_t>(cols) +
                   static_cast<size_t>(expected[2])];

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float batched_values[rows] = {};
        int batched_indices[rows] = {-1, -1, -1};
        ASSERT_TRUE(argmaxF32BatchedRows(
            d_ptr,
            rows,
            cols,
            device_id_,
            batched_values,
            batched_indices))
            << "argmaxF32BatchedRows not supported on " << GetParam();

        const auto *base = static_cast<const char *>(d_ptr);
        for (int row = 0; row < rows; ++row)
        {
            float serial_value = 0.0f;
            int serial_index = -1;
            void *row_ptr = const_cast<char *>(
                base + static_cast<size_t>(row) *
                           static_cast<size_t>(cols) * sizeof(float));
            ASSERT_TRUE(argmaxF32(row_ptr, cols, device_id_, &serial_value, &serial_index))
                << "serial argmaxF32 failed for row " << row << " on " << GetParam();

            EXPECT_EQ(batched_indices[row], serial_index) << "row=" << row;
            EXPECT_FLOAT_EQ(batched_values[row], serial_value) << "row=" << row;
            EXPECT_EQ(batched_indices[row], expected[row]) << "row=" << row;
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_PeakAtLastElement)
    {
        // Edge case: max at end of large array
        const int n = 100000;
        std::vector<float> logits(n, -1.0f);
        logits[n - 1] = 42.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, n - 1);
        EXPECT_FLOAT_EQ(out_value, 42.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_PeakAtFirstElement)
    {
        // Edge case: max at start of large array
        const int n = 100000;
        std::vector<float> logits(n, -1.0f);
        logits[0] = 42.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 0);
        EXPECT_FLOAT_EQ(out_value, 42.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_BatchedRows)
    {
        constexpr int rows = 4;
        constexpr int cols = 4096;
        std::vector<float> logits(static_cast<size_t>(rows) * static_cast<size_t>(cols), -7.0f);
        const int expected[rows] = {17, 2048, 4095, 0};
        for (int row = 0; row < rows; ++row)
        {
            logits[static_cast<size_t>(row) * static_cast<size_t>(cols) +
                   static_cast<size_t>(expected[row])] =
                100.0f + static_cast<float>(row);
        }
        logits[static_cast<size_t>(3) * static_cast<size_t>(cols) + 1234] =
            logits[static_cast<size_t>(3) * static_cast<size_t>(cols)];

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_values[rows] = {};
        int out_indices[rows] = {-1, -1, -1, -1};
        ASSERT_TRUE(argmaxF32BatchedRows(d_ptr, rows, cols, device_id_, out_values, out_indices))
            << "argmaxF32BatchedRows not supported on " << GetParam();

        for (int row = 0; row < rows; ++row)
        {
            EXPECT_EQ(out_indices[row], expected[row]) << "row=" << row;
            EXPECT_FLOAT_EQ(
                out_values[row],
                logits[static_cast<size_t>(row) * static_cast<size_t>(cols) +
                       static_cast<size_t>(expected[row])])
                << "row=" << row;
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_BatchedRowsTieBreaksToLowestTokenId)
    {
        constexpr int rows = 2;
        constexpr int cols = 8192;
        std::vector<float> logits(static_cast<size_t>(rows) * static_cast<size_t>(cols), -3.0f);
        logits[2047] = 9.0f;
        logits[4096] = 9.0f;
        logits[static_cast<size_t>(cols) + 7000] = 11.0f;
        logits[static_cast<size_t>(cols) + 123] = 11.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_values[rows] = {};
        int out_indices[rows] = {-1, -1};
        ASSERT_TRUE(argmaxF32BatchedRows(d_ptr, rows, cols, device_id_, out_values, out_indices))
            << "argmaxF32BatchedRows not supported on " << GetParam();

        EXPECT_EQ(out_indices[0], 2047);
        EXPECT_FLOAT_EQ(out_values[0], 9.0f);
        EXPECT_EQ(out_indices[1], 123);
        EXPECT_FLOAT_EQ(out_values[1], 11.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_DeviceBatchedRowsSupportsOutputStride)
    {
        /*
         * Request-batched MTP stores sampled draft tokens in request-major
         * slots: slot = draft_index + request * draft_depth.  The GPU argmax
         * primitive therefore needs strided output stores so the runner does
         * not sample on GPU and then re-upload a host shadow before verifier
         * execution.
         */
        constexpr int rows = 3;
        constexpr int cols = 64;
        constexpr int output_stride = 4;
        constexpr int output_span = 1 + (rows - 1) * output_stride;

        std::vector<float> logits(
            static_cast<size_t>(rows) * static_cast<size_t>(cols),
            -20.0f);
        logits[7] = 10.0f;
        logits[static_cast<size_t>(cols) + 31] = 12.0f;
        logits[2 * static_cast<size_t>(cols) + 9] = 11.0f;

        void *d_logits = nullptr;
        void *d_values = nullptr;
        void *d_indices = nullptr;
        void *d_partial_values = nullptr;
        void *d_partial_indices = nullptr;
        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_logits,
                d_values,
                d_indices,
                d_partial_values,
                d_partial_indices};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_values = backend_->allocate(output_span * sizeof(float), device_id_);
        d_indices = backend_->allocate(output_span * sizeof(int), device_id_);
        d_partial_values = backend_->allocate(1024 * sizeof(float), device_id_);
        d_partial_indices = backend_->allocate(1024 * sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_values, nullptr);
        ASSERT_NE(d_indices, nullptr);
        ASSERT_NE(d_partial_values, nullptr);
        ASSERT_NE(d_partial_indices, nullptr);

        std::array<float, output_span> initial_values{};
        initial_values.fill(-999.0f);
        std::array<int, output_span> initial_indices{};
        initial_indices.fill(-99);

        auto run = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_logits,
                    logits.data(),
                    logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_values,
                    initial_values.data(),
                    initial_values.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_indices,
                    initial_indices.data(),
                    initial_indices.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->enqueueArgmaxF32BatchedRowsDevice(
                    d_logits,
                    rows,
                    cols,
                    device_id_,
                    stream,
                    d_values,
                    d_indices,
                    d_partial_values,
                    d_partial_indices,
                    1024,
                    output_stride));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run(ctx);
        }

        std::array<float, output_span> values{};
        std::array<int, output_span> indices{};
        ASSERT_TRUE(copyDeviceToHost(
            values.data(),
            d_values,
            values.size() * sizeof(float),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            indices.data(),
            d_indices,
            indices.size() * sizeof(int),
            device_id_));

        EXPECT_FLOAT_EQ(values[0], 10.0f);
        EXPECT_FLOAT_EQ(values[output_stride], 12.0f);
        EXPECT_FLOAT_EQ(values[2 * output_stride], 11.0f);
        EXPECT_EQ(indices[0], 7);
        EXPECT_EQ(indices[output_stride], 31);
        EXPECT_EQ(indices[2 * output_stride], 9);
        for (int slot = 0; slot < output_span; ++slot)
        {
            if (slot == 0 || slot == output_stride || slot == 2 * output_stride)
                continue;
            EXPECT_FLOAT_EQ(values[slot], -999.0f) << "slot " << slot;
            EXPECT_EQ(indices[slot], -99) << "slot " << slot;
        }

        cleanup();
    }

    TEST_P(GPUSamplingTest, GreedySpeculativeSummaryIsGraphCapturable)
    {
        using namespace sampling_math;

        constexpr int rows = 3;
        constexpr int cols = 16;
        constexpr int compare_rows = 2;
        const int32_t draft_tokens[rows] = {2, 4, 5};
        const int stop_tokens[1] = {-1};

        std::vector<float> logits(
            static_cast<size_t>(rows) * static_cast<size_t>(cols),
            -10.0f);
        logits[4] = 9.0f;                              // accepts draft_tokens[1]
        logits[static_cast<size_t>(cols) + 7] = 11.0f; // rejects draft_tokens[2]
        logits[2 * static_cast<size_t>(cols) + 6] = 8.0f; // bonus ignored

        int expected_tokens[kSpeculativeBatchMaxOutputTokens] = {};
        int expected_meta[kSpeculativeBatchMetaCount] = {};
        const int expected_verifier_tokens[rows] = {4, 7, 6};
        summarize_greedy_speculative_verify_batch(
            draft_tokens[0],
            expected_verifier_tokens,
            draft_tokens,
            compare_rows,
            stop_tokens,
            /*stop_token_count=*/0,
            expected_tokens,
            kSpeculativeBatchMaxOutputTokens,
            expected_meta);
        ASSERT_EQ(expected_meta[kSpecBatchMetaOk], 1);

        void *d_logits = nullptr;
        void *d_argmax_values = nullptr;
        void *d_argmax_indices = nullptr;
        void *d_partial_values = nullptr;
        void *d_partial_indices = nullptr;
        void *d_draft_tokens = nullptr;
        void *d_output_tokens = nullptr;
        void *d_output_meta = nullptr;
        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_logits,
                d_argmax_values,
                d_argmax_indices,
                d_partial_values,
                d_partial_indices,
                d_draft_tokens,
                d_output_tokens,
                d_output_meta};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_argmax_values = backend_->allocate(rows * sizeof(float), device_id_);
        d_argmax_indices = backend_->allocate(rows * sizeof(int), device_id_);
        d_partial_values = backend_->allocate(1024 * sizeof(float), device_id_);
        d_partial_indices = backend_->allocate(1024 * sizeof(int), device_id_);
        d_draft_tokens = backend_->allocate(rows * sizeof(int32_t), device_id_);
        d_output_tokens =
            backend_->allocate(kSpeculativeBatchMaxOutputTokens * sizeof(int), device_id_);
        d_output_meta =
            backend_->allocate(kSpeculativeBatchMetaCount * sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_argmax_values, nullptr);
        ASSERT_NE(d_argmax_indices, nullptr);
        ASSERT_NE(d_partial_values, nullptr);
        ASSERT_NE(d_partial_indices, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_output_tokens, nullptr);
        ASSERT_NE(d_output_meta, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_logits,
                    logits.data(),
                    logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens,
                    draft_tokens,
                    sizeof(draft_tokens),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueArgmaxF32BatchedRowsDevice(
                    d_logits,
                    rows,
                    cols,
                    device_id_,
                    nullptr,
                    d_argmax_values,
                    d_argmax_indices,
                    d_partial_values,
                    d_partial_indices,
                    1024))
                    << "device-resident batched argmax must reject the legacy default/null stream";
                EXPECT_FALSE(backend_->enqueueSummarizeGreedySpeculativeVerifyBatch(
                    d_argmax_indices,
                    d_draft_tokens,
                    compare_rows,
                    draft_tokens[0],
                    stop_tokens,
                    0,
                    device_id_,
                    nullptr,
                    sampling_math::kSpeculativeBatchMaxOutputTokens,
                    d_output_tokens,
                    d_output_meta))
                    << "greedy speculative summary must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueArgmaxF32BatchedRowsDevice(
                    d_logits,
                    rows,
                    cols,
                    device_id_,
                    stream,
                    d_argmax_values,
                    d_argmax_indices,
                    d_partial_values,
                    d_partial_indices,
                    1024));
                ASSERT_TRUE(backend_->enqueueSummarizeGreedySpeculativeVerifyBatch(
                    d_argmax_indices,
                    d_draft_tokens,
                    compare_rows,
                    draft_tokens[0],
                    stop_tokens,
                    0,
                    device_id_,
                    stream,
                    sampling_math::kSpeculativeBatchMaxOutputTokens,
                    d_output_tokens,
                    d_output_meta));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        int gpu_tokens[kSpeculativeBatchMaxOutputTokens] = {};
        int gpu_meta[kSpeculativeBatchMetaCount] = {};
        ASSERT_TRUE(copyDeviceToHost(
            gpu_tokens,
            d_output_tokens,
            sizeof(gpu_tokens),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            gpu_meta,
            d_output_meta,
            sizeof(gpu_meta),
            device_id_));

        EXPECT_EQ(gpu_meta[kSpecBatchMetaOk], 1);
        for (int i = 0; i < kSpeculativeBatchMetaCount; ++i)
            EXPECT_EQ(gpu_meta[i], expected_meta[i]) << "meta index " << i;
        for (int i = 0; i < kSpeculativeBatchMaxOutputTokens; ++i)
            EXPECT_EQ(gpu_tokens[i], expected_tokens[i]) << "token index " << i;

        cleanup();
    }

    /**
     * @brief Reuses one physical greedy-verifier graph across logical depths.
     *
     * The production grouped verifier captures a bounded power-of-two row
     * bucket while the request publishes its current logical row count through
     * persistent device storage.  Rows beyond that logical count are real
     * addresses in the captured graph, but they are padding and must never
     * affect acceptance, bonus selection, compact metadata, or commit budget.
     *
     * This regression captures an eight-row reducer once and replays it for
     * logical widths five and three on both CUDA and ROCm.  The padded tokens
     * deliberately form additional accepts followed by a rejection, so using
     * the physical width would produce a visibly different compact result.
     * Replays also alternate the controller-owned leading committed count.
     * This prevents penalty-history state from being used as a proxy for the
     * response-ledger carry at transaction two and beyond.
     */
    TEST_P(
        GPUSamplingTest,
        GreedySummaryReusesPhysicalBucketAcrossLogicalDepthsByteExactly)
    {
        using namespace sampling_math;

        constexpr int physical_verifier_rows = 8;
        constexpr int physical_compare_rows = physical_verifier_rows - 1;
        const std::array<int, physical_verifier_rows> draft_tokens = {
            10, 20, 21, 22, 23, 24, 25, 26};
        const std::array<int, physical_verifier_rows> verifier_tokens = {
            20, 21, 22, 23, 24, 777, 26, 888};
        std::array<int, kSpeculativeBatchMaxStopTokens> stop_tokens{};
        stop_tokens.fill(-1);

        void *d_verifier_tokens = backend_->allocate(
            sizeof(verifier_tokens), device_id_);
        void *d_draft_tokens = backend_->allocate(
            sizeof(draft_tokens), device_id_);
        void *d_active_rows = backend_->allocate(sizeof(int), device_id_);
        void *d_stop_tokens = backend_->allocate(
            sizeof(stop_tokens), device_id_);
        void *d_next_leading_committed_output_count = backend_->allocate(
            sizeof(int), device_id_);
        void *d_output_tokens = backend_->allocate(
            kSpeculativeBatchMaxOutputTokens * sizeof(int), device_id_);
        void *d_output_meta = backend_->allocate(
            kSpeculativeBatchMetaCount * sizeof(int), device_id_);

        const std::array<void *, 7> allocations = {
            d_verifier_tokens,
            d_draft_tokens,
            d_active_rows,
            d_stop_tokens,
            d_next_leading_committed_output_count,
            d_output_tokens,
            d_output_meta};
        for (void *allocation : allocations)
            ASSERT_NE(allocation, nullptr);

        auto cleanup = [&]()
        {
            for (void *allocation : allocations)
                backend_->free(allocation, device_id_);
        };

        auto run_backend_case = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *const stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_verifier_tokens,
                    verifier_tokens.data(),
                    sizeof(verifier_tokens),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens,
                    draft_tokens.data(),
                    sizeof(draft_tokens),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_stop_tokens,
                    stop_tokens.data(),
                    sizeof(stop_tokens),
                    device_id_,
                    stream));
                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_
                        ->enqueueSummarizeGreedySpeculativeVerifyBatchDeviceControls(
                            d_verifier_tokens,
                            d_draft_tokens,
                            physical_compare_rows,
                            d_active_rows,
                            d_stop_tokens,
                            device_id_,
                            stream,
                            kSpeculativeBatchMaxOutputTokens,
                            d_output_tokens,
                            d_output_meta,
                            /*max_state_commit_rows_device=*/nullptr,
                            d_next_leading_committed_output_count));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                const auto verify_replay = [&](int logical_verifier_rows,
                                               int leading_committed_count)
                {
                    ASSERT_TRUE(copyHostToDevice(
                        d_active_rows,
                        &logical_verifier_rows,
                        sizeof(logical_verifier_rows),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_next_leading_committed_output_count,
                        &leading_committed_count,
                        sizeof(leading_committed_count),
                        device_id_,
                        stream));
                    ASSERT_TRUE(capture->launch());
                    ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                    std::array<int, kSpeculativeBatchMaxOutputTokens>
                        actual_tokens{};
                    std::array<int, kSpeculativeBatchMetaCount> actual_meta{};
                    ASSERT_TRUE(copyDeviceToHost(
                        actual_tokens.data(),
                        d_output_tokens,
                        sizeof(actual_tokens),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        actual_meta.data(),
                        d_output_meta,
                        sizeof(actual_meta),
                        device_id_,
                        stream));

                    std::array<int, kSpeculativeBatchMaxOutputTokens>
                        expected_tokens{};
                    std::array<int, kSpeculativeBatchMetaCount> expected_meta{};
                    summarize_greedy_speculative_verify_batch_at_commit_boundary(
                        draft_tokens[0],
                        verifier_tokens.data(),
                        draft_tokens.data(),
                        logical_verifier_rows - 1,
                        stop_tokens.data(),
                        static_cast<int>(stop_tokens.size()),
                        logical_verifier_rows,
                        expected_tokens.data(),
                        static_cast<int>(expected_tokens.size()),
                        expected_meta.data(),
                        leading_committed_count);

                    EXPECT_EQ(actual_tokens, expected_tokens)
                        << "logical_verifier_rows=" << logical_verifier_rows
                        << " leading_committed_count="
                        << leading_committed_count;
                    EXPECT_EQ(actual_meta, expected_meta)
                        << "logical_verifier_rows=" << logical_verifier_rows
                        << " leading_committed_count="
                        << leading_committed_count;
                };

                verify_replay(5, 0);
                verify_replay(3, 0);
                verify_replay(5, 1);
                verify_replay(3, 1);
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx =
                GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_backend_case(ctx);
        }
        else
        {
            auto &ctx =
                GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_backend_case(ctx);
        }

        cleanup();
    }

    /**
     * @brief Proves the captured grouped greedy penalty path is serial-decode exact.
     *
     * The production verifier does not replay rows through the host sampler.  It
     * reads a persistent device histogram, extends that history with only the
     * speculative prefix visible to each row, reduces every row in parallel,
     * summarizes the accepted prefix, and commits emitted tokens back into the
     * histogram.  The same capture is also exercised with a shorter resident
     * active-row scalar than its physical row bucket. Poison sentinels in the
     * inactive output suffix prove that neither reduction pass evaluates or
     * publishes those rows. This regression therefore covers both byte-exact
     * serial semantics and device-selected dynamic-depth totality.
     */
    TEST_P(GPUSamplingTest,
           MTPPenaltyGroupedOutcomeAndHistoryAreSerialDecodeByteExact)
    {
        using namespace sampling_math;

        constexpr int vocab_size = 248320;
        constexpr int partial_capacity = 1024;
        constexpr float presence_penalty = 0.75f;
        constexpr float frequency_penalty = 0.50f;
        constexpr std::array<int, 4> grouped_rows = {2, 4, 8, 16};

        auto run_backend_case = [&](IWorkerGPUContext &ctx,
                                    int rows,
                                    int active_rows,
                                    bool first_token_already_in_history)
        {
            ASSERT_GT(active_rows, 0);
            ASSERT_LE(active_rows, rows);
            const int compare_rows = active_rows - 1;
            std::vector<int> draft_tokens(static_cast<size_t>(rows));
            for (int row = 0; row < rows; ++row)
            {
                // Reusing three token ids makes later rows exercise increasing
                // frequency penalties rather than only binary presence state.
                draft_tokens[static_cast<size_t>(row)] = 20 + (row % 3);
            }

            std::vector<int> initial_counts(
                static_cast<size_t>(vocab_size),
                0);
            initial_counts[3] = 2;
            initial_counts[5] = 1;
            initial_counts[11] = 3;
            if (first_token_already_in_history)
            {
                // A pending terminal token was emitted by the preceding
                // transaction.  The grouped verifier must neither score nor
                // commit that first token twice.
                ++initial_counts[static_cast<size_t>(draft_tokens[0])];
            }

            std::vector<float> logits(
                static_cast<size_t>(rows) *
                    static_cast<size_t>(vocab_size),
                -50.0f);
            for (int row = 0; row < rows; ++row)
            {
                float *row_logits =
                    logits.data() +
                    static_cast<size_t>(row) *
                        static_cast<size_t>(vocab_size);
                if (row < compare_rows - 1)
                {
                    // Accept all early drafts.  The large margin keeps these
                    // rows focused on cumulative duplicate-token frequency.
                    row_logits[draft_tokens[static_cast<size_t>(row + 1)]] =
                        20.0f;
                    row_logits[70 + row] = 18.0f;
                }
                else if (row == compare_rows - 1)
                {
                    // Without the speculative-prefix penalty, the repeated
                    // token wins.  Serial history lowers it below the new token
                    // and deliberately terminates the accepted prefix.
                    row_logits[draft_tokens[static_cast<size_t>(row)]] = 7.10f;
                    row_logits[100 + row] = 6.00f;
                }
                else
                {
                    // The terminal bonus row also proves deterministic
                    // lowest-token tie breaking after fused penalty scoring.
                    row_logits[200] = 13.0f;
                    row_logits[201] = 13.0f;
                }
            }

            constexpr float inactive_value_sentinel = -12345.0f;
            constexpr int inactive_index_sentinel = -12345;
            std::vector<float> expected_values(
                static_cast<size_t>(rows),
                inactive_value_sentinel);
            std::vector<int> expected_indices(
                static_cast<size_t>(rows),
                inactive_index_sentinel);
            const int prefix_begin =
                first_token_already_in_history ? 1 : 0;
            for (int row = 0; row < active_rows; ++row)
            {
                float best_value = -std::numeric_limits<float>::max();
                int best_token = std::numeric_limits<int>::max();
                for (int token = 0; token < vocab_size; ++token)
                {
                    float value =
                        logits[static_cast<size_t>(row) *
                                   static_cast<size_t>(vocab_size) +
                               static_cast<size_t>(token)];
                    int count =
                        initial_counts[static_cast<size_t>(token)];
                    for (int history_index = prefix_begin;
                         history_index <= row;
                         ++history_index)
                    {
                        count +=
                            draft_tokens[static_cast<size_t>(history_index)] ==
                                    token
                                ? 1
                                : 0;
                    }
                    if (count > 0)
                    {
                        float penalty = 0.0f;
                        penalty += presence_penalty;
                        const float frequency_component =
                            frequency_penalty *
                            static_cast<float>(count);
                        penalty += frequency_component;
                        value -= penalty;
                    }
                    if (value > best_value ||
                        (value == best_value && token < best_token))
                    {
                        best_value = value;
                        best_token = token;
                    }
                }
                expected_values[static_cast<size_t>(row)] = best_value;
                expected_indices[static_cast<size_t>(row)] = best_token;
            }

            std::array<int, kSpeculativeBatchMaxOutputTokens>
                expected_output_tokens{};
            std::array<int, kSpeculativeBatchMetaCount>
                expected_output_meta{};
            summarize_greedy_speculative_verify_batch(
                draft_tokens[0],
                expected_indices.data(),
                draft_tokens.data(),
                compare_rows,
                /*stop_tokens=*/nullptr,
                /*stop_token_count=*/0,
                expected_output_tokens.data(),
                static_cast<int>(expected_output_tokens.size()),
                expected_output_meta.data());
            ASSERT_EQ(expected_output_meta[kSpecBatchMetaOk], 1);

            std::vector<int> expected_counts = initial_counts;
            const int commit_begin =
                first_token_already_in_history ? 1 : 0;
            const int output_count =
                expected_output_meta[kSpecBatchMetaOutputCount];
            ASSERT_GT(output_count, 0);
            const int accepted_state_count =
                first_token_already_in_history
                    ? output_count
                    : output_count - 1;
            constexpr int stopped_flag = 0;
            const int expected_next_pending_condition =
                output_count > accepted_state_count ? 1 : 0;
            for (int index = commit_begin; index < output_count; ++index)
            {
                const int token =
                    expected_output_tokens[static_cast<size_t>(index)];
                ASSERT_GE(token, 0);
                ASSERT_LT(token, vocab_size);
                ++expected_counts[static_cast<size_t>(token)];
            }

            void *d_logits = nullptr;
            void *d_argmax_values = nullptr;
            void *d_argmax_indices = nullptr;
            void *d_partial_values = nullptr;
            void *d_partial_indices = nullptr;
            void *d_draft_tokens = nullptr;
            void *d_output_tokens = nullptr;
            void *d_output_meta = nullptr;
            void *d_policy = nullptr;
            void *d_counts = nullptr;
            void *d_active_rows = nullptr;
            void *d_accepted_state_count = nullptr;
            void *d_stopped_flag = nullptr;
            auto cleanup = [&]()
            {
                void *ptrs[] = {
                    d_logits,
                    d_argmax_values,
                    d_argmax_indices,
                    d_partial_values,
                    d_partial_indices,
                    d_draft_tokens,
                    d_output_tokens,
                    d_output_meta,
                    d_policy,
                    d_counts,
                    d_active_rows,
                    d_accepted_state_count,
                    d_stopped_flag};
                for (void *ptr : ptrs)
                {
                    if (ptr)
                        backend_->free(ptr, device_id_);
                }
            };

            d_logits = backend_->allocate(
                logits.size() * sizeof(float),
                device_id_);
            d_argmax_values = backend_->allocate(
                static_cast<size_t>(rows) * sizeof(float),
                device_id_);
            d_argmax_indices = backend_->allocate(
                static_cast<size_t>(rows) * sizeof(int),
                device_id_);
            d_partial_values = backend_->allocate(
                partial_capacity * sizeof(float),
                device_id_);
            d_partial_indices = backend_->allocate(
                partial_capacity * sizeof(int),
                device_id_);
            d_draft_tokens = backend_->allocate(
                draft_tokens.size() * sizeof(int),
                device_id_);
            d_output_tokens = backend_->allocate(
                expected_output_tokens.size() * sizeof(int),
                device_id_);
            d_output_meta = backend_->allocate(
                expected_output_meta.size() * sizeof(int),
                device_id_);
            d_policy = backend_->allocate(
                sizeof(MTPGreedyPenaltyPolicy),
                device_id_);
            d_counts = backend_->allocate(
                initial_counts.size() * sizeof(int),
                device_id_);
            d_active_rows = backend_->allocate(
                sizeof(int),
                device_id_);
            d_accepted_state_count = backend_->allocate(
                sizeof(int),
                device_id_);
            d_stopped_flag = backend_->allocate(
                sizeof(int),
                device_id_);

            ASSERT_NE(d_logits, nullptr);
            ASSERT_NE(d_argmax_values, nullptr);
            ASSERT_NE(d_argmax_indices, nullptr);
            ASSERT_NE(d_partial_values, nullptr);
            ASSERT_NE(d_partial_indices, nullptr);
            ASSERT_NE(d_draft_tokens, nullptr);
            ASSERT_NE(d_output_tokens, nullptr);
            ASSERT_NE(d_output_meta, nullptr);
            ASSERT_NE(d_policy, nullptr);
            ASSERT_NE(d_counts, nullptr);
            ASSERT_NE(d_active_rows, nullptr);
            ASSERT_NE(d_accepted_state_count, nullptr);
            ASSERT_NE(d_stopped_flag, nullptr);

            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_logits,
                    logits.data(),
                    logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens,
                    draft_tokens.data(),
                    draft_tokens.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_counts,
                    initial_counts.data(),
                    initial_counts.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_active_rows,
                    &active_rows,
                    sizeof(active_rows),
                    device_id_,
                    stream));
                const std::vector<float> initial_values(
                    static_cast<size_t>(rows),
                    inactive_value_sentinel);
                const std::vector<int> initial_indices(
                    static_cast<size_t>(rows),
                    inactive_index_sentinel);
                ASSERT_TRUE(copyHostToDevice(
                    d_argmax_values,
                    initial_values.data(),
                    initial_values.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_argmax_indices,
                    initial_indices.data(),
                    initial_indices.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_accepted_state_count,
                    &accepted_state_count,
                    sizeof(accepted_state_count),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_stopped_flag,
                    &stopped_flag,
                    sizeof(stopped_flag),
                    device_id_,
                    stream));
                ASSERT_TRUE(
                    backend_
                        ->enqueueConfigureMTPGreedyPenaltyPolicyDevice(
                            d_policy,
                            presence_penalty,
                            frequency_penalty,
                            first_token_already_in_history,
                            device_id_,
                            stream));
                ASSERT_TRUE(
                    backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_
                        ->enqueueArgmaxF32BatchedRowsWithMTPPenaltiesDevice(
                            d_logits,
                            rows,
                            vocab_size,
                            d_draft_tokens,
                            d_counts,
                            d_policy,
                            d_active_rows,
                            device_id_,
                            stream,
                            d_argmax_values,
                            d_argmax_indices,
                            d_partial_values,
                            d_partial_indices,
                            partial_capacity));
                ASSERT_TRUE(
                    backend_->enqueueSummarizeGreedySpeculativeVerifyBatch(
                        d_argmax_indices,
                        d_draft_tokens,
                        compare_rows,
                        draft_tokens[0],
                        /*stop_tokens=*/nullptr,
                        /*stop_token_count=*/0,
                        device_id_,
                        stream,
                        static_cast<int>(expected_output_tokens.size()),
                        d_output_tokens,
                        d_output_meta));
                ASSERT_TRUE(
                    backend_->enqueueCommitMTPGreedyPenaltyHistoryDevice(
                        d_output_tokens,
                        d_output_meta,
                        d_policy,
                        d_accepted_state_count,
                        d_stopped_flag,
                        static_cast<int>(expected_output_tokens.size()),
                        vocab_size,
                        d_counts,
                        device_id_,
                        stream));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(
                    backend_->synchronizeStream(stream, device_id_));
            });

            std::vector<float> actual_values(static_cast<size_t>(rows));
            std::vector<int> actual_indices(static_cast<size_t>(rows), -1);
            std::array<int, kSpeculativeBatchMaxOutputTokens>
                actual_output_tokens{};
            std::array<int, kSpeculativeBatchMetaCount>
                actual_output_meta{};
            std::vector<int> actual_counts(
                static_cast<size_t>(vocab_size),
                -1);
            MTPGreedyPenaltyPolicy actual_policy{};
            ASSERT_TRUE(copyDeviceToHost(
                actual_values.data(),
                d_argmax_values,
                actual_values.size() * sizeof(float),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                actual_indices.data(),
                d_argmax_indices,
                actual_indices.size() * sizeof(int),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                actual_output_tokens.data(),
                d_output_tokens,
                actual_output_tokens.size() * sizeof(int),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                actual_output_meta.data(),
                d_output_meta,
                actual_output_meta.size() * sizeof(int),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                actual_counts.data(),
                d_counts,
                actual_counts.size() * sizeof(int),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                &actual_policy,
                d_policy,
                sizeof(actual_policy),
                device_id_));

            EXPECT_EQ(
                std::memcmp(
                    actual_values.data(),
                    expected_values.data(),
                    actual_values.size() * sizeof(float)),
                0)
                << "penalized argmax value bytes diverged for rows=" << rows
                << " active_rows=" << active_rows
                << " pending_first=" << first_token_already_in_history;
            EXPECT_EQ(actual_indices, expected_indices)
                << "rows=" << rows << " active_rows=" << active_rows;
            EXPECT_EQ(actual_output_tokens, expected_output_tokens);
            EXPECT_EQ(actual_output_meta, expected_output_meta);
            EXPECT_EQ(
                std::memcmp(
                    actual_counts.data(),
                    expected_counts.data(),
                    actual_counts.size() * sizeof(int)),
                0)
                << "persistent history bytes diverged for rows=" << rows
                << " pending_first=" << first_token_already_in_history;
            EXPECT_EQ(
                actual_policy.first_token_already_in_history,
                expected_next_pending_condition)
                << "accepted-state publication did not publish the next "
                   "pending-condition predicate for rows="
                << rows
                << " previous_pending="
                << first_token_already_in_history;

            cleanup();
        };

        auto execute_case = [&](int rows,
                                int active_rows,
                                bool first_token_already_in_history)
        {
            if (GetParam() == "CUDA")
            {
                auto &ctx =
                    GPUDeviceContextPool::instance().getNvidiaContext(
                        device_id_);
                run_backend_case(
                    ctx,
                    rows,
                    active_rows,
                    first_token_already_in_history);
            }
            else
            {
                auto &ctx =
                    GPUDeviceContextPool::instance().getAMDContext(
                        device_id_);
                run_backend_case(
                    ctx,
                    rows,
                    active_rows,
                    first_token_already_in_history);
            }
        };

        for (const int rows : grouped_rows)
        {
            for (const bool first_token_already_in_history :
                 {false, true})
            {
                execute_case(
                    rows,
                    rows,
                    first_token_already_in_history);
            }
        }

        for (const bool first_token_already_in_history : {false, true})
        {
            execute_case(
                /*rows=*/16,
                /*active_rows=*/5,
                first_token_already_in_history);
        }
    }

    /**
     * @brief Prove the stochastic in-place penalty transform is byte exact.
     *
     * M=1 models the next target token and consumes only durable device
     * history. Grouped M values additionally consume each verifier row's
     * branch-local prefix. Both CUDA and ROCm execute the production backend
     * entry point inside a captured graph before the bytes are compared with
     * serial host arithmetic.
     */
    TEST_P(GPUSamplingTest,
           MTPPenaltyLogitRowTransformIsSerialDecodeByteExactForAllDepths)
    {
        constexpr int vocab_size = 248320;
        constexpr float presence_penalty = 0.37f;
        constexpr float frequency_penalty = 0.13f;
        constexpr std::array<int, 5> row_cases = {1, 2, 4, 8, 16};

        auto run_case = [&](IWorkerGPUContext &ctx,
                            int rows,
                            bool use_verifier_prefix,
                            bool first_token_already_in_history)
        {
            std::vector<int> counts(static_cast<size_t>(vocab_size), 0);
            for (int token : {3, 7, 11, 19, 23})
                counts[static_cast<size_t>(token)] = 1 + (token % 3);

            std::vector<int> verifier_tokens(static_cast<size_t>(rows), 0);
            for (int row = 0; row < rows; ++row)
                verifier_tokens[static_cast<size_t>(row)] = 19 + (row % 4);
            if (use_verifier_prefix && first_token_already_in_history)
            {
                ++counts[static_cast<size_t>(verifier_tokens[0])];
            }

            std::vector<float> input(
                static_cast<size_t>(rows) * static_cast<size_t>(vocab_size));
            for (size_t index = 0; index < input.size(); ++index)
            {
                input[index] =
                    static_cast<float>(static_cast<int>(index % 97) - 48) /
                    16.0f;
            }
            std::vector<float> expected = input;
            const int prefix_begin =
                first_token_already_in_history ? 1 : 0;
            for (int row = 0; row < rows; ++row)
            {
                std::map<int, int> active_counts;
                for (int token : {3, 7, 11, 19, 23})
                {
                    active_counts[token] =
                        counts[static_cast<size_t>(token)];
                }
                if (use_verifier_prefix)
                {
                    for (int history_index = prefix_begin;
                         history_index <= row;
                         ++history_index)
                    {
                        ++active_counts[verifier_tokens[
                            static_cast<size_t>(history_index)]];
                    }
                }
                for (const auto &[token, count] : active_counts)
                {
                    if (count <= 0)
                        continue;
                    float penalty = 0.0f;
                    penalty += presence_penalty;
                    penalty += frequency_penalty * static_cast<float>(count);
                    expected[static_cast<size_t>(row) * vocab_size + token] -=
                        penalty;
                }
            }

            void *d_logits = backend_->allocate(
                input.size() * sizeof(float),
                device_id_);
            void *d_tokens = backend_->allocate(
                verifier_tokens.size() * sizeof(int),
                device_id_);
            void *d_counts = backend_->allocate(
                counts.size() * sizeof(int),
                device_id_);
            void *d_policy = backend_->allocate(
                sizeof(MTPGreedyPenaltyPolicy),
                device_id_);
            ASSERT_NE(d_logits, nullptr);
            ASSERT_NE(d_tokens, nullptr);
            ASSERT_NE(d_counts, nullptr);
            ASSERT_NE(d_policy, nullptr);

            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_logits,
                    input.data(),
                    input.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_tokens,
                    verifier_tokens.data(),
                    verifier_tokens.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_counts,
                    counts.data(),
                    counts.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(
                    backend_->enqueueConfigureMTPGreedyPenaltyPolicyDevice(
                        d_policy,
                        presence_penalty,
                        frequency_penalty,
                        first_token_already_in_history,
                        device_id_,
                        stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueApplyMTPPenaltiesToF32RowsDevice(
                        d_logits,
                        rows,
                        vocab_size,
                        vocab_size,
                        use_verifier_prefix ? d_tokens : nullptr,
                        d_counts,
                        d_policy,
                        device_id_,
                        stream));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });

            std::vector<float> actual(input.size());
            ASSERT_TRUE(copyDeviceToHost(
                actual.data(),
                d_logits,
                actual.size() * sizeof(float),
                device_id_));
            EXPECT_EQ(
                std::memcmp(
                    actual.data(),
                    expected.data(),
                    actual.size() * sizeof(float)),
                0)
                << "penalty row bytes diverged for rows=" << rows
                << " verifier_prefix=" << use_verifier_prefix
                << " pending_first=" << first_token_already_in_history;

            backend_->free(d_logits, device_id_);
            backend_->free(d_tokens, device_id_);
            backend_->free(d_counts, device_id_);
            backend_->free(d_policy, device_id_);
        };

        auto run_on_backend = [&](int rows,
                                  bool use_verifier_prefix,
                                  bool pending_first)
        {
            if (GetParam() == "CUDA")
            {
                auto &ctx =
                    GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
                run_case(ctx, rows, use_verifier_prefix, pending_first);
            }
            else
            {
                auto &ctx =
                    GPUDeviceContextPool::instance().getAMDContext(device_id_);
                run_case(ctx, rows, use_verifier_prefix, pending_first);
            }
        };

        run_on_backend(/*rows=*/1, /*use_verifier_prefix=*/false, false);
        for (int rows : row_cases)
        {
            if (rows == 1)
                continue;
            run_on_backend(rows, /*use_verifier_prefix=*/true, false);
            run_on_backend(rows, /*use_verifier_prefix=*/true, true);
        }
    }

    /**
     * @brief Prove captured verifier preparation equals its production primitives.
     *
     * The parent MTP graph consumes one typed node that optionally applies
     * durable and branch-local penalties before building every compact target
     * distribution. This regression compares that graph node byte-for-byte
     * with the same two backend entry points enqueued directly. It spans the
     * complete practical depth envelope, both first-token history policies,
     * and several Top-K geometries on CUDA and ROCm.
     *
     * The test also replays each graph after replacing logits and history in
     * place. A captured host value, stale pointer, different floating-point
     * operation order, or accidental omission of either operation therefore
     * changes logits, token ids, or probabilities and fails the comparison.
     */
    TEST_P(
        GPUSamplingTest,
        CapturedStochasticTargetPreparationIsProductionByteExactAcrossDepths)
    {
        constexpr int vocab_size = 4096;
        constexpr int output_stride = 32;
        constexpr float presence_penalty = 0.31F;
        constexpr float frequency_penalty = 0.17F;
        constexpr float top_p = 0.91F;
        constexpr float temperature = 0.73F;
        const std::array<int, 4> row_cases = {2, 4, 8, 16};
        const std::array<int, 4> top_k_cases = {1, 7, 20, 32};

        auto run_case = [&](IWorkerGPUContext &ctx,
                            int rows,
                            int top_k,
                            bool first_token_already_in_history)
        {
            const size_t logit_count =
                static_cast<size_t>(rows) * vocab_size;
            const size_t output_count =
                static_cast<size_t>(rows) * output_stride;
            const size_t scratch_capacity =
                static_cast<size_t>(rows) * 128U * output_stride;

            std::vector<float> logits(logit_count);
            std::vector<int32_t> verifier_tokens(
                static_cast<size_t>(rows));
            std::vector<int32_t> generated_counts(
                static_cast<size_t>(vocab_size),
                0);
            const std::vector<int32_t> output_id_sentinel(
                output_count,
                -777);
            const std::vector<float> output_probability_sentinel(
                output_count,
                -123.25F);
            for (int row = 0; row < rows; ++row)
            {
                verifier_tokens[static_cast<size_t>(row)] =
                    37 + (row * 53) % (vocab_size - 37);
                for (int token = 0; token < vocab_size; ++token)
                {
                    logits[static_cast<size_t>(row) * vocab_size + token] =
                        -8.0F +
                        static_cast<float>(
                            (token * 19 + row * 71) % 1009) /
                            113.0F;
                }
            }
            for (int token : {3, 37, 101, 997, 2047, 4095})
            {
                generated_counts[static_cast<size_t>(token)] =
                    1 + token % 4;
            }
            if (first_token_already_in_history)
            {
                ++generated_counts[static_cast<size_t>(
                    verifier_tokens.front())];
            }

            void *d_reference_logits = backend_->allocate(
                logit_count * sizeof(float), device_id_);
            void *d_captured_logits = backend_->allocate(
                logit_count * sizeof(float), device_id_);
            void *d_tokens = backend_->allocate(
                verifier_tokens.size() * sizeof(int32_t), device_id_);
            void *d_counts = backend_->allocate(
                generated_counts.size() * sizeof(int32_t), device_id_);
            void *d_reference_policy = backend_->allocate(
                sizeof(MTPGreedyPenaltyPolicy), device_id_);
            void *d_captured_policy = backend_->allocate(
                sizeof(MTPGreedyPenaltyPolicy), device_id_);
            void *d_reference_ids = backend_->allocate(
                output_count * sizeof(int32_t), device_id_);
            void *d_captured_ids = backend_->allocate(
                output_count * sizeof(int32_t), device_id_);
            void *d_reference_probs = backend_->allocate(
                output_count * sizeof(float), device_id_);
            void *d_captured_probs = backend_->allocate(
                output_count * sizeof(float), device_id_);
            void *d_reference_scratch_values = backend_->allocate(
                scratch_capacity * sizeof(float), device_id_);
            void *d_captured_scratch_values = backend_->allocate(
                scratch_capacity * sizeof(float), device_id_);
            void *d_reference_scratch_indices = backend_->allocate(
                scratch_capacity * sizeof(int32_t), device_id_);
            void *d_captured_scratch_indices = backend_->allocate(
                scratch_capacity * sizeof(int32_t), device_id_);

            const std::array<void *, 14> allocations = {
                d_reference_logits,
                d_captured_logits,
                d_tokens,
                d_counts,
                d_reference_policy,
                d_captured_policy,
                d_reference_ids,
                d_captured_ids,
                d_reference_probs,
                d_captured_probs,
                d_reference_scratch_values,
                d_captured_scratch_values,
                d_reference_scratch_indices,
                d_captured_scratch_indices};
            for (void *allocation : allocations)
                ASSERT_NE(allocation, nullptr);

            auto cleanup = [&]()
            {
                for (void *allocation : allocations)
                {
                    if (allocation)
                        backend_->free(allocation, device_id_);
                }
            };

            const DeviceId execution_device =
                GetParam() == "CUDA"
                    ? DeviceId::cuda(device_id_)
                    : DeviceId::rocm(device_id_);
            auto execution_context = IDeviceContext::create(execution_device);
            ASSERT_NE(execution_context, nullptr);

            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_tokens,
                    verifier_tokens.data(),
                    verifier_tokens.size() * sizeof(int32_t),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_counts,
                    generated_counts.data(),
                    generated_counts.size() * sizeof(int32_t),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_reference_logits,
                    logits.data(),
                    logit_count * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_captured_logits,
                    logits.data(),
                    logit_count * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_reference_ids,
                    output_id_sentinel.data(),
                    output_count * sizeof(int32_t),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_captured_ids,
                    output_id_sentinel.data(),
                    output_count * sizeof(int32_t),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_reference_probs,
                    output_probability_sentinel.data(),
                    output_count * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_captured_probs,
                    output_probability_sentinel.data(),
                    output_count * sizeof(float),
                    device_id_,
                    stream));

                ASSERT_TRUE(
                    backend_->enqueueConfigureMTPGreedyPenaltyPolicyDevice(
                        d_reference_policy,
                        presence_penalty,
                        frequency_penalty,
                        first_token_already_in_history,
                        device_id_,
                        stream));
                ASSERT_TRUE(
                    backend_->enqueueConfigureMTPGreedyPenaltyPolicyDevice(
                        d_captured_policy,
                        presence_penalty,
                        frequency_penalty,
                        first_token_already_in_history,
                        device_id_,
                        stream));
                ASSERT_TRUE(
                    backend_->enqueueApplyMTPPenaltiesToF32RowsDevice(
                        d_reference_logits,
                        rows,
                        vocab_size,
                        vocab_size,
                        d_tokens,
                        d_counts,
                        d_reference_policy,
                        device_id_,
                        stream));
                ASSERT_TRUE(
                    backend_->enqueueBuildTopKTopPDistributionsF32Device(
                        d_reference_logits,
                        rows,
                        vocab_size,
                        vocab_size,
                        top_k,
                        top_p,
                        temperature,
                        device_id_,
                        stream,
                        d_reference_ids,
                        output_stride,
                        d_reference_probs,
                        d_reference_scratch_values,
                        d_reference_scratch_indices,
                        scratch_capacity));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                MTPStochasticTargetDistributionStage::Params stage_params;
                stage_params.device_id = execution_device;
                stage_params.backend = backend_;
                stage_params.logits_device =
                    static_cast<float *>(d_captured_logits);
                stage_params.first_logit_row = 0;
                stage_params.row_count = rows;
                stage_params.vocab_size = vocab_size;
                stage_params.logits_row_stride = vocab_size;
                stage_params.apply_penalties = true;
                stage_params.verifier_input_tokens_device =
                    static_cast<const int32_t *>(d_tokens);
                stage_params.generated_token_counts_device =
                    static_cast<const int32_t *>(d_counts);
                stage_params.penalty_policy_device = d_captured_policy;
                stage_params.top_k = top_k;
                stage_params.top_p = top_p;
                stage_params.temperature = temperature;
                stage_params.target_token_ids_device =
                    static_cast<int32_t *>(d_captured_ids);
                stage_params.target_probs_device =
                    static_cast<float *>(d_captured_probs);
                stage_params.first_target_slot = 0;
                stage_params.target_row_stride = output_stride;
                stage_params.topk_partial_values_device =
                    static_cast<float *>(d_captured_scratch_values);
                stage_params.topk_partial_indices_device =
                    static_cast<int32_t *>(d_captured_scratch_indices);
                stage_params.topk_partial_capacity = scratch_capacity;
                MTPStochasticTargetDistributionStage stage(stage_params);
                ASSERT_TRUE(stage.getParams().apply_penalties);
                ASSERT_EQ(
                    stage.getParams().penalty_policy_device,
                    d_captured_policy);
                EXPECT_THROW(
                    stage.execute(execution_context.get()),
                    std::logic_error)
                    << "typed target preparation must fail hard on a null stream";
                stage.setGPUStream(stream);

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                const bool stage_enqueued =
                    stage.execute(execution_context.get());
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(stage_enqueued);
                ASSERT_GE(capture->nodeCount(), 3U)
                    << "target preparation capture omitted required kernels";
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                /*
                 * Replace only resident data, not captured pointers or policy.
                 * The second launch must consume the new logits/history and
                 * remain byte-identical to freshly enqueued production calls.
                 */
                for (size_t index = 0; index < logits.size(); ++index)
                {
                    logits[index] =
                        -6.0F +
                        static_cast<float>((index * 29 + rows * 41) % 1237) /
                            127.0F;
                }
                for (size_t token = 0;
                     token < generated_counts.size();
                     ++token)
                {
                    generated_counts[token] =
                        token % 509 == 0 ? 1 + static_cast<int>(token % 5)
                                         : 0;
                }
                if (first_token_already_in_history)
                {
                    ++generated_counts[static_cast<size_t>(
                        verifier_tokens.front())];
                }
                ASSERT_TRUE(copyHostToDevice(
                    d_counts,
                    generated_counts.data(),
                    generated_counts.size() * sizeof(int32_t),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_reference_logits,
                    logits.data(),
                    logit_count * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_captured_logits,
                    logits.data(),
                    logit_count * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(
                    backend_->enqueueConfigureMTPGreedyPenaltyPolicyDevice(
                        d_reference_policy,
                        presence_penalty,
                        frequency_penalty,
                        first_token_already_in_history,
                        device_id_,
                        stream));
                ASSERT_TRUE(
                    backend_->enqueueApplyMTPPenaltiesToF32RowsDevice(
                        d_reference_logits,
                        rows,
                        vocab_size,
                        vocab_size,
                        d_tokens,
                        d_counts,
                        d_reference_policy,
                        device_id_,
                        stream));
                ASSERT_TRUE(
                    backend_->enqueueBuildTopKTopPDistributionsF32Device(
                        d_reference_logits,
                        rows,
                        vocab_size,
                        vocab_size,
                        top_k,
                        top_p,
                        temperature,
                        device_id_,
                        stream,
                        d_reference_ids,
                        output_stride,
                        d_reference_probs,
                        d_reference_scratch_values,
                        d_reference_scratch_indices,
                        scratch_capacity));
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });

            std::vector<float> reference_logits(logit_count);
            std::vector<float> captured_logits(logit_count);
            std::vector<int32_t> reference_ids(output_count);
            std::vector<int32_t> captured_ids(output_count);
            std::vector<float> reference_probs(output_count);
            std::vector<float> captured_probs(output_count);
            MTPGreedyPenaltyPolicy reference_policy{};
            MTPGreedyPenaltyPolicy captured_policy{};
            ASSERT_TRUE(copyDeviceToHost(
                reference_logits.data(),
                d_reference_logits,
                logit_count * sizeof(float),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                captured_logits.data(),
                d_captured_logits,
                logit_count * sizeof(float),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                reference_ids.data(),
                d_reference_ids,
                output_count * sizeof(int32_t),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                captured_ids.data(),
                d_captured_ids,
                output_count * sizeof(int32_t),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                reference_probs.data(),
                d_reference_probs,
                output_count * sizeof(float),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                captured_probs.data(),
                d_captured_probs,
                output_count * sizeof(float),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                &reference_policy,
                d_reference_policy,
                sizeof(reference_policy),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                &captured_policy,
                d_captured_policy,
                sizeof(captured_policy),
                device_id_));

            EXPECT_EQ(
                std::memcmp(
                    &reference_policy,
                    &captured_policy,
                    sizeof(reference_policy)),
                0)
                << "captured policy bytes differ: reference={presence="
                << reference_policy.presence_penalty
                << ", frequency=" << reference_policy.frequency_penalty
                << ", first_in_history="
                << reference_policy.first_token_already_in_history
                << ", enabled=" << reference_policy.enabled
                << "} captured={presence=" << captured_policy.presence_penalty
                << ", frequency=" << captured_policy.frequency_penalty
                << ", first_in_history="
                << captured_policy.first_token_already_in_history
                << ", enabled=" << captured_policy.enabled << "}";

            const auto first_logit_mismatch = std::mismatch(
                reference_logits.begin(),
                reference_logits.end(),
                captured_logits.begin(),
                [](float lhs, float rhs)
                {
                    return std::memcmp(&lhs, &rhs, sizeof(float)) == 0;
                });
            if (first_logit_mismatch.first != reference_logits.end())
            {
                const size_t index = static_cast<size_t>(
                    first_logit_mismatch.first - reference_logits.begin());
                const size_t token = index % static_cast<size_t>(vocab_size);
                ADD_FAILURE()
                    << "first captured logit mismatch at row="
                    << index / static_cast<size_t>(vocab_size)
                    << " token=" << token
                    << " history_count=" << generated_counts[token]
                    << " reference=" << *first_logit_mismatch.first
                    << " captured=" << *first_logit_mismatch.second;
            }

            EXPECT_EQ(
                std::memcmp(
                    reference_logits.data(),
                    captured_logits.data(),
                    logit_count * sizeof(float)),
                0)
                << "captured penalty bytes differ at rows=" << rows;
            EXPECT_EQ(
                std::memcmp(
                    reference_ids.data(),
                    captured_ids.data(),
                    output_count * sizeof(int32_t)),
                0)
                << "captured target ids differ at rows=" << rows;
            EXPECT_EQ(
                std::memcmp(
                    reference_probs.data(),
                    captured_probs.data(),
                    output_count * sizeof(float)),
                0)
                << "captured target probabilities differ at rows=" << rows;
            cleanup();
        };

        for (size_t case_index = 0;
             case_index < row_cases.size();
             ++case_index)
        {
            IWorkerGPUContext &ctx =
                GetParam() == "CUDA"
                    ? GPUDeviceContextPool::instance().getNvidiaContext(
                          device_id_)
                    : GPUDeviceContextPool::instance().getAMDContext(
                          device_id_);
            run_case(
                ctx,
                row_cases[case_index],
                top_k_cases[case_index],
                case_index % 2 != 0);
        }
    }

    /**
     * @brief Prove one maximum-capacity target graph obeys every resident M.
     *
     * The production device loop must not recapture target preparation when its
     * dynamic MTP depth changes. This test captures the real penalty and compact
     * Top-K/Top-P stage once at sixteen physical rows, then changes only one
     * device-owned active-row scalar for every logical verifier width from two
     * through sixteen. A direct logical-width production enqueue is the oracle.
     *
     * Full-buffer byte comparisons are intentional. They prove that active rows
     * retain serial arithmetic while inactive logit and output suffixes remain
     * untouched, ruling out both stale-tail publication and wasted vocabulary
     * scans hidden behind a maximum launch geometry.
     */
    TEST_P(
        GPUSamplingTest,
        CapturedMaximumStochasticTargetGraphUsesDeviceOwnedMTotalActiveRows)
    {
        constexpr int physical_rows = 16;
        constexpr int vocab_size = 4096;
        constexpr int top_k = 20;
        constexpr int output_stride = 32;
        constexpr float top_p = 0.91F;
        constexpr float temperature = 0.73F;
        constexpr float presence_penalty = 0.31F;
        constexpr float frequency_penalty = 0.17F;
        constexpr size_t logit_count =
            static_cast<size_t>(physical_rows) * vocab_size;
        constexpr size_t output_count =
            static_cast<size_t>(physical_rows) * output_stride;
        constexpr size_t scratch_capacity =
            static_cast<size_t>(physical_rows) * 128U * output_stride;

        std::vector<float> logits(logit_count);
        std::vector<int32_t> verifier_tokens(physical_rows);
        std::vector<int32_t> generated_counts(vocab_size, 0);
        const std::vector<int32_t> output_id_sentinel(output_count, -777);
        const std::vector<float> output_probability_sentinel(
            output_count,
            -123.25F);
        for (int row = 0; row < physical_rows; ++row)
        {
            verifier_tokens[static_cast<size_t>(row)] =
                37 + (row * 53) % (vocab_size - 37);
            for (int token = 0; token < vocab_size; ++token)
            {
                logits[static_cast<size_t>(row) * vocab_size + token] =
                    -8.0F +
                    static_cast<float>((token * 19 + row * 71) % 1009) /
                        113.0F;
            }
        }
        for (int token : {3, 37, 101, 997, 2047, 4095})
        {
            generated_counts[static_cast<size_t>(token)] = 1 + token % 4;
        }

        void *d_reference_logits =
            backend_->allocate(logit_count * sizeof(float), device_id_);
        void *d_captured_logits =
            backend_->allocate(logit_count * sizeof(float), device_id_);
        void *d_tokens = backend_->allocate(
            verifier_tokens.size() * sizeof(int32_t), device_id_);
        void *d_counts = backend_->allocate(
            generated_counts.size() * sizeof(int32_t), device_id_);
        void *d_reference_policy = backend_->allocate(
            sizeof(MTPGreedyPenaltyPolicy), device_id_);
        void *d_captured_policy = backend_->allocate(
            sizeof(MTPGreedyPenaltyPolicy), device_id_);
        void *d_reference_ids =
            backend_->allocate(output_count * sizeof(int32_t), device_id_);
        void *d_captured_ids =
            backend_->allocate(output_count * sizeof(int32_t), device_id_);
        void *d_reference_probs =
            backend_->allocate(output_count * sizeof(float), device_id_);
        void *d_captured_probs =
            backend_->allocate(output_count * sizeof(float), device_id_);
        void *d_reference_scratch_values = backend_->allocate(
            scratch_capacity * sizeof(float), device_id_);
        void *d_captured_scratch_values = backend_->allocate(
            scratch_capacity * sizeof(float), device_id_);
        void *d_reference_scratch_indices = backend_->allocate(
            scratch_capacity * sizeof(int32_t), device_id_);
        void *d_captured_scratch_indices = backend_->allocate(
            scratch_capacity * sizeof(int32_t), device_id_);
        void *d_active_rows = backend_->allocate(sizeof(int32_t), device_id_);

        const std::array<void *, 15> allocations = {
            d_reference_logits,
            d_captured_logits,
            d_tokens,
            d_counts,
            d_reference_policy,
            d_captured_policy,
            d_reference_ids,
            d_captured_ids,
            d_reference_probs,
            d_captured_probs,
            d_reference_scratch_values,
            d_captured_scratch_values,
            d_reference_scratch_indices,
            d_captured_scratch_indices,
            d_active_rows};
        for (void *allocation : allocations)
            ASSERT_NE(allocation, nullptr);

        const DeviceId execution_device =
            GetParam() == "CUDA" ? DeviceId::cuda(device_id_)
                                 : DeviceId::rocm(device_id_);
        auto execution_context = IDeviceContext::create(execution_device);
        ASSERT_NE(execution_context, nullptr);
        IWorkerGPUContext &ctx =
            GetParam() == "CUDA"
                ? GPUDeviceContextPool::instance().getNvidiaContext(device_id_)
                : GPUDeviceContextPool::instance().getAMDContext(device_id_);

        ctx.submitAndWait([&]()
        {
            void *const stream = ctx.defaultStream();
            ASSERT_NE(stream, nullptr);
            ASSERT_TRUE(copyHostToDevice(
                d_tokens,
                verifier_tokens.data(),
                verifier_tokens.size() * sizeof(int32_t),
                device_id_,
                stream));
            ASSERT_TRUE(copyHostToDevice(
                d_counts,
                generated_counts.data(),
                generated_counts.size() * sizeof(int32_t),
                device_id_,
                stream));
            ASSERT_TRUE(
                backend_->enqueueConfigureMTPGreedyPenaltyPolicyDevice(
                    d_reference_policy,
                    presence_penalty,
                    frequency_penalty,
                    false,
                    device_id_,
                    stream));
            ASSERT_TRUE(
                backend_->enqueueConfigureMTPGreedyPenaltyPolicyDevice(
                    d_captured_policy,
                    presence_penalty,
                    frequency_penalty,
                    false,
                    device_id_,
                    stream));

            MTPStochasticTargetDistributionStage::Params stage_params;
            stage_params.device_id = execution_device;
            stage_params.backend = backend_;
            stage_params.logits_device =
                static_cast<float *>(d_captured_logits);
            stage_params.first_logit_row = 0;
            stage_params.row_count = physical_rows;
            stage_params.vocab_size = vocab_size;
            stage_params.logits_row_stride = vocab_size;
            stage_params.active_rows_device =
                static_cast<const int32_t *>(d_active_rows);
            stage_params.apply_penalties = true;
            stage_params.verifier_input_tokens_device =
                static_cast<const int32_t *>(d_tokens);
            stage_params.generated_token_counts_device =
                static_cast<const int32_t *>(d_counts);
            stage_params.penalty_policy_device = d_captured_policy;
            stage_params.top_k = top_k;
            stage_params.top_p = top_p;
            stage_params.temperature = temperature;
            stage_params.target_token_ids_device =
                static_cast<int32_t *>(d_captured_ids);
            stage_params.target_probs_device =
                static_cast<float *>(d_captured_probs);
            stage_params.first_target_slot = 0;
            stage_params.target_row_stride = output_stride;
            stage_params.topk_partial_values_device =
                static_cast<float *>(d_captured_scratch_values);
            stage_params.topk_partial_indices_device =
                static_cast<int32_t *>(d_captured_scratch_indices);
            stage_params.topk_partial_capacity = scratch_capacity;

            MTPStochasticTargetDistributionStage stage(stage_params);
            stage.setGPUStream(stream);
            auto unmasked_params = stage_params;
            unmasked_params.active_rows_device = nullptr;
            EXPECT_FALSE(stage.hasSameCaptureIdentity(unmasked_params));

            auto capture = ctx.createGraphCapture(stream);
            ASSERT_NE(capture, nullptr);
            ASSERT_TRUE(capture->beginCapture());
            const bool stage_enqueued = stage.execute(execution_context.get());
            ASSERT_TRUE(capture->endCapture());
            ASSERT_TRUE(stage_enqueued);
            ASSERT_GE(capture->nodeCount(), 3U);
            ASSERT_TRUE(capture->instantiate());

            std::vector<float> reference_logits(logit_count);
            std::vector<float> captured_logits(logit_count);
            std::vector<int32_t> reference_ids(output_count);
            std::vector<int32_t> captured_ids(output_count);
            std::vector<float> reference_probs(output_count);
            std::vector<float> captured_probs(output_count);

            for (int active_rows = 2;
                 active_rows <= physical_rows;
                 ++active_rows)
            {
                ASSERT_TRUE(copyHostToDevice(
                    d_reference_logits,
                    logits.data(),
                    logit_count * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_captured_logits,
                    logits.data(),
                    logit_count * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_reference_ids,
                    output_id_sentinel.data(),
                    output_count * sizeof(int32_t),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_captured_ids,
                    output_id_sentinel.data(),
                    output_count * sizeof(int32_t),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_reference_probs,
                    output_probability_sentinel.data(),
                    output_count * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_captured_probs,
                    output_probability_sentinel.data(),
                    output_count * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_active_rows,
                    &active_rows,
                    sizeof(active_rows),
                    device_id_,
                    stream));

                ASSERT_TRUE(
                    backend_->enqueueApplyMTPPenaltiesToF32RowsDevice(
                        d_reference_logits,
                        active_rows,
                        vocab_size,
                        vocab_size,
                        d_tokens,
                        d_counts,
                        d_reference_policy,
                        device_id_,
                        stream));
                ASSERT_TRUE(
                    backend_->enqueueBuildTopKTopPDistributionsF32Device(
                        d_reference_logits,
                        active_rows,
                        vocab_size,
                        vocab_size,
                        top_k,
                        top_p,
                        temperature,
                        device_id_,
                        stream,
                        d_reference_ids,
                        output_stride,
                        d_reference_probs,
                        d_reference_scratch_values,
                        d_reference_scratch_indices,
                        scratch_capacity));
                ASSERT_TRUE(capture->launch());

                ASSERT_TRUE(copyDeviceToHost(
                    reference_logits.data(),
                    d_reference_logits,
                    logit_count * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyDeviceToHost(
                    captured_logits.data(),
                    d_captured_logits,
                    logit_count * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyDeviceToHost(
                    reference_ids.data(),
                    d_reference_ids,
                    output_count * sizeof(int32_t),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyDeviceToHost(
                    captured_ids.data(),
                    d_captured_ids,
                    output_count * sizeof(int32_t),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyDeviceToHost(
                    reference_probs.data(),
                    d_reference_probs,
                    output_count * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyDeviceToHost(
                    captured_probs.data(),
                    d_captured_probs,
                    output_count * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_EQ(
                    std::memcmp(
                        reference_logits.data(),
                        captured_logits.data(),
                        logit_count * sizeof(float)),
                    0)
                    << "maximum target graph changed inactive logits at M="
                    << active_rows;
                EXPECT_EQ(
                    std::memcmp(
                        reference_ids.data(),
                        captured_ids.data(),
                        output_count * sizeof(int32_t)),
                    0)
                    << "maximum target graph changed inactive ids at M="
                    << active_rows;
                EXPECT_EQ(
                    std::memcmp(
                        reference_probs.data(),
                        captured_probs.data(),
                        output_count * sizeof(float)),
                    0)
                    << "maximum target graph changed inactive probabilities at M="
                    << active_rows;
            }
        });

        for (void *allocation : allocations)
        {
            if (allocation)
                backend_->free(allocation, device_id_);
        }
    }

    /**
     * @brief Prove captured proposal publication equals direct production work.
     *
     * Every tested depth binds a different persistent destination slot and a
     * correspondingly longer speculative branch.  The reference path invokes
     * the same backend penalty and argmax primitives directly; the candidate
     * path captures the typed stage, replays it after changing only resident
     * data, and must reproduce policy, logits, max value, and token bytes.
     * Disabled-penalty cases additionally prove the stochastic production lane
     * captures only deterministic argmax publication.
     */
    TEST_P(
        GPUSamplingTest,
        CapturedMTPDraftPublicationIsByteExactAcrossDepthsAndPolicies)
    {
        constexpr int vocab_size = 248320;
        constexpr int slot_capacity = 15;
        constexpr int scratch_capacity = 1024;
        constexpr float presence_penalty = 0.29F;
        constexpr float frequency_penalty = 0.11F;
        const std::array<int, 5> depth_cases = {0, 1, 3, 7, 14};

        auto run_case = [&](IWorkerGPUContext &ctx,
                            int depth,
                            bool apply_penalties,
                            bool first_token_already_in_history)
        {
            std::vector<float> logits(static_cast<size_t>(vocab_size));
            std::vector<int32_t> counts(
                static_cast<size_t>(vocab_size),
                0);
            std::vector<int32_t> drafts(
                static_cast<size_t>(slot_capacity),
                -1);
            int32_t condition_token = 37;
            for (int token = 0; token < vocab_size; ++token)
            {
                logits[static_cast<size_t>(token)] =
                    -7.0F +
                    static_cast<float>((token * 31 + depth * 17) % 1237) /
                        127.0F;
            }
            for (int token : {3, 37, 101, 997, 2047, 4095})
                counts[static_cast<size_t>(token)] = 1 + token % 4;
            if (first_token_already_in_history)
                ++counts[static_cast<size_t>(condition_token)];
            for (int prior = 0; prior < depth; ++prior)
                drafts[static_cast<size_t>(prior)] = 37 + prior % 5;

            void *d_reference_logits = backend_->allocate(
                logits.size() * sizeof(float), device_id_);
            void *d_captured_logits = backend_->allocate(
                logits.size() * sizeof(float), device_id_);
            void *d_condition = backend_->allocate(
                sizeof(condition_token), device_id_);
            void *d_drafts = backend_->allocate(
                drafts.size() * sizeof(int32_t), device_id_);
            void *d_counts = backend_->allocate(
                counts.size() * sizeof(int32_t), device_id_);
            void *d_reference_policy = backend_->allocate(
                sizeof(MTPGreedyPenaltyPolicy), device_id_);
            void *d_captured_policy = backend_->allocate(
                sizeof(MTPGreedyPenaltyPolicy), device_id_);
            void *d_reference_values = backend_->allocate(
                slot_capacity * sizeof(float), device_id_);
            void *d_captured_values = backend_->allocate(
                slot_capacity * sizeof(float), device_id_);
            void *d_reference_tokens = backend_->allocate(
                slot_capacity * sizeof(int32_t), device_id_);
            void *d_captured_tokens = backend_->allocate(
                slot_capacity * sizeof(int32_t), device_id_);
            void *d_reference_partial_values = backend_->allocate(
                scratch_capacity * sizeof(float), device_id_);
            void *d_captured_partial_values = backend_->allocate(
                scratch_capacity * sizeof(float), device_id_);
            void *d_reference_partial_indices = backend_->allocate(
                scratch_capacity * sizeof(int32_t), device_id_);
            void *d_captured_partial_indices = backend_->allocate(
                scratch_capacity * sizeof(int32_t), device_id_);
            void *d_reference_chain_token = backend_->allocate(
                sizeof(int32_t), device_id_);
            void *d_captured_chain_token = backend_->allocate(
                sizeof(int32_t), device_id_);
            void *d_reference_chain_position = backend_->allocate(
                sizeof(int32_t), device_id_);
            void *d_captured_chain_position = backend_->allocate(
                sizeof(int32_t), device_id_);

            const std::array<void *, 19> allocations = {
                d_reference_logits,
                d_captured_logits,
                d_condition,
                d_drafts,
                d_counts,
                d_reference_policy,
                d_captured_policy,
                d_reference_values,
                d_captured_values,
                d_reference_tokens,
                d_captured_tokens,
                d_reference_partial_values,
                d_captured_partial_values,
                d_reference_partial_indices,
                d_captured_partial_indices,
                d_reference_chain_token,
                d_captured_chain_token,
                d_reference_chain_position,
                d_captured_chain_position};
            for (void *allocation : allocations)
                ASSERT_NE(allocation, nullptr);

            const DeviceId execution_device =
                GetParam() == "CUDA"
                    ? DeviceId::cuda(device_id_)
                    : DeviceId::rocm(device_id_);
            auto execution_context = IDeviceContext::create(execution_device);
            ASSERT_NE(execution_context, nullptr);

            std::vector<float> value_sentinel(
                static_cast<size_t>(slot_capacity),
                -123.5F);
            std::vector<int32_t> token_sentinel(
                static_cast<size_t>(slot_capacity),
                -777);
            int32_t chain_token_sentinel = -901;
            int32_t chain_position = 409 + depth;

            ctx.submitAndWait([&]()
            {
                void *const stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_condition,
                    &condition_token,
                    sizeof(condition_token),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_drafts,
                    drafts.data(),
                    drafts.size() * sizeof(int32_t),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_counts,
                    counts.data(),
                    counts.size() * sizeof(int32_t),
                    device_id_,
                    stream));

                auto reset_inputs = [&]()
                {
                    ASSERT_TRUE(copyHostToDevice(
                        d_reference_logits,
                        logits.data(),
                        logits.size() * sizeof(float),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_captured_logits,
                        logits.data(),
                        logits.size() * sizeof(float),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_reference_values,
                        value_sentinel.data(),
                        value_sentinel.size() * sizeof(float),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_captured_values,
                        value_sentinel.data(),
                        value_sentinel.size() * sizeof(float),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_reference_tokens,
                        token_sentinel.data(),
                        token_sentinel.size() * sizeof(int32_t),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_captured_tokens,
                        token_sentinel.data(),
                        token_sentinel.size() * sizeof(int32_t),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_reference_chain_token,
                        &chain_token_sentinel,
                        sizeof(chain_token_sentinel),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_captured_chain_token,
                        &chain_token_sentinel,
                        sizeof(chain_token_sentinel),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_reference_chain_position,
                        &chain_position,
                        sizeof(chain_position),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_captured_chain_position,
                        &chain_position,
                        sizeof(chain_position),
                        device_id_,
                        stream));
                };

                auto enqueue_reference = [&]()
                {
                    if (apply_penalties)
                    {
                        ASSERT_TRUE(backend_
                            ->enqueueConfigureMTPGreedyPenaltyPolicyDevice(
                                d_reference_policy,
                                presence_penalty,
                                frequency_penalty,
                                first_token_already_in_history,
                                device_id_,
                                stream));
                        ASSERT_TRUE(backend_
                            ->enqueueApplyMTPBranchPenaltiesToF32RowDevice(
                                d_reference_logits,
                                vocab_size,
                                d_condition,
                                d_drafts,
                                depth,
                                d_counts,
                                d_reference_policy,
                                device_id_,
                                stream));
                    }
                    ASSERT_TRUE(backend_
                        ->enqueueArgmaxF32BatchedRowsAndPublishMTPChainDevice(
                        d_reference_logits,
                        /*rows=*/1,
                        vocab_size,
                        device_id_,
                        stream,
                        static_cast<float *>(d_reference_values) + depth,
                        static_cast<int32_t *>(d_reference_tokens) + depth,
                        d_reference_chain_token,
                        d_reference_chain_position,
                        /*chain_position_increment=*/1,
                        d_reference_partial_values,
                        d_reference_partial_indices,
                        scratch_capacity,
                        /*output_stride=*/1));
                };

                MTPDraftTokenPublicationStage::Params stage_params;
                stage_params.device_id = execution_device;
                stage_params.backend = backend_;
                stage_params.logits_row_device =
                    static_cast<float *>(d_captured_logits);
                stage_params.logits_row = 0;
                stage_params.vocab_size = vocab_size;
                stage_params.apply_penalties = apply_penalties;
                stage_params.first_condition_token_device =
                    static_cast<const int32_t *>(d_condition);
                stage_params.prior_draft_tokens_device =
                    static_cast<const int32_t *>(d_drafts);
                stage_params.prior_draft_count = depth;
                stage_params.generated_token_counts_device =
                    static_cast<const int32_t *>(d_counts);
                stage_params.penalty_policy_device = d_captured_policy;
                stage_params.draft_values_device =
                    static_cast<float *>(d_captured_values);
                stage_params.draft_tokens_device =
                    static_cast<int32_t *>(d_captured_tokens);
                stage_params.destination_slot = depth;
                stage_params.next_chain_condition_token_device =
                    static_cast<int32_t *>(d_captured_chain_token);
                stage_params.next_chain_position_id_device =
                    static_cast<int32_t *>(d_captured_chain_position);
                stage_params.chain_position_increment = 1;
                stage_params.argmax_partial_values_device =
                    static_cast<float *>(d_captured_partial_values);
                stage_params.argmax_partial_indices_device =
                    static_cast<int32_t *>(d_captured_partial_indices);
                stage_params.argmax_partial_capacity = scratch_capacity;
                MTPDraftTokenPublicationStage stage(stage_params);
                EXPECT_THROW(
                    stage.execute(execution_context.get()),
                    std::logic_error)
                    << "draft publication must fail hard on a null stream";
                stage.setGPUStream(stream);

                reset_inputs();
                if (apply_penalties)
                {
                    ASSERT_TRUE(backend_
                        ->enqueueConfigureMTPGreedyPenaltyPolicyDevice(
                            d_captured_policy,
                            presence_penalty,
                            frequency_penalty,
                            first_token_already_in_history,
                            device_id_,
                            stream));
                }
                enqueue_reference();
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                const bool stage_enqueued =
                    stage.execute(execution_context.get());
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(stage_enqueued);
                ASSERT_GE(
                    capture->nodeCount(),
                    apply_penalties ? 3U : 2U);
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                /*
                 * Replay against changed bytes at the same addresses.  This
                 * catches captured host values and stale branch-history reads.
                 */
                for (int token = 0; token < vocab_size; ++token)
                {
                    logits[static_cast<size_t>(token)] =
                        -9.0F +
                        static_cast<float>(
                            (token * 43 + depth * 61) % 1543) /
                            149.0F;
                    counts[static_cast<size_t>(token)] =
                        token % 503 == 0 ? 1 + token % 5 : 0;
                }
                condition_token = 101;
                if (first_token_already_in_history)
                    ++counts[static_cast<size_t>(condition_token)];
                for (int prior = 0; prior < depth; ++prior)
                    drafts[static_cast<size_t>(prior)] =
                        101 + prior % 7;
                ASSERT_TRUE(copyHostToDevice(
                    d_condition,
                    &condition_token,
                    sizeof(condition_token),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_drafts,
                    drafts.data(),
                    drafts.size() * sizeof(int32_t),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_counts,
                    counts.data(),
                    counts.size() * sizeof(int32_t),
                    device_id_,
                    stream));
                reset_inputs();
                enqueue_reference();
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });

            std::vector<float> reference_logits(logits.size());
            std::vector<float> captured_logits(logits.size());
            std::vector<float> reference_values(value_sentinel.size());
            std::vector<float> captured_values(value_sentinel.size());
            std::vector<int32_t> reference_tokens(token_sentinel.size());
            std::vector<int32_t> captured_tokens(token_sentinel.size());
            int32_t reference_chain_token = -1;
            int32_t captured_chain_token = -1;
            int32_t reference_chain_position = -1;
            int32_t captured_chain_position = -1;
            ASSERT_TRUE(copyDeviceToHost(
                reference_logits.data(),
                d_reference_logits,
                reference_logits.size() * sizeof(float),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                captured_logits.data(),
                d_captured_logits,
                captured_logits.size() * sizeof(float),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                reference_values.data(),
                d_reference_values,
                reference_values.size() * sizeof(float),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                captured_values.data(),
                d_captured_values,
                captured_values.size() * sizeof(float),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                reference_tokens.data(),
                d_reference_tokens,
                reference_tokens.size() * sizeof(int32_t),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                captured_tokens.data(),
                d_captured_tokens,
                captured_tokens.size() * sizeof(int32_t),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                &reference_chain_token,
                d_reference_chain_token,
                sizeof(reference_chain_token),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                &captured_chain_token,
                d_captured_chain_token,
                sizeof(captured_chain_token),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                &reference_chain_position,
                d_reference_chain_position,
                sizeof(reference_chain_position),
                device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                &captured_chain_position,
                d_captured_chain_position,
                sizeof(captured_chain_position),
                device_id_));

            EXPECT_EQ(
                std::memcmp(
                    reference_logits.data(),
                    captured_logits.data(),
                    reference_logits.size() * sizeof(float)),
                0)
                << "proposal logit bytes diverged at depth=" << depth
                << " penalties=" << apply_penalties;
            EXPECT_EQ(
                std::memcmp(
                    reference_values.data(),
                    captured_values.data(),
                    reference_values.size() * sizeof(float)),
                0)
                << "proposal max-value bytes diverged at depth=" << depth;
            EXPECT_EQ(
                std::memcmp(
                    reference_tokens.data(),
                    captured_tokens.data(),
                    reference_tokens.size() * sizeof(int32_t)),
                0)
                << "proposal token bytes diverged at depth=" << depth;
            EXPECT_EQ(captured_chain_token, reference_chain_token);
            EXPECT_EQ(captured_chain_token, captured_tokens[depth]);
            EXPECT_EQ(captured_chain_position, reference_chain_position);
            EXPECT_EQ(captured_chain_position, chain_position + 1);

            if (apply_penalties)
            {
                MTPGreedyPenaltyPolicy reference_policy{};
                MTPGreedyPenaltyPolicy captured_policy{};
                ASSERT_TRUE(copyDeviceToHost(
                    &reference_policy,
                    d_reference_policy,
                    sizeof(reference_policy),
                    device_id_));
                ASSERT_TRUE(copyDeviceToHost(
                    &captured_policy,
                    d_captured_policy,
                    sizeof(captured_policy),
                    device_id_));
                EXPECT_EQ(
                    std::memcmp(
                        &reference_policy,
                        &captured_policy,
                        sizeof(reference_policy)),
                    0)
                    << "proposal policy bytes diverged at depth=" << depth;
            }

            for (void *allocation : allocations)
                backend_->free(allocation, device_id_);
        };

        IWorkerGPUContext &ctx =
            GetParam() == "CUDA"
                ? GPUDeviceContextPool::instance().getNvidiaContext(device_id_)
                : GPUDeviceContextPool::instance().getAMDContext(device_id_);
        for (size_t case_index = 0;
             case_index < depth_cases.size();
             ++case_index)
        {
            run_case(
                ctx,
                depth_cases[case_index],
                /*apply_penalties=*/case_index % 2 != 0,
                /*first_token_already_in_history=*/case_index % 3 == 0);
            run_case(
                ctx,
                depth_cases[case_index],
                /*apply_penalties=*/false,
                /*first_token_already_in_history=*/false);
        }
    }

    /**
     * @brief Prove sidecar evidence is replay-live and transaction-zero owned.
     *
     * The model parent graph may execute several WHILE iterations before its
     * terminal observation.  This test captures the exact production
     * diagnostic primitive, mutates all resident inputs, and proves a later
     * transaction cannot overwrite the first record.  Re-admitting transaction
     * zero then proves the same executable reads current device bytes rather
     * than captured host values.
     */
    TEST_P(
        GPUSamplingTest,
        CapturedMTPDraftBoundaryDiagnosticRetainsOnlyTransactionZero)
    {
        using namespace sampling_math;
        constexpr int first_word_count = 4096;
        constexpr int second_word_count = 257;
        constexpr int first_boundary = static_cast<int>(
            MTPFirstTransactionDraftBoundary::TerminalHiddenInput);
        constexpr int second_boundary = static_cast<int>(
            MTPFirstTransactionDraftBoundary::Embedding);

        auto expected_hash = [](const std::vector<uint32_t> &words)
        {
            constexpr int kThreads = 256;
            std::array<uint64_t, kThreads> lane_hashes{};
            for (int lane = 0; lane < kThreads; ++lane)
            {
                uint64_t hash = 1469598103934665603ULL ^
                                static_cast<uint64_t>(lane + 1);
                for (size_t index = static_cast<size_t>(lane);
                     index < words.size();
                     index += kThreads)
                {
                    hash = append_diagnostic_u32_fnv1a(
                        hash,
                        static_cast<uint32_t>(index));
                    hash = append_diagnostic_u32_fnv1a(
                        hash,
                        words[index]);
                }
                lane_hashes[static_cast<size_t>(lane)] = hash;
            }

            uint64_t folded = 1469598103934665603ULL;
            for (uint64_t lane_hash : lane_hashes)
            {
                folded = append_diagnostic_u32_fnv1a(
                    folded,
                    static_cast<uint32_t>(lane_hash));
                folded = append_diagnostic_u32_fnv1a(
                    folded,
                    static_cast<uint32_t>(lane_hash >> 32U));
            }
            return folded;
        };

        std::vector<uint32_t> first_words(first_word_count);
        std::vector<uint32_t> second_words(second_word_count);
        for (size_t index = 0; index < first_words.size(); ++index)
            first_words[index] = static_cast<uint32_t>(index * 17U + 11U);
        for (size_t index = 0; index < second_words.size(); ++index)
            second_words[index] = static_cast<uint32_t>(index * 29U + 7U);

        std::array<int32_t, kDeviceGenerationControlCount> control{};
        control[kDeviceGenerationControlTransactionCount] = 0;
        int32_t condition_token = 271;
        int32_t position_id = 409;
        MTPFirstTransactionDiagnosticRecord initial_record{};

        void *d_first_words = backend_->allocate(
            first_words.size() * sizeof(uint32_t), device_id_);
        void *d_second_words = backend_->allocate(
            second_words.size() * sizeof(uint32_t), device_id_);
        void *d_control = backend_->allocate(sizeof(control), device_id_);
        void *d_condition = backend_->allocate(
            sizeof(condition_token), device_id_);
        void *d_position = backend_->allocate(
            sizeof(position_id), device_id_);
        void *d_record = backend_->allocate(
            sizeof(initial_record), device_id_);
        const std::array<void *, 6> allocations = {
            d_first_words,
            d_second_words,
            d_control,
            d_condition,
            d_position,
            d_record,
        };
        for (void *allocation : allocations)
            ASSERT_NE(allocation, nullptr);

        IWorkerGPUContext &ctx =
            GetParam() == "CUDA"
                ? GPUDeviceContextPool::instance().getNvidiaContext(device_id_)
                : GPUDeviceContextPool::instance().getAMDContext(device_id_);
        std::unique_ptr<IGPUGraphCapture> capture;
        ctx.submitAndWait([&]()
        {
            void *const stream = ctx.defaultStream();
            ASSERT_NE(stream, nullptr);
            ASSERT_TRUE(copyHostToDevice(
                d_first_words,
                first_words.data(),
                first_words.size() * sizeof(uint32_t),
                device_id_,
                stream));
            ASSERT_TRUE(copyHostToDevice(
                d_second_words,
                second_words.data(),
                second_words.size() * sizeof(uint32_t),
                device_id_,
                stream));
            ASSERT_TRUE(copyHostToDevice(
                d_control,
                control.data(),
                sizeof(control),
                device_id_,
                stream));
            ASSERT_TRUE(copyHostToDevice(
                d_condition,
                &condition_token,
                sizeof(condition_token),
                device_id_,
                stream));
            ASSERT_TRUE(copyHostToDevice(
                d_position,
                &position_id,
                sizeof(position_id),
                device_id_,
                stream));
            ASSERT_TRUE(copyHostToDevice(
                d_record,
                &initial_record,
                sizeof(initial_record),
                device_id_,
                stream));

            capture = ctx.createGraphCapture(stream);
            ASSERT_NE(capture, nullptr);
            ASSERT_TRUE(capture->beginCapture());
            const bool first_enqueued =
                backend_->enqueueRetainMTPFirstTransactionDraftBoundaryDevice(
                    d_first_words,
                    first_word_count,
                    first_boundary,
                    /*draft_slot=*/0,
                    d_condition,
                    d_position,
                    d_control,
                    static_cast<int>(control.size()),
                    d_record,
                    device_id_,
                    stream);
            const bool second_enqueued =
                backend_->enqueueRetainMTPFirstTransactionDraftBoundaryDevice(
                    d_second_words,
                    second_word_count,
                    second_boundary,
                    /*draft_slot=*/0,
                    d_condition,
                    d_position,
                    d_control,
                    static_cast<int>(control.size()),
                    d_record,
                    device_id_,
                    stream);
            ASSERT_TRUE(capture->endCapture());
            ASSERT_TRUE(first_enqueued);
            ASSERT_TRUE(second_enqueued);
            ASSERT_EQ(capture->nodeCount(), 2U);
            ASSERT_TRUE(capture->instantiate());
            ASSERT_TRUE(capture->launch());
            ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
        });

        MTPFirstTransactionDiagnosticRecord transaction_zero{};
        ASSERT_TRUE(copyDeviceToHost(
            &transaction_zero,
            d_record,
            sizeof(transaction_zero),
            device_id_));
        EXPECT_EQ(transaction_zero.draft_diagnostic_depth, 1);
        EXPECT_EQ(transaction_zero.draft_condition_tokens[0], condition_token);
        EXPECT_EQ(transaction_zero.draft_position_ids[0], position_id);
        EXPECT_EQ(
            transaction_zero.draft_boundary_word_counts[0][first_boundary],
            first_word_count);
        EXPECT_EQ(
            transaction_zero.draft_boundary_word_counts[0][second_boundary],
            second_word_count);
        EXPECT_EQ(
            transaction_zero.draft_boundary_hashes[0][first_boundary],
            expected_hash(first_words));
        EXPECT_EQ(
            transaction_zero.draft_boundary_hashes[0][second_boundary],
            expected_hash(second_words));

        for (size_t index = 0; index < first_words.size(); ++index)
            first_words[index] ^= static_cast<uint32_t>(index * 13U + 3U);
        for (size_t index = 0; index < second_words.size(); ++index)
            second_words[index] ^= static_cast<uint32_t>(index * 5U + 19U);
        condition_token = 999;
        position_id = 777;
        control[kDeviceGenerationControlTransactionCount] = 1;
        ctx.submitAndWait([&]()
        {
            void *const stream = ctx.defaultStream();
            ASSERT_TRUE(copyHostToDevice(
                d_first_words,
                first_words.data(),
                first_words.size() * sizeof(uint32_t),
                device_id_,
                stream));
            ASSERT_TRUE(copyHostToDevice(
                d_second_words,
                second_words.data(),
                second_words.size() * sizeof(uint32_t),
                device_id_,
                stream));
            ASSERT_TRUE(copyHostToDevice(
                d_control,
                control.data(),
                sizeof(control),
                device_id_,
                stream));
            ASSERT_TRUE(copyHostToDevice(
                d_condition,
                &condition_token,
                sizeof(condition_token),
                device_id_,
                stream));
            ASSERT_TRUE(copyHostToDevice(
                d_position,
                &position_id,
                sizeof(position_id),
                device_id_,
                stream));
            ASSERT_TRUE(capture->launch());
            ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
        });

        MTPFirstTransactionDiagnosticRecord retained{};
        ASSERT_TRUE(copyDeviceToHost(
            &retained,
            d_record,
            sizeof(retained),
            device_id_));
        EXPECT_EQ(
            std::memcmp(
                &retained,
                &transaction_zero,
                sizeof(retained)),
            0)
            << "transaction one overwrote transaction-zero sidecar evidence";

        control[kDeviceGenerationControlTransactionCount] = 0;
        ctx.submitAndWait([&]()
        {
            void *const stream = ctx.defaultStream();
            ASSERT_TRUE(copyHostToDevice(
                d_control,
                control.data(),
                sizeof(control),
                device_id_,
                stream));
            ASSERT_TRUE(capture->launch());
            ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
        });
        MTPFirstTransactionDiagnosticRecord replay_live{};
        ASSERT_TRUE(copyDeviceToHost(
            &replay_live,
            d_record,
            sizeof(replay_live),
            device_id_));
        EXPECT_EQ(replay_live.draft_condition_tokens[0], condition_token);
        EXPECT_EQ(replay_live.draft_position_ids[0], position_id);
        EXPECT_EQ(
            replay_live.draft_boundary_hashes[0][first_boundary],
            expected_hash(first_words));
        EXPECT_EQ(
            replay_live.draft_boundary_hashes[0][second_boundary],
            expected_hash(second_words));
        EXPECT_NE(
            std::memcmp(
                &replay_live,
                &transaction_zero,
                sizeof(replay_live)),
            0)
            << "captured diagnostics did not observe changed resident bytes";

        capture.reset();
        for (void *allocation : allocations)
            backend_->free(allocation, device_id_);
    }

    /**
     * @brief Proves every supported proposal depth consumes exact device history.
     *
     * A proposal row is scored after the first condition token and zero through
     * fifteen prior draft tokens.  The pending-first variant models a condition
     * already committed by the preceding transaction and proves it is not
     * counted twice.  Full-row byte equality is required because even a
     * non-winning logit may become observable after later filtering.
     */
    TEST_P(GPUSamplingTest,
           MTPBranchPenaltyProposalRowIsSerialDecodeByteExactForAllDepths)
    {
        constexpr int vocab_size = 248320;
        constexpr float presence_penalty = 0.37f;
        constexpr float frequency_penalty = 0.13f;
        constexpr int maximum_prior_drafts = 15;
        constexpr int first_condition_token = 19;

        auto run_case = [&](IWorkerGPUContext &ctx,
                            int prior_draft_count,
                            bool first_token_already_in_history)
        {
            std::vector<int> counts(static_cast<size_t>(vocab_size), 0);
            for (int token : {3, 7, 11, 19, 23})
                counts[static_cast<size_t>(token)] = 1 + (token % 3);
            if (first_token_already_in_history)
                ++counts[static_cast<size_t>(first_condition_token)];

            std::vector<int> prior_drafts(
                static_cast<size_t>(std::max(1, prior_draft_count)),
                0);
            for (int draft = 0; draft < prior_draft_count; ++draft)
            {
                prior_drafts[static_cast<size_t>(draft)] =
                    19 + (draft % 4);
            }

            std::vector<float> input(static_cast<size_t>(vocab_size));
            for (int token = 0; token < vocab_size; ++token)
            {
                input[static_cast<size_t>(token)] =
                    static_cast<float>((token % 97) - 48) / 16.0f;
            }
            std::vector<float> expected = input;
            std::map<int, int> active_counts;
            for (int token : {3, 7, 11, 19, 23})
                active_counts[token] = counts[static_cast<size_t>(token)];
            if (!first_token_already_in_history)
                ++active_counts[first_condition_token];
            for (int draft = 0; draft < prior_draft_count; ++draft)
                ++active_counts[prior_drafts[static_cast<size_t>(draft)]];
            for (const auto &[token, count] : active_counts)
            {
                if (count <= 0)
                    continue;
                float penalty = 0.0f;
                penalty += presence_penalty;
                penalty += frequency_penalty * static_cast<float>(count);
                expected[static_cast<size_t>(token)] -= penalty;
            }

            void *d_logits = backend_->allocate(
                input.size() * sizeof(float), device_id_);
            void *d_condition = backend_->allocate(sizeof(int), device_id_);
            void *d_drafts = backend_->allocate(
                prior_drafts.size() * sizeof(int), device_id_);
            void *d_counts = backend_->allocate(
                counts.size() * sizeof(int), device_id_);
            void *d_policy = backend_->allocate(
                sizeof(MTPGreedyPenaltyPolicy), device_id_);
            ASSERT_NE(d_logits, nullptr);
            ASSERT_NE(d_condition, nullptr);
            ASSERT_NE(d_drafts, nullptr);
            ASSERT_NE(d_counts, nullptr);
            ASSERT_NE(d_policy, nullptr);

            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_logits,
                    input.data(),
                    input.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_condition,
                    &first_condition_token,
                    sizeof(first_condition_token),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_drafts,
                    prior_drafts.data(),
                    prior_drafts.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_counts,
                    counts.data(),
                    counts.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(
                    backend_->enqueueConfigureMTPGreedyPenaltyPolicyDevice(
                        d_policy,
                        presence_penalty,
                        frequency_penalty,
                        first_token_already_in_history,
                        device_id_,
                        stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueApplyMTPBranchPenaltiesToF32RowDevice(
                        d_logits,
                        vocab_size,
                        d_condition,
                        d_drafts,
                        prior_draft_count,
                        d_counts,
                        d_policy,
                        device_id_,
                        stream));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });

            std::vector<float> actual(input.size());
            ASSERT_TRUE(copyDeviceToHost(
                actual.data(),
                d_logits,
                actual.size() * sizeof(float),
                device_id_));
            EXPECT_EQ(
                std::memcmp(
                    actual.data(),
                    expected.data(),
                    actual.size() * sizeof(float)),
                0)
                << "proposal penalty bytes diverged for prior_drafts="
                << prior_draft_count
                << " pending_first=" << first_token_already_in_history;

            backend_->free(d_logits, device_id_);
            backend_->free(d_condition, device_id_);
            backend_->free(d_drafts, device_id_);
            backend_->free(d_counts, device_id_);
            backend_->free(d_policy, device_id_);
        };

        for (int prior_draft_count = 0;
             prior_draft_count <= maximum_prior_drafts;
             ++prior_draft_count)
        {
            for (const bool pending_first : {false, true})
            {
                if (GetParam() == "CUDA")
                {
                    auto &ctx =
                        GPUDeviceContextPool::instance().getNvidiaContext(
                            device_id_);
                    run_case(ctx, prior_draft_count, pending_first);
                }
                else
                {
                    auto &ctx =
                        GPUDeviceContextPool::instance().getAMDContext(
                            device_id_);
                    run_case(ctx, prior_draft_count, pending_first);
                }
            }
        }
    }

    TEST_P(GPUSamplingTest, GreedySpeculativeSummaryQwen36VocabMatchesHostRows)
    {
        using namespace sampling_math;

        /*
         * Regression shape for Qwen3.6 MoE all-position MTP.  The compact
         * vLLM-style path leaves row argmax tokens on device and feeds them
         * directly into the greedy verifier summary.  Keep this equivalent to
         * the host-visible batched argmax path at the real Qwen3.6 vocab size,
         * including the final bonus-ready row.
         */
        constexpr int rows = 2;
        constexpr int cols = 248320;
        constexpr int compare_rows = 1;
        const int32_t draft_tokens[rows] = {198, 248045};
        const int expected_verifier_tokens[rows] = {248045, 248068};

        std::vector<float> logits(
            static_cast<size_t>(rows) * static_cast<size_t>(cols),
            -12.0f);
        logits[248045] = 20.0f;
        logits[74455] = 19.0f;
        logits[static_cast<size_t>(cols) + 248068] = 20.0f;
        logits[static_cast<size_t>(cols) + 74455] = 18.5f;

        int expected_tokens[kSpeculativeBatchMaxOutputTokens] = {};
        int expected_meta[kSpeculativeBatchMetaCount] = {};
        summarize_greedy_speculative_verify_batch(
            draft_tokens[0],
            expected_verifier_tokens,
            draft_tokens,
            compare_rows,
            /*stop_tokens=*/nullptr,
            /*stop_token_count=*/0,
            expected_tokens,
            kSpeculativeBatchMaxOutputTokens,
            expected_meta);
        ASSERT_EQ(expected_meta[kSpecBatchMetaOk], 1);
        ASSERT_EQ(expected_meta[kSpecBatchMetaReadyToken], 248068);

        void *d_logits = nullptr;
        void *d_argmax_values = nullptr;
        void *d_argmax_indices = nullptr;
        void *d_partial_values = nullptr;
        void *d_partial_indices = nullptr;
        void *d_draft_tokens = nullptr;
        void *d_output_tokens = nullptr;
        void *d_output_meta = nullptr;
        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_logits,
                d_argmax_values,
                d_argmax_indices,
                d_partial_values,
                d_partial_indices,
                d_draft_tokens,
                d_output_tokens,
                d_output_meta};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_argmax_values = backend_->allocate(rows * sizeof(float), device_id_);
        d_argmax_indices = backend_->allocate(rows * sizeof(int), device_id_);
        d_partial_values = backend_->allocate(1024 * sizeof(float), device_id_);
        d_partial_indices = backend_->allocate(1024 * sizeof(int), device_id_);
        d_draft_tokens = backend_->allocate(rows * sizeof(int32_t), device_id_);
        d_output_tokens =
            backend_->allocate(kSpeculativeBatchMaxOutputTokens * sizeof(int), device_id_);
        d_output_meta =
            backend_->allocate(kSpeculativeBatchMetaCount * sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_argmax_values, nullptr);
        ASSERT_NE(d_argmax_indices, nullptr);
        ASSERT_NE(d_partial_values, nullptr);
        ASSERT_NE(d_partial_indices, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_output_tokens, nullptr);
        ASSERT_NE(d_output_meta, nullptr);

        auto run_on_stream = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_logits,
                    logits.data(),
                    logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens,
                    draft_tokens,
                    sizeof(draft_tokens),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->enqueueArgmaxF32BatchedRowsDevice(
                    d_logits,
                    rows,
                    cols,
                    device_id_,
                    stream,
                    d_argmax_values,
                    d_argmax_indices,
                    d_partial_values,
                    d_partial_indices,
                    1024));
                ASSERT_TRUE(backend_->enqueueSummarizeGreedySpeculativeVerifyBatch(
                    d_argmax_indices,
                    d_draft_tokens,
                    compare_rows,
                    draft_tokens[0],
                    /*stop_tokens_host=*/nullptr,
                    /*stop_token_count=*/0,
                    device_id_,
                    stream,
                    sampling_math::kSpeculativeBatchMaxOutputTokens,
                    d_output_tokens,
                    d_output_meta));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_on_stream(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_on_stream(ctx);
        }

        int gpu_indices[rows] = {};
        int gpu_tokens[kSpeculativeBatchMaxOutputTokens] = {};
        int gpu_meta[kSpeculativeBatchMetaCount] = {};
        ASSERT_TRUE(copyDeviceToHost(
            gpu_indices,
            d_argmax_indices,
            sizeof(gpu_indices),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            gpu_tokens,
            d_output_tokens,
            sizeof(gpu_tokens),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            gpu_meta,
            d_output_meta,
            sizeof(gpu_meta),
            device_id_));

        EXPECT_EQ(gpu_indices[0], expected_verifier_tokens[0]);
        EXPECT_EQ(gpu_indices[1], expected_verifier_tokens[1]);
        for (int i = 0; i < kSpeculativeBatchMetaCount; ++i)
            EXPECT_EQ(gpu_meta[i], expected_meta[i]) << "meta index " << i;
        for (int i = 0; i < kSpeculativeBatchMaxOutputTokens; ++i)
            EXPECT_EQ(gpu_tokens[i], expected_tokens[i]) << "token index " << i;

        cleanup();
    }

    TEST_P(GPUSamplingTest, SpeculativePublicationMetadataDerivationCaptures)
    {
        using namespace sampling_math;

        constexpr int request_count = 4;
        constexpr int padded_state_rows_per_request = 4;
        constexpr int max_state_commit_rows = 3;
        constexpr int meta_stride = kSpeculativeBatchMetaCount;

        std::array<int, request_count * meta_stride> meta{};
        std::array<int, request_count * kSpeculativeBatchMaxOutputTokens>
            output_tokens{};
        std::array<int, request_count> base_cached_tokens = {
            100, 200, 300, 400};

        const int accept_rows[] = {11, 12};
        const int accept_flags[] = {1, 1};
        summarize_speculative_verify_batch(
            /*first_token=*/10,
            accept_rows,
            accept_flags,
            /*row_count=*/2,
            /*stop_tokens=*/nullptr,
            /*stop_token_count=*/0,
            /*bonus_ready_token=*/13,
            /*has_bonus_ready_token=*/1,
            output_tokens.data(),
            kSpeculativeBatchMaxOutputTokens,
            meta.data());

        const int reject_rows[] = {21, 22};
        const int reject_flags[] = {0, 1};
        summarize_speculative_verify_batch(
            /*first_token=*/20,
            reject_rows,
            reject_flags,
            /*row_count=*/2,
            /*stop_tokens=*/nullptr,
            /*stop_token_count=*/0,
            /*bonus_ready_token=*/23,
            /*has_bonus_ready_token=*/1,
            output_tokens.data() + kSpeculativeBatchMaxOutputTokens,
            kSpeculativeBatchMaxOutputTokens,
            meta.data() + meta_stride);

        meta[2 * meta_stride + kSpecBatchMetaOk] = 1;
        meta[2 * meta_stride + kSpecBatchMetaTargetVerifierStateCommitCount] =
            max_state_commit_rows + 1;
        meta[3 * meta_stride + kSpecBatchMetaOk] = 1;
        meta[3 * meta_stride + kSpecBatchMetaTargetVerifierStateCommitCount] =
            padded_state_rows_per_request + 1;

        void *d_meta = backend_->allocate(meta.size() * sizeof(int), device_id_);
        void *d_base = backend_->allocate(base_cached_tokens.size() * sizeof(int), device_id_);
        void *d_output_tokens = backend_->allocate(output_tokens.size() * sizeof(int), device_id_);
        void *d_restore_rows = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_target_cached_tokens = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_accepted_state_counts = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_ok = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_next_condition_tokens = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_next_verifier_condition_tokens =
            backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_all_drafts_accepted = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_stopped = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_shifted_target_cached_tokens =
            backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_shifted_accepted_state_counts =
            backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_shifted_ok =
            backend_->allocate(request_count * sizeof(int), device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_meta,
                d_base,
                d_output_tokens,
                d_restore_rows,
                d_target_cached_tokens,
                d_accepted_state_counts,
                d_ok,
                d_next_condition_tokens,
                d_next_verifier_condition_tokens,
                d_all_drafts_accepted,
                d_stopped,
                d_shifted_target_cached_tokens,
                d_shifted_accepted_state_counts,
                d_shifted_ok};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_meta, nullptr);
        ASSERT_NE(d_base, nullptr);
        ASSERT_NE(d_output_tokens, nullptr);
        ASSERT_NE(d_restore_rows, nullptr);
        ASSERT_NE(d_target_cached_tokens, nullptr);
        ASSERT_NE(d_accepted_state_counts, nullptr);
        ASSERT_NE(d_ok, nullptr);
        ASSERT_NE(d_next_condition_tokens, nullptr);
        ASSERT_NE(d_next_verifier_condition_tokens, nullptr);
        ASSERT_NE(d_all_drafts_accepted, nullptr);
        ASSERT_NE(d_stopped, nullptr);
        ASSERT_NE(d_shifted_target_cached_tokens, nullptr);
        ASSERT_NE(d_shifted_accepted_state_counts, nullptr);
        ASSERT_NE(d_shifted_ok, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_meta,
                    meta.data(),
                    meta.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_base,
                    base_cached_tokens.data(),
                    base_cached_tokens.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_output_tokens,
                    output_tokens.data(),
                    output_tokens.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueDeriveSpeculativePublicationMetadata(
                    d_meta,
                    meta_stride,
                    d_base,
                    request_count,
                    padded_state_rows_per_request,
                    max_state_commit_rows,
                    device_id_,
                    nullptr,
                    d_restore_rows,
                    d_target_cached_tokens,
                    d_accepted_state_counts,
                    d_ok,
                    d_next_condition_tokens,
                    d_output_tokens,
                    kSpeculativeBatchMaxOutputTokens,
                    d_all_drafts_accepted,
                    d_stopped,
                    d_next_verifier_condition_tokens))
                    << "publication metadata derivation must reject the legacy default/null stream";
                EXPECT_FALSE(
                    backend_
                        ->enqueueDeriveShiftedSpeculativePublicationMetadataFromPrimary(
                            d_base,
                            d_target_cached_tokens,
                            d_ok,
                            request_count,
                            /*mtp_depth=*/0,
                            device_id_,
                            nullptr,
                            d_shifted_target_cached_tokens,
                            d_shifted_accepted_state_counts,
                            d_shifted_ok))
                    << "shifted publication derivation must reject a null producer stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueDeriveSpeculativePublicationMetadata(
                    d_meta,
                    meta_stride,
                    d_base,
                    request_count,
                    padded_state_rows_per_request,
                    max_state_commit_rows,
                    device_id_,
                    stream,
                    d_restore_rows,
                    d_target_cached_tokens,
                    d_accepted_state_counts,
                    d_ok,
                    d_next_condition_tokens,
                    d_output_tokens,
                    kSpeculativeBatchMaxOutputTokens,
                    d_all_drafts_accepted,
                    d_stopped,
                    d_next_verifier_condition_tokens));
                ASSERT_TRUE(
                    backend_
                        ->enqueueDeriveShiftedSpeculativePublicationMetadataFromPrimary(
                            d_base,
                            d_target_cached_tokens,
                            d_ok,
                            request_count,
                            /*mtp_depth=*/0,
                            device_id_,
                            stream,
                            d_shifted_target_cached_tokens,
                            d_shifted_accepted_state_counts,
                            d_shifted_ok));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::array<int, request_count> restore_rows{};
        std::array<int, request_count> target_cached_tokens{};
        std::array<int, request_count> accepted_state_counts{};
        std::array<int, request_count> ok{};
        std::array<int, request_count> next_condition_tokens{};
        std::array<int, request_count> next_verifier_condition_tokens{};
        std::array<int, request_count> all_drafts_accepted{};
        std::array<int, request_count> stopped{};
        std::array<int, request_count> shifted_target_cached_tokens{};
        std::array<int, request_count> shifted_accepted_state_counts{};
        std::array<int, request_count> shifted_ok{};
        ASSERT_TRUE(copyDeviceToHost(
            restore_rows.data(), d_restore_rows,
            restore_rows.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            target_cached_tokens.data(), d_target_cached_tokens,
            target_cached_tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            accepted_state_counts.data(), d_accepted_state_counts,
            accepted_state_counts.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            ok.data(), d_ok,
            ok.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            next_condition_tokens.data(), d_next_condition_tokens,
            next_condition_tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            next_verifier_condition_tokens.data(),
            d_next_verifier_condition_tokens,
            next_verifier_condition_tokens.size() * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            all_drafts_accepted.data(), d_all_drafts_accepted,
            all_drafts_accepted.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            stopped.data(), d_stopped,
            stopped.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            shifted_target_cached_tokens.data(),
            d_shifted_target_cached_tokens,
            shifted_target_cached_tokens.size() * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            shifted_accepted_state_counts.data(),
            d_shifted_accepted_state_counts,
            shifted_accepted_state_counts.size() * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            shifted_ok.data(), d_shifted_ok,
            shifted_ok.size() * sizeof(int), device_id_));

        cleanup();

        EXPECT_EQ(ok[0], 1);
        EXPECT_EQ(accepted_state_counts[0], 3);
        EXPECT_EQ(restore_rows[0], 2);
        EXPECT_EQ(target_cached_tokens[0], 103);
        EXPECT_EQ(next_condition_tokens[0], 13);
        EXPECT_EQ(all_drafts_accepted[0], 1);
        EXPECT_EQ(stopped[0], 0);

        EXPECT_EQ(ok[1], 1);
        EXPECT_EQ(accepted_state_counts[1], 1);
        EXPECT_EQ(restore_rows[1], 4);
        EXPECT_EQ(target_cached_tokens[1], 201);
        EXPECT_EQ(next_condition_tokens[1], 21);
        EXPECT_EQ(all_drafts_accepted[1], 0);
        EXPECT_EQ(stopped[1], 0);

        EXPECT_EQ(ok[2], 1);
        EXPECT_EQ(accepted_state_counts[2], max_state_commit_rows);
        EXPECT_EQ(restore_rows[2], 10);
        EXPECT_EQ(target_cached_tokens[2], 303);
        EXPECT_EQ(next_condition_tokens[2], -1);
        EXPECT_EQ(all_drafts_accepted[2], 0);
        EXPECT_EQ(stopped[2], 0);

        EXPECT_EQ(ok[3], 0);
        EXPECT_EQ(accepted_state_counts[3], 0);
        EXPECT_EQ(restore_rows[3], -1);
        EXPECT_EQ(target_cached_tokens[3], 400);
        EXPECT_EQ(next_condition_tokens[3], -1);
        EXPECT_EQ(all_drafts_accepted[3], 0);
        EXPECT_EQ(stopped[3], 0);

        EXPECT_EQ(next_verifier_condition_tokens, next_condition_tokens)
            << "Every request row must publish the next verifier condition to "
               "the canonical target bank in the same captured kernel.";

        const std::array<int, request_count> expected_shifted_targets = {
            102, 200, 302, 0};
        const std::array<int, request_count> expected_shifted_accepted = {
            3, 1, 3, 0};
        const std::array<int, request_count> expected_shifted_ok = {
            1, 1, 1, 0};
        EXPECT_EQ(
            shifted_target_cached_tokens,
            expected_shifted_targets);
        EXPECT_EQ(
            shifted_accepted_state_counts,
            expected_shifted_accepted);
        EXPECT_EQ(shifted_ok, expected_shifted_ok);
    }

    // =========================================================================
    // TOP-K TESTS — mirrors Top-K Sampling from Test__Sampler.cpp
    // =========================================================================

    TEST_P(GPUSamplingTest, TopK_K1_IsArgmax)
    {
        // Top-k with k=1 should be equivalent to argmax
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = topKF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                    1, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok) << "topKF32 not supported on " << GetParam();

        EXPECT_EQ(out_index, 2) << "Top-1 should select index 2 (logit 3.0)";
        EXPECT_FLOAT_EQ(out_value, 3.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_K2_CorrectTokens)
    {
        // Top-2 of standard_logits: idx 2 (3.0), idx 1 (2.0) — descending order
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<float> values(2);
        std::vector<int> indices(2);
        bool ok = topKF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                    2, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Results should be in descending order
        EXPECT_EQ(indices[0], 2) << "Rank 0 should be index 2 (logit 3.0)";
        EXPECT_EQ(indices[1], 1) << "Rank 1 should be index 1 (logit 2.0)";
        EXPECT_FLOAT_EQ(values[0], 3.0f);
        EXPECT_FLOAT_EQ(values[1], 2.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_K3_CorrectRanking)
    {
        // Top-3 of standard_logits: idx 2 (3.0), idx 1 (2.0), idx 4 (1.5)
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<float> values(3);
        std::vector<int> indices(3);
        bool ok = topKF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                    3, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        EXPECT_EQ(indices[0], 2) << "Rank 0 should be index 2 (logit 3.0)";
        EXPECT_EQ(indices[1], 1) << "Rank 1 should be index 1 (logit 2.0)";
        EXPECT_EQ(indices[2], 4) << "Rank 2 should be index 4 (logit 1.5)";
        EXPECT_FLOAT_EQ(values[0], 3.0f);
        EXPECT_FLOAT_EQ(values[1], 2.0f);
        EXPECT_FLOAT_EQ(values[2], 1.5f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_KEqualsVocabSize)
    {
        // k equals vocab size — should return all elements sorted descending
        int n = static_cast<int>(standard_logits_.size());
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<float> values(n);
        std::vector<int> indices(n);
        bool ok = topKF32(d_ptr, n, n, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Should be sorted descending: 3.0, 2.0, 1.5, 1.0, 0.5
        EXPECT_FLOAT_EQ(values[0], 3.0f);
        EXPECT_EQ(indices[0], 2);
        EXPECT_FLOAT_EQ(values[n - 1], 0.5f);
        EXPECT_EQ(indices[n - 1], 3);

        // Verify descending order
        for (int i = 1; i < n; ++i)
        {
            EXPECT_GE(values[i - 1], values[i])
                << "Values should be in descending order at position " << i;
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_DescendingOrder)
    {
        // Verify results are always in descending value order
        std::vector<float> logits = {5.0f, 1.0f, 3.0f, 7.0f, 2.0f, 6.0f, 4.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 5;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = topKF32(d_ptr, static_cast<int>(logits.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Top-5 sorted descending: 7.0(3), 6.0(5), 5.0(0), 4.0(6), 3.0(2)
        EXPECT_FLOAT_EQ(values[0], 7.0f);
        EXPECT_FLOAT_EQ(values[1], 6.0f);
        EXPECT_FLOAT_EQ(values[2], 5.0f);
        EXPECT_FLOAT_EQ(values[3], 4.0f);
        EXPECT_FLOAT_EQ(values[4], 3.0f);

        EXPECT_EQ(indices[0], 3);
        EXPECT_EQ(indices[1], 5);
        EXPECT_EQ(indices[2], 0);
        EXPECT_EQ(indices[3], 6);
        EXPECT_EQ(indices[4], 2);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_PeakedLogits)
    {
        // Peaked distribution: top-3 should return the peak + 2 closest
        void *d_ptr = uploadLogits(peaked_logits_);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<float> values(3);
        std::vector<int> indices(3);
        bool ok = topKF32(d_ptr, static_cast<int>(peaked_logits_.size()),
                                    3, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Peak at idx 2 (10.0), then idx 1 and 4 (0.2 each), then idx 0 and 3 (0.1 each)
        EXPECT_EQ(indices[0], 2) << "Peak token should be rank 0";
        EXPECT_FLOAT_EQ(values[0], 10.0f);
        // Equal logits are part of the sampler contract: lower token id wins.
        EXPECT_EQ(indices[1], 1);
        EXPECT_EQ(indices[2], 4);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_NegativeLogits)
    {
        std::vector<float> negative = {-5.0f, -2.0f, -1.0f, -10.0f};
        void *d_ptr = uploadLogits(negative);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<float> values(3);
        std::vector<int> indices(3);
        bool ok = topKF32(d_ptr, static_cast<int>(negative.size()),
                                    3, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Descending: -1.0(2), -2.0(1), -5.0(0)
        EXPECT_EQ(indices[0], 2);
        EXPECT_EQ(indices[1], 1);
        EXPECT_EQ(indices[2], 0);
        EXPECT_FLOAT_EQ(values[0], -1.0f);
        EXPECT_FLOAT_EQ(values[1], -2.0f);
        EXPECT_FLOAT_EQ(values[2], -5.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_SingleElement)
    {
        std::vector<float> single = {7.0f};
        void *d_ptr = uploadLogits(single);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = topKF32(d_ptr, 1, 1, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 0);
        EXPECT_FLOAT_EQ(out_value, 7.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_UniformLogits)
    {
        // All same value: top-k should return the lowest token ids in order.
        void *d_ptr = uploadLogits(uniform_logits_);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 3;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = topKF32(d_ptr, static_cast<int>(uniform_logits_.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // All values should be 2.0 and ties should resolve by token id.
        for (int i = 0; i < k; ++i)
        {
            EXPECT_FLOAT_EQ(values[i], 2.0f) << "Position " << i;
            EXPECT_EQ(indices[i], i) << "Top-k ties must be deterministic";
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_AllZeros)
    {
        std::vector<float> zeros = {0.0f, 0.0f, 0.0f, 0.0f};
        void *d_ptr = uploadLogits(zeros);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 2;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = topKF32(d_ptr, static_cast<int>(zeros.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        for (int i = 0; i < k; ++i)
        {
            EXPECT_FLOAT_EQ(values[i], 0.0f);
            EXPECT_EQ(indices[i], i);
        }

        freeDevice(d_ptr);
    }

    // =========================================================================
    // TOP-K NUMERICAL STABILITY TESTS
    // =========================================================================

    TEST_P(GPUSamplingTest, TopK_VeryLargeLogits)
    {
        std::vector<float> large = {500.0f, 501.0f, 502.0f, 500.5f};
        void *d_ptr = uploadLogits(large);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 3;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = topKF32(d_ptr, static_cast<int>(large.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        EXPECT_EQ(indices[0], 2);
        EXPECT_FLOAT_EQ(values[0], 502.0f);

        // Should be descending
        for (int i = 1; i < k; ++i)
        {
            EXPECT_GE(values[i - 1], values[i]);
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_VerySmallLogits)
    {
        std::vector<float> small = {-500.0f, -501.0f, -499.0f, -500.5f};
        void *d_ptr = uploadLogits(small);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 2;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = topKF32(d_ptr, static_cast<int>(small.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Top-2: -499.0(2), -500.0(0)
        EXPECT_EQ(indices[0], 2);
        EXPECT_FLOAT_EQ(values[0], -499.0f);
        EXPECT_EQ(indices[1], 0);
        EXPECT_FLOAT_EQ(values[1], -500.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_MixedExtremeLogits)
    {
        std::vector<float> mixed = {-1000.0f, 1000.0f, -1000.0f, -1000.0f};
        void *d_ptr = uploadLogits(mixed);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 2;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = topKF32(d_ptr, static_cast<int>(mixed.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        EXPECT_EQ(indices[0], 1) << "Token 1 (1000.0) should be rank 0";
        EXPECT_FLOAT_EQ(values[0], 1000.0f);

        freeDevice(d_ptr);
    }

    // =========================================================================
    // TOP-K LARGE VOCABULARY TESTS
    // =========================================================================

    TEST_P(GPUSamplingTest, TopK_LargeVocab_50K)
    {
        // Large vocabulary with known top-3
        const int n = 50000;
        std::vector<float> logits(n, 0.0f);
        logits[100] = 5.0f;
        logits[200] = 4.0f;
        logits[300] = 3.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 3;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = topKF32(d_ptr, n, k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        EXPECT_EQ(indices[0], 100);
        EXPECT_EQ(indices[1], 200);
        EXPECT_EQ(indices[2], 300);
        EXPECT_FLOAT_EQ(values[0], 5.0f);
        EXPECT_FLOAT_EQ(values[1], 4.0f);
        EXPECT_FLOAT_EQ(values[2], 3.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_Qwen2VocabSize)
    {
        // Qwen2.5 vocab_size = 151936, realistic distribution with top-5
        const int n = 151936;
        std::vector<float> logits(n, 0.0f);
        logits[256] = 15.0f;
        logits[8159] = 14.0f;
        logits[100160] = 13.5f;
        logits[72363] = 13.0f;
        logits[105797] = 12.8f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 5;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = topKF32(d_ptr, n, k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Verify correct ranking
        EXPECT_EQ(indices[0], 256) << "Rank 0 should be token 256 (logit 15.0)";
        EXPECT_EQ(indices[1], 8159) << "Rank 1 should be token 8159 (logit 14.0)";
        EXPECT_EQ(indices[2], 100160) << "Rank 2 should be token 100160 (logit 13.5)";
        EXPECT_EQ(indices[3], 72363) << "Rank 3 should be token 72363 (logit 13.0)";
        EXPECT_EQ(indices[4], 105797) << "Rank 4 should be token 105797 (logit 12.8)";

        EXPECT_FLOAT_EQ(values[0], 15.0f);
        EXPECT_FLOAT_EQ(values[1], 14.0f);
        EXPECT_FLOAT_EQ(values[2], 13.5f);
        EXPECT_FLOAT_EQ(values[3], 13.0f);
        EXPECT_FLOAT_EQ(values[4], 12.8f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_Qwen2VocabSize_K40)
    {
        // Realistic scenario: k=40 (default) on full Qwen2 vocab
        const int n = 151936;
        const int k = 40;

        // Create a distribution with known top-40
        std::vector<float> logits(n, -10.0f);
        for (int i = 0; i < k; ++i)
        {
            logits[i * 1000] = 20.0f - static_cast<float>(i) * 0.5f;
        }
        // logits[0]=20, logits[1000]=19.5, logits[2000]=19.0, ..., logits[39000]=0.5

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = topKF32(d_ptr, n, k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Verify descending order
        for (int i = 1; i < k; ++i)
        {
            EXPECT_GE(values[i - 1], values[i])
                << "Values should be descending at position " << i;
        }

        // Verify top-1
        EXPECT_EQ(indices[0], 0) << "Rank 0 should be index 0 (logit 20.0)";
        EXPECT_FLOAT_EQ(values[0], 20.0f);

        // Verify all top-k indices are from our planted values
        for (int i = 0; i < k; ++i)
        {
            EXPECT_EQ(indices[i] % 1000, 0)
                << "Index " << indices[i] << " at rank " << i << " should be a multiple of 1000";
        }

        freeDevice(d_ptr);
    }

    // =========================================================================
    // TOP-K EDGE CASE TESTS
    // =========================================================================

    TEST_P(GPUSamplingTest, TopK_PeakAtLastElement)
    {
        // Peak at end of large array
        const int n = 10000;
        std::vector<float> logits(n, -1.0f);
        logits[n - 1] = 42.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = topKF32(d_ptr, n, 1, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, n - 1);
        EXPECT_FLOAT_EQ(out_value, 42.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_PeakAtFirstElement)
    {
        // Peak at start of large array
        const int n = 10000;
        std::vector<float> logits(n, -1.0f);
        logits[0] = 42.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = topKF32(d_ptr, n, 1, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 0);
        EXPECT_FLOAT_EQ(out_value, 42.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_DuplicateValues)
    {
        // Multiple elements with same value: lower token ids define the tie order.
        std::vector<float> dups = {5.0f, 5.0f, 5.0f, 3.0f, 3.0f, 1.0f};
        void *d_ptr = uploadLogits(dups);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 4;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = topKF32(d_ptr, static_cast<int>(dups.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Top-4 should be: three 5.0s, then one 3.0
        EXPECT_FLOAT_EQ(values[0], 5.0f);
        EXPECT_FLOAT_EQ(values[1], 5.0f);
        EXPECT_FLOAT_EQ(values[2], 5.0f);
        EXPECT_FLOAT_EQ(values[3], 3.0f);

        EXPECT_EQ(indices[0], 0);
        EXPECT_EQ(indices[1], 1);
        EXPECT_EQ(indices[2], 2);
        EXPECT_EQ(indices[3], 3);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_K128_LargeK)
    {
        // Test a large k value (128) — upper bound that fits GPU shared memory
        // Note: k=256 exceeds 48KB shared memory on some GPUs (32 threads × 256 × 8B = 64KB)
        const int n = 1000;
        const int k = 128;

        // Create descending logits so ranking is trivial
        std::vector<float> logits(n);
        for (int i = 0; i < n; ++i)
        {
            logits[i] = static_cast<float>(n - i);
        }
        // logits[0]=1000, logits[1]=999, ..., logits[999]=1

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = topKF32(d_ptr, n, k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Top element should be index 0 with value 1000
        EXPECT_EQ(indices[0], 0);
        EXPECT_FLOAT_EQ(values[0], 1000.0f);

        // Last top-k element should be index 127 with value 873
        EXPECT_EQ(indices[k - 1], k - 1);
        EXPECT_FLOAT_EQ(values[k - 1], static_cast<float>(n - k + 1));

        // Verify strict descending
        for (int i = 1; i < k; ++i)
        {
            EXPECT_GT(values[i - 1], values[i])
                << "Values should be strictly descending at position " << i;
        }

        freeDevice(d_ptr);
    }

    // =========================================================================
    // ARGMAX vs TOP-K CONSISTENCY TESTS
    // =========================================================================

    TEST_P(GPUSamplingTest, ArgmaxVsTopK1_Consistent)
    {
        // argmax and topK(k=1) should give identical results
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        float argmax_value = 0.0f;
        int argmax_index = -1;
        bool ok1 = argmaxF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                       device_id_, &argmax_value, &argmax_index);

        float topk_value = 0.0f;
        int topk_index = -1;
        bool ok2 = topKF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                     1, device_id_, &topk_value, &topk_index);

        if (ok1 && ok2)
        {
            EXPECT_EQ(argmax_index, topk_index) << "argmax and topK(1) index should match";
            EXPECT_FLOAT_EQ(argmax_value, topk_value) << "argmax and topK(1) value should match";
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, ArgmaxVsTopK1_LargeVocab)
    {
        // Consistency check on large vocab
        const int n = 151936;
        std::vector<float> logits(n, 0.0f);
        logits[75000] = 99.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float argmax_value = 0.0f, topk_value = 0.0f;
        int argmax_index = -1, topk_index = -1;

        bool ok1 = argmaxF32(d_ptr, n, device_id_, &argmax_value, &argmax_index);
        bool ok2 = topKF32(d_ptr, n, 1, device_id_, &topk_value, &topk_index);

        if (ok1 && ok2)
        {
            EXPECT_EQ(argmax_index, topk_index);
            EXPECT_FLOAT_EQ(argmax_value, topk_value);
        }

        freeDevice(d_ptr);
    }

    // =========================================================================
    // TOP-K INDEX CORRECTNESS — mirrors Token Ranking from Test__Sampler.cpp
    // =========================================================================

    TEST_P(GPUSamplingTest, TopK_IndicesAreGloballyCorrect)
    {
        // Verify that returned indices correctly map back to the original array
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        int n = static_cast<int>(standard_logits_.size());
        std::vector<float> values(n);
        std::vector<int> indices(n);
        bool ok = topKF32(d_ptr, n, n, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Each returned (index, value) pair should match the original logits
        for (int i = 0; i < n; ++i)
        {
            EXPECT_FLOAT_EQ(values[i], standard_logits_[indices[i]])
                << "Rank " << i << ": value " << values[i]
                << " should match logits[" << indices[i] << "]="
                << standard_logits_[indices[i]];
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_HigherLogitsHaveHigherRank)
    {
        // Verify ranking reflects logit magnitude
        std::vector<float> logits = {1.0f, 5.0f, 3.0f, 7.0f, 2.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 5;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = topKF32(d_ptr, static_cast<int>(logits.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Expected ranking: 7.0(3), 5.0(1), 3.0(2), 2.0(4), 1.0(0)
        EXPECT_EQ(indices[0], 3) << "Highest logit should be rank 0";
        EXPECT_EQ(indices[1], 1) << "Second highest should be rank 1";
        EXPECT_EQ(indices[2], 2) << "Third highest should be rank 2";
        EXPECT_EQ(indices[3], 4) << "Fourth highest should be rank 3";
        EXPECT_EQ(indices[4], 0) << "Lowest logit should be rank 4";

        freeDevice(d_ptr);
    }

    // =========================================================================
    // REAL-WORLD SCENARIO TESTS
    // =========================================================================

    TEST_P(GPUSamplingTest, RealWorld_Qwen2TP2_VocabLocalShard)
    {
        // In TP=2 mode, each device sees vocab_local = 76032 (half of 152064)
        // Validate correct behavior on the local shard size
        const int n = 76032;
        std::vector<float> logits(n, 0.0f);
        logits[50000] = 14.5f;
        logits[25000] = 14.0f;
        logits[75000] = 13.5f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        // Argmax on local shard
        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);
        EXPECT_EQ(out_index, 50000);
        EXPECT_FLOAT_EQ(out_value, 14.5f);

        // Top-k on local shard
        const int k = 3;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        ok = topKF32(d_ptr, n, k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        EXPECT_EQ(indices[0], 50000);
        EXPECT_EQ(indices[1], 25000);
        EXPECT_EQ(indices[2], 75000);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, RealWorld_DecodeLoopSimulation)
    {
        // Simulate repeated greedy sampling during decode loop
        // The same buffer is queried multiple times (mimicking decode iterations)
        const int n = 151936;
        std::vector<float> logits(n, 0.0f);
        logits[256] = 14.54f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        // Simulate 10 decode steps all querying the same logits
        for (int step = 0; step < 10; ++step)
        {
            float out_value = 0.0f;
            int out_index = -1;
            bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
            ASSERT_TRUE(ok) << "Decode step " << step;
            EXPECT_EQ(out_index, 256) << "Decode step " << step << " should produce consistent token";
            EXPECT_FLOAT_EQ(out_value, 14.54f);
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, RealWorld_RealisticLogitDistribution)
    {
        // Simulate a realistic logit distribution from an LLM
        // Most logits are small negative, a few tokens have positive values
        const int n = 151936;
        std::mt19937 rng(42);
        std::normal_distribution<float> noise(-5.0f, 2.0f);

        std::vector<float> logits(n);
        for (int i = 0; i < n; ++i)
        {
            logits[i] = noise(rng);
        }

        // Plant known top-5
        logits[256] = 15.0f;
        logits[8159] = 14.0f;
        logits[100160] = 13.5f;
        logits[72363] = 13.0f;
        logits[105797] = 12.8f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        // Argmax should find the planted peak
        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);
        EXPECT_EQ(out_index, 256) << "Argmax should find the planted peak in noisy data";
        EXPECT_FLOAT_EQ(out_value, 15.0f);

        // Top-5 should recover all planted peaks
        const int k = 5;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        ok = topKF32(d_ptr, n, k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        std::set<int> expected_top5 = {256, 8159, 100160, 72363, 105797};
        std::set<int> actual_top5(indices.begin(), indices.end());
        EXPECT_EQ(actual_top5, expected_top5) << "Top-5 should recover all planted peaks";

        freeDevice(d_ptr);
    }

    // =========================================================================
    // MULTIPLE ALLOCATIONS STRESS TEST
    // =========================================================================

    TEST_P(GPUSamplingTest, Stress_MultipleAllocFreeArgmax)
    {
        // Verify no leaks or corruption across multiple alloc/use/free cycles
        for (int iter = 0; iter < 20; ++iter)
        {
            const int n = 1000 + iter * 500;
            std::vector<float> logits(n, 0.0f);
            logits[iter % n] = 100.0f;

            void *d_ptr = uploadLogits(logits);
            ASSERT_NE(d_ptr, nullptr) << "Iteration " << iter;

            float out_value = 0.0f;
            int out_index = -1;
            bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
            ASSERT_TRUE(ok) << "Iteration " << iter;

            EXPECT_EQ(out_index, iter % n) << "Iteration " << iter;
            EXPECT_FLOAT_EQ(out_value, 100.0f) << "Iteration " << iter;

            freeDevice(d_ptr);
        }
    }

    TEST_P(GPUSamplingTest, Stress_MultipleAllocFreeTopK)
    {
        for (int iter = 0; iter < 20; ++iter)
        {
            const int n = 2000 + iter * 1000;
            const int k = std::min(10, n);
            std::vector<float> logits(n, -1.0f);

            // Plant k peaks at known positions
            for (int j = 0; j < k; ++j)
            {
                logits[j * (n / k)] = 50.0f - static_cast<float>(j);
            }

            void *d_ptr = uploadLogits(logits);
            ASSERT_NE(d_ptr, nullptr) << "Iteration " << iter;

            std::vector<float> values(k);
            std::vector<int> indices(k);
            bool ok = topKF32(d_ptr, n, k, device_id_, values.data(), indices.data());
            ASSERT_TRUE(ok) << "Iteration " << iter;

            // Top-1 should always be the highest planted peak
            EXPECT_EQ(indices[0], 0) << "Iteration " << iter;
            EXPECT_FLOAT_EQ(values[0], 50.0f) << "Iteration " << iter;

            freeDevice(d_ptr);
        }
    }

    // =========================================================================
    //  LOGIT PENALTY TESTS — GPU-side penalty application + sampling
    // =========================================================================

    // ------------------------------------------------------------------
    // Helper: download GPU logits back to host for verification
    // ------------------------------------------------------------------
    static std::vector<float> downloadLogits(IBackend *backend, void *d_ptr,
                                             int count, int device_id,
                                             void *stream)
    {
        std::vector<float> result(count);
        bool ok = backend->deviceToHost(result.data(), d_ptr,
                                        count * sizeof(float), device_id, stream);
        EXPECT_TRUE(ok) << "D2H transfer failed";
        return result;
    }

    static float samplingUniform01(uint64_t seed, uint64_t offset)
    {
        return sampling_math::uniform01(seed, offset);
    }

    /**
     * @brief Comparator used by CPU-side expected top-k/top-p helpers.
     *
     * The GPU kernels sort candidates by logit descending and token id ascending.
     * Keeping the test oracle on the same rule makes ties explicit and prevents
     * future sampler changes from reintroducing backend-dependent ordering.
     */
    static bool topKCandidateBefore(const std::pair<float, int> &a,
                                    const std::pair<float, int> &b)
    {
        if (a.first > b.first)
            return true;
        if (a.first < b.first)
            return false;
        return a.second < b.second;
    }

    static int expectedTopKTopPSample(const std::vector<float> &logits,
                                      int top_k,
                                      float top_p,
                                      float temperature,
                                      uint64_t seed,
                                      uint64_t offset)
    {
        std::vector<std::pair<float, int>> candidates;
        candidates.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i)
            candidates.emplace_back(logits[i], static_cast<int>(i));

        top_k = std::min<int>(top_k, static_cast<int>(candidates.size()));
        std::partial_sort(candidates.begin(),
                          candidates.begin() + top_k,
                          candidates.end(),
                          topKCandidateBefore);
        candidates.resize(static_cast<size_t>(top_k));

        std::vector<float> sorted_logits(static_cast<size_t>(top_k));
        std::vector<int> sorted_ids(static_cast<size_t>(top_k));
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            sorted_logits[i] = candidates[i].first;
            sorted_ids[i] = candidates[i].second;
        }
        std::vector<float> scratch(static_cast<size_t>(top_k), 0.0f);
        return sampling_math::sample_topk_topp_from_sorted_with_threshold(
            sorted_logits.data(),
            sorted_ids.data(),
            top_k,
            top_p,
            temperature,
            samplingUniform01(seed, offset),
            scratch.data());
    }

    struct ExpectedDistributionEntry
    {
        int token_id = -1;
        float probability = 0.0f;
    };

    static std::vector<ExpectedDistributionEntry> expectedTopKTopPDistribution(
        const std::vector<float> &logits,
        int top_k,
        float top_p,
        float temperature)
    {
        std::vector<std::pair<float, int>> candidates;
        candidates.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i)
            candidates.emplace_back(logits[i], static_cast<int>(i));

        top_k = std::min<int>(top_k, static_cast<int>(candidates.size()));
        std::partial_sort(candidates.begin(),
                          candidates.begin() + top_k,
                          candidates.end(),
                          topKCandidateBefore);
        candidates.resize(static_cast<size_t>(top_k));

        std::vector<float> sorted_logits(static_cast<size_t>(top_k));
        std::vector<int> sorted_ids(static_cast<size_t>(top_k));
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            sorted_logits[i] = candidates[i].first;
            sorted_ids[i] = candidates[i].second;
        }

        std::vector<int> out_ids(static_cast<size_t>(top_k), -1);
        std::vector<float> out_probs(static_cast<size_t>(top_k), 0.0f);
        std::vector<float> scratch(static_cast<size_t>(top_k), 0.0f);
        sampling_math::build_topk_topp_distribution_from_sorted(
            sorted_logits.data(),
            sorted_ids.data(),
            top_k,
            top_p,
            temperature,
            out_ids.data(),
            out_probs.data(),
            scratch.data());

        std::vector<ExpectedDistributionEntry> distribution(static_cast<size_t>(top_k));
        for (int i = 0; i < top_k; ++i)
        {
            distribution[static_cast<size_t>(i)].token_id = out_ids[static_cast<size_t>(i)];
            distribution[static_cast<size_t>(i)].probability = out_probs[static_cast<size_t>(i)];
        }
        return distribution;
    }

    /**
     * @brief Build the full-logit row equivalent to compact top-k/top-p sampling.
     *
     * Active nucleus tokens keep raw logits divided by temperature. Every other
     * token is non-finite, which is the processed-logit contract consumed by the
     * vLLM-style verifier and sampler.
     */
    static std::vector<float> expectedTopKTopPProcessedLogits(
        const std::vector<float> &logits,
        int top_k,
        float top_p,
        float temperature)
    {
        std::vector<float> processed(
            logits.size(),
            -std::numeric_limits<float>::infinity());
        const auto distribution =
            expectedTopKTopPDistribution(logits, top_k, top_p, temperature);
        const float temp = temperature > 0.0f ? temperature : 1.0f;
        for (const auto &entry : distribution)
        {
            if (entry.token_id >= 0 && entry.probability > 0.0f)
                processed[static_cast<size_t>(entry.token_id)] =
                    logits[static_cast<size_t>(entry.token_id)] / temp;
        }
        return processed;
    }

    static std::vector<float> expectedTemperatureOnlyProbabilities(
        const std::vector<float> &logits,
        float temperature)
    {
        const float safe_temperature =
            (std::isfinite(temperature) && temperature > 0.0f) ? temperature : 1.0f;
        float max_logit = -std::numeric_limits<float>::infinity();
        for (float logit : logits)
        {
            if (std::isfinite(logit))
                max_logit = std::max(max_logit, logit / safe_temperature);
        }

        std::vector<float> probabilities(logits.size(), 0.0f);
        if (!std::isfinite(max_logit))
            return probabilities;

        double exp_sum = 0.0;
        for (float logit : logits)
        {
            if (std::isfinite(logit))
                exp_sum += std::exp(static_cast<double>(logit / safe_temperature - max_logit));
        }
        if (!(exp_sum > 0.0))
            return probabilities;

        for (size_t token = 0; token < logits.size(); ++token)
        {
            if (std::isfinite(logits[token]))
            {
                probabilities[token] = static_cast<float>(
                    std::exp(static_cast<double>(logits[token] / safe_temperature - max_logit)) /
                    exp_sum);
            }
        }
        return probabilities;
    }

    static int expectedTemperatureOnlySampleToken(
        const std::vector<float> &probabilities,
        float threshold)
    {
        const float target_mass =
            sampling_math::clamp_unit_threshold(threshold);
        float prefix = 0.0f;
        int selected = 0;
        float best_probability = -1.0f;
        for (size_t token = 0; token < probabilities.size(); ++token)
        {
            const float probability = probabilities[token];
            if (probability > best_probability)
            {
                best_probability = probability;
                selected = static_cast<int>(token);
            }
            if (!(probability > 0.0f))
                continue;
            prefix += probability;
            if (target_mass <= prefix)
                return static_cast<int>(token);
        }
        return selected;
    }

    static float distributionProbability(
        const std::vector<ExpectedDistributionEntry> &distribution,
        int token_id)
    {
        std::vector<int> token_ids(distribution.size(), -1);
        std::vector<float> probs(distribution.size(), 0.0f);
        for (size_t i = 0; i < distribution.size(); ++i)
        {
            token_ids[i] = distribution[i].token_id;
            probs[i] = distribution[i].probability;
        }
        return sampling_math::distribution_probability(
            token_ids.data(),
            probs.data(),
            static_cast<int>(distribution.size()),
            token_id);
    }

    struct ExpectedSpeculativeVerify
    {
        int token_id = -1;
        int accepted = 0;
        float accept_probability = 0.0f;
        float accept_threshold = 0.0f;
    };

    static ExpectedSpeculativeVerify expectedSpeculativeVerifyDistributionWithThresholds(
        const std::vector<ExpectedDistributionEntry> &target,
        const std::vector<ExpectedDistributionEntry> &draft,
        int draft_token,
        float accept_threshold,
        float residual_threshold);

    static ExpectedSpeculativeVerify expectedSpeculativeVerifyOneHotDraftWithThresholds(
        const std::vector<ExpectedDistributionEntry> &target,
        int draft_token,
        float accept_threshold,
        float residual_threshold);

    static ExpectedSpeculativeVerify expectedSpeculativeVerifyOneHotDraftVLLMWithThresholds(
        const std::vector<ExpectedDistributionEntry> &target,
        int draft_token,
        float accept_threshold,
        uint64_t inverse_sample_seed,
        int logical_position,
        int vocab_size);

    static ExpectedSpeculativeVerify expectedSpeculativeVerifyDistribution(
        const std::vector<ExpectedDistributionEntry> &target,
        const std::vector<ExpectedDistributionEntry> &draft,
        int draft_token,
        uint64_t accept_seed,
        uint64_t accept_offset,
        uint64_t residual_seed,
        uint64_t residual_offset)
    {
        return expectedSpeculativeVerifyDistributionWithThresholds(
            target,
            draft,
            draft_token,
            samplingUniform01(accept_seed, accept_offset),
            samplingUniform01(residual_seed, residual_offset));
    }

    static int expectedSampleDistributionWithThreshold(
        const std::vector<ExpectedDistributionEntry> &distribution,
        float threshold)
    {
        std::vector<int> token_ids(distribution.size(), -1);
        std::vector<float> probs(distribution.size(), 0.0f);
        for (size_t i = 0; i < distribution.size(); ++i)
        {
            token_ids[i] = distribution[i].token_id;
            probs[i] = distribution[i].probability;
        }
        return sampling_math::sample_distribution_with_threshold(
            token_ids.data(),
            probs.data(),
            static_cast<int>(distribution.size()),
            threshold);
    }

    static ExpectedSpeculativeVerify expectedSpeculativeVerifyDistributionWithThresholds(
        const std::vector<ExpectedDistributionEntry> &target,
        const std::vector<ExpectedDistributionEntry> &draft,
        int draft_token,
        float accept_threshold,
        float residual_threshold)
    {
        ExpectedSpeculativeVerify result;
        std::vector<int> target_ids(target.size(), -1);
        std::vector<float> target_probs(target.size(), 0.0f);
        std::vector<int> draft_ids(draft.size(), -1);
        std::vector<float> draft_probs(draft.size(), 0.0f);
        for (size_t i = 0; i < target.size(); ++i)
        {
            target_ids[i] = target[i].token_id;
            target_probs[i] = target[i].probability;
        }
        for (size_t i = 0; i < draft.size(); ++i)
        {
            draft_ids[i] = draft[i].token_id;
            draft_probs[i] = draft[i].probability;
        }

        sampling_math::speculative_verify_with_thresholds(
            target_ids.data(),
            target_probs.data(),
            draft_ids.data(),
            draft_probs.data(),
            static_cast<int>(target.size()),
            draft_token,
            accept_threshold,
            residual_threshold,
            &result.token_id,
            &result.accepted,
            &result.accept_probability,
            &result.accept_threshold);
        return result;
    }

    static ExpectedSpeculativeVerify expectedSpeculativeVerifyOneHotDraftWithThresholds(
        const std::vector<ExpectedDistributionEntry> &target,
        int draft_token,
        float accept_threshold,
        float residual_threshold)
    {
        ExpectedSpeculativeVerify result;
        std::vector<int> target_ids(target.size(), -1);
        std::vector<float> target_probs(target.size(), 0.0f);
        for (size_t i = 0; i < target.size(); ++i)
        {
            target_ids[i] = target[i].token_id;
            target_probs[i] = target[i].probability;
        }

        sampling_math::speculative_verify_with_thresholds_one_hot_draft(
            target_ids.data(),
            target_probs.data(),
            static_cast<int>(target.size()),
            draft_token,
            accept_threshold,
            residual_threshold,
            &result.token_id,
            &result.accepted,
            &result.accept_probability,
            &result.accept_threshold);
        return result;
    }

    static ExpectedSpeculativeVerify expectedSpeculativeVerifyOneHotDraftVLLMWithThresholds(
        const std::vector<ExpectedDistributionEntry> &target,
        int draft_token,
        float accept_threshold,
        uint64_t inverse_sample_seed,
        int logical_position,
        int vocab_size)
    {
        ExpectedSpeculativeVerify result;
        std::vector<int> target_ids(target.size(), -1);
        std::vector<float> target_probs(target.size(), 0.0f);
        for (size_t i = 0; i < target.size(); ++i)
        {
            target_ids[i] = target[i].token_id;
            target_probs[i] = target[i].probability;
        }

        sampling_math::speculative_verify_with_thresholds_one_hot_draft_vllm_recovered(
            target_ids.data(),
            target_probs.data(),
            static_cast<int>(target.size()),
            vocab_size,
            draft_token,
            accept_threshold,
            inverse_sample_seed,
            logical_position,
            &result.token_id,
            &result.accepted,
            &result.accept_probability,
            &result.accept_threshold);
        return result;
    }

    TEST_P(GPUSamplingTest, Penalty_SingleToken_Subtracted)
    {
        // Apply a penalty to token 2 and verify logit is reduced
        std::vector<float> logits = {1.0f, 2.0f, 5.0f, 0.5f, 1.5f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<int> token_ids = {2};
        std::vector<float> penalties = {3.0f};

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            1, static_cast<int>(logits.size()), device_id_, stream_);
        ASSERT_TRUE(ok) << "applyLogitPenaltiesF32 not supported on " << GetParam();

        auto result = downloadLogits(backend_, d_ptr,
                                     static_cast<int>(logits.size()), device_id_,
                                     stream_);

        // Token 2: 5.0 - 3.0 = 2.0
        EXPECT_FLOAT_EQ(result[0], 1.0f) << "Unpenalized tokens should be unchanged";
        EXPECT_FLOAT_EQ(result[1], 2.0f) << "Unpenalized tokens should be unchanged";
        EXPECT_FLOAT_EQ(result[2], 2.0f) << "Token 2 should be penalized: 5.0 - 3.0 = 2.0";
        EXPECT_FLOAT_EQ(result[3], 0.5f) << "Unpenalized tokens should be unchanged";
        EXPECT_FLOAT_EQ(result[4], 1.5f) << "Unpenalized tokens should be unchanged";

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_MultipleTokens_AllSubtracted)
    {
        std::vector<float> logits = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<int> token_ids = {0, 2, 4};
        std::vector<float> penalties = {1.0f, 5.0f, 10.0f};

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            3, static_cast<int>(logits.size()), device_id_, stream_);
        ASSERT_TRUE(ok);

        auto result = downloadLogits(backend_, d_ptr,
                                     static_cast<int>(logits.size()), device_id_,
                                     stream_);

        EXPECT_FLOAT_EQ(result[0], 9.0f);  // 10 - 1
        EXPECT_FLOAT_EQ(result[1], 20.0f); // unchanged
        EXPECT_FLOAT_EQ(result[2], 25.0f); // 30 - 5
        EXPECT_FLOAT_EQ(result[3], 40.0f); // unchanged
        EXPECT_FLOAT_EQ(result[4], 40.0f); // 50 - 10

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_ZeroPenalties_NoOp)
    {
        // Empty penalty list — backends return false (no-op, nothing to do)
        // The caller (OrchestrationRunner) skips the call when the map is empty,
        // so this documents the backend contract rather than a usage pattern.
        std::vector<float> logits = {1.0f, 2.0f, 3.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, nullptr, nullptr, 0,
            static_cast<int>(logits.size()), device_id_, stream_);
        // Backends early-return false for num_penalties <= 0
        EXPECT_FALSE(ok) << "Zero penalties → backend returns false (no-op)";

        // Logits should still be unchanged
        auto result = downloadLogits(backend_, d_ptr,
                                     static_cast<int>(logits.size()), device_id_,
                                     stream_);
        EXPECT_FLOAT_EQ(result[0], 1.0f);
        EXPECT_FLOAT_EQ(result[1], 2.0f);
        EXPECT_FLOAT_EQ(result[2], 3.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_OutOfBoundsTokenId_Ignored)
    {
        // Token IDs outside [0, vocab_size) should be silently ignored
        std::vector<float> logits = {5.0f, 5.0f, 5.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<int> token_ids = {-1, 999, 1}; // -1 and 999 are OOB for vocab=3
        std::vector<float> penalties = {100.0f, 100.0f, 2.0f};

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            3, static_cast<int>(logits.size()), device_id_, stream_);
        ASSERT_TRUE(ok);

        auto result = downloadLogits(backend_, d_ptr,
                                     static_cast<int>(logits.size()), device_id_,
                                     stream_);

        EXPECT_FLOAT_EQ(result[0], 5.0f) << "OOB tokens should not corrupt logits";
        EXPECT_FLOAT_EQ(result[1], 3.0f) << "Valid token should be penalized: 5.0 - 2.0";
        EXPECT_FLOAT_EQ(result[2], 5.0f) << "OOB tokens should not corrupt logits";

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_NegativePenalty_BoostsLogit)
    {
        // Negative penalty = boost (used for negative presence penalty)
        std::vector<float> logits = {0.0f, 0.0f, 0.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<int> token_ids = {1};
        std::vector<float> penalties = {-5.0f}; // Boost by 5

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            1, static_cast<int>(logits.size()), device_id_, stream_);
        ASSERT_TRUE(ok);

        auto result = downloadLogits(backend_, d_ptr,
                                     static_cast<int>(logits.size()), device_id_,
                                     stream_);

        EXPECT_FLOAT_EQ(result[0], 0.0f);
        EXPECT_FLOAT_EQ(result[1], 5.0f) << "Negative penalty should boost: 0.0 - (-5.0) = 5.0";
        EXPECT_FLOAT_EQ(result[2], 0.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_ThenArgmax_ChangesSelection)
    {
        // End-to-end: penalty shifts argmax from token 0 to token 1
        std::vector<float> logits = {10.0f, 9.5f, 1.0f, 1.0f, 1.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        // Verify initial argmax is token 0
        float val = 0;
        int idx = -1;
        argmaxF32(d_ptr, static_cast<int>(logits.size()),
                            device_id_, &val, &idx);
        EXPECT_EQ(idx, 0) << "Before penalty, argmax should be token 0";

        // Penalize token 0 enough to make token 1 win
        std::vector<int> token_ids = {0};
        std::vector<float> penalties = {2.0f}; // 10.0 - 2.0 = 8.0 < 9.5

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            1, static_cast<int>(logits.size()), device_id_, stream_);
        ASSERT_TRUE(ok);

        // After penalty, argmax should shift to token 1
        argmaxF32(d_ptr, static_cast<int>(logits.size()),
                            device_id_, &val, &idx, stream_);
        EXPECT_EQ(idx, 1) << "After penalty, argmax should shift to token 1 (9.5 > 8.0)";
        EXPECT_FLOAT_EQ(val, 9.5f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_LargeVocab_SparseApplication)
    {
        // Real-world: Qwen2 vocab (151936) with sparse penalties
        const int vocab_size = 151936;
        std::vector<float> logits(vocab_size, 0.0f);
        logits[0] = 10.0f;
        logits[42] = 9.0f;
        logits[1000] = 8.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        // Penalize only 3 tokens out of 151K
        std::vector<int> token_ids = {0, 42, 1000};
        std::vector<float> penalties = {5.0f, 1.0f, 0.5f};

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            3, vocab_size, device_id_, stream_);
        ASSERT_TRUE(ok);

        // Verify via argmax — token 42 should now win (9.0 - 1.0 = 8.0 > 10.0 - 5.0 = 5.0)
        // token 1000: 8.0 - 0.5 = 7.5
        float val = 0;
        int idx = -1;
        argmaxF32(d_ptr, vocab_size, device_id_, &val, &idx, stream_);
        EXPECT_EQ(idx, 42) << "After penalties, token 42 (8.0) should beat token 0 (5.0)";

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_ManyPenalties_AllApplied)
    {
        // Stress test: apply 256 penalties
        const int vocab_size = 1000;
        std::vector<float> logits(vocab_size, 1.0f);
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        const int num_penalties = 256;
        std::vector<int> token_ids(num_penalties);
        std::vector<float> penalties(num_penalties);
        for (int i = 0; i < num_penalties; ++i)
        {
            token_ids[i] = i;
            penalties[i] = 0.5f;
        }

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            num_penalties, vocab_size, device_id_, stream_);
        ASSERT_TRUE(ok);

        auto result = downloadLogits(
            backend_, d_ptr, vocab_size, device_id_, stream_);

        // First 256 tokens should be 0.5, rest should be 1.0
        for (int i = 0; i < num_penalties; ++i)
        {
            EXPECT_FLOAT_EQ(result[i], 0.5f)
                << "Token " << i << " should be penalized: 1.0 - 0.5 = 0.5";
        }
        for (int i = num_penalties; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(result[i], 1.0f)
                << "Token " << i << " should be unchanged";
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_DRYStyle_ExponentialPenalty)
    {
        // Simulate DRY-style exponential penalty: multiple tokens with varying
        // penalty magnitudes that reflect repeat_len differences
        const int vocab_size = 10;
        std::vector<float> logits = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f,
                                     10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        // DRY penalties: multiplier=1.0, base=1.75
        // repeat_len=3, allowed=1 → 1.75^2 = 3.0625
        // repeat_len=5, allowed=1 → 1.75^4 = 9.3789
        float dry_3 = std::pow(1.75f, 2.0f);
        float dry_5 = std::pow(1.75f, 4.0f);

        std::vector<int> token_ids = {2, 7};
        std::vector<float> penalties = {dry_3, dry_5};

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            2, vocab_size, device_id_, stream_);
        ASSERT_TRUE(ok);

        auto result = downloadLogits(
            backend_, d_ptr, vocab_size, device_id_, stream_);

        EXPECT_NEAR(result[2], 10.0f - dry_3, 0.001f)
            << "Token 2 should have DRY penalty for repeat_len=3";
        EXPECT_NEAR(result[7], 10.0f - dry_5, 0.001f)
            << "Token 7 should have larger DRY penalty for repeat_len=5";

        // Unpenalized tokens should be unchanged
        EXPECT_FLOAT_EQ(result[0], 10.0f);
        EXPECT_FLOAT_EQ(result[5], 10.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_CombinedWithTopK_ChangesRanking)
    {
        // End-to-end: penalty → topK should reflect penalized logits
        std::vector<float> logits = {10.0f, 9.0f, 8.0f, 7.0f, 6.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        // Penalize top-2 tokens heavily
        std::vector<int> token_ids = {0, 1};
        std::vector<float> penalties = {8.0f, 7.0f}; // 10→2, 9→2

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            2, static_cast<int>(logits.size()), device_id_, stream_);
        ASSERT_TRUE(ok);

        // topK=3: should now be [2(8.0), 3(7.0), 4(6.0)] instead of [0,1,2]
        std::vector<float> values(3);
        std::vector<int> indices(3);
        ok = topKF32(d_ptr, static_cast<int>(logits.size()),
                               3, device_id_, values.data(), indices.data(),
                               stream_);
        ASSERT_TRUE(ok);

        EXPECT_EQ(indices[0], 2) << "After penalty, token 2 (8.0) should be top-1";
        EXPECT_FLOAT_EQ(values[0], 8.0f);
        EXPECT_EQ(indices[1], 3) << "Token 3 (7.0) should be top-2";
        EXPECT_EQ(indices[2], 4) << "Token 4 (6.0) should be top-3";

        freeDevice(d_ptr);
    }

    // =========================================================================
    //  DRY PENALTY CPU↔GPU PARITY TESTS
    //
    //  These tests compute DRY penalties via the CPU Sampler, then apply them
    //  to identical logit arrays on both CPU and GPU, verifying bit-exact
    //  parity. This validates the full DRY pipeline: CPU computes the sparse
    //  penalty map → GPU applies it in-place → results match CPU application.
    // =========================================================================

    // Helper: apply CPU penalties in-place (same as Sampler::apply_penalties)
    static void applyCpuPenalties(std::vector<float> &logits,
                                  const std::vector<LogitPenalty> &penalties)
    {
        for (const auto &entry : penalties)
            logits[entry.token_id] -= entry.penalty;
    }

    // Helper: apply GPU penalties, download result
    static std::vector<float> applyGpuPenalties(
        IBackend *backend, int device_id,
        const std::vector<float> &logits,
        const std::vector<LogitPenalty> &penalties,
        void *stream)
    {
        size_t bytes = logits.size() * sizeof(float);
        void *d_ptr = backend->allocate(bytes, device_id);
        EXPECT_NE(d_ptr, nullptr);
        bool ok = backend->hostToDevice(
            d_ptr, logits.data(), bytes, device_id, stream);
        EXPECT_TRUE(ok);

        if (!penalties.empty())
        {
            // Convert AoS → SoA
            std::vector<int> token_ids(penalties.size());
            std::vector<float> penalty_vals(penalties.size());
            for (size_t i = 0; i < penalties.size(); ++i)
            {
                token_ids[i] = penalties[i].token_id;
                penalty_vals[i] = penalties[i].penalty;
            }

            ok = backend->applyLogitPenaltiesF32(
                d_ptr, token_ids.data(), penalty_vals.data(),
                static_cast<int>(penalties.size()),
                static_cast<int>(logits.size()), device_id, stream);
            EXPECT_TRUE(ok);
        }

        auto result = downloadLogits(backend, d_ptr,
                                     static_cast<int>(logits.size()), device_id,
                                     stream);
        backend->free(d_ptr, device_id);
        return result;
    }

    TEST_P(GPUSamplingTest, LogitPenaltyDeviceInputsAreGraphCapturable)
    {
        const std::vector<float> logits = {1.0f, 8.0f, 4.0f, 3.0f, 6.0f};
        const std::vector<int> token_ids = {1, 4};
        const std::vector<float> penalty_vals = {7.0f, 2.5f};
        std::vector<float> expected = logits;
        expected[1] -= penalty_vals[0];
        expected[4] -= penalty_vals[1];

        void *d_logits = nullptr;
        void *d_token_ids = nullptr;
        void *d_penalties = nullptr;

        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_token_ids)
                backend_->free(d_token_ids, device_id_);
            if (d_penalties)
                backend_->free(d_penalties, device_id_);
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_token_ids = backend_->allocate(token_ids.size() * sizeof(int), device_id_);
        d_penalties = backend_->allocate(penalty_vals.size() * sizeof(float), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_token_ids, nullptr);
        ASSERT_NE(d_penalties, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_logits, logits.data(), logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_token_ids, token_ids.data(), token_ids.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_penalties, penalty_vals.data(), penalty_vals.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueLogitPenaltiesF32Device(
                    d_logits,
                    d_token_ids,
                    d_penalties,
                    static_cast<int>(token_ids.size()),
                    static_cast<int>(logits.size()),
                    device_id_,
                    stream));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        auto result = downloadLogits(
            backend_, d_logits, static_cast<int>(logits.size()), device_id_,
            stream_);
        cleanup();

        ASSERT_EQ(result.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_FLOAT_EQ(result[i], expected[i])
                << "Graph-captured penalty mismatch at token " << i;
        }
    }

    TEST_P(GPUSamplingTest, TopKTopPSampleDeviceOutputIsGraphCapturable)
    {
        const std::vector<float> logits = {0.1f, 4.5f, 3.8f, 0.0f,
                                           2.2f, 5.0f, -1.0f, 3.2f};
        constexpr int top_k = 4;
        constexpr float top_p = 0.85f;
        constexpr float temperature = 0.6f;
        constexpr uint64_t seed = 1234;
        constexpr uint64_t offset = 7;
        const int expected = expectedTopKTopPSample(
            logits, top_k, top_p, temperature, seed, offset);

        void *d_logits = nullptr;
        void *d_token = nullptr;

        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_token)
                backend_->free(d_token, device_id_);
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_token = backend_->allocate(sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_token, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_logits, logits.data(), logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                EXPECT_FALSE(backend_->enqueueSampleTopKTopPF32Device(
                    d_logits,
                    static_cast<int>(logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    seed,
                    offset,
                    device_id_,
                    nullptr,
                    d_token))
                    << "graph-capturable sampler must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueSampleTopKTopPF32Device(
                    d_logits,
                    static_cast<int>(logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    seed,
                    offset,
                    device_id_,
                    stream,
                    d_token));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        int actual = -1;
        ASSERT_TRUE(copyDeviceToHost(&actual, d_token, sizeof(int), device_id_));
        cleanup();

        EXPECT_EQ(actual, expected)
            << "Graph-captured top-k/top-p sampler selected the wrong token";
    }

    TEST_P(GPUSamplingTest, TopKTopPDistributionMatchesCPUSampler)
    {
        const std::vector<float> logits = {0.1f, 4.5f, 3.8f, 0.0f, 2.2f,
                                           5.0f, -1.0f, 3.2f, 4.1f, 1.3f};
        constexpr int top_k = 6;
        constexpr float top_p = 0.78f;
        constexpr float temperature = 0.7f;

        Sampler cpu_sampler(123);
        SamplingParams params;
        params.temperature = temperature;
        params.top_k = top_k;
        params.top_p = top_p;
        const auto cpu_distribution =
            cpu_sampler.compute_distribution(logits.data(), logits.size(), params);

        void *d_logits = nullptr;
        void *d_token_ids = nullptr;
        void *d_probs = nullptr;

        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_token_ids)
                backend_->free(d_token_ids, device_id_);
            if (d_probs)
                backend_->free(d_probs, device_id_);
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_token_ids = backend_->allocate(top_k * sizeof(int), device_id_);
        d_probs = backend_->allocate(top_k * sizeof(float), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_token_ids, nullptr);
        ASSERT_NE(d_probs, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_logits, logits.data(), logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_logits,
                    static_cast<int>(logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_token_ids,
                    d_probs));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<int> gpu_ids(top_k, -1);
        std::vector<float> gpu_probs(top_k, 0.0f);
        ASSERT_TRUE(copyDeviceToHost(gpu_ids.data(), d_token_ids, top_k * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(gpu_probs.data(), d_probs, top_k * sizeof(float), device_id_));
        cleanup();

        for (int i = 0; i < top_k; ++i)
        {
            if (i < static_cast<int>(cpu_distribution.size()))
            {
                EXPECT_EQ(gpu_ids[static_cast<size_t>(i)],
                          cpu_distribution[static_cast<size_t>(i)].token_id)
                    << "CPU/GPU compact distribution token mismatch at slot " << i;
                EXPECT_NEAR(gpu_probs[static_cast<size_t>(i)],
                            cpu_distribution[static_cast<size_t>(i)].probability,
                            1e-5f)
                    << "CPU/GPU compact distribution probability mismatch at slot " << i;
            }
            else
            {
                EXPECT_EQ(gpu_ids[static_cast<size_t>(i)], -1)
                    << "GPU should mark inactive top-p slots with token -1";
                EXPECT_FLOAT_EQ(gpu_probs[static_cast<size_t>(i)], 0.0f)
                    << "GPU should zero inactive top-p probability slots";
            }
        }
    }

    TEST_P(GPUSamplingTest, BatchedTopKTopPDistributionMatchesSerialCPUOracle)
    {
        constexpr int rows = 3;
        constexpr int vocab_size = 4096;
        constexpr int row_stride = vocab_size + 7;
        constexpr int top_k = 40;
        constexpr int out_stride = 64;
        constexpr float top_p = 0.93f;
        constexpr float temperature = 0.72f;

        std::vector<float> logits(static_cast<size_t>(rows * row_stride), -17.0f);
        for (int row = 0; row < rows; ++row)
        {
            for (int i = 0; i < vocab_size; ++i)
            {
                logits[static_cast<size_t>(row * row_stride + i)] =
                    -12.0f - 0.00011f * static_cast<float>((i * 53 + row * 97) % 997);
            }
            for (int rank = 0; rank < top_k; ++rank)
            {
                const int token = (row * 911 + rank * 73 + 17) % vocab_size;
                logits[static_cast<size_t>(row * row_stride + token)] =
                    6.0f - 0.047f * static_cast<float>(rank) +
                    0.013f * static_cast<float>(row);
            }
        }

        std::vector<std::vector<ExpectedDistributionEntry>> expected;
        expected.reserve(rows);
        for (int row = 0; row < rows; ++row)
        {
            std::vector<float> row_logits(
                logits.begin() + static_cast<ptrdiff_t>(row * row_stride),
                logits.begin() + static_cast<ptrdiff_t>(row * row_stride + vocab_size));
            expected.push_back(
                expectedTopKTopPDistribution(row_logits, top_k, top_p, temperature));
        }

        void *d_logits = nullptr;
        void *d_token_ids = nullptr;
        void *d_probs = nullptr;
        void *d_scratch_values = nullptr;
        void *d_scratch_indices = nullptr;
        constexpr int scratch_capacity = rows * 128 * top_k;

        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_token_ids)
                backend_->free(d_token_ids, device_id_);
            if (d_probs)
                backend_->free(d_probs, device_id_);
            if (d_scratch_values)
                backend_->free(d_scratch_values, device_id_);
            if (d_scratch_indices)
                backend_->free(d_scratch_indices, device_id_);
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_token_ids = backend_->allocate(rows * out_stride * sizeof(int), device_id_);
        d_probs = backend_->allocate(rows * out_stride * sizeof(float), device_id_);
        d_scratch_values = backend_->allocate(scratch_capacity * sizeof(float), device_id_);
        d_scratch_indices = backend_->allocate(scratch_capacity * sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_token_ids, nullptr);
        ASSERT_NE(d_probs, nullptr);
        ASSERT_NE(d_scratch_values, nullptr);
        ASSERT_NE(d_scratch_indices, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_logits, logits.data(), logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionsF32Device(
                    d_logits,
                    rows,
                    vocab_size,
                    row_stride,
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_token_ids,
                    out_stride,
                    d_probs,
                    d_scratch_values,
                    d_scratch_indices,
                    scratch_capacity));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<int> gpu_ids(static_cast<size_t>(rows * out_stride), -1);
        std::vector<float> gpu_probs(static_cast<size_t>(rows * out_stride), 0.0f);
        ASSERT_TRUE(copyDeviceToHost(
            gpu_ids.data(), d_token_ids, gpu_ids.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            gpu_probs.data(), d_probs, gpu_probs.size() * sizeof(float), device_id_));
        cleanup();

        for (int row = 0; row < rows; ++row)
        {
            for (int i = 0; i < top_k; ++i)
            {
                const size_t idx = static_cast<size_t>(row * out_stride + i);
                EXPECT_EQ(gpu_ids[idx], expected[static_cast<size_t>(row)][static_cast<size_t>(i)].token_id)
                    << "Batched compact distribution token mismatch at row " << row
                    << ", slot " << i;
                EXPECT_NEAR(gpu_probs[idx],
                            expected[static_cast<size_t>(row)][static_cast<size_t>(i)].probability,
                            1e-5f)
                    << "Batched compact distribution probability mismatch at row " << row
                    << ", slot " << i;
            }
        }
    }

    /**
     * @brief Prove the grouped compact-distribution builder is byte-identical
     *        to repeated production serial-row builds for every MTP depth.
     *
     * Grouped stochastic verification does not sample the verifier logits
     * directly. It first converts all verifier rows into compact Top-K/Top-P
     * distributions with enqueueBuildTopKTopPDistributionsF32Device(), whereas
     * serial decode converts one row with
     * enqueueBuildTopKTopPDistributionF32Device(). Byte-identical logits are
     * therefore not a complete batch-invariance proof unless these two
     * production builders also emit identical token ids and FP32
     * probabilities.
     *
     * The earlier batched-distribution regression compared against a relaxed
     * CPU oracle at one three-row geometry. This test instead compares the two
     * GPU production paths directly for every runtime M=1..16, using the Qwen
     * 3.6 vocabulary and production Top-K 40 policy. Tied frontier pairs also
     * prove that cooperative work partitioning retains the canonical lower-id
     * tie break. All launches are captured into one graph and reuse persistent
     * scratch; only the completed evidence is copied to the host.
     */
    TEST_P(
        GPUSamplingTest,
        BatchedTopKTopPDistributionsAreSerialRowByteExactForEveryMTPDepth)
    {
        constexpr int max_rows = 16;
        constexpr int vocab_size = 248320;
        constexpr int top_k = 40;
        constexpr float top_p = 0.9f;
        constexpr float temperature = 0.7f;
        constexpr int partial_block_capacity = 128;
        constexpr int total_swept_rows =
            max_rows * (max_rows + 1) / 2;

        /*
         * Every row has a distinct Top-K frontier made of equal-valued pairs.
         * The broad low-logit background exercises the complete Qwen-sized
         * reduction, while row-dependent hot-token positions prevent an
         * indexing or stride defect from passing because adjacent rows match.
         */
        std::vector<float> logits(
            static_cast<size_t>(max_rows) * vocab_size);
        for (int row = 0; row < max_rows; ++row)
        {
            float *row_logits =
                logits.data() + static_cast<size_t>(row) * vocab_size;
            for (int token = 0; token < vocab_size; ++token)
            {
                row_logits[token] =
                    -18.0f -
                    0.00037f *
                        static_cast<float>(
                            (token * 37 + row * 101) % 997);
            }
            for (int rank = 0; rank < top_k; ++rank)
            {
                const int token =
                    (row * 15401 + rank * 7919 + 321) % vocab_size;
                row_logits[token] =
                    6.0f -
                    0.071f * static_cast<float>(rank / 2) +
                    0.003f * static_cast<float>(row);
            }
        }

        void *d_logits = backend_->allocate(
            logits.size() * sizeof(float),
            device_id_);
        void *d_batched_ids = backend_->allocate(
            total_swept_rows * top_k * sizeof(int),
            device_id_);
        void *d_batched_probs = backend_->allocate(
            total_swept_rows * top_k * sizeof(float),
            device_id_);
        void *d_serial_ids = backend_->allocate(
            total_swept_rows * top_k * sizeof(int),
            device_id_);
        void *d_serial_probs = backend_->allocate(
            total_swept_rows * top_k * sizeof(float),
            device_id_);
        const int scratch_capacity =
            max_rows * partial_block_capacity * top_k;
        void *d_scratch_values = backend_->allocate(
            scratch_capacity * sizeof(float),
            device_id_);
        void *d_scratch_indices = backend_->allocate(
            scratch_capacity * sizeof(int),
            device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_logits,
                d_batched_ids,
                d_batched_probs,
                d_serial_ids,
                d_serial_probs,
                d_scratch_values,
                d_scratch_indices};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_batched_ids, nullptr);
        ASSERT_NE(d_batched_probs, nullptr);
        ASSERT_NE(d_serial_ids, nullptr);
        ASSERT_NE(d_serial_probs, nullptr);
        ASSERT_NE(d_scratch_values, nullptr);
        ASSERT_NE(d_scratch_indices, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_logits,
                    logits.data(),
                    logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());

                int output_row_base = 0;
                for (int row_count = 1;
                     row_count <= max_rows;
                     ++row_count)
                {
                    const size_t output_offset =
                        static_cast<size_t>(output_row_base) * top_k;
                    ASSERT_TRUE(
                        backend_
                            ->enqueueBuildTopKTopPDistributionsF32Device(
                                d_logits,
                                row_count,
                                vocab_size,
                                vocab_size,
                                top_k,
                                top_p,
                                temperature,
                                device_id_,
                                stream,
                                static_cast<int *>(d_batched_ids) +
                                    output_offset,
                                top_k,
                                static_cast<float *>(d_batched_probs) +
                                    output_offset,
                                d_scratch_values,
                                d_scratch_indices,
                                scratch_capacity))
                        << "grouped row_count=" << row_count;

                    /*
                     * Reuse the same scratch only after the grouped kernels
                     * already enqueued on this stream. Stream order makes each
                     * serial build a true repeated-row oracle without adding a
                     * synchronization edge or allocating per-row workspace.
                     */
                    for (int row = 0; row < row_count; ++row)
                    {
                        const size_t row_output_offset =
                            static_cast<size_t>(output_row_base + row) *
                            top_k;
                        ASSERT_TRUE(
                            backend_
                                ->enqueueBuildTopKTopPDistributionF32Device(
                                    static_cast<const float *>(d_logits) +
                                        static_cast<size_t>(row) *
                                            vocab_size,
                                    vocab_size,
                                    top_k,
                                    top_p,
                                    temperature,
                                    device_id_,
                                    stream,
                                    static_cast<int *>(d_serial_ids) +
                                        row_output_offset,
                                    static_cast<float *>(d_serial_probs) +
                                        row_output_offset,
                                    d_scratch_values,
                                    d_scratch_indices,
                                    scratch_capacity))
                            << "serial row_count=" << row_count
                            << " row=" << row;
                    }
                    output_row_base += row_count;
                }

                ASSERT_EQ(output_row_base, total_swept_rows);
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(
                    backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            run_capture(
                GPUDeviceContextPool::instance().getNvidiaContext(
                    device_id_));
        }
        else
        {
            run_capture(
                GPUDeviceContextPool::instance().getAMDContext(
                    device_id_));
        }

        std::vector<int> batched_ids(
            static_cast<size_t>(total_swept_rows) * top_k);
        std::vector<int> serial_ids(batched_ids.size());
        std::vector<float> batched_probs(batched_ids.size());
        std::vector<float> serial_probs(batched_ids.size());
        ASSERT_TRUE(copyDeviceToHost(
            batched_ids.data(),
            d_batched_ids,
            batched_ids.size() * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            serial_ids.data(),
            d_serial_ids,
            serial_ids.size() * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            batched_probs.data(),
            d_batched_probs,
            batched_probs.size() * sizeof(float),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            serial_probs.data(),
            d_serial_probs,
            serial_probs.size() * sizeof(float),
            device_id_));
        cleanup();

        EXPECT_EQ(
            std::memcmp(
                batched_ids.data(),
                serial_ids.data(),
                batched_ids.size() * sizeof(int)),
            0)
            << "grouped Top-K token ids differ from repeated serial rows";
        EXPECT_EQ(
            std::memcmp(
                batched_probs.data(),
                serial_probs.data(),
                batched_probs.size() * sizeof(float)),
            0)
            << "grouped Top-K/Top-P probabilities differ from repeated "
               "serial rows";
    }

    TEST_P(GPUSamplingTest, TopKTopP_Qwen36VocabTopK40_GraphCapturedDistributionAndSampleMatchCPU)
    {
        constexpr int vocab_size = 248320;
        constexpr int top_k = 40;
        constexpr float top_p = 0.95f;
        constexpr float temperature = 0.6f;
        constexpr uint64_t seed = 424242;
        constexpr uint64_t offset = 17;
        const float threshold = samplingUniform01(seed, offset);

        std::vector<float> logits(static_cast<size_t>(vocab_size));
        for (int i = 0; i < vocab_size; ++i)
            logits[static_cast<size_t>(i)] = -18.0f - 0.00037f * static_cast<float>((i * 37) % 997);

        const int hot_tokens[top_k] = {
            151936, 240001, 17, 248319, 98013,
            2048, 77777, 123456, 190000, 4096,
            222222, 31415, 65536, 101010, 88000,
            54321, 199999, 1, 135791, 246810,
            271, 13962, 96304, 3710, 5839,
            5077, 1414, 248068, 248069, 27775,
            2144, 3766, 16545, 2972, 51121,
            22527, 6157, 5757, 159034, 1503};
        for (int rank = 0; rank < top_k; ++rank)
        {
            logits[static_cast<size_t>(hot_tokens[rank])] =
                7.25f - 0.083f * static_cast<float>(rank);
        }

        const auto expected_distribution =
            expectedTopKTopPDistribution(logits, top_k, top_p, temperature);
        const int expected_sample =
            expectedSampleDistributionWithThreshold(expected_distribution, threshold);
        float expected_sample_probability = 0.0f;
        for (const auto &entry : expected_distribution)
        {
            if (entry.token_id == expected_sample)
            {
                expected_sample_probability = entry.probability;
                break;
            }
        }
        const int expected_direct_sample =
            expectedTopKTopPSample(logits, top_k, top_p, temperature, seed, offset);
        ASSERT_EQ(expected_direct_sample, expected_sample)
            << "direct CPU sample and compact-distribution CPU sample should agree";

        void *d_logits = nullptr;
        void *d_token_ids = nullptr;
        void *d_probs = nullptr;
        void *d_sample_token = nullptr;
        void *d_sample_probability = nullptr;
        void *d_direct_token = nullptr;

        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_token_ids)
                backend_->free(d_token_ids, device_id_);
            if (d_probs)
                backend_->free(d_probs, device_id_);
            if (d_sample_token)
                backend_->free(d_sample_token, device_id_);
            if (d_sample_probability)
                backend_->free(d_sample_probability, device_id_);
            if (d_direct_token)
                backend_->free(d_direct_token, device_id_);
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_token_ids = backend_->allocate(top_k * sizeof(int), device_id_);
        d_probs = backend_->allocate(top_k * sizeof(float), device_id_);
        d_sample_token = backend_->allocate(sizeof(int), device_id_);
        d_sample_probability = backend_->allocate(sizeof(float), device_id_);
        d_direct_token = backend_->allocate(sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_token_ids, nullptr);
        ASSERT_NE(d_probs, nullptr);
        ASSERT_NE(d_sample_token, nullptr);
        ASSERT_NE(d_sample_probability, nullptr);
        ASSERT_NE(d_direct_token, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_logits, logits.data(), logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_logits,
                    static_cast<int>(logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_token_ids,
                    d_probs));
                ASSERT_TRUE(backend_->enqueueSampleDistributionF32Device(
                    d_token_ids,
                    d_probs,
                    top_k,
                    threshold,
                    device_id_,
                    stream,
                    d_sample_token,
                    d_sample_probability));
                ASSERT_TRUE(backend_->enqueueSampleTopKTopPF32Device(
                    d_logits,
                    static_cast<int>(logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    seed,
                    offset,
                    device_id_,
                    stream,
                    d_direct_token));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<int> gpu_ids(top_k, -1);
        std::vector<float> gpu_probs(top_k, 0.0f);
        int sample_token = -1;
        float sample_probability = 0.0f;
        int direct_token = -1;
        ASSERT_TRUE(copyDeviceToHost(gpu_ids.data(), d_token_ids, top_k * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(gpu_probs.data(), d_probs, top_k * sizeof(float), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&sample_token, d_sample_token, sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&sample_probability, d_sample_probability, sizeof(float), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&direct_token, d_direct_token, sizeof(int), device_id_));
        cleanup();

        for (int i = 0; i < top_k; ++i)
        {
            EXPECT_EQ(gpu_ids[static_cast<size_t>(i)],
                      expected_distribution[static_cast<size_t>(i)].token_id)
                << "Qwen3.6 CPU/GPU compact distribution token mismatch at slot " << i;
            EXPECT_NEAR(gpu_probs[static_cast<size_t>(i)],
                        expected_distribution[static_cast<size_t>(i)].probability,
                        1e-5f)
                << "Qwen3.6 CPU/GPU compact distribution probability mismatch at slot " << i;
        }
        EXPECT_EQ(sample_token, expected_sample)
            << "Qwen3.6 compact distribution sample mismatch";
        EXPECT_NEAR(sample_probability, expected_sample_probability, 1e-6f)
            << "Qwen3.6 compact distribution sampled probability mismatch";
        EXPECT_EQ(direct_token, expected_direct_sample)
            << "Qwen3.6 direct top-k/top-p sample mismatch";
    }

    TEST_P(GPUSamplingTest, TopKTopP_Qwen36TopK20RepeatedGraphReplayIsStable)
    {
        /*
         * ROCm stochastic MTP repeatability is very sensitive to the first
         * token sampled from prefill logits. This test mirrors that production
         * lane: Qwen3.6 vocab size, Qwen chat-like sampling params, a compact
         * distribution build, then repeated graph replays on one explicit
         * stream. Any drift here means the sampler kernel itself is not a safe
         * building block for graph-captured MTP.
         */
        constexpr int vocab_size = 248320;
        constexpr int top_k = 20;
        constexpr float top_p = 0.95f;
        constexpr float temperature = 0.6f;
        constexpr uint64_t seed = 123;
        constexpr uint64_t offset = 4;
        constexpr int replay_count = 24;
        const float threshold = samplingUniform01(seed, offset);

        struct HotToken
        {
            int token_id;
            float logit;
        };

        const std::vector<HotToken> hot_tokens = {
            {33075, 9.0000f}, {25174, 8.9950f}, {888, 8.25f},
            {279, 8.05f},    {15217, 7.91f},   {5388, 7.80f},
            {13, 7.65f},     {198, 7.62f},     {271, 7.60f},
            {471, 7.30f},    {262, 7.18f},     {256, 7.16f},
            {2972, 7.02f},   {2425, 6.91f},    {2824, 6.80f},
            {64700, 6.69f},  {357, 6.58f},     {15352, 6.47f},
            {11, 6.36f},     {1575, 6.25f},
        };

        std::vector<float> logits(static_cast<size_t>(vocab_size), -18.0f);
        for (int i = 0; i < vocab_size; ++i)
        {
            logits[static_cast<size_t>(i)] -=
                0.00023f * static_cast<float>((i * 47) % 997);
        }
        for (const HotToken &hot : hot_tokens)
            logits[static_cast<size_t>(hot.token_id)] = hot.logit;

        const auto expected_distribution =
            expectedTopKTopPDistribution(logits, top_k, top_p, temperature);
        const int expected_sample =
            expectedSampleDistributionWithThreshold(expected_distribution, threshold);
        ASSERT_GE(expected_sample, 0);

        void *d_logits = nullptr;
        void *d_token_ids = nullptr;
        void *d_probs = nullptr;
        void *d_sample_token = nullptr;

        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_token_ids)
                backend_->free(d_token_ids, device_id_);
            if (d_probs)
                backend_->free(d_probs, device_id_);
            if (d_sample_token)
                backend_->free(d_sample_token, device_id_);
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_token_ids = backend_->allocate(top_k * sizeof(int), device_id_);
        d_probs = backend_->allocate(top_k * sizeof(float), device_id_);
        d_sample_token = backend_->allocate(sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_token_ids, nullptr);
        ASSERT_NE(d_probs, nullptr);
        ASSERT_NE(d_sample_token, nullptr);

        auto run_replays = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_logits,
                    logits.data(),
                    logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_logits,
                    vocab_size,
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_token_ids,
                    d_probs));
                ASSERT_TRUE(backend_->enqueueSampleDistributionF32Device(
                    d_token_ids,
                    d_probs,
                    top_k,
                    threshold,
                    device_id_,
                    stream,
                    d_sample_token));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                for (int replay = 0; replay < replay_count; ++replay)
                {
                    ASSERT_TRUE(capture->launch()) << "replay=" << replay;
                    ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_))
                        << "replay=" << replay;

                    int sample_token = -1;
                    ASSERT_TRUE(copyDeviceToHost(
                        &sample_token, d_sample_token, sizeof(int), device_id_))
                        << "replay=" << replay;
                    EXPECT_EQ(sample_token, expected_sample)
                        << "Qwen3.6 top-k/top-p graph replay changed sampled token at replay "
                        << replay;
                }
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_replays(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_replays(ctx);
        }

        std::vector<int> gpu_ids(top_k, -1);
        std::vector<float> gpu_probs(top_k, 0.0f);
        ASSERT_TRUE(copyDeviceToHost(
            gpu_ids.data(), d_token_ids, top_k * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            gpu_probs.data(), d_probs, top_k * sizeof(float), device_id_));
        cleanup();

        for (int i = 0; i < top_k; ++i)
        {
            EXPECT_EQ(gpu_ids[static_cast<size_t>(i)],
                      expected_distribution[static_cast<size_t>(i)].token_id)
                << "repeated graph replay compact distribution token mismatch at slot " << i;
            EXPECT_NEAR(gpu_probs[static_cast<size_t>(i)],
                        expected_distribution[static_cast<size_t>(i)].probability,
                        1e-5f)
                << "repeated graph replay compact distribution probability mismatch at slot " << i;
        }
    }

    TEST_P(GPUSamplingTest, TopKTopP_Qwen36RealLogitStyleRowsSeededSamplesMatchCPU)
    {
        constexpr int vocab_size = 248320;
        constexpr int top_k = 40;
        constexpr float top_p = 0.95f;
        constexpr float temperature = 0.6f;
        constexpr uint64_t seed = 123;

        struct HotToken
        {
            int token_id;
            float logit;
        };

        const std::vector<std::vector<HotToken>> rows = {
            {
                {262, 8.9500f}, {256, 8.9475f}, {198, 8.72f}, {471, 8.31f},
                {2972, 8.05f}, {2425, 7.92f}, {2824, 7.81f}, {64700, 7.62f},
                {357, 7.51f}, {15352, 7.42f}, {11, 7.25f}, {1575, 7.10f},
                {12, 6.96f}, {49422, 6.80f}, {6163, 6.68f}, {1358, 6.55f},
                {96220, 6.42f}, {112523, 6.30f}, {96847, 6.22f}, {104980, 6.12f},
                {98936, 6.05f}, {109120, 5.96f}, {271, 5.86f}, {248068, 5.74f},
                {8160, 5.63f}, {579, 5.51f}, {264, 5.40f}, {7047, 5.28f},
                {1817, 5.16f}, {25, 5.04f}, {16, 4.93f}, {13, 4.81f},
                {220, 4.70f}, {2014, 4.59f}, {53983, 4.47f}, {2570, 4.36f},
                {5396, 4.25f}, {1891, 4.13f}, {28758, 4.02f}, {99943, 3.91f},
            },
            {
                {256, 9.0100f}, {262, 9.0070f}, {471, 8.84f}, {2972, 8.55f},
                {1421, 8.36f}, {23398, 8.21f}, {13, 8.04f}, {198, 7.93f},
                {681, 7.80f}, {8193, 7.69f}, {883, 7.57f}, {36515, 7.44f},
                {6163, 7.32f}, {96847, 7.20f}, {1, 7.07f}, {11436, 6.95f},
                {12410, 6.84f}, {13410, 6.73f}, {1414, 6.62f}, {15613, 6.51f},
                {29223, 6.40f}, {28254, 6.29f}, {836, 6.18f}, {1919, 6.07f},
                {11, 5.96f}, {271, 5.85f}, {1835, 5.74f}, {5077, 5.63f},
                {3710, 5.52f}, {5839, 5.41f}, {5757, 5.30f}, {159034, 5.19f},
                {1503, 5.08f}, {2144, 4.97f}, {3766, 4.86f}, {16545, 4.75f},
                {51121, 4.64f}, {22527, 4.53f}, {6157, 4.42f}, {77777, 4.31f},
            },
            {
                {271, 9.15f}, {198, 8.98f}, {220, 8.77f}, {25, 8.51f},
                {16, 8.36f}, {2014, 8.24f}, {2972, 8.12f}, {579, 7.98f},
                {7047, 7.87f}, {64700, 7.74f}, {2824, 7.61f}, {2570, 7.49f},
                {262, 7.31f}, {256, 7.29f}, {11, 7.15f}, {12, 7.01f},
                {6163, 6.88f}, {49422, 6.75f}, {15352, 6.63f}, {357, 6.51f},
                {471, 6.40f}, {2425, 6.29f}, {5396, 6.18f}, {1358, 6.07f},
                {96220, 5.96f}, {112523, 5.85f}, {96847, 5.74f}, {104980, 5.63f},
                {98936, 5.52f}, {109120, 5.41f}, {248068, 5.30f}, {8160, 5.19f},
                {264, 5.08f}, {1817, 4.97f}, {13, 4.86f}, {53983, 4.75f},
                {1891, 4.64f}, {28758, 4.53f}, {99943, 4.42f}, {836, 4.31f},
            },
        };

        void *d_logits = nullptr;
        void *d_token_ids = nullptr;
        void *d_probs = nullptr;
        void *d_sample_token = nullptr;
        void *d_direct_token = nullptr;

        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_token_ids)
                backend_->free(d_token_ids, device_id_);
            if (d_probs)
                backend_->free(d_probs, device_id_);
            if (d_sample_token)
                backend_->free(d_sample_token, device_id_);
            if (d_direct_token)
                backend_->free(d_direct_token, device_id_);
        };

        d_logits = backend_->allocate(static_cast<size_t>(vocab_size) * sizeof(float), device_id_);
        d_token_ids = backend_->allocate(top_k * sizeof(int), device_id_);
        d_probs = backend_->allocate(top_k * sizeof(float), device_id_);
        d_sample_token = backend_->allocate(sizeof(int), device_id_);
        d_direct_token = backend_->allocate(sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_token_ids, nullptr);
        ASSERT_NE(d_probs, nullptr);
        ASSERT_NE(d_sample_token, nullptr);
        ASSERT_NE(d_direct_token, nullptr);

        auto run_row = [&](IWorkerGPUContext &ctx,
                           const std::vector<float> &logits,
                           float threshold,
                           uint64_t offset)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_logits,
                    logits.data(),
                    logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_logits,
                    vocab_size,
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_token_ids,
                    d_probs));
                ASSERT_TRUE(backend_->enqueueSampleDistributionF32Device(
                    d_token_ids,
                    d_probs,
                    top_k,
                    threshold,
                    device_id_,
                    stream,
                    d_sample_token));
                ASSERT_TRUE(backend_->enqueueSampleTopKTopPF32Device(
                    d_logits,
                    vocab_size,
                    top_k,
                    top_p,
                    temperature,
                    seed,
                    offset,
                    device_id_,
                    stream,
                    d_direct_token));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        for (size_t row = 0; row < rows.size(); ++row)
        {
            std::vector<float> logits(static_cast<size_t>(vocab_size), -18.0f);
            for (int i = 0; i < vocab_size; ++i)
                logits[static_cast<size_t>(i)] -=
                    0.00029f * static_cast<float>((i * 53 + static_cast<int>(row) * 17) % 997);
            for (const HotToken &hot : rows[row])
                logits[static_cast<size_t>(hot.token_id)] = hot.logit;

            const uint64_t offset = 17 + static_cast<uint64_t>(row) * 11;
            const float threshold = samplingUniform01(seed, offset);
            const auto expected_distribution =
                expectedTopKTopPDistribution(logits, top_k, top_p, temperature);
            const int expected_sample =
                expectedSampleDistributionWithThreshold(expected_distribution, threshold);
            const int expected_direct_sample =
                expectedTopKTopPSample(logits, top_k, top_p, temperature, seed, offset);
            ASSERT_EQ(expected_direct_sample, expected_sample)
                << "CPU direct/distribution sample mismatch at row " << row;

            if (GetParam() == "CUDA")
            {
                auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
                run_row(ctx, logits, threshold, offset);
            }
            else
            {
                auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
                run_row(ctx, logits, threshold, offset);
            }

            std::vector<int> gpu_ids(top_k, -1);
            std::vector<float> gpu_probs(top_k, 0.0f);
            int sample_token = -1;
            int direct_token = -1;
            ASSERT_TRUE(copyDeviceToHost(
                gpu_ids.data(), d_token_ids, top_k * sizeof(int), device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                gpu_probs.data(), d_probs, top_k * sizeof(float), device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                &sample_token, d_sample_token, sizeof(int), device_id_));
            ASSERT_TRUE(copyDeviceToHost(
                &direct_token, d_direct_token, sizeof(int), device_id_));

            for (int i = 0; i < top_k; ++i)
            {
                EXPECT_EQ(gpu_ids[static_cast<size_t>(i)],
                          expected_distribution[static_cast<size_t>(i)].token_id)
                    << "real-logit-style row " << row
                    << " compact distribution token mismatch at slot " << i;
                EXPECT_NEAR(gpu_probs[static_cast<size_t>(i)],
                            expected_distribution[static_cast<size_t>(i)].probability,
                            1e-5f)
                    << "real-logit-style row " << row
                    << " compact distribution probability mismatch at slot " << i;
            }
            EXPECT_EQ(sample_token, expected_sample)
                << "real-logit-style row " << row
                << " compact distribution sample mismatch";
            EXPECT_EQ(direct_token, expected_direct_sample)
                << "real-logit-style row " << row
                << " direct top-k/top-p sample mismatch";
        }

        cleanup();
    }

    TEST_P(GPUSamplingTest, TopKTopPProcessedLogits_Qwen36VocabTopK40_MatchesCPUAndCaptures)
    {
        constexpr int row_count = 3;
        constexpr int vocab_size = 248320;
        constexpr int row_stride = vocab_size + 7;
        constexpr int out_stride = vocab_size + 11;
        constexpr int top_k = 40;
        constexpr float top_p = 0.95f;
        constexpr float temperature = 0.6f;
        constexpr int scratch_capacity = row_count * 128 * top_k;
        const std::array<float, row_count> thresholds = {0.11f, 0.53f, 0.87f};

        struct HotToken
        {
            int token_id;
            float logit;
        };
        const std::array<std::array<HotToken, top_k>, row_count> hot_rows = {{
            {{
                {262, 8.9500f}, {256, 8.9475f}, {198, 8.72f}, {471, 8.31f},
                {2972, 8.05f}, {2425, 7.92f}, {2824, 7.81f}, {64700, 7.62f},
                {357, 7.51f}, {15352, 7.42f}, {11, 7.25f}, {1575, 7.10f},
                {12, 6.96f}, {49422, 6.80f}, {6163, 6.68f}, {1358, 6.55f},
                {96220, 6.42f}, {112523, 6.30f}, {96847, 6.22f}, {104980, 6.12f},
                {98936, 6.05f}, {109120, 5.96f}, {271, 5.86f}, {248068, 5.74f},
                {8160, 5.63f}, {579, 5.51f}, {264, 5.40f}, {7047, 5.28f},
                {1817, 5.16f}, {25, 5.04f}, {16, 4.93f}, {13, 4.81f},
                {220, 4.70f}, {2014, 4.59f}, {53983, 4.47f}, {2570, 4.36f},
                {5396, 4.25f}, {1891, 4.13f}, {28758, 4.02f}, {99943, 3.91f},
            }},
            {{
                {256, 9.0100f}, {262, 9.0070f}, {471, 8.84f}, {2972, 8.55f},
                {1421, 8.36f}, {23398, 8.21f}, {13, 8.04f}, {198, 7.93f},
                {681, 7.80f}, {8193, 7.69f}, {883, 7.57f}, {36515, 7.44f},
                {6163, 7.32f}, {96847, 7.20f}, {1, 7.07f}, {11436, 6.95f},
                {12410, 6.84f}, {13410, 6.73f}, {1414, 6.62f}, {15613, 6.51f},
                {29223, 6.40f}, {28254, 6.29f}, {836, 6.18f}, {1919, 6.07f},
                {11, 5.96f}, {271, 5.85f}, {1835, 5.74f}, {5077, 5.63f},
                {3710, 5.52f}, {5839, 5.41f}, {5757, 5.30f}, {159034, 5.19f},
                {1503, 5.08f}, {2144, 4.97f}, {3766, 4.86f}, {16545, 4.75f},
                {51121, 4.64f}, {22527, 4.53f}, {6157, 4.42f}, {77777, 4.31f},
            }},
            {{
                {271, 9.15f}, {198, 8.98f}, {220, 8.77f}, {25, 8.51f},
                {16, 8.36f}, {2014, 8.24f}, {2972, 8.12f}, {579, 7.98f},
                {7047, 7.87f}, {64700, 7.74f}, {2824, 7.61f}, {2570, 7.49f},
                {262, 7.31f}, {256, 7.29f}, {11, 7.15f}, {12, 7.01f},
                {6163, 6.88f}, {49422, 6.75f}, {15352, 6.63f}, {357, 6.51f},
                {471, 6.40f}, {2425, 6.29f}, {5396, 6.18f}, {1358, 6.07f},
                {96220, 5.96f}, {112523, 5.85f}, {96847, 5.74f}, {104980, 5.63f},
                {98936, 5.52f}, {109120, 5.41f}, {248068, 5.30f}, {8160, 5.19f},
                {264, 5.08f}, {1817, 4.97f}, {13, 4.86f}, {53983, 4.75f},
                {1891, 4.64f}, {28758, 4.53f}, {99943, 4.42f}, {836, 4.31f},
            }},
        }};

        std::vector<float> logits(static_cast<size_t>(row_count) * row_stride, -18.0f);
        std::vector<std::vector<float>> expected_processed;
        std::vector<std::vector<ExpectedDistributionEntry>> expected_distributions;
        std::array<int, row_count> expected_samples{};
        std::array<float, row_count> expected_sample_probs{};
        expected_processed.reserve(row_count);
        expected_distributions.reserve(row_count);
        for (int row = 0; row < row_count; ++row)
        {
            std::vector<float> row_logits(static_cast<size_t>(vocab_size), -18.0f);
            for (int token = 0; token < vocab_size; ++token)
            {
                row_logits[static_cast<size_t>(token)] -=
                    0.00029f * static_cast<float>((token * 53 + row * 17) % 997);
            }
            for (const HotToken &hot : hot_rows[static_cast<size_t>(row)])
                row_logits[static_cast<size_t>(hot.token_id)] = hot.logit;

            std::copy(row_logits.begin(), row_logits.end(),
                      logits.begin() + static_cast<ptrdiff_t>(row * row_stride));
            expected_distributions.push_back(
                expectedTopKTopPDistribution(row_logits, top_k, top_p, temperature));
            expected_processed.push_back(
                expectedTopKTopPProcessedLogits(row_logits, top_k, top_p, temperature));
            expected_samples[static_cast<size_t>(row)] =
                sampleMTPTokenFromProcessedLogits(
                    expected_processed.back().data(),
                    vocab_size,
                    thresholds[static_cast<size_t>(row)]);
            const MTPFullLogitRowStats stats =
                computeMTPFullLogitRowStats(
                    expected_processed.back().data(),
                    vocab_size);
            ASSERT_TRUE(stats.ok) << stats.error;
            expected_sample_probs[static_cast<size_t>(row)] =
                probabilityFromMTPFullLogits(
                    expected_processed.back().data(),
                    vocab_size,
                    stats,
                    expected_samples[static_cast<size_t>(row)]);
        }

        void *d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        void *d_processed = backend_->allocate(
            static_cast<size_t>(row_count) * out_stride * sizeof(float),
            device_id_);
        void *d_scratch_values = backend_->allocate(
            static_cast<size_t>(scratch_capacity) * sizeof(float),
            device_id_);
        void *d_scratch_indices = backend_->allocate(
            static_cast<size_t>(scratch_capacity) * sizeof(int),
            device_id_);
        void *d_samples = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_sample_probs =
            backend_->allocate(row_count * sizeof(float), device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_logits, d_processed, d_scratch_values, d_scratch_indices,
                d_samples, d_sample_probs};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_processed, nullptr);
        ASSERT_NE(d_scratch_values, nullptr);
        ASSERT_NE(d_scratch_indices, nullptr);
        ASSERT_NE(d_samples, nullptr);
        ASSERT_NE(d_sample_probs, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_logits,
                    logits.data(),
                    logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueBuildTopKTopPProcessedLogitsF32Device(
                    d_logits,
                    row_count,
                    vocab_size,
                    row_stride,
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    nullptr,
                    d_processed,
                    out_stride,
                    d_scratch_values,
                    d_scratch_indices,
                    scratch_capacity))
                    << "processed-logit top-k/top-p warper must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPProcessedLogitsF32Device(
                    d_logits,
                    row_count,
                    vocab_size,
                    row_stride,
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_processed,
                    out_stride,
                    d_scratch_values,
                    d_scratch_indices,
                    scratch_capacity));
                for (int row = 0; row < row_count; ++row)
                {
                    ASSERT_TRUE(backend_->enqueueSampleProcessedLogitsF32Device(
                        static_cast<float *>(d_processed) +
                            static_cast<size_t>(row) * out_stride,
                        vocab_size,
                        out_stride,
                        thresholds[static_cast<size_t>(row)],
                        device_id_,
                        stream,
                        static_cast<int *>(d_samples) + row,
                        static_cast<float *>(d_sample_probs) + row));
                }
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<float> gpu_processed(static_cast<size_t>(row_count) * out_stride, 0.0f);
        std::array<int, row_count> gpu_samples{};
        std::array<float, row_count> gpu_sample_probs{};
        ASSERT_TRUE(copyDeviceToHost(
            gpu_processed.data(),
            d_processed,
            gpu_processed.size() * sizeof(float),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            gpu_samples.data(),
            d_samples,
            gpu_samples.size() * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            gpu_sample_probs.data(),
            d_sample_probs,
            gpu_sample_probs.size() * sizeof(float),
            device_id_));
        cleanup();

        for (int row = 0; row < row_count; ++row)
        {
            const float *gpu_row =
                gpu_processed.data() + static_cast<size_t>(row) * out_stride;
            const auto &expected_row = expected_processed[static_cast<size_t>(row)];
            int finite_count = 0;
            int expected_finite_count = 0;
            int mismatch_count = 0;
            for (int token = 0; token < vocab_size; ++token)
            {
                const bool gpu_active = std::isfinite(gpu_row[token]);
                const bool expected_active =
                    std::isfinite(expected_row[static_cast<size_t>(token)]);
                finite_count += gpu_active ? 1 : 0;
                expected_finite_count += expected_active ? 1 : 0;
                if (gpu_active != expected_active)
                {
                    ++mismatch_count;
                    continue;
                }
                if (expected_active &&
                    std::fabs(gpu_row[token] -
                              expected_row[static_cast<size_t>(token)]) > 1e-5f)
                {
                    ++mismatch_count;
                }
            }

            EXPECT_EQ(finite_count, expected_finite_count)
                << "processed-logit active-token count mismatch at row " << row;
            EXPECT_EQ(mismatch_count, 0)
                << "processed-logit mask/value mismatch at row " << row;

            const MTPFullLogitRowStats stats =
                computeMTPFullLogitRowStats(gpu_row, vocab_size);
            ASSERT_TRUE(stats.ok) << stats.error;
            for (const auto &entry : expected_distributions[static_cast<size_t>(row)])
            {
                if (entry.token_id < 0 || !(entry.probability > 0.0f))
                    continue;
                const float gpu_probability =
                    probabilityFromMTPFullLogits(
                        gpu_row,
                        vocab_size,
                        stats,
                        entry.token_id);
                EXPECT_NEAR(gpu_probability, entry.probability, 2e-5f)
                    << "processed-logit probability mismatch at row " << row
                    << ", token " << entry.token_id;
            }

            EXPECT_EQ(gpu_samples[static_cast<size_t>(row)],
                      expected_samples[static_cast<size_t>(row)])
                << "processed-logit sample mismatch at row " << row;
            EXPECT_NEAR(gpu_sample_probs[static_cast<size_t>(row)],
                        expected_sample_probs[static_cast<size_t>(row)],
                        2e-5f)
                << "processed-logit selected probability mismatch at row "
                << row;
        }
    }

    TEST_P(GPUSamplingTest, ProcessedLogitSpeculativeVerifyMatchesReferenceAndCaptures)
    {
        constexpr int vocab_size = 16;
        constexpr int row_count = 31;
        std::vector<int> draft_tokens(static_cast<size_t>(row_count), 0);
        std::vector<float> accept_thresholds(static_cast<size_t>(row_count));
        std::vector<float> residual_thresholds(
            static_cast<size_t>(row_count),
            0.0f);
        for (int row = 0; row < row_count; ++row)
        {
            accept_thresholds[static_cast<size_t>(row)] =
                row % 2 == 0 ? 0.39f : 0.41f;
        }

        auto logits_from_probs = [](std::initializer_list<float> probs)
        {
            std::vector<float> logits;
            logits.reserve(probs.size());
            for (float p : probs)
            {
                logits.push_back(
                    p > 0.0f
                        ? std::log(p)
                        : -std::numeric_limits<float>::infinity());
            }
            return logits;
        };
        auto append_row = [](std::vector<float> &rows,
                             const std::vector<float> &row)
        {
            rows.insert(rows.end(), row.begin(), row.end());
        };

        const std::vector<float> target_row =
            logits_from_probs({0.2f, 0.8f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f});
        const std::vector<float> draft_row =
            logits_from_probs({0.5f, 0.5f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f});

        std::vector<float> target_rows;
        std::vector<float> draft_rows;
        target_rows.reserve(static_cast<size_t>(row_count) * vocab_size);
        draft_rows.reserve(static_cast<size_t>(row_count) * vocab_size);
        for (int row = 0; row < row_count; ++row)
        {
            append_row(target_rows, target_row);
            append_row(draft_rows, draft_row);
        }

        std::vector<MTPRejectionSampleRowResult> expected(
            static_cast<size_t>(row_count));
        for (int row = 0; row < row_count; ++row)
        {
            expected[static_cast<size_t>(row)] =
                sampleMTPRejectionRowFromProcessedLogits(
                    target_rows.data() + static_cast<size_t>(row) * vocab_size,
                    draft_rows.data() + static_cast<size_t>(row) * vocab_size,
                    vocab_size,
                    draft_tokens[static_cast<size_t>(row)],
                    accept_thresholds[static_cast<size_t>(row)],
                    residual_thresholds[static_cast<size_t>(row)]);
            ASSERT_TRUE(expected[static_cast<size_t>(row)].ok)
                << expected[static_cast<size_t>(row)].error;
        }
        ASSERT_TRUE(expected[0].accepted);
        ASSERT_FALSE(expected[1].accepted);

        void *d_target = backend_->allocate(target_rows.size() * sizeof(float), device_id_);
        void *d_draft = backend_->allocate(draft_rows.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(draft_tokens.size() * sizeof(int), device_id_);
        void *d_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accept_probs = backend_->allocate(row_count * sizeof(float), device_id_);
        void *d_accept_thresholds = backend_->allocate(row_count * sizeof(float), device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target, d_draft, d_draft_tokens, d_tokens,
                d_accepted, d_accept_probs, d_accept_thresholds};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target, nullptr);
        ASSERT_NE(d_draft, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_tokens, nullptr);
        ASSERT_NE(d_accepted, nullptr);
        ASSERT_NE(d_accept_probs, nullptr);
        ASSERT_NE(d_accept_thresholds, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_target, target_rows.data(),
                    target_rows.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft, draft_rows.data(),
                    draft_rows.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens, draft_tokens.data(),
                    draft_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(
                    backend_->enqueueSpeculativeVerifyProcessedLogitsF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft, row_count, vocab_size, vocab_size,
                        vocab_size, d_draft_tokens, accept_thresholds.data(),
                        residual_thresholds.data(), device_id_, nullptr,
                        d_tokens, d_accepted, d_accept_probs,
                        d_accept_thresholds))
                    << "processed-logit verifier must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProcessedLogitsF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft, row_count, vocab_size, vocab_size,
                        vocab_size, d_draft_tokens, accept_thresholds.data(),
                        residual_thresholds.data(), device_id_, stream,
                        d_tokens, d_accepted, d_accept_probs,
                        d_accept_thresholds));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<int> tokens(static_cast<size_t>(row_count));
        std::vector<int> accepted(static_cast<size_t>(row_count));
        std::vector<float> accept_probs(static_cast<size_t>(row_count));
        ASSERT_TRUE(copyDeviceToHost(tokens.data(), d_tokens,
                                           tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(accepted.data(), d_accepted,
                                           accepted.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(accept_probs.data(), d_accept_probs,
                                           accept_probs.size() * sizeof(float), device_id_));

        cleanup();

        for (int row = 0; row < row_count; ++row)
        {
            const auto &exp = expected[static_cast<size_t>(row)];
            EXPECT_EQ(tokens[static_cast<size_t>(row)], exp.token)
                << "row " << row;
            EXPECT_EQ(accepted[static_cast<size_t>(row)], exp.accepted ? 1 : 0)
                << "row " << row;
            EXPECT_NEAR(accept_probs[static_cast<size_t>(row)],
                        exp.accept_probability,
                        1e-5f)
                << "row " << row;
        }
    }

    TEST_P(GPUSamplingTest, ProcessedLogitSoftmaxMatchesReferenceAndCaptures)
    {
        constexpr int vocab_size = 8;
        constexpr int row_count = 2;
        const float neg_inf = -std::numeric_limits<float>::infinity();
        const std::vector<float> logits = {
            neg_inf, 2.0f, 1.0f, neg_inf, 0.0f, neg_inf, neg_inf, neg_inf,
            4.0f, 4.0f, neg_inf, 1.0f, neg_inf, neg_inf, 0.0f, neg_inf};
        ASSERT_EQ(
            logits.size(),
            static_cast<size_t>(row_count) * vocab_size);
        std::vector<float> expected(logits.size(), 0.0f);
        for (int row = 0; row < row_count; ++row)
        {
            const float *row_logits =
                logits.data() + static_cast<size_t>(row) * vocab_size;
            const MTPFullLogitRowStats stats =
                computeMTPFullLogitRowStats(row_logits, vocab_size);
            ASSERT_TRUE(stats.ok) << stats.error;
            for (int token = 0; token < vocab_size; ++token)
            {
                expected[static_cast<size_t>(row) * vocab_size + token] =
                    probabilityFromMTPFullLogits(
                        row_logits,
                        vocab_size,
                        stats,
                        token);
            }
        }

        void *d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        void *d_probs = backend_->allocate(expected.size() * sizeof(float), device_id_);
        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_probs)
                backend_->free(d_probs, device_id_);
        };

        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_probs, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_logits, logits.data(), logits.size() * sizeof(float),
                    device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueSoftmaxProcessedLogitsF32Device(
                    d_logits, row_count, vocab_size, vocab_size,
                    device_id_, nullptr, d_probs, vocab_size))
                    << "processed-logit softmax must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueSoftmaxProcessedLogitsF32Device(
                    d_logits, row_count, vocab_size, vocab_size,
                    device_id_, stream, d_probs, vocab_size));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<float> actual(expected.size(), -1.0f);
        ASSERT_TRUE(copyDeviceToHost(
            actual.data(), d_probs, actual.size() * sizeof(float), device_id_));
        cleanup();

        for (size_t i = 0; i < actual.size(); ++i)
        {
            EXPECT_NEAR(actual[i], expected[i], 1e-6f)
                << "probability index " << i;
        }
    }

    TEST_P(GPUSamplingTest, TemperatureDraftProposalSoftmaxSampleMatchesReferenceAndCaptures)
    {
        constexpr int vocab_size = 257;
        constexpr float temperature = 0.73f;
        constexpr float threshold = 0.61f;

        std::vector<float> logits(vocab_size, -7.0f);
        for (int token = 0; token < vocab_size; ++token)
        {
            logits[static_cast<size_t>(token)] =
                std::sin(static_cast<float>(token) * 0.13f) * 2.0f -
                static_cast<float>(token % 11) * 0.07f;
        }
        logits[3] = 8.0f;
        logits[17] = 7.5f;
        logits[199] = 6.75f;

        const std::vector<float> expected_probs =
            expectedTemperatureOnlyProbabilities(logits, temperature);
        const int expected_token =
            expectedTemperatureOnlySampleToken(expected_probs, threshold);
        ASSERT_GE(expected_token, 0);
        ASSERT_LT(expected_token, vocab_size);
        const float expected_probability =
            expected_probs[static_cast<size_t>(expected_token)];

        void *d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        void *d_probs = backend_->allocate(expected_probs.size() * sizeof(float), device_id_);
        void *d_token = backend_->allocate(sizeof(int), device_id_);
        void *d_probability = backend_->allocate(sizeof(float), device_id_);
        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_probs)
                backend_->free(d_probs, device_id_);
            if (d_token)
                backend_->free(d_token, device_id_);
            if (d_probability)
                backend_->free(d_probability, device_id_);
        };
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_probs, nullptr);
        ASSERT_NE(d_token, nullptr);
        ASSERT_NE(d_probability, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_logits, logits.data(), logits.size() * sizeof(float),
                    device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueSoftmaxAndSampleTemperatureLogitsF32Device(
                    d_logits,
                    vocab_size,
                    vocab_size,
                    temperature,
                    threshold,
                    device_id_,
                    nullptr,
                    d_probs,
                    vocab_size,
                    d_token,
                    d_probability))
                    << "temperature draft proposal must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueSoftmaxAndSampleTemperatureLogitsF32Device(
                    d_logits,
                    vocab_size,
                    vocab_size,
                    temperature,
                    threshold,
                    device_id_,
                    stream,
                    d_probs,
                    vocab_size,
                    d_token,
                    d_probability));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<float> actual_probs(expected_probs.size(), -1.0f);
        int actual_token = -1;
        float actual_probability = -1.0f;
        ASSERT_TRUE(copyDeviceToHost(
            actual_probs.data(),
            d_probs,
            actual_probs.size() * sizeof(float),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &actual_token,
            d_token,
            sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &actual_probability,
            d_probability,
            sizeof(float),
            device_id_));
        cleanup();

        float actual_sum = 0.0f;
        for (size_t i = 0; i < actual_probs.size(); ++i)
        {
            actual_sum += actual_probs[i];
            EXPECT_NEAR(actual_probs[i], expected_probs[i], 2e-6f)
                << "probability index " << i;
        }
        EXPECT_NEAR(actual_sum, 1.0f, 2e-5f);
        EXPECT_EQ(actual_token, expected_token);
        EXPECT_NEAR(actual_probability, expected_probability, 2e-6f);
    }

    TEST_P(GPUSamplingTest, TemperatureDraftLogitProposalAndVerifierMatchesReferenceAndCaptures)
    {
        constexpr int vocab_size = 257;
        constexpr int row_count = 3;
        constexpr float temperature = 0.77f;
        constexpr uint64_t inverse_sample_seed = 98765;
        constexpr int first_logical_position = 19;
        constexpr uint64_t kInverseSampleDomain = 0xA0761D6478BD642FULL;
        const float neg_inf = -std::numeric_limits<float>::infinity();

        const std::array<float, row_count> proposal_thresholds = {0.19f, 0.47f, 0.83f};
        const std::array<float, row_count> accept_thresholds = {0.25f, 0.92f, 0.10f};

        std::vector<float> draft_raw_logits(
            static_cast<size_t>(row_count) * vocab_size,
            -5.0f);
        for (int row = 0; row < row_count; ++row)
        {
            const size_t base = static_cast<size_t>(row) * vocab_size;
            for (int token = 0; token < vocab_size; ++token)
            {
                draft_raw_logits[base + static_cast<size_t>(token)] =
                    std::sin(static_cast<float>(token + row * 17) * 0.17f) * 1.7f -
                    0.015f * static_cast<float>((token + row * 31) % 29);
            }
            draft_raw_logits[base + static_cast<size_t>((13 + row * 41) % vocab_size)] = 6.1f;
            draft_raw_logits[base + static_cast<size_t>((71 + row * 29) % vocab_size)] = 5.3f;
        }

        std::vector<float> expected_draft_logits(draft_raw_logits.size(), neg_inf);
        std::vector<float> draft_probs(draft_raw_logits.size(), 0.0f);
        std::array<int, row_count> draft_tokens{};
        std::array<float, row_count> draft_token_probs{};
        for (int row = 0; row < row_count; ++row)
        {
            const size_t base = static_cast<size_t>(row) * vocab_size;
            std::vector<float> row_logits(
                draft_raw_logits.begin() + static_cast<std::ptrdiff_t>(base),
                draft_raw_logits.begin() + static_cast<std::ptrdiff_t>(base + vocab_size));
            std::vector<float> row_probs =
                expectedTemperatureOnlyProbabilities(row_logits, temperature);
            draft_tokens[static_cast<size_t>(row)] =
                expectedTemperatureOnlySampleToken(
                    row_probs, proposal_thresholds[static_cast<size_t>(row)]);
            draft_token_probs[static_cast<size_t>(row)] =
                row_probs[static_cast<size_t>(draft_tokens[static_cast<size_t>(row)])];
            for (int token = 0; token < vocab_size; ++token)
            {
                expected_draft_logits[base + static_cast<size_t>(token)] =
                    draft_raw_logits[base + static_cast<size_t>(token)] / temperature;
                draft_probs[base + static_cast<size_t>(token)] =
                    row_probs[static_cast<size_t>(token)];
            }
        }

        std::vector<float> target_logits(
            static_cast<size_t>(row_count) * vocab_size,
            neg_inf);
        std::vector<float> target_probs(
            static_cast<size_t>(row_count) * vocab_size,
            0.0f);
        auto set_target_row =
            [&](int row, float draft_prob, float alt0_prob, float alt1_prob)
        {
            ASSERT_NEAR(draft_prob + alt0_prob + alt1_prob, 1.0f, 1e-6f);
            const size_t base = static_cast<size_t>(row) * vocab_size;
            const int draft_token = draft_tokens[static_cast<size_t>(row)];
            const int alt0 = (draft_token + 17) % vocab_size;
            const int alt1 = (draft_token + 89) % vocab_size;
            target_probs[base + static_cast<size_t>(draft_token)] = draft_prob;
            target_probs[base + static_cast<size_t>(alt0)] = alt0_prob;
            target_probs[base + static_cast<size_t>(alt1)] = alt1_prob;
            target_logits[base + static_cast<size_t>(draft_token)] = std::log(draft_prob);
            target_logits[base + static_cast<size_t>(alt0)] = std::log(alt0_prob);
            target_logits[base + static_cast<size_t>(alt1)] = std::log(alt1_prob);
        };
        set_target_row(0, 0.80f, 0.12f, 0.08f);
        set_target_row(1, 0.01f, 0.70f, 0.29f);
        set_target_row(2, 0.55f, 0.25f, 0.20f);

        std::vector<float> inverse_rows(draft_raw_logits.size(), 0.0f);
        for (int row = 0; row < row_count; ++row)
        {
            for (int token = 0; token < vocab_size; ++token)
            {
                const uint64_t offset =
                    static_cast<uint64_t>(first_logical_position + row) *
                        static_cast<uint64_t>(vocab_size) +
                    static_cast<uint64_t>(token);
                const float uniform =
                    sampling_math::uniform01(
                        inverse_sample_seed ^ kInverseSampleDomain,
                        offset);
                inverse_rows[static_cast<size_t>(row) * vocab_size +
                             static_cast<size_t>(token)] =
                    sampling_math::inverse_exponential_from_uniform(uniform);
            }
        }

        std::array<MTPRejectionSampleRowResult, row_count> expected{};
        for (int row = 0; row < row_count; ++row)
        {
            const size_t base = static_cast<size_t>(row) * vocab_size;
            expected[static_cast<size_t>(row)] =
                sampleMTPRejectionRowFromProbabilities(
                    target_probs.data() + base,
                    draft_probs.data() + base,
                    inverse_rows.data() + base,
                    vocab_size,
                    draft_tokens[static_cast<size_t>(row)],
                    accept_thresholds[static_cast<size_t>(row)]);
            ASSERT_TRUE(expected[static_cast<size_t>(row)].ok)
                << expected[static_cast<size_t>(row)].error;
        }
        ASSERT_TRUE(expected[0].accepted);
        ASSERT_FALSE(expected[1].accepted);
        ASSERT_TRUE(expected[2].accepted);

        void *d_target = backend_->allocate(target_logits.size() * sizeof(float), device_id_);
        void *d_draft_raw = backend_->allocate(draft_raw_logits.size() * sizeof(float), device_id_);
        void *d_draft_logits = backend_->allocate(expected_draft_logits.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(draft_tokens.size() * sizeof(int), device_id_);
        void *d_draft_token_probs = backend_->allocate(draft_token_probs.size() * sizeof(float), device_id_);
        void *d_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accept_probs = backend_->allocate(row_count * sizeof(float), device_id_);
        void *d_accept_thresholds = backend_->allocate(row_count * sizeof(float), device_id_);
        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target, d_draft_raw, d_draft_logits, d_draft_tokens,
                d_draft_token_probs, d_tokens, d_accepted,
                d_accept_probs, d_accept_thresholds};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target, nullptr);
        ASSERT_NE(d_draft_raw, nullptr);
        ASSERT_NE(d_draft_logits, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_draft_token_probs, nullptr);
        ASSERT_NE(d_tokens, nullptr);
        ASSERT_NE(d_accepted, nullptr);
        ASSERT_NE(d_accept_probs, nullptr);
        ASSERT_NE(d_accept_thresholds, nullptr);

        std::array<int, row_count> first_tokens{};
        std::array<int, row_count> first_accepted{};
        std::array<float, row_count> first_accept_probs{};
        std::array<int, row_count> second_tokens{};
        std::array<int, row_count> second_accepted{};
        std::vector<float> actual_draft_logits(expected_draft_logits.size(), 0.0f);
        std::array<int, row_count> actual_draft_tokens{};
        std::array<float, row_count> actual_draft_token_probs{};

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_target, target_logits.data(),
                    target_logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_raw, draft_raw_logits.data(),
                    draft_raw_logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueScaleAndSampleTemperatureLogitsF32Device(
                    d_draft_raw, vocab_size, vocab_size, temperature,
                    proposal_thresholds[0], device_id_, nullptr,
                    d_draft_logits, vocab_size, d_draft_tokens,
                    d_draft_token_probs))
                    << "temperature draft-logit proposal must reject the legacy default/null stream";
                EXPECT_FALSE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftLogitsF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft_logits, row_count, vocab_size, vocab_size,
                        vocab_size, d_draft_tokens, accept_thresholds.data(),
                        inverse_sample_seed, first_logical_position,
                        device_id_, nullptr, d_tokens, d_accepted,
                        d_accept_probs, d_accept_thresholds, d_draft_token_probs))
                    << "processed-target/draft-logit verifier must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                for (int row = 0; row < row_count; ++row)
                {
                    ASSERT_TRUE(backend_->enqueueScaleAndSampleTemperatureLogitsF32Device(
                        static_cast<float *>(d_draft_raw) +
                            static_cast<size_t>(row) * vocab_size,
                        vocab_size,
                        vocab_size,
                        temperature,
                        proposal_thresholds[static_cast<size_t>(row)],
                        device_id_,
                        stream,
                        static_cast<float *>(d_draft_logits) +
                            static_cast<size_t>(row) * vocab_size,
                        vocab_size,
                        static_cast<int *>(d_draft_tokens) + row,
                        static_cast<float *>(d_draft_token_probs) + row));
                }
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftLogitsF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft_logits, row_count, vocab_size, vocab_size,
                        vocab_size, d_draft_tokens, accept_thresholds.data(),
                        inverse_sample_seed, first_logical_position,
                        device_id_, stream, d_tokens, d_accepted,
                        d_accept_probs, d_accept_thresholds, d_draft_token_probs));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                ASSERT_TRUE(copyDeviceToHost(
                    first_tokens.data(), d_tokens,
                    first_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    first_accepted.data(), d_accepted,
                    first_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    first_accept_probs.data(), d_accept_probs,
                    first_accept_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    actual_draft_logits.data(), d_draft_logits,
                    actual_draft_logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    actual_draft_tokens.data(), d_draft_tokens,
                    actual_draft_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    actual_draft_token_probs.data(), d_draft_token_probs,
                    actual_draft_token_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                const std::array<int, row_count> sentinel_tokens = {-1, -1, -1};
                const std::array<int, row_count> sentinel_accepted = {-7, -7, -7};
                ASSERT_TRUE(copyHostToDevice(
                    d_tokens, sentinel_tokens.data(),
                    sentinel_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_accepted, sentinel_accepted.data(),
                    sentinel_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                ASSERT_TRUE(copyDeviceToHost(
                    second_tokens.data(), d_tokens,
                    second_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    second_accepted.data(), d_accepted,
                    second_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        cleanup();

        EXPECT_EQ(second_tokens, first_tokens);
        EXPECT_EQ(second_accepted, first_accepted);
        for (int row = 0; row < row_count; ++row)
        {
            EXPECT_EQ(actual_draft_tokens[static_cast<size_t>(row)],
                      draft_tokens[static_cast<size_t>(row)])
                << "draft token row=" << row;
            EXPECT_NEAR(actual_draft_token_probs[static_cast<size_t>(row)],
                        draft_token_probs[static_cast<size_t>(row)],
                        2e-6f)
                << "draft probability row=" << row;

            const auto &exp = expected[static_cast<size_t>(row)];
            EXPECT_EQ(first_tokens[static_cast<size_t>(row)], exp.token)
                << "verifier token row=" << row;
            EXPECT_EQ(first_accepted[static_cast<size_t>(row)], exp.accepted ? 1 : 0)
                << "verifier accepted row=" << row;
            EXPECT_NEAR(first_accept_probs[static_cast<size_t>(row)],
                        exp.accept_probability,
                        2e-6f)
                << "accept probability row=" << row;
        }
        for (size_t i = 0; i < actual_draft_logits.size(); ++i)
        {
            EXPECT_NEAR(actual_draft_logits[i], expected_draft_logits[i], 2e-6f)
                << "draft logit index=" << i;
        }
    }

    TEST_P(GPUSamplingTest, InverseExponentialSamplesMatchSharedMathAndCapture)
    {
        constexpr int vocab_size = 17;
        constexpr int row_count = 2;
        constexpr int row_stride = 20;
        constexpr uint64_t seed = 12345;
        constexpr int first_logical_position = 7;
        constexpr uint64_t kInverseSampleDomain = 0xA0761D6478BD642FULL;

        std::vector<float> expected(static_cast<size_t>(row_count) * row_stride,
                                    -1.0f);
        for (int row = 0; row < row_count; ++row)
        {
            for (int token = 0; token < vocab_size; ++token)
            {
                const uint64_t logical_position =
                    static_cast<uint64_t>(first_logical_position + row);
                const uint64_t offset =
                    logical_position * static_cast<uint64_t>(vocab_size) +
                    static_cast<uint64_t>(token);
                const float uniform =
                    sampling_math::uniform01(seed ^ kInverseSampleDomain, offset);
                expected[static_cast<size_t>(row) * row_stride + token] =
                    sampling_math::inverse_exponential_from_uniform(uniform);
            }
        }

        void *d_samples = backend_->allocate(expected.size() * sizeof(float), device_id_);
        auto cleanup = [&]()
        {
            if (d_samples)
                backend_->free(d_samples, device_id_);
        };
        ASSERT_NE(d_samples, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                EXPECT_FALSE(backend_->enqueueFillInverseExponentialSamplesF32Device(
                    d_samples, row_count, vocab_size, row_stride,
                    seed, first_logical_position, device_id_, nullptr))
                    << "inverse-sample fill must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueFillInverseExponentialSamplesF32Device(
                    d_samples,
                    row_count,
                    vocab_size,
                    row_stride,
                    seed,
                    first_logical_position,
                    device_id_,
                    stream));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<float> actual(expected.size(), -1.0f);
        ASSERT_TRUE(copyDeviceToHost(
            actual.data(), d_samples, actual.size() * sizeof(float), device_id_));
        cleanup();

        for (int row = 0; row < row_count; ++row)
        {
            for (int token = 0; token < vocab_size; ++token)
            {
                const size_t index = static_cast<size_t>(row) * row_stride + token;
                const float tolerance =
                    std::max(1.0e-5f, std::abs(expected[index]) * 2.0e-5f);
                EXPECT_NEAR(actual[index], expected[index], tolerance)
                    << "row=" << row << " token=" << token;
            }
        }
    }

    TEST_P(GPUSamplingTest, ProcessedTargetDraftProbabilitySpeculativeVerifyMatchesReferenceAndReplay)
    {
        constexpr int vocab_size = 248320;
        constexpr int row_count = 3;
        constexpr uint64_t inverse_sample_seed = 123;
        constexpr int first_logical_position = 11;
        constexpr uint64_t kInverseSampleDomain = 0xA0761D6478BD642FULL;
        const float neg_inf = -std::numeric_limits<float>::infinity();

        const std::array<int, row_count> draft_tokens = {13, 13, 271};
        const std::array<float, row_count> accept_thresholds = {0.95f, 0.80f, 0.49f};

        std::vector<float> target_logits(
            static_cast<size_t>(row_count) * static_cast<size_t>(vocab_size),
            neg_inf);
        std::vector<float> target_probs(
            static_cast<size_t>(row_count) * static_cast<size_t>(vocab_size),
            0.0f);
        std::vector<float> draft_probs(target_probs.size(), 0.0f);
        std::vector<float> inverse_rows(target_probs.size(), 0.0f);

        auto set_target_row =
            [&](int row, std::initializer_list<std::pair<int, float>> entries)
        {
            const size_t base =
                static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
            float total = 0.0f;
            for (const auto &entry : entries)
                total += entry.second;
            ASSERT_NEAR(total, 1.0f, 1e-5f);
            for (const auto &entry : entries)
            {
                ASSERT_GE(entry.first, 0);
                ASSERT_LT(entry.first, vocab_size);
                target_probs[base + static_cast<size_t>(entry.first)] =
                    entry.second;
                target_logits[base + static_cast<size_t>(entry.first)] =
                    std::log(entry.second);
            }
        };

        auto set_draft_row =
            [&](int row, std::initializer_list<std::pair<int, float>> entries)
        {
            const size_t base =
                static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
            float special_total = 0.0f;
            std::set<int> special_tokens;
            for (const auto &entry : entries)
            {
                ASSERT_GE(entry.first, 0);
                ASSERT_LT(entry.first, vocab_size);
                special_total += entry.second;
                special_tokens.insert(entry.first);
            }
            ASSERT_LT(special_total, 1.0f);
            const float background =
                (1.0f - special_total) /
                static_cast<float>(vocab_size - static_cast<int>(special_tokens.size()));
            for (int token = 0; token < vocab_size; ++token)
            {
                draft_probs[base + static_cast<size_t>(token)] =
                    special_tokens.count(token) ? 0.0f : background;
            }
            for (const auto &entry : entries)
            {
                draft_probs[base + static_cast<size_t>(entry.first)] =
                    entry.second;
            }
        };

        set_target_row(0, {{13, 0.60f}, {271, 0.20f}, {1061, 0.10f}, {33075, 0.10f}});
        set_draft_row(0, {{13, 0.20f}, {271, 0.30f}, {1061, 0.10f}, {33075, 0.05f}});

        set_target_row(1, {{13, 0.10f}, {271, 0.55f}, {1061, 0.25f}, {88, 0.10f}});
        set_draft_row(1, {{13, 0.50f}, {271, 0.05f}, {1061, 0.05f}, {88, 0.05f}});

        set_target_row(2, {{271, 0.35f}, {1061, 0.25f}, {33075, 0.20f}, {248068, 0.20f}});
        set_draft_row(2, {{271, 0.70f}, {1061, 0.05f}, {33075, 0.05f}, {248068, 0.05f}});

        for (int row = 0; row < row_count; ++row)
        {
            for (int token = 0; token < vocab_size; ++token)
            {
                const uint64_t offset =
                    static_cast<uint64_t>(first_logical_position + row) *
                        static_cast<uint64_t>(vocab_size) +
                    static_cast<uint64_t>(token);
                const float uniform = sampling_math::uniform01(
                    inverse_sample_seed ^ kInverseSampleDomain,
                    offset);
                inverse_rows[static_cast<size_t>(row) * vocab_size +
                             static_cast<size_t>(token)] =
                    sampling_math::inverse_exponential_from_uniform(uniform);
            }
        }

        std::array<MTPRejectionSampleRowResult, row_count> expected;
        for (int row = 0; row < row_count; ++row)
        {
            const size_t base =
                static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
            expected[static_cast<size_t>(row)] =
                sampleMTPRejectionRowFromProbabilities(
                    target_probs.data() + base,
                    draft_probs.data() + base,
                    inverse_rows.data() + base,
                    vocab_size,
                    draft_tokens[static_cast<size_t>(row)],
                    accept_thresholds[static_cast<size_t>(row)]);
            ASSERT_TRUE(expected[static_cast<size_t>(row)].ok)
                << expected[static_cast<size_t>(row)].error;
        }
        ASSERT_TRUE(expected[0].accepted);
        ASSERT_FALSE(expected[1].accepted);
        ASSERT_TRUE(expected[2].accepted);

        void *d_target = backend_->allocate(target_logits.size() * sizeof(float), device_id_);
        void *d_draft = backend_->allocate(draft_probs.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(draft_tokens.size() * sizeof(int), device_id_);
        void *d_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accept_probs = backend_->allocate(row_count * sizeof(float), device_id_);
        void *d_accept_thresholds = backend_->allocate(row_count * sizeof(float), device_id_);
        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target, d_draft, d_draft_tokens, d_tokens,
                d_accepted, d_accept_probs, d_accept_thresholds};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target, nullptr);
        ASSERT_NE(d_draft, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_tokens, nullptr);
        ASSERT_NE(d_accepted, nullptr);
        ASSERT_NE(d_accept_probs, nullptr);
        ASSERT_NE(d_accept_thresholds, nullptr);

        std::array<int, row_count> first_tokens{};
        std::array<int, row_count> first_accepted{};
        std::array<float, row_count> first_accept_probs{};
        std::array<int, row_count> second_tokens{};
        std::array<int, row_count> second_accepted{};
        std::array<float, row_count> second_accept_probs{};

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_target, target_logits.data(),
                    target_logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft, draft_probs.data(),
                    draft_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens, draft_tokens.data(),
                    draft_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft, row_count, vocab_size, vocab_size,
                        vocab_size, d_draft_tokens, accept_thresholds.data(),
                        inverse_sample_seed, first_logical_position,
                        device_id_, nullptr, d_tokens, d_accepted,
                        d_accept_probs, d_accept_thresholds))
                    << "fused processed-target verifier must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft, row_count, vocab_size, vocab_size,
                        vocab_size, d_draft_tokens, accept_thresholds.data(),
                        inverse_sample_seed, first_logical_position,
                        device_id_, stream, d_tokens, d_accepted,
                        d_accept_probs, d_accept_thresholds));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                ASSERT_TRUE(copyDeviceToHost(
                    first_tokens.data(), d_tokens,
                    first_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    first_accepted.data(), d_accepted,
                    first_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    first_accept_probs.data(), d_accept_probs,
                    first_accept_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                const std::array<int, row_count> sentinel_tokens = {-1, -1, -1};
                const std::array<int, row_count> sentinel_accepted = {-7, -7, -7};
                const std::array<float, row_count> sentinel_probs = {-1.0f, -1.0f, -1.0f};
                ASSERT_TRUE(copyHostToDevice(
                    d_tokens, sentinel_tokens.data(),
                    sentinel_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_accepted, sentinel_accepted.data(),
                    sentinel_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_accept_probs, sentinel_probs.data(),
                    sentinel_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                ASSERT_TRUE(copyDeviceToHost(
                    second_tokens.data(), d_tokens,
                    second_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    second_accepted.data(), d_accepted,
                    second_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    second_accept_probs.data(), d_accept_probs,
                    second_accept_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        cleanup();

        EXPECT_EQ(second_tokens, first_tokens)
            << "captured fused verifier must replay deterministically";
        EXPECT_EQ(second_accepted, first_accepted)
            << "captured fused verifier acceptance bits must replay deterministically";
        for (int row = 0; row < row_count; ++row)
        {
            const auto &exp = expected[static_cast<size_t>(row)];
            EXPECT_EQ(first_tokens[static_cast<size_t>(row)], exp.token)
                << "first replay row=" << row;
            EXPECT_EQ(second_tokens[static_cast<size_t>(row)], exp.token)
                << "second replay row=" << row;
            EXPECT_EQ(first_accepted[static_cast<size_t>(row)], exp.accepted ? 1 : 0)
                << "first replay row=" << row;
            EXPECT_EQ(second_accepted[static_cast<size_t>(row)], exp.accepted ? 1 : 0)
                << "second replay row=" << row;
            EXPECT_NEAR(first_accept_probs[static_cast<size_t>(row)],
                        exp.accept_probability,
                        1e-5f)
                << "first replay row=" << row;
            EXPECT_NEAR(second_accept_probs[static_cast<size_t>(row)],
                        exp.accept_probability,
                        1e-5f)
                << "second replay row=" << row;
        }
    }

    TEST_P(GPUSamplingTest, ProcessedTargetGreedyDraftSpeculativeVerifyMatchesNoDraftReferenceAndReplay)
    {
        constexpr int vocab_size = 248320;
        constexpr int row_count = 3;
        constexpr uint64_t inverse_sample_seed = 98765;
        constexpr int first_logical_position = 29;
        constexpr uint64_t kInverseSampleDomain = 0xA0761D6478BD642FULL;
        const float neg_inf = -std::numeric_limits<float>::infinity();

        const std::array<int, row_count> draft_tokens = {13, 271, 1061};
        const std::array<float, row_count> accept_thresholds = {0.55f, 0.85f, 0.24f};

        std::vector<float> target_logits(
            static_cast<size_t>(row_count) * static_cast<size_t>(vocab_size),
            neg_inf);
        std::vector<float> target_probs(target_logits.size(), 0.0f);
        std::vector<float> inverse_rows(target_logits.size(), 0.0f);

        auto set_target_row =
            [&](int row, std::initializer_list<std::pair<int, float>> entries)
        {
            const size_t base =
                static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
            float total = 0.0f;
            for (const auto &entry : entries)
                total += entry.second;
            ASSERT_NEAR(total, 1.0f, 1e-5f);
            for (const auto &entry : entries)
            {
                ASSERT_GE(entry.first, 0);
                ASSERT_LT(entry.first, vocab_size);
                target_probs[base + static_cast<size_t>(entry.first)] =
                    entry.second;
                target_logits[base + static_cast<size_t>(entry.first)] =
                    std::log(entry.second);
            }
        };

        set_target_row(0, {{13, 0.60f}, {271, 0.20f}, {1061, 0.10f}, {33075, 0.10f}});
        set_target_row(1, {{271, 0.10f}, {1061, 0.45f}, {33075, 0.35f}, {88, 0.10f}});
        set_target_row(2, {{1061, 0.25f}, {271, 0.35f}, {33075, 0.25f}, {248068, 0.15f}});

        for (int row = 0; row < row_count; ++row)
        {
            for (int token = 0; token < vocab_size; ++token)
            {
                const uint64_t offset =
                    static_cast<uint64_t>(first_logical_position + row) *
                        static_cast<uint64_t>(vocab_size) +
                    static_cast<uint64_t>(token);
                const float uniform = sampling_math::uniform01(
                    inverse_sample_seed ^ kInverseSampleDomain,
                    offset);
                inverse_rows[static_cast<size_t>(row) * vocab_size +
                             static_cast<size_t>(token)] =
                    sampling_math::inverse_exponential_from_uniform(uniform);
            }
        }

        std::array<MTPRejectionSampleRowResult, row_count> expected;
        for (int row = 0; row < row_count; ++row)
        {
            const size_t base =
                static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
            expected[static_cast<size_t>(row)] =
                sampleMTPRejectionRowFromProbabilities(
                    target_probs.data() + base,
                    /*draft_probabilities=*/nullptr,
                    inverse_rows.data() + base,
                    vocab_size,
                    draft_tokens[static_cast<size_t>(row)],
                    accept_thresholds[static_cast<size_t>(row)],
                    /*no_draft_probabilities=*/true);
            ASSERT_TRUE(expected[static_cast<size_t>(row)].ok)
                << expected[static_cast<size_t>(row)].error;
        }
        ASSERT_TRUE(expected[0].accepted);
        ASSERT_FALSE(expected[1].accepted);
        ASSERT_TRUE(expected[2].accepted);

        void *d_target = backend_->allocate(target_logits.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(draft_tokens.size() * sizeof(int), device_id_);
        void *d_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accept_probs = backend_->allocate(row_count * sizeof(float), device_id_);
        void *d_accept_thresholds = backend_->allocate(row_count * sizeof(float), device_id_);
        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target, d_draft_tokens, d_tokens, d_accepted,
                d_accept_probs, d_accept_thresholds};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_tokens, nullptr);
        ASSERT_NE(d_accepted, nullptr);
        ASSERT_NE(d_accept_probs, nullptr);
        ASSERT_NE(d_accept_thresholds, nullptr);

        std::array<int, row_count> first_tokens{};
        std::array<int, row_count> first_accepted{};
        std::array<float, row_count> first_accept_probs{};
        std::array<int, row_count> second_tokens{};
        std::array<int, row_count> second_accepted{};
        std::array<float, row_count> second_accept_probs{};

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_target, target_logits.data(),
                    target_logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens, draft_tokens.data(),
                    draft_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target,
                        /*draft_probabilities_device=*/nullptr,
                        row_count,
                        vocab_size,
                        vocab_size,
                        /*draft_row_stride=*/0,
                        d_draft_tokens,
                        accept_thresholds.data(),
                        inverse_sample_seed,
                        first_logical_position,
                        device_id_,
                        nullptr,
                        d_tokens,
                        d_accepted,
                        d_accept_probs,
                        d_accept_thresholds,
                        /*no_draft_probabilities=*/true))
                    << "processed-target no-draft verifier must reject the legacy default/null stream";
                EXPECT_FALSE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target,
                        /*draft_probabilities_device=*/nullptr,
                        row_count,
                        vocab_size,
                        vocab_size,
                        /*draft_row_stride=*/0,
                        d_draft_tokens,
                        accept_thresholds.data(),
                        inverse_sample_seed,
                        first_logical_position,
                        device_id_,
                        stream,
                        d_tokens,
                        d_accepted,
                        d_accept_probs,
                        d_accept_thresholds))
                    << "null draft probabilities require explicit no-draft mode";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target,
                        /*draft_probabilities_device=*/nullptr,
                        row_count,
                        vocab_size,
                        vocab_size,
                        /*draft_row_stride=*/0,
                        d_draft_tokens,
                        accept_thresholds.data(),
                        inverse_sample_seed,
                        first_logical_position,
                        device_id_,
                        stream,
                        d_tokens,
                        d_accepted,
                        d_accept_probs,
                        d_accept_thresholds,
                        /*no_draft_probabilities=*/true));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                ASSERT_TRUE(copyDeviceToHost(
                    first_tokens.data(), d_tokens,
                    first_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    first_accepted.data(), d_accepted,
                    first_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    first_accept_probs.data(), d_accept_probs,
                    first_accept_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                const std::array<int, row_count> sentinel_tokens = {-1, -1, -1};
                const std::array<int, row_count> sentinel_accepted = {-7, -7, -7};
                ASSERT_TRUE(copyHostToDevice(
                    d_tokens, sentinel_tokens.data(),
                    sentinel_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_accepted, sentinel_accepted.data(),
                    sentinel_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                ASSERT_TRUE(copyDeviceToHost(
                    second_tokens.data(), d_tokens,
                    second_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    second_accepted.data(), d_accepted,
                    second_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyDeviceToHost(
                    second_accept_probs.data(), d_accept_probs,
                    second_accept_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        cleanup();

        EXPECT_EQ(second_tokens, first_tokens)
            << "captured processed-target no-draft verifier must replay deterministically";
        EXPECT_EQ(second_accepted, first_accepted)
            << "captured processed-target no-draft acceptance bits must replay deterministically";
        for (int row = 0; row < row_count; ++row)
        {
            const auto &exp = expected[static_cast<size_t>(row)];
            EXPECT_EQ(first_tokens[static_cast<size_t>(row)], exp.token)
                << "first replay row=" << row;
            EXPECT_EQ(second_tokens[static_cast<size_t>(row)], exp.token)
                << "second replay row=" << row;
            EXPECT_EQ(first_accepted[static_cast<size_t>(row)], exp.accepted ? 1 : 0)
                << "first replay row=" << row;
            EXPECT_EQ(second_accepted[static_cast<size_t>(row)], exp.accepted ? 1 : 0)
                << "second replay row=" << row;
            EXPECT_NEAR(first_accept_probs[static_cast<size_t>(row)],
                        exp.accept_probability,
                        1e-5f)
                << "first replay row=" << row;
            EXPECT_NEAR(second_accept_probs[static_cast<size_t>(row)],
                        exp.accept_probability,
                        1e-5f)
                << "second replay row=" << row;
        }
    }

    TEST_P(GPUSamplingTest, FullProbabilitySpeculativeVerifyMatchesVLLMReferenceAndCaptures)
    {
        constexpr int vocab_size = 8;
        constexpr int row_count = 31;
        std::vector<int> draft_tokens(static_cast<size_t>(row_count), 0);
        std::vector<float> accept_thresholds(static_cast<size_t>(row_count));
        constexpr std::array<float, 3> threshold_pattern = {
            0.9f,
            0.9f,
            0.4f};
        for (int row = 0; row < row_count; ++row)
        {
            accept_thresholds[static_cast<size_t>(row)] =
                threshold_pattern[static_cast<size_t>(row % 3)];
        }

        auto append_row = [](std::vector<float> &rows,
                             std::initializer_list<float> row)
        {
            rows.insert(rows.end(), row.begin(), row.end());
        };

        std::vector<float> target_rows;
        std::vector<float> draft_rows;
        std::vector<float> inverse_rows;
        append_row(target_rows, {0.80f, 0.05f, 0.05f, 0.05f,
                                 0.05f, 0.0f, 0.0f, 0.0f});
        append_row(draft_rows, {0.50f, 0.10f, 0.10f, 0.10f,
                                0.20f, 0.0f, 0.0f, 0.0f});
        append_row(inverse_rows, {1.0f, 1.0f, 1.0f, 1.0f,
                                  1.0f, 1.0f, 1.0f, 1.0f});

        append_row(target_rows, {0.10f, 0.20f, 0.60f, 0.10f,
                                 0.0f, 0.0f, 0.0f, 0.0f});
        append_row(draft_rows, {0.40f, 0.10f, 0.20f, 0.30f,
                                0.0f, 0.0f, 0.0f, 0.0f});
        append_row(inverse_rows, {100.0f, 1.0f, 0.5f, 20.0f,
                                  1.0f, 1.0f, 1.0f, 1.0f});

        append_row(target_rows, {0.20f, 0.80f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 0.0f, 0.0f});
        append_row(draft_rows, {0.50f, 0.50f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f});
        append_row(inverse_rows, {1.0f, 1.0f, 1.0f, 1.0f,
                                  1.0f, 1.0f, 1.0f, 1.0f});

        const std::vector<float> target_pattern = target_rows;
        const std::vector<float> draft_pattern = draft_rows;
        const std::vector<float> inverse_pattern = inverse_rows;
        for (int row = 3; row < row_count; ++row)
        {
            const size_t pattern_offset =
                static_cast<size_t>(row % 3) * vocab_size;
            target_rows.insert(
                target_rows.end(),
                target_pattern.begin() + pattern_offset,
                target_pattern.begin() + pattern_offset + vocab_size);
            draft_rows.insert(
                draft_rows.end(),
                draft_pattern.begin() + pattern_offset,
                draft_pattern.begin() + pattern_offset + vocab_size);
            inverse_rows.insert(
                inverse_rows.end(),
                inverse_pattern.begin() + pattern_offset,
                inverse_pattern.begin() + pattern_offset + vocab_size);
        }

        std::vector<MTPRejectionSampleRowResult> expected(
            static_cast<size_t>(row_count));
        for (int row = 0; row < row_count; ++row)
        {
            expected[static_cast<size_t>(row)] =
                sampleMTPRejectionRowFromProbabilities(
                    target_rows.data() + static_cast<size_t>(row) * vocab_size,
                    draft_rows.data() + static_cast<size_t>(row) * vocab_size,
                    inverse_rows.data() + static_cast<size_t>(row) * vocab_size,
                    vocab_size,
                    draft_tokens[static_cast<size_t>(row)],
                    accept_thresholds[static_cast<size_t>(row)]);
            ASSERT_TRUE(expected[static_cast<size_t>(row)].ok)
                << expected[static_cast<size_t>(row)].error;
        }
        ASSERT_TRUE(expected[0].accepted);
        ASSERT_FALSE(expected[1].accepted);
        ASSERT_TRUE(expected[2].accepted)
            << "vLLM accepts when p/q equals the uniform threshold";

        void *d_target = backend_->allocate(target_rows.size() * sizeof(float), device_id_);
        void *d_draft = backend_->allocate(draft_rows.size() * sizeof(float), device_id_);
        void *d_inverse = backend_->allocate(inverse_rows.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(draft_tokens.size() * sizeof(int), device_id_);
        void *d_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accept_probs = backend_->allocate(row_count * sizeof(float), device_id_);
        void *d_accept_thresholds = backend_->allocate(row_count * sizeof(float), device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target, d_draft, d_inverse, d_draft_tokens, d_tokens,
                d_accepted, d_accept_probs, d_accept_thresholds};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target, nullptr);
        ASSERT_NE(d_draft, nullptr);
        ASSERT_NE(d_inverse, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_tokens, nullptr);
        ASSERT_NE(d_accepted, nullptr);
        ASSERT_NE(d_accept_probs, nullptr);
        ASSERT_NE(d_accept_thresholds, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_target, target_rows.data(),
                    target_rows.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft, draft_rows.data(),
                    draft_rows.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_inverse, inverse_rows.data(),
                    inverse_rows.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens, draft_tokens.data(),
                    draft_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(
                    backend_->enqueueSpeculativeVerifyProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft, d_inverse, row_count, vocab_size,
                        vocab_size, vocab_size, vocab_size, d_draft_tokens,
                        accept_thresholds.data(), device_id_, nullptr,
                        d_tokens, d_accepted, d_accept_probs,
                        d_accept_thresholds))
                    << "full-probability verifier must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft, d_inverse, row_count, vocab_size,
                        vocab_size, vocab_size, vocab_size, d_draft_tokens,
                        accept_thresholds.data(), device_id_, stream,
                        d_tokens, d_accepted, d_accept_probs,
                        d_accept_thresholds));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<int> tokens(static_cast<size_t>(row_count));
        std::vector<int> accepted(static_cast<size_t>(row_count));
        std::vector<float> accept_probs(static_cast<size_t>(row_count));
        ASSERT_TRUE(copyDeviceToHost(tokens.data(), d_tokens,
                                           tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(accepted.data(), d_accepted,
                                           accepted.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(accept_probs.data(), d_accept_probs,
                                           accept_probs.size() * sizeof(float), device_id_));

        cleanup();

        for (int row = 0; row < row_count; ++row)
        {
            const auto &exp = expected[static_cast<size_t>(row)];
            EXPECT_EQ(tokens[static_cast<size_t>(row)], exp.token)
                << "row " << row;
            EXPECT_EQ(accepted[static_cast<size_t>(row)], exp.accepted ? 1 : 0)
                << "row " << row;
            EXPECT_NEAR(accept_probs[static_cast<size_t>(row)],
                        exp.accept_probability,
                        1e-6f)
                << "row " << row;
        }
    }

    TEST_P(GPUSamplingTest, FullProbabilitySpeculativeVerifySupportsNoDraftProbabilitiesMode)
    {
        constexpr int vocab_size = 4;
        constexpr int row_count = 1;
        const std::array<float, vocab_size> target = {0.50f, 0.20f, 0.20f, 0.10f};
        const std::array<float, vocab_size> inverse_samples = {100.0f, 1.0f, 5.0f, 1.0f};
        const std::array<int, row_count> draft_tokens = {0};
        const std::array<float, row_count> accept_thresholds = {0.99f};

        MTPRejectionSampleRowResult expected =
            sampleMTPRejectionRowFromProbabilities(
                target.data(),
                /*draft_probabilities=*/nullptr,
                inverse_samples.data(),
                vocab_size,
                draft_tokens[0],
                accept_thresholds[0],
                /*no_draft_probabilities=*/true);
        ASSERT_TRUE(expected.ok) << expected.error;
        ASSERT_FALSE(expected.accepted);
        ASSERT_EQ(expected.token, 2);

        void *d_target = backend_->allocate(target.size() * sizeof(float), device_id_);
        void *d_inverse = backend_->allocate(inverse_samples.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(draft_tokens.size() * sizeof(int), device_id_);
        void *d_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accepted = backend_->allocate(row_count * sizeof(int), device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {d_target, d_inverse, d_draft_tokens, d_tokens, d_accepted};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target, nullptr);
        ASSERT_NE(d_inverse, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_tokens, nullptr);
        ASSERT_NE(d_accepted, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_target, target.data(), target.size() * sizeof(float),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_inverse, inverse_samples.data(),
                    inverse_samples.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens, draft_tokens.data(),
                    draft_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target,
                        /*draft_probabilities_device=*/nullptr,
                        d_inverse,
                        row_count,
                        vocab_size,
                        vocab_size,
                        /*draft_row_stride=*/0,
                        vocab_size,
                        d_draft_tokens,
                        accept_thresholds.data(),
                        device_id_,
                        stream,
                        d_tokens,
                        d_accepted,
                        /*out_accept_probability_device=*/nullptr,
                        /*out_accept_threshold_device=*/nullptr,
                        /*no_draft_probabilities=*/true));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::array<int, row_count> tokens{};
        std::array<int, row_count> accepted{};
        ASSERT_TRUE(copyDeviceToHost(tokens.data(), d_tokens,
                                           tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(accepted.data(), d_accepted,
                                           accepted.size() * sizeof(int), device_id_));
        cleanup();

        EXPECT_EQ(tokens[0], expected.token);
        EXPECT_EQ(accepted[0], 0);
    }

    TEST_P(GPUSamplingTest, ProcessedLogitVerifierSamplesBonusAndSummarizes)
    {
        constexpr int vocab_size = 16;
        constexpr int row_count = 2;
        const std::array<int, row_count + 1> request_tokens = {10, 11, 12};
        const std::array<int, row_count> draft_tokens = {11, 12};
        const std::array<float, row_count> accept_thresholds = {0.0f, 0.0f};
        const std::array<float, row_count> residual_thresholds = {0.0f, 0.0f};
        constexpr float bonus_threshold = 0.5f;

        auto logits_from_hot_token = [](int hot_token)
        {
            std::vector<float> logits(vocab_size, -std::numeric_limits<float>::infinity());
            logits[static_cast<size_t>(hot_token)] = 4.0f;
            logits[static_cast<size_t>((hot_token + 1) % vocab_size)] = 2.0f;
            return logits;
        };
        auto append_row = [](std::vector<float> &rows,
                             const std::vector<float> &row)
        {
            rows.insert(rows.end(), row.begin(), row.end());
        };

        std::vector<float> target_rows;
        std::vector<float> draft_rows;
        append_row(target_rows, logits_from_hot_token(11));
        append_row(target_rows, logits_from_hot_token(12));
        append_row(draft_rows, logits_from_hot_token(11));
        append_row(draft_rows, logits_from_hot_token(12));
        const std::vector<float> bonus_row = logits_from_hot_token(7);
        std::array<float, row_count> draft_token_probabilities{};
        for (int row = 0; row < row_count; ++row)
        {
            const float *draft_row =
                draft_rows.data() + static_cast<size_t>(row) * vocab_size;
            const MTPFullLogitRowStats stats =
                computeMTPFullLogitRowStats(draft_row, vocab_size);
            ASSERT_TRUE(stats.ok) << stats.error;
            draft_token_probabilities[static_cast<size_t>(row)] =
                probabilityFromMTPFullLogits(
                    draft_row,
                    vocab_size,
                    stats,
                    draft_tokens[static_cast<size_t>(row)]);
            ASSERT_GT(draft_token_probabilities[static_cast<size_t>(row)], 0.0f);
        }

        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens.assign(request_tokens.begin(), request_tokens.end());
        MTPRejectionBatchOutcome expected =
            summarizeAllPositionMTPRejectionBatchFromProcessedLogits(
                request,
                target_rows.data(),
                draft_rows.data(),
                row_count,
                vocab_size,
                vocab_size,
                vocab_size,
                std::vector<float>(accept_thresholds.begin(), accept_thresholds.end()),
                std::vector<float>(residual_thresholds.begin(), residual_thresholds.end()),
                bonus_row.data(),
                bonus_threshold);
        ASSERT_TRUE(expected.ok) << expected.error;
        ASSERT_TRUE(expected.all_speculative_accepted);

        void *d_target = backend_->allocate(target_rows.size() * sizeof(float), device_id_);
        void *d_draft = backend_->allocate(draft_rows.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(draft_tokens.size() * sizeof(int), device_id_);
        void *d_draft_token_probs = backend_->allocate(
            draft_token_probabilities.size() * sizeof(float),
            device_id_);
        void *d_bonus = backend_->allocate(bonus_row.size() * sizeof(float), device_id_);
        void *d_verify_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_verify_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_bonus_token = backend_->allocate(sizeof(int), device_id_);
        void *d_output_tokens = backend_->allocate(
            sampling_math::kSpeculativeBatchMaxOutputTokens * sizeof(int),
            device_id_);
        void *d_output_meta = backend_->allocate(
            sampling_math::kSpeculativeBatchMetaCount * sizeof(int),
            device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target, d_draft, d_draft_tokens, d_draft_token_probs, d_bonus,
                d_verify_tokens, d_verify_accepted, d_bonus_token,
                d_output_tokens, d_output_meta};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target, nullptr);
        ASSERT_NE(d_draft, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_draft_token_probs, nullptr);
        ASSERT_NE(d_bonus, nullptr);
        ASSERT_NE(d_verify_tokens, nullptr);
        ASSERT_NE(d_verify_accepted, nullptr);
        ASSERT_NE(d_bonus_token, nullptr);
        ASSERT_NE(d_output_tokens, nullptr);
        ASSERT_NE(d_output_meta, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_target, target_rows.data(),
                    target_rows.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft, draft_rows.data(),
                    draft_rows.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens, draft_tokens.data(),
                    draft_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_token_probs, draft_token_probabilities.data(),
                    draft_token_probabilities.size() * sizeof(float),
                    device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_bonus, bonus_row.data(),
                    bonus_row.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(
                    backend_->enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
                        d_bonus, vocab_size, vocab_size, bonus_threshold,
                        d_verify_tokens, d_verify_accepted, row_count,
                        request_tokens[0],
                        /*first_token_device=*/nullptr,
                        /*stop_tokens_host=*/nullptr,
                        /*stop_token_count=*/0,
                        device_id_, nullptr, d_bonus_token))
                    << "lazy processed-logit bonus sampler must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProcessedLogitsF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft, row_count, vocab_size, vocab_size,
                        vocab_size, d_draft_tokens, accept_thresholds.data(),
                        residual_thresholds.data(), device_id_, stream,
                        d_verify_tokens, d_verify_accepted,
                        nullptr, nullptr, d_draft_token_probs));
                ASSERT_TRUE(
                    backend_->enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
                        d_bonus, vocab_size, vocab_size, bonus_threshold,
                        d_verify_tokens, d_verify_accepted, row_count,
                        request_tokens[0],
                        /*first_token_device=*/nullptr,
                        /*stop_tokens_host=*/nullptr,
                        /*stop_token_count=*/0,
                        device_id_, stream, d_bonus_token));
                ASSERT_TRUE(backend_->enqueueSummarizeSpeculativeVerifyBatch(
                    d_verify_tokens,
                    d_verify_accepted,
                    row_count,
                    request_tokens[0],
                    /*stop_tokens_host=*/nullptr,
                    /*stop_token_count=*/0,
                    d_bonus_token,
                    /*has_bonus_token=*/true,
                    device_id_,
                    stream,
                    sampling_math::kSpeculativeBatchMaxOutputTokens,
                    d_output_tokens,
                    d_output_meta));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::array<int, sampling_math::kSpeculativeBatchMaxOutputTokens> output_tokens{};
        std::array<int, sampling_math::kSpeculativeBatchMetaCount> output_meta{};
        int bonus_token = -1;
        ASSERT_TRUE(copyDeviceToHost(
            &bonus_token, d_bonus_token, sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            output_tokens.data(), d_output_tokens,
            output_tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            output_meta.data(), d_output_meta,
            output_meta.size() * sizeof(int), device_id_));

        cleanup();

        ASSERT_EQ(output_meta[sampling_math::kSpecBatchMetaOk], 1);
        EXPECT_EQ(bonus_token, expected.ready_token);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaOutputCount],
                  static_cast<int>(expected.output_tokens.size()));
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaAcceptedSpeculativePrefix],
                  expected.accepted_speculative_prefix);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaTargetVerifierStateCommitCount],
                  expected.target_verifier_state_commit_count);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaReadyToken],
                  expected.ready_token);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaSampledTerminal],
                  expected.sampled_terminal ? 1 : 0);

        for (size_t i = 0; i < expected.output_tokens.size(); ++i)
        {
            EXPECT_EQ(output_tokens[i], expected.output_tokens[i])
                << "output token " << i;
        }
    }

    TEST_P(GPUSamplingTest,
           CapturedCommitBoundaryCountsPendingCorrectionExactlyOnce)
    {
        constexpr int row_count = 4;
        constexpr int first_token = 100;
        const std::array<int, row_count> verify_tokens = {101, 102, 103, 104};
        const std::array<int, row_count> verify_accepted = {1, 1, 1, 1};
        const int bonus_token = 105;
        const uint32_t initial_committed = 0u;
        const uint32_t initial_remaining = 2u;
        const uint32_t initial_due = 0u;
        const uint32_t initial_advanced = 0u;

        void *d_verify_tokens = backend_->allocate(
            verify_tokens.size() * sizeof(int), device_id_);
        void *d_verify_accepted = backend_->allocate(
            verify_accepted.size() * sizeof(int), device_id_);
        void *d_bonus_token = backend_->allocate(sizeof(int), device_id_);
        void *d_output_tokens = backend_->allocate(
            sampling_math::kSpeculativeBatchMaxOutputTokens * sizeof(int),
            device_id_);
        void *d_output_meta = backend_->allocate(
            sampling_math::kSpeculativeBatchMetaCount * sizeof(int),
            device_id_);
        void *d_committed = backend_->allocate(sizeof(uint32_t), device_id_);
        void *d_remaining = backend_->allocate(sizeof(uint32_t), device_id_);
        void *d_due = backend_->allocate(sizeof(uint32_t), device_id_);
        void *d_advanced = backend_->allocate(sizeof(uint32_t), device_id_);

        const std::array<void *, 9> allocations = {
            d_verify_tokens,
            d_verify_accepted,
            d_bonus_token,
            d_output_tokens,
            d_output_meta,
            d_committed,
            d_remaining,
            d_due,
            d_advanced};
        for (void *allocation : allocations)
            ASSERT_NE(allocation, nullptr);

        auto cleanup = [&]()
        {
            for (void *allocation : allocations)
            {
                if (allocation)
                    backend_->free(allocation, device_id_);
            }
        };

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_verify_tokens,
                    verify_tokens.data(),
                    verify_tokens.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_verify_accepted,
                    verify_accepted.data(),
                    verify_accepted.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_bonus_token,
                    &bonus_token,
                    sizeof(bonus_token),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_committed,
                    &initial_committed,
                    sizeof(initial_committed),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_remaining,
                    &initial_remaining,
                    sizeof(initial_remaining),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_due,
                    &initial_due,
                    sizeof(initial_due),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_advanced,
                    &initial_advanced,
                    sizeof(initial_advanced),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueSummarizeSpeculativeVerifyBatch(
                    d_verify_tokens,
                    d_verify_accepted,
                    row_count,
                    first_token,
                    /*stop_tokens_host=*/nullptr,
                    /*stop_token_count=*/0,
                    d_bonus_token,
                    /*has_bonus_token=*/true,
                    device_id_,
                    stream,
                    sampling_math::kSpeculativeBatchMaxOutputTokens,
                    d_output_tokens,
                    d_output_meta,
                    d_remaining,
                    /*leading_committed_output_count=*/1));
                ASSERT_TRUE(backend_->enqueueAdvanceSpeculativeCommitBoundary(
                    d_output_meta,
                    /*request_count=*/1,
                    sampling_math::kSpeculativeBatchMetaCount,
                    d_committed,
                    d_remaining,
                    d_due,
                    d_advanced,
                    device_id_,
                    stream));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::array<int, sampling_math::kSpeculativeBatchMaxOutputTokens>
            output_tokens{};
        std::array<int, sampling_math::kSpeculativeBatchMetaCount> output_meta{};
        uint32_t committed = 0u;
        uint32_t remaining = 99u;
        uint32_t due = 0u;
        uint32_t advanced = 0u;
        ASSERT_TRUE(copyDeviceToHost(
            output_tokens.data(),
            d_output_tokens,
            output_tokens.size() * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            output_meta.data(),
            d_output_meta,
            output_meta.size() * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &committed, d_committed, sizeof(committed), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &remaining, d_remaining, sizeof(remaining), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&due, d_due, sizeof(due), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &advanced, d_advanced, sizeof(advanced), device_id_));
        cleanup();

        ASSERT_EQ(output_meta[sampling_math::kSpecBatchMetaOk], 1);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaOutputCount], 3);
        EXPECT_EQ(
            output_meta[
                sampling_math::kSpecBatchMetaTargetVerifierStateCommitCount],
            3);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaReadyToken], 103);
        EXPECT_EQ(
            output_meta[
                sampling_math::kSpecBatchMetaCommitBoundaryClipped],
            1);
        EXPECT_EQ(
            output_meta[
                sampling_math::kSpecBatchMetaAllSpeculativeAccepted],
            0);
        EXPECT_EQ(
            output_meta[sampling_math::kSpecBatchMetaSampledTerminal],
            0);
        EXPECT_EQ(
            output_meta[
                sampling_math::kSpecBatchMetaLeadingCommittedOutputCount],
            1);
        EXPECT_EQ(output_tokens[0], first_token);
        EXPECT_EQ(output_tokens[1], 101);
        EXPECT_EQ(output_tokens[2], 102);
        EXPECT_EQ(committed, 2u);
        EXPECT_EQ(remaining, 0u);
        EXPECT_EQ(due, 1u);
        EXPECT_EQ(advanced, 1u);
    }

    TEST_P(GPUSamplingTest, LazyProcessedBonusSamplerSkipsRejectedBatchAndCaptures)
    {
        constexpr int vocab_size = 16;
        constexpr int row_count = 2;
        constexpr float bonus_threshold = 0.5f;
        const std::array<int, row_count> verify_tokens = {11, 42};
        const std::array<int, row_count> verify_accepted = {1, 0};
        const std::array<int, sampling_math::kSpeculativeBatchMaxStopTokens>
            stop_tokens = {-1, -1, -1, -1, -1, -1, -1, -1};
        const int first_token = 10;

        std::vector<float> bonus_row(vocab_size, -std::numeric_limits<float>::infinity());
        bonus_row[7] = 4.0f;
        bonus_row[8] = 2.0f;

        void *d_bonus = backend_->allocate(bonus_row.size() * sizeof(float), device_id_);
        void *d_verify_tokens = backend_->allocate(verify_tokens.size() * sizeof(int), device_id_);
        void *d_verify_accepted = backend_->allocate(verify_accepted.size() * sizeof(int), device_id_);
        void *d_bonus_token = backend_->allocate(sizeof(int), device_id_);
        void *d_output_tokens = backend_->allocate(
            sampling_math::kSpeculativeBatchMaxOutputTokens * sizeof(int),
            device_id_);
        void *d_output_meta = backend_->allocate(
            sampling_math::kSpeculativeBatchMetaCount * sizeof(int),
            device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_bonus,
                d_verify_tokens,
                d_verify_accepted,
                d_bonus_token,
                d_output_tokens,
                d_output_meta};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_bonus, nullptr);
        ASSERT_NE(d_verify_tokens, nullptr);
        ASSERT_NE(d_verify_accepted, nullptr);
        ASSERT_NE(d_bonus_token, nullptr);
        ASSERT_NE(d_output_tokens, nullptr);
        ASSERT_NE(d_output_meta, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                int stale_bonus_token = 12345;
                ASSERT_TRUE(copyHostToDevice(
                    d_bonus, bonus_row.data(),
                    bonus_row.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_verify_tokens, verify_tokens.data(),
                    verify_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_verify_accepted, verify_accepted.data(),
                    verify_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_bonus_token, &stale_bonus_token, sizeof(int),
                    device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
                        d_bonus, vocab_size, vocab_size, bonus_threshold,
                        d_verify_tokens, d_verify_accepted, row_count,
                        first_token,
                        /*first_token_device=*/nullptr,
                        stop_tokens.data(),
                        /*stop_token_count=*/0,
                        device_id_, stream, d_bonus_token));
                ASSERT_TRUE(backend_->enqueueSummarizeSpeculativeVerifyBatch(
                    d_verify_tokens,
                    d_verify_accepted,
                    row_count,
                    first_token,
                    /*stop_tokens_host=*/nullptr,
                    /*stop_token_count=*/0,
                    d_bonus_token,
                    /*has_bonus_token=*/true,
                    device_id_,
                    stream,
                    sampling_math::kSpeculativeBatchMaxOutputTokens,
                    d_output_tokens,
                    d_output_meta));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        int bonus_token = 0;
        std::array<int, sampling_math::kSpeculativeBatchMaxOutputTokens> output_tokens{};
        std::array<int, sampling_math::kSpeculativeBatchMetaCount> output_meta{};
        ASSERT_TRUE(copyDeviceToHost(
            &bonus_token, d_bonus_token, sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            output_tokens.data(), d_output_tokens,
            output_tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            output_meta.data(), d_output_meta,
            output_meta.size() * sizeof(int), device_id_));

        cleanup();

        EXPECT_EQ(bonus_token, -1);
        ASSERT_EQ(output_meta[sampling_math::kSpecBatchMetaOk], 1);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaOutputCount], 3);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaAcceptedSpeculativePrefix], 1);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaReadyToken], -1);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaSampledTerminal], 0);
        EXPECT_EQ(output_tokens[0], first_token);
        EXPECT_EQ(output_tokens[1], verify_tokens[0]);
        EXPECT_EQ(output_tokens[2], verify_tokens[1]);
    }

    TEST_P(GPUSamplingTest, CompactOneHotDraftVLLMSpeculativeVerifyMatchesReferenceAndCaptures)
    {
        constexpr int top_k = 4;
        constexpr int row_count = 2;
        constexpr int distribution_stride = top_k;

        const std::vector<ExpectedDistributionEntry> target_distribution = {
            {5, 0.20f},
            {7, 0.30f},
            {11, 0.10f},
            {13, 0.40f}};
        const std::vector<int> target_ids = {
            5, 7, 11, 13,
            5, 7, 11, 13};
        const std::vector<float> target_probs = {
            0.20f, 0.30f, 0.10f, 0.40f,
            0.20f, 0.30f, 0.10f, 0.40f};
        constexpr uint64_t inverse_sample_seed = 98765;
        constexpr int resident_base_position = 3;
        constexpr int resident_verify_position_offset = 1;
        constexpr int inverse_sample_first_logical_position =
            resident_base_position + resident_verify_position_offset;
        constexpr int inverse_sample_vocab_size = 32;
        constexpr int bonus_logical_position =
            inverse_sample_first_logical_position + row_count;
        const float bonus_threshold =
            sampling_math::mtp_spec_threshold_from_seed(
                inverse_sample_seed,
                bonus_logical_position,
                0 /* MTPSpecStochasticDrawPurpose::Sample */);
        const int draft_tokens[row_count] = {7, 7};
        const float accept_thresholds[row_count] = {
            sampling_math::mtp_spec_threshold_from_seed(
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                1 /* MTPSpecStochasticDrawPurpose::Accept */),
            sampling_math::mtp_spec_threshold_from_seed(
                inverse_sample_seed,
                inverse_sample_first_logical_position + 1,
                1 /* MTPSpecStochasticDrawPurpose::Accept */)};
        const float residual_thresholds[row_count] = {
            sampling_math::mtp_spec_threshold_from_seed(
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                2 /* MTPSpecStochasticDrawPurpose::Residual */),
            sampling_math::mtp_spec_threshold_from_seed(
                inverse_sample_seed,
                inverse_sample_first_logical_position + 1,
                2 /* MTPSpecStochasticDrawPurpose::Residual */)};

        const auto expected_accept =
            expectedSpeculativeVerifyOneHotDraftVLLMWithThresholds(
                target_distribution,
                draft_tokens[0],
                accept_thresholds[0],
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                inverse_sample_vocab_size);
        const auto expected_reject =
            expectedSpeculativeVerifyOneHotDraftVLLMWithThresholds(
                target_distribution,
                draft_tokens[1],
                accept_thresholds[1],
                inverse_sample_seed,
                inverse_sample_first_logical_position + 1,
                inverse_sample_vocab_size);
        ASSERT_EQ(expected_accept.accepted, 1);
        ASSERT_EQ(expected_accept.token_id, 7);
        ASSERT_EQ(expected_reject.accepted, 0);

        void *d_target_ids = nullptr;
        void *d_target_probs = nullptr;
        void *d_draft_tokens = nullptr;
        void *d_out_tokens = nullptr;
        void *d_out_accepted = nullptr;
        void *d_out_accept_probability = nullptr;
        void *d_out_accept_threshold = nullptr;
        void *d_seeded_out_tokens = nullptr;
        void *d_seeded_out_accepted = nullptr;
        void *d_seeded_out_accept_probability = nullptr;
        void *d_seeded_out_accept_threshold = nullptr;
        void *d_resident_base_position = nullptr;
        void *d_resident_out_tokens = nullptr;
        void *d_resident_out_accepted = nullptr;
        void *d_resident_out_accept_probability = nullptr;
        void *d_resident_out_accept_threshold = nullptr;
        void *d_seeded_bonus_token = nullptr;
        void *d_seeded_bonus_probability = nullptr;
        void *d_resident_bonus_token = nullptr;
        void *d_resident_bonus_probability = nullptr;

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target_ids,
                d_target_probs,
                d_draft_tokens,
                d_out_tokens,
                d_out_accepted,
                d_out_accept_probability,
                d_out_accept_threshold,
                d_seeded_out_tokens,
                d_seeded_out_accepted,
                d_seeded_out_accept_probability,
                d_seeded_out_accept_threshold,
                d_resident_base_position,
                d_resident_out_tokens,
                d_resident_out_accepted,
                d_resident_out_accept_probability,
                d_resident_out_accept_threshold,
                d_seeded_bonus_token,
                d_seeded_bonus_probability,
                d_resident_bonus_token,
                d_resident_bonus_probability};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        d_target_ids = backend_->allocate(target_ids.size() * sizeof(int), device_id_);
        d_target_probs = backend_->allocate(target_probs.size() * sizeof(float), device_id_);
        d_draft_tokens = backend_->allocate(sizeof(draft_tokens), device_id_);
        d_out_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        d_out_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        d_out_accept_probability = backend_->allocate(row_count * sizeof(float), device_id_);
        d_out_accept_threshold = backend_->allocate(row_count * sizeof(float), device_id_);
        d_seeded_out_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        d_seeded_out_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        d_seeded_out_accept_probability = backend_->allocate(row_count * sizeof(float), device_id_);
        d_seeded_out_accept_threshold = backend_->allocate(row_count * sizeof(float), device_id_);
        d_resident_base_position = backend_->allocate(sizeof(int), device_id_);
        d_resident_out_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        d_resident_out_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        d_resident_out_accept_probability = backend_->allocate(row_count * sizeof(float), device_id_);
        d_resident_out_accept_threshold = backend_->allocate(row_count * sizeof(float), device_id_);
        d_seeded_bonus_token = backend_->allocate(sizeof(int), device_id_);
        d_seeded_bonus_probability = backend_->allocate(sizeof(float), device_id_);
        d_resident_bonus_token = backend_->allocate(sizeof(int), device_id_);
        d_resident_bonus_probability = backend_->allocate(sizeof(float), device_id_);

        ASSERT_NE(d_target_ids, nullptr);
        ASSERT_NE(d_target_probs, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_out_tokens, nullptr);
        ASSERT_NE(d_out_accepted, nullptr);
        ASSERT_NE(d_out_accept_probability, nullptr);
        ASSERT_NE(d_out_accept_threshold, nullptr);
        ASSERT_NE(d_seeded_out_tokens, nullptr);
        ASSERT_NE(d_seeded_out_accepted, nullptr);
        ASSERT_NE(d_seeded_out_accept_probability, nullptr);
        ASSERT_NE(d_seeded_out_accept_threshold, nullptr);
        ASSERT_NE(d_resident_base_position, nullptr);
        ASSERT_NE(d_resident_out_tokens, nullptr);
        ASSERT_NE(d_resident_out_accepted, nullptr);
        ASSERT_NE(d_resident_out_accept_probability, nullptr);
        ASSERT_NE(d_resident_out_accept_threshold, nullptr);
        ASSERT_NE(d_seeded_bonus_token, nullptr);
        ASSERT_NE(d_seeded_bonus_probability, nullptr);
        ASSERT_NE(d_resident_bonus_token, nullptr);
        ASSERT_NE(d_resident_bonus_probability, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_target_ids,
                    target_ids.data(),
                    target_ids.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_target_probs,
                    target_probs.data(),
                    target_probs.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens,
                    draft_tokens,
                    sizeof(draft_tokens),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_resident_base_position,
                    &resident_base_position,
                    sizeof(resident_base_position),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                    d_target_ids,
                    d_target_probs,
                    nullptr,
                    nullptr,
                    top_k,
                    distribution_stride,
                    d_draft_tokens,
                    accept_thresholds,
                    residual_thresholds,
                    row_count,
                    device_id_,
                    nullptr,
                    d_out_tokens,
                    d_out_accepted,
                    d_out_accept_probability,
                    d_out_accept_threshold,
                    /*draft_token_probabilities_device=*/nullptr,
                    inverse_sample_seed,
                    inverse_sample_first_logical_position,
                    inverse_sample_vocab_size))
                    << "one-hot compact verifier must still reject the legacy default/null stream";
                EXPECT_FALSE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                    d_target_ids,
                    d_target_probs,
                    nullptr,
                    d_target_probs,
                    top_k,
                    distribution_stride,
                    d_draft_tokens,
                    accept_thresholds,
                    residual_thresholds,
                    row_count,
                    device_id_,
                    stream,
                    d_out_tokens,
                    d_out_accepted,
                    d_out_accept_probability,
                    d_out_accept_threshold,
                    /*draft_token_probabilities_device=*/nullptr,
                    inverse_sample_seed,
                    inverse_sample_first_logical_position,
                    inverse_sample_vocab_size))
                    << "passing only one null draft-distribution pointer is invalid";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                    d_target_ids,
                    d_target_probs,
                    nullptr,
                    nullptr,
                    top_k,
                    distribution_stride,
                    d_draft_tokens,
                    accept_thresholds,
                    residual_thresholds,
                    row_count,
                    device_id_,
                    stream,
                    d_out_tokens,
                    d_out_accepted,
                    d_out_accept_probability,
                    d_out_accept_threshold,
                    /*draft_token_probabilities_device=*/nullptr,
                    inverse_sample_seed,
                    inverse_sample_first_logical_position,
                    inverse_sample_vocab_size));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());

                auto seeded_capture = ctx.createGraphCapture(stream);
                ASSERT_NE(seeded_capture, nullptr);
                ASSERT_TRUE(seeded_capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                    d_target_ids,
                    d_target_probs,
                    nullptr,
                    nullptr,
                    top_k,
                    distribution_stride,
                    d_draft_tokens,
                    /*accept_thresholds_host=*/nullptr,
                    /*residual_thresholds_host=*/nullptr,
                    row_count,
                    device_id_,
                    stream,
                    d_seeded_out_tokens,
                    d_seeded_out_accepted,
                    d_seeded_out_accept_probability,
                    d_seeded_out_accept_threshold,
                    /*draft_token_probabilities_device=*/nullptr,
                    inverse_sample_seed,
                    inverse_sample_first_logical_position,
                    inverse_sample_vocab_size));
                ASSERT_TRUE(backend_->enqueueSampleDistributionF32Device(
                    d_target_ids,
                    d_target_probs,
                    top_k,
                    bonus_threshold,
                    device_id_,
                    stream,
                    d_seeded_bonus_token,
                    d_seeded_bonus_probability));
                ASSERT_TRUE(seeded_capture->endCapture());
                ASSERT_TRUE(seeded_capture->instantiate());
                ASSERT_TRUE(seeded_capture->launch());

                auto resident_capture = ctx.createGraphCapture(stream);
                ASSERT_NE(resident_capture, nullptr);
                ASSERT_TRUE(resident_capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                    d_target_ids,
                    d_target_probs,
                    nullptr,
                    nullptr,
                    top_k,
                    distribution_stride,
                    d_draft_tokens,
                    /*accept_thresholds_host=*/nullptr,
                    /*residual_thresholds_host=*/nullptr,
                    row_count,
                    device_id_,
                    stream,
                    d_resident_out_tokens,
                    d_resident_out_accepted,
                    d_resident_out_accept_probability,
                    d_resident_out_accept_threshold,
                    /*draft_token_probabilities_device=*/nullptr,
                    inverse_sample_seed,
                    /*inverse_sample_first_logical_position=*/-1,
                    inverse_sample_vocab_size,
                    d_resident_base_position,
                    resident_verify_position_offset));
                ASSERT_TRUE(backend_->enqueueSampleDistributionF32Device(
                    d_target_ids,
                    d_target_probs,
                    top_k,
                    /*threshold=*/0.0f,
                    device_id_,
                    stream,
                    d_resident_bonus_token,
                    d_resident_bonus_probability,
                    inverse_sample_seed,
                    d_resident_base_position,
                    resident_verify_position_offset + row_count));
                ASSERT_TRUE(resident_capture->endCapture());
                ASSERT_TRUE(resident_capture->instantiate());
                ASSERT_TRUE(resident_capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<int> out_tokens(row_count, -1);
        std::vector<int> out_accepted(row_count, -1);
        std::vector<float> out_accept_probabilities(row_count, -1.0f);
        std::vector<float> out_accept_thresholds(row_count, -1.0f);
        std::vector<int> seeded_out_tokens(row_count, -1);
        std::vector<int> seeded_out_accepted(row_count, -1);
        std::vector<float> seeded_out_accept_probabilities(row_count, -1.0f);
        std::vector<float> seeded_out_accept_thresholds(row_count, -1.0f);
        std::vector<int> resident_out_tokens(row_count, -1);
        std::vector<int> resident_out_accepted(row_count, -1);
        std::vector<float> resident_out_accept_probabilities(row_count, -1.0f);
        std::vector<float> resident_out_accept_thresholds(row_count, -1.0f);
        int seeded_bonus_token = -1;
        float seeded_bonus_probability = -1.0f;
        int resident_bonus_token = -1;
        float resident_bonus_probability = -1.0f;

        ASSERT_TRUE(copyDeviceToHost(
            out_tokens.data(),
            d_out_tokens,
            row_count * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            out_accepted.data(),
            d_out_accepted,
            row_count * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            out_accept_probabilities.data(),
            d_out_accept_probability,
            row_count * sizeof(float),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            out_accept_thresholds.data(),
            d_out_accept_threshold,
            row_count * sizeof(float),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            seeded_out_tokens.data(),
            d_seeded_out_tokens,
            row_count * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            seeded_out_accepted.data(),
            d_seeded_out_accepted,
            row_count * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            seeded_out_accept_probabilities.data(),
            d_seeded_out_accept_probability,
            row_count * sizeof(float),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            seeded_out_accept_thresholds.data(),
            d_seeded_out_accept_threshold,
            row_count * sizeof(float),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            resident_out_tokens.data(),
            d_resident_out_tokens,
            row_count * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            resident_out_accepted.data(),
            d_resident_out_accepted,
            row_count * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            resident_out_accept_probabilities.data(),
            d_resident_out_accept_probability,
            row_count * sizeof(float),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            resident_out_accept_thresholds.data(),
            d_resident_out_accept_threshold,
            row_count * sizeof(float),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &seeded_bonus_token,
            d_seeded_bonus_token,
            sizeof(seeded_bonus_token),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &seeded_bonus_probability,
            d_seeded_bonus_probability,
            sizeof(seeded_bonus_probability),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &resident_bonus_token,
            d_resident_bonus_token,
            sizeof(resident_bonus_token),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &resident_bonus_probability,
            d_resident_bonus_probability,
            sizeof(resident_bonus_probability),
            device_id_));

        cleanup();

        EXPECT_EQ(out_tokens[0], expected_accept.token_id);
        EXPECT_EQ(out_accepted[0], expected_accept.accepted);
        EXPECT_NEAR(out_accept_probabilities[0], expected_accept.accept_probability, 1e-6f);
        EXPECT_NEAR(out_accept_thresholds[0], expected_accept.accept_threshold, 1e-6f);

        EXPECT_EQ(out_tokens[1], expected_reject.token_id);
        EXPECT_EQ(out_accepted[1], expected_reject.accepted);
        EXPECT_NEAR(out_accept_probabilities[1], expected_reject.accept_probability, 1e-6f);
        EXPECT_NEAR(out_accept_thresholds[1], expected_reject.accept_threshold, 1e-6f);

        EXPECT_EQ(seeded_out_tokens, out_tokens);
        EXPECT_EQ(seeded_out_accepted, out_accepted);
        EXPECT_EQ(std::memcmp(seeded_out_accept_probabilities.data(),
                              out_accept_probabilities.data(),
                              row_count * sizeof(float)),
                  0)
            << "Seed-derived scalar positions must preserve verifier probability bytes.";
        EXPECT_EQ(std::memcmp(seeded_out_accept_thresholds.data(),
                              out_accept_thresholds.data(),
                              row_count * sizeof(float)),
                  0)
            << "Seed-derived scalar positions must preserve verifier threshold bytes.";

        EXPECT_EQ(resident_out_tokens, seeded_out_tokens);
        EXPECT_EQ(resident_out_accepted, seeded_out_accepted);
        EXPECT_EQ(std::memcmp(resident_out_accept_probabilities.data(),
                              seeded_out_accept_probabilities.data(),
                              row_count * sizeof(float)),
                  0)
            << "Resident and scalar logical positions must produce identical verifier probability bytes.";
        EXPECT_EQ(std::memcmp(resident_out_accept_thresholds.data(),
                              seeded_out_accept_thresholds.data(),
                              row_count * sizeof(float)),
                  0)
            << "Resident and scalar logical positions must produce identical verifier threshold bytes.";
        EXPECT_EQ(resident_bonus_token, seeded_bonus_token);
        EXPECT_EQ(std::memcmp(&resident_bonus_probability,
                              &seeded_bonus_probability,
                              sizeof(float)),
                  0)
            << "Resident bonus sampling must be byte-identical to the same position-keyed scalar draw.";
    }

    /**
     * @brief Prove default and above-default MTP depths use captured device rows.
     *
     * The production one-hot-draft route owns sampled draft tokens and logical
     * draw positions on the GPU. This regression captures that exact route for
     * every contiguous draft depth from one through thirty-one. The first
     * fifteen rows certify the default runtime envelope; rows sixteen through
     * thirty-one prove that envelope is not a semantic kernel maximum. Each
     * grouped result is compared in native bytes with the shared serial-row
     * rejection math, so a stale scalar threshold ABI, row alias, or
     * depth-dependent RNG offset fails on both CUDA and ROCm.
     */
    TEST_P(
        GPUSamplingTest,
        RuntimeDraftDepth1To31ResidentVerifierMatchesSerialRowsByteExactAndCaptures)
    {
        using namespace sampling_math;

        constexpr int top_k = 4;
        constexpr int max_draft_rows =
            2 * kSpeculativeBatchMaxRows + 1;
        constexpr uint64_t seed = 0x51A7D3E9u;
        constexpr int base_position = 37;
        constexpr int position_offset = 1;

        std::vector<int> target_ids(
            static_cast<size_t>(max_draft_rows) * top_k);
        std::vector<float> target_probs(
            static_cast<size_t>(max_draft_rows) * top_k);
        std::vector<int> draft_tokens(static_cast<size_t>(max_draft_rows));
        for (int row = 0; row < max_draft_rows; ++row)
        {
            const size_t base = static_cast<size_t>(row) * top_k;
            target_ids[base + 0] = row * 11 + 3;
            target_ids[base + 1] = row * 11 + 5;
            target_ids[base + 2] = row * 11 + 7;
            target_ids[base + 3] = row * 11 + 9;
            target_probs[base + 0] = 0.50f;
            target_probs[base + 1] = 0.25f;
            target_probs[base + 2] = 0.15f;
            target_probs[base + 3] = 0.10f;
            draft_tokens[static_cast<size_t>(row)] =
                target_ids[base + static_cast<size_t>(row % top_k)];
        }

        std::vector<int> expected_tokens(static_cast<size_t>(max_draft_rows));
        std::vector<int> expected_accepted(static_cast<size_t>(max_draft_rows));
        std::vector<float> expected_accept_probabilities(
            static_cast<size_t>(max_draft_rows));
        std::vector<float> expected_accept_thresholds(
            static_cast<size_t>(max_draft_rows));
        for (int row = 0; row < max_draft_rows; ++row)
        {
            const int logical_position = base_position + position_offset + row;
            speculative_verify_with_thresholds_one_hot_draft(
                target_ids.data() + static_cast<size_t>(row) * top_k,
                target_probs.data() + static_cast<size_t>(row) * top_k,
                top_k,
                draft_tokens[static_cast<size_t>(row)],
                mtp_spec_threshold_from_seed(seed, logical_position, 1),
                mtp_spec_threshold_from_seed(seed, logical_position, 2),
                expected_tokens.data() + row,
                expected_accepted.data() + row,
                expected_accept_probabilities.data() + row,
                expected_accept_thresholds.data() + row);
        }

        void *d_target_ids = backend_->allocate(
            target_ids.size() * sizeof(int), device_id_);
        void *d_target_probs = backend_->allocate(
            target_probs.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(
            draft_tokens.size() * sizeof(int), device_id_);
        void *d_base_position = backend_->allocate(sizeof(int), device_id_);
        void *d_out_tokens = backend_->allocate(
            expected_tokens.size() * sizeof(int), device_id_);
        void *d_out_accepted = backend_->allocate(
            expected_accepted.size() * sizeof(int), device_id_);
        void *d_out_accept_probabilities = backend_->allocate(
            expected_accept_probabilities.size() * sizeof(float), device_id_);
        void *d_out_accept_thresholds = backend_->allocate(
            expected_accept_thresholds.size() * sizeof(float), device_id_);
        ASSERT_NE(d_target_ids, nullptr);
        ASSERT_NE(d_target_probs, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_base_position, nullptr);
        ASSERT_NE(d_out_tokens, nullptr);
        ASSERT_NE(d_out_accepted, nullptr);
        ASSERT_NE(d_out_accept_probabilities, nullptr);
        ASSERT_NE(d_out_accept_thresholds, nullptr);

        auto cleanup = [&]
        {
            for (void *ptr : {d_out_accept_thresholds,
                              d_out_accept_probabilities,
                              d_out_accepted,
                              d_out_tokens,
                              d_base_position,
                              d_draft_tokens,
                              d_target_probs,
                              d_target_ids})
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        auto run = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_target_ids,
                    target_ids.data(),
                    target_ids.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_target_probs,
                    target_probs.data(),
                    target_probs.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens,
                    draft_tokens.data(),
                    draft_tokens.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_base_position,
                    &base_position,
                    sizeof(base_position),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                for (int row_count = 1;
                     row_count <= max_draft_rows;
                     ++row_count)
                {
                    SCOPED_TRACE("draft_rows=" + std::to_string(row_count));
                    auto capture = ctx.createGraphCapture(stream);
                    ASSERT_NE(capture, nullptr);
                    ASSERT_TRUE(capture->beginCapture());
                    ASSERT_TRUE(
                        backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                            d_target_ids,
                            d_target_probs,
                            /*draft_token_ids_device=*/nullptr,
                            /*draft_probs_device=*/nullptr,
                            top_k,
                            top_k,
                            d_draft_tokens,
                            /*accept_thresholds_host=*/nullptr,
                            /*residual_thresholds_host=*/nullptr,
                            row_count,
                            device_id_,
                            stream,
                            d_out_tokens,
                            d_out_accepted,
                            d_out_accept_probabilities,
                            d_out_accept_thresholds,
                            /*draft_token_probabilities_device=*/nullptr,
                            seed,
                            /*inverse_sample_first_logical_position=*/-1,
                            /*inverse_sample_vocab_size=*/0,
                            d_base_position,
                            position_offset));
                    ASSERT_TRUE(capture->endCapture());
                    ASSERT_TRUE(capture->instantiate());
                    ASSERT_TRUE(capture->launch());

                    std::vector<int> actual_tokens(
                        static_cast<size_t>(row_count));
                    std::vector<int> actual_accepted(
                        static_cast<size_t>(row_count));
                    std::vector<float> actual_accept_probabilities(
                        static_cast<size_t>(row_count));
                    std::vector<float> actual_accept_thresholds(
                        static_cast<size_t>(row_count));
                    ASSERT_TRUE(backend_->deviceToHostFast(
                        actual_tokens.data(),
                        d_out_tokens,
                        actual_tokens.size() * sizeof(int),
                        device_id_,
                        stream));
                    ASSERT_TRUE(backend_->deviceToHostFast(
                        actual_accepted.data(),
                        d_out_accepted,
                        actual_accepted.size() * sizeof(int),
                        device_id_,
                        stream));
                    ASSERT_TRUE(backend_->deviceToHostFast(
                        actual_accept_probabilities.data(),
                        d_out_accept_probabilities,
                        actual_accept_probabilities.size() * sizeof(float),
                        device_id_,
                        stream));
                    ASSERT_TRUE(backend_->deviceToHostFast(
                        actual_accept_thresholds.data(),
                        d_out_accept_thresholds,
                        actual_accept_thresholds.size() * sizeof(float),
                        device_id_,
                        stream));
                    ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                    EXPECT_TRUE(std::equal(
                        actual_tokens.begin(),
                        actual_tokens.end(),
                        expected_tokens.begin()));
                    EXPECT_TRUE(std::equal(
                        actual_accepted.begin(),
                        actual_accepted.end(),
                        expected_accepted.begin()));
                    EXPECT_EQ(
                        std::memcmp(
                            actual_accept_probabilities.data(),
                            expected_accept_probabilities.data(),
                            actual_accept_probabilities.size() * sizeof(float)),
                        0);
                    EXPECT_EQ(
                        std::memcmp(
                            actual_accept_thresholds.data(),
                            expected_accept_thresholds.data(),
                            actual_accept_thresholds.size() * sizeof(float)),
                        0);
                }
            });
        };

        if (GetParam() == "CUDA")
            run(GPUDeviceContextPool::instance().getNvidiaContext(device_id_));
        else
            run(GPUDeviceContextPool::instance().getAMDContext(device_id_));

        cleanup();
    }

    /**
     * @brief Prove resident logical positions for processed MTP verification.
     *
     * The production stochastic request-batch path may retain processed logits
     * instead of compact Top-K tables. In that mode both the one-hot-draft
     * verifier and the lazy bonus sampler must derive their random draws from
     * the publication mailbox without reading a host position. This regression
     * captures the scalar-position oracle and the resident-position production
     * launch independently, then requires byte equality for every published
     * token, acceptance flag, probability, threshold, and bonus sample.
     */
    TEST_P(GPUSamplingTest, MTPResidentPositionProcessedVerifierAndLazyBonusAreByteExact)
    {
        constexpr int row_count = 31;
        constexpr int vocab_size = 8;
        constexpr uint64_t seed = 0xD00DFEED12345678ull;
        constexpr int resident_base_position = 17;
        constexpr int verify_position_offset = 1;
        constexpr int first_logical_position =
            resident_base_position + verify_position_offset;
        constexpr int bonus_logical_position =
            first_logical_position + row_count;

        const std::array<float, vocab_size> target_pattern_a = {
            -2.0f, -1.0f, 2.5f, 0.0f, 1.0f, -3.0f, 0.5f, -0.5f,
        };
        const std::array<float, vocab_size> target_pattern_b = {
            -1.5f, 0.2f, -0.7f, 1.3f, 2.1f, -2.2f, 0.8f, -0.1f};
        std::vector<float> target_logits;
        target_logits.reserve(static_cast<size_t>(row_count) * vocab_size);
        for (int row = 0; row < row_count; ++row)
        {
            const auto &pattern =
                row % 2 == 0 ? target_pattern_a : target_pattern_b;
            target_logits.insert(
                target_logits.end(),
                pattern.begin(),
                pattern.end());
        }
        const std::array<float, vocab_size> bonus_logits = {
            -1.0f, 0.0f, 0.5f, 1.5f, -2.0f, 0.8f, -0.5f, 2.0f};
        std::vector<int> draft_tokens(static_cast<size_t>(row_count));
        std::vector<int> bonus_verify_tokens(static_cast<size_t>(row_count));
        std::vector<int> bonus_verify_accepted(
            static_cast<size_t>(row_count),
            1);
        std::vector<float> accept_thresholds(static_cast<size_t>(row_count));
        for (int row = 0; row < row_count; ++row)
        {
            draft_tokens[static_cast<size_t>(row)] =
                row % 2 == 0 ? 2 : 4;
            bonus_verify_tokens[static_cast<size_t>(row)] =
                draft_tokens[static_cast<size_t>(row)];
            accept_thresholds[static_cast<size_t>(row)] =
                sampling_math::mtp_spec_threshold_from_seed(
                    seed,
                    first_logical_position + row,
                    1 /* MTPSpecStochasticDrawPurpose::Accept */);
        }
        ASSERT_EQ(
            target_logits.size(),
            static_cast<size_t>(row_count) * vocab_size);
        ASSERT_EQ(draft_tokens.size(), static_cast<size_t>(row_count));
        ASSERT_EQ(accept_thresholds.size(), static_cast<size_t>(row_count));
        const float bonus_threshold =
            sampling_math::mtp_spec_threshold_from_seed(
                seed,
                bonus_logical_position,
                0 /* MTPSpecStochasticDrawPurpose::Sample */);

        void *d_target_logits = nullptr;
        void *d_bonus_logits = nullptr;
        void *d_draft_tokens = nullptr;
        void *d_bonus_verify_tokens = nullptr;
        void *d_bonus_verify_accepted = nullptr;
        void *d_resident_base_position = nullptr;
        void *d_scalar_tokens = nullptr;
        void *d_scalar_accepted = nullptr;
        void *d_scalar_accept_probabilities = nullptr;
        void *d_scalar_accept_thresholds = nullptr;
        void *d_resident_tokens = nullptr;
        void *d_resident_accepted = nullptr;
        void *d_resident_accept_probabilities = nullptr;
        void *d_resident_accept_thresholds = nullptr;
        void *d_scalar_bonus_token = nullptr;
        void *d_scalar_bonus_probability = nullptr;
        void *d_resident_bonus_token = nullptr;
        void *d_resident_bonus_probability = nullptr;

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target_logits,
                d_bonus_logits,
                d_draft_tokens,
                d_bonus_verify_tokens,
                d_bonus_verify_accepted,
                d_resident_base_position,
                d_scalar_tokens,
                d_scalar_accepted,
                d_scalar_accept_probabilities,
                d_scalar_accept_thresholds,
                d_resident_tokens,
                d_resident_accepted,
                d_resident_accept_probabilities,
                d_resident_accept_thresholds,
                d_scalar_bonus_token,
                d_scalar_bonus_probability,
                d_resident_bonus_token,
                d_resident_bonus_probability};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        d_target_logits = backend_->allocate(
            target_logits.size() * sizeof(float),
            device_id_);
        d_bonus_logits = backend_->allocate(sizeof(bonus_logits), device_id_);
        d_draft_tokens = backend_->allocate(
            draft_tokens.size() * sizeof(int),
            device_id_);
        d_bonus_verify_tokens =
            backend_->allocate(
                bonus_verify_tokens.size() * sizeof(int),
                device_id_);
        d_bonus_verify_accepted =
            backend_->allocate(
                bonus_verify_accepted.size() * sizeof(int),
                device_id_);
        d_resident_base_position = backend_->allocate(sizeof(int), device_id_);
        d_scalar_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        d_scalar_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        d_scalar_accept_probabilities =
            backend_->allocate(row_count * sizeof(float), device_id_);
        d_scalar_accept_thresholds =
            backend_->allocate(row_count * sizeof(float), device_id_);
        d_resident_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        d_resident_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        d_resident_accept_probabilities =
            backend_->allocate(row_count * sizeof(float), device_id_);
        d_resident_accept_thresholds =
            backend_->allocate(row_count * sizeof(float), device_id_);
        d_scalar_bonus_token = backend_->allocate(sizeof(int), device_id_);
        d_scalar_bonus_probability = backend_->allocate(sizeof(float), device_id_);
        d_resident_bonus_token = backend_->allocate(sizeof(int), device_id_);
        d_resident_bonus_probability = backend_->allocate(sizeof(float), device_id_);

        ASSERT_NE(d_target_logits, nullptr);
        ASSERT_NE(d_bonus_logits, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_bonus_verify_tokens, nullptr);
        ASSERT_NE(d_bonus_verify_accepted, nullptr);
        ASSERT_NE(d_resident_base_position, nullptr);
        ASSERT_NE(d_scalar_tokens, nullptr);
        ASSERT_NE(d_scalar_accepted, nullptr);
        ASSERT_NE(d_scalar_accept_probabilities, nullptr);
        ASSERT_NE(d_scalar_accept_thresholds, nullptr);
        ASSERT_NE(d_resident_tokens, nullptr);
        ASSERT_NE(d_resident_accepted, nullptr);
        ASSERT_NE(d_resident_accept_probabilities, nullptr);
        ASSERT_NE(d_resident_accept_thresholds, nullptr);
        ASSERT_NE(d_scalar_bonus_token, nullptr);
        ASSERT_NE(d_scalar_bonus_probability, nullptr);
        ASSERT_NE(d_resident_bonus_token, nullptr);
        ASSERT_NE(d_resident_bonus_probability, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(copyHostToDevice(
                    d_target_logits,
                    target_logits.data(),
                    target_logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_bonus_logits,
                    bonus_logits.data(),
                    sizeof(bonus_logits),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens,
                    draft_tokens.data(),
                    draft_tokens.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_bonus_verify_tokens,
                    bonus_verify_tokens.data(),
                    bonus_verify_tokens.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_bonus_verify_accepted,
                    bonus_verify_accepted.data(),
                    bonus_verify_accepted.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_resident_base_position,
                    &resident_base_position,
                    sizeof(resident_base_position),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto scalar_capture = ctx.createGraphCapture(stream);
                ASSERT_NE(scalar_capture, nullptr);
                ASSERT_TRUE(scalar_capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target_logits,
                        /*draft_probabilities_device=*/nullptr,
                        row_count,
                        vocab_size,
                        vocab_size,
                        vocab_size,
                        d_draft_tokens,
                        accept_thresholds.data(),
                        seed,
                        first_logical_position,
                        device_id_,
                        stream,
                        d_scalar_tokens,
                        d_scalar_accepted,
                        d_scalar_accept_probabilities,
                        d_scalar_accept_thresholds,
                        /*no_draft_probabilities=*/true));
                ASSERT_TRUE(
                    backend_->enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
                        d_bonus_logits,
                        vocab_size,
                        vocab_size,
                        bonus_threshold,
                        d_bonus_verify_tokens,
                        d_bonus_verify_accepted,
                        row_count,
                        /*first_token=*/99,
                        /*first_token_device=*/nullptr,
                        /*stop_tokens_host=*/nullptr,
                        /*stop_token_count=*/0,
                        device_id_,
                        stream,
                        d_scalar_bonus_token,
                        d_scalar_bonus_probability));
                ASSERT_TRUE(scalar_capture->endCapture());
                ASSERT_TRUE(scalar_capture->instantiate());
                ASSERT_TRUE(scalar_capture->launch());

                auto resident_capture = ctx.createGraphCapture(stream);
                ASSERT_NE(resident_capture, nullptr);
                ASSERT_TRUE(resident_capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target_logits,
                        /*draft_probabilities_device=*/nullptr,
                        row_count,
                        vocab_size,
                        vocab_size,
                        vocab_size,
                        d_draft_tokens,
                        /*accept_thresholds_host=*/nullptr,
                        seed,
                        /*inverse_sample_first_logical_position=*/-1,
                        device_id_,
                        stream,
                        d_resident_tokens,
                        d_resident_accepted,
                        d_resident_accept_probabilities,
                        d_resident_accept_thresholds,
                        /*no_draft_probabilities=*/true,
                        d_resident_base_position,
                        verify_position_offset));
                ASSERT_TRUE(
                    backend_->enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
                        d_bonus_logits,
                        vocab_size,
                        vocab_size,
                        /*threshold=*/0.0f,
                        d_bonus_verify_tokens,
                        d_bonus_verify_accepted,
                        row_count,
                        /*first_token=*/99,
                        /*first_token_device=*/nullptr,
                        /*stop_tokens_host=*/nullptr,
                        /*stop_token_count=*/0,
                        device_id_,
                        stream,
                        d_resident_bonus_token,
                        d_resident_bonus_probability,
                        seed,
                        d_resident_base_position,
                        verify_position_offset + row_count));
                ASSERT_TRUE(resident_capture->endCapture());
                ASSERT_TRUE(resident_capture->instantiate());
                ASSERT_TRUE(resident_capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<int> scalar_tokens(static_cast<size_t>(row_count));
        std::vector<int> scalar_accepted(static_cast<size_t>(row_count));
        std::vector<float> scalar_accept_probabilities(
            static_cast<size_t>(row_count));
        std::vector<float> scalar_accept_thresholds(
            static_cast<size_t>(row_count));
        std::vector<int> resident_tokens(static_cast<size_t>(row_count));
        std::vector<int> resident_accepted(static_cast<size_t>(row_count));
        std::vector<float> resident_accept_probabilities(
            static_cast<size_t>(row_count));
        std::vector<float> resident_accept_thresholds(
            static_cast<size_t>(row_count));
        int scalar_bonus_token = -1;
        float scalar_bonus_probability = -1.0f;
        int resident_bonus_token = -1;
        float resident_bonus_probability = -1.0f;

        ASSERT_TRUE(copyDeviceToHost(
            scalar_tokens.data(), d_scalar_tokens,
            scalar_tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            scalar_accepted.data(), d_scalar_accepted,
            scalar_accepted.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            scalar_accept_probabilities.data(),
            d_scalar_accept_probabilities,
            scalar_accept_probabilities.size() * sizeof(float),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            scalar_accept_thresholds.data(),
            d_scalar_accept_thresholds,
            scalar_accept_thresholds.size() * sizeof(float),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            resident_tokens.data(), d_resident_tokens,
            resident_tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            resident_accepted.data(), d_resident_accepted,
            resident_accepted.size() * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            resident_accept_probabilities.data(),
            d_resident_accept_probabilities,
            resident_accept_probabilities.size() * sizeof(float),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            resident_accept_thresholds.data(),
            d_resident_accept_thresholds,
            resident_accept_thresholds.size() * sizeof(float),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &scalar_bonus_token, d_scalar_bonus_token, sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &scalar_bonus_probability,
            d_scalar_bonus_probability,
            sizeof(float),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &resident_bonus_token, d_resident_bonus_token, sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            &resident_bonus_probability,
            d_resident_bonus_probability,
            sizeof(float),
            device_id_));

        cleanup();

        EXPECT_EQ(std::memcmp(scalar_accept_thresholds.data(),
                              accept_thresholds.data(),
                              accept_thresholds.size() * sizeof(float)),
                  0)
            << "The scalar oracle must publish the exact seeded accept thresholds.";
        EXPECT_EQ(resident_tokens, scalar_tokens);
        EXPECT_EQ(resident_accepted, scalar_accepted);
        EXPECT_EQ(std::memcmp(resident_accept_probabilities.data(),
                              scalar_accept_probabilities.data(),
                              scalar_accept_probabilities.size() * sizeof(float)),
                  0);
        EXPECT_EQ(std::memcmp(resident_accept_thresholds.data(),
                              scalar_accept_thresholds.data(),
                              scalar_accept_thresholds.size() * sizeof(float)),
                  0);
        EXPECT_EQ(resident_bonus_token, scalar_bonus_token);
        EXPECT_EQ(std::memcmp(&resident_bonus_probability,
                              &scalar_bonus_probability,
                              sizeof(float)),
                  0);
    }

    TEST_P(GPUSamplingTest, CompactOneHotDraftVLLMOutcomeSummaryMatchesReferenceAndCaptures)
    {
        constexpr int top_k = 4;
        constexpr int row_count = 2;
        constexpr int distribution_stride = top_k;
        constexpr int first_token = 101;

        const std::vector<ExpectedDistributionEntry> target_distribution = {
            {5, 0.20f},
            {7, 0.30f},
            {11, 0.10f},
            {13, 0.40f}};
        const std::vector<int> target_ids = {
            5, 7, 11, 13,
            5, 7, 11, 13};
        const std::vector<float> target_probs = {
            0.20f, 0.30f, 0.10f, 0.40f,
            0.20f, 0.30f, 0.10f, 0.40f};
        constexpr uint64_t inverse_sample_seed = 98765;
        constexpr int inverse_sample_first_logical_position = 0;
        constexpr int inverse_sample_vocab_size = 32;
        const int draft_tokens[row_count] = {7, 7};
        const float accept_thresholds[row_count] = {
            sampling_math::mtp_spec_threshold_from_seed(
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                1 /* MTPSpecStochasticDrawPurpose::Accept */),
            sampling_math::mtp_spec_threshold_from_seed(
                inverse_sample_seed,
                inverse_sample_first_logical_position + 1,
                1 /* MTPSpecStochasticDrawPurpose::Accept */)};
        const float residual_thresholds[row_count] = {
            sampling_math::mtp_spec_threshold_from_seed(
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                2 /* MTPSpecStochasticDrawPurpose::Residual */),
            sampling_math::mtp_spec_threshold_from_seed(
                inverse_sample_seed,
                inverse_sample_first_logical_position + 1,
                2 /* MTPSpecStochasticDrawPurpose::Residual */)};

        const auto expected_accept =
            expectedSpeculativeVerifyOneHotDraftVLLMWithThresholds(
                target_distribution,
                draft_tokens[0],
                accept_thresholds[0],
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                inverse_sample_vocab_size);
        const auto expected_reject =
            expectedSpeculativeVerifyOneHotDraftVLLMWithThresholds(
                target_distribution,
                draft_tokens[1],
                accept_thresholds[1],
                inverse_sample_seed,
                inverse_sample_first_logical_position + 1,
                inverse_sample_vocab_size);
        ASSERT_EQ(expected_accept.accepted, 1);
        ASSERT_EQ(expected_accept.token_id, 7);
        ASSERT_EQ(expected_reject.accepted, 0);
        ASSERT_GE(expected_reject.token_id, 0);

        void *d_target_ids = backend_->allocate(target_ids.size() * sizeof(int), device_id_);
        void *d_target_probs = backend_->allocate(target_probs.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(sizeof(draft_tokens), device_id_);
        void *d_verify_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_verify_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_output_tokens = backend_->allocate(
            sampling_math::kSpeculativeBatchMaxOutputTokens * sizeof(int),
            device_id_);
        void *d_output_meta = backend_->allocate(
            sampling_math::kSpeculativeBatchMetaCount * sizeof(int),
            device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target_ids,
                d_target_probs,
                d_draft_tokens,
                d_verify_tokens,
                d_verify_accepted,
                d_output_tokens,
                d_output_meta};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target_ids, nullptr);
        ASSERT_NE(d_target_probs, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_verify_tokens, nullptr);
        ASSERT_NE(d_verify_accepted, nullptr);
        ASSERT_NE(d_output_tokens, nullptr);
        ASSERT_NE(d_output_meta, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_target_ids,
                    target_ids.data(),
                    target_ids.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_target_probs,
                    target_probs.data(),
                    target_probs.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens,
                    draft_tokens,
                    sizeof(draft_tokens),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                    d_target_ids,
                    d_target_probs,
                    /*draft_token_ids_device=*/nullptr,
                    /*draft_probs_device=*/nullptr,
                    top_k,
                    distribution_stride,
                    d_draft_tokens,
                    accept_thresholds,
                    residual_thresholds,
                    row_count,
                    device_id_,
                    stream,
                    d_verify_tokens,
                    d_verify_accepted,
                    /*out_accept_probability_device=*/nullptr,
                    /*out_accept_threshold_device=*/nullptr,
                    /*draft_token_probabilities_device=*/nullptr,
                    inverse_sample_seed,
                    inverse_sample_first_logical_position,
                    inverse_sample_vocab_size));
                ASSERT_TRUE(backend_->enqueueSummarizeSpeculativeVerifyBatch(
                    d_verify_tokens,
                    d_verify_accepted,
                    row_count,
                    first_token,
                    /*stop_tokens_host=*/nullptr,
                    /*stop_token_count=*/0,
                    /*bonus_token_device=*/nullptr,
                    /*has_bonus_token=*/false,
                    device_id_,
                    stream,
                    sampling_math::kSpeculativeBatchMaxOutputTokens,
                    d_output_tokens,
                    d_output_meta));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::array<int, sampling_math::kSpeculativeBatchMaxOutputTokens> output_tokens{};
        std::array<int, sampling_math::kSpeculativeBatchMetaCount> output_meta{};
        ASSERT_TRUE(copyDeviceToHost(
            output_tokens.data(),
            d_output_tokens,
            output_tokens.size() * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            output_meta.data(),
            d_output_meta,
            output_meta.size() * sizeof(int),
            device_id_));

        cleanup();

        ASSERT_EQ(output_meta[sampling_math::kSpecBatchMetaOk], 1);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaOutputCount], 3);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaAcceptedSpeculativePrefix], 1);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaTargetVerifierStateCommitCount], 2);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaReadyToken], -1);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaRejectedVerifiedToken],
                  expected_reject.token_id);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaStoppedOnOutput], 0);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaAllSpeculativeAccepted], 0);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaConsumedVerifierRows], 2);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaSampledTerminal], 0);

        EXPECT_EQ(output_tokens[0], first_token);
        EXPECT_EQ(output_tokens[1], expected_accept.token_id);
        EXPECT_EQ(output_tokens[2], expected_reject.token_id);
    }

    TEST_P(GPUSamplingTest, CompactTargetSerialSamplesUseResidentPositionAndMatchSerialDecodeByteExact)
    {
        using namespace sampling_math;

        /*
         * Seeded stochastic MTP has a stricter contract than ordinary vLLM
         * rejection sampling: grouped verifier rows must emit exactly the token
         * that serial decode would sample at the same logical position.  The
         * production strict path gets there by sampling each compact target row
         * directly, then handing those sampled target tokens to the greedy
         * compact summary reducer. The draw must be derived from the resident
         * pre-verifier base position, not captured as a host float. Keep this
         * graph-capture regression close to the backend primitives so either an
         * ownership regression or a rejected-row residual sample is caught.
         */
        constexpr int top_k = 4;
        constexpr int row_count = 2;
        constexpr int distribution_stride = top_k;
        constexpr int first_token = 101;
        constexpr uint64_t sample_seed = 0x123456789ABCDEF0ull;
        constexpr int verifier_base_position = 17;

        const std::vector<ExpectedDistributionEntry> target_distribution = {
            {5, 0.20f},
            {7, 0.30f},
            {11, 0.10f},
            {13, 0.40f}};
        std::vector<int> target_ids;
        std::vector<float> target_probs;
        target_ids.reserve(static_cast<size_t>(row_count + 1) * top_k);
        target_probs.reserve(static_cast<size_t>(row_count + 1) * top_k);
        for (int row = 0; row < row_count + 1; ++row)
        {
            for (const ExpectedDistributionEntry &entry : target_distribution)
            {
                target_ids.push_back(entry.token_id);
                target_probs.push_back(entry.probability);
            }
        }

        std::array<int, row_count + 1> expected_verifier_tokens{};
        for (int row = 0; row < row_count + 1; ++row)
        {
            const float threshold = mtp_spec_threshold_from_seed(
                sample_seed,
                verifier_base_position + 1 + row,
                0 /* MTPSpecStochasticDrawPurpose::Sample */);
            expected_verifier_tokens[row] =
                expectedSampleDistributionWithThreshold(
                    target_distribution,
                    threshold);
        }
        std::array<int, row_count + 1> draft_tokens = {
            first_token,
            expected_verifier_tokens[0],
            expected_verifier_tokens[1] == 5 ? 7 : 5};
        ASSERT_EQ(expected_verifier_tokens[0], draft_tokens[1]);
        ASSERT_NE(expected_verifier_tokens[1], draft_tokens[2]);

        std::array<int, kSpeculativeBatchMaxOutputTokens> expected_tokens{};
        std::array<int, kSpeculativeBatchMetaCount> expected_meta{};
        summarize_greedy_speculative_verify_batch(
            first_token,
            expected_verifier_tokens.data(),
            draft_tokens.data(),
            row_count,
            /*stop_tokens=*/nullptr,
            /*stop_token_count=*/0,
            expected_tokens.data(),
            static_cast<int>(expected_tokens.size()),
            expected_meta.data());
        ASSERT_EQ(expected_meta[kSpecBatchMetaOk], 1);
        ASSERT_EQ(expected_meta[kSpecBatchMetaOutputCount], 3);
        ASSERT_EQ(expected_meta[kSpecBatchMetaAcceptedSpeculativePrefix], 1);
        ASSERT_EQ(expected_meta[kSpecBatchMetaRejectedVerifiedToken],
                  expected_verifier_tokens[1]);

        void *d_target_ids = backend_->allocate(
            target_ids.size() * sizeof(int),
            device_id_);
        void *d_target_probs = backend_->allocate(
            target_probs.size() * sizeof(float),
            device_id_);
        void *d_draft_tokens = backend_->allocate(
            draft_tokens.size() * sizeof(int),
            device_id_);
        void *d_verify_tokens = backend_->allocate(
            expected_verifier_tokens.size() * sizeof(int),
            device_id_);
        void *d_verifier_base_position = backend_->allocate(
            sizeof(verifier_base_position),
            device_id_);
        void *d_output_tokens = backend_->allocate(
            kSpeculativeBatchMaxOutputTokens * sizeof(int),
            device_id_);
        void *d_output_meta = backend_->allocate(
            kSpeculativeBatchMetaCount * sizeof(int),
            device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target_ids,
                d_target_probs,
                d_draft_tokens,
                d_verify_tokens,
                d_verifier_base_position,
                d_output_tokens,
                d_output_meta};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target_ids, nullptr);
        ASSERT_NE(d_target_probs, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_verify_tokens, nullptr);
        ASSERT_NE(d_verifier_base_position, nullptr);
        ASSERT_NE(d_output_tokens, nullptr);
        ASSERT_NE(d_output_meta, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_target_ids,
                    target_ids.data(),
                    target_ids.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_target_probs,
                    target_probs.data(),
                    target_probs.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_tokens,
                    draft_tokens.data(),
                    draft_tokens.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_verifier_base_position,
                    &verifier_base_position,
                    sizeof(verifier_base_position),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueSampleDistributionF32Device(
                    d_target_ids,
                    d_target_probs,
                    top_k,
                    /*threshold=*/0.0f,
                    device_id_,
                    nullptr,
                    d_verify_tokens,
                    /*out_probability_device=*/nullptr,
                    sample_seed,
                    d_verifier_base_position,
                    /*threshold_position_offset=*/1))
                    << "serial target-row sampler must reject the legacy default/null stream";
                EXPECT_FALSE(backend_->enqueueSummarizeGreedySpeculativeVerifyBatch(
                    d_verify_tokens,
                    d_draft_tokens,
                    row_count,
                    first_token,
                    /*stop_tokens_host=*/nullptr,
                    /*stop_token_count=*/0,
                    device_id_,
                    nullptr,
                    kSpeculativeBatchMaxOutputTokens,
                    d_output_tokens,
                    d_output_meta))
                    << "strict serial-equivalent summary must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                for (int row = 0; row < row_count + 1; ++row)
                {
                    ASSERT_TRUE(backend_->enqueueSampleDistributionF32Device(
                        static_cast<int *>(d_target_ids) +
                            static_cast<size_t>(row) * distribution_stride,
                        static_cast<float *>(d_target_probs) +
                            static_cast<size_t>(row) * distribution_stride,
                        top_k,
                        /*threshold=*/0.0f,
                        device_id_,
                        stream,
                        static_cast<int *>(d_verify_tokens) + row,
                        /*out_probability_device=*/nullptr,
                        sample_seed,
                        d_verifier_base_position,
                        /*threshold_position_offset=*/1 + row))
                        << "row=" << row;
                }
                ASSERT_TRUE(backend_->enqueueSummarizeGreedySpeculativeVerifyBatch(
                    d_verify_tokens,
                    d_draft_tokens,
                    row_count,
                    first_token,
                    /*stop_tokens_host=*/nullptr,
                    /*stop_token_count=*/0,
                    device_id_,
                    stream,
                    kSpeculativeBatchMaxOutputTokens,
                    d_output_tokens,
                    d_output_meta));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::array<int, row_count + 1> verify_tokens{};
        std::array<int, kSpeculativeBatchMaxOutputTokens> output_tokens{};
        std::array<int, kSpeculativeBatchMetaCount> output_meta{};
        ASSERT_TRUE(copyDeviceToHost(
            verify_tokens.data(),
            d_verify_tokens,
            verify_tokens.size() * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            output_tokens.data(),
            d_output_tokens,
            output_tokens.size() * sizeof(int),
            device_id_));
        ASSERT_TRUE(copyDeviceToHost(
            output_meta.data(),
            d_output_meta,
            output_meta.size() * sizeof(int),
            device_id_));

        cleanup();

        for (int row = 0; row < row_count + 1; ++row)
        {
            EXPECT_EQ(verify_tokens[static_cast<size_t>(row)],
                      expected_verifier_tokens[row])
                << "sampled target verifier token row=" << row;
        }
        for (int i = 0; i < kSpeculativeBatchMetaCount; ++i)
        {
            EXPECT_EQ(output_meta[static_cast<size_t>(i)],
                      expected_meta[static_cast<size_t>(i)])
                << "meta index " << i;
        }
        for (int i = 0; i < kSpeculativeBatchMaxOutputTokens; ++i)
        {
            EXPECT_EQ(output_tokens[static_cast<size_t>(i)],
                      expected_tokens[static_cast<size_t>(i)])
                << "token index " << i;
        }
    }

    /**
     * @brief Prove the fused production outcome kernel is serial-row exact.
     *
     * The production stochastic MTP lane no longer launches one sampler per
     * verifier row followed by a separate summary kernel. One warp/wavefront
     * now samples every compact target row and reduces the transaction in one
     * graph node. This test covers the complete supported MTP depth family and
     * deliberately varies Top-K geometry at the same time. Every graph is
     * captured at the physical depth-fifteen ceiling even when the logical
     * transaction is smaller. It is then replayed after changing only resident
     * state, proving that both position and logical depth come from device
     * authority instead of captured host scalars.
     *
     * Every sampled token, committed output slot, and metadata byte is compared
     * with the shared SamplingMath serial oracle. A different probability scan
     * order, an omitted bonus row, stale graph parameter, or summary-policy
     * drift therefore fails on both CUDA and ROCm.
     */
    TEST_P(
        GPUSamplingTest,
        FusedSerialEquivalentOutcomeIsByteExactForEverySupportedMTPDepth)
    {
        using namespace sampling_math;

        constexpr int kMaxComparisonRows = 15;
        constexpr int kMaxSampleRows = kMaxComparisonRows + 1;
        constexpr int kMaxTopK = 32;
        constexpr int kOutputCapacity = kSpeculativeBatchMaxOutputTokens;
        constexpr uint64_t kSeed = 0xC001D00D1234ABCDull;
        constexpr int kFirstToken = 4242;
        const std::array<int, 5> depths = {1, 2, 4, 8, 15};
        const std::array<int, 5> top_ks = {1, 3, 8, 17, 32};

        using SampleRow = std::array<int32_t, kMaxSampleRows>;
        using OutputRow = std::array<int32_t, kOutputCapacity>;
        using MetaRow = std::array<int, kSpeculativeBatchMetaCount>;
        using DiagnosticRecord =
            MTPFirstTransactionDiagnosticRecord;
        struct ExpectedOutcome
        {
            SampleRow sampled{};
            OutputRow output{};
            MetaRow meta{};
        };

        std::array<int, kMaxSampleRows * kMaxTopK> target_ids{};
        std::array<float, kMaxSampleRows * kMaxTopK> target_probs{};
        std::array<int32_t, kMaxSampleRows> verifier_input{};
        std::array<int32_t, kSpeculativeBatchMaxStopTokens> stop_tokens{};
        std::array<int, kDeviceGenerationControlCount> generation_control{};
        stop_tokens.fill(-1);

        void *d_target_ids = backend_->allocate(
            sizeof(target_ids), device_id_);
        void *d_target_probs = backend_->allocate(
            sizeof(target_probs), device_id_);
        void *d_verifier_input = backend_->allocate(
            sizeof(verifier_input), device_id_);
        void *d_stop_tokens = backend_->allocate(
            sizeof(stop_tokens), device_id_);
        void *d_generation_control = backend_->allocate(
            sizeof(generation_control), device_id_);
        void *d_base_position = backend_->allocate(sizeof(int), device_id_);
        void *d_sampled = backend_->allocate(
            sizeof(SampleRow), device_id_);
        void *d_output = backend_->allocate(
            sizeof(OutputRow), device_id_);
        void *d_meta = backend_->allocate(sizeof(MetaRow), device_id_);
        void *d_first_transaction_diagnostic = backend_->allocate(
            sizeof(DiagnosticRecord),
            device_id_);
        const std::array<void *, 10> allocations = {
            d_target_ids,
            d_target_probs,
            d_verifier_input,
            d_stop_tokens,
            d_generation_control,
            d_base_position,
            d_sampled,
            d_output,
            d_meta,
            d_first_transaction_diagnostic};
        for (void *allocation : allocations)
            ASSERT_NE(allocation, nullptr);

        auto expected_for = [&](
                                int depth,
                                int top_k,
                                int base_position) -> ExpectedOutcome
        {
            ExpectedOutcome expected;
            expected.sampled.fill(-1);
            expected.output.fill(-1);
            expected.meta.fill(0);
            for (int row = 0; row < depth + 1; ++row)
            {
                const float threshold = mtp_spec_threshold_from_seed(
                    kSeed,
                    base_position + 1 + row,
                    0 /* MTPSpecStochasticDrawPurpose::Sample */);
                expected.sampled[static_cast<size_t>(row)] =
                    sample_distribution_with_threshold(
                        target_ids.data() +
                            static_cast<size_t>(row) * kMaxTopK,
                        target_probs.data() +
                            static_cast<size_t>(row) * kMaxTopK,
                        top_k,
                        threshold);
            }
            summarize_speculative_verify_batch_at_commit_boundary(
                verifier_input[0],
                expected.sampled.data(),
                /*row_accepted=*/nullptr,
                depth,
                stop_tokens.data(),
                static_cast<int>(stop_tokens.size()),
                expected.sampled[static_cast<size_t>(depth)],
                /*has_bonus_ready_token=*/1,
                generation_control[
                    kDeviceGenerationControlTransactionCommitBudget],
                expected.output.data(),
                static_cast<int>(expected.output.size()),
                expected.meta.data(),
                verifier_input.data(),
                generation_control[
                    kDeviceGenerationControlNextLeadingCommittedOutputCount]);
            return expected;
        };

        auto run_cases = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                for (size_t case_index = 0;
                     case_index < depths.size();
                     ++case_index)
                {
                    const int depth = depths[case_index];
                    const int top_k = top_ks[case_index];
                    const int base_a = 7 + static_cast<int>(case_index) * 19;
                    const int base_b = base_a + 31;

                    target_ids.fill(-1);
                    target_probs.fill(0.0f);
                    for (int row = 0; row < depth + 1; ++row)
                    {
                        int weight_sum = 0;
                        for (int column = 0; column < top_k; ++column)
                        {
                            weight_sum +=
                                1 + ((row + 3) * (column + 5) +
                                     static_cast<int>(case_index)) %
                                        23;
                        }
                        for (int column = 0; column < top_k; ++column)
                        {
                            const size_t offset =
                                static_cast<size_t>(row) * kMaxTopK +
                                static_cast<size_t>(column);
                            target_ids[offset] =
                                10000 + row * kMaxTopK + column;
                            const int weight =
                                1 + ((row + 3) * (column + 5) +
                                     static_cast<int>(case_index)) %
                                        23;
                            target_probs[offset] =
                                static_cast<float>(weight) /
                                static_cast<float>(weight_sum);
                        }
                    }

                    verifier_input.fill(-1);
                    verifier_input[0] = kFirstToken;
                    ASSERT_TRUE(initialize_device_generation_control(
                        /*max_new_tokens=*/kOutputCapacity,
                        /*response_capacity=*/kOutputCapacity,
                        DeviceGenerationDepthPolicy::fixed(depth),
                        generation_control.data()));
                    generation_control[
                        kDeviceGenerationControlTransactionCommitBudget] =
                        depth + 1;
                    generation_control[
                        kDeviceGenerationControlNextLeadingCommittedOutputCount] =
                        0;

                    /*
                     * Construct the draft row from the first resident position.
                     * Smaller depths reject after a non-empty accepted prefix;
                     * depth fifteen exercises the all-accepted bonus path.
                     */
                    const ExpectedOutcome preliminary =
                        expected_for(depth, top_k, base_a);
                    const int accepted_prefix =
                        depth == kMaxComparisonRows ? depth : depth / 2;
                    for (int row = 0; row < depth; ++row)
                    {
                        const int sampled =
                            preliminary.sampled[static_cast<size_t>(row)];
                        verifier_input[static_cast<size_t>(row + 1)] =
                            row < accepted_prefix ? sampled : sampled + 1000000;
                    }

                    const ExpectedOutcome expected_a =
                        expected_for(depth, top_k, base_a);
                    const ExpectedOutcome expected_b =
                        expected_for(depth, top_k, base_b);

                    ASSERT_TRUE(copyHostToDevice(
                        d_target_ids,
                        target_ids.data(),
                        sizeof(target_ids),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_target_probs,
                        target_probs.data(),
                        sizeof(target_probs),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_verifier_input,
                        verifier_input.data(),
                        sizeof(verifier_input),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_stop_tokens,
                        stop_tokens.data(),
                        sizeof(stop_tokens),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_generation_control,
                        generation_control.data(),
                        sizeof(generation_control),
                        device_id_,
                        stream));
                    ASSERT_TRUE(copyHostToDevice(
                        d_base_position,
                        &base_a,
                        sizeof(base_a),
                        device_id_,
                        stream));
                    ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                    EXPECT_FALSE(
                        backend_->enqueueSampleAndSummarizeSerialEquivalentSpeculativeBatchDeviceGenerationControls(
                            d_target_ids,
                            d_target_probs,
                            kMaxTopK,
                            top_k,
                            kMaxComparisonRows,
                            kSeed,
                            d_base_position,
                            /*threshold_position_offset=*/1,
                            d_verifier_input,
                            d_stop_tokens,
                            d_generation_control,
                            device_id_,
                            /*stream=*/nullptr,
                            kOutputCapacity,
                            d_sampled,
                            d_output,
                            d_meta,
                            d_first_transaction_diagnostic))
                        << "depth=" << depth << " top_k=" << top_k;

                    auto capture = ctx.createGraphCapture(stream);
                    ASSERT_NE(capture, nullptr);
                    ASSERT_TRUE(capture->beginCapture());
                    ASSERT_TRUE(
                        backend_->enqueueSampleAndSummarizeSerialEquivalentSpeculativeBatchDeviceGenerationControls(
                            d_target_ids,
                            d_target_probs,
                            kMaxTopK,
                            top_k,
                            kMaxComparisonRows,
                            kSeed,
                            d_base_position,
                            /*threshold_position_offset=*/1,
                            d_verifier_input,
                            d_stop_tokens,
                            d_generation_control,
                            device_id_,
                            stream,
                            kOutputCapacity,
                            d_sampled,
                            d_output,
                            d_meta,
                            d_first_transaction_diagnostic));
                    ASSERT_TRUE(capture->endCapture());
                    ASSERT_TRUE(capture->instantiate());

                    SampleRow sampled_a{};
                    OutputRow output_a{};
                    MetaRow meta_a{};
                    DiagnosticRecord diagnostic_a{};
                    SampleRow sampled_b{};
                    OutputRow output_b{};
                    MetaRow meta_b{};
                    DiagnosticRecord diagnostic_b{};
                    ASSERT_TRUE(capture->launch());
                    ASSERT_TRUE(copyDeviceToHost(
                        sampled_a.data(), d_sampled, sizeof(sampled_a),
                        device_id_, stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        output_a.data(), d_output, sizeof(output_a),
                        device_id_, stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        meta_a.data(), d_meta, sizeof(meta_a),
                        device_id_, stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        &diagnostic_a,
                        d_first_transaction_diagnostic,
                        sizeof(diagnostic_a),
                        device_id_,
                        stream));

                    ASSERT_TRUE(copyHostToDevice(
                        d_base_position,
                        &base_b,
                        sizeof(base_b),
                        device_id_,
                        stream));
                    generation_control[
                        kDeviceGenerationControlTransactionCount] = 1;
                    ASSERT_TRUE(copyHostToDevice(
                        d_generation_control,
                        generation_control.data(),
                        sizeof(generation_control),
                        device_id_,
                        stream));
                    ASSERT_TRUE(capture->launch());
                    ASSERT_TRUE(copyDeviceToHost(
                        sampled_b.data(), d_sampled, sizeof(sampled_b),
                        device_id_, stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        output_b.data(), d_output, sizeof(output_b),
                        device_id_, stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        meta_b.data(), d_meta, sizeof(meta_b),
                        device_id_, stream));
                    ASSERT_TRUE(copyDeviceToHost(
                        &diagnostic_b,
                        d_first_transaction_diagnostic,
                        sizeof(diagnostic_b),
                        device_id_,
                        stream));
                    ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                    EXPECT_EQ(
                        std::memcmp(
                            sampled_a.data(),
                            expected_a.sampled.data(),
                            static_cast<size_t>(depth + 1) *
                                sizeof(sampled_a[0])),
                        0)
                        << "sample rows A depth=" << depth
                        << " top_k=" << top_k;
                    EXPECT_EQ(
                        std::memcmp(
                            output_a.data(),
                            expected_a.output.data(),
                            sizeof(output_a)),
                        0)
                        << "output A depth=" << depth
                        << " top_k=" << top_k;
                    EXPECT_EQ(
                        std::memcmp(
                            meta_a.data(),
                            expected_a.meta.data(),
                            sizeof(meta_a)),
                        0)
                        << "metadata A depth=" << depth
                        << " top_k=" << top_k;
                    EXPECT_EQ(
                        std::memcmp(
                            sampled_b.data(),
                            expected_b.sampled.data(),
                            static_cast<size_t>(depth + 1) *
                                sizeof(sampled_b[0])),
                        0)
                        << "sample rows B depth=" << depth
                        << " top_k=" << top_k;
                    EXPECT_EQ(
                        std::memcmp(
                            output_b.data(),
                            expected_b.output.data(),
                            sizeof(output_b)),
                        0)
                        << "output B depth=" << depth
                        << " top_k=" << top_k;
                    EXPECT_EQ(
                        std::memcmp(
                            meta_b.data(),
                            expected_b.meta.data(),
                            sizeof(meta_b)),
                        0)
                        << "metadata B depth=" << depth
                        << " top_k=" << top_k;

                    EXPECT_EQ(diagnostic_a.valid, 1u);
                    EXPECT_EQ(
                        diagnostic_a.version,
                        kMTPFirstTransactionDiagnosticVersion);
                    EXPECT_EQ(diagnostic_a.threshold_seed, kSeed);
                    EXPECT_EQ(
                        diagnostic_a.threshold_base_position,
                        base_a);
                    EXPECT_EQ(diagnostic_a.threshold_position_offset, 1);
                    EXPECT_EQ(diagnostic_a.comparison_row_count, depth);
                    EXPECT_EQ(diagnostic_a.top_k, top_k);
                    EXPECT_EQ(diagnostic_a.transaction_count, 0);
                    EXPECT_EQ(
                        diagnostic_a.transaction_commit_budget,
                        depth + 1);
                    EXPECT_EQ(
                        diagnostic_a.leading_committed_output_count,
                        0);
                    for (int row = 0; row < depth + 1; ++row)
                    {
                        const size_t row_index =
                            static_cast<size_t>(row);
                        const float threshold =
                            mtp_spec_threshold_from_seed(
                                kSeed,
                                base_a + 1 + row,
                                0 /* MTPSpecStochasticDrawPurpose::Sample */);
                        EXPECT_EQ(
                            diagnostic_a.verifier_input_tokens[row_index],
                            verifier_input[row_index]);
                        EXPECT_EQ(
                            diagnostic_a.sampled_target_tokens[row_index],
                            expected_a.sampled[row_index]);
                        EXPECT_EQ(
                            diagnostic_a.threshold_bits[row_index],
                            sampling_float_bits(threshold));
                        EXPECT_EQ(
                            diagnostic_a.target_distribution_hashes[row_index],
                            hash_compact_distribution_exact(
                                target_ids.data() +
                                    row_index * kMaxTopK,
                                target_probs.data() +
                                    row_index * kMaxTopK,
                                top_k));
                        if (row < depth)
                        {
                            EXPECT_EQ(
                                diagnostic_a
                                    .sampled_matches_verifier_input[row_index],
                                expected_a.sampled[row_index] ==
                                        verifier_input[row_index + 1]
                                    ? 1
                                    : 0);
                        }
                        else
                        {
                            EXPECT_EQ(
                                diagnostic_a
                                    .sampled_matches_verifier_input[row_index],
                                -1);
                        }
                    }
                    EXPECT_EQ(
                        std::memcmp(
                            diagnostic_a.output_tokens,
                            expected_a.output.data(),
                            sizeof(expected_a.output)),
                        0);
                    EXPECT_EQ(
                        std::memcmp(
                            diagnostic_a.output_meta,
                            expected_a.meta.data(),
                            sizeof(expected_a.meta)),
                        0);

                    /*
                     * Replay with a non-zero resident transaction index must
                     * update ordinary outputs while preserving transaction-zero
                     * evidence byte-for-byte. This is the exact lifetime needed
                     * by a device-controlled WHILE graph whose fatal host
                     * observation happens only after all iterations complete.
                     */
                    EXPECT_EQ(
                        std::memcmp(
                            &diagnostic_a,
                            &diagnostic_b,
                            sizeof(DiagnosticRecord)),
                        0)
                        << "first transaction diagnostic was overwritten by replay"
                        << " depth=" << depth << " top_k=" << top_k;
                }
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx =
                GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_cases(ctx);
        }
        else
        {
            auto &ctx =
                GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_cases(ctx);
        }

        for (void *allocation : allocations)
            backend_->free(allocation, device_id_);
    }

    TEST_P(GPUSamplingTest, SpeculativeVerifyDistributionsAreGraphCapturable)
    {
        const std::vector<float> target_logits = {0.1f, 3.2f, 2.0f, 1.2f,
                                                  4.5f, 0.5f, 2.6f, 3.7f};
        const std::vector<float> draft_accept_logits = {0.3f, 3.8f, 1.9f, 0.7f,
                                                        2.6f, 0.1f, 2.1f, 3.0f};
        const std::vector<float> draft_reject_logits = {0.2f, 5.2f, 1.8f, 0.4f,
                                                        2.0f, 0.3f, 2.4f, 3.3f};
        constexpr int top_k = 4;
        constexpr float top_p = 0.95f;
        constexpr float temperature = 0.7f;
        constexpr uint64_t accept_seed_accept_case = 1234;
        constexpr uint64_t accept_offset_accept_case = 7;
        constexpr uint64_t accept_seed_reject_case = 1;
        constexpr uint64_t accept_offset_reject_case = 0;
        constexpr uint64_t residual_seed = 999;
        constexpr uint64_t residual_offset = 11;
        constexpr int accept_draft_token = 7;
        constexpr int reject_draft_token = 1;

        const auto expected_target =
            expectedTopKTopPDistribution(target_logits, top_k, top_p, temperature);
        const auto expected_draft_accept =
            expectedTopKTopPDistribution(draft_accept_logits, top_k, top_p, temperature);
        const auto expected_draft_reject =
            expectedTopKTopPDistribution(draft_reject_logits, top_k, top_p, temperature);
        const auto expected_accept = expectedSpeculativeVerifyDistribution(
            expected_target,
            expected_draft_accept,
            accept_draft_token,
            accept_seed_accept_case,
            accept_offset_accept_case,
            residual_seed,
            residual_offset);
        const auto expected_reject = expectedSpeculativeVerifyDistribution(
            expected_target,
            expected_draft_reject,
            reject_draft_token,
            accept_seed_reject_case,
            accept_offset_reject_case,
            residual_seed,
            residual_offset);
        ASSERT_EQ(expected_accept.accepted, 1);
        ASSERT_EQ(expected_reject.accepted, 0);

        void *d_target_logits = nullptr;
        void *d_draft_accept_logits = nullptr;
        void *d_draft_reject_logits = nullptr;
        void *d_target_ids = nullptr;
        void *d_target_probs = nullptr;
        void *d_draft_accept_ids = nullptr;
        void *d_draft_accept_probs = nullptr;
        void *d_draft_reject_ids = nullptr;
        void *d_draft_reject_probs = nullptr;
        void *d_accept_token = nullptr;
        void *d_accept_flag = nullptr;
        void *d_accept_probability = nullptr;
        void *d_accept_threshold = nullptr;
        void *d_reject_token = nullptr;
        void *d_reject_flag = nullptr;
        void *d_reject_probability = nullptr;
        void *d_reject_threshold = nullptr;
        void *d_threshold_sample_token = nullptr;
        void *d_threshold_verify_token = nullptr;
        void *d_threshold_verify_flag = nullptr;
        void *d_threshold_verify_probability = nullptr;
        void *d_threshold_verify_threshold = nullptr;
        void *d_batch_target_ids = nullptr;
        void *d_batch_target_probs = nullptr;
        void *d_batch_draft_ids = nullptr;
        void *d_batch_draft_probs = nullptr;
        void *d_batch_verify_tokens = nullptr;
        void *d_batch_accept_flags = nullptr;
        void *d_batch_accept_probabilities = nullptr;
        void *d_batch_accept_thresholds = nullptr;
        void *d_batch_sampled_draft_tokens = nullptr;
        void *d_batch_sampled_draft_probabilities = nullptr;
        void *d_batch_device_token_verify_tokens = nullptr;
        void *d_batch_device_token_accept_flags = nullptr;
        void *d_batch_device_token_accept_probabilities = nullptr;
        void *d_batch_device_token_accept_thresholds = nullptr;

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target_logits,
                d_draft_accept_logits,
                d_draft_reject_logits,
                d_target_ids,
                d_target_probs,
                d_draft_accept_ids,
                d_draft_accept_probs,
                d_draft_reject_ids,
                d_draft_reject_probs,
                d_accept_token,
                d_accept_flag,
                d_accept_probability,
                d_accept_threshold,
                d_reject_token,
                d_reject_flag,
                d_reject_probability,
                d_reject_threshold,
                d_threshold_sample_token,
                d_threshold_verify_token,
                d_threshold_verify_flag,
                d_threshold_verify_probability,
                d_threshold_verify_threshold,
                d_batch_target_ids,
                d_batch_target_probs,
                d_batch_draft_ids,
                d_batch_draft_probs,
                d_batch_verify_tokens,
                d_batch_accept_flags,
                d_batch_accept_probabilities,
                d_batch_accept_thresholds,
                d_batch_sampled_draft_tokens,
                d_batch_sampled_draft_probabilities,
                d_batch_device_token_verify_tokens,
                d_batch_device_token_accept_flags,
                d_batch_device_token_accept_probabilities,
                d_batch_device_token_accept_thresholds};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        d_target_logits = backend_->allocate(target_logits.size() * sizeof(float), device_id_);
        d_draft_accept_logits = backend_->allocate(draft_accept_logits.size() * sizeof(float), device_id_);
        d_draft_reject_logits = backend_->allocate(draft_reject_logits.size() * sizeof(float), device_id_);
        d_target_ids = backend_->allocate(top_k * sizeof(int), device_id_);
        d_target_probs = backend_->allocate(top_k * sizeof(float), device_id_);
        d_draft_accept_ids = backend_->allocate(top_k * sizeof(int), device_id_);
        d_draft_accept_probs = backend_->allocate(top_k * sizeof(float), device_id_);
        d_draft_reject_ids = backend_->allocate(top_k * sizeof(int), device_id_);
        d_draft_reject_probs = backend_->allocate(top_k * sizeof(float), device_id_);
        d_accept_token = backend_->allocate(sizeof(int), device_id_);
        d_accept_flag = backend_->allocate(sizeof(int), device_id_);
        d_accept_probability = backend_->allocate(sizeof(float), device_id_);
        d_accept_threshold = backend_->allocate(sizeof(float), device_id_);
        d_reject_token = backend_->allocate(sizeof(int), device_id_);
        d_reject_flag = backend_->allocate(sizeof(int), device_id_);
        d_reject_probability = backend_->allocate(sizeof(float), device_id_);
        d_reject_threshold = backend_->allocate(sizeof(float), device_id_);
        d_threshold_sample_token = backend_->allocate(sizeof(int), device_id_);
        d_threshold_verify_token = backend_->allocate(sizeof(int), device_id_);
        d_threshold_verify_flag = backend_->allocate(sizeof(int), device_id_);
        d_threshold_verify_probability = backend_->allocate(sizeof(float), device_id_);
        d_threshold_verify_threshold = backend_->allocate(sizeof(float), device_id_);
        d_batch_target_ids = backend_->allocate(2 * top_k * sizeof(int), device_id_);
        d_batch_target_probs = backend_->allocate(2 * top_k * sizeof(float), device_id_);
        d_batch_draft_ids = backend_->allocate(2 * top_k * sizeof(int), device_id_);
        d_batch_draft_probs = backend_->allocate(2 * top_k * sizeof(float), device_id_);
        d_batch_verify_tokens = backend_->allocate(2 * sizeof(int), device_id_);
        d_batch_accept_flags = backend_->allocate(2 * sizeof(int), device_id_);
        d_batch_accept_probabilities = backend_->allocate(2 * sizeof(float), device_id_);
        d_batch_accept_thresholds = backend_->allocate(2 * sizeof(float), device_id_);
        d_batch_sampled_draft_tokens = backend_->allocate(2 * sizeof(int), device_id_);
        d_batch_sampled_draft_probabilities = backend_->allocate(2 * sizeof(float), device_id_);
        d_batch_device_token_verify_tokens = backend_->allocate(2 * sizeof(int), device_id_);
        d_batch_device_token_accept_flags = backend_->allocate(2 * sizeof(int), device_id_);
        d_batch_device_token_accept_probabilities = backend_->allocate(2 * sizeof(float), device_id_);
        d_batch_device_token_accept_thresholds = backend_->allocate(2 * sizeof(float), device_id_);

        ASSERT_NE(d_target_logits, nullptr);
        ASSERT_NE(d_draft_accept_logits, nullptr);
        ASSERT_NE(d_draft_reject_logits, nullptr);
        ASSERT_NE(d_target_ids, nullptr);
        ASSERT_NE(d_target_probs, nullptr);
        ASSERT_NE(d_draft_accept_ids, nullptr);
        ASSERT_NE(d_draft_accept_probs, nullptr);
        ASSERT_NE(d_draft_reject_ids, nullptr);
        ASSERT_NE(d_draft_reject_probs, nullptr);
        ASSERT_NE(d_accept_token, nullptr);
        ASSERT_NE(d_accept_flag, nullptr);
        ASSERT_NE(d_accept_probability, nullptr);
        ASSERT_NE(d_accept_threshold, nullptr);
        ASSERT_NE(d_reject_token, nullptr);
        ASSERT_NE(d_reject_flag, nullptr);
        ASSERT_NE(d_reject_probability, nullptr);
        ASSERT_NE(d_reject_threshold, nullptr);
        ASSERT_NE(d_threshold_sample_token, nullptr);
        ASSERT_NE(d_threshold_verify_token, nullptr);
        ASSERT_NE(d_threshold_verify_flag, nullptr);
        ASSERT_NE(d_threshold_verify_probability, nullptr);
        ASSERT_NE(d_threshold_verify_threshold, nullptr);
        ASSERT_NE(d_batch_target_ids, nullptr);
        ASSERT_NE(d_batch_target_probs, nullptr);
        ASSERT_NE(d_batch_draft_ids, nullptr);
        ASSERT_NE(d_batch_draft_probs, nullptr);
        ASSERT_NE(d_batch_verify_tokens, nullptr);
        ASSERT_NE(d_batch_accept_flags, nullptr);
        ASSERT_NE(d_batch_accept_probabilities, nullptr);
        ASSERT_NE(d_batch_accept_thresholds, nullptr);
        ASSERT_NE(d_batch_sampled_draft_tokens, nullptr);
        ASSERT_NE(d_batch_sampled_draft_probabilities, nullptr);
        ASSERT_NE(d_batch_device_token_verify_tokens, nullptr);
        ASSERT_NE(d_batch_device_token_accept_flags, nullptr);
        ASSERT_NE(d_batch_device_token_accept_probabilities, nullptr);
        ASSERT_NE(d_batch_device_token_accept_thresholds, nullptr);

        const int batch_draft_tokens[2] = {accept_draft_token, reject_draft_token};
        const float batch_accept_thresholds[2] = {0.0f, 0.99f};
        const float batch_residual_thresholds[2] = {0.0f, 0.0f};
        const float batch_draft_token_probabilities[2] = {
            distributionProbability(expected_draft_accept, accept_draft_token) * 2.0f,
            distributionProbability(expected_draft_reject, reject_draft_token)};
        const auto expected_batch_accept =
            expectedSpeculativeVerifyDistributionWithThresholds(
                expected_target,
                expected_draft_accept,
                accept_draft_token,
                batch_accept_thresholds[0],
                batch_residual_thresholds[0]);
        const auto expected_batch_reject =
            expectedSpeculativeVerifyDistributionWithThresholds(
                expected_target,
                expected_draft_reject,
                reject_draft_token,
                batch_accept_thresholds[1],
                batch_residual_thresholds[1]);
        ASSERT_EQ(expected_batch_accept.accepted, 1);
        ASSERT_EQ(expected_batch_reject.accepted, 0);
        const float expected_device_token_accept_probability =
            std::min(
                1.0f,
                distributionProbability(expected_target, accept_draft_token) /
                    batch_draft_token_probabilities[0]);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(copyHostToDevice(
                    d_target_logits,
                    target_logits.data(),
                    target_logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_accept_logits,
                    draft_accept_logits.data(),
                    draft_accept_logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_draft_reject_logits,
                    draft_reject_logits.data(),
                    draft_reject_logits.size() * sizeof(float),
                    device_id_,
                    stream));
                // Device-token verifier regression setup: sampled MTP draft
                // tokens must already live in device scratch before capture.
                ASSERT_TRUE(copyHostToDevice(
                    d_batch_sampled_draft_tokens,
                    batch_draft_tokens,
                    sizeof(batch_draft_tokens),
                    device_id_,
                    stream));
                ASSERT_TRUE(copyHostToDevice(
                    d_batch_sampled_draft_probabilities,
                    batch_draft_token_probabilities,
                    sizeof(batch_draft_token_probabilities),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_target_logits,
                    static_cast<int>(target_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    nullptr,
                    d_target_ids,
                    d_target_probs))
                    << "distribution builder must reject the legacy default/null stream";
                EXPECT_FALSE(backend_->enqueueSpeculativeVerifyDistributionsF32Device(
                    d_target_ids,
                    d_target_probs,
                    d_draft_accept_ids,
                    d_draft_accept_probs,
                    top_k,
                    accept_draft_token,
                    accept_seed_accept_case,
                    accept_offset_accept_case,
                    residual_seed,
                    residual_offset,
                    device_id_,
                    nullptr,
                    d_accept_token,
                    d_accept_flag,
                    d_accept_probability,
                    d_accept_threshold))
                    << "speculative verifier must reject the legacy default/null stream";
                EXPECT_FALSE(backend_->enqueueSampleDistributionF32Device(
                    d_target_ids,
                    d_target_probs,
                    top_k,
                    0.25f,
                    device_id_,
                    nullptr,
                    d_threshold_sample_token))
                    << "compact distribution sampler must reject the legacy default/null stream";
                EXPECT_FALSE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholds(
                    d_target_ids,
                    d_target_probs,
                    d_draft_reject_ids,
                    d_draft_reject_probs,
                    top_k,
                    reject_draft_token,
                    0.99f,
                    0.0f,
                    device_id_,
                    nullptr,
                    d_threshold_verify_token,
                    d_threshold_verify_flag,
                    d_threshold_verify_probability,
                    d_threshold_verify_threshold))
                    << "threshold verifier must reject the legacy default/null stream";
                EXPECT_FALSE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatch(
                    d_batch_target_ids,
                    d_batch_target_probs,
                    d_batch_draft_ids,
                    d_batch_draft_probs,
                    top_k,
                    top_k,
                    batch_draft_tokens,
                    batch_accept_thresholds,
                    batch_residual_thresholds,
                    2,
                    device_id_,
                    nullptr,
                    d_batch_verify_tokens,
                    d_batch_accept_flags,
                    d_batch_accept_probabilities,
                    d_batch_accept_thresholds))
                    << "batched speculative verifier must reject the legacy default/null stream";
                EXPECT_FALSE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                    d_batch_target_ids,
                    d_batch_target_probs,
                    d_batch_draft_ids,
                    d_batch_draft_probs,
                    top_k,
                    top_k,
                    d_batch_sampled_draft_tokens,
                    batch_accept_thresholds,
                    batch_residual_thresholds,
                    2,
                    device_id_,
                    nullptr,
                    d_batch_device_token_verify_tokens,
                    d_batch_device_token_accept_flags,
                    d_batch_device_token_accept_probabilities,
                    d_batch_device_token_accept_thresholds,
                    d_batch_sampled_draft_probabilities))
                    << "device-token batched verifier must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_target_logits,
                    static_cast<int>(target_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_target_ids,
                    d_target_probs));
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_draft_accept_logits,
                    static_cast<int>(draft_accept_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_draft_accept_ids,
                    d_draft_accept_probs));
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_draft_reject_logits,
                    static_cast<int>(draft_reject_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_draft_reject_ids,
                    d_draft_reject_probs));
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32Device(
                    d_target_ids,
                    d_target_probs,
                    d_draft_accept_ids,
                    d_draft_accept_probs,
                    top_k,
                    accept_draft_token,
                    accept_seed_accept_case,
                    accept_offset_accept_case,
                    residual_seed,
                    residual_offset,
                    device_id_,
                    stream,
                    d_accept_token,
                    d_accept_flag,
                    d_accept_probability,
                    d_accept_threshold));
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32Device(
                    d_target_ids,
                    d_target_probs,
                    d_draft_reject_ids,
                    d_draft_reject_probs,
                    top_k,
                    reject_draft_token,
                    accept_seed_reject_case,
                    accept_offset_reject_case,
                    residual_seed,
                    residual_offset,
                    device_id_,
                    stream,
                    d_reject_token,
                    d_reject_flag,
                    d_reject_probability,
                    d_reject_threshold));
                ASSERT_TRUE(backend_->enqueueSampleDistributionF32Device(
                    d_target_ids,
                    d_target_probs,
                    top_k,
                    0.25f,
                    device_id_,
                    stream,
                    d_threshold_sample_token));
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholds(
                    d_target_ids,
                    d_target_probs,
                    d_draft_reject_ids,
                    d_draft_reject_probs,
                    top_k,
                    reject_draft_token,
                    0.99f,
                    0.0f,
                    device_id_,
                    stream,
                    d_threshold_verify_token,
                    d_threshold_verify_flag,
                    d_threshold_verify_probability,
                    d_threshold_verify_threshold));
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_target_logits,
                    static_cast<int>(target_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    static_cast<int *>(d_batch_target_ids),
                    static_cast<float *>(d_batch_target_probs)));
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_target_logits,
                    static_cast<int>(target_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    static_cast<int *>(d_batch_target_ids) + top_k,
                    static_cast<float *>(d_batch_target_probs) + top_k));
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_draft_accept_logits,
                    static_cast<int>(draft_accept_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    static_cast<int *>(d_batch_draft_ids),
                    static_cast<float *>(d_batch_draft_probs)));
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_draft_reject_logits,
                    static_cast<int>(draft_reject_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    static_cast<int *>(d_batch_draft_ids) + top_k,
                    static_cast<float *>(d_batch_draft_probs) + top_k));
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatch(
                    d_batch_target_ids,
                    d_batch_target_probs,
                    d_batch_draft_ids,
                    d_batch_draft_probs,
                    top_k,
                    top_k,
                    batch_draft_tokens,
                    batch_accept_thresholds,
                    batch_residual_thresholds,
                    2,
                    device_id_,
                    stream,
                    d_batch_verify_tokens,
                    d_batch_accept_flags,
                    d_batch_accept_probabilities,
                    d_batch_accept_thresholds));
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                    d_batch_target_ids,
                    d_batch_target_probs,
                    d_batch_draft_ids,
                    d_batch_draft_probs,
                    top_k,
                    top_k,
                    d_batch_sampled_draft_tokens,
                    batch_accept_thresholds,
                    batch_residual_thresholds,
                    2,
                    device_id_,
                    stream,
                    d_batch_device_token_verify_tokens,
                    d_batch_device_token_accept_flags,
                    d_batch_device_token_accept_probabilities,
                    d_batch_device_token_accept_thresholds,
                    d_batch_sampled_draft_probabilities));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<int> target_ids(top_k, -1);
        std::vector<float> target_probs(top_k, 0.0f);
        int accept_token = -1;
        int accept_flag = -1;
        float accept_probability = -1.0f;
        float accept_threshold = -1.0f;
        int reject_token = -1;
        int reject_flag = -1;
        float reject_probability = -1.0f;
        float reject_threshold = -1.0f;
        int threshold_sample_token = -1;
        int threshold_verify_token = -1;
        int threshold_verify_flag = -1;
        float threshold_verify_probability = -1.0f;
        float threshold_verify_threshold = -1.0f;
        std::vector<int> batch_verify_tokens(2, -1);
        std::vector<int> batch_accept_flags(2, -1);
        std::vector<float> batch_accept_probabilities(2, -1.0f);
        std::vector<float> batch_accept_threshold_results(2, -1.0f);
        std::vector<int> batch_device_token_verify_tokens(2, -1);
        std::vector<int> batch_device_token_accept_flags(2, -1);
        std::vector<float> batch_device_token_accept_probabilities(2, -1.0f);
        std::vector<float> batch_device_token_accept_threshold_results(2, -1.0f);

        ASSERT_TRUE(copyDeviceToHost(target_ids.data(), d_target_ids, top_k * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(target_probs.data(), d_target_probs, top_k * sizeof(float), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&accept_token, d_accept_token, sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&accept_flag, d_accept_flag, sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&accept_probability, d_accept_probability, sizeof(float), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&accept_threshold, d_accept_threshold, sizeof(float), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&reject_token, d_reject_token, sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&reject_flag, d_reject_flag, sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&reject_probability, d_reject_probability, sizeof(float), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&reject_threshold, d_reject_threshold, sizeof(float), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&threshold_sample_token, d_threshold_sample_token, sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&threshold_verify_token, d_threshold_verify_token, sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&threshold_verify_flag, d_threshold_verify_flag, sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&threshold_verify_probability, d_threshold_verify_probability, sizeof(float), device_id_));
        ASSERT_TRUE(copyDeviceToHost(&threshold_verify_threshold, d_threshold_verify_threshold, sizeof(float), device_id_));
        ASSERT_TRUE(copyDeviceToHost(batch_verify_tokens.data(), d_batch_verify_tokens, 2 * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(batch_accept_flags.data(), d_batch_accept_flags, 2 * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(batch_accept_probabilities.data(), d_batch_accept_probabilities, 2 * sizeof(float), device_id_));
        ASSERT_TRUE(copyDeviceToHost(batch_accept_threshold_results.data(), d_batch_accept_thresholds, 2 * sizeof(float), device_id_));
        ASSERT_TRUE(copyDeviceToHost(batch_device_token_verify_tokens.data(), d_batch_device_token_verify_tokens, 2 * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(batch_device_token_accept_flags.data(), d_batch_device_token_accept_flags, 2 * sizeof(int), device_id_));
        ASSERT_TRUE(copyDeviceToHost(batch_device_token_accept_probabilities.data(), d_batch_device_token_accept_probabilities, 2 * sizeof(float), device_id_));
        ASSERT_TRUE(copyDeviceToHost(batch_device_token_accept_threshold_results.data(), d_batch_device_token_accept_thresholds, 2 * sizeof(float), device_id_));

        cleanup();

        for (int i = 0; i < top_k; ++i)
        {
            EXPECT_EQ(target_ids[static_cast<size_t>(i)],
                      expected_target[static_cast<size_t>(i)].token_id)
                << "target compact distribution token mismatch at slot " << i;
            EXPECT_NEAR(target_probs[static_cast<size_t>(i)],
                        expected_target[static_cast<size_t>(i)].probability,
                        1e-5f)
                << "target compact distribution probability mismatch at slot " << i;
        }

        EXPECT_EQ(accept_flag, expected_accept.accepted);
        EXPECT_EQ(accept_token, expected_accept.token_id);
        EXPECT_NEAR(accept_probability, expected_accept.accept_probability, 1e-5f);
        EXPECT_NEAR(accept_threshold, expected_accept.accept_threshold, 1e-6f);

        EXPECT_EQ(reject_flag, expected_reject.accepted);
        EXPECT_EQ(reject_token, expected_reject.token_id);
        EXPECT_NEAR(reject_probability, expected_reject.accept_probability, 1e-5f);
        EXPECT_NEAR(reject_threshold, expected_reject.accept_threshold, 1e-6f);

        const int expected_threshold_sample =
            expectedSampleDistributionWithThreshold(expected_target, 0.25f);
        const auto expected_threshold_verify =
            expectedSpeculativeVerifyDistributionWithThresholds(
                expected_target,
                expected_draft_reject,
                reject_draft_token,
                0.99f,
                0.0f);
        EXPECT_EQ(threshold_sample_token, expected_threshold_sample);
        EXPECT_EQ(threshold_verify_flag, expected_threshold_verify.accepted);
        EXPECT_EQ(threshold_verify_token, expected_threshold_verify.token_id);
        EXPECT_NEAR(threshold_verify_probability, expected_threshold_verify.accept_probability, 1e-5f);
        EXPECT_NEAR(threshold_verify_threshold, expected_threshold_verify.accept_threshold, 1e-6f);

        EXPECT_EQ(batch_verify_tokens[0], expected_batch_accept.token_id);
        EXPECT_EQ(batch_accept_flags[0], expected_batch_accept.accepted);
        EXPECT_NEAR(batch_accept_probabilities[0], expected_batch_accept.accept_probability, 1e-5f);
        EXPECT_NEAR(batch_accept_threshold_results[0], expected_batch_accept.accept_threshold, 1e-6f);
        EXPECT_EQ(batch_device_token_verify_tokens[0], expected_batch_accept.token_id);
        EXPECT_EQ(batch_device_token_accept_flags[0], expected_batch_accept.accepted);
        EXPECT_NEAR(batch_device_token_accept_probabilities[0], expected_device_token_accept_probability, 1e-5f);
        EXPECT_NEAR(batch_device_token_accept_threshold_results[0], expected_batch_accept.accept_threshold, 1e-6f);

        EXPECT_EQ(batch_verify_tokens[1], expected_batch_reject.token_id);
        EXPECT_EQ(batch_accept_flags[1], expected_batch_reject.accepted);
        EXPECT_NEAR(batch_accept_probabilities[1], expected_batch_reject.accept_probability, 1e-5f);
        EXPECT_NEAR(batch_accept_threshold_results[1], expected_batch_reject.accept_threshold, 1e-6f);
        EXPECT_EQ(batch_device_token_verify_tokens[1], expected_batch_reject.token_id);
        EXPECT_EQ(batch_device_token_accept_flags[1], expected_batch_reject.accepted);
        EXPECT_NEAR(batch_device_token_accept_probabilities[1], expected_batch_reject.accept_probability, 1e-5f);
        EXPECT_NEAR(batch_device_token_accept_threshold_results[1], expected_batch_reject.accept_threshold, 1e-6f);

        EXPECT_EQ(batch_device_token_verify_tokens, batch_verify_tokens)
            << "device-token batch verifier must match host-token verifier tokens";
        EXPECT_EQ(batch_device_token_accept_flags, batch_accept_flags)
            << "device-token batch verifier must match host-token verifier accept flags";
    }

    TEST_P(GPUSamplingTest, DRYParity_SimpleRepeat)
    {
        // History: [A, B, C, A, B, C] — "A B C" repeated
        // DRY should penalize token A (extending the repeat)
        const int vocab_size = 100;
        const int A = 10, B = 20, C = 30;

        Sampler sampler(42);
        for (int token : {A, B, C, A, B, C})
            sampler.record_token(token);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_base = 1.75f;
        params.dry_allowed_length = 1;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);
        ASSERT_FALSE(penalties.empty()) << "DRY should produce penalties for repeated pattern";

        // Identical logits for CPU and GPU
        std::vector<float> logits(vocab_size, 5.0f);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(
            backend_, device_id_, logits, penalties, stream_);

        for (int i = 0; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                << "CPU↔GPU mismatch at token " << i;
        }
    }

    TEST_P(GPUSamplingTest, DRYParity_ExponentialScaling)
    {
        // History: [A, B, C, D, A, B, C, D] — repeat of length 4
        // Produces penalty = 2.0 * 1.75^(4-1) = 10.72 on token A
        const int vocab_size = 100;
        const int A = 10, B = 20, C = 30, D = 40;

        Sampler sampler(42);
        for (int token : {A, B, C, D, A, B, C, D})
            sampler.record_token(token);

        SamplingParams params;
        params.dry_multiplier = 2.0f;
        params.dry_base = 1.75f;
        params.dry_allowed_length = 1;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);
        ASSERT_FALSE(penalties.empty());

        std::vector<float> logits(vocab_size, 10.0f);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(
            backend_, device_id_, logits, penalties, stream_);

        for (int i = 0; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                << "CPU↔GPU mismatch at token " << i;
        }

        // Sanity: token A should have the expected exponential penalty
        float expected_penalty = 2.0f * std::pow(1.75f, 3.0f);
        EXPECT_NEAR(gpu_result[A], 10.0f - expected_penalty, 0.01f);
    }

    TEST_P(GPUSamplingTest, DRYParity_CombinedWithPresenceFrequency)
    {
        // DRY + presence + frequency penalties all combined
        const int vocab_size = 100;
        const int A = 10;

        Sampler sampler(42);
        for (int i = 0; i < 4; ++i)
            sampler.record_token(A);

        SamplingParams params;
        params.presence_penalty = 1.0f;
        params.frequency_penalty = 0.5f;
        params.dry_multiplier = 1.0f;
        params.dry_base = 1.75f;
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);
        ASSERT_FALSE(penalties.empty());

        std::vector<float> logits(vocab_size, 5.0f);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(
            backend_, device_id_, logits, penalties, stream_);

        for (int i = 0; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                << "CPU↔GPU mismatch at token " << i;
        }

        // Verify the combined penalty is additive (pres+freq + DRY)
        float pf_penalty = 1.0f + 0.5f * 4.0f; // 3.0
        EXPECT_GT(5.0f - gpu_result[A], pf_penalty)
            << "Combined penalty should exceed presence+frequency alone";
    }

    TEST_P(GPUSamplingTest, DRYParity_SequenceBreakers)
    {
        // History with a breaker in the middle — should NOT penalize across it
        const int vocab_size = 100;
        const int A = 10, NEWLINE = 50;

        Sampler sampler(42);
        sampler.initDryBreakers({"\n"}, [&](const std::string &) -> std::vector<int> {
            return {NEWLINE};
        });
        for (int token : {A, NEWLINE, A})
            sampler.record_token(token);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_allowed_length = 1;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);

        // With breaker, A should not be penalized — penalty map may be empty
        std::vector<float> logits(vocab_size, 5.0f);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(
            backend_, device_id_, logits, penalties, stream_);

        for (int i = 0; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                << "CPU↔GPU mismatch at token " << i;
        }

        // Token A should be unpenalized (breaker prevents detection)
        EXPECT_FLOAT_EQ(gpu_result[A], 5.0f)
            << "Sequence breaker should prevent DRY penalty on token A";
    }

    TEST_P(GPUSamplingTest, DRYParity_SingleTokenBreakerExemption)
    {
        // Token that is itself a single-token breaker should be exempt
        const int vocab_size = 100;
        const int A = 10, NEWLINE = 50;

        Sampler sampler(42);
        sampler.initDryBreakers({"\n"}, [&](const std::string &) -> std::vector<int> {
            return {NEWLINE};
        });
        // NEWLINE A NEWLINE A NEWLINE — repeat pattern, but NEWLINE is a breaker
        for (int token : {NEWLINE, A, NEWLINE, A, NEWLINE})
            sampler.record_token(token);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);

        std::vector<float> logits(vocab_size, 5.0f);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(
            backend_, device_id_, logits, penalties, stream_);

        for (int i = 0; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                << "CPU↔GPU mismatch at token " << i;
        }

        // NEWLINE should be exempt (single-token breaker)
        EXPECT_FLOAT_EQ(gpu_result[NEWLINE], 5.0f)
            << "Single-token breaker should be exempt from DRY penalty";
    }

    TEST_P(GPUSamplingTest, DRYParity_OverflowProtection)
    {
        // Large repeat count with base=2.0 — should not overflow to inf
        const int vocab_size = 100;
        const int A = 10;

        Sampler sampler(42);
        for (int i = 0; i < 50; ++i)
            sampler.record_token(A);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_base = 2.0f;
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);
        ASSERT_FALSE(penalties.empty());

        // Verify no overflow in CPU computation
        for (const auto &p : penalties)
        {
            EXPECT_FALSE(std::isinf(p.penalty)) << "CPU penalty should not overflow";
            EXPECT_FALSE(std::isnan(p.penalty)) << "CPU penalty should not be NaN";
        }

        std::vector<float> logits(vocab_size, 100.0f);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(
            backend_, device_id_, logits, penalties, stream_);

        for (int i = 0; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                << "CPU↔GPU mismatch at token " << i;
        }
    }

    TEST_P(GPUSamplingTest, DRYParity_WindowLimitsDetection)
    {
        // With a small window, long repeats outside the window should not be detected
        const int vocab_size = 100;

        Sampler sampler(42);
        for (int token : {1, 2, 3, 4, 5, 1, 2, 3, 4, 5})
            sampler.record_token(token);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = 3; // Only see last 3 tokens

        auto penalties = sampler.compute_penalty_map(params, vocab_size);

        std::vector<float> logits(vocab_size, 5.0f);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(
            backend_, device_id_, logits, penalties, stream_);

        for (int i = 0; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                << "CPU↔GPU mismatch at token " << i;
        }

        // With only 3-token window, cannot detect the full 5-length repeat
        float full_penalty = std::pow(1.75f, 4.0f);
        for (const auto &p : penalties)
        {
            EXPECT_LT(p.penalty, full_penalty)
                << "Window should prevent detection of full repeat";
        }
    }

    TEST_P(GPUSamplingTest, DRYParity_ArgmaxShift)
    {
        // End-to-end: DRY penalty shifts GPU argmax to match CPU argmax
        const int vocab_size = 10;

        Sampler sampler(42);
        // Token 5 and 7 alternate — so token 5 would extend the repeat
        for (int token : {5, 7, 5, 7})
            sampler.record_token(token);

        SamplingParams params;
        params.dry_multiplier = 5.0f;
        params.dry_base = 1.75f;
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);
        ASSERT_FALSE(penalties.empty());

        // Token 5 has highest logit but will be penalized by DRY
        std::vector<float> logits = {0.0f, 0.0f, 0.0f, 9.5f, 0.0f,
                                     10.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        // CPU: apply penalties and find argmax
        auto cpu_logits = logits;
        applyCpuPenalties(cpu_logits, penalties);
        int cpu_argmax = static_cast<int>(
            std::max_element(cpu_logits.begin(), cpu_logits.end()) - cpu_logits.begin());

        // GPU: apply penalties and find argmax
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<int> token_ids(penalties.size());
        std::vector<float> penalty_vals(penalties.size());
        for (size_t i = 0; i < penalties.size(); ++i)
        {
            token_ids[i] = penalties[i].token_id;
            penalty_vals[i] = penalties[i].penalty;
        }

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalty_vals.data(),
            static_cast<int>(penalties.size()), vocab_size, device_id_,
            stream_);
        ASSERT_TRUE(ok);

        float gpu_val = 0;
        int gpu_argmax = -1;
        ok = argmaxF32(
            d_ptr, vocab_size, device_id_, &gpu_val, &gpu_argmax,
            stream_);
        ASSERT_TRUE(ok);

        EXPECT_EQ(gpu_argmax, cpu_argmax)
            << "GPU argmax should match CPU argmax after DRY penalties";
        EXPECT_EQ(gpu_argmax, 3)
            << "Token 3 (9.5) should win after token 5 is penalized by DRY";

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, DRYParity_LargeVocab_RealisticScenario)
    {
        // Realistic scenario: Qwen2 vocab size, natural-looking token history
        const int vocab_size = 151936;
        std::mt19937 rng(12345);

        Sampler sampler(42);
        // Simulate a conversation with some repetitive patterns
        std::vector<int> history = {
            100, 200, 300, 400, 500,   // unique intro
            100, 200, 300, 400, 500,   // exact repeat
            600, 700, 800,             // break
            100, 200, 300, 400, 500,   // another repeat
            900, 1000                  // end
        };
        for (int token : history)
            sampler.record_token(token);

        SamplingParams params;
        params.dry_multiplier = 1.5f;
        params.dry_base = 1.75f;
        params.dry_allowed_length = 2;
        params.dry_penalty_last_n = -1;
        params.presence_penalty = 0.5f;
        params.frequency_penalty = 0.3f;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);
        ASSERT_FALSE(penalties.empty());

        // Generate logits with some structure
        std::vector<float> logits(vocab_size);
        std::uniform_real_distribution<float> dist(-5.0f, 15.0f);
        for (auto &l : logits)
            l = dist(rng);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(
            backend_, device_id_, logits, penalties, stream_);

        // Spot-check penalized tokens
        for (const auto &p : penalties)
        {
            EXPECT_FLOAT_EQ(cpu_result[p.token_id], gpu_result[p.token_id])
                << "CPU↔GPU mismatch at penalized token " << p.token_id;
        }

        // Spot-check unpenalized tokens
        std::set<int> penalized_ids;
        for (const auto &p : penalties)
            penalized_ids.insert(p.token_id);

        int checked = 0;
        for (int i = 0; i < vocab_size && checked < 100; ++i)
        {
            if (penalized_ids.find(i) == penalized_ids.end())
            {
                EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                    << "Unpenalized token " << i << " should be unchanged";
                checked++;
            }
        }
    }

    /**
     * @brief Prove the real captured dispatch-ticket publication on both GPUs.
     *
     * Admission initializes ticket identity on the fixture stream. Publication
     * is then captured there but replayed on a second explicit scheduler stream
     * after that stream changes controller/depth state. The observed ticket must
     * reflect replay-time device bytes, proving that hosted HIP scheduling does
     * not capture a host shadow or depend on the original capture stream.
     */
    TEST_P(
        GPUSamplingTest,
        DeviceGenerationDispatchTicketIsCapturedAndExplicitStreamOrdered)
    {
        using namespace sampling_math;

        constexpr uint64_t session_epoch = 0x123456789ABCDEF0ull;
        constexpr uint64_t workspace_generation = 0x0FEDCBA987654321ull;
        std::array<int, kDeviceGenerationControlCount> control{};
        DeviceGenerationDepthPolicy policy;
        policy.mode = DeviceGenerationDepthPolicyMode::Dynamic;
        policy.initial_depth = 2;
        policy.minimum_depth = 1;
        policy.maximum_depth =
            DeviceGenerationDepthPolicy::kMaximumSupportedDraftDepth;
        ASSERT_TRUE(policy.valid());
        ASSERT_TRUE(initialize_device_generation_control(
            /*max_new_tokens=*/64,
            /*response_capacity=*/64,
            policy,
            control.data()));

        void *const control_device = backend_->allocate(
            sizeof(control),
            device_id_);
        void *const ticket_device = backend_->allocate(
            sizeof(DeviceGenerationDispatchTicket),
            device_id_);
        void *const maintenance_due_device = backend_->allocate(
            sizeof(uint32_t),
            device_id_);
        ASSERT_NE(control_device, nullptr);
        ASSERT_NE(ticket_device, nullptr);
        ASSERT_NE(maintenance_due_device, nullptr);

        ASSERT_TRUE(copyHostToDevice(
            control_device,
            control.data(),
            sizeof(control),
            device_id_,
            stream_));
        ASSERT_TRUE(
            backend_->enqueueInitializeDeviceGenerationDispatchTicket(
                session_epoch,
                workspace_generation,
                control_device,
                kDeviceGenerationControlCount,
                /*request_count=*/1,
                ticket_device,
                device_id_,
                stream_));
        ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

        const DeviceId device =
            GetParam() == "CUDA" ? DeviceId::cuda(device_id_)
                                 : DeviceId::rocm(device_id_);
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        auto publication = context.createGraphCapture(stream_);
        ASSERT_NE(publication, nullptr);
        ASSERT_TRUE(publication->beginCapture());
        ASSERT_TRUE(backend_->enqueuePublishDeviceGenerationDispatchTickets(
            control_device,
            kDeviceGenerationControlCount,
            /*request_count=*/1,
            maintenance_due_device,
            ticket_device,
            device_id_,
            stream_));
        ASSERT_TRUE(publication->endCapture());
        ASSERT_TRUE(publication->instantiate());

        control[kDeviceGenerationControlTransactionCount] = 9;
        control[kDeviceGenerationControlCurrentDraftDepth] = 15;
        control[kDeviceGenerationControlActiveVerifierRowCount] = 16;
        const uint32_t maintenance_due = 1;
        void *const scheduler_stream = backend_->createStream(device_id_);
        ASSERT_NE(scheduler_stream, nullptr);
        ASSERT_TRUE(copyHostToDevice(
            control_device,
            control.data(),
            sizeof(control),
            device_id_,
            scheduler_stream));
        ASSERT_TRUE(copyHostToDevice(
            maintenance_due_device,
            &maintenance_due,
            sizeof(maintenance_due),
            device_id_,
            scheduler_stream));
        ASSERT_TRUE(publication->launchOnStream(scheduler_stream));

        DeviceGenerationDispatchTicket observed{};
        ASSERT_TRUE(copyDeviceToHost(
            &observed,
            ticket_device,
            sizeof(observed),
            device_id_,
            scheduler_stream));
        EXPECT_TRUE(observed.matchesLifecycle(
            session_epoch,
            workspace_generation));
        EXPECT_EQ(observed.transaction_count, 9);
        EXPECT_EQ(observed.next_draft_depth, 15);
        EXPECT_EQ(observed.maintenance_due, 1);
        EXPECT_EQ(observed.healthy, 1);
        EXPECT_EQ(
            observed.error_code,
            static_cast<int>(DeviceGenerationError::None));

        backend_->destroyStream(scheduler_stream, device_id_);
        backend_->free(maintenance_due_device, device_id_);
        backend_->free(ticket_device, device_id_);
        backend_->free(control_device, device_id_);
    }

} // anonymous namespace
