/**
 * @file CUDAGDNMergedDecodeRegression.h
 * @brief Captured production CUDA merged-M1 coordinate and state-owner regression.
 *
 * Included by the existing CUDA GDN integration fixture to reuse its native
 * buffers, captured graph owner, strict byte helper and backend discovery.
 * The candidate enters the mandatory tensor API without any layout workspace.
 * An independent host coordinate copy supplies the existing planar backend
 * oracle. Graph inspection authenticates every borrowed native input/state
 * binding, while retained dirty replays test lifecycle without recapture.
 */
#pragma once
#ifdef HAVE_CUDA
#include <limits>

namespace
{
    /**
     * @brief Compare complete FP32 vectors through the existing strict slice oracle.
     * @param label Diagnostic context for a size or byte mismatch.
     * @param actual Observed complete physical output or state bank.
     * @param expected Independently produced complete physical reference.
     */
    void expectByteExactEquivalent(
        const char *label, const std::vector<float> &actual,
        const std::vector<float> &expected)
    {
        ASSERT_EQ(actual.size(), expected.size()) << label;
        expectByteExactEquivalent(label, actual, expected, 0, expected.size());
    }

    /**
     * @brief Retire already-submitted integration work before any graph/buffer dies.
     * @details This terminal-only event owner is declared after captured graphs
     * and before their first replay. Its exceptional-exit join protects assertion
     * and exception paths; no such host wait exists in production execution.
     */
    class CudaMergedReplayRetirement final
    {
    public:
        /** @brief Create a terminal observer outside capture on the exact stream.
         * @param stream Non-default test execution stream. */
        explicit CudaMergedReplayRetirement(cudaStream_t stream) : stream_(stream)
        {
            checkCuda(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming),
                "cudaEventCreateWithFlags(merged terminal)");
        }
        /** @brief Join any pending replay before the earlier native owners retire. */
        ~CudaMergedReplayRetirement()
        {
            if (!joined_)
            {
                (void)cudaEventRecord(done_, stream_);
                (void)cudaEventSynchronize(done_);
            }
            (void)cudaEventDestroy(done_);
        }
        /** @brief Surface one immutable terminal completion before host observation. */
        void finish()
        {
            checkCuda(cudaEventRecord(done_, stream_), "cudaEventRecord(merged terminal)");
            checkCuda(cudaEventSynchronize(done_), "cudaEventSynchronize(merged terminal)");
            joined_ = true;
        }
        CudaMergedReplayRetirement(const CudaMergedReplayRetirement&) = delete;
        CudaMergedReplayRetirement& operator=(const CudaMergedReplayRetirement&) = delete;
    private:
        cudaStream_t stream_; ///< Exact execution producer stream.
        cudaEvent_t done_ = nullptr; ///< Terminal test-only publication.
        bool joined_ = false; ///< Whether completion was authenticated explicitly.
    };
}

TEST_F(Test__CUDAGDNPaddedRealLength,
       MergedM1DecodeUsesSingleCapturedNativeKernelAndPreservesStateBinding)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "cudaSetDevice(merged M1)");
    CudaStreamHandle stream;
    constexpr int key_heads = 2, heads = 6;
    const std::array<int, 5> offsets{
        0, 5, -5, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()};
    for (const int dk : {64, 128})
    for (const int dv : {64, 65, 128, 129})
    for (const bool normalize : {false, true})
    for (const int offset : offsets)
    {
        const int key_span = key_heads * dk;
        const int stride = 2 * key_span + heads * dv;
        const int state_floats = heads * dk * dv;
        SCOPED_TRACE("dk=" + std::to_string(dk) + " dv=" + std::to_string(dv) +
            " offset=" + std::to_string(offset) + " norm=" + std::to_string(normalize));
        auto merged = makeInitialState(static_cast<size_t>(stride), 0.0125f);
        auto initial = makeInitialState(static_cast<size_t>(state_floats), 0.0002f);
        merged[0] = -0.0f;
        merged[1] = std::numeric_limits<float>::denorm_min();
        initial[0] = -0.0f;
        std::vector<float> q(static_cast<size_t>(heads) * dk);
        std::vector<float> k(q.size()), v(static_cast<size_t>(heads) * dv);
        for (int head = 0; head < heads; ++head)
        {
            auto source = (std::int64_t{head} + offset) % key_heads;
            if (source < 0) source += key_heads;
            std::copy_n(merged.data() + source * dk, dk, q.data() + head * dk);
            std::copy_n(merged.data() + key_span + source * dk, dk, k.data() + head * dk);
            std::copy_n(merged.data() + 2 * key_span + head * dv, dv, v.data() + head * dv);
        }
        std::vector<float> alpha(heads), beta(heads), a_log(heads, -0.5f), bias(heads, 0.1f);
        for (int head = 0; head < heads; ++head)
        {
            alpha[head] = 19.9f + static_cast<float>(head % 3 - 1) * 0.000002f;
            beta[head] = static_cast<float>(head - 3) * 0.15f;
        }
        std::vector<float> ordinary_terminal;
        for (const int snapshot_rows : {0, 1, 4})
        {
            const bool verifier = snapshot_rows > 0;
            SCOPED_TRACE("snapshot_rows=" + std::to_string(snapshot_rows));
            const size_t snapshot_floats = initial.size() *
                static_cast<size_t>(std::max(1, snapshot_rows));
            CudaFloatBuffer d_merged(merged), d_q(q), d_k(k), d_v(v);
            CudaFloatBuffer d_alpha(alpha), d_beta(beta), d_a_log(a_log), d_bias(bias);
            CudaFloatBuffer d_live(initial), d_control_live(initial);
            CudaFloatBuffer d_out(v.size(), 111.0f), d_control_out(v.size(), 222.0f);
            CudaFloatBuffer d_snapshot(snapshot_floats, 333.0f);
            CudaFloatBuffer d_control_snapshot(snapshot_floats, 333.0f);
            CudaFloatBuffer d_work(initial.size(), 444.0f), d_control_work(initial.size(), 444.0f);
            CUDAGatedDeltaNet candidate(cuda_ordinal_), control(cuda_ordinal_);
            ASSERT_TRUE(candidate.bindDeviceState(GDNDeviceStateBinding{
                .primary_state = d_live.ptr,
                .primary_state_floats = state_floats,
                .request_state_bank = d_live.ptr,
                .request_state_bank_floats = d_live.count,
                .request_capacity = 1,
            }));
            ASSERT_TRUE(control.bindDeviceState(GDNDeviceStateBinding{
                .primary_state = d_control_live.ptr,
                .primary_state_floats = state_floats,
                .request_state_bank = d_control_live.ptr,
                .request_state_bank_floats = d_control_live.count,
                .request_capacity = 1,
            }));
            candidate.setGPUStream(stream.stream);
            control.setGPUStream(stream.stream);
            if (verifier)
            {
                candidate.bindVerifierStateCaptureWorkspace(d_snapshot.ptr, snapshot_rows, state_floats);
                control.bindVerifierStateCaptureWorkspace(d_control_snapshot.ptr, snapshot_rows, state_floats);
                candidate.bindSpeculativeStateWorkspace(d_work.ptr, state_floats);
                control.bindSpeculativeStateWorkspace(d_control_work.ptr, state_floats);
            }
            // No deinterleave workspace is bound: a planar-copy route would be
            // an invalid capture rather than a second successful implementation.
            ITensorGatedDeltaNet& api = candidate;
            // All negative source/output intervals remain inside one admitted
            // test allocation. They must fail before any recurrence is queued.
            CudaFloatBuffer alias_storage(static_cast<size_t>(stride) + v.size() + 2, 555.0f);
            {
                CudaMergedReplayRetirement rejection_retirement(stream.stream);
                const auto reject_alias = [&](const float* source, float* destination) {
                    return api.chunkForwardMergedQKV(source, stride,
                        d_alpha.ptr, d_beta.ptr, d_a_log.ptr, d_bias.ptr,
                        destination, nullptr, 1, key_heads, heads, dk, dv,
                        offset, 64, normalize);
                };
                ASSERT_FALSE(reject_alias(alias_storage.ptr, alias_storage.ptr));
                ASSERT_FALSE(reject_alias(alias_storage.ptr, alias_storage.ptr + 1));
                ASSERT_FALSE(reject_alias(alias_storage.ptr + 1, alias_storage.ptr));
                rejection_retirement.finish();
            }
            CudaCapturedGraph merged_graph(stream.stream, [&]() {
                return api.chunkForwardMergedQKV(d_merged.ptr, stride,
                    d_alpha.ptr, d_beta.ptr, d_a_log.ptr, d_bias.ptr,
                    d_out.ptr, nullptr, 1, key_heads, heads, dk, dv, offset,
                    64, normalize);
            });
            CudaCapturedGraph control_graph(stream.stream, [&]() {
                return control.recurrent_step(d_q.ptr, d_k.ptr, d_v.ptr,
                    d_alpha.ptr, d_beta.ptr, d_a_log.ptr, d_bias.ptr,
                    d_control_out.ptr, nullptr, heads, dk, dv, normalize);
            });
            cudaKernelNodeParams merged_params{}, control_params{};
            const auto inspect = [&](const CudaCapturedGraph& graph,
                                     cudaKernelNodeParams& params) {
                size_t count = 0;
                checkCuda(cudaGraphGetNodes(graph.graph, nullptr, &count), "cudaGraphGetNodes(count)");
                ASSERT_EQ(count, 1u);
                cudaGraphNode_t node = nullptr;
                checkCuda(cudaGraphGetNodes(graph.graph, &node, &count), "cudaGraphGetNodes(node)");
                cudaGraphNodeType type;
                checkCuda(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType");
                ASSERT_EQ(type, cudaGraphNodeTypeKernel);
                checkCuda(cudaGraphKernelNodeGetParams(node, &params), "cudaGraphKernelNodeGetParams");
                ASSERT_EQ(params.gridDim.x, static_cast<unsigned>(heads));
                ASSERT_EQ(params.gridDim.y, static_cast<unsigned>(1 + (dv - 1) / 32));
                ASSERT_EQ(params.gridDim.z, 1u);
                ASSERT_EQ(params.blockDim.x, 256u);
                ASSERT_EQ(params.blockDim.y, 1u);
                ASSERT_EQ(params.blockDim.z, 1u);
                ASSERT_EQ(params.sharedMemBytes, static_cast<unsigned>((2 * dk + 512) * sizeof(float)));
            };
            ASSERT_NO_FATAL_FAILURE(inspect(merged_graph, merged_params));
            ASSERT_NO_FATAL_FAILURE(inspect(control_graph, control_params));
            ASSERT_NE(merged_params.func, control_params.func);
            ASSERT_NE(merged_params.kernelParams, nullptr);
            const auto input = [&](int index) {
                return *static_cast<const float* const*>(merged_params.kernelParams[index]);
            };
            const auto integer = [&](int index) {
                return *static_cast<const int*>(merged_params.kernelParams[index]);
            };
            ASSERT_EQ(input(0), d_merged.ptr);
            ASSERT_EQ(input(1), nullptr);
            ASSERT_EQ(input(2), nullptr);
            ASSERT_EQ(input(3), d_alpha.ptr);
            ASSERT_EQ(input(4), d_beta.ptr);
            ASSERT_EQ(input(5), d_a_log.ptr);
            ASSERT_EQ(input(6), d_bias.ptr);
            ASSERT_EQ(input(7), d_out.ptr);
            ASSERT_EQ(input(8), d_live.ptr);
            ASSERT_EQ(input(9), verifier ? d_work.ptr : d_live.ptr);
            ASSERT_EQ(integer(10), 1);
            ASSERT_EQ(integer(11), 1);
            ASSERT_EQ(integer(12), heads);
            ASSERT_EQ(integer(13), dv);
            ASSERT_EQ(*static_cast<const bool*>(merged_params.kernelParams[14]), normalize);
            ASSERT_EQ(*static_cast<const int* const*>(merged_params.kernelParams[15]), nullptr);
            ASSERT_EQ(integer(16), 0);
            ASSERT_EQ(input(17), verifier ? d_snapshot.ptr : nullptr);
            ASSERT_EQ(integer(18), verifier ? state_floats : 0);
            ASSERT_EQ(integer(19), snapshot_rows);
            ASSERT_EQ(integer(20), key_heads);
            ASSERT_EQ(integer(21), offset);
            for (int replay = 0; replay < 2; ++replay)
            {
                d_live.copyFrom(initial);
                d_control_live.copyFrom(initial);
                d_out.fill(111.0f + replay);
                d_control_out.fill(222.0f + replay);
                d_snapshot.fill(333.0f + replay);
                d_control_snapshot.fill(333.0f + replay);
                d_work.fill(444.0f + replay);
                d_control_work.fill(444.0f + replay);
                CudaMergedReplayRetirement retirement(stream.stream);
                merged_graph.launch(stream.stream);
                control_graph.launch(stream.stream);
                retirement.finish();
                expectByteExactEquivalent("merged output", d_out.toHost(), d_control_out.toHost());
                expectByteExactEquivalent("merged live owner", d_live.toHost(), d_control_live.toHost());
                expectByteExactEquivalent("merged snapshot", d_snapshot.toHost(), d_control_snapshot.toHost());
                expectByteExactEquivalent("merged work bank", d_work.toHost(), d_control_work.toHost());
                expectByteExactEquivalent("merged source immutable", d_merged.toHost(), merged);
                expectByteExactEquivalent("planar Q immutable", d_q.toHost(), q);
                expectByteExactEquivalent("planar K immutable", d_k.toHost(), k);
                expectByteExactEquivalent("planar V immutable", d_v.toHost(), v);
                expectByteExactEquivalent("alpha immutable", d_alpha.toHost(), alpha);
                expectByteExactEquivalent("beta immutable", d_beta.toHost(), beta);
                if (!verifier)
                    ordinary_terminal = d_control_live.toHost();
                else
                {
                    expectByteExactEquivalent("verifier live unchanged", d_live.toHost(), initial);
                    // A retained verifier bank may admit more rows than this
                    // ordinary M1 transaction advances. Only row0 is published;
                    // every other admitted row must retain its dirty sentinel.
                    std::vector<float> expected_snapshots(snapshot_floats, 333.0f + replay);
                    std::copy(ordinary_terminal.begin(), ordinary_terminal.end(),
                        expected_snapshots.begin());
                    expectByteExactEquivalent("whole retained snapshot envelope",
                        d_snapshot.toHost(), expected_snapshots);
                    CudaMergedReplayRetirement publication(stream.stream);
                    ASSERT_TRUE(candidate.restoreVerifierStateCaptureRow(nullptr, 0, stream.stream));
                    publication.finish();
                    expectByteExactEquivalent("accepted snapshot published to live owner",
                        d_live.toHost(), ordinary_terminal);
                }
            }
        }
    }
}
#endif
