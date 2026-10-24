/**
 * @file CUDAGDNMergedDecodeTiming.h
 * @brief Opt-in unprofiled native-event timing of complete CUDA merged-M1 boundaries.
 *
 * Both sides call production backend methods and advance separately bound live
 * state in place. Control captures real deinterleave plus planar recurrence;
 * candidate captures the mandatory merged API. Existing native owners retain
 * all buffers and graphs. Full byte checks and capture authentication stay
 * outside event intervals; no CUPTI, Nsight or profiler API is used.
 */
#pragma once
#ifdef HAVE_CUDA
#include <cstdlib>
#include <iostream>

namespace
{
    /**
     * @brief Read one optional bounded private fixture selector without recovery.
     * @param name Exact selector name, private to this opt-in test.
     * @param absent Value when no selector was supplied.
     * @param maximum Largest admitted positive value.
     * @return Parsed positive integer or the explicit absent policy.
     * @throws std::invalid_argument For malformed, nonpositive or excessive input.
     */
    int cudaMergedTimingSelector(const char* name, int absent, int maximum)
    {
        const char* value = std::getenv(name);
        if (!value) return absent;
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end || parsed <= 0 || parsed > maximum)
            throw std::invalid_argument(std::string("Invalid private CUDA timing selector: ") + name);
        return static_cast<int>(parsed);
    }

    /** @brief Own reusable native timing events outside capture and native graphs. */
    class CudaMergedBoundaryEvents final
    {
    public:
        /** @brief Create timing-enabled events before any measurement starts. */
        CudaMergedBoundaryEvents()
        {
            checkCuda(cudaEventCreateWithFlags(&begin_, cudaEventDefault),
                "cudaEventCreateWithFlags(merged begin)");
            checkCuda(cudaEventCreateWithFlags(&end_, cudaEventDefault),
                "cudaEventCreateWithFlags(merged end)");
        }
        /** @brief Retire timing observers after the later terminal guard joined work. */
        ~CudaMergedBoundaryEvents()
        {
            (void)cudaEventDestroy(end_);
            (void)cudaEventDestroy(begin_);
        }
        /**
         * @brief Measure one captured complete-boundary graph on its exact stream.
         * @param graph Retained graph containing repeats complete transactions.
         * @param stream Exact non-default execution stream.
         * @param repeats Original logical transaction count, independent of node count.
         * @return Native microseconds per complete transaction.
         * @details Waiting observes terminal test completion only. No reset,
         * upload, parity read, capture or observer creation enters this interval.
         */
        double measure(const CudaCapturedGraph& graph, cudaStream_t stream, int repeats)
        {
            checkCuda(cudaEventRecord(begin_, stream), "cudaEventRecord(merged begin)");
            graph.launch(stream);
            checkCuda(cudaEventRecord(end_, stream), "cudaEventRecord(merged end)");
            checkCuda(cudaEventSynchronize(end_), "cudaEventSynchronize(merged timed terminal)");
            float milliseconds = 0.0f;
            checkCuda(cudaEventElapsedTime(&milliseconds, begin_, end_),
                "cudaEventElapsedTime(merged complete boundary)");
            return 1000.0 * static_cast<double>(milliseconds) / repeats;
        }
        CudaMergedBoundaryEvents(const CudaMergedBoundaryEvents&) = delete;
        CudaMergedBoundaryEvents& operator=(const CudaMergedBoundaryEvents&) = delete;
    private:
        cudaEvent_t begin_ = nullptr; ///< Native timing start, outside capture.
        cudaEvent_t end_ = nullptr; ///< Exact-stream terminal timestamp.
    };
}

TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_MergedM1CompleteBoundaryEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "cudaSetDevice(merged timing)");
    const int repeats = cudaMergedTimingSelector("QFN_GDN177_CUDA_REPEATS", 64, 256);
    const int selected_heads = cudaMergedTimingSelector("QFN_GDN177_CUDA_HEADS", 0, 24);
    ASSERT_TRUE(selected_heads == 0 || selected_heads == 12 || selected_heads == 24);
    CudaStreamHandle stream;
    for (const int heads : {12, 24})
    {
        if (selected_heads && heads != selected_heads) continue;
        constexpr int dk = 128, dv = 128, offset = 0;
        constexpr bool normalize = true;
        const int key_heads = heads / 3;
        const int merged_stride = 2 * key_heads * dk + heads * dv;
        const int state_floats = heads * dk * dv;
        const int output_floats = heads * dv;
        const int plane_floats = 2 * heads * dk + output_floats;
        SCOPED_TRACE("heads=" + std::to_string(heads));
        const auto merged = makeInitialState(static_cast<size_t>(merged_stride), 0.0125f);
        const auto initial = makeInitialState(static_cast<size_t>(state_floats), 0.0002f);
        const std::vector<float> alpha(heads, 0.03f), beta(heads, -0.15f);
        const std::vector<float> a_log(heads, -0.5f), bias(heads, 0.1f);
        std::vector<float> expected_planes(static_cast<size_t>(plane_floats));
        for (int head = 0; head < heads; ++head)
        {
            const int source = head % key_heads;
            std::copy_n(merged.data() + source * dk, dk,
                expected_planes.data() + head * dk);
            std::copy_n(merged.data() + key_heads * dk + source * dk, dk,
                expected_planes.data() + heads * dk + head * dk);
            std::copy_n(merged.data() + 2 * key_heads * dk + head * dv, dv,
                expected_planes.data() + 2 * heads * dk + head * dv);
        }
        CudaFloatBuffer d_merged(merged), d_alpha(alpha), d_beta(beta);
        CudaFloatBuffer d_a_log(a_log), d_bias(bias);
        CudaFloatBuffer d_candidate_state(initial), d_control_state(initial);
        CudaFloatBuffer d_candidate_output(static_cast<size_t>(output_floats), 111.0f);
        CudaFloatBuffer d_control_output(static_cast<size_t>(output_floats), 222.0f);
        CudaFloatBuffer d_control_planes(static_cast<size_t>(plane_floats), 333.0f);
        CUDAGatedDeltaNet candidate(cuda_ordinal_), control(cuda_ordinal_);
        ASSERT_TRUE(candidate.bindDeviceState(GDNDeviceStateBinding{
            .primary_state = d_candidate_state.ptr,
            .primary_state_floats = state_floats,
            .request_state_bank = d_candidate_state.ptr,
            .request_state_bank_floats = d_candidate_state.count,
            .request_capacity = 1,
        }));
        ASSERT_TRUE(control.bindDeviceState(GDNDeviceStateBinding{
            .primary_state = d_control_state.ptr,
            .primary_state_floats = state_floats,
            .request_state_bank = d_control_state.ptr,
            .request_state_bank_floats = d_control_state.count,
            .request_capacity = 1,
        }));
        candidate.setGPUStream(stream.stream);
        control.setGPUStream(stream.stream);
        control.bindDeinterleaveWorkspace(d_control_planes.ptr, d_control_planes.count);
        // The candidate has no layout scratch. Both recurrences use ordinary
        // in-place live state, with no verifier capture or speculative bank.
        ITensorGatedDeltaNet& candidate_api = candidate;
        CudaCapturedGraph control_graph(stream.stream, [&]() {
            for (int repeat = 0; repeat < repeats; ++repeat)
            {
                float *q = nullptr, *k = nullptr, *v = nullptr;
                if (!control.deinterleave_qkv_device(d_merged.ptr, q, k, v,
                        DeviceRequestRowRanges::fullyActive(1), key_heads, heads,
                        dk, dv, offset)) return false;
                if (!control.recurrent_step(q, k, v, d_alpha.ptr, d_beta.ptr,
                        d_a_log.ptr, d_bias.ptr, d_control_output.ptr, nullptr,
                        heads, dk, dv, normalize)) return false;
            }
            return true;
        });
        CudaCapturedGraph candidate_graph(stream.stream, [&]() {
            for (int repeat = 0; repeat < repeats; ++repeat)
                if (!candidate_api.chunkForwardMergedQKV(d_merged.ptr, merged_stride,
                        d_alpha.ptr, d_beta.ptr, d_a_log.ptr, d_bias.ptr,
                        d_candidate_output.ptr, nullptr, 1, key_heads, heads,
                        dk, dv, offset, 64, normalize)) return false;
            return true;
        });
        std::array<void*, 3> functions{}; // Copier, planar recurrence, merged recurrence.
        const auto inspect = [&](const CudaCapturedGraph& graph, bool is_candidate) {
            size_t count = 0;
            checkCuda(cudaGraphGetNodes(graph.graph, nullptr, &count), "cudaGraphGetNodes(timing count)");
            const size_t expected = static_cast<size_t>(repeats) * (is_candidate ? 1 : 2);
            ASSERT_EQ(count, expected);
            count = 0;
            checkCuda(cudaGraphGetRootNodes(graph.graph, nullptr, &count), "cudaGraphGetRootNodes(count)");
            ASSERT_EQ(count, 1u);
            cudaGraphNode_t node = nullptr;
            checkCuda(cudaGraphGetRootNodes(graph.graph, &node, &count), "cudaGraphGetRootNodes(node)");
            for (size_t index = 0; index < expected; ++index)
            {
                cudaGraphNodeType type;
                checkCuda(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType(timing)");
                ASSERT_EQ(type, cudaGraphNodeTypeKernel);
                cudaKernelNodeParams p{};
                checkCuda(cudaGraphKernelNodeGetParams(node, &p), "cudaGraphKernelNodeGetParams(timing)");
                ASSERT_NE(p.kernelParams, nullptr);
                const auto input = [&](int argument) {
                    return *static_cast<const float* const*>(p.kernelParams[argument]);
                };
                const auto integer = [&](int argument) {
                    return *static_cast<const int*>(p.kernelParams[argument]);
                };
                const bool copier = !is_candidate && index % 2 == 0;
                const int role = is_candidate ? 2 : copier ? 0 : 1;
                if (!functions[role])
                {
                    functions[role] = p.func;
                    cudaFuncAttributes attributes{};
                    checkCuda(cudaFuncGetAttributes(&attributes, p.func), "cudaFuncGetAttributes(timing role)");
                    ASSERT_EQ(attributes.localSizeBytes, 0u);
                    std::cout << "CUDA_GDN_MERGED_NATIVE heads=" << heads << " role=" << role
                        << " registers=" << attributes.numRegs << " static_shared=" << attributes.sharedSizeBytes
                        << " dynamic_shared=" << p.sharedMemBytes << " private=" << attributes.localSizeBytes
                        << " function=" << reinterpret_cast<std::uintptr_t>(p.func) << '\n';
                }
                ASSERT_EQ(p.func, functions[role]);
                ASSERT_EQ(p.blockDim.x, 256u);
                ASSERT_EQ(p.blockDim.y, 1u);
                ASSERT_EQ(p.blockDim.z, 1u);
                ASSERT_EQ(p.gridDim.z, 1u);
                if (copier)
                {
                    ASSERT_EQ(p.gridDim.x, static_cast<unsigned>(1 + (plane_floats - 1) / 256));
                    ASSERT_EQ(p.gridDim.y, 1u);
                    ASSERT_EQ(p.sharedMemBytes, 0u);
                    ASSERT_EQ(input(0), d_merged.ptr);
                    ASSERT_EQ(input(1), d_control_planes.ptr);
                    ASSERT_EQ(input(2), d_control_planes.ptr + heads * dk);
                    ASSERT_EQ(input(3), d_control_planes.ptr + 2 * heads * dk);
                    const auto& rows = *static_cast<const DeviceRequestRowRanges*>(p.kernelParams[4]);
                    ASSERT_EQ(rows.requests(), 1);
                    ASSERT_EQ(rows.rowsPerRequest(), 1);
                    ASSERT_EQ(rows.lengthOwner(), nullptr);
                    ASSERT_EQ(integer(5), key_heads);
                    ASSERT_EQ(integer(6), heads);
                    ASSERT_EQ(integer(7), dk);
                    ASSERT_EQ(integer(8), dv);
                    ASSERT_EQ(integer(9), offset);
                }
                else
                {
                    ASSERT_EQ(p.gridDim.x, static_cast<unsigned>(heads));
                    ASSERT_EQ(p.gridDim.y, 4u);
                    ASSERT_EQ(p.sharedMemBytes, 3072u);
                    ASSERT_EQ(input(0), is_candidate ? d_merged.ptr : d_control_planes.ptr);
                    ASSERT_EQ(input(1), is_candidate ? nullptr : d_control_planes.ptr + heads * dk);
                    ASSERT_EQ(input(2), is_candidate ? nullptr : d_control_planes.ptr + 2 * heads * dk);
                    ASSERT_EQ(input(3), d_alpha.ptr);
                    ASSERT_EQ(input(4), d_beta.ptr);
                    ASSERT_EQ(input(5), d_a_log.ptr);
                    ASSERT_EQ(input(6), d_bias.ptr);
                    ASSERT_EQ(input(7), is_candidate ? d_candidate_output.ptr : d_control_output.ptr);
                    const float* state = is_candidate ? d_candidate_state.ptr : d_control_state.ptr;
                    ASSERT_EQ(input(8), state);
                    ASSERT_EQ(input(9), state);
                    ASSERT_EQ(integer(10), 1);
                    ASSERT_EQ(integer(11), 1);
                    ASSERT_EQ(integer(12), heads);
                    ASSERT_EQ(integer(13), dv);
                    ASSERT_EQ(*static_cast<const bool*>(p.kernelParams[14]), normalize);
                    ASSERT_EQ(*static_cast<const int* const*>(p.kernelParams[15]), nullptr);
                    ASSERT_EQ(integer(18), 0);
                    ASSERT_EQ(integer(19), 0);
                    ASSERT_EQ(input(17), nullptr);
                    ASSERT_EQ(integer(16), 0);
                    if (is_candidate)
                    {
                        ASSERT_EQ(integer(20), key_heads);
                        ASSERT_EQ(integer(21), offset);
                    }
                }
                size_t next_count = 0;
                checkCuda(cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &next_count),
                    "cudaGraphNodeGetDependentNodes(count)");
                ASSERT_EQ(next_count, index + 1 == expected ? 0u : 1u);
                if (next_count)
                    checkCuda(cudaGraphNodeGetDependentNodes(node, &node, nullptr, &next_count),
                        "cudaGraphNodeGetDependentNodes(next)");
            }
            std::cout << "CUDA_GDN_MERGED_CAPTURE side=" << (is_candidate ? "candidate" : "control")
                << " heads=" << heads << " key_heads=" << key_heads << " DK=" << dk << " DV=" << dv
                << " repeats=" << repeats << " nodes=" << expected << '\n';
        };
        ASSERT_NO_FATAL_FAILURE(inspect(control_graph, false));
        ASSERT_NO_FATAL_FAILURE(inspect(candidate_graph, true));
        ASSERT_NE(functions[0], functions[1]);
        ASSERT_NE(functions[1], functions[2]);
        const auto reset = [&]() {
            d_candidate_state.copyFrom(initial);
            d_control_state.copyFrom(initial);
            d_candidate_output.fill(111.0f);
            d_control_output.fill(222.0f);
            d_control_planes.fill(333.0f);
        };
        const auto bytes = [&]() {
            const auto out = d_candidate_output.toHost(), ref = d_control_output.toHost();
            const auto state = d_candidate_state.toHost(), ref_state = d_control_state.toHost();
            const auto planes = d_control_planes.toHost(), source = d_merged.toHost();
            ASSERT_EQ(std::memcmp(out.data(), ref.data(), out.size() * sizeof(float)), 0);
            ASSERT_EQ(std::memcmp(state.data(), ref_state.data(), state.size() * sizeof(float)), 0);
            ASSERT_EQ(std::memcmp(planes.data(), expected_planes.data(), planes.size() * sizeof(float)), 0);
            ASSERT_EQ(std::memcmp(source.data(), merged.data(), source.size() * sizeof(float)), 0);
            expectByteExactEquivalent("timing alpha immutable", d_alpha.toHost(), alpha);
            expectByteExactEquivalent("timing beta immutable", d_beta.toHost(), beta);
        };
        // Two dirty reset/replays prove complete bytes before events begin.
        for (int replay = 0; replay < 2; ++replay)
        {
            reset();
            CudaMergedReplayRetirement retirement(stream.stream);
            control_graph.launch(stream.stream);
            candidate_graph.launch(stream.stream);
            retirement.finish();
            ASSERT_NO_FATAL_FAILURE(bytes());
            ASSERT_FALSE(::testing::Test::HasFailure());
        }
        for (int warmup = 0; warmup < 5; ++warmup)
        {
            reset();
            CudaMergedReplayRetirement retirement(stream.stream);
            control_graph.launch(stream.stream);
            candidate_graph.launch(stream.stream);
            retirement.finish();
        }
        CudaMergedBoundaryEvents events;
        std::vector<double> control_samples, candidate_samples;
        for (int sample = 0; sample < 12; ++sample)
        {
            reset();
            CudaMergedReplayRetirement retirement(stream.stream);
            double control_us, candidate_us;
            if (sample % 2 == 0)
            {
                control_us = events.measure(control_graph, stream.stream, repeats);
                candidate_us = events.measure(candidate_graph, stream.stream, repeats);
            }
            else
            {
                candidate_us = events.measure(candidate_graph, stream.stream, repeats);
                control_us = events.measure(control_graph, stream.stream, repeats);
            }
            retirement.finish();
            control_samples.push_back(control_us);
            candidate_samples.push_back(candidate_us);
            std::cout << std::setprecision(9) << "CUDA_GDN_MERGED_SAMPLE heads=" << heads
                << " sample=" << sample << " order=" << (sample % 2 ? "candidate_first" : "control_first")
                << " control_us=" << control_us << " candidate_us=" << candidate_us << '\n';
            ASSERT_NO_FATAL_FAILURE(bytes());
            ASSERT_FALSE(::testing::Test::HasFailure());
        }
        const auto median = [](std::vector<double> samples) {
            std::sort(samples.begin(), samples.end());
            return (samples[5] + samples[6]) / 2.0;
        };
        std::cout << std::setprecision(9) << "CUDA_GDN_MERGED_PAIR heads=" << heads
            << " key_heads=" << key_heads << " DK=" << dk << " DV=" << dv
            << " state=in_place live=1 norm=1 repeats=" << repeats
            << " samples=12 warmups=5 unit=us_per_complete_boundary control_us=" << median(control_samples)
            << " candidate_us=" << median(candidate_samples) << '\n';
    }
}
#endif
