/**
 * @file Perf__GPUExpertPipeline.cpp
 * @brief Captured all-format routed-expert pipeline economy on both GPUs.
 *
 * Each independently selectable case has real 122B or Qwen 3.6 35B projection
 * geometry. Qwen shard probes additionally isolate complete public phases and
 * the fixed-down column bank, under shared uniform/hotset/power-law profiles;
 * these are communication-free local costs, not multi-GPU speedups. Every
 * downstream phase receives valid producer data prepared outside its timer.
 * Timing includes device grouping, gate/up/SwiGLU, canonical down
 * publication and reduction. Weight preparation, graph construction and result
 * downloads are outside event timing. Numerical certification belongs to the
 * independent serial oracle in MoELiveRows; this fixture additionally requires
 * the timed replay sequence to leave the same finite output bits. No performance
 * threshold or benchmark case is added to the production-preflight gate.
 * Opt-in gfx1100 tournaments use the same source/geometry owners, compare both
 * native tiles against explicit ordered Dot4 after poisoned retained replay,
 * and keep profiler launches separate from paired native-event samples.
 * Hybrid experiments retain all row-floor graphs while route data crosses the
 * thresholds. Skewed replay independently removes each down writer family;
 * neither a matching symbol nor a repaired final result can certify ownership.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "execution/moe/MoERuntimeTable.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "utils/GpuPreparedGemmHarness.h"
#include "utils/TestTensorFactory.h"
#include "utils/QuantizedVerifierFormats.h"
#include "performance/kernels/native_quantised_dispatch/NativeQuantisedMoERoutingProfiles.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include "backends/rocm/HIPGraphCapture.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedWmma.h"
#endif
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    using native_quantised_dispatch::MoERoutingProfile;

    /** One backend/storage/physical-row shape; no weight conversion is timed. */
    struct Case
    {
        DeviceId device;
        TensorType format;
        int rows;
        int live_rows;
        int local_routes;
        std::string name;
        int hidden = 3072;
        int intermediate = 1024;
        int experts = 8;
        int ownership_degree = 0; ///< Nonzero selects ordinal whole-expert ownership.
        std::optional<TensorType> down_format;
        MoERoutingProfile route_profile = MoERoutingProfile::Uniform;
    };

    /** @return Symmetric storage, prompt/verifier geometry and sparse live-work cases. */
    std::vector<Case> cases()
    {
        std::vector<Case> result;
        for (auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
        {
            // Qwen's expert matrices remain whole; TP divides owned routes,
            // not each expert's N/K dimensions. Mixed source formats reproduce
            // the IQ3_S GGUF's common routed-expert arithmetic surface.
            for (const auto profile : native_quantised_dispatch::kMoERoutingProfiles)
                for (const int degree : {1, 2, 4})
                    for (const int rows : {64, 512})
                        result.push_back({device, TensorType::IQ2_S, rows, rows == 512 ? 448 : rows, 8,
                            std::string(device.is_cuda() ? "CUDA_" : "ROCm_") + "Qwen36WholeExpertShard" +
                            std::to_string(degree) + "Rows" + std::to_string(rows) +
                            (profile == MoERoutingProfile::Uniform ? "" :
                                "_Route_" + std::string(native_quantised_dispatch::moeRoutingProfileName(profile))),
                            2048, 512, 256, degree, TensorType::IQ4_XS, profile});
            for (auto format : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
                for (int rows : {1, 16, 128, 512})
                    result.push_back({device, format, rows, rows, 8,
                        std::string(device.is_cuda() ? "CUDA_" : "ROCm_") +
                        (format == TensorType::FP16 ? "FP16_" : format == TensorType::BF16 ? "BF16_" : "FP32_") +
                        "Rows" + std::to_string(rows)});
            // Capacity and live work are independent under graph capture. Sparse
            // cases model a participant that owns only some of a token's routes;
            // the full case guards against optimizing only mostly-empty graphs.
            for (const auto &format : quantizedMoEVerifierFormats())
            {
                for (int live : {0, 1, 3, 16})
                    result.push_back({device, format.tensor_type, 16, live, live == 16 ? 8 : 3,
                        std::string(device.is_cuda() ? "CUDA_" : "ROCm_") + format.label +
                        "Rows16Live" + std::to_string(live)});
                // Long rows select the expert-tiled producer rather than the
                // verifier family. Compact matrices keep this all-codebook
                // economy sweep cheap, while 65/512 live rows exercise both
                // partial and full row tiles under the installed dispatch.
                for (const int rows : {65, 512})
                    result.push_back({device, format.tensor_type, rows, rows, 8,
                        std::string(device.is_cuda() ? "CUDA_" : "ROCm_") +
                        "AllCodebookPrefill_" + format.label + "Rows" + std::to_string(rows),
                        512, 256, 8});
            }
        }
        return result;
    }

    /** @return Source bytes rounded once to the requested weight storage. */
    std::unique_ptr<TensorBase> weight(TensorType type, const std::vector<size_t> &shape, uint32_t seed)
    {
        if (type == TensorType::FP16) return TestTensorFactory::createFP16Random(shape, -0.125f, 0.125f, seed);
        if (type == TensorType::BF16) return TestTensorFactory::createBF16Random(shape, -0.125f, 0.125f, seed);
        if (type == TensorType::FP32) return TestTensorFactory::createFP32Random(shape, -0.125f, 0.125f, seed);
        for (const auto &format : quantizedMoEVerifierFormats())
            if (format.tensor_type == type) return format.create(shape, seed);
        throw std::invalid_argument("unregistered expert benchmark weight format");
    }

    /** Captured, event-timed whole-operation fixture; supports one-case profiling. */
    class GPUExpertPipeline : public ::testing::TestWithParam<Case> {};

    /** @brief One complete benchmark operation, with profiler runs kept untimed. */
    enum class ExpertBenchmark
    {
        Installed, WmmaReplay, WmmaTournament, Wmma32Profile, Wmma64Profile,
        WmmaHybridReplay, WmmaHybridTournament, WmmaHybrid32Profile
    };

#ifdef HAVE_ROCM
    /** @brief One retained dot policy; names bind every sample to its row floor. */
    struct GroupedDotCandidate
    {
        rocm::MoEGroupedPrefillEngine engine;
        rocm::MoEGroupedWmmaRowFloor floor;
        std::string name;
    };

    /** @param hybrid Include all explicit whole-expert admission experiments.
     * @return Ordered oracle, both pure native tiles and requested hybrid policies. */
    std::vector<GroupedDotCandidate> groupedDotCandidates(bool hybrid)
    {
        using enum rocm::MoEGroupedPrefillEngine;
        using enum rocm::MoEGroupedWmmaRowFloor;
        std::vector<GroupedDotCandidate> result{
            {OrderedDot4,EveryExpert,"Dot4"},
            {WmmaM16N32,EveryExpert,"N32"}, {WmmaM16N64,EveryExpert,"N64"}};
        if (hybrid)
            for (const auto floor : {Four,Eight,Twelve,Sixteen})
                for (const auto engine : {WmmaM16N32,WmmaM16N64})
                    result.push_back({engine,floor,
                        std::string(engine == WmmaM16N32 ? "N32" : "N64") +
                        "Floor" + std::to_string(static_cast<int>(floor))});
        return result;
    }
#endif

    /** @brief Reuse production geometry and source preparation for native tournaments.
     * @param parameter Exact existing model/control case. @param mode Timing or isolated profile owner. */
    void runExpertPipeline(Case parameter, ExpertBenchmark mode = ExpertBenchmark::Installed)
    {
        const bool native = mode != ExpertBenchmark::Installed;
        const bool hybrid = mode == ExpertBenchmark::WmmaHybridReplay ||
                            mode == ExpertBenchmark::WmmaHybridTournament ||
                            mode == ExpertBenchmark::WmmaHybrid32Profile;
        // One retained capacity spans every live threshold. The canonical case
        // still owns the matrix shapes, formats, ownership and route profile.
        if (hybrid) parameter.live_rows = parameter.rows;
#ifdef HAVE_ROCM
        const auto dot_candidates = groupedDotCandidates(hybrid);
        if (native)
        {
            initializeROCmHardwareMetadata();
            int ordinal = -1;
            for (int i = 0; i < int(rocmHardwareMetadata().size()); ++i)
            {
                const auto &isa = rocmHardwareMetadata().at(i).isa_target;
                if (isa == "gfx1100" || isa.starts_with("gfx1100:")) { ordinal = i; break; }
            }
            if (ordinal < 0) GTEST_SKIP() << "Native gfx1100 is not visible";
            parameter.device = DeviceId::rocm(ordinal);
            const auto &h = rocmHardwareMetadata().at(ordinal);
            std::printf("MOE_WMMA_DEVICE,%d,%s,%s,",ordinal,h.isa_target.c_str(),h.pci_address.c_str());
            for (auto byte : h.uuid) std::printf("%02x",unsigned(byte));
            std::printf("\n");
        }
#else
        if (native) GTEST_SKIP() << "ROCm support is not compiled";
#endif
        auto *backend = getBackendFor(parameter.device);
        if (!backend) GTEST_SKIP() << "Backend is not compiled into this build";
        if (backend->deviceCount() <= parameter.device.ordinal) GTEST_SKIP();
        auto &context = GPUDeviceContextPool::instance().getContext(parameter.device);
        context.submitAndWait([&] {
            const int hidden = parameter.hidden, intermediate = parameter.intermediate;
            const int experts = parameter.experts;
            constexpr int top_k = 8;
            auto *stream = context.defaultStream();
            ASSERT_NE(stream, nullptr);
            auto tensor = [&](int m, int n) {
                auto result = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(m), static_cast<size_t>(n)}, -0.125f, 0.125f, 721);
                if (!result->ensureOnDevice(parameter.device, stream))
                    throw std::runtime_error("floating economy tensor preparation failed");
                return result;
            };
            // Source views borrow shared ownership when preparing down-column
            // slices, so retain the source through TensorBase's shared lifetime.
            std::vector<std::shared_ptr<TensorBase>> sources;
            std::vector<GpuPreparedGemm> prepared;
            std::vector<DeviceMoEFloatingMatrixDesc> gates(experts), ups(experts), downs(experts);
            std::vector<DeviceNativeQuantisedMatrixDesc> quant_gates(experts), quant_ups(experts), quant_downs(experts);
            const bool floating = parameter.format == TensorType::FP16 ||
                parameter.format == TensorType::BF16 || parameter.format == TensorType::FP32;
            DeviceMoEWeightFormat format{};
            for (int expert = 0; expert < experts; ++expert)
            {
                for (int projection = 0; projection < 3; ++projection)
                {
                    const auto shape = projection == 2
                        ? std::vector<size_t>{static_cast<size_t>(hidden), static_cast<size_t>(intermediate)}
                        : std::vector<size_t>{static_cast<size_t>(intermediate), static_cast<size_t>(hidden)};
                    sources.push_back(weight(projection == 2 ? parameter.down_format.value_or(parameter.format)
                                                            : parameter.format,
                        shape, 1000 + expert * 3 + projection));
                    const auto name = "expert.economy." + std::to_string(expert) + "." + std::to_string(projection);
                    prepared.push_back(floating
                        ? makeGpuPreparedFloatingPointGemm(sources.back().get(), parameter.device, name)
                        : makeGpuPreparedGemm(sources.back().get(), parameter.device, name));
                }
                DeviceMoEExpertDescriptor descriptor{};
                ASSERT_TRUE(exportDeviceMoEExpertWeightDescriptors(prepared[expert * 3].kernel,
                    prepared[expert * 3 + 1].kernel, prepared[expert * 3 + 2].kernel,
                    hidden, intermediate, descriptor));
                gates[expert] = descriptor.floating_gate;
                ups[expert] = descriptor.floating_up;
                downs[expert] = descriptor.floating_down;
                quant_gates[expert] = descriptor.gate;
                quant_ups[expert] = descriptor.up;
                quant_downs[expert] = descriptor.down;
                format = descriptor.weight_format;
            }
            const auto requirements = parameter.device.is_cuda()
                ? MoEWorkspaceBuffers::cudaMoE(parameter.rows, hidden, intermediate, experts, top_k)
                : MoEWorkspaceBuffers::rocmMoE(parameter.rows, hidden, intermediate, experts, top_k);
            DeviceWorkspaceManager workspace(parameter.device,
                requirements.total_bytes_with_alignment() + 4 * 1024 * 1024);
            ASSERT_TRUE(workspace.allocate(requirements));
            auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(parameter.device);
            ASSERT_NE(kernel, nullptr);
            kernel->setGPUStream(stream);
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
            ASSERT_NE(consumer, nullptr);
            consumer->bindWorkspace(&workspace);
            const int gate_table = floating
                ? kernel->uploadGroupedExpertFloatingGateUpDescriptorTables(
                    gates.data(), ups.data(), format, experts, hidden, intermediate)
                : kernel->uploadGroupedExpertGateUpDescriptorTables(
                    quant_gates.data(), quant_ups.data(), experts, hidden, intermediate);
            const int down_table = floating
                ? kernel->uploadGroupedExpertFloatingDownDescriptorTable(
                    downs.data(), format, experts, hidden, intermediate)
                : kernel->uploadGroupedExpertDownDescriptorTable(
                    quant_downs.data(), experts, hidden, intermediate);
            ASSERT_GE(gate_table, 0);
            ASSERT_GE(down_table, 0);
            auto input = tensor(parameter.rows, hidden);
            auto ids = tensor(parameter.rows, top_k), probabilities = tensor(parameter.rows, top_k);
            auto output = tensor(parameter.rows, hidden), routes = tensor(parameter.rows * top_k, hidden);
            // Retain the complete route matrix: after timing owner-local
            // producers, column-owned down needs these SAME global routes.
            // Inventing new routes there would no longer test the same input.
            const auto global_routes = native_quantised_dispatch::makeMoERoutingIndices(
                parameter.route_profile, parameter.rows, top_k, experts);
            std::vector<float> route_ids(global_routes), route_weights(route_ids.size(), 0.125f);
            size_t active_routes = 0;
            for (size_t index = 0; index < route_ids.size(); ++index)
            {
                const int expert = static_cast<int>(global_routes[index]);
                const bool local = index / top_k < static_cast<size_t>(parameter.live_rows) &&
                    index % top_k < static_cast<size_t>(parameter.local_routes) &&
                    (parameter.ownership_degree == 0 || expert % parameter.ownership_degree == 0);
                route_ids[index] = local ? static_cast<float>(expert) : -1.0f;
                if (!local) route_weights[index] = 0.0f;
                else ++active_routes;
            }
            // Only uniform routing promises exactly equal local work. Skew is
            // intentional in the other profiles; retain its live route count
            // rather than quietly rebalancing the benchmark input.
            if (parameter.ownership_degree > 0 && parameter.route_profile == MoERoutingProfile::Uniform)
                ASSERT_EQ(active_routes, static_cast<size_t>(parameter.live_rows * top_k / parameter.ownership_degree));
            if (parameter.ownership_degree > 0)
                std::printf("moe_route_profile,%s,%s,%d,%d,%d,%zu\n",
                    parameter.device.is_cuda() ? "CUDA" : "ROCm",
                    std::string(native_quantised_dispatch::moeRoutingProfileName(parameter.route_profile)).c_str(),
                    parameter.ownership_degree, parameter.rows, parameter.live_rows, active_routes);
            ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(), route_ids.data(), route_ids.size() * sizeof(float), parameter.device.ordinal, stream));
            ASSERT_TRUE(backend->hostToDevice(probabilities->gpu_data_ptr(), route_weights.data(), route_weights.size() * sizeof(float), parameter.device.ordinal, stream));
            auto enqueue = [&] {
                return kernel->prepareExpertGroupsAsync(ids.get(), probabilities.get(), parameter.rows, experts, top_k) &&
                    kernel->executeGroupedPrefillPipeline(input.get(), output.get(), gate_table, down_table,
                        parameter.rows, hidden, intermediate, experts, top_k, routes.get()) &&
                    kernel->reduceCanonicalRouteContributions(routes.get(), output.get(), parameter.rows, top_k, hidden);
            };
            {
#ifdef HAVE_ROCM
                rocm::ScopedMoEGroupedPrefillEngine baseline(native ? rocm::MoEGroupedPrefillEngine::OrderedDot4
                                                                   : rocm::MoEGroupedPrefillEngine::Automatic);
#endif
                ASSERT_TRUE(enqueue());
            }
            std::vector<float> expected(parameter.rows * hidden), actual(expected.size());
            ASSERT_TRUE(backend->deviceToHost(expected.data(), output->gpu_data_ptr(), expected.size() * sizeof(float), parameter.device.ordinal, stream));
            ASSERT_TRUE(backend->synchronizeStream(stream, parameter.device.ordinal));
            ASSERT_TRUE(std::all_of(expected.begin(), expected.end(), [](float value) { return std::isfinite(value); }));
            if (parameter.live_rows > 0)
                ASSERT_TRUE(std::any_of(expected.begin(), expected.end(), [](float value) { return value != 0.0f; }));
            else
                ASSERT_TRUE(std::all_of(expected.begin(), expected.end(), [](float value) { return value == 0.0f; }));
            auto graph = context.createGraphCapture(stream);
            {
                ScopedBackendGraphCapture recording(context, *graph, "expert economy complete pipeline");
                ASSERT_TRUE(recording.begin());
#ifdef HAVE_ROCM
                rocm::ScopedMoEGroupedPrefillEngine baseline(native ? rocm::MoEGroupedPrefillEngine::OrderedDot4
                                                                   : rocm::MoEGroupedPrefillEngine::Automatic);
#endif
                for (int repeat = 0; repeat < (native ? 8 : 1); ++repeat) ASSERT_TRUE(enqueue());
                recording.finish();
            }
            ASSERT_TRUE(graph->instantiate());
            void *start = context.createEvent(GPUEventPurpose::Timing);
            void *end = context.createEvent(GPUEventPurpose::Timing);
            ASSERT_NE(start, nullptr);
            ASSERT_NE(end, nullptr);
            auto retire = [&](void *) {
                (void)backend->synchronizeStream(stream, parameter.device.ordinal);
                context.destroyEvent(end);
                context.destroyEvent(start);
            };
            std::unique_ptr<void, decltype(retire)> timing_owner(&context, retire);
#ifdef HAVE_ROCM
            if (native)
            {
                const bool profile = mode == ExpertBenchmark::Wmma32Profile || mode == ExpertBenchmark::Wmma64Profile ||
                                     mode == ExpertBenchmark::WmmaHybrid32Profile;
                if (profile && ::testing::UnitTest::GetInstance()->test_to_run_count() != 1)
                    throw std::invalid_argument("grouped WMMA profiling requires one exact selected case");
                constexpr int repeats = 8, sample_count = 31;
                std::vector<decltype(graph)> native_graphs;
                std::vector<decltype(graph.get())> graphs{graph.get()};
                // Retire submissions before these local native graphs disappear,
                // including fatal assertions after a launch or event failure.
                const auto join_graphs = [&](void *) { (void)backend->synchronizeStream(stream,parameter.device.ordinal); };
                std::unique_ptr<void,decltype(join_graphs)> graph_retirement(&context,join_graphs);
                for (size_t path = 1; path < dot_candidates.size(); ++path)
                {
                    native_graphs.push_back(context.createGraphCapture(stream));
                    graphs.push_back(native_graphs.back().get());
                    rocm::ScopedMoEGroupedPrefillEngine candidate(
                        dot_candidates[path].engine,dot_candidates[path].floor);
                    ScopedBackendGraphCapture capture(context,*graphs[path],"native grouped MoE component");
                    ASSERT_TRUE(capture.begin());
                    // Profiles isolate one complete operation, timing batches eight.
                    for (int repeat = 0; repeat < (profile ? 1 : repeats); ++repeat) ASSERT_TRUE(enqueue());
                    capture.finish();
                    std::vector<GPUGraphKernelNodeInfo> nodes;
                    ASSERT_TRUE(graphs[path]->inspectKernelNodes(nodes));
                    int native_nodes = 0;
                    std::set<std::tuple<uintptr_t,uint32_t,uint32_t,uint32_t,uint32_t>> seen;
                    for (const auto &node : nodes)
                    {
                        ASSERT_EQ(node.local_memory_bytes_per_thread,0u) << node.name;
                        if (node.name.find("grouped_wmma_detail") == std::string::npos) continue;
                        ++native_nodes;
                        if (seen.emplace(node.function_identity,node.grid_x,node.grid_y,node.grid_z,node.block_x).second)
                            std::printf("MOE_WMMA_RESOURCE,%s,%zu,%u,%zu,%u,%u,%u,%u,%u,%s\n",parameter.name.c_str(),path,
                                node.registers_per_thread,node.static_shared_memory_bytes,node.max_active_blocks_per_sm,
                                node.grid_x,node.grid_y,node.grid_z,node.block_x,node.name.c_str());
                    }
                    ASSERT_GE(native_nodes,2 * (profile ? 1 : repeats));
                    ASSERT_TRUE(graphs[path]->instantiate());
                }
                const auto poison = [&] {
                    for (const char *name : {MoEWorkspaceBuffers::PREFILL_A_INT8,MoEWorkspaceBuffers::PREFILL_A_SCALES,
                            MoEWorkspaceBuffers::PREFILL_GATE,MoEWorkspaceBuffers::PREFILL_UP,
                            MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8,MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES,
                            MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY})
                    {
                        void *bank = workspace.getBuffer(name);
                        const size_t bytes = workspace.getBufferSize(name);
                        if (!bank || !bytes || hipMemsetAsync(bank,0xa5,bytes,static_cast<hipStream_t>(stream)) != hipSuccess)
                            throw std::runtime_error("grouped WMMA scratch poison failed");
                    }
                    if (hipMemsetAsync(output->gpu_data_ptr(),0xa5,output->numel()*sizeof(float),static_cast<hipStream_t>(stream)) != hipSuccess ||
                        hipMemsetAsync(routes->gpu_data_ptr(),0xa5,routes->numel()*sizeof(float),static_cast<hipStream_t>(stream)) != hipSuccess)
                        throw std::runtime_error("grouped WMMA publication poison failed");
                };
                const auto verify = [&](int live) {
                    auto replay_ids = route_ids;
                    for (size_t i = static_cast<size_t>(live)*top_k; i < replay_ids.size(); ++i) replay_ids[i] = -1.0f;
                    ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(),replay_ids.data(),replay_ids.size()*sizeof(float),parameter.device.ordinal,stream));
                    auto reference = expected;
                    std::fill(reference.begin() + static_cast<size_t>(live)*hidden,reference.end(),0.0f);
                    for (size_t path = 0; path < graphs.size(); ++path)
                    {
                        poison();
                        // A missing replay must fail the independently established
                        // full-row reference, including empty positive-zero output.
                        ASSERT_TRUE(backend->deviceToHost(actual.data(),output->gpu_data_ptr(),actual.size()*sizeof(float),parameter.device.ordinal,stream));
                        ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                        ASSERT_NE(std::memcmp(reference.data(),actual.data(),actual.size()*sizeof(float)),0);
                        ASSERT_TRUE(graphs[path]->launch());
                        ASSERT_TRUE(backend->deviceToHost(actual.data(),output->gpu_data_ptr(),actual.size()*sizeof(float),parameter.device.ordinal,stream));
                        ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                        ASSERT_EQ(std::memcmp(reference.data(),actual.data(),actual.size()*sizeof(float)),0)
                            << "path=" << path << " live=" << live;
                    }
                };
                for (int live : {parameter.live_rows,0,1,std::min(8,parameter.live_rows),parameter.live_rows})
                { ASSERT_NO_FATAL_FAILURE(verify(live)); }
                for (int warmup = 0; warmup < 5; ++warmup)
                    for (auto *candidate : graphs) ASSERT_TRUE(candidate->launch());
                if (profile)
                {
                    ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                    ASSERT_GE(roctxRangePushA("ROCmGroupedMoEWmma"),0);
                    ASSERT_EQ(roctxProfilerResume(0),0);
                    const std::string profile_name = mode == ExpertBenchmark::WmmaHybrid32Profile ? "N32Floor8" :
                        mode == ExpertBenchmark::Wmma32Profile ? "N32" : "N64";
                    const auto selected = std::find_if(dot_candidates.begin(),dot_candidates.end(),
                        [&](const auto &candidate) { return candidate.name == profile_name; });
                    ASSERT_NE(selected,dot_candidates.end());
                    const bool launched = graphs[selected-dot_candidates.begin()]->launch();
                    const bool joined = backend->synchronizeStream(stream,parameter.device.ordinal);
                    const int paused = roctxProfilerPause(0), popped = roctxRangePop();
                    ASSERT_TRUE(launched && joined && paused == 0 && popped >= 0);
                    return;
                }
                const auto verify_both_writers = [&] {
                    std::vector<int> counts(experts);
                    for (float id : route_ids) if (id >= 0) ++counts[static_cast<int>(id)];
                    int mixed_policies = 0;
                    for (size_t path = 3; path < graphs.size(); ++path)
                    {
                        const int floor = static_cast<int>(dot_candidates[path].floor);
                        if (!std::any_of(counts.begin(),counts.end(),[&](int n) { return n >= floor; }) ||
                            !std::any_of(counts.begin(),counts.end(),[&](int n) { return n > 0 && n < floor; }))
                            continue;
                        ++mixed_policies;
                        auto *capture = dynamic_cast<HIPGraphCapture *>(graphs[path]);
                        ASSERT_NE(capture,nullptr);
                        size_t count = 0;
                        ASSERT_EQ(hipGraphGetNodes(capture->graph(),nullptr,&count),hipSuccess);
                        std::vector<hipGraphNode_t> handles(count);
                        ASSERT_EQ(hipGraphGetNodes(capture->graph(),handles.data(),&count),hipSuccess);
                        std::array<std::vector<hipGraphNode_t>,2> down_nodes;
                        for (auto handle : handles)
                        {
                            hipGraphNodeType type;
                            ASSERT_EQ(hipGraphNodeGetType(handle,&type),hipSuccess);
                            if (type != hipGraphNodeTypeKernel) continue;
                            hipKernelNodeParams params{};
                            ASSERT_EQ(hipGraphKernelNodeGetParams(handle,&params),hipSuccess);
                            const char *name = hipKernelNameRefByPtr(params.func,static_cast<hipStream_t>(stream));
                            ASSERT_NE(name,nullptr);
                            const std::string_view symbol(name);
                            if (symbol.find("moe_grouped_down_batch_invariant_tiled_prefill_kernel") != std::string_view::npos)
                                down_nodes[0].push_back(handle);
                            if (symbol.find("grouped_wmma_detail") != std::string_view::npos && params.gridDim.z == 1)
                                down_nodes[1].push_back(handle);
                        }
                        // Independently remove each writer from every repeated
                        // operation. The other family must not repair its rows.
                        for (const auto &family : down_nodes)
                        {
                            ASSERT_FALSE(family.empty());
                            for (auto node : family)
                                ASSERT_EQ(hipGraphNodeSetEnabled(capture->executable(),node,0),hipSuccess);
                            poison();
                            ASSERT_TRUE(graphs[path]->launch());
                            ASSERT_TRUE(backend->deviceToHost(actual.data(),output->gpu_data_ptr(),actual.size()*sizeof(float),parameter.device.ordinal,stream));
                            ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                            for (auto node : family)
                                ASSERT_EQ(hipGraphNodeSetEnabled(capture->executable(),node,1),hipSuccess);
                            ASSERT_NE(std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float)),0)
                                << "missing writer was repaired: " << dot_candidates[path].name;
                        }
                    }
                    ASSERT_GT(mixed_policies,0);
                };
                if (hybrid && parameter.route_profile == MoERoutingProfile::PowerLaw)
                    ASSERT_NO_FATAL_FAILURE(verify_both_writers());
                if (hybrid && parameter.ownership_degree == 0 && experts == top_k)
                {
                    const auto original_ids = route_ids;
                    const auto original_reference = expected;
                    // E8/top-k8 uniform routing gives every expert the same
                    // count. This separate partial-owner histogram keeps the
                    // tiled family active while straddling every native floor.
                    // Each row still contains at most one route to each expert.
                    constexpr std::array counts{3,4,7,8,12,16,17,31};
                    for (size_t i = 0; i < route_ids.size(); ++i)
                        if (route_ids[i] >= 0 && i/top_k >= static_cast<size_t>(counts[static_cast<int>(route_ids[i])]))
                            route_ids[i] = -1.0f;
                    ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(),route_ids.data(),route_ids.size()*sizeof(float),parameter.device.ordinal,stream));
                    ASSERT_TRUE(graphs[0]->launch());
                    ASSERT_TRUE(backend->deviceToHost(expected.data(),output->gpu_data_ptr(),expected.size()*sizeof(float),parameter.device.ordinal,stream));
                    ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                    ASSERT_TRUE(std::all_of(expected.begin(),expected.end(),[](float n) { return std::isfinite(n); }));
                    ASSERT_TRUE(std::any_of(expected.begin(),expected.end(),[](float n) { return n != 0; }));
                    ASSERT_NO_FATAL_FAILURE(verify(parameter.live_rows));
                    ASSERT_NO_FATAL_FAILURE(verify_both_writers());
                    for (int live : {0,1,17,31,parameter.live_rows})
                        ASSERT_NO_FATAL_FAILURE(verify(live));
                    route_ids = original_ids;
                    expected = original_reference;
                }
                const bool replay_only = mode == ExpertBenchmark::WmmaReplay || mode == ExpertBenchmark::WmmaHybridReplay;
                std::vector<int> live_points{parameter.live_rows};
                if (hybrid)
                {
                    live_points = {0,1,4,8,12,16,17,31,32,48,64,68,124,128,256,384,512};
                    std::erase_if(live_points,[&](int n) { return n > parameter.live_rows; });
                }
                for (int live : live_points)
                {
                    ASSERT_NO_FATAL_FAILURE(verify(live));
                    if (replay_only) continue;
                    auto reference = expected;
                    std::fill(reference.begin()+static_cast<size_t>(live)*hidden,reference.end(),0.0f);
                    for (int warmup = 0; warmup < 3; ++warmup)
                        for (auto *candidate : graphs) ASSERT_TRUE(candidate->launch());
                    std::vector<std::array<double,sample_count>> times(graphs.size());
                    for (int sample = 0; sample < sample_count; ++sample)
                    for (size_t order = 0; order < graphs.size(); ++order)
                    {
                        const size_t path = (sample + (sample % 2 ? graphs.size()-1-order : order)) % graphs.size();
                        ASSERT_TRUE(context.recordEventChecked(start,stream));
                        ASSERT_TRUE(graphs[path]->launch());
                        ASSERT_TRUE(context.recordEventChecked(end,stream));
                        ASSERT_TRUE(context.synchronizeEventChecked(end));
                        const double us = context.eventElapsedTime(start,end)*1000.0/repeats;
                        ASSERT_TRUE(std::isfinite(us) && us > 0);
                        times[path][sample] = us;
                        // Inspect this candidate before the next can repair its output.
                        ASSERT_TRUE(backend->deviceToHost(actual.data(),output->gpu_data_ptr(),actual.size()*sizeof(float),parameter.device.ordinal,stream));
                        ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                        ASSERT_EQ(std::memcmp(reference.data(),actual.data(),actual.size()*sizeof(float)),0)
                            << dot_candidates[path].name << " live=" << live;
                    }
                    for (int sample = 0; sample < sample_count; ++sample)
                        if (hybrid)
                            for (size_t path = 0; path < graphs.size(); ++path)
                                std::printf("MOE_WMMA_HYBRID_SAMPLE,%s,%d,%s,%d,%.9f\n",parameter.name.c_str(),live,
                                    dot_candidates[path].name.c_str(),sample,times[path][sample]);
                        else
                            std::printf("MOE_WMMA_SAMPLE,%s,%d,%.9f,%.9f,%.9f\n",parameter.name.c_str(),sample,
                                times[0][sample],times[1][sample],times[2][sample]);
                    for (auto &values : times) std::sort(values.begin(),values.end());
                    if (hybrid)
                        for (size_t path = 0; path < graphs.size(); ++path)
                            std::printf("MOE_WMMA_HYBRID_MEDIAN,%s,%d,%s,%.9f,%.6f\n",parameter.name.c_str(),live,
                                dot_candidates[path].name.c_str(),times[path][15],times[0][15]/times[path][15]);
                    else
                        std::printf("MOE_WMMA_MEDIAN,%s,%.9f,%.9f,%.9f,%.6f,%.6f\n",parameter.name.c_str(),
                            times[0][15],times[1][15],times[2][15],times[0][15]/times[1][15],times[0][15]/times[2][15]);
                }
                ASSERT_NO_FATAL_FAILURE(verify(0));
                ASSERT_NO_FATAL_FAILURE(verify(parameter.live_rows));
            }
#endif
            std::array<float, 9> samples{};
            for (size_t sample = 0; !native && sample < samples.size() + 3u; ++sample)
            {
                ASSERT_TRUE(context.recordEventChecked(start, stream));
                ASSERT_TRUE(graph->launchOnStream(stream));
                ASSERT_TRUE(context.recordEventChecked(end, stream));
                ASSERT_TRUE(context.synchronizeEventChecked(end));
                if (sample >= 3u) samples[sample - 3u] = context.eventElapsedTime(start, end);
            }
            if (!native && parameter.ownership_degree > 0)
                for (size_t sample = 0; sample < samples.size(); ++sample)
                    std::printf("moe_whole_shard_sample,%s,%d,%d,%d,%zu,%.9f\n",
                        parameter.device.is_cuda() ? "CUDA" : "ROCm", parameter.ownership_degree,
                        parameter.rows, parameter.live_rows, sample, samples[sample] * 1000.0);
            std::sort(samples.begin(), samples.end());
            if (!native) std::printf("MOE_EXPERT_ECONOMY,%s,%.3f\n", parameter.name.c_str(), samples[samples.size() / 2u] * 1000.0f);
            ASSERT_TRUE(backend->deviceToHost(actual.data(), output->gpu_data_ptr(), actual.size() * sizeof(float), parameter.device.ordinal, stream));
            ASSERT_TRUE(backend->synchronizeStream(stream, parameter.device.ordinal));
            EXPECT_EQ(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)), 0);

            if (parameter.ownership_degree == 0 && !native)
                return;
            const auto original_ids = route_ids, original_weights = route_weights;
            const auto original_expected = expected;
            // All candidates bind the same immutable source-prepared slice.
            // Repacking per policy would repeat setup work and obscure the
            // retained-buffer ownership this replay is intended to qualify.
            const int columns = hidden / (parameter.ownership_degree ? parameter.ownership_degree : 2);
            const int first_column = hidden - columns;
            std::vector<std::shared_ptr<TensorBase>> column_sources;
            std::vector<GpuPreparedGemm> column_prepared;
            std::vector<DeviceNativeQuantisedMatrixDesc> column_descriptors(experts);
            for (int expert = 0; expert < experts; ++expert)
            {
                // Slice SOURCE rows before packing. Reinterpreting a complete
                // packed matrix with a smaller N would use the wrong layout.
                column_sources.push_back(sources[expert * 3 + 2]->create_view(
                    {static_cast<size_t>(columns), static_cast<size_t>(intermediate)},
                    static_cast<size_t>(first_column) * intermediate));
                column_prepared.push_back(makeGpuPreparedGemm(column_sources.back().get(), parameter.device,
                    "expert.economy.fixed_down." + std::to_string(expert)));
                ASSERT_TRUE(column_prepared.back().kernel->exportNativeQuantisedMatrixDesc(column_descriptors[expert]));
            }
            const int column_table = kernel->uploadGroupedExpertDownDescriptorTable(
                column_descriptors.data(), experts, columns, intermediate);
            ASSERT_GE(column_table, 0);
            auto column_routes = tensor(parameter.rows * top_k, columns);
            auto column_output = tensor(parameter.rows, columns);
            const auto check_phases = [&](const char *engine_label) {
            route_ids = original_ids; route_weights = original_weights; expected = original_expected;
            ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(),route_ids.data(),route_ids.size()*sizeof(float),parameter.device.ordinal,stream));
            ASSERT_TRUE(backend->hostToDevice(probabilities->gpu_data_ptr(),route_weights.data(),route_weights.size()*sizeof(float),parameter.device.ordinal,stream));
            ASSERT_TRUE(enqueue());
            // Public projection boundaries preserve the same grouped route
            // plan and Q8 intermediates as the complete production operation.
            // Time a retained batch, never individual arbitrary graph nodes:
            // grouping's count/scatter kernels require their preceding clears.
            const auto measure_phase = [&](const char *ownership, const char *phase, const auto &operation) {
                // Preserve the historical uniform record identity, but never
                // merge samples from different route distributions in a curve.
                const std::string curve_ownership = std::string(engine_label) + ownership +
                    (parameter.route_profile == MoERoutingProfile::Uniform ? "" :
                        "_" + std::string(native_quantised_dispatch::moeRoutingProfileName(parameter.route_profile)));
                constexpr int repetitions = 8;
                // Join the phase's exact external producer before recording.
                // In particular, the first column fold must import the down
                // graph's publication; a device-wide join does not replace
                // TransferEngine's explicit producer/consumer contract.
                ASSERT_TRUE(operation());
                auto phase_graph = context.createGraphCapture(stream);
                {
                    ScopedBackendGraphCapture recording(context, *phase_graph, "expert economy phase");
                    ASSERT_TRUE(recording.begin());
                    for (int repetition = 0; repetition < repetitions; ++repetition)
                        ASSERT_TRUE(operation());
                    recording.finish();
                }
                ASSERT_TRUE(phase_graph->instantiate());
                std::vector<GPUGraphKernelNodeInfo> nodes;
                std::string inspection_error;
                ASSERT_TRUE(phase_graph->inspectKernelNodes(nodes, &inspection_error)) << inspection_error;
                std::set<std::tuple<uintptr_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> reported;
                for (const auto &node : nodes)
                {
                    ASSERT_TRUE(node.valid());
                    if (reported.emplace(node.function_identity, node.grid_x, node.grid_y, node.grid_z,
                            node.block_x, node.block_y, node.block_z).second)
                        std::printf("moe_phase_resource,%s,%s,%s,%d,%d,%d,%u,%u,%u,%zu,%u,%s\n",
                            parameter.device.is_cuda() ? "CUDA" : "ROCm", curve_ownership.c_str(), phase,
                            parameter.ownership_degree, parameter.rows, parameter.live_rows,
                            node.grid_x * node.grid_y * node.grid_z,
                            node.block_x * node.block_y * node.block_z,
                            node.registers_per_thread, node.local_memory_bytes_per_thread,
                            node.max_active_blocks_per_sm, node.name.c_str());
                }
                for (int sample = -3; sample < (native ? 0 : 9); ++sample)
                {
                    ASSERT_TRUE(context.recordEventChecked(start, stream));
                    ASSERT_TRUE(phase_graph->launchOnStream(stream));
                    ASSERT_TRUE(context.recordEventChecked(end, stream));
                    ASSERT_TRUE(context.synchronizeEventChecked(end));
                    if (sample >= 0)
                        std::printf("moe_phase_sample,%s,%s,%s,%d,%d,%d,%d,%.9f\n",
                            parameter.device.is_cuda() ? "CUDA" : "ROCm", curve_ownership.c_str(), phase,
                            parameter.ownership_degree, parameter.rows, parameter.live_rows, sample,
                            context.eventElapsedTime(start, end) * 1000.0 / repetitions);
                }
            };
            const auto group = [&] {
                return kernel->prepareExpertGroupsAsync(ids.get(), probabilities.get(), parameter.rows, experts, top_k);
            };
            const auto gate_up = [&] {
                return kernel->executeGroupedPrefillProjection(input.get(), nullptr, gate_table, down_table,
                    parameter.rows, hidden, intermediate, experts, top_k,
                    MoEPrefillProjectionExecution::gateUp(hidden));
            };
            const auto down = [&] {
                return kernel->executeGroupedPrefillProjection(nullptr, nullptr, gate_table, down_table,
                    parameter.rows, hidden, intermediate, experts, top_k,
                    MoEPrefillProjectionExecution::down(hidden, 0, hidden), routes.get());
            };
            const auto fold = [&] {
                return kernel->reduceCanonicalRouteContributions(routes.get(), output.get(), parameter.rows, top_k, hidden);
            };
            ASSERT_NO_FATAL_FAILURE(measure_phase("whole_expert", "pipeline", enqueue));
            ASSERT_NO_FATAL_FAILURE(measure_phase("whole_expert", "grouping", group));
            ASSERT_NO_FATAL_FAILURE(measure_phase("whole_expert", "gate_up", gate_up));
            ASSERT_NO_FATAL_FAILURE(measure_phase("whole_expert", "down", down));
            ASSERT_NO_FATAL_FAILURE(measure_phase("whole_expert", "fold", fold));
            ASSERT_TRUE(backend->deviceToHost(actual.data(), output->gpu_data_ptr(), actual.size() * sizeof(float), parameter.device.ordinal, stream));
            ASSERT_TRUE(backend->synchronizeStream(stream, parameter.device.ordinal));
            ASSERT_EQ(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)), 0)
                << "Separately captured public phases changed the complete pipeline result";

            // Column ownership has DIFFERENT route geometry: every participant
            // computes down columns for ALL routes after intermediate exchange.
            // Prepare that complete intermediate state once, outside timing.
            // This local probe does not pretend to measure transport/assembly.
            for (size_t index = 0; index < route_ids.size(); ++index)
            {
                const bool live = index / top_k < static_cast<size_t>(parameter.live_rows);
                route_ids[index] = live ? global_routes[index] : -1.f;
                route_weights[index] = live ? 0.125f : 0.f;
            }
            ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(), route_ids.data(), route_ids.size() * sizeof(float), parameter.device.ordinal, stream));
            ASSERT_TRUE(backend->hostToDevice(probabilities->gpu_data_ptr(), route_weights.data(), route_weights.size() * sizeof(float), parameter.device.ordinal, stream));
            {
#ifdef HAVE_ROCM
                rocm::ScopedMoEGroupedPrefillEngine oracle(native ? rocm::MoEGroupedPrefillEngine::OrderedDot4
                                                                 : rocm::MoEGroupedPrefillEngine::Automatic);
#endif
                ASSERT_TRUE(enqueue());
            }
            ASSERT_TRUE(backend->deviceToHost(expected.data(), output->gpu_data_ptr(), expected.size() * sizeof(float), parameter.device.ordinal, stream));
            ASSERT_TRUE(backend->synchronizeStream(stream, parameter.device.ordinal));

            const auto column_down = [&] {
                return kernel->executeGroupedPrefillProjection(nullptr, nullptr, gate_table, column_table,
                    parameter.rows, hidden, intermediate, experts, top_k,
                    MoEPrefillProjectionExecution::down(hidden, first_column, columns), column_routes.get());
            };
            const auto column_fold = [&] {
                return kernel->reduceCanonicalRouteContributions(column_routes.get(), column_output.get(),
                    parameter.rows, top_k, columns);
            };
            ASSERT_NO_FATAL_FAILURE(measure_phase("down_columns", "grouping", group));
            // Grouping may legitimately reorder the packed rows on another
            // replay. Production imports intermediates through the NEW route
            // map after grouping; this transport-free probe regenerates that
            // producer once, outside down timing. Never consume values packed
            // for an earlier grouping, even when the router IDs are identical.
            ASSERT_TRUE(gate_up());
            ASSERT_NO_FATAL_FAILURE(measure_phase("down_columns", "down", column_down));
            ASSERT_NO_FATAL_FAILURE(measure_phase("down_columns", "fold", column_fold));
            std::vector<float> actual_columns(parameter.rows * columns);
            ASSERT_TRUE(backend->deviceToHost(actual_columns.data(), column_output->gpu_data_ptr(),
                actual_columns.size() * sizeof(float), parameter.device.ordinal, stream));
            ASSERT_TRUE(backend->synchronizeStream(stream, parameter.device.ordinal));
            for (int row = 0; row < parameter.rows; ++row)
                ASSERT_EQ(std::memcmp(expected.data() + row * hidden + first_column,
                    actual_columns.data() + row * columns, columns * sizeof(float)), 0)
                    << "Column-owned down disagrees with full-width oracle at row " << row;
            };
#ifdef HAVE_ROCM
            if (native)
            {
                for (size_t path = 1; path < dot_candidates.size(); ++path)
                {
                    rocm::ScopedMoEGroupedPrefillEngine selected(dot_candidates[path].engine,dot_candidates[path].floor);
                    ASSERT_NO_FATAL_FAILURE(check_phases((dot_candidates[path].name+"_").c_str()));
                }
            }
            else
#endif
                ASSERT_NO_FATAL_FAILURE(check_phases(""));
        });
    }

    /** @test Preserve the existing installed-policy economy surface. */
    TEST_P(GPUExpertPipeline, CapturedPipeline) { runExpertPipeline(GetParam()); }

    /** @brief Opt-in native tournament reuses existing prefill shapes and source formats. */
    class ROCmGroupedWmmaPerf : public ::testing::TestWithParam<Case> {};
    /** @return Existing quantized ROCm prefill cases, without a private model inventory. */
    std::vector<Case> groupedWmmaPerfCases()
    {
        auto result = cases();
        std::erase_if(result,[](const Case &c) {
            return !c.device.is_rocm() || c.rows < 64 || c.format == TensorType::FP16 ||
                c.format == TensorType::BF16 || c.format == TensorType::FP32;
        });
        return result;
    }
    /** @test Replay both native engines through split phases and source-prepared down slices. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_CapturedSplitReplay)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaReplay); }
    /** @test Pair whole-component native-event samples only after dirty replay passes. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_CapturedTournament)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaTournament); }
    /** @test Cross whole-expert thresholds in retained graphs and prove both writers contribute. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_HybridCapturedReplay)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaHybridReplay); }
    /** @test Time complete hybrid policies across device-published live work. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_HybridCapturedTournament)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaHybridTournament); }
    /** @test Isolate one N32 complete projection transaction outside canonical timing. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_IsolatedN32Profile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::Wmma32Profile); }
    /** @test Isolate one N64 complete projection transaction outside canonical timing. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_IsolatedN64Profile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::Wmma64Profile); }
    /** @test Profile one complete mixed-writer transaction, independently of event timing. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_IsolatedHybridN32Floor8Profile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaHybrid32Profile); }
    INSTANTIATE_TEST_SUITE_P(NativeGfx1100,ROCmGroupedWmmaPerf,::testing::ValuesIn(groupedWmmaPerfCases()),
        [](const ::testing::TestParamInfo<Case> &info) { return info.param.name; });

    INSTANTIATE_TEST_SUITE_P(ProductionShape, GPUExpertPipeline, ::testing::ValuesIn(cases()),
        [](const ::testing::TestParamInfo<Case> &info) { return info.param.name; });

    /**
     * @brief Isolate canonical route folding without expert-weight setup.
     *
     * One captured production call folds FP32 route contributions in router
     * order. Cancellation-sensitive inputs make reassociated arithmetic fail
     * the independent CPU oracle. These probes cover partial columns, full and
     * partial load windows, verifier rows, and the main/suffix prefill shapes.
     * No timing threshold is a correctness assertion. The complete imported
     * route tensor is prepared outside timing; no collective is executed.
     *
     * @param device Exact participant used for the entire local shape curve.
     * @param row_counts Captured row extents; one value isolates profiler work.
     * @param column_counts Local output widths, not allocation capacities.
     * @param route_counts Live routes folded in their unchanged serial order.
     */
    void measureCanonicalRouteFold(DeviceId device,
        std::initializer_list<int> row_counts = {1, 16, 64, 512},
        std::initializer_list<int> column_counts = {65, 512, 1024, 2048},
        std::initializer_list<int> route_counts = {1, 3, 7, 8, 9, 17})
    {
        auto *backend = getBackendFor(device);
        if (!backend || backend->deviceCount() <= device.ordinal) GTEST_SKIP();
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&] {
            void *stream = context.defaultStream();
            ASSERT_NE(stream, nullptr);
            auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
            ASSERT_NE(kernel, nullptr);
            kernel->setGPUStream(stream);
            for (const int rows : row_counts)
                for (const int columns : column_counts)
                    for (const int top_k : route_counts)
                    {
                        SCOPED_TRACE(::testing::Message() << device.to_string()
                            << " rows=" << rows << " columns=" << columns << " top_k=" << top_k);
                        auto input = TestTensorFactory::createFP32Zeros(
                            {static_cast<size_t>(rows * top_k), static_cast<size_t>(columns)});
                        auto output = TestTensorFactory::createFP32Zeros(
                            {static_cast<size_t>(rows), static_cast<size_t>(columns)});
                        // Different signs/magnitudes expose a parallel tree or
                        // an accidental reset between consecutive load windows.
                        constexpr std::array<float, 8> pattern{
                            16777216.0f, 1.0f, -16777216.0f, 0.25f,
                            -0.5f, 3.0f, -0.125f, 0.0625f};
                        auto *source = input->mutable_data();
                        std::vector<float> expected(static_cast<size_t>(rows) * columns);
                        for (int row = 0; row < rows; ++row)
                            for (int column = 0; column < columns; ++column)
                            {
                                float sum = 0.0f;
                                for (int route = 0; route < top_k; ++route)
                                {
                                    const float value = pattern[(route + row + column) % pattern.size()];
                                    source[(static_cast<size_t>(row) * top_k + route) * columns + column] = value;
                                    sum += value;
                                }
                                expected[static_cast<size_t>(row) * columns + column] = sum;
                            }
                        ASSERT_TRUE(input->ensureOnDevice(device, stream));
                        ASSERT_TRUE(output->ensureOnDevice(device, stream));
                        const auto enqueue = [&] {
                            return kernel->reduceCanonicalRouteContributions(
                                input.get(), output.get(), rows, top_k, columns);
                        };
                        ASSERT_TRUE(enqueue());
                        auto graph = context.createGraphCapture(stream);
                        {
                            ScopedBackendGraphCapture recording(context, *graph, "canonical route-fold economy");
                            ASSERT_TRUE(recording.begin());
                            for (int repeat = 0; repeat < 8; ++repeat) ASSERT_TRUE(enqueue());
                            recording.finish();
                        }
                        ASSERT_TRUE(graph->instantiate());
                        std::vector<GPUGraphKernelNodeInfo> nodes;
                        std::string inspection_error;
                        ASSERT_TRUE(graph->inspectKernelNodes(nodes, &inspection_error)) << inspection_error;
                        ASSERT_EQ(nodes.size(), 8u);
                        const auto &node = nodes.front();
                        ASSERT_TRUE(node.valid());
                        std::printf("route_fold_resource,%s,%d,%d,%d,%u,%u,%u,%zu,%u,%s\n",
                            device.is_cuda() ? "CUDA" : "ROCm", rows, columns, top_k,
                            node.grid_x * node.grid_y * node.grid_z,
                            node.block_x * node.block_y * node.block_z,
                            node.registers_per_thread, node.local_memory_bytes_per_thread,
                            node.max_active_blocks_per_sm, node.name.c_str());
                        void *start = context.createEvent(GPUEventPurpose::Timing);
                        void *end = context.createEvent(GPUEventPurpose::Timing);
                        ASSERT_NE(start, nullptr);
                        ASSERT_NE(end, nullptr);
                        auto retire = [&](void *) {
                            (void)backend->synchronizeStream(stream, device.ordinal);
                            context.destroyEvent(end);
                            context.destroyEvent(start);
                        };
                        std::unique_ptr<void, decltype(retire)> timing_owner(&context, retire);
                        for (int sample = -3; sample < 9; ++sample)
                        {
                            ASSERT_TRUE(context.recordEventChecked(start, stream));
                            ASSERT_TRUE(graph->launchOnStream(stream));
                            ASSERT_TRUE(context.recordEventChecked(end, stream));
                            ASSERT_TRUE(context.synchronizeEventChecked(end));
                            if (sample >= 0)
                                std::printf("route_fold_sample,%s,%d,%d,%d,%d,%.9f\n",
                                    device.is_cuda() ? "CUDA" : "ROCm", rows, columns, top_k,
                                    sample, context.eventElapsedTime(start, end) * 1000.0 / 8);
                        }
                        std::vector<float> actual(expected.size());
                        ASSERT_TRUE(backend->deviceToHost(actual.data(), output->gpu_data_ptr(),
                            actual.size() * sizeof(float), device.ordinal, stream));
                        ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                        ASSERT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0)
                            << "Captured fold changed the independent serial router-order result";
                    }
        });
    }

    /** @brief CUDA communication-free route-fold byte/economy shape sweep. */
    TEST(GPUExpertPipelinePerf, CUDACanonicalRouteFoldCapturedScaling)
    {
        measureCanonicalRouteFold(DeviceId::cuda(0));
    }

    /** @brief ROCm counterpart with the same source bits and local shapes. */
    TEST(GPUExpertPipelinePerf, ROCmCanonicalRouteFoldCapturedScaling)
    {
        measureCanonicalRouteFold(DeviceId::rocm(0));
    }

    /** @brief One exact ROCm main-prefill shard, separate from canonical timing. */
    TEST(GPUExpertPipelinePerf, ROCmCanonicalRouteFoldExactProfilerLaunch)
    {
        measureCanonicalRouteFold(DeviceId::rocm(0), {512}, {1024}, {8});
    }

    /** @brief Model-free route preparation timing, separate from expert arithmetic. */
    class GPURuntimeGroupingPerf : public ::testing::TestWithParam<std::tuple<std::string, int>> {};

    /**
     * @test Time the public captured grouping operation, including clear/count/scatter.
     *
     * The balanced input visits every expert; no weight upload or projection is
     * timed. Every grouped route is checked against the host's stable ordering
     * after timing. One selectable shape/backend per process supports isolated
     * profiler evidence without contaminating the canonical event samples.
     */
    TEST_P(GPURuntimeGroupingPerf, CapturedStableGrouping)
    {
        const auto [backend_name, rows] = GetParam();
        const auto device = backend_name == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&] {
            constexpr int experts = 256, top_k = 8;
            const int slots = rows * top_k;
            auto *stream = context.defaultStream();
            ASSERT_NE(stream, nullptr);
            DeviceMoERuntimeTable::Config config;
            config.device_id = device;
            config.num_layers = 1;
            config.num_experts = experts;
            config.top_k = top_k;
            config.mirror_to_device = true;
            config.prefill_token_capacity = rows;
            MoERuntimeTable runtime(config);
            const auto state = runtime.hostLayerState(0);
            const auto requirements = device.is_cuda()
                ? MoEWorkspaceBuffers::cudaMoE(rows, 256, 256, experts, top_k)
                : MoEWorkspaceBuffers::rocmMoE(rows, 256, 256, experts, top_k);
            DeviceWorkspaceManager workspace(device,
                requirements.total_bytes_with_alignment() + 4 * 1024 * 1024);
            ASSERT_TRUE(workspace.allocate(requirements));
            auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
            ASSERT_NE(kernel, nullptr);
            kernel->setGPUStream(stream);
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
            ASSERT_NE(consumer, nullptr);
            consumer->bindWorkspace(&workspace);
            auto indices = TestTensorFactory::createFP32({static_cast<size_t>(rows), top_k});
            auto weights = TestTensorFactory::createFP32({static_cast<size_t>(rows), top_k});
            for (int slot = 0; slot < slots; ++slot)
            {
                indices->mutable_data()[slot] = static_cast<float>((slot * 37) % experts);
                weights->mutable_data()[slot] = static_cast<float>(slot + 1) / 65536.0f;
            }
            ASSERT_TRUE(indices->ensureOnDevice(device, stream));
            ASSERT_TRUE(weights->ensureOnDevice(device, stream));
            const auto enqueue = [&] {
                return kernel->groupPrefillRoutes(runtime.deviceLayerState(0),
                    indices.get(), weights.get(), rows, rows, experts, top_k);
            };
            ASSERT_TRUE(enqueue());
            ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
            auto graph = context.createGraphCapture(stream);
            ASSERT_NE(graph, nullptr);
            {
                GraphCaptureGuard recording;
                ASSERT_TRUE(graph->beginCapture());
                ASSERT_TRUE(enqueue());
                ASSERT_TRUE(graph->endCapture());
            }
            ASSERT_TRUE(graph->instantiate());
            void *start = context.createEvent(GPUEventPurpose::Timing);
            void *end = context.createEvent(GPUEventPurpose::Timing);
            ASSERT_NE(start, nullptr);
            ASSERT_NE(end, nullptr);
            auto retire = [&](void *) {
                (void)backend->synchronizeStream(stream, device.ordinal);
                context.destroyEvent(end);
                context.destroyEvent(start);
            };
            std::unique_ptr<void, decltype(retire)> timing_owner(&context, retire);
            std::array<float, 31> samples{};
            constexpr int replays_per_sample = 20;
            for (size_t sample = 0; sample < samples.size() + 5u; ++sample)
            {
                ASSERT_TRUE(context.recordEventChecked(start, stream));
                for (int replay = 0; replay < replays_per_sample; ++replay)
                    ASSERT_TRUE(graph->launchOnStream(stream));
                ASSERT_TRUE(context.recordEventChecked(end, stream));
                ASSERT_TRUE(context.synchronizeEventChecked(end));
                if (sample >= 5u)
                    samples[sample - 5u] = context.eventElapsedTime(start, end) * 1000.0f / replays_per_sample;
            }
            std::sort(samples.begin(), samples.end());
            std::printf("MOE_GROUPING_ECONOMY,%s,%d,%.3f\n", backend_name.c_str(), rows, samples[samples.size() / 2u]);
            std::vector<int32_t> grouped_rows(slots);
            std::vector<float> grouped_weights(slots);
            ASSERT_TRUE(backend->deviceToHost(grouped_rows.data(), state.grouped_token_ids,
                slots * sizeof(int32_t), device.ordinal, stream));
            ASSERT_TRUE(backend->deviceToHost(grouped_weights.data(), state.grouped_route_weights,
                slots * sizeof(float), device.ordinal, stream));
            ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
            int grouped = 0;
            for (int expert = 0; expert < experts; ++expert)
                for (int slot = 0; slot < slots; ++slot)
                    if ((slot * 37) % experts == expert)
                    {
                        ASSERT_EQ(grouped_rows[grouped], slot);
                        ASSERT_EQ(grouped_weights[grouped], static_cast<float>(slot + 1) / 65536.0f);
                        ++grouped;
                    }
            ASSERT_EQ(grouped, slots);
        });
    }

    INSTANTIATE_TEST_SUITE_P(Grouping, GPURuntimeGroupingPerf,
        ::testing::Combine(::testing::Values("CUDA", "ROCm"), ::testing::Values(32, 33, 64, 512, 2048)),
        [](const ::testing::TestParamInfo<std::tuple<std::string, int>> &info) {
            return std::get<0>(info.param) + "_Rows" + std::to_string(std::get<1>(info.param));
        });
}
