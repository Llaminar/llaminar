/**
 * @file Perf__GPUExpertPipeline.cpp
 * @brief Captured all-format routed-expert pipeline economy on both GPUs.
 *
 * Each independently selectable case has real 122B or Qwen 3.6 35B projection
 * geometry. Qwen shard probes additionally isolate complete public phases and
 * the fixed-down column bank, under shared uniform/hotset/power-law profiles;
 * these are communication-free local costs, not multi-GPU speedups. Every
 * downstream phase receives valid producer data prepared outside its timer.
 * Ordinary-prefill probes retain the real router Q8 producer and direct token
 * publication, with explicit injected expert routes for controlled histograms.
 * Those timings include router work; they are not complete model measurements.
 * Timing includes device grouping, gate/up/SwiGLU, canonical down
 * publication and reduction. Weight preparation, graph construction and result
 * downloads are outside event timing. Numerical certification belongs to the
 * independent serial oracle in MoELiveRows; this fixture additionally requires
 * the timed replay sequence to leave the same finite output bits. No performance
 * threshold is a correctness gate; focused event-owner replay is explicitly
 * registered in production preflight.
 * Opt-in gfx1100 tournaments use the same source/geometry owners, compare both
 * native tiles against explicit ordered Dot4 after poisoned retained replay,
 * and keep profiler launches separate from paired native-event samples.
 * Hybrid experiments retain all row-floor graphs while route data crosses the
 * thresholds. Skewed replay independently suppresses each selected down branch;
 * neither a matching symbol nor a repaired final result can certify ownership.
 * Native tournaments wrap complete retained operation batches in internal
 * timeline events. Each sample must overwrite poisoned state before another
 * candidate can repair it; host submission gaps stay outside the interval.
 * Seeded host sources are synthesized independently on physical CPU workers;
 * quantized preparation batches the original matrices through one production
 * loader while retaining individual prepared handles and source lifetimes.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/HardwareInventory.h"
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
#include "utils/PerfStatsCollector.h"
#include "performance/kernels/native_quantised_dispatch/NativeQuantisedMoERoutingProfiles.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include "backends/rocm/HIPGraphCapture.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedWmma.h"
#include "utils/ROCmGroupedBranchMutation.h"
#include "utils/ROCmGroupedPopulationCaptureProof.h"
#endif
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>
#include <omp.h>

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

    /** @brief Generate independent original tensors before GPU preparation.
     * @param parameter Complete source geometry, format and expert ownership case.
     * @return Sources in the original expert-major gate/up/down order.
     * @throws std::runtime_error For absent physical-worker inventory; forwards factory failures.
     *
     * Each task owns its RNG and complete tensor. Parallel scheduling preserves
     * every original seed and byte; GPU loading and capture remain on their
     * existing device worker. Physical cores and the selected OpenMP budget
     * jointly bound this cold team. No work is split within a seeded tensor. */
    std::vector<std::shared_ptr<TensorBase>> synthesizeExpertSources(const Case &parameter)
    {
        const auto *hardware = DeviceManager::instance().hardware();
        if (!hardware || hardware->total_physical_cores() <= 0 || omp_get_max_threads() <= 0)
            throw std::runtime_error("Expert source synthesis requires physical-worker inventory");
        if (parameter.experts <= 0 || parameter.experts > std::numeric_limits<int>::max()/3)
            throw std::invalid_argument("Expert source synthesis task count is invalid");
        const int tasks = parameter.experts*3;
        const int workers = std::min({tasks,hardware->total_physical_cores(),omp_get_max_threads()});
        std::vector<std::shared_ptr<TensorBase>> sources(tasks);
        std::vector<std::exception_ptr> errors(tasks);
        const auto started = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic,1) num_threads(workers)
        for (int task = 0; task < tasks; ++task)
            try
            {
                const bool down = task%3 == 2;
                sources[task] = weight(down ? parameter.down_format.value_or(parameter.format) : parameter.format,
                    {static_cast<size_t>(down ? parameter.hidden : parameter.intermediate),
                     static_cast<size_t>(down ? parameter.intermediate : parameter.hidden)},1000+task);
            }
            catch (...) { errors[task] = std::current_exception(); }
        for (const auto &error : errors) if (error) std::rethrow_exception(error);
        std::printf("MOE_SETUP,source_synthesis,matrices=%d,workers=%d,seconds=%.6f\n",tasks,workers,
            std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count());
        return sources;
    }

    /** Captured, event-timed whole-operation fixture; supports one-case profiling. */
    class GPUExpertPipeline : public ::testing::TestWithParam<Case> {};

    /** @brief Scope only the existing collector policy; restore original optional environment values. */
    class RouterQ8SubmissionObservation final {
        std::optional<std::string> json_, filter_;
        /**
         * @param name One of the two collector environment keys owned by this scope.
         * @return Exact prior value, distinguishing an empty value from absence.
         */
        static std::optional<std::string> prior(const char *name) {
            const char *value = std::getenv(name);
            return value ? std::optional<std::string>{value} : std::nullopt;
        }
        /**
         * @brief Restore exactly the saved value or original absence; failed restoration is fatal.
         * @param name Collector environment key being retired.
         * @param value Original value, or absence to restore with unsetenv.
         */
        static void restore(const char *name, const std::optional<std::string> &value) {
            const int result = value ? setenv(name,value->c_str(),1) : unsetenv(name);
            if (result != 0) std::terminate();
        }
    public:
        /** @brief Select accepted host submissions without enabling timers or another domain. */
        RouterQ8SubmissionObservation()
            : json_(prior("LLAMINAR_PERF_STATS_JSON")),
              filter_(prior("LLAMINAR_PERF_STATS_FILTER")) {
            if (setenv("LLAMINAR_PERF_STATS_JSON","1",1) != 0 ||
                setenv("LLAMINAR_PERF_STATS_FILTER","kernel.rocm_moe_grouped_prefill_router_q8_reuse_calls,kernel.rocm_moe_grouped_prefill_batch_invariant_calls",1) != 0) {
                restore("LLAMINAR_PERF_STATS_FILTER",filter_);
                restore("LLAMINAR_PERF_STATS_JSON",json_);
                throw std::runtime_error("Cannot select router Q8 submission observation environment");
            }
            PerfStatsCollector::reloadConfigurationFromEnvironment();
            PerfStatsCollector::reset();
        }
        /** @brief Retire fixture evidence, restore exact environment absence/values and reload it. */
        ~RouterQ8SubmissionObservation() {
            PerfStatsCollector::reset();
            restore("LLAMINAR_PERF_STATS_FILTER",filter_);
            restore("LLAMINAR_PERF_STATS_JSON",json_);
            PerfStatsCollector::reloadConfigurationFromEnvironment();
        }
        RouterQ8SubmissionObservation(const RouterQ8SubmissionObservation&) = delete;
        RouterQ8SubmissionObservation& operator=(const RouterQ8SubmissionObservation&) = delete;
    };

    /** @brief Public operation boundary whose immutable publication shape is timed. */
    enum class ExpertPublication { CanonicalRoutesThenFold, RouterQ8TokenRows };

    /** @brief Physical producer whose retained graph the diagnostic profiler submits. */
    enum class PopulationProfileProducer { Automatic, Ordered };
    /** @brief Immutable diagnostic workload; capacity and device-owned runtime policy stay intact. */
    struct PopulationProfileWork
    {
        int live_rows;
        PopulationProfileProducer producer;
    };

    /** @brief One complete benchmark operation, with profiler runs kept untimed. */
    enum class ExpertBenchmark
    {
        Installed, WmmaReplay, WmmaTournament, Wmma32Profile, Wmma64Profile,
        WmmaHybridReplay, WmmaHybridTournament, WmmaHybrid32Profile, WmmaHybridTimingReplay,
        OrderedProfile, OrderedColdProfile, WmmaAdaptive32ColdProfile,
        WmmaMacroReplay, WmmaMacroTournament, WmmaMacroTimingReplay, WmmaMacroProfile, WmmaMacroColdProfile,
        PopulationReplay, PopulationTournament, PopulationProfile
    };

#ifdef HAVE_ROCM
    /** @brief Inspect and temporarily omit every real router-Q8 producer in a retained graph.
     * The lease borrows its graph and stream; both outlive restoration. Publication
     * metadata authenticates capture arguments, never device values or live counts. */
    class RouterQ8CaptureProof final
    {
        HIPGraphCapture &graph_;
        hipStream_t stream_;
        std::vector<hipGraphNode_t> producers_;
        bool omitted_ = false;
        /** @param status Native inspection outcome. @throws std::runtime_error On failure. */
        static void require(hipError_t status)
        {
            if (status != hipSuccess) throw std::runtime_error("Router Q8 capture inspection failed");
        }
        /** @param params Captured kernel ABI. @param index Argument ordinal.
         * @return Copied immutable argument. @throws std::runtime_error For unsupported ABI. */
        template<class T> static T argument(const hipKernelNodeParams &params,size_t index)
        {
            static_assert(std::is_trivially_copyable_v<T>);
            if (!params.kernelParams || !params.kernelParams[index] || params.extra)
                throw std::runtime_error("Router Q8 capture argument ABI is missing");
            std::array<std::byte,sizeof(T)> bytes;
            std::memcpy(bytes.data(),params.kernelParams[index],bytes.size());
            return std::bit_cast<T>(bytes);
        }
    public:
        /** @param graph Actual instantiated parent, including imported timing nodes.
         * @param stream Exact execution stream. @param publication Real router metadata.
         * @param live Device-owned live count. @param grouped_ids Grouping authority.
         * @param rows Physical bucket. @param width Hidden width. @param intermediate GU width.
         * @param repeats Expected complete operation count. @param projection_consumers Exact retained GU consumers per complete operation.
         * @throws std::runtime_error For missing producers, duplicate quantization or wrong bindings. */
        RouterQ8CaptureProof(HIPGraphCapture &graph,hipStream_t stream,
            const MoERouterQ8HiddenPublication &publication,const int *live,const int *grouped_ids,
            int rows,int width,int intermediate,int repeats,int projection_consumers)
            : graph_(graph),stream_(stream)
        {
            if (!graph.graph() || !graph.executable() || !stream ||
                stream == hipStreamLegacy || stream == hipStreamPerThread ||
                !publication.capture_recorded || !publication.source_rows || !live ||
                publication.published_rows != rows || publication.d_model_capacity < width)
                throw std::runtime_error("Router Q8 proof requires a complete retained publication");
            size_t count = 0;
            require(hipGraphGetNodes(graph.graph(),nullptr,&count));
            std::vector<hipGraphNode_t> nodes(count);
            require(hipGraphGetNodes(graph.graph(),nodes.data(),&count));
            int consumers = 0;
            for (auto node : nodes)
            {
                hipGraphNodeType type;
                require(hipGraphNodeGetType(node,&type));
                if (type != hipGraphNodeTypeKernel) continue;
                hipKernelNodeParams params{};
                require(hipGraphKernelNodeGetParams(node,&params));
                const char *name = hipKernelNameRefByPtr(params.func,stream);
                if (!name) throw std::runtime_error("Router Q8 proof has an unnamed kernel");
                const std::string_view symbol(name);
                if (symbol.find("moe_prefill_hidden_quantize_rows_kernel") != std::string_view::npos)
                    throw std::runtime_error("Expert duplicated the router Q8 producer");
                if (symbol.find("rocm_moe_router_hidden_quantize_q8_rows_kernel") != std::string_view::npos)
                {
                    if (argument<const float *>(params,0) != publication.source_rows ||
                        argument<int8_t *>(params,1) != publication.quantized_rows ||
                        argument<float *>(params,2) != publication.row_scales ||
                        argument<int>(params,3) != rows || argument<int>(params,4) != width ||
                        argument<const int *>(params,5) != live)
                        throw std::runtime_error("Router Q8 producer bindings changed");
                    producers_.push_back(node);
                }
                if (symbol.find("grouped_wmma_detail") != std::string_view::npos && params.gridDim.z == 2)
                {
                    const auto args = argument<rocm::MoEGroupedWmmaArguments>(params,0);
                    if (args.input != rocm::MoEGroupedWmmaInput::OriginalTokens ||
                        args.activations != publication.quantized_rows ||
                        args.activation_scales != publication.row_scales ||
                        args.group_token_indices != grouped_ids || args.rows != rows ||
                        args.k != width || args.n != intermediate)
                        throw std::runtime_error("GU does not consume the real router publication");
                    ++consumers;
                }
            }
            if (producers_.size() != static_cast<size_t>(repeats) || consumers != projection_consumers * repeats)
                throw std::runtime_error("Router Q8 producer/consumer inventory is incomplete");
        }
        /** @brief Restore all producer nodes before the borrowed graph retires. */
        ~RouterQ8CaptureProof() { restore(); }
        RouterQ8CaptureProof(const RouterQ8CaptureProof &) = delete;
        RouterQ8CaptureProof &operator=(const RouterQ8CaptureProof &) = delete;
        /** @brief Omit every repeated producer; partial mutation is restored by the live lease. */
        void omit()
        {
            if (omitted_) throw std::runtime_error("Router Q8 producer is already omitted");
            omitted_ = true;
            for (auto node : producers_) require(hipGraphNodeSetEnabled(graph_.executable(),node,0));
        }
        /** @brief Drain the exact stream and restore all producer nodes, including on assertion exit. */
        void restore() noexcept
        {
            if (!omitted_) return;
            if (hipStreamSynchronize(stream_) != hipSuccess) std::terminate();
            for (auto node : producers_)
                if (hipGraphNodeSetEnabled(graph_.executable(),node,1) != hipSuccess) std::terminate();
            omitted_ = false;
        }
    };

    /** @brief One retained dot policy; names bind every sample to its row floor. */
    struct GroupedDotCandidate
    {
        rocm::MoEGroupedPrefillEngine engine;
        rocm::MoEGroupedWmmaRowFloor floor;
        std::string name;
    };

    /** @param hybrid Include all explicit whole-expert admission experiments.
     * @param macro Use the separate wide-candidate roster without expanding old suites.
     * @param router_q8 Include both adaptive native tiles for ordinary-operation comparison.
     * @return Ordered oracle, both pure native tiles and requested hybrid policies. */
    std::vector<GroupedDotCandidate> groupedDotCandidates(bool hybrid, bool macro, bool router_q8)
    {
        using enum rocm::MoEGroupedPrefillEngine;
        using enum rocm::MoEGroupedWmmaRowFloor;
        std::vector<GroupedDotCandidate> result{
            {OrderedDot4,EveryExpert,"Dot4"},
            {WmmaM16N32,EveryExpert,"N32"}, {WmmaM16N64,EveryExpert,"N64"}};
        if (macro)
            result.push_back({WideDot4Wmma,Sixteen,"WideFloor16"});
        else if (hybrid)
            for (const auto floor : {Four,Eight,Twelve,Sixteen})
                for (const auto engine : {WmmaM16N32,WmmaM16N64})
                    result.push_back({engine,floor,
                        std::string(engine == WmmaM16N32 ? "N32" : "N64") +
                        "Floor" + std::to_string(static_cast<int>(floor))});
        if (router_q8)
        {
            result.push_back({WmmaM16N32,Sixteen,"N32Floor16"});
            result.push_back({WmmaM16N64,Sixteen,"N64Floor16"});
        }
        return result;
    }

    /** @brief Prove the selected profile actually admits its tiled down consumer.
     * @param graph Retained production operation, outside the profiler region.
     * @param stream Exact stream used to resolve native kernel identities.
     * @param active_routes Count derived from this fixture's immutable input IDs.
     * @throws std::runtime_error For absent nodes or inactive/incompatible admission.
     *
     * The captured admission object owns all shape/codebook exceptions. A generic
     * route-count threshold could profile an early-return kernel instead. Reading
     * these immutable capture arguments never reads device group counts. */
    void requireGroupedProfileTiledAdmission(HIPGraphCapture &graph, hipStream_t stream,
                                             size_t active_routes)
    {
        if (active_routes > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Grouped profile route count exceeds the captured ABI");
        const auto require = [](hipError_t status) {
            if (status != hipSuccess)
                throw std::runtime_error("Grouped profile capture inspection failed");
        };
        const auto argument = []<typename T>(const hipKernelNodeParams &params, size_t index) {
            static_assert(std::is_trivially_copyable_v<T>);
            if (!params.kernelParams || !params.kernelParams[index] || params.extra)
                throw std::runtime_error("Grouped profile has incompatible capture arguments");
            std::array<std::byte,sizeof(T)> bytes;
            std::memcpy(bytes.data(),params.kernelParams[index],bytes.size());
            return std::bit_cast<T>(bytes);
        };
        size_t count = 0;
        require(hipGraphGetNodes(graph.graph(),nullptr,&count));
        std::vector<hipGraphNode_t> nodes(count);
        require(hipGraphGetNodes(graph.graph(),nodes.data(),&count));
        int witnesses = 0;
        for (auto node : nodes)
        {
            hipGraphNodeType type;
            require(hipGraphNodeGetType(node,&type));
            if (type != hipGraphNodeTypeKernel) continue;
            hipKernelNodeParams params{};
            require(hipGraphKernelNodeGetParams(node,&params));
            const char *name = hipKernelNameRefByPtr(params.func,stream);
            if (!name) throw std::runtime_error("Grouped profile has no native symbol");
            const std::string_view symbol(name);
            std::optional<rocm::ROCmMoEGroupedRouteAdmission> admission;
            if (symbol.find("grouped_wmma_detail") != std::string_view::npos && params.gridDim.z == 1)
            {
                const auto packet = argument.template operator()<rocm::MoEGroupedWmmaArguments>(params,0);
                if (packet.second_descriptors || packet.second_output ||
                    packet.input != rocm::MoEGroupedWmmaInput::GroupedRows)
                    throw std::runtime_error("Grouped profile native witness is not a down consumer");
                admission = packet.admission;
            }
            else if (symbol.find("moe_grouped_down_batch_invariant_tiled_prefill_kernel") != std::string_view::npos)
                admission = argument.template operator()<rocm::ROCmMoEGroupedRouteAdmission>(params,13);
            if (!admission) continue;
            if (admission->strategyForSlots(static_cast<int>(active_routes)) != rocm::ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled)
                throw std::runtime_error("Grouped profile requires the actual captured ExpertTiled route policy");
            ++witnesses;
        }
        if (!witnesses) throw std::runtime_error("Grouped profile has no tiled down admission witness");
    }
#endif

    /** @brief Reuse production geometry and source preparation for native tournaments.
     * @param parameter Exact existing model/control case. @param mode Timing or isolated profile owner.
     * @param publication_mode Complete public operation boundary retained by every candidate.
     * @param profile_work Optional exact live work and producer for the population profiler only. */
    void runExpertPipeline(Case parameter, ExpertBenchmark mode = ExpertBenchmark::Installed,
        ExpertPublication publication_mode = ExpertPublication::CanonicalRoutesThenFold,
        std::optional<PopulationProfileWork> profile_work = std::nullopt)
    {
        const bool token_publication = publication_mode == ExpertPublication::RouterQ8TokenRows;
        std::optional<RouterQ8SubmissionObservation> observation;
        if (token_publication) observation.emplace();
        const bool native = mode != ExpertBenchmark::Installed;
        if (token_publication && !native)
            throw std::invalid_argument("Router Q8 token probes require a native tournament mode");
        const bool population = mode == ExpertBenchmark::PopulationReplay ||
            mode == ExpertBenchmark::PopulationTournament || mode == ExpertBenchmark::PopulationProfile;
        if (population && !token_publication)
            throw std::invalid_argument("Population probes require the complete router Q8 token operation");
        const bool macro = population || mode == ExpertBenchmark::WmmaMacroReplay || mode == ExpertBenchmark::WmmaMacroTournament ||
            mode == ExpertBenchmark::WmmaMacroTimingReplay || mode == ExpertBenchmark::WmmaMacroProfile ||
            mode == ExpertBenchmark::WmmaMacroColdProfile;
        const bool timing_replay = mode == ExpertBenchmark::PopulationReplay || mode == ExpertBenchmark::WmmaHybridTimingReplay || mode == ExpertBenchmark::WmmaMacroTimingReplay;
        const bool cold_profile = mode == ExpertBenchmark::WmmaAdaptive32ColdProfile ||
                                  mode == ExpertBenchmark::OrderedColdProfile || mode == ExpertBenchmark::WmmaMacroColdProfile;
        const bool profile = mode == ExpertBenchmark::PopulationProfile || mode == ExpertBenchmark::Wmma32Profile || mode == ExpertBenchmark::Wmma64Profile ||
            mode == ExpertBenchmark::WmmaHybrid32Profile || mode == ExpertBenchmark::OrderedProfile ||
            mode == ExpertBenchmark::WmmaMacroProfile || cold_profile;
        if (profile && ::testing::UnitTest::GetInstance()->test_to_run_count() != 1)
            throw std::invalid_argument("grouped WMMA profiling requires one exact selected case");
        const bool hybrid = mode == ExpertBenchmark::WmmaHybridReplay ||
                            mode == ExpertBenchmark::WmmaHybridTournament ||
                            mode == ExpertBenchmark::WmmaHybrid32Profile || mode == ExpertBenchmark::WmmaAdaptive32ColdProfile || timing_replay || macro;
        // One retained capacity spans every live threshold. The canonical case
        // still owns the matrix shapes, formats, ownership and route profile.
        if (hybrid || mode == ExpertBenchmark::OrderedProfile) parameter.live_rows = parameter.rows;
        if (cold_profile)
        {
            ASSERT_EQ(parameter.rows,512) << "Cold profiles retain the full 512-row capture capacity";
            parameter.live_rows = 64;
        }
        if (profile_work)
        {
            if (mode != ExpertBenchmark::PopulationProfile || profile_work->live_rows <= 0 ||
                profile_work->live_rows > parameter.rows ||
                (profile_work->producer != PopulationProfileProducer::Automatic &&
                 profile_work->producer != PopulationProfileProducer::Ordered))
                throw std::invalid_argument("Invalid exact population profile workload");
            parameter.live_rows = profile_work->live_rows;
        }
#ifdef HAVE_ROCM
        auto dot_candidates = groupedDotCandidates(hybrid,macro,token_publication);
        if (population) dot_candidates.push_back({rocm::MoEGroupedPrefillEngine::Automatic,
            rocm::MoEGroupedWmmaRowFloor::EveryExpert,"Automatic"});
        // Independent fixture admission: source TensorTypes, public operation and
        // physical case geometry are available before the production selector runs.
        const bool qwen_population = population && parameter.rows == 512 && parameter.hidden == 2048 &&
            parameter.intermediate == 512 && parameter.experts == 256 && parameter.format == TensorType::IQ2_S &&
            parameter.down_format == TensorType::IQ4_XS;
        const bool compact_population = population && parameter.rows == 512 && parameter.hidden == 512 &&
            parameter.intermediate == 256 && parameter.experts == 8 && !parameter.down_format &&
            (parameter.format == TensorType::Q4_0 || parameter.format == TensorType::IQ2_S);
        // Compact's TM16/TN256 ordered geometry needs its own fused qualification.
        const auto population_domain = qwen_population ? rocm::MoEGroupedPopulationDomain::Qwen :
            rocm::MoEGroupedPopulationDomain::Unmeasured;
        const int automatic_native_families = qwen_population ? 1 : 0;
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
        // Raw backend discovery does not publish the CPU worker authority.
        // Initialize the canonical inventory once, under the existing startup
        // vendor policy, before any source task consumes its physical budget.
        if (!DeviceManager::instance().hardware()) DeviceManager::instance().initialize(-1,false);
        const auto sources = synthesizeExpertSources(parameter);
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
            // The enclosing sources outlive every prepared handle and borrowed
            // down-column view, including cleanup on this exact device worker.
            std::vector<GpuPreparedGemm> prepared;
            std::vector<DeviceMoEFloatingMatrixDesc> gates(experts), ups(experts), downs(experts);
            std::vector<DeviceNativeQuantisedMatrixDesc> quant_gates(experts), quant_ups(experts), quant_downs(experts);
            const bool floating = parameter.format == TensorType::FP16 ||
                parameter.format == TensorType::BF16 || parameter.format == TensorType::FP32;
            DeviceMoEWeightFormat format{};
            if (floating)
            {
                for (int index = 0; index < experts*3; ++index)
                    prepared.push_back(makeGpuPreparedFloatingPointGemm(sources[index].get(),parameter.device,
                        "expert.economy."+std::to_string(index/3)+"."+std::to_string(index%3)));
            }
            else
            {
                std::vector<TensorBase *> jobs;
                jobs.reserve(sources.size());
                for (const auto &source : sources) jobs.push_back(source.get());
                prepared = makeGpuPreparedGemmBatch(jobs,parameter.device,"expert.economy");
            }
            for (int expert = 0; expert < experts; ++expert)
            {
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
            // The real router produces original-basis Q8 rows on this exact
            // stream. Separate router outputs leave the controlled expert route
            // distribution explicit rather than pretending the router chose it.
            std::shared_ptr<MoERouterQ8HiddenPublication> router_publication;
            std::unique_ptr<TensorBase> router_gate, router_ids, router_weights;
            std::unique_ptr<INT32Tensor> device_live;
            MoERoutingResult unused_router_result;
            if (token_publication)
            {
                router_gate = tensor(experts,hidden);
                router_ids = tensor(parameter.rows,top_k);
                router_weights = tensor(parameter.rows,top_k);
                device_live = TestTensorFactory::createINT32({1});
                device_live->mutable_int32_data()[0] = parameter.live_rows;
                ASSERT_TRUE(device_live->ensureOnDevice(parameter.device,stream));
                router_publication = std::make_shared<MoERouterQ8HiddenPublication>();
                ASSERT_TRUE(kernel->bindRouterQ8HiddenPublication(router_publication,
                    MoERouterQ8PublicationAccess::ProducerAndConsumer));
                ASSERT_TRUE(kernel->prepareRouteLaunch(router_gate.get(),{
                    .kind=MoERouteLaunchKind::GroupedPrefill,.physical_rows=parameter.rows,
                    .d_model=hidden,.num_experts=experts,.top_k=top_k}));
                std::printf("MOE_OPERATION,router_q8_token_rows,injected_expert_routes,%s\n",parameter.name.c_str());
            }
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
            if (mode == ExpertBenchmark::PopulationProfile)
            {
                std::vector<int> counts(experts);
                for (float id : route_ids) if (id >= 0) ++counts[static_cast<int>(id)];
                int positive = 0, minimum = std::numeric_limits<int>::max(), maximum = 0;
                for (int count : counts) if (count > 0)
                { ++positive; minimum = std::min(minimum,count); maximum = std::max(maximum,count); }
                ASSERT_GT(positive,0);
                // Fifteen live tokens in uniform Qwen routing means120 one-row
                // experts. It is not the fifteen-rows-per-expert threshold.
                if (qwen_population && parameter.route_profile == MoERoutingProfile::Uniform &&
                    parameter.live_rows == 15)
                { ASSERT_EQ(active_routes,120u); ASSERT_EQ(positive,120); ASSERT_EQ(minimum,1); ASSERT_EQ(maximum,1); }
                std::printf("population_profile_work,capacity=%d,live=%d,routes=%zu,experts=%d,min=%d,max=%d,producer=%s\n",
                    parameter.rows,parameter.live_rows,active_routes,positive,minimum,maximum,
                    profile_work && profile_work->producer == PopulationProfileProducer::Ordered ? "Dot4" : "Automatic");
            }
            if (cold_profile)
            {
                std::vector<int> counts(experts);
                for (float id : route_ids) if (id >= 0) ++counts[static_cast<int>(id)];
                ASSERT_TRUE(std::all_of(counts.begin(),counts.end(),[](int n) { return n < 8; }))
                    << "Cold adaptive profile requires every live expert below floor8";
            }
            ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(), route_ids.data(), route_ids.size() * sizeof(float), parameter.device.ordinal, stream));
            ASSERT_TRUE(backend->hostToDevice(probabilities->gpu_data_ptr(), route_weights.data(), route_weights.size() * sizeof(float), parameter.device.ordinal, stream));
            const auto reuse_count = [&] {
                double count = 0;
                for (const auto &record : PerfStatsCollector::snapshot(
                         {"kernel.rocm_moe_grouped_prefill_router_q8_reuse_calls"}))
                    if (record.tags.contains("seq_len") && record.tags.at("seq_len") == std::to_string(parameter.rows) &&
                        record.tags.contains("top_k") && record.tags.at("top_k") == std::to_string(top_k) &&
                        record.tags.contains("descriptor_source") && record.tags.at("descriptor_source") == "static_table")
                        count += record.value;
                return count;
            };
            if (token_publication) ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("kernel"));
            auto enqueue = [&] {
                if (token_publication && !kernel->routeWithTensorsEffectiveSeqLen(
                        input.get(),router_gate.get(),parameter.rows,hidden,experts,top_k,true,
                        router_ids.get(),router_weights.get(),unused_router_result,
                        static_cast<const int *>(device_live->gpu_data_ptr()))) return false;
                if (token_publication)
                    return kernel->prepareExpertGroupsAsync(ids.get(),probabilities.get(),parameter.rows,experts,top_k) &&
                        kernel->executeGroupedPrefillPipeline(input.get(),output.get(),gate_table,down_table,
                            parameter.rows,hidden,intermediate,experts,top_k,nullptr);
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
            std::vector<unsigned char> expected_q8, expected_scales;
            if (token_publication)
            {
                ASSERT_EQ(router_publication->source_rows,input->gpu_data_ptr());
                ASSERT_EQ(router_publication->backend,DeviceType::ROCm);
                ASSERT_EQ(router_publication->device_ordinal,parameter.device.ordinal);
                expected_q8.resize(static_cast<size_t>(parameter.rows)*hidden);
                expected_scales.resize(static_cast<size_t>(parameter.rows)*(hidden/32)*sizeof(float));
                ASSERT_TRUE(backend->deviceToHost(expected_q8.data(),router_publication->quantized_rows,
                    expected_q8.size(),parameter.device.ordinal,stream));
                ASSERT_TRUE(backend->deviceToHost(expected_scales.data(),router_publication->row_scales,
                    expected_scales.size(),parameter.device.ordinal,stream));
                ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
            }
            const double reuse_before_capture = token_publication ? reuse_count() : 0;
            auto graph = context.createGraphCapture(stream);
            {
                ScopedBackendGraphCapture recording(context, *graph, "expert economy complete pipeline");
                ASSERT_TRUE(recording.begin());
#ifdef HAVE_ROCM
                rocm::ScopedMoEGroupedPrefillEngine baseline(native ? rocm::MoEGroupedPrefillEngine::OrderedDot4
                                                                   : rocm::MoEGroupedPrefillEngine::Automatic);
#endif
                for (int repeat = 0; repeat < (native && !profile ? 8 : 1); ++repeat) ASSERT_TRUE(enqueue());
                recording.finish();
            }
            ASSERT_TRUE(graph->instantiate());
            void *start = context.createEvent(GPUEventPurpose::Timing);
            void *end = context.createEvent(GPUEventPurpose::Timing);
            ASSERT_NE(start, nullptr);
            ASSERT_NE(end, nullptr);
            auto retire = [&](void *) {
                if (!backend->synchronizeStream(stream, parameter.device.ordinal)) std::terminate();
                context.destroyEvent(end);
                context.destroyEvent(start);
            };
            std::unique_ptr<void, decltype(retire)> timing_owner(&context, retire);
#ifdef HAVE_ROCM
            if (native)
            {
                constexpr int repeats = 8, sample_count = 31;
                std::vector<decltype(graph)> native_graphs;
                // Parents import the existing source DAGs, preserving every
                // operation and parameter. Retire parents before their sources.
                std::vector<decltype(graph)> timed_graphs;
                std::vector<decltype(graph.get())> graphs{graph.get()};
                // Retire submissions before these local native graphs disappear,
                // including fatal assertions after a launch or event failure.
                const auto join_graphs = [&](void *) {
                    if (!backend->synchronizeStream(stream,parameter.device.ordinal)) std::terminate();
                };
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
                    int native_nodes = 0, ordered_population_nodes = 0, gate_up_nodes = 0, down_nodes = 0;
                    std::set<std::tuple<uintptr_t,uint32_t,uint32_t,uint32_t,uint32_t>> seen;
                    for (const auto &node : nodes)
                    {
                        ASSERT_EQ(node.local_memory_bytes_per_thread,0u) << node.name;
                        if (dot_candidates[path].engine != rocm::MoEGroupedPrefillEngine::Automatic)
                        {
                            ASSERT_EQ(node.name.find("moe_grouped_gate_up_batch_invariant_tiled_prefill_kernel"),std::string::npos);
                            ASSERT_EQ(node.name.find("moe_grouped_down_batch_invariant_tiled_prefill_kernel"),std::string::npos);
                        }
                        if (node.name.find("grouped_wmma_detail") == std::string::npos) continue;
                        const bool automatic = dot_candidates[path].engine == rocm::MoEGroupedPrefillEngine::Automatic;
                        if (automatic)
                            ASSERT_NE(node.name.find("populationProjection"),std::string::npos) << node.name;
                        else
                        {
                            const bool adaptive = dot_candidates[path].floor != rocm::MoEGroupedWmmaRowFloor::EveryExpert;
                            ASSERT_NE(node.name.find(adaptive ? "ELb1EE" : "ELb0EE"),std::string::npos) << node.name;
                        }
                        if (dot_candidates[path].engine == rocm::MoEGroupedPrefillEngine::WideDot4Wmma)
                        {
                            ASSERT_NE(node.name.find("wideProjection"),std::string::npos);
                            ASSERT_TRUE(node.grid_z == 1 || node.grid_z == 2);
                            const unsigned columns = node.grid_z == 2 ? 256u : 128u;
                            const unsigned n = node.grid_z == 2 ? intermediate : hidden;
                            ASSERT_EQ(node.block_x,columns);
                            const unsigned native_columns = columns / 4;
                            ASSERT_EQ(node.grid_x,(n+native_columns-1)/native_columns);
                        }
                        if (automatic && capturedPopulationArithmetic(node.name,node.grid_z == 2) == PopulationCapturedArithmetic::Ordered)
                            ++ordered_population_nodes;
                        else ++native_nodes;
                        if (node.grid_z == 2) ++gate_up_nodes;
                        else if (node.grid_z == 1) ++down_nodes;
                        else FAIL() << "Unexpected grouped projection role";
                        if (seen.emplace(node.function_identity,node.grid_x,node.grid_y,node.grid_z,node.block_x).second)
                            std::printf("MOE_WMMA_RESOURCE,%s,%zu,%u,%zu,%u,%u,%u,%u,%u,%s\n",parameter.name.c_str(),path,
                                node.registers_per_thread,node.static_shared_memory_bytes,node.max_active_blocks_per_sm,
                                node.grid_x,node.grid_y,node.grid_z,node.block_x,node.name.c_str());
                    }
                    const int native_families = dot_candidates[path].engine == rocm::MoEGroupedPrefillEngine::Automatic
                        ? automatic_native_families : 1;
                    ASSERT_EQ(native_nodes,2 * native_families * (profile ? 1 : repeats));
                    const int ordered_families = dot_candidates[path].engine == rocm::MoEGroupedPrefillEngine::Automatic
                        ? automatic_native_families : 0;
                    ASSERT_EQ(ordered_population_nodes,ordered_families * (profile ? 1 : repeats));
                    ASSERT_EQ(gate_up_nodes,(native_families + ordered_families) * (profile ? 1 : repeats));
                    ASSERT_EQ(down_nodes,native_families * (profile ? 1 : repeats));
                    ASSERT_TRUE(graphs[path]->instantiate());
                    if (dot_candidates[path].engine == rocm::MoEGroupedPrefillEngine::WideDot4Wmma)
                    {
                        auto *captured = dynamic_cast<HIPGraphCapture *>(graphs[path]);
                        ASSERT_NE(captured,nullptr);
                        std::array<size_t,2> roles;
                        ASSERT_NO_THROW(roles = inspectROCmGroupedWideCapture(*captured,
                            static_cast<hipStream_t>(stream),dot_candidates[path].floor));
                        EXPECT_EQ(roles,(std::array<size_t,2>{size_t(profile ? 1 : repeats),size_t(profile ? 1 : repeats)}));
                    }
                }
                const bool internal_timing = timing_replay || mode == ExpertBenchmark::PopulationTournament || mode == ExpertBenchmark::WmmaTournament ||
                    mode == ExpertBenchmark::WmmaHybridTournament || mode == ExpertBenchmark::WmmaMacroTournament;
                if (internal_timing)
                {
                    for (size_t path = 0; path < graphs.size(); ++path)
                    {
                        timed_graphs.push_back(context.createGraphCapture(stream));
                        auto &parent = *timed_graphs.back();
                        const std::array<GPUOrderedTimelineStep,1> steps{{{
                            .name = dot_candidates[path].name.c_str(),
                            .kind = GPUOrderedTimelineStepKind::CapturedFragment,
                            .capture = graphs[path]}}};
                        ASSERT_TRUE(parent.buildOrderedTimelineTransaction(steps,GPUOrderedTimelineInstrumentation::PerStepEvents));
                        ASSERT_EQ(parent.nodeCount(),graphs[path]->nodeCount()+2);
                        ASSERT_EQ(parent.consumeOrderedTimelineTiming().state,GPUOrderedTimelineTimingState::AwaitingLaunch);
                        std::vector<GPUGraphKernelNodeInfo> source_nodes, parent_nodes;
                        ASSERT_TRUE(graphs[path]->inspectKernelNodes(source_nodes));
                        ASSERT_TRUE(parent.inspectKernelNodes(parent_nodes));
                        ASSERT_FALSE(source_nodes.empty());
                        ASSERT_EQ(source_nodes.size(),parent_nodes.size());
                        // Native enumeration order is not a semantic graph edge.
                        // Compare the complete launch inventory as a multiset;
                        // replay bytes additionally authenticate cloned arguments.
                        const auto key = [](const GPUGraphKernelNodeInfo &node) {
                            return std::tuple{node.name,node.function_identity,node.grid_x,node.grid_y,node.grid_z,
                                node.block_x,node.block_y,node.block_z,node.static_shared_memory_bytes,
                                node.dynamic_shared_memory_bytes,node.local_memory_bytes_per_thread};
                        };
                        std::multiset<decltype(key(source_nodes.front()))> source_inventory, parent_inventory;
                        for (const auto &node : source_nodes) { ASSERT_TRUE(node.valid()); source_inventory.insert(key(node)); }
                        for (const auto &node : parent_nodes) { ASSERT_TRUE(node.valid()); parent_inventory.insert(key(node)); }
                        ASSERT_EQ(source_inventory,parent_inventory);
                        ASSERT_TRUE(parent.instantiate());
                        if (dot_candidates[path].engine == rocm::MoEGroupedPrefillEngine::WideDot4Wmma)
                        {
                            auto *captured = dynamic_cast<HIPGraphCapture *>(&parent);
                            ASSERT_NE(captured,nullptr);
                            std::array<size_t,2> roles;
                            ASSERT_NO_THROW(roles = inspectROCmGroupedWideCapture(*captured,
                                static_cast<hipStream_t>(stream),dot_candidates[path].floor));
                            EXPECT_EQ(roles,(std::array<size_t,2>{repeats,repeats}));
                        }
                        if (timing_replay && path == 0)
                        {
                            auto disabled = context.createGraphCapture(stream);
                            ASSERT_TRUE(disabled->buildOrderedTimelineTransaction(steps,GPUOrderedTimelineInstrumentation::Disabled));
                            const auto observation = disabled->consumeOrderedTimelineTiming();
                            ASSERT_EQ(observation.state,GPUOrderedTimelineTimingState::Disabled);
                            ASSERT_TRUE(observation.samples.empty());
                        }
                        // Dirty replay and omitted-writer controls must act on
                        // the actual timed executable, whose imported nodes are
                        // distinct from the source executable's nodes.
                        graphs[path] = &parent;
                    }
                }
                std::vector<std::unique_ptr<RouterQ8CaptureProof>> router_proofs;
                const double reuse_after_capture = token_publication ? reuse_count() : 0;
                const auto submitted_count = [&] {
                    double count = 0;
                    for (const auto &record : PerfStatsCollector::snapshot({"kernel.rocm_moe_grouped_prefill_batch_invariant_calls"}))
                        count += record.value;
                    return count;
                };
                const double submissions_after_capture = token_publication ? submitted_count() : 0;
                if (population)
                {
                    double automatic_submissions = 0;
                    std::vector<GPUGraphKernelNodeInfo> actual_nodes;
                    ASSERT_TRUE(graphs.back()->inspectKernelNodes(actual_nodes));
                    int gateup_leaves = 0, down_leaves = 0;
                    for (const auto &node : actual_nodes)
                    {
                        if (node.name.find("grouped_wmma_detail") != std::string::npos)
                            (node.grid_z == 2 ? gateup_leaves : down_leaves)++;
                        else if (node.name.find("moe_grouped_gate_up_batch_invariant_tiled_prefill_kernel") != std::string::npos) ++gateup_leaves;
                        else if (node.name.find("moe_grouped_down_batch_invariant_tiled_prefill_kernel") != std::string::npos) ++down_leaves;
                    }
                    for (const auto &record : PerfStatsCollector::snapshot({"kernel.rocm_moe_grouped_prefill_batch_invariant_calls"}))
                    {
                        ASSERT_EQ(record.tags.at("counter_semantics"),"host_submissions");
                        ASSERT_EQ(record.tags.at("hidden_preparation"),"router_q8");
                        if (record.tags.at("dot_policy") != "device_population_v1") continue;
                        automatic_submissions += record.value;
                        ASSERT_EQ(std::stoi(record.tags.at("gateup_submitted_tiled_leaves"))*(profile ? 1 : repeats),gateup_leaves);
                        ASSERT_EQ(std::stoi(record.tags.at("down_submitted_tiled_leaves"))*(profile ? 1 : repeats),down_leaves);
                        ASSERT_EQ(record.tags.at("gateup_submitted_tiles"),
                            "dot4:12x128/threads=128/ordered_rows=0/ordered_columns=0/floor=0/leaves=1;"
                            "wmma_n32:16x32/threads=128/ordered_rows=0/ordered_columns=0/floor=0/leaves=1");
                        ASSERT_EQ(record.tags.at("down_submitted_tiles"),
                            "device_population_v1:16x32/threads=128/ordered_rows=16/ordered_columns=128/floor=16/leaves=1");
                        ASSERT_EQ(record.tags.at("gateup_submitted_tiles").find("wide_dot4_wmma"),std::string::npos);
                    }
                    ASSERT_EQ(automatic_submissions,automatic_native_families ? (profile ? 1 : repeats) : 0);
                }
                if (token_publication)
                {
                    ASSERT_EQ(router_publication->source_rows,input->gpu_data_ptr());
                    ASSERT_EQ(router_publication->backend,DeviceType::ROCm);
                    ASSERT_EQ(router_publication->device_ordinal,parameter.device.ordinal);
                    ASSERT_EQ(reuse_after_capture-reuse_before_capture,graphs.size()*(profile ? 1 : repeats));
                    for (size_t path = 0; path < graphs.size(); ++path)
                    {
                        auto *captured = dynamic_cast<HIPGraphCapture *>(graphs[path]);
                        ASSERT_NE(captured,nullptr);
                        router_proofs.push_back(std::make_unique<RouterQ8CaptureProof>(*captured,
                            static_cast<hipStream_t>(stream),*router_publication,
                            static_cast<const int *>(device_live->gpu_data_ptr()),
                            static_cast<const int *>(workspace.getBuffer(MoEWorkspaceBuffers::GROUP_TOKEN_INDICES)),
                            parameter.rows,hidden,intermediate,profile ? 1 : repeats,
                            dot_candidates[path].engine == rocm::MoEGroupedPrefillEngine::Automatic ? 2 * automatic_native_families : path != 0));
                    }
                }
                std::unique_ptr<ROCmGroupedPopulationCaptureProof> population_proof;
                auto *population_selection = static_cast<uint32_t *>(workspace.getBuffer(
                    MoEWorkspaceBuffers::ROCM_GROUPED_POPULATION_SELECTION));
                if (population)
                {
                    auto *captured = dynamic_cast<HIPGraphCapture *>(graphs.back());
                    ASSERT_NE(captured,nullptr);
                    ASSERT_NO_THROW(population_proof = std::make_unique<ROCmGroupedPopulationCaptureProof>(
                        *captured,static_cast<hipStream_t>(stream),population_domain,population_selection,
                        static_cast<const int *>(workspace.getBuffer(MoEWorkspaceBuffers::GROUP_COUNTS)),
                        static_cast<const uint32_t *>(workspace.getBuffer(MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY)),
                        experts,profile ? 1 : repeats));
                }
                if (mode == ExpertBenchmark::PopulationReplay && population_domain == rocm::MoEGroupedPopulationDomain::Qwen)
                {
                    // Probe the real submission boundary with captured production
                    // packets. These auxiliary graphs are inspected, never launched.
                    for (const auto role : {rocm::MoEPrefillProjectionRole::GateUp,rocm::MoEPrefillProjectionRole::Down})
                    {
                        const auto [args,control] = population_proof->capturedRoleBindings(role);
                        const size_t required = rocm::MoEGroupedPopulationDirectoryLayout(args.directory_capacity).words();
                        for (const bool exact : {false,true})
                        {
                            auto probe = context.createGraphCapture(stream);
                            rocm::MoEGroupedPrefillRoleSubmission submitted;
                            submitted.record({rocm::MoEGroupedPrefillEngine::PopulationAdaptive,
                                rocm::MoEGroupedWmmaRowFloor::Sixteen,16,32,128,12,7,128});
                            bool accepted = false;
                            {
                                ScopedBackendGraphCapture capture(context,*probe,"population workspace bound");
                                ASSERT_TRUE(capture.begin());
                                accepted = rocm::submitMoEGroupedPopulation(role,
                                    {population_selection,static_cast<uint32_t *>(workspace.getBuffer(
                                        MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY)),required - (exact ? 0 : 1)},
                                    uint32_t{1} << (role == rocm::MoEPrefillProjectionRole::GateUp ? 13 : 24),
                                    args,control,stream,&submitted);
                                capture.finish();
                            }
                            ASSERT_EQ(accepted,exact);
                            const unsigned leaves = role == rocm::MoEPrefillProjectionRole::GateUp ? 2u : 1u;
                            ASSERT_EQ(submitted.leaves(),exact ? leaves : 0u);
                            ASSERT_EQ(submitted.size,exact ? leaves : 0u);
                            std::vector<GPUGraphKernelNodeInfo> nodes;
                            ASSERT_TRUE(probe->inspectKernelNodes(nodes));
                            ASSERT_EQ(nodes.size(),exact ? leaves + 1u : 0u);
                        }
                    }
                }
                if (timing_replay)
                    for (auto *candidate : graphs)
                    {
                        auto *captured = dynamic_cast<HIPGraphCapture *>(candidate);
                        ASSERT_NE(captured,nullptr);
                        ASSERT_NO_THROW(requireGroupedProfileTiledAdmission(*captured,
                            static_cast<hipStream_t>(stream),active_routes));
                        // An empty route plan uses RouteOwned even though the
                        // same retained topology includes tiled projection nodes.
                        ASSERT_THROW(requireGroupedProfileTiledAdmission(*captured,
                            static_cast<hipStream_t>(stream),0),std::runtime_error);
                        ASSERT_THROW(requireGroupedProfileTiledAdmission(*captured,
                            static_cast<hipStream_t>(stream),
                            static_cast<size_t>(std::numeric_limits<int>::max())+1),std::runtime_error);
                    }
                const auto consume_timing = [&](size_t path) {
                    const auto observation = graphs[path]->consumeOrderedTimelineTiming();
                    if (observation.state != GPUOrderedTimelineTimingState::Complete || observation.samples.size() != 1 ||
                        observation.samples.front().name != dot_candidates[path].name ||
                        observation.samples.front().kind != GPUOrderedTimelineStepKind::CapturedFragment ||
                        !std::isfinite(observation.samples.front().elapsed_ms) || observation.samples.front().elapsed_ms <= 0 ||
                        graphs[path]->consumeOrderedTimelineTiming().state != GPUOrderedTimelineTimingState::AwaitingLaunch)
                        throw std::runtime_error("grouped timing sample is missing, stale or belongs to another policy");
                    return observation.samples.front().elapsed_ms*1000.0/repeats;
                };
                const auto poison = [&] {
                    for (const char *name : {MoEWorkspaceBuffers::PREFILL_A_INT8,MoEWorkspaceBuffers::PREFILL_A_SCALES,
                            MoEWorkspaceBuffers::PREFILL_GATE,MoEWorkspaceBuffers::PREFILL_UP,
                            MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8,MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES,
                            MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY,
                            MoEWorkspaceBuffers::ROCM_GROUPED_POPULATION_SELECTION})
                    {
                        void *bank = workspace.getBuffer(name);
                        const size_t bytes = workspace.getBufferSize(name);
                        if (!bank || !bytes || hipMemsetAsync(bank,0xa5,bytes,static_cast<hipStream_t>(stream)) != hipSuccess)
                            throw std::runtime_error("grouped WMMA scratch poison failed");
                    }
                    if (token_publication)
                        for (const char *name : {MoEWorkspaceBuffers::ROCM_ROUTER_Q8_HIDDEN,
                                                MoEWorkspaceBuffers::ROCM_ROUTER_Q8_SCALES})
                            if (!workspace.getBuffer(name) || !workspace.getBufferSize(name) ||
                                hipMemsetAsync(workspace.getBuffer(name),0xa5,workspace.getBufferSize(name),
                                    static_cast<hipStream_t>(stream)) != hipSuccess)
                                throw std::runtime_error("router Q8 publication poison failed");
                    if (hipMemsetAsync(output->gpu_data_ptr(),0xa5,output->numel()*sizeof(float),static_cast<hipStream_t>(stream)) != hipSuccess ||
                        hipMemsetAsync(routes->gpu_data_ptr(),0xa5,routes->numel()*sizeof(float),static_cast<hipStream_t>(stream)) != hipSuccess)
                        throw std::runtime_error("grouped WMMA publication poison failed");
                };
                const auto verify_q8_publication = [&](int live) {
                    if (!token_publication) return;
                    std::vector<unsigned char> q8(expected_q8.size()), scales(expected_scales.size());
                    const auto drain_q8 = [&](void *) {
                        if (!backend->synchronizeStream(stream,parameter.device.ordinal)) std::terminate();
                    };
                    std::unique_ptr<void,decltype(drain_q8)> q8_retirement(&context,drain_q8);
                    ASSERT_TRUE(backend->deviceToHost(q8.data(),router_publication->quantized_rows,
                        q8.size(),parameter.device.ordinal,stream));
                    ASSERT_TRUE(backend->deviceToHost(scales.data(),router_publication->row_scales,
                        scales.size(),parameter.device.ordinal,stream));
                    ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                    const size_t active_q8 = static_cast<size_t>(live)*hidden;
                    const size_t active_scales = static_cast<size_t>(live)*(hidden/32)*sizeof(float);
                    ASSERT_EQ(std::memcmp(q8.data(),expected_q8.data(),active_q8),0);
                    ASSERT_EQ(std::memcmp(scales.data(),expected_scales.data(),active_scales),0);
                    ASSERT_TRUE(std::all_of(q8.begin()+active_q8,q8.end(),[](auto v) { return v == 0xa5; }));
                    ASSERT_TRUE(std::all_of(scales.begin()+active_scales,scales.end(),[](auto v) { return v == 0xa5; }));
                    ASSERT_EQ(reuse_count(),reuse_after_capture);
                    ASSERT_EQ(submitted_count(),submissions_after_capture);
                };
                // Expected decisions derive from the fixture's routes, never a
                // device count download. This scalar oracle is outside timing.
                const auto expected_population = [&](int live) {
                    using enum rocm::MoEGroupedPopulationFamily;
                    std::vector<int> counts(experts);
                    int slots = 0;
                    for (size_t i = 0; i < std::min(route_ids.size(),static_cast<size_t>(live)*top_k); ++i)
                        if (route_ids[i] >= 0) { ++counts[static_cast<int>(route_ids[i])]; ++slots; }
                    if (!population_proof->tiled(slots)) return Dot4;
                    std::erase(counts,0);
                    if (counts.empty()) return Dot4;
                    const auto [low,high] = std::minmax_element(counts.begin(),counts.end());
                    if (compact_population)
                        return counts.size() == 8 && *low == *high && *low >= 12 ? N32 : Dot4;
                    if (*low >= 16 || (counts.size() <= 32 && *low >= 7)) return N32;
                    if (*high >= 16 || *high >= 5 * *low || (*low == 12 && *high == 12)) return WideFloor16;
                    return Dot4;
                };
                const auto verify_population = [&](int live) {
                    if (!population || !automatic_native_families) return;
                    uint32_t selected = UINT32_MAX;
                    const auto drain_selection = [&](void *) {
                        if (!backend->synchronizeStream(stream,parameter.device.ordinal)) std::terminate();
                    };
                    std::unique_ptr<void,decltype(drain_selection)> selection_retirement(&context,drain_selection);
                    ASSERT_TRUE(backend->deviceToHost(&selected,population_selection,sizeof(selected),parameter.device.ordinal,stream));
                    ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                    ASSERT_EQ(selected,static_cast<uint32_t>(expected_population(live))) << "live=" << live;
                };
                const auto verify = [&](int live) {
                    auto replay_ids = route_ids;
                    const auto drain_publication = [&](void *) {
                        if (!backend->synchronizeStream(stream,parameter.device.ordinal)) std::terminate();
                    };
                    std::unique_ptr<void,decltype(drain_publication)> publication_retirement(&context,drain_publication);
                    for (size_t i = static_cast<size_t>(live)*top_k; i < replay_ids.size(); ++i) replay_ids[i] = -1.0f;
                    ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(),replay_ids.data(),replay_ids.size()*sizeof(float),parameter.device.ordinal,stream));
                    if (token_publication)
                        ASSERT_TRUE(backend->hostToDevice(device_live->gpu_data_ptr(),&live,sizeof(live),
                            parameter.device.ordinal,stream));
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
                        ASSERT_NO_FATAL_FAILURE(verify_q8_publication(live));
                        if (population && path + 1 == graphs.size()) ASSERT_NO_FATAL_FAILURE(verify_population(live));
                    }
                };
                const auto verify_population_writers = [&](int live) {
                    if (!population || !automatic_native_families) return;
                    int slots = 0;
                    for (size_t i = 0; i < std::min(route_ids.size(),static_cast<size_t>(live)*top_k); ++i)
                        slots += route_ids[i] >= 0;
                    const bool tiled = population_proof->tiled(slots);
                    const auto selected = expected_population(live);
                    auto reference = expected;
                    std::fill(reference.begin()+static_cast<size_t>(live)*hidden,reference.end(),0.0f);
                    for (auto family : {rocm::MoEGroupedPopulationFamily::Dot4,
                            rocm::MoEGroupedPopulationFamily::N32,rocm::MoEGroupedPopulationFamily::WideFloor16})
                    {
                        if (!population_proof->contains(family)) continue;
                        population_proof->omit(family);
                        poison();
                        ASSERT_TRUE(graphs.back()->launch());
                        ASSERT_TRUE(backend->deviceToHost(actual.data(),output->gpu_data_ptr(),actual.size()*sizeof(float),parameter.device.ordinal,stream));
                        ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                        ASSERT_EQ(std::memcmp(reference.data(),actual.data(),actual.size()*sizeof(float)) != 0,
                            tiled && family == selected) << "live=" << live << " omitted=" << static_cast<int>(family);
                        population_proof->restore();
                        for (int restored = 0; restored < 2; ++restored)
                        {
                            poison();
                            ASSERT_TRUE(graphs.back()->launch());
                            ASSERT_TRUE(backend->deviceToHost(actual.data(),output->gpu_data_ptr(),actual.size()*sizeof(float),parameter.device.ordinal,stream));
                            ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                            ASSERT_EQ(std::memcmp(reference.data(),actual.data(),actual.size()*sizeof(float)),0);
                            ASSERT_NO_FATAL_FAILURE(verify_population(live));
                        }
                    }
                };
                const auto verify_population_producer_omissions = [&] {
                    if (!qwen_population) return;
                    const auto selected_family = expected_population(parameter.rows);
                    std::vector<int> counts(experts);
                    for (float id : route_ids) if (id >= 0) ++counts[static_cast<int>(id)];
                    if (!population_proof->tiled(std::accumulate(counts.begin(),counts.end(),0))) return;
                    for (auto branch : {rocm::MoEGroupedBranchSuppression::Ordered,rocm::MoEGroupedBranchSuppression::Wmma})
                    {
                        population_proof->omitArithmetic(branch);
                        poison();
                        ASSERT_TRUE(graphs.back()->launch());
                        ASSERT_TRUE(backend->deviceToHost(actual.data(),output->gpu_data_ptr(),actual.size()*sizeof(float),parameter.device.ordinal,stream));
                        ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                        int surviving_tokens = 0, missing_tokens = 0;
                        bool expect_missing = false, expect_surviving = false;
                        for (int token=0;token<parameter.rows;++token)
                        {
                            bool omitted = false, surviving = false;
                            for (int slot=0;slot<top_k;++slot)
                            {
                                const int expert = static_cast<int>(route_ids[token*top_k+slot]);
                                if (expert < 0) continue;
                                const bool is_ordered = selected_family == rocm::MoEGroupedPopulationFamily::Dot4 ||
                                    (selected_family == rocm::MoEGroupedPopulationFamily::WideFloor16 && counts[expert] < 16);
                                const bool removed = is_ordered == (branch == rocm::MoEGroupedBranchSuppression::Ordered);
                                omitted |= removed; surviving |= !removed;
                            }
                            const int different = std::memcmp(actual.data()+token*hidden,expected.data()+token*hidden,hidden*sizeof(float));
                            if (!omitted) {
                                ASSERT_EQ(different,0) << "surviving token=" << token;
                                surviving_tokens += surviving; expect_surviving |= surviving;
                            }
                            else { expect_missing = true; if (different) ++missing_tokens; }
                        }
                        if (expect_surviving) ASSERT_GT(surviving_tokens,0);
                        if (expect_missing) ASSERT_GT(missing_tokens,0);
                        else ASSERT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),0);
                        population_proof->restore();
                        for (int restored=0;restored<2;++restored)
                        {
                            poison();
                            ASSERT_TRUE(graphs.back()->launch());
                            ASSERT_TRUE(backend->deviceToHost(actual.data(),output->gpu_data_ptr(),actual.size()*sizeof(float),parameter.device.ordinal,stream));
                            ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                            ASSERT_EQ(std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float)),0);
                        }
                    }
                };
                for (int live : {parameter.live_rows,0,1,std::min(8,parameter.live_rows),parameter.live_rows})
                { ASSERT_NO_FATAL_FAILURE(verify(live)); }
                if (token_publication)
                    for (size_t path = 0; path < graphs.size(); ++path)
                    {
                        router_proofs[path]->omit();
                        poison();
                        ASSERT_TRUE(graphs[path]->launch());
                        ASSERT_TRUE(backend->deviceToHost(actual.data(),output->gpu_data_ptr(),
                            actual.size()*sizeof(float),parameter.device.ordinal,stream));
                        ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                        ASSERT_NE(std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float)),0);
                        router_proofs[path]->restore();
                        for (int restored = 0; restored < 2; ++restored)
                        {
                            poison();
                            ASSERT_TRUE(graphs[path]->launch());
                            ASSERT_TRUE(backend->deviceToHost(actual.data(),output->gpu_data_ptr(),
                                actual.size()*sizeof(float),parameter.device.ordinal,stream));
                            ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                            ASSERT_EQ(std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float)),0);
                        }
                    }
                for (int warmup = 0; warmup < 5; ++warmup)
                    for (auto *candidate : graphs) ASSERT_TRUE(candidate->launch());
                if (profile)
                {
                    ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                    const std::string profile_name = mode == ExpertBenchmark::PopulationProfile ?
                        (profile_work && profile_work->producer == PopulationProfileProducer::Ordered ? "Dot4" : "Automatic") :
                        (mode == ExpertBenchmark::OrderedProfile || mode == ExpertBenchmark::OrderedColdProfile) ? "Dot4" :
                        (mode == ExpertBenchmark::WmmaHybrid32Profile || mode == ExpertBenchmark::WmmaAdaptive32ColdProfile) ? "N32Floor8" :
                        macro ? "WideFloor16" : mode == ExpertBenchmark::Wmma32Profile ? "N32" : "N64";
                    const auto selected = std::find_if(dot_candidates.begin(),dot_candidates.end(),
                        [&](const auto &candidate) { return candidate.name == profile_name; });
                    ASSERT_NE(selected,dot_candidates.end());
                    auto *captured = dynamic_cast<HIPGraphCapture *>(graphs[selected-dot_candidates.begin()]);
                    ASSERT_NE(captured,nullptr);
                    ASSERT_NO_THROW(requireGroupedProfileTiledAdmission(*captured,static_cast<hipStream_t>(stream),
                        active_routes));
                    ASSERT_GE(roctxRangePushA("ROCmGroupedMoEWmma"),0);
                    ASSERT_EQ(roctxProfilerResume(0),0);
                    const bool launched = graphs[selected-dot_candidates.begin()]->launch();
                    const bool joined = backend->synchronizeStream(stream,parameter.device.ordinal);
                    const int paused = roctxProfilerPause(0), popped = roctxRangePop();
                    ASSERT_TRUE(launched && joined && paused == 0 && popped >= 0);
                    return;
                }
                const auto verify_both_writers = [&] {
                    std::vector<int> counts(experts);
                    for (float id : route_ids) if (id >= 0) ++counts[static_cast<int>(id)];
                    if (token_publication)
                    {
                        int mixed_policies = 0;
                        for (size_t path = 3; path < graphs.size(); ++path)
                        {
                            const int floor = static_cast<int>(dot_candidates[path].floor);
                            if (!std::any_of(counts.begin(),counts.end(),[&](int n) { return n >= floor; }) ||
                                !std::any_of(counts.begin(),counts.end(),[&](int n) { return n > 0 && n < floor; })) continue;
                            ++mixed_policies;
                            auto *capture = dynamic_cast<HIPGraphCapture *>(graphs[path]);
                            ASSERT_NE(capture,nullptr);
                            ROCmGroupedBranchMutation mutation(*capture,static_cast<hipStream_t>(stream));
                            for (auto omitted : {rocm::MoEGroupedBranchSuppression::Ordered,rocm::MoEGroupedBranchSuppression::Wmma})
                            {
                                mutation.suppress(omitted);
                                poison();
                                ASSERT_TRUE(graphs[path]->launch());
                                ASSERT_TRUE(backend->deviceToHost(actual.data(),output->gpu_data_ptr(),
                                    actual.size()*sizeof(float),parameter.device.ordinal,stream));
                                ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                                ASSERT_NE(std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float)),0);
                                mutation.restore();
                                for (int restored = 0; restored < 2; ++restored)
                                {
                                    poison();
                                    ASSERT_TRUE(graphs[path]->launch());
                                    ASSERT_TRUE(backend->deviceToHost(actual.data(),output->gpu_data_ptr(),
                                        actual.size()*sizeof(float),parameter.device.ordinal,stream));
                                    ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                                    ASSERT_EQ(std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float)),0);
                                }
                            }
                        }
                        ASSERT_GT(mixed_policies,0);
                        return;
                    }
                    std::vector<float> reference_routes(routes->numel()), observed_routes(routes->numel());
                    const auto drain_observations = [&](void *) {
                        if (!backend->synchronizeStream(stream,parameter.device.ordinal)) std::terminate();
                    };
                    std::unique_ptr<void,decltype(drain_observations)> observation_retirement(&context,drain_observations);
                    // Only the explicit Dot4 graph establishes the route oracle.
                    // Public output mixes experts and cannot certify the intact
                    // branch when another expert's contribution is omitted.
                    ASSERT_TRUE(graphs[0]->launch());
                    ASSERT_TRUE(backend->deviceToHost(reference_routes.data(),routes->gpu_data_ptr(),
                        reference_routes.size()*sizeof(float),parameter.device.ordinal,stream));
                    ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
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
                        ROCmGroupedBranchMutation mutation(*capture,static_cast<hipStream_t>(stream));
                        ASSERT_GE(mutation.nodeCount(),static_cast<size_t>(repeats));
                        for (auto omitted : {rocm::MoEGroupedBranchSuppression::Ordered,rocm::MoEGroupedBranchSuppression::Wmma})
                        {
                            mutation.suppress(omitted);
                            ASSERT_NO_FATAL_FAILURE(poison());
                            ASSERT_TRUE(graphs[path]->launch());
                            ASSERT_TRUE(backend->deviceToHost(observed_routes.data(),routes->gpu_data_ptr(),
                                observed_routes.size()*sizeof(float),parameter.device.ordinal,stream));
                            ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                            mutation.restore();
                            std::vector<int> changed(experts);
                            int surviving = 0, missing = 0;
                            for (size_t slot = 0; slot < route_ids.size(); ++slot)
                            {
                                const int expert = static_cast<int>(route_ids[slot]);
                                const bool omitted_expert = expert >= 0 &&
                                    ((counts[expert] < floor) == (omitted == rocm::MoEGroupedBranchSuppression::Ordered));
                                const int comparison = std::memcmp(observed_routes.data()+slot*hidden,
                                    reference_routes.data()+slot*hidden,hidden*sizeof(float));
                                if (omitted_expert)
                                {
                                    ++missing;
                                    if (comparison != 0) ++changed[expert];
                                }
                                else
                                {
                                    ASSERT_EQ(comparison,0) << dot_candidates[path].name << " surviving slot=" << slot;
                                    if (expert >= 0) ++surviving;
                                }
                            }
                            ASSERT_GT(missing,0); ASSERT_GT(surviving,0);
                            for (int expert = 0; expert < experts; ++expert)
                                if (counts[expert] > 0 &&
                                    ((counts[expert] < floor) == (omitted == rocm::MoEGroupedBranchSuppression::Ordered)))
                                    ASSERT_GT(changed[expert],0) << dot_candidates[path].name << " omitted expert=" << expert;
                            // Restoration must recover the complete real path
                            // after fresh poison, without another candidate run.
                            ASSERT_NO_FATAL_FAILURE(poison());
                            ASSERT_TRUE(graphs[path]->launch());
                            ASSERT_TRUE(backend->deviceToHost(observed_routes.data(),routes->gpu_data_ptr(),
                                observed_routes.size()*sizeof(float),parameter.device.ordinal,stream));
                            ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                            ASSERT_EQ(std::memcmp(observed_routes.data(),reference_routes.data(),
                                reference_routes.size()*sizeof(float)),0);
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
                    const std::array counts = macro ? std::array{3,4,7,13,15,16,17,31}
                                                    : std::array{3,4,7,8,12,16,17,31};
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
                const bool replay_only = mode == ExpertBenchmark::WmmaReplay || mode == ExpertBenchmark::WmmaHybridReplay ||
                    mode == ExpertBenchmark::WmmaMacroReplay;
                std::vector<int> live_points{parameter.live_rows};
                if (hybrid)
                {
                    live_points = {0,1,4,8,12,16,17,31,32,48,64,68,124,128,256,384,512};
                    std::erase_if(live_points,[&](int n) { return n > parameter.live_rows; });
                }
                if (timing_replay)
                {
                    live_points = {parameter.live_rows,0,1,3,4,7,8,11,12,15,16,17,31,parameter.live_rows,
                        0,1,16,17,31,parameter.live_rows};
                    std::erase_if(live_points,[&](int n) { return n > parameter.live_rows; });
                }
                if (population)
                {
                    live_points = {0,1,3,4,7,8,9,11,12,13,15,16,17,31,32,39,40,41,48,57,58,60,63,64,65,68,
                        124,128,256,384,480,481,496,511,512};
                    std::erase_if(live_points,[&](int n) { return n > parameter.live_rows; });
                }
                for (int live : live_points)
                {
                    ASSERT_NO_FATAL_FAILURE(verify(live));
                    ASSERT_NO_FATAL_FAILURE(verify_population_writers(live));
                    if (replay_only) continue;
                    auto reference = expected;
                    std::fill(reference.begin()+static_cast<size_t>(live)*hidden,reference.end(),0.0f);
                    for (int warmup = 0; warmup < 3; ++warmup)
                        for (auto *candidate : graphs) ASSERT_TRUE(candidate->launch());
                    std::vector<std::array<double,sample_count>> times(graphs.size());
                    for (int sample = 0; sample < (timing_replay ? 1 : sample_count); ++sample)
                    for (size_t order = 0; order < graphs.size(); ++order)
                    {
                        const size_t path = (sample + (sample % 2 ? graphs.size()-1-order : order)) % graphs.size();
                        poison();
                        ASSERT_TRUE(graphs[path]->launch());
                        ASSERT_TRUE(context.recordEventChecked(end,stream));
                        ASSERT_TRUE(context.synchronizeEventChecked(end));
                        const double us = consume_timing(path);
                        ASSERT_TRUE(std::isfinite(us) && us > 0);
                        times[path][sample] = us;
                        // Inspect this candidate before the next can repair its output.
                        ASSERT_TRUE(backend->deviceToHost(actual.data(),output->gpu_data_ptr(),actual.size()*sizeof(float),parameter.device.ordinal,stream));
                        ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                        ASSERT_EQ(std::memcmp(reference.data(),actual.data(),actual.size()*sizeof(float)),0)
                            << dot_candidates[path].name << " live=" << live;
                        ASSERT_NO_FATAL_FAILURE(verify_q8_publication(live));
                    }
                    if (timing_replay) continue;
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
                if (population && automatic_native_families)
                {
                    using Family = rocm::MoEGroupedPopulationFamily;
                    struct Histogram { std::vector<int> counts; Family expected; };
                    std::vector<Histogram> histograms;
                    const auto balanced = [&](int active,int count,Family family) {
                        histograms.push_back({std::vector<int>(active,count),family});
                    };
                    if (qwen_population)
                    {
                        balanced(32,6,Family::Dot4); balanced(32,7,Family::N32); balanced(32,8,Family::N32);
                        balanced(31,7,Family::N32); balanced(33,7,Family::Dot4);
                        balanced(64,11,Family::Dot4); balanced(64,12,Family::WideFloor16); balanced(64,13,Family::Dot4);
                        balanced(64,15,Family::Dot4); balanced(64,16,Family::N32); balanced(64,17,Family::N32);
                        for (int maximum : {4,5,6})
                        {
                            std::vector<int> counts(64,1); counts.back() = maximum;
                            histograms.push_back({counts,maximum == 4 ? Family::Dot4 : Family::WideFloor16});
                        }
                        auto near_cold = std::vector<int>(64,12); near_cold.back() = 13;
                        histograms.push_back({near_cold,Family::Dot4});
                        auto mixed = std::vector<int>(64,15); mixed.back() = 16;
                        histograms.push_back({mixed,Family::WideFloor16});
                        // Cross the retained worker bound in both roles with
                        // cold and native writers, then replay smaller traces.
                        mixed.assign(128,15); mixed.back() = 16;
                        histograms.push_back({mixed,Family::WideFloor16});
                        balanced(0,0,Family::Dot4); balanced(1,1,Family::Dot4);
                        balanced(63,1,Family::Dot4); balanced(64,1,Family::Dot4); balanced(65,1,Family::Dot4);
                        balanced(256,16,Family::N32); balanced(0,0,Family::Dot4);
                        balanced(256,12,Family::WideFloor16); balanced(33,7,Family::Dot4); balanced(256,16,Family::N32);
                        // Direct one-row GU work, cold12-row tails and retained
                        // direct/staged transitions share the production graph.
                        balanced(256,13,Family::Dot4);
                        balanced(256,14,Family::Dot4); // Two-row cold tails use the direct producer.
                        auto single_and_pair = std::vector<int>(256,1);
                        std::fill(single_and_pair.begin()+128,single_and_pair.end(),2);
                        histograms.push_back({single_and_pair,Family::Dot4});
                        std::reverse(single_and_pair.begin(),single_and_pair.end());
                        histograms.push_back({single_and_pair,Family::Dot4});
                        auto single_and_hot = std::vector<int>(128,1);
                        std::fill(single_and_hot.begin()+64,single_and_hot.end(),16);
                        histograms.push_back({single_and_hot,Family::WideFloor16});
                        // A native tile's one-row tail must stay on its native
                        // producer, even though it has the same live row count.
                        balanced(128,17,Family::N32);
                        balanced(128,18,Family::N32); // Two-row native tails retain WMMA ownership.
                        // Interior holes cross four-expert lane groups, the
                        // midpoint and the last lane. Mixed row steps must keep
                        // exact expert-major offsets through every zero gap.
                        std::vector<int> holed(256,0);
                        for (const auto [expert,count] : std::array<std::pair<int,int>,11>{{
                            {0,13},{2,15},{3,1},{4,16},{7,2},{8,12},
                            {127,13},{128,1},{251,16},{252,2},{255,17}}})
                            holed[expert] = count;
                        histograms.push_back({holed,Family::WideFloor16});
                    }
                    else
                    {
                        balanced(8,11,Family::Dot4); balanced(8,12,Family::N32); balanced(8,13,Family::N32);
                        balanced(7,16,Family::Dot4); balanced(8,512,Family::N32); balanced(0,0,Family::Dot4);
                        histograms.push_back({{15,16,16,16,16,16,16,16},Family::Dot4});
                        balanced(8,512,Family::N32);
                    }
                    const auto original_routes = route_ids, original_expected = expected;
                    const auto original_q8 = expected_q8, original_scales = expected_scales;
                    std::vector<float> original_input(input->numel()), changed_input(original_input.size());
                    const int live = parameter.rows;
                    const auto drain_trace = [&](void *) {
                        if (!backend->synchronizeStream(stream,parameter.device.ordinal)) std::terminate();
                    };
                    std::unique_ptr<void,decltype(drain_trace)> trace_retirement(&context,drain_trace);
                    ASSERT_TRUE(backend->deviceToHost(original_input.data(),input->gpu_data_ptr(),original_input.size()*sizeof(float),parameter.device.ordinal,stream));
                    ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                    for (size_t trace = 0; trace < histograms.size(); ++trace)
                    {
                        route_ids.assign(route_ids.size(),-1.0f);
                        std::vector<int> filled(parameter.rows);
                        int cold_span = parameter.rows, hot_span = 0;
                        if (histograms[trace].expected == Family::WideFloor16)
                        {
                            int cold_slots=0,hot_slots=0,cold_max=0,hot_max=0;
                            for (int count : histograms[trace].counts)
                                if (count < 16) { cold_slots+=count; cold_max=std::max(cold_max,count); }
                                else { hot_slots+=count; hot_max=std::max(hot_max,count); }
                            if (cold_slots && hot_slots)
                            {
                                cold_span=std::max(cold_max,(cold_slots+top_k-1)/top_k);
                                hot_span=std::max(hot_max,(hot_slots+top_k-1)/top_k);
                                ASSERT_LE(cold_span+hot_span,parameter.rows);
                            }
                        }
                        // Mixed arithmetic owns disjoint token spans so omission
                        // proves both missing outputs and untouched surviving bytes.
                        // Allocate each expert to distinct least-filled token rows;
                        // the holdout has no duplicate expert in a token's top-k.
                        for (size_t expert = 0; expert < histograms[trace].counts.size(); ++expert)
                        {
                            const bool hot = hot_span && histograms[trace].counts[expert] >= 16;
                            std::vector<int> rows(hot ? hot_span : cold_span);
                            std::iota(rows.begin(),rows.end(),hot ? cold_span : 0);
                            std::stable_sort(rows.begin(),rows.end(),[&](int a,int b) { return filled[a] < filled[b]; });
                            for (int n = 0; n < histograms[trace].counts[expert]; ++n)
                            {
                                const int row = rows[n];
                                ASSERT_LT(filled[row],top_k);
                                route_ids[row*top_k+filled[row]++] = static_cast<float>(expert);
                            }
                        }
                        for (size_t i = 0; i < original_input.size(); ++i)
                            changed_input[i] = original_input[i] * (trace % 2 ? -0.5f : 1.0f);
                        ASSERT_TRUE(backend->hostToDevice(input->gpu_data_ptr(),changed_input.data(),changed_input.size()*sizeof(float),parameter.device.ordinal,stream));
                        ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(),route_ids.data(),route_ids.size()*sizeof(float),parameter.device.ordinal,stream));
                        ASSERT_TRUE(backend->hostToDevice(device_live->gpu_data_ptr(),&live,sizeof(live),parameter.device.ordinal,stream));
                        poison();
                        ASSERT_TRUE(graphs[0]->launch());
                        ASSERT_TRUE(backend->deviceToHost(expected.data(),output->gpu_data_ptr(),expected.size()*sizeof(float),parameter.device.ordinal,stream));
                        ASSERT_TRUE(backend->deviceToHost(expected_q8.data(),router_publication->quantized_rows,expected_q8.size(),parameter.device.ordinal,stream));
                        ASSERT_TRUE(backend->deviceToHost(expected_scales.data(),router_publication->row_scales,expected_scales.size(),parameter.device.ordinal,stream));
                        ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                        ASSERT_EQ(expected_population(live),histograms[trace].expected) << "holdout=" << trace;
                        ASSERT_NO_FATAL_FAILURE(verify(live));
                        ASSERT_NO_FATAL_FAILURE(verify_population_writers(live));
                        ASSERT_NO_FATAL_FAILURE(verify_population_producer_omissions());
                    }
                    route_ids = original_routes; expected = original_expected;
                    expected_q8 = original_q8; expected_scales = original_scales;
                    ASSERT_TRUE(backend->hostToDevice(input->gpu_data_ptr(),original_input.data(),original_input.size()*sizeof(float),parameter.device.ordinal,stream));
                }
                ASSERT_NO_FATAL_FAILURE(verify(0));
                ASSERT_NO_FATAL_FAILURE(verify(parameter.live_rows));
                if (timing_replay || token_publication) return;
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
            const int columns = macro ? 96 : hidden / (parameter.ownership_degree ? parameter.ownership_degree : 2);
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
            const auto check_phases = [&](const char *engine_label, bool wide_candidate) {
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
#ifdef HAVE_ROCM
            if (macro)
            {
                // Warm phase values cannot certify a retained writer. This
                // transaction owns down projection, route publication and fold;
                // its existing grouped SwiGLU inputs are deliberately preserved.
                auto transaction = context.createGraphCapture(stream);
                {
                    ScopedBackendGraphCapture recording(context,*transaction,"dirty column-owned down and fold");
                    ASSERT_TRUE(recording.begin());
                    ASSERT_TRUE(column_down());
                    ASSERT_TRUE(column_fold());
                    recording.finish();
                }
                ASSERT_TRUE(transaction->instantiate());
                auto *captured = dynamic_cast<HIPGraphCapture *>(transaction.get());
                ASSERT_NE(captured,nullptr);
                if (wide_candidate)
                {
                    std::array<size_t,2> roles;
                    ASSERT_NO_THROW(roles = inspectROCmGroupedWideCapture(*captured,
                        static_cast<hipStream_t>(stream),rocm::MoEGroupedWmmaRowFloor::Sixteen));
                    EXPECT_EQ(roles,(std::array<size_t,2>{0,1}));
                }
                auto join = [&](void *) {
                    if (!backend->synchronizeStream(stream,parameter.device.ordinal)) std::terminate();
                };
                std::unique_ptr<void,decltype(join)> completion(&context,join);
                const auto poison_columns = [&] {
                    void *scratch = workspace.getBuffer(MoEWorkspaceBuffers::PREFILL_GATE);
                    const size_t bytes = workspace.getBufferSize(MoEWorkspaceBuffers::PREFILL_GATE);
                    ASSERT_NE(scratch,nullptr); ASSERT_GT(bytes,0u);
                    ASSERT_EQ(hipMemsetAsync(scratch,0xa5,bytes,static_cast<hipStream_t>(stream)),hipSuccess);
                    ASSERT_EQ(hipMemsetAsync(column_routes->gpu_data_ptr(),0xa5,column_routes->numel()*sizeof(float),
                        static_cast<hipStream_t>(stream)),hipSuccess);
                    ASSERT_EQ(hipMemsetAsync(column_output->gpu_data_ptr(),0xa5,column_output->numel()*sizeof(float),
                        static_cast<hipStream_t>(stream)),hipSuccess);
                };
                const auto replay_columns = [&] {
                    ASSERT_TRUE(transaction->launch());
                    ASSERT_TRUE(backend->deviceToHost(actual_columns.data(),column_output->gpu_data_ptr(),
                        actual_columns.size()*sizeof(float),parameter.device.ordinal,stream));
                    ASSERT_TRUE(backend->synchronizeStream(stream,parameter.device.ordinal));
                };
                const auto matching_columns = [&] {
                    for (int row = 0; row < parameter.rows; ++row)
                        if (std::memcmp(expected.data()+row*hidden+first_column,actual_columns.data()+row*columns,
                            columns*sizeof(float)) != 0) return false;
                    return true;
                };
                size_t count = 0;
                ASSERT_EQ(hipGraphGetNodes(captured->graph(),nullptr,&count),hipSuccess);
                std::vector<hipGraphNode_t> handles(count),down_writers,fold_writers;
                ASSERT_EQ(hipGraphGetNodes(captured->graph(),handles.data(),&count),hipSuccess);
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
                    if (symbol.find("grouped_wmma_detail") != std::string_view::npos)
                    {
                        ASSERT_EQ(params.gridDim.z,1u);
                        down_writers.push_back(handle);
                    }
                    if (symbol.find("moe_reduce_canonical_route_contributions_kernel") != std::string_view::npos)
                        fold_writers.push_back(handle);
                }
                ASSERT_EQ(down_writers.size(),1u); ASSERT_EQ(fold_writers.size(),1u);
                for (auto omitted : {down_writers.front(),fold_writers.front()})
                {
                    ASSERT_EQ(hipGraphNodeSetEnabled(captured->executable(),omitted,0),hipSuccess);
                    ASSERT_NO_FATAL_FAILURE(poison_columns());
                    ASSERT_NO_FATAL_FAILURE(replay_columns());
                    ASSERT_EQ(hipGraphNodeSetEnabled(captured->executable(),omitted,1),hipSuccess);
                    ASSERT_FALSE(matching_columns()) << "Omitted captured slice writer was masked by old bytes";
                    for (int replay = 0; replay < 2; ++replay)
                    {
                        ASSERT_NO_FATAL_FAILURE(poison_columns());
                        ASSERT_NO_FATAL_FAILURE(replay_columns());
                        ASSERT_TRUE(matching_columns()) << "Dirty retained slice changed source-exact bytes";
                    }
                }
            }
#endif
            };
#ifdef HAVE_ROCM
            if (native)
            {
                for (size_t path = 1; path < dot_candidates.size(); ++path)
                {
                    rocm::ScopedMoEGroupedPrefillEngine selected(dot_candidates[path].engine,dot_candidates[path].floor);
                    ASSERT_NO_FATAL_FAILURE(check_phases((dot_candidates[path].name+"_").c_str(),
                        dot_candidates[path].engine == rocm::MoEGroupedPrefillEngine::WideDot4Wmma));
                }
            }
            else
#endif
                ASSERT_NO_FATAL_FAILURE(check_phases("",false));
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
    /** @test Replay all internal event owners through partial, empty and threshold resets. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_HybridCapturedTimingReplay)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaHybridTimingReplay); }
    /** @test Qualify role-specific wide rows, branch omission and a 96-column down slice. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_MacroCapturedSplitReplay)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaMacroReplay); }
    /** @test Measure complete wide-candidate operations against both existing native tiles and Dot4. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_MacroCapturedTournament)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaMacroTournament); }
    /** @test Authenticate actual imported wide timing nodes through dirty threshold replays. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_MacroCapturedTimingReplay)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaMacroTimingReplay); }
    /** @test Replay ordinary router-Q8/direct-token publication through retained thresholds. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_PopulationReplay)
    { runExpertPipeline(GetParam(),ExpertBenchmark::PopulationReplay,ExpertPublication::RouterQ8TokenRows); }
    /** @brief Pair actual device-selected complete operations against all explicit candidates. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_PopulationTournament)
    { runExpertPipeline(GetParam(),ExpertBenchmark::PopulationTournament,ExpertPublication::RouterQ8TokenRows); }
    /** @brief Profile one complete Automatic operation outside canonical timings. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_PopulationProfile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::PopulationProfile,ExpertPublication::RouterQ8TokenRows); }
    /** @brief Attribute the measured fifteen-token Auto loss with unchanged512-row capacity. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_PopulationColdProfile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::PopulationProfile,ExpertPublication::RouterQ8TokenRows,
        PopulationProfileWork{.live_rows=15,.producer=PopulationProfileProducer::Automatic}); }
    /** @brief Profile sparse paired experts at the actual forty-token live workload. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_PopulationPairProfile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::PopulationProfile,ExpertPublication::RouterQ8TokenRows,
        PopulationProfileWork{.live_rows=40,.producer=PopulationProfileProducer::Automatic}); }
    /** @brief Profile the matching ordered baseline with the same router and live route histogram. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_PopulationOrderedColdProfile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::PopulationProfile,ExpertPublication::RouterQ8TokenRows,
        PopulationProfileWork{.live_rows=15,.producer=PopulationProfileProducer::Ordered}); }
    /** @brief Preserve the original explicit-candidate router publication replay. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_RouterQ8TokenReplay)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaMacroTimingReplay,ExpertPublication::RouterQ8TokenRows); }
    /** @test Time real routing, injected expert grouping and direct token publication together. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_RouterQ8TokenTournament)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaMacroTournament,ExpertPublication::RouterQ8TokenRows); }
    /** @test Profile one native direct-token operation including its real router producer. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_RouterQ8TokenProfile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaMacroProfile,ExpertPublication::RouterQ8TokenRows); }
    /** @test Isolate one complete full-capacity wide candidate for exact profiling. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_IsolatedMacroProfile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaMacroProfile); }
    /** @test Isolate wide cold work in the original full-capacity retained graph. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_IsolatedMacroColdProfile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaMacroColdProfile); }
    /** @test Isolate one N32 complete projection transaction outside canonical timing. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_IsolatedN32Profile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::Wmma32Profile); }
    /** @test Isolate full-capacity N32 without changing existing partial profile cases. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_IsolatedFullN32Profile)
    {
        auto parameter = GetParam();
        parameter.live_rows = parameter.rows;
        runExpertPipeline(parameter,ExpertBenchmark::Wmma32Profile);
    }
    /** @test Isolate one N64 complete projection transaction outside canonical timing. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_IsolatedN64Profile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::Wmma64Profile); }
    /** @test Profile one complete mixed-writer transaction, independently of event timing. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_IsolatedHybridN32Floor8Profile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaHybrid32Profile); }
    /** @test Isolate one complete explicit Dot4 baseline operation. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_IsolatedDot4Profile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::OrderedProfile); }
    /** @test Profile the Dot4 baseline at the same sparse live work and full capacity. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_IsolatedColdDot4Profile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::OrderedColdProfile); }
    /** @test Isolate one adaptive transaction with every tiled expert below floor8. */
    TEST_P(ROCmGroupedWmmaPerf, DISABLED_IsolatedAdaptiveColdN32Floor8Profile)
    { runExpertPipeline(GetParam(),ExpertBenchmark::WmmaAdaptive32ColdProfile); }
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
