/**
 * @file Perf__CUDAPersistentVerifierGeometry.cu
 * @brief Tests whether one cooperative CUDA geometry can economically execute a complete verifier schedule.
 *
 * The Qwen3.6-35B MoE all-position verifier currently captures 1,376 CUDA
 * kernel nodes. Those nodes use eight physical block sizes and range from one
 * logical CTA to thousands of CTAs. Replacing that graph with one monolithic
 * kernel is attractive only if a fixed cooperative grid can virtualize the
 * heterogeneous launch geometry more cheaply than CUDA Graph dispatches the
 * native launches.
 *
 * This benchmark isolates that architectural question before production
 * kernels are refactored into shared device tile bodies. The baseline graph
 * reproduces the measured node count, logical CTA count, and block-size mix.
 * Persistent candidates execute the same ordered phase program in one
 * cooperative kernel, with a grid-wide barrier between every former launch.
 * Candidate outputs and phase-to-phase state are compared byte-for-byte with
 * the native-geometry baseline on device; only one mismatch word is copied to
 * the host after a complete run.
 *
 * Three deliberately bounded proxy workloads separate the scheduler floor
 * from data-plane effects:
 *
 * - `cta-token` executes one deterministic item per logical CTA.
 * - `lane-token` executes one deterministic item per logical CUDA thread.
 * - `lane-alu8` executes the full lane population with a heavier integer body.
 *
 * These proxies do not claim to predict final verifier throughput. They answer
 * whether cooperative barriers and a fixed resident grid have a plausible
 * economy floor. If they win, the next experiment must call shared production
 * tile bodies and repeat correctness/resource profiling. If even the lower
 * bound loses, a one-kernel verifier is not justified by launch elimination.
 *
 * All measured graphs use persistent allocations and one explicit non-default
 * stream. The timed path contains no allocation, deallocation, transfer,
 * host callback, stream synchronization, or device synchronization. GPU events
 * delimit one complete graph replay. Set
 * `LLAMINAR_CUDA_PERSISTENT_VERIFIER_PROFILE` to `cta`, `lane`, or `alu8` to
 * reduce the run to one workload for Nsight Compute. Optional
 * `LLAMINAR_CUDA_PERSISTENT_VERIFIER_BLOCK` and
 * `LLAMINAR_CUDA_PERSISTENT_VERIFIER_BLOCKS_PER_SM` select one candidate.
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{
namespace
{
    namespace cg = cooperative_groups;

    constexpr int kWarmupReplays = 4;
    constexpr int kMeasuredSamples = 15;
    constexpr uint32_t kExpectedKernelNodes = 1376u;
    constexpr uint32_t kExpectedLogicalCTAs = 293290u;
    constexpr uint32_t kExpectedLogicalThreads = 45306146u;
    constexpr uint32_t kInitialPhaseState = 0x6a09e667u;

    /**
     * @brief Selects how much deterministic work each virtual launch performs.
     */
    enum class ProxyWorkload : uint32_t
    {
        CTAToken = 0, ///< One item per logical CTA; scheduler/barrier lower bound.
        LaneToken,    ///< One item per logical thread with a two-round body.
        LaneALU8      ///< One item per logical thread with an eight-round body.
    };

    /**
     * @brief One compact group of equivalent production launch geometries.
     *
     * The values were exported from the CUDA:0 depth-three, all-position
     * verifier inventory after the fused MoE grouping change. `logical_ctas`
     * is the flattened captured grid volume and `instances` is the number of
     * graph nodes with that geometry. Keeping groups compact makes the fixture
     * auditable while `buildPhaseProgram()` still expands the exact 1,376 phase
     * barriers seen by a monolithic executor.
     */
    struct GeometryGroup
    {
        uint32_t logical_ctas;
        uint32_t virtual_threads;
        uint32_t instances;
    };

    constexpr std::array<GeometryGroup, 59> kProductionGeometryGroups{{
        {4u, 1024u, 2u},
        {4u, 1024u, 79u},
        {128u, 128u, 30u},
        {512u, 128u, 39u},
        {512u, 128u, 1u},
        {64u, 128u, 41u},
        {128u, 128u, 10u},
        {5820u, 128u, 1u},
        {256u, 128u, 40u},
        {512u, 128u, 30u},
        {128u, 128u, 40u},
        {128u, 128u, 2u},
        {128u, 128u, 18u},
        {2048u, 128u, 1u},
        {2048u, 128u, 39u},
        {256u, 128u, 1u},
        {256u, 128u, 39u},
        {640u, 128u, 1u},
        {512u, 128u, 39u},
        {1u, 1u, 10u},
        {64u, 256u, 30u},
        {192u, 256u, 30u},
        {1u, 256u, 40u},
        {1u, 256u, 40u},
        {1u, 256u, 40u},
        {1u, 256u, 40u},
        {8u, 256u, 18u},
        {8u, 256u, 2u},
        {32u, 256u, 1u},
        {32u, 256u, 30u},
        {32u, 256u, 1u},
        {64u, 256u, 10u},
        {128u, 256u, 39u},
        {128u, 256u, 1u},
        {256u, 256u, 30u},
        {4u, 256u, 40u},
        {4u, 256u, 40u},
        {8u, 256u, 10u},
        {64u, 256u, 10u},
        {64u, 256u, 10u},
        {64u, 256u, 10u},
        {8u, 256u, 10u},
        {8u, 256u, 20u},
        {256u, 256u, 40u},
        {32u, 256u, 40u},
        {32u, 256u, 39u},
        {32u, 256u, 1u},
        {3880u, 256u, 1u},
        {1280u, 256u, 10u},
        {128u, 256u, 30u},
        {256u, 288u, 1u},
        {256u, 288u, 37u},
        {256u, 288u, 42u},
        {512u, 32u, 40u},
        {64u, 32u, 40u},
        {256u, 32u, 40u},
        {72u, 32u, 10u},
        {1u, 4u, 10u},
        {1u, 8u, 30u},
    }};

    /**
     * @brief Device-resident instruction for one former graph kernel node.
     */
    struct PhaseDescriptor
    {
        uint32_t logical_ctas = 0;
        uint32_t virtual_threads = 0;
        uint32_t cta_output_offset = 0;
        uint32_t lane_output_offset = 0;
    };

    /**
     * @brief Latency distribution for complete verifier schedule replays.
     */
    struct LatencySummary
    {
        double median_us = 0.0;
        double p95_us = 0.0;
        double minimum_us = 0.0;
        double maximum_us = 0.0;
    };

    /**
     * @brief CUDA compiler/runtime resource report for one persistent kernel.
     */
    struct KernelResources
    {
        int registers_per_thread = 0;
        size_t static_shared_bytes = 0;
        size_t local_bytes_per_thread = 0;
        int max_threads_per_block = 0;
        int max_active_blocks_per_sm = 0;
    };

    /**
     * @brief Own one captured graph and its instantiated executable.
     */
    class CapturedGraph final
    {
    public:
        CapturedGraph() = default;

        CapturedGraph(cudaGraph_t graph, cudaGraphExec_t executable, size_t nodes)
            : graph_(graph), executable_(executable), node_count_(nodes)
        {
        }

        ~CapturedGraph()
        {
            reset();
        }

        CapturedGraph(const CapturedGraph &) = delete;
        CapturedGraph &operator=(const CapturedGraph &) = delete;

        CapturedGraph(CapturedGraph &&other) noexcept
        {
            *this = std::move(other);
        }

        CapturedGraph &operator=(CapturedGraph &&other) noexcept
        {
            if (this == &other)
                return *this;
            reset();
            graph_ = std::exchange(other.graph_, nullptr);
            executable_ = std::exchange(other.executable_, nullptr);
            node_count_ = std::exchange(other.node_count_, 0u);
            return *this;
        }

        [[nodiscard]] cudaGraphExec_t executable() const noexcept
        {
            return executable_;
        }

        [[nodiscard]] size_t nodeCount() const noexcept
        {
            return node_count_;
        }

    private:
        void reset() noexcept
        {
            if (executable_ != nullptr)
                (void)cudaGraphExecDestroy(executable_);
            if (graph_ != nullptr)
                (void)cudaGraphDestroy(graph_);
            executable_ = nullptr;
            graph_ = nullptr;
            node_count_ = 0;
        }

        cudaGraph_t graph_ = nullptr;
        cudaGraphExec_t executable_ = nullptr;
        size_t node_count_ = 0;
    };

    /**
     * @brief Throw a precise diagnostic for a failed CUDA setup operation.
     */
    void requireCuda(cudaError_t status, const char *operation)
    {
        if (status == cudaSuccess)
            return;
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }

    /**
     * @brief Return one nearest-rank percentile from sorted samples.
     */
    double percentile(const std::vector<double> &sorted, double quantile)
    {
        if (sorted.empty())
            return 0.0;
        const double rank = quantile * static_cast<double>(sorted.size() - 1u);
        const size_t index = static_cast<size_t>(std::ceil(rank));
        return sorted[std::min(index, sorted.size() - 1u)];
    }

    /**
     * @brief Summarize complete-transaction latency samples.
     */
    LatencySummary summarize(std::vector<double> samples)
    {
        std::sort(samples.begin(), samples.end());
        if (samples.empty())
            return {};
        return LatencySummary{
            .median_us = percentile(samples, 0.50),
            .p95_us = percentile(samples, 0.95),
            .minimum_us = samples.front(),
            .maximum_us = samples.back(),
        };
    }

    /**
     * @brief Parse one positive profiler selector, returning zero when absent.
     */
    int positiveEnvironmentInteger(const char *name)
    {
        const char *raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0')
            return 0;
        const long value = std::strtol(raw, nullptr, 10);
        if (value <= 0 || value > std::numeric_limits<int>::max())
            return 0;
        return static_cast<int>(value);
    }

    /**
     * @brief Return the stable display name for a proxy workload.
     */
    constexpr const char *workloadName(ProxyWorkload workload)
    {
        switch (workload)
        {
        case ProxyWorkload::CTAToken:
            return "cta-token";
        case ProxyWorkload::LaneToken:
            return "lane-token";
        case ProxyWorkload::LaneALU8:
            return "lane-alu8";
        }
        return "unknown";
    }

    /**
     * @brief Determine whether profiler mode selected this workload.
     */
    bool workloadSelected(ProxyWorkload workload)
    {
        const char *raw =
            std::getenv("LLAMINAR_CUDA_PERSISTENT_VERIFIER_PROFILE");
        if (raw == nullptr || *raw == '\0')
            return true;
        const std::string selected(raw);
        if (selected == "cta")
            return workload == ProxyWorkload::CTAToken;
        if (selected == "lane")
            return workload == ProxyWorkload::LaneToken;
        if (selected == "alu8")
            return workload == ProxyWorkload::LaneALU8;
        throw std::runtime_error(
            "LLAMINAR_CUDA_PERSISTENT_VERIFIER_PROFILE must be cta, lane, or alu8");
    }

    /**
     * @brief Mix one word with a compile-time arithmetic depth.
     *
     * Integer arithmetic gives baseline and persistent executors one exact
     * comparison oracle. Inputs remain data-dependent, preventing the compiler
     * from folding the proxy body into constants.
     */
    template <int Rounds>
    __device__ __forceinline__ uint32_t mixWord(
        uint32_t value,
        uint32_t phase,
        uint32_t item)
    {
        uint32_t mixed = value ^ (phase * 0x9e3779b9u) ^ item;
#pragma unroll
        for (int round = 0; round < Rounds; ++round)
        {
            mixed ^= mixed >> 16;
            mixed *= 0x7feb352du + static_cast<uint32_t>(round * 2);
            mixed ^= mixed >> 15;
            mixed *= 0x846ca68bu + static_cast<uint32_t>(round * 4);
            mixed ^= mixed >> 16;
        }
        return mixed;
    }

    /**
     * @brief Apply the selected proxy body without a runtime arithmetic branch.
     */
    template <ProxyWorkload Workload>
    __device__ __forceinline__ uint32_t executeProxyBody(
        uint32_t value,
        uint32_t phase,
        uint32_t item)
    {
        if constexpr (Workload == ProxyWorkload::LaneALU8)
            return mixWord<8>(value, phase, item);
        return mixWord<2>(value, phase, item);
    }

    /**
     * @brief Initialize the immutable input corpus entirely on the device.
     */
    __global__ void initializeInputKernel(uint32_t *input, uint32_t item_count)
    {
        const uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t stride = blockDim.x * gridDim.x;
        for (uint32_t item = first; item < item_count; item += stride)
            input[item] = mixWord<2>(0x243f6a88u, 0u, item);
    }

    /**
     * @brief Initialize only the first state word consumed by a schedule.
     */
    __global__ void initializePhaseStateKernel(uint32_t *state)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
            state[0] = kInitialPhaseState;
    }

    /**
     * @brief Execute one former graph node at its native CUDA block geometry.
     *
     * CUDA stream ordering publishes `phase_state[phase + 1]` before the next
     * graph node begins. The persistent executor below implements the same edge
     * with a cooperative grid barrier.
     */
    template <int VirtualThreads, ProxyWorkload Workload>
    __global__ void nativePhaseKernel(
        PhaseDescriptor descriptor,
        uint32_t phase,
        const uint32_t *input,
        uint32_t *output,
        uint32_t *phase_state)
    {
        static_assert(VirtualThreads > 0);
        const uint32_t seed = phase_state[phase];

        if constexpr (Workload == ProxyWorkload::CTAToken)
        {
            if (threadIdx.x == 0)
            {
                const uint32_t item = blockIdx.x;
                const uint32_t output_index =
                    descriptor.cta_output_offset + item;
                output[output_index] = executeProxyBody<Workload>(
                    input[output_index], phase, item);
            }
        }
        else
        {
            const uint32_t item =
                blockIdx.x * static_cast<uint32_t>(VirtualThreads) +
                threadIdx.x;
            const uint32_t output_index =
                descriptor.lane_output_offset + item;
            output[output_index] = executeProxyBody<Workload>(
                input[output_index], phase, item);
        }

        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            phase_state[phase + 1u] = executeProxyBody<Workload>(
                seed,
                phase,
                descriptor.logical_ctas ^ descriptor.virtual_threads);
        }
    }

    /**
     * @brief Execute every verifier phase inside one fixed cooperative grid.
     *
     * Physical CUDA threads cooperatively traverse each phase's logical item
     * population. `grid.sync()` is the exact global producer/consumer boundary
     * that replaces stream ordering between native graph nodes. No host state,
     * child launch, dynamic allocation, transfer, or device synchronization is
     * involved.
     */
    template <int PhysicalThreads, ProxyWorkload Workload>
    __global__ void persistentVerifierGeometryKernel(
        const PhaseDescriptor *phases,
        uint32_t phase_count,
        const uint32_t *input,
        uint32_t *output,
        uint32_t *phase_state)
    {
        static_assert(PhysicalThreads > 0);
        const cg::grid_group grid = cg::this_grid();
        const uint32_t physical_thread =
            blockIdx.x * static_cast<uint32_t>(PhysicalThreads) + threadIdx.x;
        const uint32_t physical_stride =
            gridDim.x * static_cast<uint32_t>(PhysicalThreads);

        for (uint32_t phase = 0; phase < phase_count; ++phase)
        {
            const PhaseDescriptor descriptor = phases[phase];
            const uint32_t seed = phase_state[phase];
            const uint32_t items_per_cta =
                Workload == ProxyWorkload::CTAToken
                    ? 1u
                    : descriptor.virtual_threads;
            const uint32_t item_count =
                descriptor.logical_ctas * items_per_cta;
            const uint32_t output_offset =
                Workload == ProxyWorkload::CTAToken
                    ? descriptor.cta_output_offset
                    : descriptor.lane_output_offset;

            for (uint32_t item = physical_thread;
                 item < item_count;
                 item += physical_stride)
            {
                const uint32_t output_index = output_offset + item;
                output[output_index] = executeProxyBody<Workload>(
                    input[output_index], phase, item);
            }

            if (physical_thread == 0)
            {
                phase_state[phase + 1u] = executeProxyBody<Workload>(
                    seed,
                    phase,
                    descriptor.logical_ctas ^ descriptor.virtual_threads);
            }

            /*
             * Every physical block must observe both the scalar publication and
             * all phase data writes before any thread consumes the next phase.
             * Cooperative launch guarantees that the complete grid is resident,
             * so this barrier cannot deadlock through unscheduled blocks.
             */
            grid.sync();
        }
    }

    /**
     * @brief Compare two device buffers and publish one terminal mismatch flag.
     *
     * The atomic belongs only to post-benchmark diagnostic validation. It is
     * not captured, timed, or part of the proposed production executor.
     */
    __global__ void compareExactKernel(
        const uint32_t *expected,
        const uint32_t *actual,
        uint32_t item_count,
        uint32_t *mismatch)
    {
        const uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t stride = blockDim.x * gridDim.x;
        for (uint32_t item = first; item < item_count; item += stride)
        {
            if (expected[item] != actual[item])
            {
                atomicExch(mismatch, 1u);
                return;
            }
        }
    }

    /**
     * @brief Launch one baseline phase with its captured native block size.
     */
    template <ProxyWorkload Workload>
    void launchNativePhase(
        const PhaseDescriptor &descriptor,
        uint32_t phase,
        const uint32_t *input,
        uint32_t *output,
        uint32_t *phase_state,
        cudaStream_t stream)
    {
        const dim3 grid(descriptor.logical_ctas);
        switch (descriptor.virtual_threads)
        {
        case 1u:
            nativePhaseKernel<1, Workload><<<grid, 1, 0, stream>>>(
                descriptor, phase, input, output, phase_state);
            return;
        case 4u:
            nativePhaseKernel<4, Workload><<<grid, 4, 0, stream>>>(
                descriptor, phase, input, output, phase_state);
            return;
        case 8u:
            nativePhaseKernel<8, Workload><<<grid, 8, 0, stream>>>(
                descriptor, phase, input, output, phase_state);
            return;
        case 32u:
            nativePhaseKernel<32, Workload><<<grid, 32, 0, stream>>>(
                descriptor, phase, input, output, phase_state);
            return;
        case 128u:
            nativePhaseKernel<128, Workload><<<grid, 128, 0, stream>>>(
                descriptor, phase, input, output, phase_state);
            return;
        case 256u:
            nativePhaseKernel<256, Workload><<<grid, 256, 0, stream>>>(
                descriptor, phase, input, output, phase_state);
            return;
        case 288u:
            nativePhaseKernel<288, Workload><<<grid, 288, 0, stream>>>(
                descriptor, phase, input, output, phase_state);
            return;
        case 1024u:
            nativePhaseKernel<1024, Workload><<<grid, 1024, 0, stream>>>(
                descriptor, phase, input, output, phase_state);
            return;
        default:
            throw std::runtime_error(
                "production geometry contains an unsupported block size");
        }
    }

    /**
     * @brief Expand compact geometry groups into the exact phase program.
     */
    std::vector<PhaseDescriptor> buildPhaseProgram()
    {
        std::vector<PhaseDescriptor> phases;
        phases.reserve(kExpectedKernelNodes);
        uint64_t cta_offset = 0;
        uint64_t lane_offset = 0;

        for (const GeometryGroup &group : kProductionGeometryGroups)
        {
            for (uint32_t instance = 0; instance < group.instances; ++instance)
            {
                if (cta_offset > std::numeric_limits<uint32_t>::max() ||
                    lane_offset > std::numeric_limits<uint32_t>::max())
                {
                    throw std::runtime_error(
                        "production verifier proxy offsets exceed uint32_t");
                }
                phases.push_back(PhaseDescriptor{
                    .logical_ctas = group.logical_ctas,
                    .virtual_threads = group.virtual_threads,
                    .cta_output_offset = static_cast<uint32_t>(cta_offset),
                    .lane_output_offset = static_cast<uint32_t>(lane_offset),
                });
                cta_offset += group.logical_ctas;
                lane_offset +=
                    static_cast<uint64_t>(group.logical_ctas) *
                    group.virtual_threads;
            }
        }

        if (phases.size() != kExpectedKernelNodes ||
            cta_offset != kExpectedLogicalCTAs ||
            lane_offset != kExpectedLogicalThreads)
        {
            throw std::runtime_error(
                "production verifier geometry fixture totals are inconsistent");
        }
        return phases;
    }

    /**
     * @brief Fixture owning all persistent buffers and explicit timing objects.
     */
    class Perf__CUDAPersistentVerifierGeometry : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
                device_count <= 0)
            {
                GTEST_SKIP() << "A CUDA device is required";
            }

            requireCuda(cudaSetDevice(0), "cudaSetDevice");
            requireCuda(
                cudaGetDeviceProperties(&device_properties_, 0),
                "cudaGetDeviceProperties");
            if (device_properties_.cooperativeLaunch == 0)
            {
                GTEST_SKIP() << "The CUDA device does not support cooperative launch";
            }

            phases_ = buildPhaseProgram();
            requireCuda(
                cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags");
            requireCuda(
                cudaEventCreate(&start_event_),
                "cudaEventCreate(start)");
            requireCuda(
                cudaEventCreate(&stop_event_),
                "cudaEventCreate(stop)");
            requireCuda(
                cudaEventCreateWithFlags(&completion_event_, cudaEventDisableTiming),
                "cudaEventCreate(completion)");

            const size_t descriptor_bytes =
                phases_.size() * sizeof(PhaseDescriptor);
            const size_t output_bytes =
                static_cast<size_t>(kExpectedLogicalThreads) * sizeof(uint32_t);
            const size_t state_bytes =
                (phases_.size() + 1u) * sizeof(uint32_t);
            requireCuda(
                cudaMalloc(reinterpret_cast<void **>(&device_phases_), descriptor_bytes),
                "cudaMalloc(phases)");
            requireCuda(
                cudaMalloc(reinterpret_cast<void **>(&device_input_), output_bytes),
                "cudaMalloc(input)");
            requireCuda(
                cudaMalloc(reinterpret_cast<void **>(&baseline_output_), output_bytes),
                "cudaMalloc(baseline output)");
            requireCuda(
                cudaMalloc(reinterpret_cast<void **>(&persistent_output_), output_bytes),
                "cudaMalloc(persistent output)");
            requireCuda(
                cudaMalloc(reinterpret_cast<void **>(&baseline_state_), state_bytes),
                "cudaMalloc(baseline state)");
            requireCuda(
                cudaMalloc(reinterpret_cast<void **>(&persistent_state_), state_bytes),
                "cudaMalloc(persistent state)");
            requireCuda(
                cudaMalloc(reinterpret_cast<void **>(&device_mismatch_), sizeof(uint32_t)),
                "cudaMalloc(mismatch)");

            requireCuda(
                cudaMemcpyAsync(
                    device_phases_,
                    phases_.data(),
                    descriptor_bytes,
                    cudaMemcpyHostToDevice,
                    stream_),
                "cudaMemcpyAsync(phases)");
            constexpr uint32_t kInitThreads = 256u;
            constexpr uint32_t kInitBlocks = 1024u;
            initializeInputKernel<<<kInitBlocks, kInitThreads, 0, stream_>>>(
                device_input_, kExpectedLogicalThreads);
            initializePhaseStateKernel<<<1, 1, 0, stream_>>>(baseline_state_);
            initializePhaseStateKernel<<<1, 1, 0, stream_>>>(persistent_state_);
            requireCuda(cudaGetLastError(), "CUDA setup kernels");
            requireCuda(
                cudaEventRecord(completion_event_, stream_),
                "cudaEventRecord(setup completion)");
            requireCuda(
                cudaEventSynchronize(completion_event_),
                "cudaEventSynchronize(setup completion)");
        }

        void TearDown() override
        {
            if (device_mismatch_ != nullptr)
                (void)cudaFree(device_mismatch_);
            if (persistent_state_ != nullptr)
                (void)cudaFree(persistent_state_);
            if (baseline_state_ != nullptr)
                (void)cudaFree(baseline_state_);
            if (persistent_output_ != nullptr)
                (void)cudaFree(persistent_output_);
            if (baseline_output_ != nullptr)
                (void)cudaFree(baseline_output_);
            if (device_input_ != nullptr)
                (void)cudaFree(device_input_);
            if (device_phases_ != nullptr)
                (void)cudaFree(device_phases_);
            if (completion_event_ != nullptr)
                (void)cudaEventDestroy(completion_event_);
            if (stop_event_ != nullptr)
                (void)cudaEventDestroy(stop_event_);
            if (start_event_ != nullptr)
                (void)cudaEventDestroy(start_event_);
            if (stream_ != nullptr)
                (void)cudaStreamDestroy(stream_);
        }

        /**
         * @brief Instantiate a graph after capture and count its native nodes.
         */
        CapturedGraph finishCapture()
        {
            cudaGraph_t graph = nullptr;
            requireCuda(
                cudaStreamEndCapture(stream_, &graph),
                "cudaStreamEndCapture");

            cudaGraphExec_t executable = nullptr;
            const cudaError_t instantiate_status = cudaGraphInstantiate(
                &executable, graph, nullptr, nullptr, 0);
            if (instantiate_status != cudaSuccess)
            {
                (void)cudaGraphDestroy(graph);
                requireCuda(instantiate_status, "cudaGraphInstantiate");
            }

            size_t node_count = 0;
            const cudaError_t node_status =
                cudaGraphGetNodes(graph, nullptr, &node_count);
            if (node_status != cudaSuccess)
            {
                (void)cudaGraphExecDestroy(executable);
                (void)cudaGraphDestroy(graph);
                requireCuda(node_status, "cudaGraphGetNodes");
            }
            return CapturedGraph(graph, executable, node_count);
        }

        /**
         * @brief Capture the complete native-geometry proxy schedule.
         */
        template <ProxyWorkload Workload>
        CapturedGraph captureBaseline()
        {
            requireCuda(
                cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
                "cudaStreamBeginCapture(baseline)");
            for (uint32_t phase = 0;
                 phase < static_cast<uint32_t>(phases_.size());
                 ++phase)
            {
                launchNativePhase<Workload>(
                    phases_[phase],
                    phase,
                    device_input_,
                    baseline_output_,
                    baseline_state_,
                    stream_);
            }
            requireCuda(cudaGetLastError(), "native phase graph capture");
            return finishCapture();
        }

        /**
         * @brief Query compiler resources and occupancy for one candidate.
         */
        template <int PhysicalThreads, ProxyWorkload Workload>
        KernelResources queryPersistentResources()
        {
            cudaFuncAttributes attributes{};
            requireCuda(
                cudaFuncGetAttributes(
                    &attributes,
                    persistentVerifierGeometryKernel<PhysicalThreads, Workload>),
                "cudaFuncGetAttributes(persistent verifier)");
            int active_blocks = 0;
            requireCuda(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &active_blocks,
                    persistentVerifierGeometryKernel<PhysicalThreads, Workload>,
                    PhysicalThreads,
                    0),
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
            return KernelResources{
                .registers_per_thread = attributes.numRegs,
                .static_shared_bytes = attributes.sharedSizeBytes,
                .local_bytes_per_thread = attributes.localSizeBytes,
                .max_threads_per_block = attributes.maxThreadsPerBlock,
                .max_active_blocks_per_sm = active_blocks,
            };
        }

        /**
         * @brief Capture one cooperative persistent candidate as one graph node.
         */
        template <int PhysicalThreads, ProxyWorkload Workload>
        CapturedGraph capturePersistent(int blocks_per_sm)
        {
            const KernelResources resources =
                queryPersistentResources<PhysicalThreads, Workload>();
            if (blocks_per_sm <= 0 ||
                blocks_per_sm > resources.max_active_blocks_per_sm)
            {
                throw std::runtime_error(
                    "persistent candidate exceeds its occupancy ceiling");
            }

            uint32_t phase_count =
                static_cast<uint32_t>(phases_.size());
            const uint32_t grid_blocks = static_cast<uint32_t>(
                device_properties_.multiProcessorCount * blocks_per_sm);
            const PhaseDescriptor *phases = device_phases_;
            const uint32_t *input = device_input_;
            uint32_t *output = persistent_output_;
            uint32_t *state = persistent_state_;
            void *arguments[] = {
                &phases,
                &phase_count,
                &input,
                &output,
                &state,
            };

            requireCuda(
                cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
                "cudaStreamBeginCapture(persistent)");
            const cudaError_t launch_status = cudaLaunchCooperativeKernel(
                reinterpret_cast<const void *>(
                    persistentVerifierGeometryKernel<PhysicalThreads, Workload>),
                dim3(grid_blocks),
                dim3(PhysicalThreads),
                arguments,
                0,
                stream_);
            if (launch_status != cudaSuccess)
            {
                cudaGraph_t abandoned_graph = nullptr;
                (void)cudaStreamEndCapture(stream_, &abandoned_graph);
                if (abandoned_graph != nullptr)
                    (void)cudaGraphDestroy(abandoned_graph);
                requireCuda(
                    launch_status,
                    "cudaLaunchCooperativeKernel during graph capture");
            }
            return finishCapture();
        }

        /**
         * @brief Measure complete graph replays with explicit CUDA events.
         */
        LatencySummary measure(const CapturedGraph &captured)
        {
            const bool profiler_mode =
                std::getenv("LLAMINAR_CUDA_PERSISTENT_VERIFIER_PROFILE") != nullptr;
            const int warmups = profiler_mode ? 1 : kWarmupReplays;
            const int samples = profiler_mode ? 1 : kMeasuredSamples;
            for (int replay = 0; replay < warmups; ++replay)
            {
                requireCuda(
                    cudaGraphLaunch(captured.executable(), stream_),
                    "cudaGraphLaunch(warmup)");
            }

            std::vector<double> latency_samples;
            latency_samples.reserve(static_cast<size_t>(samples));
            for (int sample = 0; sample < samples; ++sample)
            {
                requireCuda(
                    cudaEventRecord(start_event_, stream_),
                    "cudaEventRecord(start)");
                requireCuda(
                    cudaGraphLaunch(captured.executable(), stream_),
                    "cudaGraphLaunch(timed)");
                requireCuda(
                    cudaEventRecord(stop_event_, stream_),
                    "cudaEventRecord(stop)");
                requireCuda(
                    cudaEventSynchronize(stop_event_),
                    "cudaEventSynchronize(complete transaction)");
                float elapsed_ms = 0.0f;
                requireCuda(
                    cudaEventElapsedTime(
                        &elapsed_ms, start_event_, stop_event_),
                    "cudaEventElapsedTime");
                latency_samples.push_back(
                    static_cast<double>(elapsed_ms) * 1000.0);
            }
            return summarize(std::move(latency_samples));
        }

        /**
         * @brief Prove one persistent result is byte-identical to baseline.
         */
        void expectExactResult(ProxyWorkload workload)
        {
            const uint32_t output_items =
                workload == ProxyWorkload::CTAToken
                    ? kExpectedLogicalCTAs
                    : kExpectedLogicalThreads;
            requireCuda(
                cudaMemsetAsync(
                    device_mismatch_, 0, sizeof(uint32_t), stream_),
                "cudaMemsetAsync(mismatch)");
            constexpr uint32_t kCompareThreads = 256u;
            constexpr uint32_t kCompareBlocks = 1024u;
            compareExactKernel<<<kCompareBlocks, kCompareThreads, 0, stream_>>>(
                baseline_output_,
                persistent_output_,
                output_items,
                device_mismatch_);
            compareExactKernel<<<32, kCompareThreads, 0, stream_>>>(
                baseline_state_,
                persistent_state_,
                static_cast<uint32_t>(phases_.size() + 1u),
                device_mismatch_);
            requireCuda(cudaGetLastError(), "compareExactKernel");

            uint32_t mismatch = 1u;
            requireCuda(
                cudaMemcpyAsync(
                    &mismatch,
                    device_mismatch_,
                    sizeof(mismatch),
                    cudaMemcpyDeviceToHost,
                    stream_),
                "cudaMemcpyAsync(final mismatch)");
            requireCuda(
                cudaEventRecord(completion_event_, stream_),
                "cudaEventRecord(comparison completion)");
            requireCuda(
                cudaEventSynchronize(completion_event_),
                "cudaEventSynchronize(comparison completion)");
            EXPECT_EQ(mismatch, 0u)
                << workloadName(workload)
                << " persistent schedule is not byte-identical to baseline";
        }

        /**
         * @brief Measure and print one fixed-block/fixed-residency candidate.
         */
        template <int PhysicalThreads, ProxyWorkload Workload>
        void measureCandidate(
            int blocks_per_sm,
            const LatencySummary &baseline)
        {
            const int selected_block = positiveEnvironmentInteger(
                "LLAMINAR_CUDA_PERSISTENT_VERIFIER_BLOCK");
            const int selected_residency = positiveEnvironmentInteger(
                "LLAMINAR_CUDA_PERSISTENT_VERIFIER_BLOCKS_PER_SM");
            if ((selected_block > 0 && selected_block != PhysicalThreads) ||
                (selected_residency > 0 && selected_residency != blocks_per_sm))
            {
                return;
            }

            const KernelResources resources =
                queryPersistentResources<PhysicalThreads, Workload>();
            ASSERT_LE(blocks_per_sm, resources.max_active_blocks_per_sm);
            ASSERT_LE(PhysicalThreads, resources.max_threads_per_block);
            EXPECT_EQ(resources.local_bytes_per_thread, 0u)
                << "persistent scheduler spilled before production bodies were added";

            CapturedGraph candidate =
                capturePersistent<PhysicalThreads, Workload>(blocks_per_sm);
            ASSERT_EQ(candidate.nodeCount(), 1u)
                << "the persistent candidate must capture as one kernel node";
            const LatencySummary measured = measure(candidate);
            expectExactResult(Workload);

            const int grid_blocks =
                device_properties_.multiProcessorCount * blocks_per_sm;
            const double resident_thread_fraction =
                100.0 * static_cast<double>(
                            blocks_per_sm * PhysicalThreads) /
                static_cast<double>(
                    device_properties_.maxThreadsPerMultiProcessor);
            const double speedup = measured.median_us > 0.0
                                       ? baseline.median_us / measured.median_us
                                       : 0.0;
            std::cout << std::left << std::setw(12) << workloadName(Workload)
                      << std::right << std::setw(8) << PhysicalThreads
                      << std::setw(10) << blocks_per_sm
                      << std::setw(10) << grid_blocks
                      << std::setw(8) << resources.registers_per_thread
                      << std::setw(10) << resources.local_bytes_per_thread
                      << std::setw(11) << std::fixed << std::setprecision(1)
                      << resident_thread_fraction
                      << std::setw(13) << std::setprecision(3)
                      << measured.median_us
                      << std::setw(13) << measured.p95_us
                      << std::setw(11) << std::setprecision(3) << speedup
                      << '\n';
        }

        /**
         * @brief Sweep distinct useful residency levels for one block size.
         */
        template <int PhysicalThreads, ProxyWorkload Workload>
        void sweepBlockSize(const LatencySummary &baseline)
        {
            const KernelResources resources =
                queryPersistentResources<PhysicalThreads, Workload>();
            std::set<int> residencies{
                1,
                std::min(2, resources.max_active_blocks_per_sm),
                resources.max_active_blocks_per_sm,
            };
            for (int blocks_per_sm : residencies)
            {
                if (blocks_per_sm > 0)
                {
                    measureCandidate<PhysicalThreads, Workload>(
                        blocks_per_sm, baseline);
                }
            }
        }

        /**
         * @brief Benchmark one proxy workload across the geometry tournament.
         */
        template <ProxyWorkload Workload>
        void runWorkload()
        {
            if (!workloadSelected(Workload))
                return;

            CapturedGraph baseline = captureBaseline<Workload>();
            ASSERT_EQ(baseline.nodeCount(), phases_.size());
            const LatencySummary baseline_latency = measure(baseline);
            std::cout << "baseline " << workloadName(Workload)
                      << ": nodes=" << baseline.nodeCount()
                      << " median_us=" << std::fixed << std::setprecision(3)
                      << baseline_latency.median_us
                      << " p95_us=" << baseline_latency.p95_us << '\n';

            sweepBlockSize<128, Workload>(baseline_latency);
            sweepBlockSize<256, Workload>(baseline_latency);
            sweepBlockSize<288, Workload>(baseline_latency);
            sweepBlockSize<512, Workload>(baseline_latency);
        }

        cudaDeviceProp device_properties_{};
        cudaStream_t stream_ = nullptr;
        cudaEvent_t start_event_ = nullptr;
        cudaEvent_t stop_event_ = nullptr;
        cudaEvent_t completion_event_ = nullptr;
        std::vector<PhaseDescriptor> phases_;
        PhaseDescriptor *device_phases_ = nullptr;
        uint32_t *device_input_ = nullptr;
        uint32_t *baseline_output_ = nullptr;
        uint32_t *persistent_output_ = nullptr;
        uint32_t *baseline_state_ = nullptr;
        uint32_t *persistent_state_ = nullptr;
        uint32_t *device_mismatch_ = nullptr;
    };

    /**
     * @brief Measure the fixed-geometry lower bound for the production schedule.
     */
    TEST_F(
        Perf__CUDAPersistentVerifierGeometry,
        Qwen36MoEAllPositionVerifierGeometryTournament)
    {
        std::cout << "CUDA device: " << device_properties_.name
                  << " SMs=" << device_properties_.multiProcessorCount
                  << " verifier_nodes=" << phases_.size()
                  << " logical_ctas=" << kExpectedLogicalCTAs
                  << " logical_threads=" << kExpectedLogicalThreads << '\n'
                  << std::left << std::setw(12) << "workload"
                  << std::right << std::setw(8) << "block"
                  << std::setw(10) << "blocks/SM"
                  << std::setw(10) << "grid"
                  << std::setw(8) << "regs"
                  << std::setw(10) << "local_B"
                  << std::setw(11) << "threads_%"
                  << std::setw(13) << "median_us"
                  << std::setw(13) << "p95_us"
                  << std::setw(11) << "speedup"
                  << '\n';

        runWorkload<ProxyWorkload::CTAToken>();
        runWorkload<ProxyWorkload::LaneToken>();
        runWorkload<ProxyWorkload::LaneALU8>();
    }
} // namespace
} // namespace llaminar2

#endif // HAVE_CUDA
