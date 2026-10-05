/**
 * @file CUDACapturedKernelProbe.h
 * @brief Communication-free timings for exact model-free CUDA kernel fixtures.
 *
 * This diagnostic copies the warmed fixture's physical kernel nodes into
 * independent sequential graphs. It retains every argument and node attribute,
 * including embedded buffer addresses, and never enters conditional bodies.
 * Timings include native graph scheduling; compiler occupancy is a ceiling,
 * not achieved occupancy. Callers retain all buffers and replay their complete
 * fixture afterward before certifying output bytes. This is not a trainer or
 * a production execution mode.
 */
#pragma once

#include <cuda_runtime.h>
#include "../../../external/vendor/nlohmann/json.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test
{
    /** Explicit witnesses for fixtures whose repeated writes are independent. */
    enum class CUDAIndependentKernelFixture
    {
        StatelessProjection, ///< Immutable weights/activations, overwritten outputs.
        ZeroRecurrentFixedPoint, ///< Initialized zero state remains exactly zero.
        ReadOnlyAttentionCache, ///< Immutable Q/K/V and overwritten scratch/output.
    };

    /** Exact workload identity supplied by the fixture owning the buffers. */
    struct CUDACapturedKernelProbeIdentity
    {
        std::string operation; ///< Projection or recurrence role, not a kernel name.
        std::string format; ///< Actual source precision/format.
        int m = 0; ///< Participant-local physical rows.
        int n = 0; ///< Participant-local output width or head count.
        int k = 0; ///< Participant-local reduction or context width.
        int tp_degree = 1; ///< Local geometry label; the probe runs on one device.
    };

    /** Native-event repetition policy, independent of any production default. */
    struct CUDACapturedKernelProbeConfig
    {
        int warmup_runs = 5; ///< Untimed isolated graph submissions.
        int sample_count = 31; ///< Retained independent event observations.
        int operations_per_sample = 16; ///< Sequential identical physical nodes.
    };

    namespace cuda_kernel_probe_detail
    {
        /** Throw the exact runtime failure instead of changing measurement mode.
         * @param result CUDA operation status.
         * @param operation Named setup, replay, or observation boundary.
         */
        inline void require(cudaError_t result, const char *operation)
        {
            if (result != cudaSuccess)
                throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
        }

        /** Own one timing event until its retained sample batch completes. */
        struct TimingEvent
        {
            cudaEvent_t handle = nullptr;
            /** Create a real timing event, outside capture and measurement. */
            TimingEvent() { require(cudaEventCreateWithFlags(&handle, cudaEventDefault), "create probe timing event"); }
            /** CUDA defers destruction of a pending event until its consumers retire. */
            ~TimingEvent() { if (handle) (void)cudaEventDestroy(handle); }
            TimingEvent(const TimingEvent &) = delete;
            TimingEvent &operator=(const TimingEvent &) = delete;
        };

        /** Retain one isolated graph and executable through terminal observation. */
        struct GraphOwner
        {
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            /** Create the native graph before any measured submission. */
            GraphOwner() { require(cudaGraphCreate(&graph, 0), "create isolated probe graph"); }
            /** Retire executable references before the graph's argument storage. */
            ~GraphOwner()
            {
                if (executable) (void)cudaGraphExecDestroy(executable);
                if (graph) (void)cudaGraphDestroy(graph);
            }
            GraphOwner(const GraphOwner &) = delete;
            GraphOwner &operator=(const GraphOwner &) = delete;
        };

        /** Copy every compiled argument byte for complete capture identity.
         * @param params Parameters borrowed from the retained parent graph.
         * @return Argument values, including pointers and policy scalars.
         * @throws std::runtime_error for opaque/unsupported argument storage.
         */
        inline std::vector<std::uint8_t> argumentIdentity(const cudaKernelNodeParams &params)
        {
            if (!params.kernelParams || params.extra)
                throw std::runtime_error("isolated probe requires explicit kernel parameter storage");
            std::vector<std::uint8_t> identity;
            for (std::size_t index = 0; index < 128; ++index)
            {
                std::size_t offset = 0, bytes = 0;
                const auto result = cudaFuncGetParamInfo(params.func, index, &offset, &bytes);
                if (result == cudaErrorInvalidValue)
                {
                    // End of the compiler's parameter inventory is an expected
                    // API result. Consume only that expected runtime error;
                    // a pending execution failure is never hidden here.
                    const auto pending = cudaGetLastError();
                    if (pending != cudaSuccess && pending != cudaErrorInvalidValue)
                        require(pending, "probe parameter inventory completion");
                    return identity;
                }
                require(result, "inspect exact probe kernel parameter");
                if (!bytes || !params.kernelParams[index])
                    throw std::runtime_error("isolated probe encountered an incomplete kernel parameter");
                const auto *value = static_cast<const std::uint8_t *>(params.kernelParams[index]);
                identity.insert(identity.end(), value, value + bytes);
            }
            throw std::runtime_error("isolated probe kernel parameter inventory exceeds its explicit limit");
        }
    }

    /** Measure each physical kernel node of one warmed independent fixture.
     * @param parent Retained flat graph, owning the exact kernel arguments.
     * @param stream Exact non-null producer stream of the warmed fixture.
     * @param fixture Explicit repeated-write independence witness.
     * @param identity Source format and participant-local physical geometry.
     * @param config Untimed warmups and native-event sampling policy.
     * @return Diagnostic records, including all samples and compiler resources.
     * @throws std::runtime_error for conditional/collective graphs, missing
     *         arguments, nonzero local memory, or any failed CUDA operation.
     *
     * Memory nodes are inventoried but not timed as kernels. Every isolated
     * kernel consumes the intermediates published by the complete warmed
     * fixture; no quantizer, reducer, transport, or sibling operation runs
     * inside its timing graph. Independent timings cannot be summed into a
     * production critical path because their cache/overlap regimes differ.
     * Repeated nodes remain separate witnesses: equal arguments and launch
     * geometry do not establish equality of every native node attribute.
     */
    inline std::vector<nlohmann::json> probeIndependentCUDAKernels(
        cudaGraph_t parent, cudaStream_t stream,
        CUDAIndependentKernelFixture fixture,
        const CUDACapturedKernelProbeIdentity &identity,
        const CUDACapturedKernelProbeConfig &config = {})
    {
        using namespace cuda_kernel_probe_detail;
        if (!parent || !stream || identity.operation.empty() || identity.format.empty() ||
            identity.m <= 0 || identity.n <= 0 || identity.k <= 0 || identity.tp_degree <= 0 ||
            config.warmup_runs <= 0 || config.sample_count <= 0 || config.operations_per_sample <= 0)
            throw std::invalid_argument("isolated CUDA probe requires a complete fixture, exact stream, and positive geometry");
        switch (fixture)
        {
        case CUDAIndependentKernelFixture::StatelessProjection:
        case CUDAIndependentKernelFixture::ZeroRecurrentFixedPoint:
        case CUDAIndependentKernelFixture::ReadOnlyAttentionCache: break;
        default: throw std::invalid_argument("isolated CUDA probe has no independence witness");
        }
        struct Entry { cudaKernelNodeParams params; cudaGraphNode_t source; std::size_t argument_bytes; };
        std::vector<Entry> kernels;
        std::size_t count = 0, memory_nodes = 0;
        require(cudaGraphGetNodes(parent, nullptr, &count), "count retained probe nodes");
        std::vector<cudaGraphNode_t> nodes(count);
        require(cudaGraphGetNodes(parent, nodes.data(), &count), "inspect retained probe nodes");
        for (auto node : nodes)
        {
            cudaGraphNodeType type{};
            require(cudaGraphNodeGetType(node, &type), "inspect probe node type");
            if (type == cudaGraphNodeTypeEmpty) continue;
            if (type == cudaGraphNodeTypeMemset || type == cudaGraphNodeTypeMemcpy)
            {
                ++memory_nodes;
                continue;
            }
            if (type != cudaGraphNodeTypeKernel)
                throw std::runtime_error("isolated CUDA probe rejects opaque node type " + std::to_string(static_cast<int>(type)));
            cudaKernelNodeParams params{};
            require(cudaGraphKernelNodeGetParams(node, &params), "inspect exact probe launch parameters");
            const auto argument_bytes = argumentIdentity(params).size();
            kernels.push_back({params, node, argument_bytes});
        }
        if (kernels.empty()) throw std::runtime_error("isolated CUDA probe contains no physical kernel");
        std::vector<nlohmann::json> records;
        for (std::size_t node_index = 0; node_index < kernels.size(); ++node_index)
        {
            const auto &entry = kernels[node_index];
            const auto &params = entry.params;
            const char *symbol = nullptr;
            require(cudaFuncGetName(&symbol, params.func), "name exact probe kernel");
            const std::string name = symbol ? symbol : "";
            if (name.empty() || name.find("nccl") != std::string::npos || name.find("rccl") != std::string::npos)
                throw std::runtime_error("isolated CUDA probe requires a named independent compute kernel");
            cudaFuncAttributes attributes{};
            require(cudaFuncGetAttributes(&attributes, params.func), "inspect probe compiler resources");
            if (attributes.localSizeBytes != 0)
                throw std::runtime_error("isolated CUDA probe found local memory in " + name);
            const int threads = params.blockDim.x * params.blockDim.y * params.blockDim.z;
            int occupancy = 0;
            require(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, params.func, threads, params.sharedMemBytes),
                    "inspect probe occupancy ceiling");
            GraphOwner isolated;
            cudaGraphNode_t previous = nullptr;
            for (int repeat = 0; repeat < config.operations_per_sample; ++repeat)
            {
                cudaGraphNode_t current = nullptr;
                require(cudaGraphAddKernelNode(&current, isolated.graph, previous ? &previous : nullptr,
                    previous ? 1 : 0, &params), "retain isolated physical kernel");
                require(cudaGraphKernelNodeCopyAttributes(current, entry.source), "retain exact probe node attributes");
                previous = current;
            }
            require(cudaGraphInstantiateWithFlags(&isolated.executable, isolated.graph, cudaGraphInstantiateFlagUseNodePriority),
                    "instantiate isolated probe");
            for (int warmup = 0; warmup < config.warmup_runs; ++warmup)
                require(cudaGraphLaunch(isolated.executable, stream), "warm isolated probe");
            std::vector<TimingEvent> starts(config.sample_count), stops(config.sample_count);
            for (int sample = 0; sample < config.sample_count; ++sample)
            {
                require(cudaEventRecord(starts[sample].handle, stream), "start isolated probe sample");
                require(cudaGraphLaunch(isolated.executable, stream), "replay isolated probe sample");
                require(cudaEventRecord(stops[sample].handle, stream), "stop isolated probe sample");
            }
            require(cudaEventSynchronize(stops.back().handle), "observe terminal isolated probe event");
            std::vector<double> samples;
            for (int sample = 0; sample < config.sample_count; ++sample)
            {
                float ms = 0;
                require(cudaEventElapsedTime(&ms, starts[sample].handle, stops[sample].handle), "read isolated probe sample");
                samples.push_back(ms * 1000.0 / config.operations_per_sample);
            }
            records.push_back({{"diagnostic_only", true}, {"mode", "isolated_native_graph"},
                {"operation", identity.operation}, {"format", identity.format}, {"tp_degree", identity.tp_degree},
                {"m", identity.m}, {"n", identity.n}, {"k", identity.k}, {"kernel", name},
                {"grid", {params.gridDim.x, params.gridDim.y, params.gridDim.z}},
                {"block", {params.blockDim.x, params.blockDim.y, params.blockDim.z}},
                {"registers", attributes.numRegs}, {"local_bytes", attributes.localSizeBytes},
                {"shared_bytes", attributes.sharedSizeBytes + params.sharedMemBytes},
                {"active_blocks_per_sm_ceiling", occupancy}, {"physical_node_index", node_index},
                {"argument_bytes", entry.argument_bytes},
                {"untimed_memory_nodes", memory_nodes}, {"operations_per_sample", config.operations_per_sample},
                {"samples_us", samples}});
        }
        return records;
    }

    /** Append complete diagnostic records after timing has finished.
     * @param path Explicit diagnostic destination, never a training corpus.
     * @param records Exact physical-kernel observations to retain.
     * @throws std::runtime_error if the evidence cannot be written completely.
     */
    inline void appendCUDAKernelProbeRecords(
        const std::string &path, const std::vector<nlohmann::json> &records)
    {
        std::ofstream output(path, std::ios::app);
        if (!output) throw std::runtime_error("cannot open isolated CUDA probe evidence " + path);
        for (const auto &record : records) output << record.dump() << '\n';
        output.flush();
        if (!output) throw std::runtime_error("cannot finish isolated CUDA probe evidence " + path);
    }
}
