/**
 * @file native_graph_event_trace.cpp
 * @brief Diagnostic-only CUDA graph event timeline, without CUPTI attachment.
 *
 * Split each existing completion edge around event records; do not replace any
 * computation, collective, data or launch policy. Conditional bodies remain
 * untouched and are timed only as opaque parent nodes. Events are queried at
 * the existing executable-retirement point, never synchronized per node. This
 * adds overhead and is NOT benchmark timing. Brackets include scheduling and
 * event overhead; they are not hardware-counter kernel service times.
 *
 * Build as a standalone LD_PRELOAD diagnostic, never as an engine dependency.
 * See references/native-graph-events.md for commands and limitations.
 */
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <unistd.h>

namespace {
using Json = nlohmann::json;
/**
 * @brief Stop on incomplete evidence instead of guessing at missing nodes.
 * @param status Exact CUDA driver result from the preceding observation.
 * @param operation Diagnostic name of the operation that must have succeeded.
 */
void require(CUresult status, const char *operation) {
    if (status == CUDA_SUCCESS) return;
    const char *name = nullptr;
    cuGetErrorName(status, &name);
    std::fprintf(stderr, "native event trace: %s: %s\n", operation, name ? name : "unknown");
    std::abort();
}
/** @brief Keep timing handles alive through every replay of one executable. */
struct Trace {
    unsigned id{};
    int device{};
    CUcontext context{};
    CUevent begin{}, end{};
    std::vector<CUevent> before, after;
    Json nodes;
    std::string graph_selector;
    std::vector<std::string> node_filters;
    unsigned launches{};
    bool selected_nodes_only{};
};
std::mutex traces_mutex;
std::unordered_map<cudaGraphExec_t, std::unique_ptr<Trace>> traces;
std::atomic<unsigned> next_id{0};

/**
 * @brief Add one owned timing event node with only the stated producer edges.
 * @param graph Selected parent, never a conditional body.
 * @param event Receives the event retained until executable retirement.
 * @param parents Exact producer completions required before the event.
 * @return The added graph node; failures abort the diagnostic process.
 */
CUgraphNode record(CUgraph graph, CUevent &event, const std::vector<CUgraphNode> &parents) {
    require(cuEventCreate(&event, CU_EVENT_DEFAULT), "create timing event");
    CUgraphNode node{};
    require(cuGraphAddEventRecordNode(&node, graph, parents.data(), parents.size(), event), "add event record");
    return node;
}

/**
 * @brief Instrument a selected prefill parent without entering conditional bodies.
 * @param graph Uninstantiated native parent; original work/edges are retained.
 * @return Owned event lifetime, or null for a disabled/unsupported/unmatched graph.
 */
std::unique_ptr<Trace> instrument(CUgraph graph) {
    if (!std::getenv("LLAMINAR_NATIVE_EVENT_TRACE_DIR")) return {};
    std::size_t size = 0;
    require(cuGraphGetNodes(graph, nullptr, &size), "count nodes");
    if (size >= 1000)
        std::fprintf(stderr, "native event trace inspecting pid=%d nodes=%zu\n", getpid(), size);
    const std::size_t minimum = std::getenv("LLAMINAR_NATIVE_EVENT_TRACE_MIN_NODES")
        ? std::stoul(std::getenv("LLAMINAR_NATIVE_EVENT_TRACE_MIN_NODES")) : 1000;
    if (size < minimum) return {};
    std::vector<CUgraphNode> nodes(size);
    require(cuGraphGetNodes(graph, nodes.data(), &size), "get nodes");
    std::unordered_map<CUgraphNode, std::size_t> index;
    auto trace = std::make_unique<Trace>();
    trace->nodes = Json::array();
    const char *graph_selector = std::getenv("LLAMINAR_NATIVE_EVENT_TRACE_GRAPH_CONTAINS");
    trace->graph_selector = graph_selector ? graph_selector : "gdn_chunk_forward_kernel";
    if (trace->graph_selector.empty()) std::abort();
    bool selected = std::getenv("LLAMINAR_NATIVE_EVENT_TRACE_ALL_FLAT") != nullptr;
    for (std::size_t i = 0; i < size; ++i) {
        index[nodes[i]] = i;
        CUgraphNodeType type{};
        require(cuGraphNodeGetType(nodes[i], &type), "node type");
        // Events belong only to this parent. In particular, do not insert event
        // nodes inside conditional bodies, which have a stricter CUDA contract.
        // Report their complete interval as opaque rather than inventing an
        // attribution to the kernels or communication hidden inside the body.
        if (type != CU_GRAPH_NODE_TYPE_KERNEL && type != CU_GRAPH_NODE_TYPE_EMPTY &&
            type != CU_GRAPH_NODE_TYPE_MEMSET && type != CU_GRAPH_NODE_TYPE_MEMCPY &&
            type != CU_GRAPH_NODE_TYPE_CONDITIONAL) {
            std::fprintf(stderr, "native event trace skipped pid=%d nodes=%zu node=%zu unsupported_type=%d\n",
                         getpid(), size, i, int(type));
            return {};
        }
        Json item{{"id", i}, {"type", int(type)}, {"parents", Json::array()}, {"name", "empty"}};
        if (type == CU_GRAPH_NODE_TYPE_MEMSET) item["name"] = "memset";
        if (type == CU_GRAPH_NODE_TYPE_MEMCPY) item["name"] = "memcpy";
        if (type == CU_GRAPH_NODE_TYPE_CONDITIONAL) item["name"] = "conditional_body_opaque";
        if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
            CUDA_KERNEL_NODE_PARAMS params{};
            require(cuGraphKernelNodeGetParams(nodes[i], &params), "kernel parameters");
            const char *name = nullptr;
            if (params.func) require(cuFuncGetName(&name, params.func), "function name");
            else if (params.kern) require(cuKernelGetName(&name, params.kern), "kernel name");
            else std::abort();
            item["name"] = name;
            item["grid"] = {params.gridDimX, params.gridDimY, params.gridDimZ};
            item["block"] = {params.blockDimX, params.blockDimY, params.blockDimZ};
            selected |= std::string(name).find(trace->graph_selector) != std::string::npos;
        }
        trace->nodes.push_back(std::move(item));
    }
    if (!selected) {
        std::fprintf(stderr, "native event trace skipped pid=%d nodes=%zu reason=no_graph_selector_match\n", getpid(), size);
        return {};
    }
    trace->id = next_id++;
    require(cuCtxGetCurrent(&trace->context), "trace context");
    require(cuCtxGetDevice(&trace->device), "trace device");
    std::vector<std::vector<CUgraphNode>> parents(size), children(size);
    for (std::size_t i = 0; i < size; ++i) {
        std::size_t count = 0;
        require(cuGraphNodeGetDependencies(nodes[i], nullptr, nullptr, &count), "dependency count");
        parents[i].resize(count);
        std::vector<CUgraphEdgeData> properties(count);
        require(cuGraphNodeGetDependencies(nodes[i], parents[i].data(), properties.data(), &count), "dependencies");
        for (std::size_t p = 0; p < count; ++p) {
            if (properties[p].from_port || properties[p].to_port || properties[p].type) std::abort();
            trace->nodes[i]["parents"].push_back(index.at(parents[i][p]));
            children[index.at(parents[i][p])].push_back(nodes[i]);
        }
    }
    std::vector<std::string> filters;
    if (const char *filter = std::getenv("LLAMINAR_NATIVE_EVENT_TRACE_KERNEL_CONTAINS")) {
        std::istringstream values(filter);
        std::string value;
        while (std::getline(values, value, ',')) {
            if (value.empty()) std::abort();
            filters.push_back(value);
        }
        if (filters.empty()) std::abort();
        trace->selected_nodes_only = true;
    }
    trace->node_filters = filters;
    std::vector<bool> measured(size, filters.empty());
    std::size_t measured_count = 0;
    for (std::size_t i = 0; i < size; ++i) {
        const std::string name = trace->nodes[i]["name"];
        for (const auto &filter : filters)
            if (name.find(filter) != std::string::npos) measured[i] = true;
        trace->nodes[i]["instrumented"] = bool(measured[i]);
        measured_count += measured[i] ? 1 : 0;
    }
    // A selector miss must be visible and must not modify the graph at all.
    if (!measured_count) {
        std::fprintf(stderr, "native event trace skipped pid=%d nodes=%zu reason=no_node_selector_match\n", getpid(), size);
        return {};
    }
    // Bracketing preserves the original producer/consumer partial order. CUDA
    // may reschedule branches after adding these nodes; no queue/overlap
    // equivalence is implied. The terminal supplies one device-clock interval.
    const auto begin = record(graph, trace->begin, {});
    for (std::size_t i = 0; i < size; ++i)
        if (parents[i].empty())
            require(cuGraphAddDependencies(graph, &begin, &nodes[i], nullptr, 1), "origin dependency");
    trace->before.resize(size);
    trace->after.resize(size);
    std::vector<CUgraphNode> starts(size), stops(size);
    for (std::size_t i = 0; i < size; ++i) {
        if (!measured[i]) continue;
        auto input = parents[i];
        if (input.empty()) input.push_back(begin);
        starts[i] = record(graph, trace->before[i], input);
        require(cuGraphAddDependencies(graph, &starts[i], &nodes[i], nullptr, 1), "start dependency");
        stops[i] = record(graph, trace->after[i], {nodes[i]});
    }
    for (std::size_t i = 0; i < size; ++i) {
        if (!stops[i]) continue;
        for (auto child : children[i]) {
            const auto consumer = starts[index.at(child)] ? starts[index.at(child)] : child;
            require(cuGraphAddDependencies(graph, &stops[i], &consumer, nullptr, 1), "stop dependency");
        }
    }
    std::vector<CUgraphNode> terminal;
    for (std::size_t i = 0; i < size; ++i)
        if (children[i].empty()) terminal.push_back(stops[i] ? stops[i] : nodes[i]);
    record(graph, trace->end, terminal);
    std::fprintf(stderr, "native event trace instrumented device=%d graph=%u nodes=%zu measured=%zu\n", trace->device, trace->id, size, measured_count);
    return trace;
}

/**
 * @brief Observe the final completed replay at normal retirement; do not wait.
 * @param trace Event owner whose native executable has just been destroyed.
 * @details Queries require ordinary terminal completion. A pending event is an
 * error, not permission for the observer to add synchronization to inference.
 */
void retire(Trace &trace) {
    require(cuCtxPushCurrent(trace.context), "select trace context");
    if (trace.launches) {
        require(cuEventQuery(trace.end), "normal retirement must follow terminal event");
        float duration = 0;
        require(cuEventElapsedTime(&duration, trace.begin, trace.end), "graph time");
        for (std::size_t i = 0; i < trace.before.size(); ++i) {
            if (!trace.before[i]) continue;
            float begin = 0, end = 0;
            require(cuEventElapsedTime(&begin, trace.begin, trace.before[i]), "node begin");
            require(cuEventElapsedTime(&end, trace.begin, trace.after[i]), "node end");
            trace.nodes[i]["start_ms"] = begin;
            trace.nodes[i]["end_ms"] = end;
        }
        Json report{{"schema_version", 1}, {"diagnostic_only", true}, {"source", "native_cuda_graph_events"},
            {"intervals_include_scheduling_and_event_overhead", true},
            {"conditional_bodies_timed_as_opaque", true},
            {"selected_nodes_only", trace.selected_nodes_only},
            {"graph_selector", trace.graph_selector}, {"node_filters", trace.node_filters},
            {"device", trace.device}, {"graph", trace.id}, {"launches", trace.launches},
            {"snapshot", "last_completed_replay"}, {"duration_ms", duration}, {"nodes", trace.nodes}};
        const std::string path = std::string(std::getenv("LLAMINAR_NATIVE_EVENT_TRACE_DIR")) +
            "/cuda-event-" + std::to_string(getpid()) + "-" + std::to_string(trace.id) + ".json";
        std::ofstream output(path);
        output << report.dump(2) << '\n';
        if (!output) std::abort();
        std::fprintf(stderr, "native event trace saved %s\n", path.c_str());
    }
    for (auto event : trace.before) if (event) require(cuEventDestroy(event), "retire start");
    for (auto event : trace.after) if (event) require(cuEventDestroy(event), "retire stop");
    require(cuEventDestroy(trace.begin), "retire origin");
    require(cuEventDestroy(trace.end), "retire terminal");
    CUcontext restored{};
    require(cuCtxPopCurrent(&restored), "restore trace context");
}
}

/**
 * @brief Retain production instantiation flags and track only selected graphs.
 * @param exec Receives the actual runtime executable, unchanged by the wrapper.
 * @param graph Parent to inspect immediately before ordinary instantiation.
 * @param flags Original runtime flags, including node-priority policy.
 * @return The native runtime result; selected instrumentation fails fatally.
 */
extern "C" cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *exec, cudaGraph_t graph, unsigned long long flags) {
    using Function = cudaError_t (*)(cudaGraphExec_t *, cudaGraph_t, unsigned long long);
    static auto native = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "cudaGraphInstantiateWithFlags"));
    if (!native) std::abort();
    auto trace = instrument(reinterpret_cast<CUgraph>(graph));
    auto status = native(exec, graph, flags);
    if (trace && status != cudaSuccess) std::abort();
    if (trace) { std::lock_guard guard(traces_mutex); traces.emplace(*exec, std::move(trace)); }
    return status;
}

/**
 * @brief Count real submissions without adding ordering to the launch stream.
 * @param exec The production executable, potentially carrying observer events.
 * @param stream The caller's exact stream, forwarded without substitution.
 * @return The native launch result, never retried by the observer.
 */
extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
    using Function = cudaError_t (*)(cudaGraphExec_t, cudaStream_t);
    static auto native = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "cudaGraphLaunch"));
    if (!native) std::abort();
    auto status = native(exec, stream);
    if (status == cudaSuccess) {
        std::lock_guard guard(traces_mutex);
        if (auto found = traces.find(exec); found != traces.end()) ++found->second->launches;
    }
    return status;
}

/**
 * @brief Export after the caller has completed its ordinary graph lifetime.
 * @param exec Executable retired by its production owner after terminal completion.
 * @return The native retirement result; diagnostic event retirement follows it.
 */
extern "C" cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) {
    using Function = cudaError_t (*)(cudaGraphExec_t);
    static auto native = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "cudaGraphExecDestroy"));
    if (!native) std::abort();
    std::unique_ptr<Trace> trace;
    { std::lock_guard guard(traces_mutex);
      if (auto found = traces.find(exec); found != traces.end()) {
          trace = std::move(found->second); traces.erase(found);
      }
    }
    // The runtime already completed its terminal wait. Retire the executable
    // before its diagnostic event handles; no extra synchronization is added.
    const auto status = native(exec);
    if (trace && status != cudaSuccess) std::abort();
    if (trace) retire(*trace);
    return status;
}
