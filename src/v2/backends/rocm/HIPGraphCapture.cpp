/**
 * @file HIPGraphCapture.cpp
 * @brief HIP graph capture, replay, composition, and resident-memory accounting.
 *
 * Retained HIP executables keep opaque driver-owned state outside BufferArena.
 * Instantiation therefore records its setup-only free-VRAM delta alongside the
 * exact native node count. Capacity planning consumes the same graph-family
 * model as CUDA, while these observations prove that the backend-specific
 * driver remains inside the admitted envelope. No memory query occurs during
 * graph replay or inference.
 */

#ifdef HAVE_ROCM

#include "HIPGraphCapture.h"
#include "HIPGraphTimelineKernels.h"
#include "../../utils/Logger.h"
#include "../../utils/VramBillOfMaterials.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Result of importing one captured HIP graph into another graph.
         *
         * A graph fragment may end in more than one independent leaf. Keeping
         * the complete leaf frontier lets the next timeline operation depend
         * on every outstanding branch instead of inventing a serial tail.
         */
        struct ImportedHIPGraph
        {
            std::vector<hipGraphNode_t> leaves;
            std::size_t node_count = 0u;
        };

        /**
         * @brief Append a node once while retaining deterministic order.
         * @param nodes Destination dependency/frontier list.
         * @param node Non-null HIP graph node to append if it is not present.
         */
        void appendUniqueNode(
            std::vector<hipGraphNode_t> &nodes,
            hipGraphNode_t node)
        {
            if (node != nullptr &&
                std::find(nodes.begin(), nodes.end(), node) == nodes.end())
            {
                nodes.push_back(node);
            }
        }

        /**
         * @brief Clone one non-child HIP node into a destination graph.
         *
         * The accepted node families are exactly the stable device-resident
         * families permitted in retained inference fragments. Host callbacks,
         * graph allocation/free, and HIP's beta batch-memory nodes are rejected:
         * importing them would hide a hot-path ownership, allocation, or
         * dependency-ordering violation.
         *
         * @param source_node Immutable node whose launch parameters are read.
         * @param destination_graph Graph receiving the new native node.
         * @param dependencies Destination-graph nodes that must complete first.
         * @param destination_node Receives the newly-created node.
         * @param error Receives a precise diagnostic on failure.
         * @param node_path Human-readable source location for diagnostics.
         * @return true when the node was cloned successfully.
         */
        bool cloneSupportedHipNode(
            hipGraphNode_t source_node,
            hipGraph_t destination_graph,
            const std::vector<hipGraphNode_t> &dependencies,
            hipGraphNode_t &destination_node,
            std::string &error,
            const std::string &node_path)
        {
            hipGraphNodeType type = hipGraphNodeTypeCount;
            hipError_t status = hipGraphNodeGetType(source_node, &type);
            if (status != hipSuccess)
            {
                error = "hipGraphNodeGetType failed at " + node_path +
                        ": " + hipGetErrorString(status);
                return false;
            }

            const hipGraphNode_t *dependency_data =
                dependencies.empty() ? nullptr : dependencies.data();
            const std::size_t dependency_count = dependencies.size();
            switch (type)
            {
            case hipGraphNodeTypeKernel:
            {
                hipKernelNodeParams params{};
                status = hipGraphKernelNodeGetParams(source_node, &params);
                if (status == hipSuccess)
                {
                    status = hipGraphAddKernelNode(
                        &destination_node,
                        destination_graph,
                        dependency_data,
                        dependency_count,
                        &params);
                }
                break;
            }
            case hipGraphNodeTypeMemcpy:
            {
                hipMemcpy3DParms params{};
                status = hipGraphMemcpyNodeGetParams(source_node, &params);
                if (status == hipSuccess)
                {
                    const bool is_linear_copy =
                        params.srcArray == nullptr &&
                        params.dstArray == nullptr &&
                        params.srcPtr.ptr != nullptr &&
                        params.dstPtr.ptr != nullptr &&
                        params.srcPos.x == 0u &&
                        params.srcPos.y == 0u &&
                        params.srcPos.z == 0u &&
                        params.dstPos.x == 0u &&
                        params.dstPos.y == 0u &&
                        params.dstPos.z == 0u &&
                        params.extent.width > 0u &&
                        params.extent.height == 1u &&
                        params.extent.depth == 1u;
                    if (is_linear_copy)
                    {
                        /* Stream capture represents hipMemcpyAsync as HIP's
                         * specialized 1D node. Re-importing that node through
                         * the generic 3D API loses the runtime's effective
                         * peer/mapped-memory classification. On gfx906 the
                         * resulting node advertises AQL packet capture but can
                         * produce no packet, causing ROCm 7.1 to instantiate a
                         * successful executable that replays no work. Preserve
                         * the linear operation as a 1D node so HIP selects the
                         * exact copy engine and packet-capture policy from the
                         * source and destination allocations. */
                        status = hipGraphAddMemcpyNode1D(
                            &destination_node,
                            destination_graph,
                            dependency_data,
                            dependency_count,
                            params.dstPtr.ptr,
                            params.srcPtr.ptr,
                            params.extent.width,
                            params.kind);
                    }
                    else
                    {
                        status = hipGraphAddMemcpyNode(
                            &destination_node,
                            destination_graph,
                            dependency_data,
                            dependency_count,
                            &params);
                    }
                }
                break;
            }
            case hipGraphNodeTypeMemset:
            {
                hipMemsetParams params{};
                status = hipGraphMemsetNodeGetParams(source_node, &params);
                if (status == hipSuccess)
                {
                    status = hipGraphAddMemsetNode(
                        &destination_node,
                        destination_graph,
                        dependency_data,
                        dependency_count,
                        &params);
                }
                break;
            }
            case hipGraphNodeTypeEmpty:
                status = hipGraphAddEmptyNode(
                    &destination_node,
                    destination_graph,
                    dependency_data,
                    dependency_count);
                break;
            case hipGraphNodeTypeWaitEvent:
            {
                hipEvent_t event = nullptr;
                status = hipGraphEventWaitNodeGetEvent(source_node, &event);
                if (status == hipSuccess && event != nullptr)
                {
                    status = hipGraphAddEventWaitNode(
                        &destination_node,
                        destination_graph,
                        dependency_data,
                        dependency_count,
                        event);
                }
                else if (status == hipSuccess)
                {
                    status = hipErrorInvalidValue;
                }
                break;
            }
            case hipGraphNodeTypeEventRecord:
            {
                hipEvent_t event = nullptr;
                status = hipGraphEventRecordNodeGetEvent(source_node, &event);
                if (status == hipSuccess && event != nullptr)
                {
                    status = hipGraphAddEventRecordNode(
                        &destination_node,
                        destination_graph,
                        dependency_data,
                        dependency_count,
                        event);
                }
                else if (status == hipSuccess)
                {
                    status = hipErrorInvalidValue;
                }
                break;
            }
            case hipGraphNodeTypeBatchMemOp:
                error = "HIP retained fragment contains a forbidden beta "
                        "batch-memory node at " +
                        node_path;
                return false;
            case hipGraphNodeTypeGraph:
                error = "internal error: nested child reached scalar HIP node "
                        "cloner at " +
                        node_path;
                return false;
            case hipGraphNodeTypeHost:
                error = "HIP retained fragment contains a forbidden host "
                        "callback at " +
                        node_path;
                return false;
            case hipGraphNodeTypeMemAlloc:
            case hipGraphNodeTypeMemFree:
                error = "HIP retained fragment contains forbidden hot-path "
                        "allocation/free at " +
                        node_path;
                return false;
            default:
                error = "HIP retained fragment contains unsupported node type " +
                        std::to_string(static_cast<int>(type)) + " at " +
                        node_path;
                return false;
            }

            if (status != hipSuccess || destination_node == nullptr)
            {
                error = "failed to clone HIP node type " +
                        std::to_string(static_cast<int>(type)) + " at " +
                        node_path + ": " + hipGetErrorString(status);
                return false;
            }
            return true;
        }

        /**
         * @brief Flatten a retained HIP fragment into a parent dependency DAG.
         *
         * HIP child-graph dependencies can gate completion of the child wrapper
         * without preventing the runtime from issuing the child's internal
         * operations early. That is invalid for a transaction whose preceding
         * node acquires bytes published by another device. This importer
         * reconstructs the source DAG in the parent graph so each source root
         * directly depends on the incoming timeline frontier.
         *
         * Nested child graphs are recursively flattened. Original parallelism
         * is retained: incoming edges are reproduced, and all terminal leaves
         * are returned as the frontier for the next transaction step.
         *
         * @param source_graph Immutable retained graph to import.
         * @param destination_graph Parent graph receiving native nodes.
         * @param predecessors Nodes that every source root must follow.
         * @param result Receives imported node count and terminal frontier.
         * @param error Receives a precise diagnostic on failure.
         * @param graph_path Human-readable source location for diagnostics.
         * @return true when the complete source DAG was imported.
         */
        bool importHipGraphDAG(
            hipGraph_t source_graph,
            hipGraph_t destination_graph,
            const std::vector<hipGraphNode_t> &predecessors,
            ImportedHIPGraph &result,
            std::string &error,
            const std::string &graph_path)
        {
            result = {};
            size_t node_count = 0u;
            hipError_t status =
                hipGraphGetNodes(source_graph, nullptr, &node_count);
            if (status != hipSuccess)
            {
                error = "hipGraphGetNodes(count) failed at " + graph_path +
                        ": " + hipGetErrorString(status);
                return false;
            }
            if (node_count == 0u)
            {
                error = "cannot import an empty HIP retained fragment at " +
                        graph_path;
                return false;
            }

            std::vector<hipGraphNode_t> source_nodes(node_count);
            status = hipGraphGetNodes(
                source_graph, source_nodes.data(), &node_count);
            if (status != hipSuccess)
            {
                error = "hipGraphGetNodes(nodes) failed at " + graph_path +
                        ": " + hipGetErrorString(status);
                return false;
            }
            source_nodes.resize(node_count);

            std::unordered_map<hipGraphNode_t, std::size_t> source_indices;
            source_indices.reserve(source_nodes.size());
            for (std::size_t index = 0u; index < source_nodes.size(); ++index)
            {
                if (!source_indices.emplace(source_nodes[index], index).second)
                {
                    error = "HIP graph returned a duplicate node at " +
                            graph_path;
                    return false;
                }
            }

            size_t edge_count = 0u;
            status = hipGraphGetEdges(
                source_graph, nullptr, nullptr, &edge_count);
            if (status != hipSuccess)
            {
                error = "hipGraphGetEdges(count) failed at " + graph_path +
                        ": " + hipGetErrorString(status);
                return false;
            }
            std::vector<hipGraphNode_t> edge_sources(edge_count);
            std::vector<hipGraphNode_t> edge_targets(edge_count);
            if (edge_count > 0u)
            {
                status = hipGraphGetEdges(
                    source_graph,
                    edge_sources.data(),
                    edge_targets.data(),
                    &edge_count);
                if (status != hipSuccess)
                {
                    error = "hipGraphGetEdges(edges) failed at " + graph_path +
                            ": " + hipGetErrorString(status);
                    return false;
                }
                edge_sources.resize(edge_count);
                edge_targets.resize(edge_count);
            }

            std::vector<std::vector<std::size_t>> parents(source_nodes.size());
            std::vector<std::vector<std::size_t>> children(source_nodes.size());
            for (std::size_t edge = 0u; edge < edge_count; ++edge)
            {
                const auto source = source_indices.find(edge_sources[edge]);
                const auto target = source_indices.find(edge_targets[edge]);
                if (source == source_indices.end() ||
                    target == source_indices.end() ||
                    source->second == target->second)
                {
                    error = "HIP graph returned an invalid edge at " +
                            graph_path;
                    return false;
                }
                parents[target->second].push_back(source->second);
                children[source->second].push_back(target->second);
            }

            // Kahn ordering makes parent destination nodes available before a
            // child is reconstructed. The original edge set, rather than this
            // traversal order, determines which branches may run concurrently.
            std::vector<std::size_t> remaining_parents(source_nodes.size());
            std::vector<std::size_t> ready;
            ready.reserve(source_nodes.size());
            for (std::size_t index = 0u; index < source_nodes.size(); ++index)
            {
                remaining_parents[index] = parents[index].size();
                if (remaining_parents[index] == 0u)
                    ready.push_back(index);
            }

            std::vector<std::vector<hipGraphNode_t>> completion_frontiers(
                source_nodes.size());
            std::size_t ready_cursor = 0u;
            std::size_t visited = 0u;
            while (ready_cursor < ready.size())
            {
                const std::size_t index = ready[ready_cursor++];
                ++visited;

                std::vector<hipGraphNode_t> dependencies;
                if (parents[index].empty())
                {
                    dependencies = predecessors;
                }
                else
                {
                    for (const std::size_t parent : parents[index])
                    {
                        for (const hipGraphNode_t dependency :
                             completion_frontiers[parent])
                        {
                            appendUniqueNode(dependencies, dependency);
                        }
                    }
                }

                hipGraphNodeType type = hipGraphNodeTypeCount;
                status = hipGraphNodeGetType(source_nodes[index], &type);
                if (status != hipSuccess)
                {
                    error = "hipGraphNodeGetType failed at " + graph_path +
                            "/node[" + std::to_string(index) + "]: " +
                            hipGetErrorString(status);
                    return false;
                }

                const std::string node_path =
                    graph_path + "/node[" + std::to_string(index) + "]";
                if (type == hipGraphNodeTypeGraph)
                {
                    hipGraph_t child_graph = nullptr;
                    status = hipGraphChildGraphNodeGetGraph(
                        source_nodes[index], &child_graph);
                    if (status != hipSuccess || child_graph == nullptr)
                    {
                        error = "hipGraphChildGraphNodeGetGraph failed at " +
                                node_path + ": " + hipGetErrorString(status);
                        return false;
                    }
                    ImportedHIPGraph child_result;
                    if (!importHipGraphDAG(
                            child_graph,
                            destination_graph,
                            dependencies,
                            child_result,
                            error,
                            node_path + "/child"))
                    {
                        return false;
                    }
                    completion_frontiers[index] =
                        std::move(child_result.leaves);
                    result.node_count += child_result.node_count;
                }
                else
                {
                    hipGraphNode_t destination_node = nullptr;
                    if (!cloneSupportedHipNode(
                            source_nodes[index],
                            destination_graph,
                            dependencies,
                            destination_node,
                            error,
                            node_path))
                    {
                        return false;
                    }
                    completion_frontiers[index].push_back(destination_node);
                    ++result.node_count;
                }

                if (completion_frontiers[index].empty())
                {
                    error = "HIP graph node produced no completion frontier at " +
                            node_path;
                    return false;
                }
                for (const std::size_t child : children[index])
                {
                    if (remaining_parents[child] == 0u)
                    {
                        error = "HIP graph contains a duplicate dependency at " +
                                graph_path;
                        return false;
                    }
                    --remaining_parents[child];
                    if (remaining_parents[child] == 0u)
                        ready.push_back(child);
                }
            }

            if (visited != source_nodes.size())
            {
                error = "HIP retained fragment is cyclic at " + graph_path;
                return false;
            }
            for (std::size_t index = 0u; index < source_nodes.size(); ++index)
            {
                if (!children[index].empty())
                    continue;
                for (const hipGraphNode_t leaf : completion_frontiers[index])
                    appendUniqueNode(result.leaves, leaf);
            }
            if (result.leaves.empty() || result.node_count == 0u)
            {
                error = "HIP retained fragment produced no native leaf nodes at " +
                        graph_path;
                return false;
            }
            return true;
        }

        /**
         * @brief Recursively append every HIP kernel node in one captured graph.
         *
         * Child graphs are traversed because production stage captures commonly
         * compose nested graph fragments. The routine reads immutable graph
         * metadata only and performs no launch, synchronization, allocation on
         * the device, or device/host transfer.
         */
        bool inspectHipKernelNodesRecursive(
            hipGraph_t graph,
            hipStream_t stream,
            const std::string &graph_path,
            size_t nesting_depth,
            std::vector<GPUGraphKernelNodeInfo> &kernel_nodes,
            std::string &error)
        {
            size_t node_count = 0;
            hipError_t status = hipGraphGetNodes(graph, nullptr, &node_count);
            if (status != hipSuccess)
            {
                error = "hipGraphGetNodes(count) failed at " + graph_path +
                        ": " + hipGetErrorString(status);
                return false;
            }

            std::vector<hipGraphNode_t> nodes(node_count);
            if (node_count > 0)
            {
                status = hipGraphGetNodes(graph, nodes.data(), &node_count);
                if (status != hipSuccess)
                {
                    error = "hipGraphGetNodes(nodes) failed at " + graph_path +
                            ": " + hipGetErrorString(status);
                    return false;
                }
                nodes.resize(node_count);
            }

            for (size_t node_index = 0; node_index < nodes.size(); ++node_index)
            {
                const std::string node_path =
                    graph_path + "/node[" + std::to_string(node_index) + "]";
                hipGraphNodeType type = hipGraphNodeTypeCount;
                status = hipGraphNodeGetType(nodes[node_index], &type);
                if (status != hipSuccess)
                {
                    error = "hipGraphNodeGetType failed at " + node_path +
                            ": " + hipGetErrorString(status);
                    return false;
                }

                if (type == hipGraphNodeTypeKernel)
                {
                    hipKernelNodeParams params{};
                    status = hipGraphKernelNodeGetParams(
                        nodes[node_index], &params);
                    if (status != hipSuccess || params.func == nullptr)
                    {
                        error = "hipGraphKernelNodeGetParams failed at " +
                                node_path + ": " + hipGetErrorString(status);
                        return false;
                    }

                    const char *runtime_name =
                        hipKernelNameRefByPtr(params.func, stream);
                    const bool name_resolved =
                        runtime_name != nullptr && runtime_name[0] != '\0';
                    hipFuncAttributes attributes{};
                    status = hipFuncGetAttributes(&attributes, params.func);
                    if (status != hipSuccess)
                    {
                        error = "hipFuncGetAttributes failed at " + node_path +
                                ": " + hipGetErrorString(status);
                        return false;
                    }

                    const uint64_t captured_block_threads =
                        static_cast<uint64_t>(params.blockDim.x) *
                        static_cast<uint64_t>(params.blockDim.y) *
                        static_cast<uint64_t>(params.blockDim.z);
                    if (captured_block_threads == 0 ||
                        captured_block_threads >
                            static_cast<uint64_t>(
                                std::numeric_limits<int>::max()))
                    {
                        error = "HIP kernel node has invalid flattened block "
                                "size at " +
                                node_path;
                        return false;
                    }
                    int max_active_blocks_per_sm = 0;
                    status = hipOccupancyMaxActiveBlocksPerMultiprocessor(
                        &max_active_blocks_per_sm,
                        params.func,
                        static_cast<int>(captured_block_threads),
                        params.sharedMemBytes);
                    if (status != hipSuccess ||
                        max_active_blocks_per_sm <= 0)
                    {
                        error = "hipOccupancyMaxActiveBlocksPerMultiprocessor "
                                "failed at " +
                                node_path + ": " + hipGetErrorString(status);
                        return false;
                    }
                    kernel_nodes.push_back(GPUGraphKernelNodeInfo{
                        .name = name_resolved ? runtime_name : "unresolved_kernel",
                        .graph_path = node_path,
                        .function_identity = reinterpret_cast<uintptr_t>(params.func),
                        .grid_x = params.gridDim.x,
                        .grid_y = params.gridDim.y,
                        .grid_z = params.gridDim.z,
                        .block_x = params.blockDim.x,
                        .block_y = params.blockDim.y,
                        .block_z = params.blockDim.z,
                        .dynamic_shared_memory_bytes = params.sharedMemBytes,
                        .static_shared_memory_bytes = attributes.sharedSizeBytes,
                        .local_memory_bytes_per_thread = attributes.localSizeBytes,
                        .registers_per_thread = static_cast<uint32_t>(
                            std::max(0, attributes.numRegs)),
                        .max_threads_per_block = static_cast<uint32_t>(
                            std::max(0, attributes.maxThreadsPerBlock)),
                        .max_active_blocks_per_sm = static_cast<uint32_t>(
                            max_active_blocks_per_sm),
                        .nesting_depth = nesting_depth,
                        .name_resolved = name_resolved,
                    });
                    continue;
                }

                if (type == hipGraphNodeTypeGraph)
                {
                    hipGraph_t child_graph = nullptr;
                    status = hipGraphChildGraphNodeGetGraph(
                        nodes[node_index], &child_graph);
                    if (status != hipSuccess || child_graph == nullptr)
                    {
                        error = "hipGraphChildGraphNodeGetGraph failed at " +
                                node_path + ": " +
                                hipGetErrorString(status);
                        return false;
                    }
                    if (!inspectHipKernelNodesRecursive(
                            child_graph,
                            stream,
                            node_path + "/child",
                            nesting_depth + 1,
                            kernel_nodes,
                            error))
                    {
                        return false;
                    }
                }
            }
            return true;
        }
    } // namespace


    // Best-effort error logging for HIP cleanup paths (destructors, reset(),
    // resource-clear-before-reuse). Failure here typically means the GPU state is
    // already corrupted, but we are tearing down anyway — logging at WARN keeps
    // diagnostics visible without escalating during shutdown/error rollback,
    // where throwing or logging at ERROR could mask the real failure or trigger
    // std::terminate from a destructor on stack unwind.
#define HIP_WARN_IF_FAIL(call)                                                    \
    do                                                                            \
    {                                                                             \
        hipError_t _err = (call);                                                 \
        if (_err != hipSuccess)                                                   \
        {                                                                         \
            LOG_WARN("[HIPGraphCapture] " << #call << " failed: "                 \
                                          << hipGetErrorString(_err) << " ("      \
                                          << __FILE__ << ":" << __LINE__ << ")"); \
        }                                                                         \
    } while (0)

    HIPGraphCapture::HIPGraphCapture(
        hipStream_t stream,
        int device_ordinal)
        : stream_(stream), device_ordinal_(device_ordinal)
    {
        if (!stream_ || device_ordinal_ < 0)
        {
            throw std::invalid_argument(
                "HIPGraphCapture requires an explicit stream and ROCm ordinal");
        }
    }

    HIPGraphCapture::~HIPGraphCapture() { reset(); }

    HIPGraphCapture::HIPGraphCapture(HIPGraphCapture &&other) noexcept
        : stream_(other.stream_), device_ordinal_(other.device_ordinal_),
          graph_(other.graph_), exec_(other.exec_),
          node_count_(other.node_count_),
          resident_memory_bytes_(other.resident_memory_bytes_),
          ordered_timeline_timing_events_(
              std::move(other.ordered_timeline_timing_events_)),
          ordered_timeline_timing_pending_(
              other.ordered_timeline_timing_pending_)
    {
        other.stream_ = nullptr;
        other.device_ordinal_ = -1;
        other.graph_ = nullptr;
        other.exec_ = nullptr;
        other.node_count_ = 0;
        other.resident_memory_bytes_ = 0u;
        other.ordered_timeline_timing_pending_ = false;
    }

    HIPGraphCapture &HIPGraphCapture::operator=(HIPGraphCapture &&other) noexcept
    {
        if (this != &other)
        {
            reset();
            stream_ = other.stream_;
            device_ordinal_ = other.device_ordinal_;
            graph_ = other.graph_;
            exec_ = other.exec_;
            node_count_ = other.node_count_;
            resident_memory_bytes_ = other.resident_memory_bytes_;
            ordered_timeline_timing_events_ =
                std::move(other.ordered_timeline_timing_events_);
            ordered_timeline_timing_pending_ =
                other.ordered_timeline_timing_pending_;
            other.stream_ = nullptr;
            other.device_ordinal_ = -1;
            other.graph_ = nullptr;
            other.exec_ = nullptr;
            other.node_count_ = 0;
            other.resident_memory_bytes_ = 0u;
            other.ordered_timeline_timing_pending_ = false;
        }
        return *this;
    }

    bool HIPGraphCapture::activateOwner(const char *operation) const noexcept
    {
        if (!stream_ || device_ordinal_ < 0)
        {
            LOG_ERROR("[HIPGraphCapture] "
                      << (operation ? operation : "graph operation")
                      << " has no valid ROCm owner"
                      << " stream=" << static_cast<void *>(stream_)
                      << " device=" << device_ordinal_);
            return false;
        }
        const hipError_t error = hipSetDevice(device_ordinal_);
        if (error != hipSuccess)
        {
            LOG_ERROR("[HIPGraphCapture] hipSetDevice(" << device_ordinal_
                      << ") failed before "
                      << (operation ? operation : "graph operation") << ": "
                      << hipGetErrorString(error));
            return false;
        }
        return true;
    }

    bool HIPGraphCapture::beginCapture()
    {
        if (!activateOwner("beginCapture"))
            return false;
        // Destroy any previous graph (but keep exec_ for tryUpdate)
        if (graph_)
        {
            const hipError_t destroy_error = hipGraphDestroy(graph_);
            if (destroy_error != hipSuccess)
            {
                LOG_ERROR("[HIPGraphCapture] Cannot begin a new capture because "
                          "destroying the prior captured graph failed: "
                          << hipGetErrorString(destroy_error));
                return false;
            }
            graph_ = nullptr;
            node_count_ = 0;
            destroyOrderedTimelineTimingEvents();
        }

        hipError_t err = hipStreamBeginCapture(stream_, hipStreamCaptureModeRelaxed);
        if (err != hipSuccess)
        {
            LOG_ERROR("[HIPGraphCapture] hipStreamBeginCapture failed: " << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool HIPGraphCapture::endCapture()
    {
        if (!activateOwner("endCapture"))
            return false;
        const auto end_capture_begin = std::chrono::steady_clock::now();
        hipError_t err = hipStreamEndCapture(stream_, &graph_);
        const auto end_capture_elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - end_capture_begin);
        if (err != hipSuccess)
        {
            LOG_ERROR("[HIPGraphCapture] hipStreamEndCapture failed: " << hipGetErrorString(err));
            graph_ = nullptr;
            return false;
        }
        if (!graph_)
        {
            LOG_ERROR("[HIPGraphCapture] hipStreamEndCapture produced null graph");
            return false;
        }

        // Cache node count
        size_t count = 0;
        err = hipGraphGetNodes(graph_, nullptr, &count);
        if (err == hipSuccess)
        {
            node_count_ = count;
        }
        const double end_capture_ms =
            static_cast<double>(end_capture_elapsed.count()) / 1000.0;
        if (end_capture_ms >= 1000.0)
        {
            LOG_WARN(
                "[HIPGraphCapture] Slow hipStreamEndCapture"
                << " device=ROCm:" << device_ordinal_
                << " nodes=" << node_count_
                << " elapsed_ms=" << end_capture_ms);
        }
        else
        {
            LOG_TRACE(
                "[HIPGraphCapture] Captured graph"
                << " device=ROCm:" << device_ordinal_
                << " nodes=" << node_count_
                << " elapsed_ms=" << end_capture_ms);
        }
        return true;
    }

    bool HIPGraphCapture::instantiate()
    {
        if (!activateOwner("instantiate"))
            return false;
        if (!graph_)
        {
            LOG_ERROR("[HIPGraphCapture] Cannot instantiate: no captured graph");
            return false;
        }
        // Destroy old executable
        if (exec_)
        {
            const hipError_t destroy_error = hipGraphExecDestroy(exec_);
            if (destroy_error != hipSuccess)
            {
                LOG_ERROR("[HIPGraphCapture] Cannot replace graph executable because "
                          "destroying the prior executable failed: "
                          << hipGetErrorString(destroy_error));
                return false;
            }
            exec_ = nullptr;
            resident_memory_bytes_ = 0u;
        }

        std::size_t free_bytes_before = 0u;
        std::size_t total_bytes_before = 0u;
        const hipError_t memory_before_status = hipMemGetInfo(
            &free_bytes_before,
            &total_bytes_before);
        if (memory_before_status != hipSuccess)
        {
            LOG_ERROR(
                "[HIPGraphCapture] Cannot account graph-executable VRAM before instantiation: "
                << hipGetErrorString(memory_before_status));
            return false;
        }

        const auto instantiate_begin = std::chrono::steady_clock::now();
        hipError_t err = hipGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0);
        const auto instantiate_elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - instantiate_begin);
        if (err != hipSuccess)
        {
            LOG_ERROR("[HIPGraphCapture] hipGraphInstantiate failed: " << hipGetErrorString(err));
            exec_ = nullptr;
            return false;
        }

        std::size_t free_bytes_after = 0u;
        std::size_t total_bytes_after = 0u;
        const hipError_t memory_after_status = hipMemGetInfo(
            &free_bytes_after,
            &total_bytes_after);
        if (memory_after_status != hipSuccess ||
            total_bytes_after != total_bytes_before)
        {
            LOG_ERROR(
                "[HIPGraphCapture] Cannot account graph-executable VRAM after instantiation: status="
                << hipGetErrorString(memory_after_status)
                << " total_before=" << total_bytes_before
                << " total_after=" << total_bytes_after);
            const hipError_t destroy_error = hipGraphExecDestroy(exec_);
            if (destroy_error != hipSuccess)
            {
                LOG_ERROR(
                    "[HIPGraphCapture] Failed to destroy an executable after its VRAM accounting failed: "
                    << hipGetErrorString(destroy_error));
            }
            exec_ = nullptr;
            return false;
        }
        /*
         * HIP owns this storage and exposes no allocation handle. The positive
         * free-memory delta is evidence for complete-family certification, not
         * an allocation attributable to this executable: HIP may grow a shared
         * pool now and satisfy later graph owners from it. Physical admission
         * therefore commits the complete family extent. Clamp an apparent
         * increase to zero because unrelated deferred driver state may be
         * retired at the same boundary.
         */
        const std::size_t resident_delta_bytes =
            free_bytes_before > free_bytes_after
                ? free_bytes_before - free_bytes_after
                : 0u;
        resident_memory_bytes_ = resident_delta_bytes;
        if (vramBomEnabled())
        {
            logVramBomLine(
                "hip_graph_executable",
                "action=instantiate device=" +
                    std::to_string(device_ordinal_) +
                    " nodes=" + std::to_string(node_count_) +
                    " resident_delta_bytes=" +
                    std::to_string(resident_delta_bytes) +
                    " free_before_bytes=" +
                    std::to_string(free_bytes_before) +
                    " free_after_bytes=" +
                    std::to_string(free_bytes_after));
        }
        const double instantiate_ms =
            static_cast<double>(instantiate_elapsed.count()) / 1000.0;
        if (instantiate_ms >= 1000.0)
        {
            LOG_WARN(
                "[HIPGraphCapture] Slow hipGraphInstantiate"
                << " device=ROCm:" << device_ordinal_
                << " nodes=" << node_count_
                << " elapsed_ms=" << instantiate_ms
                << " resident_delta_bytes=" << resident_delta_bytes);
        }
        else
        {
            LOG_TRACE(
                "[HIPGraphCapture] Instantiated graph executable"
                << " device=ROCm:" << device_ordinal_
                << " nodes=" << node_count_
                << " elapsed_ms=" << instantiate_ms
                << " resident_delta_bytes=" << resident_delta_bytes
                << " free_bytes_after=" << free_bytes_after);
        }
        return true;
    }

    bool HIPGraphCapture::launch()
    {
        return launchOnStream(static_cast<void *>(stream_));
    }

    bool HIPGraphCapture::launchOnStream(void *stream) const
    {
        if (!activateOwner("launchOnStream") || !exec_ || !stream)
        {
            LOG_ERROR("[HIPGraphCapture] Cannot launch: executable or explicit stream is missing");
            return false;
        }
        hipError_t err = hipGraphLaunch(
            exec_,
            static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            LOG_ERROR("[HIPGraphCapture] hipGraphLaunch failed: " << hipGetErrorString(err));
            return false;
        }

        if (!ordered_timeline_timing_events_.empty())
            ordered_timeline_timing_pending_ = true;

        return true;
    }

    bool HIPGraphCapture::buildOrderedTimelineTransaction(
        std::span<const GPUOrderedTimelineStep> ordered_steps,
        GPUOrderedTimelineInstrumentation instrumentation)
    {
        if (!activateOwner("buildOrderedTimelineTransaction") ||
            ordered_steps.empty())
        {
            LOG_ERROR("[HIPGraphCapture] Ordered timeline transaction requires an exact owner and non-empty steps");
            return false;
        }

        std::vector<const HIPGraphCapture *> fragments(
            ordered_steps.size(), nullptr);
        for (std::size_t index = 0u; index < ordered_steps.size(); ++index)
        {
            const auto &step = ordered_steps[index];
            if (!step.valid())
            {
                LOG_ERROR("[HIPGraphCapture] Invalid ordered timeline step index="
                          << index);
                return false;
            }
            if (step.kind !=
                GPUOrderedTimelineStepKind::CapturedFragment)
            {
                continue;
            }
            const auto *fragment =
                dynamic_cast<const HIPGraphCapture *>(step.capture);
            if (!fragment || fragment == this ||
                fragment->deviceOrdinal() != device_ordinal_ ||
                !fragment->graph() || fragment->nodeCount() == 0u)
            {
                LOG_ERROR("[HIPGraphCapture] Ordered timeline fragment has incompatible ownership"
                          << " index=" << index
                          << " name=" << step.name);
                return false;
            }
            fragments[index] = fragment;
        }

        reset();
        hipError_t error = hipGraphCreate(&graph_, 0u);
        if (error != hipSuccess)
        {
            LOG_ERROR("[HIPGraphCapture] hipGraphCreate failed for ordered timeline transaction: "
                      << hipGetErrorString(error));
            return false;
        }
        auto fail = [&](const char *operation, hipError_t operation_error)
        {
            LOG_ERROR("[HIPGraphCapture] " << operation
                      << " failed for ordered timeline transaction: "
                      << hipGetErrorString(operation_error));
            reset();
            return false;
        };

        if (instrumentation ==
            GPUOrderedTimelineInstrumentation::PerStepEvents)
        {
            ordered_timeline_timing_events_.reserve(ordered_steps.size());
            for (const auto &step : ordered_steps)
            {
                OrderedTimelineTimingEvents timing{
                    .name = step.name,
                    .kind = step.kind,
                };
                error = hipEventCreate(&timing.start);
                if (error != hipSuccess)
                    return fail("hipEventCreate(timeline start)", error);
                error = hipEventCreate(&timing.stop);
                if (error != hipSuccess)
                {
                    (void)hipEventDestroy(timing.start);
                    return fail("hipEventCreate(timeline stop)", error);
                }
                ordered_timeline_timing_events_.push_back(
                    std::move(timing));
            }
        }

        std::vector<hipGraphNode_t> frontier;
        std::size_t expected_node_count = 0u;
        for (std::size_t index = 0u; index < ordered_steps.size(); ++index)
        {
            const auto &step = ordered_steps[index];
            if (!ordered_timeline_timing_events_.empty())
            {
                hipGraphNode_t timing_start = nullptr;
                error = hipGraphAddEventRecordNode(
                    &timing_start,
                    graph_,
                    frontier.empty() ? nullptr : frontier.data(),
                    frontier.size(),
                    ordered_timeline_timing_events_[index].start);
                if (error != hipSuccess)
                    return fail("hipGraphAddEventRecordNode(timeline start)", error);
                frontier.assign(1u, timing_start);
                ++expected_node_count;
            }

            hipGraphNode_t node = nullptr;
            if (step.kind ==
                GPUOrderedTimelineStepKind::CapturedFragment)
            {
                ImportedHIPGraph imported;
                std::string import_error;
                if (!importHipGraphDAG(
                        fragments[index]->graph(),
                        graph_,
                        frontier,
                        imported,
                        import_error,
                        "ordered_step[" + std::to_string(index) + "](" +
                            step.name + ")"))
                {
                    LOG_ERROR("[HIPGraphCapture] Cannot flatten ordered "
                              "timeline fragment: "
                              << import_error);
                    reset();
                    return false;
                }
                frontier = std::move(imported.leaves);
                expected_node_count += imported.node_count;
            }
            else if (step.kind == GPUOrderedTimelineStepKind::WaitValue64)
            {
                // HIP's beta batch-memory graph wait may report a dependency
                // while issuing successor work before peer bytes are visible.
                // An ordinary one-wave kernel is the wait authority so native
                // graph dependencies gate the actual imported fragment roots.
                error = hip_graph_timeline::addSystemWaitValue64Node(
                    &node,
                    graph_,
                    frontier.empty() ? nullptr : frontier.data(),
                    frontier.size(),
                    step.signal,
                    step.value);
                if (error != hipSuccess)
                    return fail("addSystemWaitValue64Node", error);
            }
            else
            {
                // A system-release kernel orders every producer frontier and
                // publishes the leased timeline without a host submission.
                error = hip_graph_timeline::addSystemReleaseValue64Node(
                    &node,
                    graph_,
                    frontier.empty() ? nullptr : frontier.data(),
                    frontier.size(),
                    step.signal,
                    step.value);
                if (error != hipSuccess)
                    return fail(
                        "addSystemReleaseValue64Node", error);
                frontier.assign(1u, node);
                ++expected_node_count;
            }
            if (step.kind == GPUOrderedTimelineStepKind::WaitValue64)
            {
                frontier.assign(1u, node);
                ++expected_node_count;
            }

            if (!ordered_timeline_timing_events_.empty())
            {
                hipGraphNode_t timing_stop = nullptr;
                error = hipGraphAddEventRecordNode(
                    &timing_stop,
                    graph_,
                    frontier.data(),
                    frontier.size(),
                    ordered_timeline_timing_events_[index].stop);
                if (error != hipSuccess)
                    return fail("hipGraphAddEventRecordNode(timeline stop)", error);
                frontier.assign(1u, timing_stop);
                ++expected_node_count;
            }
        }

        std::size_t count = 0u;
        error = hipGraphGetNodes(graph_, nullptr, &count);
        if (error != hipSuccess)
            return fail("hipGraphGetNodes", error);
        node_count_ = count;
        return !frontier.empty() && node_count_ == expected_node_count;
    }

    GPUOrderedTimelineTimingSnapshot
    HIPGraphCapture::consumeOrderedTimelineTiming()
    {
        GPUOrderedTimelineTimingSnapshot snapshot;
        if (ordered_timeline_timing_events_.empty())
            return snapshot;
        if (!ordered_timeline_timing_pending_)
        {
            snapshot.state =
                GPUOrderedTimelineTimingState::AwaitingLaunch;
            return snapshot;
        }
        if (!activateOwner("consumeOrderedTimelineTiming"))
        {
            snapshot.state = GPUOrderedTimelineTimingState::Failed;
            snapshot.error = "could not activate the immutable HIP graph owner";
            return snapshot;
        }

        const hipError_t query =
            hipEventQuery(ordered_timeline_timing_events_.back().stop);
        if (query == hipErrorNotReady)
        {
            snapshot.state = GPUOrderedTimelineTimingState::Pending;
            return snapshot;
        }
        if (query != hipSuccess)
        {
            snapshot.state = GPUOrderedTimelineTimingState::Failed;
            snapshot.error = std::string("hipEventQuery failed: ") +
                             hipGetErrorString(query);
            return snapshot;
        }

        snapshot.samples.reserve(ordered_timeline_timing_events_.size());
        for (const auto &timing : ordered_timeline_timing_events_)
        {
            float elapsed_ms = 0.0f;
            const hipError_t elapsed =
                hipEventElapsedTime(&elapsed_ms, timing.start, timing.stop);
            if (elapsed != hipSuccess)
            {
                snapshot.samples.clear();
                snapshot.state = GPUOrderedTimelineTimingState::Failed;
                snapshot.error = std::string("hipEventElapsedTime failed: ") +
                                 hipGetErrorString(elapsed);
                return snapshot;
            }
            snapshot.samples.push_back({
                .name = timing.name,
                .kind = timing.kind,
                .elapsed_ms = static_cast<double>(elapsed_ms),
            });
        }
        ordered_timeline_timing_pending_ = false;
        snapshot.state = GPUOrderedTimelineTimingState::Complete;
        return snapshot;
    }

    void HIPGraphCapture::destroyOrderedTimelineTimingEvents() noexcept
    {
        for (auto &timing : ordered_timeline_timing_events_)
        {
            if (timing.start)
                HIP_WARN_IF_FAIL(hipEventDestroy(timing.start));
            if (timing.stop)
                HIP_WARN_IF_FAIL(hipEventDestroy(timing.stop));
            timing.start = nullptr;
            timing.stop = nullptr;
        }
        ordered_timeline_timing_events_.clear();
        ordered_timeline_timing_pending_ = false;
    }

    GraphUpdateResult HIPGraphCapture::tryUpdate()
    {
        LOG_ERROR("[HIPGraphCapture] tryUpdate() called even though HIP graph "
                  "executable update is not an advertised backend capability");
        return GraphUpdateResult::Failed;
    }

    bool HIPGraphCapture::hasExecutable() const
    {
        return exec_ != nullptr;
    }

    size_t HIPGraphCapture::nodeCount() const
    {
        return node_count_;
    }

    bool HIPGraphCapture::inspectKernelNodes(
        std::vector<GPUGraphKernelNodeInfo> &kernel_nodes,
        std::string *error) const
    {
        if (!activateOwner("inspectKernelNodes"))
            return false;
        kernel_nodes.clear();
        if (error)
            error->clear();
        if (graph_ == nullptr)
        {
            if (error)
                *error = "cannot inspect HIP kernel nodes without a captured graph";
            return false;
        }

        std::string inspection_error;
        if (!inspectHipKernelNodesRecursive(
                graph_,
                stream_,
                "root",
                0,
                kernel_nodes,
                inspection_error))
        {
            kernel_nodes.clear();
            if (error)
                *error = std::move(inspection_error);
            return false;
        }
        return true;
    }

    void HIPGraphCapture::reset()
    {
        if ((exec_ || graph_) && !activateOwner("reset"))
        {
            LOG_ERROR("[HIPGraphCapture] Cannot release graph resources without their immutable ROCm owner");
            std::terminate();
        }

        const bool account_vram = vramBomEnabled() && (exec_ || graph_);
        const std::size_t released_node_count = node_count_;
        std::size_t free_bytes_before = 0u;
        std::size_t total_bytes_before = 0u;
        hipError_t memory_before_status = hipSuccess;
        if (account_vram)
        {
            memory_before_status = hipMemGetInfo(
                &free_bytes_before,
                &total_bytes_before);
        }
        if (exec_)
        {
            HIP_WARN_IF_FAIL(hipGraphExecDestroy(exec_));
            exec_ = nullptr;
        }
        resident_memory_bytes_ = 0u;
        if (graph_)
        {
            HIP_WARN_IF_FAIL(hipGraphDestroy(graph_));
            graph_ = nullptr;
        }
        destroyOrderedTimelineTimingEvents();

        if (account_vram)
        {
            std::size_t free_bytes_after = 0u;
            std::size_t total_bytes_after = 0u;
            const hipError_t memory_after_status = hipMemGetInfo(
                &free_bytes_after,
                &total_bytes_after);
            const std::size_t released_resident_bytes =
                memory_before_status == hipSuccess &&
                        memory_after_status == hipSuccess &&
                        total_bytes_before == total_bytes_after &&
                        free_bytes_after > free_bytes_before
                    ? free_bytes_after - free_bytes_before
                    : 0u;
            logVramBomLine(
                "hip_graph_executable",
                "action=reset device=" +
                    std::to_string(device_ordinal_) +
                    " nodes=" + std::to_string(released_node_count) +
                    " released_resident_bytes=" +
                    std::to_string(released_resident_bytes) +
                    " free_before_bytes=" +
                    std::to_string(free_bytes_before) +
                    " free_after_bytes=" +
                    std::to_string(free_bytes_after) +
                    " memory_before_status=" +
                    std::to_string(static_cast<int>(memory_before_status)) +
                    " memory_after_status=" +
                    std::to_string(static_cast<int>(memory_after_status)));
        }
        node_count_ = 0;
    }

} // namespace llaminar2

#endif // HAVE_ROCM
