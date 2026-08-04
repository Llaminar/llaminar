#ifdef HAVE_ROCM

#include "HIPGraphCapture.h"
#include "../../utils/Logger.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace
    {
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

    HIPGraphCapture::HIPGraphCapture(hipStream_t stream) : stream_(stream) {}

    HIPGraphCapture::~HIPGraphCapture() { reset(); }

    HIPGraphCapture::HIPGraphCapture(HIPGraphCapture &&other) noexcept
        : stream_(other.stream_), graph_(other.graph_), exec_(other.exec_),
          node_count_(other.node_count_)
    {
        other.stream_ = nullptr;
        other.graph_ = nullptr;
        other.exec_ = nullptr;
        other.node_count_ = 0;
    }

    HIPGraphCapture &HIPGraphCapture::operator=(HIPGraphCapture &&other) noexcept
    {
        if (this != &other)
        {
            reset();
            stream_ = other.stream_;
            graph_ = other.graph_;
            exec_ = other.exec_;
            node_count_ = other.node_count_;
            other.stream_ = nullptr;
            other.graph_ = nullptr;
            other.exec_ = nullptr;
            other.node_count_ = 0;
        }
        return *this;
    }

    bool HIPGraphCapture::beginCapture()
    {
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
        hipError_t err = hipStreamEndCapture(stream_, &graph_);
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
        LOG_DEBUG("[HIPGraphCapture] Captured graph with " << node_count_ << " nodes");
        return true;
    }

    bool HIPGraphCapture::instantiate()
    {
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
        }

        hipError_t err = hipGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0);
        if (err != hipSuccess)
        {
            LOG_ERROR("[HIPGraphCapture] hipGraphInstantiate failed: " << hipGetErrorString(err));
            exec_ = nullptr;
            return false;
        }
        LOG_DEBUG("[HIPGraphCapture] Instantiated graph executable (" << node_count_ << " nodes)");
        return true;
    }

    bool HIPGraphCapture::launch()
    {
        if (!exec_)
        {
            LOG_ERROR("[HIPGraphCapture] Cannot launch: no instantiated executable");
            return false;
        }
        hipError_t err = hipGraphLaunch(exec_, stream_);
        if (err != hipSuccess)
        {
            LOG_ERROR("[HIPGraphCapture] hipGraphLaunch failed: " << hipGetErrorString(err));
            return false;
        }
        return true;
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
        if (exec_)
        {
            HIP_WARN_IF_FAIL(hipGraphExecDestroy(exec_));
            exec_ = nullptr;
        }
        if (graph_)
        {
            HIP_WARN_IF_FAIL(hipGraphDestroy(graph_));
            graph_ = nullptr;
        }
        node_count_ = 0;
    }

} // namespace llaminar2

#endif // HAVE_ROCM
