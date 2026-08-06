/**
 * @file CUDAGraphCapture.cu
 * @brief CUDA graph capture, replay, and device-controlled graph composition.
 *
 * CUDA conditional nodes accept a narrower set of body-node types than an
 * ordinary captured graph. Event record/wait nodes, host nodes, allocation
 * nodes, and nested conditionals are forbidden inside Llaminar's transaction
 * fragments. Parent construction authenticates every source graph recursively
 * and fails before composition when a producer exposes one of those nodes.
 * Ordering must be expressed by the graph's native dependency edges at the
 * producer; graph composition never rewrites a captured lifecycle after the
 * fact.
 */

#ifdef HAVE_CUDA

#include "CUDAGraphCapture.h"
#include "../../utils/Logger.h"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2
{

    namespace
    {
        /**
         * @brief Return a stable diagnostic name for one CUDA graph node type.
         *
         * CUDA's instantiation log frequently reports only `invalid argument`.
         * Keeping the native enum and a readable name together makes a fragment
         * contract failure actionable without relying on driver-specific text.
         */
        const char *cudaGraphNodeTypeName(cudaGraphNodeType type) noexcept
        {
            switch (type)
            {
            case cudaGraphNodeTypeKernel:
                return "kernel";
            case cudaGraphNodeTypeMemcpy:
                return "memcpy";
            case cudaGraphNodeTypeMemset:
                return "memset";
            case cudaGraphNodeTypeHost:
                return "host";
            case cudaGraphNodeTypeGraph:
                return "child_graph";
            case cudaGraphNodeTypeEmpty:
                return "empty";
            case cudaGraphNodeTypeWaitEvent:
                return "event_wait";
            case cudaGraphNodeTypeEventRecord:
                return "event_record";
            case cudaGraphNodeTypeExtSemaphoreSignal:
                return "external_semaphore_signal";
            case cudaGraphNodeTypeExtSemaphoreWait:
                return "external_semaphore_wait";
            case cudaGraphNodeTypeMemAlloc:
                return "memory_allocation";
            case cudaGraphNodeTypeMemFree:
                return "memory_free";
            case cudaGraphNodeTypeConditional:
                return "conditional";
            case cudaGraphNodeTypeCount:
                return "invalid";
            }
            return "unknown";
        }

        /**
         * @brief Recursively authenticate one captured CUDA conditional fragment.
         *
         * CUDA WHILE/IF/SWITCH bodies admit only kernels, memcpy, memset, empty,
         * and child-graph nodes in the reusable fragments composed here. Child
         * graphs are traversed recursively so a forbidden event or nested
         * conditional cannot pass the parent contract and fail later with an
         * opaque instantiation error.
         *
         * This validation runs only while publishing a new executable. Its small
         * host vectors cannot enter captured replay or the inference hot path.
         *
         * @param graph Graph (or nested child graph) being authenticated.
         * @param fragment Semantic producer role supplied by the composition API.
         * @param fragment_index Position in producer-to-consumer transaction order.
         * @param graph_path Recursive child path used in fatal diagnostics.
         * @return true only when every recursively reachable node is legal.
         */
        bool validateCudaConditionalBodyFragmentGraph(
            cudaGraph_t graph,
            const char *fragment,
            size_t fragment_index,
            const std::string &graph_path)
        {
            size_t node_count = 0;
            cudaError_t error = cudaGraphGetNodes(graph, nullptr, &node_count);
            if (error != cudaSuccess)
            {
                LOG_ERROR("[CUDAGraphCapture] Cannot enumerate conditional-body fragment"
                          << " fragment=" << fragment
                          << " index=" << fragment_index
                          << " path=" << graph_path
                          << " error=" << cudaGetErrorString(error));
                return false;
            }
            if (node_count == 0)
            {
                LOG_ERROR("[CUDAGraphCapture] Conditional-body fragment contains an empty graph"
                          << " fragment=" << fragment
                          << " index=" << fragment_index
                          << " path=" << graph_path);
                return false;
            }

            std::vector<cudaGraphNode_t> nodes(node_count);
            error = cudaGraphGetNodes(graph, nodes.data(), &node_count);
            if (error != cudaSuccess)
            {
                LOG_ERROR("[CUDAGraphCapture] Cannot read conditional-body fragment nodes"
                          << " fragment=" << fragment
                          << " index=" << fragment_index
                          << " path=" << graph_path
                          << " error=" << cudaGetErrorString(error));
                return false;
            }
            nodes.resize(node_count);

            for (size_t node_index = 0; node_index < nodes.size(); ++node_index)
            {
                cudaGraphNodeType type = cudaGraphNodeTypeCount;
                error = cudaGraphNodeGetType(nodes[node_index], &type);
                if (error != cudaSuccess)
                {
                    LOG_ERROR("[CUDAGraphCapture] Cannot classify conditional-body fragment node"
                              << " fragment=" << fragment
                              << " index=" << fragment_index
                              << " path=" << graph_path
                              << " node=" << node_index
                              << " error=" << cudaGetErrorString(error));
                    return false;
                }

                switch (type)
                {
                case cudaGraphNodeTypeKernel:
                case cudaGraphNodeTypeMemcpy:
                case cudaGraphNodeTypeMemset:
                case cudaGraphNodeTypeEmpty:
                    break;
                case cudaGraphNodeTypeGraph:
                {
                    cudaGraph_t child_graph = nullptr;
                    error = cudaGraphChildGraphNodeGetGraph(
                        nodes[node_index], &child_graph);
                    if (error != cudaSuccess || !child_graph)
                    {
                        LOG_ERROR("[CUDAGraphCapture] Cannot inspect nested conditional-body child graph"
                                  << " fragment=" << fragment
                                  << " index=" << fragment_index
                                  << " path=" << graph_path
                                  << " node=" << node_index
                                  << " error=" << cudaGetErrorString(error));
                        return false;
                    }
                    const std::string child_path =
                        graph_path + "/child[" +
                        std::to_string(node_index) + "]";
                    if (!validateCudaConditionalBodyFragmentGraph(
                            child_graph,
                            fragment,
                            fragment_index,
                            child_path))
                    {
                        return false;
                    }
                    break;
                }
                case cudaGraphNodeTypeConditional:
                    LOG_ERROR("[CUDAGraphCapture] Conditional-body fragment cannot be cloned as a child because it already contains a conditional node"
                              << " fragment=" << fragment
                              << " index=" << fragment_index
                              << " path=" << graph_path
                              << " node=" << node_index);
                    return false;
                default:
                    LOG_ERROR("[CUDAGraphCapture] Conditional-body fragment contains a forbidden node"
                              << " fragment=" << fragment
                              << " index=" << fragment_index
                              << " path=" << graph_path
                              << " node=" << node_index
                              << " node_type=" << cudaGraphNodeTypeName(type)
                              << " node_type_value=" << static_cast<int>(type));
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Render a CUDA driver result without losing its symbolic name.
         *
         * Graphs captured through libraries such as NCCL may contain driver
         * kernel handles that the CUDA runtime graph-query API cannot decode.
         * Inventory therefore uses the driver graph API uniformly and reports
         * both the stable driver enum and the optional explanatory string.
         */
        std::string cudaDriverErrorString(CUresult status)
        {
            const char *name = nullptr;
            const char *description = nullptr;
            (void)cuGetErrorName(status, &name);
            (void)cuGetErrorString(status, &description);
            return std::string(name ? name : "CUDA_ERROR_UNKNOWN") +
                   " (" + (description ? description : "no description") + ")";
        }

        /**
         * @brief Recursively append every CUDA kernel node in one captured graph.
         *
         * CUDA represents a captured stage hierarchy with child-graph nodes. A
         * shallow walk would therefore undercount the production workload and
         * hide precisely the small launches needed for fusion analysis. This
         * routine follows every child and records launch metadata without
         * instantiating, replaying, synchronizing, or reading device memory.
         *
         * @param graph Current graph in the recursive traversal.
         * @param graph_path Stable diagnostic path rooted at `root`.
         * @param nesting_depth Number of child-graph edges already traversed.
         * @param kernel_nodes Destination inventory.
         * @param error Receives a precise native-API failure.
         * @return true only when every reachable node was inspected.
         */
        bool inspectCudaKernelNodesRecursive(
            CUgraph graph,
            const std::string &graph_path,
            size_t nesting_depth,
            std::vector<GPUGraphKernelNodeInfo> &kernel_nodes,
            std::string &error)
        {
            size_t node_count = 0;
            CUresult status = cuGraphGetNodes(graph, nullptr, &node_count);
            if (status != CUDA_SUCCESS)
            {
                error = "cuGraphGetNodes(count) failed at " + graph_path +
                        ": " + cudaDriverErrorString(status);
                return false;
            }

            std::vector<CUgraphNode> nodes(node_count);
            if (node_count > 0)
            {
                status = cuGraphGetNodes(graph, nodes.data(), &node_count);
                if (status != CUDA_SUCCESS)
                {
                    error = "cuGraphGetNodes(nodes) failed at " + graph_path +
                            ": " + cudaDriverErrorString(status);
                    return false;
                }
                nodes.resize(node_count);
            }

            for (size_t node_index = 0; node_index < nodes.size(); ++node_index)
            {
                const std::string node_path =
                    graph_path + "/node[" + std::to_string(node_index) + "]";
                CUgraphNodeType type = CU_GRAPH_NODE_TYPE_EMPTY;
                status = cuGraphNodeGetType(nodes[node_index], &type);
                if (status != CUDA_SUCCESS)
                {
                    error = "cuGraphNodeGetType failed at " + node_path +
                            ": " + cudaDriverErrorString(status);
                    return false;
                }

                if (type == CU_GRAPH_NODE_TYPE_KERNEL)
                {
                    CUDA_KERNEL_NODE_PARAMS params{};
                    status = cuGraphKernelNodeGetParams(
                        nodes[node_index], &params);
                    if (status != CUDA_SUCCESS)
                    {
                        error = "cuGraphKernelNodeGetParams failed at " +
                                node_path + ": " +
                                cudaDriverErrorString(status);
                        return false;
                    }

                    const char *driver_name = nullptr;
                    CUresult name_status = CUDA_ERROR_INVALID_HANDLE;
                    uintptr_t function_identity = 0;
                    CUfunction function = nullptr;
                    if (params.func != nullptr)
                    {
                        function = params.func;
                        function_identity =
                            reinterpret_cast<uintptr_t>(params.func);
                        name_status = cuFuncGetName(&driver_name, params.func);
                    }
#if CUDA_VERSION >= 12000
                    else if (params.kern != nullptr)
                    {
                        function_identity =
                            reinterpret_cast<uintptr_t>(params.kern);
                        name_status = cuKernelGetName(&driver_name, params.kern);
                        status = cuKernelGetFunction(&function, params.kern);
                        if (status != CUDA_SUCCESS || function == nullptr)
                        {
                            error = "cuKernelGetFunction failed at " +
                                    node_path + ": " +
                                    cudaDriverErrorString(status);
                            return false;
                        }
                    }
#endif
                    else
                    {
                        error = "CUDA kernel node has neither CUfunction nor "
                                "CUkernel identity at " + node_path;
                        return false;
                    }
                    const bool name_resolved =
                        name_status == CUDA_SUCCESS && driver_name != nullptr &&
                        driver_name[0] != '\0';

                    auto requireFunctionAttribute =
                        [&](CUfunction_attribute attribute,
                            const char *attribute_name,
                            int &value) -> bool
                    {
                        status = cuFuncGetAttribute(
                            &value,
                            attribute,
                            function);
                        if (status == CUDA_SUCCESS)
                            return true;
                        error = std::string("cuFuncGetAttribute(") +
                                attribute_name + ") failed at " + node_path +
                                ": " + cudaDriverErrorString(status);
                        return false;
                    };

                    int registers_per_thread = 0;
                    int static_shared_memory_bytes = 0;
                    int local_memory_bytes_per_thread = 0;
                    int max_threads_per_block = 0;
                    if (!requireFunctionAttribute(
                            CU_FUNC_ATTRIBUTE_NUM_REGS,
                            "NUM_REGS",
                            registers_per_thread) ||
                        !requireFunctionAttribute(
                            CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                            "SHARED_SIZE_BYTES",
                            static_shared_memory_bytes) ||
                        !requireFunctionAttribute(
                            CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                            "LOCAL_SIZE_BYTES",
                            local_memory_bytes_per_thread) ||
                        !requireFunctionAttribute(
                            CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                            "MAX_THREADS_PER_BLOCK",
                            max_threads_per_block))
                    {
                        return false;
                    }

                    const uint64_t captured_block_threads =
                        static_cast<uint64_t>(params.blockDimX) *
                        static_cast<uint64_t>(params.blockDimY) *
                        static_cast<uint64_t>(params.blockDimZ);
                    if (captured_block_threads == 0 ||
                        captured_block_threads >
                            static_cast<uint64_t>(
                                std::numeric_limits<int>::max()))
                    {
                        error = "CUDA kernel node has invalid flattened block "
                                "size at " +
                                node_path;
                        return false;
                    }
                    int max_active_blocks_per_sm = 0;
                    status = cuOccupancyMaxActiveBlocksPerMultiprocessor(
                        &max_active_blocks_per_sm,
                        function,
                        static_cast<int>(captured_block_threads),
                        params.sharedMemBytes);
                    if (status != CUDA_SUCCESS ||
                        max_active_blocks_per_sm <= 0)
                    {
                        error = "cuOccupancyMaxActiveBlocksPerMultiprocessor "
                                "failed at " +
                                node_path + ": " +
                                cudaDriverErrorString(status);
                        return false;
                    }
                    kernel_nodes.push_back(GPUGraphKernelNodeInfo{
                        .name = name_resolved ? driver_name : "unresolved_kernel",
                        .graph_path = node_path,
                        .function_identity = function_identity,
                        .grid_x = params.gridDimX,
                        .grid_y = params.gridDimY,
                        .grid_z = params.gridDimZ,
                        .block_x = params.blockDimX,
                        .block_y = params.blockDimY,
                        .block_z = params.blockDimZ,
                        .dynamic_shared_memory_bytes = params.sharedMemBytes,
                        .static_shared_memory_bytes = static_cast<size_t>(
                            std::max(0, static_shared_memory_bytes)),
                        .local_memory_bytes_per_thread = static_cast<size_t>(
                            std::max(0, local_memory_bytes_per_thread)),
                        .registers_per_thread = static_cast<uint32_t>(
                            std::max(0, registers_per_thread)),
                        .max_threads_per_block = static_cast<uint32_t>(
                            std::max(0, max_threads_per_block)),
                        .max_active_blocks_per_sm = static_cast<uint32_t>(
                            max_active_blocks_per_sm),
                        .nesting_depth = nesting_depth,
                        .name_resolved = name_resolved,
                    });
                    continue;
                }

                if (type == CU_GRAPH_NODE_TYPE_GRAPH)
                {
                    CUgraph child_graph = nullptr;
                    status = cuGraphChildGraphNodeGetGraph(
                        nodes[node_index], &child_graph);
                    if (status != CUDA_SUCCESS || child_graph == nullptr)
                    {
                        error = "cuGraphChildGraphNodeGetGraph failed at " +
                                node_path + ": " +
                                cudaDriverErrorString(status);
                        return false;
                    }
                    if (!inspectCudaKernelNodesRecursive(
                            child_graph,
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

    }

#if CUDART_VERSION >= 12030
    /**
     * @brief Publish the continuation value for one CUDA conditional loop.
     *
     * This deliberately tiny control kernel runs once after a complete MTP
     * transaction graph.  One thread is cheaper than a reduction for the
     * production single-request lane, while the bounded request-row scan also
     * handles request batching without atomics or temporary storage.
     */
    __global__ void updateDeviceControlledLoopCondition(
        cudaGraphConditionalHandle handle,
        const int *control_rows,
        int control_stride,
        int request_count,
        int healthy_index,
        int complete_index)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0)
            return;

        unsigned int continue_loop = 0;
        for (int request = 0; request < request_count; ++request)
        {
            const int *row =
                control_rows + static_cast<size_t>(request) * control_stride;
            if (row[healthy_index] == 0)
            {
                continue_loop = 0;
                break;
            }
            continue_loop |= row[complete_index] == 0 ? 1u : 0u;
        }
        cudaGraphSetConditional(handle, continue_loop);
    }

    /**
     * @brief Publish one fragment-local IF condition from persistent device state.
     *
     * A non-zero word admits the complete captured fragment. This deliberately
     * treats fatal sentinel values as true as well: the fragment owning that
     * state must execute and publish its precise terminal diagnostic rather
     * than allowing the parent to skip a poisoned lifecycle edge.
     */
    __global__ void updateDeviceControlledFragmentCondition(
        cudaGraphConditionalHandle handle,
        const uint32_t *condition_word)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0)
            return;
        cudaGraphSetConditional(
            handle,
            condition_word && *condition_word != 0u ? 1u : 0u);
    }

    namespace
    {
        /**
         * @brief Result of lowering one typed transaction fragment.
         *
         * CUDA reports failures through an error code, while a malformed
         * conditional body can also be discovered after a successful node
         * insertion. Carrying the operation name keeps the parent builder's
         * fatal diagnostic precise without duplicating the lowering policy.
         */
        struct DeviceControlledFragmentAppendResult
        {
            cudaGraphNode_t tail = nullptr;
            cudaError_t error = cudaSuccess;
            const char *operation = nullptr;

            /** @return true when @ref tail identifies the complete fragment. */
            [[nodiscard]] bool succeeded() const noexcept
            {
                return error == cudaSuccess && operation == nullptr && tail;
            }
        };

        /**
         * @brief Append one typed fragment to a native conditional transaction.
         *
         * This is the single lowering authority shared by fixed-depth WHILE
         * bodies and dynamic-depth SWITCH branches. An unconditional fragment
         * becomes one ordered child graph. A conditional fragment becomes an
         * exact device-word read followed by a native IF whose body owns the
         * child graph. No host observation, launch split, or alternate replay
         * path is introduced.
         *
         * @param transaction_graph WHILE body or SWITCH branch receiving work.
         * @param dependency Tail of the preceding transaction fragment.
         * @param fragment Typed execution policy and persistent predicate.
         * @param cuda_fragment Validated CUDA capture supplying the child graph.
         * @return Complete tail identity or a precise fatal CUDA operation.
         */
        DeviceControlledFragmentAppendResult appendDeviceControlledFragment(
            cudaGraph_t transaction_graph,
            cudaGraphNode_t dependency,
            const DeviceControlledLoopFragment &fragment,
            const CUDAGraphCapture &cuda_fragment)
        {
            DeviceControlledFragmentAppendResult result;
            if (fragment.execution ==
                DeviceControlledLoopFragmentExecution::Always)
            {
                result.error = cudaGraphAddChildGraphNode(
                    &result.tail,
                    transaction_graph,
                    dependency ? &dependency : nullptr,
                    dependency ? 1 : 0,
                    cuda_fragment.graph());
                if (result.error != cudaSuccess)
                    result.operation =
                        "cudaGraphAddChildGraphNode(transaction fragment)";
                return result;
            }

            cudaGraphConditionalHandle condition = 0;
            result.error = cudaGraphConditionalHandleCreate(
                &condition,
                transaction_graph,
                /*defaultLaunchValue=*/0,
                cudaGraphCondAssignDefault);
            if (result.error != cudaSuccess)
            {
                result.operation =
                    "cudaGraphConditionalHandleCreate(fragment IF)";
                return result;
            }

            cudaGraphConditionalHandle condition_arg = condition;
            const uint32_t *condition_word_arg =
                fragment.condition_word_device;
            void *condition_args[] = {
                &condition_arg,
                &condition_word_arg};
            cudaKernelNodeParams condition_params{};
            condition_params.func = reinterpret_cast<void *>(
                updateDeviceControlledFragmentCondition);
            condition_params.gridDim = dim3(1, 1, 1);
            condition_params.blockDim = dim3(1, 1, 1);
            condition_params.sharedMemBytes = 0;
            condition_params.kernelParams = condition_args;
            condition_params.extra = nullptr;

            cudaGraphNode_t condition_node = nullptr;
            result.error = cudaGraphAddKernelNode(
                &condition_node,
                transaction_graph,
                dependency ? &dependency : nullptr,
                dependency ? 1 : 0,
                &condition_params);
            if (result.error != cudaSuccess)
            {
                result.operation =
                    "cudaGraphAddKernelNode(fragment IF predicate)";
                return result;
            }

            cudaGraphNodeParams if_params{};
            if_params.type = cudaGraphNodeTypeConditional;
            if_params.conditional.handle = condition;
            if_params.conditional.type = cudaGraphCondTypeIf;
            if_params.conditional.size = 1;
            result.error = cudaGraphAddNode(
                &result.tail,
                transaction_graph,
                &condition_node,
                /*dependencyData=*/nullptr,
                /*numDependencies=*/1,
                &if_params);
            if (result.error != cudaSuccess)
            {
                result.operation = "cudaGraphAddNode(fragment IF)";
                return result;
            }
            if (!if_params.conditional.phGraph_out ||
                !if_params.conditional.phGraph_out[0])
            {
                result.error = cudaErrorInvalidValue;
                result.operation = "cudaGraphAddNode(fragment IF body)";
                return result;
            }

            cudaGraphNode_t child_node = nullptr;
            result.error = cudaGraphAddChildGraphNode(
                &child_node,
                if_params.conditional.phGraph_out[0],
                /*dependencies=*/nullptr,
                /*numDependencies=*/0,
                cuda_fragment.graph());
            if (result.error != cudaSuccess)
            {
                result.operation =
                    "cudaGraphAddChildGraphNode(fragment IF body)";
            }
            return result;
        }
    }
#endif

#if CUDART_VERSION >= 13000
    /**
     * @brief Validate and publish one transaction selector for a CUDA SWITCH node.
     *
     * A native SWITCH controls the whole request batch, so active rows must
     * agree on one selector. On malformed input this kernel invalidates every
     * active row and publishes an out-of-range handle value; CUDA consequently
     * executes no branch, and the following loop predicate terminates replay.
     */
    __global__ void updateDeviceControlledSwitchCondition(
        cudaGraphConditionalHandle handle,
        int *control_rows,
        int control_stride,
        int request_count,
        int healthy_index,
        int complete_index,
        int selector_index,
        int error_index,
        int minimum_selector,
        int maximum_selector,
        int invalid_selector_error,
        unsigned int branch_count)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0)
            return;

        int selected = -1;
        bool invalid = false;
        for (int request = 0; request < request_count; ++request)
        {
            int *row =
                control_rows + static_cast<size_t>(request) * control_stride;
            if (row[healthy_index] == 0 || row[complete_index] != 0)
                continue;

            const int candidate = row[selector_index];
            if (candidate < minimum_selector ||
                candidate > maximum_selector ||
                static_cast<unsigned int>(candidate) >= branch_count ||
                (selected >= 0 && selected != candidate))
            {
                invalid = true;
                break;
            }
            selected = candidate;
        }

        if (selected < 0)
            invalid = true;
        if (invalid)
        {
            for (int request = 0; request < request_count; ++request)
            {
                int *row =
                    control_rows + static_cast<size_t>(request) * control_stride;
                if (row[complete_index] == 0)
                {
                    row[healthy_index] = 0;
                    row[complete_index] = 1;
                    row[error_index] = invalid_selector_error;
                }
            }
            cudaGraphSetConditional(handle, branch_count);
            return;
        }

        cudaGraphSetConditional(handle, static_cast<unsigned int>(selected));
    }
#endif

    // Best-effort error logging for CUDA cleanup paths (destructors, reset(),
    // resource-clear-before-reuse). Failure here typically means the GPU state is
    // already corrupted, but we are tearing down anyway — logging at WARN keeps
    // diagnostics visible without escalating during shutdown/error rollback,
    // where throwing or logging at ERROR could mask the real failure or trigger
    // std::terminate from a destructor on stack unwind.
    //
    // cudaErrorCudartUnloading is silenced because it's expected during process
    // exit when the CUDA runtime tears down before our cleanup code runs.
#define CUDA_WARN_IF_FAIL(call)                                                             \
    do                                                                                      \
    {                                                                                       \
        cudaError_t _err = (call);                                                          \
        if (_err != cudaSuccess)                                                            \
        {                                                                                   \
            if (_err == cudaErrorCudartUnloading)                                           \
            {                                                                               \
                LOG_TRACE("[CUDAGraphCapture] " << #call                                    \
                                                << " skipped: CUDA runtime shutting down"); \
            }                                                                               \
            else                                                                            \
            {                                                                               \
                LOG_WARN("[CUDAGraphCapture] " << #call << " failed: "                      \
                                               << cudaGetErrorString(_err) << " ("          \
                                               << __FILE__ << ":" << __LINE__ << ")");      \
            }                                                                               \
        }                                                                                   \
    } while (0)

    CUDAGraphCapture::CUDAGraphCapture(
        cudaStream_t stream,
        int device_ordinal)
        : stream_(stream), device_ordinal_(device_ordinal)
    {
        if (!stream_ || device_ordinal_ < 0)
        {
            throw std::invalid_argument(
                "CUDAGraphCapture requires an explicit stream and CUDA ordinal");
        }
    }

    CUDAGraphCapture::~CUDAGraphCapture() { reset(); }

    CUDAGraphCapture::CUDAGraphCapture(CUDAGraphCapture &&other) noexcept
        : stream_(other.stream_), device_ordinal_(other.device_ordinal_),
          graph_(other.graph_), exec_(other.exec_), node_count_(other.node_count_)
    {
        other.stream_ = nullptr;
        other.device_ordinal_ = -1;
        other.graph_ = nullptr;
        other.exec_ = nullptr;
        other.node_count_ = 0;
    }

    CUDAGraphCapture &CUDAGraphCapture::operator=(CUDAGraphCapture &&other) noexcept
    {
        if (this != &other)
        {
            reset();
            stream_ = other.stream_;
            device_ordinal_ = other.device_ordinal_;
            graph_ = other.graph_;
            exec_ = other.exec_;
            node_count_ = other.node_count_;
            other.stream_ = nullptr;
            other.device_ordinal_ = -1;
            other.graph_ = nullptr;
            other.exec_ = nullptr;
            other.node_count_ = 0;
        }
        return *this;
    }

    bool CUDAGraphCapture::activateOwner(const char *operation) const noexcept
    {
        if (!stream_ || device_ordinal_ < 0)
        {
            LOG_ERROR("[CUDAGraphCapture] "
                      << (operation ? operation : "graph operation")
                      << " has no valid CUDA owner"
                      << " stream=" << static_cast<void *>(stream_)
                      << " device=" << device_ordinal_);
            return false;
        }

        const cudaError_t error = cudaSetDevice(device_ordinal_);
        if (error != cudaSuccess)
        {
            LOG_ERROR("[CUDAGraphCapture] cudaSetDevice(" << device_ordinal_
                      << ") failed before "
                      << (operation ? operation : "graph operation") << ": "
                      << cudaGetErrorString(error));
            return false;
        }
        return true;
    }

    bool CUDAGraphCapture::beginCapture()
    {
        if (!activateOwner("beginCapture"))
            return false;

        // Destroy any previous graph (but keep exec_ for tryUpdate)
        if (graph_)
        {
            const cudaError_t destroy_error = cudaGraphDestroy(graph_);
            if (destroy_error != cudaSuccess)
            {
                LOG_ERROR("[CUDAGraphCapture] Cannot begin a new capture because "
                          "destroying the prior captured graph failed: "
                          << cudaGetErrorString(destroy_error));
                return false;
            }
            graph_ = nullptr;
            node_count_ = 0;
        }

        cudaError_t err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAGraphCapture] cudaStreamBeginCapture failed: " << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool CUDAGraphCapture::endCapture()
    {
        if (!activateOwner("endCapture"))
            return false;

        cudaError_t err = cudaStreamEndCapture(stream_, &graph_);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAGraphCapture] cudaStreamEndCapture failed: " << cudaGetErrorString(err));
            graph_ = nullptr;
            return false;
        }
        if (!graph_)
        {
            LOG_ERROR("[CUDAGraphCapture] cudaStreamEndCapture produced null graph");
            return false;
        }

        // Cache node count
        size_t count = 0;
        err = cudaGraphGetNodes(graph_, nullptr, &count);
        if (err == cudaSuccess)
        {
            node_count_ = count;
        }
        LOG_DEBUG("[CUDAGraphCapture] Captured graph with " << node_count_ << " nodes");
        return true;
    }

    bool CUDAGraphCapture::instantiate()
    {
        if (!activateOwner("instantiate"))
            return false;

        if (!graph_)
        {
            LOG_ERROR("[CUDAGraphCapture] Cannot instantiate: no captured graph");
            return false;
        }
        // Destroy old executable
        if (exec_)
        {
            const cudaError_t destroy_error = cudaGraphExecDestroy(exec_);
            if (destroy_error != cudaSuccess)
            {
                LOG_ERROR("[CUDAGraphCapture] Cannot replace graph executable because "
                          "destroying the prior executable failed: "
                          << cudaGetErrorString(destroy_error));
                return false;
            }
            exec_ = nullptr;
        }

        /*
         * Graph composition errors are otherwise reported only as
         * cudaErrorInvalidValue, which loses the structural reason and the
         * offending node. Instantiation is a cold graph-publication operation,
         * so a bounded stack log has no inference-path allocation or transfer
         * cost. Keeping these outputs populated makes unsupported nodes inside
         * nested conditional bodies diagnosable from an ordinary E2E log.
         */
        cudaGraphNode_t error_node = nullptr;
        std::array<char, 8192> instantiate_log{};
        cudaError_t err = cudaGraphInstantiate(
            &exec_,
            graph_,
            &error_node,
            instantiate_log.data(),
            instantiate_log.size());
        if (err != cudaSuccess)
        {
            cudaGraphNodeType error_node_type = cudaGraphNodeTypeCount;
            const cudaError_t type_error =
                error_node
                    ? cudaGraphNodeGetType(error_node, &error_node_type)
                    : cudaErrorInvalidValue;
            LOG_ERROR("[CUDAGraphCapture] cudaGraphInstantiate failed: "
                      << cudaGetErrorString(err)
                      << " error_node=" << static_cast<const void *>(error_node)
                      << " error_node_type="
                      << (type_error == cudaSuccess
                              ? static_cast<int>(error_node_type)
                              : -1)
                      << " detail='" << instantiate_log.data() << "'");
            exec_ = nullptr;
            return false;
        }
        LOG_DEBUG("[CUDAGraphCapture] Instantiated graph executable (" << node_count_ << " nodes)");
        return true;
    }

    bool CUDAGraphCapture::launch()
    {
        return launchOnStream(static_cast<void *>(stream_));
    }

    bool CUDAGraphCapture::launchOnStream(void *stream) const
    {
        if (!activateOwner("launchOnStream"))
            return false;

        if (!exec_ || !stream)
        {
            LOG_ERROR("[CUDAGraphCapture] Cannot launch: executable or explicit stream is missing");
            return false;
        }
        cudaError_t err = cudaGraphLaunch(
            exec_,
            static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAGraphCapture] cudaGraphLaunch failed: " << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool CUDAGraphCapture::buildDeviceControlledWhileLoop(
        std::span<const DeviceControlledLoopFragment> ordered_body_fragments,
        const DeviceControlledLoopPredicate &predicate)
    {
#if CUDART_VERSION < 12030
        (void)ordered_body_fragments;
        (void)predicate;
        LOG_ERROR("[CUDAGraphCapture] Device-controlled graph loops require CUDA 12.3 or newer");
        return false;
#else
        if (!activateOwner("buildDeviceControlledWhileLoop"))
            return false;

        if (ordered_body_fragments.empty() || !predicate.valid())
        {
            LOG_ERROR("[CUDAGraphCapture] Invalid device-controlled loop contract"
                      << " fragments=" << ordered_body_fragments.size()
                      << " predicate_valid=" << predicate.valid());
            return false;
        }

        size_t transaction_node_count = 0;
        std::vector<const CUDAGraphCapture *> validated_fragments;
        validated_fragments.reserve(ordered_body_fragments.size());
        for (size_t fragment_index = 0;
             fragment_index < ordered_body_fragments.size();
             ++fragment_index)
        {
            const DeviceControlledLoopFragment &fragment =
                ordered_body_fragments[fragment_index];
            const auto *cuda_fragment =
                dynamic_cast<const CUDAGraphCapture *>(fragment.capture);
            if (!fragment.valid() || !cuda_fragment || cuda_fragment == this ||
                cuda_fragment->deviceOrdinal() != device_ordinal_ ||
                !cuda_fragment->graph() || cuda_fragment->nodeCount() == 0)
            {
                LOG_ERROR(
                    "[CUDAGraphCapture] Invalid device-loop fragment"
                    << " index=" << fragment_index
                    << " name=" << (fragment.name ? fragment.name : "<unnamed>")
                    << " execution=" << static_cast<int>(fragment.execution)
                    << " condition_word="
                    << static_cast<const void *>(fragment.condition_word_device)
                    << " backend="
                    << (fragment.capture
                            ? fragment.capture->backendName()
                            : "<null>")
                    << " nodes="
                    << (fragment.capture
                            ? fragment.capture->nodeCount()
                            : 0)
                    << " self=" << (cuda_fragment == this)
                    << " parent_device=" << device_ordinal_
                    << " fragment_device="
                    << (cuda_fragment ? cuda_fragment->deviceOrdinal() : -1));
                return false;
            }
            if (!validateCudaConditionalBodyFragmentGraph(
                    cuda_fragment->graph(),
                    fragment.name,
                    fragment_index,
                    "root"))
            {
                return false;
            }
            validated_fragments.push_back(cuda_fragment);
            transaction_node_count += cuda_fragment->nodeCount();
        }

        reset();
        cudaError_t error = cudaGraphCreate(&graph_, 0);
        if (error != cudaSuccess)
        {
            LOG_ERROR("[CUDAGraphCapture] cudaGraphCreate failed for device loop: "
                      << cudaGetErrorString(error));
            return false;
        }

        auto fail = [&](const char *operation, cudaError_t operation_error)
        {
            LOG_ERROR("[CUDAGraphCapture] " << operation
                      << " failed for device loop: "
                      << cudaGetErrorString(operation_error));
            reset();
            return false;
        };

        cudaGraphConditionalHandle condition = 0;
        error = cudaGraphConditionalHandleCreate(
            &condition,
            graph_,
            /*defaultLaunchValue=*/0,
            cudaGraphCondAssignDefault);
        if (error != cudaSuccess)
            return fail("cudaGraphConditionalHandleCreate", error);

        /*
         * A reusable parent graph can be launched after its controller became
         * terminal (for example when the capture/materialization transaction
         * consumed the final response budget). Evaluate the live device rows
         * before entering the WHILE node so terminal input executes zero body
         * iterations. A default-true conditional would append one stale extra
         * transaction on every such launch.
         */
        cudaGraphConditionalHandle condition_arg = condition;
        const int *control_rows_arg = predicate.control_rows_device;
        int control_stride_arg = predicate.control_stride;
        int request_count_arg = predicate.request_count;
        int healthy_index_arg = predicate.healthy_index;
        int complete_index_arg = predicate.complete_index;
        void *kernel_args[] = {
            &condition_arg,
            &control_rows_arg,
            &control_stride_arg,
            &request_count_arg,
            &healthy_index_arg,
            &complete_index_arg};

        cudaKernelNodeParams predicate_params{};
        predicate_params.func =
            reinterpret_cast<void *>(updateDeviceControlledLoopCondition);
        predicate_params.gridDim = dim3(1, 1, 1);
        predicate_params.blockDim = dim3(1, 1, 1);
        predicate_params.sharedMemBytes = 0;
        predicate_params.kernelParams = kernel_args;
        predicate_params.extra = nullptr;

        cudaGraphNode_t initial_predicate_node = nullptr;
        error = cudaGraphAddKernelNode(
            &initial_predicate_node,
            graph_,
            /*dependencies=*/nullptr,
            /*numDependencies=*/0,
            &predicate_params);
        if (error != cudaSuccess)
            return fail("cudaGraphAddKernelNode(initial loop predicate)", error);

        cudaGraphNodeParams conditional_params{};
        conditional_params.type = cudaGraphNodeTypeConditional;
        conditional_params.conditional.handle = condition;
        conditional_params.conditional.type = cudaGraphCondTypeWhile;
        conditional_params.conditional.size = 1;

        cudaGraphNode_t conditional_node = nullptr;
        error = cudaGraphAddNode(
            &conditional_node,
            graph_,
            &initial_predicate_node,
            /*dependencyData=*/nullptr,
            /*numDependencies=*/1,
            &conditional_params);
        if (error != cudaSuccess)
            return fail("cudaGraphAddNode(WHILE)", error);
        if (!conditional_params.conditional.phGraph_out ||
            !conditional_params.conditional.phGraph_out[0])
        {
            LOG_ERROR("[CUDAGraphCapture] CUDA did not return a WHILE body graph");
            reset();
            return false;
        }

        cudaGraph_t loop_body = conditional_params.conditional.phGraph_out[0];
        cudaGraphNode_t transaction_tail = nullptr;
        size_t conditional_fragment_count = 0;
        for (size_t fragment_index = 0;
             fragment_index < ordered_body_fragments.size();
             ++fragment_index)
        {
            const DeviceControlledLoopFragment &fragment =
                ordered_body_fragments[fragment_index];
            const CUDAGraphCapture *cuda_fragment =
                validated_fragments[fragment_index];
            const DeviceControlledFragmentAppendResult appended =
                appendDeviceControlledFragment(
                    loop_body,
                    transaction_tail,
                    fragment,
                    *cuda_fragment);
            if (!appended.succeeded())
                return fail(appended.operation, appended.error);
            transaction_tail = appended.tail;
            conditional_fragment_count +=
                fragment.execution ==
                DeviceControlledLoopFragmentExecution::IfDeviceWordNonZero
                    ? 1u
                    : 0u;
        }

        cudaGraphNode_t predicate_node = nullptr;
        error = cudaGraphAddKernelNode(
            &predicate_node,
            loop_body,
            &transaction_tail,
            /*numDependencies=*/1,
            &predicate_params);
        if (error != cudaSuccess)
            return fail("cudaGraphAddKernelNode(loop predicate)", error);

        size_t count = 0;
        error = cudaGraphGetNodes(graph_, nullptr, &count);
        if (error != cudaSuccess)
            return fail("cudaGraphGetNodes", error);
        node_count_ = count;
        LOG_DEBUG("[CUDAGraphCapture] Built device-controlled WHILE graph"
                  << " parent_nodes=" << node_count_
                  << " fragments=" << ordered_body_fragments.size()
                  << " conditional_fragments=" << conditional_fragment_count
                  << " transaction_nodes=" << transaction_node_count
                  << " requests=" << predicate.request_count);
        return true;
#endif
    }

    bool CUDAGraphCapture::buildDeviceControlledSwitchWhileLoop(
        std::span<const DeviceControlledLoopBranch> branches,
        const DeviceControlledLoopPredicate &predicate,
        const DeviceControlledLoopSwitch &switch_policy)
    {
#if CUDART_VERSION < 13000
        (void)branches;
        (void)predicate;
        (void)switch_policy;
        LOG_ERROR("[CUDAGraphCapture] Device-controlled SWITCH/WHILE graphs require CUDA 13.0 or newer");
        return false;
#else
        if (!activateOwner("buildDeviceControlledSwitchWhileLoop"))
            return false;

        const bool shared_control_authority =
            predicate.control_rows_device == switch_policy.control_rows_device &&
            predicate.control_stride == switch_policy.control_stride &&
            predicate.request_count == switch_policy.request_count &&
            predicate.healthy_index == switch_policy.healthy_index &&
            predicate.complete_index == switch_policy.complete_index;
        if (branches.empty() || !predicate.valid() ||
            !switch_policy.valid() || !shared_control_authority ||
            switch_policy.maximum_selector >=
                static_cast<int>(branches.size()))
        {
            LOG_ERROR("[CUDAGraphCapture] Invalid device-controlled SWITCH/WHILE contract"
                      << " branches=" << branches.size()
                      << " predicate_valid=" << predicate.valid()
                      << " switch_valid=" << switch_policy.valid()
                      << " shared_authority=" << shared_control_authority
                      << " selector_range=["
                      << switch_policy.minimum_selector << ','
                      << switch_policy.maximum_selector << ']');
            return false;
        }

        size_t transaction_node_count = 0;
        std::vector<std::vector<const CUDAGraphCapture *>> validated_branches(
            branches.size());
        for (size_t branch_index = 0;
             branch_index < branches.size();
             ++branch_index)
        {
            const bool implemented =
                static_cast<int>(branch_index) >=
                    switch_policy.minimum_selector &&
                static_cast<int>(branch_index) <=
                    switch_policy.maximum_selector;
            const auto fragments = branches[branch_index].ordered_fragments;
            if (implemented != !fragments.empty())
            {
                LOG_ERROR("[CUDAGraphCapture] SWITCH branch implementation does not match its declared selector interval"
                          << " branch=" << branch_index
                          << " implemented=" << implemented
                          << " fragments=" << fragments.size());
                return false;
            }

            for (size_t fragment_index = 0;
                 fragment_index < fragments.size();
                 ++fragment_index)
            {
                const DeviceControlledLoopFragment &fragment =
                    fragments[fragment_index];
                const auto *cuda_fragment =
                    dynamic_cast<const CUDAGraphCapture *>(fragment.capture);
                if (!fragment.valid() || !cuda_fragment ||
                    cuda_fragment == this ||
                    cuda_fragment->deviceOrdinal() != device_ordinal_ ||
                    !cuda_fragment->graph() ||
                    cuda_fragment->nodeCount() == 0)
                {
                    LOG_ERROR("[CUDAGraphCapture] Invalid SWITCH transaction fragment"
                              << " branch=" << branch_index
                              << " fragment=" << fragment_index
                              << " name="
                              << (fragment.name ? fragment.name : "<unnamed>")
                              << " backend="
                              << (fragment.capture
                                      ? fragment.capture->backendName()
                                      : "<null>")
                              << " nodes="
                              << (fragment.capture
                                      ? fragment.capture->nodeCount()
                                      : 0)
                              << " parent_device=" << device_ordinal_
                              << " fragment_device="
                              << (cuda_fragment
                                      ? cuda_fragment->deviceOrdinal()
                                      : -1));
                    return false;
                }
                if (!validateCudaConditionalBodyFragmentGraph(
                        cuda_fragment->graph(),
                        fragment.name,
                        fragment_index,
                        "branch[" + std::to_string(branch_index) + "]"))
                {
                    return false;
                }
                validated_branches[branch_index].push_back(cuda_fragment);
                transaction_node_count += cuda_fragment->nodeCount();
            }
        }

        reset();
        cudaError_t error = cudaGraphCreate(&graph_, 0);
        if (error != cudaSuccess)
        {
            LOG_ERROR("[CUDAGraphCapture] cudaGraphCreate failed for device SWITCH/WHILE: "
                      << cudaGetErrorString(error));
            return false;
        }

        auto fail = [&](const char *operation, cudaError_t operation_error)
        {
            LOG_ERROR("[CUDAGraphCapture] " << operation
                      << " failed for device SWITCH/WHILE: "
                      << cudaGetErrorString(operation_error));
            reset();
            return false;
        };

        cudaGraphConditionalHandle loop_condition = 0;
        error = cudaGraphConditionalHandleCreate(
            &loop_condition,
            graph_,
            /*defaultLaunchValue=*/0,
            cudaGraphCondAssignDefault);
        if (error != cudaSuccess)
            return fail("cudaGraphConditionalHandleCreate(WHILE)", error);

        cudaGraphConditionalHandle loop_condition_arg = loop_condition;
        const int *predicate_rows_arg = predicate.control_rows_device;
        int predicate_stride_arg = predicate.control_stride;
        int predicate_requests_arg = predicate.request_count;
        int predicate_healthy_arg = predicate.healthy_index;
        int predicate_complete_arg = predicate.complete_index;
        void *predicate_args[] = {
            &loop_condition_arg,
            &predicate_rows_arg,
            &predicate_stride_arg,
            &predicate_requests_arg,
            &predicate_healthy_arg,
            &predicate_complete_arg};

        cudaKernelNodeParams predicate_params{};
        predicate_params.func =
            reinterpret_cast<void *>(updateDeviceControlledLoopCondition);
        predicate_params.gridDim = dim3(1, 1, 1);
        predicate_params.blockDim = dim3(1, 1, 1);
        predicate_params.sharedMemBytes = 0;
        predicate_params.kernelParams = predicate_args;
        predicate_params.extra = nullptr;

        cudaGraphNode_t initial_predicate_node = nullptr;
        error = cudaGraphAddKernelNode(
            &initial_predicate_node,
            graph_,
            /*dependencies=*/nullptr,
            /*numDependencies=*/0,
            &predicate_params);
        if (error != cudaSuccess)
            return fail("cudaGraphAddKernelNode(initial WHILE predicate)", error);

        cudaGraphNodeParams while_params{};
        while_params.type = cudaGraphNodeTypeConditional;
        while_params.conditional.handle = loop_condition;
        while_params.conditional.type = cudaGraphCondTypeWhile;
        while_params.conditional.size = 1;

        cudaGraphNode_t while_node = nullptr;
        error = cudaGraphAddNode(
            &while_node,
            graph_,
            &initial_predicate_node,
            /*dependencyData=*/nullptr,
            /*numDependencies=*/1,
            &while_params);
        if (error != cudaSuccess)
            return fail("cudaGraphAddNode(WHILE)", error);
        if (!while_params.conditional.phGraph_out ||
            !while_params.conditional.phGraph_out[0])
        {
            LOG_ERROR("[CUDAGraphCapture] CUDA did not return a SWITCH/WHILE body graph");
            reset();
            return false;
        }

        cudaGraph_t loop_body = while_params.conditional.phGraph_out[0];
        cudaGraphConditionalHandle switch_condition = 0;
        error = cudaGraphConditionalHandleCreate(
            &switch_condition,
            loop_body,
            /*defaultLaunchValue=*/0,
            cudaGraphCondAssignDefault);
        if (error != cudaSuccess)
            return fail("cudaGraphConditionalHandleCreate(SWITCH)", error);

        cudaGraphConditionalHandle switch_condition_arg = switch_condition;
        int *switch_rows_arg = switch_policy.control_rows_device;
        int switch_stride_arg = switch_policy.control_stride;
        int switch_requests_arg = switch_policy.request_count;
        int switch_healthy_arg = switch_policy.healthy_index;
        int switch_complete_arg = switch_policy.complete_index;
        int switch_selector_arg = switch_policy.selector_index;
        int switch_error_arg = switch_policy.error_index;
        int switch_minimum_arg = switch_policy.minimum_selector;
        int switch_maximum_arg = switch_policy.maximum_selector;
        int switch_invalid_error_arg =
            switch_policy.invalid_selector_error;
        unsigned int branch_count_arg =
            static_cast<unsigned int>(branches.size());
        void *switch_args[] = {
            &switch_condition_arg,
            &switch_rows_arg,
            &switch_stride_arg,
            &switch_requests_arg,
            &switch_healthy_arg,
            &switch_complete_arg,
            &switch_selector_arg,
            &switch_error_arg,
            &switch_minimum_arg,
            &switch_maximum_arg,
            &switch_invalid_error_arg,
            &branch_count_arg};

        cudaKernelNodeParams selector_params{};
        selector_params.func =
            reinterpret_cast<void *>(updateDeviceControlledSwitchCondition);
        selector_params.gridDim = dim3(1, 1, 1);
        selector_params.blockDim = dim3(1, 1, 1);
        selector_params.sharedMemBytes = 0;
        selector_params.kernelParams = switch_args;
        selector_params.extra = nullptr;

        cudaGraphNode_t selector_node = nullptr;
        error = cudaGraphAddKernelNode(
            &selector_node,
            loop_body,
            /*dependencies=*/nullptr,
            /*numDependencies=*/0,
            &selector_params);
        if (error != cudaSuccess)
            return fail("cudaGraphAddKernelNode(SWITCH selector)", error);

        cudaGraphNodeParams switch_params{};
        switch_params.type = cudaGraphNodeTypeConditional;
        switch_params.conditional.handle = switch_condition;
        switch_params.conditional.type = cudaGraphCondTypeSwitch;
        switch_params.conditional.size =
            static_cast<unsigned int>(branches.size());

        cudaGraphNode_t switch_node = nullptr;
        error = cudaGraphAddNode(
            &switch_node,
            loop_body,
            &selector_node,
            /*dependencyData=*/nullptr,
            /*numDependencies=*/1,
            &switch_params);
        if (error != cudaSuccess)
            return fail("cudaGraphAddNode(SWITCH)", error);
        if (!switch_params.conditional.phGraph_out)
        {
            LOG_ERROR("[CUDAGraphCapture] CUDA did not return SWITCH body graphs");
            reset();
            return false;
        }

        size_t conditional_fragment_count = 0;
        for (int branch_index = switch_policy.minimum_selector;
             branch_index <= switch_policy.maximum_selector;
             ++branch_index)
        {
            cudaGraph_t branch_graph =
                switch_params.conditional.phGraph_out[branch_index];
            if (!branch_graph)
            {
                LOG_ERROR("[CUDAGraphCapture] CUDA returned a null SWITCH branch graph"
                          << " branch=" << branch_index);
                reset();
                return false;
            }

            cudaGraphNode_t branch_tail = nullptr;
            const auto branch_fragments =
                branches[static_cast<size_t>(branch_index)].ordered_fragments;
            const auto &validated_fragments =
                validated_branches[static_cast<size_t>(branch_index)];
            for (size_t fragment_index = 0;
                 fragment_index < branch_fragments.size();
                 ++fragment_index)
            {
                const DeviceControlledLoopFragment &fragment =
                    branch_fragments[fragment_index];
                const DeviceControlledFragmentAppendResult appended =
                    appendDeviceControlledFragment(
                    branch_graph,
                    branch_tail,
                    fragment,
                    *validated_fragments[fragment_index]);
                if (!appended.succeeded())
                    return fail(appended.operation, appended.error);
                branch_tail = appended.tail;
                conditional_fragment_count +=
                    fragment.execution ==
                    DeviceControlledLoopFragmentExecution::
                        IfDeviceWordNonZero
                        ? 1u
                        : 0u;
            }
        }

        cudaGraphNode_t predicate_node = nullptr;
        error = cudaGraphAddKernelNode(
            &predicate_node,
            loop_body,
            &switch_node,
            /*numDependencies=*/1,
            &predicate_params);
        if (error != cudaSuccess)
            return fail("cudaGraphAddKernelNode(loop predicate)", error);

        size_t count = 0;
        error = cudaGraphGetNodes(graph_, nullptr, &count);
        if (error != cudaSuccess)
            return fail("cudaGraphGetNodes", error);
        node_count_ = count;
        LOG_DEBUG("[CUDAGraphCapture] Built device-controlled SWITCH/WHILE graph"
                  << " parent_nodes=" << node_count_
                  << " branches=" << branches.size()
                  << " conditional_fragments=" << conditional_fragment_count
                  << " selector_range=["
                  << switch_policy.minimum_selector << ','
                  << switch_policy.maximum_selector << ']'
                  << " transaction_nodes=" << transaction_node_count
                  << " requests=" << predicate.request_count);
        return true;
#endif
    }

    GraphUpdateResult CUDAGraphCapture::tryUpdate()
    {
        if (!activateOwner("tryUpdate"))
            return GraphUpdateResult::Failed;

        if (!exec_ || !graph_)
        {
            return GraphUpdateResult::Failed;
        }

        // Use 4-arg version with cudaGraphExecUpdateResult for CUDA 10-12 compatibility
        cudaGraphExecUpdateResult update_result = cudaGraphExecUpdateError;
        cudaError_t err = cudaGraphExecUpdate(exec_, graph_, nullptr, &update_result);

        if (err == cudaSuccess && update_result == cudaGraphExecUpdateSuccess)
        {
            LOG_TRACE("[CUDAGraphCapture] Graph executable updated in-place");
            return GraphUpdateResult::Success;
        }

        if (update_result == cudaGraphExecUpdateErrorTopologyChanged ||
            update_result == cudaGraphExecUpdateErrorNodeTypeChanged ||
            update_result == cudaGraphExecUpdateErrorNotSupported)
        {
            LOG_WARN("[CUDAGraphCapture] Graph update needs reinstantiation: result="
                     << static_cast<int>(update_result));
            return GraphUpdateResult::NeedsReinstantiate;
        }

        LOG_WARN("[CUDAGraphCapture] Graph update failed: " << cudaGetErrorString(err)
                                                            << " result=" << static_cast<int>(update_result));
        return GraphUpdateResult::Failed;
    }

    bool CUDAGraphCapture::hasExecutable() const
    {
        return exec_ != nullptr;
    }

    size_t CUDAGraphCapture::nodeCount() const
    {
        return node_count_;
    }

    bool CUDAGraphCapture::inspectKernelNodes(
        std::vector<GPUGraphKernelNodeInfo> &kernel_nodes,
        std::string *error) const
    {
        kernel_nodes.clear();
        if (error)
            error->clear();
        if (!activateOwner("inspect kernel nodes"))
        {
            if (error)
                *error = "failed to activate the CUDA graph owner device";
            return false;
        }
        if (graph_ == nullptr)
        {
            if (error)
                *error = "cannot inspect CUDA kernel nodes without a captured graph";
            return false;
        }

        std::string inspection_error;
        if (!inspectCudaKernelNodesRecursive(
                reinterpret_cast<CUgraph>(graph_),
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

    void CUDAGraphCapture::reset()
    {
        if ((exec_ || graph_) && device_ordinal_ >= 0)
        {
            CUDA_WARN_IF_FAIL(cudaSetDevice(device_ordinal_));
        }
        if (exec_)
        {
            CUDA_WARN_IF_FAIL(cudaGraphExecDestroy(exec_));
            exec_ = nullptr;
        }
        if (graph_)
        {
            CUDA_WARN_IF_FAIL(cudaGraphDestroy(graph_));
            graph_ = nullptr;
        }
        node_count_ = 0;
    }

} // namespace llaminar2

#endif // HAVE_CUDA
