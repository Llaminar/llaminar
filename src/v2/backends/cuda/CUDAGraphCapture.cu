/**
 * @file CUDAGraphCapture.cu
 * @brief CUDA graph capture, replay, and device-controlled graph composition.
 *
 * CUDA conditional nodes accept a narrower set of body-node types than an
 * ordinary captured graph. Event record/wait nodes, host nodes, allocation
 * nodes, and nested conditionals cannot be cloned into transaction bodies.
 * Retained ordered timelines instead record fragments directly in their final
 * parent, preserving conditional handle identity without cloning or recapture.
 * Cloned conditional-body compositions authenticate source graphs recursively.
 * Ordering must be expressed by the graph's native dependency edges at the
 * producer; graph composition never rewrites a captured lifecycle after the
 * fact.
 * Mapped peer waits occupy a single sleeping warp rather than a batch-memory
 * scheduling channel: outstanding waits from distinct graph executables must
 * not prevent an independent controller or DMA producer from making progress.
 */

#ifdef HAVE_CUDA

#include "CUDAGraphCapture.h"
#include "../NativeParallelGraphBranch.h"
#include "../../utils/Logger.h"

#include <cuda.h>
#include <cuda/atomic>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace llaminar2
{
    /**
     * @brief One native graph definition shared by its owner and recording views.
     *
     * Conditional handles are born in this definition and never cloned. Views
     * keep storage alive but cannot instantiate it. Only the owner seals the
     * complete fragment set, then creates the sole executable.
     */
    struct CUDAGraphCapture::OrderedTimelineRecording
    {
        enum class State { Open, Recording, Sealed, Failed };
        cudaGraph_t graph = nullptr;
        int device_ordinal = -1;
        State state = State::Open;
        std::size_t fragment_count = 0u;

        /** @brief Release the definition once the last owner/view is retired. */
        ~OrderedTimelineRecording()
        {
            if (graph)
            {
                if (cudaSetDevice(device_ordinal) != cudaSuccess ||
                    cudaGraphDestroy(graph) != cudaSuccess)
                {
                    LOG_ERROR("[CUDAGraphCapture] Failed to retire ordered timeline definition");
                    std::terminate();
                }
            }
        }
    };

    namespace
    {
        /**
         * @brief Acquire a peer timeline without blocking a CUDA work channel.
         *
         * Native batch-memory waits can starve unrelated submissions when
         * distinct retained graphs expose several pending peers. A one-thread
         * kernel leaves copy engines and other SMs available to the producer.
         * The sleep limits polling traffic; the final system-acquire load
         * orders the peer's payload before dependent graph work consumes it.
         */
        __global__ void waitSystemAcquireValue64(
            std::uint64_t *signal,
            std::uint64_t value)
        {
            cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> word(*signal);
            while (word.load(cuda::memory_order_acquire) < value)
                __nanosleep(128u);
        }

        /**
         * @brief Append one single-warp system-acquire wait at an exact frontier.
         * @return CUDA validation or graph construction status.
         *
         * Active capture and explicit timeline composition use this same
         * lowering so transaction construction cannot change progress semantics.
         */
        cudaError_t addSystemWaitValue64Node(
            cudaGraphNode_t *node,
            cudaGraph_t graph,
            const cudaGraphNode_t *dependencies,
            std::size_t dependency_count,
            void *signal,
            std::uint64_t value) noexcept
        {
            if (!node || !graph || !signal || value == 0u ||
                (dependency_count > 0u && !dependencies) ||
                (reinterpret_cast<std::uintptr_t>(signal) &
                 (alignof(std::uint64_t) - 1u)) != 0u)
                return cudaErrorInvalidValue;
            auto *signal_argument = static_cast<std::uint64_t *>(signal);
            void *arguments[] = {&signal_argument, &value};
            cudaKernelNodeParams params{};
            params.func = reinterpret_cast<void *>(waitSystemAcquireValue64);
            params.gridDim = dim3(1u, 1u, 1u);
            params.blockDim = dim3(1u, 1u, 1u);
            params.kernelParams = arguments;
            return cudaGraphAddKernelNode(
                node, graph, dependencies, dependency_count, &params);
        }

        /**
         * @brief Publish mapped timeline state after flushing prior GPU writes.
         *
         * A driver batch-memory write can make its signal visible to another
         * vendor before mapped writes inside a preceding child graph have
         * reached system scope. The explicit fence and aligned volatile store
         * make the publication edge part of the retained device transaction.
         */
        __global__ void publishSystemReleaseValue64(
            std::uint64_t *signal,
            std::uint64_t value)
        {
            if (blockIdx.x == 0u && threadIdx.x == 0u)
            {
                __threadfence_system();
                *reinterpret_cast<volatile std::uint64_t *>(signal) = value;
            }
        }

        /**
         * @brief Append one system-release mapped publication kernel node.
         * @return CUDA status from validation or graph construction.
         */
        cudaError_t addSystemReleaseValue64Node(
            cudaGraphNode_t *node,
            cudaGraph_t graph,
            const cudaGraphNode_t *dependencies,
            std::size_t dependency_count,
            void *signal,
            std::uint64_t value) noexcept
        {
            if (!node || !graph || !signal || value == 0u ||
                (dependency_count > 0u && !dependencies) ||
                (reinterpret_cast<std::uintptr_t>(signal) &
                 (alignof(std::uint64_t) - 1u)) != 0u)
            {
                return cudaErrorInvalidValue;
            }
            auto *signal_argument = static_cast<std::uint64_t *>(signal);
            std::uint64_t value_argument = value;
            void *arguments[] = {&signal_argument, &value_argument};
            cudaKernelNodeParams params{};
            params.func = reinterpret_cast<void *>(publishSystemReleaseValue64);
            params.gridDim = dim3(1u, 1u, 1u);
            params.blockDim = dim3(1u, 1u, 1u);
            params.sharedMemBytes = 0u;
            params.kernelParams = arguments;
            params.extra = nullptr;
            return cudaGraphAddKernelNode(
                node,
                graph,
                dependencies,
                dependency_count,
                &params);
        }
    } // namespace

    bool appendCUDAActiveCaptureTimelineWait64(
        cudaStream_t stream,
        void *signal,
        std::uint64_t value) noexcept
    {
        if (!stream || !signal || value == 0u ||
            (reinterpret_cast<std::uintptr_t>(signal) &
             (alignof(std::uint64_t) - 1u)) != 0u)
        {
            LOG_ERROR(
                "[CUDAGraphCapture] Active timeline wait has an invalid stream, signal, or value");
            return false;
        }

        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        unsigned long long capture_id = 0u;
        cudaGraph_t graph = nullptr;
        const cudaGraphNode_t *dependencies = nullptr;
        const cudaGraphEdgeData *edge_data = nullptr;
        std::size_t dependency_count = 0u;
        cudaError_t runtime_status = cudaStreamGetCaptureInfo(
            stream,
            &capture_status,
            &capture_id,
            &graph,
            &dependencies,
            &edge_data,
            &dependency_count);
        if (runtime_status != cudaSuccess ||
            capture_status != cudaStreamCaptureStatusActive || !graph ||
            capture_id == 0u ||
            (dependency_count != 0u && !dependencies))
        {
            LOG_ERROR(
                "[CUDAGraphCapture] Active timeline wait could not resolve its capture frontier: "
                << cudaGetErrorString(runtime_status));
            return false;
        }

        cudaGraphNode_t runtime_node = nullptr;
        runtime_status = addSystemWaitValue64Node(
            &runtime_node, graph, dependencies, dependency_count, signal, value);
        if (runtime_status != cudaSuccess || !runtime_node)
        {
            LOG_ERROR(
                "[CUDAGraphCapture] addSystemWaitValue64Node(active timeline wait) failed: "
                << cudaGetErrorString(runtime_status));
            return false;
        }
        runtime_status = cudaStreamUpdateCaptureDependencies(
            stream,
            &runtime_node,
            /*dependencyData=*/nullptr,
            1u,
            cudaStreamSetCaptureDependencies);
        if (runtime_status != cudaSuccess)
        {
            LOG_ERROR(
                "[CUDAGraphCapture] cudaStreamUpdateCaptureDependencies(active timeline wait) failed: "
                << cudaGetErrorString(runtime_status));
            return false;
        }
        return true;
    }

    bool appendCUDAActiveCaptureTimelinePublish64(
        cudaStream_t stream,
        void *signal,
        std::uint64_t value) noexcept
    {
        if (!stream || !signal || value == 0u ||
            (reinterpret_cast<std::uintptr_t>(signal) &
             (alignof(std::uint64_t) - 1u)) != 0u)
        {
            LOG_ERROR(
                "[CUDAGraphCapture] Active timeline publication has an invalid stream, signal, or value");
            return false;
        }

        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        unsigned long long capture_id = 0u;
        cudaGraph_t graph = nullptr;
        const cudaGraphNode_t *dependencies = nullptr;
        const cudaGraphEdgeData *edge_data = nullptr;
        std::size_t dependency_count = 0u;
        cudaError_t status = cudaStreamGetCaptureInfo(
            stream,
            &capture_status,
            &capture_id,
            &graph,
            &dependencies,
            &edge_data,
            &dependency_count);
        if (status != cudaSuccess ||
            capture_status != cudaStreamCaptureStatusActive || !graph ||
            capture_id == 0u ||
            (dependency_count != 0u && !dependencies))
        {
            LOG_ERROR(
                "[CUDAGraphCapture] Active timeline publication could not resolve its capture frontier: "
                << cudaGetErrorString(status));
            return false;
        }

        cudaGraphNode_t node = nullptr;
        status = addSystemReleaseValue64Node(
            &node,
            graph,
            dependencies,
            dependency_count,
            signal,
            value);
        if (status != cudaSuccess || !node)
        {
            LOG_ERROR(
                "[CUDAGraphCapture] addSystemReleaseValue64Node(active timeline publication) failed: "
                << cudaGetErrorString(status));
            return false;
        }
        status = cudaStreamUpdateCaptureDependencies(
            stream,
            &node,
            /*dependencyData=*/nullptr,
            1u,
            cudaStreamSetCaptureDependencies);
        if (status != cudaSuccess)
        {
            LOG_ERROR(
                "[CUDAGraphCapture] cudaStreamUpdateCaptureDependencies(active timeline publication) failed: "
                << cudaGetErrorString(status));
            return false;
        }
        return true;
    }

#if CUDART_VERSION >= 12030
    CUDAActiveCaptureConditional::CUDAActiveCaptureConditional(
        cudaStream_t stream,
        CUDAActiveCaptureConditionalKind kind)
        : stream_(stream), kind_(kind)
    {
        if (!stream_)
        {
            (void)fail(
                "CUDA active-capture conditional requires an explicit non-default stream");
            return;
        }

        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        const cudaGraphNode_t *dependencies = nullptr;
        const cudaGraphEdgeData *edge_data = nullptr;
        std::size_t dependency_count = 0;
        cudaError_t status = cudaStreamGetCaptureInfo(
            stream_,
            &capture_status,
            &capture_id_,
            &parent_graph_,
            &dependencies,
            &edge_data,
            &dependency_count);
        if (status != cudaSuccess)
        {
            (void)fail("cudaStreamGetCaptureInfo(active conditional)", status);
            return;
        }
        if (capture_status != cudaStreamCaptureStatusActive || !parent_graph_)
        {
            (void)fail(
                "CUDA active-capture conditional was requested outside an active stream capture");
            return;
        }
        if (dependency_count > 0 && (!dependencies || !edge_data))
        {
            (void)fail(
                "CUDA active-capture dependency frontier omitted node or edge metadata");
            return;
        }

        if (dependency_count > 0)
        {
            incoming_dependencies_.assign(
                dependencies,
                dependencies + dependency_count);
            incoming_edge_data_.assign(
                edge_data,
                edge_data + dependency_count);
        }

        status = cudaGraphConditionalHandleCreate(
            &condition_,
            parent_graph_,
            /*defaultLaunchValue=*/0,
            cudaGraphCondAssignDefault);
        if (status != cudaSuccess)
        {
            (void)fail(
                "cudaGraphConditionalHandleCreate(active conditional)",
                status);
            return;
        }
        state_ = State::AwaitingPredicate;
    }

    bool CUDAActiveCaptureConditional::ready() const noexcept
    {
        return state_ == State::AwaitingPredicate && condition_ != 0 &&
               parent_graph_ != nullptr && stream_ != nullptr;
    }

    bool CUDAActiveCaptureConditional::bindPublishedPredicate()
    {
        if (!ready())
            return fail(
                "CUDA active-capture conditional predicate frontier was bound out of lifecycle order");

        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        unsigned long long current_capture_id = 0;
        cudaGraph_t current_graph = nullptr;
        const cudaGraphNode_t *dependencies = nullptr;
        const cudaGraphEdgeData *edge_data = nullptr;
        std::size_t dependency_count = 0;
        cudaError_t status = cudaStreamGetCaptureInfo(
            stream_,
            &capture_status,
            &current_capture_id,
            &current_graph,
            &dependencies,
            &edge_data,
            &dependency_count);
        if (status != cudaSuccess)
            return fail(
                "cudaStreamGetCaptureInfo(bind published predicate)",
                status);
        if (capture_status != cudaStreamCaptureStatusActive ||
            current_capture_id != capture_id_ || current_graph != parent_graph_)
        {
            return fail(
                "CUDA active-capture conditional predicate publisher lost its parent capture identity");
        }
        if (dependency_count == 0 || !dependencies || !edge_data)
            return fail(
                "CUDA active-capture conditional predicate publisher produced no dependency frontier");

        const bool frontier_unchanged =
            dependency_count == incoming_dependencies_.size() &&
            std::equal(
                dependencies,
                dependencies + dependency_count,
                incoming_dependencies_.begin());
        if (frontier_unchanged)
            return fail(
                "CUDA active-capture conditional predicate publisher enqueued no graph operation");

        state_ = State::AwaitingBranches;
        return true;
    }

    bool CUDAActiveCaptureConditional::beginBranches()
    {
        if (state_ != State::AwaitingBranches)
            return fail(
                "CUDA active-capture conditional branches were begun out of lifecycle order");

        std::vector<cudaGraphNode_t> dependencies;
        std::vector<cudaGraphEdgeData> edge_data;
        if (!queryCurrentFrontier(
                dependencies,
                edge_data,
                "begin conditional branches"))
        {
            return false;
        }
        return materializeBranches(dependencies, edge_data);
    }

    bool CUDAActiveCaptureConditional::beginBranchesWithConcurrentRootKernel(
        const cudaKernelNodeParams &params,
        const char *semantic_name)
    {
        if (state_ != State::AwaitingBranches ||
            kind_ != CUDAActiveCaptureConditionalKind::IfOnly ||
            concurrent_root_node_)
        {
            return fail(
                "CUDA concurrent-root conditional was begun out of lifecycle order or with a two-body topology");
        }
        if (!semantic_name || semantic_name[0] == '\0' || !params.func ||
            params.gridDim.x == 0 || params.gridDim.y == 0 ||
            params.gridDim.z == 0 || params.blockDim.x == 0 ||
            params.blockDim.y == 0 || params.blockDim.z == 0 ||
            (params.kernelParams && params.extra))
        {
            return fail(
                "CUDA concurrent-root conditional has invalid semantic identity or launch parameters");
        }

        std::vector<cudaGraphNode_t> dependencies;
        std::vector<cudaGraphEdgeData> edge_data;
        if (!queryCurrentFrontier(
                dependencies,
                edge_data,
                "begin concurrent-root conditional branches"))
        {
            return false;
        }

        const cudaError_t root_status = cudaGraphAddKernelNode(
            &concurrent_root_node_,
            parent_graph_,
            dependencies.data(),
            dependencies.size(),
            &params);
        if (root_status != cudaSuccess)
        {
            return fail(
                std::string("cudaGraphAddKernelNode(concurrent root '") +
                semantic_name + "'): " + cudaGetErrorString(root_status));
        }

        return materializeBranches(dependencies, edge_data);
    }

    bool CUDAActiveCaptureConditional::queryCurrentFrontier(
        std::vector<cudaGraphNode_t> &dependencies_out,
        std::vector<cudaGraphEdgeData> &edge_data_out,
        const char *operation)
    {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        unsigned long long current_capture_id = 0;
        cudaGraph_t current_graph = nullptr;
        const cudaGraphNode_t *dependencies = nullptr;
        const cudaGraphEdgeData *edge_data = nullptr;
        std::size_t dependency_count = 0;
        cudaError_t status = cudaStreamGetCaptureInfo(
            stream_,
            &capture_status,
            &current_capture_id,
            &current_graph,
            &dependencies,
            &edge_data,
            &dependency_count);
        if (status != cudaSuccess)
        {
            const std::string capture_operation =
                std::string("cudaStreamGetCaptureInfo(") +
                (operation ? operation : "query current frontier") + ")";
            return fail(capture_operation.c_str(), status);
        }
        if (capture_status != cudaStreamCaptureStatusActive ||
            current_capture_id != capture_id_ || current_graph != parent_graph_)
        {
            return fail(
                "CUDA active-capture conditional lost its parent identity before branch construction");
        }
        if (dependency_count == 0 || !dependencies || !edge_data)
            return fail(
                "CUDA active-capture conditional has no prepared branch-input frontier");

        dependencies_out.assign(
            dependencies,
            dependencies + dependency_count);
        edge_data_out.assign(edge_data, edge_data + dependency_count);
        return true;
    }

    bool CUDAActiveCaptureConditional::materializeBranches(
        const std::vector<cudaGraphNode_t> &dependencies,
        const std::vector<cudaGraphEdgeData> &edge_data)
    {
        if (state_ != State::AwaitingBranches || dependencies.empty() ||
            dependencies.size() != edge_data.size())
        {
            return fail(
                "CUDA active-capture conditional received an invalid prepared frontier");
        }
        cudaGraphNodeParams conditional_params{};
        conditional_params.type = cudaGraphNodeTypeConditional;
        conditional_params.conditional.handle = condition_;
        conditional_params.conditional.type = cudaGraphCondTypeIf;
        conditional_params.conditional.size =
            kind_ == CUDAActiveCaptureConditionalKind::IfOnly ? 1 : 2;
        const cudaError_t status = cudaGraphAddNode(
            &conditional_node_,
            parent_graph_,
            dependencies.data(),
            edge_data.data(),
            dependencies.size(),
            &conditional_params);
        if (status != cudaSuccess)
            return fail("cudaGraphAddNode(active conditional)", status);
        if (!conditional_params.conditional.phGraph_out ||
            !conditional_params.conditional.phGraph_out[0] ||
            (kind_ == CUDAActiveCaptureConditionalKind::IfElse &&
             !conditional_params.conditional.phGraph_out[1]))
        {
            return fail(
                "CUDA did not materialize the declared active-capture conditional bodies");
        }

        branch_graphs_[branchIndex(
            CUDAActiveCaptureConditionalBranch::IfNonZero)] =
            conditional_params.conditional.phGraph_out[0];
        if (kind_ == CUDAActiveCaptureConditionalKind::IfElse)
        {
            branch_graphs_[branchIndex(
                CUDAActiveCaptureConditionalBranch::ElseZero)] =
                conditional_params.conditional.phGraph_out[1];
        }
        state_ = State::BuildingBranches;
        return true;
    }

    bool CUDAActiveCaptureConditional::appendKernel(
        CUDAActiveCaptureConditionalBranch branch,
        const cudaKernelNodeParams &params,
        const char *semantic_name)
    {
        const std::size_t index = branchIndex(branch);
        if (state_ != State::BuildingBranches || index >= branch_graphs_.size() ||
            (branch == CUDAActiveCaptureConditionalBranch::ElseZero &&
             kind_ == CUDAActiveCaptureConditionalKind::IfOnly))
            return fail(
                "CUDA active-capture conditional branch kernel was appended out of lifecycle order");
        if (!semantic_name || semantic_name[0] == '\0' || !branch_graphs_[index] ||
            !params.func || params.gridDim.x == 0 || params.gridDim.y == 0 ||
            params.gridDim.z == 0 || params.blockDim.x == 0 ||
            params.blockDim.y == 0 || params.blockDim.z == 0 ||
            (params.kernelParams && params.extra))
        {
            return fail(
                "CUDA active-capture conditional branch has invalid semantic identity or launch parameters");
        }

        cudaGraphNode_t node = nullptr;
        const cudaGraphNode_t dependency = branch_tails_[index];
        const cudaError_t status = cudaGraphAddKernelNode(
            &node,
            branch_graphs_[index],
            dependency ? &dependency : nullptr,
            dependency ? 1 : 0,
            &params);
        if (status != cudaSuccess)
        {
            return fail(
                std::string("cudaGraphAddKernelNode(active branch '") +
                semantic_name + "'): " + cudaGetErrorString(status));
        }
        branch_tails_[index] = node;
        ++branch_node_counts_[index];
        return true;
    }

    bool CUDAActiveCaptureConditional::commit()
    {
        if (state_ != State::BuildingBranches || !conditional_node_)
            return fail(
                "CUDA active-capture conditional was committed out of lifecycle order");
        if (branch_node_counts_[0] == 0 ||
            (kind_ == CUDAActiveCaptureConditionalKind::IfElse &&
             branch_node_counts_[1] == 0))
            return fail(
                "CUDA active-capture conditional requires every declared body to be non-empty");

        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        unsigned long long current_capture_id = 0;
        cudaGraph_t current_graph = nullptr;
        cudaError_t status = cudaStreamGetCaptureInfo(
            stream_,
            &capture_status,
            &current_capture_id,
            &current_graph,
            /*dependencies_out=*/nullptr,
            /*edgeData_out=*/nullptr,
            /*numDependencies_out=*/nullptr);
        if (status != cudaSuccess)
            return fail("cudaStreamGetCaptureInfo(commit conditional)", status);
        if (capture_status != cudaStreamCaptureStatusActive ||
            current_capture_id != capture_id_ || current_graph != parent_graph_)
        {
            return fail(
                "CUDA active-capture conditional lost its parent capture identity before commit");
        }

        std::array<cudaGraphNode_t, 2> new_tails{
            conditional_node_,
            concurrent_root_node_,
        };
        const std::size_t new_tail_count =
            concurrent_root_node_ ? 2 : 1;
        status = cudaStreamUpdateCaptureDependencies(
            stream_,
            new_tails.data(),
            /*dependencyData=*/nullptr,
            new_tail_count,
            cudaStreamSetCaptureDependencies);
        if (status != cudaSuccess)
        {
            return fail(
                "cudaStreamUpdateCaptureDependencies(commit conditional)",
                status);
        }
        state_ = State::Committed;
        return true;
    }

    bool CUDAActiveCaptureConditional::fail(
        const char *operation,
        cudaError_t status)
    {
        return fail(
            std::string(operation ? operation : "unknown CUDA operation") +
            ": " + cudaGetErrorString(status));
    }

    bool CUDAActiveCaptureConditional::fail(const std::string &message)
    {
        if (error_.empty())
        {
            error_ = message;
            LOG_ERROR("[CUDAActiveCaptureConditional] " << error_);
        }
        state_ = State::Failed;
        return false;
    }
#endif

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
         * @param selected_nodes Optional fragment-local roots/nodes; nested child
         *        graphs are still traversed recursively. Null inspects the whole graph.
         * @return true only when every reachable node was inspected.
         */
        bool inspectCudaKernelNodesRecursive(
            CUgraph graph,
            const std::string &graph_path,
            size_t nesting_depth,
            std::vector<GPUGraphKernelNodeInfo> &kernel_nodes,
            std::string &error,
            const std::vector<cudaGraphNode_t> *selected_nodes = nullptr)
        {
            size_t node_count = 0;
            CUresult status = selected_nodes ? CUDA_SUCCESS :
                cuGraphGetNodes(graph, nullptr, &node_count);
            if (status != CUDA_SUCCESS)
            {
                error = "cuGraphGetNodes(count) failed at " + graph_path +
                        ": " + cudaDriverErrorString(status);
                return false;
            }

            std::vector<CUgraphNode> nodes(node_count);
            if (selected_nodes)
                for (const auto node : *selected_nodes)
                    nodes.push_back(reinterpret_cast<CUgraphNode>(node));
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

    /**
     * @brief Publish whether a validated transaction selector reaches one threshold.
     *
     * The full definition follows the common fragment appender. Declaring the
     * exact kernel signature here lets the appender reject selector predicates;
     * the selector-aware composer groups adjacent fragments behind one launch.
     */
    __global__ void updateDeviceControlledSelectorThresholdCondition(
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
        int fragment_threshold);

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

            if (fragment.execution ==
                DeviceControlledLoopFragmentExecution::
                    IfDeviceSelectorAtLeast)
            {
                result.error = cudaErrorInvalidValue;
                result.operation =
                    "selector-gated fragment requires selector-aware composer";
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

#if CUDART_VERSION >= 12030
    namespace
    {
        /**
         * @brief Validate one selector batch and poison every live row on error.
         *
         * Both the transaction-entry validator and every monotonic depth gate
         * use this one arithmetic authority. Revalidation is deliberate: if a
         * prefix fragment illegally mutates the selector, the next gate turns
         * that lifecycle violation into the same stable terminal error instead
         * of silently changing transaction width midway through an iteration.
         *
         * @return The agreed legal selector, or -1 after publishing failure.
         */
        __device__ int validateDeviceControlledSelector(
            int *control_rows,
            int control_stride,
            int request_count,
            int healthy_index,
            int complete_index,
            int selector_index,
            int error_index,
            int minimum_selector,
            int maximum_selector,
            int invalid_selector_error)
        {
            int selected = -1;
            bool invalid = false;
            for (int request = 0; request < request_count; ++request)
            {
                int *row = control_rows +
                    static_cast<size_t>(request) * control_stride;
                if (row[healthy_index] == 0 || row[complete_index] != 0)
                    continue;

                const int candidate = row[selector_index];
                if (candidate < minimum_selector ||
                    candidate > maximum_selector ||
                    (selected >= 0 && selected != candidate))
                {
                    invalid = true;
                    break;
                }
                selected = candidate;
            }

            if (selected < 0)
                invalid = true;
            if (!invalid)
                return selected;

            for (int request = 0; request < request_count; ++request)
            {
                int *row = control_rows +
                    static_cast<size_t>(request) * control_stride;
                if (row[complete_index] == 0)
                {
                    row[healthy_index] = 0;
                    row[complete_index] = 1;
                    row[error_index] = invalid_selector_error;
                }
            }
            return -1;
        }
    }

    /**
     * @brief Admit one complete linear transaction only for a legal selector.
     *
     * Failure marks the controller terminal before CUDA enters the transaction
     * IF body, so neither the unconditional prefix nor the shared tail can run.
     */
    __global__ void updateDeviceControlledSelectorValidityCondition(
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
        int invalid_selector_error)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0)
            return;
        const int selected = validateDeviceControlledSelector(
            control_rows,
            control_stride,
            request_count,
            healthy_index,
            complete_index,
            selector_index,
            error_index,
            minimum_selector,
            maximum_selector,
            invalid_selector_error);
        cudaGraphSetConditional(handle, selected >= 0 ? 1u : 0u);
    }

    /**
     * @brief Publish whether the current legal selector reaches one width gate.
     *
     * Adjacent fragments with the same threshold share this predicate and one
     * native IF body. The selector is revalidated before the group so an
     * accidental prefix mutation fails closed without executing a partial group.
     */
    __global__ void updateDeviceControlledSelectorThresholdCondition(
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
        int fragment_threshold)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0)
            return;
        const int selected = validateDeviceControlledSelector(
            control_rows,
            control_stride,
            request_count,
            healthy_index,
            complete_index,
            selector_index,
            error_index,
            minimum_selector,
            maximum_selector,
            invalid_selector_error);
        cudaGraphSetConditional(
            handle,
            selected >= fragment_threshold ? 1u : 0u);
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
          graph_(other.graph_), exec_(other.exec_), node_count_(other.node_count_),
          resident_memory_bytes_(other.resident_memory_bytes_),
          ordered_timeline_timing_events_(
              std::move(other.ordered_timeline_timing_events_)),
          ordered_timeline_timing_pending_(
              other.ordered_timeline_timing_pending_)
    {
        timeline_recording_ = std::move(other.timeline_recording_);
        recording_role_ = other.recording_role_;
        fragment_state_ = other.fragment_state_;
        fragment_entry_ = other.fragment_entry_;
        fragment_exit_ = other.fragment_exit_;
        fragment_nodes_ = std::move(other.fragment_nodes_);
        other.stream_ = nullptr;
        other.device_ordinal_ = -1;
        other.graph_ = nullptr;
        other.exec_ = nullptr;
        other.node_count_ = 0;
        other.resident_memory_bytes_ = 0u;
        other.ordered_timeline_timing_pending_ = false;
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
            resident_memory_bytes_ = other.resident_memory_bytes_;
            ordered_timeline_timing_events_ =
                std::move(other.ordered_timeline_timing_events_);
            ordered_timeline_timing_pending_ =
                other.ordered_timeline_timing_pending_;
            timeline_recording_ = std::move(other.timeline_recording_);
            recording_role_ = other.recording_role_;
            fragment_state_ = other.fragment_state_;
            fragment_entry_ = other.fragment_entry_;
            fragment_exit_ = other.fragment_exit_;
            fragment_nodes_ = std::move(other.fragment_nodes_);
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
        if (recording_role_ == RecordingRole::Fragment && !timeline_recording_)
            return false;

        if (timeline_recording_)
        {
            if (recording_role_ != RecordingRole::Fragment ||
                fragment_state_ != FragmentState::Unrecorded ||
                timeline_recording_->state != OrderedTimelineRecording::State::Open)
                return false;
            // Each unit starts as an independent branch in the final owner.
            // Sealing later joins these exact frontiers in declared order; no
            // node or conditional handle is copied or submitted a second time.
            auto error = cudaGraphAddEmptyNode(&fragment_entry_, graph_, nullptr, 0u);
            if (error == cudaSuccess)
                error = cudaStreamBeginCaptureToGraph(
                    stream_, graph_, &fragment_entry_, nullptr, 1u,
                    cudaStreamCaptureModeRelaxed);
            if (error != cudaSuccess)
            {
                timeline_recording_->state = OrderedTimelineRecording::State::Failed;
                LOG_ERROR("[CUDAGraphCapture] Cannot record into ordered timeline: "
                          << cudaGetErrorString(error));
                return false;
            }
            fragment_state_ = FragmentState::Recording;
            timeline_recording_->state = OrderedTimelineRecording::State::Recording;
            return true;
        }

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
            destroyOrderedTimelineTimingEvents();
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
        if (recording_role_ == RecordingRole::Fragment && !timeline_recording_)
            return false;

        if (timeline_recording_)
        {
            if (recording_role_ != RecordingRole::Fragment ||
                fragment_state_ != FragmentState::Recording ||
                timeline_recording_->state != OrderedTimelineRecording::State::Recording)
                return false;
            cudaStreamCaptureStatus capture_status{};
            cudaGraph_t captured = nullptr;
            const cudaGraphNode_t *frontier = nullptr;
            size_t frontier_count = 0u;
            auto error = cudaStreamGetCaptureInfo(
                stream_, &capture_status, nullptr, &captured, &frontier, nullptr, &frontier_count);
            if (error == cudaSuccess && captured == graph_ && frontier_count != 0u)
                error = cudaGraphAddEmptyNode(
                    &fragment_exit_, graph_, frontier, frontier_count);
            else if (error == cudaSuccess)
                error = cudaErrorInvalidValue;
            const auto end_error = cudaStreamEndCapture(stream_, &captured);
            if (error == cudaSuccess)
                error = end_error;
            if (error != cudaSuccess || captured != graph_)
            {
                timeline_recording_->state = OrderedTimelineRecording::State::Failed;
                LOG_ERROR("[CUDAGraphCapture] Cannot seal ordered timeline fragment: "
                          << cudaGetErrorString(error));
                return false;
            }

            // Walk only this entry's descendants, not the growing whole graph.
            // This keeps construction O(total nodes) across many model layers.
            fragment_nodes_ = {fragment_entry_};
            std::unordered_set<cudaGraphNode_t> visited{fragment_entry_};
            for (size_t index = 0u; index < fragment_nodes_.size(); ++index)
            {
                size_t count = 0u;
                error = cudaGraphNodeGetDependentNodes(fragment_nodes_[index], nullptr, nullptr, &count);
                std::vector<cudaGraphNode_t> children(count);
                if (error == cudaSuccess && count != 0u)
                    error = cudaGraphNodeGetDependentNodes(
                        fragment_nodes_[index], children.data(), nullptr, &count);
                if (error != cudaSuccess)
                {
                    timeline_recording_->state = OrderedTimelineRecording::State::Failed;
                    return false;
                }
                for (const auto child : children)
                    if (visited.insert(child).second)
                        fragment_nodes_.push_back(child);
            }
            node_count_ = fragment_nodes_.size();
            fragment_state_ = FragmentState::Recorded;
            timeline_recording_->state = OrderedTimelineRecording::State::Open;
            return true;
        }

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
        if (recording_role_ == RecordingRole::Fragment)
            return false;
        if (timeline_recording_ &&
            (recording_role_ != RecordingRole::Owner ||
             timeline_recording_->state != OrderedTimelineRecording::State::Sealed))
            return false;
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
            resident_memory_bytes_ = 0u;
        }

        /*
         * CUDA normally assigns every graph kernel the launch stream's
         * priority, discarding the priorities of streams that participated in
         * capture. That would make a latency-critical transfer-progress branch
         * sit behind the main inference wave again. Preserve each kernel node's
         * captured stream priority for every production executable; ordinary
         * single-stream graphs retain exactly their prior scheduling semantics.
         *
         * `cudaGraphInstantiateWithParams` currently accepts this flag on CUDA
         * 13 but publishes an executable whose queried flags omit it. Use the
         * dedicated flags entrypoint that the latency proof exercises. Older
         * headers retain the bounded diagnostic buffer because they cannot
         * express node-priority instantiation.
         */
        cudaGraphNode_t error_node = nullptr;
        std::array<char, 8192> instantiate_log{};
        std::size_t free_bytes_before = 0u;
        std::size_t total_bytes_before = 0u;
        const cudaError_t memory_before_status = cudaMemGetInfo(
            &free_bytes_before,
            &total_bytes_before);
        if (memory_before_status != cudaSuccess)
        {
            LOG_ERROR(
                "[CUDAGraphCapture] Cannot account graph-executable VRAM before instantiation: "
                << cudaGetErrorString(memory_before_status));
            return false;
        }
#if CUDART_VERSION >= 11040
        cudaError_t err = cudaGraphInstantiateWithFlags(
            &exec_, graph_, cudaGraphInstantiateFlagUseNodePriority);
#else
        cudaError_t err = cudaGraphInstantiate(
            &exec_,
            graph_,
            &error_node,
            instantiate_log.data(),
            instantiate_log.size());
#endif
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
        std::size_t free_bytes_after = 0u;
        std::size_t total_bytes_after = 0u;
        const cudaError_t memory_after_status = cudaMemGetInfo(
            &free_bytes_after,
            &total_bytes_after);
        if (memory_after_status != cudaSuccess ||
            total_bytes_after != total_bytes_before)
        {
            LOG_ERROR(
                "[CUDAGraphCapture] Cannot account graph-executable VRAM after instantiation: status="
                << cudaGetErrorString(memory_after_status)
                << " total_before=" << total_bytes_before
                << " total_after=" << total_bytes_after);
            const cudaError_t destroy_error = cudaGraphExecDestroy(exec_);
            if (destroy_error != cudaSuccess)
            {
                LOG_ERROR(
                    "[CUDAGraphCapture] Failed to destroy an executable after its VRAM accounting failed: "
                    << cudaGetErrorString(destroy_error));
            }
            exec_ = nullptr;
            return false;
        }
        /*
         * Native graph storage is opaque to Llaminar's allocators. Measuring
         * the setup-only free-memory delta gives complete-family certification
         * a concrete observation without adding a query to replay. CUDA owns a
         * shared graph pool: this delta is pool growth triggered by the current
         * executable, not memory attributable to that executable alone. The
         * PhysicalMemoryAuthority therefore commits the admitted family extent;
         * comparing this value with one per-slot unit would falsely reject the
         * first graph that grows storage later consumed by its siblings. An
         * apparent free-memory increase is reported as zero because the driver
         * may retire unrelated deferred state at this exact setup boundary.
         */
        const std::size_t resident_delta_bytes =
            free_bytes_before > free_bytes_after
                ? free_bytes_before - free_bytes_after
                : 0u;
        resident_memory_bytes_ = resident_delta_bytes;
        LOG_DEBUG(
            "[CUDAGraphCapture] Instantiated graph executable ("
            << node_count_
            << " nodes, captured node priorities requested, resident_delta_bytes="
            << resident_delta_bytes
            << ", free_bytes_after=" << free_bytes_after << ")");
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
        if (!ordered_timeline_timing_events_.empty())
            ordered_timeline_timing_pending_ = true;
        return true;
    }

    bool CUDAGraphCapture::buildDeviceControlledTransaction(
        std::span<const DeviceControlledLoopFragment> ordered_fragments)
    {
        if (timeline_recording_ || recording_role_ != RecordingRole::Owner)
            return false;
#if CUDART_VERSION < 12030
        (void)ordered_fragments;
        LOG_ERROR("[CUDAGraphCapture] Device-controlled graph transactions require CUDA 12.3 or newer");
        return false;
#else
        if (!activateOwner("buildDeviceControlledTransaction"))
            return false;
        if (ordered_fragments.empty())
        {
            LOG_ERROR("[CUDAGraphCapture] A device-controlled transaction requires at least one fragment");
            return false;
        }

        size_t transaction_node_count = 0;
        std::vector<const CUDAGraphCapture *> validated_fragments;
        validated_fragments.reserve(ordered_fragments.size());
        for (size_t fragment_index = 0;
             fragment_index < ordered_fragments.size();
             ++fragment_index)
        {
            const DeviceControlledLoopFragment &fragment =
                ordered_fragments[fragment_index];
            const auto *cuda_fragment =
                dynamic_cast<const CUDAGraphCapture *>(fragment.capture);
            if (!fragment.valid() || !cuda_fragment || cuda_fragment == this ||
                cuda_fragment->deviceOrdinal() != device_ordinal_ ||
                !cuda_fragment->graph() || cuda_fragment->nodeCount() == 0)
            {
                LOG_ERROR(
                    "[CUDAGraphCapture] Invalid one-shot transaction fragment"
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
            LOG_ERROR("[CUDAGraphCapture] cudaGraphCreate failed for one-shot transaction: "
                      << cudaGetErrorString(error));
            return false;
        }

        auto fail = [&](const char *operation, cudaError_t operation_error)
        {
            LOG_ERROR("[CUDAGraphCapture] " << operation
                      << " failed for one-shot transaction: "
                      << cudaGetErrorString(operation_error));
            reset();
            return false;
        };

        cudaGraphNode_t transaction_tail = nullptr;
        size_t conditional_fragment_count = 0;
        for (size_t fragment_index = 0;
             fragment_index < ordered_fragments.size();
             ++fragment_index)
        {
            const DeviceControlledLoopFragment &fragment =
                ordered_fragments[fragment_index];
            const DeviceControlledFragmentAppendResult appended =
                appendDeviceControlledFragment(
                    graph_,
                    transaction_tail,
                    fragment,
                    *validated_fragments[fragment_index]);
            if (!appended.succeeded())
                return fail(appended.operation, appended.error);
            transaction_tail = appended.tail;
            conditional_fragment_count +=
                fragment.execution ==
                        DeviceControlledLoopFragmentExecution::
                            IfDeviceWordNonZero
                    ? 1u
                    : 0u;
        }

        size_t count = 0;
        error = cudaGraphGetNodes(graph_, nullptr, &count);
        if (error != cudaSuccess)
            return fail("cudaGraphGetNodes", error);
        node_count_ = count;
        LOG_DEBUG("[CUDAGraphCapture] Built one-shot device-controlled transaction"
                  << " parent_nodes=" << node_count_
                  << " fragments=" << ordered_fragments.size()
                  << " conditional_fragments=" << conditional_fragment_count
                  << " transaction_nodes=" << transaction_node_count);
        return transaction_tail != nullptr;
#endif
    }

    std::unique_ptr<IGPUGraphCapture> CUDAGraphCapture::createOrderedTimelineFragment()
    {
        if (!activateOwner("createOrderedTimelineFragment") || exec_ ||
            recording_role_ != RecordingRole::Owner || node_count_ != 0u ||
            (timeline_recording_ &&
             timeline_recording_->state != OrderedTimelineRecording::State::Open))
            return nullptr;
        if (!timeline_recording_)
        {
            if (graph_)
                return nullptr;
            auto recording = std::make_shared<OrderedTimelineRecording>();
            recording->device_ordinal = device_ordinal_;
            if (cudaGraphCreate(&recording->graph, 0u) != cudaSuccess)
                return nullptr;
            timeline_recording_ = std::move(recording);
            graph_ = timeline_recording_->graph;
        }
        auto fragment = std::make_unique<CUDAGraphCapture>(stream_, device_ordinal_);
        fragment->timeline_recording_ = timeline_recording_;
        fragment->recording_role_ = RecordingRole::Fragment;
        fragment->graph_ = graph_;
        ++timeline_recording_->fragment_count;
        return fragment;
    }

    bool CUDAGraphCapture::buildOrderedTimelineTransaction(
        std::span<const GPUOrderedTimelineStep> ordered_steps,
        GPUOrderedTimelineInstrumentation instrumentation)
    {
        if (recording_role_ != RecordingRole::Owner ||
            !activateOwner("buildOrderedTimelineTransaction") ||
            ordered_steps.empty())
        {
            LOG_ERROR("[CUDAGraphCapture] Ordered timeline transaction requires an exact owner and non-empty steps");
            return false;
        }

        if (timeline_recording_ &&
            (recording_role_ != RecordingRole::Owner ||
             timeline_recording_->state != OrderedTimelineRecording::State::Open))
            return false;
        std::unordered_set<const CUDAGraphCapture *> recorded_fragments;
        std::vector<const CUDAGraphCapture *> fragments(
            ordered_steps.size(), nullptr);
        for (std::size_t index = 0u; index < ordered_steps.size(); ++index)
        {
            const auto &step = ordered_steps[index];
            if (!step.valid())
            {
                LOG_ERROR("[CUDAGraphCapture] Invalid ordered timeline step index="
                          << index);
                return false;
            }
            if (step.kind !=
                GPUOrderedTimelineStepKind::CapturedFragment)
            {
                continue;
            }
            const auto *fragment =
                dynamic_cast<const CUDAGraphCapture *>(step.capture);
            if (!fragment || fragment == this ||
                fragment->deviceOrdinal() != device_ordinal_ ||
                !fragment->graph() || fragment->nodeCount() == 0u)
            {
                LOG_ERROR("[CUDAGraphCapture] Ordered timeline fragment has incompatible ownership"
                          << " index=" << index
                          << " name=" << step.name);
                return false;
            }
            fragments[index] = fragment;
            if (fragment->timeline_recording_ != timeline_recording_)
                return false;
            if (timeline_recording_ &&
                (fragment->timeline_recording_ != timeline_recording_ ||
                 fragment->recording_role_ != RecordingRole::Fragment ||
                 fragment->fragment_state_ != FragmentState::Recorded ||
                 !recorded_fragments.insert(fragment).second))
                return false;
        }

        if (timeline_recording_ &&
            recorded_fragments.size() != timeline_recording_->fragment_count)
            return false;
        if (!timeline_recording_)
            reset();
        cudaError_t runtime_error = timeline_recording_ ? cudaSuccess :
            cudaGraphCreate(&graph_, 0u);
        if (runtime_error != cudaSuccess)
        {
            LOG_ERROR("[CUDAGraphCapture] cudaGraphCreate failed for ordered timeline transaction: "
                      << cudaGetErrorString(runtime_error));
            return false;
        }
        auto failRuntime = [&](const char *operation, cudaError_t error)
        {
            LOG_ERROR("[CUDAGraphCapture] " << operation
                      << " failed for ordered timeline transaction: "
                      << cudaGetErrorString(error));
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
                runtime_error = cudaEventCreate(&timing.start);
                if (runtime_error != cudaSuccess)
                    return failRuntime("cudaEventCreate(timeline start)", runtime_error);
                runtime_error = cudaEventCreate(&timing.stop);
                if (runtime_error != cudaSuccess)
                {
                    (void)cudaEventDestroy(timing.start);
                    return failRuntime("cudaEventCreate(timeline stop)", runtime_error);
                }
                ordered_timeline_timing_events_.push_back(
                    std::move(timing));
            }
        }

        cudaGraphNode_t tail = nullptr;
        for (std::size_t index = 0u; index < ordered_steps.size(); ++index)
        {
            const auto &step = ordered_steps[index];
            if (!ordered_timeline_timing_events_.empty())
            {
                cudaGraphNode_t timing_start = nullptr;
                runtime_error = cudaGraphAddEventRecordNode(
                    &timing_start,
                    graph_,
                    tail ? &tail : nullptr,
                    tail ? 1u : 0u,
                    ordered_timeline_timing_events_[index].start);
                if (runtime_error != cudaSuccess)
                    return failRuntime(
                        "cudaGraphAddEventRecordNode(timeline start)",
                        runtime_error);
                tail = timing_start;
            }
            if (step.kind ==
                GPUOrderedTimelineStepKind::CapturedFragment)
            {
                if (timeline_recording_)
                {
                    const auto *fragment = fragments[index];
                    if (tail)
                        runtime_error = cudaGraphAddDependencies(
                            graph_, &tail, &fragment->fragment_entry_, nullptr, 1u);
                    if (runtime_error != cudaSuccess)
                        return failRuntime("cudaGraphAddDependencies(fragment entry)", runtime_error);
                    tail = fragment->fragment_exit_;
                }
                else
                {
                cudaGraphNode_t node = nullptr;
                runtime_error = cudaGraphAddChildGraphNode(
                    &node,
                    graph_,
                    tail ? &tail : nullptr,
                    tail ? 1u : 0u,
                    fragments[index]->graph());
                if (runtime_error != cudaSuccess)
                {
                    // A fragment name ties a native composition error back to
                    // its declarative stage. Inventory only on failure: graph
                    // diagnostics must not add work to successful replay.
                    LOG_ERROR("[CUDAGraphCapture] Rejected ordered timeline child"
                              << " index=" << index << " name=" << step.name
                              << " nodes=" << fragments[index]->nodeCount());
                    size_t count = 0u;
                    if (cudaGraphGetNodes(fragments[index]->graph(), nullptr,
                                          &count) == cudaSuccess)
                    {
                        std::vector<cudaGraphNode_t> nodes(count);
                        if (cudaGraphGetNodes(fragments[index]->graph(),
                                              nodes.data(), &count) == cudaSuccess)
                        {
                            std::array<size_t, cudaGraphNodeTypeCount> census{};
                            for (const auto source_node : nodes)
                            {
                                cudaGraphNodeType type = cudaGraphNodeTypeCount;
                                if (cudaGraphNodeGetType(source_node, &type) ==
                                        cudaSuccess && type < cudaGraphNodeTypeCount)
                                    ++census[type];
                            }
                            for (size_t type = 0u; type < census.size(); ++type)
                                if (census[type] != 0u)
                                    LOG_ERROR("[CUDAGraphCapture] Rejected child node inventory"
                                              << " name=" << step.name
                                              << " type=" << cudaGraphNodeTypeName(
                                                     static_cast<cudaGraphNodeType>(type))
                                              << " count=" << census[type]);
                        }
                    }
                    return failRuntime(
                        "cudaGraphAddChildGraphNode", runtime_error);
                }
                tail = node;
                }
            }
            else
            {
                if (step.kind == GPUOrderedTimelineStepKind::WaitValue64)
                {
                    cudaGraphNode_t node = nullptr;
                    runtime_error = addSystemWaitValue64Node(
                        &node, graph_, tail ? &tail : nullptr,
                        tail ? 1u : 0u, step.signal, step.value);
                    if (runtime_error != cudaSuccess)
                        return failRuntime("addSystemWaitValue64Node", runtime_error);
                    tail = node;
                }
                else
                {
                    cudaGraphNode_t node = nullptr;
                    runtime_error = addSystemReleaseValue64Node(
                        &node,
                        graph_,
                        tail ? &tail : nullptr,
                        tail ? 1u : 0u,
                        step.signal,
                        step.value);
                    if (runtime_error != cudaSuccess)
                    {
                        return failRuntime(
                            "addSystemReleaseValue64Node", runtime_error);
                    }
                    tail = node;
                }
            }

            if (!ordered_timeline_timing_events_.empty())
            {
                cudaGraphNode_t timing_stop = nullptr;
                runtime_error = cudaGraphAddEventRecordNode(
                    &timing_stop,
                    graph_,
                    &tail,
                    1u,
                    ordered_timeline_timing_events_[index].stop);
                if (runtime_error != cudaSuccess)
                    return failRuntime(
                        "cudaGraphAddEventRecordNode(timeline stop)",
                        runtime_error);
                tail = timing_stop;
            }
        }

        std::size_t count = 0u;
        runtime_error = cudaGraphGetNodes(graph_, nullptr, &count);
        if (runtime_error != cudaSuccess)
            return failRuntime("cudaGraphGetNodes", runtime_error);
        node_count_ = count;
        if (timeline_recording_)
        {
            timeline_recording_->state = OrderedTimelineRecording::State::Sealed;
            return tail != nullptr && node_count_ != 0u;
        }
        const std::size_t expected_node_count =
            ordered_steps.size() *
            (ordered_timeline_timing_events_.empty() ? 1u : 3u);
        return tail != nullptr && node_count_ == expected_node_count;
    }

    namespace
    {
        /** @brief Exact CUDA operations for the shared cold branch-DAG builder. */
        struct CUDAParallelGraphAPI
        {
            using Graph = cudaGraph_t;
            using Node = cudaGraphNode_t;
            using Error = cudaError_t;
            static constexpr Error success = cudaSuccess;
            static constexpr Error invalid = cudaErrorInvalidValue;
            /** @brief Query the owner's nodes without executing device work. */
            static Error nodes(Graph g, Node *n, std::size_t *s) { return cudaGraphGetNodes(g, n, s); }
            /** @brief Query a node's complete incoming frontier. */
            static Error dependencies(Node n, Node *d, std::size_t *s) { return cudaGraphNodeGetDependencies(n, d, nullptr, s); }
            /** @brief Query a node's complete outgoing frontier. */
            static Error dependents(Node n, Node *d, std::size_t *s) { return cudaGraphNodeGetDependentNodes(n, d, nullptr, s); }
            /** @brief Clone only a small branch fragment, never the original body. */
            static Error child(Node *n, Graph g, const Node *d, std::size_t s, Graph c) { return cudaGraphAddChildGraphNode(n, g, d, s, c); }
            /** @brief Bind an explicit native producer/consumer edge set. */
            static Error edges(Graph g, const Node *f, const Node *t, std::size_t s) { return cudaGraphAddDependencies(g, f, t, nullptr, s); }
            /** @brief Join the bounded worker and the inference-owned Close. */
            static Error empty(Node *n, Graph g, const Node *d, std::size_t s) { return cudaGraphAddEmptyNode(n, g, d, s); }
        };
    }

    bool CUDAGraphCapture::appendParallelBranch(
        const GPUCapturedParallelBranch &branch)
    {
        if (!activateOwner("appendParallelBranch") || !graph_ || exec_ ||
            node_count_ == 0u || recording_role_ != RecordingRole::Owner)
        {
            LOG_ERROR("[CUDAGraphCapture] Parallel decoration requires a sealed uninstantiated graph owner");
            return false;
        }
        cudaStreamCaptureStatus status{};
        if (cudaStreamIsCapturing(stream_, &status) != cudaSuccess ||
            status != cudaStreamCaptureStatusNone)
        {
            LOG_ERROR("[CUDAGraphCapture] Parallel decoration cannot mutate an active recording");
            return false;
        }
        const IGPUGraphCapture *sources[] = {&branch.open, &branch.worker, &branch.close};
        std::array<cudaGraph_t, 3> fragments{};
        for (std::size_t i = 0u; i < fragments.size(); ++i)
        {
            const auto *source = dynamic_cast<const CUDAGraphCapture *>(sources[i]);
            if (!source || source == this || source->deviceOrdinal() != device_ordinal_ ||
                !source->graph() || source->graph() == graph_ || source->nodeCount() == 0u ||
                source->hasExecutable())
            {
                LOG_ERROR("[CUDAGraphCapture] Parallel fragment has incompatible graph/device ownership index=" << i);
                return false;
            }
            fragments[i] = source->graph();
        }
        const auto result = detail::appendNativeParallelBranch<CUDAParallelGraphAPI>(graph_, fragments);
        if (result.error != cudaSuccess)
        {
            LOG_ERROR("[CUDAGraphCapture] Parallel graph assembly failed operation=" <<
                      result.operation << " error=" << cudaGetErrorString(result.error));
            return false;
        }
        std::size_t count = 0u;
        if (cudaGraphGetNodes(graph_, nullptr, &count) != cudaSuccess)
            return false;
        node_count_ = count;
        return true;
    }

    GPUOrderedTimelineTimingSnapshot
    CUDAGraphCapture::consumeOrderedTimelineTiming()
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
            snapshot.error = "could not activate the immutable CUDA graph owner";
            return snapshot;
        }

        const cudaError_t query =
            cudaEventQuery(ordered_timeline_timing_events_.back().stop);
        if (query == cudaErrorNotReady)
        {
            snapshot.state = GPUOrderedTimelineTimingState::Pending;
            return snapshot;
        }
        if (query != cudaSuccess)
        {
            snapshot.state = GPUOrderedTimelineTimingState::Failed;
            snapshot.error = std::string("cudaEventQuery failed: ") +
                             cudaGetErrorString(query);
            return snapshot;
        }

        snapshot.samples.reserve(ordered_timeline_timing_events_.size());
        for (const auto &timing : ordered_timeline_timing_events_)
        {
            float elapsed_ms = 0.0f;
            const cudaError_t elapsed =
                cudaEventElapsedTime(&elapsed_ms, timing.start, timing.stop);
            if (elapsed != cudaSuccess)
            {
                snapshot.samples.clear();
                snapshot.state = GPUOrderedTimelineTimingState::Failed;
                snapshot.error = std::string("cudaEventElapsedTime failed: ") +
                                 cudaGetErrorString(elapsed);
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

    bool CUDAGraphCapture::buildDeviceControlledWhileLoop(
        std::span<const DeviceControlledLoopFragment> ordered_body_fragments,
        const DeviceControlledLoopPredicate &predicate)
    {
        if (timeline_recording_ || recording_role_ != RecordingRole::Owner)
            return false;
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

    bool CUDAGraphCapture::buildDeviceControlledSelectorWhileLoop(
        std::span<const DeviceControlledLoopFragment> ordered_body_fragments,
        const DeviceControlledLoopPredicate &predicate,
        const DeviceControlledLoopSelector &selector_policy)
    {
        if (timeline_recording_ || recording_role_ != RecordingRole::Owner)
            return false;
#if CUDART_VERSION < 12030
        (void)ordered_body_fragments;
        (void)predicate;
        (void)selector_policy;
        LOG_ERROR("[CUDAGraphCapture] Device-controlled selector/WHILE graphs require CUDA 12.3 or newer");
        return false;
#else
        if (!activateOwner("buildDeviceControlledSelectorWhileLoop"))
            return false;

        const bool shared_control_authority =
            predicate.control_rows_device ==
                selector_policy.control_rows_device &&
            predicate.control_stride == selector_policy.control_stride &&
            predicate.request_count == selector_policy.request_count &&
            predicate.healthy_index == selector_policy.healthy_index &&
            predicate.complete_index == selector_policy.complete_index;
        if (ordered_body_fragments.empty() || !predicate.valid() ||
            !selector_policy.valid() || !shared_control_authority ||
            selector_policy.maximum_selector <=
                selector_policy.minimum_selector)
        {
            LOG_ERROR("[CUDAGraphCapture] Invalid device-controlled selector/WHILE contract"
                      << " fragments=" << ordered_body_fragments.size()
                      << " predicate_valid=" << predicate.valid()
                      << " selector_valid=" << selector_policy.valid()
                      << " shared_authority=" << shared_control_authority
                      << " selector_range=["
                      << selector_policy.minimum_selector << ','
                      << selector_policy.maximum_selector << ']');
            return false;
        }

        size_t transaction_node_count = 0;
        size_t selector_gated_fragment_count = 0;
        size_t selector_group_count = 0;
        size_t device_word_fragment_count = 0;
        bool selector_region_started = false;
        bool selector_region_closed = false;
        int previous_threshold = selector_policy.minimum_selector;
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
            if (!fragment.valid() || !cuda_fragment ||
                cuda_fragment == this ||
                cuda_fragment->deviceOrdinal() != device_ordinal_ ||
                !cuda_fragment->graph() ||
                cuda_fragment->nodeCount() == 0)
            {
                LOG_ERROR("[CUDAGraphCapture] Invalid selector/WHILE transaction fragment"
                          << " fragment=" << fragment_index
                          << " name="
                          << (fragment.name ? fragment.name : "<unnamed>")
                          << " execution="
                          << static_cast<int>(fragment.execution)
                          << " threshold=" << fragment.minimum_selector
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

            if (fragment.execution ==
                DeviceControlledLoopFragmentExecution::
                    IfDeviceSelectorAtLeast)
            {
                if (selector_region_closed ||
                    fragment.minimum_selector <=
                        selector_policy.minimum_selector ||
                    fragment.minimum_selector >
                        selector_policy.maximum_selector ||
                    fragment.minimum_selector < previous_threshold)
                {
                    LOG_ERROR("[CUDAGraphCapture] Selector-gated fragments must form one monotonic region"
                              << " fragment=" << fragment_index
                              << " threshold=" << fragment.minimum_selector
                              << " previous_threshold=" << previous_threshold
                              << " region_closed=" << selector_region_closed
                              << " selector_range=["
                              << selector_policy.minimum_selector << ','
                              << selector_policy.maximum_selector << ']');
                    return false;
                }
                if (!selector_region_started ||
                    fragment.minimum_selector != previous_threshold)
                {
                    ++selector_group_count;
                }
                selector_region_started = true;
                previous_threshold = fragment.minimum_selector;
                ++selector_gated_fragment_count;
            }
            else
            {
                if (selector_region_started)
                    selector_region_closed = true;
                if (fragment.execution ==
                    DeviceControlledLoopFragmentExecution::
                        IfDeviceWordNonZero)
                {
                    ++device_word_fragment_count;
                }
            }

            if (!validateCudaConditionalBodyFragmentGraph(
                    cuda_fragment->graph(),
                    fragment.name,
                    fragment_index,
                    "selector_linear_transaction"))
            {
                return false;
            }
            validated_fragments.push_back(cuda_fragment);
            transaction_node_count += cuda_fragment->nodeCount();
        }
        if (!selector_region_started || selector_group_count == 0)
        {
            LOG_ERROR("[CUDAGraphCapture] Selector/WHILE transaction has no selector-gated prefix groups");
            return false;
        }

        reset();
        cudaError_t error = cudaGraphCreate(&graph_, 0);
        if (error != cudaSuccess)
        {
            LOG_ERROR("[CUDAGraphCapture] cudaGraphCreate failed for device selector/WHILE: "
                      << cudaGetErrorString(error));
            return false;
        }

        auto fail = [&](const char *operation, cudaError_t operation_error)
        {
            LOG_ERROR("[CUDAGraphCapture] " << operation
                      << " failed for device selector/WHILE: "
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
            return fail(
                "cudaGraphAddKernelNode(initial WHILE predicate)",
                error);

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
            LOG_ERROR("[CUDAGraphCapture] CUDA did not return a selector/WHILE body graph");
            reset();
            return false;
        }

        cudaGraph_t loop_body = while_params.conditional.phGraph_out[0];
        cudaGraphConditionalHandle transaction_condition = 0;
        error = cudaGraphConditionalHandleCreate(
            &transaction_condition,
            loop_body,
            /*defaultLaunchValue=*/0,
            cudaGraphCondAssignDefault);
        if (error != cudaSuccess)
        {
            return fail(
                "cudaGraphConditionalHandleCreate(selector validation IF)",
                error);
        }

        cudaGraphConditionalHandle transaction_condition_arg =
            transaction_condition;
        int *selector_rows_arg = selector_policy.control_rows_device;
        int selector_stride_arg = selector_policy.control_stride;
        int selector_requests_arg = selector_policy.request_count;
        int selector_healthy_arg = selector_policy.healthy_index;
        int selector_complete_arg = selector_policy.complete_index;
        int selector_index_arg = selector_policy.selector_index;
        int selector_error_arg = selector_policy.error_index;
        int selector_minimum_arg = selector_policy.minimum_selector;
        int selector_maximum_arg = selector_policy.maximum_selector;
        int selector_invalid_error_arg =
            selector_policy.invalid_selector_error;
        void *selector_validation_args[] = {
            &transaction_condition_arg,
            &selector_rows_arg,
            &selector_stride_arg,
            &selector_requests_arg,
            &selector_healthy_arg,
            &selector_complete_arg,
            &selector_index_arg,
            &selector_error_arg,
            &selector_minimum_arg,
            &selector_maximum_arg,
            &selector_invalid_error_arg};

        cudaKernelNodeParams selector_validation_params{};
        selector_validation_params.func = reinterpret_cast<void *>(
            updateDeviceControlledSelectorValidityCondition);
        selector_validation_params.gridDim = dim3(1, 1, 1);
        selector_validation_params.blockDim = dim3(1, 1, 1);
        selector_validation_params.sharedMemBytes = 0;
        selector_validation_params.kernelParams =
            selector_validation_args;
        selector_validation_params.extra = nullptr;

        cudaGraphNode_t selector_validation_node = nullptr;
        error = cudaGraphAddKernelNode(
            &selector_validation_node,
            loop_body,
            /*dependencies=*/nullptr,
            /*numDependencies=*/0,
            &selector_validation_params);
        if (error != cudaSuccess)
        {
            return fail(
                "cudaGraphAddKernelNode(selector validation)",
                error);
        }

        cudaGraphNodeParams transaction_if_params{};
        transaction_if_params.type = cudaGraphNodeTypeConditional;
        transaction_if_params.conditional.handle = transaction_condition;
        transaction_if_params.conditional.type = cudaGraphCondTypeIf;
        transaction_if_params.conditional.size = 1;

        cudaGraphNode_t transaction_if_node = nullptr;
        error = cudaGraphAddNode(
            &transaction_if_node,
            loop_body,
            &selector_validation_node,
            /*dependencyData=*/nullptr,
            /*numDependencies=*/1,
            &transaction_if_params);
        if (error != cudaSuccess)
            return fail("cudaGraphAddNode(selector validation IF)", error);
        if (!transaction_if_params.conditional.phGraph_out ||
            !transaction_if_params.conditional.phGraph_out[0])
        {
            LOG_ERROR("[CUDAGraphCapture] CUDA did not return the validated transaction IF body");
            reset();
            return false;
        }

        cudaGraph_t transaction_graph =
            transaction_if_params.conditional.phGraph_out[0];
        cudaGraphNode_t transaction_tail = nullptr;
        size_t fragment_index = 0;
        while (fragment_index < ordered_body_fragments.size())
        {
            const DeviceControlledLoopFragment &fragment =
                ordered_body_fragments[fragment_index];
            if (fragment.execution !=
                DeviceControlledLoopFragmentExecution::
                    IfDeviceSelectorAtLeast)
            {
                const DeviceControlledFragmentAppendResult appended =
                    appendDeviceControlledFragment(
                        transaction_graph,
                        transaction_tail,
                        fragment,
                        *validated_fragments[fragment_index]);
                if (!appended.succeeded())
                    return fail(appended.operation, appended.error);
                transaction_tail = appended.tail;
                ++fragment_index;
                continue;
            }

            const int threshold = fragment.minimum_selector;
            const size_t group_begin = fragment_index;
            size_t group_end = group_begin;
            while (group_end < ordered_body_fragments.size() &&
                   ordered_body_fragments[group_end].execution ==
                       DeviceControlledLoopFragmentExecution::
                           IfDeviceSelectorAtLeast &&
                   ordered_body_fragments[group_end].minimum_selector ==
                       threshold)
            {
                ++group_end;
            }

            cudaGraphConditionalHandle group_condition = 0;
            error = cudaGraphConditionalHandleCreate(
                &group_condition,
                transaction_graph,
                /*defaultLaunchValue=*/0,
                cudaGraphCondAssignDefault);
            if (error != cudaSuccess)
            {
                return fail(
                    "cudaGraphConditionalHandleCreate(selector threshold IF)",
                    error);
            }

            cudaGraphConditionalHandle group_condition_arg = group_condition;
            int group_threshold_arg = threshold;
            void *group_condition_args[] = {
                &group_condition_arg,
                &selector_rows_arg,
                &selector_stride_arg,
                &selector_requests_arg,
                &selector_healthy_arg,
                &selector_complete_arg,
                &selector_index_arg,
                &selector_error_arg,
                &selector_minimum_arg,
                &selector_maximum_arg,
                &selector_invalid_error_arg,
                &group_threshold_arg};

            cudaKernelNodeParams group_condition_params{};
            group_condition_params.func = reinterpret_cast<void *>(
                updateDeviceControlledSelectorThresholdCondition);
            group_condition_params.gridDim = dim3(1, 1, 1);
            group_condition_params.blockDim = dim3(1, 1, 1);
            group_condition_params.sharedMemBytes = 0;
            group_condition_params.kernelParams = group_condition_args;
            group_condition_params.extra = nullptr;

            cudaGraphNode_t group_condition_node = nullptr;
            error = cudaGraphAddKernelNode(
                &group_condition_node,
                transaction_graph,
                transaction_tail ? &transaction_tail : nullptr,
                transaction_tail ? 1 : 0,
                &group_condition_params);
            if (error != cudaSuccess)
            {
                return fail(
                    "cudaGraphAddKernelNode(selector threshold)",
                    error);
            }

            cudaGraphNodeParams group_if_params{};
            group_if_params.type = cudaGraphNodeTypeConditional;
            group_if_params.conditional.handle = group_condition;
            group_if_params.conditional.type = cudaGraphCondTypeIf;
            group_if_params.conditional.size = 1;

            cudaGraphNode_t group_if_node = nullptr;
            error = cudaGraphAddNode(
                &group_if_node,
                transaction_graph,
                &group_condition_node,
                /*dependencyData=*/nullptr,
                /*numDependencies=*/1,
                &group_if_params);
            if (error != cudaSuccess)
                return fail("cudaGraphAddNode(selector threshold IF)", error);
            if (!group_if_params.conditional.phGraph_out ||
                !group_if_params.conditional.phGraph_out[0])
            {
                LOG_ERROR("[CUDAGraphCapture] CUDA did not return a selector threshold IF body");
                reset();
                return false;
            }

            cudaGraph_t group_graph =
                group_if_params.conditional.phGraph_out[0];
            cudaGraphNode_t group_tail = nullptr;
            for (size_t grouped = group_begin;
                 grouped < group_end;
                 ++grouped)
            {
                cudaGraphNode_t child_node = nullptr;
                error = cudaGraphAddChildGraphNode(
                    &child_node,
                    group_graph,
                    group_tail ? &group_tail : nullptr,
                    group_tail ? 1 : 0,
                    validated_fragments[grouped]->graph());
                if (error != cudaSuccess)
                {
                    return fail(
                        "cudaGraphAddChildGraphNode(selector threshold group)",
                        error);
                }
                group_tail = child_node;
            }

            transaction_tail = group_if_node;
            fragment_index = group_end;
        }

        cudaGraphNode_t predicate_node = nullptr;
        error = cudaGraphAddKernelNode(
            &predicate_node,
            loop_body,
            &transaction_if_node,
            /*numDependencies=*/1,
            &predicate_params);
        if (error != cudaSuccess)
            return fail("cudaGraphAddKernelNode(loop predicate)", error);

        size_t count = 0;
        error = cudaGraphGetNodes(graph_, nullptr, &count);
        if (error != cudaSuccess)
            return fail("cudaGraphGetNodes", error);
        node_count_ = count;
        LOG_DEBUG("[CUDAGraphCapture] Built device-controlled selector/WHILE graph"
                  << " parent_nodes=" << node_count_
                  << " fragments=" << ordered_body_fragments.size()
                  << " selector_gated_fragments="
                  << selector_gated_fragment_count
                  << " selector_groups=" << selector_group_count
                  << " device_word_fragments="
                  << device_word_fragment_count
                  << " selector_range=["
                  << selector_policy.minimum_selector << ','
                  << selector_policy.maximum_selector << ']'
                  << " transaction_nodes=" << transaction_node_count
                  << " requests=" << predicate.request_count);
        return transaction_tail != nullptr;
#endif
    }
    GraphUpdateResult CUDAGraphCapture::tryUpdate()
    {
        if (!supportsExecutableUpdate())
            return GraphUpdateResult::Failed;
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

    bool CUDAGraphCapture::validateFlatHelperNodeKinds(std::string *error) const
    {
        const auto fail = [&](const std::string &detail)
        {
            if (error)
                *error = "CUDA bounded helper shape: " + detail;
            return false;
        };
        if (!activateOwner("validateFlatHelperNodeKinds") || !graph_)
            return fail("missing owner or native graph");
        size_t count = 0u;
        auto status = cudaGraphGetNodes(graph_, nullptr, &count);
        if (status != cudaSuccess)
            return fail(cudaGetErrorString(status));
        if (count != nodeCount() || count == 0u ||
            count > GPUGraphMemoryContract::kBoundedFlatHelperMaxNodes)
            return fail("helper must own one complete bounded native graph");
        // Inspect only node kinds: resolving kernel symbols/occupancy here
        // would turn a tiny setup guard into unnecessary profiler work.
        std::array<cudaGraphNode_t, GPUGraphMemoryContract::kBoundedFlatHelperMaxNodes> nodes{};
        status = cudaGraphGetNodes(graph_, nodes.data(), &count);
        if (status != cudaSuccess)
            return fail(cudaGetErrorString(status));
        for (size_t index = 0; index < count; ++index)
        {
            cudaGraphNodeType kind;
            status = cudaGraphNodeGetType(nodes[index], &kind);
            if (status != cudaSuccess)
                return fail(cudaGetErrorString(status));
            if (kind != cudaGraphNodeTypeKernel && kind != cudaGraphNodeTypeMemcpy &&
                kind != cudaGraphNodeTypeMemset)
                return fail("forbidden nested/control node at index=" + std::to_string(index) +
                            " kind=" + std::to_string(static_cast<int>(kind)));
        }
        return true;
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
                inspection_error,
                recording_role_ == RecordingRole::Fragment ? &fragment_nodes_ : nullptr))
        {
            kernel_nodes.clear();
            if (error)
                *error = std::move(inspection_error);
            return false;
        }
        return true;
    }

    void CUDAGraphCapture::destroyOrderedTimelineTimingEvents() noexcept
    {
        for (auto &timing : ordered_timeline_timing_events_)
        {
            if (timing.start)
                CUDA_WARN_IF_FAIL(cudaEventDestroy(timing.start));
            if (timing.stop)
                CUDA_WARN_IF_FAIL(cudaEventDestroy(timing.stop));
            timing.start = nullptr;
            timing.stop = nullptr;
        }
        ordered_timeline_timing_events_.clear();
        ordered_timeline_timing_pending_ = false;
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
        resident_memory_bytes_ = 0u;
        if (graph_)
        {
            if (!timeline_recording_)
                CUDA_WARN_IF_FAIL(cudaGraphDestroy(graph_));
            graph_ = nullptr;
        }
        // Invalidating the owner revokes any surviving fragment views. Their
        // shared definition remains alive only for safe diagnostic destruction.
        if (timeline_recording_ && recording_role_ == RecordingRole::Owner)
            timeline_recording_->state = OrderedTimelineRecording::State::Failed;
        timeline_recording_.reset();
        fragment_nodes_.clear();
        fragment_entry_ = nullptr;
        fragment_exit_ = nullptr;
        fragment_state_ = FragmentState::Unrecorded;
        destroyOrderedTimelineTimingEvents();
        node_count_ = 0;
    }

} // namespace llaminar2

#endif // HAVE_CUDA
