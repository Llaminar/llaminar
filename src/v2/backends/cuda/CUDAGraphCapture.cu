#ifdef HAVE_CUDA

#include "CUDAGraphCapture.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

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

    CUDAGraphCapture::CUDAGraphCapture(cudaStream_t stream) : stream_(stream) {}

    CUDAGraphCapture::~CUDAGraphCapture() { reset(); }

    CUDAGraphCapture::CUDAGraphCapture(CUDAGraphCapture &&other) noexcept
        : stream_(other.stream_), graph_(other.graph_), exec_(other.exec_),
          node_count_(other.node_count_)
    {
        other.stream_ = nullptr;
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

    bool CUDAGraphCapture::beginCapture()
    {
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

        // Use 5-arg version for compatibility with CUDA 10-12+
        cudaError_t err = cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAGraphCapture] cudaGraphInstantiate failed: " << cudaGetErrorString(err));
            exec_ = nullptr;
            return false;
        }
        LOG_DEBUG("[CUDAGraphCapture] Instantiated graph executable (" << node_count_ << " nodes)");
        return true;
    }

    bool CUDAGraphCapture::launch()
    {
        if (!exec_)
        {
            LOG_ERROR("[CUDAGraphCapture] Cannot launch: no instantiated executable");
            return false;
        }
        cudaError_t err = cudaGraphLaunch(exec_, stream_);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAGraphCapture] cudaGraphLaunch failed: " << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool CUDAGraphCapture::buildDeviceControlledWhileLoop(
        std::span<const IGPUGraphCapture *const> ordered_body_fragments,
        const DeviceControlledLoopPredicate &predicate)
    {
#if CUDART_VERSION < 12030
        (void)ordered_body_fragments;
        (void)predicate;
        LOG_ERROR("[CUDAGraphCapture] Device-controlled graph loops require CUDA 12.3 or newer");
        return false;
#else
        if (ordered_body_fragments.empty() || !predicate.valid())
        {
            LOG_ERROR("[CUDAGraphCapture] Invalid device-controlled loop contract"
                      << " fragments=" << ordered_body_fragments.size()
                      << " predicate_valid=" << predicate.valid());
            return false;
        }

        size_t transaction_node_count = 0;
        for (size_t fragment_index = 0;
             fragment_index < ordered_body_fragments.size();
             ++fragment_index)
        {
            const IGPUGraphCapture *fragment =
                ordered_body_fragments[fragment_index];
            const auto *cuda_fragment =
                dynamic_cast<const CUDAGraphCapture *>(fragment);
            if (!cuda_fragment || cuda_fragment == this ||
                !cuda_fragment->graph() || cuda_fragment->nodeCount() == 0)
            {
                LOG_ERROR(
                    "[CUDAGraphCapture] Invalid device-loop fragment"
                    << " index=" << fragment_index
                    << " backend="
                    << (fragment ? fragment->backendName() : "<null>")
                    << " nodes=" << (fragment ? fragment->nodeCount() : 0)
                    << " self=" << (cuda_fragment == this));
                return false;
            }
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
            /*defaultLaunchValue=*/1,
            cudaGraphCondAssignDefault);
        if (error != cudaSuccess)
            return fail("cudaGraphConditionalHandleCreate", error);

        cudaGraphNodeParams conditional_params{};
        conditional_params.type = cudaGraphNodeTypeConditional;
        conditional_params.conditional.handle = condition;
        conditional_params.conditional.type = cudaGraphCondTypeWhile;
        conditional_params.conditional.size = 1;

        cudaGraphNode_t conditional_node = nullptr;
        error = cudaGraphAddNode(
            &conditional_node,
            graph_,
            /*dependencies=*/nullptr,
            /*dependencyData=*/nullptr,
            /*numDependencies=*/0,
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
        for (size_t fragment_index = 0;
             fragment_index < ordered_body_fragments.size();
             ++fragment_index)
        {
            const auto *cuda_fragment = static_cast<const CUDAGraphCapture *>(
                ordered_body_fragments[fragment_index]);
            cudaGraphNode_t fragment_node = nullptr;
            error = cudaGraphAddChildGraphNode(
                &fragment_node,
                loop_body,
                transaction_tail ? &transaction_tail : nullptr,
                transaction_tail ? 1 : 0,
                cuda_fragment->graph());
            if (error != cudaSuccess)
                return fail("cudaGraphAddChildGraphNode(transaction fragment)", error);
            transaction_tail = fragment_node;
        }

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
                  << " transaction_nodes=" << transaction_node_count
                  << " requests=" << predicate.request_count);
        return true;
#endif
    }

    GraphUpdateResult CUDAGraphCapture::tryUpdate()
    {
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

    void CUDAGraphCapture::reset()
    {
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
