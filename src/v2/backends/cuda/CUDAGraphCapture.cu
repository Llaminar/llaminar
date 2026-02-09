#ifdef HAVE_CUDA

#include "CUDAGraphCapture.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    CUDAGraphCapture::CUDAGraphCapture(cudaStream_t stream) : stream_(stream) {}

    CUDAGraphCapture::~CUDAGraphCapture() { reset(); }

    CUDAGraphCapture::CUDAGraphCapture(CUDAGraphCapture &&other) noexcept
        : stream_(other.stream_), graph_(other.graph_), exec_(other.exec_),
          node_count_(other.node_count_),
          consecutive_update_failures_(other.consecutive_update_failures_)
    {
        other.stream_ = nullptr;
        other.graph_ = nullptr;
        other.exec_ = nullptr;
        other.node_count_ = 0;
        other.consecutive_update_failures_ = 0;
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
            consecutive_update_failures_ = other.consecutive_update_failures_;
            other.stream_ = nullptr;
            other.graph_ = nullptr;
            other.exec_ = nullptr;
            other.node_count_ = 0;
            other.consecutive_update_failures_ = 0;
        }
        return *this;
    }

    bool CUDAGraphCapture::beginCapture()
    {
        // Destroy any previous graph (but keep exec_ for tryUpdate)
        if (graph_)
        {
            cudaGraphDestroy(graph_);
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
            cudaGraphExecDestroy(exec_);
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
        consecutive_update_failures_ = 0;
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

    GraphUpdateResult CUDAGraphCapture::tryUpdate()
    {
        if (!exec_ || !graph_)
        {
            return GraphUpdateResult::Failed;
        }

        // Use 4-arg version with cudaGraphExecUpdateResult for CUDA 10-12 compatibility
        cudaGraphExecUpdateResult update_result;
        cudaError_t err = cudaGraphExecUpdate(exec_, graph_, nullptr, &update_result);

        if (err == cudaSuccess && update_result == cudaGraphExecUpdateSuccess)
        {
            consecutive_update_failures_ = 0;
            LOG_TRACE("[CUDAGraphCapture] Graph executable updated in-place");
            return GraphUpdateResult::Success;
        }

        consecutive_update_failures_++;

        if (update_result == cudaGraphExecUpdateErrorTopologyChanged ||
            update_result == cudaGraphExecUpdateErrorNodeTypeChanged ||
            update_result == cudaGraphExecUpdateErrorNotSupported)
        {
            LOG_WARN("[CUDAGraphCapture] Graph update needs reinstantiation: result="
                     << static_cast<int>(update_result)
                     << " (failure " << consecutive_update_failures_ << ")");
            return GraphUpdateResult::NeedsReinstantiate;
        }

        LOG_WARN("[CUDAGraphCapture] Graph update failed: " << cudaGetErrorString(err)
                                                            << " result=" << static_cast<int>(update_result)
                                                            << " (failure " << consecutive_update_failures_ << ")");
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
            cudaGraphExecDestroy(exec_);
            exec_ = nullptr;
        }
        if (graph_)
        {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
        node_count_ = 0;
        consecutive_update_failures_ = 0;
    }

} // namespace llaminar2

#endif // HAVE_CUDA
