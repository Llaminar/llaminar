#ifdef HAVE_ROCM

#include "HIPGraphCapture.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    // Best-effort error logging for HIP cleanup paths (destructors, reset(),
    // resource-clear-before-reuse). Failure here typically means the GPU state is
    // already corrupted, but we are tearing down anyway — logging at WARN keeps
    // diagnostics visible without escalating during shutdown/error rollback,
    // where throwing or logging at ERROR could mask the real failure or trigger
    // std::terminate from a destructor on stack unwind.
#define HIP_WARN_IF_FAIL(call)                                                      \
    do                                                                              \
    {                                                                               \
        hipError_t _err = (call);                                                   \
        if (_err != hipSuccess)                                                     \
        {                                                                           \
            LOG_WARN("[HIPGraphCapture] " << #call << " failed: "                   \
                                          << hipGetErrorString(_err) << " ("       \
                                          << __FILE__ << ":" << __LINE__ << ")");  \
        }                                                                           \
    } while (0)

    HIPGraphCapture::HIPGraphCapture(hipStream_t stream) : stream_(stream) {}

    HIPGraphCapture::~HIPGraphCapture() { reset(); }

    HIPGraphCapture::HIPGraphCapture(HIPGraphCapture &&other) noexcept
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

    HIPGraphCapture &HIPGraphCapture::operator=(HIPGraphCapture &&other) noexcept
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

    bool HIPGraphCapture::beginCapture()
    {
        // Destroy any previous graph (but keep exec_ for tryUpdate)
        if (graph_)
        {
            HIP_WARN_IF_FAIL(hipGraphDestroy(graph_));
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
            HIP_WARN_IF_FAIL(hipGraphExecDestroy(exec_));
            exec_ = nullptr;
        }

        hipError_t err = hipGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0);
        if (err != hipSuccess)
        {
            LOG_ERROR("[HIPGraphCapture] hipGraphInstantiate failed: " << hipGetErrorString(err));
            exec_ = nullptr;
            return false;
        }
        consecutive_update_failures_ = 0;
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
        if (!exec_ || !graph_)
        {
            return GraphUpdateResult::Failed;
        }

        hipGraphExecUpdateResult update_result;
        hipError_t err = hipGraphExecUpdate(exec_, graph_, nullptr, &update_result);

        if (err == hipSuccess && update_result == hipGraphExecUpdateSuccess)
        {
            consecutive_update_failures_ = 0;
            LOG_TRACE("[HIPGraphCapture] Graph executable updated in-place");
            return GraphUpdateResult::Success;
        }

        consecutive_update_failures_++;

        if (update_result == hipGraphExecUpdateErrorTopologyChanged ||
            update_result == hipGraphExecUpdateErrorNodeTypeChanged ||
            update_result == hipGraphExecUpdateErrorNotSupported)
        {
            LOG_WARN("[HIPGraphCapture] Graph update needs reinstantiation: result="
                     << static_cast<int>(update_result)
                     << " (failure " << consecutive_update_failures_ << ")");
            return GraphUpdateResult::NeedsReinstantiate;
        }

        LOG_WARN("[HIPGraphCapture] Graph update failed: " << hipGetErrorString(err)
                                                           << " result=" << static_cast<int>(update_result)
                                                           << " (failure " << consecutive_update_failures_ << ")");
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
        consecutive_update_failures_ = 0;
    }

} // namespace llaminar2

#endif // HAVE_ROCM
