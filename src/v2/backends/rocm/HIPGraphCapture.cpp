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
