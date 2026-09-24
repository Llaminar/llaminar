/**
 * @file GraphCaptureStageActivity.h
 * @brief Exception-safe publication of stage-owned native-capture activity.
 *
 * Native capture is a stream lifetime, not merely a backend API call. Some
 * stages lend that same stream to a background authority which may otherwise
 * attempt to record an external event while CUDA or HIP is capturing it. This
 * owner publishes one typed Entering edge before native capture and guarantees
 * one Completed or Aborted terminal edge on every exit path.
 */

#pragma once

#include "../../compute_stages/IComputeStage.h"
#include "../../../utils/Logger.h"

#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace llaminar2
{
    /**
     * @brief RAII owner for a set of stage capture-activity transitions.
     *
     * The caller resolves stage pointers before `beginCapture()` and keeps the
     * backing span alive for this object's lifetime. Entering proceeds in graph
     * order; terminal publication proceeds in reverse order. If one Entering
     * transition fails, all previously entered stages are aborted before the
     * method returns. The destructor provides the same rollback for exceptions
     * raised by graph recording or `endCapture()` bookkeeping.
     */
    class ScopedGraphCaptureStageActivity final
    {
    public:
        /**
         * @brief Bind the stages and exact stream for one future capture.
         *
         * @param stages Stable stage-pointer span in native recording order.
         * @param ctx Device context that owns @p stream.
         * @param stream Exact non-null CUDA/HIP capture stream.
         * @param operation Stable diagnostic name for rollback failures.
         */
        ScopedGraphCaptureStageActivity(
            std::span<IComputeStage *const> stages,
            IDeviceContext *ctx,
            void *stream,
            std::string operation)
            : stages_(stages),
              ctx_(ctx),
              stream_(stream),
              operation_(std::move(operation))
        {
        }

        ~ScopedGraphCaptureStageActivity()
        {
            if (!active_)
                return;
            if (!publishTerminal(GraphCaptureActivityTransition::Aborted))
            {
                LOG_ERROR(
                    "[ScopedGraphCaptureStageActivity] Failed to publish "
                    "aborted stage-capture activity for "
                    << operation_);
            }
        }

        ScopedGraphCaptureStageActivity(
            const ScopedGraphCaptureStageActivity &) = delete;
        ScopedGraphCaptureStageActivity &operator=(
            const ScopedGraphCaptureStageActivity &) = delete;

        /**
         * @brief Publish Entering to every stage before backend capture begins.
         *
         * @return true when every stage accepted the transition; false after
         *         rolling back the accepted prefix.
         */
        bool begin()
        {
            if (active_ || entered_count_ != 0u)
            {
                throw std::logic_error(
                    "ScopedGraphCaptureStageActivity cannot begin twice");
            }
            if (!ctx_ || !stream_)
            {
                LOG_ERROR(
                    "[ScopedGraphCaptureStageActivity] Missing device context "
                    "or exact stream for "
                    << operation_);
                return false;
            }

            for (IComputeStage *stage : stages_)
            {
                if (!stage ||
                    !stage->transitionGraphCaptureActivity(
                        ctx_,
                        stream_,
                        GraphCaptureActivityTransition::Entering))
                {
                    if (!publishEnteredPrefixAbort())
                    {
                        LOG_ERROR(
                            "[ScopedGraphCaptureStageActivity] Failed to roll "
                            "back a partial Entering transition for "
                            << operation_);
                    }
                    return false;
                }
                ++entered_count_;
            }
            active_ = true;
            return true;
        }

        /**
         * @brief Close the published activity after native capture ends.
         *
         * @param transition Completed for a retained unit or Aborted for a
         *        discarded/failed recording.
         * @return true when every entered stage accepted its terminal edge.
         */
        bool finish(GraphCaptureActivityTransition transition)
        {
            if (!active_ ||
                (transition != GraphCaptureActivityTransition::Completed &&
                 transition != GraphCaptureActivityTransition::Aborted))
            {
                throw std::logic_error(
                    "ScopedGraphCaptureStageActivity received an invalid terminal transition");
            }
            return publishTerminal(transition);
        }

    private:
        /** Publish Aborted for a failed prefix before active ownership begins. */
        bool publishEnteredPrefixAbort() noexcept
        {
            bool ok = true;
            while (entered_count_ > 0u)
            {
                --entered_count_;
                IComputeStage *stage = stages_[entered_count_];
                try
                {
                    ok = stage &&
                             stage->transitionGraphCaptureActivity(
                                 ctx_,
                                 stream_,
                                 GraphCaptureActivityTransition::Aborted) &&
                         ok;
                }
                catch (...)
                {
                    ok = false;
                }
            }
            return ok;
        }

        /** Publish one terminal edge to the full entered set in reverse order. */
        bool publishTerminal(
            GraphCaptureActivityTransition transition) noexcept
        {
            bool ok = true;
            while (entered_count_ > 0u)
            {
                --entered_count_;
                IComputeStage *stage = stages_[entered_count_];
                try
                {
                    ok = stage &&
                             stage->transitionGraphCaptureActivity(
                                 ctx_, stream_, transition) &&
                         ok;
                }
                catch (...)
                {
                    ok = false;
                }
            }
            active_ = false;
            return ok;
        }

        std::span<IComputeStage *const> stages_;
        IDeviceContext *ctx_ = nullptr;
        void *stream_ = nullptr;
        std::string operation_;
        std::size_t entered_count_ = 0u;
        bool active_ = false;
    };

} // namespace llaminar2
