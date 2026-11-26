/**
 * @file IEarlyPacketPublication.h
 * @brief Backend-neutral native capture frontier splice for early route packets.
 *
 * A retained GPU route prefix and its local FFN suffix live inside one native
 * graph capture. Independent outbound packet readers (mapped remote-lane
 * dispatch and host dispatch tickets) depend only on the route prefix's
 * completion, allowing remote CPU/GPU participants to start while the local
 * FFN suffix computes. Both branches rejoin at the original graph cutpoint
 * without host synchronization, callbacks, or graph recapture.
 */

#pragma once

#include <functional>
#include <memory>

namespace llaminar2
{
    /**
     * @brief Own one early packet chain's native producer and consumer edges.
     */
    class IEarlyPacketPublication
    {
    public:
        virtual ~IEarlyPacketPublication() = default;

        /**
         * @brief Bind the exact non-default capture stream before recording.
         * @param stream Exact cold preparation stream.
         */
        virtual void prepare(void *stream) = 0;

        /**
         * @brief Snapshot the active capture frontier after the route producer.
         * @param stream Exact stream immediately after recording the producer node.
         */
        virtual void recordCompletedProducer(void *stream) = 0;

        /**
         * @brief Record a packet consumer against the early producer frontier
         *        and rejoin its completion with the concurrent local arithmetic.
         * @param stream Exact native capture stream bound to the producer.
         * @param record Construction-only packet submission reading the producer.
         * @return True when the packet and arithmetic edges are joined.
         */
        virtual bool recordConsumer(void *stream, const std::function<bool()> &record) = 0;
    };

#ifdef HAVE_CUDA
    /** @brief Construct one CUDA native capture early-packet publication owner. */
    std::shared_ptr<IEarlyPacketPublication> makeCUDAEarlyPacketPublication();
#endif

#ifdef HAVE_ROCM
    /** @brief Construct one HIP native capture early-packet publication owner. */
    std::shared_ptr<IEarlyPacketPublication> makeHIPEarlyPacketPublication();
#endif

} // namespace llaminar2
