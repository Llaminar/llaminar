/**
 * @file IMoEGroupedVerifierHistogramPublisher.h
 * @brief Typed accepted-state publication contracts for grouped MoE routing.
 *
 * Grouped MTP verification computes routes for physical rows that may later be
 * rejected.  A graph stage that retains those routes implements this interface
 * so the accepted-state transaction can publish only the serial-visible prefix
 * into the production routing histograms.  The interface deliberately exposes
 * no host route data and requires the exact GPU publication stream.
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace llaminar2
{
    enum class RuntimeHistogramProducerCaptureTransition : std::uint8_t;

    /**
     * @brief Explicit lifecycle role at a grouped-verifier history boundary.
     *
     * Every routed MoE layer in a main verifier graph selects exactly one
     * router or expert stage as its history boundary.  Static placement still
     * declares that boundary, but deliberately performs no history update.
     * Dynamic/Observe placement retains physical verifier routes there and
     * publishes only rows accepted by the later state transaction.  Keeping
     * `NotOwner` distinct from `StaticNoPublication` lets graph validation
     * reject a missing Dynamic publisher without pretending that Static needs
     * an unused stream or histogram bank.
     */
    enum class MoEGroupedVerifierHistogramRole : std::uint8_t
    {
        /** This stage is not the selected per-layer history boundary. */
        NotOwner,
        /** This is the boundary, but Static placement owns no route history. */
        StaticNoPublication,
        /** This boundary defers production history until acceptance is known. */
        DeferredAcceptedRows,
    };

    /**
     * @brief Device-ledger owner participating in accepted MTP publication.
     *
     * Exactly one implementation must be selected for each routed MoE layer in
     * a main verifier graph.  Ordinary homogeneous graphs use the expert stage,
     * which retains final expert and participant assignments while grouping.
     * A heterogeneous ExpertOverlay graph uses its router, which retains the
     * selected expert IDs at the explicit ticket boundary and records no local
     * participant ownership because execution occurs in other device domains.
     */
    class IMoEGroupedVerifierHistogramPublisher
    {
    public:
        virtual ~IMoEGroupedVerifierHistogramPublisher() = default;

        /**
         * @brief Return this stage's explicit per-layer history role.
         *
         * Exactly one router or expert stage per routed verifier layer must
         * return a role other than @ref MoEGroupedVerifierHistogramRole::NotOwner.
         * A deferred role requires a complete table-owned publication stream;
         * a static role forbids one and makes the accepted-state transaction a
         * deliberate no-op for MoE demand.
         *
         * @return Typed ownership/publication role for this concrete stage.
         */
        [[nodiscard]] virtual MoEGroupedVerifierHistogramRole
        groupedVerifierHistogramRole() const noexcept = 0;

        /**
         * @brief Return the model layer represented by this retained ledger.
         *
         * @return Non-negative routed MoE layer index for an active publisher.
         */
        [[nodiscard]] virtual int
        groupedVerifierHistogramLayerIndex() const noexcept = 0;

        /**
         * @brief Return a stable diagnostic label without allocating.
         *
         * @return Static-lifetime label identifying the publishing stage kind.
         */
        [[nodiscard]] virtual std::string_view
        groupedVerifierHistogramPublisherName() const noexcept = 0;

        /**
         * @brief Return the sole model-lifetime accepted-publication stream.
         *
         * The runtime-table authority creates and admits this stream before
         * asynchronous histogram maintenance can seal its producer topology.
         * Every accepted-state graph identity must borrow the returned stream;
         * creating an identity-local replacement is an invalid lifecycle.
         *
         * @return Exact non-null CUDA/HIP stream for an active publisher.
         */
        [[nodiscard]] virtual void *
        groupedVerifierHistogramPublicationStream() const = 0;

        /**
         * @brief Admit the exact accepted-state producer before graph capture.
         *
         * The stream must equal @ref groupedVerifierHistogramPublicationStream.
         * Model setup has already allocated its reusable events and installed
         * the initialization edge, so this call is an allocation-free identity
         * certification at the enclosing graph's `prepareGraphLaunch()` edge.
         *
         * @param producer_stream Exact non-null CUDA/HIP capture/replay stream.
         * @return true when the producer identity is permanently admitted.
         */
        virtual bool prepareGroupedVerifierHistogramProducer(
            void *producer_stream) = 0;

        /**
         * @brief Publish the native-capture activity of the admitted stream.
         *
         * The accepted-state graph calls Entering immediately before backend
         * capture and exactly one Completed or Aborted edge afterward. This is
         * host-only lifecycle bookkeeping which prevents asynchronous
         * histogram maintenance from recording an external event into an
         * active CUDA/HIP capture interval.
         *
         * @param producer_stream Exact pre-admitted publication stream.
         * @param transition Typed producer-capture lifecycle edge.
         * @return true when the runtime-table authority accepted the edge.
         */
        virtual bool transitionGroupedVerifierHistogramProducerCapture(
            void *producer_stream,
            RuntimeHistogramProducerCaptureTransition transition) = 0;

        /**
         * @brief Enqueue accepted-row histogram publication on the exact stream.
         *
         * The implementation joins its immutable-address route ledger with the
         * device-owned accepted prefix for every request.  It must neither read
         * acceptance on the host nor synchronize the producer stream.
         *
         * @param accepted_state_counts_device Accepted rows per request.
         * @param publication_ok_flags_device Metadata-valid flag per request.
         * @param request_count Number of request-major verifier groups.
         * @param rows_per_request Physical verifier rows in each request group.
         * @param producer_stream Exact CUDA/HIP accepted-publication stream.
         * The implementation must also validate that @p producer_stream was
         * admitted by @ref prepareGroupedVerifierHistogramProducer; that
         * validation performs no backend operation and is capture-safe.
         *
         * @return true after the graph-capturable commit was enqueued.
         */
        virtual bool enqueueCommittedGroupedVerifierHistograms(
            const int32_t *accepted_state_counts_device,
            const int32_t *publication_ok_flags_device,
            int request_count,
            int rows_per_request,
            void *producer_stream) = 0;
    };

} // namespace llaminar2
