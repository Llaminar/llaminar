#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>

namespace llaminar2
{

    /// Result of an in-place graph executable update
    enum class GraphUpdateResult
    {
        Success,            ///< Graph executable updated in-place
        NeedsReinstantiate, ///< Topology changed, needs full re-instantiation
        Failed              ///< Update failed; the owning execution path must fail
    };

    /**
     * @brief Device-resident predicate for a graph-owned request loop.
     *
     * The predicate addresses a row-major controller table that remains on the
     * accelerator for the complete graph launch.  A loop iteration is admitted
     * only while every request is healthy and at least one request remains
     * incomplete.  A controller failure therefore stops device execution and is
     * surfaced by the caller's terminal validation; it cannot spin forever or
     * silently switch to host scheduling.
     *
     * The field indices deliberately remain explicit.  The graph abstraction
     * does not own the generation-controller ABI, while its caller can bind the
     * exact health and completion fields from that ABI without manufacturing a
     * second device flag or a host-visible mirror.
     */
    struct DeviceControlledLoopPredicate
    {
        const int *control_rows_device = nullptr; ///< First word of row zero on device.
        int control_stride = 0;                   ///< Controller words between request rows.
        int request_count = 0;                    ///< Number of active request rows.
        int healthy_index = -1;                   ///< Non-zero means the row remains valid.
        int complete_index = -1;                  ///< Non-zero means the row is terminal.

        [[nodiscard]] bool valid() const noexcept
        {
            return control_rows_device != nullptr &&
                   control_stride > 0 &&
                   request_count > 0 &&
                   healthy_index >= 0 &&
                   healthy_index < control_stride &&
                   complete_index >= 0 &&
                   complete_index < control_stride;
        }
    };

    /// Abstract interface for GPU graph capture and replay.
    /// Abstracts over HIP Graphs (ROCm) and CUDA Graphs (NVIDIA).
    ///
    /// Lifecycle:
    ///   1. beginCapture() — stream enters capture mode
    ///   2. (launch kernels into the captured stream)
    ///   3. endCapture() — stop capture, produce graph object
    ///   4. instantiate() — compile graph into executable
    ///   5. launch() — replay the entire captured workload
    ///
    /// For subsequent decode steps on a backend that supports executable updates:
    ///   1. beginCapture() → launch kernels → endCapture()
    ///   2. tryUpdate() — in-place update of the executable
    ///   3. launch()
    ///
    /// If tryUpdate() returns NeedsReinstantiate:
    ///   1. instantiate() to create a new executable from the latest capture
    ///   2. launch()
    ///
    /// On a backend that does not support executable updates:
    ///   1. beginCapture() → launch kernels → endCapture()
    ///   2. instantiate() to replace the executable from the latest capture
    ///   3. launch()
    class IGPUGraphCapture
    {
    public:
        virtual ~IGPUGraphCapture() = default;

        /// Begin stream capture. All subsequent kernel launches on the associated
        /// stream will be recorded into a graph rather than executed.
        /// @return true on success
        virtual bool beginCapture() = 0;

        /// End stream capture and produce a graph object from the recorded operations.
        /// @return true on success
        virtual bool endCapture() = 0;

        /// Compile the captured graph into an executable.
        /// Must be called after endCapture() before the first launch().
        /// @return true on success
        virtual bool instantiate() = 0;

        /// Launch (replay) the instantiated graph executable on the associated stream.
        /// @return true on success
        virtual bool launch() = 0;

        /**
         * @brief Report whether this graph owner supports a device-controlled WHILE node.
         *
         * This is a static backend/runtime property.  A caller selecting a
         * fully device-resident generation architecture must require it (or a
         * backend-native equivalent) before admitting the request; this method
         * is not permission to retry through a host loop.
         */
        [[nodiscard]] virtual bool supportsDeviceControlledWhileLoop() const noexcept
        {
            return false;
        }

        /**
         * @brief Replace this graph with a device-controlled repetition of captured fragments.
         *
         * The implementation clones every element of @p ordered_body_fragments
         * into one conditional WHILE body, adds an explicit dependency from each
         * fragment to its successor, and appends the device predicate update
         * after the final fragment. The first iteration is admitted by request
         * admission; subsequent iterations are controlled exclusively by
         * @p predicate. No D2H copy, host callback, allocation, or stream
         * synchronization is permitted in the generated graph.
         *
         * On success the graph is built but not instantiated.  The caller must
         * call instantiate() exactly as it would after endCapture().
         *
         * @param ordered_body_fragments Captured, non-empty transaction fragments
         *        in producer-to-consumer order. Every pointer must remain valid
         *        through this call and must identify the same backend/device
         *        context as this graph owner. Source captures may own different
         *        streams: child-graph cloning discards launch-stream identity,
         *        and the parent dependencies establish transaction ordering.
         * @param predicate Persistent device controller binding.
         * @return true when this object owns a complete loop graph.
         */
        virtual bool buildDeviceControlledWhileLoop(
            std::span<const IGPUGraphCapture *const> ordered_body_fragments,
            const DeviceControlledLoopPredicate &predicate)
        {
            (void)ordered_body_fragments;
            (void)predicate;
            return false;
        }

        /**
         * @brief Convenience overload for a transaction captured as one graph.
         *
         * This overload deliberately delegates to the ordered-fragment contract,
         * so backends have exactly one composition implementation and monolithic
         * transactions cannot acquire subtly different loop semantics.
         */
        bool buildDeviceControlledWhileLoop(
            const IGPUGraphCapture &body,
            const DeviceControlledLoopPredicate &predicate)
        {
            const IGPUGraphCapture *fragments[] = {&body};
            return buildDeviceControlledWhileLoop(fragments, predicate);
        }

        /**
         * @brief Return the exact stream owned by this capture lifecycle.
         *
         * Capture, replay, and post-launch completion publication must all use
         * this stream. Exposing the non-owning identity through the backend
         * interface prevents callers from accepting a second, potentially
         * different stream and silently publishing an event against work that
         * was launched elsewhere.
         *
         * @return Opaque non-null stream identity for a usable GPU capture.
         */
        [[nodiscard]] virtual void *executionStream() const noexcept = 0;

        /// Attempt to update the existing executable in-place with a newly captured graph.
        /// Call this after endCapture() on subsequent iterations where the graph topology
        /// is unchanged but kernel parameters differ.
        ///
        /// This operation is valid only when supportsExecutableUpdate() returns true.
        /// Calling it on a backend that does not advertise the operation is a programming
        /// error and returns GraphUpdateResult::Failed. Callers must instantiate the newly
        /// captured graph directly for such backends; they must not probe support by
        /// invoking this method and interpreting a runtime error.
        /// @return GraphUpdateResult indicating success, need for reinstantiation, or failure
        virtual GraphUpdateResult tryUpdate() = 0;

        /**
         * @brief Report whether this backend can update an instantiated executable in place.
         *
         * Graph recapture always produces a new graph object. Backends that return true may
         * attempt to transplant the new graph's parameters and topology into the existing
         * executable through tryUpdate(). Backends that return false require instantiate()
         * to replace the executable directly. This is a static backend capability, not a
         * transient runtime condition, so orchestration code can select one explicit
         * lifecycle without failed API calls, sticky-error clearing, or retry behavior.
         *
         * @return true when tryUpdate() is a supported active lifecycle operation.
         */
        [[nodiscard]] virtual bool supportsExecutableUpdate() const noexcept = 0;

        /// @return true if an instantiated executable exists and is ready for launch()
        virtual bool hasExecutable() const = 0;

        /// @return Number of nodes in the last captured graph (0 if no capture done)
        virtual size_t nodeCount() const = 0;

        /// Destroy all captured graph and executable resources.
        /// Safe to call multiple times or on an empty object.
        virtual void reset() = 0;

        /// @return Human-readable backend name ("HIP" or "CUDA")
        virtual const char *backendName() const = 0;

    protected:
        IGPUGraphCapture() = default;
        // Non-copyable
        IGPUGraphCapture(const IGPUGraphCapture &) = delete;
        IGPUGraphCapture &operator=(const IGPUGraphCapture &) = delete;
    };

} // namespace llaminar2
