/**
 * @file IGPUGraphCapture.h
 * @brief Backend-neutral ownership and composition API for native GPU graphs.
 *
 * The interface exposes graph lifecycle operations together with the small set
 * of native conditional compositions used by fully device-owned generation.
 * Conditional predicates and selectors always name persistent device rows;
 * implementations may not materialize those values on the host, insert host
 * callbacks, synchronize a stream, or allocate replay-time storage.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>

namespace llaminar2
{

    class IGPUGraphCapture;

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

    /**
     * @brief Device-resident selector and fatal validation policy for a loop switch.
     *
     * A single native SWITCH node controls one uniformly shaped request batch.
     * Every healthy, incomplete request must publish the same selector. Values
     * outside `[minimum_selector, maximum_selector]`, or disagreement between
     * active rows, invalidate all active rows before any branch executes. The
     * selected branch index is the selector value itself, keeping the control
     * ABI visible and avoiding a second host-authored lookup table.
     */
    struct DeviceControlledLoopSwitch
    {
        int *control_rows_device = nullptr; ///< Mutable first word of row zero.
        int control_stride = 0;             ///< Controller words between rows.
        int request_count = 0;              ///< Uniformly switched request rows.
        int healthy_index = -1;             ///< Non-zero means the row is valid.
        int complete_index = -1;            ///< Non-zero means the row is terminal.
        int selector_index = -1;            ///< Branch index published by the device.
        int error_index = -1;                ///< Fatal diagnostic word to publish.
        int minimum_selector = 0;            ///< First implemented branch index.
        int maximum_selector = -1;           ///< Last implemented branch index.
        int invalid_selector_error = 0;      ///< Non-zero stable terminal error code.

        /**
         * @brief Validate scalar layout independently of the branch table.
         */
        [[nodiscard]] bool valid() const noexcept
        {
            return control_rows_device != nullptr && control_stride > 0 &&
                   request_count > 0 && healthy_index >= 0 &&
                   healthy_index < control_stride && complete_index >= 0 &&
                   complete_index < control_stride && selector_index >= 0 &&
                   selector_index < control_stride && error_index >= 0 &&
                   error_index < control_stride && minimum_selector >= 0 &&
                   maximum_selector >= minimum_selector &&
                   invalid_selector_error != 0;
        }
    };

    /**
     * @brief One named native-graph fragment in a device-controlled transaction.
     *
     * The semantic name is part of the composition contract rather than an
     * incidental log string. Native conditional APIs impose stricter recursive
     * node rules than ordinary graph replay, so a rejected fragment must identify
     * the exact producer role that exported an incompatible graph. The name and
     * capture are borrowed only for the duration of parent construction.
     */
    struct DeviceControlledLoopFragment
    {
        const char *name = nullptr;                  ///< Stable non-empty producer role.
        const IGPUGraphCapture *capture = nullptr;   ///< Borrowed captured graph owner.

        /** @brief Verify that the fragment carries complete diagnostic identity. */
        [[nodiscard]] bool valid() const noexcept
        {
            return name != nullptr && name[0] != '\0' && capture != nullptr;
        }
    };

    /**
     * @brief One complete transaction body selected by a device control value.
     *
     * Fragments are cloned in producer-to-consumer order. A branch inside the
     * declared selector interval must be non-empty: accepting a selector and
     * executing no transaction would leave the outer WHILE spinning forever.
     */
    struct DeviceControlledLoopBranch
    {
        std::span<const DeviceControlledLoopFragment> ordered_fragments;
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
         * @brief Report support for a device-selected transaction inside WHILE.
         *
         * This is stronger than @ref supportsDeviceControlledWhileLoop: the
         * backend must support a nested native SWITCH whose selector is updated
         * by a device kernel on every loop iteration.
         */
        [[nodiscard]] virtual bool supportsDeviceControlledSwitchWhileLoop() const noexcept
        {
            return false;
        }

        /**
         * @brief Replace this graph with a device-controlled repetition of captured fragments.
         *
         * The implementation clones every element of @p ordered_body_fragments
         * into one conditional WHILE body, adds an explicit dependency from each
         * fragment to its successor, and appends the device predicate update
         * after the final fragment. Backend-native composers may lower internal
         * multi-stream event handoffs to equivalent direct dependency edges when
         * conditional bodies do not admit event nodes. A root wait or terminal
         * record crosses the fragment boundary and must remain a hard error. The
         * first iteration is admitted by request admission; subsequent iterations
         * are controlled exclusively by @p predicate. No D2H copy, host callback,
         * allocation, or stream synchronization is permitted in the generated
         * graph.
         *
         * On success the graph is built but not instantiated.  The caller must
         * call instantiate() exactly as it would after endCapture().
         *
         * @param ordered_body_fragments Named, captured, non-empty transaction
         *        fragments in producer-to-consumer order. Every capture pointer
         *        must remain valid through this call and identify the same
         *        backend/device context as this graph owner. Source captures may
         *        own different streams: child-graph cloning discards launch-stream
         *        identity, and parent dependencies establish transaction ordering.
         * @param predicate Persistent device controller binding.
         * @return true when this object owns a complete loop graph.
         */
        virtual bool buildDeviceControlledWhileLoop(
            std::span<const DeviceControlledLoopFragment> ordered_body_fragments,
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
            const DeviceControlledLoopFragment fragments[] = {{
                .name = "complete transaction",
                .capture = &body,
            }};
            return buildDeviceControlledWhileLoop(fragments, predicate);
        }

        /**
         * @brief Replace this graph with a device-controlled WHILE of SWITCH bodies.
         *
         * The implementation first evaluates @p predicate to decide whether the
         * loop may begin. Each iteration validates @p switch_policy on device,
         * executes exactly one complete branch, then reevaluates @p predicate.
         * Invalid or divergent selectors execute no branch and make the request
         * terminally unhealthy. No partial common tail is permitted outside the
         * branches because it could mutate state after selector validation failed.
         *
         * Branch array index is the device selector value. Entries outside the
         * declared selector interval may be empty; every entry inside it must own
         * at least one valid captured fragment.
         *
         * @param branches Complete transaction bodies indexed by selector value.
         * @param predicate Device-owned loop continuation policy.
         * @param switch_policy Device-owned selector and fatal validation policy.
         * @return true when this object owns a built, uninstantiated parent graph.
         */
        virtual bool buildDeviceControlledSwitchWhileLoop(
            std::span<const DeviceControlledLoopBranch> branches,
            const DeviceControlledLoopPredicate &predicate,
            const DeviceControlledLoopSwitch &switch_policy)
        {
            (void)branches;
            (void)predicate;
            (void)switch_policy;
            return false;
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
