/**
 * @file GraphCaptureGuard.h
 * @brief Structural ownership for GPU graph recording and its data dependencies.
 *
 * Native CUDA/HIP graph capture records work without executing that work.  A
 * tensor written by an earlier recorded stage is therefore a valid input to a
 * later recorded stage even though the tensor's globally visible coherence
 * state must not claim that the write has completed.  This file keeps those two
 * facts separate:
 *
 * - GraphCaptureDependencyLedger describes the exact topological stage order
 *   and graph-internal inputs for one capture transaction.
 * - GraphCaptureGuard publishes that ledger only to the recording thread.
 * - ScopedGraphCaptureStage advances the ledger through the canonical stage
 *   runner.
 * - ScopedBackendGraphCapture owns the indivisible backend begin/end interval.
 *
 * The resulting contract makes an unrecorded internal producer, a different
 * stream, a different device, or an out-of-order stage a hard failure.  It also
 * avoids pretending that recorded-but-unlaunched bytes are globally device
 * authoritative.
 */

#pragma once

#include "../../../backends/DeviceId.h"
#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{
    class TensorBase;

    /**
     * @brief Precomputed producer/consumer proof for one native graph capture.
     *
     * The graph executor builds this ledger before beginCapture(), while arena
     * contracts and BufferIds are still available.  During recording,
     * TransferEngine only sees tensor identities, so each internal input stores
     * the canonical transfer-storage owner and the earlier stage index that
     * produces it.  No container grows inside the capture window.
     */
    class GraphCaptureDependencyLedger final
    {
    public:
        /**
         * @brief Authority required for declared inputs entering this capture unit.
         *
         * A normal capture immediately submits the resulting executable, so all
         * graph-frontier inputs must already own valid bytes. Setup
         * materialization records an executable before request admission and may
         * therefore consume only the permanent address of a declared arena
         * input. Tensor identities absent from StagePlan::external_inputs remain
         * strict in both modes; this keeps weights and undeclared stage metadata
         * from borrowing the setup exception.
         */
        enum class ExternalInputAuthority : uint8_t
        {
            RequireReadyBytes = 0, ///< The executable may launch in this transaction.
            BindDeclaredAddressesOnly, ///< Setup records declared frontier pointers only.
        };

        /** @brief One tensor consumed from an earlier stage in this transaction. */
        struct InternalInput
        {
            const TensorBase *tensor = nullptr; ///< Canonical storage owner.
            size_t producer_stage_index = 0;    ///< Earlier stage that records the write.
        };

        /** @brief Immutable dependency description for one topological stage. */
        struct StagePlan
        {
            const void *stage_identity = nullptr; ///< Exact IComputeStage object identity.
            std::string stage_name;               ///< Stable diagnostic name.
            std::vector<const TensorBase *> external_inputs; ///< Inputs joined before capture.
            std::vector<InternalInput> internal_inputs;       ///< Inputs produced in this graph.
            /**
             * Inputs produced by an earlier child of the same retained parent.
             * The child graph records only a stable pointer read; parent
             * composition later installs the exact child-to-child device edge.
             */
            std::vector<const TensorBase *> retained_parent_inputs;
            std::vector<const TensorBase *> outputs;          ///< Declared arena write owners.
        };

        /** @brief How TransferEngine must validate one current-stage input. */
        enum class InputDisposition
        {
            StrictExternal, ///< Require globally valid residency and a prejoined event.
            InternalRecorded, ///< Earlier producer is ordered by this native graph stream.
            RetainedParentRecorded, ///< Earlier retained child is ordered by the composed parent.
            SetupAddressOnlyExternal, ///< Declared setup frontier owns storage but no request bytes.
        };

        /**
         * @brief Construct a frozen dependency plan before native capture begins.
         *
         * @param device Exact GPU owning the capture transaction.
         * @param stream Exact non-null native capture stream.
         * @param stages Stages in the precise order in which they will be recorded.
         * @param context Stable diagnostic label for lifecycle failures.
         * @param external_input_authority Whether declared graph-frontier reads
         *        require live bytes or setup may bind their stable addresses.
         */
        GraphCaptureDependencyLedger(
            DeviceId device,
            void *stream,
            std::vector<StagePlan> stages,
            std::string context,
            ExternalInputAuthority external_input_authority =
                ExternalInputAuthority::RequireReadyBytes)
            : device_(device),
              stream_(stream),
              stages_(std::move(stages)),
              context_(std::move(context)),
              external_input_authority_(external_input_authority)
        {
            if (!device_.is_gpu())
                throw std::invalid_argument(
                    "GraphCaptureDependencyLedger requires a GPU device");
            if (!stream_)
                throw std::invalid_argument(
                    "GraphCaptureDependencyLedger requires an exact non-null stream");
            for (const auto &stage : stages_)
            {
                if (!stage.stage_identity)
                    throw std::invalid_argument(
                        "GraphCaptureDependencyLedger contains a null stage identity");
                max_stage_outputs_ =
                    std::max(max_stage_outputs_, stage.outputs.size());
            }
            current_stage_publications_.resize(max_stage_outputs_, 0);
        }

        /** @brief Arm the frozen plan for one recording pass. */
        void beginCapture()
        {
            if (active_)
                throw std::logic_error(
                    "GraphCaptureDependencyLedger cannot begin twice: " + context_);
            active_ = true;
            failed_ = false;
            stage_cursor_ = 0;
            current_stage_active_ = false;
        }

        /** @brief Clear thread-local recording state without publishing tensor authority. */
        void endCapture() noexcept
        {
            active_ = false;
            current_stage_active_ = false;
            stage_cursor_ = 0;
        }

        /**
         * @brief Enter the next exact stage in the precomputed topological plan.
         * @param stage_identity Exact IComputeStage object about to record work.
         */
        void beginStage(const void *stage_identity)
        {
            if (!active_ || current_stage_active_ || stage_cursor_ >= stages_.size())
                throw std::logic_error(
                    "Graph capture stage lifecycle is invalid: " + context_);
            const auto &expected = stages_[stage_cursor_];
            if (expected.stage_identity != stage_identity)
            {
                throw std::logic_error(
                    "Graph capture stage order mismatch: expected '" +
                    expected.stage_name + "' in " + context_);
            }
            std::fill(
                current_stage_publications_.begin(),
                current_stage_publications_.begin() +
                    static_cast<std::ptrdiff_t>(expected.outputs.size()),
                uint8_t{0});
            current_stage_active_ = true;
        }

        /** @brief Commit the current stage after every required launch was recorded. */
        void completeStage(const void *stage_identity)
        {
            requireCurrentStage(stage_identity);
            current_stage_active_ = false;
            ++stage_cursor_;
        }

        /** @brief Mark a failed stage so capture closure can remain exception-safe. */
        void abortCurrentStage() noexcept
        {
            if (current_stage_active_)
            {
                current_stage_active_ = false;
                failed_ = true;
            }
        }

        /**
         * @brief Classify one tensor read made by the currently recording stage.
         *
         * An internal input is accepted when either an earlier stage owns its
         * producer edge or the current compound stage has already published that
         * exact declared output after recording its producer sub-operation.
         * Unknown tensors, including weights and stage-owned metadata, remain
         * strict external inputs.
         */
        InputDisposition classifyInput(
            const TensorBase *tensor,
            DeviceId device,
            void *stream) const
        {
            requireCurrentTransaction(device, stream);
            if (!current_stage_active_ || stage_cursor_ >= stages_.size())
                throw std::logic_error(
                    "Graph capture input was requested outside a stage: " + context_);

            const auto &stage = stages_[stage_cursor_];
            for (size_t output_index = 0;
                 output_index < stage.outputs.size();
                 ++output_index)
            {
                if (stage.outputs[output_index] != tensor)
                    continue;
                if (current_stage_publications_[output_index] != 0)
                    return InputDisposition::InternalRecorded;

                /*
                 * Inouts legitimately read their prior generation before the
                 * current stage publishes a replacement. Pure outputs do not:
                 * consuming one before publication is an invalid intra-stage
                 * producer/consumer ordering, not an external dependency.
                 */
                const bool declared_read =
                    std::find(stage.external_inputs.begin(),
                              stage.external_inputs.end(),
                              tensor) != stage.external_inputs.end() ||
                    std::any_of(
                        stage.internal_inputs.begin(),
                        stage.internal_inputs.end(),
                        [tensor](const InternalInput &input)
                        {
                            return input.tensor == tensor;
                        }) ||
                    std::find(
                        stage.retained_parent_inputs.begin(),
                        stage.retained_parent_inputs.end(),
                        tensor) != stage.retained_parent_inputs.end();
                if (!declared_read)
                {
                    throw std::logic_error(
                        "Graph capture stage '" + stage.stage_name +
                        "' consumed a declared output before publishing its "
                        "producer sub-operation (" + context_ + ")");
                }
                break;
            }

            for (const auto &input : stage.internal_inputs)
            {
                if (input.tensor != tensor)
                    continue;
                if (input.producer_stage_index >= stage_cursor_)
                {
                    throw std::logic_error(
                        "Graph capture internal input precedes its producer in stage '" +
                        stage.stage_name + "' (" + context_ + ")");
                }
                return InputDisposition::InternalRecorded;
            }
            if (std::find(
                    stage.retained_parent_inputs.begin(),
                    stage.retained_parent_inputs.end(),
                    tensor) != stage.retained_parent_inputs.end())
            {
                return InputDisposition::RetainedParentRecorded;
            }
            if (external_input_authority_ ==
                    ExternalInputAuthority::BindDeclaredAddressesOnly &&
                std::find(
                    stage.external_inputs.begin(),
                    stage.external_inputs.end(),
                    tensor) != stage.external_inputs.end())
            {
                /*
                 * The setup owner has proven stable exact-device storage for
                 * this typed arena frontier before beginCapture(). The kernel
                 * only records that address now; transaction zero will perform
                 * the strict producer/event preflight before any launch.
                 */
                return InputDisposition::SetupAddressOnlyExternal;
            }
            return InputDisposition::StrictExternal;
        }

        /**
         * @brief Record one declared stage-local publication without an event.
         *
         * Stage kernels historically called publishDeviceWrite() immediately
         * after enqueueing work.  Inside native capture that work has only been
         * recorded, so the transaction records the exact intra-stage producer
         * edge and defers real authority/event publication to the post-launch
         * graph boundary. The preallocated publication bitmap makes this legal in
         * a native capture window without dynamic storage or event nodes.
         */
        void recordStagePublication(
            const TensorBase *tensor,
            DeviceId device,
            void *stream)
        {
            requireCurrentTransaction(device, stream);
            if (!current_stage_active_ || stage_cursor_ >= stages_.size())
                throw std::logic_error(
                    "Graph capture publication occurred outside a stage: " + context_);

            const auto &stage = stages_[stage_cursor_];
            for (size_t output_index = 0;
                 output_index < stage.outputs.size();
                 ++output_index)
            {
                if (stage.outputs[output_index] != tensor)
                    continue;
                current_stage_publications_[output_index] = 1;
                return;
            }

            throw std::logic_error(
                "Graph capture stage '" + stage.stage_name +
                "' published a tensor absent from its declared output contract (" +
                context_ + ")");
        }

        /** @brief True when every planned stage was recorded successfully. */
        bool allStagesRecorded() const noexcept
        {
            return !current_stage_active_ && stage_cursor_ == stages_.size();
        }

        /** @brief True when a stage launch rejected the current transaction. */
        bool failed() const noexcept { return failed_; }

    private:
        void requireCurrentTransaction(DeviceId device, void *stream) const
        {
            if (!active_)
                throw std::logic_error(
                    "Graph capture dependency ledger is not active: " + context_);
            if (device != device_ || stream != stream_)
            {
                throw std::logic_error(
                    "Graph capture dependency used a different device or stream: " +
                    context_ + " expected_device=" + device_.toString() +
                    " observed_device=" + device.toString() +
                    " expected_stream=" +
                    std::to_string(reinterpret_cast<std::uintptr_t>(stream_)) +
                    " observed_stream=" +
                    std::to_string(reinterpret_cast<std::uintptr_t>(stream)));
            }
        }

        void requireCurrentStage(const void *stage_identity) const
        {
            if (!active_ || !current_stage_active_ || stage_cursor_ >= stages_.size() ||
                stages_[stage_cursor_].stage_identity != stage_identity)
            {
                throw std::logic_error(
                    "Graph capture stage completion does not match the active stage: " +
                    context_);
            }
        }

        DeviceId device_;
        void *stream_ = nullptr;
        std::vector<StagePlan> stages_;
        std::string context_;
        ExternalInputAuthority external_input_authority_ =
            ExternalInputAuthority::RequireReadyBytes;
        std::vector<uint8_t> current_stage_publications_;
        size_t max_stage_outputs_ = 0;
        size_t stage_cursor_ = 0;
        bool active_ = false;
        bool current_stage_active_ = false;
        bool failed_ = false;
    };

    /**
     * @brief Thread-local flag indicating that the current thread is inside
     *        a HIP/CUDA graph capture recording window.
     *
     * During graph capture (between beginCapture/endCapture), many HIP/CUDA
     * operations are illegal:
     *   - hipDeviceSynchronize / cudaDeviceSynchronize
     *   - hipStreamSynchronize / cudaStreamSynchronize (on capture stream)
     *   - hipMemcpy (synchronous variants)
     *   - hipEventSynchronize
     *
     * Code that might call these operations can check isGraphCaptureActive()
     * and reject the invalid operation with a precise diagnostic. It must not
     * substitute a weaker execution path.
     *
     * Usage:
     *   // In a low-level capture harness (production uses
     *   // ScopedBackendGraphCapture plus DeviceGraphExecutor::runStage()):
     *   {
     *       GraphCaptureGuard guard;  // sets flag true
     *       record_kernel_body_on_the_exact_stream();
     *   }  // guard destructor sets flag false
     *
     *   // In tensor code:
     *   if (isGraphCaptureActive() && operation_requires_host_wait)
     *       throw std::logic_error("host wait is forbidden during capture");
     */

    /// Thread-local flag: true when inside a graph capture recording window.
    inline thread_local bool tls_graph_capture_active = false;

    /// Exact dependency ledger for the current production capture transaction.
    inline thread_local GraphCaptureDependencyLedger *
        tls_graph_capture_dependency_ledger = nullptr;

    /// Query whether the current thread is recording into a GPU graph.
    inline bool isGraphCaptureActive() { return tls_graph_capture_active; }

    /** @brief Return the current typed dependency transaction, if one is installed. */
    inline GraphCaptureDependencyLedger *currentGraphCaptureDependencyLedger()
    {
        return tls_graph_capture_dependency_ledger;
    }

    /**
     * @brief RAII guard that sets the graph-capture-active flag for the
     *        duration of a capture recording window.
     */
    class GraphCaptureGuard
    {
    public:
        explicit GraphCaptureGuard(
            GraphCaptureDependencyLedger *dependency_ledger = nullptr)
            : prev_(tls_graph_capture_active),
              prev_dependency_ledger_(tls_graph_capture_dependency_ledger),
              owned_dependency_ledger_(dependency_ledger)
        {
            if (dependency_ledger && prev_dependency_ledger_ &&
                dependency_ledger != prev_dependency_ledger_)
            {
                throw std::logic_error(
                    "Nested GPU graph capture cannot replace the active dependency ledger");
            }
            if (owned_dependency_ledger_)
                owned_dependency_ledger_->beginCapture();
            tls_graph_capture_active = true;
            tls_graph_capture_dependency_ledger =
                dependency_ledger ? dependency_ledger : prev_dependency_ledger_;
        }

        ~GraphCaptureGuard()
        {
            if (owned_dependency_ledger_)
                owned_dependency_ledger_->endCapture();
            tls_graph_capture_dependency_ledger = prev_dependency_ledger_;
            tls_graph_capture_active = prev_;
        }

        // Non-copyable, non-movable
        GraphCaptureGuard(const GraphCaptureGuard &) = delete;
        GraphCaptureGuard &operator=(const GraphCaptureGuard &) = delete;

    private:
        bool prev_; ///< Previous graph-capture flag (for nested guard support).
        GraphCaptureDependencyLedger *prev_dependency_ledger_ = nullptr;
        GraphCaptureDependencyLedger *owned_dependency_ledger_ = nullptr;
    };

    /**
     * @brief RAII stage cursor for the canonical executor while capture is active.
     *
     * A failed or exceptional stage aborts the ledger entry in the destructor.
     * Successful callers must invoke complete() exactly once after every launch
     * and publication check has succeeded.
     */
    class ScopedGraphCaptureStage final
    {
    public:
        explicit ScopedGraphCaptureStage(const void *stage_identity)
            : ledger_(currentGraphCaptureDependencyLedger()),
              stage_identity_(stage_identity)
        {
            if (ledger_)
                ledger_->beginStage(stage_identity_);
        }

        ~ScopedGraphCaptureStage()
        {
            if (ledger_ && !completed_)
                ledger_->abortCurrentStage();
        }

        ScopedGraphCaptureStage(const ScopedGraphCaptureStage &) = delete;
        ScopedGraphCaptureStage &operator=(const ScopedGraphCaptureStage &) = delete;

        /** @brief Commit the current stage to the capture dependency sequence. */
        void complete()
        {
            if (!ledger_)
                return;
            if (completed_)
                throw std::logic_error(
                    "ScopedGraphCaptureStage cannot complete twice");
            ledger_->completeStage(stage_identity_);
            completed_ = true;
        }

    private:
        GraphCaptureDependencyLedger *ledger_ = nullptr;
        const void *stage_identity_ = nullptr;
        bool completed_ = false;
    };

    /**
     * @brief Own one exact HIP/CUDA backend stream-capture transaction.
     *
     * A capture interval is indivisible: after `beginCapture()` succeeds,
     * exactly one `endCapture()` must execute before the stream can be queried,
     * synchronized, destroyed, or reused. Keeping the worker-context flag,
     * thread-local guard, and backend graph owner in separate call-site code
     * allowed early returns to strand a native stream in capture mode.
     *
     * This owner makes that lifecycle structural. `finish()` closes the normal
     * interval, while the destructor closes one abandoned by an exception or
     * early return. Backend `endCapture()` failure is process-fatal because the
     * native stream's ownership state is then unknowable; eager recovery or a
     * stream synchronization would itself be illegal.
     */
    class ScopedBackendGraphCapture final
    {
    public:
        /**
         * @brief Bind a software and backend capture owner.
         *
         * @param gpu_context Worker context whose capture-active state protects
         *        backend operations on this device.
         * @param capture Backend graph object bound to the exact native stream.
         * @param operation Stable diagnostic name for fatal lifecycle failures.
         */
        ScopedBackendGraphCapture(
            IWorkerGPUContext &gpu_context,
            IGPUGraphCapture &capture,
            std::string operation,
            GraphCaptureDependencyLedger *dependency_ledger = nullptr)
            : gpu_context_(&gpu_context),
              capture_(capture),
              operation_(std::move(operation)),
              dependency_ledger_(dependency_ledger)
        {
        }

        /**
         * @brief Bind a backend capture when no worker-context flag is available.
         *
         * Legacy single-graph callers still receive the same structural
         * begin/end pairing and thread-local protection. New execution paths
         * should pass the worker context through the primary constructor.
         */
        ScopedBackendGraphCapture(
            IGPUGraphCapture &capture,
            std::string operation,
            GraphCaptureDependencyLedger *dependency_ledger = nullptr)
            : capture_(capture),
              operation_(std::move(operation)),
              dependency_ledger_(dependency_ledger)
        {
        }

        ~ScopedBackendGraphCapture()
        {
            if (!active_)
                return;

            clearSoftwareCaptureState();
            active_ = false;
            if (!capture_.endCapture())
            {
                LOG_ERROR(
                    "[ScopedBackendGraphCapture] Fatal failure closing "
                    "abandoned backend graph capture for "
                    << operation_);
                std::terminate();
            }
        }

        ScopedBackendGraphCapture(
            const ScopedBackendGraphCapture &) = delete;
        ScopedBackendGraphCapture &operator=(
            const ScopedBackendGraphCapture &) = delete;

        /**
         * @brief Enter backend stream capture and publish software state.
         *
         * @return true after the backend accepted capture; false before any
         *         capture interval became active.
         */
        bool begin()
        {
            if (active_)
                throw std::logic_error(
                    "ScopedBackendGraphCapture cannot begin twice");

            if (gpu_context_)
            {
                gpu_context_->setGraphCaptureActive(true);
            }
            if (!capture_.beginCapture())
            {
                if (gpu_context_)
                    gpu_context_->setGraphCaptureActive(false);
                return false;
            }

            /*
             * Publish backend ownership before constructing software guards. If
             * ledger validation throws, stack unwinding must still make this
             * object's destructor close the already-open native capture.
             */
            active_ = true;
            capture_guard_.emplace(dependency_ledger_);
            return true;
        }

        /**
         * @brief Close the exact backend capture interval.
         *
         * Backend failure leaves the native stream in an unknowable state, so
         * this method terminates instead of exposing a recoverable result.
         */
        void finish()
        {
            if (!active_)
                throw std::logic_error(
                    "ScopedBackendGraphCapture cannot finish an inactive capture");

            const bool incomplete_dependency_plan =
                dependency_ledger_ && !dependency_ledger_->failed() &&
                !dependency_ledger_->allStagesRecorded();

            clearSoftwareCaptureState();
            active_ = false;
            if (!capture_.endCapture())
            {
                LOG_ERROR(
                    "[ScopedBackendGraphCapture] Fatal backend endCapture "
                    "failure for "
                    << operation_);
                std::terminate();
            }
            if (incomplete_dependency_plan)
            {
                throw std::logic_error(
                    "GPU graph capture closed before every dependency-planned stage "
                    "was recorded for " +
                    operation_);
            }
        }

    private:
        void clearSoftwareCaptureState() noexcept
        {
            capture_guard_.reset();
            if (gpu_context_)
                gpu_context_->setGraphCaptureActive(false);
        }

        IWorkerGPUContext *gpu_context_ = nullptr;
        IGPUGraphCapture &capture_;
        std::string operation_;
        GraphCaptureDependencyLedger *dependency_ledger_ = nullptr;
        std::optional<GraphCaptureGuard> capture_guard_;
        bool active_ = false;
    };

} // namespace llaminar2
