/**
 * @file ICollectiveCoordinator.h
 * @brief Abstract base interface for collective communication coordinators
 * @author David Sanftenberg
 * @date February 2026
 *
 * Each collective backend gets a dedicated coordinator thread that owns all
 * communication state and serializes operations. This solves the threading
 * mismatch where comms were created on worker threads but used from the caller
 * thread.
 *
 * Architecture:
 * - Each coordinator owns a dedicated thread that serializes all collective
 *   operations. This ensures proper threading semantics for NCCL/RCCL group
 *   operations and prevents context corruption.
 * - Work is submitted via submitAndWait() or submitAsync() which queue
 *   lambdas to the coordinator thread.
 * - Implementations override enqueueWork() to provide the thread-safe queue.
 *
 * Thread safety: All public methods are thread-safe (work is queued).
 *
 * @see NCCLCoordinator, RCCLCoordinator
 */

#pragma once

#include <functional>
#include <future>
#include <memory>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Abstract base for collective communication coordinators
     *
     * Each coordinator owns a dedicated thread that serializes all collective
     * operations. This ensures proper threading semantics for NCCL/RCCL group
     * operations and prevents context corruption.
     *
     * Thread safety: All public methods are thread-safe (work is queued).
     *
     * Usage example:
     * @code
     *   // Concrete coordinator (e.g., NCCLCoordinator)
     *   auto coord = std::make_unique<NCCLCoordinator>();
     *   coord->initialize({0, 1});  // GPUs 0 and 1
     *
     *   // From any thread - synchronous submission:
     *   bool success = coord->submitAndWait([&]() {
     *       return performCollectiveOp();
     *   });
     *
     *   // Asynchronous submission:
     *   auto future = coord->submitAsync([&]() {
     *       performCollectiveOp();
     *   });
     *   future.wait();  // Wait when ready
     *
     *   // Device worker synchronization:
     *   cudaStreamWaitEvent(compute_stream, coord->getCompletionEvent(gpu_idx));
     * @endcode
     */
    class ICollectiveCoordinator
    {
    public:
        virtual ~ICollectiveCoordinator() = default;

        // Non-copyable, non-movable (owns thread state)
        ICollectiveCoordinator(const ICollectiveCoordinator &) = delete;
        ICollectiveCoordinator &operator=(const ICollectiveCoordinator &) = delete;
        ICollectiveCoordinator(ICollectiveCoordinator &&) = delete;
        ICollectiveCoordinator &operator=(ICollectiveCoordinator &&) = delete;

        // =========================================================================
        // Lifecycle
        // =========================================================================

        /**
         * @brief Initialize the coordinator with the given device ordinals
         *
         * Spawns the coordinator thread and initializes per-device state
         * (streams, events, communicators).
         *
         * @param device_ordinals Local GPU ordinals to manage (e.g., {0, 1} for GPUs 0 and 1)
         * @return true if initialization succeeded
         */
        virtual bool initialize(const std::vector<int> &device_ordinals) = 0;

        /**
         * @brief Shutdown the coordinator
         *
         * Signals the coordinator thread to exit, waits for pending work to complete,
         * and releases all resources (communicators, streams, events).
         *
         * Safe to call multiple times (idempotent).
         */
        virtual void shutdown() = 0;

        /**
         * @brief Check if the coordinator is initialized and ready
         *
         * @return true if initialize() succeeded and shutdown() hasn't been called
         */
        virtual bool isInitialized() const = 0;

        // =========================================================================
        // Synchronization with Device Workers
        // =========================================================================

        /**
         * @brief Get completion event for a device after last collective
         *
         * Device workers call cudaStreamWaitEvent(stream, getCompletionEvent(device))
         * before launching kernels that depend on collective results.
         *
         * The returned event is recorded after each collective operation completes
         * on the coordinator's internal stream for that device.
         *
         * @param device_idx Local device index (0 to num_devices-1)
         * @return Opaque event handle (cudaEvent_t or hipEvent_t cast to void*)
         * @throws std::out_of_range if device_idx is invalid
         */
        virtual void *getCompletionEvent(int device_idx) const = 0;

        /**
         * @brief Wait for device worker's stream before collective
         *
         * Coordinator calls this internally to ensure all prior device work is done
         * before starting a collective that reads from device buffers.
         *
         * Device workers record an event after their last kernel and pass it here.
         * The coordinator's stream will wait on this event before proceeding.
         *
         * @param device_idx Local device index
         * @param worker_event Event recorded after device worker's last kernel
         */
        virtual void waitForDeviceEvent(int device_idx, void *worker_event) = 0;

        /**
         * @brief Register compute streams for stream-level pre-collective sync
         *
         * When set, the coordinator uses hipStreamWaitEvent to establish a
         * stream-level dependency between each compute stream and its RCCL/NCCL
         * stream, replacing the heavyweight hipDeviceSynchronize() pre-sync.
         *
         * @param compute_streams One compute stream handle per device (same order as device_ordinals)
         */
        virtual void setComputeStreams(const std::vector<void *> &compute_streams) = 0;

        // =========================================================================
        // Submission API
        // =========================================================================

        /**
         * @brief Submit work and wait for completion (synchronous)
         *
         * Queues the work lambda to the coordinator thread and blocks until
         * the work completes. The return value from work() is returned.
         *
         * Thread-safe: May be called from any thread.
         *
         * @tparam F Callable type (lambda, function, etc.)
         * @param work The work to execute on the coordinator thread
         * @return The return value of work()
         *
         * @code
         *   // Submit and get result
         *   bool success = coord->submitAndWait([&]() {
         *       return performAllreduce(buffer, count);
         *   });
         *
         *   // Void work
         *   coord->submitAndWait([&]() {
         *       setupCommunicators();
         *   });
         * @endcode
         */
        template <typename F>
        auto submitAndWait(F &&work) -> decltype(work())
        {
            using ReturnType = decltype(work());

            if constexpr (std::is_void_v<ReturnType>)
            {
                // Void return type - use promise<void>
                std::promise<void> promise;
                auto future = promise.get_future();

                enqueueWork([&promise, work = std::forward<F>(work)]() mutable
                            {
                try {
                    work();
                    promise.set_value();
                } catch (...) {
                    promise.set_exception(std::current_exception());
                } });

                future.get(); // Block until complete (may throw)
            }
            else
            {
                // Non-void return type
                std::promise<ReturnType> promise;
                auto future = promise.get_future();

                enqueueWork([&promise, work = std::forward<F>(work)]() mutable
                            {
                try {
                    promise.set_value(work());
                } catch (...) {
                    promise.set_exception(std::current_exception());
                } });

                return future.get(); // Block and return result (may throw)
            }
        }

        /**
         * @brief Submit work without waiting (asynchronous)
         *
         * Queues the work lambda to the coordinator thread and returns immediately
         * with a future that will hold the result when complete.
         *
         * Thread-safe: May be called from any thread.
         *
         * @tparam F Callable type (lambda, function, etc.)
         * @param work The work to execute on the coordinator thread
         * @return A future that will hold the result of work()
         *
         * @code
         *   // Submit async and continue
         *   auto future = coord->submitAsync([&]() {
         *       return performAllreduce(buffer, count);
         *   });
         *
         *   // Do other work...
         *
         *   // Wait for result when ready
         *   bool success = future.get();
         * @endcode
         */
        template <typename F>
        auto submitAsync(F &&work) -> std::future<decltype(work())>
        {
            using ReturnType = decltype(work());

            // Use shared_ptr<packaged_task> so it can be captured by the lambda
            // and still be valid when the coordinator thread executes it
            auto task = std::make_shared<std::packaged_task<ReturnType()>>(
                std::forward<F>(work));

            auto future = task->get_future();

            enqueueWork([task]()
                        {
                            (*task)(); // Execute the packaged task
                        });

            return future;
        }

    protected:
        // Default constructor for derived classes
        ICollectiveCoordinator() = default;

        /**
         * @brief Enqueue work to the coordinator thread (implementation-specific)
         *
         * Implementations must override this to provide thread-safe queueing
         * to their coordinator thread. The work function must be executed
         * on the coordinator thread in FIFO order.
         *
         * This is called by submitAndWait() and submitAsync() with packaged
         * work that includes promise/future machinery.
         *
         * @param work The work function to execute on the coordinator thread
         */
        virtual void enqueueWork(std::function<void()> work) = 0;
    };

} // namespace llaminar2
