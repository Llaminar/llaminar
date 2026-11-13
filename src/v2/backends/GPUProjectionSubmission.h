/**
 * @file GPUProjectionSubmission.h
 * @brief Context-owned fork/join resources and exclusive native capture scopes.
 *
 * Streams remain native capture members until capture ends. One context owns
 * this fixed resource set; a host lease therefore spans an entire capture or
 * eager fork/join submission, never just one recorded projection. No physical
 * capacity is estimated or accounted here: backend opaque admission owns it.
 */
#pragma once
#include <array>
#include <exception>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

namespace llaminar2
{
    class IWorkerGPUContext;
    /** @brief Native interval protected by a projection submission lease. */
    enum class GPUProjectionSubmissionKind { Eager, Capture };

    /** @brief Backend-admitted native population, fixed for one context generation. */
    enum class GPUProjectionStreamCount { Four = 4, Eight = 8 };

    /** @brief Backend operations on exact unpublished context-owned handles. */
    class GPUProjectionResourceOperations
    {
    public:
        virtual ~GPUProjectionResourceOperations() = default;
        /** @brief Create a nondefault nonblocking stream, returning null on failure. */
        virtual void *createProjectionStream() = 0;
        /** @brief Create a dependency-only event, returning null on failure. */
        virtual void *createProjectionEvent() = 0;
        /** @brief Destroy the exact stream; false means retirement is incomplete. */
        virtual bool destroyProjectionStream(void *) noexcept = 0;
        /** @brief Destroy the exact event; false means retirement is incomplete. */
        virtual bool destroyProjectionEvent(void *) noexcept = 0;
    };

    /** @brief Fully constructed resources borrowed only through an active lease. */
    struct GPUProjectionResourceView
    {
        static constexpr int MAX_STREAMS = 8;
        std::array<void *, MAX_STREAMS> streams{};
        std::array<void *, MAX_STREAMS> completion{};
        void *quant_ready = nullptr;
        const int count;
        /** @brief Bind exact admitted population; spare array slots own no handles. */
        explicit GPUProjectionResourceView(
            GPUProjectionStreamCount population = GPUProjectionStreamCount::Eight)
            : count(static_cast<int>(population))
        {
            if (population != GPUProjectionStreamCount::Four && population != GPUProjectionStreamCount::Eight)
                throw std::invalid_argument("Projection stream population must be exactly four or eight");
        }
    };
    class GPUProjectionResources;

    /** @brief Move-only host submission ownership; does not synchronize a GPU. */
    class GPUProjectionSubmission final
    {
    public:
        GPUProjectionSubmission() = default;
        GPUProjectionSubmission(const GPUProjectionSubmission &) = delete;
        GPUProjectionSubmission &operator=(const GPUProjectionSubmission &) = delete;
        /** @brief Transfer ownership on the recording/submission thread. */
        GPUProjectionSubmission(GPUProjectionSubmission &&other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), context_owner_(std::move(other.context_owner_)),
              context_owner_bound_(other.context_owner_bound_), lock_(std::move(other.lock_)),
              owns_interval_(other.owns_interval_), thread_(other.thread_) {}
        GPUProjectionSubmission &operator=(GPUProjectionSubmission &&) = delete;
        /** @brief Release only the interval this token opened. */
        ~GPUProjectionSubmission();
        /** @brief Obtain complete handles; capture itself may begin before preparation. */
        [[nodiscard]] const GPUProjectionResourceView &resources() const;
    private:
        friend class GPUProjectionResources;
        friend class IWorkerGPUContext;
        GPUProjectionSubmission(GPUProjectionResources &owner,
                                std::unique_lock<std::recursive_mutex> lock,
                                bool owns_interval)
            : owner_(&owner), lock_(std::move(lock)), owns_interval_(owns_interval),
              thread_(std::this_thread::get_id()) {}
        GPUProjectionResources *owner_ = nullptr;
        std::shared_ptr<IWorkerGPUContext> context_owner_;
        bool context_owner_bound_ = false;
        std::unique_lock<std::recursive_mutex> lock_;
        bool owns_interval_ = false;
        std::thread::id thread_{};
    };

    /** @brief One exact worker-context generation's projection resource owner. */
    class GPUProjectionResources final
    {
    public:
        /** @brief Bind the exact context; device-free standalone replay has no context. */
        explicit GPUProjectionResources(
            IWorkerGPUContext *context = nullptr,
            GPUProjectionStreamCount population = GPUProjectionStreamCount::Eight)
            : context_identity_(context), view_(population) {}
        GPUProjectionResources(const GPUProjectionResources &) = delete;
        GPUProjectionResources &operator=(const GPUProjectionResources &) = delete;
        /** @brief Native handles must be retired while their context is still alive. */
        ~GPUProjectionResources() { if (ready_ || active_) std::terminate(); }

        /** @brief Construct cold resources atomically, cleaning every partial handle. */
        void prepare(GPUProjectionResourceOperations &operations)
        {
            std::lock_guard lock(mutex_);
            if (retiring_ || active_) throw std::logic_error("Projection preparation requires an idle live context");
            if (ready_) return;
            try
            {
                for (int i = 0; i < view_.count; ++i)
                {
                    view_.streams[i] = operations.createProjectionStream();
                    if (!view_.streams[i]) throw std::runtime_error("Projection stream construction failed");
                    view_.completion[i] = operations.createProjectionEvent();
                    if (!view_.completion[i]) throw std::runtime_error("Projection completion event construction failed");
                }
                view_.quant_ready = operations.createProjectionEvent();
                if (!view_.quant_ready) throw std::runtime_error("Projection fork event construction failed");
                ready_ = true;
            }
            catch (...) { destroyHandles(operations); throw; }
        }

        /**
         * @brief Claim a whole capture/eager interval or borrow its matching capture.
         * @param root Exact non-null root stream.
         * @param kind Native capture or a complete eager fork/join submission.
         * @param native_capturing Actual root stream capture status for eager callers.
         */
        [[nodiscard]] GPUProjectionSubmission acquire(
            void *root, GPUProjectionSubmissionKind kind, bool native_capturing = false)
        {
            if (!root) throw std::invalid_argument("Projection submission requires an exact non-null root stream");
            std::unique_lock lock(mutex_);
            if (retiring_) throw std::logic_error("Projection context is retiring");
            if (active_)
            {
                if (kind != GPUProjectionSubmissionKind::Eager ||
                    kind_ != GPUProjectionSubmissionKind::Capture || root != root_ ||
                    thread_ != std::this_thread::get_id() || !native_capturing)
                    throw std::logic_error("Projection submission does not match its whole native capture owner");
                if (!ready_) throw std::logic_error("Projection resources were not prepared before capture");
                ++borrowers_;
                return GPUProjectionSubmission(*this, std::move(lock), false);
            }
            if (native_capturing) throw std::logic_error("Standalone native capture cannot borrow shared projection resources");
            if (kind == GPUProjectionSubmissionKind::Eager && !ready_)
                throw std::logic_error("Projection resources require explicit cold preparation");
            active_ = true; kind_ = kind; root_ = root; thread_ = std::this_thread::get_id();
            return GPUProjectionSubmission(*this, std::move(lock), true);
        }

        /**
         * @brief Run cold named-resource maintenance without recycling projection handles.
         *
         * The same mutex excludes native capture begin until the complete operation
         * returns. Marking the interval active also rejects recursive acquisition
         * on this host thread; readiness, population and native handles never change.
         * @param operation Exact context-owned maintenance action; may throw.
         */
        template<class Operation>
        void withIdleOperation(Operation &&operation)
        {
            std::unique_lock lock(mutex_, std::try_to_lock);
            if (!lock.owns_lock() || active_)
                throw std::logic_error("Auxiliary reset requires an idle projection context");
            active_ = true;
            try { std::forward<Operation>(operation)(); }
            catch (...) { active_ = false; throw; }
            active_ = false;
        }

        /** @brief Fail before removing a context whose exact submission is live. */
        void assertRetirable()
        {
            std::unique_lock lock(mutex_, std::try_to_lock);
            if (!lock.owns_lock() || active_)
                throw std::logic_error("Cannot retire a context with a live projection submission");
        }
        /** @brief Close acquisition before the context pool removes this generation. */
        void beginRetirement()
        {
            std::unique_lock lock(mutex_, std::try_to_lock);
            if (!lock.owns_lock() || active_)
                throw std::logic_error("Cannot retire a context with a live projection submission");
            retiring_ = true;
        }
        /** @brief Destroy fixed handles only during exact context cleanup; never hide failure. */
        void retire(GPUProjectionResourceOperations &operations)
        {
            std::unique_lock lock(mutex_, std::try_to_lock);
            if (!lock.owns_lock() || active_)
                throw std::logic_error("Cannot reset a context with a live projection submission");
            destroyHandles(operations); ready_ = false;
        }
    private:
        friend class GPUProjectionSubmission;
        friend class IWorkerGPUContext;
        /** @brief Each published or partially built native handle has exactly one release. */
        void destroyHandles(GPUProjectionResourceOperations &operations) noexcept
        {
            if (view_.quant_ready && !operations.destroyProjectionEvent(view_.quant_ready)) std::terminate();
            view_.quant_ready = nullptr;
            for (int i = view_.count - 1; i >= 0; --i)
            {
                if (view_.completion[i] && !operations.destroyProjectionEvent(view_.completion[i])) std::terminate();
                view_.completion[i] = nullptr;
                if (view_.streams[i] && !operations.destroyProjectionStream(view_.streams[i])) std::terminate();
                view_.streams[i] = nullptr;
            }
        }
        IWorkerGPUContext *const context_identity_;
        std::recursive_mutex mutex_;
        GPUProjectionResourceView view_;
        bool ready_ = false, active_ = false, retiring_ = false;
        unsigned borrowers_ = 0;
        GPUProjectionSubmissionKind kind_ = GPUProjectionSubmissionKind::Eager;
        void *root_ = nullptr;
        std::thread::id thread_{};
    };
    inline GPUProjectionSubmission::~GPUProjectionSubmission()
    {
        if (!owner_) return;
        if (thread_ != std::this_thread::get_id()) std::terminate();
        if (owns_interval_)
        {
            if (owner_->borrowers_ != 0) std::terminate();
            owner_->active_ = false; owner_->root_ = nullptr; owner_->thread_ = {};
        }
        else
        {
            if (owner_->borrowers_ == 0) std::terminate();
            --owner_->borrowers_;
        }
    }
    inline const GPUProjectionResourceView &GPUProjectionSubmission::resources() const
    {
        if (!owner_ || thread_ != std::this_thread::get_id() || !owner_->ready_)
            throw std::logic_error("Projection resource view requires a complete live submission lease");
        return owner_->view_;
    }
}
