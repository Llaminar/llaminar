/**
 * @file PrefixArchivePersistence.cpp
 * @brief Background FIFO publication for durable prefix puts and tombstones.
 *
 * Completion releases the writer's source alias before publishing its receipt.
 * A cache can then evict its own RAM alias, while PMA remains the sole authority
 * for request-held or DMA-held physical bytes. Failures seal the writer and are
 * published to every outstanding consumer; no mutation is silently retried.
 */
#include "execution/prefix_cache/PrefixArchivePersistence.h"

#include "execution/prefix_cache/DiskPrefixStorageBackend.h"
#include "utils/PerfStatsCollector.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    uint64_t PrefixArchivePersistenceTicket::identity() const noexcept
    {
        return state_ ? state_->identity : 0u;
    }

    PrefixArchiveMutationPhase PrefixArchivePersistenceTicket::phase() const
    {
        if (!state_)
            return PrefixArchiveMutationPhase::Empty;
        std::lock_guard lock(state_->mutex);
        return state_->phase;
    }

    std::shared_ptr<const PrefixArchivePublication>
    PrefixArchivePersistenceTicket::publication() const
    {
        if (!state_)
            return {};
        std::lock_guard lock(state_->mutex);
        return state_->publication;
    }

    std::shared_ptr<const PrefixArchivePublication>
    PrefixArchivePersistenceTicket::waitForPublication() const
    {
        if (!state_)
            throw std::logic_error("cannot join an empty prefix archive receipt");
        std::unique_lock lock(state_->mutex);
        state_->completed.wait(lock, [&] { return state_->publication != nullptr; });
        return state_->publication;
    }

    PrefixArchivePersistence::PrefixArchivePersistence(DiskPrefixStorageBackend &backend)
        : backend_(backend), writer_([this] { run(); })
    {
    }

    PrefixArchivePersistence::~PrefixArchivePersistence()
    {
        {
            std::lock_guard lock(mutex_);
            if (lifecycle_ == Lifecycle::Accepting)
                lifecycle_ = Lifecycle::Sealing;
        }
        changed_.notify_all();
        if (writer_.joinable())
            writer_.join();
    }

    PrefixArchivePersistenceTicket PrefixArchivePersistence::enqueueLocked(
        std::variant<Put, Retire> operation)
    {
        if (lifecycle_ != Lifecycle::Accepting)
            throw std::runtime_error(failure_.empty()
                ? "prefix archive writer has sealed mutation admission" : failure_);
        if (next_identity_ == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("prefix archive persistence identity overflow");
        auto state = std::make_shared<PrefixArchivePersistenceTicket::State>(next_identity_++);
        pending_.push_back({std::move(operation), state});
        changed_.notify_all();
        return PrefixArchivePersistenceTicket(std::move(state));
    }

    PrefixArchivePersistenceTicket PrefixArchivePersistence::write(PrefixBlockHandle handle)
    {
        if (!handle.valid() || !backend_.canStore(handle.total_bytes))
            throw std::invalid_argument("invalid prefix owner or payload exceeds durable tier capacity");
        std::lock_guard lock(mutex_);
        return enqueueLocked(Put{std::move(handle)});
    }

    PrefixArchivePersistenceTicket PrefixArchivePersistence::retire(const PrefixCacheKey &key)
    {
        if (!key.valid())
            throw std::invalid_argument("invalid prefix key for durable retirement");
        std::lock_guard lock(mutex_);
        // The queue is the only unstarted-owner authority. The worker retires
        // their aliases; the request never frees a potentially large payload.
        for (auto &mutation : pending_)
        {
            if (auto *put = std::get_if<Put>(&mutation.operation);
                put && put->handle.key == key)
                put->disposition = PutDisposition::Superseded;
        }
        return enqueueLocked(Retire{key});
    }

    void PrefixArchivePersistence::publish(
        const std::shared_ptr<PrefixArchivePersistenceTicket::State> &receipt,
        PrefixArchivePublication result)
    {
        auto publication = std::make_shared<const PrefixArchivePublication>(std::move(result));
        {
            std::lock_guard lock(receipt->mutex);
            if (!receipt->publication)
            {
                receipt->publication = std::move(publication);
                receipt->phase = PrefixArchiveMutationPhase::Published;
            }
        }
        receipt->completed.notify_all();
    }

    bool PrefixArchivePersistence::waitUntilIdle(std::string *error)
    {
        std::unique_lock lock(mutex_);
        changed_.wait(lock, [&] { return pending_.empty() && !executing_; });
        if (!failure_.empty() && error)
            *error = failure_;
        return failure_.empty();
    }

    void PrefixArchivePersistence::run() noexcept
    {
        while (true)
        {
            std::optional<Mutation> mutation;
            std::string prior_failure;
            {
                std::unique_lock lock(mutex_);
                changed_.wait(lock, [&]
                {
                    return !pending_.empty() || lifecycle_ != Lifecycle::Accepting;
                });
                if (pending_.empty())
                    return;
                mutation.emplace(std::move(pending_.front()));
                pending_.pop_front();
                executing_ = mutation->receipt->identity;
                prior_failure = failure_;
            }

            std::optional<PrefixArchivePublication> result;
            std::string error;
            try
            {
                if (!prior_failure.empty())
                    result = PrefixArchivePersistenceFailure{prior_failure};
                else if (auto *put = std::get_if<Put>(&mutation->operation))
                {
                    {
                        std::lock_guard lock(mutation->receipt->mutex);
                        mutation->receipt->phase = PrefixArchiveMutationPhase::Executing;
                    }
                    if (put->disposition == PutDisposition::Superseded)
                    {
                        // Cancellation is terminal only after mutation.reset()
                        // releases the original PMA alias below. Publishing it
                        // on the enqueueing thread would lie about completion.
                        result = PrefixArchiveCancelledPublication{};
                    }
                    else
                    {
                        PrefixArchiveWritePublication written;
                        if (!backend_.writeBlock(put->handle, &written.disk_handle,
                                                 &written.evicted_keys, &error))
                            throw std::runtime_error(error.empty() ? "prefix archive put failed" : error);
                        written.executor = std::this_thread::get_id();
                        PerfStatsCollector::addCounter(
                            "prefix_archive", "background_write_blocks", 1.0,
                            "prefix_persistence", "CPU");
                        PerfStatsCollector::addCounter(
                            "prefix_archive", "background_write_bytes",
                            static_cast<double>(put->handle.total_bytes),
                            "prefix_persistence", "CPU");
                        result = std::move(written);
                    }
                }
                else
                {
                    {
                        std::lock_guard lock(mutation->receipt->mutex);
                        mutation->receipt->phase = PrefixArchiveMutationPhase::Executing;
                    }
                    const auto key = std::get<Retire>(mutation->operation).key;
                    PrefixBlockHandle handle;
                    handle.key = key;
                    handle.tier = PrefixStorageTier::Disk;
                    if (!backend_.release(handle))
                        throw std::runtime_error("prefix archive tombstone publication failed");
                    result = PrefixArchiveRetirementPublication{key};
                }
            }
            catch (const std::exception &exception)
            {
                error = exception.what();
                result = PrefixArchivePersistenceFailure{error};
            }
            catch (...)
            {
                error = "unknown prefix archive persistence failure";
                result = PrefixArchivePersistenceFailure{error};
            }

            const auto receipt = mutation->receipt;
            // Drop the full source before observers can interpret Published as
            // permission to reclaim the cache's alias. No second block exists.
            mutation.reset();
            publish(receipt, std::move(*result));
            {
                std::lock_guard lock(mutex_);
                executing_.reset();
                if (!error.empty())
                {
                    failure_ = std::move(error);
                    lifecycle_ = Lifecycle::Failed;
                }
            }
            changed_.notify_all();
        }
    }
}
