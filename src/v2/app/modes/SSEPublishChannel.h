/**
 * @file SSEPublishChannel.h
 * @brief Bounded producer/consumer SSE line channel (design §10.1).
 *
 * The channel itself is transport-agnostic and touches no socket: a producer
 * pushes rendered lines; a consumer drains them. The §10.1 INDEPENDENT-WRITER
 * integration (the inference submission owner as producer-only, the httplib
 * provider thread as the network writer) is a SEPARATE wiring step — this
 * component only provides the bounded queue semantics.
 *
 * The channel is bounded — a slow consumer exerts backpressure on the producer —
 * and closes exactly once; after close, drains return what was published and
 * then report closure so the consumer can terminate.
 *
 * Ownership: the producer owns the close (after production); consumers observe
 * closure and may not push; the channel is non-copyable.
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace llaminar2
{
    /** @brief Queued-payload window quota for one SSE publish channel.
     *
     *  `max_entries` is the CONFIGURED ENTRY QUOTA (the backpressure authority
     *  for this channel); it is NOT a derived handler publication count and NOT
     *  an output-window credit. The production quota policy is the provisional
     *  policy below — handler publication geometry is not derived here.
     *  `max_buffered_bytes == 0` means BYTES UNBOUNDED — the
     *  backpressure authority is then the entry count only, and `bufferedBytes()`
     *  remains an OBSERVED logical quantity. A POSITIVE byte bound is an
     *  explicit cap (focused tests; available to a future admitted window
     *  authority) — it is NOT claimed as PMA-backed admission: physical
     *  output-window byte admission remains a separate open authority. */
    struct SSEPublishChannelBudget
    {
        std::size_t max_entries = 0;
        /** @brief 0 = unbounded (observation only); >0 = explicit byte cap. */
        std::size_t max_buffered_bytes = 0;
        /** @brief Handling of one line larger than the whole byte cap. */
        enum class OversizedLine
        {
            Reject,              ///< Typed std::length_error (default).
            AdmitAloneWhenEmpty, ///< Wait until the queue drained, then queue it alone:
                                 ///< buffered bytes stay <= max(cap, that already-built line).
        };
        OversizedLine oversized_line = OversizedLine::Reject;

        /** @brief PRODUCTION QUOTA POLICY (PROVISIONAL — NOT certified geometry).
         *
         *  `max_entries` = response-limit tokens × 2 (reasoning AND answer can
         *  both publish for one committed token) + a PROVISIONAL framing-line
         *  allowance (role, finish, usage, final DONE, tool/flush). This is an
         *  honest provisional quota policy: the exact maximum handler
         *  publication count per window (reasoning/answer double emission,
         *  usage, tool-call/flush geometry) is NOT yet derived or counted
         *  here, and certification of the framing geometry is an OPEN item —
         *  this quota must not be claimed as real geometry. Bytes are
         *  UNBOUNDED (no guessed per-token byte constant; physical/admitted
         *  byte authority is a separate open piece), so no possible line is
         *  ever rejected by a fabricated cap. */
        [[nodiscard]] static SSEPublishChannelBudget forRequest(int response_limit_tokens)
        {
            constexpr std::size_t kDefaultWindowTokens = 16;
            constexpr std::size_t kProvisionalFramingLines = 6;
            // 2 publication lines per window token (reasoning + answer) is the
            // provisional allowance; see the doc comment above.
            constexpr std::size_t kProvisionalLinesPerToken = 2;
            const std::size_t tokens = response_limit_tokens > 0
                                           ? static_cast<std::size_t>(response_limit_tokens)
                                           : kDefaultWindowTokens;
            return SSEPublishChannelBudget{/*max_entries*/
                                           tokens * kProvisionalLinesPerToken +
                                               kProvisionalFramingLines,
                                           /*max_buffered_bytes*/ 0};
        }
    };

    class SSEPublishChannel final
    {
    public:
        explicit SSEPublishChannel(SSEPublishChannelBudget budget)
            : budget_(budget)
        {
            if (budget_.max_entries == 0)
                throw std::invalid_argument("SSE publish channel entry bound must be positive");
            // max_buffered_bytes == 0 is the UNBOUNDED (observation-only) mode.
        }
        explicit SSEPublishChannel(std::size_t capacity = 64)
            : budget_(SSEPublishChannelBudget{capacity, checkedLegacyBytes(capacity)})
        {
            if (capacity == 0)
                throw std::invalid_argument("SSE publish channel capacity must be positive");
        }
        SSEPublishChannel(const SSEPublishChannel &) = delete;
        SSEPublishChannel &operator=(const SSEPublishChannel &) = delete;

        /** @brief Producer enqueue (bounded; blocks while the writer is slow).
         *         Returns false once the channel is closed (transport retired). */
        bool push(const std::string &line)
        {
            // BYTE-CAP PATH ONLY: distinguish an IMPOSSIBLE line before waiting.
            // A line larger than the whole byte window can never fit (no pop can
            // help), so waiting would deadlock producer/consumer; it is a
            // deterministic fatal rejection (typed error), never a silent false.
            // With max_buffered_bytes == 0 (production/unbounded) no line is ever
            // rejected by a fabricated cap.
            const bool byte_capped = budget_.max_buffered_bytes != 0;
            const bool oversized = byte_capped && line.size() > budget_.max_buffered_bytes;
            if (oversized && budget_.oversized_line != SSEPublishChannelBudget::OversizedLine::AdmitAloneWhenEmpty)
            {
                throw std::length_error(
                    "SSE publish line exceeds the request's queued-payload byte window; "
                    "the operation is impossible and is rejected before waiting");
            }
            std::unique_lock lock(mutex_);
            // SCOPED registration: only THIS call's own wait registers (and its
            // own destructor decrements) — a different caller that never waited
            // can never steal or falsely clear this registration.
            // Stack-scoped registration (no per-wait heap traffic).
            std::optional<WaitRegistration> registration;
            // Overflow-safe remaining-capacity comparison: on the cap path,
            // buffered_bytes_ <= max_buffered_bytes always holds (every admitted
            // line passed the same predicate), so the subtraction cannot wrap.
            const auto blocked = [&] {
                if (lines_.size() >= budget_.max_entries)
                    return true;
                if (oversized)
                    return !lines_.empty(); // admitted alone into an empty queue
                return byte_capped &&
                       (buffered_bytes_ >= budget_.max_buffered_bytes ||
                        line.size() > budget_.max_buffered_bytes - buffered_bytes_);
            };
            if (blocked() && !closed_)
                registration.emplace(blocked_pushers_);
            not_full_.wait(lock, [&] { return closed_ || !blocked(); });
            if (closed_)
                return false;
            lines_.push_back(line);
            buffered_bytes_ += line.size();
            not_empty_.notify_one();
            return true;
        }

        /** @brief NON-BLOCKING push: succeeds only when a slot is free; NEVER
         *         registers a wait (the controlled-concurrency discriminator).
         *         @return true if enqueued, false when full or closed (no wait). */
        bool tryPush(const std::string &line)
        {
            const bool byte_capped = budget_.max_buffered_bytes != 0;
            if (byte_capped && line.size() > budget_.max_buffered_bytes)
                return false; // impossible fit: definite refuse, no wait, no throw
            std::lock_guard lock(mutex_);
            if (closed_ || lines_.size() >= budget_.max_entries)
                return false;
            if (byte_capped && (buffered_bytes_ >= budget_.max_buffered_bytes ||
                        line.size() > budget_.max_buffered_bytes - buffered_bytes_))
                return false;
            lines_.push_back(line);
            buffered_bytes_ += line.size();
            not_empty_.notify_one();
            return true;
        }

        /** @brief NON-BLOCKING drain: returns a line only when one is available;
         *         NEVER registers a wait (consumer-symmetry discriminator). */
        std::optional<std::string> tryPop()
        {
            std::lock_guard lock(mutex_);
            if (lines_.empty())
                return std::nullopt;
            std::string line = std::move(lines_.front());
            lines_.pop_front();
            buffered_bytes_ -= line.size();
            not_full_.notify_all();
            return line;
        }

        /** @brief Writer drain: returns the next published line, or null when the
         *         producer closed AND the queue is drained (stream complete). */
        std::optional<std::string> popOrClosed()
        {
            std::unique_lock lock(mutex_);
            std::optional<WaitRegistration> registration;
            if (lines_.empty() && !closed_)
                registration.emplace(blocked_consumers_);
            not_empty_.wait(lock, [&] { return closed_ || !lines_.empty(); });
            if (lines_.empty())
                return std::nullopt; // closed and drained
            std::string line = std::move(lines_.front());
            lines_.pop_front();
            buffered_bytes_ -= line.size();
            not_full_.notify_all();
            return line;
        }

        /** @brief Result of a bounded writer drain. */
        enum class PopStatus { Line, Closed, TimedOut };

        /**
         * @brief Writer drain bounded by @p timeout, so the writer can check
         *        its transport (client disconnect) while no output exists.
         */
        PopStatus popOrClosedFor(std::chrono::milliseconds timeout, std::string &line)
        {
            std::unique_lock lock(mutex_);
            std::optional<WaitRegistration> registration;
            if (lines_.empty() && !closed_)
                registration.emplace(blocked_consumers_);
            if (!not_empty_.wait_for(lock, timeout, [&] { return closed_ || !lines_.empty(); }))
                return PopStatus::TimedOut;
            if (lines_.empty())
                return PopStatus::Closed;
            line = std::move(lines_.front());
            lines_.pop_front();
            buffered_bytes_ -= line.size();
            not_full_.notify_all();
            return PopStatus::Line;
        }

        /** @brief Producer close (exactly once; all waiting drains wake). */
        void close()
        {
            {
                std::lock_guard lock(mutex_);
                if (closed_)
                    return;
                closed_ = true;
            }
            not_empty_.notify_all();
            not_full_.notify_all();
        }

        [[nodiscard]] bool closed() const
        {
            std::lock_guard lock(mutex_);
            return closed_;
        }
        [[nodiscard]] std::size_t blockedPushers() const
        {
            std::lock_guard lock(mutex_);
            return blocked_pushers_;
        }
        [[nodiscard]] std::size_t blockedConsumers() const
        {
            std::lock_guard lock(mutex_);
            return blocked_consumers_;
        }
        [[nodiscard]] std::size_t size() const
        {
            std::lock_guard lock(mutex_);
            return lines_.size();
        }
        /** @brief Currently buffered queued payload bytes (observed quantity;
         *  NOT an admitted physical byte authority). */
        [[nodiscard]] std::size_t bufferedBytes() const
        {
            std::lock_guard lock(mutex_);
            return buffered_bytes_;
        }
        [[nodiscard]] const SSEPublishChannelBudget &budget() const noexcept { return budget_; }

    private:
        /** @brief SHARED registration authority: ONE type for pushers and
         *         consumers; each call's own registration increments exactly
         *         once and its destructor decrements exactly once (paired). */
        struct WaitRegistration
        {
            std::size_t &count;
            explicit WaitRegistration(std::size_t &c) : count(c) { ++count; }
            ~WaitRegistration() { --count; }
        };

        /** @brief Overflow-safe legacy byte computation (capacity x per-entry). */
        static std::size_t checkedLegacyBytes(std::size_t capacity)
        {
            constexpr std::size_t kLegacyBytesPerEntry = 4096;
            if (capacity > std::numeric_limits<std::size_t>::max() / kLegacyBytesPerEntry)
                throw std::overflow_error("SSE publish channel legacy byte bound overflows");
            return capacity * kLegacyBytesPerEntry;
        }

        SSEPublishChannelBudget budget_;
        std::size_t buffered_bytes_ = 0;
        bool closed_ = false;
        mutable std::mutex mutex_;
        std::condition_variable not_empty_;
        std::condition_variable not_full_;
        std::deque<std::string> lines_;
        std::size_t blocked_pushers_ = 0;   ///< Threads inside not_full_.wait (wait-entry proof).
        std::size_t blocked_consumers_ = 0; ///< Threads inside not_empty_.wait (wait-entry proof).
    };
} // namespace llaminar2
