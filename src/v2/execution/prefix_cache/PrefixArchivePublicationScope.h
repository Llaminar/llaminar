/**
 * @file PrefixArchivePublicationScope.h
 * @brief Exception-safe completion of an already admitted prefix payload transaction.
 *
 * The callback uses the existing event and storage authority. This scope owns
 * no event, payload, allocation or ledger; it ensures that every early return
 * and exception publishes completion before local source/destination leases
 * unwind. A missing completion proof is fatal rather than a blocking cleanup.
 */
#pragma once
#include <exception>
#include <type_traits>
#include <utility>
namespace llaminar2
{
/** @brief Commit publishes a complete archive; abandonment protects partially queued work only. */
enum class PrefixArchiveCompletion : unsigned char { Commit, Abandon };
/** @brief Exactly-once event publication, declared after the leases its callback protects. */
template<class Publish> class PrefixArchivePublicationScope final
{
public:
    /**
     * @brief Bind an existing authority before the first possible asynchronous copy.
     * @param publish Callable returning true only after exact event/owner publication succeeds.
     */
    explicit PrefixArchivePublicationScope(Publish publish) : publish_(std::move(publish)) {}
    PrefixArchivePublicationScope(const PrefixArchivePublicationScope &)=delete;
    PrefixArchivePublicationScope &operator=(const PrefixArchivePublicationScope &)=delete;
    /** @brief Publish partial progress before exception or early-return lease destruction. */
    ~PrefixArchivePublicationScope() noexcept { if(armed_)finish(PrefixArchiveCompletion::Abandon); }
    /** @brief Publish normal completion once; duplicate commit is a lifecycle violation. */
    void commit() noexcept
    {
        if(!armed_)std::terminate();
        finish(PrefixArchiveCompletion::Commit);
    }
private:
    /** @brief Never permit failed publication or a throwing observer to unwind physical owners. */
    void finish(PrefixArchiveCompletion reason) noexcept
    {
        try { if(!publish_(reason))std::terminate(); }
        catch(...) { std::terminate(); }
        armed_=false;
    }
    Publish publish_;
    bool armed_=true;
};
} // namespace llaminar2
