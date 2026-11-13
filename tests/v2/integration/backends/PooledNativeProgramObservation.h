/**
 * @file PooledNativeProgramObservation.h
 * @brief Test-target-only host observations around the unchanged native lookup graph APIs.
 *
 * Receipt emission calls no CUDA/HIP API and certifies no resource bounds.
 * Explicit diagnostic adapters additionally inspect the cold native graph;
 * their query/storage overhead remains in observations. These objects are
 * excluded from production/timing targets. Receipts use fixed stack buffers.
 */
#pragma once
#include <cstdint>
namespace llaminar2::test::pooled_program_observation {
struct HostSample;
/** @brief Fixed TLS call ancestry; observer state is not a resource ledger or native allowance. */
class CallScope final {
public:
    /** @brief Link this actual API invocation to the enclosing observed invocation on this thread. */
    CallScope(const char* backend,const char* operation,std::uint64_t ordinal)noexcept;
    /** @brief Restore the original enclosing observation before returning to the SDK caller. */
    ~CallScope()noexcept;
    CallScope(const CallScope&)=delete;
    CallScope& operator=(const CallScope&)=delete;
private:
    friend void record(const char*,const char*,const char*,std::uint64_t,const void*,const void*,int,const HostSample&,const void*,std::uint64_t)noexcept;
    const char* backend_;
    const char* operation_;
    std::uint64_t ordinal_,thread_;
    unsigned depth_;
    CallScope* parent_;
    static thread_local CallScope* current_;
};

/** @brief Fixed host-only snapshot; byte values are observations, never PMA allowances. */
struct HostSample {std::uint64_t rss,peak,allocator_live,allocator_arena,allocator_mapped,monotonic_ns;};
/** @return Existing process host counters without a device API or callback. */
HostSample sample()noexcept;
/** @brief Emit one bounded diagnostic receipt; failure to write aborts the diagnostic. */
void record(const char* backend,const char* operation,const char* phase,std::uint64_t ordinal,
    const void* handle,const void* caller,int result,const HostSample&,const void* stream=nullptr,std::uint64_t flags=0)noexcept;
/** @return Whether this explicit diagnostic injects an original instantiate return-code failure.
 * @param ordinal Exact per-backend/API invocation ordinal.
 * @param backend Actual CUDA/ROCm API family; a backend-qualified control requires it.
 */
bool rejectInstantiation(std::uint64_t ordinal,const char* backend=nullptr)noexcept;
}
namespace llaminar2::test::pooled_program_observation {
/** @brief Preserve actual existing SDK graph membership/type inspection without issuing another query. */
void topology(const char* backend,const char* observation,const void* graph,const void* node,int kind,int result)noexcept;
}
