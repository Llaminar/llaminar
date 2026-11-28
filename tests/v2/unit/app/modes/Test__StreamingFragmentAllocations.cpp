/**
 * @file Test__StreamingFragmentAllocations.cpp
 * @brief Scoped allocation witness around the actual production delta builder.
 *
 * Inputs and assertions stay outside the measured interval. This dedicated
 * executable replaces C++ allocation only locally; it does not assert anything
 * about tokenizer/parser storage or total HTTP response memory.
 */
#include "app/modes/StreamingDeltaFragments.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <new>

namespace {
thread_local bool counting = false;
thread_local std::size_t allocated_bytes = 0;
/** @brief Record requested bytes only on the witnessed thread. */
void observe(std::size_t bytes) noexcept { if (counting) allocated_bytes += bytes; }
/** @brief Scope counters so failed or throwing calls cannot contaminate later tests. */
struct Witness {
    Witness() { allocated_bytes = 0; counting = true; }
    ~Witness() { counting = false; }
};
}
/** @brief Instrument ordinary allocation in this standalone test process. */
void *operator new(std::size_t n) { if (auto *p = std::malloc(std::max(n, std::size_t{1}))) { observe(n); return p; } throw std::bad_alloc(); }
/** @brief Instrument arrays. */
void *operator new[](std::size_t n) { return ::operator new(n); }
/** @brief Instrument aligned storage without changing alignment. */
void *operator new(std::size_t n, std::align_val_t a) { void *p = nullptr; if (posix_memalign(&p, std::size_t(a), std::max(n, std::size_t{1}))) throw std::bad_alloc(); observe(n); return p; }
/** @brief Instrument aligned arrays. */
void *operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
/** @brief Release ordinary storage. */
void operator delete(void *p) noexcept { std::free(p); }
/** @brief Release array storage. */
void operator delete[](void *p) noexcept { std::free(p); }
/** @brief Release sized storage. */
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
/** @brief Release sized array storage. */
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }
/** @brief Release aligned storage. */
void operator delete(void *p, std::align_val_t) noexcept { std::free(p); }
/** @brief Release aligned arrays. */
void operator delete[](void *p, std::align_val_t) noexcept { std::free(p); }
/** @brief Release sized aligned storage. */
void operator delete(void *p, std::size_t, std::align_val_t) noexcept { std::free(p); }
/** @brief Release sized aligned arrays. */
void operator delete[](void *p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace llaminar2::test {
using json = nlohmann::json;
/** @brief Build representative fields before measuring publication. */
json input(const std::string &field, std::size_t size)
{
    if (field != "tool_calls") return json{{field, std::string(size, 'x')}};
    return json{{"tool_calls", json::array({json{{"index", 0}, {"id", "call_1"}, {"type", "function"},
        {"function", {{"name", "write"}, {"arguments", std::string(size, 'x')}}}}})}};
}
TEST(StreamingFragmentAllocations, RejectFirstHasInputIndependentAllocation)
{
    for (const std::string field : {"content", "reasoning_content", "tool_calls"})
    {
        SCOPED_TRACE(field);
        const auto small = input(field, 8192);
        const auto large = input(field, 2 * 1024 * 1024);
        std::size_t measured[2]{};
        for (int i = 0; i != 2; ++i)
        {
            int callbacks = 0;
            bool delivered = true;
            {
                Witness witness;
                delivered = streaming_detail::publishDeltaFragments(i == 0 ? small : large, nullptr,
                    [&](const json &, const char *) { ++callbacks; return false; });
                measured[i] = allocated_bytes;
            }
            EXPECT_FALSE(delivered);
            EXPECT_EQ(callbacks, 1);
        }
        EXPECT_EQ(measured[0], measured[1]);
        EXPECT_LT(measured[1], 64 * 1024u);
        // Original whole-field cloning must be observable by the same witness.
        std::size_t cloned = 0;
        {
            Witness witness;
            const json copy = large;
            cloned = allocated_bytes;
        }
        EXPECT_GE(cloned, 2 * 1024 * 1024u);
    }
}
TEST(StreamingFragmentAllocations, BorrowedPiecesAliasSourceAndStopImmediately)
{
    const std::string source(2 * 1024 * 1024, 'x');
    const char *data = nullptr;
    std::size_t bytes = 0, length = 0;
    int calls = 0;
    {
        Witness witness;
        streaming_detail::publishUtf8Pieces(source, 2048, [&](std::string_view piece, bool) {
            ++calls; data = piece.data(); length = piece.size(); return false;
        });
        bytes = allocated_bytes;
    }
    EXPECT_EQ(data, source.data());
    EXPECT_EQ(length, 2048u);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(bytes, 0u);
}
} // namespace llaminar2::test
