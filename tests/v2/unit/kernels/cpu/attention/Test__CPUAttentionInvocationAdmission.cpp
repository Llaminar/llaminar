/**
 * @file Test__CPUAttentionInvocationAdmission.cpp
 * @brief Device-free contract of the ISA-neutral CPU attention invocation admission.
 *
 * The row programs address banks without per-call checks only because this
 * admission proved, once per invocation, that every addressed row exists,
 * heads map inside the declared K/V heads and the output is disjoint from
 * every input. Each rule here has a negative control.
 */
#include "kernels/cpu/attention/CPUAttentionInvocationAdmission.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace
{
using llaminar2::cpu::CPUAttentionInvocationAdmission;
using llaminar2::cpu::CPUAttentionInvocationRequest;

/** @brief Disjoint banks of exactly the declared extents for a small GQA geometry. */
struct Fixture
{
    static constexpr int rows = 3, kv_rows = 5, heads = 8, kv_heads = 2, head_dim = 64;
    std::vector<float> query = std::vector<float>(rows * heads * head_dim);
    std::vector<float> output = std::vector<float>(rows * heads * head_dim);
    std::vector<std::uint16_t> key = std::vector<std::uint16_t>(kv_rows * kv_heads * head_dim);
    std::vector<std::uint16_t> value = std::vector<std::uint16_t>(kv_rows * kv_heads * head_dim);

    [[nodiscard]] CPUAttentionInvocationRequest request() const
    {
        const std::size_t kv_row = std::size_t(kv_heads) * head_dim * sizeof(std::uint16_t);
        return {
            .query = {query.data(), query.size() * sizeof(float)},
            .output = {output.data(), output.size() * sizeof(float)},
            .key = {key.data(), key.size() * sizeof(std::uint16_t)},
            .value = {value.data(), value.size() * sizeof(std::uint16_t)},
            .query_rows = rows, .kv_physical_rows = kv_rows, .query_heads = heads, .kv_heads = kv_heads,
            .head_dim = head_dim, .head_start = 0, .gqa_n_rep = 0,
            .key_row_bytes = kv_row, .value_row_bytes = kv_row,
        };
    }
};

/** @brief Expect rejection and a diagnostic. */
void rejects(const CPUAttentionInvocationRequest &request, const char *trace)
{
    std::string error;
    EXPECT_FALSE(CPUAttentionInvocationAdmission::admit(request, &error).has_value()) << trace;
    EXPECT_FALSE(error.empty()) << trace;
}
} // namespace

/** @test Exact extents and disjoint banks are admitted; the head width is retained. */
TEST(CPUAttentionInvocationAdmission, ExactDisjointBanksAreAdmitted)
{
    const Fixture fixture;
    const auto admission = CPUAttentionInvocationAdmission::admit(fixture.request());
    ASSERT_TRUE(admission.has_value());
    EXPECT_EQ(admission->headWidth(), Fixture::head_dim);
}

/** @test Every geometry rule has a negative control. */
TEST(CPUAttentionInvocationAdmission, GeometryViolationsAreRejected)
{
    const Fixture fixture;
    for (int field = 0; field < 5; ++field)
    {
        auto request = fixture.request();
        int *target[] = {&request.query_rows, &request.kv_physical_rows, &request.query_heads, &request.kv_heads,
                         &request.head_dim};
        *target[field] = 0;
        rejects(request, "nonpositive dimension");
    }
    auto request = fixture.request();
    request.head_dim = CPUAttentionInvocationAdmission::kMaxHeadDim + 1;
    rejects(request, "head width");
    request = fixture.request();
    request.query_heads = 7; // 7 query heads over 2 K/V heads, no explicit ratio.
    rejects(request, "uneven heads");
    request = fixture.request();
    request.gqa_n_rep = 4;
    request.head_start = 1; // Global heads 1..8 over ratio 4 reach K/V head 2 of 2.
    rejects(request, "head mapping");
    request.head_start = 0;
    EXPECT_TRUE(CPUAttentionInvocationAdmission::admit(request).has_value());
    request = fixture.request();
    request.query_rows = std::numeric_limits<int>::max();
    request.query_heads = std::numeric_limits<int>::max() / 2 * 2;
    request.kv_heads = 2;
    rejects(request, "unrepresentable extent");
}

/** @test Banks shorter than their declared rows, or missing, are rejected. */
TEST(CPUAttentionInvocationAdmission, ShortOrMissingBanksAreRejected)
{
    const Fixture fixture;
    for (int bank = 0; bank < 4; ++bank)
    {
        auto request = fixture.request();
        auto *target = bank == 0 ? &request.query : bank == 1 ? &request.output : bank == 2 ? &request.key : &request.value;
        target->bytes -= 1;
        rejects(request, "short bank");
        target->bytes += 1;
        target->data = nullptr;
        rejects(request, "missing bank");
    }
    auto request = fixture.request();
    request.kv_physical_rows += 1; // A ring capacity beyond the cache allocation.
    rejects(request, "physical rows beyond K/V");
}

/** @test The output may not overlap the query or either K/V bank; adjacency is fine. */
TEST(CPUAttentionInvocationAdmission, OutputAliasingIsRejected)
{
    Fixture fixture;
    auto request = fixture.request();
    request.output.data = fixture.query.data();
    rejects(request, "output aliases query");
    // One allocation where the K (then V) bank starts 64 bytes before the output ends.
    std::vector<std::uint8_t> overlapping(fixture.output.size() * sizeof(float) + fixture.key.size() * sizeof(std::uint16_t));
    for (const bool key_bank : {true, false})
    {
        request = fixture.request();
        request.output = {overlapping.data(), fixture.output.size() * sizeof(float)};
        auto &bank = key_bank ? request.key : request.value;
        bank = {overlapping.data() + request.output.bytes - 64, fixture.key.size() * sizeof(std::uint16_t)};
        rejects(request, key_bank ? "output aliases key" : "output aliases value");
    }

    // One contiguous allocation holding output immediately followed by value: disjoint.
    std::vector<std::uint8_t> shared(fixture.output.size() * sizeof(float) + fixture.value.size() * sizeof(std::uint16_t));
    request = fixture.request();
    request.output = {shared.data(), fixture.output.size() * sizeof(float)};
    request.value = {shared.data() + request.output.bytes, fixture.value.size() * sizeof(std::uint16_t)};
    EXPECT_TRUE(CPUAttentionInvocationAdmission::admit(request).has_value());
}
