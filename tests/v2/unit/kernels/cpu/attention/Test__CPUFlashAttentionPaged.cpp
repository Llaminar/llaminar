/**
 * @file Test__CPUFlashAttentionPaged.cpp
 * @brief Native optimized FP32 attention over actual permuted PMA page backing.
 *
 * Contiguous attention is a numerical control only. The paged call reads its
 * original retained page addresses directly; fixture packing is not a runtime
 * full-history gather adapter or a production paged append implementation.
 */
#include "v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using namespace llaminar2;

/** @test Page permutations/partial tails preserve the existing optimized arithmetic bytes. */
TEST(CPUFlashAttentionPaged, PermutedPagesMatchContiguousBytes)
{
    constexpr int page_tokens = 16;
    for (const int heads : {1, 4})
    for (const int dimension : {64, 128})
    for (const int tokens : {1, 15, 16, 17, 257, 513})
    for (const int queries : {1, 3})
    {
        if (queries > tokens) continue;
        SCOPED_TRACE(::testing::Message() << heads << '/' << dimension << '/' << tokens << '/' << queries);
        const int kv_heads = heads == 1 ? 1 : 2;
        const size_t width = static_cast<size_t>(kv_heads * dimension);
        const size_t page_count = static_cast<size_t>((tokens + page_tokens - 1) / page_tokens);
        const PhysicalPagePoolGeometry geometry{page_count, page_tokens * 2 * width * sizeof(float), 1, page_count};
        const auto device = DeviceId::cpu();
        const auto bytes = PhysicalPagePool::physicalBytes(geometry, device);
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
            PhysicalMemoryOwner::KVCache, bytes);
        auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
        auto pool = authority->createCPUPagePool(device, geometry, CPUStoragePlacement::local());
        auto request = pool->tryAdmit(page_count);
        ASSERT_TRUE(request);
        std::vector<PhysicalPagePool::FrameHandle> frames;
        for (size_t i = 0; i < page_count; ++i) frames.push_back(request->assignFrame());
        std::mt19937 random(42);
        std::uniform_real_distribution<float> values(-0.5f, 0.5f);
        std::vector<float> q(static_cast<size_t>(queries * heads * dimension));
        std::vector<float> k(static_cast<size_t>(tokens) * width), v(k.size());
        for (auto *bank : {&q, &k, &v}) for (auto &value : *bank) value = values(random);
        std::vector<CPUFP32KVPage> pages;
        pages.reserve(page_count);
        for (size_t page = 0; page < page_count; ++page)
        {
            const size_t first = page * page_tokens;
            const size_t count = std::min(static_cast<size_t>(page_tokens), static_cast<size_t>(tokens) - first);
            std::vector<float> fixture(count * 2 * width);
            for (size_t row = 0; row < count; ++row)
            {
                std::copy_n(k.data() + (first + row) * width, width, fixture.data() + row * 2 * width);
                std::copy_n(v.data() + (first + row) * width, width, fixture.data() + row * 2 * width + width);
            }
            auto write = request->prepareAppend(frames[(page + 1) % page_count], fixture.size() * sizeof(float));
            ASSERT_TRUE(write);
            write->copyFrom(std::as_bytes(std::span(fixture)));
            pages.emplace_back(write->commit(), kv_heads, dimension);
        }
        CPUFP32PagedKVView view(pages, page_tokens);
        // Publication lifetime, not the original request handle, retains bytes.
        request.reset();
        pool.reset();
        std::vector<float> expected(q.size(), -99.0f), actual(q.size(), -88.0f);
        CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
        ASSERT_TRUE(kernel.compute_decode(q.data(), k.data(), v.data(), expected.data(), queries,
            tokens, heads, kv_heads, dimension, true, tokens - queries));
        ASSERT_TRUE(kernel.compute_paged_fp32(q, view, actual, queries, heads, true, -1, tokens - queries));
        EXPECT_EQ(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)), 0);
        EXPECT_EQ(view.key(0, 0), pages.front().key(0, 0));
        std::fill(actual.begin(), actual.end(), -77.0f);
        EXPECT_FALSE(kernel.compute_paged_fp32(std::span<const float>(q.data(), q.size() - 1),
            view, actual, queries, heads, true));
        EXPECT_TRUE(std::all_of(actual.begin(), actual.end(), [](float value) { return value == -77.0f; }));
        EXPECT_FALSE(kernel.compute_paged_fp32(q, view, std::span<float>(q), queries, heads, true));
    }
}
