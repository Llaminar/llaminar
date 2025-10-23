// Standalone RoPE corruption test matching MPIAttentionOperator dimensions
#include <gtest/gtest.h>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "operators/common/AttentionPrimitives.h"
#include "utils/DebugEnv.h"

using namespace llaminar::attn;

// Fill with same pattern as fillSequential in test_mpi_attention_kernel_clean
static void fillSequential(std::vector<float> &v, float scale)
{
    for (size_t i = 0; i < v.size(); ++i)
    {
        int val_idx = static_cast<int>(i % 101);
        v[i] = scale * static_cast<float>(val_idx - 50);
    }
}

// Test exact dimensions from failing MPIAttentionOperator test
TEST(RoPECorruption, ExactMPIAttentionDimensions)
{
    const int seq_len = 4;
    const int heads = 2;
    const int head_dim = 64;
    const int n_past = 0;
    const float freq_base = 10000.0f;

    // Total elements: seq_len * heads * head_dim = 4 * 2 * 64 = 512
    const size_t total_elems = seq_len * heads * head_dim;

    std::cout << "Testing RoPE with seq_len=" << seq_len
              << " heads=" << heads
              << " head_dim=" << head_dim
              << " total_elems=" << total_elems << std::endl;

    // Test with different fill patterns
    for (int pattern = 0; pattern < 5; ++pattern)
    {
        std::vector<float> q(total_elems);
        std::vector<float> k(total_elems);

        switch (pattern)
        {
        case 0:
            // Sequential pattern (like our failing test)
            fillSequential(q, 0.001f);
            fillSequential(k, 0.00125f);
            std::cout << "  Pattern 0: Sequential fill" << std::endl;
            break;
        case 1:
            // All zeros
            std::fill(q.begin(), q.end(), 0.0f);
            std::fill(k.begin(), k.end(), 0.0f);
            std::cout << "  Pattern 1: All zeros" << std::endl;
            break;
        case 2:
            // Small random values
            for (size_t i = 0; i < total_elems; ++i)
            {
                q[i] = (float(i % 101) / 101.0f) - 0.5f;
                k[i] = (float(i % 103) / 103.0f) - 0.5f;
            }
            std::cout << "  Pattern 2: Small random" << std::endl;
            break;
        case 3:
            // Ones
            std::fill(q.begin(), q.end(), 1.0f);
            std::fill(k.begin(), k.end(), 1.0f);
            std::cout << "  Pattern 3: All ones" << std::endl;
            break;
        case 4:
            // Alternating positive/negative
            for (size_t i = 0; i < total_elems; ++i)
            {
                q[i] = (i % 2 == 0) ? 0.5f : -0.5f;
                k[i] = (i % 2 == 0) ? 0.3f : -0.3f;
            }
            std::cout << "  Pattern 4: Alternating" << std::endl;
            break;
        }

        // Check for corruption before RoPE
        for (size_t i = 0; i < total_elems; ++i)
        {
            ASSERT_FALSE(std::isnan(q[i])) << "Q has NaN BEFORE RoPE at index " << i;
            ASSERT_FALSE(std::isinf(q[i])) << "Q has Inf BEFORE RoPE at index " << i;
            ASSERT_FALSE(std::isnan(k[i])) << "K has NaN BEFORE RoPE at index " << i;
            ASSERT_FALSE(std::isinf(k[i])) << "K has Inf BEFORE RoPE at index " << i;
            ASSERT_LT(std::abs(q[i]), 1e10f) << "Q has huge value BEFORE RoPE at index " << i << ": " << q[i];
            ASSERT_LT(std::abs(k[i]), 1e10f) << "K has huge value BEFORE RoPE at index " << i << ": " << k[i];
        }

        // Apply RoPE (default path, whatever the environment decides)
        apply_rope(q.data(), k.data(), seq_len, head_dim, heads, heads, n_past, freq_base);

        // Check for corruption after RoPE
        bool q_corrupted = false;
        bool k_corrupted = false;

        for (size_t i = 0; i < total_elems; ++i)
        {
            if (std::isnan(q[i]) || std::isinf(q[i]) || std::abs(q[i]) > 1e10f)
            {
                std::cerr << "    ⚠️ Q CORRUPTED at index " << i << ": " << q[i] << std::endl;
                q_corrupted = true;
                FAIL() << "Q corrupted after RoPE at index " << i << ": " << q[i];
            }
            if (std::isnan(k[i]) || std::isinf(k[i]) || std::abs(k[i]) > 1e10f)
            {
                std::cerr << "    ⚠️ K CORRUPTED at index " << i << ": " << k[i] << std::endl;
                k_corrupted = true;
                FAIL() << "K corrupted after RoPE at index " << i << ": " << k[i];
            }
        }

        if (!q_corrupted && !k_corrupted)
        {
            std::cout << "    ✓ No corruption detected" << std::endl;
        }
    }
}

// Run the exact same dimensions many times to detect non-determinism
TEST(RoPECorruption, RepeatedRuns)
{
    const int seq_len = 4;
    const int heads = 2;
    const int head_dim = 64;
    const size_t total_elems = seq_len * heads * head_dim;

    int corruption_count = 0;
    const int num_runs = 100;

    std::cout << "Running RoPE " << num_runs << " times to detect non-determinism..." << std::endl;

    for (int run = 0; run < num_runs; ++run)
    {
        std::vector<float> q(total_elems);
        std::vector<float> k(total_elems);

        fillSequential(q, 0.001f);
        fillSequential(k, 0.00125f);

        apply_rope(q.data(), k.data(), seq_len, head_dim, heads, heads, 0, 10000.0f);

        bool corrupted = false;
        for (size_t i = 0; i < total_elems; ++i)
        {
            if (std::isnan(q[i]) || std::isinf(q[i]) || std::abs(q[i]) > 1e10f ||
                std::isnan(k[i]) || std::isinf(k[i]) || std::abs(k[i]) > 1e10f)
            {
                corrupted = true;
                corruption_count++;
                if (corruption_count <= 5) // Only print first few
                {
                    std::cerr << "Run " << run << ": CORRUPTED at index " << i
                              << " q=" << q[i] << " k=" << k[i] << std::endl;
                }
                break;
            }
        }

        if (!corrupted && run % 20 == 0)
        {
            std::cout << "  Run " << run << ": OK" << std::endl;
        }
    }

    std::cout << "Corruption rate: " << corruption_count << "/" << num_runs << std::endl;
    ASSERT_EQ(corruption_count, 0) << "RoPE produced corruption in " << corruption_count << " out of " << num_runs << " runs";
}

// Test with larger buffer to see if it's a buffer size issue
TEST(RoPECorruption, LargerBuffer)
{
    const int seq_len = 8; // Larger than failing test
    const int heads = 4;
    const int head_dim = 128;
    const size_t total_elems = seq_len * heads * head_dim;

    std::cout << "Testing larger buffer: seq_len=" << seq_len
              << " heads=" << heads
              << " head_dim=" << head_dim
              << " total_elems=" << total_elems << std::endl;

    std::vector<float> q(total_elems);
    std::vector<float> k(total_elems);

    fillSequential(q, 0.001f);
    fillSequential(k, 0.00125f);

    apply_rope(q.data(), k.data(), seq_len, head_dim, heads, heads, 0, 10000.0f);

    for (size_t i = 0; i < total_elems; ++i)
    {
        ASSERT_FALSE(std::isnan(q[i])) << "Q has NaN at index " << i;
        ASSERT_FALSE(std::isinf(q[i])) << "Q has Inf at index " << i;
        ASSERT_LT(std::abs(q[i]), 1e10f) << "Q has huge value at index " << i << ": " << q[i];

        ASSERT_FALSE(std::isnan(k[i])) << "K has NaN at index " << i;
        ASSERT_FALSE(std::isinf(k[i])) << "K has Inf at index " << i;
        ASSERT_LT(std::abs(k[i]), 1e10f) << "K has huge value at index " << i << ": " << k[i];
    }

    std::cout << "  ✓ No corruption in larger buffer" << std::endl;
}

// Test with unaligned sizes
TEST(RoPECorruption, UnalignedSizes)
{
    // Test various "odd" sizes that might trigger alignment issues
    std::vector<std::tuple<int, int, int>> test_cases = {
        {3, 1, 63},  // Odd seq_len, odd head_dim
        {5, 3, 65},  // Prime numbers
        {4, 2, 64},  // Exact failing case
        {1, 1, 128}, // Minimal
        {7, 5, 127}, // Large primes
    };

    for (const auto &[seq_len, heads, head_dim] : test_cases)
    {
        const size_t total_elems = seq_len * heads * head_dim;
        std::cout << "Testing seq_len=" << seq_len
                  << " heads=" << heads
                  << " head_dim=" << head_dim << std::endl;

        std::vector<float> q(total_elems);
        std::vector<float> k(total_elems);

        fillSequential(q, 0.001f);
        fillSequential(k, 0.00125f);

        apply_rope(q.data(), k.data(), seq_len, head_dim, heads, heads, 0, 10000.0f);

        for (size_t i = 0; i < total_elems; ++i)
        {
            ASSERT_LT(std::abs(q[i]), 1e10f) << "Q corrupted at index " << i << ": " << q[i];
            ASSERT_LT(std::abs(k[i]), 1e10f) << "K corrupted at index " << i << ": " << k[i];
        }

        std::cout << "  ✓ OK" << std::endl;
    }
}
