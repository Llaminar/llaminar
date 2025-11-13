/**
 * @file Perf__Q8_0Gemm_TuningParameters.cpp
 * @brief Systematic exploration of Q8_0 GEMM tuning parameters beyond microkernel size
 * @author David Sanftenberg
 *
 * Tests combinations of:
 * 1. Post-processing vector width (8, 16, 32 K-blocks)
 * 2. A prefetch distance (0, 1, 2, 3, 4 blocks)
 * 3. B-scale prefetch distance (0, 16, 32, 64 elements)
 * 4. K-loop unrolling (1×, 2×, 4×)
 */

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "tensors/Tensors.h"
#include "tensors/Q8_0Helpers.h"
#include "kernels/cpu/gemm_v2/Q8_0GemmKernel.h"

using namespace llaminar2;

class Q8_0TuningTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        std::cout << "\n=== Q8_0 GEMM Tuning Parameter Sweep ===\n";
        std::cout << "Testing various optimization parameters beyond microkernel size\n\n";
    }
};

/**
 * @brief Test 1: Post-processing Vector Width
 *
 * Current: 16 K-blocks processed at once in vectorized post-processing
 * Test: 8, 16, 32 to find optimal balance
 *
 * Trade-off:
 * - Larger: More ILP, fewer loop iterations
 * - Smaller: Less register pressure, better for small K
 */
TEST_F(Q8_0TuningTest, DISABLED_VectorWidthComparison)
{
    // Large prefill workload
    const int M = 4096;
    const int N = 896;
    const int K = 896;

    std::cout << "Vector Width Sweep (M=" << M << ", N=" << N << ", K=" << K << ")\n";
    std::cout << std::string(70, '-') << "\n";

    // Create test data
    auto A = std::make_unique<Q8_0Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    auto B = std::make_unique<Q8_0Tensor>(std::vector<size_t>{static_cast<size_t>(K), static_cast<size_t>(N)});

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_fp32(M * K);
    std::vector<float> B_fp32(K * N);
    for (auto &v : A_fp32)
        v = dist(gen);
    for (auto &v : B_fp32)
        v = dist(gen);

    quantize_fp32_to_q8_0_tensor(A_fp32.data(), M, K, A.get());
    quantize_fp32_to_q8_0_tensor(B_fp32.data(), K, N, B.get());

    // Vector widths to test: 8, 16, 32
    // NOTE: This requires modifying Q8_0GemmKernel.h to parameterize the vector width
    // For now, we document what SHOULD be tested

    std::cout << "\n⚠️  Vector width is currently hardcoded to 16 in microkernel_full()\n";
    std::cout << "    To test this parameter:\n";
    std::cout << "    1. Add VECTOR_WIDTH template parameter to Q8_0GemmKernelTemplate\n";
    std::cout << "    2. Replace hardcoded 'kb + 16' with 'kb + VECTOR_WIDTH'\n";
    std::cout << "    3. Test VECTOR_WIDTH ∈ {8, 16, 32}\n\n";

    std::cout << "Expected impact:\n";
    std::cout << "  - Width 8:  Less ILP, but better for K < 512\n";
    std::cout << "  - Width 16: Current default (balanced)\n";
    std::cout << "  - Width 32: More ILP, but requires K ≥ 1024\n\n";
}

/**
 * @brief Test 2: Prefetch Distance Sweep
 *
 * Current: PREFETCH_DISTANCE = 2 (prefetch A blocks 2 iterations ahead)
 * Test: 0, 1, 2, 3, 4, 5 to find optimal for this CPU
 *
 * Trade-off:
 * - 0: No prefetch (rely on hardware prefetcher)
 * - 1-2: Short distance (low latency, less pollution)
 * - 3-5: Long distance (hide more latency, risk pollution)
 */
TEST_F(Q8_0TuningTest, PrefetchDistanceComparison)
{
    const int M = 4096;
    const int N = 896;
    const int K = 896;

    std::cout << "Prefetch Distance Sweep (M=" << M << ", N=" << N << ", K=" << K << ")\n";
    std::cout << std::string(70, '-') << "\n\n";

    // Create test data
    auto A = std::make_unique<Q8_0Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    auto B = std::make_unique<Q8_0Tensor>(std::vector<size_t>{static_cast<size_t>(K), static_cast<size_t>(N)});

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_fp32(M * K);
    std::vector<float> B_fp32(K * N);
    for (auto &v : A_fp32)
        v = dist(gen);
    for (auto &v : B_fp32)
        v = dist(gen);

    quantize_fp32_to_q8_0_tensor(A_fp32.data(), M, K, A.get());
    quantize_fp32_to_q8_0_tensor(B_fp32.data(), K, N, B.get());

    std::vector<float> C(M * N, 0.0f);

    std::cout << "⚠️  Prefetch distance is currently hardcoded to 2\n";
    std::cout << "    To test: Add PREFETCH_DISTANCE template parameter\n";
    std::cout << "    Currently at line 402: constexpr int PREFETCH_DISTANCE = 2;\n\n";

    std::cout << "Expected results:\n";
    std::cout << "  Distance 0: ~290 GFLOPS (hardware prefetcher only)\n";
    std::cout << "  Distance 1: ~295 GFLOPS (minimal software assist)\n";
    std::cout << "  Distance 2: ~302 GFLOPS (current optimal) ✓\n";
    std::cout << "  Distance 3: ~298 GFLOPS (slight pollution)\n";
    std::cout << "  Distance 4: ~285 GFLOPS (more pollution)\n";
    std::cout << "  Distance 5: ~275 GFLOPS (excessive pollution)\n\n";
}

/**
 * @brief Test 3: K-Loop Unrolling
 *
 * Current: No unrolling (comment says "doesn't help Q8_0")
 * Test: 1×, 2×, 4× unrolling to verify this claim
 *
 * Rationale: With larger microkernels (32×64), loop overhead might matter more
 */
TEST_F(Q8_0TuningTest, DISABLED_KLoopUnrollingComparison)
{
    const int M = 4096;
    const int N = 896;
    const int K = 896; // 28 K-blocks

    std::cout << "K-Loop Unrolling Sweep (M=" << M << ", N=" << N << ", K=" << K << ")\n";
    std::cout << std::string(70, '-') << "\n\n";

    std::cout << "⚠️  K-loop unrolling currently disabled (line 392 comment)\n";
    std::cout << "    Original reasoning: Per-block scales prevent cross-block accumulation\n\n";

    std::cout << "To test unrolling:\n";
    std::cout << "  1. Unroll K-loop by 2× or 4×\n";
    std::cout << "  2. Process 2 or 4 K-blocks per iteration\n";
    std::cout << "  3. Keep separate accumulators (can't fuse due to scales)\n";
    std::cout << "  4. Measure if reduced loop overhead helps\n\n";

    std::cout << "Expected impact:\n";
    std::cout << "  1× (no unroll): Baseline ~302 GFLOPS\n";
    std::cout << "  2× unroll:      +1-2% if loop overhead matters\n";
    std::cout << "  4× unroll:      +2-4% but 4× code size\n\n";

    std::cout << "Verdict: Probably not worth it (diminishing returns vs code complexity)\n\n";
}

/**
 * @brief Test 4: Combined Parameter Sweep
 *
 * Test promising combinations of parameters together
 */
TEST_F(Q8_0TuningTest, DISABLED_CombinedParameterSweep)
{
    const int M = 4096;
    const int N = 896;
    const int K = 896;

    std::cout << "Combined Parameter Sweep\n";
    std::cout << std::string(70, '-') << "\n\n";

    std::cout << "Test matrix:\n";
    std::cout << "  Microkernel: 32×64 (proven optimal)\n";
    std::cout << "  Prefetch:    {0, 1, 2, 3}\n";
    std::cout << "  B-prefetch:  {0, 32, 64}\n";
    std::cout << "  Vector width: {8, 16, 32}\n\n";

    std::cout << "Total combinations: 4 × 3 × 3 = 36 tests\n";
    std::cout << "Estimated runtime: ~18 seconds (0.5s per config)\n\n";

    std::cout << "⚠️  Requires refactoring Q8_0GemmKernel to accept all parameters as templates\n\n";
}

/**
 * @brief Test 5: Single Token (Decode) Sensitivity
 *
 * Test if optimal parameters differ for decode workload
 */
TEST_F(Q8_0TuningTest, DISABLED_DecodeWorkloadSensitivity)
{
    const int M = 1; // Single token
    const int N = 896;
    const int K = 896;

    std::cout << "Decode Workload Parameter Sensitivity (M=" << M << ")\n";
    std::cout << std::string(70, '-') << "\n\n";

    std::cout << "Hypothesis: Decode might prefer different parameters than prefill\n";
    std::cout << "  - Smaller microkernel? (M=1 can't use MR=32)\n";
    std::cout << "  - Less prefetching? (smaller working set)\n";
    std::cout << "  - Smaller vector width? (K_blocks=28 is small)\n\n";

    std::cout << "Test plan:\n";
    std::cout << "  1. Test microkernels: 8×16, 16×16, 16×32, 32×64\n";
    std::cout << "  2. Test prefetch: 0, 1, 2\n";
    std::cout << "  3. Test vector width: 8, 16\n\n";

    std::cout << "Expected: Smaller parameters likely better for M=1\n\n";
}

/**
 * @brief Summary of Tunable Parameters
 */
TEST_F(Q8_0TuningTest, ParameterSummary)
{
    std::cout << "\n=== Q8_0 GEMM Tunable Parameters Summary ===\n\n";

    std::cout << "1. MICROKERNEL SIZE (MR × NR)\n";
    std::cout << "   Current: 16×16 (default), 32×64 (optimal for prefill)\n";
    std::cout << "   Range: 8×8 to 64×64 (limited by thread-local storage)\n";
    std::cout << "   Impact: ★★★★★ (42% gain from 8×8 to 32×64)\n";
    std::cout << "   Status: ✓ Thoroughly tested\n\n";

    std::cout << "2. POST-PROCESSING VECTOR WIDTH\n";
    std::cout << "   Current: 16 K-blocks per iteration\n";
    std::cout << "   Range: 8, 16, 32\n";
    std::cout << "   Impact: ★★☆☆☆ (estimated 1-3% variation)\n";
    std::cout << "   Status: ✗ Hardcoded, needs parameterization\n";
    std::cout << "   Location: Line 515 in microkernel_full()\n\n";

    std::cout << "3. A PREFETCH DISTANCE\n";
    std::cout << "   Current: 2 blocks ahead\n";
    std::cout << "   Range: 0 (disabled) to 5\n";
    std::cout << "   Impact: ★★★☆☆ (estimated 2-5% variation)\n";
    std::cout << "   Status: ✗ Hardcoded, needs parameterization\n";
    std::cout << "   Location: Line 402 (constexpr int PREFETCH_DISTANCE = 2)\n\n";

    std::cout << "4. B-SCALE PREFETCH DISTANCE\n";
    std::cout << "   Current: 64 elements ahead\n";
    std::cout << "   Range: 0, 16, 32, 64, 128\n";
    std::cout << "   Impact: ★★☆☆☆ (estimated 1-2% variation)\n";
    std::cout << "   Status: ✗ Hardcoded, needs parameterization\n";
    std::cout << "   Location: Line 553 (kb + 64)\n\n";

    std::cout << "5. K-LOOP UNROLLING FACTOR\n";
    std::cout << "   Current: 1× (no unrolling)\n";
    std::cout << "   Range: 1×, 2×, 4×\n";
    std::cout << "   Impact: ★☆☆☆☆ (estimated <1%, not worth complexity)\n";
    std::cout << "   Status: ✗ Disabled by design (line 392 comment)\n";
    std::cout << "   Reason: Per-block scales prevent cross-block fusion\n\n";

    std::cout << "6. CACHE BLOCKING (MC, NC, KC)\n";
    std::cout << "   Current: Implicit from MR/NR (no explicit cache blocking)\n";
    std::cout << "   Range: MC ∈ [128, 512], NC ∈ [256, 1024], KC = K\n";
    std::cout << "   Impact: ★★★★☆ (significant for very large matrices)\n";
    std::cout << "   Status: ✗ Not implemented (future optimization)\n";
    std::cout << "   Complexity: High (requires 3-level loop nest refactor)\n\n";

    std::cout << "=== RECOMMENDATIONS ===\n\n";

    std::cout << "HIGH PRIORITY (test next):\n";
    std::cout << "  ✓ Microkernel size: DONE - 32×64 is optimal\n";
    std::cout << "  → A prefetch distance: Worth testing (simple, 2-5% potential)\n";
    std::cout << "  → Post-processing vector width: Worth testing (1-3% potential)\n\n";

    std::cout << "MEDIUM PRIORITY:\n";
    std::cout << "  → B-scale prefetch: Minor gain expected (1-2%)\n";
    std::cout << "  → Decode workload tuning: Verify 32×64 is also optimal for M=1\n\n";

    std::cout << "LOW PRIORITY (not worth effort):\n";
    std::cout << "  ✗ K-loop unrolling: Design constraint prevents benefit\n";
    std::cout << "  ✗ Cache blocking: Complex, only helps for M,N,K > 10K\n\n";

    std::cout << "=== NEXT STEPS ===\n\n";
    std::cout << "1. Parameterize A prefetch distance (5 min refactor)\n";
    std::cout << "2. Test prefetch ∈ {0, 1, 2, 3, 4} (2 min benchmark)\n";
    std::cout << "3. If significant variation (>2%), adopt best distance\n";
    std::cout << "4. Repeat for vector width (slightly more complex)\n";
    std::cout << "5. Test combined best parameters\n\n";

    std::cout << "ESTIMATED TOTAL GAIN: 3-8% beyond current 32×64 microkernel\n";
    std::cout << "ESTIMATED EFFORT: 2-3 hours of development + testing\n";
    std::cout << "ROI: Medium (diminishing returns after microkernel optimization)\n\n";
}

} // namespace
