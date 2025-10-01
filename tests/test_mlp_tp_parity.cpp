// Tensor-parallel MLP parity test (replicated weights, executor-based column partition)
// Validates that enabling LLAMINAR_TP_MLP_ENABLE with partitions>1 matches reference output.
// Single authoritative reference is computed on rank 0 then broadcast for comparison.

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <string>
#include "../src/kernels/MPIMLPKernel.h"
#include "../src/tensors/tensor_factory.h"
#include "../src/utils/debug_env.h"

using namespace llaminar;

namespace
{
    void ref_matmul(const float *A, const float *B, float *C, int M, int K, int N)
    {
        for (int i = 0; i < M; ++i)
        {
            const float *ar = A + i * K;
            float *cr = C + i * N;
            for (int j = 0; j < N; ++j)
            {
                double acc = 0.0;
                for (int k = 0; k < K; ++k)
                    acc += (double)ar[k] * (double)B[k * N + j];
                cr[j] = (float)acc;
            }
        }
    }
    inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
    void fill(const std::shared_ptr<TensorBase> &t, float s)
    {
        for (int i = 0; i < t->size(); ++i)
            t->data()[i] = s * float((i % 97) - 48);
    }
}

// Global MPI environment (initialize once, finalize after all tests)
class MPIGlobalEnv : public ::testing::Environment
{
public:
    void SetUp() override
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            int argc = 0;
            char **argv = nullptr;
            int provided = 0;
            ASSERT_EQ(MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided), MPI_SUCCESS);
        }
    }
    void TearDown() override
    {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
        }
    }
};
static ::testing::Environment *const mpi_env = ::testing::AddGlobalTestEnvironment(new MPIGlobalEnv());

class MLP_TP_Parity : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Refresh TP env for each test (allow different shapes / dynamic world size)
        setenv("LLAMINAR_TP_MLP_ENABLE", "1", 1);
        int world = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &world);
        // If caller didn't preset size, map to world size
        if (!std::getenv("LLAMINAR_TP_MLP_SIZE"))
        {
            std::string sz = std::to_string(world);
            setenv("LLAMINAR_TP_MLP_SIZE", sz.c_str(), 1);
        }
        setenv("LLAMINAR_TP_MLP_VALIDATE", "1", 1); // enable internal kernel parity logging
        debugEnvRefresh();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    void TearDown() override { MPI_Barrier(MPI_COMM_WORLD); }
    void run_case(int seq_len, int d_model, int d_ff, const std::string &tag, double atol = 1e-5, double rtol = 1e-5)
    {
        int rank = 0, world = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world);
        if (world < 2)
        {
            if (rank == 0)
                GTEST_SKIP() << "Need >=2 ranks";
            return;
        }
        auto input = TensorFactory::create_simple({seq_len, d_model});
        auto w_gate = TensorFactory::create_simple({d_model, d_ff});
        auto w_up = TensorFactory::create_simple({d_model, d_ff});
        auto w_down = TensorFactory::create_simple({d_ff, d_model});
        auto output = TensorFactory::create_simple({seq_len, d_model});
        fill(input, 0.01f);
        fill(w_gate, 0.02f);
        fill(w_up, 0.03f);
        fill(w_down, 0.04f);
        std::vector<float> ref_out((size_t)seq_len * d_model, 0.f);
        if (rank == 0)
        {
            std::vector<float> gate(seq_len * d_ff), up(seq_len * d_ff), act(seq_len * d_ff);
            ref_matmul(input->data(), w_gate->data(), gate.data(), seq_len, d_model, d_ff);
            ref_matmul(input->data(), w_up->data(), up.data(), seq_len, d_model, d_ff);
            for (size_t i = 0; i < act.size(); ++i)
                act[i] = silu(gate[i]) * up[i];
            ref_matmul(act.data(), w_down->data(), ref_out.data(), seq_len, d_ff, d_model);
        }
        MPI_Bcast(ref_out.data(), (int)ref_out.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        auto kernel = std::make_unique<MPIMLPKernel>();
        std::vector<std::shared_ptr<TensorBase>> inputs = {input, w_gate, w_up, w_down};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        ASSERT_TRUE(kernel->execute(inputs, outputs)) << "Kernel execute failed for case " << tag;
        if (rank == 0)
        {
            double max_abs = 0.0, diff_sq = 0.0, ref_sq = 0.0;
            size_t total = (size_t)seq_len * d_model;
            for (size_t i = 0; i < total; ++i)
            {
                double d = (double)output->data()[i] - (double)ref_out[i];
                double ad = fabs(d);
                if (ad > max_abs)
                    max_abs = ad;
                diff_sq += d * d;
                ref_sq += (double)ref_out[i] * (double)ref_out[i];
            }
            double rel_l2 = ref_sq > 0.0 ? std::sqrt(diff_sq) / std::sqrt(ref_sq) : 0.0;
            EXPECT_LE(max_abs, atol) << " tag=" << tag << " rel_l2=" << rel_l2;
            EXPECT_LE(rel_l2, rtol) << " tag=" << tag << " max_abs=" << max_abs;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

TEST_F(MLP_TP_Parity, SmallOddDff70) { run_case(5, 48, 70, "odd70_ragged"); }
TEST_F(MLP_TP_Parity, EvenDff64) { run_case(5, 48, 64, "even64_balanced"); }
TEST_F(MLP_TP_Parity, SmallEdgeDff33) { run_case(2, 16, 33, "edge33_small"); }
TEST_F(MLP_TP_Parity, RaggedDff65) { run_case(7, 32, 65, "ragged65", 2e-5, 5e-5); }
TEST_F(MLP_TP_Parity, LargerEvenDff128) { run_case(32, 64, 128, "larger128", 2e-5, 2e-5); }
// Stress: larger seq_len near prefill threshold boundary but still below typical COSMA threshold (keeps runtime low)
TEST_F(MLP_TP_Parity, StressLongPrefill) { run_case(256, 96, 192, "stress_prefill", 2e-3, 1e-4); }

// 3-way partition variants (run with -np 3)
TEST_F(MLP_TP_Parity, TP3_RaggedDff97)
{
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 3)
    {
        GTEST_SKIP() << "Requires 3 ranks";
    }
    // Relaxed tolerances for 3-way ragged split of d_ff=97 (slice pattern 33|32|32).
    // Rationale: slight amplification of rounding error due to uneven first slice and
    // accumulation ordering; recent runs showed max_abs ≈8.6e-4, rel_l2 ≈1.5e-7.
    // Tightening these would create intermittent flakes; re‑evaluate after any GEMM or
    // reduce/parity refactor that might lower per-slice drift.
    run_case(6, 48, 97, "tp3_ragged97", 1e-3, 2e-4);
}
TEST_F(MLP_TP_Parity, TP3_EvenDff90)
{
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 3)
    {
        GTEST_SKIP() << "Requires 3 ranks";
    }
    run_case(6, 48, 90, "tp3_even90", 2e-5, 5e-5);
}

// 4-way partition variants (run with -np 4)
TEST_F(MLP_TP_Parity, TP4_EvenDff128)
{
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 4)
    {
        GTEST_SKIP() << "Requires 4 ranks";
    }
    run_case(8, 64, 128, "tp4_even128", 2e-5, 5e-5);
}
TEST_F(MLP_TP_Parity, TP4_RaggedDff130)
{
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 4)
    {
        GTEST_SKIP() << "Requires 4 ranks";
    }
    run_case(8, 64, 130, "tp4_ragged130", 4e-5, 6e-5);
}

// COSMA-prefill path trigger (simulate by lowering threshold & using prefill-like seq length)
TEST_F(MLP_TP_Parity, PrefillCosmaMode)
{
    // Lower threshold so moderate seq_len triggers prefill path
    setenv("LLAMINAR_COSMA_PREFILL_THRESHOLD", "8", 1);
    debugEnvRefresh();
    run_case(16, 64, 96, "prefill_cosma_mode", 5e-4, 1e-4);
    // (No restoration needed for later tests; they reinitialize env each SetUp)
}

// Negative test: deliberately alter one element post-execution on rank 0 to force failure; we expect ASSERT/EXPECT to trip.
TEST_F(MLP_TP_Parity, NegativeMismatch)
{
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world < 2)
    {
        if (rank == 0)
            GTEST_SKIP() << "Need >=2 ranks";
        return;
    }
    const int seq_len = 4, d_model = 16, d_ff = 24;
    auto input = TensorFactory::create_simple({seq_len, d_model});
    auto w_gate = TensorFactory::create_simple({d_model, d_ff});
    auto w_up = TensorFactory::create_simple({d_model, d_ff});
    auto w_down = TensorFactory::create_simple({d_ff, d_model});
    auto output = TensorFactory::create_simple({seq_len, d_model});
    fill(input, 0.01f);
    fill(w_gate, 0.02f);
    fill(w_up, 0.03f);
    fill(w_down, 0.04f);
    std::vector<float> ref_out((size_t)seq_len * d_model, 0.f);
    if (rank == 0)
    {
        std::vector<float> gate(seq_len * d_ff), up(seq_len * d_ff), act(seq_len * d_ff);
        ref_matmul(input->data(), w_gate->data(), gate.data(), seq_len, d_model, d_ff);
        ref_matmul(input->data(), w_up->data(), up.data(), seq_len, d_model, d_ff);
        for (size_t i = 0; i < act.size(); ++i)
            act[i] = silu(gate[i]) * up[i];
        ref_matmul(act.data(), w_down->data(), ref_out.data(), seq_len, d_ff, d_model);
    }
    MPI_Bcast(ref_out.data(), (int)ref_out.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    auto kernel = std::make_unique<MPIMLPKernel>();
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, w_gate, w_up, w_down};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel->execute(inputs, outputs));
    if (rank == 0)
    {                               // introduce mismatch
        output->data()[3] += 1e-3f; // perturb one value
        double max_abs = 0.0, diff_sq = 0.0, ref_sq = 0.0;
        size_t total = (size_t)seq_len * d_model;
        for (size_t i = 0; i < total; ++i)
        {
            double d = (double)output->data()[i] - (double)ref_out[i];
            double ad = fabs(d);
            if (ad > max_abs)
                max_abs = ad;
            diff_sq += d * d;
            ref_sq += (double)ref_out[i] * (double)ref_out[i];
        }
        double rel_l2 = ref_sq > 0.0 ? std::sqrt(diff_sq) / std::sqrt(ref_sq) : 0.0;
        EXPECT_GT(max_abs, 1e-4) << "negative test should exceed tolerance";
        EXPECT_GT(rel_l2, 1e-6) << "negative test should exceed rel tolerance";
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
