#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>

#include "../src/tensors/tp_generic_matmul_executor.h"
#include "../src/adaptive_matmul.h" // corrected path
#include "../src/utils/debug_env.h"

using namespace llaminar;

namespace
{
    struct MPIGlobalEnvGemm : public ::testing::Environment
    {
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
    static ::testing::Environment *const mpi_env_gemm = ::testing::AddGlobalTestEnvironment(new MPIGlobalEnvGemm());

    void fill_random(std::vector<float> &v, uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto &x : v)
            x = dist(rng);
    }

    struct GemmCase
    {
        int M;
        int K;
        int N;
        int tp_size;
        TPGemmExecConfig::Mode mode;
        const char *tag;
        double atol;
        double rtol;
    };

    void run_gemm_case(const GemmCase &gc)
    {
        int rank = 0, world = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world);
        if (world != gc.tp_size)
        {
            if (rank == 0)
                GTEST_SKIP() << "Requires tp_size=" << gc.tp_size << " ranks";
            return;
        }
        // Allocate A[M,K], B[K,N]
        std::vector<float> A((size_t)gc.M * gc.K);
        std::vector<float> B((size_t)gc.K * gc.N);
        fill_random(A, 123 + gc.M + gc.N + gc.K);
        fill_random(B, 321 + gc.N + gc.K + gc.M);

        // Reference full C
        std::vector<float> C_ref((size_t)gc.M * gc.N, 0.f);
        // Simple reference matmul (double accumulation) on rank 0 then Bcast
        if (rank == 0)
        {
            for (int m = 0; m < gc.M; ++m)
            {
                for (int n = 0; n < gc.N; ++n)
                {
                    double acc = 0.0;
                    const float *arow = A.data() + m * gc.K;
                    for (int k = 0; k < gc.K; ++k)
                        acc += (double)arow[k] * (double)B[k * gc.N + n];
                    C_ref[m * gc.N + n] = (float)acc;
                }
            }
        }
        MPI_Bcast(C_ref.data(), (int)C_ref.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        // Executor using adaptive_matmul backend
        TPGemmExecConfig cfg;
        cfg.mode = gc.mode;
        cfg.tp_size = gc.tp_size;
        cfg.tp_rank = rank;
        auto fn = [](const float *A, const float *B, float *C, size_t M, size_t N, size_t K) -> bool
        {
            return adaptive_matmul(A, B, C, (int)M, (int)N, (int)K, false);
        };
        TPGemmExecutor exec(fn, cfg, gc.M, gc.N, gc.K);
        auto local = exec.run(A.data(), B.data());

        // Gather all local pieces to rank 0 for reconstruction
        // Serialize local buffer size then contents
        int local_elems = (int)local.buffer.size();
        std::vector<int> recv_counts(world, 0);
        MPI_Gather(&local_elems, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<int> displs;
        std::vector<float> all_buf;
        if (rank == 0)
        {
            displs.resize(world, 0);
            int total = 0;
            for (int i = 0; i < world; ++i)
            {
                displs[i] = total;
                total += recv_counts[i];
            }
            all_buf.resize((size_t)total);
        }
        MPI_Gatherv(local.buffer.data(), local_elems, MPI_FLOAT, rank == 0 ? all_buf.data() : nullptr, recv_counts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            // Reconstruct
            std::vector<TPGemmLocalResult> parts(world);
            int cursor = 0;
            for (int r = 0; r < world; ++r)
            {
                parts[r].partA = (cfg.mode == TPGemmExecConfig::Mode::Row) ? compute_tp_partition(gc.M, gc.tp_size, r, TPPartitionSpec::Axis::Row) : TPPartitionSpec{};
                parts[r].partB = (cfg.mode == TPGemmExecConfig::Mode::Column) ? compute_tp_partition(gc.N, gc.tp_size, r, TPPartitionSpec::Axis::Col) : TPPartitionSpec{};
                size_t m_loc = (cfg.mode == TPGemmExecConfig::Mode::Row) ? parts[r].partA.local_dim : gc.M;
                size_t n_loc = (cfg.mode == TPGemmExecConfig::Mode::Column) ? parts[r].partB.local_dim : gc.N;
                size_t elems = m_loc * n_loc;
                parts[r].buffer.assign(all_buf.begin() + cursor, all_buf.begin() + cursor + elems);
                parts[r].M_local = m_loc;
                parts[r].N_local = n_loc;
                cursor += (int)elems;
            }
            std::vector<float> C_recon((size_t)gc.M * gc.N, 0.f);
            if (cfg.mode == TPGemmExecConfig::Mode::Column)
                TPGemmExecutor::reconstruct_columns(parts, C_recon.data(), gc.M, gc.N);
            else
                TPGemmExecutor::reconstruct_rows(parts, C_recon.data(), gc.M, gc.N);
            double max_abs = 0.0, diff_sq = 0.0, ref_sq = 0.0;
            size_t total = (size_t)gc.M * gc.N;
            for (size_t i = 0; i < total; ++i)
            {
                double d = (double)C_recon[i] - (double)C_ref[i];
                double ad = fabs(d);
                if (ad > max_abs)
                    max_abs = ad;
                diff_sq += d * d;
                ref_sq += (double)C_ref[i] * (double)C_ref[i];
            }
            double rel_l2 = ref_sq > 0.0 ? std::sqrt(diff_sq) / std::sqrt(ref_sq) : 0.0;
            EXPECT_LE(max_abs, gc.atol) << gc.tag << " rel_l2=" << rel_l2;
            EXPECT_LE(rel_l2, gc.rtol) << gc.tag << " max_abs=" << max_abs;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
} // namespace

// Test cases
TEST(TPGemmParity, ColumnEvenSplit_2way) { run_gemm_case({64, 48, 96, 2, TPGemmExecConfig::Mode::Column, "col_even_64x48x96_t2", 1e-5, 1e-6}); }
TEST(TPGemmParity, ColumnRaggedPrime_3way) { run_gemm_case({71, 53, 89, 3, TPGemmExecConfig::Mode::Column, "col_ragged_71x53x89_t3", 2e-5, 2e-6}); }
TEST(TPGemmParity, ColumnEven_4way) { run_gemm_case({64, 64, 128, 4, TPGemmExecConfig::Mode::Column, "col_even_64x64x128_t4", 2e-5, 2e-6}); }
TEST(TPGemmParity, RowEvenSplit_2way) { run_gemm_case({96, 64, 48, 2, TPGemmExecConfig::Mode::Row, "row_even_96x64x48_t2", 1e-5, 1e-6}); }
TEST(TPGemmParity, RowRaggedPrime_3way) { run_gemm_case({89, 71, 53, 3, TPGemmExecConfig::Mode::Row, "row_ragged_89x71x53_t3", 2e-5, 2e-6}); }
TEST(TPGemmParity, RowEven_4way) { run_gemm_case({128, 64, 64, 4, TPGemmExecConfig::Mode::Row, "row_even_128x64x64_t4", 2e-5, 2e-6}); }
// Degenerate single-rank (skip if world !=1). Useful for ensuring path doesn't break when tp_size=1.
TEST(TPGemmParity, DegenerateSingle) { run_gemm_case({33, 17, 25, 1, TPGemmExecConfig::Mode::Column, "degenerate_single", 1e-6, 1e-6}); }

// ---------------- Additional Shapes ----------------
TEST(TPGemmParity, ColumnWide_4way) { run_gemm_case({32, 64, 192, 4, TPGemmExecConfig::Mode::Column, "col_wide_32x64x192_t4", 3e-5, 3e-6}); }
TEST(TPGemmParity, RowTall_4way) { run_gemm_case({192, 64, 32, 4, TPGemmExecConfig::Mode::Row, "row_tall_192x64x32_t4", 3e-5, 3e-6}); }
TEST(TPGemmParity, ColumnModerate_2way) { run_gemm_case({128, 96, 160, 2, TPGemmExecConfig::Mode::Column, "col_mod_128x96x160_t2", 2e-5, 2e-6}); }
TEST(TPGemmParity, RowModerate_2way) { run_gemm_case({160, 96, 128, 2, TPGemmExecConfig::Mode::Row, "row_mod_160x96x128_t2", 2e-5, 2e-6}); }
TEST(TPGemmParity, ColumnRaggedLarge_3way) { run_gemm_case({73, 57, 101, 3, TPGemmExecConfig::Mode::Column, "col_rag_large_73x57x101_t3", 3e-5, 3e-6}); }
TEST(TPGemmParity, RowRaggedLarge_3way) { run_gemm_case({101, 57, 73, 3, TPGemmExecConfig::Mode::Row, "row_rag_large_101x57x73_t3", 3e-5, 3e-6}); }

// ---------------- Batched GEMM Support ----------------
namespace
{
    struct BatchedGemmCase
    {
        int batch;
        int M;
        int K;
        int N;
        int tp_size;
        TPGemmExecConfig::Mode mode;
        const char *tag;
        double atol;
        double rtol;
    };

    void fill_random_batch(std::vector<float> &v, uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto &x : v)
            x = dist(rng);
    }

    void run_batched_case(const BatchedGemmCase &bc)
    {
        int rank = 0, world = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world);
        if (world != bc.tp_size)
        {
            if (rank == 0)
                GTEST_SKIP() << "Requires tp_size=" << bc.tp_size << " ranks";
            return;
        }
        // Allocate A_all[batch,M,K], B_all[batch,K,N]
        std::vector<float> A_all((size_t)bc.batch * bc.M * bc.K);
        std::vector<float> B_all((size_t)bc.batch * bc.K * bc.N);
        for (int b = 0; b < bc.batch; ++b)
        {
            std::vector<float> tmpA((size_t)bc.M * bc.K), tmpB((size_t)bc.K * bc.N);
            fill_random_batch(tmpA, 1000 + b * 17 + bc.M + bc.N + bc.K);
            fill_random_batch(tmpB, 2000 + b * 31 + bc.M + bc.N + bc.K);
            std::memcpy(A_all.data() + (size_t)b * bc.M * bc.K, tmpA.data(), sizeof(float) * tmpA.size());
            std::memcpy(B_all.data() + (size_t)b * bc.K * bc.N, tmpB.data(), sizeof(float) * tmpB.size());
        }
        // Reference outputs per batch on rank 0
        std::vector<float> C_ref((size_t)bc.batch * bc.M * bc.N, 0.f);
        if (rank == 0)
        {
            for (int b = 0; b < bc.batch; ++b)
            {
                const float *A = A_all.data() + (size_t)b * bc.M * bc.K;
                const float *B = B_all.data() + (size_t)b * bc.K * bc.N;
                float *C = C_ref.data() + (size_t)b * bc.M * bc.N;
                for (int m = 0; m < bc.M; ++m)
                {
                    const float *arow = A + m * bc.K;
                    for (int n = 0; n < bc.N; ++n)
                    {
                        double acc = 0.0;
                        for (int k = 0; k < bc.K; ++k)
                            acc += (double)arow[k] * (double)B[k * bc.N + n];
                        C[m * bc.N + n] = (float)acc;
                    }
                }
            }
        }
        MPI_Bcast(C_ref.data(), (int)C_ref.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        TPGemmExecConfig cfg;
        cfg.mode = bc.mode;
        cfg.tp_size = bc.tp_size;
        cfg.tp_rank = rank;
        auto fn = [](const float *A, const float *B, float *C, size_t M, size_t N, size_t K) -> bool
        { return adaptive_matmul(A, B, C, (int)M, (int)N, (int)K, false); };

        for (int b = 0; b < bc.batch; ++b)
        {
            TPGemmExecutor exec(fn, cfg, bc.M, bc.N, bc.K);
            const float *A = A_all.data() + (size_t)b * bc.M * bc.K;
            const float *B = B_all.data() + (size_t)b * bc.K * bc.N;
            auto local = exec.run(A, B);
            int local_elems = (int)local.buffer.size();
            std::vector<int> recv_counts(world, 0);
            MPI_Gather(&local_elems, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
            std::vector<int> displs;
            std::vector<float> all_buf;
            if (rank == 0)
            {
                displs.resize(world, 0);
                int total = 0;
                for (int r = 0; r < world; ++r)
                {
                    displs[r] = total;
                    total += recv_counts[r];
                }
                all_buf.resize(total);
            }
            MPI_Gatherv(local.buffer.data(), local_elems, MPI_FLOAT, rank == 0 ? all_buf.data() : nullptr, recv_counts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
            if (rank == 0)
            {
                std::vector<TPGemmLocalResult> parts(world);
                int cursor = 0;
                for (int r = 0; r < world; ++r)
                {
                    parts[r].partA = (cfg.mode == TPGemmExecConfig::Mode::Row) ? compute_tp_partition(bc.M, bc.tp_size, r, TPPartitionSpec::Axis::Row) : TPPartitionSpec{};
                    parts[r].partB = (cfg.mode == TPGemmExecConfig::Mode::Column) ? compute_tp_partition(bc.N, bc.tp_size, r, TPPartitionSpec::Axis::Col) : TPPartitionSpec{};
                    size_t m_loc = (cfg.mode == TPGemmExecConfig::Mode::Row) ? parts[r].partA.local_dim : bc.M;
                    size_t n_loc = (cfg.mode == TPGemmExecConfig::Mode::Column) ? parts[r].partB.local_dim : bc.N;
                    size_t elems = m_loc * n_loc;
                    parts[r].buffer.assign(all_buf.begin() + cursor, all_buf.begin() + cursor + elems);
                    parts[r].M_local = m_loc;
                    parts[r].N_local = n_loc;
                    cursor += (int)elems;
                }
                std::vector<float> C_recon((size_t)bc.M * bc.N, 0.f);
                if (cfg.mode == TPGemmExecConfig::Mode::Column)
                    TPGemmExecutor::reconstruct_columns(parts, C_recon.data(), bc.M, bc.N);
                else
                    TPGemmExecutor::reconstruct_rows(parts, C_recon.data(), bc.M, bc.N);
                const float *Cgold = C_ref.data() + (size_t)b * bc.M * bc.N;
                double max_abs = 0.0, diff_sq = 0.0, ref_sq = 0.0;
                size_t total = (size_t)bc.M * bc.N;
                for (size_t i = 0; i < total; ++i)
                {
                    double d = (double)C_recon[i] - (double)Cgold[i];
                    double ad = fabs(d);
                    if (ad > max_abs)
                        max_abs = ad;
                    diff_sq += d * d;
                    ref_sq += (double)Cgold[i] * (double)Cgold[i];
                }
                double rel_l2 = ref_sq > 0.0 ? std::sqrt(diff_sq) / std::sqrt(ref_sq) : 0.0;
                EXPECT_LE(max_abs, bc.atol) << bc.tag << " batch_index=" << b << " rel_l2=" << rel_l2;
                EXPECT_LE(rel_l2, bc.rtol) << bc.tag << " batch_index=" << b << " max_abs=" << max_abs;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
} // anonymous namespace for batched helpers

// Batched GEMM test cases
TEST(TPGemmParity, BatchRowEvenSplit_2way) { run_batched_case({3, 64, 64, 64, 2, TPGemmExecConfig::Mode::Row, "batch_row_even_t2", 2e-5, 2e-6}); }
TEST(TPGemmParity, BatchColEvenSplit_2way) { run_batched_case({5, 48, 32, 96, 2, TPGemmExecConfig::Mode::Column, "batch_col_even_t2", 2e-5, 2e-6}); }
