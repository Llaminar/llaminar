#include "TestTensorUtils.h"
#include "TestReferenceImpls.h"
#include "TestTimeoutGuard.h"
#include "operators/MPILinearOperator.h"
#include "tensors/TensorFactory.h"
#include "TestMpiUtils.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <cmath>
#include <memory>
#include <cstdlib>

using namespace llaminar;

namespace
{
    struct ScopedMPIInit
    {
        ScopedMPIInit()
        {
            int flag;
            MPI_Initialized(&flag);
            if (!flag)
            {
                int provided;
                MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
            }
        }
        ~ScopedMPIInit()
        {
            // Do not finalize; shared across tests
        }
    };
}

class LinearOrientationTestFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mpi_guard = std::make_unique<ScopedMPIInit>();
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        world_rank = rank;
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        world_size = size;
    }
    std::unique_ptr<ScopedMPIInit> mpi_guard;
    int world_rank = 0;
    int world_size = 1;
};

static std::shared_ptr<TensorBase> make_simple(const std::vector<int> &shape)
{
    return TensorFactory::create_simple(shape);
}

static void fill_sequential(std::shared_ptr<TensorBase> &t, float offset = 0.f)
{
    for (int i = 0; i < t->size(); ++i)
        t->data()[i] = offset + float(i % 97) * 0.01f; // bounded predictable
}

TEST_F(LinearOrientationTestFixture, ParitySmallShapes)
{
    // A small battery of (M,K,N) shapes to catch orientation mistakes early.
    struct Case
    {
        int M, K, N;
    };
    std::vector<Case> cases = {
        {1, 4, 4}, {2, 3, 5}, {3, 7, 2}, {4, 8, 6}, {5, 5, 5}};

    MPILinearOperator kernel;

    const double max_abs_tol = 1e-6;
    const double rel_l2_tol = 1e-6;

    for (auto c : cases)
    {
        auto input = make_simple({c.M, c.K});
        auto weight = make_simple({c.N, c.K});
        auto bias = make_simple({c.N});
        auto output = make_simple({c.M, c.N});

        fill_sequential(input, 0.1f);
        fill_sequential(weight, 0.2f);
        fill_sequential(bias, 0.3f);

        std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight, bias};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        ASSERT_TRUE(kernel.execute(inputs, outputs));

        // Reference matmul + bias (single-process notion)
        // Weight is [N,K] so use transposed version
        auto ref = testref::matmul_with_transposed_b(input->data(), weight->data(), c.M, c.K, c.N);
        for (int m = 0; m < c.M; ++m)
        {
            for (int n = 0; n < c.N; ++n)
            {
                ref[m * c.N + n] += bias->data()[n];
            }
        }

        // Gather all ranks' outputs (kernel may allgather internally already, but ensure parity check on each rank)
        std::vector<float> produced(output->data(), output->data() + output->size());

        auto stats = testutils::diff(produced, ref);
        if (!(testutils::within(stats, max_abs_tol, rel_l2_tol)))
        {
            ADD_FAILURE() << "Shape (" << c.M << "," << c.K << "," << c.N << ") diff exceeded tolerance: "
                          << testutils::summarize(stats);
        }
    }
}

TEST_F(LinearOrientationTestFixture, RandomizedBatched)
{
    testutils::PRNG rng(1234);
    MPILinearOperator kernel;

    // Random shape palette (kept small for unit test runtime)
    for (int iter = 0; iter < 10; ++iter)
    {
        int M = 1 + (iter % 6);      // 1..6
        int K = 4 + (iter * 3) % 16; // vary
        int N = 2 + (iter * 5) % 20; // vary
        auto input = make_simple({M, K});
        auto weight = make_simple({N, K});
        auto bias = make_simple({N});
        auto output = make_simple({M, N});

        // Fill with uniform random data
        auto rnd_in = rng.uniform(M * K, 0.5f);
        std::copy(rnd_in.begin(), rnd_in.end(), input->data());
        auto rnd_w = rng.uniform(K * N, 0.5f);
        std::copy(rnd_w.begin(), rnd_w.end(), weight->data());
        auto rnd_b = rng.uniform(N, 0.1f);
        std::copy(rnd_b.begin(), rnd_b.end(), bias->data());

        std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight, bias};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        ASSERT_TRUE(kernel.execute(inputs, outputs));

        // Weight is [N,K] so use transposed version
        auto ref = testref::matmul_with_transposed_b(input->data(), weight->data(), M, K, N);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n)
                ref[m * N + n] += bias->data()[n];

        std::vector<float> produced(output->data(), output->data() + output->size());
        auto stats = testutils::diff(produced, ref);
        EXPECT_LE(stats.max_abs, 1e-5) << testutils::summarize(stats); // looser for random accumulation variability
        EXPECT_LE(stats.rel_l2, 1e-6) << testutils::summarize(stats);
    }
}

LLAMINAR_DEFINE_GTEST_MPI_MAIN();
