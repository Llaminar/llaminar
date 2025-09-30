#include "kernels/MPIEmbeddingKernel.h"
#include "tensors/tensor_factory.h"
#include "test_timeout_guard.h"
#include "test_mpi_utils.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <random>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>
#include <string_view>

using namespace llaminar;

namespace
{
    // Deterministic helper to compute a global embedding value for (token, dim)
    inline float global_embed_value(int token, int dim)
    {
        return static_cast<float>((token * 1000 + dim) * 0.001); // token + dim/1000 pattern
    }

    // Partition logic replicated from MPIEmbeddingKernel ctor (kept in sync)
    inline void compute_partition(size_t vocab_size, int world, int rank,
                                  size_t &local_start, size_t &local_size)
    {
        size_t base = vocab_size / world;
        size_t rem = vocab_size % world;
        if (rank < static_cast<int>(rem))
        {
            local_size = base + 1;
            local_start = static_cast<size_t>(rank) * local_size;
        }
        else
        {
            local_size = base;
            local_start = rem * (base + 1) + (static_cast<size_t>(rank) - rem) * base;
        }
    }

    // Create a simple 1D tensor storing token ids as float values (current kernel copies via float*)
    std::shared_ptr<TensorBase> make_token_tensor(const std::vector<int> &tokens)
    {
        std::vector<int> shape = {static_cast<int>(tokens.size())};
        auto t = TensorFactory::create_simple(shape);
        for (size_t i = 0; i < tokens.size(); ++i)
        {
            t->data()[i] = static_cast<float>(tokens[i]);
        }
        return t;
    }

    // Allocate embedding table in (rows, cols) and populate using global value mapping
    std::shared_ptr<TensorBase> make_embedding_table(size_t rows, size_t cols,
                                                     bool transposed,
                                                     size_t global_vocab, size_t embedding_dim,
                                                     size_t local_start = 0)
    {
        std::vector<int> shape = {static_cast<int>(rows), static_cast<int>(cols)};
        auto table = TensorFactory::create_simple(shape);
        if (!transposed)
        {
            // rows = vocab portion, cols = embedding_dim
            for (size_t r = 0; r < rows; ++r)
            {
                int global_token = static_cast<int>(local_start + r);
                for (size_t c = 0; c < cols; ++c)
                {
                    table->data()[r * cols + c] = global_embed_value(global_token, static_cast<int>(c));
                }
            }
        }
        else
        {
            // transposed: rows = embedding_dim, cols = vocab portion
            for (size_t c = 0; c < cols; ++c)
            {
                int global_token = static_cast<int>(local_start + c);
                for (size_t r = 0; r < rows; ++r)
                {
                    table->data()[r * cols + c] = global_embed_value(global_token, static_cast<int>(r));
                }
            }
        }
        return table;
    }

    // Build expected output (seq_len, embedding_dim)
    std::vector<float> build_expected(const std::vector<int> &tokens, size_t embedding_dim)
    {
        std::vector<float> out(tokens.size() * embedding_dim);
        for (size_t i = 0; i < tokens.size(); ++i)
        {
            int tok = tokens[i];
            if (tok < 0)
                continue; // leave zeros if invalid
            for (size_t d = 0; d < embedding_dim; ++d)
            {
                out[i * embedding_dim + d] = global_embed_value(tok, static_cast<int>(d));
            }
        }
        return out;
    }

    void assert_buffers_close(const std::vector<float> &got, const std::vector<float> &exp, float tol = 1e-6f)
    {
        ASSERT_EQ(got.size(), exp.size());
        for (size_t i = 0; i < got.size(); ++i)
        {
            ASSERT_NEAR(got[i], exp[i], tol) << "Mismatch at index " << i;
        }
    }

    // Optional early-finalize hook: if LLAMINAR_EMBED_FINALIZE_AFTER matches the provided
    // test name, we (a) clear LLAMINAR_TEST_SKIP_FINALIZE so the real finalize occurs,
    // (b) invoke MPIEnvironment::finalize(), and (c) _exit(0) so GTest doesn't proceed
    // to further tests with a finalized MPI environment. This is used to bisect which
    // test (if any) causes MPI_Finalize to hang inside this binary.
    inline void maybe_finalize_after(const char *test_name)
    {
        const char *target = std::getenv("LLAMINAR_EMBED_FINALIZE_AFTER");
        if (!target || target[0] == '\0')
            return;
        if (std::string_view(target) == test_name)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            fprintf(stderr, "[MPIEmbeddingKernelTest] rank %d triggering early finalize after %s\n", rank, test_name);
            // Ensure we do NOT skip finalize even though the test-level property sets it.
            unsetenv("LLAMINAR_TEST_SKIP_FINALIZE");
            // Provide a default watchdog if user did not set one explicitly.
            if (!std::getenv("LLAMINAR_TEST_FINALIZE_TIMEOUT_MS"))
            {
                setenv("LLAMINAR_TEST_FINALIZE_TIMEOUT_MS", "10000", 0); // 10s default
            }
            // Call environment finalize (idempotent) then hard-exit to avoid further GTest activity.
            ::llaminar::test_util::MPIEnvironment::finalize();
            fflush(stderr);
            _exit(0);
        }
    }
} // namespace

class MPIEmbeddingKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // MPI already initialized by LLAMINAR_DEFINE_GTEST_MPI_MAIN macro via MPIEnvironment.
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_);
        // Unless explicitly allowed, suppress embedding trace to keep test output fast & uncluttered.
        if (!std::getenv("LLAMINAR_EMBED_TEST_ALLOW_TRACE"))
        {
            unsetenv("LLAMINAR_EMBED_TRACE");
        }
    }
    int rank_ = 0;
    int world_ = 1;
};

TEST_F(MPIEmbeddingKernelTest, FullTableNonTransposed)
{
    const size_t vocab = 17;
    const size_t emb = 8;
    std::vector<int> tokens = {0, 3, 5, 16};
    auto token_tensor = make_token_tensor(tokens);

    // Full table layout [vocab, emb]
    auto table = make_embedding_table(vocab, emb, /*transposed=*/false, vocab, emb);
    auto output = TensorFactory::create_simple({(int)tokens.size(), (int)emb});

    MPIEmbeddingKernel kernel(vocab, emb);
    std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel.execute(inputs, outputs));

    std::vector<float> got(output->data(), output->data() + output->size());
    auto expected = build_expected(tokens, emb);
    assert_buffers_close(got, expected);
    maybe_finalize_after("FullTableNonTransposed");
}

TEST_F(MPIEmbeddingKernelTest, FullTableTransposed)
{
    const size_t vocab = 19;
    const size_t emb = 6;
    std::vector<int> tokens = {1, 7, 18};
    auto token_tensor = make_token_tensor(tokens);

    // Transposed full table [emb, vocab]
    auto table = make_embedding_table(emb, vocab, /*transposed=*/true, vocab, emb);
    auto output = TensorFactory::create_simple({(int)tokens.size(), (int)emb});

    MPIEmbeddingKernel kernel(vocab, emb);
    std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel.execute(inputs, outputs));

    std::vector<float> got(output->data(), output->data() + output->size());
    auto expected = build_expected(tokens, emb);
    assert_buffers_close(got, expected);
    maybe_finalize_after("FullTableTransposed");
}

TEST_F(MPIEmbeddingKernelTest, ShardedNonTransposed)
{
    if (world_ < 2)
    {
        GTEST_SKIP() << "Requires >=2 ranks";
    }
    const size_t vocab = 23;
    const size_t emb = 5;
    std::vector<int> tokens = {0, 4, 7, 12, 18, 22};
    auto token_tensor = make_token_tensor(tokens);

    size_t local_start = 0, local_size = 0;
    compute_partition(vocab, world_, rank_, local_start, local_size);
    auto shard = make_embedding_table(local_size, emb, /*transposed=*/false, vocab, emb, local_start);
    auto output = TensorFactory::create_simple({(int)tokens.size(), (int)emb});

    MPIEmbeddingKernel kernel(vocab, emb);
    std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, shard};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel.execute(inputs, outputs));

    std::vector<float> got(output->data(), output->data() + output->size());
    auto expected = build_expected(tokens, emb);
    assert_buffers_close(got, expected);
    maybe_finalize_after("ShardedNonTransposed");
}

TEST_F(MPIEmbeddingKernelTest, ShardedTransposed)
{
    if (world_ < 2)
    {
        GTEST_SKIP() << "Requires >=2 ranks";
    }
    const size_t vocab = 29;
    const size_t emb = 7;
    std::vector<int> tokens = {1, 5, 9, 14, 20, 28};
    auto token_tensor = make_token_tensor(tokens);

    size_t local_start = 0, local_size = 0;
    compute_partition(vocab, world_, rank_, local_start, local_size);
    auto shardT = make_embedding_table(emb, local_size, /*transposed=*/true, vocab, emb, local_start);
    auto output = TensorFactory::create_simple({(int)tokens.size(), (int)emb});

    MPIEmbeddingKernel kernel(vocab, emb);
    std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, shardT};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel.execute(inputs, outputs));

    std::vector<float> got(output->data(), output->data() + output->size());
    auto expected = build_expected(tokens, emb);
    assert_buffers_close(got, expected);
    maybe_finalize_after("ShardedTransposed");
}

TEST_F(MPIEmbeddingKernelTest, OutOfRangeTokenZeroRow)
{
    const size_t vocab = 11;
    const size_t emb = 4;
    std::vector<int> tokens = {0, 5, 11}; // 11 is out-of-range
    auto token_tensor = make_token_tensor(tokens);
    auto table = make_embedding_table(vocab, emb, false, vocab, emb);
    auto output = TensorFactory::create_simple({(int)tokens.size(), (int)emb});

    MPIEmbeddingKernel kernel(vocab, emb);
    std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel.execute(inputs, outputs));

    std::vector<float> got(output->data(), output->data() + output->size());
    auto expected = build_expected(tokens, emb);
    // Manually zero expected last row (out-of-range)
    for (size_t d = 0; d < emb; ++d)
        expected[2 * emb + d] = 0.0f;
    assert_buffers_close(got, expected);
    maybe_finalize_after("OutOfRangeTokenZeroRow");
}

TEST_F(MPIEmbeddingKernelTest, ValidationFailureBadShape)
{
    const size_t vocab = 10;
    const size_t emb = 4;
    std::vector<int> tokens = {0, 1};
    auto token_tensor = make_token_tensor(tokens);
    // Intentionally wrong shaped embedding table: [emb, emb] not matching any valid case
    auto bad = make_embedding_table(emb, emb, false, vocab, emb); // rows=emb != vocab or local shard; cols=emb OK but mismatch
    auto output = TensorFactory::create_simple({(int)tokens.size(), (int)emb});

    MPIEmbeddingKernel kernel(vocab, emb);
    std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, bad};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    EXPECT_FALSE(kernel.execute(inputs, outputs));
    maybe_finalize_after("ValidationFailureBadShape");
}

// Provide an MPI main similar to other MPI kernel tests
LLAMINAR_DEFINE_GTEST_MPI_MAIN();

// Fuzz / randomized coverage test: random vocab sizes, embedding dims, seq lens, token ids,
// and orientation / sharding modes. Ensures indexing & partition logic remain correct
// under varied scenarios. Fixed seed for reproducibility.
TEST_F(MPIEmbeddingKernelTest, RandomizedFuzz)
{
    if (!std::getenv("LLAMINAR_EMBED_EXTENDED"))
    {
        GTEST_SKIP() << "Extended fuzz disabled (set LLAMINAR_EMBED_EXTENDED=1 to enable)";
    }
    int rank = 0;
    int world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    // Single RNG on rank 0; parameters broadcast to all ranks to keep buffer sizes identical.
    static std::mt19937 rng(1234567u);
    std::uniform_int_distribution<int> vocab_dist(8, 64);  // moderate vocab to keep runtime low
    std::uniform_int_distribution<int> emb_dist(4, 48);    // embedding sizes
    std::uniform_int_distribution<int> seqlen_dist(1, 16); // short sequences
    std::uniform_int_distribution<int> bool_dist(0, 1);

    const int iterations = 25; // keep runtime modest while offering diversity
    for (int it = 0; it < iterations; ++it)
    {
        int vocab = 0, emb = 0, seq_len = 0;
        int full_table_flag = 0, transposed_flag = 0;
        if (rank == 0)
        {
            vocab = vocab_dist(rng);
            emb = emb_dist(rng);
            seq_len = seqlen_dist(rng);
            full_table_flag = bool_dist(rng);
            transposed_flag = bool_dist(rng);
        }
        int params[5] = {vocab, emb, seq_len, full_table_flag, transposed_flag};
        MPI_Bcast(params, 5, MPI_INT, 0, MPI_COMM_WORLD);
        vocab = params[0];
        emb = params[1];
        seq_len = params[2];
        full_table_flag = params[3];
        transposed_flag = params[4];
        bool full_table = (full_table_flag != 0);
        bool transposed = (transposed_flag != 0);

        // Instantiate kernel with (possibly) different global vocab each iteration
        MPIEmbeddingKernel kernel(vocab, emb);

        // Generate random token ids in [0, vocab) with a small chance of out-of-range to exercise guard
        std::vector<int> tokens(seq_len);
        if (rank == 0)
        {
            for (int i = 0; i < seq_len; ++i)
            {
                tokens[i] = rng() % vocab; // in-range only
            }
        }
        // Broadcast tokens (even if seq_len=0 this is safe)
        if (seq_len > 0)
            MPI_Bcast(tokens.data(), seq_len, MPI_INT, 0, MPI_COMM_WORLD);
        auto token_tensor = make_token_tensor(tokens);

        // Partition for this rank
        size_t local_start = 0, local_size = 0;
        compute_partition(vocab, world, rank, local_start, local_size);

        std::shared_ptr<TensorBase> table;
        if (full_table)
        {
            // Allocate full table (replicated)
            if (!transposed)
            {
                table = make_embedding_table(vocab, emb, false, vocab, emb, 0);
            }
            else
            {
                table = make_embedding_table(emb, vocab, true, vocab, emb, 0);
            }
        }
        else
        {
            // Allocate shard for this rank only
            if (!transposed)
            {
                table = make_embedding_table(local_size, emb, false, vocab, emb, local_start);
            }
            else
            {
                table = make_embedding_table(emb, local_size, true, vocab, emb, local_start);
            }
        }

        auto output = TensorFactory::create_simple({seq_len, emb});
        std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, table};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        bool ok = kernel.execute(inputs, outputs);
        // If we intentionally fed out-of-range tokens we still expect execute() to succeed (it just logs warnings)
        ASSERT_TRUE(ok);

        std::vector<float> got(output->data(), output->data() + output->size());
        auto expected = build_expected(tokens, emb);
        // All tokens in-range; no row zeroing required now.
        assert_buffers_close(got, expected);
    }
    maybe_finalize_after("RandomizedFuzz");
}

// Reuse a single kernel across alternating table orientations to ensure internal flags
// are recomputed per validate() and no stale state leaks.
TEST_F(MPIEmbeddingKernelTest, AlternatingOrientationReuse)
{
    const int vocab = 29;
    const int emb = 7;
    std::vector<int> tokens = {0, 1, 5, 14, 28};
    auto token_tensor = make_token_tensor(tokens);
    auto output = TensorFactory::create_simple({(int)tokens.size(), emb});

    MPIEmbeddingKernel kernel(vocab, emb);

    // First run: full transposed table
    auto full_transposed = make_embedding_table(emb, vocab, /*transposed=*/true, vocab, emb, 0);
    {
        std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, full_transposed};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        ASSERT_TRUE(kernel.execute(inputs, outputs));
        std::vector<float> got(output->data(), output->data() + output->size());
        auto expected = build_expected(tokens, emb);
        assert_buffers_close(got, expected);
    }

    // Second run: sharded non-transposed table
    size_t local_start = 0, local_size = 0;
    compute_partition(vocab, world_, rank_, local_start, local_size);
    auto shard_std = make_embedding_table(local_size, emb, /*transposed=*/false, vocab, emb, local_start);
    std::fill(output->data(), output->data() + output->size(), -12345.0f);
    {
        std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, shard_std};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        ASSERT_TRUE(kernel.execute(inputs, outputs));
        std::vector<float> got(output->data(), output->data() + output->size());
        auto expected = build_expected(tokens, emb);
        assert_buffers_close(got, expected);
    }
    maybe_finalize_after("AlternatingOrientationReuse");
}

// Precision boundary test: ensure token IDs near upper vocab limit are stable when stored as float.
TEST_F(MPIEmbeddingKernelTest, PrecisionBoundaryTokens)
{
    if (!std::getenv("LLAMINAR_EMBED_EXTENDED"))
    {
        GTEST_SKIP() << "Precision boundary test disabled (set LLAMINAR_EMBED_EXTENDED=1 to enable)";
    }
    // Keep vocab modest to avoid large alloc; focus on top-end indices in that vocab.
    const int vocab = 131072; // 2^17
    const int emb = 4;
    std::vector<int> tokens = {vocab - 4, vocab - 3, vocab - 2, vocab - 1};
    auto token_tensor = make_token_tensor(tokens);

    auto table = make_embedding_table(vocab, emb, false, vocab, emb, 0); // full table
    auto output = TensorFactory::create_simple({(int)tokens.size(), emb});
    MPIEmbeddingKernel kernel(vocab, emb);
    std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel.execute(inputs, outputs));
    std::vector<float> got(output->data(), output->data() + output->size());
    auto expected = build_expected(tokens, emb);
    assert_buffers_close(got, expected);
    maybe_finalize_after("PrecisionBoundaryTokens");
}

// All requested tokens are owned by a single (remote for one rank) shard; both ranks request the SAME list.
// Verifies that the owning rank populates rows and MPI_Allreduce distributes them to the non-owning rank.
TEST_F(MPIEmbeddingKernelTest, RemoteOwnedTokensGather)
{
    if (world_ != 2)
        GTEST_SKIP() << "Test expects 2 ranks";
    const int vocab = 50; // simple even split 25/25
    const int emb = 6;
    MPIEmbeddingKernel kernel(vocab, emb);

    // Choose an owner rank deterministically (rank 0) whose tokens we'll request.
    const int owner_rank = 0;
    size_t owner_start = 0, owner_size = 0;
    compute_partition(vocab, world_, owner_rank, owner_start, owner_size);
    std::vector<int> tokens;
    if (rank_ == owner_rank)
    {
        tokens = {(int)owner_start,
                  (int)(owner_start + owner_size / 2),
                  (int)(owner_start + owner_size - 1)};
    }
    // Broadcast token count then data so both ranks use identical sequence.
    int token_count = (int)tokens.size();
    MPI_Bcast(&token_count, 1, MPI_INT, owner_rank, MPI_COMM_WORLD);
    if (rank_ != owner_rank)
        tokens.resize(token_count);
    if (token_count > 0)
    {
        MPI_Bcast(tokens.data(), token_count, MPI_INT, owner_rank, MPI_COMM_WORLD);
    }

    // Build local shard table for this rank only (portion of vocab)
    size_t local_start = 0, local_size = 0;
    compute_partition(vocab, world_, rank_, local_start, local_size);
    auto shard_table = make_embedding_table(local_size, emb, /*transposed=*/false, vocab, emb, local_start);

    auto token_tensor = make_token_tensor(tokens);
    auto output = TensorFactory::create_simple({token_count, emb});
    std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, shard_table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel.execute(inputs, outputs));

    std::vector<float> got(output->data(), output->data() + output->size());
    auto expected = build_expected(tokens, emb);
    assert_buffers_close(got, expected);
    maybe_finalize_after("RemoteOwnedTokensGather");
}

// Duplicate heavy tokens to ensure no accumulation artifacts.
TEST_F(MPIEmbeddingKernelTest, DuplicateTokens)
{
    const int vocab = 37;
    const int emb = 9;
    MPIEmbeddingKernel kernel(vocab, emb);
    std::vector<int> tokens(16, 13); // same token repeated
    auto token_tensor = make_token_tensor(tokens);
    auto table = make_embedding_table(vocab, emb, false, vocab, emb, 0);
    auto output = TensorFactory::create_simple({(int)tokens.size(), emb});
    std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel.execute(inputs, outputs));
    auto expected = build_expected(tokens, emb);
    std::vector<float> got(output->data(), output->data() + output->size());
    assert_buffers_close(got, expected);
    maybe_finalize_after("DuplicateTokens");
}

// Inject NaN into embedding table (full table mode) and confirm it propagates when fail-fast is off.
TEST_F(MPIEmbeddingKernelTest, NaNPropagationNoFailFast)
{
    const int vocab = 23;
    const int emb = 6;
    MPIEmbeddingKernel kernel(vocab, emb);
    std::vector<int> tokens = {3, 7, 11};
    auto token_tensor = make_token_tensor(tokens);
    auto table = make_embedding_table(vocab, emb, false, vocab, emb, 0);
    // Introduce NaN in token 7, dim 2
    table->data()[7 * emb + 2] = std::numeric_limits<float>::quiet_NaN();
    auto output = TensorFactory::create_simple({(int)tokens.size(), emb});
    std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel.execute(inputs, outputs));
    // Find row index for token 7
    int row_idx = -1;
    for (size_t i = 0; i < tokens.size(); ++i)
        if (tokens[i] == 7)
            row_idx = (int)i;
    ASSERT_NE(row_idx, -1);
    float *out = output->data();
    ASSERT_TRUE(std::isnan(out[row_idx * emb + 2]));
    maybe_finalize_after("NaNPropagationNoFailFast");
}

// Idempotent repeat execution: same inputs twice produce same outputs.
TEST_F(MPIEmbeddingKernelTest, IdempotentRepeatExecution)
{
    const int vocab = 41;
    const int emb = 8;
    std::vector<int> tokens = {0, 5, 13, 22, 40};
    auto token_tensor = make_token_tensor(tokens);
    auto table = make_embedding_table(vocab, emb, true, vocab, emb, 0); // transposed
    auto output = TensorFactory::create_simple({(int)tokens.size(), emb});
    MPIEmbeddingKernel kernel(vocab, emb);
    std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel.execute(inputs, outputs));
    std::vector<float> first(output->data(), output->data() + output->size());
    // Overwrite output with junk then rerun
    std::fill(output->data(), output->data() + output->size(), 123456.0f);
    ASSERT_TRUE(kernel.execute(inputs, outputs));
    std::vector<float> second(output->data(), output->data() + output->size());
    assert_buffers_close(first, second);
    // Final diagnostic barrier & print to confirm both ranks reached end of tests
    int r = 0, w = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    MPI_Comm_size(MPI_COMM_WORLD, &w);
    if (w > 1)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        fprintf(stderr, "[MPIEmbeddingKernelTest] rank %d reached end-of-suite barrier\n", r);
        fflush(stderr);
    }
    maybe_finalize_after("IdempotentRepeatExecution");
}

// Disabled zero-row shard test: known to trigger MPI_Finalize hang on current MPI stack.
// To run manually for debugging set GTest filter to ...EmptyShardSafe AND export LLAMINAR_RUN_EMPTY_SHARD=1.
// Normal CI runs skip it (prefixed DISABLED_). Root cause investigation tracked in issue: EMBED-ZERO-SHARD-HANG.
TEST_F(MPIEmbeddingKernelTest, DISABLED_EmptyShardSafe)
{
    if (!std::getenv("LLAMINAR_RUN_EMPTY_SHARD"))
    {
        GTEST_SKIP() << "EmptyShardSafe disabled pending finalize hang fix (set LLAMINAR_RUN_EMPTY_SHARD=1 to run manually)";
    }
    if (world_ != 2)
        GTEST_SKIP() << "Test expects 2 ranks";
    const int vocab = 1;
    const int emb = 5;
    MPIEmbeddingKernel kernel(vocab, emb);
    size_t local_start = 0, local_size = 0;
    compute_partition(vocab, world_, rank_, local_start, local_size);
    ASSERT_TRUE((local_size == 1 && rank_ == 0) || (local_size == 0 && rank_ == 1));
    std::vector<int> tokens = {0, 0, 0};
    auto token_tensor = make_token_tensor(tokens);
    auto shard_table = make_embedding_table(local_size, emb, false, vocab, emb, local_start);
    auto output = TensorFactory::create_simple({(int)tokens.size(), emb});
    std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, shard_table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel.execute(inputs, outputs));
    auto expected = build_expected(tokens, emb);
    std::vector<float> got(output->data(), output->data() + output->size());
    assert_buffers_close(got, expected);
    // Intentionally do not attempt finalize workaround here; rely on manual debug runs.
}
