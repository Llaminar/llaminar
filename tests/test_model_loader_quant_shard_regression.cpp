// Regression test for quantized shard support (phase 1 full-load slicing fallback).
// Validates that ModelLoader::loadTensorColumnShards and loadTensorRowShard
// produce exact fp32 slices matching a full tensor load for a representative
// quantized 2D tensor. Ensures future optimizations (partial block decode)
// preserve correctness.

#include <gtest/gtest.h>
#include <filesystem>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <random>
#include <mpi.h>

#include "logger.h"
#include "model_loader.h"

using namespace llaminar;

namespace
{
    std::string pick_model()
    {
        if (const char *env = std::getenv("LLAMINAR_SHARD_TEST_MODEL"))
            return env;
        namespace fs = std::filesystem;
        fs::path dir{"models"};
        if (!fs::exists(dir))
            return {};
        // Pick smallest .gguf file (heuristic for faster CI) containing at least one quantized 2D tensor.
        uintmax_t best_size = (std::numeric_limits<uintmax_t>::max)();
        std::string best_path;
        for (auto &p : fs::directory_iterator(dir))
        {
            if (!p.is_regular_file() || p.path().extension() != ".gguf")
                continue;
            auto sz = std::filesystem::file_size(p.path());
            if (sz >= best_size)
                continue; // only consider better candidates
            ModelLoader loader;
            if (!loader.loadModel(p.path().string()))
                continue;
            bool has_quant2d = false;
            for (auto &ti : loader.getModel().tensors)
            {
                if (ti.dimensions.size() == 2 && ti.isQuantized())
                {
                    has_quant2d = true;
                    break;
                }
            }
            if (has_quant2d)
            {
                best_size = sz;
                best_path = p.path().string();
            }
        }
        return best_path;
    }
}

TEST(ModelLoaderQuantShardRegression, ColumnAndRowShardParity)
{
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    if (!mpi_init)
    {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SINGLE, &provided);
    }
    std::string model_path = pick_model();
    if (model_path.empty())
    {
        GTEST_SKIP() << "No model available for shard regression test (set LLAMINAR_SHARD_TEST_MODEL)";
    }
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path)) << "Failed to load model: " << model_path;

    const char *forced_tensor = std::getenv("LLAMINAR_SHARD_TEST_TENSOR");
    const GGUFTensorInfo *chosen = nullptr;
    if (forced_tensor)
    {
        chosen = loader.getModel().findTensor(forced_tensor);
        if (!chosen)
            GTEST_SKIP() << "Forced tensor '" << forced_tensor << "' not found";
        if (!(chosen->isQuantized() && chosen->dimensions.size() == 2))
        {
            GTEST_SKIP() << "Forced tensor not quantized 2D";
        }
    }
    else
    {
        // Select first suitable quantized 2D tensor under size limits for quick test.
        const size_t ELEM_LIMIT = 5'000'000; // safeguard against huge embeddings
        for (auto &ti : loader.getModel().tensors)
        {
            if (ti.dimensions.size() != 2 || !ti.isQuantized())
                continue;
            uint64_t rows = ti.dimensions[0];
            uint64_t cols = ti.dimensions[1];
            if (rows == 0 || cols == 0)
                continue;
            if (rows * cols > ELEM_LIMIT)
                continue;
            if (cols < 8 || rows < 2)
                continue; // need enough width for multi-shard
            chosen = &ti;
            break;
        }
    }
    if (!chosen)
    {
        GTEST_SKIP() << "No suitable quantized 2D tensor found in model (consider smaller model or set LLAMINAR_SHARD_TEST_TENSOR)";
    }

    // Full load reference (fp32) via generic API (dequant + convert).
    auto full_tensor = loader.loadTensor(chosen->name);
    ASSERT_TRUE(full_tensor) << "Full tensor load failed for " << chosen->name;
    ASSERT_EQ(full_tensor->shape().size(), 2u);
    int rows = full_tensor->shape()[0];
    int cols = full_tensor->shape()[1];
    ASSERT_GT(cols, 2);

    // --- Column shard parity ---
    // Partition columns into three shards with uneven sizes covering the entire width.
    int c0 = std::max(1, cols / 5);
    int c1 = std::max(1, cols / 7);
    if (c0 + c1 >= cols)
        c1 = std::max(1, (cols - c0) / 2);
    if (c0 + c1 >= cols)
        c1 = 1; // final safeguard
    int c2 = cols - c0 - c1;
    if (c2 <= 0)
    {
        c2 = 1;
        if (c0 > 1)
            --c0;
        else if (c1 > 1)
            --c1;
    }
    ASSERT_EQ(c0 + c1 + c2, cols);
    std::vector<int> col_offsets = {0, c0, c0 + c1};
    std::vector<int> col_counts = {c0, c1, c2};
    std::vector<std::vector<float>> shard_bufs(3);
    for (int i = 0; i < 3; ++i)
        shard_bufs[i].resize((size_t)rows * col_counts[i]);
    std::vector<float *> dest_ptrs = {shard_bufs[0].data(), shard_bufs[1].data(), shard_bufs[2].data()};
    ASSERT_TRUE(loader.loadTensorColumnShards(chosen->name, col_offsets, col_counts, dest_ptrs))
        << "Column shard load failed for tensor " << chosen->name;

    // Reconstruct full matrix from shards.
    std::vector<float> recon_cols((size_t)rows * cols, -1234.f);
    for (int r = 0; r < rows; ++r)
    {
        float *row_dst = recon_cols.data() + (size_t)r * cols;
        // Copy in order of offsets
        std::memcpy(row_dst + col_offsets[0], shard_bufs[0].data() + (size_t)r * c0, sizeof(float) * c0);
        std::memcpy(row_dst + col_offsets[1], shard_bufs[1].data() + (size_t)r * c1, sizeof(float) * c1);
        std::memcpy(row_dst + col_offsets[2], shard_bufs[2].data() + (size_t)r * c2, sizeof(float) * c2);
    }
    // Compare
    const float *full = full_tensor->data();
    double max_abs_col = 0.0, diff_sq_col = 0.0, ref_sq_col = 0.0;
    for (size_t i = 0; i < recon_cols.size(); ++i)
    {
        double d = (double)recon_cols[i] - full[i];
        if (std::fabs(d) > max_abs_col)
            max_abs_col = std::fabs(d);
        diff_sq_col += d * d;
        ref_sq_col += (double)full[i] * full[i];
    }
    double rel_l2_col = (ref_sq_col > 0.0) ? std::sqrt(diff_sq_col / ref_sq_col) : 0.0;
    EXPECT_LT(max_abs_col, 1e-9) << "Column shard parity max_abs mismatch";
    EXPECT_LT(rel_l2_col, 1e-10) << "Column shard parity rel_l2 mismatch";

    // --- Row shard parity ---
    int row_offset = rows / 5;
    int row_count = std::max(2, rows / 4);
    if (row_offset + row_count > rows)
        row_count = rows - row_offset;
    std::vector<float> row_shard((size_t)row_count * cols, 0.f);
    ASSERT_TRUE(loader.loadTensorRowShard(chosen->name, row_offset, row_count, row_shard.data()))
        << "Row shard load failed";
    double max_abs_row = 0.0, diff_sq_row = 0.0, ref_sq_row = 0.0;
    for (int r = 0; r < row_count; ++r)
    {
        const float *ref_row = full + (size_t)(row_offset + r) * cols;
        const float *sh_row = row_shard.data() + (size_t)r * cols;
        for (int c = 0; c < cols; ++c)
        {
            double d = (double)sh_row[c] - ref_row[c];
            if (std::fabs(d) > max_abs_row)
                max_abs_row = std::fabs(d);
            diff_sq_row += d * d;
            ref_sq_row += (double)ref_row[c] * ref_row[c];
        }
    }
    double rel_l2_row = (ref_sq_row > 0.0) ? std::sqrt(diff_sq_row / ref_sq_row) : 0.0;
    EXPECT_LT(max_abs_row, 1e-9) << "Row shard parity max_abs mismatch";
    EXPECT_LT(rel_l2_row, 1e-10) << "Row shard parity rel_l2 mismatch";

    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized)
    {
        int was_init = mpi_init;
        MPI_Barrier(MPI_COMM_WORLD);
        if (!was_init)
            MPI_Finalize();
    }
}

// Randomized stress variant: multiple random column partition patterns and row slices.
// Skips if the selected tensor is too large to avoid long CI runtimes.
TEST(ModelLoaderQuantShardRegression, RandomizedMultiPatternParity)
{
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    if (!mpi_init)
    {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SINGLE, &provided);
    }
    std::string model_path = pick_model();
    if (model_path.empty())
    {
        GTEST_SKIP() << "No model available for randomized shard test";
    }
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path));
    const GGUFTensorInfo *chosen = nullptr;
    const size_t ELEM_LIMIT = 1'500'000; // tighter limit for multi-iteration test
    for (auto &ti : loader.getModel().tensors)
    {
        if (ti.dimensions.size() != 2 || !ti.isQuantized())
            continue;
        uint64_t rows = ti.dimensions[0];
        uint64_t cols = ti.dimensions[1];
        if (rows == 0 || cols == 0)
            continue;
        if (rows * cols > ELEM_LIMIT)
            continue;
        if (cols < 8 || rows < 2)
            continue;
        chosen = &ti;
        break;
    }
    if (!chosen)
    {
        GTEST_SKIP() << "No suitably small quant 2D tensor under element limit";
    }
    auto full_tensor = loader.loadTensor(chosen->name);
    ASSERT_TRUE(full_tensor);
    ASSERT_EQ(full_tensor->shape().size(), 2u);
    int rows = full_tensor->shape()[0];
    int cols = full_tensor->shape()[1];
    const float *full = full_tensor->data();

    std::mt19937 rng(123456); // deterministic for reproducibility
    std::uniform_int_distribution<int> n_shards_dist(2, 5);
    std::uniform_int_distribution<int> row_off_dist(0, std::max(0, rows - 2));
    std::uniform_int_distribution<int> row_len_dist(1, std::max(1, rows / 2));

    const int ITER = 5; // number of randomized patterns
    for (int iter = 0; iter < ITER; ++iter)
    {
        int n_shards = std::min(n_shards_dist(rng), cols); // cannot exceed cols
        // Generate n_shards positive lengths summing to cols.
        // Strategy: sample n_shards-1 cut points, sort, derive segment lengths.
        std::vector<int> cuts;
        cuts.reserve(n_shards - 1);
        std::uniform_int_distribution<int> cut_dist(1, cols - 1);
        while ((int)cuts.size() < n_shards - 1)
        {
            int c = cut_dist(rng);
            if (std::find(cuts.begin(), cuts.end(), c) == cuts.end())
                cuts.push_back(c);
        }
        std::sort(cuts.begin(), cuts.end());
        std::vector<int> offsets;
        offsets.reserve(n_shards);
        std::vector<int> counts;
        counts.reserve(n_shards);
        int prev = 0;
        for (int c : cuts)
        {
            counts.push_back(c - prev);
            offsets.push_back(prev);
            prev = c;
        }
        counts.push_back(cols - prev);
        offsets.push_back(prev);
        // Ensure no zero lengths
        bool zero = false;
        for (int c : counts)
            if (c <= 0)
                zero = true;
        if (zero)
        {
            --iter;
            continue;
        }
        std::vector<std::vector<float>> buffers(n_shards);
        std::vector<float *> dests;
        dests.reserve(n_shards);
        for (int s = 0; s < n_shards; ++s)
        {
            buffers[s].resize((size_t)rows * counts[s]);
            dests.push_back(buffers[s].data());
        }
        ASSERT_TRUE(loader.loadTensorColumnShards(chosen->name, offsets, counts, dests)) << "Column shard load failed iter=" << iter;
        // Reconstruct
        std::vector<float> recon((size_t)rows * cols, -777.f);
        for (int r = 0; r < rows; ++r)
        {
            for (int s = 0; s < n_shards; ++s)
            {
                std::memcpy(recon.data() + (size_t)r * cols + offsets[s], buffers[s].data() + (size_t)r * counts[s], sizeof(float) * counts[s]);
            }
        }
        double max_abs = 0.0, diff_sq = 0.0, ref_sq = 0.0;
        for (size_t i = 0; i < recon.size(); ++i)
        {
            double d = (double)recon[i] - full[i];
            if (std::fabs(d) > max_abs)
                max_abs = std::fabs(d);
            diff_sq += d * d;
            ref_sq += (double)full[i] * full[i];
        }
        double rel_l2 = (ref_sq > 0.0) ? std::sqrt(diff_sq / ref_sq) : 0.0;
        EXPECT_LT(max_abs, 1e-9) << "Iter=" << iter << " column shard max_abs mismatch";
        EXPECT_LT(rel_l2, 1e-10) << "Iter=" << iter << " column shard rel_l2 mismatch";

        // Random row slice parity
        int row_off = row_off_dist(rng);
        int row_len = std::min(row_len_dist(rng), rows - row_off);
        if (row_len <= 0)
            row_len = 1;
        std::vector<float> row_buf((size_t)row_len * cols, 0.f);
        ASSERT_TRUE(loader.loadTensorRowShard(chosen->name, row_off, row_len, row_buf.data())) << "Row shard load failed iter=" << iter;
        double max_abs_r = 0.0, diff_sq_r = 0.0, ref_sq_r = 0.0;
        for (int r = 0; r < row_len; ++r)
        {
            const float *ref_row = full + (size_t)(row_off + r) * cols;
            const float *got_row = row_buf.data() + (size_t)r * cols;
            for (int c = 0; c < cols; ++c)
            {
                double d = (double)got_row[c] - ref_row[c];
                if (std::fabs(d) > max_abs_r)
                    max_abs_r = std::fabs(d);
                diff_sq_r += d * d;
                ref_sq_r += (double)ref_row[c] * ref_row[c];
            }
        }
        double rel_l2_r = (ref_sq_r > 0.0) ? std::sqrt(diff_sq_r / ref_sq_r) : 0.0;
        EXPECT_LT(max_abs_r, 1e-9) << "Iter=" << iter << " row shard max_abs mismatch";
        EXPECT_LT(rel_l2_r, 1e-10) << "Iter=" << iter << " row shard rel_l2 mismatch";
        if (::testing::Test::HasFailure())
            break; // early exit to reduce noise if failing
    }
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized)
    {
        int was_init = mpi_init;
        MPI_Barrier(MPI_COMM_WORLD);
        if (!was_init)
            MPI_Finalize();
    }
}
