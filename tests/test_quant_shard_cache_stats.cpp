// Verifies quant shard cache statistics (hits/misses/evictions) behave as expected.
// Strategy:
// 1. Load a model and pick a small quant 2D tensor.
// 2. Clear cache, request a column shard twice -> expect 1 miss then 1 hit.
// 3. Force eviction by setting a tiny cache max (env) and loading additional tensors until original evicted.
// 4. Re-request original shard -> miss count increments.

#include <gtest/gtest.h>
#include <filesystem>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include "model_loader.h"
#include "logger.h"

using namespace llaminar;

// Helper: select a small quant 2D tensor and optionally additional ones.
static std::vector<const GGUFTensorInfo *> collect_quant_2d(const GGUFModel &model)
{
    std::vector<const GGUFTensorInfo *> out;
    for (auto &ti : model.tensors)
    {
        if (ti.isQuantized() && ti.dimensions.size() == 2)
            out.push_back(&ti);
    }
    // stable order is fine
    return out;
}

static std::string pick_model_path()
{
    if (const char *env = std::getenv("LLAMINAR_SHARD_TEST_MODEL"))
        return env;
    namespace fs = std::filesystem;
    fs::path dir{"models"};
    if (!fs::exists(dir))
        return {};
    uintmax_t best = (std::numeric_limits<uintmax_t>::max)();
    std::string path;
    for (auto &p : fs::directory_iterator(dir))
    {
        if (!p.is_regular_file() || p.path().extension() != ".gguf")
            continue;
        auto sz = fs::file_size(p.path());
        if (sz < best)
        {
            best = sz;
            path = p.path().string();
        }
    }
    return path;
}

TEST(QuantShardCacheStats, BasicHitsAndEvictions)
{
    std::string model_path = pick_model_path();
    if (model_path.empty())
        GTEST_SKIP() << "No model found for cache stats test";
    // Constrain cache to a very small size so multiple tensors trigger eviction quickly.
    setenv("LLAMINAR_SHARD_CACHE_MAX_MB", "4", 1); // 4MB budget

    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path));
    auto tensors = collect_quant_2d(loader.getModel());
    if (tensors.size() < 2)
        GTEST_SKIP() << "Need >=2 quant 2D tensors for eviction scenario";
    const GGUFTensorInfo *primary = tensors[0];

    loader.clearQuantShardCache();
    auto stats0 = loader.getQuantShardCacheStats();
    EXPECT_EQ(stats0.loads, 0u);
    EXPECT_EQ(stats0.cache_hits, 0u);
    EXPECT_EQ(stats0.cache_misses, 0u);

    // Request a small column shard twice.
    int rows = (int)primary->dimensions[0];
    int cols = (int)primary->dimensions[1];
    int take = std::min(8, cols); // small slice
    std::vector<float> shard_buf((size_t)rows * take);
    std::vector<int> offs = {0};
    std::vector<int> counts = {take};
    std::vector<float *> dests = {shard_buf.data()};
    ASSERT_TRUE(loader.loadTensorColumnShards(primary->name, offs, counts, dests));
    auto stats1 = loader.getQuantShardCacheStats();
    EXPECT_EQ(stats1.loads, 1u);
    EXPECT_EQ(stats1.cache_misses, 1u);
    EXPECT_EQ(stats1.cache_hits, 0u);
    // Second call (should be hit)
    ASSERT_TRUE(loader.loadTensorColumnShards(primary->name, offs, counts, dests));
    auto stats2 = loader.getQuantShardCacheStats();
    EXPECT_EQ(stats2.loads, 2u);
    EXPECT_EQ(stats2.cache_misses, 1u);
    EXPECT_EQ(stats2.cache_hits, 1u);

    // Now load additional distinct quant tensors to force eviction of primary.
    size_t loads_before = stats2.loads;
    for (size_t i = 1; i < tensors.size(); ++i)
    {
        const auto *ti = tensors[i];
        // row shard to vary access pattern
        int slice_rows = std::min<int>((int)ti->dimensions[0], 4);
        std::vector<float> tmp((size_t)slice_rows * (int)ti->dimensions[1]);
        if (!loader.loadTensorRowShard(ti->name, 0, slice_rows, tmp.data()))
            break;
        auto st = loader.getQuantShardCacheStats();
        if (st.evictions > 0)
            break; // stop once we see an eviction
    }
    auto stats_after_eviction = loader.getQuantShardCacheStats();
    if (stats_after_eviction.evictions == 0)
    {
        GTEST_SKIP() << "Eviction did not occur (model tensors too small to exceed 4MB); adjust cache budget or use larger model";
    }
    // Access primary again -> should be a miss since evicted
    ASSERT_TRUE(loader.loadTensorColumnShards(primary->name, offs, counts, dests));
    auto stats_final = loader.getQuantShardCacheStats();
    EXPECT_EQ(stats_final.cache_misses, stats_after_eviction.cache_misses + 1) << "Expected miss after eviction";
}
