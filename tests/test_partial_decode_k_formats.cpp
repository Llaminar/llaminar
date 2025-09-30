// Parity test for partial decode of K-format blocks (currently Q4_K / Q4_K_M) vs full-cache path.
// Activates partial decode via LLAMINAR_PARTIAL_DECODE and compares with reference after clearing cache.
// Skips gracefully if no suitable aligned tensor (2D, width % 256 == 0) of required type is present.

#include <gtest/gtest.h>
#include <filesystem>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include "model_loader.h"

using namespace llaminar;

namespace
{
    static std::string pick_model()
    {
        if (const char *env = std::getenv("LLAMINAR_SHARD_TEST_MODEL"))
            return env;
        namespace fs = std::filesystem;
        fs::path dir{"models"};
        if (!fs::exists(dir))
            return {};
        for (auto &p : fs::directory_iterator(dir))
        {
            if (!p.is_regular_file() || p.path().extension() != ".gguf")
                continue;
            return p.path().string();
        }
        return {};
    }

    static const std::vector<GGUFTensorType> kKTypes = {GGUFTensorType::Q4_K, GGUFTensorType::Q4_K_M};

    TEST(PartialDecodeKFormats, ParityVsFull)
    {
        std::string model_path = pick_model();
        if (model_path.empty())
            GTEST_SKIP() << "No model available";
        ModelLoader loader;
        ASSERT_TRUE(loader.loadModel(model_path));

        std::vector<const GGUFTensorInfo *> targets;
        for (auto &ti : loader.getModel().tensors)
        {
            if (ti.dimensions.size() != 2)
                continue;
            if (ti.dimensions[1] % 256 != 0)
                continue;
            if (std::find(kKTypes.begin(), kKTypes.end(), ti.type) == kKTypes.end())
                continue;
            targets.push_back(&ti);
        }
        if (targets.empty())
            GTEST_SKIP() << "No aligned Q4_K / Q4_K_M tensors present";

        for (const GGUFTensorInfo *target : targets)
        {
            int rows = (int)target->dimensions[0];
            int cols = (int)target->dimensions[1];
            if (cols < 256)
                continue; // skip tiny
            // Choose three sparse shards, each small (16 columns or less) to stay well below 25% coverage.
            int shard_w = std::min(16, std::max(8, cols / 128));
            int off1 = 0;
            int off2 = cols / 3;
            int off3 = (2 * cols) / 3 + 32;
            if (off3 + shard_w > cols)
                off3 = cols - shard_w - 1;
            if (off3 < 0)
                off3 = cols - shard_w;
            std::vector<int> offs = {off1, off2, off3};
            std::vector<int> counts = {shard_w, shard_w, shard_w};
            // Enable partial decode
            setenv("LLAMINAR_PARTIAL_DECODE", "1", 1);
            std::vector<std::vector<float>> shards(3, std::vector<float>((size_t)rows * shard_w, 0.f));
            std::vector<float *> dest;
            for (int i = 0; i < 3; ++i)
                dest.push_back(shards[i].data());
            ASSERT_TRUE(loader.loadTensorColumnShards(target->name, offs, counts, dest)) << "Partial path failed for type";
            // Reference path
            unsetenv("LLAMINAR_PARTIAL_DECODE");
            loader.clearQuantShardCache();
            std::vector<std::vector<float>> shards_ref(3, std::vector<float>((size_t)rows * shard_w, 0.f));
            std::vector<float *> dest_ref;
            for (int i = 0; i < 3; ++i)
                dest_ref.push_back(shards_ref[i].data());
            ASSERT_TRUE(loader.loadTensorColumnShards(target->name, offs, counts, dest_ref));
            for (size_t s = 0; s < offs.size(); ++s)
            {
                for (int r = 0; r < rows; ++r)
                {
                    const float *a = shards[s].data() + (size_t)r * shard_w;
                    const float *b = shards_ref[s].data() + (size_t)r * shard_w;
                    for (int c = 0; c < shard_w; ++c)
                    {
                        ASSERT_FLOAT_EQ(a[c], b[c]) << "Mismatch k-format tensor='" << target->name << "' shard=" << s << " row=" << r << " col=" << c;
                    }
                }
            }
            loader.clearQuantShardCache();
        }
    }

} // namespace
