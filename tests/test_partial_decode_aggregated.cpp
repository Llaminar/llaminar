// Aggregated partial decode parity test iterating over supported 32-value block quant formats.
// Formats covered: Q4_0, Q5_0, Q8_0 (legacy Q4_1/Q5_1/Q8_1 removed).
// Strategy: For each format present in the loaded model (2D tensor, width % 32 == 0),
// perform a sparse multi-shard column extraction with partial decode enabled, then
// compare against the cache (full tensor decode) reference after clearing env + cache.

#include <gtest/gtest.h>
#include <filesystem>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_set>
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

    static const std::vector<GGUFTensorType> kFormats = {
        GGUFTensorType::Q4_0,
        GGUFTensorType::Q5_0,
        GGUFTensorType::Q8_0};

    struct FmtInfo
    {
        GGUFTensorType t;
        const char *name;
    };

    static const char *fmt_name(GGUFTensorType t)
    {
        switch (t)
        {
        case GGUFTensorType::Q4_0:
            return "Q4_0";
        case GGUFTensorType::Q5_0:
            return "Q5_0";
        case GGUFTensorType::Q8_0:
            return "Q8_0";
        default:
            return "?";
        }
    }

    TEST(PartialDecodeAggregated, ParityAllPresentFormats)
    {
        std::string model_path = pick_model();
        if (model_path.empty())
            GTEST_SKIP() << "No model available";
        ModelLoader loader;
        ASSERT_TRUE(loader.loadModel(model_path));
        // Collect a representative tensor per supported format.
        struct Candidate
        {
            const GGUFTensorInfo *info;
        };
        std::vector<const GGUFTensorInfo *> targets;
        std::unordered_set<int> seen;
        for (auto &ti : loader.getModel().tensors)
        {
            if (ti.dimensions.size() != 2)
                continue;
            if (ti.dimensions[1] % 32 != 0)
                continue;
            if (std::find(kFormats.begin(), kFormats.end(), ti.type) == kFormats.end())
                continue;
            if (seen.insert((int)ti.type).second)
            {
                targets.push_back(&ti);
            }
        }
        if (targets.empty())
            GTEST_SKIP() << "No supported quant tensors present";

        for (const GGUFTensorInfo *target : targets)
        {
            // Enable partial decode
            setenv("LLAMINAR_PARTIAL_DECODE", "1", 1);
            int rows = (int)target->dimensions[0];
            int cols = (int)target->dimensions[1];
            if (cols < 96)
            {
                unsetenv("LLAMINAR_PARTIAL_DECODE");
                continue;
            }
            int s1 = std::min(8, cols / 8);
            int s2 = std::min(8, cols / 8);
            int s3 = std::min(8, cols / 8);
            int off1 = 0;
            int off2 = cols / 3;
            int off3 = (2 * cols) / 3 + 4;
            if (off3 + s3 > cols)
                off3 = cols - s3 - 1;
            std::vector<int> offs = {off1, off2, off3};
            std::vector<int> counts = {s1, s2, s3};
            std::vector<std::vector<float>> buffers(3, std::vector<float>((size_t)rows * 8, 0.f));
            std::vector<float *> dest;
            for (int i = 0; i < 3; ++i)
                dest.push_back(buffers[i].data());
            ASSERT_TRUE(loader.loadTensorColumnShards(target->name, offs, counts, dest)) << "Partial path failed for format=" << fmt_name(target->type);
            unsetenv("LLAMINAR_PARTIAL_DECODE");
            loader.clearQuantShardCache();
            std::vector<std::vector<float>> buffers_ref(3, std::vector<float>((size_t)rows * 8, 0.f));
            std::vector<float *> dest_ref;
            for (int i = 0; i < 3; ++i)
                dest_ref.push_back(buffers_ref[i].data());
            ASSERT_TRUE(loader.loadTensorColumnShards(target->name, offs, counts, dest_ref)) << "Cache path failed for format=" << fmt_name(target->type);
            for (size_t s = 0; s < offs.size(); ++s)
            {
                int cnt = counts[s];
                for (int r = 0; r < rows; ++r)
                {
                    const float *a = buffers[s].data() + (size_t)r * cnt;
                    const float *b = buffers_ref[s].data() + (size_t)r * cnt;
                    for (int c = 0; c < cnt; ++c)
                    {
                        ASSERT_FLOAT_EQ(a[c], b[c]) << "Mismatch format=" << fmt_name(target->type) << " shard=" << s << " row=" << r << " col=" << c;
                    }
                }
            }
            // Reset cache between formats to avoid cross-contamination of timing/coverage (already cleared above for reference load).
            loader.clearQuantShardCache();
        }
    }

} // namespace
