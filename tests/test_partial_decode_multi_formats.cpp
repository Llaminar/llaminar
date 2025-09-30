// Parity tests for partial column decode across selected supported quant formats.
// Removed legacy formats: Q4_1, Q5_1, Q8_1.
// Ensures partial decode output matches full-cache slice when enabled via LLAMINAR_PARTIAL_DECODE.
// Skips gracefully if no aligned tensor for a given format is present in current test model.

#include <gtest/gtest.h>
#include <filesystem>
#include <cstdlib>
#include <vector>
#include <string>
#include <limits>
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

    struct FormatCase
    {
        GGUFTensorType type;
        const char *name;
    };

    static void run_parity_for(GGUFTensorType ttype)
    {
        std::string model_path = pick_model();
        if (model_path.empty())
            GTEST_SKIP() << "No model available";
        setenv("LLAMINAR_PARTIAL_DECODE", "1", 1);
        ModelLoader loader;
        ASSERT_TRUE(loader.loadModel(model_path));
        const GGUFTensorInfo *target = nullptr;
        for (auto &ti : loader.getModel().tensors)
        {
            if (ti.type == ttype && ti.dimensions.size() == 2 && (ti.dimensions[1] % 32) == 0)
            {
                target = &ti;
                break;
            }
        }
        if (!target)
            GTEST_SKIP() << "No aligned tensor found for format enum=" << (int)ttype;
        int rows = (int)target->dimensions[0];
        int cols = (int)target->dimensions[1];
        if (cols < 96)
            GTEST_SKIP() << "Need >=96 cols to form three shards";
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
        ASSERT_TRUE(loader.loadTensorColumnShards(target->name, offs, counts, dest));
        unsetenv("LLAMINAR_PARTIAL_DECODE");
        loader.clearQuantShardCache();
        std::vector<std::vector<float>> buffers_ref(3, std::vector<float>((size_t)rows * 8, 0.f));
        std::vector<float *> dest_ref;
        for (int i = 0; i < 3; ++i)
            dest_ref.push_back(buffers_ref[i].data());
        ASSERT_TRUE(loader.loadTensorColumnShards(target->name, offs, counts, dest_ref));

        for (size_t s = 0; s < offs.size(); ++s)
        {
            int cnt = counts[s];
            for (int r = 0; r < rows; ++r)
            {
                const float *a = buffers[s].data() + (size_t)r * cnt;
                const float *b = buffers_ref[s].data() + (size_t)r * cnt;
                for (int c = 0; c < cnt; ++c)
                {
                    ASSERT_FLOAT_EQ(a[c], b[c]) << "Mismatch fmt=" << (int)ttype << " shard=" << s << " row=" << r << " col=" << c;
                }
            }
        }
    }
} // namespace

TEST(PartialDecodeMulti, Q5_0_Parity) { run_parity_for(GGUFTensorType::Q5_0); }
// Add more supported formats here as needed (e.g., Q4_0, Q8_0, K formats) if they have aligned tensors in test models.
