// Validates partial decode prototype for Q4_0 column shards matches full-cache path.
// Requirements: a model containing at least one Q4_0 2D tensor with width multiple of 32.

#include <gtest/gtest.h>
#include <filesystem>
#include <cstdlib>
#include <vector>
#include <string>
#include <limits>
#include "model_loader.h"

using namespace llaminar;

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

TEST(PartialDecodeQ4_0, ParityVsFullCache)
{
    std::string model_path = pick_model();
    if (model_path.empty())
        GTEST_SKIP() << "No model available";
    setenv("LLAMINAR_PARTIAL_DECODE_Q4_0", "1", 1);
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path));
    const GGUFTensorInfo *target = nullptr;
    for (auto &ti : loader.getModel().tensors)
    {
        if (ti.type == GGUFTensorType::Q4_0 && ti.dimensions.size() == 2 && (ti.dimensions[1] % 32) == 0)
        {
            target = &ti;
            break;
        }
    }
    if (!target)
        GTEST_SKIP() << "No Q4_0 aligned tensor found";
    int rows = (int)target->dimensions[0];
    int cols = (int)target->dimensions[1];
    if (cols < 96)
        GTEST_SKIP() << "Need >=96 cols to form three shards";
    // Define three small disjoint shards covering sparse columns
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
    std::vector<std::vector<float>> buffers(3, std::vector<float>((size_t)rows * 8, 0.f)); // oversize ok
    std::vector<float *> dest;
    for (int i = 0; i < 3; ++i)
        dest.push_back(buffers[i].data());

    // First call uses partial decode (coverage likely < threshold)
    ASSERT_TRUE(loader.loadTensorColumnShards(target->name, offs, counts, dest));
    // Clear env to force fallback to cache path (which will populate & slice full tensor)
    unsetenv("LLAMINAR_PARTIAL_DECODE_Q4_0");
    // Clear cache to ensure full-load path executes
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
                ASSERT_FLOAT_EQ(a[c], b[c]) << "Mismatch shard=" << s << " row=" << r << " col=" << c;
            }
        }
    }
}
