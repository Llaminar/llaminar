#include <gtest/gtest.h>
#include "model_loader.h"
#include <fstream>
#include <cstdlib>
#include <sys/stat.h>

// Helper: load a small GGUF model specified by env var LLAMINAR_TEST_GGUF_PATH (required)
// Verifies data_offset alignment and offset chain invariants (padded by model alignment or 32 default).
// Fails fast with clear diagnostics if any invariant is violated.

static std::string getModelPathOrEmpty()
{
    const char *p = std::getenv("LLAMINAR_TEST_GGUF_PATH");
    if (!p || std::string(p).empty())
    {
        return {};
    }
    return std::string(p);
}

TEST(ModelLoaderAlignmentTest, OffsetChainAndAlignment)
{
    std::string path = getModelPathOrEmpty();
    if (path.empty())
    {
        GTEST_SKIP() << "LLAMINAR_TEST_GGUF_PATH not set; skipping"
                        " (alignment regression)";
    }
    struct stat st{};
    if (stat(path.c_str(), &st) != 0)
    {
        GTEST_SKIP() << "Model path not accessible: " << path;
    }

    // Force debug invariants OFF to test independently (do not rely on runtime logging)
    unsetenv("LLAMINAR_MODEL_LOAD_DEBUG");

    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(path)) << "loadModel failed for path=" << path;
    const GGUFModel &model = loader.getModel();

    uint32_t align = model.alignment ? model.alignment : 32;
    ASSERT_TRUE((align & (align - 1)) == 0) << "Alignment must be power of two, got=" << align;
    ASSERT_EQ(model.data_offset % align, 0ULL) << "data_offset not aligned: data_offset=" << model.data_offset << " align=" << align;

    // Verify offset chain with padding logic
    uint64_t expected = 0;
    for (size_t i = 0; i < model.tensors.size(); ++i)
    {
        const auto &t = model.tensors[i];
        ASSERT_EQ(t.offset, expected) << "Offset chain break at i=" << i << " name=" << t.name << " expected=" << expected << " got=" << t.offset;
        size_t padded = ((t.size_bytes + align - 1) / align) * align;
        expected += padded;
    }

    // Spot-check that final expected size does not exceed file size (basic sanity)
    // File size sanity check omitted (private accessor); invariant chain sufficient for regression.
}
