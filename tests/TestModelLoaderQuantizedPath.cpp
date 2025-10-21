/**
 * @file TestModelLoaderQuantizedPath.cpp
 * @brief Verifies Phase 2 quantized loader early-return creates QuantizedTensor.
 *
 * Test Strategy:
 *  1. Pick a small quantized GGUF test model (env LLAMINAR_GOLDEN_MODEL or smallest in models/).
 *  2. Enable quant flags: LLAMINAR_QUANT_ENABLE=1 LLAMINAR_LOAD_QUANTIZED=1.
 *  3. Force FP32 override off (ensure LLAMINAR_FORCE_FP32_WEIGHTS unset).
 *  4. Load a known quantized tensor (e.g. first projection weight) and assert factory returns QuantizedTensor.
 *  5. Load an FP16 tensor (e.g. rmsnorm gamma) and assert still SimpleTensor.
 *  6. Disable quant flags and ensure same quantized tensor now loads as SimpleTensor (dequantized path).
 *
 * NOTE: This test does not validate decode correctness – decodeBlock is still a placeholder.
 *       It strictly validates gating and type selection logic in ModelLoader::loadTensor.
 */
#include "ModelLoader.h"
#include "tensors/TensorFactory.h"
#include "Logger.h"
#include <gtest/gtest.h>
#include <filesystem>

using namespace llaminar;

// Helper: find a model path (prefers env override)
static std::optional<std::string> pick_model()
{
    if (const char *m = std::getenv("LLAMINAR_GOLDEN_MODEL"))
    {
        if (*m)
            return std::string(m);
    }
    std::string chosen;
    for (auto &p : std::filesystem::directory_iterator("models"))
    {
        if (!p.is_regular_file() || p.path().extension() != ".gguf")
            continue;
        auto sz = std::filesystem::file_size(p.path());
        if (chosen.empty() || sz < std::filesystem::file_size(chosen))
            chosen = p.path().string();
    }
    if (chosen.empty())
        return std::nullopt;
    return chosen;
}

// Helper: identify first quantized and first float tensor names
static void find_candidate_tensors(const GGUFModel &model, std::string &quant_name, std::string &float_name)
{
    for (const auto &ti : model.tensors)
    {
        if (quant_name.empty() && ti.type != GGUFTensorType::F32 && ti.type != GGUFTensorType::F16)
        {
            quant_name = ti.name;
        }
        if (float_name.empty() && (ti.type == GGUFTensorType::F32 || ti.type == GGUFTensorType::F16))
        {
            float_name = ti.name;
        }
        if (!quant_name.empty() && !float_name.empty())
            break;
    }
}

TEST(ModelLoaderQuantizedPath, GatedQuantizedTensorCreation)
{
    auto maybe_model = pick_model();
    ASSERT_TRUE(maybe_model.has_value()) << "No .gguf model found under models/.";
    const std::string model_path = *maybe_model;

    // Ensure quant flags active for first phase
    setenv("LLAMINAR_QUANT_ENABLE", "1", 1);
    setenv("LLAMINAR_LOAD_QUANTIZED", "1", 1);
    unsetenv("LLAMINAR_FORCE_FP32_WEIGHTS");
    debugEnvRefresh(); // refresh snapshot

    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path)) << "Failed to load model: " << model_path;
    const auto &gguf = loader.getModel();

    std::string quant_tensor_name, float_tensor_name;
    find_candidate_tensors(gguf, quant_tensor_name, float_tensor_name);
    ASSERT_FALSE(quant_tensor_name.empty()) << "No quantized tensor found in model.";
    ASSERT_FALSE(float_tensor_name.empty()) << "No float tensor found in model.";

    auto q_tensor = loader.loadTensor(quant_tensor_name);
    ASSERT_TRUE(q_tensor) << "Quantized tensor load returned nullptr.";
    EXPECT_TRUE(TensorFactory::is_quantized(q_tensor)) << "Expected QuantizedTensor when quant flags enabled for '" << quant_tensor_name << "'.";

    auto f_tensor = loader.loadTensor(float_tensor_name);
    ASSERT_TRUE(f_tensor) << "Float tensor load returned nullptr.";
    EXPECT_FALSE(TensorFactory::is_quantized(f_tensor)) << "Float tensor should not be quantized: '" << float_tensor_name << "'.";

    // Disable quant path and verify fallback to dequantized SimpleTensor
    unsetenv("LLAMINAR_QUANT_ENABLE");
    unsetenv("LLAMINAR_LOAD_QUANTIZED");
    debugEnvRefresh();

    ModelLoader loader2; // new loader to avoid cached snapshots or data reuse
    ASSERT_TRUE(loader2.loadModel(model_path));
    auto q_tensor_fp32 = loader2.loadTensor(quant_tensor_name);
    ASSERT_TRUE(q_tensor_fp32);
    EXPECT_FALSE(TensorFactory::is_quantized(q_tensor_fp32)) << "Expected dequantized FP32 tensor after disabling quant flags.";
}
