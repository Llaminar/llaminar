#include "model_loader.h"
#include "logger.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <random>
#include <map>
#include <iostream>

using namespace llaminar;

// Helper function to enumerate quantization types in a model
static void enumerate_quantization_types(const GGUFModel &model)
{
    std::map<int, int> type_counts;

    for (const auto &tensor : model.tensors)
    {
        int type_enum = static_cast<int>(tensor.type);
        type_counts[type_enum]++;
    }

    std::cout << "\nQuantization type distribution in model:\n";
    for (const auto &[type_enum, count] : type_counts)
    {
        const char *type_name = "Unknown";
        switch (static_cast<GGUFTensorType>(type_enum))
        {
        case GGUFTensorType::F32:
            type_name = "F32";
            break;
        case GGUFTensorType::F16:
            type_name = "F16";
            break;
        case GGUFTensorType::Q4_0:
            type_name = "Q4_0";
            break;
        case GGUFTensorType::Q4_K:
            type_name = "Q4_K";
            break;
        case GGUFTensorType::Q5_0:
            type_name = "Q5_0";
            break;
        case GGUFTensorType::Q5_K:
            type_name = "Q5_K";
            break;
        case GGUFTensorType::Q8_0:
            type_name = "Q8_0";
            break;
        case GGUFTensorType::Q2_K:
            type_name = "Q2_K";
            break;
        case GGUFTensorType::Q3_K:
            type_name = "Q3_K";
            break;
        case GGUFTensorType::Q6_K:
            type_name = "Q6_K";
            break;
        default:
            break;
        }
        std::cout << "  " << type_name << " (enum " << type_enum << "): " << count << " tensors\n";
    }
}
// Helper: locate a GGUF model that contains Q4_K tensors.
// We'll scan the provided models directory for a file name containing "q4_k" (common naming) or fall back to q4_0 to skip.
static std::string find_q4k_model()
{
    std::filesystem::path models_dir{"models"};
    if (!std::filesystem::exists(models_dir))
        return {};
    std::string fallback;
    for (auto &p : std::filesystem::directory_iterator(models_dir))
    {
        if (!p.is_regular_file())
            continue;
        auto name = p.path().filename().string();
        if (name.find(".gguf") == std::string::npos)
            continue;
        if (fallback.empty())
            fallback = p.path().string();
        // Prefer explicit q4_k markers (common conventions: q4_k, q4_k_m, q4_k_s, etc.)
        std::string lower = name;
        for (auto &c : lower)
            c = (char)std::tolower(c);
        if (lower.find("q4_k") != std::string::npos)
            return p.path().string();
    }
    return fallback;
}

TEST(IntegrationQ4K, DecodeRealTensorsBasicStats)
{
    std::string model_path = find_q4k_model();
    if (model_path.empty())
    {
        GTEST_SKIP() << "No GGUF model file found in models/ directory";
    }

    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path)) << "Failed to load GGUF model: " << model_path;

    const auto &model = loader.getModel();

    // Enumerate all quantization types in the model
    enumerate_quantization_types(model);

    size_t q4k_tensor_count = 0;
    size_t checked = 0;
    double global_mean = 0.0;
    double global_var_acc = 0.0;
    size_t global_n = 0;

    for (const auto &t : model.tensors)
    {
        if (t.type != GGUFTensorType::Q4_K)
            continue;
        ++q4k_tensor_count;
        // Load and dequantize
        auto tensor = loader.loadTensor(t.name);
        ASSERT_TRUE(tensor) << "Failed to load tensor: " << t.name;
        auto shape = tensor->shape();
        size_t n = 1;
        for (auto d : shape)
            n *= d;
        ASSERT_GT(n, 0u);
        // Access raw data
        const float *data_f = reinterpret_cast<const float *>(tensor->data());
        ASSERT_NE(data_f, nullptr);
        double sum = 0.0;
        double sumsq = 0.0;
        float vmin = std::numeric_limits<float>::max();
        float vmax = std::numeric_limits<float>::lowest();
        size_t sample_n = std::min<size_t>(n, 4096);
        for (size_t i = 0; i < sample_n; ++i)
        {
            float v = data_f[i];
            ASSERT_FALSE(std::isnan(v));
            sum += v;
            sumsq += double(v) * double(v);
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
        double mean = sum / sample_n;
        double var = (sumsq / sample_n) - mean * mean;
        // Basic sanity: variance > 0 (not all constants) and range not degenerate
        EXPECT_GT(var, 0.0) << "Tensor appears constant: " << t.name;
        EXPECT_LT(vmin, vmax) << "Degenerate range in tensor: " << t.name;
        // Accumulate global stats
        global_mean += sum;
        global_var_acc += sumsq;
        global_n += sample_n;
        if (++checked >= 3)
            break; // Limit runtime; sample first few Q4_K tensors
    }

    if (q4k_tensor_count == 0)
    {
        GTEST_SKIP() << "Loaded model has no Q4_K tensors (perhaps different quantization)";
    }

    ASSERT_GT(checked, 0u) << "No Q4_K tensors were sampled for stats";
    double gmean = global_mean / global_n;
    double gvar = (global_var_acc / global_n) - gmean * gmean;
    EXPECT_GT(gvar, 0.0) << "Global variance zero across sampled tensors";
}

TEST(IntegrationQ4K, TestAllQuantizationTypes)
{
    std::string model_path = find_q4k_model();
    if (model_path.empty())
    {
        GTEST_SKIP() << "No GGUF model found for comprehensive quantization test";
        return;
    }

    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path)) << "Failed to load model: " << model_path;

    std::map<GGUFTensorType, int> successful_loads;
    std::map<GGUFTensorType, int> failed_loads;
    std::map<GGUFTensorType, std::string> first_tensor_names;

    const auto &model = loader.getModel();

    std::cout << "\nTesting all quantization types in model: " << model_path << "\n";
    enumerate_quantization_types(model);

    for (const auto &tensor : model.tensors)
    {
        // Test if we support this quantization type
        if (!loader.supportsQuantization(tensor.type))
        {
            continue;
        }

        try
        {
            auto tensor_ptr = loader.loadTensor(tensor.name);
            if (tensor_ptr)
            {
                // Derive element count from tensor shape since TensorBase does not expose numElements()
                size_t tensor_size = 1;
                const auto &shape = tensor_ptr->shape();
                for (auto d : shape)
                    tensor_size *= static_cast<size_t>(d);
                if (tensor_size == 0)
                {
                    failed_loads[tensor.type]++;
                    std::cout << "  Empty result for tensor " << tensor.name
                              << " (type " << static_cast<int>(tensor.type) << ")\n";
                    continue;
                }
                const float *data = tensor_ptr->data();
                bool has_nan = false;
                bool has_inf = false;
                double sum = 0.0;
                size_t sample_size = std::min<size_t>(tensor_size, static_cast<size_t>(1000));

                for (size_t i = 0; i < sample_size; ++i)
                {
                    float val = data[i];
                    if (std::isnan(val))
                    {
                        has_nan = true;
                        break;
                    }
                    if (std::isinf(val))
                    {
                        has_inf = true;
                        break;
                    }
                    sum += val;
                }

                if (!has_nan && !has_inf)
                {
                    successful_loads[tensor.type]++;
                    if (first_tensor_names.find(tensor.type) == first_tensor_names.end())
                    {
                        first_tensor_names[tensor.type] = tensor.name;
                    }
                }
                else
                {
                    failed_loads[tensor.type]++;
                    std::cout << "  Invalid values in tensor " << tensor.name
                              << " (type " << static_cast<int>(tensor.type) << ")\n";
                }
            }
            else
            {
                failed_loads[tensor.type]++;
                std::cout << "  Null tensor pointer for " << tensor.name
                          << " (type " << static_cast<int>(tensor.type) << ")\n";
            }
        }
        catch (const std::exception &e)
        {
            failed_loads[tensor.type]++;
            std::cout << "  Exception loading tensor " << tensor.name
                      << " (type " << static_cast<int>(tensor.type) << "): " << e.what() << "\n";
        }
    }

    std::cout << "\nQuantization type test results:\n";
    for (const auto &[type, count] : successful_loads)
    {
        const char *type_name = "Unknown";
        switch (type)
        {
        case GGUFTensorType::F32:
            type_name = "F32";
            break;
        case GGUFTensorType::F16:
            type_name = "F16";
            break;
        case GGUFTensorType::Q4_0:
            type_name = "Q4_0";
            break;
        case GGUFTensorType::Q4_K:
            type_name = "Q4_K";
            break;
        case GGUFTensorType::Q5_0:
            type_name = "Q5_0";
            break;
        case GGUFTensorType::Q5_K:
            type_name = "Q5_K";
            break;
        case GGUFTensorType::Q8_0:
            type_name = "Q8_0";
            break;
        case GGUFTensorType::Q2_K:
            type_name = "Q2_K";
            break;
        case GGUFTensorType::Q3_K:
            type_name = "Q3_K";
            break;
        case GGUFTensorType::Q6_K:
            type_name = "Q6_K";
            break;
        default:
            break;
        }
        std::cout << "  " << type_name << ": " << count << " successful";
        if (first_tensor_names.find(type) != first_tensor_names.end())
        {
            std::cout << " (e.g., " << first_tensor_names[type] << ")";
        }
        std::cout << "\n";
    }

    for (const auto &[type, count] : failed_loads)
    {
        if (count > 0)
        {
            std::cout << "  Type " << static_cast<int>(type) << ": " << count << " failed\n";
        }
    }

    // Ensure we tested at least some tensors successfully
    int total_successful = 0;
    for (const auto &[type, count] : successful_loads)
    {
        total_successful += count;
    }

    EXPECT_GT(total_successful, 0) << "No tensors were successfully loaded and validated";
}
