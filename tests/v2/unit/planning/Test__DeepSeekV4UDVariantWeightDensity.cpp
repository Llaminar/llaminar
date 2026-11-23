/**
 * @file Test__DeepSeekV4UDVariantWeightDensity.cpp
 * @brief Every Unsloth UD variant that fits one GB10 is admitted at file density on CUDA.
 *
 * The fixture is the exact tensor inventory (name, GGUF type, shape, layers) of
 * the published DeepSeek-V4-Flash-Vision UD-IQ1_S, UD-IQ1_M, UD-IQ2_XXS,
 * UD-Q2_K_XL, UD-IQ3_XXS and UD-IQ3_S GGUFs (the variants whose weights fit a
 * 128 GB GB10), extracted from their shard headers by tensor directory only.
 * Each tensor is replayed through the production WeightMemoryEstimator on a
 * CUDA participant. The admitted device bytes may exceed the GGUF bytes only
 * by the pool's 256-byte region alignment of a matrix's bit-packed code plane;
 * any other growth (per-block metadata planes, FP32 widening of BF16 router
 * weights, padded expert slots) fails. Host-resident integer routing tables
 * may be admitted below their file bytes. Device-free and model-free.
 */

#include <gtest/gtest.h>

#include "planning/ModelMemoryProfile.h"
#include "planning/WeightMemoryEstimator.h"
#include "backends/DeviceId.h"
#include "DeepSeekV4UDInventory.generated.h"

#include <nlohmann/json.hpp>

#include <map>
#include <string>
#include <vector>

namespace llaminar2
{
namespace
{
/** @return GGUF bytes of @p elements values of @p type (block element count, block bytes). */
std::size_t ggufBytes(const std::string &type, std::size_t elements)
{
    static const std::map<std::string, std::pair<std::size_t, std::size_t>> blocks{
        {"F32", {1, 4}},       {"BF16", {1, 2}},      {"F16", {1, 2}},       {"I32", {1, 4}},
        {"Q8_0", {32, 34}},    {"MXFP4", {32, 17}},   {"Q4_K", {256, 144}},  {"Q5_K", {256, 176}},
        {"Q6_K", {256, 210}},  {"Q2_K", {256, 84}},   {"Q3_K", {256, 110}},  {"IQ2_XXS", {256, 66}},
        {"IQ2_XS", {256, 74}}, {"IQ2_S", {256, 82}},  {"IQ3_XXS", {256, 98}}, {"IQ3_S", {256, 110}},
        {"IQ1_S", {256, 50}},  {"IQ1_M", {256, 56}},  {"IQ4_XS", {256, 136}}};
    const auto &[block_elements, block_bytes] = blocks.at(type);
    EXPECT_EQ(elements % block_elements, 0u) << type;
    return elements / block_elements * block_bytes;
}

/** @return Whether CUDA stores this format's per-block code in a separate bit-packed plane. */
bool hasSeparateCodePlane(const std::string &type)
{
    return type == "Q4_K" || type == "Q5_K" || type == "Q6_K" || type == "Q2_K" || type == "Q3_K" ||
           type == "IQ2_S" || type == "IQ3_S" || type == "IQ1_M" || type == "IQ4_XS";
}

/** @return One-tensor DeepSeek-V4 profile with the real model's global geometry. */
ModelMemoryProfile profileOf(const TensorSizeInfo &tensor)
{
    ModelMemoryProfile profile;
    profile.architecture = "deepseek4";
    profile.n_layers = 43;
    profile.d_model = 4096;
    profile.vocab_size = 129280;
    profile.expert_count = 256;
    profile.tensors = {tensor};
    profile.total_native_bytes = tensor.native_bytes;
    return profile;
}
} // namespace

TEST(DeepSeekV4UDVariantWeightDensity, EverySparkFittingVariantIsAdmittedAtFileDensityOnCUDA)
{
    const auto inventory = nlohmann::json::parse(test::dsv4::kUDInventory);
    ASSERT_EQ(inventory.size(), 6u);
    for (const auto &[variant, families] : inventory.items())
    {
        SCOPED_TRACE(variant);
        std::size_t file_total = 0, device_total = 0, coded_planes = 0, host_tables = 0;
        for (const auto &family : families)
        {
            const std::string name = family[0], type = family[1], layer_list = family[3];
            const auto dims = family[2].get<std::vector<std::size_t>>();
            std::size_t elements = 1;
            for (const auto d : dims) elements *= d;
            std::vector<int> layers;
            if (layer_list.empty())
                layers.push_back(-1);
            else
                for (std::size_t at = 0; at < layer_list.size();)
                {
                    const auto comma = layer_list.find(',', at);
                    layers.push_back(std::stoi(layer_list.substr(at, comma - at)));
                    at = comma == std::string::npos ? layer_list.size() : comma + 1;
                }
            for (const int layer : layers)
            {
                std::string tensor_name = name;
                if (layer >= 0) tensor_name.replace(tensor_name.find(".N."), 3, "." + std::to_string(layer) + ".");
                SCOPED_TRACE(tensor_name + " " + type);
                const std::size_t file = ggufBytes(type, elements);
                const TensorSizeInfo tensor{tensor_name, file, type, elements, dims.at(0), std::max(0, layer)};
                const auto estimate = WeightMemoryEstimator::estimate(profileOf(tensor), DeviceId::cuda(0));
                file_total += file;
                device_total += estimate.device_bytes;
                if (type == "I32")
                {
                    // Integer routing tables are host-resident lookups.
                    EXPECT_LE(estimate.device_bytes, file);
                    host_tables += file - estimate.device_bytes;
                    continue;
                }
                const std::size_t allowance = hasSeparateCodePlane(type) ? 256u : 0u;
                coded_planes += allowance ? 1u : 0u;
                EXPECT_GE(estimate.device_bytes, file) << "prepared bytes below the file lose information";
                EXPECT_LE(estimate.device_bytes, file + allowance)
                    << "CUDA admits more than the GGUF stores beyond code-plane alignment";
            }
        }
        EXPECT_LE(device_total, file_total + 256u * coded_planes);
        EXPECT_GE(device_total + host_tables, file_total);
        RecordProperty(variant + "_file_bytes", std::to_string(file_total));
        RecordProperty(variant + "_cuda_bytes", std::to_string(device_total));
    }
}
} // namespace llaminar2
