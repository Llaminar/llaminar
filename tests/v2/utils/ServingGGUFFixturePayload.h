/**
 * @file ServingGGUFFixturePayload.h
 * @brief Nondegenerate synthetic source bytes for tiny full-attention serving fixtures.
 *
 * PlanningGGUFFixture intentionally leaves sparse zero holes. A test which
 * executes real CPU stages with the ordinary verifier must instead publish
 * finite embeddings, normalization scales and projections. These bytes are
 * test data, never model-quality/reference evidence or an inference fallback.
 */
#pragma once
#include "PlanningGGUFFixture.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include <algorithm>
#include <fstream>
#include <vector>

namespace llaminar2::test
{
    /**
     * @brief Populate every authored tensor before any real model loader retains it.
     * @param fixture Tiny fixture whose native payloads are FP32 or Q4_0.
     * @throws std::invalid_argument For a source format outside this explicit fixture contract.
     * @throws std::runtime_error When the source directory cannot be read.
     *
     * All norm scales are one. Other FP32 sources and Q4_0 expert blocks have
     * small positive nonuniform values, keeping residuals nonzero and finite
     * through every declared layer. The directory and all extents are unchanged.
     * Validate the whole format inventory before writing any payload.
     */
    inline void populateServingGGUFFixture(const PlanningGGUFFixture &fixture)
    {
        ModelLoader directory;
        directory.setUseMmap(false);
        if (!directory.loadModel(fixture.path()))
            throw std::runtime_error("Cannot read serving fixture source directory");
        const auto &model = directory.getModel();
        for (const auto &tensor : model.tensors)
            if ((tensor.type != GGUFTensorType::F32 && tensor.type != GGUFTensorType::Q4_0) ||
                (tensor.type == GGUFTensorType::F32 && tensor.size_bytes % sizeof(float)) ||
                (tensor.type == GGUFTensorType::Q4_0 && tensor.size_bytes % sizeof(Q4_0Block)))
                throw std::invalid_argument("Serving fixture payload requires exact FP32/Q4_0 native tensors");
        std::fstream stream(fixture.path(), std::ios::binary | std::ios::in | std::ios::out);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        for (const auto &tensor : model.tensors)
        {
            stream.seekp(model.data_offset + tensor.offset);
            if (tensor.type == GGUFTensorType::F32)
            {
                const bool norm = tensor.name.find("norm.weight") != std::string::npos;
                const bool embedding = tensor.name == "token_embd.weight";
                std::vector<float> values(tensor.size_bytes / sizeof(float));
                for (std::size_t index = 0; index < values.size(); ++index)
                    values[index] = norm ? 1.0F :
                        (embedding ? 0.01F : 0.001F) * static_cast<float>(1 + index % 29) / 29.0F;
                stream.write(reinterpret_cast<const char *>(values.data()), tensor.size_bytes);
            }
            else
            {
                std::vector<Q4_0Block> blocks(tensor.size_bytes / sizeof(Q4_0Block));
                for (std::size_t index = 0; index < blocks.size(); ++index)
                {
                    blocks[index].d = fp32_to_fp16(1.0F / 2048.0F);
                    // Both native nibble halves decode to positive1/2; this
                    // exact constant-scale source is valid on every backend.
                    std::fill_n(blocks[index].qs, 16, index % 2 ? 0x9a : 0xa9);
                }
                stream.write(reinterpret_cast<const char *>(blocks.data()), tensor.size_bytes);
            }
        }
    }
}
