/**
 * @file PlanningSampleWire.h
 * @brief Shared private native-matrix codec for planning source publications.
 *
 * Both ordinary matrices and expert triplets validate the same GGUF block
 * geometry here. This is immutable source metadata, not an allocation ledger,
 * prepared representation, backend capability table or service estimate.
 */
#pragma once
#include "PlanningModelSample.h"
#include "loaders/ModelLoader.h"
#include <nlohmann/json.hpp>
#include <limits>

namespace llaminar2::planning_sample_wire
{
    /** @return Nonnegative bounded integer without JSON's signed/float conversion shortcuts. */
    inline size_t unsignedField(const nlohmann::json &value, size_t maximum)
    {
        if (!value.is_number_integer() ||
            (!value.is_number_unsigned() && value.get<int64_t>() < 0) || value.get<uint64_t>() > maximum)
            throw std::invalid_argument("Planning sample integer is negative, fractional or out of range");
        return value.get<size_t>();
    }

    /** @return Canonical GGUF spelling; runtime type names otherwise remain unchanged. */
    inline std::string sourceFormat(TensorType type)
    {
        if (type == TensorType::FP32) return "F32";
        if (type == TensorType::FP16) return "F16";
        return tensorTypeName(type);
    }

    /** @brief Decoded source description, validated before any payload allocation. */
    struct NativeMatrix
    {
        std::string name;
        uint32_t gguf_type;
        PlanningModelSampleGeometry geometry;
        std::string format;
    };

    /** @return Exact native-matrix fields shared by single-matrix and FFN publications. */
    inline nlohmann::json encode(const NativeMatrix &matrix)
    {
        return {{"name", matrix.name}, {"type", matrix.gguf_type}, {"n", matrix.geometry.n},
            {"k", matrix.geometry.k}, {"bytes", matrix.geometry.source_bytes}, {"format", matrix.format}};
    }

    /** @return Complete native layout, rejecting inconsistent formats, extents and block boundaries. */
    inline NativeMatrix decode(const nlohmann::json &wire)
    {
        if (!wire.is_object() || wire.size() != 6)
            throw std::invalid_argument("Malformed planning source matrix");
        NativeMatrix result{wire.at("name").get<std::string>(),
            static_cast<uint32_t>(unsignedField(wire.at("type"), std::numeric_limits<uint32_t>::max())),
            {unsignedField(wire.at("n"), std::numeric_limits<int>::max()),
             unsignedField(wire.at("k"), std::numeric_limits<int>::max()),
             unsignedField(wire.at("bytes"), std::numeric_limits<size_t>::max())},
            wire.at("format").get<std::string>()};
        GGUFTensorInfo info{};
        info.type = static_cast<GGUFTensorType>(result.gguf_type);
        const auto type = ggufToTensorType(info.type);
        const size_t block = std::max(size_t{1}, info.getBlockSize()), block_bytes = info.getTypeSize();
        const auto &geometry = result.geometry;
        if (result.name.empty() || !geometry.n || !geometry.k || !block_bytes || geometry.k % block ||
            sourceFormat(type) != result.format || geometry.n > std::numeric_limits<size_t>::max() / (geometry.k / block))
            throw std::invalid_argument("Planning sample has invalid native geometry or format");
        const size_t blocks = geometry.n * (geometry.k / block);
        if (blocks > std::numeric_limits<size_t>::max() / block_bytes || geometry.source_bytes != blocks * block_bytes)
            throw std::invalid_argument("Planning sample native byte extent disagrees with geometry");
        return result;
    }
}
