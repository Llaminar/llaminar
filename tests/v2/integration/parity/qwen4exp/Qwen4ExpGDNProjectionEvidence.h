/**
 * @file Qwen4ExpGDNProjectionEvidence.h
 * @brief Original-source validation of GDN projection submission/capture evidence.
 *
 * These host-authored route records identify the prepared native operation
 * submitted or recorded. Their counts are not token counts or GPU replay
 * counts; canonical production_path.csv independently proves actual replay.
 */
#pragma once
#include "models/qwen4exp/Qwen4ExpConfig.h"
#include "utils/PerfStatsCollector.h"
#include "tensors/NativeVnniFormatInfo.h"
#include <charconv>
#include <cmath>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <stdexcept>

namespace llaminar2::test::parity::qwen4exp
{
/** @brief Exact operation family; no broad kernel/timing export is required. */
inline constexpr const char *kGDNProjectionRouteFilter = "kernel.gdn_projection_route";
/**
 * @brief Require actual selected route records against the original main GGUF directory.
 * @param records Unmodified exact-family observations, retained as raw CSV before validation.
 * @param directory Original model-owned source directory; no tensor payload is read.
 * @throws std::invalid_argument For missing, malformed, foreign or source-incompatible selected evidence.
 * @details Main GDN N/K and Q8_0 format come from original tensor metadata,
 * not a second layout formula, requested selector, or matrix-label guess.
 */
inline void requireQwen4ExpGDNQKVRouteEvidence(std::span<const PerfStatRecord> records,
    const GGUFModel &directory)
{
    const Qwen4ExpConfig model(directory);
    std::set<std::pair<std::string, std::string>> original_shapes;
    for (uint32_t layer = 0; layer < model.mainLayers(); ++layer)
    {
        if (model.mixers()[layer] != Qwen4ExpMixer::GatedDeltaNet) continue;
        const auto *source = directory.findTensor("blk." + std::to_string(layer) + ".attn_qkv.weight");
        if (!source || source->type != GGUFTensorType::Q8_0 || source->dimensions.size() != 2 ||
            !source->dimensions[0] || !source->dimensions[1])
            throw std::invalid_argument("Selected QFN QKV route lacks its original main native Q8_0 source metadata");
        original_shapes.emplace(std::to_string(source->dimensions[0]), std::to_string(source->dimensions[1]));
    }
    bool selected = false;
    for (const auto &record : records)
    {
        if (record.domain != "kernel" || record.name != "gdn_projection_route" ||
            record.kind != PerfStatRecord::Kind::Counter || !record.count ||
            record.count > 9007199254740991ULL || !std::isfinite(record.value) ||
            record.value != static_cast<double>(record.count))
            throw std::invalid_argument("Malformed QFN native submission/capture route record");
        const auto tag = [&](const char *name) -> const std::string &
        {
            const auto found = record.tags.find(name);
            if (found == record.tags.end() || found->second.empty())
                throw std::invalid_argument("QFN native route record is missing an original operation tag");
            return found->second;
        };
        if (tag("route") != "qkv_two_term_q8") continue;
        const auto &rows = tag("m");
        int m = 0;
        const auto parsed = std::from_chars(rows.data(), rows.data() + rows.size(), m);
        if (parsed.ec != std::errc{} || parsed.ptr != rows.data() + rows.size() || m <= 0 ||
            tag("projections") != "1" || tag("names") != "qkv" || tag("codebooks") != std::to_string(native_vnni_formats::Q8_0.codebook_id) ||
            !original_shapes.contains({tag("n"), tag("k")}))
            throw std::invalid_argument("Selected QFN QKV route does not match its original native operation/source N/K");
        selected = true;
    }
    if (!selected)
        throw std::invalid_argument("Missing QFN qkv_two_term_q8 native submission/capture route evidence");
}
} // namespace llaminar2::test::parity::qwen4exp
