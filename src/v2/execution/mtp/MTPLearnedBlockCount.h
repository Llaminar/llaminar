/**
 * @file MTPLearnedBlockCount.h
 * @brief One source-directory calculation of learned predictor cardinality.
 *
 * Draft-token depth is not a learned-block count. Discovery, main-layer
 * boundaries and physical-memory BOM inputs share this metadata-only query so
 * an omitted optional GGUF key cannot erase real predictor state. A count is
 * not an availability certificate: requireMTPWeightManifest must still verify
 * every mandatory source role before MTP admission.
 */
#pragma once

#include <string>

namespace llaminar2
{
    class IModelLoader;
    struct GGUFModel;

    /**
     * @brief Resolve declared or directory-inferred learned blocks without I/O.
     * @param loader Immutable source metadata and tensor-name directory.
     * @param architecture Namespace of the source's GGUF metadata keys.
     * @param block_count Raw block count or an already known main-layer boundary.
     * @return Learned predictor count, including an incomplete declared block.
     * @throws std::invalid_argument For a count that cannot fit runtime geometry.
     */
    [[nodiscard]] int mtpLearnedBlockCount(
        const IModelLoader &loader, const std::string &architecture, int block_count);

    /**
     * @brief Project the same count directly from a parsed GGUF directory.
     * @param model Exact model header and tensor inventory used by memory planning.
     * @return Same learned count as the loader view of that source.
     * @throws std::invalid_argument For an unrepresentable block/count value.
     *
     * This overload borrows the directory; it does not construct another loader,
     * read weights, allocate runtime state or create a second accounting ledger.
     */
    [[nodiscard]] int mtpLearnedBlockCount(const GGUFModel &model);
}
