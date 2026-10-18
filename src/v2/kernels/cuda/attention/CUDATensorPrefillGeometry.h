/**
 * @file CUDATensorPrefillGeometry.h
 * @brief Device-free admission of complete and partial warp-owned query buckets.
 *
 * Each native CTA owns 128 queries. Larger buckets add CTAs without changing
 * registers or shared storage; a partial final tile owns no padded global
 * operand or summary bytes. This policy is shared by capture and Unit tests.
 */
#pragma once
#include "../../attention/AttentionExecutionPolicy.h"
#include "../../../tensors/TensorType.h"
#include <cstdint>
#include <optional>

namespace llaminar2::cuda::tensor_prefill {
/** Complete admitted query/native-KV head mapping, independent of model names. */
enum class HeadGeometry : std::uint8_t {
    Local8x1, Local12x2, Local16x2, Local24x4, Replicated8x2, Replicated12x4
};

/** Immutable native operand and visibility facts carried into capture. */
struct Geometry {
    int batch_size = 0; ///< Captured requests.
    int query_rows = 0; ///< Physical query extent; no tile divisibility requirement.
    int head_dim = 0; ///< Actual head width.
    int query_heads = 0; ///< Local query heads.
    int kv_heads = 0; ///< Visible native KV heads.
    int head_start = 0; ///< Global offset for replicated KV.
    int gqa_repetition = 0; ///< Zero for sharded KV, otherwise global GQA ratio.
    TensorType kv_format = TensorType::FP32; ///< Native execution operands.
    attention::AttentionVisibilityPolicy visibility = attention::AttentionVisibilityPolicy::General;
};

/**
 * @brief Admit the qualified tensor architecture without a maximum query bucket.
 * @param geometry Complete immutable native head/operand geometry.
 * @return Exact compiled head entry, or absence when its contract does not apply.
 * M512 is the smallest qualified bucket. Larger and irregular buckets retain
 * the same numerical entry and fixed per-block memory, rather than dispatching
 * to the older architecture solely because M increased.
 */
[[nodiscard]] constexpr std::optional<HeadGeometry> selectHeadGeometry(const Geometry &geometry) noexcept {
    if (geometry.batch_size != 1 || geometry.query_rows < 512 || geometry.head_dim != 256 ||
        geometry.kv_format != TensorType::FP16 ||
        geometry.visibility != attention::AttentionVisibilityPolicy::CausalFullContext)
        return std::nullopt;
    if (geometry.gqa_repetition == 0) {
        // Sharded KV is indexed locally. Its published global query offset is
        // meaningful for other stages but must not exclude this native entry.
        if (geometry.query_heads == 8 && geometry.kv_heads == 1) return HeadGeometry::Local8x1;
        if (geometry.query_heads == 12 && geometry.kv_heads == 2) return HeadGeometry::Local12x2;
        if (geometry.query_heads == 16 && geometry.kv_heads == 2) return HeadGeometry::Local16x2;
        if (geometry.query_heads == 24 && geometry.kv_heads == 4) return HeadGeometry::Local24x4;
    } else {
        const auto global_heads = static_cast<std::int64_t>(geometry.kv_heads) * geometry.gqa_repetition;
        if (geometry.head_start < 0 || static_cast<std::int64_t>(geometry.head_start) + geometry.query_heads > global_heads)
            return std::nullopt;
        if (geometry.query_heads == 8 && geometry.kv_heads == 2 && geometry.gqa_repetition == 8)
            return HeadGeometry::Replicated8x2;
        if (geometry.query_heads == 12 && geometry.kv_heads == 4 && geometry.gqa_repetition == 6)
            return HeadGeometry::Replicated12x4;
    }
    return std::nullopt;
}
} // namespace llaminar2::cuda::tensor_prefill
