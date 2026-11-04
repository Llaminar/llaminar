/**
 * @file AttentionStateOwner.h
 * @brief Typed ownership of a model's per-layer attention state.
 *
 * Most architectures keep attention K/V in the canonical KV cache. DeepSeek-V4
 * keeps its raw window, compressor state and compressed/indexer pools in
 * model-owned arena banks; its canonical cache carries only the sequence
 * frontier (every layer is a `model_state` hybrid layer). This single
 * authority drives cache construction (DeviceGraphOrchestrator), runtime
 * capability gates (TPPPValidator: prefix caching cannot archive the banks
 * yet) and graph configuration, so no consumer branches on model names.
 */
#pragma once

#include <cstdint>
#include <string_view>

namespace llaminar2
{
/** @brief Who owns a model's attention state. */
enum class AttentionStateOwner : std::uint8_t
{
    KVCache,    ///< Canonical KV cache payload (full attention / GDN state).
    ModelBanks, ///< Model arena banks; the cache owns only the sequence frontier.
};

/**
 * @brief The one architecture -> ownership mapping.
 * @param architecture GGUF `general.architecture`.
 */
constexpr AttentionStateOwner attentionStateOwnerOf(std::string_view architecture) noexcept
{
    return architecture == "deepseek4" ? AttentionStateOwner::ModelBanks : AttentionStateOwner::KVCache;
}
} // namespace llaminar2
