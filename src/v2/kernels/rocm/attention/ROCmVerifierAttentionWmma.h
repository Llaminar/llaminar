/**
 * @file ROCmVerifierAttentionWmma.h
 * @brief Explicit recording choices for serial-row-equivalent grouped attention.
 *
 * Decode owns two half-dot updates per original lane and a lane-specific XOR
 * tree. Its matrix candidate is separate from ordinary-prefill Q/K engines.
 * Automatic retains the installed ordered producer until complete-operation
 * measurements qualify a native domain. Diagnostic scopes require fresh graph
 * recording on the original GPU worker; retained graphs never read host policy.
 */
#pragma once

#include <string_view>

namespace llaminar2::rocm
{
/** @brief Grouped decode producers; no choice changes the serial arithmetic contract. */
enum class VerifierAttentionEngine { Automatic, OrderedDot2, WmmaLaneExact };

/** @brief Immutable shared-KV launch geometry, independent of device row metadata. */
struct VerifierAttentionGeometry
{
    int rows;
    int kv_capacity;
    int query_heads;
    int kv_heads;
    int head_dim;
    int split_capacity;
    int head_start;
    int replicated_gqa;
};

/** @param engine Declared recording choice. @return Stable diagnostic/counter name. */
[[nodiscard]] constexpr const char *verifierAttentionEngineName(VerifierAttentionEngine engine) noexcept
{
    switch (engine)
    {
    case VerifierAttentionEngine::Automatic: return "automatic";
    case VerifierAttentionEngine::OrderedDot2: return "ordered_dot2";
    case VerifierAttentionEngine::WmmaLaneExact: return "wmma_lane_exact";
    }
    return "invalid";
}

/** @brief Nestable worker-local diagnostic recording policy.
 * A caller must record a fresh graph within this scope. It is not a mutable
 * serving configuration or permission to reuse a differently selected graph.
 */
class ScopedVerifierAttentionEngine final
{
public:
    /** @brief Select the producer for subsequent fresh recordings on this worker.
     * @param engine Installed selection or explicit native candidate.
     * @throws std::invalid_argument For an undeclared enum value. */
    explicit ScopedVerifierAttentionEngine(VerifierAttentionEngine engine);
    /** @brief Restore the enclosing worker's recording policy. */
    ~ScopedVerifierAttentionEngine();
    ScopedVerifierAttentionEngine(const ScopedVerifierAttentionEngine &) = delete;
    ScopedVerifierAttentionEngine &operator=(const ScopedVerifierAttentionEngine &) = delete;
private:
    VerifierAttentionEngine previous_;
};

/** @brief Record a successful shared-KV producer submission without device observations.
 * @param engine Actual embedded producer. @param geometry Immutable launch envelope.
 * @param native_isa Authenticated physical ISA. @param device Owning visible ordinal.
 * This counts submissions during setup/recording, not executions during replay.
 */
void recordVerifierAttentionSubmission(VerifierAttentionEngine engine,
    const VerifierAttentionGeometry &geometry, std::string_view native_isa, int device);
}
