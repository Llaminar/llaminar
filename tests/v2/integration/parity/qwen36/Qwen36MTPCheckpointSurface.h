/**
 * @file Qwen36MTPCheckpointSurface.h
 * @brief Production-observable recursive checkpoint surfaces for Qwen3.6.
 *
 * A reference model can materialize algebraic intermediates that an optimized
 * production graph fuses away.  The typed parity definition must therefore
 * name the complete observable cut through the live graph: every declared
 * checkpoint is mandatory, while values absent from this list are never
 * manufactured by a test-only execution path.  Dense Qwen3.6 exposes gate and
 * up inputs plus the fused down-projection result, so the unmaterialized
 * SwiGLU temporary is deliberately outside this surface.
 */

#pragma once

#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2::test::parity::qwen36
{
    /** Model-block checkpoints retained by the optimized dense MTP graph. */
    inline constexpr std::array<std::string_view, 22>
        kQwen36DenseMTPModelStageSuffixes = {
            "EMBEDDING",
            "NORM_HIDDEN",
            "NORM_EMBEDDING",
            "CONCAT",
            "FC",
            "ATTENTION_NORM",
            "Q_PROJECTION",
            "FA_GATE",
            "K_PROJECTION",
            "V_PROJECTION",
            "Q_NORM",
            "K_NORM",
            "ATTENTION_CONTEXT",
            "ATTENTION_CONTEXT_GATED",
            "ATTENTION_OUTPUT",
            "FFN_NORM",
            "FFN_GATE",
            "FFN_UP",
            "FFN_DOWN",
            "FFN_RESIDUAL",
            "FINAL_NORM",
            "LM_HEAD",
        };

    /**
     * @brief Build the complete typed reference/runtime comparison surface.
     * @return Terminal-hidden selection followed by every live model stage.
     */
    inline std::vector<std::string> qwen36DenseMTPCheckpointSurface()
    {
        std::vector<std::string> surface;
        surface.reserve(kQwen36DenseMTPModelStageSuffixes.size() + 1u);
        surface.emplace_back("TERMINAL_HIDDEN_ROW_SELECT");
        for (const std::string_view suffix :
             kQwen36DenseMTPModelStageSuffixes)
        {
            surface.emplace_back(suffix);
        }
        return surface;
    }
} // namespace llaminar2::test::parity::qwen36
