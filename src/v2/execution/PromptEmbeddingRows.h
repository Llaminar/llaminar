/**
 * @file PromptEmbeddingRows.h
 * @brief Request-owned replacement embedding rows for prompt positions (multimodal input).
 *
 * A multimodal request carries rows produced outside the text graph (for
 * DeepSeek-V4 vision: the learned image-sentinel vectors and the encoded image
 * patches) together with a per-position map naming which row replaces the
 * token embedding at that prompt position. The request layer builds this once
 * from host inputs; the participant admits it with the request's other inputs
 * (one H2D at request admission on a GPU), and the model's embedding graph
 * applies it. The value is immutable after construction and shared by the
 * request pipeline, so admission never copies or mutates it.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace llaminar2
{
/** @brief Immutable replacement rows and their prompt-position map. */
struct PromptEmbeddingRows
{
    int width = 0;                          ///< Row width (the model's d_model).
    std::vector<float> rows;                ///< [rowCount(), width] replacement rows.
    /**
     * @brief Per prompt position: replacement row index + 1, or 0 to keep the token embedding.
     * @details Covers the whole prompt (index = absolute position). Model graphs
     * may restrict replacement further, e.g. to sentinel token ids.
     */
    std::vector<std::int32_t> position_rows;
    /**
     * @brief Prompt [begin, end) ranges one prefill chunk must hold whole.
     * @details Model-declared (DeepSeek-V4: each image block START..END, which
     * attends bidirectionally). Sorted and disjoint.
     */
    std::vector<std::pair<int, int>> atomic_spans;

    /** @return Number of replacement rows. */
    std::size_t rowCount() const noexcept { return width > 0 ? rows.size() / static_cast<std::size_t>(width) : 0; }

    /** @return Whether the geometry is self-consistent (every map entry names a real row). */
    bool valid() const noexcept
    {
        if (width <= 0 || rows.size() % static_cast<std::size_t>(width) != 0) return false;
        for (const auto v : position_rows)
            if (v < 0 || static_cast<std::size_t>(v) > rowCount()) return false;
        int previous_end = 0;
        for (const auto &[begin, end] : atomic_spans)
        {
            if (begin < previous_end || end <= begin || static_cast<std::size_t>(end) > position_rows.size())
                return false;
            previous_end = end;
        }
        return true;
    }
};
} // namespace llaminar2
