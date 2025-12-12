/**
 * @file OnlineSoftmax.h
 * @brief Microkernel μK2: Streaming online softmax computation
 * @author David Sanftenberg
 * 
 * Implements online softmax algorithm that processes scores one at a time
 * without needing to store all scores. This enables fused attention where
 * we compute Q·K, update softmax, and accumulate V in a single pass.
 * 
 * Reference: Flash Attention (Dao et al., 2022)
 */

#pragma once

#include <cmath>
#include <limits>

namespace llaminar::v2::kernels::microkernels {

/**
 * @brief State for streaming online softmax computation
 * 
 * Maintains running maximum and sum of exponentials for numerically
 * stable softmax computation without storing all scores.
 */
struct OnlineSoftmaxState {
    float max_score;    ///< Running maximum score seen so far
    float sum_exp;      ///< Running sum of exp(score - max_score)
    bool initialized;   ///< Whether first score has been processed
    
    OnlineSoftmaxState() 
        : max_score(-std::numeric_limits<float>::infinity())
        , sum_exp(0.0f)
        , initialized(false) {}
};

/**
 * @brief Initialize online softmax state
 * @return Fresh state ready for first score
 */
inline OnlineSoftmaxState online_softmax_init() {
    return OnlineSoftmaxState();
}

/**
 * @brief Update softmax state with a new score
 * 
 * Returns the weight for this score (before final normalization).
 * If the new score becomes the new maximum, existing accumulations
 * need to be corrected using online_softmax_correction().
 * 
 * @param state Softmax state to update (modified in place)
 * @param score New score to incorporate
 * @return Weight for this score (exp(score - max) before normalization)
 */
float online_softmax_update(OnlineSoftmaxState& state, float score);

/**
 * @brief Compute correction factor when maximum changes
 * 
 * When a new score becomes the maximum, all previously accumulated
 * weighted values need to be multiplied by this correction factor.
 * 
 * correction = exp(old_max - new_max)
 * 
 * @param old_max Previous maximum score
 * @param new_max New maximum score (must be > old_max)
 * @return Correction factor to apply to existing accumulations
 */
inline float online_softmax_correction(float old_max, float new_max) {
    return std::exp(old_max - new_max);
}

/**
 * @brief Finalize softmax and get normalization factor
 * 
 * After all scores have been processed, call this to get 1/sum_exp
 * for normalizing the accumulated weighted values.
 * 
 * @param state Final softmax state
 * @return 1.0 / sum_exp (multiply accumulated values by this)
 */
inline float online_softmax_finalize(const OnlineSoftmaxState& state) {
    return (state.sum_exp > 0.0f) ? (1.0f / state.sum_exp) : 0.0f;
}

/**
 * @brief Check if maximum changed during last update
 * 
 * Useful for determining whether correction factor needs to be applied.
 * 
 * @param state Current state
 * @param prev_max Maximum before last update
 * @return true if max_score changed
 */
inline bool online_softmax_max_changed(const OnlineSoftmaxState& state, float prev_max) {
    return state.max_score != prev_max;
}

} // namespace llaminar::v2::kernels::microkernels
