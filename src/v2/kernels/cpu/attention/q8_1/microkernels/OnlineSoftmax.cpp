/**
 * @file OnlineSoftmax.cpp
 * @brief Implementation of streaming online softmax
 * @author David Sanftenberg
 */

#include "OnlineSoftmax.h"
#include "FastExp.h"

namespace llaminar::v2::kernels::microkernels {

float online_softmax_update(OnlineSoftmaxState& state, float score) {
    if (!state.initialized) {
        // First score becomes the baseline
        state.max_score = score;
        state.sum_exp = 1.0f;  // exp(0) = 1
        state.initialized = true;
        return 1.0f;  // Weight before final normalization
    }
    
    if (score > state.max_score) {
        // New maximum found - need to rescale existing accumulation
        // All previous exp(old_score - old_max) become exp(old_score - new_max)
        // = exp(old_score - old_max) * exp(old_max - new_max)
        // = old_weight * correction
        
        float correction = fast_exp(state.max_score - score);
        state.sum_exp *= correction;
        state.max_score = score;
        
        // This score's weight is exp(score - score) = 1
        state.sum_exp += 1.0f;
        return 1.0f;
    } else {
        // Normal case: compute weight relative to current max
        float weight = fast_exp(score - state.max_score);
        state.sum_exp += weight;
        return weight;
    }
}

} // namespace llaminar::v2::kernels::microkernels
