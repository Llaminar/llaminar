/**
 * @file backend_selector.h
 * @brief Centralized matmul / attention backend selection for prefill vs decode stages.
 */
#pragma once
#include <string>
#include <cstdint>

namespace llaminar
{

    enum class BackendKind : uint8_t
    {
        OpenBLAS_SingleThread,
        OpenBLAS_MultiThread,
        COSMA_Prefill,
    };

    struct BackendDecision
    {
        BackendKind kind;
        std::string reason;
        bool use_cosma() const { return kind == BackendKind::COSMA_Prefill; }
    };

    struct BackendContext
    {
        bool is_prefill = true;
        int seq_len = 0; // current sequence length
        int d_model = 0; // hidden size
        int n_layers = 0;
        int world = 1; // MPI world size
    };

    // Heuristic selection combining environment flags + basic size thresholds.
    BackendDecision selectAttentionBackend(const BackendContext &ctx);

} // namespace llaminar
