// Instrumentation counters for quant linear fused kernel
#pragma once
#include <atomic>
namespace llaminar
{
    struct QuantLinearInstrumentation
    {
        std::atomic<long long> decoded_blocks{0};
        std::atomic<long long> applied_blocks{0};
        std::atomic<long long> simd_path{0};
        std::atomic<long long> scalar_path{0};
        std::atomic<long long> partial_blocks{0};
        void reset()
        {
            decoded_blocks.store(0, std::memory_order_relaxed);
            applied_blocks.store(0, std::memory_order_relaxed);
            simd_path.store(0, std::memory_order_relaxed);
            scalar_path.store(0, std::memory_order_relaxed);
            partial_blocks.store(0, std::memory_order_relaxed);
        }
    };
    QuantLinearInstrumentation &quantLinearInstr();
}
