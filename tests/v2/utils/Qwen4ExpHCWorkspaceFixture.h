/**
 * @file Qwen4ExpHCWorkspaceFixture.h
 * @brief Native-source metadata fixture with an explicit SingleQ8 delta baseline.
 *
 * The fixture isolates selected activation banks while retaining positive
 * attention and context geometry; it loads no weights and discovers no device.
 * Delta oracles author their baseline independently of production defaults.
 * Default-selection tests opt into the actual default explicitly so promotion
 * cannot silently erase the source-backed residual allocation requirement.
 */
#pragma once
#include "CPUExecutionTestGeometry.h"
#include "planning/WorkspaceMemoryEstimator.h"

namespace llaminar2::test
{
/** @return A two-layer profile with one original Q8_0 HC-up source. */
inline ModelMemoryProfile qfnHCWorkspaceProfile()
{
    ModelMemoryProfile value;
    value.architecture = "qwen4exp";
    value.n_layers = 2;
    value.d_model = 128;
    value.d_ff = 256;
    value.n_heads = 4;
    value.n_kv_heads = 2;
    value.head_dim = 32;
    value.vocab_size = 32;
    value.max_seq_len = 4096;
    value.flash_next = Qwen4ExpMemoryGeometry{
        .residual_width = 512, .terminal_head_width = 128,
        .residual_branches = 4, .residual_gate_rank = 32};
    value.tensors.push_back({.name = "blk.0.hc_attn_up.weight",
        .native_bytes = 512u * 34, .quant_type = "Q8_0",
        .elements = 512u * 32, .K = 32, .layer_index = 0});
    return value;
}

/**
 * @brief Declare a complete participant without querying its runtime.
 * @param device Backend whose native workspace contract is under test.
 * @param rows Admitted physical prompt bucket.
 * @return Explicit host/GPU geometry; runtime dispatch remains disabled.
 */
inline WorkspaceMemoryGeometry qfnHCWorkspaceGeometry(DeviceId device, int rows)
{
    WorkspaceMemoryGeometry geometry{.device = device, .device_compute_units = device.is_cpu() ? 1 : 64,
        .cpu_execution = device.is_cpu() ? kSyntheticCPUExecutionGeometry : CPUExecutionGeometry{},
        .batch_size = 1, .resident_graph_rows = rows, .max_context_rows = 4096,
        .local_d_ff = 256, .local_query_heads = 4, .local_kv_heads = 2,
        .local_vocab = 32, .first_layer = 0, .last_layer = 1, .total_shards = 1};
    // This fixture explicitly authors the SingleQ8 baseline for delta oracles.
    geometry.qfn_hc_up_activation = QuantizedActivationOperandPrecision::SingleQ8;
    return geometry;
}
} // namespace llaminar2::test
