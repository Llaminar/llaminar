/**
 * @file Test__CUDAWorkspaceMemoryEstimator.cpp
 * @brief Device-backed regressions for CUDA graph-family memory admission.
 *
 * CUDA NativeVNNI prefill scratch is selected from the concrete device launch
 * policy, so its final admission contract cannot be certified by a device-free
 * Unit test.  These focused tests use only synthetic tensor metadata: they do
 * not load a model or allocate model weights, but they query the same CUDA
 * workspace authority used immediately before production weight admission.
 */

#include "backends/BackendManager.h"
#include "planning/WorkspaceMemoryEstimator.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>

using namespace llaminar2;

namespace
{

/**
 * @brief Build the fixed non-tensor geometry around a two-layer Qwen MoE graph.
 * @return Profile whose only varying bytes come from the installed matrices.
 */
ModelMemoryProfile profile()
{
    ModelMemoryProfile value;
    value.architecture = "qwen35moe";
    value.n_layers = 2;
    value.d_model = 2048;
    value.d_ff = 512;
    value.n_heads = 16;
    value.n_kv_heads = 2;
    value.head_dim = 128;
    value.vocab_size = 248320;
    value.max_seq_len = 4096;
    return value;
}

/**
 * @brief Build the exact local CUDA policy query used during admission.
 * @return Complete participant geometry for CUDA ordinal zero.
 */
WorkspaceMemoryGeometry geometry()
{
    return WorkspaceMemoryGeometry{
        .device = DeviceId::cuda(0),
        .device_compute_units = 82,
        .batch_size = 1,
        .resident_graph_rows = 512,
        .max_context_rows = 4096,
        .local_d_ff = 512,
        .first_layer = 0,
        .last_layer = 1,
        .total_shards = 1,
        .runtime_device_policy_available = true,
    };
}

/**
 * @brief Add one fused GDN projection bundle with production Q8 source format.
 * @param value Profile receiving the layer-local tensor metadata.
 * @param layer Layer index encoded in every tensor name and ownership record.
 */
void addFusedBundle(ModelMemoryProfile& value, int layer)
{
    const auto add_matrix = [&](std::string suffix,
                                std::size_t output_columns)
    {
        TensorSizeInfo tensor;
        tensor.name = "blk." + std::to_string(layer) + suffix;
        tensor.quant_type = "Q8_0";
        tensor.elements = output_columns * std::size_t{2048};
        tensor.K = 2048;
        tensor.layer_index = layer;
        value.tensors.push_back(std::move(tensor));
    };
    add_matrix(".attn_qkv.weight", 8192);
    add_matrix(".attn_gate.weight", 4096);
    add_matrix(".ssm_alpha.weight", 32);
    add_matrix(".ssm_beta.weight", 32);
}

} // namespace

/**
 * @test Serial layers share one named CUDA fused-projection scratch arena.
 *
 * The runtime workspace manager merges stable names and materializes the widest
 * extent.  Admission must do the same.  Directly appending each layer's fused
 * descriptors inflated a 48-layer production estimate by several GiB and
 * unnecessarily displaced experts from the continuation tier.
 */
TEST(Test__CUDAWorkspaceMemoryEstimator,
     FusedProjectionWorkspaceCanonicalizesAcrossSerialLayers)
{
    if (!getCUDABackend())
        GTEST_SKIP() << "CUDA backend is unavailable";

    auto value = profile();
    addFusedBundle(value, 0);
    const std::size_t one_layer =
        WorkspaceMemoryEstimator::estimate(value, geometry());

    addFusedBundle(value, 1);
    const std::size_t two_layers =
        WorkspaceMemoryEstimator::estimate(value, geometry());

    EXPECT_EQ(two_layers, one_layer)
        << "Serial CUDA layers must merge their stable fused workspace names "
           "instead of multiplying physical admission bytes.";
}
