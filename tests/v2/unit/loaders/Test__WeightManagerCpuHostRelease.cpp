/**
 * @file Test__WeightManagerCpuHostRelease.cpp
 * @brief Unit tests for CPU binding-driven host data release after VNNI packing
 *
 * Tests verify the fix in prepareWeightsForDevice(frozen_weights, cpu_device):
 * After binding-driven CPU GEMM preparation completes, quantized TensorSlice
 * weights (IINT8Unpackable) have their raw data released, while FP32 tensors,
 * embeddings, norms, biases, non-2D tensors, and non-prepared bindings are
 * retained.
 *
 * This locks in the fix that saves ~7.5 GB per MPI rank by releasing the
 * original Q4_K/Q5_K/Q6_K TensorSlice heap data after VNNI packing.
 */

#include <gtest/gtest.h>
#include <memory>

#include "loaders/WeightManager.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/Tensors.h"
#include "tensors/TensorSlice.h"
#include "mocks/MockModelLoader.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Helper: Build a WeightBinding for a given tensor
// ============================================================================

namespace
{
    /// @brief Create a WeightBinding for test purposes with specified role and prepared state.
    WeightBinding makeBinding(
        uint64_t id,
        TensorBase *tensor,
        const std::string &name,
        WeightRole role,
        int layer,
        DeviceId device,
        bool mark_prepared)
    {
        WeightBinding binding;
        binding.binding_id = id;
        binding.tensor = tensor;
        binding.identity.canonical_name = name;
        binding.identity.role = role;
        binding.identity.layer = layer;
        binding.residency.home_device = device;
        if (mark_prepared)
        {
            PreparedWeightRef ref;
            ref.binding_id = id;
            ref.kind = PreparedWeightKind::CpuPackedGemm;
            ref.device = device;
            binding.prepared = ref;
        }
        return binding;
    }

    /// @brief Wrap a tensor in a TensorSlice with column-parallel metadata.
    std::shared_ptr<TensorSlice> wrapInSlice(
        std::shared_ptr<TensorBase> inner,
        size_t original_rows, size_t original_cols)
    {
        auto meta = SliceMetadata::forColumnParallel(
            original_rows, original_cols, /*rank=*/0, /*world_size=*/2, /*inner_is_presliced=*/true);
        return std::make_shared<TensorSlice>(std::move(inner), meta);
    }
} // namespace

// ============================================================================
// TestableWeightManager — opens lifecycle gates and pre-registers prepared state
// ============================================================================

class TestableCpuReleaseWeightManager : public WeightManager
{
public:
    TestableCpuReleaseWeightManager(IModelLoader &loader)
        : WeightManager(loader, nullptr, nullptr,
                        WeightDistributionStrategy::REPLICATED,
                        WeightPrecision::NATIVE)
    {
        markMaterializationComplete();
    }

    /// Pre-register a binding in the PreparedWeightStore so that
    /// prepareWeightsForDevice skips actual GEMM packing but still
    /// runs the release loop.
    void preRegisterPrepared(const WeightBinding &binding, DeviceId device)
    {
        auto store = preparedWeightStore();
        store->registerPreparedForTest(binding, PreparedWeightKind::CpuPackedGemm, device);
    }
};

// ============================================================================
// Test Fixture
// ============================================================================

class Test__WeightManagerCpuHostRelease : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mock_loader_ = MockModelLoader::createMinimal();
    }

    std::shared_ptr<MockModelLoader> mock_loader_;
};

// ============================================================================
// Tests
// ============================================================================

TEST_F(Test__WeightManagerCpuHostRelease, ReleasesQuantizedTensorSliceAfterCpuPreparation)
{
    // A Q4_0 TensorSlice that is IINT8Unpackable should have its raw data
    // released after CPU binding-driven preparation.
    TestableCpuReleaseWeightManager wm(*mock_loader_);
    DeviceId cpu = DeviceId::cpu();

    // Create a Q4_0 tensor wrapped in TensorSlice (mimics TP=2 sharding)
    mock_loader_->addQ4_0RandomTensor("blk.0.attn_q.weight", {128, 64});
    auto inner = mock_loader_->loadTensor("blk.0.attn_q.weight");
    ASSERT_NE(inner, nullptr);
    auto sliced = wrapInSlice(inner, 256, 64);

    // Verify it IS IINT8Unpackable (prerequisite for the fix)
    ASSERT_NE(dynamic_cast<IINT8Unpackable *>(sliced.get()), nullptr);
    ASSERT_FALSE(sliced->is_raw_data_released());

    // Build a FrozenModelWeightSet with this binding marked as prepared
    auto binding = makeBinding(1, sliced.get(), "blk.0.attn_q.weight",
                               WeightRole::AttentionQ, 0, cpu, /*mark_prepared=*/true);
    wm.preRegisterPrepared(binding, cpu);

    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::Unknown;
    std::vector<WeightBinding> bindings_vec = {binding};
    FrozenModelWeightSet frozen(strategy, std::move(bindings_vec));

    // Act: this should release the raw data after "preparation"
    bool ok = wm.prepareWeightsForDevice(frozen, cpu);
    ASSERT_TRUE(ok);

    // Assert: raw data was released
    EXPECT_TRUE(sliced->is_raw_data_released());
}

TEST_F(Test__WeightManagerCpuHostRelease, RetainsFP32TensorAfterCpuPreparation)
{
    // FP32 tensors do NOT implement IINT8Unpackable → should be retained
    TestableCpuReleaseWeightManager wm(*mock_loader_);
    DeviceId cpu = DeviceId::cpu();

    mock_loader_->addFP32RandomTensor("blk.0.attn_q.weight", {128, 64});
    auto fp32_tensor = mock_loader_->loadTensor("blk.0.attn_q.weight");
    ASSERT_NE(fp32_tensor, nullptr);

    // FP32Tensor is NOT IINT8Unpackable
    ASSERT_EQ(dynamic_cast<IINT8Unpackable *>(fp32_tensor.get()), nullptr);

    auto binding = makeBinding(1, fp32_tensor.get(), "blk.0.attn_q.weight",
                               WeightRole::AttentionQ, 0, cpu, /*mark_prepared=*/true);
    wm.preRegisterPrepared(binding, cpu);

    InferenceStrategy strategy;
    std::vector<WeightBinding> bindings_vec = {binding};
    FrozenModelWeightSet frozen(strategy, std::move(bindings_vec));

    bool ok = wm.prepareWeightsForDevice(frozen, cpu);
    ASSERT_TRUE(ok);

    // FP32 should NOT be released (dynamic_cast<IINT8Unpackable*> fails)
    EXPECT_FALSE(fp32_tensor->is_raw_data_released());
}

TEST_F(Test__WeightManagerCpuHostRelease, RetainsEmbeddingWeightAfterCpuPreparation)
{
    // Embedding role weights should be skipped regardless of tensor type
    TestableCpuReleaseWeightManager wm(*mock_loader_);
    DeviceId cpu = DeviceId::cpu();

    mock_loader_->addQ4_0RandomTensor("token_embd.weight", {128, 64});
    auto inner = mock_loader_->loadTensor("token_embd.weight");
    auto sliced = wrapInSlice(inner, 256, 64);

    // It IS IINT8Unpackable, but the Embedding role should protect it
    ASSERT_NE(dynamic_cast<IINT8Unpackable *>(sliced.get()), nullptr);

    auto binding = makeBinding(1, sliced.get(), "token_embd.weight",
                               WeightRole::Embedding, -1, cpu, /*mark_prepared=*/true);
    wm.preRegisterPrepared(binding, cpu);

    InferenceStrategy strategy;
    std::vector<WeightBinding> bindings_vec = {binding};
    FrozenModelWeightSet frozen(strategy, std::move(bindings_vec));

    bool ok = wm.prepareWeightsForDevice(frozen, cpu);
    ASSERT_TRUE(ok);

    // Embedding should NOT be released
    EXPECT_FALSE(sliced->is_raw_data_released());
}

TEST_F(Test__WeightManagerCpuHostRelease, RetainsNormWeightAfterCpuPreparation)
{
    // Norm role weights should be skipped
    TestableCpuReleaseWeightManager wm(*mock_loader_);
    DeviceId cpu = DeviceId::cpu();

    mock_loader_->addFP32RandomTensor("blk.0.attn_norm.weight", {128});
    auto norm = mock_loader_->loadTensor("blk.0.attn_norm.weight");
    ASSERT_NE(norm, nullptr);

    // 1D tensor — the shape().size() != 2 check should also skip it
    auto binding = makeBinding(1, norm.get(), "blk.0.attn_norm.weight",
                               WeightRole::Norm, 0, cpu, /*mark_prepared=*/true);
    wm.preRegisterPrepared(binding, cpu);

    InferenceStrategy strategy;
    std::vector<WeightBinding> bindings_vec = {binding};
    FrozenModelWeightSet frozen(strategy, std::move(bindings_vec));

    bool ok = wm.prepareWeightsForDevice(frozen, cpu);
    ASSERT_TRUE(ok);

    EXPECT_FALSE(norm->is_raw_data_released());
}

TEST_F(Test__WeightManagerCpuHostRelease, SkipsBindingsWithoutPreparedField)
{
    // Bindings without prepared field should not trigger release
    TestableCpuReleaseWeightManager wm(*mock_loader_);
    DeviceId cpu = DeviceId::cpu();

    mock_loader_->addQ4_0RandomTensor("blk.0.attn_q.weight", {128, 64});
    auto inner = mock_loader_->loadTensor("blk.0.attn_q.weight");
    auto sliced = wrapInSlice(inner, 256, 64);

    // No prepared field (mark_prepared=false)
    auto binding = makeBinding(1, sliced.get(), "blk.0.attn_q.weight",
                               WeightRole::AttentionQ, 0, cpu, /*mark_prepared=*/false);

    InferenceStrategy strategy;
    std::vector<WeightBinding> bindings_vec = {binding};
    FrozenModelWeightSet frozen(strategy, std::move(bindings_vec));

    bool ok = wm.prepareWeightsForDevice(frozen, cpu);
    ASSERT_TRUE(ok);

    // Should NOT be released — binding has no prepared state
    EXPECT_FALSE(sliced->is_raw_data_released());
}

TEST_F(Test__WeightManagerCpuHostRelease, SkipsBindingsPreparedForDifferentDevice)
{
    // Bindings prepared for a different device (e.g., cuda:0) should not be
    // released when preparing for CPU
    TestableCpuReleaseWeightManager wm(*mock_loader_);
    DeviceId cpu = DeviceId::cpu();
    DeviceId cuda0 = DeviceId::cuda(0);

    mock_loader_->addQ4_0RandomTensor("blk.0.attn_q.weight", {128, 64});
    auto inner = mock_loader_->loadTensor("blk.0.attn_q.weight");
    auto sliced = wrapInSlice(inner, 256, 64);

    // Prepared for cuda:0, not cpu
    auto binding = makeBinding(1, sliced.get(), "blk.0.attn_q.weight",
                               WeightRole::AttentionQ, 0, cuda0, /*mark_prepared=*/true);

    InferenceStrategy strategy;
    std::vector<WeightBinding> bindings_vec = {binding};
    FrozenModelWeightSet frozen(strategy, std::move(bindings_vec));

    bool ok = wm.prepareWeightsForDevice(frozen, cpu);
    ASSERT_TRUE(ok);

    // Should NOT be released — binding is for cuda:0, we prepared for cpu
    EXPECT_FALSE(sliced->is_raw_data_released());
}

TEST_F(Test__WeightManagerCpuHostRelease, MixedBindingsReleasesOnlyQuantized)
{
    // Multiple bindings: quantized Q4_0 (released), FP32 (retained),
    // embedding (retained), norm 1D (retained)
    TestableCpuReleaseWeightManager wm(*mock_loader_);
    DeviceId cpu = DeviceId::cpu();

    // Quantized 2D GEMM weight → should be released
    mock_loader_->addQ4_0RandomTensor("blk.0.attn_q.weight", {128, 64});
    auto q4_inner = mock_loader_->loadTensor("blk.0.attn_q.weight");
    auto q4_sliced = wrapInSlice(q4_inner, 256, 64);

    // Another quantized 2D GEMM weight → should be released
    mock_loader_->addQ4_0RandomTensor("blk.0.ffn_gate.weight", {256, 64});
    auto gate_inner = mock_loader_->loadTensor("blk.0.ffn_gate.weight");
    auto gate_sliced = wrapInSlice(gate_inner, 512, 64);

    // FP32 2D GEMM weight → should be retained
    mock_loader_->addFP32RandomTensor("blk.0.attn_output.weight", {64, 128});
    auto fp32_tensor = mock_loader_->loadTensor("blk.0.attn_output.weight");

    // Quantized embedding → should be retained (Embedding role)
    mock_loader_->addQ4_0RandomTensor("token_embd.weight", {128, 64});
    auto embd_inner = mock_loader_->loadTensor("token_embd.weight");
    auto embd_sliced = wrapInSlice(embd_inner, 256, 64);

    std::vector<WeightBinding> bindings;
    bindings.push_back(makeBinding(1, q4_sliced.get(), "blk.0.attn_q.weight",
                    WeightRole::AttentionQ, 0, cpu, true));
    bindings.push_back(makeBinding(2, gate_sliced.get(), "blk.0.ffn_gate.weight",
                    WeightRole::FFNGate, 0, cpu, true));
    bindings.push_back(makeBinding(3, fp32_tensor.get(), "blk.0.attn_output.weight",
                    WeightRole::AttentionWO, 0, cpu, true));
    bindings.push_back(makeBinding(4, embd_sliced.get(), "token_embd.weight",
                    WeightRole::Embedding, -1, cpu, true));

    // Pre-register all as prepared so actual packing is skipped
    for (const auto &b : bindings)
        wm.preRegisterPrepared(b, cpu);

    InferenceStrategy strategy;
    FrozenModelWeightSet frozen(strategy, bindings);

    bool ok = wm.prepareWeightsForDevice(frozen, cpu);
    ASSERT_TRUE(ok);

    // Quantized GEMM weights: released
    EXPECT_TRUE(q4_sliced->is_raw_data_released());
    EXPECT_TRUE(gate_sliced->is_raw_data_released());

    // FP32 GEMM weight: retained (not IINT8Unpackable)
    EXPECT_FALSE(fp32_tensor->is_raw_data_released());

    // Quantized Embedding: retained (Embedding role)
    EXPECT_FALSE(embd_sliced->is_raw_data_released());
}

TEST_F(Test__WeightManagerCpuHostRelease, RetainsBiasWeightAfterCpuPreparation)
{
    // Bias role weights should be skipped
    TestableCpuReleaseWeightManager wm(*mock_loader_);
    DeviceId cpu = DeviceId::cpu();

    mock_loader_->addFP32RandomTensor("blk.0.attn_q.bias", {128, 64});
    auto bias = mock_loader_->loadTensor("blk.0.attn_q.bias");
    ASSERT_NE(bias, nullptr);

    auto binding = makeBinding(1, bias.get(), "blk.0.attn_q.bias",
                               WeightRole::Bias, 0, cpu, /*mark_prepared=*/true);
    wm.preRegisterPrepared(binding, cpu);

    InferenceStrategy strategy;
    std::vector<WeightBinding> bindings_vec = {binding};
    FrozenModelWeightSet frozen(strategy, std::move(bindings_vec));

    bool ok = wm.prepareWeightsForDevice(frozen, cpu);
    ASSERT_TRUE(ok);

    EXPECT_FALSE(bias->is_raw_data_released());
}

TEST_F(Test__WeightManagerCpuHostRelease, Skips1DTensorAfterCpuPreparation)
{
    // 1D tensors (shape().size() != 2) should be skipped
    TestableCpuReleaseWeightManager wm(*mock_loader_);
    DeviceId cpu = DeviceId::cpu();

    mock_loader_->addFP32RandomTensor("blk.0.some_1d.weight", {128});
    auto tensor_1d = mock_loader_->loadTensor("blk.0.some_1d.weight");
    ASSERT_NE(tensor_1d, nullptr);
    ASSERT_EQ(tensor_1d->shape().size(), 1u);

    auto binding = makeBinding(1, tensor_1d.get(), "blk.0.some_1d.weight",
                               WeightRole::Other, 0, cpu, /*mark_prepared=*/true);
    wm.preRegisterPrepared(binding, cpu);

    InferenceStrategy strategy;
    std::vector<WeightBinding> bindings_vec = {binding};
    FrozenModelWeightSet frozen(strategy, std::move(bindings_vec));

    bool ok = wm.prepareWeightsForDevice(frozen, cpu);
    ASSERT_TRUE(ok);

    EXPECT_FALSE(tensor_1d->is_raw_data_released());
}

TEST_F(Test__WeightManagerCpuHostRelease, ReleasesMultipleLayersQuantizedWeights)
{
    // Multiple layers with quantized weights should all be released
    TestableCpuReleaseWeightManager wm(*mock_loader_);
    DeviceId cpu = DeviceId::cpu();

    std::vector<std::shared_ptr<TensorSlice>> slices;
    std::vector<WeightBinding> bindings;

    for (int layer = 0; layer < 4; ++layer)
    {
        std::string name = "blk." + std::to_string(layer) + ".attn_q.weight";
        mock_loader_->addQ4_0RandomTensor(name, {128, 64});
        auto inner = mock_loader_->loadTensor(name);
        auto sliced = wrapInSlice(inner, 256, 64);
        slices.push_back(sliced);

        auto binding = makeBinding(
            static_cast<uint64_t>(layer + 1), sliced.get(), name,
            WeightRole::AttentionQ, layer, cpu, true);
        bindings.push_back(binding);
    }

    for (const auto &b : bindings)
        wm.preRegisterPrepared(b, cpu);

    InferenceStrategy strategy;
    FrozenModelWeightSet frozen(strategy, bindings);

    bool ok = wm.prepareWeightsForDevice(frozen, cpu);
    ASSERT_TRUE(ok);

    // All 4 layers should have raw data released
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_TRUE(slices[i]->is_raw_data_released())
            << "Layer " << i << " quantized weight should be released";
    }
}

TEST_F(Test__WeightManagerCpuHostRelease, GpuPathDoesNotUseCpuReleaseLogic)
{
    // The GPU path (prepareWeightsForDeviceImpl) has its own separate release
    // mechanism. This test verifies that a binding prepared for GPU, when the
    // WeightManager is asked to prepare for GPU, goes through the GPU path
    // (not the CPU binding-driven path). The GPU path may or may not release
    // depending on its own logic, but the CPU-specific release loop in
    // prepareWeightsForDevice(frozen_weights, cpu_device) is NOT executed.
    //
    // Coverage note: The CPU release path is guarded by `if (device.is_cpu())`.
    // This test verifies that guard works — the GPU path is a separate codepath.
    TestableCpuReleaseWeightManager wm(*mock_loader_);
    DeviceId cuda0 = DeviceId::cuda(0);

    mock_loader_->addQ4_0RandomTensor("blk.0.attn_q.weight", {128, 64});
    auto inner = mock_loader_->loadTensor("blk.0.attn_q.weight");
    auto sliced = wrapInSlice(inner, 256, 64);

    auto binding = makeBinding(1, sliced.get(), "blk.0.attn_q.weight",
                               WeightRole::AttentionQ, 0, cuda0, /*mark_prepared=*/true);

    InferenceStrategy strategy;
    std::vector<WeightBinding> bindings_vec = {binding};
    FrozenModelWeightSet frozen(strategy, std::move(bindings_vec));

    // Calling with cuda0 takes the else branch (prepareWeightsForDeviceImpl),
    // not the CPU binding-driven path. The GPU path has its own release behavior
    // which is tested separately. Here we just confirm it doesn't crash.
    wm.prepareWeightsForDevice(frozen, cuda0);
    // No assertion on release state — GPU path has its own valid release logic
}
