/**
 * @file Test__Qwen4ExpFP16Policy.cpp
 * @brief Real QFN source materialization, prepared GDN stage and independent router/expert operands.
 *
 * Small synthetic original Q8_0/F32 sources exercise the existing production
 * loader/store/stages without model files or accelerator discovery. This is a
 * component ownership proof, not full-model capacity or numerical qualification.
 */
#include "../../utils/CPUPreparedStoreTestAdmission.h"
#include "../../mocks/MockModelLoader.h"
#include "../../utils/CPUProjectionTestWorkspace.h"
#include "../../utils/TestTensorFactory.h"
#include "../models/qwen4exp/Qwen4ExpMetadataFixture.h"
#include "execution/compute_stages/stages/GDNProjectionStage.h"
#include "execution/compute_stages/stages/MoERoutingStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/cpu/moe/CPUMoEKernel.h"
#include "../../utils/CPURouterKernelTestAdmission.h"
#include "loaders/PreparedHalfWeights.h"
#include "loaders/PreparedWeightRepresentationContract.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/WeightManager.h"
#include "loaders/WeightPlan.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/PhysicalMemoryCapacityExhausted.h"
#include "planning/WeightMemoryEstimator.h"
#include "utils/DebugEnv.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace llaminar2::test
{
namespace
{
/** @brief Restore immutable process test selection and the short CPU team. */
struct PolicyScope final
{
    GemmConfig &setup = mutableDebugEnv().gemm;
    GDNProjectionArithmetic gdn = setup.gdn_projection_arithmetic;
    MoERouterArithmetic router = setup.moe_router_arithmetic;
    std::array<QuantizedActivationOperandPrecision, 4> residual{
        setup.qfn_hc_up_activation, setup.qfn_gdn_output_activation, setup.qfn_fa_output_activation,
        setup.qfn_gdn_qkv_activation};
    int workers = omp_get_max_threads();
    /** @brief Select only this component's serial CPU team and existing SingleQ8 controls. */
    PolicyScope()
    {
        omp_set_num_threads(1);
        setup.qfn_hc_up_activation = setup.qfn_gdn_output_activation = setup.qfn_fa_output_activation =
            setup.qfn_gdn_qkv_activation = QuantizedActivationOperandPrecision::SingleQ8;
    }
    /** @brief Restore every process setting so later tests retain their original policy. */
    ~PolicyScope()
    {
        setup.gdn_projection_arithmetic = gdn;
        setup.moe_router_arithmetic = router;
        setup.qfn_hc_up_activation = residual[0];
        setup.qfn_gdn_output_activation = residual[1];
        setup.qfn_fa_output_activation = residual[2];
        setup.qfn_gdn_qkv_activation = residual[3];
        omp_set_num_threads(workers);
    }
};
/**
 * @brief Exercise one actual frozen original-source policy combination.
 * @param gdn Existing selective-controls or full-half GDN selection.
 * @param router Existing independent native or physical-half router selection.
 */
void prove(GDNProjectionArithmetic gdn, MoERouterArithmetic router)
{
    PolicyScope policy;
    policy.setup.gdn_projection_arithmetic = gdn;
    policy.setup.moe_router_arithmetic = router;
    ASSERT_EQ(policy.setup.qfn_hc_up_activation, QuantizedActivationOperandPrecision::SingleQ8);
    ASSERT_EQ(policy.setup.qfn_gdn_output_activation, QuantizedActivationOperandPrecision::SingleQ8);
    ASSERT_EQ(policy.setup.qfn_fa_output_activation, QuantizedActivationOperandPrecision::SingleQ8);
    ASSERT_EQ(policy.setup.qfn_gdn_qkv_activation, QuantizedActivationOperandPrecision::SingleQ8);
    for (const auto name :
         {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight",
          "blk.0.ssm_conv1d.weight", "blk.3.attn_q.weight", "blk.3.attn_output.weight", "blk.0.hc_attn_up.weight"})
    {
        const auto role = inferWeightRole(name);
        EXPECT_EQ(PreparedWeightRepresentationContract::resolve(role, TensorType::Q8_0, name, "qwen4exp", gdn, router),
                  ModelPreparedWeightRepresentation::SourceNative);
        EXPECT_EQ(PreparedWeightRepresentationContract::operandPrecision(role, name, "qwen4exp", gdn, router),
                  FloatingActivationOperandPrecision::Native);
    }
    if (!hasCPUBackend())
        initCPUBackend(-1);
    constexpr int K = 64, E = 16, Top = 4;
    const std::array<int, 7> columns{128, 64, 4, 4, 64, E, 64};
    const std::array<const char *, 7> suffixes{"attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight",
                                               "ssm_beta.weight", "ssm_out.weight",   "ffn_gate_inp.weight",
                                               "ffn_up.weight"};
    std::array<std::shared_ptr<TensorBase>, 7> original;
    std::array<std::vector<std::byte>, 7> original_bytes;
    MockModelLoaderBuilder builder;
    builder.setArchitecture("qwen4exp");
    for (size_t i = 0; i < original.size(); ++i)
    {
        if (i == 2 || i == 3 || i == 5)
        {
            auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{size_t(columns[i]), K});
            for (size_t j = 0; j < tensor->numel(); ++j)
                tensor->mutable_data()[j] = float(int((j * 7 + i) % 29) - 14) / 256.f;
            original[i] = tensor;
        }
        else
            original[i] = TestTensorFactory::createQ8_0Random({size_t(columns[i]), K}, 46100 + i);
        const auto *bytes = static_cast<const std::byte *>(original[i]->raw_data());
        original_bytes[i].assign(bytes, bytes + original[i]->size_bytes());
        builder.addTensor(std::string("blk.0.") + suffixes[i], original[i]);
    }
    auto loader = builder.build();
    WeightManager manager(*loader);
    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::SingleDevice;
    strategy.model_id = ModelContextId{46100};
    strategy.devices = {DeviceId::cpu()};
    WeightPlan plan(strategy);
    for (const auto suffix : suffixes)
    {
        WeightRequirement requirement;
        requirement.canonical_name = std::string("blk.0.") + suffix;
        requirement.target_device = DeviceId::cpu();
        requirement.lookup_device = DeviceId::cpu();
        requirement.host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
        plan.add(requirement);
    }
    const auto frozen = manager.materialize(plan);
    PreparedWeightStore store(strategy.model_id);
    std::vector<const TensorBase *> preparedMatrices;
    for (size_t i = 0; i < 4; ++i) preparedMatrices.push_back(frozen.layer(0, suffixes[i]).tensor);
    test::admitCPUPreparedStoreForTest(store, preparedMatrices);
    std::array<PreparedWeightRef, 4> refs;
    for (size_t i = 0; i < original.size(); ++i)
    {
        const auto &binding = frozen.layer(0, suffixes[i]);
        const bool half = i == 2 || i == 3 || (i < 5 && gdn == GDNProjectionArithmetic::FP16) ||
                          (i == 5 && resolveMoERouterArithmetic(router, "qwen4exp") == MoERouterArithmetic::FP16);
        EXPECT_EQ(binding.tensor->native_type(), half ? TensorType::FP16 : original[i]->native_type());
        EXPECT_EQ(binding.floating_operand_precision,
                  half ? FloatingActivationOperandPrecision::FP16RNE : FloatingActivationOperandPrecision::Native);
        if (half)
        {
            ASSERT_NE(binding.tensor, original[i].get());
            std::vector<float> decoded(original[i]->numel());
            original[i]->to_fp32(decoded.data());
            const auto *words = static_cast<const uint16_t *>(binding.tensor->raw_data());
            for (size_t j = 0; j < decoded.size(); ++j)
                ASSERT_EQ(words[j], canonicalPreparedHalfBits(decoded[j]));
        }
        else
            EXPECT_EQ(binding.tensor, original[i].get());
        EXPECT_EQ(std::memcmp(original[i]->raw_data(), original_bytes[i].data(), original_bytes[i].size()), 0);
        // The production estimator consumes the exact same original role,
        // name and source format. Compare the selected representation with its
        // physical half metadata rather than copying backend packing formulas.
        for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
        {
            auto profile = ModelMemoryProfile::fromGGUF(qwen4MetadataFixture(false));
            profile.tensors = {{.name = std::string("blk.0.") + suffixes[i],
                                .native_bytes = original[i]->size_bytes(),
                                .quant_type = (i == 2 || i == 3 || i == 5) ? "F32" : "Q8_0",
                                .elements = original[i]->numel(),
                                .K = K,
                                .layer_index = 0}};
            const auto estimate = WeightMemoryEstimator::estimate(profile, device);
            EXPECT_EQ(estimate.native_bytes, original[i]->size_bytes());
            ASSERT_GT(estimate.device_bytes, 0u);
            if (half)
            {
                profile.tensors[0].quant_type = "F16";
                profile.tensors[0].native_bytes = original[i]->numel() * sizeof(uint16_t);
                EXPECT_EQ(estimate.device_bytes, WeightMemoryEstimator::estimate(profile, device).device_bytes);
            }
            PhysicalMemoryPlanBuilder exact;
            exact.add({.world_rank = 0,
                       .device = device,
                       .total_bytes = estimate.device_bytes,
                       .admission_available_bytes = estimate.device_bytes},
                      PhysicalMemoryOwner::PrimaryModelWeights, estimate.device_bytes);
            EXPECT_NO_THROW((void)PhysicalMemoryPlanAdmissionCertificate(exact.build()));
            PhysicalMemoryPlanBuilder short_plan;
            short_plan.add({.world_rank = 0,
                            .device = device,
                            .total_bytes = estimate.device_bytes - 1,
                            .admission_available_bytes = estimate.device_bytes - 1},
                           PhysicalMemoryOwner::PrimaryModelWeights, estimate.device_bytes);
            EXPECT_THROW((void)PhysicalMemoryPlanAdmissionCertificate(short_plan.build()),
                         PhysicalMemoryCapacityExhausted);
        }
        if (i < 4)
            refs[i] = store.prepareGemm(binding);
    }
    CPUDeviceContext context(DeviceId::cpu(), 1);
    FP32Tensor input({9, K});
    for (size_t i = 0; i < input.numel(); ++i)
        input.mutable_data()[i] = float(4099 + 4 * (i % 31)) / 4096.f;
    const std::vector<float> input_bytes(input.data(), input.data() + input.numel());
    std::array<std::unique_ptr<FP32Tensor>, 4> outputs;
    for (size_t i = 0; i < outputs.size(); ++i)
        outputs[i] = std::make_unique<FP32Tensor>(std::vector<size_t>{9, size_t(columns[i])});
    for (const int rows : {1, 3, 9})
    {
        GDNProjectionStage::Params p;
        p.device_id = DeviceId::cpu();
        p.prepared_store = &store;
        p.input = &input;
        p.m = rows;
        p.k = K;
        p.w_qkv = frozen.layer(0, suffixes[0]).tensor;
        p.output_qkv = outputs[0].get();
        p.n_qkv = columns[0];
        p.prepared_ref_qkv = refs[0];
        p.w_z = frozen.layer(0, suffixes[1]).tensor;
        p.output_z = outputs[1].get();
        p.n_z = columns[1];
        p.prepared_ref_z = refs[1];
        p.w_a = frozen.layer(0, suffixes[2]).tensor;
        p.output_a = outputs[2].get();
        p.n_a = columns[2];
        p.prepared_ref_a = refs[2];
        p.w_b = frozen.layer(0, suffixes[3]).tensor;
        p.output_b = outputs[3].get();
        p.n_b = columns[3];
        p.prepared_ref_b = refs[3];
        p.force_decode_equivalent_verifier_prefill = rows > 1;
        GDNProjectionStage stage(p);
        const auto requirements = stage.getWorkspaceRequirements(rows);
        // CPU floating projections declare no scratch. Keep that exact empty
        // contract instead of admitting a zero-capacity resource or spare bytes;
        // quantized projections still receive every declared invocation buffer.
        std::unique_ptr<CPUProjectionTestWorkspace> workspace_owner;
        if (!requirements.buffers.empty())
            workspace_owner = std::make_unique<CPUProjectionTestWorkspace>(requirements);
        auto *workspace = workspace_owner ? workspace_owner->get() : nullptr;
        stage.bindWorkspace(workspace);
        struct Revoke
        {
            GDNProjectionStage &stage;
            /** @brief Revoke every prepared-engine borrow before its exact admitted arena retires. */
            ~Revoke() { stage.unbindWorkspace(); }
        } revoke{stage};
        std::array<std::vector<float>, 4> expected;
        FP32Tensor serial_input({1, K});
        for (size_t i = 0; i < 4; ++i)
        {
            auto *engine = store.gemmKernel(refs[i]);
            ASSERT_NE(engine, nullptr);
            FP32Tensor serial_output({1, size_t(columns[i])});
            expected[i].resize(size_t(rows) * columns[i]);
            for (int row = 0; row < rows; ++row)
            {
                std::copy_n(input.data() + row * K, K, serial_input.mutable_data());
                ASSERT_TRUE(engine->multiply_tensor(&serial_input, &serial_output, 1, columns[i], K, true, 1.f, 0.f,
                                                    nullptr, nullptr, -1, workspace));
                std::copy_n(serial_output.data(), columns[i], expected[i].data() + row * columns[i]);
            }
            std::fill_n(outputs[i]->mutable_data(), outputs[i]->numel(), -777.f);
        }
        if (gdn == GDNProjectionArithmetic::FP16)
        {
            auto incompatible = p;
            incompatible.qkv_operand_precision = QuantizedActivationOperandPrecision::TwoTermQ8;
            incompatible.qkv_row_range = DeviceRowRange::fullyActive(rows);
            EXPECT_THROW((void)GDNProjectionStage(incompatible), std::invalid_argument);
            for (const auto &output : outputs)
                for (size_t j = 0; j < output->numel(); ++j)
                    ASSERT_EQ(output->data()[j], -777.f);
        }
        ASSERT_TRUE(stage.execute(&context));
        for (size_t i = 0; i < 4; ++i)
        {
            EXPECT_EQ(std::memcmp(outputs[i]->data(), expected[i].data(), expected[i].size() * sizeof(float)), 0);
            for (size_t j = expected[i].size(); j < outputs[i]->numel(); ++j)
                EXPECT_EQ(outputs[i]->data()[j], -777.f);
        }
    }
    FP32Tensor ids({9, Top}), weights({9, Top});
    auto owner = std::make_shared<MoERoutedPipelineKernelOwner>();
    auto cpu = std::make_unique<CPUMoEKernel>();
    admitCPURouterKernelForTest(*cpu,9,K);
    auto *actual = cpu.get();
    owner->kernel = std::move(cpu);
    MoERoutingStage::Params rp;
    rp.device_id = DeviceId::cpu();
    rp.input = &input;
    rp.gate_weights = frozen.layer(0, suffixes[5]).tensor;
    rp.seq_len = 9;
    rp.d_model = K;
    rp.num_experts = E;
    rp.top_k = Top;
    rp.output_indices = &ids;
    rp.output_weights = &weights;
    rp.routed_pipeline_kernel_owner = owner;
    MoERoutingStage routing(rp);
    ASSERT_TRUE(routing.execute(&context));
    CPUMoEKernel reference, rounded;
    admitCPURouterKernelForTest(reference,9,K);
    admitCPURouterKernelForTest(rounded,9,K);
    ASSERT_TRUE(reference.publishTransportedRouterQ8Hidden(input.data(), 9, K));
    const auto *published = actual->publishedRouterQ8Hidden(input.data(), 9, K);
    const auto *required = reference.publishedRouterQ8Hidden(input.data(), 9, K);
    ASSERT_NE(published, nullptr);
    ASSERT_NE(required, nullptr);
    EXPECT_EQ(std::memcmp(published, required, 9 * (K / 32) * sizeof(Q8_1Block)), 0);
    std::vector<float> half_input(input.numel());
    for (size_t i = 0; i < half_input.size(); ++i)
        half_input[i] = fp16_to_fp32(canonicalPreparedHalfBits(input.data()[i]));
    ASSERT_TRUE(rounded.publishTransportedRouterQ8Hidden(half_input.data(), 9, K));
    const auto *wrong = rounded.publishedRouterQ8Hidden(half_input.data(), 9, K);
    ASSERT_NE(wrong, nullptr);
    EXPECT_NE(std::memcmp(published, wrong, 9 * (K / 32) * sizeof(Q8_1Block)), 0)
        << "The witness must discriminate original expert operands from half-rounded router operands";
    EXPECT_EQ(std::memcmp(input.data(), input_bytes.data(), input.numel() * sizeof(float)), 0);
}
} // namespace

/** @test Omitted QFN routing selects real half sources/operands while expert bytes stay original. */
TEST(Qwen4ExpFP16Policy, ModelDefaultOriginalSourcesPreparedStageAndExpertOperands)
{
    prove(kDefaultGDNProjectionArithmetic, MoERouterArithmetic::ModelDefault);
}

/** @test Original Q8 wide sources plus half controls and native router retain baseline behavior. */
TEST(Qwen4ExpFP16Policy, BaselineOriginalSourcesPreparedStageAndExpertOperands)
{
    prove(GDNProjectionArithmetic::FP16MoEAlphaBeta, MoERouterArithmetic::Native);
}
/** @test Router-only half leaves native wide GDN and original expert Q8 unchanged. */
TEST(Qwen4ExpFP16Policy, RouterOnlyOriginalSourcesPreparedStageAndExpertOperands)
{
    prove(GDNProjectionArithmetic::FP16MoEAlphaBeta, MoERouterArithmetic::FP16);
}
/** @test Full half GDN derives all four operands once from their original distinct sources. */
TEST(Qwen4ExpFP16Policy, GDNOnlyOriginalSourcesPreparedStageAndExpertOperands)
{
    prove(GDNProjectionArithmetic::FP16, MoERouterArithmetic::Native);
}
/** @test Independent selections compose without changing original hidden/source authority. */
TEST(Qwen4ExpFP16Policy, BothOriginalSourcesPreparedStageAndExpertOperands)
{
    prove(GDNProjectionArithmetic::FP16, MoERouterArithmetic::FP16);
}
} // namespace llaminar2::test
