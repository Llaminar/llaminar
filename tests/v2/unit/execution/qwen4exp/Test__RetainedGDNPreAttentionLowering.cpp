/**
 * @file Test__RetainedGDNPreAttentionLowering.cpp
 * @brief Host-only topology and workspace regression using the real local-stage lowerer.
 *
 * Actual production stage declarations are lowered without materializing any
 * GPU storage. The prepared-store engine stands in only for host ownership
 * checks. Native arithmetic and address admission are exercised by the
 * separate ROCm integration replay.
 */
#include <gtest/gtest.h>
#include "models/qwen4exp/Qwen4ExpRetainedGDNPreAttentionPolicy.h"
#include "execution/compute_stages/stages/qwen4exp/RetainedGDNPreAttentionStage.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "utils/PreparedWeightTestHarness.h"
#include "execution/compute_stages/IGraphOutputDemand.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "memory/BufferArena.h"
#include "utils/DebugEnv.h"
#include "tensors/Tensors.h"
#include <algorithm>

namespace llaminar2
{
namespace
{
using HC = Qwen4ExpPrimitiveStage;
using Float = Qwen4ExpFloatBinding;
using Factory = llaminar::v2::kernels::KernelFactory;

/** @brief Construct original local HC and GDN declarations with inspectable topology. */
struct Fixture
{
    static constexpr int h = 2560;
    static constexpr int rank = 256;
    const DeviceId device = DeviceId::rocm(0);
    FP32Tensor residual{{1, 4*h}, DeviceId::cpu()}, trunk{{1, 4*h}, DeviceId::cpu()},
        block{{1, h}, DeviceId::cpu()}, gate{{1, 4}, DeviceId::cpu()}, gain{{4, h}, DeviceId::cpu()},
        norm{{1, 4*h}, DeviceId::cpu()}, down{{1, rank}, DeviceId::cpu()}, activation{{1, rank}, DeviceId::cpu()},
        logits{{1, 4*h}, DeviceId::cpu()}, mixed{{1, h}, DeviceId::cpu()}, injection{{1, 4}, DeviceId::cpu()},
        qkv{{1, 2560}, DeviceId::cpu()}, z{{1, 1536}, DeviceId::cpu()},
        alpha{{1, 12}, DeviceId::cpu()}, beta{{1, 12}, DeviceId::cpu()},
        injectionWeights{{4, 4*h}, DeviceId::cpu()}, hostEngineSource{{1, 32}, DeviceId::cpu()};
    Q8_0Tensor wd{q8(rank, 4*h)}, wu{q8(4*h, rank)}, wq{q8(2560, h)}, wz{q8(1536, h)};
    FP16Tensor wa{{12, h}}, wb{{12, h}};
    test::PreparedGemmFixture hostEngine{test::makePreparedGemmFixture(&hostEngineSource, DeviceId::cpu(), "cold.engine")};
    PreparedWeightStore store{ModelContextId{54021}};
    ComputeGraph graph;

    /** @return An original-format test tensor; its zeroed physical bytes never reach a kernel. */
    static Q8_0Tensor q8(std::size_t n, std::size_t k)
    {
        return {{n, k}, std::vector<std::uint8_t>(n*k/32*sizeof(Q8_0Block))};
    }

    /** @brief Attach a host-only executable handle for cold lifetime and contract queries. */
    PreparedWeightRef declare(TensorBase &source, const char *name)
    {
        auto binding = test::makePreparedWeightTestBinding(&source, device, name, store.modelId());
        auto handle = std::make_shared<Factory::PreparedGemmHandle>();
        handle->tensor = &source;
        handle->device_id = device;
        handle->kind = Factory::GemmPreparationKind::ROCM_INT8_PACKED;
        handle->prepared_weights = std::make_shared<Factory::PreparedGemmWeights>();
        handle->prepared_weights->kind = handle->kind;
        handle->prepared_weights->owned_kernel = hostEngine.store->gemmKernelLifetime(hostEngine.ref);
        if (&source == &wu)
            handle->prepared_weights->owned_kernel = std::make_shared<cpu::native_quantised::CPUNativeQuantisedGemmKernel>(&wu);
        handle->prepared_weights->kernel = handle->prepared_weights->owned_kernel.get();
        return store.registerPreparedGemmHandle(binding, test::preparedKindForDevice(device), device, std::move(handle));
    }

    /** @brief Create the same production primitive declaration without executing it. */
    void primitive(const char *name, HC::Operation operation)
    {
        graph.addNode(name, std::make_unique<HC>(HC::Params{
            .device_id = device, .rows = {1, {}}, .operation = std::move(operation)}), device);
    }

    /** @brief Wire exact original dependencies and independent injection. */
    explicit Fixture(bool previousWrite)
    {
        const Float trunkPort(trunk, BufferId::HIDDEN_STATE);
        const Float normPort(norm, BufferId::NORMALIZED);
        const Float mixedPort(mixed, BufferId::ATTN_OUTPUT);
        const Float logitsPort(logits, BufferId::GATE_PROJ);
        if (previousWrite)
            primitive("frontier", HC::ResidualWrite{h, {residual, BufferId::RESIDUAL},
                {block, BufferId::FFN_OUTPUT}, {gate, BufferId::V_PROJ}, trunkPort});
        else
            primitive("frontier", HC::ResidualExpand{h, {block, BufferId::FFN_OUTPUT}, trunkPort});
        primitive("norm", HC::BranchRMSNorm{h, 1e-6f, trunkPort, Qwen4ExpImmutableWeightBinding(gain), normPort});
        primitive("read", HC::ResidualRead{h, normPort, logitsPort, mixedPort});
        GEMMStage::Params downParams{};
        downParams.device_id = device;
        downParams.A = &norm; downParams.B = &wd; downParams.C = &down;
        downParams.m = 1; downParams.n = rank; downParams.k = 4*h;
        downParams.a_buffer_id = BufferId::NORMALIZED; downParams.c_buffer_id = BufferId::Q_PROJ;
        downParams.prepared_store = &store; downParams.prepared_ref = declare(wd, "graph.down");
        graph.addNode("down", std::make_unique<GEMMStage>(downParams), device);

        auto upParams = downParams;
        upParams.A = &down; upParams.B = &wu; upParams.C = &logits;
        upParams.n = 4*h; upParams.k = rank;
        upParams.a_buffer_id = BufferId::Q_PROJ; upParams.c_buffer_id = BufferId::GATE_PROJ;
        upParams.prepared_ref = declare(wu, "graph.up");
        upParams.quantized_operand_precision = QuantizedActivationOperandPrecision::TwoTermQ8;
        upParams.hc_residual_down = GEMMStage::HCResidualDownPublication{&activation, BufferId::K_PROJ};
        graph.addNode("up", std::make_unique<GEMMStage>(upParams), device);

        auto injectParams = downParams;
        injectParams.n = 4; injectParams.C = &injection;
        injectParams.B = &injectionWeights;
        injectParams.prepared_ref = declare(injectionWeights, "graph.injection");
        injectParams.c_buffer_id = BufferId::V_PROJ;
        graph.addNode("injection", std::make_unique<GEMMStage>(injectParams), device);

        GDNProjectionStage::Params gdn{};
        gdn.device_id = device; gdn.input = &mixed; gdn.m = 1; gdn.k = h;
        gdn.input_buffer_id = BufferId::ATTN_OUTPUT;
        gdn.w_qkv = &wq; gdn.output_qkv = &qkv; gdn.n_qkv = 2560; gdn.output_qkv_buffer_id = BufferId::GDN_QKV;
        gdn.w_z = &wz; gdn.output_z = &z; gdn.n_z = 1536; gdn.output_z_buffer_id = BufferId::GDN_Z;
        gdn.w_a = &wa; gdn.output_a = &alpha; gdn.n_a = 12; gdn.output_a_buffer_id = BufferId::GDN_ALPHA;
        gdn.w_b = &wb; gdn.output_b = &beta; gdn.n_b = 12; gdn.output_b_buffer_id = BufferId::GDN_BETA;
        gdn.prepared_store = &store; gdn.prepared_ref_qkv = declare(wq, "graph.qkv");
        gdn.prepared_ref_z = declare(wz, "graph.z"); gdn.prepared_ref_a = declare(wa, "graph.alpha");
        gdn.prepared_ref_b = declare(wb, "graph.beta");
        graph.addNode("gdn", std::make_unique<GDNProjectionStage>(gdn), device);
        graph.addDependency("norm", "frontier");
        graph.addDependency("down", "norm");
        graph.addDependency("up", "down");
        graph.addDependency("read", "norm");
        graph.addDependency("read", "up");
        graph.addDependency("injection", "norm");
        graph.addDependency("gdn", "read");
        graph.setTerminalNode("injection");
    }
};

/** @test Two different participants in one process each retain the policy of their own placement. */
TEST(RetainedGDNPreAttentionLowering, ParticipantDefaultsAndDiagnosticOverridesRemainIndependent)
{
    const auto expected = [](const char *name, bool measured) {
        const char *value = std::getenv(name);
        return value ? std::string_view(value) == "1" : measured;
    };
    for (const bool measured : {false, true, false, true}) {
        EXPECT_EQ(qfnRetainedM1HCFFNEnabled(measured),
                  expected("LLAMINAR_QFN_RETAINED_M1_HC_FFN_CANDIDATE", measured));
        EXPECT_EQ(qfnRetainedGDNPreAttentionEnabled(measured),
                  expected("LLAMINAR_QFN_RETAINED_GDN_PREATTENTION", measured));
    }
}

/** @test The PMA quote and graph hook agree on phase, device, placement, and scalar geometry. */
TEST(RetainedGDNPreAttentionLowering, SelectsOnlyTheMeasuredOrdinaryInvocation)
{
    using Phase = ForwardExecutionPhase;
    using Role = ForwardExecutionRole;
    using Placement = RoutedExpertComputePolicy;
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
        for (const auto placement : {Placement::TensorSharded, Placement::Apportioned, Placement::Replicated})
            for (const auto phase : {Phase::Prefill, Phase::Decode})
                for (const auto role : {Role::MainInference, Role::GroupedMTPVerifier, Role::MTPCondition})
                    for (int rows : {1, 2, 4, 8, 512})
                        for (int requests : {1, 2})
                        {
                            EXPECT_FALSE(usesRetainedGDNPreAttention(false, device, placement, phase, role, rows, requests));
                            EXPECT_EQ(usesRetainedGDNPreAttention(true, device, placement, phase, role, rows, requests),
                                device.is_rocm() && placement == Placement::TensorSharded && phase == Phase::Decode &&
                                role == Role::MainInference && rows == 1 && requests == 1);
                        }
}

TEST(RetainedGDNPreAttentionLowering, PreservesOriginalFrontiersAndIndependentInjection)
{
    for (bool previousWrite : {true, false})
    {
        SCOPED_TRACE(previousWrite ? "previous write" : "published trunk");
        Fixture f(previousWrite);
        const auto *originalInjection = f.graph.getNode("injection")->stage.get();
        ASSERT_NO_THROW(lowerRetainedQwenGDNPreAttention(f.graph, "gdn", "frontier"));
        EXPECT_EQ(f.graph.size(), previousWrite ? 2u : 3u);
        EXPECT_EQ(f.graph.getNode("frontier") != nullptr, !previousWrite);
        EXPECT_EQ(f.graph.getNode("injection")->stage.get(), originalInjection);
        EXPECT_EQ(f.graph.getNode("injection")->dependencies, std::vector<std::string>{"gdn"});
        const auto &gdn = *f.graph.getNode("gdn");
        EXPECT_EQ(gdn.dependencies, previousWrite ? std::vector<std::string>{} : std::vector<std::string>{"frontier"});
        const auto contract = gdn.stage->bufferContract();
        const auto published = [&](BufferId id) {
            return std::any_of(contract.outputs.begin(), contract.outputs.end(),
                [=](const auto &binding) { return binding.id == id; });
        };
        EXPECT_TRUE(published(BufferId::NORMALIZED));
        EXPECT_TRUE(published(BufferId::GDN_BETA));
        const auto *workspace = dynamic_cast<const IWorkspaceConsumer *>(gdn.stage.get());
        ASSERT_NE(workspace, nullptr);
        const auto declared = workspace->getWorkspaceRequirements(1, 2560, 2560);
        const auto canonical = retainedGDNPreAttentionWorkspace();
        ASSERT_EQ(canonical.buffers.size(), 1u);
        const auto &bank = canonical.buffers.front();
        const auto sameBank = [&](const auto &b) {
            return b.name == bank.name && b.size_bytes == bank.size_bytes && b.alignment == bank.alignment &&
                b.content_lifetime == bank.content_lifetime;
        };
        EXPECT_EQ(std::count_if(declared.buffers.begin(), declared.buffers.end(), sameBank), 1);
        auto *demand = dynamic_cast<IGraphOutputDemand *>(gdn.stage.get());
        ASSERT_NE(demand, nullptr);
        BufferArena arena;
        const std::array<std::string, 1> diagnostics{"read_logits"};
        EXPECT_THROW(demand->sealGraphOutputDemand(f.graph, arena, diagnostics, 1), std::invalid_argument);
    }
}

TEST(RetainedGDNPreAttentionLowering, RejectsForeignFrontierWithoutMutation)
{
    Fixture f(true);
    const auto originalOrder = f.graph.getExecutionOrder();
    EXPECT_THROW(lowerRetainedQwenGDNPreAttention(f.graph, "gdn", "injection"), std::invalid_argument);
    EXPECT_EQ(f.graph.getExecutionOrder(), originalOrder);
}
}
}
