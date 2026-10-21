/**
 * @file Test__ExpertGemmRegistry.cpp
 * @brief Device-free prepared engine identity, scope and lifetime proofs.
 *
 * Mock engines exercise registration and model-context publication without
 * initializing drivers. Projection tests prevent full-expert callers from
 * borrowing or retiring fixed output slices under an ambiguous logical key.
 * Static source fixtures invoke actual frozen matrix lowering and retain real
 * disjoint CPU source roots; all declared GPU identities remain metadata only.
 */
#include <gtest/gtest.h>
#include "loaders/ExpertGemmRegistry.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/TensorKernels.h"
#include "../../utils/TestTensorFactory.h"

#include <array>
#include <thread>
#include <vector>

using namespace llaminar2;

namespace
{

    class MockGemm : public ITensorGemm
    {
    public:
        int tag = 0;
        explicit MockGemm(int t) : tag(t) {}

        bool supports_device(int /*device_idx*/) const override { return true; }

        bool multiply_tensor(
            const TensorBase * /*A*/, TensorBase * /*C*/,
            int /*m*/, int /*n*/, int /*k*/,
            bool /*transpose_B*/,
            float /*alpha*/, float /*beta*/,
            const TensorBase * /*bias*/,
            const IMPIContext * /*mpi_ctx*/,
            int /*device_idx*/,
            DeviceWorkspaceManager * /*workspace*/,
            int /*activation_row_offset*/) override
        {
            return false;
        }
    };

    using Role = ExpertGemmRegistry::WeightRole;

    std::shared_ptr<MockGemm> registerMockEngine(ExpertGemmRegistry &reg,
                                                 DeviceId device,
                                                 int layer,
                                                 int expert,
                                                 Role role,
                                                 int tag)
    {
        auto engine = std::make_shared<MockGemm>(tag);
        reg.registerEngine(device, layer, expert, role, engine.get(), engine);
        return engine;
    }

    std::vector<std::shared_ptr<MockGemm>> registerCompleteLayer(ExpertGemmRegistry &reg,
                                                                 DeviceId device,
                                                                 int layer,
                                                                 int num_experts)
    {
        std::vector<std::shared_ptr<MockGemm>> engines;
        engines.reserve(static_cast<size_t>(num_experts) * 3);

        for (int e = 0; e < num_experts; ++e)
        {
            engines.push_back(registerMockEngine(reg, device, layer, e, Role::GATE, 100 + e));
            engines.push_back(registerMockEngine(reg, device, layer, e, Role::UP, 200 + e));
            engines.push_back(registerMockEngine(reg, device, layer, e, Role::DOWN, 300 + e));
        }

        return engines;
    }

} // namespace

TEST(Test__ExpertGemmRegistry, RegisterAndRetrieve)
{
    ExpertGemmRegistry reg;
    MockGemm gemm(1);
    auto sp = std::make_shared<MockGemm>(1);

    reg.registerEngine(DeviceId::cuda(0), 0, 0, Role::GATE, &gemm, sp);
    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 0, Role::GATE), &gemm);
}

TEST(Test__ExpertGemmRegistry, RetrieveNonExistent)
{
    ExpertGemmRegistry reg;
    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 0, Role::GATE), nullptr);
}

TEST(Test__ExpertGemmRegistry, MultipleDevices)
{
    ExpertGemmRegistry reg;
    MockGemm g1(1), g2(2);
    auto sp1 = std::make_shared<MockGemm>(1);
    auto sp2 = std::make_shared<MockGemm>(2);

    reg.registerEngine(DeviceId::cuda(0), 0, 0, Role::GATE, &g1, sp1);
    reg.registerEngine(DeviceId::cuda(1), 0, 0, Role::GATE, &g2, sp2);

    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 0, Role::GATE), &g1);
    EXPECT_EQ(reg.getEngine(DeviceId::cuda(1), 0, 0, Role::GATE), &g2);
}

TEST(Test__ExpertGemmRegistry, DomainScopedEntriesOnSamePhysicalDeviceDoNotOverwrite)
{
    ExpertGemmRegistry reg;
    auto fast = std::make_shared<MockGemm>(101);
    auto warm = std::make_shared<MockGemm>(202);
    auto legacy = std::make_shared<MockGemm>(303);
    const DeviceId device = DeviceId::cuda(0);

    reg.registerEngineForDomain("cuda_fast", device, 0, 1, Role::GATE, fast.get(), fast);
    reg.registerEngineForDomain("cuda_warm", device, 0, 1, Role::GATE, warm.get(), warm);
    reg.registerEngine(device, 0, 1, Role::GATE, legacy.get(), legacy);

    EXPECT_EQ(reg.getEngineForDomain("cuda_fast", device, 0, 1, Role::GATE), fast.get());
    EXPECT_EQ(reg.getEngineForDomain("cuda_warm", device, 0, 1, Role::GATE), warm.get());
    EXPECT_EQ(reg.getEngine(device, 0, 1, Role::GATE), legacy.get());
    EXPECT_EQ(reg.countEnginesForLayerInDomain("cuda_fast", device, 0), 1u);
    EXPECT_EQ(reg.countEnginesForLayerInDomain("cuda_warm", device, 0), 1u);
    EXPECT_EQ(reg.countEnginesForLayer(device, 0), 1u);
    EXPECT_EQ(reg.size(), 3u);
}

TEST(Test__ExpertGemmRegistry, ParticipantScopedEntriesOnSameDomainDeviceDoNotOverwrite)
{
    ExpertGemmRegistry reg;
    auto rank0 = std::make_shared<MockGemm>(401);
    auto rank1 = std::make_shared<MockGemm>(402);
    auto domain_default = std::make_shared<MockGemm>(499);
    const DeviceId device = DeviceId::cpu();

    reg.registerEngineForParticipant("cpu_cold", device, 0, 0, 1, 2, Role::GATE, rank0.get(), rank0);
    reg.registerEngineForParticipant("cpu_cold", device, 1, 1, 1, 2, Role::GATE, rank1.get(), rank1);
    reg.registerEngineForDomain("cpu_cold", device, 1, 2, Role::GATE, domain_default.get(), domain_default);

    EXPECT_EQ(reg.getEngineForParticipant("cpu_cold", device, 0, 0, 1, 2, Role::GATE), rank0.get());
    EXPECT_EQ(reg.getEngineForParticipant("cpu_cold", device, 1, 1, 1, 2, Role::GATE), rank1.get());
    EXPECT_EQ(reg.getEngineForDomain("cpu_cold", device, 1, 2, Role::GATE), domain_default.get());
    EXPECT_EQ(reg.size(), 3u);
}

TEST(Test__ExpertGemmRegistry,
     ParticipantResidencyReplacementAtomicallyRebindsExactTripletsAndDomainAliases)
{
    ExpertGemmRegistry reg;
    const DeviceId device = DeviceId::cpu();
    const ExpertGemmRegistry::ParticipantLayerScope participant0{
        .domain_name = "cpu_tier",
        .device = device,
        .participant_world_rank = 0,
        .participant_index = 0,
        .layer = 3,
    };
    const ExpertGemmRegistry::ParticipantLayerScope participant1{
        .domain_name = "cpu_tier",
        .device = device,
        .participant_world_rank = 0,
        .participant_index = 1,
        .layer = 3,
    };

    auto stale = std::make_shared<MockGemm>(10);
    for (const Role role : {Role::GATE, Role::UP, Role::DOWN})
    {
        reg.registerEngineForParticipant(
            "cpu_tier", device, 0, 0, 3, 0, role, stale.get(), stale);
        reg.registerEngineForDomain(
            "cpu_tier", device, 3, 0, role, stale.get(), stale);
    }
    auto unrelated = std::make_shared<MockGemm>(99);
    reg.registerEngine(
        device, 8, 7, Role::GATE, unrelated.get(), unrelated);

    const auto triplet = [](int base)
    {
        ExpertGemmRegistry::ParticipantExpertBinding binding;
        binding.gate = std::make_shared<MockGemm>(base + 1);
        binding.up = std::make_shared<MockGemm>(base + 2);
        binding.down = std::make_shared<MockGemm>(base + 3);
        return binding;
    };
    auto expert1 = triplet(100);
    expert1.scope = participant0;
    expert1.expert = 1;
    auto expert0 = triplet(200);
    expert0.scope = participant1;
    expert0.expert = 0;
    const std::array scopes{participant0, participant1};
    const std::array bindings{expert1, expert0};

    std::string error;
    ASSERT_TRUE(reg.replaceParticipantResidency(scopes, bindings, &error))
        << error;
    EXPECT_EQ(
        reg.getEngineForParticipant(
            "cpu_tier", device, 0, 0, 3, 0, Role::GATE),
        nullptr);
    EXPECT_EQ(
        reg.getEngineForParticipant(
            "cpu_tier", device, 0, 0, 3, 1, Role::GATE),
        expert1.gate.get());
    EXPECT_EQ(
        reg.getEngineForParticipant(
            "cpu_tier", device, 0, 1, 3, 0, Role::DOWN),
        expert0.down.get());
    EXPECT_EQ(
        reg.getEngineForDomain(
            "cpu_tier", device, 3, 1, Role::UP),
        expert1.up.get());
    EXPECT_EQ(
        reg.getEngineForDomain(
            "cpu_tier", device, 3, 0, Role::DOWN),
        expert0.down.get());
    EXPECT_EQ(
        reg.getEngine(device, 8, 7, Role::GATE),
        unrelated.get())
        << "non-overlay registry scopes must not be touched";

    auto invalid = expert1;
    invalid.down.reset();
    const std::array invalid_bindings{invalid};
    EXPECT_FALSE(reg.replaceParticipantResidency(
        scopes, invalid_bindings, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(
        reg.getEngineForParticipant(
            "cpu_tier", device, 0, 0, 3, 1, Role::GATE),
        expert1.gate.get())
        << "a rejected replacement must leave the prior publication intact";
}

TEST(Test__ExpertGemmRegistry,
     ScopedLifetimeLookupRetainsOnlyTheExactRegisteredEngine)
{
    ExpertGemmRegistry reg;
    const DeviceId device = DeviceId::cpu();
    auto participant = std::make_shared<MockGemm>(450);
    reg.registerEngineForParticipant(
        "cpu_cold",
        device,
        1,
        3,
        2,
        7,
        Role::DOWN,
        participant.get(),
        participant);

    auto lifetime = reg.getEngineLifetimeForParticipant(
        "cpu_cold", device, 1, 3, 2, 7, Role::DOWN);
    ASSERT_NE(lifetime, nullptr);
    EXPECT_EQ(lifetime.get(), participant.get());
    EXPECT_EQ(
        reg.getEngineLifetimeForDomain(
            "cpu_cold", device, 2, 7, Role::DOWN),
        nullptr);

    MockGemm borrowed(451);
    auto unrelated_owner = std::make_shared<MockGemm>(452);
    reg.registerEngine(
        device,
        2,
        8,
        Role::DOWN,
        &borrowed,
        unrelated_owner);
    EXPECT_EQ(
        reg.getEngineLifetime(device, 2, 8, Role::DOWN),
        nullptr);
}

TEST(Test__ExpertGemmRegistry, AliasDomainAndParticipantFromBaseDeviceReusesEngine)
{
    ExpertGemmRegistry reg;
    const DeviceId device = DeviceId::cuda(0);
    auto base = std::make_shared<MockGemm>(515);

    reg.registerEngine(device, 2, 7, Role::UP, base.get(), base);

    EXPECT_TRUE(reg.aliasEngineForDomainFromDevice("cuda_hot", device, 2, 7, Role::UP));
    EXPECT_TRUE(reg.aliasEngineForParticipantFromDevice("cuda_hot", device, 0, 1, 2, 7, Role::UP));

    EXPECT_EQ(reg.getEngine(device, 2, 7, Role::UP), base.get());
    EXPECT_EQ(reg.getEngineForDomain("cuda_hot", device, 2, 7, Role::UP), base.get());
    EXPECT_EQ(reg.getEngineForParticipant("cuda_hot", device, 0, 1, 2, 7, Role::UP), base.get());
    EXPECT_EQ(reg.size(), 3u);
}

TEST(Test__ExpertGemmRegistry, PopulateExpertEnginesForParticipantUsesOnlyThatParticipant)
{
    ExpertGemmRegistry reg;
    std::vector<std::shared_ptr<MockGemm>> keepalive;
    const DeviceId device = DeviceId::cpu();

    for (int expert : {0, 2})
    {
        keepalive.push_back(std::make_shared<MockGemm>(100 + expert));
        reg.registerEngineForParticipant("cpu_cold", device, 0, 0, 1, expert, Role::GATE,
                                         keepalive.back().get(), keepalive.back());
        keepalive.push_back(std::make_shared<MockGemm>(200 + expert));
        reg.registerEngineForParticipant("cpu_cold", device, 0, 0, 1, expert, Role::UP,
                                         keepalive.back().get(), keepalive.back());
        keepalive.push_back(std::make_shared<MockGemm>(300 + expert));
        reg.registerEngineForParticipant("cpu_cold", device, 0, 0, 1, expert, Role::DOWN,
                                         keepalive.back().get(), keepalive.back());
    }

    auto rank1_gate = std::make_shared<MockGemm>(999);
    reg.registerEngineForParticipant("cpu_cold", device, 1, 1, 1, 0, Role::GATE,
                                     rank1_gate.get(), rank1_gate);

    std::vector<ITensorGemm *> gate, up, down;
    EXPECT_FALSE(reg.populateExpertEnginesForParticipant(
        "cpu_cold", device, 0, 0, 1, 3, gate, up, down));
    ASSERT_EQ(gate.size(), 3u);
    EXPECT_EQ(gate[0], keepalive[0].get());
    EXPECT_EQ(up[0], keepalive[1].get());
    EXPECT_EQ(down[0], keepalive[2].get());
    EXPECT_EQ(gate[1], nullptr);
    EXPECT_EQ(gate[2], keepalive[3].get());

    EXPECT_FALSE(reg.populateExpertEnginesForParticipant(
        "cpu_cold", device, 1, 1, 1, 3, gate, up, down));
    EXPECT_EQ(gate[0], rank1_gate.get());
    EXPECT_EQ(up[0], nullptr);
    EXPECT_EQ(gate[2], nullptr);
}

TEST(Test__ExpertGemmRegistry, MultipleLayers)
{
    ExpertGemmRegistry reg;
    MockGemm g1(1), g2(2);
    auto sp1 = std::make_shared<MockGemm>(1);
    auto sp2 = std::make_shared<MockGemm>(2);

    reg.registerEngine(DeviceId::cuda(0), 0, 0, Role::GATE, &g1, sp1);
    reg.registerEngine(DeviceId::cuda(0), 5, 0, Role::GATE, &g2, sp2);

    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 0, Role::GATE), &g1);
    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 5, 0, Role::GATE), &g2);
}

TEST(Test__ExpertGemmRegistry, MultipleExperts)
{
    ExpertGemmRegistry reg;
    MockGemm g0(0), g3(3), g7(7);
    auto sp0 = std::make_shared<MockGemm>(0);
    auto sp3 = std::make_shared<MockGemm>(3);
    auto sp7 = std::make_shared<MockGemm>(7);

    reg.registerEngine(DeviceId::cuda(0), 0, 0, Role::GATE, &g0, sp0);
    reg.registerEngine(DeviceId::cuda(0), 0, 3, Role::GATE, &g3, sp3);
    reg.registerEngine(DeviceId::cuda(0), 0, 7, Role::GATE, &g7, sp7);

    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 0, Role::GATE), &g0);
    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 3, Role::GATE), &g3);
    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 7, Role::GATE), &g7);
    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 1, Role::GATE), nullptr);
}

TEST(Test__ExpertGemmRegistry, AllRoles)
{
    ExpertGemmRegistry reg;
    MockGemm gGate(1), gUp(2), gDown(3);
    auto spGate = std::make_shared<MockGemm>(1);
    auto spUp = std::make_shared<MockGemm>(2);
    auto spDown = std::make_shared<MockGemm>(3);

    reg.registerEngine(DeviceId::cuda(0), 0, 0, Role::GATE, &gGate, spGate);
    reg.registerEngine(DeviceId::cuda(0), 0, 0, Role::UP, &gUp, spUp);
    reg.registerEngine(DeviceId::cuda(0), 0, 0, Role::DOWN, &gDown, spDown);

    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 0, Role::GATE), &gGate);
    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 0, Role::UP), &gUp);
    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 0, Role::DOWN), &gDown);
}

TEST(Test__ExpertGemmRegistry, PopulateExpertEngines)
{
    ExpertGemmRegistry reg;
    MockGemm gGate0(10), gUp0(20), gDown0(30);
    MockGemm gGate2(12), gUp2(22), gDown2(32);
    auto spGate0 = std::make_shared<MockGemm>(10);
    auto spUp0 = std::make_shared<MockGemm>(20);
    auto spDown0 = std::make_shared<MockGemm>(30);
    auto spGate2 = std::make_shared<MockGemm>(12);
    auto spUp2 = std::make_shared<MockGemm>(22);
    auto spDown2 = std::make_shared<MockGemm>(32);

    // Register experts 0 and 2 (leaving expert 1 as a gap)
    reg.registerEngine(DeviceId::cuda(0), 3, 0, Role::GATE, &gGate0, spGate0);
    reg.registerEngine(DeviceId::cuda(0), 3, 0, Role::UP, &gUp0, spUp0);
    reg.registerEngine(DeviceId::cuda(0), 3, 0, Role::DOWN, &gDown0, spDown0);
    reg.registerEngine(DeviceId::cuda(0), 3, 2, Role::GATE, &gGate2, spGate2);
    reg.registerEngine(DeviceId::cuda(0), 3, 2, Role::UP, &gUp2, spUp2);
    reg.registerEngine(DeviceId::cuda(0), 3, 2, Role::DOWN, &gDown2, spDown2);

    std::vector<ITensorGemm *> gate, up, down;
    bool found = reg.populateExpertEngines(DeviceId::cuda(0), 3, 4, gate, up, down);

    EXPECT_FALSE(found);
    ASSERT_EQ(gate.size(), 4u);
    ASSERT_EQ(up.size(), 4u);
    ASSERT_EQ(down.size(), 4u);

    EXPECT_EQ(gate[0], &gGate0);
    EXPECT_EQ(up[0], &gUp0);
    EXPECT_EQ(down[0], &gDown0);

    EXPECT_EQ(gate[1], nullptr);
    EXPECT_EQ(up[1], nullptr);
    EXPECT_EQ(down[1], nullptr);

    EXPECT_EQ(gate[2], &gGate2);
    EXPECT_EQ(up[2], &gUp2);
    EXPECT_EQ(down[2], &gDown2);

    EXPECT_EQ(gate[3], nullptr);
    EXPECT_EQ(up[3], nullptr);
    EXPECT_EQ(down[3], nullptr);
}

TEST(Test__ExpertGemmRegistry, PopulateExpertEnginesForDomainUsesOnlyThatDomain)
{
    ExpertGemmRegistry reg;
    std::vector<std::shared_ptr<MockGemm>> keepalive;

    for (int expert : {0, 1})
    {
        keepalive.push_back(std::make_shared<MockGemm>(10 + expert));
        reg.registerEngineForDomain("cuda_fast", DeviceId::cuda(0), 2, expert, Role::GATE,
                                    keepalive.back().get(), keepalive.back());
        keepalive.push_back(std::make_shared<MockGemm>(20 + expert));
        reg.registerEngineForDomain("cuda_fast", DeviceId::cuda(0), 2, expert, Role::UP,
                                    keepalive.back().get(), keepalive.back());
        keepalive.push_back(std::make_shared<MockGemm>(30 + expert));
        reg.registerEngineForDomain("cuda_fast", DeviceId::cuda(0), 2, expert, Role::DOWN,
                                    keepalive.back().get(), keepalive.back());
    }

    auto warm_gate = std::make_shared<MockGemm>(999);
    reg.registerEngineForDomain("cuda_warm", DeviceId::cuda(0), 2, 0, Role::GATE,
                                warm_gate.get(), warm_gate);

    std::vector<ITensorGemm *> gate, up, down;
    EXPECT_TRUE(reg.populateExpertEnginesForDomain("cuda_fast", DeviceId::cuda(0), 2, 2, gate, up, down));
    EXPECT_EQ(gate[0], keepalive[0].get());
    EXPECT_EQ(up[0], keepalive[1].get());
    EXPECT_EQ(down[0], keepalive[2].get());
    EXPECT_EQ(gate[1], keepalive[3].get());
    EXPECT_EQ(up[1], keepalive[4].get());
    EXPECT_EQ(down[1], keepalive[5].get());

    EXPECT_FALSE(reg.populateExpertEnginesForDomain("cuda_warm", DeviceId::cuda(0), 2, 2, gate, up, down));
    EXPECT_EQ(gate[0], warm_gate.get());
    EXPECT_EQ(up[0], nullptr);
    EXPECT_EQ(down[0], nullptr);
}

TEST(Test__ExpertGemmRegistry, CompleteLayerReturnsTrue)
{
    ExpertGemmRegistry reg;
    auto keepalive = registerCompleteLayer(reg, DeviceId::cuda(0), 2, 4);

    EXPECT_TRUE(reg.hasCompleteRole(DeviceId::cuda(0), 2, 4, Role::GATE));
    EXPECT_TRUE(reg.hasCompleteRole(DeviceId::cuda(0), 2, 4, Role::UP));
    EXPECT_TRUE(reg.hasCompleteRole(DeviceId::cuda(0), 2, 4, Role::DOWN));
    EXPECT_TRUE(reg.hasCompleteLayer(DeviceId::cuda(0), 2, 4));
}

TEST(Test__ExpertGemmRegistry, MissingRoleMakesLayerIncomplete)
{
    ExpertGemmRegistry reg;
    std::vector<std::shared_ptr<MockGemm>> keepalive;
    keepalive.reserve(5);

    keepalive.push_back(registerMockEngine(reg, DeviceId::cuda(0), 1, 0, Role::GATE, 10));
    keepalive.push_back(registerMockEngine(reg, DeviceId::cuda(0), 1, 0, Role::UP, 20));
    keepalive.push_back(registerMockEngine(reg, DeviceId::cuda(0), 1, 0, Role::DOWN, 30));
    keepalive.push_back(registerMockEngine(reg, DeviceId::cuda(0), 1, 1, Role::GATE, 11));
    keepalive.push_back(registerMockEngine(reg, DeviceId::cuda(0), 1, 1, Role::DOWN, 31));

    EXPECT_TRUE(reg.hasCompleteRole(DeviceId::cuda(0), 1, 2, Role::GATE));
    EXPECT_FALSE(reg.hasCompleteRole(DeviceId::cuda(0), 1, 2, Role::UP));
    EXPECT_TRUE(reg.hasCompleteRole(DeviceId::cuda(0), 1, 2, Role::DOWN));
    EXPECT_FALSE(reg.hasCompleteLayer(DeviceId::cuda(0), 1, 2));
}

TEST(Test__ExpertGemmRegistry, MissingExpertMakesRoleIncomplete)
{
    ExpertGemmRegistry reg;
    auto keepalive = registerCompleteLayer(reg, DeviceId::cuda(0), 4, 3);
    reg.removeEngine(DeviceId::cuda(0), 4, 1, Role::GATE);

    EXPECT_FALSE(reg.hasCompleteRole(DeviceId::cuda(0), 4, 3, Role::GATE));
    EXPECT_TRUE(reg.hasCompleteRole(DeviceId::cuda(0), 4, 3, Role::UP));
    EXPECT_TRUE(reg.hasCompleteRole(DeviceId::cuda(0), 4, 3, Role::DOWN));
    EXPECT_FALSE(reg.hasCompleteLayer(DeviceId::cuda(0), 4, 3));
}

TEST(Test__ExpertGemmRegistry, SubsetCompletenessAndCountsAreDeviceScoped)
{
    ExpertGemmRegistry reg;
    std::vector<std::shared_ptr<MockGemm>> keepalive;

    for (int expert : {0, 2})
    {
        keepalive.push_back(registerMockEngine(reg, DeviceId::cuda(0), 3, expert, Role::GATE, 10 + expert));
        keepalive.push_back(registerMockEngine(reg, DeviceId::cuda(0), 3, expert, Role::UP, 20 + expert));
        keepalive.push_back(registerMockEngine(reg, DeviceId::cuda(0), 3, expert, Role::DOWN, 30 + expert));
    }
    keepalive.push_back(registerMockEngine(reg, DeviceId::rocm(0), 3, 1, Role::GATE, 101));
    keepalive.push_back(registerMockEngine(reg, DeviceId::rocm(0), 3, 1, Role::UP, 201));
    keepalive.push_back(registerMockEngine(reg, DeviceId::rocm(0), 3, 1, Role::DOWN, 301));

    EXPECT_TRUE(reg.hasCompleteRoleForExperts(DeviceId::cuda(0), 3, {0, 2}, Role::GATE));
    EXPECT_FALSE(reg.hasCompleteRoleForExperts(DeviceId::cuda(0), 3, {0, 1, 2}, Role::GATE));
    EXPECT_EQ(reg.completeExpertsForLayer(DeviceId::cuda(0), 3, 4), (std::vector<int>{0, 2}));
    EXPECT_EQ(reg.countCompleteExpertsForLayer(DeviceId::cuda(0), 3, 4), 2u);
    EXPECT_EQ(reg.countEnginesForLayer(DeviceId::cuda(0), 3), 6u);
    EXPECT_EQ(reg.countEnginesForDevice(DeviceId::rocm(0)), 3u);
}

TEST(Test__ExpertGemmRegistry, ModelLifetimeAccountingIncludesOverlayScopes)
{
    ExpertGemmRegistry reg;
    auto domain = std::make_shared<MockGemm>(10);
    auto participant = std::make_shared<MockGemm>(20);
    auto sibling = std::make_shared<MockGemm>(30);

    reg.registerEngineForDomain(
        "secondary",
        DeviceId::rocm(0),
        2,
        7,
        Role::GATE,
        domain.get(),
        domain);
    reg.registerEngineForParticipant(
        "secondary",
        DeviceId::rocm(0),
        1,
        3,
        2,
        7,
        Role::UP,
        participant.get(),
        participant);
    reg.registerEngineForDomain(
        "secondary",
        DeviceId::rocm(1),
        2,
        7,
        Role::DOWN,
        sibling.get(),
        sibling);

    EXPECT_EQ(reg.countEnginesForDevice(DeviceId::rocm(0)), 0u)
        << "Legacy unscoped lookup must remain scope-specific";
    EXPECT_EQ(
        reg.countOwnedEnginesForDeviceAcrossScopes(DeviceId::rocm(0)),
        2u);
    EXPECT_EQ(
        reg.countOwnedEnginesForDeviceAcrossScopes(DeviceId::rocm(1)),
        1u);
    EXPECT_EQ(
        reg.countOwnedEnginesForDeviceAcrossScopes(DeviceId::cuda(0)),
        0u);
}

TEST(Test__ExpertGemmRegistry, RemovalAndReplacementUpdateCompleteness)
{
    ExpertGemmRegistry reg;
    auto keepalive = registerCompleteLayer(reg, DeviceId::cuda(0), 7, 2);
    ASSERT_TRUE(reg.hasCompleteLayer(DeviceId::cuda(0), 7, 2));

    EXPECT_TRUE(reg.removeEngine(DeviceId::cuda(0), 7, 1, Role::DOWN));
    EXPECT_FALSE(reg.hasCompleteLayer(DeviceId::cuda(0), 7, 2));

    auto replacement = std::make_shared<MockGemm>(999);
    reg.replaceEngine(DeviceId::cuda(0), 7, 1, Role::DOWN, replacement.get(), replacement);
    EXPECT_TRUE(reg.hasCompleteLayer(DeviceId::cuda(0), 7, 2));
}

TEST(Test__ExpertGemmRegistry, PopulateTrueGuaranteesNoNullEntries)
{
    ExpertGemmRegistry reg;
    auto keepalive = registerCompleteLayer(reg, DeviceId::cuda(0), 9, 3);

    std::vector<ITensorGemm *> gate, up, down;
    EXPECT_TRUE(reg.populateExpertEngines(DeviceId::cuda(0), 9, 3, gate, up, down));

    ASSERT_EQ(gate.size(), 3u);
    ASSERT_EQ(up.size(), 3u);
    ASSERT_EQ(down.size(), 3u);
    for (int e = 0; e < 3; ++e)
    {
        EXPECT_NE(gate[e], nullptr);
        EXPECT_NE(up[e], nullptr);
        EXPECT_NE(down[e], nullptr);
    }
}

TEST(Test__ExpertGemmRegistry, PopulateExpertEnginesEmpty)
{
    ExpertGemmRegistry reg;
    std::vector<ITensorGemm *> gate, up, down;
    bool found = reg.populateExpertEngines(DeviceId::cuda(0), 0, 3, gate, up, down);

    EXPECT_FALSE(found);
    ASSERT_EQ(gate.size(), 3u);
    ASSERT_EQ(up.size(), 3u);
    ASSERT_EQ(down.size(), 3u);
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(gate[i], nullptr);
        EXPECT_EQ(up[i], nullptr);
        EXPECT_EQ(down[i], nullptr);
    }
}

TEST(Test__ExpertGemmRegistry, ReplaceEngine)
{
    ExpertGemmRegistry reg;
    MockGemm g1(1), g2(2);
    auto sp1 = std::make_shared<MockGemm>(1);
    auto sp2 = std::make_shared<MockGemm>(2);

    reg.registerEngine(DeviceId::cuda(0), 0, 0, Role::GATE, &g1, sp1);
    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 0, Role::GATE), &g1);
    EXPECT_EQ(reg.size(), 1u);

    reg.replaceEngine(DeviceId::cuda(0), 0, 0, Role::GATE, &g2, sp2);
    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 0, Role::GATE), &g2);
    EXPECT_EQ(reg.size(), 1u);
}

TEST(Test__ExpertGemmRegistry, RemoveEngine)
{
    ExpertGemmRegistry reg;
    MockGemm g1(1);
    auto sp1 = std::make_shared<MockGemm>(1);

    reg.registerEngine(DeviceId::cuda(0), 0, 0, Role::GATE, &g1, sp1);
    EXPECT_EQ(reg.size(), 1u);

    bool removed = reg.removeEngine(DeviceId::cuda(0), 0, 0, Role::GATE);
    EXPECT_TRUE(removed);
    EXPECT_EQ(reg.size(), 0u);
    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 0, Role::GATE), nullptr);
}

TEST(Test__ExpertGemmRegistry, RemoveNonExistent)
{
    ExpertGemmRegistry reg;
    bool removed = reg.removeEngine(DeviceId::cuda(0), 0, 0, Role::GATE);
    EXPECT_FALSE(removed);
}

TEST(Test__ExpertGemmRegistry, HasEnginesForLayer)
{
    ExpertGemmRegistry reg;
    MockGemm g1(1);
    auto sp1 = std::make_shared<MockGemm>(1);

    reg.registerEngine(DeviceId::cuda(0), 5, 0, Role::GATE, &g1, sp1);

    EXPECT_TRUE(reg.hasEnginesForLayer(DeviceId::cuda(0), 5));
    EXPECT_FALSE(reg.hasEnginesForLayer(DeviceId::cuda(0), 6));
    EXPECT_FALSE(reg.hasEnginesForLayer(DeviceId::cuda(1), 5));
}

TEST(Test__ExpertGemmRegistry, Clear)
{
    ExpertGemmRegistry reg;
    MockGemm g1(1), g2(2), g3(3);
    auto sp1 = std::make_shared<MockGemm>(1);
    auto sp2 = std::make_shared<MockGemm>(2);
    auto sp3 = std::make_shared<MockGemm>(3);

    reg.registerEngine(DeviceId::cuda(0), 0, 0, Role::GATE, &g1, sp1);
    reg.registerEngine(DeviceId::cuda(0), 0, 1, Role::UP, &g2, sp2);
    reg.registerEngine(DeviceId::cuda(1), 1, 0, Role::DOWN, &g3, sp3);
    EXPECT_EQ(reg.size(), 3u);

    reg.clear();
    EXPECT_EQ(reg.size(), 0u);
    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 0, Role::GATE), nullptr);
    EXPECT_EQ(reg.getEngine(DeviceId::cuda(0), 0, 1, Role::UP), nullptr);
    EXPECT_EQ(reg.getEngine(DeviceId::cuda(1), 1, 0, Role::DOWN), nullptr);
}

TEST(Test__ExpertGemmRegistry, Size)
{
    ExpertGemmRegistry reg;
    MockGemm g1(1), g2(2), g3(3);
    auto sp1 = std::make_shared<MockGemm>(1);
    auto sp2 = std::make_shared<MockGemm>(2);
    auto sp3 = std::make_shared<MockGemm>(3);

    EXPECT_EQ(reg.size(), 0u);

    reg.registerEngine(DeviceId::cuda(0), 0, 0, Role::GATE, &g1, sp1);
    EXPECT_EQ(reg.size(), 1u);

    reg.registerEngine(DeviceId::cuda(0), 0, 0, Role::UP, &g2, sp2);
    EXPECT_EQ(reg.size(), 2u);

    reg.registerEngine(DeviceId::cuda(0), 0, 1, Role::GATE, &g3, sp3);
    EXPECT_EQ(reg.size(), 3u);

    reg.removeEngine(DeviceId::cuda(0), 0, 0, Role::GATE);
    EXPECT_EQ(reg.size(), 2u);
}

TEST(Test__ExpertGemmRegistry, ThreadSafety_ConcurrentReads)
{
    ExpertGemmRegistry reg;
    constexpr int NUM_EXPERTS = 8;
    constexpr int NUM_THREADS = 8;
    constexpr int READS_PER_THREAD = 1000;

    std::vector<MockGemm> gemms;
    std::vector<std::shared_ptr<MockGemm>> sps;
    gemms.reserve(NUM_EXPERTS);
    sps.reserve(NUM_EXPERTS);

    for (int e = 0; e < NUM_EXPERTS; ++e)
    {
        gemms.emplace_back(e);
        sps.push_back(std::make_shared<MockGemm>(e));
    }

    for (int e = 0; e < NUM_EXPERTS; ++e)
    {
        reg.registerEngine(DeviceId::cuda(0), 0, e, Role::GATE, &gemms[e], sps[e]);
    }

    std::vector<std::thread> threads;
    std::atomic<int> errors{0};

    for (int t = 0; t < NUM_THREADS; ++t)
    {
        threads.emplace_back([&, t]()
                             {
            for (int i = 0; i < READS_PER_THREAD; ++i)
            {
                int expert = (t + i) % NUM_EXPERTS;
                auto* result = reg.getEngine(DeviceId::cuda(0), 0, expert, Role::GATE);
                if (result != &gemms[expert])
                    errors.fetch_add(1, std::memory_order_relaxed);
            } });
    }

    for (auto &th : threads)
        th.join();

    EXPECT_EQ(errors.load(), 0);
}

TEST(ExpertProjectionPreparedIdentity, RejectsWholeExpertAndNeighbourSliceAdoption)
{
    for (const auto device : {DeviceId::cuda(3), DeviceId::rocm(3)})
    {
        ExpertGemmRegistry registry;
        const auto identity = MoEExpertProjectionOwnership::gateUpOwnedDownColumns({4, 512, 256}, 1, 2);
        const auto neighbour = MoEExpertProjectionOwnership::gateUpOwnedDownColumns({4, 512, 256}, 0, 2);
        for (const auto role : {Role::GATE, Role::UP, Role::DOWN})
        {
            auto engine = std::make_shared<MockGemm>(1);
            registry.registerEngineForParticipant("compute", device, 2, 1, 0, 0, role, engine.get(), engine, identity);
            registry.registerEngineForDomain("compute", device, 0, 0, role, engine.get(), engine, identity);
            EXPECT_EQ(registry.getEngineForParticipant("compute", device, 2, 1, 0, 0, role, identity), engine.get());
            EXPECT_EQ(registry.getEngineLifetimeForDomain("compute", device, 0, 0, role, identity), engine);
            EXPECT_THROW(registry.getEngineForParticipant("compute", device, 2, 1, 0, 0, role), std::logic_error);
            EXPECT_THROW(registry.getEngineForDomain("compute", device, 0, 0, role, neighbour), std::logic_error);
            EXPECT_THROW((void)registry.getEngineLifetimeForParticipant("compute", device, 2, 1, 0, 0, role), std::logic_error);
            EXPECT_THROW(registry.registerEngineForDomain("compute", device, 0, 0, role, engine.get(), engine), std::logic_error);
            EXPECT_FALSE(registry.hasCompleteRoleForExpertsInDomain("compute", device, 0, {0}, role));
            EXPECT_THROW(registry.replaceEngineForDomain("compute", device, 0, 0, role, engine.get(), engine), std::logic_error);
            EXPECT_THROW(registry.removeEngineForDomain("compute", device, 0, 0, role), std::logic_error);

            auto legacy = std::make_shared<MockGemm>(2);
            registry.registerEngine(device, 0, 0, role, legacy.get(), legacy);
            EXPECT_THROW(registry.aliasEngineForDomainFromDevice("compute", device, 0, 0, role), std::logic_error);
            EXPECT_EQ(registry.getEngineForDomain("compute", device, 0, 0, role, identity), engine.get());
        }
        EXPECT_FALSE(registry.hasCompleteLayerInDomain("compute", device, 0, 1));
        EXPECT_TRUE(registry.completeExpertsForLayerInDomain("compute", device, 0, 1).empty());
        std::vector<ITensorGemm *> gates, ups, downs;
        EXPECT_THROW(registry.populateExpertEnginesForParticipant("compute", device, 2, 1, 0, 1, gates, ups, downs), std::logic_error);
        const ExpertGemmRegistry::ParticipantLayerScope scope{"compute", device, 2, 1, 0};
        std::string error;
        EXPECT_FALSE(registry.replaceParticipantResidency(std::array{scope}, {}, &error));
        EXPECT_NE(error.find("fixed projection"), std::string::npos);
        EXPECT_NE(registry.getEngineForParticipant("compute", device, 2, 1, 0, 0, Role::DOWN, identity), nullptr);
        auto other = std::make_shared<MockGemm>(3);
        EXPECT_THROW(registry.registerEngineForParticipant("compute", device, 2, 0, 0, 0,
            Role::DOWN, other.get(), other, identity), std::invalid_argument);
    }
}


namespace
{
    /** @brief Geometry-aware mock: no device initialization, allocation or arithmetic. */
    class StaticSourceGemm final : public MockGemm
    {
    public:
        /** @brief Preserve actual local matrix dimensions for registration validation. */
        StaticSourceGemm(int n, int k, const void *data) : MockGemm(55), n_(n), k_(k), data_(data) {}
        int get_n() const override { return n_; }
        int get_k() const override { return k_; }
        /** @brief Real CPU-backed metadata descriptor; no fabricated device address or device operation. */
        bool exportContiguousFloatingPointWeights(ContiguousFloatingPointWeightDescriptor &out) const override
        {
            out = {data_, TensorType::FP32, n_, k_, size_t(n_) * size_t(k_) * sizeof(float)};
            return out.valid();
        }
    private:
        int n_;
        int k_;
        const void *data_;
    };

    /** @brief Build original compact CPU tensors, then invoke actual frozen native-view lowering. */
    std::vector<WeightBinding> staticSources(DeviceId target, int experts = 512,
        int participant = 0, std::vector<MoEExpertProjectionOwnership::IntermediateInterval> intervals =
            {{0,4},{4,4},{8,4},{12,8}}, bool native_q8 = false)
    {
        const auto ownership = MoEExpertProjectionOwnership::withinExpertIntermediateSlices(
            {experts,native_q8 ? 32 : 8,native_q8 ? 128 : 20}, participant, std::move(intervals));
        const InferenceStrategy strategy{.model_id=ModelContextId{55}};
        ModelWeightSetBuilder builder(strategy);
        for (const auto role : {llaminar2::WeightRole::MoEExpertGate,
                               llaminar2::WeightRole::MoEExpertUp, llaminar2::WeightRole::MoEExpertDown})
        {
            const auto projection = ownership.projection(role);
            const std::string suffix = role == llaminar2::WeightRole::MoEExpertGate ? "gate"
                : role == llaminar2::WeightRole::MoEExpertUp ? "up" : "down";
            WeightBinding parent;
            parent.binding_id = 100 + static_cast<unsigned>(role);
            parent.identity = makeSourceWeightIdentity("blk.7.ffn_" + suffix + "_exps.weight", strategy.model_id, parent.binding_id);
            parent.identity.role = role;
            parent.identity.derivation = WeightDerivationKind::ExpertSlice;
            parent.identity.layer = 7;
            parent.identity.overlay_domain = "static_original";
            parent.identity.overlay_participant_world_rank = 0;
            parent.identity.overlay_participant_index = participant;
            parent.identity.tp_rank_or_device_index = participant;
            parent.projection_ownership = ownership;
            const std::vector<size_t> shape{size_t(projection.columns), size_t(projection.rows), size_t(experts)};
            if (native_q8)
            {
                const size_t blocks = size_t(projection.columns / 32) * size_t(projection.rows) * size_t(experts);
                parent.tensor_owner = std::make_shared<Q8_0Tensor>(shape, std::vector<uint8_t>(blocks * sizeof(Q8_0Block), 0));
            }
            else
            {
                parent.tensor_owner = llaminar2::test::TestTensorFactory::createFP32Zeros(shape);
            }
            parent.tensor = parent.tensor_owner.get();
            parent.slice.source_rows = size_t(projection.source_columns);
            parent.slice.source_cols = size_t(projection.source_rows);
            parent.slice.row_start = size_t(projection.first_column);
            parent.slice.row_count = size_t(projection.columns);
            parent.slice.col_start = size_t(projection.first_row);
            parent.slice.col_count = size_t(projection.rows);
            parent.slice.expert_count = size_t(experts);
            parent.slice.inner_is_presliced = true;
            parent.residency.home_device = target;
            parent.residency.resident_device = target;
            parent.residency.host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
            builder.addBinding(std::move(parent));
        }
        const FrozenModelWeightSet original(strategy, builder.freezeBindings(), PhysicalMemoryOwner::RoutedExpertWeights);
        const auto projected = original.projectedExpertMatrices();
        return projected.bindings();
    }

    /** @brief Publish geometry-aware mock handles through the actual static registry API. */
    void registerStaticSources(ExpertGemmRegistry &registry, std::span<const WeightBinding> sources)
    {
        for (const auto &source : sources)
        {
            auto engine = std::make_shared<StaticSourceGemm>(int(source.tensor->shape()[0]), int(source.tensor->shape()[1]), source.tensor->raw_data());
            registry.registerStaticProjection(source, engine.get(), engine);
        }
    }
}

TEST(StaticProjectionPreparedSource, AllOriginalExpertsAndIndependentOwnerLifetime)
{
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        ExpertGemmRegistry registry;
        auto sources = staticSources(device);
        const auto scope = ExpertGemmRegistry::staticProjectionScope(sources.front());
        const auto projection = *sources.front().projection_ownership;
        EXPECT_FALSE(registry.acquireStaticProjectionLayer(scope, projection, sources));
        auto unscoped = std::make_shared<StaticSourceGemm>(4,8,sources.front().tensor->raw_data());
        EXPECT_THROW(registry.registerEngineForParticipant(scope.domain_name, device, 0, 0, 7, 0, Role::GATE,
            unscoped.get(), unscoped, projection), std::invalid_argument);
        EXPECT_EQ(registry.size(), 0u);
        registerStaticSources(registry, sources);
        EXPECT_EQ(registry.size(), 512u * 3);
        EXPECT_FALSE(registry.hasCompleteLayer(device, 7, 512));
        EXPECT_EQ(registry.getEngine(device, 7, 511, Role::DOWN), nullptr);
        EXPECT_EQ(registry.getEngineForDomain(scope.domain_name, device, 7, 511, Role::DOWN, projection), nullptr);
        EXPECT_THROW((void)registry.getEngineForParticipant(scope.domain_name, device, 0, 0, 7, 511, Role::DOWN, projection), std::logic_error);
        auto acquired = registry.acquireStaticProjectionLayer(scope, projection, sources);
        ASSERT_TRUE(acquired);
        ASSERT_EQ(acquired->experts.size(), 512u);
        EXPECT_TRUE(acquired->experts[511].complete());
        EXPECT_EQ(acquired->experts[511].gate->get_n(), 4);
        EXPECT_EQ(acquired->experts[511].gate->get_k(), 8);
        EXPECT_EQ(acquired->experts[511].down->get_n(), 8);
        EXPECT_EQ(acquired->experts[511].down->get_k(), 4);
        std::weak_ptr<ITensorGemm> engine = acquired->experts[511].down;
        std::weak_ptr<TensorBase> original = sources.front().projection_source_owner;
        const auto other_device = device.is_cuda() ? DeviceId::cuda(1) : DeviceId::rocm(1);
        auto other_sources = staticSources(other_device);
        const auto other_scope = ExpertGemmRegistry::staticProjectionScope(other_sources.front());
        registerStaticSources(registry, other_sources);
        auto other = registry.acquireStaticProjectionLayer(other_scope, projection, other_sources);
        ASSERT_TRUE(other);
        std::weak_ptr<ITensorGemm> other_engine = other->experts[511].down;
        std::weak_ptr<TensorBase> other_original = other_sources.front().projection_source_owner;
        EXPECT_NE(other->experts[511].down.get(), acquired->experts[511].down.get());
        EXPECT_NE(other_original.lock().get(), original.lock().get());
        sources.clear();
        other_sources.clear();
        registry.clear();
        EXPECT_FALSE(engine.expired());
        EXPECT_FALSE(original.expired());
        EXPECT_FALSE(other_engine.expired());
        acquired.reset();
        EXPECT_TRUE(engine.expired());
        EXPECT_TRUE(original.expired());
        EXPECT_FALSE(other_original.expired());
        EXPECT_FALSE(other_engine.expired());
        other.reset();
        EXPECT_TRUE(other_original.expired());
        EXPECT_TRUE(other_engine.expired());
    }
}

TEST(StaticProjectionPreparedSource, RejectsForeignScopeAndEqualLocalDifferentGlobalPartition)
{
    ExpertGemmRegistry registry;
    auto sources = staticSources(DeviceId::rocm(1), 3);
    const auto scope = ExpertGemmRegistry::staticProjectionScope(sources.front());
    const auto original = *sources.front().projection_ownership;
    registerStaticSources(registry, sources);
    const auto other = MoEExpertProjectionOwnership::withinExpertIntermediateSlices(
        original.geometry(), 0, {{0,4},{4,2},{6,6},{12,8}});
    EXPECT_EQ(original.projection(llaminar2::WeightRole::MoEExpertDown).columns,
              other.projection(llaminar2::WeightRole::MoEExpertDown).columns);
    auto foreign = sources;
    for (auto &source : foreign) source.projection_ownership = other;
    EXPECT_THROW((void)registry.acquireStaticProjectionLayer(scope, other, foreign), std::logic_error);
    for (int mutation=0; mutation<5; ++mutation)
    {
        foreign = sources;
        for (auto &source : foreign)
        {
            if (mutation==0) source.identity.overlay_domain = "foreign";
            else if (mutation==1) source.identity.overlay_participant_world_rank = 1;
            else if (mutation==2) source.identity.overlay_participant_index = 1;
            else if (mutation==3) source.identity.model_id.value = 56;
            else
            {
                source.residency.home_device = DeviceId::rocm(2);
                source.residency.resident_device = DeviceId::rocm(2);
                source.prepared->device = DeviceId::rocm(2);
            }
        }
        if (mutation==3)
        {
            for (auto &source : foreign) source.prepared->model_id.value = 56;
            EXPECT_THROW((void)registry.acquireStaticProjectionLayer(scope, original, foreign), std::logic_error);
        }
        else
        {
            EXPECT_THROW((void)registry.acquireStaticProjectionLayer(scope, original, foreign), std::invalid_argument);
        }
    }
    auto absent = scope;
    absent.domain_name = "not_this_domain";
    foreign = sources;
    for (auto &source : foreign) source.identity.overlay_domain = absent.domain_name;
    EXPECT_FALSE(registry.acquireStaticProjectionLayer(absent, original, foreign));
}

TEST(StaticProjectionPreparedSource, RejectsDuplicateMissingHighExpertAndPartialPublication)
{
    ExpertGemmRegistry registry;
    auto sources = staticSources(DeviceId::cuda(0));
    const auto scope = ExpertGemmRegistry::staticProjectionScope(sources.front());
    const auto projection = *sources.front().projection_ownership;
    auto missing = sources;
    missing.pop_back(); // Actual last original down source is expert511.
    EXPECT_THROW((void)registry.acquireStaticProjectionLayer(scope, projection, missing), std::invalid_argument);
    auto duplicate = sources;
    duplicate.back() = duplicate[duplicate.size()-2];
    EXPECT_THROW((void)registry.acquireStaticProjectionLayer(scope, projection, duplicate), std::invalid_argument);
    registerStaticSources(registry, missing);
    EXPECT_THROW((void)registry.acquireStaticProjectionLayer(scope, projection, sources), std::logic_error);
    const auto &last = sources.back();
    auto engine = std::make_shared<StaticSourceGemm>(8,4,last.tensor->raw_data());
    registry.registerStaticProjection(last, engine.get(), engine);
    EXPECT_TRUE(registry.acquireStaticProjectionLayer(scope, projection, sources));
    EXPECT_THROW(registry.registerStaticProjection(last, engine.get(), engine), std::logic_error);
    EXPECT_EQ(registry.size(), 512u*3);
}

TEST(StaticProjectionPreparedSource, RejectsMalformedSourceAndStaleOwnerWithoutMutation)
{
    ExpertGemmRegistry registry;
    auto sources = staticSources(DeviceId::rocm(0), 3);
    const auto scope = ExpertGemmRegistry::staticProjectionScope(sources.front());
    const auto projection = *sources.front().projection_ownership;
    for (int mutation=0; mutation<7; ++mutation)
    {
        auto source = sources.front();
        auto engine = std::make_shared<StaticSourceGemm>(4,8,source.tensor->raw_data());
        switch (mutation)
        {
        case 0: source.identity.overlay_domain.clear(); break;
        case 1: source.projection_source_owner.reset(); break;
        case 2: source.slice.col_start=1; break;
        case 3: source.identity.expert=3; break;
        case 4: source.prepared->binding_id ^= 1; break;
        case 5: source.tensor_owner=llaminar2::test::TestTensorFactory::createFP32Zeros({4,8}); source.tensor=source.tensor_owner.get(); break;
        case 6: engine=std::make_shared<StaticSourceGemm>(8,4,source.tensor->raw_data()); break;
        }
        EXPECT_THROW(registry.registerStaticProjection(source, engine.get(), engine), std::invalid_argument) << mutation;
        EXPECT_EQ(registry.size(), 0u);
    }
    registerStaticSources(registry, sources);
    auto stale = sources;
    stale.front().tensor_owner = std::shared_ptr<TensorBase>(stale.front().tensor, [](TensorBase *) {});
    EXPECT_THROW((void)registry.acquireStaticProjectionLayer(scope, projection, stale), std::logic_error);
    stale = sources;
    stale.front().projection_source_owner = std::shared_ptr<TensorBase>(stale.front().projection_source_owner.get(), [](TensorBase *) {});
    EXPECT_THROW((void)registry.acquireStaticProjectionLayer(scope, projection, stale), std::logic_error);
    EXPECT_TRUE(registry.acquireStaticProjectionLayer(scope, projection, sources));
}

TEST(StaticProjectionPreparedSource, UnequalPositivePartitionsKeepOriginalParticipants)
{
    for (const int participants : {2,4,8})
    {
        std::vector<MoEExpertProjectionOwnership::IntermediateInterval> intervals;
        int origin=0;
        for (int p=0; p<participants; ++p)
        {
            const int width = p+1==participants ? 20-origin : 1;
            intervals.push_back({origin,width});
            origin+=width;
        }
        for (int participant=0; participant<participants; ++participant)
        {
            ExpertGemmRegistry registry;
            const auto sources=staticSources(DeviceId::cuda(participant),3,participant,intervals);
            const auto scope=ExpertGemmRegistry::staticProjectionScope(sources.front());
            const auto projection=*sources.front().projection_ownership;
            registerStaticSources(registry,sources);
            const auto layer=registry.acquireStaticProjectionLayer(scope,projection,sources);
            ASSERT_TRUE(layer);
            EXPECT_EQ(layer->scope.participant_index,participant);
            EXPECT_EQ(layer->projection.intermediateIntervals().size(),size_t(participants));
            EXPECT_EQ(layer->experts[2].gate->get_n(),intervals[size_t(participant)].columns);
            EXPECT_EQ(layer->experts[2].down->get_k(),intervals[size_t(participant)].columns);
        }
    }
}


TEST(StaticProjectionPreparedSource, PreparedStoreCannotAdoptAnotherOriginalRoot)
{
    const auto sources = staticSources(DeviceId::cuda(0), 3);
    const auto &original = sources.front();
    PreparedWeightStore store(ModelContextId{55});
    // Metadata-only registration exercises adoption identity without a native
    // handle/backend; executable geometry is tested separately by this suite.
    store.registerPreparedForTest(original, original.prepared->kind, original.prepared->device);
    auto foreign = original;
    foreign.projection_source_owner = std::shared_ptr<TensorBase>(original.projection_source_owner.get(), [](TensorBase *) {});
    EXPECT_THROW((void)store.adoptPreparedGemmForBinding(foreign, original.prepared->device), std::runtime_error);
    auto same = original;
    EXPECT_TRUE(store.adoptPreparedGemmForBinding(same, original.prepared->device));
    EXPECT_EQ(same.tensor, original.tensor);
    EXPECT_EQ(same.projection_source_owner, original.projection_source_owner);
}


namespace
{
    /** @brief Actual native-source metadata mock with real CPU backing, never a fabricated GPU pointer. */
    class StaticNativeSourceGemm final : public MockGemm
    {
    public:
        /** @brief Borrow source storage only for descriptor validation; registry separately owns that source. */
        explicit StaticNativeSourceGemm(const WeightBinding &source, int width_error=0, int codebook_error=0)
            : MockGemm(56), data_(static_cast<const uint8_t *>(source.tensor->raw_data())),
              n_(int(source.tensor->shape()[0])), k_(int(source.tensor->shape()[1])+width_error)
        {
            const auto *format=dynamic_cast<const IINT8Unpackable *>(source.tensor)->vnniFormatInfo();
            original_={uint8_t(format->codebook_id+codebook_error),format->is_superblock,true};
        }
        /** @brief Native path deliberately retains generic get_n/get_k==0, as actual GPU quantized engines do. */
        bool exportNativeVNNIMatrixDesc(DeviceNativeVNNIMatrixDesc &out) override
        {
            out={};
            out.payload=data_;
            out.scales=&scale_;
            out.n=n_;
            out.k=k_;
            out.blocks_per_row=uint32_t(k_/32);
            out.codebook_id=original_.codebook_id;
            out.source_codebook_id=original_.codebook_id;
            out.source_is_superblock=uint8_t(original_.is_superblock);
            out.source_identity_present=1;
            return out.valid();
        }
        bool exportNativeVNNISourceIdentity(NativeVnniSourceIdentity &out) const override
        {
            out=original_;
            return true;
        }
    private:
        const uint8_t *data_;
        int n_;
        int k_;
        uint16_t scale_=0;
        NativeVnniSourceIdentity original_;
    };
}

TEST(StaticProjectionPreparedSource, NativeDescriptorOwnsGeometryAndOriginalCodebook)
{
    for (const auto device : {DeviceId::cuda(0),DeviceId::rocm(0)})
    {
        ExpertGemmRegistry registry;
        const auto sources=staticSources(device,3,0,{{0,32},{32,32},{64,32},{96,32}},true);
        const auto scope=ExpertGemmRegistry::staticProjectionScope(sources.front());
        const auto projection=*sources.front().projection_ownership;
        auto wrong_width=std::make_shared<StaticNativeSourceGemm>(sources.front(),32,0);
        EXPECT_THROW(registry.registerStaticProjection(sources.front(),wrong_width.get(),wrong_width),std::invalid_argument);
        auto wrong_original=std::make_shared<StaticNativeSourceGemm>(sources.front(),0,1);
        EXPECT_THROW(registry.registerStaticProjection(sources.front(),wrong_original.get(),wrong_original),std::invalid_argument);
        EXPECT_EQ(registry.size(),0u);
        for (const auto &source:sources)
        {
            auto engine=std::make_shared<StaticNativeSourceGemm>(source);
            EXPECT_EQ(engine->get_n(),0);
            registry.registerStaticProjection(source,engine.get(),engine);
        }
        const auto layer=registry.acquireStaticProjectionLayer(scope,projection,sources);
        ASSERT_TRUE(layer);
        EXPECT_EQ(layer->experts.size(),3u);
        EXPECT_EQ(layer->sources.front().tensor->native_type(),TensorType::Q8_0);
    }
}
