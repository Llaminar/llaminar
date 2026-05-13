#include <gtest/gtest.h>
#include "loaders/ExpertGemmRegistry.h"
#include "tensors/TensorKernels.h"

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
