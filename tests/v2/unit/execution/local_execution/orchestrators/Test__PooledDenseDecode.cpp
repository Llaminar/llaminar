/**
 * @file Test__PooledDenseDecode.cpp
 * @brief P2 production caller: pooled decode through the actual runner/graph.
 */
#include <gtest/gtest.h>
#include <cstring>
#include <typeinfo>
#include <memory>
#include <vector>
#include "models/qwen/QwenStandardGraph.h"
#include "planning/PagedKVInstance.h"
#include "utils/CPUForwardSamplingTestFixture.h"
#include "utils/CPUOrchestratorTestAdmission.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"

using namespace llaminar2;

/** @brief Run one stage expression, naming the stage on failure/throw. */
#define ASSERT_OK(stage, expr) \
    do { \
        try { \
            if (!(expr)) FAIL() << stage << " returned false"; \
        } catch (const std::exception &e) { \
            FAIL() << stage << " threw " << typeid(e).name() << ": " << e.what(); \
        } \
    } while (false)

/** @test P2 graph caller (R24 M1-M6): the pooled runner decode equals the
 * uninterrupted ring control across page_tokens+1 decode steps with the real
 * graph wiring (distinct prior/next cells, pre-admitted step grant, all-or-none
 * layer publication via advanceStep). The pooled prompt (3 rows) is written
 * into pooled pages at prefill; each decode step then shares every prior full
 * page and COW-extends the partial last page. Logits are byte-compared.
 */
TEST(PooledDenseDecode, RunnerPooledDecodeMatchesUninterruptedControl)
{
    const auto device = DeviceId::cpu();
    const std::array<int32_t, 3> prompt{1, 3, 5};
    // page_tokens+1 = 9 decode steps: 3-row prefill + 9 rows crosses one page
    // boundary at row 8 (full page then COW/extend + fresh page) and proves the
    // M2 history retention across full pages with a partial tail.
    constexpr int decode_steps = 9;
    const std::array<int32_t, decode_steps> decode{2, 4, 6, 8, 10, 12, 14, 16, 18};

    // The PreparedWeightStore is move-only: the runner and its store must be
    // created together and kept in one scope for the full decode sequence.
    auto make_runner = [&](test::TinyQwenForwardFixture &fx,
                           std::unique_ptr<DeviceGraphOrchestrator> &owner_out,
                           PreparedWeightStore &store_out,
                           std::shared_ptr<QwenStandardGraph> &builder_out)
    {
        auto builder = std::make_shared<QwenStandardGraph>(fx.config, fx.mpi);
        auto deps = test::cpuInferenceFixtureDependencies(builder, fx.mpi, 1,
            fx.config.max_seq_len, fx.config.max_seq_len,
            test::cpuFixtureTensorInventory(fx.modelWeights(), fx.config.n_layers));
        owner_out = DeviceGraphOrchestrator::createForTest(std::move(deps));
        auto &runner = *owner_out;
        if (!runner.initializeInferenceStateFromArena(1, fx.config.max_seq_len, device))
            throw std::runtime_error("initializeInferenceStateFromArena failed");
        runner.setWeights(fx.modelWeights());
        test::prepareDenseForwardWeights(runner, *builder, store_out, device);
        builder_out = std::move(builder);
    };

    test::TinyQwenForwardFixture control_fx(device, KVCachePrecision::FP32);
    control_fx.config.mtp.enabled = false;
    control_fx.config.mtp.graph_capacity_draft_tokens = 0;
    // 3-row prompt + 9 decode rows = 12 live rows; the fixture default horizon
    // (8) is too small, so extend it before admission.
    control_fx.config.max_seq_len = 16;
    test::TinyQwenForwardFixture pooled_fx(device, KVCachePrecision::FP32);
    pooled_fx.config.mtp.enabled = false;
    pooled_fx.config.mtp.graph_capacity_draft_tokens = 0;
    pooled_fx.config.max_seq_len = 16;

    // Control runner: uninterrupted ring-cache decode.
    std::unique_ptr<DeviceGraphOrchestrator> control_owner;
    PreparedWeightStore control_store;
    std::shared_ptr<QwenStandardGraph> control_builder;
    make_runner(control_fx, control_owner, control_store, control_builder);
    auto &control = *control_owner;
    ASSERT_OK("control prefill", control.forward(prompt.data(), 3, 1) != nullptr);

    // Pooled instance (fresh PMA authority) admitted, then pooled runner.
    PagedKVTierConfig tier_config; // Auto pool.
    PagedKVPoolBudget budget;
    budget.remaining_budget_bytes = 2 * 1024 * 1024;
    budget.frame_bytes = 4 * 1024;
    budget.request_records = 4;
    budget.max_pages_per_request = 64;
    auto plan = planPagedKVInstance(tier_config, budget,
        PagedKVInstancePolicy{/*page_tokens=*/8, /*bank_count=*/8,
            /*readers_per_bank=*/4, /*request_records=*/16},
        device);
    ASSERT_TRUE(plan);
    PhysicalMemoryPlanBuilder mbuilder;
    mbuilder.add({.world_rank=0, .device=device,
        .total_bytes=plan->total_bytes,
        .admission_available_bytes=plan->total_bytes},
        PhysicalMemoryOwner::KVCache, plan->total_bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(mbuilder.build()), 0);
    auto created = PagedKVInstance::create(authority, *plan, device);
    ASSERT_TRUE(created);
    PagedKVInstance instance = std::move(*created);

    std::unique_ptr<DeviceGraphOrchestrator> pooled_owner;
    PreparedWeightStore pooled_store;
    std::shared_ptr<QwenStandardGraph> pooled_builder;
    make_runner(pooled_fx, pooled_owner, pooled_store, pooled_builder);
    auto &pooled = *pooled_owner;
    PooledKVDecodeBinding binding(&instance.pool(), &instance.table(),
        /*request_id=*/77, /*slot=*/0,
        pooled_fx.config.n_kv_heads, pooled_fx.config.head_dim,
        /*page_tokens=*/8, pooled_fx.config.n_layers);
    pooled.setPooledDecodeBinding(&binding);

    // M1/M4: pooled prefill writes the prompt (3 rows; partial first page).
    ASSERT_OK("pooled prefill (pooled)", pooled.forward(prompt.data(), 3, 1) != nullptr);
    ASSERT_FALSE(binding.stepPrepared()); // grant consumed and released.
    auto *prior0 = binding.priorFor(0);
    ASSERT_NE(prior0, nullptr);
    ASSERT_EQ(prior0->snapshot.view().rows(), 3);

    const std::size_t vocab = static_cast<std::size_t>(pooled_fx.config.vocab_size);
    for (int step = 0; step < decode_steps; ++step)
    {
        const auto *control_next = control.forward(&decode[step], 1, 1);
        const auto *pooled_next = pooled.forward(&decode[step], 1, 1);
        ASSERT_NE(control_next, nullptr);
        ASSERT_NE(pooled_next, nullptr);
        EXPECT_EQ(std::memcmp(pooled_next, control_next, vocab * sizeof(float)), 0)
            << "pooled decode diverged from the control at step " << step;
        // M1: after advanceStep the new publication is the prior of the next
        // step; the pooled history grew by exactly one row per decode step.
        auto *prior = binding.priorFor(0);
        ASSERT_NE(prior, nullptr) << "prior lost after step " << step;
        const int expected_rows = 3 + step + 1;
        EXPECT_EQ(prior->snapshot.view().rows(), expected_rows)
            << "pooled history rows at step " << step;
        EXPECT_FALSE(binding.stepPrepared());
    }
}

/** @test R24 (maxim 13): CPU forward graphs are rebuilt per forward, so a
 * binding attached between forwards is observed by the NEXT graph; a graph
 * cached across forwards would keep writing the OLD binding's cells. The
 * negative: after switching the binding to a second instance, the next pooled
 * forward must publish into the SECOND instance's table only. */
TEST(PooledDenseDecode, BindingToggleIsObservedByNextForwardGraph)
{
    const auto device = DeviceId::cpu();
    const std::array<int32_t, 2> prompt{1, 3};

    auto make_pooled_instance = [&](PagedKVInstance &out) -> bool
    {
        PagedKVPoolBudget budget;
        budget.remaining_budget_bytes = 2 * 1024 * 1024;
        budget.frame_bytes = 4 * 1024;
        budget.request_records = 4;
        budget.max_pages_per_request = 64;
        auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget,
            PagedKVInstancePolicy{/*page_tokens=*/8, /*bank_count=*/8,
                /*readers_per_bank=*/4, /*request_records=*/16},
            device);
        if (!plan) return false;
        PhysicalMemoryPlanBuilder mb;
        mb.add({.world_rank=0, .device=device,
            .total_bytes=plan->total_bytes,
            .admission_available_bytes=plan->total_bytes},
            PhysicalMemoryOwner::KVCache, plan->total_bytes);
        auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(mb.build()), 0);
        auto created = PagedKVInstance::create(authority, *plan, device);
        if (!created) return false;
        out = std::move(*created);
        return true;
    };

    test::TinyQwenForwardFixture fx(device, KVCachePrecision::FP32);
    fx.config.mtp.enabled = false;
    fx.config.mtp.graph_capacity_draft_tokens = 0;
    fx.config.max_seq_len = 16;
    auto builder = std::make_shared<QwenStandardGraph>(fx.config, fx.mpi);
    auto deps = test::cpuInferenceFixtureDependencies(builder, fx.mpi, 1,
        fx.config.max_seq_len, fx.config.max_seq_len,
        test::cpuFixtureTensorInventory(fx.modelWeights(), fx.config.n_layers));
    auto owner = DeviceGraphOrchestrator::createForTest(std::move(deps));
    auto &runner = *owner;
    ASSERT_TRUE(runner.initializeInferenceStateFromArena(1, fx.config.max_seq_len, device));
    runner.setWeights(fx.modelWeights());
    PreparedWeightStore store;
    test::prepareDenseForwardWeights(runner, *builder, store, device);

    PagedKVInstance instance_a;
    ASSERT_TRUE(make_pooled_instance(instance_a));
    PagedKVInstance instance_b;
    ASSERT_TRUE(make_pooled_instance(instance_b));
    PooledKVDecodeBinding binding_a(&instance_a.pool(), &instance_a.table(),
        /*request_id=*/55, /*slot=*/0,
        fx.config.n_kv_heads, fx.config.head_dim, /*page_tokens=*/8, fx.config.n_layers);
    PooledKVDecodeBinding binding_b(&instance_b.pool(), &instance_b.table(),
        /*request_id=*/56, /*slot=*/0,
        fx.config.n_kv_heads, fx.config.head_dim, /*page_tokens=*/8, fx.config.n_layers);

    // First forward: binding A attached.
    runner.setPooledDecodeBinding(&binding_a);
    ASSERT_OK("prefill A", runner.forward(prompt.data(), 2, 1) != nullptr);
    auto *prior_a = binding_a.priorFor(0);
    ASSERT_NE(prior_a, nullptr);
    EXPECT_EQ(prior_a->snapshot.view().rows(), 2);
    auto *prior_b0 = binding_b.priorFor(0);
    EXPECT_EQ(prior_b0, nullptr);

    // Second forward: switch the binding to B, a DIFFERENT instance whose
    // publication starts at position 0 while the host cursor is 2. A graph
    // CACHED from forward 1 (built with A) would check A's nextPosition (=2),
    // agree with the cursor and silently publish into A. The fresh CPU graph
    // consults the binding attached NOW (B), sees the position disagreement
    // and refuses fail-closed before any graph runs. That refusal IS the
    // maxim-13 proof that the forward graph is rebuilt and observes the
    // current binding; no cached graph can see B.
    runner.setPooledDecodeBinding(&binding_b);
    const int32_t token = 2;
    EXPECT_EQ(runner.forward(&token, 1, 1), nullptr)
        << "pooled forward with a fresh binding must refuse on position authority";
    EXPECT_TRUE(binding_b.priorFor(0) == nullptr); // B never published.
    EXPECT_EQ(binding_a.priorFor(0)->snapshot.view().rows(), 2); // A untouched.

    // Third forward: switch back to A; the next fresh graph sees A again and
    // the position authority agrees (publication=2, cursor=2).
    runner.setPooledDecodeBinding(&binding_a);
    const int32_t token2 = 4;
    ASSERT_OK("decode A again", runner.forward(&token2, 1, 1) != nullptr);
    auto *prior_a2 = binding_a.priorFor(0);
    ASSERT_NE(prior_a2, nullptr);
    EXPECT_EQ(prior_a2->snapshot.view().rows(), 3);
    EXPECT_TRUE(binding_b.priorFor(0) == nullptr); // Still never published.
}
