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
    // N2: the policy derives FROM the admitted envelope (3+9 forwards = 12
    // issuances, 1 layer), not test constants.
    const auto policy = PooledKVDecodeBinding::requestPolicy(/*page_tokens=*/8,
        /*layers=*/pooled_fx.config.n_layers,
        /*admitted_issuances=*/12, /*readers_per_bank=*/4);
    auto plan = planPagedKVInstance(tier_config, budget, policy, device);
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
        /*page_tokens=*/8, pooled_fx.config.n_layers,
        /*admitted_issuances=*/12);
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

    auto make_pooled_instance = [&](int layers, PagedKVInstance &out) -> bool
    {
        PagedKVPoolBudget budget;
        budget.remaining_budget_bytes = 2 * 1024 * 1024;
        budget.frame_bytes = 4 * 1024;
        budget.request_records = 4;
        budget.max_pages_per_request = 64;
        const auto policy = PooledKVDecodeBinding::requestPolicy(/*page_tokens=*/8,
            /*layers=*/layers,
            /*admitted_issuances=*/8, /*readers_per_bank=*/4);
        auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget, policy, device);
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
    ASSERT_TRUE(make_pooled_instance(fx.config.n_layers, instance_a));
    PagedKVInstance instance_b;
    ASSERT_TRUE(make_pooled_instance(fx.config.n_layers, instance_b));
    PooledKVDecodeBinding binding_a(&instance_a.pool(), &instance_a.table(),
        /*request_id=*/55, /*slot=*/0,
        fx.config.n_kv_heads, fx.config.head_dim, /*page_tokens=*/8, fx.config.n_layers,
        /*admitted_issuances=*/8);
    PooledKVDecodeBinding binding_b(&instance_b.pool(), &instance_b.table(),
        /*request_id=*/56, /*slot=*/0,
        fx.config.n_kv_heads, fx.config.head_dim, /*page_tokens=*/8, fx.config.n_layers,
        /*admitted_issuances=*/8);

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

/** @test P2 in-place private tail (design line 43): with the persistent tail
 * grant, a decode step extends the request's OWN private tail FRAME in place
 * instead of COW-copying the prior partial page. The pooled decode stays
 * byte-identical to the contiguous control across a page boundary, and the
 * in-place counter (not the COW counter) records the extension bytes. */
TEST(PooledDenseDecode, RunnerInPlacePrivateTailMatchesControl)
{
    const auto device = DeviceId::cpu();
    const std::array<int32_t, 3> prompt{1, 3, 5};
    constexpr int decode_steps = 9;
    const std::array<int32_t, decode_steps> decode{2, 4, 6, 8, 10, 12, 14, 18, 20};

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
    control_fx.config.max_seq_len = 16;
    test::TinyQwenForwardFixture pooled_fx(device, KVCachePrecision::FP32);
    pooled_fx.config.mtp.enabled = false;
    pooled_fx.config.mtp.graph_capacity_draft_tokens = 0;
    pooled_fx.config.max_seq_len = 16;

    std::unique_ptr<DeviceGraphOrchestrator> control_owner;
    PreparedWeightStore control_store;
    std::shared_ptr<QwenStandardGraph> control_builder;
    make_runner(control_fx, control_owner, control_store, control_builder);
    auto &control = *control_owner;
    ASSERT_OK("control prefill", control.forward(prompt.data(), 3, 1) != nullptr);

    PagedKVPoolBudget budget;
    budget.remaining_budget_bytes = 2 * 1024 * 1024;
    budget.frame_bytes = 4 * 1024;
    budget.request_records = 4;
    budget.max_pages_per_request = 64;
    const auto policy = PooledKVDecodeBinding::requestPolicy(/*page_tokens=*/8,
        /*layers=*/pooled_fx.config.n_layers,
        /*admitted_issuances=*/12, /*readers_per_bank=*/4);
    auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget, policy, device);
    ASSERT_TRUE(plan);
    // Slightly larger table so the binding's tail grant and all banks seat.
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
    // persistent_tail=true enables the in-place private tail extension.
    PooledKVDecodeBinding binding(&instance.pool(), &instance.table(),
        /*request_id=*/88, /*slot=*/0,
        pooled_fx.config.n_kv_heads, pooled_fx.config.head_dim,
        /*page_tokens=*/8, pooled_fx.config.n_layers,
        /*admitted_issuances=*/12, /*persistent_tail=*/true);
    ASSERT_TRUE(binding.hasTailGrant());
    pooled.setPooledDecodeBinding(&binding);

    ASSERT_OK("pooled prefill (in-place tail)", pooled.forward(prompt.data(), 3, 1) != nullptr);
    const std::size_t vocab = static_cast<std::size_t>(pooled_fx.config.vocab_size);
    for (int step = 0; step < decode_steps; ++step)
    {
        const auto *control_next = control.forward(&decode[step], 1, 1);
        const auto *pooled_next = pooled.forward(&decode[step], 1, 1);
        ASSERT_NE(control_next, nullptr);
        ASSERT_NE(pooled_next, nullptr);
        EXPECT_EQ(std::memcmp(pooled_next, control_next, vocab * sizeof(float)), 0)
            << "in-place pooled decode diverged from the control at step " << step;
        auto *prior = binding.priorFor(0);
        ASSERT_NE(prior, nullptr) << "prior lost after step " << step;
        // Rows grow exactly like the COW caller (the in-place path appends in
        // the same frame; row arithmetic is unchanged).
        EXPECT_EQ(prior->snapshot.view().rows(), 3 + step + 1);
        EXPECT_FALSE(binding.stepPrepared());
    }
    // The in-place tail grant must still be live (the request owns its tail).
    EXPECT_TRUE(binding.hasTailGrant());
    // Rows pass the page boundary: 3 prefill + 9 decode = 12 rows, crossing the
    // 8-row page at step 4 — the in-place extension filled the first page and
    // a rolled fresh tail frame took the next partial page.
    EXPECT_EQ(binding.priorFor(0)->snapshot.view().rows(), 12);
}

/** @test P2 multi-request on a SHARED pool: two live pooled requests use one
 * PMA-backed pool+table instance (separate request records, separate
 * repositories per request), interleaved decode, each byte-identical to its
 * own uninterrupted ring control. The shared pool edits no logical request
 * state across the two. */
TEST(PooledDenseDecode, TwoRequestsShareOnePoolWithoutCrossTalk)
{
    const auto device = DeviceId::cpu();
    const std::array<int32_t, 2> prompt_a{1, 5};
    const std::array<int32_t, 3> prompt_b{2, 3, 5};
    const std::array<int32_t, 5> decode_a{2, 4, 6, 8, 10};
    const std::array<int32_t, 4> decode_b{1, 3, 5, 7};

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

    test::TinyQwenForwardFixture ctl_a(device, KVCachePrecision::FP32);
    ctl_a.config.mtp.enabled = false; ctl_a.config.mtp.graph_capacity_draft_tokens = 0;
    ctl_a.config.max_seq_len = 16;
    test::TinyQwenForwardFixture ctl_b(device, KVCachePrecision::FP32);
    ctl_b.config.mtp.enabled = false; ctl_b.config.mtp.graph_capacity_draft_tokens = 0;
    ctl_b.config.max_seq_len = 16;

    std::unique_ptr<DeviceGraphOrchestrator> ctl_a_owner, ctl_b_owner;
    PreparedWeightStore ctl_a_store, ctl_b_store;
    std::shared_ptr<QwenStandardGraph> ctl_a_builder, ctl_b_builder;
    make_runner(ctl_a, ctl_a_owner, ctl_a_store, ctl_a_builder);
    make_runner(ctl_b, ctl_b_owner, ctl_b_store, ctl_b_builder);
    ASSERT_OK("control A prefill", ctl_a_owner->forward(prompt_a.data(), 2, 1) != nullptr);
    ASSERT_OK("control B prefill", ctl_b_owner->forward(prompt_b.data(), 3, 1) != nullptr);

    // ONE shared instance seats both requests (2 records, banks for 2x2).
    PagedKVPoolBudget budget;
    budget.remaining_budget_bytes = 3 * 1024 * 1024;
    budget.frame_bytes = 4 * 1024;
    budget.request_records = 4;
    budget.max_pages_per_request = 64;
    const auto policy = PooledKVDecodeBinding::requestPolicy(/*page_tokens=*/8,
        /*layers=*/ctl_a.config.n_layers, /*admitted_issuances=*/8, /*readers_per_bank=*/4);
    auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget, policy, device);
    ASSERT_TRUE(plan);
    // Table needs 2 bindings x 2 banks/layer; widen it explicitly.
    auto table_geom = plan->table_geometry;
    table_geom.bank_count = 2 * table_geom.bank_count;
    table_geom.issuance_headroom = 4 * table_geom.issuance_headroom;
    table_geom.issuance_budget = 2 * table_geom.issuance_headroom;
    auto widened = std::make_shared<PagedKVInstancePlan>(*plan);
    widened->table_geometry = table_geom;
    widened->total_bytes = PhysicalPagePool::physicalBytes(plan->pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geom, device);
    PhysicalMemoryPlanBuilder mbuilder;
    mbuilder.add({.world_rank=0, .device=device,
        .total_bytes=widened->total_bytes,
        .admission_available_bytes=widened->total_bytes},
        PhysicalMemoryOwner::KVCache, widened->total_bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(mbuilder.build()), 0);
    auto created = PagedKVInstance::create(authority, *widened, device);
    ASSERT_TRUE(created);
    PagedKVInstance instance = std::move(*created);

    std::unique_ptr<DeviceGraphOrchestrator> pooled_a_owner, pooled_b_owner;
    PreparedWeightStore pooled_a_store, pooled_b_store;
    std::shared_ptr<QwenStandardGraph> pooled_a_builder, pooled_b_builder;
    make_runner(ctl_a, pooled_a_owner, pooled_a_store, pooled_a_builder);
    make_runner(ctl_b, pooled_b_owner, pooled_b_store, pooled_b_builder);
    PooledKVDecodeBinding binding_a(&instance.pool(), &instance.table(),
        /*request_id=*/101, /*slot=*/0,
        ctl_a.config.n_kv_heads, ctl_a.config.head_dim,
        /*page_tokens=*/8, ctl_a.config.n_layers, /*admitted_issuances=*/8);
    PooledKVDecodeBinding binding_b(&instance.pool(), &instance.table(),
        /*request_id=*/102, /*slot=*/1,
        ctl_b.config.n_kv_heads, ctl_b.config.head_dim,
        /*page_tokens=*/8, ctl_b.config.n_layers, /*admitted_issuances=*/8);
    pooled_a_owner->setPooledDecodeBinding(&binding_a);
    pooled_b_owner->setPooledDecodeBinding(&binding_b);

    ASSERT_OK("pooled A prefill", pooled_a_owner->forward(prompt_a.data(), 2, 1) != nullptr);
    ASSERT_OK("pooled B prefill", pooled_b_owner->forward(prompt_b.data(), 3, 1) != nullptr);
    const std::size_t vocab = static_cast<std::size_t>(ctl_a.config.vocab_size);
    // Interleave decodes: A then B, A, B, ... — both live at once. Each pooled
    // token is byte-compared against the control's matching token IMMEDIATELY
    // (same position, same token), so a divergence is located exactly.
    for (int step = 0; step < static_cast<int>(decode_a.size()); ++step)
    {
        const auto *ctl_a = ctl_a_owner->forward(&decode_a[step], 1, 1);
        const auto *pooled_a = pooled_a_owner->forward(&decode_a[step], 1, 1);
        ASSERT_NE(ctl_a, nullptr) << "control A forward " << step;
        ASSERT_NE(pooled_a, nullptr) << "pooled A forward " << step;
        EXPECT_EQ(std::memcmp(pooled_a, ctl_a, vocab * sizeof(float)), 0)
            << "shared-pool request A diverged at step " << step;
        if (step < static_cast<int>(decode_b.size()))
        {
            const auto *ctl_b = ctl_b_owner->forward(&decode_b[step], 1, 1);
            const auto *pooled_b = pooled_b_owner->forward(&decode_b[step], 1, 1);
            ASSERT_NE(ctl_b, nullptr) << "control B forward " << step;
            ASSERT_NE(pooled_b, nullptr) << "pooled B forward " << step;
            EXPECT_EQ(std::memcmp(pooled_b, ctl_b, vocab * sizeof(float)), 0)
                << "shared-pool request B diverged at step " << step;
        }
    }
    // Each request's publication is independent and grew exactly.
    EXPECT_EQ(binding_a.priorFor(0)->snapshot.view().rows(), 2 + 5);
    EXPECT_EQ(binding_b.priorFor(0)->snapshot.view().rows(), 3 + 4);
}


/** @test P2 runner-level truncate/export/import: a live pooled request exports
 * its history, truncates mid-page, and continues decode; the byte oracle is a
 * SECOND pooled request that prefilled the same truncated prefix and decoded
 * the same tokens from row 0 of an identical history and stream pace — the
 * two end at the same row count with identical K/V bytes and logits. */
TEST(PooledDenseDecode, RunnerTruncateExportImportRoundTrip)
{
    const auto device = DeviceId::cpu();
    const std::array<int32_t, 4> prompt{1, 3, 5, 7};
    const std::array<int32_t, 4> decode_after{2, 4, 6, 8};
    const int truncated_rows = 4; // Mid-page (4 < page_tokens 8).

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

    test::TinyQwenForwardFixture fx(device, KVCachePrecision::FP32);
    fx.config.mtp.enabled = false; fx.config.mtp.graph_capacity_draft_tokens = 0;
    fx.config.max_seq_len = 24;
    auto make_instance = [&](PagedKVInstance &out) -> bool
    {
        PagedKVPoolBudget budget;
        budget.remaining_budget_bytes = 3 * 1024 * 1024;
        budget.frame_bytes = 4 * 1024;
        budget.request_records = 4;
        budget.max_pages_per_request = 64;
        const auto policy = PooledKVDecodeBinding::requestPolicy(/*page_tokens=*/8,
            /*layers=*/fx.config.n_layers, /*admitted_issuances=*/10, /*readers_per_bank=*/4);
        auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget, policy, device);
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

    // Target: prefill 4 + decode 2 = 6 rows, truncate to 4, decode 4 more = 8.
    PagedKVInstance target_instance;
    ASSERT_TRUE(make_instance(target_instance));
    std::unique_ptr<DeviceGraphOrchestrator> target_owner;
    PreparedWeightStore target_store;
    std::shared_ptr<QwenStandardGraph> target_builder;
    make_runner(fx, target_owner, target_store, target_builder);
    // The truncate test exercises the COW/truncation path with a fixed
    // geometry; the persistent in-place tail is opted out explicitly.
    PooledKVDecodeBinding target(&target_instance.pool(), &target_instance.table(),
        /*request_id=*/21, /*slot=*/0,
        fx.config.n_kv_heads, fx.config.head_dim, /*page_tokens=*/8, fx.config.n_layers,
        /*admitted_issuances=*/10, /*persistent_tail=*/false);
    target_owner->setPooledDecodeBinding(&target);

    ASSERT_OK("target prefill", target_owner->forward(prompt.data(), 4, 1) != nullptr);
    const std::array<int32_t, 2> pre_decode{2, 4};
    ASSERT_OK("target decode 0", target_owner->forward(&pre_decode[0], 1, 1) != nullptr);
    ASSERT_OK("target decode 1", target_owner->forward(&pre_decode[1], 1, 1) != nullptr);
    EXPECT_EQ(target.priorFor(0)->snapshot.view().rows(), 6);
    auto exported = target.exportLayer(0);
    ASSERT_TRUE(exported.has_value());
    EXPECT_EQ(static_cast<int>(exported->meta().rows), 6);
    EXPECT_TRUE(target.truncateLayer(0, truncated_rows));
    EXPECT_EQ(target.priorFor(0)->snapshot.view().rows(), truncated_rows);
    // Typed cursor lifecycle: no host setter — the DGO adopts the publication
    // row count because the binding's cursor epoch advanced.
    EXPECT_EQ(target.cursorEpoch(), 1u);
    for (const auto token : decode_after)
        ASSERT_OK("target decode after truncate", target_owner->forward(&token, 1, 1) != nullptr);
    EXPECT_EQ(target.priorFor(0)->snapshot.view().rows(),
        truncated_rows + static_cast<int>(decode_after.size()));

    // Oracle: a second pooled request that prefilled the SAME 4-row prefix and
    // decoded the same 4 tokens — identical history bytes at row 8.
    PagedKVInstance oracle_instance;
    ASSERT_TRUE(make_instance(oracle_instance));
    std::unique_ptr<DeviceGraphOrchestrator> oracle_owner;
    PreparedWeightStore oracle_store;
    std::shared_ptr<QwenStandardGraph> oracle_builder;
    make_runner(fx, oracle_owner, oracle_store, oracle_builder);
    PooledKVDecodeBinding oracle(&oracle_instance.pool(), &oracle_instance.table(),
        /*request_id=*/22, /*slot=*/0,
        fx.config.n_kv_heads, fx.config.head_dim, /*page_tokens=*/8, fx.config.n_layers,
        /*admitted_issuances=*/10);
    oracle_owner->setPooledDecodeBinding(&oracle);
    ASSERT_OK("oracle prefill", oracle_owner->forward(prompt.data(), 4, 1) != nullptr);
    for (const auto token : decode_after)
        ASSERT_OK("oracle decode", oracle_owner->forward(&token, 1, 1) != nullptr);
    EXPECT_EQ(oracle.priorFor(0)->snapshot.view().rows(), 8);

    // Both are at row 8 with the same 4-token prefix + the same decode tokens:
    // compare the NEXT token's logits (identical attention inputs).
    const int32_t cmp = 10;
    const auto *target_cmp = target_owner->forward(&cmp, 1, 1);
    const auto *oracle_cmp = oracle_owner->forward(&cmp, 1, 1);
    ASSERT_NE(target_cmp, nullptr);
    ASSERT_NE(oracle_cmp, nullptr);
    const std::size_t vocab = static_cast<std::size_t>(fx.config.vocab_size);
    EXPECT_EQ(std::memcmp(target_cmp, oracle_cmp, vocab * sizeof(float)), 0)
        << "truncated request diverged from the identical-prefix oracle";
    // Byte-identity of the retained history: same pages, same rows.
    auto target_view = target.priorFor(0)->snapshot.view();
    auto oracle_view = oracle.priorFor(0)->snapshot.view();
    ASSERT_EQ(target_view.rows(), oracle_view.rows());
    const int rows = target_view.rows();
    for (int row = 0; row < rows; ++row)
    {
        EXPECT_EQ(std::memcmp(target_view.key(row, 0), oracle_view.key(row, 0),
                static_cast<std::size_t>(fx.config.head_dim) * sizeof(float)), 0)
            << "history K diverged at row " << row;
        EXPECT_EQ(std::memcmp(target_view.value(row, 0), oracle_view.value(row, 0),
                static_cast<std::size_t>(fx.config.head_dim) * sizeof(float)), 0)
            << "history V diverged at row " << row;
    }
}

/** @test R3: truncate/import on ONE grouped member re-seats only that
 * member's typed cursor epoch — the sibling member's epoch and publication
 * stay independent, and the DGO adopts the truncated member's publication
 * position in a later grouped forward. */
TEST(PooledDenseDecode, GroupedMemberTruncateReseatsOnlyThatMember)
{
    const auto device = DeviceId::cpu();
    const std::array<int32_t, 3> prompt_a{1, 5, 3};
    const std::array<int32_t, 4> prompt_b{2, 3, 5};
    const int truncated_rows = 3; // A: prefill 3 + 2 decodes = 5 then truncate to 3.

    auto make_runner = [&](test::TinyQwenForwardFixture &fx,
                           std::unique_ptr<DeviceGraphOrchestrator> &owner_out,
                           PreparedWeightStore &store_out,
                           std::shared_ptr<QwenStandardGraph> &builder_out,
                           int batch)
    {
        auto builder = std::make_shared<QwenStandardGraph>(fx.config, fx.mpi);
        auto deps = test::cpuInferenceFixtureDependencies(builder, fx.mpi, batch,
            fx.config.max_seq_len, fx.config.max_seq_len,
            test::cpuFixtureTensorInventory(fx.modelWeights(), fx.config.n_layers));
        owner_out = DeviceGraphOrchestrator::createForTest(std::move(deps));
        auto &runner = *owner_out;
        if (!runner.initializeInferenceStateFromArena(batch, fx.config.max_seq_len, device))
            throw std::runtime_error("initializeInferenceStateFromArena failed");
        runner.setWeights(fx.modelWeights());
        test::prepareDenseForwardWeights(runner, *builder, store_out, device);
        builder_out = std::move(builder);
    };

    test::TinyQwenForwardFixture fx(device, KVCachePrecision::FP32);
    fx.config.mtp.enabled = false; fx.config.mtp.graph_capacity_draft_tokens = 0;
    fx.config.max_seq_len = 16;

    PagedKVPoolBudget budget;
    budget.remaining_budget_bytes = 3 * 1024 * 1024;
    budget.frame_bytes = 4 * 1024;
    // Enough live grant records for two steps + the truncate's private bank.
    budget.request_records = 8;
    budget.max_pages_per_request = 64;
    const auto policy = PooledKVDecodeBinding::requestPolicy(/*page_tokens=*/8,
        /*layers=*/fx.config.n_layers, /*admitted_issuances=*/16, /*readers_per_bank=*/4);
    auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget, policy, device);
    ASSERT_TRUE(plan);
    auto table_geom = plan->table_geometry;
    table_geom.bank_count = 2 * table_geom.bank_count;
    table_geom.issuance_headroom = 4 * table_geom.issuance_headroom;
    table_geom.issuance_budget = 2 * table_geom.issuance_headroom;
    // The R3 truncation admits ONE private frame for the materialized bank
    // plus per-step grants for two live requests; give the physical pool
    // generous frame headroom (only frames, never the geometry policy) so
    // the admissions stay deterministic.
    auto pool_geom = plan->pool_geometry;
    pool_geom.frames = 64 + 8 * pool_geom.frames;
    // Multiple live grants coexist: per-step grants, the truncate's private
    // publication grant (held for the bank lifetime) and the evergreen
    // reservation records. Do NOT derive records from the tiny policy here.
    pool_geom.request_records = 8;
    auto widened = std::make_shared<PagedKVInstancePlan>(*plan);
    widened->pool_geometry = pool_geom;
    widened->table_geometry = table_geom;
    widened->total_bytes = PhysicalPagePool::physicalBytes(pool_geom, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geom, device);
    PhysicalMemoryPlanBuilder mbuilder;
    mbuilder.add({.world_rank=0, .device=device,
        .total_bytes=widened->total_bytes,
        .admission_available_bytes=widened->total_bytes},
        PhysicalMemoryOwner::KVCache, widened->total_bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(mbuilder.build()), 0);
    auto created = PagedKVInstance::create(authority, *widened, device);
    ASSERT_TRUE(created);
    PagedKVInstance instance = std::move(*created);

    std::unique_ptr<DeviceGraphOrchestrator> grouped_owner;
    PreparedWeightStore grouped_store;
    std::shared_ptr<QwenStandardGraph> grouped_builder;
    make_runner(fx, grouped_owner, grouped_store, grouped_builder, 2);
    // Truncation independence is the R3 contract; exercise it on the COW
    // arm (two binding tail grants would participate in the same frame count
    // and the truncate admission is a normal grant — persistent_tail=false
    // matches the round-trip test's geometry and keeps the truncate grant
    // admission deterministic).
    PooledKVDecodeBinding binding_a(&instance.pool(), &instance.table(),
        /*request_id=*/311, /*slot=*/0,
        fx.config.n_kv_heads, fx.config.head_dim,
        /*page_tokens=*/8, fx.config.n_layers, /*admitted_issuances=*/16,
        /*persistent_tail=*/false);
    PooledKVDecodeBinding binding_b(&instance.pool(), &instance.table(),
        /*request_id=*/312, /*slot=*/1,
        fx.config.n_kv_heads, fx.config.head_dim,
        /*page_tokens=*/8, fx.config.n_layers, /*admitted_issuances=*/16,
        /*persistent_tail=*/false);
    grouped_owner->setPooledDecodeBinding(&binding_a);
    grouped_owner->setPooledDecodePositions({0, 0});
    ASSERT_OK("pooled A prefill", grouped_owner->forward(prompt_a.data(), 3, 1) != nullptr);
    grouped_owner->setPooledDecodeBinding(nullptr);
    grouped_owner->setPooledDecodeBinding(&binding_b);
    grouped_owner->setPooledDecodePositions({0, 0});
    ASSERT_OK("pooled B prefill", grouped_owner->forward(prompt_b.data(), 4, 1) != nullptr);
    grouped_owner->setPooledDecodeBinding(nullptr);
    grouped_owner->setPooledDecodeGroup({&binding_a, &binding_b});

    // Two grouped decode steps: A rows 3+2=5, B rows 4+2=6.
    const int32_t tk_a[2] = {2, 4};
    const int32_t tk_b[2] = {1, 3};
    for (int s = 0; s < 2; ++s)
    {
        grouped_owner->setPooledDecodePositions({3 + s, 4 + s});
        const int32_t batch[2] = {tk_a[s], tk_b[s]};
        ASSERT_OK("grouped decode", grouped_owner->forward(batch, 1, 2) != nullptr);
    }
    ASSERT_EQ(binding_a.priorFor(0)->snapshot.view().rows(), 5);
    ASSERT_EQ(binding_b.priorFor(0)->snapshot.view().rows(), 6);
    // Truncate ONLY member A to its prefill length (3 rows): its epoch
    // advances; B keeps epoch 0 and its 6-row publication untouched.
    EXPECT_EQ(binding_a.cursorEpoch(), 0u);
    EXPECT_EQ(binding_b.cursorEpoch(), 0u);
    ASSERT_TRUE(binding_a.truncateLayer(0, truncated_rows));
    EXPECT_EQ(binding_a.cursorEpoch(), 1u);
    EXPECT_EQ(binding_b.cursorEpoch(), 0u);
    EXPECT_EQ(binding_b.priorFor(0)->snapshot.view().rows(), 6);
    // A grouped forward with BOTH members: the DGO adopts A's re-seated
    // position (3, not 5) from the publication while B stays at 6.
    grouped_owner->setPooledDecodePositions({3, 6});
    const int32_t batch[2] = {6, 5};
    ASSERT_OK("grouped decode after member truncate",
        grouped_owner->forward(batch, 1, 2) != nullptr);
    EXPECT_EQ(binding_a.priorFor(0)->snapshot.view().rows(), 4);
    EXPECT_EQ(binding_b.priorFor(0)->snapshot.view().rows(), 7);
}
/** @test R2: the grouped gate refuses when the host cursor vector is too
 * small for the batch — the indexed positions access is guarded BEFORE any
 * out-of-range read (adoption only re-seats existing entries). */
TEST(PooledDenseDecode, GroupedGateRefusesShortCursorVector)
{
    const auto device = DeviceId::cpu();
    const std::array<int32_t, 3> prompt_a{1, 5, 3};
    const std::array<int32_t, 4> prompt_b{2, 3, 5};

    auto make_runner = [&](test::TinyQwenForwardFixture &fx,
                           std::unique_ptr<DeviceGraphOrchestrator> &owner_out,
                           PreparedWeightStore &store_out,
                           std::shared_ptr<QwenStandardGraph> &builder_out,
                           int batch)
    {
        auto builder = std::make_shared<QwenStandardGraph>(fx.config, fx.mpi);
        auto deps = test::cpuInferenceFixtureDependencies(builder, fx.mpi, batch,
            fx.config.max_seq_len, fx.config.max_seq_len,
            test::cpuFixtureTensorInventory(fx.modelWeights(), fx.config.n_layers));
        owner_out = DeviceGraphOrchestrator::createForTest(std::move(deps));
        auto &runner = *owner_out;
        if (!runner.initializeInferenceStateFromArena(batch, fx.config.max_seq_len, device))
            throw std::runtime_error("initializeInferenceStateFromArena failed");
        runner.setWeights(fx.modelWeights());
        test::prepareDenseForwardWeights(runner, *builder, store_out, device);
        builder_out = std::move(builder);
    };

    test::TinyQwenForwardFixture fx(device, KVCachePrecision::FP32);
    fx.config.mtp.enabled = false; fx.config.mtp.graph_capacity_draft_tokens = 0;
    fx.config.max_seq_len = 16;

    PagedKVPoolBudget budget;
    budget.remaining_budget_bytes = 3 * 1024 * 1024;
    budget.frame_bytes = 4 * 1024;
    budget.request_records = 8;
    budget.max_pages_per_request = 64;
    const auto policy = PooledKVDecodeBinding::requestPolicy(/*page_tokens=*/8,
        /*layers=*/fx.config.n_layers, /*admitted_issuances=*/16, /*readers_per_bank=*/4);
    auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget, policy, device);
    ASSERT_TRUE(plan);
    auto table_geom = plan->table_geometry;
    table_geom.bank_count = 2 * table_geom.bank_count;
    table_geom.issuance_headroom = 4 * table_geom.issuance_headroom;
    table_geom.issuance_budget = 2 * table_geom.issuance_headroom;
    auto pool_geom = plan->pool_geometry;
    pool_geom.frames = 64 + 8 * pool_geom.frames;
    pool_geom.request_records = 8;
    auto widened = std::make_shared<PagedKVInstancePlan>(*plan);
    widened->pool_geometry = pool_geom;
    widened->table_geometry = table_geom;
    widened->total_bytes = PhysicalPagePool::physicalBytes(pool_geom, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geom, device);
    PhysicalMemoryPlanBuilder mbuilder;
    mbuilder.add({.world_rank=0, .device=device,
        .total_bytes=widened->total_bytes,
        .admission_available_bytes=widened->total_bytes},
        PhysicalMemoryOwner::KVCache, widened->total_bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(mbuilder.build()), 0);
    auto created = PagedKVInstance::create(authority, *widened, device);
    ASSERT_TRUE(created);
    PagedKVInstance instance = std::move(*created);

    std::unique_ptr<DeviceGraphOrchestrator> grouped_owner;
    PreparedWeightStore grouped_store;
    std::shared_ptr<QwenStandardGraph> grouped_builder;
    make_runner(fx, grouped_owner, grouped_store, grouped_builder, 2);
    PooledKVDecodeBinding binding_a(&instance.pool(), &instance.table(),
        /*request_id=*/321, /*slot=*/0,
        fx.config.n_kv_heads, fx.config.head_dim,
        /*page_tokens=*/8, fx.config.n_layers, /*admitted_issuances=*/16,
        /*persistent_tail=*/false);
    PooledKVDecodeBinding binding_b(&instance.pool(), &instance.table(),
        /*request_id=*/322, /*slot=*/1,
        fx.config.n_kv_heads, fx.config.head_dim,
        /*page_tokens=*/8, fx.config.n_layers, /*admitted_issuances=*/16,
        /*persistent_tail=*/false);
    grouped_owner->setPooledDecodeBinding(&binding_a);
    grouped_owner->setPooledDecodePositions({0, 0});
    ASSERT_OK("pooled A prefill", grouped_owner->forward(prompt_a.data(), 3, 1) != nullptr);
    grouped_owner->setPooledDecodeBinding(nullptr);
    grouped_owner->setPooledDecodeBinding(&binding_b);
    grouped_owner->setPooledDecodePositions({0, 0});
    ASSERT_OK("pooled B prefill", grouped_owner->forward(prompt_b.data(), 4, 1) != nullptr);
    grouped_owner->setPooledDecodeBinding(nullptr);
    grouped_owner->setPooledDecodeGroup({&binding_a, &binding_b});
    // Shrink the host cursor vector BELOW the batch: the grouped gate must
    // refuse (indexed access would be out of range).
    grouped_owner->setPooledDecodePositions({3});
    const int32_t batch[2] = {2, 1};
    EXPECT_EQ(grouped_owner->forward(batch, 1, 2), nullptr)
        << "grouped forward must refuse with a short host cursor vector";
}

/** @test P2 cancellation under a running pooled step + exact borrower
 * retirement: a request cancelled between steps publishes nothing new, its
 * accepted-work envelope and tail frames retire to the pool, and a resolved
 * borrower of its LAST published snapshot stays valid (reads the exact bytes)
 * until it releases — the pool never reuses a frame a borrower still pins.
 * A replacement request on the SAME instance then runs cleanly. */
TEST(PooledDenseDecode, RunnerCancellationRetiresBorrowersExactly)
{
    const auto device = DeviceId::cpu();
    const std::array<int32_t, 3> prompt{1, 3, 5};
    const std::array<int32_t, 4> decode{2, 4, 6, 8};

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

    auto make_instance = [&](PagedKVInstance &out) -> bool
    {
        PagedKVPoolBudget budget;
        budget.remaining_budget_bytes = 3 * 1024 * 1024;
        budget.frame_bytes = 4 * 1024;
        budget.request_records = 4;
        budget.max_pages_per_request = 64;
        const auto policy = PooledKVDecodeBinding::requestPolicy(/*page_tokens=*/8,
            /*layers=*/1, /*admitted_issuances=*/8, /*readers_per_bank=*/4);
        auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget, policy, device);
        if (!plan) return false;
        // Two persistent-tail bindings: 2 banks/layer x 2 bindings + headroom
        // >= layers x issuances x 2. Widen the table explicitly.
        auto table_geom = plan->table_geometry;
        table_geom.bank_count = 2 * table_geom.bank_count;
        table_geom.issuance_headroom = 4 * table_geom.issuance_headroom;
        table_geom.issuance_budget = 2 * table_geom.issuance_headroom;
        auto widened = std::make_shared<PagedKVInstancePlan>(*plan);
        widened->table_geometry = table_geom;
        widened->total_bytes = PhysicalPagePool::physicalBytes(plan->pool_geometry, device) +
            CPUFP32PagedKVTable::physicalBytes(table_geom, device);
        PhysicalMemoryPlanBuilder mb;
        mb.add({.world_rank=0, .device=device,
            .total_bytes=widened->total_bytes,
            .admission_available_bytes=widened->total_bytes},
            PhysicalMemoryOwner::KVCache, widened->total_bytes);
        auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(mb.build()), 0);
        auto created = PagedKVInstance::create(authority, *widened, device);
        if (!created) return false;
        out = std::move(*created);
        return true;
    };

    test::TinyQwenForwardFixture fx(device, KVCachePrecision::FP32);
    fx.config.mtp.enabled = false; fx.config.mtp.graph_capacity_draft_tokens = 0;
    fx.config.max_seq_len = 16;

    PagedKVInstance instance;
    ASSERT_TRUE(make_instance(instance));
    std::unique_ptr<DeviceGraphOrchestrator> owner;
    PreparedWeightStore store;
    std::shared_ptr<QwenStandardGraph> builder;
    make_runner(fx, owner, store, builder);
    PooledKVDecodeBinding binding(&instance.pool(), &instance.table(),
        /*request_id=*/201, /*slot=*/0,
        fx.config.n_kv_heads, fx.config.head_dim, /*page_tokens=*/8, /*layers=*/1,
        /*admitted_issuances=*/8, /*persistent_tail=*/true);
    owner->setPooledDecodeBinding(&binding);
    ASSERT_OK("prefill", owner->forward(prompt.data(), 3, 1) != nullptr);
    ASSERT_OK("decode 0", owner->forward(&decode[0], 1, 1) != nullptr);
    ASSERT_OK("decode 1", owner->forward(&decode[1], 1, 1) != nullptr);

    // A real borrower of the LAST published snapshot: resolve the live ref.
    auto *prior = binding.priorFor(0);
    ASSERT_NE(prior, nullptr);
    ASSERT_TRUE(prior->ref.valid());
    auto borrowed = instance.table().resolveSnapshotRef(prior->ref);
    ASSERT_TRUE(borrowed.has_value());
    auto borrowed_view = borrowed->view();
    const int borrowed_rows = borrowed_view.rows();
    EXPECT_EQ(borrowed_rows, 5);

    // Cancel the request; the last publication stays alive while the borrower
    // holds it (frames retire after the last pin).
    binding.cancelRequest();
    EXPECT_TRUE(binding.cancelled());
    EXPECT_FALSE(binding.hasLivePublication());
    EXPECT_EQ(binding.remainingIssuances(), 0u);
    // The borrowing snapshot is still readable byte-exact.
    EXPECT_EQ(borrowed->view().rows(), borrowed_rows);
    for (int row = 0; row < borrowed_rows; ++row)
        (void)borrowed_view.key(row, 0);
    // A NEW request on the SAME instance runs cleanly after the cancellation.
    std::unique_ptr<DeviceGraphOrchestrator> owner2;
    PreparedWeightStore store2;
    std::shared_ptr<QwenStandardGraph> builder2;
    make_runner(fx, owner2, store2, builder2);
    PooledKVDecodeBinding binding2(&instance.pool(), &instance.table(),
        /*request_id=*/202, /*slot=*/1,
        fx.config.n_kv_heads, fx.config.head_dim, /*page_tokens=*/8, /*layers=*/1,
        /*admitted_issuances=*/8, /*persistent_tail=*/true);
    owner2->setPooledDecodeBinding(&binding2);
    ASSERT_OK("replacement prefill", owner2->forward(prompt.data(), 3, 1) != nullptr);
    ASSERT_OK("replacement decode", owner2->forward(&decode[2], 1, 1) != nullptr);
    EXPECT_EQ(binding2.priorFor(0)->snapshot.view().rows(), 4);
    // Release the borrower; the replacement's own writes remain valid.
    borrowed.reset();
}

/** @test P2 GENUINELY grouped pooled decode: N requests, ONE graph. Two
 * bindings on one shared pool/table are attached to ONE graph via
 * setPooledDecodeGroup; a single forward(tokens, 1, 2) appends both rows'
 * K/V to their OWN publications and computes both rows against their own
 * exact refs through ONE grouped pooled kernel entry (no stage-level
 * row-replay loop, no per-request graphs). Every decode row's logits are
 * byte-compared against that request's own uninterrupted ring control at the
 * same position, and each publication grows independently. */
TEST(PooledDenseDecode, GroupedPooledDecodeTwoRequestsOneGraph)
{
    const auto device = DeviceId::cpu();
    const std::array<int32_t, 3> prompt_a{1, 5, 3};
    const std::array<int32_t, 4> prompt_b{2, 3, 5};
    const std::array<int32_t, 6> decode_a{2, 4, 6, 8, 10, 12};
    const std::array<int32_t, 5> decode_b{1, 3, 5, 7, 9};

    auto make_runner = [&](test::TinyQwenForwardFixture &fx,
                           std::unique_ptr<DeviceGraphOrchestrator> &owner_out,
                           PreparedWeightStore &store_out,
                           std::shared_ptr<QwenStandardGraph> &builder_out,
                           int batch)
    {
        auto builder = std::make_shared<QwenStandardGraph>(fx.config, fx.mpi);
        auto deps = test::cpuInferenceFixtureDependencies(builder, fx.mpi, batch,
            fx.config.max_seq_len, fx.config.max_seq_len,
            test::cpuFixtureTensorInventory(fx.modelWeights(), fx.config.n_layers));
        owner_out = DeviceGraphOrchestrator::createForTest(std::move(deps));
        auto &runner = *owner_out;
        if (!runner.initializeInferenceStateFromArena(batch, fx.config.max_seq_len, device))
            throw std::runtime_error("initializeInferenceStateFromArena failed");
        runner.setWeights(fx.modelWeights());
        test::prepareDenseForwardWeights(runner, *builder, store_out, device);
        builder_out = std::move(builder);
    };

    test::TinyQwenForwardFixture fx(device, KVCachePrecision::FP32);
    fx.config.mtp.enabled = false; fx.config.mtp.graph_capacity_draft_tokens = 0;
    fx.config.max_seq_len = 16;

    // RING SLOT ORACLE: one 2-slot runner; every decode step advances BOTH
    // slots in ONE batched serving graph (row 0 = A, row 1 = B). This gives a
    // same-M=2 graph shape with the ORDINARY ring cache — the only difference
    // vs the pooled group is the KV source (pooled pages vs ring slots), so a
    // byte-exact match proves the pooled grouped machinery produced identical
    // results for a genuinely grouped one-graph decode.
    std::unique_ptr<DeviceGraphOrchestrator> ring_owner;
    PreparedWeightStore ring_store;
    std::shared_ptr<QwenStandardGraph> ring_builder;
    make_runner(fx, ring_owner, ring_store, ring_builder, 2);
    // Ring prefills through the SAME slot batched API (slot 0 = A, slot 1 = B):
    // the ordinary forward() would place both in request 0.
    ASSERT_OK("ring A prefill", ring_owner->forwardServingRows(
        {0}, {std::vector<int>(prompt_a.begin(), prompt_a.end())}));
    ASSERT_OK("ring B prefill", ring_owner->forwardServingRows(
        {1}, {std::vector<int>(prompt_b.begin(), prompt_b.end())}));

    // Pooled grouped: ONE shared instance, TWO bindings, batch=2 one-graph.
    PagedKVPoolBudget budget;
    budget.remaining_budget_bytes = 3 * 1024 * 1024;
    budget.frame_bytes = 4 * 1024;
    budget.request_records = 4;
    budget.max_pages_per_request = 64;
    const auto policy = PooledKVDecodeBinding::requestPolicy(/*page_tokens=*/8,
        /*layers=*/fx.config.n_layers, /*admitted_issuances=*/8, /*readers_per_bank=*/4);
    auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget, policy, device);
    ASSERT_TRUE(plan);
    auto table_geom = plan->table_geometry;
    table_geom.bank_count = 2 * table_geom.bank_count;
    table_geom.issuance_headroom = 4 * table_geom.issuance_headroom;
    table_geom.issuance_budget = 2 * table_geom.issuance_headroom;
    auto widened = std::make_shared<PagedKVInstancePlan>(*plan);
    widened->table_geometry = table_geom;
    widened->total_bytes = PhysicalPagePool::physicalBytes(plan->pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geom, device);
    PhysicalMemoryPlanBuilder mbuilder;
    mbuilder.add({.world_rank=0, .device=device,
        .total_bytes=widened->total_bytes,
        .admission_available_bytes=widened->total_bytes},
        PhysicalMemoryOwner::KVCache, widened->total_bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(mbuilder.build()), 0);
    auto created = PagedKVInstance::create(authority, *widened, device);
    ASSERT_TRUE(created);
    PagedKVInstance instance = std::move(*created);

    std::unique_ptr<DeviceGraphOrchestrator> grouped_owner;
    PreparedWeightStore grouped_store;
    std::shared_ptr<QwenStandardGraph> grouped_builder;
    make_runner(fx, grouped_owner, grouped_store, grouped_builder, 2);
    PooledKVDecodeBinding binding_a(&instance.pool(), &instance.table(),
        /*request_id=*/301, /*slot=*/0,
        fx.config.n_kv_heads, fx.config.head_dim,
        /*page_tokens=*/8, fx.config.n_layers, /*admitted_issuances=*/8);
    PooledKVDecodeBinding binding_b(&instance.pool(), &instance.table(),
        /*request_id=*/302, /*slot=*/1,
        fx.config.n_kv_heads, fx.config.head_dim,
        /*page_tokens=*/8, fx.config.n_layers, /*admitted_issuances=*/8);
    // Prefill each request through its own single-binding forward (identical
    // M=1 math to the ring prefills above, no cross-request pooling yet).
    grouped_owner->setPooledDecodeBinding(&binding_a);
    grouped_owner->setPooledDecodePositions({0, 0});
    ASSERT_OK("pooled A prefill", grouped_owner->forward(prompt_a.data(), 3, 1) != nullptr);
    grouped_owner->setPooledDecodeBinding(nullptr);
    grouped_owner->setPooledDecodeBinding(&binding_b);
    grouped_owner->setPooledDecodePositions({0, 0});
    ASSERT_OK("pooled B prefill", grouped_owner->forward(prompt_b.data(), 4, 1) != nullptr);
    grouped_owner->setPooledDecodeBinding(nullptr);
    // GENUINELY GROUPED decode: both bindings, batch=2, ONE graph per step.
    grouped_owner->setPooledDecodeGroup({&binding_a, &binding_b});

    const std::size_t vocab = static_cast<std::size_t>(fx.config.vocab_size);
    for (int step = 0; step < static_cast<int>(decode_a.size()); ++step)
    {
        const bool b_live = step < static_cast<int>(decode_b.size());
        grouped_owner->setPooledDecodePositions({3 + step, 4 + step});
        // Ring slot oracle: advance A (and B when live) in one batched graph.
        std::vector<int> ring_slots{0};
        std::vector<std::vector<int>> ring_rows{{std::vector<int>{decode_a[step]}}};
        if (b_live)
        {
            ring_slots.push_back(1);
            ring_rows.push_back({std::vector<int>{decode_b[step]}});
        }
        ASSERT_OK("ring batched", ring_owner->forwardServingRows(ring_slots, ring_rows));
        const float *ring_logits_a = ring_owner->servingRowLogits(0);
        ASSERT_NE(ring_logits_a, nullptr) << "ring A logits " << step;
        const float *grouped_row_a = nullptr;
        if (b_live)
        {
            const float *ring_logits_b = ring_owner->servingRowLogits(1);
            ASSERT_NE(ring_logits_b, nullptr) << "ring B logits " << step;
            int32_t batch_tokens[2] = {decode_a[step], decode_b[step]};
            const auto *grouped = grouped_owner->forward(batch_tokens, 1, 2);
            ASSERT_NE(grouped, nullptr) << "grouped forward " << step;
            grouped_row_a = grouped;
            const float *grouped_row_b = grouped + vocab;
            EXPECT_EQ(std::memcmp(ring_logits_b, grouped_row_b, vocab * sizeof(float)), 0)
                << "grouped request B diverged at step " << step;
        }
        else
        {
            grouped_owner->setPooledDecodeGroup({});
            grouped_owner->setPooledDecodeBinding(&binding_a);
            // A has 3+5=8 rows after the grouped steps; its cursor is 8.
            grouped_owner->setPooledDecodePositions({8, 9});
            const auto *grouped = grouped_owner->forward(&decode_a[step], 1, 1);
            ASSERT_NE(grouped, nullptr);
            grouped_row_a = grouped;
            grouped_owner->setPooledDecodeBinding(nullptr);
            grouped_owner->setPooledDecodeGroup({&binding_a, &binding_b});
        }
        EXPECT_EQ(std::memcmp(ring_logits_a, grouped_row_a, vocab * sizeof(float)), 0)
            << "grouped request A diverged at step " << step;
    }
    EXPECT_EQ(binding_a.priorFor(0)->snapshot.view().rows(), 9);
    EXPECT_EQ(binding_b.priorFor(0)->snapshot.view().rows(), 9);
}

/** @test P2 tail economy measurement: the persistent in-place private tail
 * copies strictly fewer bytes per decode step than the COW extension (the
 * prior tail rows are never copied on the in-place path). This is the
 * producer-side measurement that decides the persistent-tail DEFAULT before
 * P2 exit; both paths stay byte-identical to the ring control (asserted in
 * RunnerInPlacePrivateTailMatchesControl and the COW matrix). */
TEST(PooledDenseDecode, TailCopyBytesMeasurement)
{
    const auto device = DeviceId::cpu();
    // Enable the pooled_kv_decode PerfStats domain (immutable policy is
    // process-start state; re-publish for the measurement scope).
    if (const char *prior = std::getenv("LLAMINAR_PERF_STATS_FILTER"))
        (void)prior;
    setenv("LLAMINAR_PERF_STATS_FILTER", "pooled_kv_decode", 1);
    setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
    PerfStatsCollector::reloadConfigurationFromEnvironment();
    const std::array<int32_t, 3> prompt{1, 5, 3};
    std::vector<int32_t> decode;
    for (int i = 0; i < 40; ++i)
        decode.push_back(2 + i * 2);

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
    auto make_instance = [&](PagedKVInstance &out) -> bool
    {
        PagedKVPoolBudget budget;
        budget.remaining_budget_bytes = 4 * 1024 * 1024;
        budget.frame_bytes = 4 * 1024;
        budget.request_records = 4;
        budget.max_pages_per_request = 128;
        const auto policy = PooledKVDecodeBinding::requestPolicy(/*page_tokens=*/8,
            /*layers=*/1, /*admitted_issuances=*/96, /*readers_per_bank=*/4);
        auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget, policy, device);
        if (!plan) return false;
        auto table_geom = plan->table_geometry;
        table_geom.bank_count = 2 * table_geom.bank_count;
        table_geom.issuance_headroom = 4 * table_geom.issuance_headroom;
        table_geom.issuance_budget = 2 * table_geom.issuance_headroom;
        auto widened = std::make_shared<PagedKVInstancePlan>(*plan);
        widened->table_geometry = table_geom;
        widened->total_bytes = PhysicalPagePool::physicalBytes(plan->pool_geometry, device) +
            CPUFP32PagedKVTable::physicalBytes(table_geom, device);
        PhysicalMemoryPlanBuilder mb;
        mb.add({.world_rank=0, .device=device,
            .total_bytes=widened->total_bytes,
            .admission_available_bytes=widened->total_bytes},
            PhysicalMemoryOwner::KVCache, widened->total_bytes);
        auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(mb.build()), 0);
        auto created = PagedKVInstance::create(authority, *widened, device);
        if (!created) return false;
        out = std::move(*created);
        return true;
    };

    test::TinyQwenForwardFixture fx(device, KVCachePrecision::FP32);
    fx.config.mtp.enabled = false; fx.config.mtp.graph_capacity_draft_tokens = 0;
    fx.config.max_seq_len = 64;

    // IN-PLACE measurement: persistent_tail => in_place_tail_bytes recorded.
    PagedKVInstance inplace_instance;
    ASSERT_TRUE(make_instance(inplace_instance));
    std::unique_ptr<DeviceGraphOrchestrator> inplace_owner;
    PreparedWeightStore inplace_store;
    std::shared_ptr<QwenStandardGraph> inplace_builder;
    make_runner(fx, inplace_owner, inplace_store, inplace_builder);
    PooledKVDecodeBinding inplace_binding(&inplace_instance.pool(), &inplace_instance.table(),
        /*request_id=*/501, /*slot=*/0,
        fx.config.n_kv_heads, fx.config.head_dim, /*page_tokens=*/8, fx.config.n_layers,
        /*admitted_issuances=*/96, /*persistent_tail=*/true);
    inplace_owner->setPooledDecodeBinding(&inplace_binding);
    ASSERT_OK("in-place prefill", inplace_owner->forward(prompt.data(), 3, 1) != nullptr);
    for (const auto token : decode)
        ASSERT_OK("in-place decode", inplace_owner->forward(&token, 1, 1) != nullptr);

    // COW measurement: persistent_tail=false (default) => cow_tail_bytes.
    PagedKVInstance cow_instance;
    ASSERT_TRUE(make_instance(cow_instance));
    std::unique_ptr<DeviceGraphOrchestrator> cow_owner;
    PreparedWeightStore cow_store;
    std::shared_ptr<QwenStandardGraph> cow_builder;
    make_runner(fx, cow_owner, cow_store, cow_builder);
    PooledKVDecodeBinding cow_binding(&cow_instance.pool(), &cow_instance.table(),
        /*request_id=*/502, /*slot=*/0,
        fx.config.n_kv_heads, fx.config.head_dim, /*page_tokens=*/8, fx.config.n_layers,
        /*admitted_issuances=*/96, /*persistent_tail=*/false);
    cow_owner->setPooledDecodeBinding(&cow_binding);
    ASSERT_OK("cow prefill", cow_owner->forward(prompt.data(), 3, 1) != nullptr);
    for (const auto token : decode)
        ASSERT_OK("cow decode", cow_owner->forward(&token, 1, 1) != nullptr);

    // Both decoded the same 3+40 rows; byte-identity is covered elsewhere.
    EXPECT_EQ(inplace_binding.priorFor(0)->snapshot.view().rows(), 3 + 40);
    EXPECT_EQ(cow_binding.priorFor(0)->snapshot.view().rows(), 3 + 40);

    // Aggregate the per-step bytes copied.
    const auto inplace_records = PerfStatsCollector::snapshot({"pooled_kv_decode"});
    double inplace_bytes = 0.0, cow_bytes = 0.0;
    int inplace_steps = 0, cow_steps = 0;
    for (const auto &record : inplace_records)
    {
        if (record.domain != "pooled_kv_decode") continue;
        if (record.name == "in_place_tail_bytes_per_step" && record.count > 0)
        {
            // Snapshot aggregates: record.value is the SUM over record.count
            // observations (never multiply again).
            inplace_bytes += record.value;
            inplace_steps += static_cast<int>(record.count);
        }
        if (record.name == "cow_tail_bytes_per_step" && record.count > 0)
        {
            cow_bytes += record.value;
            cow_steps += static_cast<int>(record.count);
        }
    }
    EXPECT_GT(inplace_steps, 0) << "in-place tail counters were not recorded";
    EXPECT_GT(cow_steps, 0) << "COW tail counters were not recorded";
    if (inplace_steps > 0 && cow_steps > 0)
    {
        const double inplace_mean = inplace_bytes / inplace_steps;
        const double cow_mean = cow_bytes / cow_steps;
        std::cerr << "[tail-measure] in_place_steps=" << inplace_steps
                  << " cow_steps=" << cow_steps
                  << " in_place_mean=" << inplace_mean
                  << " cow_mean=" << cow_mean << std::endl;
        // The in-place path never re-copies the prior tail rows; with many
        // partial-tail steps it must be strictly smaller than COW.
        EXPECT_LT(inplace_mean, cow_mean)
            << "in-place tail must copy fewer bytes per step than COW";
        // Economic decision evidence: the in-place path writes only the fresh
        // rows for that step (row_stride = kv_heads*head_dim*8 bytes);
        // COW writes the whole prior tail page plus the fresh rows.
        const double row_stride = static_cast<double>(
            fx.config.n_kv_heads * fx.config.head_dim * 2 * sizeof(float));
        EXPECT_LE(inplace_mean, row_stride * 1.05)
            << "in-place tail must copy only the fresh row(s) (stride=" << row_stride << ")";
        EXPECT_GE(cow_mean, row_stride)
            << "COW tail must copy at least the fresh row(s)";
    }
}

/** @test P3 M1: shared-prefix attach deduplicates immutable frames.
 * A fresh binding attaches an exact prefix of another live request's
 * publication as SHARED frames (full pages adopted verbatim, one private
 * COW page clipped to the partial remainder), then both requests diverge
 * and continue — each stays byte-exact against its own uninterrupted
 * control. The source keeps decoding around the shared frames, proving
 * the attach never mutates the source and the source never mutates the
 * attached prefix. */
TEST(PooledDenseDecode, SharedPrefixAttachDeduplicatesAndDiverges)
{
    const auto device = DeviceId::cpu();
    const std::array<int32_t, 10> prompt_src{1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    const std::array<int32_t, 2> src_after{2, 4};
    const std::array<int32_t, 3> dst_after{6, 8, 10};

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

    test::TinyQwenForwardFixture fx(device, KVCachePrecision::FP32);
    fx.config.mtp.enabled = false; fx.config.mtp.graph_capacity_draft_tokens = 0;
    fx.config.max_seq_len = 24;

    auto make_instance = [&](PagedKVInstance &out) -> bool
    {
        PagedKVPoolBudget budget;
        budget.remaining_budget_bytes = 4 * 1024 * 1024;
        budget.frame_bytes = 4 * 1024;
        budget.request_records = 8;
        budget.max_pages_per_request = 64;
        const auto policy = PooledKVDecodeBinding::requestPolicy(/*page_tokens=*/8,
            /*layers=*/fx.config.n_layers, /*admitted_issuances=*/10, /*readers_per_bank=*/4);
        auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget, policy, device);
        if (!plan) return false;
        auto table_geom = plan->table_geometry;
        table_geom.bank_count = 2 * table_geom.bank_count;
        table_geom.issuance_headroom = 4 * table_geom.issuance_headroom;
        table_geom.issuance_budget = 2 * table_geom.issuance_headroom;
        auto pool_geom = plan->pool_geometry;
        pool_geom.frames = 64 + 8 * pool_geom.frames;
        pool_geom.request_records = 8;
        auto widened = std::make_shared<PagedKVInstancePlan>(*plan);
        widened->pool_geometry = pool_geom;
        widened->table_geometry = table_geom;
        widened->total_bytes = PhysicalPagePool::physicalBytes(pool_geom, device) +
            CPUFP32PagedKVTable::physicalBytes(table_geom, device);
        PhysicalMemoryPlanBuilder mb;
        mb.add({.world_rank=0, .device=device,
            .total_bytes=widened->total_bytes,
            .admission_available_bytes=widened->total_bytes},
            PhysicalMemoryOwner::KVCache, widened->total_bytes);
        auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(mb.build()), 0);
        auto created = PagedKVInstance::create(authority, *widened, device);
        if (!created) return false;
        out = std::move(*created);
        return true;
    };

    // SOURCE: prefill 10 (page 8 full + 2 partial), then 2 decodes = 12 rows.
    PagedKVInstance src_instance;
    ASSERT_TRUE(make_instance(src_instance));
    std::unique_ptr<DeviceGraphOrchestrator> src_owner;
    PreparedWeightStore src_store;
    std::shared_ptr<QwenStandardGraph> src_builder;
    make_runner(fx, src_owner, src_store, src_builder);
    PooledKVDecodeBinding src(&src_instance.pool(), &src_instance.table(),
        /*request_id=*/401, /*slot=*/0,
        fx.config.n_kv_heads, fx.config.head_dim,
        /*page_tokens=*/8, fx.config.n_layers, /*admitted_issuances=*/10,
        /*persistent_tail=*/false);
    src_owner->setPooledDecodeBinding(&src);
    ASSERT_OK("source prefill", src_owner->forward(prompt_src.data(), 10, 1) != nullptr);
    ASSERT_OK("source decode 0", src_owner->forward(&src_after[0], 1, 1) != nullptr);
    ASSERT_OK("source decode 1", src_owner->forward(&src_after[1], 1, 1) != nullptr);
    EXPECT_EQ(src.priorFor(0)->snapshot.view().rows(), 12);
    const auto &src_pages = src.priorFor(0)->snapshot.view().pages();
    ASSERT_EQ(static_cast<int>(src_pages.size()), 2) << "10-row prefill + 2 decode = page 8 + page 4";

    // DESTINATION: attached prefix of 9 rows (page 8 full + 1 row COW).
    // This is the SAME instance authority (dedup only within one authority).
    PagedKVInstance dst_instance;
    ASSERT_TRUE(make_instance(dst_instance));
    std::unique_ptr<DeviceGraphOrchestrator> dst_owner;
    PreparedWeightStore dst_store;
    std::shared_ptr<QwenStandardGraph> dst_builder;
    make_runner(fx, dst_owner, dst_store, dst_builder);
    // DELIBERATE NEGATIVE: attach across authorities must refuse before any
    // byte is shared (the pooled authority is the isolation domain).
    {
        PooledKVDecodeBinding foreign(&dst_instance.pool(), &dst_instance.table(),
            /*request_id=*/402, /*slot=*/0,
            fx.config.n_kv_heads, fx.config.head_dim,
            /*page_tokens=*/8, fx.config.n_layers, /*admitted_issuances=*/10,
            /*persistent_tail=*/false);
        EXPECT_FALSE(foreign.attachSharedPrefix(src, 9))
            << "cross-authority prefix attach must refuse";
        EXPECT_FALSE(foreign.hasLivePublication());
    }
    // Same-authority attach, mid-page prefix (9 = 8 full + 1 partial).
    PooledKVDecodeBinding dst(&src_instance.pool(), &src_instance.table(),
        /*request_id=*/403, /*slot=*/1,
        fx.config.n_kv_heads, fx.config.head_dim,
        /*page_tokens=*/8, fx.config.n_layers, /*admitted_issuances=*/10,
        /*persistent_tail=*/false);
    ASSERT_TRUE(dst.attachSharedPrefix(src, 9));
    EXPECT_EQ(dst.cursorEpoch(), 1u);
    EXPECT_EQ(dst.nextPosition(), 9);
    const auto &ancestry = dst.prefixAncestryRefs();
    ASSERT_EQ(ancestry.size(), 1u);
    EXPECT_EQ(ancestry[0].request_id, 401u);
    EXPECT_EQ(ancestry[0].slot, 0);
    // Register the destination binding on its runner (decode below reads
    // the attached pooled publication, never the ring).
    dst_owner->setPooledDecodeBinding(&dst);
    // Dedup proof: page 0 is a SHARED reference to the source's frame; the
    // partial page is a PRIVATE COW clip (never the whole shared page).
    const auto &dst_pages = dst.priorFor(0)->snapshot.view().pages();
    ASSERT_EQ(static_cast<int>(dst_pages.size()), 2);
    EXPECT_TRUE(dst_pages[0].sharedBacked()) << "full prefix pages are deduplicated shared frames";
    EXPECT_FALSE(dst_pages[1].sharedBacked()) << "partial prefix remainder is a private COW clip";
    EXPECT_EQ(dst_pages[1].rows(), 1);
    // Byte-identity of the attached prefix vs the source's matching rows.
    {
        const auto src_view = src.priorFor(0)->snapshot.view();
        const auto dst_view = dst.priorFor(0)->snapshot.view();
        for (int row = 0; row < 9; ++row)
        {
            for (int h = 0; h < fx.config.n_kv_heads; ++h)
            {
                EXPECT_EQ(std::memcmp(dst_view.key(row, h), src_view.key(row, h),
                        static_cast<std::size_t>(fx.config.head_dim) * sizeof(float)), 0)
                    << "attached prefix K diverged at row " << row;
                EXPECT_EQ(std::memcmp(dst_view.value(row, h), src_view.value(row, h),
                        static_cast<std::size_t>(fx.config.head_dim) * sizeof(float)), 0)
                    << "attached prefix V diverged at row " << row;
            }
        }
    }
    // Negative: attach again on a destination with live history refuses.
    EXPECT_FALSE(dst.attachSharedPrefix(src, 9))
        << "attach into a destination with live history must refuse";
    // Negative: an out-of-range prefix refuses; the destination is unchanged.
    EXPECT_FALSE(dst.attachSharedPrefix(src, 13));
    EXPECT_EQ(dst.nextPosition(), 9);

    // DIVERGENCE: source decodes 2 more; destination decodes 3 different
    // tokens. Each request is compared to its OWN uninterrupted control.
    PagedKVInstance ctrl_src_instance;
    ASSERT_TRUE(make_instance(ctrl_src_instance));
    std::unique_ptr<DeviceGraphOrchestrator> ctrl_src_owner;
    PreparedWeightStore ctrl_src_store;
    std::shared_ptr<QwenStandardGraph> ctrl_src_builder;
    make_runner(fx, ctrl_src_owner, ctrl_src_store, ctrl_src_builder);
    PooledKVDecodeBinding ctrl_src(&ctrl_src_instance.pool(), &ctrl_src_instance.table(),
        /*request_id=*/404, /*slot=*/0,
        fx.config.n_kv_heads, fx.config.head_dim,
        /*page_tokens=*/8, fx.config.n_layers, /*admitted_issuances=*/10,
        /*persistent_tail=*/false);
    ctrl_src_owner->setPooledDecodeBinding(&ctrl_src);
    ASSERT_OK("control source prefill", ctrl_src_owner->forward(prompt_src.data(), 10, 1) != nullptr);
    // Control for DST: an uninterrupted request with the same attach prefix
    // tokens (prompt 1..9) then the same divergence tokens.
    PagedKVInstance ctrl_dst_instance;
    ASSERT_TRUE(make_instance(ctrl_dst_instance));
    std::unique_ptr<DeviceGraphOrchestrator> ctrl_dst_owner;
    PreparedWeightStore ctrl_dst_store;
    std::shared_ptr<QwenStandardGraph> ctrl_dst_builder;
    make_runner(fx, ctrl_dst_owner, ctrl_dst_store, ctrl_dst_builder);
    PooledKVDecodeBinding ctrl_dst(&ctrl_dst_instance.pool(), &ctrl_dst_instance.table(),
        /*request_id=*/405, /*slot=*/0,
        fx.config.n_kv_heads, fx.config.head_dim,
        /*page_tokens=*/8, fx.config.n_layers, /*admitted_issuances=*/10,
        /*persistent_tail=*/false);
    ctrl_dst_owner->setPooledDecodeBinding(&ctrl_dst);
    ASSERT_OK("control dst prefill", ctrl_dst_owner->forward(prompt_src.data(), 9, 1) != nullptr);

    const std::size_t vocab = static_cast<std::size_t>(fx.config.vocab_size);
    // Source control: replicate the source's COMPLETE history first (prefill
    // already done; the 2 pre-attach decode tokens), then compare the same
    // step's logits for the divergence tokens.
    for (int t = 0; t < static_cast<int>(src_after.size()); ++t)
    {
        ASSERT_NE(ctrl_src_owner->forward(&src_after[t], 1, 1), nullptr)
            << "control source pre-decode " << t;
    }
    for (int t = 0; t < static_cast<int>(src_after.size()); ++t)
    {
        const float *src_logits = src_owner->forward(&src_after[t], 1, 1);
        ASSERT_NE(src_logits, nullptr);
        const float *ctrl_logits = ctrl_src_owner->forward(&src_after[t], 1, 1);
        ASSERT_NE(ctrl_logits, nullptr);
        EXPECT_EQ(std::memcmp(src_logits, ctrl_logits, vocab * sizeof(float)), 0)
            << "source diverged after shared-prefix attach at token " << t;
    }
    // Destination control: prefill the same 9-token prefix, then the same
    // 3 divergence tokens — an uninterrupted request with the same history.
    const float *ctrl_dst_logits = nullptr;
    for (int t = 0; t < static_cast<int>(dst_after.size()); ++t)
    {
        ctrl_dst_logits = ctrl_dst_owner->forward(&dst_after[t], 1, 1);
        ASSERT_NE(ctrl_dst_logits, nullptr);
    }
    const float *dst_logits = nullptr;
    for (int t = 0; t < static_cast<int>(dst_after.size()); ++t)
    {
        dst_logits = dst_owner->forward(&dst_after[t], 1, 1);
        ASSERT_NE(dst_logits, nullptr);
    }
    ASSERT_NE(ctrl_dst_logits, nullptr);
    EXPECT_EQ(std::memcmp(dst_logits, ctrl_dst_logits, vocab * sizeof(float)), 0)
        << "attached-prefix request diverged from its uninterrupted control";
    const auto src_after_view = src.priorFor(0)->snapshot.view();
    const auto dst_after_view = dst.priorFor(0)->snapshot.view();
    for (int row = 0; row < 8; ++row)
    {
        EXPECT_EQ(std::memcmp(dst_after_view.key(row, 0), src_after_view.key(row, 0),
                static_cast<std::size_t>(fx.config.head_dim) * sizeof(float)), 0)
            << "shared full page mutated after source continued at row " << row;
    }
}
