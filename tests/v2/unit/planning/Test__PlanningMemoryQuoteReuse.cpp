/**
 * @file Test__PlanningMemoryQuoteReuse.cpp
 * @brief Device-free exact logical quote identity, source scope and deterministic replay.
 */
#include "planning/PlanningMemoryQuoteReuse.h"
#include "../../utils/PlanningGGUFFixture.h"
#include <gtest/gtest.h>
#include <thread>
#include <atomic>
using namespace llaminar2;
/** @test Pure quotes retain all actual geometry fields; lexical scopes cannot leak or alias another source. */
TEST(PlanningMemoryQuoteReuse, FullGeometryIdentityAndSourceLifetimeAreExact)
{
    test::PlanningGGUFFixture file;PlanningModelSource source(file.path());
    const auto &profile=source.metadata().memoryProfile();
    WorkspaceMemoryGeometry geometry;
    geometry.source_tensor_index=source.metadata().tensorIndex();
    geometry.device=DeviceId::rocm(7);geometry.device_compute_units=64;
    geometry.resident_graph_rows=32;geometry.max_context_rows=128;
    const auto canonical=WorkspaceMemoryEstimator::estimate(profile,geometry);
    {
        PlanningMemoryQuoteReuse quotes(source.metadata().tensorIndex());
        for(int replay=0;replay<20;++replay)EXPECT_EQ(PlanningMemoryQuoteReuse::workspace(profile,geometry),canonical);
        EXPECT_EQ(quotes.counts().workspace_builds,1u);EXPECT_EQ(quotes.counts().workspace_hits,19u);
        auto changed=geometry;changed.resident_graph_rows=64;
        EXPECT_EQ(PlanningMemoryQuoteReuse::workspace(profile,changed),WorkspaceMemoryEstimator::estimate(profile,changed));
        changed=geometry;changed.max_context_rows=256;
        EXPECT_EQ(PlanningMemoryQuoteReuse::workspace(profile,changed),WorkspaceMemoryEstimator::estimate(profile,changed));
        changed=geometry;changed.device_compute_units=80;
        EXPECT_EQ(PlanningMemoryQuoteReuse::workspace(profile,changed),WorkspaceMemoryEstimator::estimate(profile,changed));
        changed=geometry;changed.device=DeviceId::rocm(2);
        EXPECT_EQ(PlanningMemoryQuoteReuse::workspace(profile,changed),WorkspaceMemoryEstimator::estimate(profile,changed));
        changed=geometry;changed.qfn_fa_output_activation=QuantizedActivationOperandPrecision::TwoTermQ8;
        EXPECT_EQ(PlanningMemoryQuoteReuse::workspace(profile,changed),WorkspaceMemoryEstimator::estimate(profile,changed));
        EXPECT_EQ(quotes.counts().workspace_builds,6u);
        changed=geometry;changed.planning_policy.cuda_fa2_tile_kv+=1;
        EXPECT_EQ(PlanningMemoryQuoteReuse::workspace(profile,changed),WorkspaceMemoryEstimator::estimate(profile,changed));
        changed=geometry;changed.cpu_execution.maximum_native_row_tile=4;
        EXPECT_EQ(PlanningMemoryQuoteReuse::workspace(profile,changed),WorkspaceMemoryEstimator::estimate(profile,changed));
        EXPECT_EQ(quotes.counts().workspace_builds,8u);
        changed=geometry;changed.total_shards=2;changed.shard_index=1;
        EXPECT_EQ(PlanningMemoryQuoteReuse::workspace(profile,changed),WorkspaceMemoryEstimator::estimate(profile,changed));
        EXPECT_EQ(quotes.counts().workspace_builds,9u);
        changed=geometry;changed.generation_request_capacity=2;
        EXPECT_FALSE(changed==geometry);
        changed=geometry;changed.mtp_target_query_rows=2;
        EXPECT_FALSE(changed==geometry);
        changed=geometry;changed.gdn_live_state_handoff=gdn_live_state_workspace::Geometry{};
        EXPECT_FALSE(changed==geometry); // Invalid nested policy cannot alias the previously sealed input.
        std::atomic<bool> isolated=false;
        std::thread peer([&]{
            isolated=!PlanningMemoryQuoteReuse::sourceIndex(profile);
            PlanningMemoryQuoteReuse own_thread(source.metadata().tensorIndex());
            EXPECT_EQ(own_thread.counts().workspace_requests,0u);
        });
        peer.join();EXPECT_TRUE(isolated);
        EXPECT_THROW(PlanningMemoryQuoteReuse(source.metadata().tensorIndex()),std::invalid_argument);
        test::PlanningGGUFFixture other;PlanningModelSource foreign(other.path());
        EXPECT_THROW(PlanningMemoryQuoteReuse::sourceIndex(foreign.metadata().memoryProfile()),std::invalid_argument);
        changed=geometry;changed.source_tensor_index=foreign.metadata().tensorIndex();
        EXPECT_THROW(PlanningMemoryQuoteReuse::workspace(profile,changed),std::invalid_argument);
        EXPECT_EQ(PlanningMemoryQuoteReuse::workspace(profile,geometry),canonical); // Nested/foreign rejection preserves this live owner.
    }
    EXPECT_FALSE(PlanningMemoryQuoteReuse::sourceIndex(profile));
    try { PlanningMemoryQuoteReuse exceptional(source.metadata().tensorIndex()); throw std::runtime_error("recorded failure"); }
    catch(const std::runtime_error&) {}
    EXPECT_FALSE(PlanningMemoryQuoteReuse::sourceIndex(profile));
    { PlanningMemoryQuoteReuse replay(source.metadata().tensorIndex());
      EXPECT_EQ(PlanningMemoryQuoteReuse::workspace(profile,geometry),canonical);
      EXPECT_EQ(replay.counts().workspace_builds,1u); }
}

/** @test Arena and prepared service quotes replay canonical bytes without retaining admission or source identity across searches. */
TEST(PlanningMemoryQuoteReuse, ProjectionArenaAndPreparedServiceReuseExactOriginalInputs)
{
    test::PlanningGGUFFixture file(true,false);
    PlanningModelSource source(file.path());
    const auto &profile=source.metadata().memoryProfile();
    WorkspaceMemoryGeometry geometry;
    geometry.source_tensor_index=source.metadata().tensorIndex();
    geometry.device=DeviceId::rocm(7);
    geometry.device_compute_units=64;
    geometry.resident_graph_rows=32;
    geometry.max_context_rows=128;
    geometry.first_layer=0;
    geometry.last_layer=source.metadata().mainLayerCount()-1;
    geometry.apportioned_routed_experts=true;
    const auto canonical_service=WorkspaceMemoryEstimator::estimatePreparedExpertService(profile,geometry);
    const auto compare_arena=[&](const MoEProjectionArenaGeometry &actual,int degree) {
        const auto expected=MoEProjectionArenaGeometry::resolve(profile,degree);
        for(size_t bank=0;bank<expected.banks().size();++bank) {
            EXPECT_EQ(actual.banks()[bank].id,expected.banks()[bank].id);
            EXPECT_EQ(actual.banks()[bank].name,expected.banks()[bank].name);
            EXPECT_EQ(actual.banks()[bank].words_per_row,expected.banks()[bank].words_per_row);
        }
        for(const size_t rows:{1u,16u,64u}) {
            EXPECT_EQ(actual.bytes(rows),expected.bytes(rows));
            EXPECT_EQ(actual.packetCapacityBytes(rows),expected.packetCapacityBytes(rows));
        }
    };
    {
        PlanningMemoryQuoteReuse quotes(source.metadata().tensorIndex());
        for(int replay=0;replay<20;++replay) {
            compare_arena(PlanningMemoryQuoteReuse::projectionArena(profile,2),2);
            EXPECT_EQ(PlanningMemoryQuoteReuse::preparedExpertService(profile,geometry),canonical_service);
        }
        compare_arena(PlanningMemoryQuoteReuse::projectionArena(profile,4),4);
        auto changed=geometry;changed.resident_graph_rows=64;
        EXPECT_EQ(PlanningMemoryQuoteReuse::preparedExpertService(profile,changed),
            WorkspaceMemoryEstimator::estimatePreparedExpertService(profile,changed));
        changed=geometry;changed.planning_policy.moe_router_arithmetic=geometry.planning_policy.moe_router_arithmetic==MoERouterArithmetic::Native
            ? MoERouterArithmetic::FP16 : MoERouterArithmetic::Native;
        EXPECT_EQ(PlanningMemoryQuoteReuse::preparedExpertService(profile,changed),
            WorkspaceMemoryEstimator::estimatePreparedExpertService(profile,changed));
        changed=geometry;changed.first_layer=1;
        EXPECT_EQ(PlanningMemoryQuoteReuse::preparedExpertService(profile,changed),
            WorkspaceMemoryEstimator::estimatePreparedExpertService(profile,changed));
        auto invalid=geometry;invalid.device=DeviceId::cpu();
        EXPECT_THROW(PlanningMemoryQuoteReuse::preparedExpertService(profile,invalid),std::invalid_argument);
        EXPECT_THROW(PlanningMemoryQuoteReuse::projectionArena(profile,3),std::invalid_argument);
        test::PlanningGGUFFixture other(true,false);PlanningModelSource foreign(other.path());
        EXPECT_THROW(PlanningMemoryQuoteReuse::projectionArena(foreign.metadata().memoryProfile(),2),std::invalid_argument);
        EXPECT_THROW(PlanningMemoryQuoteReuse::preparedExpertService(foreign.metadata().memoryProfile(),geometry),std::invalid_argument);
        auto foreign_geometry=geometry;foreign_geometry.source_tensor_index=foreign.metadata().tensorIndex();
        EXPECT_THROW(PlanningMemoryQuoteReuse::preparedExpertService(profile,foreign_geometry),std::invalid_argument);
        EXPECT_EQ(quotes.counts().projection_arena_builds,2u);
        EXPECT_EQ(quotes.counts().projection_arena_hits,19u);
        EXPECT_EQ(quotes.counts().prepared_service_builds,4u);
        EXPECT_EQ(quotes.counts().prepared_service_hits,19u);
    }
    compare_arena(PlanningMemoryQuoteReuse::projectionArena(profile,2),2);
    EXPECT_EQ(PlanningMemoryQuoteReuse::preparedExpertService(profile,geometry),canonical_service);
    {
        PlanningMemoryQuoteReuse replay(source.metadata().tensorIndex());
        compare_arena(PlanningMemoryQuoteReuse::projectionArena(profile,2),2);
        EXPECT_EQ(PlanningMemoryQuoteReuse::preparedExpertService(profile,geometry),canonical_service);
        EXPECT_EQ(replay.counts().projection_arena_builds,1u);
        EXPECT_EQ(replay.counts().prepared_service_builds,1u);
    }
}

/** @test Complete activation policy identity includes fields whose bytes may happen to agree for this source. */
TEST(PlanningMemoryQuoteReuse, ActivationQuotesReplayCompleteGeometryAndExactDevice)
{
    test::PlanningGGUFFixture file(true,false);
    PlanningModelSource source(file.path());
    const auto &profile=source.metadata().memoryProfile();
    ActivationGraphMemoryGeometry geometry;
    geometry.resident_graph_rows=32;
    geometry.context_tokens=128;
    geometry.local_d_ff=profile.d_ff;
    geometry.local_n_heads=profile.n_heads;
    geometry.local_n_kv_heads=profile.n_kv_heads;
    geometry.local_vocab=profile.vocab_size;
    const auto device=DeviceId::rocm(7);
    const auto canonical=ActivationMemoryEstimator::estimate(profile,geometry,device);
    {
        PlanningMemoryQuoteReuse quotes(source.metadata().tensorIndex());
        for(int replay=0;replay<20;++replay)
            EXPECT_EQ(PlanningMemoryQuoteReuse::activation(profile,geometry,device),canonical);
        std::vector<ActivationGraphMemoryGeometry> variants;
        auto changed=geometry;changed.resident_graph_rows=64;variants.push_back(changed);
        changed=geometry;changed.context_tokens=256;variants.push_back(changed);
        changed=geometry;changed.mtp_target_query_rows=8;variants.push_back(changed);
        changed=geometry;changed.local_hot_terminal_observation=true;variants.push_back(changed);
        for(const auto &variant:variants)
            EXPECT_EQ(PlanningMemoryQuoteReuse::activation(profile,variant,device),
                ActivationMemoryEstimator::estimate(profile,variant,device));
        EXPECT_EQ(PlanningMemoryQuoteReuse::activation(profile,geometry,DeviceId::rocm(2)),
            ActivationMemoryEstimator::estimate(profile,geometry,DeviceId::rocm(2)));
        EXPECT_EQ(quotes.counts().activation_builds,6u);
        EXPECT_EQ(quotes.counts().activation_hits,19u);
        changed=geometry;changed.resident_graph_rows=0;
        EXPECT_THROW(PlanningMemoryQuoteReuse::activation(profile,changed,device),std::invalid_argument);
        EXPECT_EQ(quotes.counts().activation_builds,6u);
        test::PlanningGGUFFixture other(true,false);PlanningModelSource foreign(other.path());
        EXPECT_THROW(PlanningMemoryQuoteReuse::activation(foreign.metadata().memoryProfile(),geometry,device),std::invalid_argument);
    }
    EXPECT_EQ(PlanningMemoryQuoteReuse::activation(profile,geometry,device),canonical);
    {
        PlanningMemoryQuoteReuse replay(source.metadata().tensorIndex());
        EXPECT_EQ(PlanningMemoryQuoteReuse::activation(profile,geometry,device),canonical);
        EXPECT_EQ(replay.counts().activation_builds,1u);
    }
}
