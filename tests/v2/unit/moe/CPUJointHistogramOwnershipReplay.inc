/**
 * @file CPUJointHistogramOwnershipReplay.inc
 * @brief Original-source complete placement publication over the same logical histogram banks.
 *
 * Widths2..8 describe metadata geometry only. No native participant, socket,
 * collective, weight bank or readiness is fabricated by this device-free replay.
 */
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEExpertShardCohort.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/CPUJointSourceTestPlan.h"
#include "tensors/TensorFactory.h"
namespace {
/** @brief Own genuine tiny source metadata and the original canonical joint map. */
struct HistogramJointSource {
    llaminar2::test::PlanningGGUFFixture file;
    std::shared_ptr<MPIContext> mpi=MPIContextFactory::self();
    TensorFactory factory{*mpi};ModelLoader loader{&factory};
    MoERoutedExpertPlacementPlan plan;
    std::shared_ptr<const MoEExpertOverlayPreparationPlan> preparation;
    /** @param width Declared metadata width, never a native hardware claim. */
    explicit HistogramJointSource(int width):file(true,false,GGUFTensorType::F32,64,std::nullopt,8,
        llaminar2::test::PlanningGGUFFixture::Geometry::FullAttentionGraph,8,4) {
        if(!loader.loadModel(file.path()))throw std::runtime_error("Histogram source metadata parse failed");
        plan=llaminar2::test::cpuJointSourceTestPlan();
        plan.routed_tiers[0].max_experts_per_layer=1;plan.routed_tiers[1].max_experts_per_layer=3;
        plan.placements={{.layer=7,.routed_expert_tier={0,1,1,1}}};
        auto& domain=plan.domains[1];domain.participants.clear();domain.world_ranks.clear();
        for(int member=0;member<width;++member){domain.participants.push_back(GlobalDeviceAddress::cpu(member));domain.world_ranks.push_back(member);}
        const auto original=std::make_shared<const MoERoutedExpertPlacementPlan>(plan);
        const auto runtime=resolveMoEExpertOverlayRuntimePlan(original,{.current_world_rank=0,.validate_mvp_root_reachability=false});
        preparation=std::make_shared<const MoEExpertOverlayPreparationPlan>(MoEExpertOverlayPreparationPlan::build(*runtime,loader));
    }
};
}
/** @test Repeated canonical ownership publication never changes logical demand or duplicates cohort hotness. */
TEST(Test__DecodeExpertHistogram, CompleteTargetOwnershipPublicationReplay)
{
    for(int width=2;width<=8;++width) {
        SCOPED_TRACE(width);
        auto source=std::make_shared<HistogramJointSource>(width);
        const auto original=std::shared_ptr<const MoEExpertOwnerMap>(source,source->preparation->cpuJointOwnerMap().get());
        auto plan=source->plan;std::swap(plan.placements[0].routed_expert_tier[0],plan.placements[0].routed_expert_tier[1]);
        struct Candidate {std::shared_ptr<HistogramJointSource> source;MoEExpertOwnerMap map;};
        auto candidate_owner=std::make_shared<Candidate>(Candidate{source,MoEExpertOwnerMap::buildShardTransition(plan,*original,*original)});
        auto candidate=std::shared_ptr<const MoEExpertOwnerMap>(candidate_owner,&candidate_owner->map);
        DecodeExpertHistogramConfig cfg{.num_layers=1,.num_experts=4,.top_k=2,.window_size=4,
            .transaction_demand=ExpertHistogramTransactionConfig{{4,4,2},transactionMemory(65536)},.placement_ownership=original};
        DecodeExpertHistogram histogram(cfg);
        EXPECT_EQ(histogram.firstModelLayer(),7);
        EXPECT_TRUE(histogram.config().sockets.empty());
        EXPECT_EQ(histogram.placementOwnership(),original);
        EXPECT_THROW(histogram.socketLoads(7),std::logic_error);
        EXPECT_THROW(histogram.updateOwnership(makeOwnership(1,2,{0,1,0,1})),std::logic_error);
        auto mixed=cfg;mixed.sockets={DeviceId::cpu()};EXPECT_THROW(DecodeExpertHistogram{mixed},std::invalid_argument);
        auto foreign=std::make_shared<HistogramJointSource>(width==8?2:width+1);
        EXPECT_THROW(histogram.updatePlacementOwnership(std::shared_ptr<const MoEExpertOwnerMap>(foreign,foreign->preparation->cpuJointOwnerMap().get())),std::invalid_argument);
        const int routes[]{0,1,1,2,1,3,1,2};std::array<uint64_t,4> scratch{};
        const RoutedExpertHistogramMerge batch{.source=ExpertHistogramSource::PrefillChunk,.layer_idx=7,
            .real_token_count=4,.bucket_token_count=4,.top_k=2,.route_stride=2,.count_window_tokens=true};
        for(int replay=0;replay<20;++replay) {
            const auto merged=histogram.mergeRoutedExpertRows(routes,batch,scratch);
            ASSERT_TRUE(merged) << merged.error;
            const auto counts=histogram.layerHistogram(7);
            EXPECT_EQ(counts,(std::vector<uint64_t>{1,4,2,1}));
            for(const bool changed:{false,true,false,true}) {
                const auto ownership=changed?candidate:original;
                histogram.updatePlacementOwnership(ownership);
                EXPECT_EQ(histogram.placementOwnership(),ownership);
                EXPECT_EQ(histogram.config().placement_ownership,nullptr);
                EXPECT_EQ(histogram.layerHistogram(7),counts);
                const auto loads=histogram.targetLoads(7);ASSERT_EQ(loads.size(),2u);
                uint64_t total=0;
                for(const auto& load:loads){total+=load.activations;EXPECT_EQ(load.activations,changed?4u:load.target.isCPUCohort()?7u:1u);}
                EXPECT_EQ(total,8u);
                const auto imbalance=histogram.currentPlacementImbalance();ASSERT_TRUE(imbalance.valid);
                EXPECT_EQ(imbalance.total_activations,8u);
                EXPECT_DOUBLE_EQ(imbalance.worst_spread,changed?0.0:6.0/7.0);
                EXPECT_NE(histogram.layerSummary(7).find("target_loads="),std::string::npos);
            }
            auto frozen=histogram.freezeAndRotateWindow();ASSERT_TRUE(frozen.valid());
            EXPECT_EQ(frozen.expert_counts,counts);EXPECT_EQ(frozen.token_count,4u);
            ASSERT_NE(frozen.transaction_demand,nullptr);
            const auto demand=frozen.transaction_demand->routes(7,0);
            EXPECT_EQ(demand.logical_rows,4u);EXPECT_TRUE(std::equal(routes,routes+8,demand.expert_ids));
            EXPECT_EQ(histogram.layerHistogram(7),(std::vector<uint64_t>(4,0)));
            histogram.resetWindow();EXPECT_EQ(histogram.layerHistogram(7),(std::vector<uint64_t>(4,0)));
        }
        // The immutable settings view must not pin the first publication after
        // the live authority moves on. This wrapper owns the actual map borrow,
        // so its weak lifetime distinguishes settings retention from map reuse.
        struct Publication {std::shared_ptr<const MoEExpertOwnerMap> map;};
        auto first_publication=std::make_shared<Publication>(Publication{original});
        std::weak_ptr<Publication> first_lifetime=first_publication;
        auto handoff_config=cfg;
        handoff_config.placement_ownership=std::shared_ptr<const MoEExpertOwnerMap>(
            first_publication,first_publication->map.get());
        DecodeExpertHistogram handoff(std::move(handoff_config));
        first_publication.reset();
        EXPECT_FALSE(first_lifetime.expired());
        EXPECT_EQ(handoff.config().placement_ownership,nullptr);
        handoff.updatePlacementOwnership(candidate);
        EXPECT_TRUE(first_lifetime.expired());
        handoff.updatePlacementOwnership(original);

        histogram.updatePlacementOwnership(candidate);
        std::weak_ptr<Candidate> lifetime=candidate_owner;candidate_owner.reset();candidate.reset();
        EXPECT_FALSE(lifetime.expired());
        histogram.updatePlacementOwnership(original);
        EXPECT_TRUE(lifetime.expired());
    }
}
