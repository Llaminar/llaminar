/**
 * @file PhysicalMeasurementProducerRoleContract.inc
 * @brief Exercise the actual fabric identity producer with valid mixed-format role permutations.
 *
 * The original owner map and manifest are passive source declarations. This
 * contract creates no prepared source permit, events or execution readiness.
 */
namespace llaminar2::test {
TEST(PhysicalPieceMeasurementContract, ActualProducerPreservesMixedFormatsUnderEveryRolePermutation)
{
    MoERoutedExpertPlacementPlan plan;plan.enabled=true;plan.topology=RoutedExpertPlacementTopology::TieredOverlay;
    plan.residency_policy=RoutedExpertResidencyPolicy::StaticById;
    plan.first_model_layer=7;plan.continuation_domain=plan.base_model_domain=plan.shared_expert_domain="hot";
    RoutedExpertDomain hot;hot.name="hot";hot.scope=ExecutionDomainScope::SINGLE;hot.backend=CollectiveBackendType::HOST;
    hot.owner_rank=0;hot.participants={GlobalDeviceAddress::cpu(0)};
    auto cold=hot;cold.name="cold";cold.participants={GlobalDeviceAddress::cpu(1)};
    plan.domains={hot,cold};plan.routed_tiers={{.name="hot",.domain="hot",.priority=0,.max_experts_per_layer=1},
        {.name="cold",.domain="cold",.priority=1,.max_experts_per_layer=1,.fallback=true}};
    plan.placements={{.layer=7,.routed_expert_tier={0,1}}};
    const auto native=[](std::size_t index){const auto *m=native_quantised_formats::kAllSourceFormats[index].metadata;
        return ExpertWeightFormat::nativeQuantised({.codebook_id=m->codebook_id,.is_superblock=m->is_superblock,.present=true});};
    MoEOverlayLayerWeightManifest source{7,{{{ExpertTierWeightProjection::Gate,512,256,native(0)},
        {ExpertTierWeightProjection::Up,512,256,ExpertWeightFormat::floating(TensorType::FP16)},
        {ExpertTierWeightProjection::Down,256,512,native(20)}}}};
    const auto original=source;
    for(bool joint:{false,true}) {
    auto declared=plan;
    if(joint) {
        auto &domain=declared.domains[1];domain.scope=ExecutionDomainScope::NODE_LOCAL;
        domain.backend=CollectiveBackendType::UPI;domain.owner_rank=-1;
        domain.routed_compute_policy=RoutedExpertComputePolicy::TensorSharded;
        domain.participants={GlobalDeviceAddress::cpu(1),GlobalDeviceAddress::cpu(2)};domain.world_ranks={1,2};
        domain.routed_phase_policy=RoutedExpertPhasePolicy::Uniform;
        domain.routed_decode_assignment_policy=domain.routed_prefill_assignment_policy=RoutedExpertAssignmentPolicy::StaticOwner;
    }
    const auto owners=joint?MoEExpertOwnerMap::buildShardPlacement(declared,{original}):MoEExpertOwnerMap::build(declared);
    auto destination=*owners.placementFor(7,1);destination.expert_id=0;
    const MoEOverlayTierMigration edge{.layer_idx=7,.expert_id=0,.source=*owners.placementFor(7,0),.destination=destination};
    source=original;
    do {
        ASSERT_TRUE(source.valid());
        const auto geometry=makePhysicalMoEOverlayMeasurementGeometry(edge,source,2,owners);
        ASSERT_EQ(geometry->pieces().size(),joint?6u:3u);
        for(const auto &projection:original.projections) {
            const auto role=unsigned(projection.projection);
            EXPECT_EQ(geometry->formats()[role],projection.format);
            for(const auto &piece:geometry->pieces())
                if(piece.projection==projection.projection)EXPECT_EQ(piece.format,projection.format);
        }
    }while(std::next_permutation(source.projections.begin(),source.projections.end(),[](const auto&a,const auto&b){return unsigned(a.projection)<unsigned(b.projection);}));
    }
}
}
