/**
 * @file CPUJointInitialBankPrepassContract.inc
 * @brief Genuine multi-layer cold preparation before first fragment graph construction.
 *
 * All source engines come from the real tiny GGUF/common reader/Manager/PMA
 * pipeline. The second socket stays metadata; this is one local actor, not MPI
 * or multi-socket execution. No partial source layer is execution readiness.
 */
namespace
{
    /** @return Same genuine source topology with two fully authored original layers. */
    StaticJointOriginal multiLayerJointOriginal()
    {
        using namespace llaminar2;
        auto base = staticJointOriginal();
        auto file = std::make_shared<JointOriginalFile>();
        {
            std::ofstream out(file->path, std::ios::binary | std::ios::trunc);
            out.write("GGUF", 4);
            jointScalar(out, uint32_t{3});
            jointScalar(out, uint64_t{6});
            jointScalar(out, uint64_t{3});
            jointString(out, "general.architecture");
            jointScalar(out, uint32_t{8});
            jointString(out, "llama");
            jointString(out, "llama.expert_count");
            jointScalar(out, uint32_t{4});
            jointScalar(out, uint32_t{512});
            jointString(out, "llama.expert_used_count");
            jointScalar(out, uint32_t{4});
            jointScalar(out, uint32_t{4});
            for (int layer = 7; layer < 9; ++layer)
                for (int role = 0; role < 3; ++role)
                {
                    jointString(out, "blk." + std::to_string(layer) + ".ffn_" +
                                         std::string(role == 0   ? "gate"
                                                     : role == 1 ? "up"
                                                                 : "down") +
                                         "_exps.weight");
                    jointScalar(out, uint32_t{3});
                    jointScalar(out, uint64_t(role == 2 ? 20 : 8));
                    jointScalar(out, uint64_t(role == 2 ? 8 : 20));
                    jointScalar(out, uint64_t{512});
                    jointScalar(out, uint32_t{0});
                    jointScalar(out, uint64_t((layer - 7) * 3 + role) * 512 * 160 * sizeof(float));
                }
            const auto at = size_t(out.tellp());
            std::vector<char> padding((at + 31) / 32 * 32 - at, 0);
            out.write(padding.data(), padding.size());
            for (int layer = 7; layer < 9; ++layer)
                for (int role = 0; role < 3; ++role)
                    for (int expert = 0; expert < 512; ++expert)
                        for (int element = 0; element < 160; ++element)
                            jointScalar(out, float(expert + layer - 7));
            if (!out)
                throw std::runtime_error("Multi-layer original source write failed");
        }
        auto loader = std::make_shared<ModelLoader>(base.factory.get());
        if (!loader->loadModel(file->path.string()))
            throw std::runtime_error("Multi-layer original source parse failed");
        auto plan = *base.plan;
        auto second = plan.placements.front();
        second.layer = 8;
        plan.placements.push_back(second);
        auto original = std::make_shared<const MoERoutedExpertPlacementPlan>(std::move(plan));
        const auto runtime = resolveMoEExpertOverlayRuntimePlan(
            original, {.current_world_rank = 0, .validate_mvp_root_reachability = false});
        auto preparation = std::make_shared<const MoEExpertOverlayPreparationPlan>(
            MoEExpertOverlayPreparationPlan::build(*runtime, *loader));
        return {original, preparation, file, base.mpi, base.factory, loader};
    }
} // namespace

/** @test Nested source fixtures own distinct files and retire only their own source. */
TEST(CPUJointInitialBankPrepassContract, IndependentSourcesSurvivePeerRetirement)
{
    JointOriginalFile retained;
    std::ofstream(retained.path) << "retained source";
    std::filesystem::path departed;
    {
        JointOriginalFile temporary;
        ASSERT_NE(retained.path, temporary.path);
        std::ofstream(temporary.path) << "temporary source";
        departed = temporary.path;
        ASSERT_TRUE(std::filesystem::exists(retained.path));
        ASSERT_TRUE(std::filesystem::exists(departed));
    }
    EXPECT_TRUE(std::filesystem::exists(retained.path));
    EXPECT_FALSE(std::filesystem::exists(departed));
}

TEST(CPUJointInitialBankPrepassContract,
     FirstGraphBorrowRequiresAllRealLayersAndCompletedReentryNeverResurrects)
{
    using namespace llaminar2;
    if (!hasCPUBackend())
        initCPUBackend(-1);
    const auto original = multiLayerJointOriginal();
    auto context = coldSourceContext(original);
    auto manager = context->concreteWeightManager();
    const ModelContextId model{reinterpret_cast<uint64_t>(context.get())};
    const auto ids = staticJointMembers(original);
    ASSERT_FALSE(ids.empty());
    const auto id = ids.front();
    const auto *participant = original.preparation->cpuJointOwnerMap()->participantForId(id);
    ASSERT_TRUE(participant);
    InferenceStrategy strategy{.model_id = model};
    WeightPlan full(strategy, PhysicalMemoryOwner::RoutedExpertWeights);
    WeightPlan first(strategy, PhysicalMemoryOwner::RoutedExpertWeights);
    WeightPlan later(strategy, PhysicalMemoryOwner::RoutedExpertWeights);
    for (int layer = 7; layer < 9; ++layer)
    {
        const auto selection = original.preparation->cpuShardSourceFor(layer, *participant);
        for (const auto role :
             {WeightRole::MoEExpertGate, WeightRole::MoEExpertUp, WeightRole::MoEExpertDown})
        {
            const auto name = "blk." + std::to_string(layer) + ".ffn_" +
                              std::string(role == WeightRole::MoEExpertGate ? "gate"
                                          : role == WeightRole::MoEExpertUp ? "up"
                                                                            : "down") +
                              "_exps.weight";
            const auto request = test::nativeJointRequirementForTest(selection, role, name);
            full.add(request);
            (layer == 7 ? first : later).add(request);
        }
    }
    // This original participant uses NUMA-backed final roots. The generic
    // source-reader fixture admits local heap storage, whose physical extent
    // differs from one page-aligned allocation per actual fragment projection.
    const auto placement = CPUStoragePlacement::onNode(participant->address.numa_node);
    size_t final_bytes = 0;
    for (int layer = 7; layer < 9; ++layer)
    {
        const auto selection = original.preparation->cpuShardSourceFor(layer, *participant);
        for (const auto &projection :
             selection->cohort()->localManifest(selection->participantIndex()).projections)
            final_bytes += selection->expertIds().size() *
                placement.physicalBytes<float>(size_t(projection.N) * projection.K);
    }
    PhysicalMemoryPlanBuilder budget;
    const PhysicalMemoryResource host{participant->world_rank, participant->device,
                                      final_bytes, final_bytes};
    budget.add(host, PhysicalMemoryOwner::RoutedExpertWeights, final_bytes);
    manager->installPhysicalMemoryAuthority(std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()), host.world_rank));
    ASSERT_EQ(manager->materialize(first).bindings().size(), 1344u);
    MoEOverlayParticipantResidencyRegistry banks({.local_participant_ids = {id},
                                                  .num_layers = 2,
                                                  .num_experts = 512,
                                                  .initial_epoch = 1,
                                                  .projection_preparation = original.preparation,
                                                  .first_model_layer = 7,
                                                  .original_model = model});
    std::string error;
    ASSERT_TRUE(banks.bindOriginalPreparedContext(context, original.preparation, &error)) << error;
    // An actual missing later source cannot leave a graph-constructible first layer.
    EXPECT_FALSE(banks.registerInitialLayerFromPreparedRegistry(
        manager->expertGemmRegistry(), 7, &error));
    EXPECT_FALSE(banks.endpoint(id)->acquire(1));
    EXPECT_FALSE(banks.allInitialBanksReady());
    EXPECT_THROW(banks.initialCPUJointLayer(id, 7), std::logic_error);
    ASSERT_EQ(manager->materialize(later).bindings().size(), 1344u);
    ASSERT_TRUE(banks.registerInitialLayerFromPreparedRegistry(
        manager->expertGemmRegistry(), 7, &error)) << error;
    ASSERT_TRUE(banks.allInitialBanksReady());
    auto first_frame = banks.initialCPUJointLayer(id, 7);
    auto first_construction = first_frame->acquireInitialConstruction();
    ASSERT_TRUE(first_construction);
    ASSERT_TRUE(first_construction->containsModelLayer(8));
    EXPECT_TRUE(first_construction->layerForModelLayer(7).cpu_fragments[71].valid());
    EXPECT_TRUE(first_construction->layerForModelLayer(8).cpu_fragments[71].valid());
    const auto later_frame = banks.initialCPUJointLayer(id, 8);
    ASSERT_TRUE(later_frame->acquireInitialConstruction());
    EXPECT_EQ(later_frame->sourceSelection()->layer(), 8);
    EXPECT_THROW(banks.initialCPUJointLayer(id, 9), std::invalid_argument);
    ExpertGemmRegistry foreign;
    EXPECT_FALSE(banks.registerInitialCPUJointBank(foreign, model, id, &error));
    EXPECT_FALSE(banks.registerInitialCPUJointBank(manager->expertGemmRegistry(), ModelContextId{1},
                                                   id, &error));
    // Source lookup entries may retire after the one immutable bank installs.
    // Every retained graph re-entry borrows that actual bank, never rescans all
    // source layers or constructs another publication.
    manager->expertGemmRegistry().clear();
    for (int repeat = 0; repeat < 20; ++repeat)
    {
        ASSERT_TRUE(
            banks.registerInitialCPUJointBank(manager->expertGemmRegistry(), model, id, &error))
            << error;
        EXPECT_EQ(banks.initialCPUJointLayer(id, 7), first_frame);
    }
    first_construction.reset();
    ASSERT_TRUE(banks.endpoint(id)->retire(1));
    EXPECT_FALSE(first_frame->acquireInitialConstruction());
    EXPECT_FALSE(
        banks.registerInitialCPUJointBank(manager->expertGemmRegistry(), model, id, &error));
    EXPECT_NE(error.find("resurrect"), std::string::npos);
}
