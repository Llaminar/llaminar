/**
 * @file MoEIntermediateShardFFNFinalizationContract.inc
 * @brief Structural final-FFN proof over caller-supplied genuine local routed/shared producers.
 *
 * Called by the focused original-source native replay after its real shared
 * FFN weights are materialized under the same ModelContext/Store and original
 * communicator. This helper constructs no synthetic source, metadata engine,
 * transport or replacement router. It proves declaration topology only; actual
 * native replay follows this declaration in the same fixture; full-model numerical gates remain required.
 * A complete verifier also owns an explicit static history boundary. The
 * negative control removes that declaration from the actual original router
 * before finalization, reproducing the former late MTP publication failure.
 */

/**
 * @brief Verify exactly one explicit full-H FFN sum and lossless native metadata.
 * @param graph Original participant-local graph containing real routed/shared producers.
 * @param routed Actual typed handle from appendMoEIntermediateShardPipeline.
 * @param declaration Same source-authenticated shared FFN and original final sum policy.
 */
void proveIntermediateShardFFNFinalizationDeclaration(ComputeGraph &graph,
    const MoEIntermediateShardLocalPartial &routed, const MoEIntermediateShardFFNFinalization &declaration)
{
    const auto before = graph.size();
    const auto count_sums = [&] {
        int count = 0;
        for (const auto &name : graph.getExecutionOrder())
            count += graph.getNode(name)->stage->type() == ComputeStageType::ALLREDUCE;
        return count;
    };
    const int original_sums = count_sums();
    auto *router_node = graph.getNode("router");
    auto *router = router_node ? dynamic_cast<MoERoutingStage *>(router_node->stage.get()) : nullptr;
    require(router != nullptr, "finalizer fixture retains its actual router");
    if (router->declaration().force_decode_equivalent_verifier_prefill)
    {
        auto missing_boundary = router->declaration();
        missing_boundary.grouped_verifier_histogram_role = MoEGroupedVerifierHistogramRole::NotOwner;
        auto original = std::move(router_node->stage);
        router_node->stage = std::make_unique<MoERoutingStage>(missing_boundary);
        bool rejected = false;
        try { (void)finalizeMoEIntermediateShardFFN(graph, routed, declaration); }
        catch (const std::invalid_argument &) { rejected = true; }
        router_node->stage = std::move(original);
        require(rejected && graph.size() == before,
            "missing static MTP history boundary must fail before graph mutation");

        MoEGroupedVerifierHistogramBoundarySet boundaries;
        MoEGroupedVerifierHistogramBoundaryResolution resolution;
        std::string error;
        require(boundaries.registerRoutedLayer(router->layerIndex(), "router", &error) &&
            boundaries.registerStage(router, "router", &error) && boundaries.resolve(resolution, &error),
            "complete static MTP history boundary: " + error);
        require(resolution.publishers.empty() && !resolution.publication_stream,
            "static MTP history adds no histogram publisher or stream");
    }
    const auto output = finalizeMoEIntermediateShardFFN(graph, routed, declaration);
    EXPECT_EQ(graph.size(), before + 2);
    EXPECT_EQ(count_sums(), original_sums + 1);
    const auto *gate_node = graph.getNode(declaration.prefix + "local_shared_gate_add");
    const auto *sum_node = graph.getNode(output.producer);
    ASSERT_NE(gate_node, nullptr); ASSERT_NE(sum_node, nullptr);
    const auto *gate = dynamic_cast<const SharedExpertGateStage *>(gate_node->stage.get());
    const auto *sum = dynamic_cast<const TPAllreduceStage *>(sum_node->stage.get());
    ASSERT_NE(gate, nullptr); ASSERT_NE(sum, nullptr);
    EXPECT_FALSE(gate->requiresAllreduce());
    EXPECT_EQ(gate_node->dependencies, (std::vector<std::string>{routed.producer,declaration.shared_ffn_node}));
    EXPECT_EQ(sum_node->dependencies, (std::vector<std::string>{gate_node->name}));
    EXPECT_EQ(gate->declaration().output_role, SharedExpertOutputRole::ParticipantLocalPartial);
    EXPECT_EQ(gate->declaration().phase, routed.phase);
    EXPECT_EQ(sum->params().arithmetic_policy, TPAllreduceArithmeticPolicy::CanonicalRankOrder);
    EXPECT_EQ(sum->params().precision, "fp32");
    EXPECT_EQ(sum->params().count, std::size_t(routed.rows) * routed.model_columns);
    EXPECT_EQ(sum->params().sideband_device_index, declaration.final_sum.sideband_device_index);
    EXPECT_EQ(sum->params().sidebands.size(), declaration.final_sum.sidebands.size());
    for (std::size_t index = 0; index < declaration.final_sum.sidebands.size(); ++index)
    {
        const auto &actual = sum->params().sidebands.at(index), &original = declaration.final_sum.sidebands[index];
        EXPECT_EQ(actual.kind, original.kind); EXPECT_EQ(actual.send_buffer, original.send_buffer);
        EXPECT_EQ(actual.recv_buffer, original.recv_buffer); EXPECT_EQ(actual.element_count, original.element_count);
        EXPECT_EQ(actual.dtype, original.dtype); EXPECT_EQ(actual.root_device_index, original.root_device_index);
        EXPECT_EQ(actual.name, original.name);
    }
    EXPECT_EQ(sum->params().sideband_workspace_bindings.size(), declaration.final_sum.sideband_workspace_bindings.size());
    if (routed.live_rows)
    {
        ASSERT_TRUE(sum->params().live_rows);
        EXPECT_EQ(sum->params().live_rows->rows().captureIdentity(), routed.live_rows->captureIdentity());
        EXPECT_EQ(sum->params().live_rows->elementsPerRow(), std::size_t(routed.model_columns));
    }
    if (declaration.final_sum.live_rows)
        EXPECT_EQ(sum->params().live_rows->payloadReceipt(), declaration.final_sum.live_rows->payloadReceipt());
    EXPECT_EQ(output.output.tensor, declaration.local_combined.tensor);
    EXPECT_EQ(output.output.id, declaration.local_combined.id);
    EXPECT_EQ(output.family, routed.family);
    EXPECT_EQ(output.phase, routed.phase);
    EXPECT_THROW((void)finalizeMoEIntermediateShardFFN(graph, routed, declaration), std::logic_error);
    EXPECT_EQ(graph.size(), before + 2);
    auto wrong_math = declaration;
    wrong_math.final_sum.arithmetic_policy = TPAllreduceArithmeticPolicy::NativeCollective;
    EXPECT_THROW((void)finalizeMoEIntermediateShardFFN(graph, routed, wrong_math), std::invalid_argument);
    auto wrong_precision = declaration;
    wrong_precision.final_sum.precision = "bf16";
    EXPECT_THROW((void)finalizeMoEIntermediateShardFFN(graph, routed, wrong_precision), std::invalid_argument);
    auto wrong_handle = routed;
    wrong_handle.phase = MoEOverlayServicePhaseHint::Auto;
    EXPECT_THROW((void)finalizeMoEIntermediateShardFFN(graph, wrong_handle, declaration), std::logic_error);
    EXPECT_EQ(graph.size(), before + 2);
    ASSERT_NE(declaration.shared_weights, nullptr);
    const auto &original_plan = *declaration.shared_weights;
    auto foreign_strategy = original_plan.strategy();
    ++foreign_strategy.model_id.value;
    const FrozenModelWeightSet foreign_model(foreign_strategy, original_plan.bindings(), original_plan.physicalMemoryOwner());
    auto foreign = declaration; foreign.shared_weights = &foreign_model;
    EXPECT_THROW((void)finalizeMoEIntermediateShardFFN(graph, routed, foreign), std::invalid_argument);
    auto missing_gate = original_plan.bindings();
    std::erase_if(missing_gate, [&](const auto &binding) {
        return binding.identity.role == WeightRole::SharedExpertInputGate &&
            binding.residency.home_device == gate->device();
    });
    const FrozenModelWeightSet incomplete(original_plan.strategy(), std::move(missing_gate), original_plan.physicalMemoryOwner());
    foreign.shared_weights = &incomplete;
    EXPECT_THROW((void)finalizeMoEIntermediateShardFFN(graph, routed, foreign), std::invalid_argument);
    if (!declaration.shared_partition)
    {
        EXPECT_EQ(original_plan.strategy().tp_degree, 1);
        EXPECT_EQ(original_plan.strategy().mode, WeightInferenceMode::SingleDevice);
    }
    if (declaration.shared_partition && declaration.shared_partition->worldSize() > 1)
    {
        auto foreign_cuts = declaration.shared_partition->assignments();
        ASSERT_GT(foreign_cuts.back().d_ff_count, 32);
        foreign_cuts.front().d_ff_count += 32;
        foreign_cuts.back().d_ff_count -= 32;
        int cursor = 0;
        for (auto &assignment : foreign_cuts)
        { assignment.d_ff_start = cursor; cursor += assignment.d_ff_count; }
        // A valid complete partition can still be foreign to these original
        // prepared source cuts. No peer tensor binding is synthesized.
        const TensorParallelConfig other_partition(std::move(foreign_cuts));
        foreign = declaration; foreign.shared_partition = &other_partition;
        EXPECT_THROW((void)finalizeMoEIntermediateShardFFN(graph, routed, foreign), std::invalid_argument);
    }
    EXPECT_EQ(graph.size(), before + 2);
}
