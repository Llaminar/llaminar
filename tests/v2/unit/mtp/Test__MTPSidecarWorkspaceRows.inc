/**
 * @file Test__MTPSidecarWorkspaceRows.inc
 * @brief Prove that sidecar scratch follows invocation rows, never KV capacity.
 *
 * Sparse ExpertOverlay stages borrow routed packets rather than declaring an
 * arena input shape. Their workspace therefore relies on the enclosing DGO's
 * exact row hint. A shape-free probe appended to real CPU predictor arithmetic
 * isolates this contract without loading a large model or mocking execution.
 */

namespace
{
    /** @brief Passive observations; none of these values controls inference. */
    struct SidecarWorkspaceRows
    {
        std::vector<int> queries; ///< Actual allocator-to-consumer row hints.
        int executions = 0; ///< Completed workspace writes by the probe.
    };

    /** @brief Emulate a routed packet consumer with invocation-sized scratch. */
    class SparseRowWorkspaceProbe final
        : public llaminar2::testing::MockComputeStage, public IWorkspaceConsumer
    {
    public:
        /** @brief Retain immutable invocation geometry and externally owned evidence. */
        SparseRowWorkspaceProbe(int rows, SidecarWorkspaceRows &evidence)
            : rows_(rows), evidence_(evidence) {}

        /** @brief Advertise scratch without an arena tensor from which to infer M. */
        WorkspaceRequirements getWorkspaceRequirements(int rows, int = 0, int = 0) const override
        {
            evidence_.queries.push_back(rows);
            WorkspaceRequirements result;
            // A sparse packet also owns its physical row extent, as in
            // MoELocalExpertStage. Setup must cover that extent even when the
            // enclosing family's primary participant has fewer rows.
            result.buffers.emplace_back("sidecar_sparse_row_probe",
                size_t(std::max(rows, rows_)) * sizeof(float));
            return result;
        }
        /** @brief Borrow the same family-owned manager as production sparse stages. */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override { workspace_ = workspace; }
        /** @return Whether setup supplied the persistent scratch owner. */
        bool hasWorkspace() const override { return workspace_ != nullptr; }
        /** @return Borrowed manager, never a separately allocated scratch buffer. */
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }
        /** @brief Touch every live row, proving that the declared scratch is usable. */
        bool execute(IDeviceContext *) override
        {
            if (!workspace_ || workspace_->getBufferSize("sidecar_sparse_row_probe") < size_t(rows_) * sizeof(float))
                return false;
            auto *scratch = static_cast<float *>(workspace_->getBuffer("sidecar_sparse_row_probe"));
            if (!scratch) return false;
            std::fill_n(scratch, rows_, 1.0f);
            ++evidence_.executions;
            return true;
        }
    private:
        int rows_; ///< Physical rows in this exact sidecar invocation.
        SidecarWorkspaceRows &evidence_; ///< Outlives all retained graph objects.
        DeviceWorkspaceManager *workspace_ = nullptr; ///< Borrowed serial-family storage.
    };

    /** @brief Preserve real predictor arithmetic while exposing its sparse-row contract. */
    class SidecarWorkspaceQwenGraph final : public Qwen35Graph
    {
    public:
        using Qwen35Graph::Qwen35Graph;
        SidecarWorkspaceRows evidence; ///< Graph-family setup and execution observations.

        /** @copydoc Qwen35Graph::buildMTPGraph */
        ComputeGraph buildMTPGraph(int depth, const MTPDepthWeightBindings &weights,
                                  const MTPForwardInput &input, MTPForwardOutput &output) override
        {
            auto graph = Qwen35Graph::buildMTPGraph(depth, weights, input, output);
            const auto terminal = graph.getExecutionOrder().back();
            graph.addNode("sparse_row_workspace_probe", std::make_unique<SparseRowWorkspaceProbe>(
                input.batch_size * input.seq_len, evidence), input.device);
            graph.addDependency("sparse_row_workspace_probe", terminal);
            return graph;
        }
    };
}

/** @test Single/chained and request-batched sidecars reuse the prepared bounded family. */
TEST(Test__MTPGraphConstruction, SidecarWorkspaceUsesInvocationRowsNotContextCapacity)
{
    DeviceManager::instance().initialize(-1, false);
    for (const int batch : {1, 2, 3})
    {
        SCOPED_TRACE(batch);
        TinyQwen35MTPForwardFixture fixture;
        fixture.config.max_seq_len = 8192;
        fixture.config.mtp.graph_capacity_draft_tokens = 15;
        fixture.config.mtp.max_request_batch = batch;
        auto graph = std::make_shared<SidecarWorkspaceQwenGraph>(fixture.config, fixture.mpi);
        PreparedWeightStore store; // All graph/kernel borrows retire before their immutable owner.
        DeviceGraphOrchestrator runner(graph, fixture.mpi);
        ASSERT_TRUE(runner.initializeInferenceStateFromArena(batch, fixture.config.max_seq_len,
            DeviceId::cpu(), {.activation_seq_len = 512}));
        runner.setFrozenWeightSet(makeTinyQwen35MTPFrozenWeightSet(fixture));
        prepareFrozenGemmWeightsForCPU(*runner.frozenWeightSet(), store);
        graph->setPreparedWeightStore(&store);
        ASSERT_TRUE(runner.materializeForwardGraphForShape(1, batch));
        auto &workspace_host = static_cast<IForwardExecutionHost &>(runner);
        const auto generation = workspace_host.workspaceGeneration(DeviceId::cpu());
        graph->evidence.queries.clear();

        auto hidden = runner.inferenceState().hidden;
        ASSERT_NE(hidden, nullptr);
        std::fill_n(hidden->mutable_data(), hidden->numel(), 0.01f);
        runner.markMainForwardHiddenProducedForTesting(1, batch, std::vector<int>(batch, 1));
        std::vector<int32_t> tokens(batch, 3), drafts(batch, -1);
        std::vector<int> positions(batch, 0);
        if (batch == 1)
        {
            ASSERT_TRUE(runner.forwardMTPAndSampleGreedy(tokens.front(), drafts.data()));
            ASSERT_TRUE(runner.forwardMTPFromLastDraftAndSampleGreedy(drafts.front(), 1, tokens.data()));
        }
        else
        {
            ASSERT_TRUE(runner.forwardMTPBatchAndSampleGreedy(tokens.data(), positions.data(), batch, drafts.data()));
            std::fill(positions.begin(), positions.end(), 1);
            ASSERT_TRUE(runner.forwardMTPBatchFromLastDraftAndSampleGreedy(
                drafts.data(), positions.data(), batch, tokens.data()));
        }
        ASSERT_EQ(graph->evidence.executions, 2);
        ASSERT_FALSE(graph->evidence.queries.empty());
        for (int rows : graph->evidence.queries)
            EXPECT_EQ(rows, batch) << "A KV context or prefill capacity cannot size a live sidecar";
        EXPECT_EQ(workspace_host.workspaceGeneration(DeviceId::cpu()), generation);
        for (int32_t token : tokens)
        {
            EXPECT_GE(token, 0);
            EXPECT_LT(token, fixture.config.vocab_size);
        }
    }
}
