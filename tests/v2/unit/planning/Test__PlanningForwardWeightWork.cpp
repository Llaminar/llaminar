/**
 * @file Test__PlanningForwardWeightWork.cpp
 * @brief Device-free work ownership proofs using the real compiler and GGUF metadata.
 *
 * Sparse fixtures never execute inference. They distinguish resident capacity
 * from issued main-forward work, exact TP axes from element fractions, both
 * rank namespaces, and continuation operands from routed-only service. No
 * synthetic inventory or work count is claimed as measured performance.
 */
#include "planning/PlanningForwardWeightWork.h"
#include "planning/OrchestrationCandidateAdmission.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include "tensors/NativeVnniFormatInfo.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

using namespace llaminar2;

namespace
{
    /** @return Two physically distinct hosts with GPUs only on discovery rank one. */
    ClusterInventory inventory(DeviceType backend, int gpu_count = 1)
    {
        ClusterInventory result;
        result.world_size = 2;
        for (int rank = 0; rank < 2; ++rank)
        {
            RankInventory record;
            record.rank = rank;
            record.node_id = 7 + rank;
            record.local_rank = 0;
            record.hostname = "work-host-" + std::to_string(rank);
            record.cpu.numa_node = 3;
            record.cpu.memory_bytes = 64ull << 30;
            record.cpu.free_memory_bytes = 48ull << 30;
            record.cpu.compute_units = record.cpu_cores = 8;
            record.cpu_worker_threads = 8;
            record.cpu_execution = test::kSyntheticCPUExecutionGeometry;
            record.numa_nodes = 1;
            if (rank == 1 && backend != DeviceType::CPU)
                for (int index = 0; index < gpu_count; ++index)
                    record.gpus.push_back({.type = backend, .local_device_id = index + 2,
                        .memory_bytes = 32ull << 30, .free_memory_bytes = 24ull << 30,
                        .compute_units = 64, .uuid = "card-" + std::to_string(index), .numa_node = 3});
            result.ranks.push_back(std::move(record));
        }
        result.buildNodeAggregations();
        return result;
    }

    /** @return Last complete proposal of one family, preserving sparse discovery identity. */
    AutomaticOrchestrationCandidate proposal(const PlanningModelSource &source, DeviceType backend,
        OrchestrationStrategy strategy, int gpu_count = 1)
    {
        OrchestrationConfig request;
        request.model_path = source.path();
        request.max_seq_len = 1024;
        request.mtp.enabled = false;
        request.mtp.graph_capacity_draft_tokens = source.metadata().memoryProfile().mtp_layer_count > 0 ? 15 : 0;
        request.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
        request.automatic_planning.only_backends = strategy == OrchestrationStrategy::ExpertOverlay
            ? std::vector{backend, DeviceType::CPU} : std::vector{backend};
        request.automatic_planning.only_strategies = {strategy};
        std::optional<AutomaticOrchestrationCandidate> result;
        visitAutomaticOrchestrationCandidates(request, source.metadata(), inventory(backend, gpu_count), [&](auto value) {
            if (strategy != OrchestrationStrategy::ExpertOverlay || value.membership.discoveryRanks().size() == 2)
            {
                if (strategy == OrchestrationStrategy::ExpertOverlay && gpu_count > 1 &&
                    std::none_of(value.config.moe_routed_expert_plan->domains.begin(),
                        value.config.moe_routed_expert_plan->domains.end(), [&](const auto &domain) {
                            return domain.participants.size() == static_cast<size_t>(gpu_count) &&
                                domain.participants.front().isGPU();
                        })) return;
                result = std::move(value);
            }
        });
        if (!result) throw std::logic_error("No fixture candidate");
        return std::move(*result);
    }

    /** @return Complete admission with a deliberately small GPU capture ladder. */
    AdmittedOrchestrationCandidate admit(AutomaticOrchestrationCandidate candidate, const PlanningModelSource &source)
    {
        return AdmittedOrchestrationCandidate::admit(std::move(candidate), source,
            {.prefill_bucket_rows = {32, 64}, .minimum_prefill_sequence_rows = 1, .maximum_cached_prefill_buckets = 8});
    }

    /** @return Semantic operand under any ordinary operation alternative. */
    const PlanningWeightOperand &operand(const PlanningOrdinaryWeightWork &work)
    {
        return std::visit([](const auto &operation) -> const PlanningWeightOperand & { return operation.weight; }, work);
    }

    /** @return Unique semantic/source use; throws instead of returning an unrelated shape. */
    const PlanningWeightOperand &find(const PlanningParticipantWeightWork &work, const std::string &name, WeightRole role)
    {
        const auto found = std::find_if(work.ordinary.begin(), work.ordinary.end(), [&](const auto &item) {
            return operand(item).source_name == name && operand(item).role == role;
        });
        if (found == work.ordinary.end()) throw std::logic_error("Missing work: " + name);
        return operand(*found);
    }
}

TEST(PlanningForwardWeightWork, MainInvocationExcludesRetainedDepthFifteenAndPreservesRankIdentity)
{
    test::PlanningGGUFFixture file(false, true);
    PlanningModelSource source(file.path());
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        const auto candidate = admit(proposal(source, backend, OrchestrationStrategy::SingleDevice), source);
        for (auto phase : {PlanningMainForwardPhase::Prefill, PlanningMainForwardPhase::Decode})
        {
            const auto work = compilePlanningForwardWeightWork(source.metadata(), candidate, phase);
            ASSERT_EQ(work.size(), 1u);
            EXPECT_EQ(work[0].execution_rank, 0);
            EXPECT_EQ(work[0].discovery_rank, 1);
            EXPECT_EQ(work[0].first_layer, 0);
            EXPECT_EQ(work[0].last_layer, 1);
            EXPECT_EQ(work[0].activation_rows, candidate.devicePlans()[0].activation_seq_len);
            EXPECT_TRUE(work[0].routed.empty());
            size_t projections = 0, lookups = 0;
            for (const auto &item : work[0].ordinary)
            {
                EXPECT_LT(operand(item).layer, source.metadata().mainLayerCount());
                EXPECT_FALSE(std::holds_alternative<PlanningUnclassifiedWeight>(item));
                projections += std::holds_alternative<PlanningProjectionWeight>(item);
                lookups += std::holds_alternative<PlanningEmbeddingWeight>(item);
            }
            EXPECT_EQ(projections, 15u); // 2 * (Q/K/V/O + gate/up/down), plus terminal head.
            EXPECT_EQ(lookups, 1u);
        }
    }
}

TEST(PlanningForwardWeightWork, NativeFormatsAndCanonicalFP32OverridesRemainDistinct)
{
    test::PlanningGGUFFixture file(false, true);
    PlanningModelSource source(file.path());
    std::vector<std::string> formats{"F32", "F16", "BF16"};
    for (const auto &entry : native_vnni_formats::kAllSourceFormats) formats.emplace_back(entry.quant_type);
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        const auto candidate = admit(proposal(source, backend, OrchestrationStrategy::SingleDevice), source);
        for (const auto &format : formats)
        {
            auto profile = source.metadata().memoryProfile();
            for (auto &tensor : profile.tensors)
                if (inferWeightRole(tensor.name) == WeightRole::FFNGate) tensor.quant_type = format;
            profile.tensors.push_back({"blk.0.ssm_alpha.weight", 256 * 8 * 4, format, 256 * 8, 256, 0});
            profile.tensors.push_back({"blk.0.ffn_gate_inp_shexp.weight", 256 * 4, format, 256, 256, 0});
            profile.tensors.push_back({"blk.0.ssm_out.weight", 256 * 512 * 4, format, 256 * 512, 512, 0});
            profile.tensors.push_back({"blk.0.attn_gate.weight", 256 * 512 * 4, format, 256 * 512, 256, 0});
            const auto work = compilePlanningForwardWeightWork(PlanningModelMetadata(profile, 2), candidate, PlanningMainForwardPhase::Decode);
            const auto &native = find(work[0], "blk.0.ffn_gate.weight", WeightRole::FFNGate);
            EXPECT_EQ(native.executionFormat(), format);
            EXPECT_EQ(native.executionFormat().data(), native.source_format.data())
                << "Native execution format must borrow retained storage, not a conditional-expression temporary";
            EXPECT_EQ(native.representation, ModelPreparedWeightRepresentation::SourceNative);
            const auto &alpha = find(work[0], "blk.0.ssm_alpha.weight", WeightRole::GDNAlphaBetaProjection);
            EXPECT_EQ(alpha.executionFormat(), format == "Q8_0" ? "F32" : format);
            const auto &gate = find(work[0], "blk.0.ffn_gate_inp_shexp.weight", WeightRole::SharedExpertInputGate);
            EXPECT_EQ(gate.executionFormat(), "F32");
            EXPECT_EQ(find(work[0], "blk.0.ssm_out.weight", WeightRole::GDNProjection).executionFormat(), format);
            EXPECT_EQ(find(work[0], "blk.0.attn_gate.weight", WeightRole::GDNProjection).executionFormat(), format);
        }
    }
}

TEST(PlanningForwardWeightWork, TPUsesExactAxesAndMirroredHeadOnBothPhasesEvenWithMTPOff)
{
    test::PlanningGGUFFixture file(false, true);
    PlanningModelSource source(file.path());
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (auto head : {MTPTerminalHeadPolicy::MirroredFullVocabulary, MTPTerminalHeadPolicy::VocabularySharded})
    {
        auto request = proposal(source, backend, OrchestrationStrategy::TensorParallel, 2);
        request.config.mtp.terminal_head_policy = head;
        const auto candidate = admit(std::move(request), source);
        ASSERT_EQ(candidate.devicePlans().size(), 2u);
        for (auto phase : {PlanningMainForwardPhase::Prefill, PlanningMainForwardPhase::Decode})
        {
            const auto work = compilePlanningForwardWeightWork(source.metadata(), candidate, phase);
            ASSERT_EQ(work.size(), 2u);
            for (const auto &participant : work)
            {
                EXPECT_EQ(find(participant, "blk.0.ffn_gate.weight", WeightRole::FFNGate).geometry.matrix(),
                    (WeightShardMatrix{256, 256, 1}));
                EXPECT_EQ(find(participant, "blk.0.ffn_down.weight", WeightRole::FFNDown).geometry.matrix(),
                    (WeightShardMatrix{256, 256, 1}));
                EXPECT_EQ(find(participant, "output.weight", WeightRole::LMHead).geometry.matrix()->rows,
                    head == MTPTerminalHeadPolicy::MirroredFullVocabulary ? 320u : 160u);
            }
        }
    }
}

TEST(PlanningForwardWeightWork, PhaseSplitMirrorsReplaceWorkRatherThanAddingAnotherInvocation)
{
    test::PlanningGGUFFixture file(true, true);
    PlanningModelSource source(file.path());
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (const auto policy : {DenseParallelPolicy::PrefillTensorParallelDecodeReplicated,
                             DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding})
    {
        auto request = proposal(source, backend, OrchestrationStrategy::ExpertOverlay, 2);
        request.config.moe_routed_expert_plan->continuation_dense_policy_intent = MoEContinuationDensePolicyIntent::Explicit;
        request.config.moe_routed_expert_plan->continuation_domain_spec.setDensePolicy(policy);
        const auto candidate = admit(std::move(request), source);
        const auto prefill = compilePlanningForwardWeightWork(source.metadata(), candidate, PlanningMainForwardPhase::Prefill);
        const auto decode = compilePlanningForwardWeightWork(source.metadata(), candidate, PlanningMainForwardPhase::Decode);
        ASSERT_EQ(prefill.size(), 3u);
        ASSERT_EQ(decode.size(), prefill.size());
        for (size_t index = 0; index < prefill.size(); ++index)
        {
            const auto &before = prefill[index];
            const auto &after = decode[index];
            ASSERT_EQ(before.ordinary.size(), after.ordinary.size());
            if (before.device.is_cpu()) continue;
            EXPECT_EQ(find(before, "blk.0.attn_q.weight", WeightRole::AttentionQ).geometry.matrix()->rows, 128u);
            EXPECT_EQ(find(after, "blk.0.attn_q.weight", WeightRole::AttentionQ).geometry.matrix()->rows,
                policy == DenseParallelPolicy::PrefillTensorParallelDecodeReplicated ? 256u : 128u);
            EXPECT_EQ(find(before, "token_embd.weight", WeightRole::Embedding).geometry.matrix()->rows, 160u);
            EXPECT_EQ(find(after, "token_embd.weight", WeightRole::Embedding).geometry.matrix()->rows, 320u);
            EXPECT_EQ(find(before, "output.weight", WeightRole::LMHead).geometry.matrix()->rows, 320u);
            EXPECT_EQ(find(after, "output.weight", WeightRole::LMHead).geometry.matrix()->rows, 320u);
        }
    }
}

TEST(PlanningForwardWeightWork, PipelineOwnsEachLayerAndGlobalOperandOnlyOnce)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    const auto candidate = admit(proposal(source, DeviceType::CPU, OrchestrationStrategy::PipelineParallel), source);
    const auto work = compilePlanningForwardWeightWork(source.metadata(), candidate, PlanningMainForwardPhase::Prefill);
    ASSERT_EQ(work.size(), 2u);
    std::map<std::string, int> uses;
    for (const auto &participant : work)
        for (const auto &item : participant.ordinary)
        {
            const auto &weight = operand(item);
            ++uses[weight.source_name];
            if (weight.layer >= 0)
            {
                EXPECT_GE(weight.layer, participant.first_layer);
                EXPECT_LE(weight.layer, participant.last_layer);
            }
        }
    EXPECT_EQ(uses.size(), source.metadata().memoryProfile().tensors.size());
    for (const auto &[name, count] : uses) EXPECT_EQ(count, 1) << name;
}

TEST(PlanningForwardWeightWork, RemoteExpertServiceIsWholeFFNsNotDenseWorkOrResidentCopies)
{
    test::PlanningGGUFFixture file(true, true, GGUFTensorType::BF16, 512);
    PlanningModelSource source(file.path());
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        const auto candidate = admit(proposal(source, backend, OrchestrationStrategy::ExpertOverlay), source);
        for (auto phase : {PlanningMainForwardPhase::Prefill, PlanningMainForwardPhase::Decode})
        {
            const auto work = compilePlanningForwardWeightWork(source.metadata(), candidate, phase);
            ASSERT_EQ(work.size(), 2u);
            for (const auto &participant : work)
            {
                EXPECT_EQ(participant.ordinary.empty(), participant.device.is_cpu());
                EXPECT_EQ(std::count_if(participant.ordinary.begin(), participant.ordinary.end(), [](const auto &item) {
                    return std::holds_alternative<PlanningRouterWeight>(item);
                }), participant.device.is_cpu() ? 0 : 2);
                ASSERT_EQ(participant.routed.size(), 2u);
                for (const auto &expert : participant.routed)
                {
                    EXPECT_EQ(expert.expert_count, 8);
                    EXPECT_EQ(expert.routes_per_token, 2);
                    EXPECT_EQ(expert.gate_up_down[0].geometry.matrix(), (WeightShardMatrix{512, 256, 1}));
                    EXPECT_EQ(expert.gate_up_down[1].geometry.matrix(), (WeightShardMatrix{512, 256, 1}));
                    EXPECT_EQ(expert.gate_up_down[2].geometry.matrix(), (WeightShardMatrix{256, 512, 1}));
                    for (const auto &weight : expert.gate_up_down) EXPECT_EQ(weight.executionFormat(), "BF16");
                }
            }
        }
    }
}

TEST(PlanningForwardWeightWork, TiedHeadRetainsSourceIdentityAndUnknownWorkCannotDisappear)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    const auto candidate = admit(proposal(source, DeviceType::CPU, OrchestrationStrategy::SingleDevice), source);
    auto profile = source.metadata().memoryProfile();
    std::erase_if(profile.tensors, [](const auto &tensor) { return tensor.name == "output.weight"; });
    profile.tensors.push_back({"blk.0.future_op.weight", 256 * 4, "F32", 256, 256, 0});
    const auto work = compilePlanningForwardWeightWork(PlanningModelMetadata(profile, 2), candidate, PlanningMainForwardPhase::Decode);
    EXPECT_EQ(find(work[0], "token_embd.weight", WeightRole::LMHead).geometry.matrix(), (WeightShardMatrix{320, 256, 1}));
    EXPECT_EQ(find(work[0], "token_embd.weight", WeightRole::Embedding).geometry.matrix(), (WeightShardMatrix{320, 256, 1}));
    EXPECT_EQ(std::count_if(work[0].ordinary.begin(), work[0].ordinary.end(), [](const auto &item) {
        return std::holds_alternative<PlanningUnclassifiedWeight>(item);
    }), 1);
}

TEST(PlanningForwardWeightWork, DenseOnlyContinuationDoesNotAcquireAnExpertServiceEndpoint)
{
    test::PlanningGGUFFixture file(true);
    PlanningModelSource source(file.path());
    for (auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        auto request = proposal(source, backend, OrchestrationStrategy::ExpertOverlay);
        auto &overlay = *request.config.moe_routed_expert_plan;
        const auto continuation = overlay.continuation_domain;
        // Keep the normalized hardware declaration: continuation still uses
        // that domain. Only tier membership grants routed expert ownership.
        std::erase_if(overlay.routed_tiers, [&](const auto &tier) { return tier.domain == continuation; });
        const auto candidate = admit(std::move(request), source);
        const auto work = compilePlanningForwardWeightWork(source.metadata(), candidate, PlanningMainForwardPhase::Decode);
        ASSERT_EQ(work.size(), 2u);
        for (const auto &participant : work)
        {
            EXPECT_EQ(participant.routed.empty(), participant.device.is_gpu());
            EXPECT_EQ(participant.ordinary.empty(), participant.device.is_cpu());
        }
    }
}

TEST(PlanningForwardWeightWork, IncompleteOrMalformedRoutedWorkIsFatal)
{
    test::PlanningGGUFFixture file(true);
    PlanningModelSource source(file.path());
    const auto candidate = admit(proposal(source, DeviceType::CPU, OrchestrationStrategy::SingleDevice), source);
    for (int defect : {0, 1, 2, 3})
    {
        auto profile = source.metadata().memoryProfile();
        auto up = std::find_if(profile.tensors.begin(), profile.tensors.end(), [](const auto &tensor) {
            return tensor.name == "blk.0.ffn_up_exps.weight";
        });
        ASSERT_NE(up, profile.tensors.end());
        if (defect == 0) profile.tensors.erase(up);
        if (defect == 1) up->K = 0;
        if (defect == 2) up->K *= 2;
        if (defect == 3) profile.expert_used_count = profile.expert_count + 1;
        EXPECT_THROW(compilePlanningForwardWeightWork(PlanningModelMetadata(profile, 2), candidate,
            PlanningMainForwardPhase::Decode), std::invalid_argument);
    }
}

TEST(PlanningForwardWeightWork, UniformRoutingExpectationMatchesExhaustiveDistinctTopK)
{
    // Exhaust all 6^3 equally likely three-token router streams. Within each
    // token the two experts are distinct; assuming independent slots would
    // produce a different nonempty-expert count and fail this oracle.
    const std::vector<std::pair<int, int>> routes{{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
    double rows = 0, active = 0;
    for (auto first : routes) for (auto second : routes) for (auto third : routes)
    {
        std::array<int, 4> histogram{};
        for (auto route : {first, second, third})
        {
            ++histogram[route.first];
            ++histogram[route.second];
        }
        rows += histogram[0] + histogram[1];
        active += (histogram[0] != 0) + (histogram[1] != 0);
    }
    const auto expected = PlanningExpertExecutionShare(PlanningExpertExecution::OwnedExperts, 2)
        .uniformExpectation(4, 2, 3);
    EXPECT_DOUBLE_EQ(expected.routed_rows, rows / 216);
    EXPECT_DOUBLE_EQ(expected.nonempty_experts, active / 216);
    EXPECT_LT(expected.nonempty_experts, expected.routed_rows);
}

TEST(PlanningForwardWeightWork, UniformRoutingBoundariesAndReplicaSemanticsAreExplicit)
{
    const auto full = PlanningExpertExecutionShare(PlanningExpertExecution::ReplicatedExperts, 8);
    const auto assigned = PlanningExpertExecutionShare(PlanningExpertExecution::BalancedReplicaAssignment, 8, 2);
    for (size_t rows : {0u, 1u, 32u, 8192u})
    {
        const auto replicated = full.uniformExpectation(8, 2, rows);
        const auto apportioned = assigned.uniformExpectation(8, 2, rows);
        EXPECT_DOUBLE_EQ(replicated.routed_rows, 2.0 * rows);
        EXPECT_DOUBLE_EQ(apportioned.routed_rows, replicated.routed_rows / 2);
        EXPECT_DOUBLE_EQ(apportioned.nonempty_experts, replicated.nonempty_experts / 2);
        EXPECT_LE(replicated.nonempty_experts, 8);
        EXPECT_LE(replicated.nonempty_experts, replicated.routed_rows);
    }
    EXPECT_DOUBLE_EQ(full.uniformExpectation(8, 8, 7).nonempty_experts, 8);
    EXPECT_DOUBLE_EQ(full.uniformExpectation(8, 8, 0).nonempty_experts, 0);
    EXPECT_DOUBLE_EQ(PlanningExpertExecutionShare(PlanningExpertExecution::OwnedExperts, 0)
        .uniformExpectation(8, 2, 8192).routed_rows, 0);
    const int large = std::numeric_limits<int>::max();
    const auto rare = PlanningExpertExecutionShare(PlanningExpertExecution::OwnedExperts, large)
        .uniformExpectation(large, 1, 1);
    EXPECT_NEAR(rare.nonempty_experts, 1, 1e-14);
    EXPECT_THROW(PlanningExpertExecutionShare(PlanningExpertExecution::ReplicatedExperts, 8, 2), std::invalid_argument);
    EXPECT_THROW(PlanningExpertExecutionShare(PlanningExpertExecution::OwnedExperts, 8, 2), std::invalid_argument);
    EXPECT_THROW(PlanningExpertExecutionShare(PlanningExpertExecution::OwnedExperts, -1), std::invalid_argument);
    EXPECT_THROW(PlanningExpertExecutionShare(PlanningExpertExecution::BalancedReplicaAssignment, 8, 0), std::invalid_argument);
    EXPECT_THROW(full.uniformExpectation(0, 1, 1), std::invalid_argument);
    EXPECT_THROW(full.uniformExpectation(7, 2, 1), std::invalid_argument);
    EXPECT_THROW(full.uniformExpectation(8, 9, 1), std::invalid_argument);
    EXPECT_THROW(full.uniformExpectation(8, 0, 0), std::invalid_argument);
}

TEST(PlanningForwardWeightWork, AdmittedApportionmentConservesRoutesAcrossDevicesAndRemoteTier)
{
    // Format changes must not change route ownership. Sampling remains a later
    // operation; these sparse GGUF fixtures never allocate or use an accelerator.
    using T = GGUFTensorType;
    const std::vector formats{T::F32, T::F16, T::BF16, T::Q4_0, T::Q4_1, T::Q5_0, T::Q5_1, T::Q8_0,
        T::Q2_K, T::Q3_K, T::Q4_K, T::Q5_K, T::Q6_K, T::Q8_K, T::IQ1_S, T::IQ1_M,
        T::IQ2_XXS, T::IQ2_XS, T::IQ2_S, T::IQ3_XXS, T::IQ3_S, T::IQ4_NL, T::IQ4_XS};
    for (auto format : formats)
    for (auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        test::PlanningGGUFFixture file(true, false, format, 512);
        PlanningModelSource source(file.path());
        auto request = proposal(source, backend, OrchestrationStrategy::ExpertOverlay, 2);
        auto &plan = *request.config.moe_routed_expert_plan;
        for (auto &tier : plan.routed_tiers)
            if (tier.domain == plan.continuation_domain) tier.max_experts_per_layer = 6;
        const auto candidate = admit(std::move(request), source);
        for (auto phase : {PlanningMainForwardPhase::Prefill, PlanningMainForwardPhase::Decode})
        {
            const auto work = compilePlanningForwardWeightWork(source.metadata(), candidate, phase);
            ASSERT_EQ(work.size(), 3u);
            std::map<int, double> total;
            for (const auto &participant : work)
            for (const auto &expert : participant.routed)
            {
                ASSERT_EQ(expert.execution_shares.size(), 1u);
                const auto &share = expert.execution_shares.front();
                EXPECT_EQ(share.execution(), PlanningExpertExecution::OwnedExperts);
                EXPECT_EQ(share.experts(), participant.device.is_cpu() ? 2 : 3);
                const auto expected = expert.uniformExpectation(32);
                EXPECT_DOUBLE_EQ(expected.routed_rows, share.experts() * 8.0);
                total[expert.layer] += expected.routed_rows;
            }
            ASSERT_EQ(total.size(), 2u);
            for (const auto &[layer, rows] : total) EXPECT_DOUBLE_EQ(rows, 32 * 2);
        }
    }
}

TEST(PlanningForwardWeightWork, AdmittedReplicaDecodeDoesNotClaimPrefillAssignmentSpeedup)
{
    test::PlanningGGUFFixture file(true);
    PlanningModelSource source(file.path());
    for (auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (auto split : {RoutedExpertPhasePolicy::Uniform, RoutedExpertPhasePolicy::PrefillApportionedDecodeReplicated})
    {
        auto request = proposal(source, backend, OrchestrationStrategy::ExpertOverlay, 2);
        auto &plan = *request.config.moe_routed_expert_plan;
        for (auto &tier : plan.routed_tiers)
            if (tier.domain == plan.continuation_domain) tier.max_experts_per_layer = 6;
        for (auto &domain : plan.domains)
            if (domain.name == plan.continuation_domain)
            {
                domain.routed_compute_policy = RoutedExpertComputePolicy::Replicated;
                domain.routed_phase_policy = split;
                // The candidate intentionally retains its declarative hardware
                // definitions. Change the fixture intent consistently before
                // asking the production compiler to authenticate both views.
                for (auto &definition : request.config.domain_definitions)
                    if (definition.name == domain.name)
                        definition = DomainDefinition::fromExecutionDomainDefinition(domain.toExecutionDomainDefinition());
            }
        const auto candidate = admit(std::move(request), source);
        for (auto phase : {PlanningMainForwardPhase::Prefill, PlanningMainForwardPhase::Decode})
        for (const auto &participant : compilePlanningForwardWeightWork(source.metadata(), candidate, phase))
        for (const auto &expert : participant.routed)
        {
            const auto expected = expert.uniformExpectation(32);
            if (participant.device.is_cpu()) EXPECT_DOUBLE_EQ(expected.routed_rows, 16);
            else
            {
                const bool assigned = phase == PlanningMainForwardPhase::Prefill &&
                    split == RoutedExpertPhasePolicy::PrefillApportionedDecodeReplicated;
                ASSERT_EQ(expert.execution_shares.size(), 1u);
                EXPECT_EQ(expert.execution_shares.front().experts(), 6);
                EXPECT_EQ(expert.execution_shares.front().assignmentParticipants(), assigned ? 2 : 1);
                EXPECT_DOUBLE_EQ(expected.routed_rows, assigned ? 24 : 48);
            }
        }
    }
}
