/**
 * @file Test__PlanningForwardWeightWork.cpp
 * @brief Device-free work ownership proofs using the real compiler and GGUF metadata.
 *
 * Sparse fixtures never execute inference. They distinguish resident capacity
 * from issued main-forward work, exact TP axes from element fractions, both
 * rank namespaces, and continuation operands from routed-only service. No
 * synthetic inventory or work count is claimed as measured performance.
 */
#include "../../utils/ScopedGDNProjectionArithmetic.h"
#include "planning/PlanningForwardWeightWork.h"
#include "planning/OrchestrationCandidateAdmission.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/PlanningHostServiceFixture.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include "../../utils/QuantizedVerifierFormats.h"
#include "tensors/NativeQuantisedFormatInfo.h"
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
            {
                for (int index = 0; index < gpu_count; ++index)
                    record.gpus.push_back({.type = backend, .local_device_id = index + 2,
                        .memory_bytes = 32ull << 30, .free_memory_bytes = 24ull << 30,
                        .compute_units = 64, .uuid = "card-" + std::to_string(index), .numa_node = 3});
                // A typed projection candidate needs a complete topology
                // observation. This fixture declares no off-diagonal P2P;
                // it does not infer a link from its synthetic timing/capacity.
                auto &matrix = backend == DeviceType::CUDA ? record.p2p_cuda : record.p2p_rocm;
                auto &count = backend == DeviceType::CUDA ? record.p2p_cuda_count : record.p2p_rocm_count;
                count = gpu_count;
                matrix.assign(static_cast<size_t>(gpu_count) * gpu_count, false);
                for (int index = 0; index < gpu_count; ++index)
                    matrix[static_cast<size_t>(index) * gpu_count + index] = true;
            }
            result.ranks.push_back(std::move(record));
        }
        result.buildNodeAggregations();
        return result;
    }

    /**
     * @brief Enumerate a real proposal without confusing visible capacity with requested degree.
     * @param source Original model metadata used by the production candidate compiler.
     * @param backend Backend of the fixture's observed devices.
     * @param strategy Requested candidate family.
     * @param gpu_count Number of physically distinct visible fixture GPUs.
     * @param routed_policy Actual expert projection ownership policy.
     * @param required_gpu_count Optional public cardinality constraint on selected compute.
     * @return Last complete proposal of the requested family, preserving discovery identity.
     * @throws std::logic_error When the declared inventory cannot produce that proposal.
     */
    AutomaticOrchestrationCandidate proposal(const PlanningModelSource &source, DeviceType backend,
        OrchestrationStrategy strategy, int gpu_count = 1,
        RoutedExpertComputePolicy routed_policy = RoutedExpertComputePolicy::Apportioned,
        std::optional<int> required_gpu_count = std::nullopt)
    {
        OrchestrationConfig request;
        request.model_path = source.path();
        request.max_seq_len = 1024;
        request.mtp.enabled = false;
        request.mtp_activation_policy = MTPActivationPolicy::Disabled;
        request.mtp.graph_capacity_draft_tokens = source.metadata().memoryProfile().mtp_layer_count > 0 ? 15 : 0;
        request.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
        request.routed_expert_compute_policy = routed_policy;
        if (routed_policy == RoutedExpertComputePolicy::GateUpOwnedDownColumns)
            request.automatic_planning.device_counts = {{backend, gpu_count}};
        if (required_gpu_count)
            request.automatic_planning.device_counts = {{backend, *required_gpu_count}};
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
        if (!result) throw std::logic_error(std::string("No fixture candidate: backend=") + deviceTypeToString(backend) +
            " strategy=" + std::to_string(static_cast<int>(strategy)) + " model=" + source.path());
        return std::move(*result);
    }

    /** @return Complete admission with a deliberately small GPU capture ladder. */
    AdmittedOrchestrationCandidate admit(AutomaticOrchestrationCandidate candidate, const PlanningModelSource &source)
    {
        OrchestrationCandidateMemoryPolicy policy{
            .prefill_bucket_rows = {32, 64}, .minimum_prefill_sequence_rows = 1, .maximum_cached_prefill_buckets = 8};
        policy.retained_host_services = test::planningHostServiceBudgets(2);
        return AdmittedOrchestrationCandidate::admit(std::move(candidate), source,
            policy);
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
    const test::ScopedGDNProjectionArithmetic arithmetic(GDNProjectionArithmetic::Native);
    test::PlanningGGUFFixture file(false, true);
    PlanningModelSource source(file.path());
    std::vector<std::string> formats{"F32", "F16", "BF16"};
    for (const auto &entry : native_quantised_formats::kAllSourceFormats) formats.emplace_back(entry.quant_type);
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

/**
 * @test An MTP GGUF with explicit serial execution admits only its main consumers.
 *
 * Sweep every native codebook and CPU/CUDA/ROCm through single-device, TP, PP,
 * and routed multi-device MoE/CPU-overlay candidates. The real compiler and PMA
 * admission run without loading payloads. Main-forward operands may never name
 * the trailing sidecar, including when its retained envelope is requested.
 */
TEST(PlanningForwardWeightWork, DisabledMTPGGUFAdmissionAllFormatsBackendsAndTopologies)
{
    for (const bool moe : {false, true})
    {
        std::vector<TensorType> formats{TensorType::FP32, TensorType::FP16, TensorType::BF16};
        for (const auto &format : test::quantizedVerifierFormats())
            formats.push_back(format.tensor_type);
        for (const auto format : formats)
        {
            if (format == TensorType::Q8_1)
            {
                // Q8_1 is accepted by runtime preparation but has no supported
                // GGUF source encoding. Prove that boundary rather than forge one.
                EXPECT_THROW(test::PlanningGGUFFixture::sourceType(format), std::invalid_argument);
                continue;
            }
            test::PlanningGGUFFixture file(moe, true, test::PlanningGGUFFixture::sourceType(format));
            PlanningModelSource source(file.path());
            for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
                for (const auto strategy : {OrchestrationStrategy::SingleDevice,
                     OrchestrationStrategy::TensorParallel, OrchestrationStrategy::PipelineParallel,
                     OrchestrationStrategy::ExpertOverlay})
                {
                    if (moe && strategy == OrchestrationStrategy::PipelineParallel)
                        continue; // The canonical candidate inventory admits PP for dense models.
                    if (strategy == OrchestrationStrategy::ExpertOverlay && (!moe || backend == DeviceType::CPU))
                        continue; // A routed overlay requires routed experts and distinct tier backends.
                    SCOPED_TRACE(::testing::Message() << "moe=" << moe << " format=" << static_cast<int>(format)
                        << " backend=" << deviceTypeToString(backend) << " strategy=" << static_cast<int>(strategy));
                    for (const int retained_depth : {0, 15})
                    {
                        auto request = proposal(source, backend, strategy, 2);
                        request.config.mtp.graph_capacity_draft_tokens = retained_depth;
                        const auto admitted = admit(std::move(request), source);
                        EXPECT_FALSE(admitted.config().mtp.enabled);
                        EXPECT_EQ(admitted.config().mtp_activation_policy, MTPActivationPolicy::Disabled);
                        EXPECT_EQ(admitted.config().mtp.graph_capacity_draft_tokens, retained_depth);
                        EXPECT_GT(admitted.physicalAdmission().plan().totalBytes(), 0u);
                        for (const auto &rank : admitted.rankPlans())
                        {
                            EXPECT_FALSE(rank.runtime.mtp.enabled);
                            EXPECT_LT(rank.last_layer, 2);
                        }
                        for (const auto phase : {PlanningMainForwardPhase::Prefill, PlanningMainForwardPhase::Decode})
                            for (const auto &participant : compilePlanningForwardWeightWork(source.metadata(), admitted, phase))
                            {
                                for (const auto &operation : participant.ordinary)
                                {
                                    const auto &weight = operand(operation);
                                    EXPECT_LT(weight.layer, 2);
                                    EXPECT_FALSE(weight.source_name.starts_with("blk.2."));
                                }
                                for (const auto &routed : participant.routed)
                                    for (const auto &weight : routed.gate_up_down)
                                    {
                                        EXPECT_LT(weight.layer, 2);
                                        EXPECT_FALSE(weight.source_name.starts_with("blk.2."));
                                    }
                            }
                    }
                }
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

/**
 * @brief The installed projection layout must have a real automatic work description.
 *
 * Metadata-only admission already supports this homogeneous GPU mode. Its
 * costing compiler must preserve that mode rather than reject it as an unknown
 * whole-expert policy. Every source format uses the same ownership authority;
 * neither payload conversion nor a device is needed to prove this contract.
 */
TEST(PlanningForwardWeightWork, ProjectionOwnershipCompilesBothPhasesAllFormats)
{
    using T = GGUFTensorType;
    const std::vector formats{T::F32, T::F16, T::BF16, T::Q4_0, T::Q4_1, T::Q5_0, T::Q5_1, T::Q8_0,
        T::Q2_K, T::Q3_K, T::Q4_K, T::Q5_K, T::Q6_K, T::Q8_K, T::IQ1_S, T::IQ1_M,
        T::IQ2_XXS, T::IQ2_XS, T::IQ2_S, T::IQ3_XXS, T::IQ3_S, T::IQ4_NL, T::IQ4_XS};
    for (const auto format : formats)
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (const int degree : {2, 4, 8})
    {
        SCOPED_TRACE(::testing::Message() << int(format) << "/" << int(backend) << "/" << degree);
        // Eight-way GQA needs at least two Q heads per participant when the
        // two KV heads are replicated. Keep source matrices/normalizers
        // consistent with sixteen heads rather than weakening shard validation.
        test::PlanningGGUFFixture file(true, false, format, 512, std::nullopt, 16);
        PlanningModelSource source(file.path());
        const auto candidate = admit(proposal(source, backend,
            OrchestrationStrategy::TensorParallel, degree,
            RoutedExpertComputePolicy::GateUpOwnedDownColumns), source);
        ASSERT_EQ(candidate.devicePlans().size(), static_cast<size_t>(degree));
        for (const auto phase : {PlanningMainForwardPhase::Prefill, PlanningMainForwardPhase::Decode})
        {
            std::vector<PlanningParticipantWeightWork> work;
            ASSERT_NO_THROW(work = compilePlanningForwardWeightWork(source.metadata(), candidate, phase));
            ASSERT_EQ(work.size(), static_cast<size_t>(degree));
            double owned_routes = 0;
            for (size_t index = 0; index < work.size(); ++index)
            {
                EXPECT_EQ(candidate.devicePlans()[index].routed_compute_policy,
                    RoutedExpertComputePolicy::GateUpOwnedDownColumns);
                EXPECT_EQ(work[index].routed.size(), 2u);
                for (const auto &expert : work[index].routed)
                {
                    ASSERT_TRUE(expert.projection_ownership);
                    EXPECT_EQ(*expert.projection_ownership,
                        MoEExpertProjectionOwnership::gateUpOwnedDownColumns(
                            {8, 256, 512}, static_cast<int>(index), degree));
                    EXPECT_EQ(expert.projectionMatrix(WeightRole::MoEExpertGate), (WeightShardMatrix{512, 256, 1}));
                    EXPECT_EQ(expert.projectionMatrix(WeightRole::MoEExpertUp), (WeightShardMatrix{512, 256, 1}));
                    EXPECT_EQ(expert.projectionMatrix(WeightRole::MoEExpertDown),
                        (WeightShardMatrix{static_cast<size_t>(256 / degree), 512, 1}));
                    const auto gate = expert.projectionExpectation(WeightRole::MoEExpertGate, 33);
                    const auto down = expert.projectionExpectation(WeightRole::MoEExpertDown, 33);
                    EXPECT_DOUBLE_EQ(gate.routed_rows, expert.uniformExpectation(33).routed_rows);
                    EXPECT_DOUBLE_EQ(down.routed_rows, 66);
                    EXPECT_DOUBLE_EQ(expert.projectionExpectation(WeightRole::MoEExpertDown, 0).routed_rows, 0);
                    if (expert.layer == 0) owned_routes += gate.routed_rows;
                    // Ownership has one authority. A forged source shape or
                    // semantic role must not become a cheaper candidate.
                    auto corrupt = expert;
                    corrupt.gate_up_down[2] = expert.gate_up_down[0];
                    EXPECT_THROW(corrupt.projectionMatrix(WeightRole::MoEExpertDown), std::invalid_argument);
                    EXPECT_THROW(expert.projectionMatrix(WeightRole::FFNDown), std::invalid_argument);
                }
            }
            EXPECT_DOUBLE_EQ(owned_routes, 66);
        }
    }
}

/**
 * @brief Replay QFN source-directory HC/QSA/PLE roles through canonical work compilation.
 *
 * The compact dense GGUF supplies admitted CPU/CUDA/ROCm topology fixtures;
 * appended metadata preserves actual QFN extension N/K/format observations
 * without loading their payloads or advertising a synthetic model certificate.
 */
TEST(PlanningForwardWeightWork, QFNExtensionRolesPreserveGeometryRowsAndTerminalStage)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    auto profile = source.metadata().memoryProfile();
    profile.architecture = "qwen4exp";
    const auto add = [&](const std::string &name, size_t n, size_t k, const std::string &format, int layer) {
        const size_t elements = n * k;
        const size_t bytes = format == "Q8_0" ? (elements / 32u) * 34u : elements * (format == "BF16" ? 2u : 4u);
        profile.tensors.push_back({name, bytes, format, elements, k, layer});
    };
    add("output_hc_down.weight", 320, 10240, "Q8_0", -1);
    add("output_hc_up.weight", 10240, 320, "Q8_0", -1);
    add("output_hc_norm.weight", 1, 10240, "F32", -1);
    for (int layer = 0; layer < 2; ++layer)
        for (const auto branch : {"attn", "ffn"})
        {
            const auto prefix = "blk." + std::to_string(layer) + ".hc_" + branch;
            add(prefix + "_down.weight", 320, 10240, "Q8_0", layer);
            add(prefix + "_up.weight", 10240, 320, "Q8_0", layer);
            add(prefix + "_inject.weight", 4, 10240, "F32", layer);
            add(prefix + "_norm.weight", 1, 10240, "F32", layer);
        }
    // Source indexer query gains are reused per head; no hidden-width shard
    // may replace the actual N512/N128 projection or 128-element gain.
    add("blk.1.indexer.q_proj.weight", 512, 2560, "BF16", 1);
    add("blk.1.indexer.k_proj.weight", 128, 2560, "BF16", 1);
    add("blk.1.indexer.q_norm.weight", 1, 128, "F32", 1);
    add("blk.1.indexer.k_norm.weight", 1, 128, "F32", 1);
    add("blk.1.ple_key.weight", 10240, 2560, "Q8_0", 1);
    add("blk.1.ple_value.weight", 2560, 2560, "Q8_0", 1);
    for (const auto suffix : {"key", "query", "conv"})
        add(std::string("blk.1.ple_norm_") + suffix + ".weight", 1, 10240, "F32", 1);
    add("blk.1.ple_conv1d.weight", 10240, 4, "F32", 1);
    const PlanningModelMetadata model(profile, 2);
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        for (const auto strategy : {OrchestrationStrategy::SingleDevice,
                                   backend == DeviceType::CPU ? OrchestrationStrategy::PipelineParallel : OrchestrationStrategy::TensorParallel})
        {
            SCOPED_TRACE(std::string(deviceTypeToString(backend)) + "/" + std::to_string(int(strategy)));
            // Four visible cards do not imply a four-card proposal: ordinary
            // enumeration also emits smaller subsets. Declare the actual
            // tested degree through the same public hint as production.
            const auto candidate = admit(proposal(source, backend, strategy,
                strategy == OrchestrationStrategy::TensorParallel ? 4 : 1,
                RoutedExpertComputePolicy::Apportioned,
                strategy == OrchestrationStrategy::TensorParallel ? std::optional<int>{4} : std::nullopt), source);
            for (const auto phase : {PlanningMainForwardPhase::Prefill, PlanningMainForwardPhase::Decode})
            {
                const auto work = compilePlanningForwardWeightWork(model, candidate, phase);
                std::size_t terminal_uses = 0;
                for (const auto &participant : work)
                {
                    for (const auto &operation : participant.ordinary)
                    {
                        const auto &weight = operand(operation);
                        EXPECT_FALSE(std::holds_alternative<PlanningUnclassifiedWeight>(operation)) << weight.source_name;
                        if (weight.role == WeightRole::HyperConnectionProjection || weight.role == WeightRole::SparseIndexerProjection ||
                            weight.role == WeightRole::PerLayerEmbeddingProjection)
                            EXPECT_TRUE(std::holds_alternative<PlanningProjectionWeight>(operation)) << weight.source_name;
                        if (weight.source_name.starts_with("output_hc_"))
                        {
                            ++terminal_uses;
                            EXPECT_EQ(participant.last_layer, 1);
                            EXPECT_TRUE(planningTerminalWeight(weight.role, weight.layer));
                            EXPECT_EQ(planningWeightInvocationRows(weight.role, phase == PlanningMainForwardPhase::Prefill ? 512 : 1),
                                      phase == PlanningMainForwardPhase::Prefill ? 512 : 1);
                        }
                    }
                    if (participant.last_layer == 1)
                    {
                        EXPECT_EQ(find(participant, "output_hc_down.weight", WeightRole::HyperConnectionProjection).geometry.matrix(),
                                  (WeightShardMatrix{320, 10240, 1}));
                        EXPECT_EQ(find(participant, "output_hc_up.weight", WeightRole::HyperConnectionProjection).geometry.matrix(),
                                  (WeightShardMatrix{10240, 320, 1}));
                        EXPECT_EQ(find(participant, "output_hc_norm.weight", WeightRole::HyperConnectionNorm).geometry.elements(), 10240u);
                    }
                    if (participant.first_layer <= 1 && participant.last_layer >= 1)
                    {
                        EXPECT_EQ(find(participant, "blk.1.ple_key.weight", WeightRole::PerLayerEmbeddingProjection).geometry.matrix(),
                                  (WeightShardMatrix{10240, 2560, 1}));
                        EXPECT_EQ(find(participant, "blk.1.indexer.q_proj.weight", WeightRole::SparseIndexerProjection).geometry.matrix(),
                                  (WeightShardMatrix{512, 2560, 1}));
                        EXPECT_EQ(find(participant, "blk.1.ple_conv1d.weight", WeightRole::PerLayerEmbeddingConvolution).geometry.elements(), 40960u);
                    }
                }
                EXPECT_EQ(terminal_uses, strategy == OrchestrationStrategy::TensorParallel ? 12u : 3u);
            }
        }
    EXPECT_EQ(planningWeightInvocationRows(WeightRole::LMHead, 512), 1);
    EXPECT_EQ(planningWeightInvocationRows(WeightRole::OutputNorm, 512), 1);
    EXPECT_FALSE(planningTerminalWeight(WeightRole::HyperConnectionProjection, 0));
    EXPECT_THROW((void)planningWeightInvocationRows(WeightRole::HyperConnectionProjection, 0), std::invalid_argument);
}

/** @test Immutable source indices preserve canonical roles through copies and wire reconstruction. */
TEST(PlanningForwardWeightWork, RetainedClassificationIsSourceBoundAndIndexesOwnedLayers) {
    test::PlanningGGUFFixture file(true,true,GGUFTensorType::Q4_0);
    PlanningModelSource source(file.path());
    const auto &model=source.metadata();
    const auto check=[](const PlanningModelMetadata &metadata) {
        const auto &profile=metadata.memoryProfile();std::vector<size_t> visits(profile.tensors.size(),0);
        for(const size_t index:metadata.globalTensorIndices()) {
            EXPECT_LT(index,profile.tensors.size());EXPECT_LT(profile.tensors[index].layer_index,0);++visits[index];
        }
        for(int layer=0;layer<profile.n_layers;++layer)
            for(const size_t index:metadata.layerTensorIndices(layer)) {
                EXPECT_LT(index,profile.tensors.size());EXPECT_EQ(profile.tensors[index].layer_index,layer);++visits[index];
            }
        for(size_t index=0;index<profile.tensors.size();++index) {
            const auto view=metadata.classifiedTensor(index);
            EXPECT_EQ(&view.profile(),&profile);EXPECT_EQ(&view.tensor(),&profile.tensors[index]);
            EXPECT_EQ(view.index(),index);EXPECT_EQ(view.role(),inferWeightRole(profile.tensors[index].name));
            EXPECT_EQ(visits[index],1u);
        }
        EXPECT_THROW(metadata.classifiedTensor(profile.tensors.size()),std::out_of_range);
        EXPECT_THROW(metadata.layerTensorIndices(-1),std::invalid_argument);
        EXPECT_THROW(metadata.layerTensorIndices(profile.n_layers),std::invalid_argument);
    };
    check(model);auto copy=model;check(copy);
    EXPECT_EQ(copy.tensorIndex(),model.tensorIndex());
    std::shared_ptr<const PlanningTensorIndex> retained;
    { PlanningModelMetadata ephemeral(model.memoryProfile(),model.mainLayerCount());retained=ephemeral.tensorIndex(); }
    EXPECT_EQ(retained->profile().tensors.size(),model.memoryProfile().tensors.size());
    EXPECT_NE(retained->findTensor(model.memoryProfile().tensors.front().name),nullptr);
    EXPECT_THROW(retained->requireProfile(model.memoryProfile()),std::invalid_argument);
    const auto restored=PlanningModelMetadata::deserialize(model.serialize());check(restored);
    ASSERT_TRUE(model.firstTensorWithRole(WeightRole::Embedding));
    EXPECT_EQ(model.classifiedTensor(*model.firstTensorWithRole(WeightRole::Embedding)).role(),WeightRole::Embedding);
    auto extended=model.memoryProfile();const auto original_layer_size=model.layerTensorIndices(0).size();
    for(size_t index=0;index<1000;++index)extended.tensors.push_back({
        .name="blk.1.additional_norm_"+std::to_string(index)+".weight",.native_bytes=4,.quant_type="F32",.elements=1,.K=1,.layer_index=1});
    const PlanningModelMetadata larger(std::move(extended),model.mainLayerCount());
    EXPECT_EQ(larger.layerTensorIndices(0).size(),original_layer_size);
    EXPECT_EQ(larger.layerTensorIndices(1).size(),model.layerTensorIndices(1).size()+1000);
    EXPECT_EQ(larger.globalTensorIndices().size(),model.globalTensorIndices().size());
    EXPECT_EQ(model.memoryProfile().tensors.size()+1000,larger.memoryProfile().tensors.size());
}

/** @test Retained typed roles use the same sharding arithmetic and cannot cross source authorities. */
TEST(PlanningForwardWeightWork, PreclassifiedGeometryMatchesCanonicalResolverAndRejectsForeignViews) {
    test::PlanningGGUFFixture file(true,true,GGUFTensorType::Q4_0);
    PlanningModelSource source(file.path());const auto &model=source.metadata();const PlanningModelMetadata foreign(model.memoryProfile(),model.mainLayerCount());
    for(const auto backend:{DeviceType::CPU,DeviceType::CUDA,DeviceType::ROCm})
    for(const int degree:{1,2})for(int shard=0;shard<degree;++shard) {
        SCOPED_TRACE(::testing::Message()<<int(backend)<<"/"<<degree<<"/"<<shard);
        const WeightShardGeometryResolver resolver(model.memoryProfile(),DeviceId(backend,0),shard,degree);
        for(size_t index=0;index<model.memoryProfile().tensors.size();++index) {
            const auto view=model.classifiedTensor(index);
            const auto original=resolver.resolve(view.tensor());const auto retained=resolver.resolve(view);
            EXPECT_EQ(retained.elements(),original.elements());EXPECT_EQ(retained.matrix(),original.matrix());
            if(isRoutedExpertRole(view.role())) {
                const auto expert=resolver.resolve(view.tensor(),1);const auto typed=resolver.resolve(view,1);
                EXPECT_EQ(typed.elements(),expert.elements());EXPECT_EQ(typed.matrix(),expert.matrix());
            }
        }
        EXPECT_THROW(resolver.resolve(foreign.classifiedTensor(0)),std::invalid_argument);
    }
}

/** @test Retained source facts preserve every canonical byte quote and reject foreign profile/lifetime reuse. */
TEST(PlanningForwardWeightWork, RetainedSourceWorkspaceAndWeightQuotesMatchCanonicalBytes) {
    test::PlanningGGUFFixture file(true,true,GGUFTensorType::Q4_0);
    PlanningModelSource source(file.path());const auto &model=source.metadata();const auto &profile=model.memoryProfile();
    for(const auto backend:{DeviceType::CPU,DeviceType::CUDA,DeviceType::ROCm})
    for(const int rows:{1,32}) {
        SCOPED_TRACE(::testing::Message()<<int(backend)<<"/"<<rows);
        WorkspaceMemoryGeometry plain;
        plain.device=DeviceId(backend,0);plain.device_compute_units=backend==DeviceType::CPU?2:64;
        plain.cpu_execution=test::kSyntheticCPUExecutionGeometry;plain.resident_graph_rows=rows;plain.max_context_rows=128;
        plain.owns_embedding=true;plain.first_layer=0;plain.last_layer=profile.n_layers-1;
        auto indexed=plain;indexed.source_tensor_index=model.tensorIndex();
        EXPECT_EQ(WorkspaceMemoryEstimator::estimate(profile,plain),WorkspaceMemoryEstimator::estimate(profile,indexed));
        auto policy=PlanningWeightSourcePolicy::current();policy.source_tensor_index=model.tensorIndex();
        const auto original=WeightMemoryEstimator::estimate(profile,plain.device);
        const auto retained=WeightMemoryEstimator::estimate(profile,plain.device,0,1,0,-1,{}, {},
            WeightComponentScope::EmbeddingAndTerminal,CPUStoragePlacement::local(),policy);
        EXPECT_EQ(original.native_bytes,retained.native_bytes);EXPECT_EQ(original.device_bytes,retained.device_bytes);
        EXPECT_EQ(original.prepared_embedding_bytes,retained.prepared_embedding_bytes);
        EXPECT_EQ(original.lm_head_bytes,retained.lm_head_bytes);EXPECT_EQ(original.tied_lm_head_bytes,retained.tied_lm_head_bytes);
        const PlanningModelMetadata foreign(profile,model.mainLayerCount());
        auto wrong=indexed;wrong.source_tensor_index=foreign.tensorIndex();
        EXPECT_THROW(WorkspaceMemoryEstimator::estimate(profile,wrong),std::invalid_argument);
        policy.source_tensor_index=foreign.tensorIndex();
        EXPECT_THROW(WeightMemoryEstimator::estimate(profile,plain.device,0,1,0,-1,{}, {},
            WeightComponentScope::EmbeddingAndTerminal,CPUStoragePlacement::local(),policy),std::invalid_argument);
    }
    // The handle, not a startup-local metadata address, retains the single source.
    std::shared_ptr<const PlanningTensorIndex> retained;
    { PlanningModelMetadata local(profile,model.mainLayerCount());retained=local.tensorIndex(); }
    WorkspaceMemoryGeometry lifetime;lifetime.source_tensor_index=retained;lifetime.device=DeviceId::rocm(0);
    lifetime.device_compute_units=64;lifetime.resident_graph_rows=1;lifetime.max_context_rows=128;
    lifetime.first_layer=0;lifetime.last_layer=retained->profile().n_layers-1;
    EXPECT_GT(WorkspaceMemoryEstimator::estimate(retained->profile(),lifetime),0u);
    auto changed=profile;changed.tensors.front().name="different-source.weight";
    EXPECT_THROW(retained->requireProfile(changed),std::invalid_argument);
}

/** @test Explicit appended predictor intervals remain present even beyond the nominal main-layer count. */
TEST(PlanningForwardWeightWork, RetainedSourceWeightQuotesIncludeAppendedSidecarLayers)
{
    test::PlanningGGUFFixture file(true,true,GGUFTensorType::Q4_0);
    PlanningModelSource source(file.path());
    auto original=source.metadata().memoryProfile();
    const int sidecar=source.metadata().mainLayerCount();
    ASSERT_GT(original.n_layers,sidecar);
    // Some bound runtime profiles retain the main count while their original
    // directory also contains an explicitly requested appended predictor.
    original.n_layers=sidecar;
    const PlanningModelMetadata metadata(std::move(original),sidecar);
    const auto &profile=metadata.memoryProfile();
    ASSERT_FALSE(metadata.layerTensorIndices(sidecar).empty());
    for(const size_t index:metadata.layerTensorIndices(sidecar))EXPECT_EQ(profile.tensors[index].layer_index,sidecar);
    size_t visited=0;
    metadata.tensorIndex()->forEachLayerTensor(sidecar,sidecar,[&](size_t){++visited;});
    EXPECT_EQ(visited,metadata.layerTensorIndices(sidecar).size());
    auto policy=PlanningWeightSourcePolicy::current();policy.source_tensor_index=metadata.tensorIndex();
    for(const auto backend:{DeviceType::CPU,DeviceType::CUDA,DeviceType::ROCm})
    for(const auto scope:{WeightComponentScope::LayersOnly,WeightComponentScope::Terminal,
                         WeightComponentScope::EmbeddingAndTerminal})
    for(const int first:{sidecar-1,sidecar})
    {
        SCOPED_TRACE(::testing::Message()<<int(backend)<<"/"<<int(scope)<<"/"<<first);
        const auto canonical=WeightMemoryEstimator::estimate(profile,DeviceId(backend,0),0,1,first,sidecar,{}, {},scope);
        const auto indexed=WeightMemoryEstimator::estimate(profile,DeviceId(backend,0),0,1,first,sidecar,{}, {},scope,
            CPUStoragePlacement::local(),policy);
        EXPECT_GT(canonical.device_bytes,0u);
        EXPECT_EQ(indexed.native_bytes,canonical.native_bytes);
        EXPECT_EQ(indexed.device_bytes,canonical.device_bytes);
        EXPECT_EQ(indexed.prepared_embedding_bytes,canonical.prepared_embedding_bytes);
        EXPECT_EQ(indexed.lm_head_bytes,canonical.lm_head_bytes);
        EXPECT_EQ(indexed.tied_lm_head_bytes,canonical.tied_lm_head_bytes);
    }
    // An omitted upper bound keeps its original canonical nominal-layer rule.
    const auto canonical=WeightMemoryEstimator::estimate(profile,DeviceId::rocm(0),0,1,0,-1,{}, {},WeightComponentScope::LayersOnly);
    const auto indexed=WeightMemoryEstimator::estimate(profile,DeviceId::rocm(0),0,1,0,-1,{}, {},WeightComponentScope::LayersOnly,
        CPUStoragePlacement::local(),policy);
    EXPECT_EQ(indexed.native_bytes,canonical.native_bytes);
    EXPECT_EQ(indexed.device_bytes,canonical.device_bytes);
}
