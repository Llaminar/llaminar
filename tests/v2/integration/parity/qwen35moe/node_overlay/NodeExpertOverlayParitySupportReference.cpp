/**
 * @file NodeExpertOverlayParitySupportReference.cpp
 * @brief SupportReference implementation of the shared node-overlay parity fixture.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "NodeExpertOverlayParitySupport.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

    /**
     * @brief Authenticate fresh-versus-reuse admission across every MPI rank.
     *
     * A retained ModelContext is rank-local, but the production runner's
     * initialization phases are collective. Consequently, one rank may not
     * enter the reuse constructor while a peer enters the fresh constructor.
     * The selected path is encoded in the phase identity itself: a partial
     * cache hit becomes a typed phase-identity mismatch before either runner
     * can issue a production collective.
     *
     * @param local_cache_hit Whether this rank holds the requested authority.
     * @param local_error Rank-local cache validation or retirement failure.
     * @return One unanimous admission or a precise collective diagnostic.
     */
    Qwen122CampaignModelAdmissionResult
    reachQwen122CampaignModelAdmission(
        MPI_Comm control_communicator,
        bool local_cache_hit,
        std::string_view local_error)
    {
        const bool local_valid = local_error.empty();
        const std::string_view phase_name =
            !local_valid
                ? "qwen122CampaignModelInvalid"
                : local_cache_hit
                      ? "qwen122CampaignModelReuse"
                      : "qwen122CampaignModelFresh";
        const auto consensus = MPIRankInitializationConsensus::reach(
            control_communicator,
            RankInitializationPhaseIdentity{
                .ordinal = 0u,
                .name = phase_name,
            },
            local_valid
                ? RankInitializationLocalOutcome::Succeeded
                : RankInitializationLocalOutcome::ReturnedFailure);

        if (consensus.outcome !=
            RankInitializationConsensusOutcome::AllRanksSucceeded)
        {
            std::ostringstream diagnostic;
            diagnostic
                << "122B campaign model admission was not rank-unanimous";
            if (!local_error.empty())
                diagnostic << ": local=" << local_error;
            if (!consensus.detail.empty())
                diagnostic << "; consensus=" << consensus.detail;
            return {
                .admission = Qwen122CampaignModelAdmission::Fresh,
                .succeeded = false,
                .diagnostic = diagnostic.str(),
            };
        }

        return {
            .admission = local_cache_hit
                             ? Qwen122CampaignModelAdmission::Reuse
                             : Qwen122CampaignModelAdmission::Fresh,
            .succeeded = true,
            .diagnostic = {},
        };
    }

    /**
     * @brief Require every MPI rank to choose the same retained-runner path.
     * @param control_communicator Per-cell test-control communicator.
     * @param local_cache_hit Whether this process owns the exact runner.
     * @param local_error Rank-local cache validation failure.
     * @return One unanimous fresh/retained decision or a fatal diagnostic.
     */
    Qwen122CampaignRunnerAdmissionResult
    reachQwen122CampaignRunnerAdmission(
        MPI_Comm control_communicator,
        bool local_cache_hit,
        std::string_view local_error)
    {
        const bool local_valid = local_error.empty();
        const std::string_view phase_name =
            !local_valid
                ? "qwen122CampaignRunnerInvalid"
                : local_cache_hit
                      ? "qwen122CampaignRunnerRetained"
                      : "qwen122CampaignRunnerFresh";
        const auto consensus = MPIRankInitializationConsensus::reach(
            control_communicator,
            RankInitializationPhaseIdentity{
                .ordinal = 0u,
                .name = phase_name,
            },
            local_valid
                ? RankInitializationLocalOutcome::Succeeded
                : RankInitializationLocalOutcome::ReturnedFailure);
        if (consensus.outcome !=
            RankInitializationConsensusOutcome::AllRanksSucceeded)
        {
            std::ostringstream diagnostic;
            diagnostic
                << "122B campaign runner admission was not rank-unanimous";
            if (!local_error.empty())
                diagnostic << ": local=" << local_error;
            if (!consensus.detail.empty())
                diagnostic << "; consensus=" << consensus.detail;
            return {
                .admission = Qwen122CampaignRunnerAdmission::Fresh,
                .succeeded = false,
                .diagnostic = diagnostic.str(),
            };
        }
        return {
            .admission = local_cache_hit
                             ? Qwen122CampaignRunnerAdmission::Retained
                             : Qwen122CampaignRunnerAdmission::Fresh,
            .succeeded = true,
            .diagnostic = {},
        };
    }

    /** @return The sole bounded 122B prepared-weight cache in this MPI process. */
    Qwen122OverlayModelContextCampaignCache &
    qwen122OverlayModelContextCampaignCache()
    {
        static Qwen122OverlayModelContextCampaignCache cache;
        return cache;
    }

    /** @return The one bounded deferred-reference queue in this test process. */
    DeferredMTPBranchCampaign &deferredMTPBranchCampaign()
    {
        static DeferredMTPBranchCampaign campaign;
        return campaign;
    }

    /**
     * @brief Retire the cache's final model owner and prove exact GPU recovery.
     *
     * The caller holds the cache mutex. Every ticket is captured while the
     * model and reusable workspace allocations are still live; only then is
     * the complete contract destroyed. Completion trims scoped runtime caches
     * and proves the admitted per-device bytes became driver-visible again.
     * CPU arenas are trimmed at this same terminal owner edge.
     *
     * @param cache Locked process-local campaign cache.
     * @param error Receives the first ownership or reclamation defect.
     * @return True when the cache was empty or every owner retired completely.
     */
    bool retireQwen122OverlayModelContextCampaignCacheLocked(
        Qwen122OverlayModelContextCampaignCache &cache,
        std::string *error)
    {
        if (error)
            error->clear();
        if (!cache.contract)
        {
            cache.runner.reset();
            cache.runner_identity.reset();
            cache.physical_identity.reset();
            cache.model_path.clear();
            return true;
        }

        try
        {
            /* A yielded runner still owns the reuse authority exclusively.
             * Destroy it first on every MPI rank so the ordinary production
             * seal publishes the contract consumed below. No graph or stream
             * owner may survive the final model retirement. */
            cache.runner.reset();
            cache.runner_identity.reset();
            /* The core final-owner transition validates the reusable contract,
             * captures its exact device BOM, releases every model/workspace
             * owner, and retires the complete participant set atomically. The
             * campaign cache deliberately has no second lifecycle protocol. */
            (void)retireExclusiveModelContextReuseContract(
                *cache.contract);
            cache.contract.reset();
            cache.runner_identity.reset();
            cache.physical_identity.reset();
            cache.model_path.clear();
        }
        catch (const std::exception &exception)
        {
            if (error)
            {
                *error = "122B campaign exact model retirement was incomplete: " +
                         std::string(exception.what());
            }
            return false;
        }
        return true;
    }

    /**
     * @brief Release the final immutable 122B authority before Python loads it.
     * @param error Receives a precise retirement failure.
     * @return True when all model-lifetime allocations were certified free.
     */
    bool releaseQwen122OverlayModelContextCampaignCache(std::string *error)
    {
        auto &cache = qwen122OverlayModelContextCampaignCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        return retireQwen122OverlayModelContextCampaignCacheLocked(
            cache, error);
    }

    /** @return A shell-safe single argument preserving every input byte. */
    std::string quoteDeferredReferenceArgument(const std::string &value)
    {
        std::string quoted = "'";
        for (const char character : value)
        {
            if (character == '\'')
                quoted += "'\"'\"'";
            else
                quoted += character;
        }
        quoted += '\'';
        return quoted;
    }

    /**
     * @brief Load one FP32/FP64 NPY tensor without constructing a parity fixture.
     * @param path Exact authenticated reference tensor path.
     * @param error Receives a precise load/type failure.
     * @return FP32 values, or an empty optional on failure.
     */
    std::optional<std::vector<float>> loadDeferredReferenceTensor(
        const std::filesystem::path &path,
        std::string *error)
    {
        try
        {
            const cnpy::NpyArray array = cnpy::npy_load(path.string());
            std::vector<float> values(array.num_vals);
            if (array.word_size == sizeof(float))
            {
                const float *const data = array.data<float>();
                std::copy(data, data + array.num_vals, values.begin());
            }
            else if (array.word_size == sizeof(double))
            {
                const double *const data = array.data<double>();
                std::transform(
                    data,
                    data + array.num_vals,
                    values.begin(),
                    [](double value) { return static_cast<float>(value); });
            }
            else
            {
                if (error)
                {
                    *error = "unsupported NPY word size " +
                             std::to_string(array.word_size) + " at " +
                             path.string();
                }
                return std::nullopt;
            }
            return values;
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = path.string() + ": " + exception.what();
            return std::nullopt;
        }
    }

    /**
     * @brief Compare sparse routing weights after aligning them by expert ID.
     * @return Mean sparse-vector cosine and maximum absolute expert-mass error.
     */
    std::pair<float, float> compareDeferredRoutingWeights(
        const std::vector<float> &actual_weights,
        const std::vector<float> &expected_weights,
        const std::vector<float> &actual_indices,
        const std::vector<float> &expected_indices,
        int top_k,
        int num_experts)
    {
        if (top_k <= 0 || num_experts <= 0 || actual_weights.empty() ||
            actual_weights.size() != expected_weights.size() ||
            actual_weights.size() != actual_indices.size() ||
            actual_weights.size() != expected_indices.size() ||
            actual_weights.size() % static_cast<size_t>(top_k) != 0u)
        {
            return {0.0f, std::numeric_limits<float>::infinity()};
        }

        const size_t rows =
            actual_weights.size() / static_cast<size_t>(top_k);
        double total_cosine = 0.0;
        float maximum_error = 0.0f;
        for (size_t row = 0; row < rows; ++row)
        {
            std::vector<float> actual_sparse(
                static_cast<size_t>(num_experts), 0.0f);
            std::vector<float> expected_sparse(
                static_cast<size_t>(num_experts), 0.0f);
            for (int index = 0; index < top_k; ++index)
            {
                const size_t offset =
                    row * static_cast<size_t>(top_k) +
                    static_cast<size_t>(index);
                const int actual_expert =
                    static_cast<int>(actual_indices[offset]);
                const int expected_expert =
                    static_cast<int>(expected_indices[offset]);
                if (actual_expert >= 0 && actual_expert < num_experts)
                {
                    actual_sparse[static_cast<size_t>(actual_expert)] =
                        actual_weights[offset];
                }
                if (expected_expert >= 0 && expected_expert < num_experts)
                {
                    expected_sparse[static_cast<size_t>(expected_expert)] =
                        expected_weights[offset];
                }
            }
            total_cosine += computeCosineSimilarity(
                actual_sparse.data(),
                expected_sparse.data(),
                actual_sparse.size());
            for (size_t expert = 0; expert < actual_sparse.size(); ++expert)
            {
                maximum_error = std::max(
                    maximum_error,
                    std::abs(actual_sparse[expert] - expected_sparse[expert]));
            }
        }
        return {
            static_cast<float>(total_cosine / static_cast<double>(rows)),
            maximum_error};
    }

    /**
     * @brief Complete one deferred context with the same route-aware math gate.
     * @param context Immutable live evidence copied before runner teardown.
     * @param error Receives every failed invariant for campaign diagnostics.
     * @return True only when all tensors and the aggregate semantic gate pass.
     */
    bool compareDeferredMTPBranchContext(
        const DeferredMTPBranchContext &context,
        std::string *error)
    {
        std::string reference_prefix =
            "decode_step" + std::to_string(context.reference_step) +
            "_BRANCH";
        for (const int32_t token : context.condition_tokens)
            reference_prefix += "_" + std::to_string(token);
        reference_prefix +=
            "_MTP" + std::to_string(context.reference_depth) + "_";

        std::map<std::string, std::vector<float>> references;
        for (const auto &checkpoint : context.checkpoints)
        {
            const auto path = std::filesystem::path(context.snapshot_dir) /
                              (reference_prefix + checkpoint.stage + ".npy");
            auto loaded = loadDeferredReferenceTensor(path, error);
            if (!loaded || loaded->empty())
                return false;
            if (loaded->size() > checkpoint.actual.size() &&
                !checkpoint.actual.empty() &&
                loaded->size() % checkpoint.actual.size() == 0u)
            {
                loaded->erase(
                    loaded->begin(),
                    loaded->end() -
                        static_cast<ptrdiff_t>(checkpoint.actual.size()));
            }
            if (loaded->size() != checkpoint.actual.size())
            {
                if (error)
                {
                    *error = context.test_name + " " + checkpoint.stage +
                             " element mismatch actual=" +
                             std::to_string(checkpoint.actual.size()) +
                             " reference=" +
                             std::to_string(loaded->size());
                }
                return false;
            }
            references.emplace(checkpoint.stage, std::move(*loaded));
        }

        std::ofstream csv(context.snapshot_csv_path, std::ios::app);
        if (!csv.is_open())
        {
            if (error)
                *error = "could not append " + context.snapshot_csv_path.string();
            return false;
        }

        bool context_finite = true;
        bool routing_indices_exact = true;
        bool routing_top1_match = true;
        float routing_overlap = 1.0f;
        bool routing_weights_equivalent = false;
        bool routed_expert_output_equivalent = false;
        bool lm_head_passed = false;
        ReferenceTopKContainmentResult lm_head_topk;
        MoERoutingBoundaryResult routing_boundary;
        double router_symmetric_kl =
            std::numeric_limits<double>::infinity();
        double numerical_cosine_sum = 0.0;
        size_t numerical_stage_count = 0u;
        size_t compared_stages = 0u;

        const auto actual_indices_it = std::find_if(
            context.checkpoints.begin(),
            context.checkpoints.end(),
            [](const DeferredMTPCheckpoint &checkpoint)
            { return checkpoint.stage == "MOE_ROUTING_INDICES"; });
        const auto reference_indices_it = references.find(
            "MOE_ROUTING_INDICES");
        const auto actual_router_it = std::find_if(
            context.checkpoints.begin(),
            context.checkpoints.end(),
            [](const DeferredMTPCheckpoint &checkpoint)
            { return checkpoint.stage == "MOE_ROUTER_OUTPUT"; });
        const auto reference_router_it = references.find(
            "MOE_ROUTER_OUTPUT");

        for (const auto &checkpoint : context.checkpoints)
        {
            const auto reference_it = references.find(checkpoint.stage);
            if (reference_it == references.end())
                return false;
            const auto &actual = checkpoint.actual;
            const auto &expected = reference_it->second;
            bool finite = true;
            bool exact_indices = true;
            double maximum_absolute_error = 0.0;
            for (size_t index = 0; index < actual.size(); ++index)
            {
                finite = finite && std::isfinite(actual[index]) &&
                         std::isfinite(expected[index]);
                maximum_absolute_error = std::max(
                    maximum_absolute_error,
                    std::abs(
                        static_cast<double>(actual[index]) -
                        static_cast<double>(expected[index])));
                if (checkpoint.stage == "MOE_ROUTING_INDICES")
                    exact_indices = exact_indices &&
                                    actual[index] == expected[index];
            }

            float cosine = computeCosineSimilarity(
                actual.data(), expected.data(), actual.size());
            float stage_routing_overlap = 1.0f;
            bool stage_routing_top1_match = true;
            float kl = 0.0f;
            bool passed = finite;
            if (checkpoint.stage == "MOE_ROUTING_INDICES")
            {
                if (context.top_k <= 0 ||
                    actual.size() % static_cast<size_t>(context.top_k) != 0u)
                {
                    passed = false;
                }
                else
                {
                    const size_t rows =
                        actual.size() / static_cast<size_t>(context.top_k);
                    double overlap_sum = 0.0;
                    size_t top1_matches = 0u;
                    for (size_t row = 0; row < rows; ++row)
                    {
                        std::set<int> actual_experts;
                        std::set<int> expected_experts;
                        for (int slot = 0; slot < context.top_k; ++slot)
                        {
                            const size_t offset =
                                row * static_cast<size_t>(context.top_k) +
                                static_cast<size_t>(slot);
                            actual_experts.insert(
                                static_cast<int>(actual[offset]));
                            expected_experts.insert(
                                static_cast<int>(expected[offset]));
                        }
                        size_t intersection = 0u;
                        for (const int expert : actual_experts)
                        {
                            if (expected_experts.contains(expert))
                                ++intersection;
                        }
                        overlap_sum +=
                            static_cast<double>(intersection) /
                            static_cast<double>(context.top_k);
                        top1_matches +=
                            actual[row * static_cast<size_t>(context.top_k)] ==
                            expected[row * static_cast<size_t>(context.top_k)];
                    }
                    stage_routing_overlap = static_cast<float>(
                        overlap_sum / static_cast<double>(rows));
                    stage_routing_top1_match = top1_matches == rows;
                    cosine = stage_routing_overlap;
                    maximum_absolute_error = 1.0 - stage_routing_overlap;
                    const float minimum_overlap =
                        1.0f - 1.0f / static_cast<float>(context.top_k);
                    routing_boundary =
                        actual_router_it != context.checkpoints.end() &&
                                reference_router_it != references.end()
                            ? compareMoERoutingBoundarySelections(
                                  expected.data(),
                                  actual.data(),
                                  actual.size(),
                                  reference_router_it->second.data(),
                                  reference_router_it->second.size(),
                                  actual_router_it->actual.data(),
                                  actual_router_it->actual.size(),
                                  static_cast<size_t>(context.top_k))
                            : MoERoutingBoundaryResult{};
                    passed = finite &&
                             stage_routing_top1_match &&
                             stage_routing_overlap >= minimum_overlap &&
                             routing_boundary.evaluated &&
                             routing_boundary.equivalent;
                }
                routing_indices_exact = exact_indices;
                routing_top1_match =
                    routing_top1_match && stage_routing_top1_match;
                routing_overlap = std::min(
                    routing_overlap, stage_routing_overlap);
            }
            else if (checkpoint.stage == "MOE_ROUTING_WEIGHTS")
            {
                if (actual_indices_it == context.checkpoints.end() ||
                    reference_indices_it == references.end())
                {
                    passed = false;
                }
                else
                {
                    const auto [sparse_cosine, sparse_max_error] =
                        compareDeferredRoutingWeights(
                            actual,
                            expected,
                            actual_indices_it->actual,
                            reference_indices_it->second,
                            context.top_k,
                            context.num_experts);
                    cosine = sparse_cosine;
                    maximum_absolute_error = sparse_max_error;
                    stage_routing_overlap = sparse_cosine;
                    passed = finite &&
                             sparse_cosine >= context.cosine_threshold;
                }
                routing_weights_equivalent = passed;
            }
            else
            {
                const float threshold =
                    checkpoint.stage == "MOE_EXPERT_OUTPUT" &&
                            !routing_indices_exact
                        ? context.cosine_threshold
                        : context.decode_cosine_threshold;
                passed = finite && cosine >= threshold;
                if (checkpoint.stage == "MOE_ROUTER_OUTPUT")
                {
                    router_symmetric_kl =
                        symmetricProbabilityKLDivergence(
                            expected.data(),
                            actual.data(),
                            actual.size(),
                            static_cast<size_t>(context.num_experts));
                    kl = static_cast<float>(router_symmetric_kl);
                    passed = passed &&
                             router_symmetric_kl <= context.kl_threshold;
                }
                if (checkpoint.stage == "MOE_EXPERT_OUTPUT")
                    routed_expert_output_equivalent = passed;
            }

            if (checkpoint.stage == "LM_HEAD")
            {
                if (context.vocab_size <= 0 ||
                    actual.size() %
                            static_cast<size_t>(context.vocab_size) !=
                        0u)
                {
                    passed = false;
                }
                else
                {
                    kl = computeKLDivergence(
                        actual.data(),
                        expected.data(),
                        actual.size(),
                        static_cast<size_t>(context.vocab_size));
                    lm_head_topk = evaluateReferenceTopKContainment(
                        actual.data(),
                        expected.data(),
                        actual.size(),
                        static_cast<size_t>(context.vocab_size),
                        context.pytorch_top1_in_topk);
                    passed = passed && kl < context.kl_threshold &&
                             lm_head_topk.passed;
                }
                lm_head_passed = passed;
            }

            context_finite = context_finite && finite;
            if (parityStageContributesToLayerCosine(
                    checkpoint.stage, routing_indices_exact))
            {
                numerical_cosine_sum += cosine;
                ++numerical_stage_count;
            }

            csv << context.call << ',' << context.reference_step << ','
                << context.reference_depth << ','
                << checkpoint.production_key << ','
                << reference_prefix << checkpoint.stage << ','
                << actual.size() << ',' << cosine << ','
                << maximum_absolute_error << ',' << kl << ','
                << (exact_indices ? 1 : 0) << ','
                << stage_routing_overlap << ','
                << (stage_routing_top1_match ? 1 : 0) << ','
                << (finite ? 1 : 0) << ',' << (passed ? 1 : 0) << '\n';
            ++compared_stages;
        }
        csv.flush();
        if (!csv.good())
        {
            if (error)
                *error = "failed while appending " + context.snapshot_csv_path.string();
            return false;
        }

        const MoERoutedContributionResult routed_contribution =
            adjudicateMoERoutedContribution({
                .routing_boundary = routing_boundary,
                .sparse_routing_weights_equivalent =
                    routing_weights_equivalent,
                .routed_expert_output_equivalent =
                    routed_expert_output_equivalent,
                .router_symmetric_kl = router_symmetric_kl,
                .maximum_router_symmetric_kl = context.kl_threshold,
            });
        const bool routed_contribution_equivalent =
            routed_contribution.evaluated &&
            routed_contribution.equivalent;
        const double numerical_cosine =
            numerical_stage_count > 0u
                ? numerical_cosine_sum /
                      static_cast<double>(numerical_stage_count)
                : 0.0;
        const bool passed = compared_stages > 0u && context_finite &&
                            routing_top1_match &&
                            routed_contribution_equivalent &&
                            productionRecursiveMTPAggregatePasses(
                                numerical_cosine,
                                context.decode_cosine_threshold,
                                context.mtp_recursive_aggregate_cosine_floor
                                    .value_or(
                                        kMinimumProductionRecursiveMTPAggregateCosine)) &&
                            lm_head_passed;
        if (!passed && error)
        {
            std::ostringstream detail;
            detail << context.test_name
                   << " deferred recursive MTP parity failed"
                   << " step=" << context.reference_step
                   << " depth=" << context.reference_depth
                   << " compared_stages=" << compared_stages
                   << " finite=" << context_finite
                   << " routing_top1=" << routing_top1_match
                   << " routing_overlap=" << routing_overlap
                   << " routed_value=" << routed_contribution_equivalent
                   << " router_symmetric_kl=" << router_symmetric_kl
                   << " routed_contribution_authority="
                   << static_cast<int>(routed_contribution.authority)
                   << " numerical_cosine=" << numerical_cosine
                   << " required_numerical_cosine="
                   << productionRecursiveMTPAggregateRequiredCosine(
                          context.decode_cosine_threshold,
                          context.mtp_recursive_aggregate_cosine_floor
                              .value_or(
                                  kMinimumProductionRecursiveMTPAggregateCosine))
                   << " lm_head=" << lm_head_passed
                   << " lm_head_top_k=" << lm_head_topk.configured_top_k
                   << " reference_top1_in_production="
                   << lm_head_topk.reference_top1_in_production
                   << " production_top1_in_reference="
                   << lm_head_topk.production_top1_in_reference
                   << " csv=" << context.snapshot_csv_path;
            *error = detail.str();
        }
        return passed;
    }

    /**
     * @brief Generate all missing branches with one loaded HF model and compare.
     * @param contexts Immutable queue removed from the live campaign.
     * @param error Receives the generator output or first numerical failure.
     */
    bool resolveDeferredMTPBranchCampaign(
        const std::vector<DeferredMTPBranchContext> &contexts,
        std::string *error)
    {
        if (contexts.empty())
            return true;

        const auto &identity = contexts.front();
        int maximum_depth = 1;
        int decode_steps = identity.decode_steps;
        std::set<std::pair<int, std::vector<int32_t>>> unique_branches;
        for (const auto &context : contexts)
        {
            if (context.model_path != identity.model_path ||
                context.prompt != identity.prompt ||
                context.snapshot_dir != identity.snapshot_dir)
            {
                if (error)
                    *error = "deferred MTP campaign mixed reference identities";
                return false;
            }
            maximum_depth = std::max(
                maximum_depth,
                static_cast<int>(context.condition_tokens.size()) + 1);
            decode_steps = std::max(decode_steps, context.decode_steps);
            unique_branches.emplace(
                context.reference_step, context.condition_tokens);
        }

        const std::filesystem::path request_path =
            contexts.front().snapshot_csv_path.parent_path() /
            "mtp_hf_branch_campaign_requests.json";
        std::ofstream request(request_path, std::ios::trunc);
        if (!request.is_open())
        {
            if (error)
                *error = "could not write " + request_path.string();
            return false;
        }
        request << "[\n";
        size_t branch_index = 0u;
        for (const auto &[step, tokens] : unique_branches)
        {
            if (branch_index++ != 0u)
                request << ",\n";
            request << "  {\"" << step << "\": [";
            for (size_t token_index = 0; token_index < tokens.size(); ++token_index)
            {
                if (token_index != 0u)
                    request << ", ";
                request << tokens[token_index];
            }
            request << "]}";
        }
        request << "\n]\n";
        request.flush();
        if (!request.good())
        {
            if (error)
                *error = "failed while writing " + request_path.string();
            return false;
        }

        std::ostringstream script;
        script
            << "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS "
               "OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
            << "if [ -f /workspaces/llaminar/.venv/bin/activate ]; then "
               "source /workspaces/llaminar/.venv/bin/activate; fi; "
            << "python3 python/reference/"
               "generate_qwen35_moe_pipeline_snapshots.py"
            << " --model "
            << quoteDeferredReferenceArgument(identity.model_path)
            << " --prompt "
            << quoteDeferredReferenceArgument(identity.prompt)
            << " --output "
            << quoteDeferredReferenceArgument(identity.snapshot_dir)
            << " --decode-steps " << decode_steps
            << " --mtp-sidecar-snapshots --mtp-max-draft-depth "
            << maximum_depth << " --mtp-branch-overrides "
            << quoteDeferredReferenceArgument(request_path.string());
        const std::string command =
            "bash -c " + quoteDeferredReferenceArgument(script.str()) +
            " 2>&1";

        LOG_INFO(
            "[Qwen3.5 MoE MTP Parity] Resolving "
            << unique_branches.size()
            << " deferred branch trajectories with one HF model load after "
               "production residency retirement");
        FILE *pipe = popen(command.c_str(), "r");
        if (!pipe)
        {
            if (error)
                *error = "could not start deferred HF branch generator";
            return false;
        }
        std::string output;
        std::array<char, 512> buffer{};
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
            output += buffer.data();
        const int exit_code = pclose(pipe);
        if (exit_code != 0)
        {
            if (error)
                *error = "deferred HF branch generation failed:\n" + output;
            return false;
        }

        for (const auto &context : contexts)
        {
            if (!compareDeferredMTPBranchContext(context, error))
                return false;
        }
        return true;
    }

}
