/**
 * @file Test__MoEExpertOverlay_MultiAcceleratorTiers.cpp
 * @brief Bridge Phase 5A model-light audit coverage for multi-accelerator MoE expert-overlay tiers.
 */

#include <gtest/gtest.h>

#include "backends/ComputeBackend.h"
#include "backends/HardwareInventory.h"
#include "collective/ILocalTPContext.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoEExpertParallelReduceStage.h"
#include "execution/moe/MoEExpertOverlayDenseReduce.h"
#include "execution/moe/MoEExpertOverlayLocalTPExecutor.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "execution/moe/MoEExpertParallelPlanner.h"
#include "execution/moe/MoEExpertTokenRowTransfer.h"
#include "kernels/KernelFactory.h"
#include "mocks/MockComputeStage.h"
#include "tensors/Tensors.h"

#include <algorithm>
#include <exception>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        constexpr int kSeqLen = 4;
        constexpr int kDModel = 8;
        constexpr int kIntermediate = 4;
        constexpr int kNumExperts = 6;
        constexpr int kTopK = 2;

        std::shared_ptr<FP32Tensor> fp32(std::vector<size_t> shape)
        {
            return std::make_shared<FP32Tensor>(std::move(shape));
        }

        void fillHidden(FP32Tensor *tensor)
        {
            float *data = tensor->mutable_data();
            for (int row = 0; row < kSeqLen; ++row)
            {
                for (int col = 0; col < kDModel; ++col)
                {
                    data[static_cast<size_t>(row) * kDModel + col] =
                        0.04f * static_cast<float>(row + 1) +
                        0.015f * static_cast<float>(col + 1);
                }
            }
        }

        void fillRouting(FP32Tensor *indices, FP32Tensor *weights)
        {
            const float route_indices[] = {
                0.0f,
                2.0f,
                1.0f,
                3.0f,
                4.0f,
                5.0f,
                2.0f,
                5.0f,
            };
            const float route_weights[] = {
                0.70f,
                0.30f,
                0.55f,
                0.45f,
                0.60f,
                0.40f,
                0.25f,
                0.75f,
            };

            std::copy(std::begin(route_indices), std::end(route_indices), indices->mutable_data());
            std::copy(std::begin(route_weights), std::end(route_weights), weights->mutable_data());
        }

        void fillExpertTensor(FP32Tensor *tensor, float scale)
        {
            const auto &shape = tensor->shape();
            ASSERT_EQ(shape.size(), 3u);
            const size_t cols = shape[0];
            const size_t rows = shape[1];
            const size_t experts = shape[2];
            float *data = tensor->mutable_data();

            for (size_t expert = 0; expert < experts; ++expert)
            {
                for (size_t row = 0; row < rows; ++row)
                {
                    for (size_t col = 0; col < cols; ++col)
                    {
                        const size_t offset = expert * rows * cols + row * cols + col;
                        data[offset] = scale * static_cast<float>(expert + 1) +
                                       0.0025f * static_cast<float>(row + 1) +
                                       0.0009f * static_cast<float>(col + 1);
                    }
                }
            }
        }

        struct ExpertWeights
        {
            std::shared_ptr<FP32Tensor> gate;
            std::shared_ptr<FP32Tensor> up;
            std::shared_ptr<FP32Tensor> down;
        };

        using KernelFactory = llaminar::v2::kernels::KernelFactory;

        struct PreparedLocalTPFixture
        {
            std::vector<std::shared_ptr<TensorBase>> views;
            std::vector<std::shared_ptr<KernelFactory::PreparedGemmHandle>> handles;
            std::vector<MoEExpertOverlayLocalTPPreparedParticipant> participants;
            std::vector<FP32Tensor *> partial_views;
        };

        ExpertWeights makeWeights()
        {
            ExpertWeights weights;
            weights.gate = fp32({kDModel, kIntermediate, kNumExperts});
            weights.up = fp32({kDModel, kIntermediate, kNumExperts});
            weights.down = fp32({kIntermediate, kDModel, kNumExperts});
            fillExpertTensor(weights.gate.get(), 0.009f);
            fillExpertTensor(weights.up.get(), 0.011f);
            fillExpertTensor(weights.down.get(), 0.007f);
            return weights;
        }

        std::shared_ptr<TensorBase> makeExpertView(
            const std::shared_ptr<FP32Tensor> &tensor,
            int expert,
            int rows,
            int cols)
        {
            return tensor->create_view(
                {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                static_cast<size_t>(expert) * static_cast<size_t>(rows) * static_cast<size_t>(cols));
        }

        ITensorGemm *prepareEngine(
            const std::shared_ptr<TensorBase> &view,
            DeviceId device,
            PreparedLocalTPFixture *fixture)
        {
            fixture->views.push_back(view);
            if (device.is_gpu() && !view->ensureOnDevice(device))
                throw std::runtime_error("Failed to upload expert view to " + device.to_string());
            auto handle = KernelFactory::prepareGemmHandleLocal(view.get(), device);
            ITensorGemm *engine = KernelFactory::getOrCreateGemmEngine(handle.get());
            fixture->handles.push_back(std::move(handle));
            return engine;
        }

        PreparedLocalTPFixture prepareLocalTPParticipants(
            const ExpertWeights &weights,
            const MoEOverlayRuntimeDomain &domain,
            const std::vector<bool> &expert_mask)
        {
            PreparedLocalTPFixture fixture;
            fixture.participants.reserve(domain.participants.size());

            for (size_t participant_index = 0; participant_index < domain.participants.size(); ++participant_index)
            {
                const DeviceId device = domain.participants[participant_index].local_device;
                MoEExpertOverlayLocalTPPreparedParticipant participant;
                participant.participant_index = static_cast<int>(participant_index);
                participant.device = device;
                participant.gate_gemm.assign(kNumExperts, nullptr);
                participant.up_gemm.assign(kNumExperts, nullptr);
                participant.down_gemm.assign(kNumExperts, nullptr);
                participant.input_scratch = fp32({kSeqLen, kDModel});
                participant.batch_scratch = fp32({kSeqLen, kDModel});
                participant.gate_scratch = fp32({kSeqLen, kIntermediate});
                participant.up_scratch = fp32({kSeqLen, kIntermediate});
                participant.partial_scratch = fp32({kSeqLen, kDModel});
                if (device.is_gpu())
                {
                    participant.input_scratch->allocateOnDevice(device);
                    participant.batch_scratch->allocateOnDevice(device);
                    participant.gate_scratch->allocateOnDevice(device);
                    participant.up_scratch->allocateOnDevice(device);
                    participant.partial_scratch->allocateOnDevice(device);
                }
                participant.token_indices.reserve(static_cast<size_t>(kSeqLen) * kTopK);
                participant.token_weights.reserve(static_cast<size_t>(kSeqLen) * kTopK);

                for (int expert = 0; expert < kNumExperts; ++expert)
                {
                    if (!expert_mask[static_cast<size_t>(expert)])
                        continue;
                    participant.gate_gemm[static_cast<size_t>(expert)] = prepareEngine(
                        makeExpertView(weights.gate, expert, kIntermediate, kDModel), device, &fixture);
                    participant.up_gemm[static_cast<size_t>(expert)] = prepareEngine(
                        makeExpertView(weights.up, expert, kIntermediate, kDModel), device, &fixture);
                    participant.down_gemm[static_cast<size_t>(expert)] = prepareEngine(
                        makeExpertView(weights.down, expert, kDModel, kIntermediate), device, &fixture);
                }

                fixture.participants.push_back(std::move(participant));
            }

            fixture.partial_views.reserve(fixture.participants.size());
            for (auto &participant : fixture.participants)
                fixture.partial_views.push_back(participant.partial_scratch.get());

            return fixture;
        }

        ExpertComputeDomain cudaFastDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "cuda_fast";
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::AUTO;
            domain.participants = {GlobalDeviceAddress::cuda(0, 0)};
            domain.world_ranks = {0};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain rocmHotDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "rocm_hot";
            domain.kind = ExpertDomainKind::LocalTP;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0, 0),
                                   GlobalDeviceAddress::rocm(1, 0)};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
            return domain;
        }

        ExpertComputeDomain cpuColdDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "cpu_cold";
            domain.kind = ExpertDomainKind::NodeLocalTP;
            domain.backend = CollectiveBackendType::UPI;
            domain.participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
            domain.world_ranks = {1, 2};
            domain.owner_rank = 1;
            domain.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
            return domain;
        }

        ExpertRoutedTier tier(
            const std::string &name,
            const std::string &domain,
            int priority,
            int max_experts_per_layer,
            bool fallback = false)
        {
            ExpertRoutedTier result;
            result.name = name;
            result.domain = domain;
            result.priority = priority;
            result.max_experts_per_layer = max_experts_per_layer;
            result.fallback = fallback;
            return result;
        }

        MoEExpertModelMetadata metadata()
        {
            MoEExpertModelMetadata model;
            model.num_layers = 1;
            model.num_experts = kNumExperts;
            model.d_model = kDModel;
            model.routed_intermediate_size = kIntermediate;
            model.has_shared_expert = false;
            model.routed_quant_type = "F32";
            return model;
        }

        MoEExpertParallelPlan makePhase8Plan()
        {
            MoEExpertParallelPlan plan;
            plan.enabled = true;
            plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan.continuation_domain = "cuda_fast";
            plan.shared_expert_domain = "cuda_fast";
            plan.residency_policy = ExpertResidencyPolicy::StaticById;
            plan.domains = {cudaFastDomain(), rocmHotDomain(), cpuColdDomain()};
            plan.routed_tiers = {
                tier("hottest", "cuda_fast", 0, 2),
                tier("hot", "rocm_hot", 1, 2),
                tier("cold", "cpu_cold", 2, 0, true),
            };

            return MoEExpertParallelPlanner::plan(plan, metadata()).planned_plan;
        }

        const ExpertComputeDomain *findDomain(const MoEExpertParallelPlan &plan, const std::string &name)
        {
            auto it = std::find_if(plan.domains.begin(), plan.domains.end(), [&](const auto &domain)
                                   { return domain.name == name; });
            return it == plan.domains.end() ? nullptr : &(*it);
        }

        std::vector<bool> maskForTier(const ExpertLayerPlacement &placement, int tier_index)
        {
            std::vector<bool> mask(static_cast<size_t>(kNumExperts), false);
            for (int expert = 0; expert < kNumExperts; ++expert)
                mask[static_cast<size_t>(expert)] =
                    placement.routed_expert_tier[static_cast<size_t>(expert)] == tier_index;
            return mask;
        }

        bool runExpertCompute(
            IDeviceContext *ctx,
            TensorBase *input,
            TensorBase *routing_indices,
            TensorBase *routing_weights,
            const ExpertWeights &weights,
            const std::vector<bool> &expert_mask,
            TensorBase *output)
        {
            MoEExpertComputeStage::Params params;
            params.device_id = DeviceId::cpu();
            params.input = input;
            params.seq_len = kSeqLen;
            params.d_model = kDModel;
            params.num_experts = kNumExperts;
            params.top_k = kTopK;
            params.gate_exps = weights.gate.get();
            params.up_exps = weights.up.get();
            params.down_exps = weights.down.get();
            params.expert_intermediate = kIntermediate;
            params.layer_idx = 0;
            params.expert_mask = expert_mask;
            params.routing_indices = routing_indices;
            params.routing_weights = routing_weights;
            params.output = output;

            if (!MoEExpertComputeStage::extractExpertViews(params) ||
                !MoEExpertComputeStage::prepareExpertGemmEngines(params))
            {
                return false;
            }

            MoEExpertComputeStage stage(std::move(params));
            return stage.execute(ctx);
        }

        bool runDispatch(
            IDeviceContext *ctx,
            TensorBase *hidden,
            TensorBase *routing_indices,
            TensorBase *routing_weights,
            const MoEExpertParallelPlan &plan,
            MoEExpertDispatchOutput *output)
        {
            if (plan.placements.empty())
                return false;

            MoEExpertDispatchStage::Params params;
            params.device_id = DeviceId::cpu();
            params.routing_indices = routing_indices;
            params.routing_weights = routing_weights;
            params.hidden = hidden;
            params.seq_len = kSeqLen;
            params.top_k = kTopK;
            params.d_model = kDModel;
            params.continuation_domain = plan.continuation_domain;
            params.placement = plan.placements.front();
            params.routed_tiers = plan.routed_tiers;
            params.output = output;

            MoEExpertDispatchStage stage(std::move(params));
            return stage.execute(ctx);
        }

        void expectSparseVolume(
            const MoEExpertTierDispatch &tier,
            size_t selected_rows,
            MoEExpertTransferMode mode)
        {
            const size_t hidden_row_bytes = static_cast<size_t>(kDModel) * sizeof(float);
            const size_t routing_row_bytes = static_cast<size_t>(kTopK) * 2u * sizeof(float);
            ASSERT_TRUE(tier.transfer_required);
            EXPECT_EQ(tier.transfer_mode, mode);
            EXPECT_EQ(tier.transfer_volume.selected_rows, selected_rows);
            EXPECT_EQ(tier.transfer_volume.outbound_bytes,
                      selected_rows * (hidden_row_bytes + routing_row_bytes));
            EXPECT_EQ(tier.transfer_volume.return_bytes,
                      selected_rows * hidden_row_bytes);
            EXPECT_EQ(tier.transfer_volume.dense_outbound_bytes,
                      static_cast<size_t>(kSeqLen) * (hidden_row_bytes + routing_row_bytes));
            EXPECT_EQ(tier.transfer_volume.dense_return_bytes,
                      static_cast<size_t>(kSeqLen) * hidden_row_bytes);
        }

        void expectTensorNear(const TensorBase *actual, const TensorBase *expected, float tolerance = 1e-4f)
        {
            ASSERT_EQ(actual->numel(), expected->numel());
            const float *actual_data = actual->data();
            const float *expected_data = expected->data();
            for (size_t i = 0; i < actual->numel(); ++i)
                EXPECT_NEAR(actual_data[i], expected_data[i], tolerance) << "index=" << i;
        }

        std::string joinErrors(const std::vector<std::string> &errors)
        {
            std::ostringstream out;
            for (const auto &error : errors)
                out << "\n - " << error;
            return out.str();
        }

        void expectPhase8Topology(const MoEExpertParallelPlan &plan)
        {
            ASSERT_TRUE(plan.isTieredOverlay());
            EXPECT_EQ(plan.continuation_domain, "cuda_fast");
            EXPECT_EQ(plan.shared_expert_domain, "cuda_fast");
            ASSERT_EQ(plan.domains.size(), 3u);
            ASSERT_EQ(plan.routed_tiers.size(), 3u);
            ASSERT_EQ(plan.placements.size(), 1u);
            EXPECT_EQ(plan.placements.front().routed_expert_tier,
                      (std::vector<int>{0, 0, 1, 1, 2, 2}));

            const auto *cuda = findDomain(plan, "cuda_fast");
            ASSERT_NE(cuda, nullptr);
            EXPECT_EQ(cuda->kind, ExpertDomainKind::SingleDevice);
            EXPECT_EQ(cuda->backend, CollectiveBackendType::AUTO);
            EXPECT_EQ(cuda->compute_kind, ExpertDomainComputeKind::ReplicatedExperts);
            ASSERT_EQ(cuda->participants.size(), 1u);
            EXPECT_EQ(cuda->participants[0].device_type, DeviceType::CUDA);
            EXPECT_EQ(cuda->world_ranks, (std::vector<int>{0}));

            const auto *rocm = findDomain(plan, "rocm_hot");
            ASSERT_NE(rocm, nullptr);
            EXPECT_EQ(rocm->kind, ExpertDomainKind::LocalTP);
            EXPECT_EQ(rocm->backend, CollectiveBackendType::RCCL);
            EXPECT_EQ(rocm->compute_kind, ExpertDomainComputeKind::TensorParallelExperts);
            ASSERT_EQ(rocm->participants.size(), 2u);
            EXPECT_EQ(rocm->participants[0].device_type, DeviceType::ROCm);
            EXPECT_EQ(rocm->participants[1].device_type, DeviceType::ROCm);

            const auto *cpu = findDomain(plan, "cpu_cold");
            ASSERT_NE(cpu, nullptr);
            EXPECT_EQ(cpu->kind, ExpertDomainKind::NodeLocalTP);
            EXPECT_EQ(cpu->backend, CollectiveBackendType::UPI);
            EXPECT_EQ(cpu->compute_kind, ExpertDomainComputeKind::TensorParallelExperts);
            ASSERT_EQ(cpu->participants.size(), 2u);
            EXPECT_EQ(cpu->participants[0].device_type, DeviceType::CPU);
            EXPECT_EQ(cpu->participants[1].device_type, DeviceType::CPU);
            EXPECT_EQ(cpu->world_ranks, (std::vector<int>{1, 2}));
        }

        void expectDispatchMatchesPhase8Plan(
            const MoEExpertDispatchOutput &output,
            const MoEExpertParallelPlan &plan)
        {
            ASSERT_EQ(output.tiers.size(), plan.routed_tiers.size());
            const auto &placement = plan.placements.front();
            std::vector<size_t> entry_counts(output.tiers.size(), 0);

            for (size_t tier_index = 0; tier_index < output.tiers.size(); ++tier_index)
            {
                EXPECT_EQ(output.tiers[tier_index].tier_index, static_cast<int>(tier_index));
                EXPECT_EQ(output.tiers[tier_index].tier_name, plan.routed_tiers[tier_index].name);
                EXPECT_EQ(output.tiers[tier_index].domain, plan.routed_tiers[tier_index].domain);

                for (const auto &entry : output.tiers[tier_index].entries)
                {
                    ASSERT_GE(entry.expert_id, 0);
                    ASSERT_LT(entry.expert_id, kNumExperts);
                    EXPECT_EQ(placement.routed_expert_tier[static_cast<size_t>(entry.expert_id)],
                              static_cast<int>(tier_index));
                    ++entry_counts[tier_index];
                }
            }

            EXPECT_EQ(entry_counts, (std::vector<size_t>{2u, 3u, 3u}));
            EXPECT_EQ(output.tiers[0].token_rows, (std::vector<int>{0, 1}));
            EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{0, 1, 3}));
            EXPECT_EQ(output.tiers[2].token_rows, (std::vector<int>{2, 3}));
        }

        std::string realAcceleratorSkipReason()
        {
#if !defined(HAVE_ROCM)
            return "Built without ROCm support (-DHAVE_ROCM=OFF)";
#elif !defined(HAVE_RCCL)
            return "Built without RCCL support";
#else
            const auto hardware = HardwareInventory::detect();
            if (hardware.rocm_device_count() < 2)
                return "Hardware availability: Bridge Phase 5C accelerator LocalTP execution requires at least 2 ROCm devices for rocm_hot LocalTP";
            return {};
#endif
        }

        std::string rocmContinuationSkipReason()
        {
#if !defined(HAVE_ROCM)
            return "Built without ROCm support (-DHAVE_ROCM=OFF)";
#else
            const auto hardware = HardwareInventory::detect();
            if (hardware.rocm_device_count() < 1)
                return "Hardware availability: Bridge Phase 7A Layout A continuation residency requires at least 1 ROCm device";
            return {};
#endif
        }

        std::string cudaAndRocmContinuationSkipReason()
        {
#if !defined(HAVE_CUDA)
            return "Built without CUDA support (-DHAVE_CUDA=OFF)";
#elif !defined(HAVE_ROCM)
            return "Built without ROCm support (-DHAVE_ROCM=OFF)";
#else
            const auto hardware = HardwareInventory::detect();
            if (hardware.cuda_device_count() < 1)
                return "Hardware availability: Bridge Phase 7A Layout B continuation residency requires at least 1 CUDA device";
            if (hardware.rocm_device_count() < 1)
                return "Hardware availability: Bridge Phase 7A Layout B cross-domain reduce requires at least 1 ROCm device";
            return {};
#endif
        }

        std::string initializeDeviceManagerForTransferTests()
        {
            try
            {
                DeviceManager::instance().initialize(-1);
            }
            catch (const std::exception &e)
            {
                return std::string("Hardware initialization: Bridge Phase 7A continuation residency test could not initialize DeviceManager: ") + e.what();
            }
            return {};
        }

    } // namespace

    TEST(Test__MoEExpertOverlay_MultiAcceleratorTiers, SyntheticThreeTierExecutionReducesToCudaContinuationReference)
    {
        const auto plan = makePhase8Plan();
        expectPhase8Topology(plan);

        auto ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
        auto weights = makeWeights();

        auto hidden = fp32({kSeqLen, kDModel});
        auto routing_indices = fp32({kSeqLen, kTopK});
        auto routing_weights = fp32({kSeqLen, kTopK});
        fillHidden(hidden.get());
        fillRouting(routing_indices.get(), routing_weights.get());

        auto reference = fp32({kSeqLen, kDModel});
        ASSERT_TRUE(runExpertCompute(ctx.get(),
                                     hidden.get(),
                                     routing_indices.get(),
                                     routing_weights.get(),
                                     weights,
                                     std::vector<bool>{true, true, true, true, true, true},
                                     reference.get()));

        MoEExpertDispatchOutput dispatch;
        ASSERT_TRUE(runDispatch(ctx.get(), hidden.get(), routing_indices.get(), routing_weights.get(), plan, &dispatch));
        expectDispatchMatchesPhase8Plan(dispatch, plan);

        const auto &placement = plan.placements.front();
        auto cuda_partial = fp32({kSeqLen, kDModel});
        auto rocm_partial = fp32({kSeqLen, kDModel});
        auto cpu_partial = fp32({kSeqLen, kDModel});
        ASSERT_TRUE(runExpertCompute(ctx.get(), hidden.get(), routing_indices.get(), routing_weights.get(), weights,
                                     maskForTier(placement, 0), cuda_partial.get()));
        ASSERT_TRUE(runExpertCompute(ctx.get(), hidden.get(), routing_indices.get(), routing_weights.get(), weights,
                                     maskForTier(placement, 1), rocm_partial.get()));
        ASSERT_TRUE(runExpertCompute(ctx.get(), hidden.get(), routing_indices.get(), routing_weights.get(), weights,
                                     maskForTier(placement, 2), cpu_partial.get()));

        auto reduced = fp32({kSeqLen, kDModel});
        MoEExpertOverlayDenseReduceRequest request;
        request.plan = &plan;
        request.partials = {
            {.tier_name = "hottest", .source_domain = "cuda_fast", .tensor = cuda_partial.get()},
            {.tier_name = "hot", .source_domain = "rocm_hot", .tensor = rocm_partial.get()},
            {.tier_name = "cold", .source_domain = "cpu_cold", .tensor = cpu_partial.get()},
        };
        request.output = reduced.get();
        request.rows = kSeqLen;
        request.cols = kDModel;

        const auto request_errors = MoEExpertOverlayDenseReduce::validateRequest(request);
        ASSERT_TRUE(request_errors.empty()) << joinErrors(request_errors);
        ASSERT_TRUE(MoEExpertOverlayDenseReduce::reduceToContinuation(request, ctx.get()));

        expectTensorNear(reduced.get(), reference.get());
    }

    TEST(Test__MoEExpertOverlay_MultiAcceleratorTiers, RocmLocalTPPreparationTargetsBothParticipantsAndRuntimeIsExecutable)
    {
        auto plan = std::make_shared<MoEExpertParallelPlan>(makePhase8Plan());
        auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);
        ASSERT_NE(runtime_plan, nullptr);

        const auto *rocm_domain = runtime_plan->domainForName("rocm_hot");
        ASSERT_NE(rocm_domain, nullptr);
        EXPECT_EQ(rocm_domain->kind, ExpertDomainKind::LocalTP);
        EXPECT_EQ(rocm_domain->compute_kind, ExpertDomainComputeKind::TensorParallelExperts);
        ASSERT_EQ(rocm_domain->participants.size(), 2u);
        EXPECT_EQ(rocm_domain->participants[0].local_device, DeviceId::rocm(0));
        EXPECT_EQ(rocm_domain->participants[1].local_device, DeviceId::rocm(1));
        EXPECT_FALSE(rocm_domain->multi_participant_execution_pending);
        EXPECT_TRUE(rocm_domain->domain_scoped_collective_context_ready);
        EXPECT_TRUE(rocm_domain->pending_reason.empty()) << rocm_domain->pending_reason;

        const auto prep = MoEExpertOverlayPreparationPlan::build(*runtime_plan, 256);
        const auto devices = prep.acceleratorDevices();
        EXPECT_EQ(devices, (std::vector<DeviceId>{DeviceId::cuda(0), DeviceId::rocm(0), DeviceId::rocm(1)}));

        using Role = ExpertGemmRegistry::WeightRole;
        EXPECT_TRUE(prep.shouldPrepare(DeviceId::rocm(0), 0, 2, Role::GATE));
        EXPECT_TRUE(prep.shouldPrepare(DeviceId::rocm(0), 0, 3, Role::DOWN));
        EXPECT_TRUE(prep.shouldPrepare(DeviceId::rocm(1), 0, 2, Role::GATE));
        EXPECT_TRUE(prep.shouldPrepare(DeviceId::rocm(1), 0, 3, Role::DOWN));

        const auto *rocm0 = prep.diagnostics().domainStats("rocm_hot", DeviceId::rocm(0));
        const auto *rocm1 = prep.diagnostics().domainStats("rocm_hot", DeviceId::rocm(1));
        ASSERT_NE(rocm0, nullptr);
        ASSERT_NE(rocm1, nullptr);
        EXPECT_EQ(rocm0->assigned_routed_experts, rocm1->assigned_routed_experts);
        EXPECT_EQ(rocm0->planned_engine_count, rocm1->planned_engine_count);
    }

    TEST(Test__MoEExpertOverlay_MultiAcceleratorTiers, SparsePrefillTransferVolumeScalesWithRoutedRowsAndScatterAddsReturn)
    {
        const auto plan = makePhase8Plan();
        auto ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);

        auto hidden = fp32({kSeqLen, kDModel});
        auto routing_indices = fp32({kSeqLen, kTopK});
        auto routing_weights = fp32({kSeqLen, kTopK});
        fillHidden(hidden.get());
        fillRouting(routing_indices.get(), routing_weights.get());

        MoEExpertDispatchOutput dispatch;
        ASSERT_TRUE(runDispatch(ctx.get(), hidden.get(), routing_indices.get(), routing_weights.get(), plan, &dispatch));
        expectDispatchMatchesPhase8Plan(dispatch, plan);

        ASSERT_EQ(dispatch.tiers.size(), 3u);
        EXPECT_FALSE(dispatch.tiers[0].transfer_required);
        EXPECT_EQ(dispatch.tiers[0].transfer_mode, MoEExpertTransferMode::None);
        expectSparseVolume(dispatch.tiers[1], 3u, MoEExpertTransferMode::SparseTokenRows);
        expectSparseVolume(dispatch.tiers[2], 2u, MoEExpertTransferMode::SparseTokenRows);
        EXPECT_LT(dispatch.estimatedTransferBytes(), dispatch.denseTransferBytes());

        MoEExpertSparseTransferBuffers cold_buffers;
        MoEExpertTransferVolume cold_volume;
        ASSERT_TRUE(MoEExpertTokenRowTransfer::gatherRows(hidden.get(),
                                                          routing_indices.get(),
                                                          routing_weights.get(),
                                                          dispatch.tiers[2].token_rows,
                                                          kSeqLen,
                                                          kTopK,
                                                          kDModel,
                                                          &cold_buffers,
                                                          &cold_volume,
                                                          dispatch.tiers[2].transfer_mode));
        EXPECT_EQ(cold_volume.totalBytes(), dispatch.tiers[2].transfer_volume.totalBytes());
        EXPECT_EQ(cold_buffers.hidden->shape(), (std::vector<size_t>{2u, static_cast<size_t>(kDModel)}));
        EXPECT_EQ(cold_buffers.routing_indices->shape(), (std::vector<size_t>{2u, static_cast<size_t>(kTopK)}));

        float *compact_output = cold_buffers.output->mutable_data();
        for (size_t sparse_row = 0; sparse_row < dispatch.tiers[2].token_rows.size(); ++sparse_row)
        {
            for (int col = 0; col < kDModel; ++col)
            {
                compact_output[sparse_row * static_cast<size_t>(kDModel) + static_cast<size_t>(col)] =
                    10.0f * static_cast<float>(sparse_row + 1) + static_cast<float>(col);
            }
        }

        auto cuda_partial = fp32({kSeqLen, kDModel});
        auto rocm_partial = fp32({kSeqLen, kDModel});
        auto cpu_partial = fp32({kSeqLen, kDModel});
        std::fill_n(cuda_partial->mutable_data(), cuda_partial->numel(), 1.0f);
        std::fill_n(rocm_partial->mutable_data(), rocm_partial->numel(), 2.0f);
        std::fill_n(cpu_partial->mutable_data(), cpu_partial->numel(), 0.0f);

        ASSERT_TRUE(MoEExpertTokenRowTransfer::scatterAddRows(cold_buffers.output.get(),
                                                              dispatch.tiers[2].token_rows,
                                                              cpu_partial.get(),
                                                              kSeqLen,
                                                              kDModel));
        const float *cpu_data = cpu_partial->data();
        for (int row = 0; row < kSeqLen; ++row)
        {
            for (int col = 0; col < kDModel; ++col)
            {
                const size_t offset = static_cast<size_t>(row) * kDModel + static_cast<size_t>(col);
                if (row == 2)
                    EXPECT_FLOAT_EQ(cpu_data[offset], 10.0f + static_cast<float>(col));
                else if (row == 3)
                    EXPECT_FLOAT_EQ(cpu_data[offset], 20.0f + static_cast<float>(col));
                else
                    EXPECT_FLOAT_EQ(cpu_data[offset], 0.0f);
            }
        }

        auto reduced = fp32({kSeqLen, kDModel});
        MoEExpertOverlayDenseReduceRequest request;
        request.plan = &plan;
        request.partials = {
            {.tier_name = "hottest", .source_domain = "cuda_fast", .tensor = cuda_partial.get()},
            {.tier_name = "hot", .source_domain = "rocm_hot", .tensor = rocm_partial.get()},
            {.tier_name = "cold", .source_domain = "cpu_cold", .tensor = cpu_partial.get()},
        };
        request.output = reduced.get();
        request.rows = kSeqLen;
        request.cols = kDModel;
        ASSERT_TRUE(MoEExpertOverlayDenseReduce::reduceToContinuation(request, ctx.get()));

        const float *reduced_data = reduced->data();
        for (int row = 0; row < kSeqLen; ++row)
        {
            for (int col = 0; col < kDModel; ++col)
            {
                const size_t offset = static_cast<size_t>(row) * kDModel + static_cast<size_t>(col);
                const float expected_cpu = row == 2   ? 10.0f + static_cast<float>(col)
                                           : row == 3 ? 20.0f + static_cast<float>(col)
                                                      : 0.0f;
                EXPECT_FLOAT_EQ(reduced_data[offset], 3.0f + expected_cpu);
            }
        }
    }

    TEST(Test__MoEExpertOverlay_MultiAcceleratorTiers, LayoutAReducesCpuFallbackPartialBackToRocmContinuation)
    {
        const std::string skip_reason = rocmContinuationSkipReason();
        if (!skip_reason.empty())
            GTEST_SKIP() << skip_reason;
        const std::string init_skip_reason = initializeDeviceManagerForTransferTests();
        if (!init_skip_reason.empty())
            GTEST_SKIP() << init_skip_reason;

        auto ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
        auto shared = fp32({kSeqLen, kDModel});
        auto hot_rocm = fp32({kSeqLen, kDModel});
        auto cold_cpu = fp32({kSeqLen, kDModel});
        auto output = fp32({kSeqLen, kDModel});
        std::fill_n(shared->mutable_data(), shared->numel(), 1.0f);
        std::fill_n(hot_rocm->mutable_data(), hot_rocm->numel(), 2.0f);
        std::fill_n(cold_cpu->mutable_data(), cold_cpu->numel(), 3.0f);
        ASSERT_TRUE(shared->ensureOnDevice(DeviceId::rocm(0)));
        shared->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, DeviceId::rocm(0));
        ASSERT_TRUE(hot_rocm->ensureOnDevice(DeviceId::rocm(0)));
        hot_rocm->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, DeviceId::rocm(0));

        MoEExpertParallelReduceDiagnostics diagnostics;
        MoEExpertParallelReduceStage::Params params;
        params.device_id = DeviceId::rocm(0);
        params.partials = {shared.get(), hot_rocm.get(), cold_cpu.get()};
        params.partial_infos = {
            {.name = "shared_expert", .source_domain = "rocm_hot", .source_device = DeviceId::rocm(0)},
            {.name = "hot", .source_domain = "rocm_hot", .source_device = DeviceId::rocm(0)},
            {.name = "cold", .source_domain = "cpu_cold", .source_device = DeviceId::cpu()},
        };
        params.output = output.get();
        params.rows = kSeqLen;
        params.cols = kDModel;
        params.mode = MoEExpertParallelReduceMode::ContinuationDeviceOptimized;
        params.continuation_domain = "rocm_hot";
        params.continuation_device = DeviceId::rocm(0);
        params.diagnostics = &diagnostics;

        MoEExpertParallelReduceStage stage(std::move(params));
        ASSERT_TRUE(stage.execute(ctx.get()));

        EXPECT_TRUE(output->is_on_device(DeviceId::rocm(0)));
        EXPECT_EQ(diagnostics.mode, MoEExpertParallelReduceMode::ContinuationDeviceOptimized);
        EXPECT_FALSE(diagnostics.host_staged);
        EXPECT_EQ(diagnostics.continuation_domain, "rocm_hot");
        EXPECT_EQ(diagnostics.continuation_device, DeviceId::rocm(0));
        EXPECT_TRUE(diagnostics.output_resident_on_continuation);
        EXPECT_EQ(diagnostics.partial_count, 3u);
        EXPECT_EQ(diagnostics.host_staged_read_bytes, 0u);
        EXPECT_EQ(diagnostics.device_to_host_bytes, 0u);
        EXPECT_EQ(diagnostics.host_to_device_bytes, static_cast<size_t>(kSeqLen) * kDModel * sizeof(float));
        EXPECT_EQ(diagnostics.total_transfer_bytes, static_cast<size_t>(kSeqLen) * kDModel * sizeof(float));
        ASSERT_EQ(diagnostics.partials.size(), 3u);
        EXPECT_TRUE(diagnostics.partials[0].source_is_continuation);
        EXPECT_TRUE(diagnostics.partials[1].source_is_continuation);
        EXPECT_FALSE(diagnostics.partials[2].source_is_continuation);
        EXPECT_EQ(diagnostics.partials[0].accumulation_path,
                  MoEExpertParallelReducePartialAccumulationPath::ContinuationDeviceAccumulated);
        EXPECT_EQ(diagnostics.partials[1].accumulation_path,
                  MoEExpertParallelReducePartialAccumulationPath::ContinuationDeviceAccumulated);
        EXPECT_EQ(diagnostics.partials[2].accumulation_path,
                  MoEExpertParallelReducePartialAccumulationPath::HostStagedThenDeviceAccumulated);

        const float *data = output->data();
        for (size_t i = 0; i < output->numel(); ++i)
            EXPECT_FLOAT_EQ(data[i], 6.0f) << "index=" << i;
    }

    TEST(Test__MoEExpertOverlay_MultiAcceleratorTiers, LayoutBReducesRocmAndCpuPartialsBackToCudaContinuation)
    {
        const std::string skip_reason = cudaAndRocmContinuationSkipReason();
        if (!skip_reason.empty())
            GTEST_SKIP() << skip_reason;
        const std::string init_skip_reason = initializeDeviceManagerForTransferTests();
        if (!init_skip_reason.empty())
            GTEST_SKIP() << init_skip_reason;

        auto ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
        auto shared = fp32({kSeqLen, kDModel});
        auto rocm_hot = fp32({kSeqLen, kDModel});
        auto cpu_cold = fp32({kSeqLen, kDModel});
        auto output = fp32({kSeqLen, kDModel});
        std::fill_n(shared->mutable_data(), shared->numel(), 0.5f);
        std::fill_n(rocm_hot->mutable_data(), rocm_hot->numel(), 4.0f);
        std::fill_n(cpu_cold->mutable_data(), cpu_cold->numel(), -1.5f);
        ASSERT_TRUE(shared->ensureOnDevice(DeviceId::cuda(0)));
        shared->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, DeviceId::cuda(0));
        ASSERT_TRUE(rocm_hot->ensureOnDevice(DeviceId::rocm(0)));
        rocm_hot->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, DeviceId::rocm(0));

        MoEExpertParallelReduceDiagnostics diagnostics;
        MoEExpertParallelReduceStage::Params params;
        params.device_id = DeviceId::cuda(0);
        params.partials = {shared.get(), rocm_hot.get(), cpu_cold.get()};
        params.partial_infos = {
            {.name = "shared_expert", .source_domain = "cuda_fast", .source_device = DeviceId::cuda(0)},
            {.name = "hot", .source_domain = "rocm_hot", .source_device = DeviceId::rocm(0)},
            {.name = "cold", .source_domain = "cpu_cold", .source_device = DeviceId::cpu()},
        };
        params.output = output.get();
        params.rows = kSeqLen;
        params.cols = kDModel;
        params.mode = MoEExpertParallelReduceMode::ContinuationDeviceOptimized;
        params.continuation_domain = "cuda_fast";
        params.continuation_device = DeviceId::cuda(0);
        params.diagnostics = &diagnostics;

        MoEExpertParallelReduceStage stage(std::move(params));
        ASSERT_TRUE(stage.execute(ctx.get()));

        EXPECT_TRUE(output->is_on_device(DeviceId::cuda(0)));
        EXPECT_EQ(diagnostics.mode, MoEExpertParallelReduceMode::ContinuationDeviceOptimized);
        EXPECT_FALSE(diagnostics.host_staged);
        EXPECT_EQ(diagnostics.continuation_domain, "cuda_fast");
        EXPECT_EQ(diagnostics.continuation_device, DeviceId::cuda(0));
        EXPECT_TRUE(diagnostics.output_resident_on_continuation);
        EXPECT_EQ(diagnostics.partial_count, 3u);
        const size_t live_bytes = static_cast<size_t>(kSeqLen) * kDModel * sizeof(float);
        EXPECT_EQ(diagnostics.host_staged_read_bytes, 0u);
        EXPECT_EQ(diagnostics.device_to_host_bytes, live_bytes);
        EXPECT_EQ(diagnostics.host_to_device_bytes, 2u * live_bytes);
        EXPECT_EQ(diagnostics.total_transfer_bytes, 3u * live_bytes);
        ASSERT_EQ(diagnostics.partials.size(), 3u);
        EXPECT_TRUE(diagnostics.partials[0].source_is_continuation);
        EXPECT_FALSE(diagnostics.partials[1].source_is_continuation);
        EXPECT_FALSE(diagnostics.partials[2].source_is_continuation);
        EXPECT_EQ(diagnostics.partials[0].accumulation_path,
                  MoEExpertParallelReducePartialAccumulationPath::ContinuationDeviceAccumulated);
        EXPECT_EQ(diagnostics.partials[1].accumulation_path,
                  MoEExpertParallelReducePartialAccumulationPath::HostStagedThenDeviceAccumulated);
        EXPECT_EQ(diagnostics.partials[2].accumulation_path,
                  MoEExpertParallelReducePartialAccumulationPath::HostStagedThenDeviceAccumulated);

        const float *data = output->data();
        for (size_t i = 0; i < output->numel(); ++i)
            EXPECT_FLOAT_EQ(data[i], 3.0f) << "index=" << i;
    }

    TEST(Test__MoEExpertOverlay_MultiAcceleratorTiers, DenseTransferCompatibilityModeKeepsFullSequenceVolume)
    {
        const auto plan = makePhase8Plan();
        auto ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);

        auto hidden = fp32({kSeqLen, kDModel});
        auto routing_indices = fp32({kSeqLen, kTopK});
        auto routing_weights = fp32({kSeqLen, kTopK});
        fillHidden(hidden.get());
        fillRouting(routing_indices.get(), routing_weights.get());

        MoEExpertDispatchStage::Params params;
        params.device_id = DeviceId::cpu();
        params.routing_indices = routing_indices.get();
        params.routing_weights = routing_weights.get();
        params.hidden = hidden.get();
        params.seq_len = kSeqLen;
        params.top_k = kTopK;
        params.d_model = kDModel;
        params.continuation_domain = plan.continuation_domain;
        params.transfer_mode = MoEExpertTransferMode::DenseFullSequence;
        params.placement = plan.placements.front();
        params.routed_tiers = plan.routed_tiers;
        MoEExpertDispatchOutput dispatch;
        params.output = &dispatch;

        MoEExpertDispatchStage stage(std::move(params));
        ASSERT_TRUE(stage.execute(ctx.get()));

        ASSERT_EQ(dispatch.tiers.size(), 3u);
        EXPECT_EQ(dispatch.tiers[2].transfer_mode, MoEExpertTransferMode::DenseFullSequence);
        EXPECT_EQ(dispatch.tiers[2].transfer_volume.totalBytes(),
                  dispatch.tiers[2].transfer_volume.denseTotalBytes());
        EXPECT_GT(dispatch.tiers[2].transfer_volume.totalBytes(),
                  dispatch.tiers[2].transfer_volume.sparseTotalBytes());
    }

    TEST(Test__MoEExpertOverlay_MultiAcceleratorTiers, DecodeFallbackTransferUsesOneHiddenRowAndTopKMetadata)
    {
        MoEExpertParallelPlan plan;
        plan.enabled = true;
        plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan.continuation_domain = "cuda_fast";
        plan.shared_expert_domain = "cuda_fast";
        plan.domains = {cudaFastDomain(), cpuColdDomain()};
        plan.routed_tiers = {
            tier("hottest", "cuda_fast", 0, 2),
            tier("cold", "cpu_cold", 1, 0, true),
        };
        plan.placements = {ExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 1}}};

        auto ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
        auto hidden = fp32({1, kDModel});
        auto routing_indices = fp32({1, kTopK});
        auto routing_weights = fp32({1, kTopK});
        std::fill_n(hidden->mutable_data(), hidden->numel(), 1.0f);
        routing_indices->mutable_data()[0] = 0.0f;
        routing_indices->mutable_data()[1] = 1.0f;
        routing_weights->mutable_data()[0] = 0.25f;
        routing_weights->mutable_data()[1] = 0.75f;

        MoEExpertDispatchStage::Params params;
        params.device_id = DeviceId::cpu();
        params.routing_indices = routing_indices.get();
        params.routing_weights = routing_weights.get();
        params.hidden = hidden.get();
        params.seq_len = 1;
        params.top_k = kTopK;
        params.d_model = kDModel;
        params.continuation_domain = plan.continuation_domain;
        params.placement = plan.placements.front();
        params.routed_tiers = plan.routed_tiers;
        MoEExpertDispatchOutput dispatch;
        params.output = &dispatch;

        MoEExpertDispatchStage stage(std::move(params));
        ASSERT_TRUE(stage.execute(ctx.get()));

        ASSERT_EQ(dispatch.tiers.size(), 2u);
        const auto &cold = dispatch.tiers[1];
        EXPECT_TRUE(cold.transfer_required);
        EXPECT_EQ(cold.token_rows, (std::vector<int>{0}));
        EXPECT_EQ(cold.transfer_mode, MoEExpertTransferMode::DecodeOneToken);
        EXPECT_EQ(cold.transfer_volume.outbound_bytes,
                  static_cast<size_t>(kDModel) * sizeof(float) + static_cast<size_t>(kTopK) * 2u * sizeof(float));
        EXPECT_EQ(cold.transfer_volume.return_bytes,
                  static_cast<size_t>(kDModel) * sizeof(float));
    }

    TEST(Test__MoEExpertOverlay_MultiAcceleratorTiers, RocmLocalTPTensorParallelExpertsExecutesBothParticipantsAndMatchesReference)
    {
        const std::string skip_reason = realAcceleratorSkipReason();
        if (!skip_reason.empty())
        {
            GTEST_SKIP() << skip_reason
                         << "; model-light dense FP32 dispatch/reduce semantics are covered by SyntheticThreeTierExecutionReducesToCudaContinuationReference";
        }

        auto plan = std::make_shared<MoEExpertParallelPlan>(makePhase8Plan());
        auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);
        ASSERT_NE(runtime_plan, nullptr);

        const auto *rocm_domain = runtime_plan->domainForName("rocm_hot");
        ASSERT_NE(rocm_domain, nullptr);
        EXPECT_FALSE(rocm_domain->multi_participant_execution_pending);
        EXPECT_TRUE(rocm_domain->domain_scoped_collective_context_ready);

        std::vector<GlobalDeviceAddress> devices;
        devices.reserve(rocm_domain->participants.size());
        for (const auto &participant : rocm_domain->participants)
            devices.push_back(participant.address);

        std::unique_ptr<ILocalTPContext> local_tp_context;
        try
        {
            local_tp_context = createLocalTPContext(std::move(devices), {}, CollectiveBackendType::RCCL);
        }
        catch (const std::exception &e)
        {
            GTEST_SKIP() << "Unable to create RCCL LocalTPContext: " << e.what();
        }
        ASSERT_NE(local_tp_context, nullptr);

        auto ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
        auto weights = makeWeights();
        auto hidden = fp32({kSeqLen, kDModel});
        auto routing_indices = fp32({kSeqLen, kTopK});
        auto routing_weights = fp32({kSeqLen, kTopK});
        fillHidden(hidden.get());
        fillRouting(routing_indices.get(), routing_weights.get());

        const auto &placement = plan->placements.front();
        const auto tier_mask = maskForTier(placement, 1);
        auto reference = fp32({kSeqLen, kDModel});
        ASSERT_TRUE(runExpertCompute(ctx.get(),
                                     hidden.get(),
                                     routing_indices.get(),
                                     routing_weights.get(),
                                     weights,
                                     tier_mask,
                                     reference.get()));

        auto actual = fp32({kSeqLen, kDModel});
        MoEExpertOverlayLocalTPRunParams run;
        run.input = hidden.get();
        run.routing_indices = routing_indices.get();
        run.routing_weights = routing_weights.get();
        run.gate_exps = weights.gate.get();
        run.up_exps = weights.up.get();
        run.down_exps = weights.down.get();
        run.output = actual.get();
        run.seq_len = kSeqLen;
        run.d_model = kDModel;
        run.num_experts = kNumExperts;
        run.top_k = kTopK;
        run.expert_intermediate = kIntermediate;
        run.layer_idx = 0;
        run.expert_mask = tier_mask;
        auto prepared_fixture = prepareLocalTPParticipants(weights, *rocm_domain, tier_mask);
        run.prepared_participants = &prepared_fixture.participants;
        run.prepared_partial_views = &prepared_fixture.partial_views;

        std::string reason;
        ASSERT_TRUE(MoEExpertOverlayLocalTPExecutor::canExecute(*rocm_domain, *local_tp_context, &reason)) << reason;

        MoEExpertOverlayLocalTPDiagnostics diagnostics;
        ASSERT_TRUE(MoEExpertOverlayLocalTPExecutor::runTensorParallelExperts(
            *rocm_domain,
            *local_tp_context,
            run,
            &diagnostics));

        expectTensorNear(actual.get(), reference.get(), 2e-4f);

        EXPECT_EQ(diagnostics.domain_name, "rocm_hot");
        EXPECT_EQ(diagnostics.backend, CollectiveBackendType::RCCL);
        EXPECT_EQ(diagnostics.degree, 2);
        EXPECT_EQ(diagnostics.total_routed_entries, 3);
        ASSERT_EQ(diagnostics.participants.size(), 2u);

        EXPECT_EQ(diagnostics.participants[0].participant_index, 0);
        EXPECT_EQ(diagnostics.participants[0].device, DeviceId::rocm(0));
        EXPECT_EQ(diagnostics.participants[0].intermediate_start, 0);
        EXPECT_EQ(diagnostics.participants[0].intermediate_end, 2);
        EXPECT_EQ(diagnostics.participants[0].routed_entry_count, 3);
        EXPECT_EQ(diagnostics.participants[0].expert_allreduce_count, 2);
        EXPECT_EQ(diagnostics.participants[0].executed_expert_ids, (std::vector<int>{2, 3}));

        EXPECT_EQ(diagnostics.participants[1].participant_index, 1);
        EXPECT_EQ(diagnostics.participants[1].device, DeviceId::rocm(1));
        EXPECT_EQ(diagnostics.participants[1].intermediate_start, 2);
        EXPECT_EQ(diagnostics.participants[1].intermediate_end, 4);
        EXPECT_EQ(diagnostics.participants[1].routed_entry_count, 3);
        EXPECT_EQ(diagnostics.participants[1].expert_allreduce_count, 2);
        EXPECT_EQ(diagnostics.participants[1].executed_expert_ids, (std::vector<int>{2, 3}));
    }

} // namespace llaminar2::test