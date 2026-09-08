/**
 * @file Test__Qwen35MoEGraphNativeProductionLowering.cpp
 * @brief Production graph-native lowering checks for Qwen3.5 MoE overlay tiers.
 *
 * The fixture verifies declarative Qwen3.5 MoE graph lowering without
 * substituting a test-only execution topology.  CPU-only cases inspect stable
 * graph ownership and route-buffer identities; explicit CUDA/ROCm integration
 * campaigns exercise the same lowering with real model weights and devices.
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoEGPUCurrentBatchLLEPStage.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoEOverlayActivationPacketStages.h"
#include "execution/compute_stages/stages/MoEOverlayTicketConsumeStage.h"
#include "execution/compute_stages/stages/MoERankBatchSparseStages.h"
#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "execution/compute_stages/stages/TPAllreduceStage.h"
#include "execution/local_execution/graph/DeviceGraphCaptureController.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "execution/moe/MoEOverlayInferenceTransaction.h"
#include "execution/moe/MoEOverlayDeviceControllerTopology.h"
#include "execution/moe/MoEOverlayNodeLocalDeviceControllerFabric.h"
#include "execution/moe/MoEOverlayNodeLocalRouteExchange.h"
#include "execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/moe/MoEOverlayRetainedParentComposer.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/MoERuntimeTable.h"
#include "loaders/ExpertGemmRegistry.h"
#include "loaders/ModelContext.h"
#include "models/qwen35moe/Qwen35MoEGraph.h"
#include "mocks/MockLocalTPContext.h"
#include "mocks/MockMPIContext.h"
#include "mocks/MockMPITopology.h"
#include "tensors/TensorKernels.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

namespace llaminar2::test
{
    namespace
    {
        constexpr int kDModel = 32;
        constexpr int kIntermediate = 32;
        constexpr int kNumExperts = 6;
        constexpr int kTopK = 2;
        constexpr int kSeqLen = 3;
        constexpr int kBatchSize = 1;

        /**
         * @brief Own the exact stream used to publish graph-build device state.
         *
         * These tests lower production GPU graphs and may initialize the
         * device-resident MoE runtime table while doing so. Keeping a real
         * backend stream beside each graph builder exercises the same explicit
         * producer contract as the orchestrator. The test-only synchronization
         * runs during fixture cleanup, before the graph builder releases the
         * table storage; it is not part of the production graph-build path.
         */
        class ScopedDevicePublicationStream
        {
        public:
            explicit ScopedDevicePublicationStream(DeviceId device)
                : device_(device),
                  backend_(getBackendFor(device))
            {
                if (!device_.is_gpu())
                    return;
                if (!backend_)
                {
                    throw std::runtime_error(
                        "No backend is available for graph publication on " +
                        device_.to_string());
                }
                stream_ = backend_->createStream(device_.toKernelDeviceIndex());
                if (!stream_)
                {
                    throw std::runtime_error(
                        "Failed to create graph publication stream on " +
                        device_.to_string());
                }
            }

            ~ScopedDevicePublicationStream()
            {
                if (!stream_)
                    return;
                const int ordinal = device_.toKernelDeviceIndex();
                if (!backend_->synchronizeStream(stream_, ordinal))
                {
                    ADD_FAILURE()
                        << "Failed to synchronize graph publication stream on "
                        << device_.to_string();
                }
                if (row_count_device_)
                    backend_->free(row_count_device_, ordinal);
                backend_->destroyStream(stream_, ordinal);
            }

            ScopedDevicePublicationStream(
                const ScopedDevicePublicationStream &) = delete;
            ScopedDevicePublicationStream &operator=(
                const ScopedDevicePublicationStream &) = delete;

            [[nodiscard]] void *get() const { return stream_; }

            /**
             * @brief Publish a test row count to a real device-owned scalar.
             *
             * Graph-lowering tests inspect pointer identity but do not execute
             * stages. Using backend-owned storage still exercises the same
             * ownership contract as production without introducing fake GPU
             * addresses into graph metadata.
             */
            [[nodiscard]] const int32_t *publishRowCount(int32_t value)
            {
                if (!device_.is_gpu() || !backend_ || !stream_)
                    throw std::runtime_error("Device row-count publication requires a GPU stream");
                const int ordinal = device_.toKernelDeviceIndex();
                if (!row_count_device_)
                    row_count_device_ = backend_->allocate(sizeof(value), ordinal);
                if (!row_count_device_ ||
                    !backend_->hostToDevice(
                        row_count_device_, &value, sizeof(value), ordinal, stream_))
                {
                    throw std::runtime_error(
                        "Failed to publish graph-test row count on " +
                        device_.to_string());
                }
                return static_cast<const int32_t *>(row_count_device_);
            }

        private:
            DeviceId device_;
            IBackend *backend_ = nullptr;
            void *stream_ = nullptr;
            void *row_count_device_ = nullptr;
        };

        class ScopedDebugEnv
        {
        public:
            explicit ScopedDebugEnv(std::initializer_list<std::pair<const char *, const char *>> values)
            {
                for (const auto &[name, value] : values)
                {
                    Entry entry;
                    entry.name = name;
                    if (const char *old_value = std::getenv(name))
                    {
                        entry.had_value = true;
                        entry.old_value = old_value;
                    }
                    entries_.push_back(entry);
                    ::setenv(name, value, 1);
                }
                mutableDebugEnv().reload();
            }

            ~ScopedDebugEnv()
            {
                for (const auto &entry : entries_)
                {
                    if (entry.had_value)
                        ::setenv(entry.name.c_str(), entry.old_value.c_str(), 1);
                    else
                        ::unsetenv(entry.name.c_str());
                }
                mutableDebugEnv().reload();
            }

            ScopedDebugEnv(const ScopedDebugEnv &) = delete;
            ScopedDebugEnv &operator=(const ScopedDebugEnv &) = delete;

        private:
            struct Entry
            {
                std::string name;
                bool had_value = false;
                std::string old_value;
            };

            std::vector<Entry> entries_;
        };

        class TestExpertGemm : public ITensorGemm
        {
        public:
            TestExpertGemm(int tag, ExpertGemmRegistry::WeightRole role)
                : tag_(tag), role_(role)
            {
                payload_[0] = static_cast<uint8_t>(tag_ & 0xff);
                scale_[0] = 1.0f;
            }

            bool supports_device(int /*device_idx*/) const override { return true; }

            bool multiply_tensor(
                const TensorBase * /*A*/,
                TensorBase * /*C*/,
                int /*m*/,
                int /*n*/,
                int /*k*/,
                bool /*transpose_B*/,
                float /*alpha*/,
                float /*beta*/,
                const TensorBase * /*bias*/,
                const IMPIContext * /*mpi_ctx*/,
                int /*device_idx*/,
                DeviceWorkspaceManager * /*workspace*/,
                int /*activation_row_offset*/) override
            {
                return false;
            }

            bool exportNativeVNNIMatrixDesc(DeviceNativeVNNIMatrixDesc &out) override
            {
                out = {};
                out.payload = payload_;
                out.scales = scale_;
                out.blocks_per_row = 1;
                out.codebook_id = 4;
                if (role_ == ExpertGemmRegistry::WeightRole::DOWN)
                {
                    out.n = kDModel;
                    out.k = kIntermediate;
                }
                else
                {
                    out.n = kIntermediate;
                    out.k = kDModel;
                }
                return true;
            }

            bool exportNativeVNNISourceIdentity(
                NativeVnniSourceIdentity &out) const override
            {
                out = NativeVnniSourceIdentity{
                    .codebook_id = 4,
                    .is_superblock = false,
                    .present = true,
                };
                return true;
            }

        private:
            int tag_ = 0;
            ExpertGemmRegistry::WeightRole role_ = ExpertGemmRegistry::WeightRole::GATE;
            uint8_t payload_[16] = {};
            float scale_[1] = {};
        };

        using ExpertRole = ExpertGemmRegistry::WeightRole;

        class TensorArena
        {
        public:
            FP32Tensor *fp32(std::vector<size_t> shape)
            {
                auto tensor = std::make_shared<FP32Tensor>(std::move(shape));
                auto *ptr = tensor.get();
                tensors_.push_back(std::move(tensor));
                return ptr;
            }

        private:
            std::vector<std::shared_ptr<TensorBase>> tensors_;
        };

        void fill(FP32Tensor *tensor, float value)
        {
            std::fill_n(tensor->mutable_data(), tensor->numel(), value);
        }

        RoutedExpertDomain domain(const std::string &name, GlobalDeviceAddress participant)
        {
            RoutedExpertDomain result;
            result.name = name;
            result.scope = ExecutionDomainScope::SINGLE;
            result.backend = CollectiveBackendType::HOST;
            result.participants = {std::move(participant)};
            result.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            result.owner_rank = 0;
            return result;
        }

        RoutedExpertDomain localTPDomain(
            const std::string &name,
            std::vector<GlobalDeviceAddress> participants)
        {
            RoutedExpertDomain result;
            result.name = name;
            result.scope = ExecutionDomainScope::RANK_LOCAL;
            result.backend = CollectiveBackendType::RCCL;
            result.participants = std::move(participants);
            result.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            result.owner_rank = 0;
            return result;
        }

        RoutedExpertTier tier(const std::string &name, const std::string &domain_name, int priority, bool fallback = false)
        {
            RoutedExpertTier result;
            result.name = name;
            result.domain = domain_name;
            result.priority = priority;
            result.fallback = fallback;
            return result;
        }

        std::shared_ptr<MoERoutedExpertPlacementPlan> makeProductionStylePlan()
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan->continuation_domain = "continuation";
            plan->base_model_domain = "continuation";
            plan->shared_expert_domain = "continuation";
            plan->continuation_domain_spec.domain = "continuation";
            plan->continuation_domain_spec.logical_root_participant = 0;
            plan->residency_policy = RoutedExpertResidencyPolicy::ExplicitMasks;
            plan->domains = {
                domain("continuation", GlobalDeviceAddress::cpu(0)),
                domain("hot_domain", GlobalDeviceAddress::cpu(0)),
                domain("warm_domain", GlobalDeviceAddress::cpu(1)),
                domain("cold_domain", GlobalDeviceAddress::cpu(2)),
            };
            plan->routed_tiers = {
                tier("hot", "hot_domain", 0),
                tier("warm", "warm_domain", 1),
                tier("cold", "cold_domain", 99, true),
            };
            plan->placements.push_back(RoutedExpertLayerPlacement{
                .layer = 0,
                .routed_expert_tier = {0, 1, 2, 0, 1, 2},
            });
            validateMoERoutedExpertPlacementPlanOrThrow(
                *plan,
                {.layer_count = 1, .routed_expert_count = kNumExperts});
            return plan;
        }

        std::shared_ptr<MoERoutedExpertPlacementPlan> makeLocalTPApportionedHotPlan(int layer_count = 1)
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan->continuation_domain = "hot_domain";
            plan->base_model_domain = "hot_domain";
            plan->shared_expert_domain = "hot_domain";
            plan->continuation_domain_spec.domain = "hot_domain";
            plan->continuation_domain_spec.logical_root_participant = 0;
            plan->residency_policy = RoutedExpertResidencyPolicy::ExplicitMasks;
            plan->domains = {
                localTPDomain(
                    "hot_domain",
                    {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)}),
            };
            plan->routed_tiers = {
                tier("hot", "hot_domain", 0),
            };
            for (int layer = 0; layer < layer_count; ++layer)
            {
                plan->placements.push_back(RoutedExpertLayerPlacement{
                    .layer = layer,
                    .routed_expert_tier = {0, 0, 0, 0, 0, 0},
                });
            }
            validateMoERoutedExpertPlacementPlanOrThrow(
                *plan,
                {.layer_count = layer_count, .routed_expert_count = kNumExperts});
            return plan;
        }

        /**
         * @brief Build a two-rank overlay with a two-GPU LocalTP continuation.
         *
         * Rank zero owns two ROCm continuation participants; rank one owns four
         * remote ROCm expert participants. This is the production
         * two-accelerator/four-accelerator shape. Its dense path is TP during
         * prefill and replicated during decode, while routed experts remain
         * apportioned across the two continuation devices in both phases. The
         * test proves that those same-rank peers stay in captured local graph
         * branches and four remote endpoints lower to one retained rank
         * transaction rather than serialized sparse collectives.
         */
        std::shared_ptr<MoERoutedExpertPlacementPlan>
        makeDistributedLocalTPContinuationPlan(int layer_count = 1)
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology =
                RoutedExpertPlacementTopology::TieredOverlay;
            plan->continuation_domain = "continuation_domain";
            plan->base_model_domain = "continuation_domain";
            plan->shared_expert_domain = "continuation_domain";
            plan->continuation_domain_spec.domain =
                "continuation_domain";
            plan->continuation_domain_spec.logical_root_participant = 0;
            plan->continuation_domain_spec.setDensePolicy(
                DenseParallelPolicy::PrefillTensorParallelDecodeReplicated);
            plan->residency_policy =
                RoutedExpertResidencyPolicy::ExplicitMasks;

            auto continuation = localTPDomain(
                "continuation_domain",
                {GlobalDeviceAddress::rocm(0),
                 GlobalDeviceAddress::rocm(1)});
            continuation.owner_rank = 0;
            auto remote = localTPDomain(
                "remote_domain",
                {GlobalDeviceAddress::rocm(0),
                 GlobalDeviceAddress::rocm(1),
                 GlobalDeviceAddress::rocm(2),
                 GlobalDeviceAddress::rocm(3)});
            remote.owner_rank = 1;
            plan->domains = {
                std::move(continuation),
                std::move(remote),
            };
            plan->routed_tiers = {
                tier("priority_0", "continuation_domain", 0),
                tier("priority_1", "remote_domain", 1, true),
            };
            for (int layer = 0; layer < layer_count; ++layer)
            {
                plan->placements.push_back(RoutedExpertLayerPlacement{
                    .layer = layer,
                    .routed_expert_tier = {0, 0, 1, 1, 1, 1},
                });
            }
            validateMoERoutedExpertPlacementPlanOrThrow(
                *plan,
                {.layer_count = layer_count,
                 .routed_expert_count = kNumExperts});
            return plan;
        }

        /**
         * @brief Build the live two-rank CUDA-continuation/ROCm-follower shape.
         *
         * The continuation participant is intentionally a one-device domain.
         * Its experts must remain in the CUDA graph while the ROCm tier uses a
         * mapped activation lane owned by rank one. This is the smallest
         * production topology that proves adding another device type does not
         * demote continuation-local GPU work to a host sparse stage.
         */
        std::shared_ptr<MoERoutedExpertPlacementPlan>
        makeDistributedSingleGPUContinuationPlan()
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology =
                RoutedExpertPlacementTopology::TieredOverlay;
            plan->continuation_domain = "continuation_domain";
            plan->base_model_domain = "continuation_domain";
            plan->shared_expert_domain = "continuation_domain";
            plan->continuation_domain_spec.domain =
                "continuation_domain";
            plan->continuation_domain_spec.logical_root_participant = 0;
            plan->residency_policy =
                RoutedExpertResidencyPolicy::ExplicitMasks;

            auto continuation = domain(
                "continuation_domain", GlobalDeviceAddress::cuda(0));
            continuation.backend = CollectiveBackendType::NCCL;
            continuation.owner_rank = 0;
            auto remote = domain(
                "remote_domain", GlobalDeviceAddress::rocm(0));
            remote.backend = CollectiveBackendType::RCCL;
            remote.owner_rank = 1;
            plan->domains = {
                std::move(continuation),
                std::move(remote),
            };
            plan->routed_tiers = {
                tier("priority_0", "continuation_domain", 0),
                tier("priority_1", "remote_domain", 1, true),
            };
            plan->placements.push_back(RoutedExpertLayerPlacement{
                .layer = 0,
                .routed_expert_tier = {0, 0, 0, 1, 1, 1},
            });
            validateMoERoutedExpertPlacementPlanOrThrow(
                *plan,
                {.layer_count = 1,
                 .routed_expert_count = kNumExperts});
            return plan;
        }

        GraphConfig makeConfig(
            std::shared_ptr<MoERoutedExpertPlacementPlan> plan,
            int layer_count = 2)
        {
            GraphConfig config;
            config.n_layers = layer_count;
            config.total_n_layers = layer_count;
            config.d_model = kDModel;
            config.n_heads = 2;
            config.n_kv_heads = 2;
            config.head_dim = 4;
            config.d_ff = 16;
            config.vocab_size = 32;
            config.rms_norm_eps = 1e-6f;
            config.default_device = DeviceId::cpu();
            config.moe.num_experts = kNumExperts;
            config.moe.top_k = kTopK;
            config.moe.intermediate_size = kIntermediate;
            config.moe.norm_topk_prob = true;
            config.moe.routed_expert_plan = std::move(plan);
            config.moe.rebalance_config.mode =
                MoERebalanceRuntimeMode::Off;
            return config;
        }

        void fillExpert3D(FP32Tensor *tensor, int rows, int cols, float scale)
        {
            float *data = tensor->mutable_data();
            for (int expert = 0; expert < kNumExperts; ++expert)
            {
                for (int row = 0; row < rows; ++row)
                {
                    for (int col = 0; col < cols; ++col)
                    {
                        const size_t offset = static_cast<size_t>(expert) * rows * cols +
                                              static_cast<size_t>(row) * cols +
                                              static_cast<size_t>(col);
                        data[offset] = scale * static_cast<float>((expert + 1) * 3 + row + 1) +
                                       0.001f * static_cast<float>(col + 1);
                    }
                }
            }
        }

        LayerWeights makeLayerWeights(TensorArena &arena)
        {
            LayerWeights layer;
            layer.ffn_norm = arena.fp32({kDModel});
            fill(static_cast<FP32Tensor *>(layer.ffn_norm), 1.0f);

            layer.moe_gate = arena.fp32({kNumExperts, kDModel});
            fill(static_cast<FP32Tensor *>(layer.moe_gate), 0.25f);

            layer.moe_gate_exps = arena.fp32({kDModel, kIntermediate, kNumExperts});
            layer.moe_up_exps = arena.fp32({kDModel, kIntermediate, kNumExperts});
            layer.moe_down_exps = arena.fp32({kIntermediate, kDModel, kNumExperts});
            fillExpert3D(static_cast<FP32Tensor *>(layer.moe_gate_exps), kIntermediate, kDModel, 0.010f);
            fillExpert3D(static_cast<FP32Tensor *>(layer.moe_up_exps), kIntermediate, kDModel, 0.012f);
            fillExpert3D(static_cast<FP32Tensor *>(layer.moe_down_exps), kDModel, kIntermediate, 0.008f);
            return layer;
        }

        void registerDomainExpertEngine(
            ExpertGemmRegistry &registry,
            const std::string &domain_name,
            DeviceId device,
            int layer_idx,
            int expert,
            ExpertRole role,
            int tag)
        {
            auto engine = std::make_shared<TestExpertGemm>(tag, role);
            registry.registerEngineForDomain(
                domain_name,
                device,
                layer_idx,
                expert,
                role,
                engine.get(),
                engine);
        }

        void registerCompleteDomainExpertLayer(
            ExpertGemmRegistry &registry,
            const std::string &domain_name,
            DeviceId device,
            int layer_idx)
        {
            for (int expert = 0; expert < kNumExperts; ++expert)
            {
                const int tag_base = 1000 * device.ordinal + 10 * expert;
                registerDomainExpertEngine(
                    registry, domain_name, device, layer_idx, expert, ExpertRole::GATE, tag_base + 1);
                registerDomainExpertEngine(
                    registry, domain_name, device, layer_idx, expert, ExpertRole::UP, tag_base + 2);
                registerDomainExpertEngine(
                    registry, domain_name, device, layer_idx, expert, ExpertRole::DOWN, tag_base + 3);
            }
        }

        void registerOwnedParticipantExpertLayers(
            ExpertGemmRegistry &registry,
            const MoEExpertOwnerMap &owner_map,
            int layer_count)
        {
            for (const auto &participant : owner_map.participants())
            {
                const int world_rank = participant.world_rank_known
                                           ? participant.world_rank
                                           : -1;
                for (int layer = 0; layer < layer_count; ++layer)
                {
                    const auto mask = owner_map.expertMaskForParticipant(
                        layer, participant.participant_id, kNumExperts);
                    for (int expert = 0; expert < kNumExperts; ++expert)
                    {
                        if (!mask[static_cast<std::size_t>(expert)])
                            continue;
                        for (const ExpertRole role : {
                                 ExpertRole::GATE,
                                 ExpertRole::UP,
                                 ExpertRole::DOWN,
                             })
                        {
                            const int tag =
                                10000 * participant.participant_id +
                                100 * layer + 10 * expert +
                                static_cast<int>(role) + 1;
                            auto engine = std::make_shared<TestExpertGemm>(
                                tag, role);
                            registry.registerEngineForParticipant(
                                participant.domain_name,
                                participant.device,
                                world_rank,
                                participant.domain_participant_index,
                                layer,
                                expert,
                                role,
                                engine.get(),
                                engine);
                        }
                    }
                }
            }
        }

        void registerCompleteDeviceExpertLayer(
            ExpertGemmRegistry &registry,
            DeviceId device,
            int layer_idx)
        {
            for (int expert = 0; expert < kNumExperts; ++expert)
            {
                for (const ExpertRole role : {
                         ExpertRole::GATE,
                         ExpertRole::UP,
                         ExpertRole::DOWN,
                     })
                {
                    const int tag =
                        100 * expert + static_cast<int>(role) + 1;
                    auto engine = std::make_shared<TestExpertGemm>(tag, role);
                    registry.registerEngine(
                        device,
                        layer_idx,
                        expert,
                        role,
                        engine.get(),
                        engine);
                }
            }
        }

        /**
         * @brief Build a model context whose ROCm hot domain has complete prepared experts.
         *
         * The lowering tests deliberately use registry-owned GEMM handles rather
         * than raw parent tensors.  Production ExpertOverlay graphs have the
         * same requirement: expert execution must resolve through the prepared
         * registry for the participant that owns the routes.
         */
        std::shared_ptr<ModelContext> makeTestingModelContextWithHotDomainExperts(int layer_count = 1)
        {
            auto model_ctx = ModelContext::createForTesting(
                "test.gguf",
                nullptr,
                static_cast<uint32_t>(std::max(1, layer_count)),
                /*with_weight_manager=*/true);
            if (!model_ctx || !model_ctx->concreteWeightManager())
                throw std::runtime_error("ModelContext test WeightManager was not created");

            auto &registry = model_ctx->concreteWeightManager()->expertGemmRegistry();
            for (int layer = 0; layer < std::max(1, layer_count); ++layer)
            {
                registerCompleteDomainExpertLayer(registry, "hot_domain", DeviceId::rocm(0), layer);
                registerCompleteDomainExpertLayer(registry, "hot_domain", DeviceId::rocm(1), layer);
            }
            return model_ctx;
        }

        /**
         * @brief Build a CPU tiered-overlay context with every tier's prepared engines.
         *
         * `DeviceId` intentionally identifies CPU execution rather than a
         * socket.  The tier/domain name is therefore the registry key that
         * distinguishes the logical hot, warm, and cold CPU participants in
         * this graph-lowering fixture.  This mirrors production's
         * domain-qualified registry lookup and avoids testing an invalid
         * raw-weight fallback path.
         */
        std::shared_ptr<ModelContext> makeTestingModelContextWithTieredOverlayExperts(
            int layer_count = 1)
        {
            auto model_ctx = ModelContext::createForTesting(
                "test.gguf",
                nullptr,
                static_cast<uint32_t>(std::max(1, layer_count)),
                /*with_weight_manager=*/true);
            if (!model_ctx || !model_ctx->concreteWeightManager())
            {
                throw std::runtime_error(
                    "ModelContext test WeightManager was not created");
            }

            auto &registry = model_ctx->concreteWeightManager()->expertGemmRegistry();
            for (int layer = 0; layer < std::max(1, layer_count); ++layer)
            {
                for (const char *domain_name : {
                         "hot_domain",
                         "warm_domain",
                         "cold_domain",
                     })
                {
                    registerCompleteDomainExpertLayer(
                        registry,
                        domain_name,
                        DeviceId::cpu(),
                        layer);
                }
            }
            return model_ctx;
        }

        ActivationBuffers makeActivationBuffers(
            TensorArena &arena,
            int row_count = kSeqLen)
        {
            if (row_count <= 0)
            {
                throw std::invalid_argument(
                    "Activation-buffer row count must be positive");
            }
            const size_t rows = static_cast<size_t>(row_count);

            ActivationBuffers buffers;
            buffers.attn_proj = arena.fp32({rows, kDModel});
            buffers.current_hidden = arena.fp32({rows, kDModel});
            buffers.normalized = arena.fp32({rows, kDModel});
            fill(static_cast<FP32Tensor *>(buffers.attn_proj), 0.0f);
            fill(static_cast<FP32Tensor *>(buffers.current_hidden), 0.0f);
            fill(static_cast<FP32Tensor *>(buffers.normalized), 0.0f);

            buffers.extensions[BufferId::MOE_EXPERT_INDICES] = arena.fp32({rows, kTopK});
            buffers.extensions[BufferId::MOE_EXPERT_WEIGHTS] = arena.fp32({rows, kTopK});
            buffers.extensions[BufferId::MOE_COMBINED_OUTPUT] = arena.fp32({rows, kDModel});
            /*
             * BufferArena preserves the leading row axis and flattens every
             * trailing schema axis into matrix columns. Model the production
             * representation here so live-prefix collective checks cannot
             * pass only against a direct rank-three test tensor.
             */
            buffers.extensions[BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS] =
                arena.fp32(
                    {rows,
                     static_cast<size_t>(kTopK + 2) *
                         static_cast<size_t>(kDModel)});
            buffers.extensions[BufferId::MOE_SHARED_EXPERT_OUTPUT] = arena.fp32({rows, kDModel});
            buffers.extensions[BufferId::MOE_GATE_SCRATCH] = arena.fp32({rows, kIntermediate});
            buffers.extensions[BufferId::MOE_UP_SCRATCH] = arena.fp32({rows, kIntermediate});
            return buffers;
        }

        /**
         * @brief Build one complete MoE predictor block for GPU graph lowering.
         *
         * The sidecar borrows the routed/shared expert tensors from @p moe_layer
         * and adds the attention and predictor-only weights required by the
         * public `buildMTPGraph()` entrypoint. Tensor values are irrelevant to
         * declarative lowering, but every pointer and shape is production-valid.
         */
        MTPDepthWeights makeMTPSidecarWeights(
            TensorArena &arena,
            const LayerWeights &moe_layer,
            int source_layer_index)
        {
            const size_t d_model = static_cast<size_t>(kDModel);
            const size_t q_dim = 8u;
            const size_t kv_dim = 8u;

            MTPDepthWeights weights;
            weights.depth_index = 0;
            weights.source_layer_index = source_layer_index;
            weights.nextn_block_layout = true;
            weights.fc = arena.fp32({d_model, 2u * d_model});
            weights.pre_fc_norm_hidden = arena.fp32({d_model});
            weights.pre_fc_norm_embedding = arena.fp32({d_model});
            weights.final_norm = arena.fp32({d_model});
            weights.fa_block = moe_layer;
            weights.fa_block.attn_norm = arena.fp32({d_model});
            weights.fa_block.wq = arena.fp32({2u * q_dim, d_model});
            weights.fa_block.wk = arena.fp32({kv_dim, d_model});
            weights.fa_block.wv = arena.fp32({kv_dim, d_model});
            weights.fa_block.wo = arena.fp32({d_model, q_dim});
            weights.fa_block.q_norm = arena.fp32({4u});
            weights.fa_block.k_norm = arena.fp32({4u});
            return weights;
        }

        /**
         * @brief Allocate every stable output bound by a one-row MoE sidecar.
         * @param arena Owner whose lifetime encloses the constructed graph.
         * @return Complete non-owning MTP output bundle.
         */
        MTPForwardOutput makeMTPSidecarOutput(TensorArena &arena)
        {
            constexpr size_t rows = 1u;
            constexpr size_t q_dim = 8u;
            constexpr size_t kv_dim = 8u;
            MTPForwardOutput output;
            output.logits = arena.fp32({rows, 32u});
            output.hidden = arena.fp32({rows, kDModel});
            output.embedding = arena.fp32({rows, kDModel});
            output.norm_hidden = arena.fp32({rows, kDModel});
            output.norm_embedding = arena.fp32({rows, kDModel});
            output.concat = arena.fp32({rows, 2u * kDModel});
            output.projected = arena.fp32({rows, kDModel});
            output.q = arena.fp32({rows, q_dim});
            output.k = arena.fp32({rows, kv_dim});
            output.v = arena.fp32({rows, kv_dim});
            output.q_raw = arena.fp32({rows, 2u * q_dim});
            output.q_gate = arena.fp32({rows, q_dim});
            output.attn_output = arena.fp32({rows, q_dim});
            output.attn_proj = arena.fp32({rows, kDModel});
            output.gate = arena.fp32({rows, 16u});
            output.up = arena.fp32({rows, 16u});
            output.ffn_output = arena.fp32({rows, kDModel});
            output.moe_expert_indices = arena.fp32({rows, kTopK});
            output.moe_expert_weights = arena.fp32({rows, kTopK});
            output.moe_combined_output = arena.fp32({rows, kDModel});
            output.moe_canonical_route_contributions = arena.fp32(
                {rows,
                 static_cast<size_t>(kTopK + 2) *
                     static_cast<size_t>(kDModel)});
            output.moe_shared_expert_output =
                arena.fp32({rows, kDModel});
            output.moe_gate_scratch =
                arena.fp32({rows, kIntermediate});
            output.moe_up_scratch =
                arena.fp32({rows, kIntermediate});
            return output;
        }

        size_t countStagesOfType(const ComputeGraph &graph, ComputeStageType type)
        {
            size_t count = 0;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                const auto *node = graph.getNode(node_name);
                if (node && node->stage->type() == type)
                    ++count;
            }
            return count;
        }

        bool hasDependency(
            const ComputeGraph &graph,
            const std::string &node_name,
            const std::string &dependency)
        {
            const auto *node = graph.getNode(node_name);
            if (!node)
                return false;
            return std::find(
                       node->dependencies.begin(),
                       node->dependencies.end(),
                       dependency) != node->dependencies.end();
        }

        std::vector<std::string> stageNamesOfType(const ComputeGraph &graph, ComputeStageType type)
        {
            std::vector<std::string> names;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                const auto *node = graph.getNode(node_name);
                if (node && node->stage->type() == type)
                    names.push_back(node_name);
            }
            return names;
        }

        /**
         * @brief Flatten active and passive native-capture waves in rendezvous order.
         *
         * LocalTP participants may own different packet stages, but they must
         * present exactly the same capture lifecycle to their shared
         * collective context. Unannotated common segments use their stable
         * stage range; root-only annotated segments are matched by passive
         * sibling waves.
         */
        std::vector<std::string> captureWaveSchedule(
            const DeviceGraphExecutor::GraphSegmentCache &cache)
        {
            std::vector<std::string> result;
            for (const auto &segment : cache.segments)
            {
                if (segment.capturable)
                {
                    if (!segment.capture_wave_identity.empty())
                    {
                        result.push_back(
                            "explicit:" +
                            segment.capture_wave_identity);
                    }
                    else
                    {
                        if (segment.stage_names.empty())
                            throw std::logic_error(
                                "capturable test segment has no stages");
                        result.push_back(
                            "implicit:" + segment.stage_names.front() +
                            ".." + segment.stage_names.back());
                    }
                }
                for (const auto &passive :
                     segment.passive_capture_waves_after)
                {
                    result.push_back("explicit:" + passive.identity);
                }
            }
            return result;
        }

        const MoESparseDispatchStage *firstSparseDispatchStage(const ComputeGraph &graph)
        {
            for (const auto &node_name : graph.getExecutionOrder())
            {
                const auto *node = graph.getNode(node_name);
                if (!node)
                    continue;
                if (const auto *stage = dynamic_cast<const MoESparseDispatchStage *>(node->stage.get()))
                    return stage;
            }
            return nullptr;
        }

        std::vector<const MoESparseDispatchStage *> sparseDispatchStages(const ComputeGraph &graph)
        {
            std::vector<const MoESparseDispatchStage *> stages;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                const auto *node = graph.getNode(node_name);
                if (!node)
                    continue;
                if (const auto *stage = dynamic_cast<const MoESparseDispatchStage *>(node->stage.get()))
                    stages.push_back(stage);
            }
            return stages;
        }

        std::vector<const MoELocalExpertStage *> localExpertStages(const ComputeGraph &graph)
        {
            std::vector<const MoELocalExpertStage *> stages;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                const auto *node = graph.getNode(node_name);
                if (!node)
                    continue;
                if (const auto *stage = dynamic_cast<const MoELocalExpertStage *>(node->stage.get()))
                    stages.push_back(stage);
            }
            return stages;
        }

        const MoEExpertComputeStage *expertComputeStage(const ComputeGraph &graph, const std::string &node_name)
        {
            const auto *node = graph.getNode(node_name);
            if (!node)
                return nullptr;
            return dynamic_cast<const MoEExpertComputeStage *>(node->stage.get());
        }

        /** @return Typed GPU current-batch LLEP phase at @p node_name. */
        const MoEGPUCurrentBatchLLEPStage *gpuCurrentBatchLLEPStage(
            const ComputeGraph &graph,
            const std::string &node_name)
        {
            const auto *node = graph.getNode(node_name);
            if (!node)
                return nullptr;
            return dynamic_cast<const MoEGPUCurrentBatchLLEPStage *>(
                node->stage.get());
        }

    } // namespace

    TEST(Test__Qwen35MoEGraphNativeProductionLowering, TieredOverlayDefaultsToGraphNativeSparseStages)
    {
        auto plan = makeProductionStylePlan();
        const auto owner_map = MoEExpertOwnerMap::build(*plan);
        GraphConfig config = makeConfig(plan);
        ASSERT_EQ(config.moe.expert_overlay_runtime_plan, nullptr);

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        TensorArena activation_arena;
        auto buffers = makeActivationBuffers(activation_arena);

        auto model_ctx = makeTestingModelContextWithTieredOverlayExperts(
            config.n_layers);
        Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
        ComputeGraph graph = graph_builder.buildFFNGraph(
            layer,
            buffers,
            0,
            kSeqLen,
            kBatchSize,
            DeviceId::cpu(),
            /*device_state_publication_stream=*/nullptr);

        const auto *dispatch_node = graph.getNode("layer0_moe_expert_dispatch");
        ASSERT_NE(dispatch_node, nullptr);
        const auto *dispatch_stage =
            dynamic_cast<const MoEExpertDispatchStage *>(dispatch_node->stage.get());
        ASSERT_NE(dispatch_stage, nullptr);
        EXPECT_EQ(dispatch_stage->params().routing_indices_buffer_id, BufferId::MOE_EXPERT_INDICES);
        EXPECT_EQ(dispatch_stage->params().routing_weights_buffer_id, BufferId::MOE_EXPERT_WEIGHTS);
        EXPECT_EQ(dispatch_stage->params().hidden_buffer_id, BufferId::NORMALIZED);
        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::MOE_EXPERT_DISPATCH), 1u);
        EXPECT_GT(countStagesOfType(graph, ComputeStageType::MOE_SPARSE_DISPATCH), 0u);
        EXPECT_GT(countStagesOfType(graph, ComputeStageType::MOE_LOCAL_EXPERT), 0u);
        EXPECT_GT(countStagesOfType(graph, ComputeStageType::MOE_SPARSE_RETURN_REDUCE), 0u);
        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::MOE_EXPERT_FFN), 0u);

        const auto *sparse_dispatch = firstSparseDispatchStage(graph);
        ASSERT_NE(sparse_dispatch, nullptr);
        const auto &params = sparse_dispatch->params();
        EXPECT_TRUE(params.key.isValid());
        EXPECT_NE(params.workspace, nullptr);
        EXPECT_NE(params.collective_context, nullptr);
        EXPECT_GE(params.source_participant, 0);
        EXPECT_GE(params.target_participant, 0);

        const auto participant_ids = owner_map.participantIdsForTier(params.key.tier_idx);
        EXPECT_NE(std::find(participant_ids.begin(), participant_ids.end(), params.target_participant),
                  participant_ids.end());

        bool found_root_sparse_dispatch = false;
        for (const auto *stage : sparseDispatchStages(graph))
        {
            const auto &sparse_params = stage->params();
            if (!sparse_params.hidden)
                continue;
            found_root_sparse_dispatch = true;
            EXPECT_EQ(sparse_params.hidden_buffer_id, BufferId::NORMALIZED);
            EXPECT_EQ(sparse_params.routing_indices_buffer_id, BufferId::MOE_EXPERT_INDICES);
            EXPECT_EQ(sparse_params.routing_weights_buffer_id, BufferId::MOE_EXPERT_WEIGHTS);
        }
        EXPECT_TRUE(found_root_sparse_dispatch);
    }

    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         ProductionAuthorityBindsEveryLocalStageToItsExactInitialEpochBank)
    {
        auto plan = makeProductionStylePlan();
        const auto owner_map = MoEExpertOwnerMap::build(*plan);
        GraphConfig config = makeConfig(plan, /*layer_count=*/1);
        auto authority = std::make_shared<MoEOverlayResidencyAuthority>(
            MoEOverlayResidencyAuthority::Config{
                .initial_plan = *plan,
                .model_metadata = {
                    .num_layers = 1,
                    .num_experts = kNumExperts,
                    .d_model = kDModel,
                    .routed_intermediate_size = kIntermediate,
                },
                .maintenance_mode = MoERebalanceRuntimeMode::Off,
                .histogram = nullptr,
                .perf_device = "cpu_test",
            });
        const auto snapshot = authority->snapshot();
        ASSERT_NE(snapshot, nullptr);
        ASSERT_TRUE(snapshot->valid());

        std::vector<int> local_participants;
        for (const auto &participant : owner_map.participants())
            local_participants.push_back(participant.participant_id);
        auto participant_residency =
            std::make_shared<MoEOverlayParticipantResidencyRegistry>(
                MoEOverlayParticipantResidencyRegistry::Config{
                    .owner_map = snapshot->owner_map,
                    .local_participant_ids = local_participants,
                    .num_layers = 1,
                    .num_experts = kNumExperts,
                    .initial_epoch = snapshot->epoch,
                });
        config.moe.expert_overlay_residency_authority = authority;
        config.moe.durable_residency_authority =
            MoEDurableResidencyAuthorityKind::ExpertOverlayRCU;
        config.moe.expert_overlay_participant_residency =
            participant_residency;

        auto model_ctx = makeTestingModelContextWithTieredOverlayExperts(1);
        registerOwnedParticipantExpertLayers(
            model_ctx->concreteWeightManager()->expertGemmRegistry(),
            owner_map,
            /*layer_count=*/1);
        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        TensorArena activation_arena;
        auto buffers = makeActivationBuffers(activation_arena);

        Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
        ComputeGraph graph = graph_builder.buildFFNGraph(
            layer,
            buffers,
            /*layer_idx=*/0,
            kSeqLen,
            kBatchSize,
            DeviceId::cpu(),
            /*device_state_publication_stream=*/nullptr);

        const auto stages = localExpertStages(graph);
        ASSERT_FALSE(stages.empty());
        EXPECT_TRUE(participant_residency->allInitialBanksReady());
        for (const auto *stage : stages)
        {
            ASSERT_NE(stage, nullptr);
            const int participant = stage->participantIndex();
            ASSERT_NE(stage->params().overlay_participant_residency, nullptr);
            EXPECT_EQ(
                stage->params().overlay_participant_residency,
                participant_residency->endpoint(participant));
            const auto bank =
                stage->params().overlay_participant_residency->acquire(
                    snapshot->epoch);
            ASSERT_NE(bank, nullptr);
            EXPECT_EQ(bank->epoch, snapshot->epoch);
            EXPECT_EQ(bank->participant_id, participant);
        }
    }

    /**
     * @brief Graph variants share compact tensors only within one participant's serial family.
     *
     * This is a production graph-builder test, not a synthetic pointer cache:
     * both graphs are lowered through the ordinary sparse collective topology.
     * The test also proves that different CPU participants keep independent
     * packet storage, preserving the future ability to execute them in
     * parallel even though their current graph nodes are serially ordered.
     */
    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         TieredOverlaySerialVariantsSharePerParticipantCompactBuffers)
    {
        auto plan = makeProductionStylePlan();
        GraphConfig config = makeConfig(plan);
        config.max_seq_len = 8;
        /*
         * Production MemoryPlanner publishes flattened graph rows separately
         * from the full KV horizon.  Make the difference visible here so a
         * later graph builder cannot accidentally allocate compact packets at
         * max_seq_len after admission selected a smaller captured family.
         */
        config.max_activation_rows = 5;

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        TensorArena small_activation_arena;
        auto small_buffers = makeActivationBuffers(
            small_activation_arena, /*row_count=*/1);
        TensorArena large_activation_arena;
        auto large_buffers = makeActivationBuffers(
            large_activation_arena, /*row_count=*/kSeqLen);

        auto model_ctx = makeTestingModelContextWithTieredOverlayExperts(
            config.n_layers);
        Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
        ComputeGraph small_graph = graph_builder.buildFFNGraph(
            layer,
            small_buffers,
            /*layer_idx=*/0,
            /*seq_len=*/1,
            kBatchSize,
            DeviceId::cpu(),
            /*device_state_publication_stream=*/nullptr);
        ComputeGraph large_graph = graph_builder.buildFFNGraph(
            layer,
            large_buffers,
            /*layer_idx=*/0,
            /*seq_len=*/kSeqLen,
            kBatchSize,
            DeviceId::cpu(),
            /*device_state_publication_stream=*/nullptr);

        const auto small_stages = localExpertStages(small_graph);
        const auto large_stages = localExpertStages(large_graph);
        ASSERT_FALSE(small_stages.empty());
        ASSERT_EQ(small_stages.size(), large_stages.size());

        std::map<int, const MoELocalExpertStage *> small_by_participant;
        std::map<int, const MoELocalExpertStage *> large_by_participant;
        for (const auto *stage : small_stages)
        {
            ASSERT_NE(stage, nullptr);
            ASSERT_NE(stage->params().serial_compact_buffer_arena, nullptr);
            small_by_participant.emplace(stage->participantIndex(), stage);
        }
        for (const auto *stage : large_stages)
        {
            ASSERT_NE(stage, nullptr);
            ASSERT_NE(stage->params().serial_compact_buffer_arena, nullptr);
            large_by_participant.emplace(stage->participantIndex(), stage);
        }
        ASSERT_EQ(small_by_participant.size(), large_by_participant.size());

        for (const auto &[participant, small_stage] : small_by_participant)
        {
            const auto large_it = large_by_participant.find(participant);
            ASSERT_NE(large_it, large_by_participant.end());
            const auto *large_stage = large_it->second;
            ASSERT_NE(large_stage, nullptr);
            EXPECT_EQ(
                small_stage->params().serial_compact_buffer_arena.get(),
                large_stage->params().serial_compact_buffer_arena.get())
                << "participant=" << participant;
            ASSERT_NE(
                small_stage->params().serial_compact_buffer_arena
                    ->cpuGroupedWorkspace(),
                nullptr)
                << "A production CPU overlay participant must first-touch its "
                   "grouped execution workspace during graph setup";
            EXPECT_EQ(
                small_stage->params().serial_compact_buffer_arena
                    ->cpuGroupedWorkspace().get(),
                large_stage->params().serial_compact_buffer_arena
                    ->cpuGroupedWorkspace().get())
                << "Every serial bucket for one participant must share one "
                   "grouped execution workspace; participant="
                << participant;
            EXPECT_NE(
                small_stage->compactHiddenTensorForDiagnostics(),
                large_stage->compactHiddenTensorForDiagnostics())
                << "A decode-size packet and a prefill-size packet require "
                   "independent tensor coherence; participant="
                << participant;
            EXPECT_EQ(
                small_stage->compactRowCapacityForDiagnostics(),
                1u);
            const auto *expected_large_family =
                small_stage->params().serial_compact_buffer_arena
                    ->smallestFamilySupporting(
                        static_cast<size_t>(kSeqLen));
            ASSERT_NE(expected_large_family, nullptr);
            EXPECT_EQ(
                large_stage->compactRowCapacityForDiagnostics(),
                expected_large_family->row_capacity)
                << "The captured prefill graph must bind the smallest exact "
                   "family that contains its live row envelope";
            EXPECT_EQ(expected_large_family->row_capacity, 4u);
            EXPECT_GE(
                small_stage->params().serial_compact_buffer_arena->rowCapacity(),
                static_cast<size_t>(config.max_activation_rows));
            EXPECT_EQ(
                small_stage->params().serial_compact_buffer_arena->rowCapacity(),
                static_cast<size_t>(config.max_activation_rows));
            EXPECT_EQ(
                small_stage->params().serial_compact_buffer_arena->familyCount(),
                4u)
                << "The bounded five-row envelope retains the complete "
                   "{1,2,4,5} capture family ladder";
            ASSERT_NE(
                small_stage->params().serial_compact_buffer_arena
                    ->smallestFamilySupporting(1),
                nullptr);
            EXPECT_EQ(
                small_stage->params().serial_compact_buffer_arena
                    ->smallestFamilySupporting(1)
                    ->row_capacity,
                1u);
        }

        ASSERT_GE(small_by_participant.size(), 2u);
        auto first = small_by_participant.begin();
        auto second = std::next(first);
        EXPECT_NE(
            first->second->params().serial_compact_buffer_arena.get(),
            second->second->params().serial_compact_buffer_arena.get())
            << "Independent overlay participants must not alias a compact packet arena";
        EXPECT_NE(
            first->second->params().serial_compact_buffer_arena
                ->cpuGroupedWorkspace().get(),
            second->second->params().serial_compact_buffer_arena
                ->cpuGroupedWorkspace().get())
            << "Independent CPU participants must not alias execution scratch";
    }

    /**
     * @brief A single GPU continuation remains device-owned beside a remote tier.
     *
     * This is the production CUDA/ROCm topology used by the real-weight parity
     * campaign. The root graph must contain one direct CUDA expert stage and
     * one mapped ROCm packet lane; no host dispatch, rank-batch stage, or
     * continuation-local sparse round trip is permitted.
     */
    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         DistributedDynamicSingleGPUContinuationUsesMappedAdmissionSelector)
    {
        auto plan = makeDistributedSingleGPUContinuationPlan();
        plan->residency_policy =
            RoutedExpertResidencyPolicy::RoutedTierRebalanced;
        plan->authority_execution =
            MoEOverlayAuthorityExecutionKind::DeviceResident;
        const auto owner_map = MoEExpertOwnerMap::build(*plan);
        auto overlay_mpi =
            std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/2);
        overlay_mpi->set_topology(
            MockMPITopology::createSimple(
                /*rank=*/0,
                /*world_size=*/2,
                /*ranks_per_node=*/2));
        auto peer_mpi =
            std::make_shared<MockMPIContext>(/*rank=*/1, /*world_size=*/2);
        peer_mpi->set_topology(
            MockMPITopology::createSimple(
                /*rank=*/1,
                /*world_size=*/2,
                /*ranks_per_node=*/2));

        auto controller_topology_value =
            resolveMoEOverlayDeviceControllerTopology(
                *plan,
                owner_map,
                {.world_rank_node_ids = std::array<int, 2>{7, 7}});
        /*
         * Mock MPI run namespaces are stable. Salt this fixture's mapped ABI
         * identity so an interrupted prior process cannot be mistaken for the
         * peer half of the current bilateral first-touch handshake.
         */
        controller_topology_value.topology_fingerprint ^=
            static_cast<std::uint64_t>(::getpid()) << 17u;
        if (controller_topology_value.topology_fingerprint == 0u)
            controller_topology_value.topology_fingerprint = 1u;
        auto controller_topology = std::make_shared<
            const MoEOverlayDeviceControllerTopology>(
            std::move(controller_topology_value));
        const auto initial_layered_ownership = owner_map.layeredOwnership(
            /*num_layers=*/1,
            /*num_experts=*/kNumExperts);
        std::vector<std::uint32_t> initial_owner_participants;
        initial_owner_participants.reserve(kNumExperts);
        for (int expert = 0; expert < kNumExperts; ++expert)
        {
            initial_owner_participants.push_back(
                static_cast<std::uint32_t>(
                    initial_layered_ownership.owner(0, expert)));
        }
        const auto controller_config =
            [&](const std::shared_ptr<MockMPIContext> &mpi)
        {
            return MoEOverlayNodeLocalDeviceControllerFabric::Config{
                .mpi_ctx = mpi,
                .topology = controller_topology,
                .num_layers = 1u,
                .num_experts = static_cast<std::uint32_t>(kNumExperts),
                .command_capacity = 4u,
                .initial_durable_epoch = 1u,
                .payload_bytes_per_layer = {4096u},
                .initial_owner_participants =
                    initial_owner_participants,
                .minimum_window_activations = 1u,
                .maximum_cycles_per_wave = 1u,
            };
        };
        std::shared_ptr<MoEOverlayNodeLocalDeviceControllerFabric>
            root_controller_fabric;
        std::shared_ptr<MoEOverlayNodeLocalDeviceControllerFabric>
            peer_controller_fabric;
        std::exception_ptr root_controller_error;
        std::exception_ptr peer_controller_error;
        std::jthread root_controller_builder(
            [&]
            {
                try
                {
                    root_controller_fabric = std::make_shared<
                        MoEOverlayNodeLocalDeviceControllerFabric>(
                        controller_config(overlay_mpi));
                }
                catch (...)
                {
                    root_controller_error = std::current_exception();
                }
            });
        std::jthread peer_controller_builder(
            [&]
            {
                try
                {
                    peer_controller_fabric = std::make_shared<
                        MoEOverlayNodeLocalDeviceControllerFabric>(
                        controller_config(peer_mpi));
                }
                catch (...)
                {
                    peer_controller_error = std::current_exception();
                }
            });
        root_controller_builder.join();
        peer_controller_builder.join();
        if (root_controller_error)
            std::rethrow_exception(root_controller_error);
        if (peer_controller_error)
            std::rethrow_exception(peer_controller_error);
        ASSERT_NE(root_controller_fabric, nullptr);
        ASSERT_NE(peer_controller_fabric, nullptr);

        auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(
            plan,
            MoEExpertOverlayRuntimeResolverOptions{
                .current_world_rank = 0,
            });

        GraphConfig config = makeConfig(plan, /*layer_count=*/1);
        config.max_activation_rows = kSeqLen;
        config.default_device = DeviceId::cuda(0);
        config.moe.overlay_mpi_ctx = overlay_mpi;
        config.moe.expert_overlay_runtime_plan = runtime_plan;
        config.moe.rebalance_config.mode =
            MoERebalanceRuntimeMode::Dynamic;
        config.moe.durable_residency_authority =
            MoEDurableResidencyAuthorityKind::ExpertOverlayRCU;
        config.moe.authority_execution =
            MoEOverlayAuthorityExecutionKind::DeviceResident;
        config.moe.device_controller_fabric = root_controller_fabric;

        DecodeExpertHistogramConfig histogram_config;
        histogram_config.num_layers = 1;
        histogram_config.num_experts = kNumExperts;
        histogram_config.top_k = kTopK;
        histogram_config.window_size = 64;
        histogram_config.token_boundary_layer_idx = 0;
        histogram_config.sockets = {
            DeviceId::cuda(0), DeviceId::rocm(0)};
        histogram_config.ownership =
            owner_map.layeredOwnership(/*num_layers=*/1, kNumExperts);
        auto histogram = std::make_shared<DecodeExpertHistogram>(
            std::move(histogram_config));
        auto residency_authority = std::make_shared<
            MoEOverlayResidencyAuthority>(
            MoEOverlayResidencyAuthority::Config{
                .initial_plan = *plan,
                .model_metadata = {
                    .num_layers = 1,
                    .num_experts = kNumExperts,
                    .d_model = kDModel,
                    .routed_intermediate_size = kIntermediate,
                },
                .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
                .histogram = histogram.get(),
                .perf_device = "cuda_rocm_dynamic_lowering",
            });
        const auto initial_snapshot = residency_authority->snapshot();
        ASSERT_NE(initial_snapshot, nullptr);
        ASSERT_TRUE(initial_snapshot->valid());
        auto participant_residency = std::make_shared<
            MoEOverlayParticipantResidencyRegistry>(
            MoEOverlayParticipantResidencyRegistry::Config{
                .owner_map = initial_snapshot->owner_map,
                .local_participant_ids = {0},
                .num_layers = 1,
                .num_experts = kNumExperts,
                .initial_epoch = initial_snapshot->epoch,
            });
        config.moe.decode_histogram = histogram.get();
        config.moe.expert_overlay_residency_authority =
            residency_authority;
        config.moe.expert_overlay_participant_residency =
            participant_residency;

        auto model_ctx = ModelContext::createForTesting(
            "test.gguf", nullptr, 1, /*with_weight_manager=*/true);
        ASSERT_NE(model_ctx, nullptr);
        ASSERT_NE(model_ctx->concreteWeightManager(), nullptr);
        registerCompleteDomainExpertLayer(
            model_ctx->concreteWeightManager()->expertGemmRegistry(),
            "continuation_domain",
            DeviceId::cuda(0),
            /*layer_idx=*/0);
        registerOwnedParticipantExpertLayers(
            model_ctx->concreteWeightManager()->expertGemmRegistry(),
            owner_map,
            /*layer_count=*/1);

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        TensorArena decode_arena;
        auto decode_buffers = makeActivationBuffers(decode_arena);
        TensorArena prefill_arena;
        auto prefill_buffers = makeActivationBuffers(prefill_arena);
        ScopedDevicePublicationStream stream(DeviceId::cuda(0));
        const int32_t *const prefill_live_rows =
            stream.publishRowCount(kSeqLen);

        /*
         * The mapped rank-pair channel has a directional first-touch setup
         * handshake. Production runs the follower on world rank one; this
         * direct graph-builder test creates that setup peer concurrently so
         * it exercises the same retained channel instead of timing out while
         * waiting for a rank that the fixture never launched.
         */
        const auto graph_family =
            resolveMoEOverlayInferenceGraphFamilyIdentity(
                model_ctx->concreteLoader(),
                model_ctx->architecture(),
                model_ctx->totalBlockCount(),
                MoEOverlayMTPGraphFamilyPolicy::MainOnly,
                /*graph_family_generation=*/1,
                /*max_graph_rows=*/kSeqLen,
                /*max_decode_rows=*/1,
                config.max_request_count,
                /*max_mtp_draft_depth=*/0);
        const auto transaction_topology =
            makeMoEOverlayInferenceTopologyIdentity(
                owner_map,
                graph_family,
                /*source_world_rank=*/0,
                /*target_world_rank=*/1);
        auto peer_workspace =
            std::make_shared<MoEOverlayRankBatchWireWorkspace>(
                MoEOverlayRankBatchWireWorkspace::Config{
                    .participant_ids = {1},
                    .max_total_rows =
                        static_cast<std::size_t>(kSeqLen * kTopK),
                    .max_total_entries =
                        static_cast<std::size_t>(kSeqLen * kTopK),
                    .d_model = kDModel,
                    .top_k = kTopK,
                });
        std::shared_ptr<IMoEOverlayRankBatchTransport> peer_transport;
        std::exception_ptr peer_error;
        std::jthread peer_builder(
            [&]
            {
                try
                {
                    peer_transport = createMoEOverlayRankBatchTransport(
                        MoEOverlayRankBatchTransportConfig{
                            .mpi_ctx = peer_mpi,
                            .source_world_rank = 0,
                            .target_world_rank = 1,
                            .workspace = peer_workspace,
                            .max_rows_per_participant = kSeqLen,
                            .max_entries_per_participant =
                                static_cast<std::size_t>(kSeqLen * kTopK),
                            .d_model = kDModel,
                            .top_k = kTopK,
                            .tier_index = 1,
                            .domain_ordinal = 1,
                            .channel_identity =
                                "tier1#domain1#rank0to1#p1,",
                            .transaction_slot_count = 4096,
                            .transaction_topology = transaction_topology,
                            .source_endpoint = {
                                .world_rank = 0,
                                .participant_id = 0,
                                .tier_priority = 0,
                                .domain_ordinal = 0,
                            },
                            .target_tier_priority = 1,
                            .activation_graph_families =
                                makeMoEOverlayActivationGraphFamilyManifests(
                                    graph_family),
                            .local_lanes = {{
                                .participant_id = 1,
                                .device = DeviceId::cpu(),
                            }},
                        });
                }
                catch (...)
                {
                    peer_error = std::current_exception();
                }
            });
        auto source_workspace =
            std::make_shared<MoEOverlayRankBatchWireWorkspace>(
                MoEOverlayRankBatchWireWorkspace::Config{
                    .participant_ids = {1},
                    .max_total_rows =
                        static_cast<std::size_t>(kSeqLen * kTopK),
                    .max_total_entries =
                        static_cast<std::size_t>(kSeqLen * kTopK),
                    .d_model = kDModel,
                    .top_k = kTopK,
                });
        auto source_transport = createMoEOverlayRankBatchTransport(
            MoEOverlayRankBatchTransportConfig{
                .mpi_ctx = overlay_mpi,
                .source_world_rank = 0,
                .target_world_rank = 1,
                .workspace = std::move(source_workspace),
                .max_rows_per_participant = kSeqLen,
                .max_entries_per_participant =
                    static_cast<std::size_t>(kSeqLen * kTopK),
                .d_model = kDModel,
                .top_k = kTopK,
                .tier_index = 1,
                .domain_ordinal = 1,
                .channel_identity = "tier1#domain1#rank0to1#p1,",
                .transaction_slot_count = 4096,
                .transaction_topology = transaction_topology,
                .source_endpoint = {
                    .world_rank = 0,
                    .participant_id = 0,
                    .tier_priority = 0,
                    .domain_ordinal = 0,
                },
                .target_tier_priority = 1,
                .activation_graph_families =
                    makeMoEOverlayActivationGraphFamilyManifests(
                        graph_family),
                .local_lanes = {{
                    .participant_id = 1,
                    .device = DeviceId::cuda(0),
                }},
            });
        peer_builder.join();
        if (peer_error)
            std::rethrow_exception(peer_error);
        ASSERT_NE(peer_transport, nullptr);
        auto channel_registry =
            std::make_shared<MoEOverlayRankBatchTransportRegistry>();
        channel_registry->install(
            "tier1#domain1#rank0to1#p1,",
            std::move(source_transport));
        config.moe.rank_batch_transport_registry =
            std::move(channel_registry);
        Qwen35MoEGraph builder(model_ctx, overlay_mpi, config);

        ComputeGraph decode_graph = builder.buildFFNGraph(
            layer,
            decode_buffers,
            /*layer_idx=*/0,
            /*seq_len=*/1,
            kBatchSize,
            DeviceId::cuda(0),
            stream.get());
        ComputeGraph prefill_graph = builder.buildFFNGraph(
            layer,
            prefill_buffers,
            /*layer_idx=*/0,
            kSeqLen,
            kBatchSize,
            DeviceId::cuda(0),
            stream.get(),
            prefill_live_rows);
        const int32_t *const one_row_prefill_live_rows =
            stream.publishRowCount(1);
        ComputeGraph one_row_prefill_graph = builder.buildFFNGraph(
            layer,
            decode_buffers,
            /*layer_idx=*/0,
            /*seq_len=*/1,
            kBatchSize,
            DeviceId::cuda(0),
            stream.get(),
            one_row_prefill_live_rows);

        const std::string local_name =
            "layer0_moe_expert_ffn_overlay_continuation_local";
        const auto *decode_local = expertComputeStage(
            decode_graph, local_name);
        ASSERT_NE(decode_local, nullptr);
        ASSERT_NE(decode_local->moeRuntimeTableForTesting(), nullptr);
        EXPECT_EQ(
            decode_local->runtimeDecodeDescriptorSourceForTesting(),
            MoEDecodeDescriptorSource::RuntimePlacementTable)
            << "A heterogeneous Dynamic continuation must read the exact "
               "descriptor bank selected by its admitted request epoch; the "
               "capture-time owner mask cannot expose promoted experts";
        const auto *prefill_local = expertComputeStage(
            prefill_graph, local_name);
        ASSERT_NE(prefill_local, nullptr);
        EXPECT_TRUE(prefill_local->usesRuntimeRowGroupingForTesting())
            << "Dynamic captured continuation prefill must group rows from "
               "the live placement bank; a construction-time expert mask "
               "would keep the retained graph on stale owners after an epoch";
        ASSERT_NE(prefill_local->moeRuntimeTableForTesting(), nullptr);
        EXPECT_EQ(
            prefill_local->moeRuntimeTableForTesting(),
            decode_local->moeRuntimeTableForTesting())
            << "Prefill and decode must consume one durable placement "
               "authority so promotion or skew rebalance becomes visible to "
               "both retained graph families at the same admitted epoch";
        const auto *const runtime_table = dynamic_cast<
            const DeviceMoERuntimeTable *>(
            decode_local->moeRuntimeTableForTesting());
        ASSERT_NE(runtime_table, nullptr);
        ASSERT_NE(runtime_table->overlayEpochArena(), nullptr);
        const auto root_controller_binding =
            root_controller_fabric->participantBinding(0);
        ASSERT_TRUE(root_controller_binding.valid());
        ASSERT_NE(root_controller_binding.controller, nullptr);
        EXPECT_EQ(
            runtime_table->overlayEpochArena()->externalAdmissionEpoch(),
            &root_controller_binding.controller->admission_epoch)
            << "Dynamic continuation capture must select only the globally "
               "admitted mapped epoch; a participant-local host selector is "
               "not a valid substitute";
        const auto admission_barrier =
            runtime_table->overlayEpochArena()->admissionBarrier();
        ASSERT_TRUE(admission_barrier.valid());
        EXPECT_EQ(
            admission_barrier.record,
            root_controller_binding.inference_epoch_record)
            << "Every continuation graph must embed the one topology-wide "
               "transaction epoch record rather than a participant-local copy";
        EXPECT_EQ(
            admission_barrier.participant_id,
            static_cast<std::uint32_t>(
                root_controller_binding.participant_id));
        const auto &runtime_state = runtime_table->hostLayerState(0);
        ASSERT_LE(runtime_state.active_bank, 1u);
        EXPECT_EQ(runtime_state.participant_count, 1u);
        const auto &runtime_bank =
            runtime_state.banks[runtime_state.active_bank];
        for (int expert = 0; expert < kNumExperts; ++expert)
        {
            const auto *expected_owner = owner_map.ownerFor(0, expert);
            ASSERT_NE(expected_owner, nullptr);
            EXPECT_EQ(
                runtime_bank.overlay_route_participant[
                    static_cast<size_t>(expert)],
                expected_owner->owner_participant);
        }
        EXPECT_TRUE(
            runtime_table->overlayRoutePlacementBinding(0).valid());

        const auto *one_row_prefill_local = expertComputeStage(
            one_row_prefill_graph, local_name);
        ASSERT_NE(one_row_prefill_local, nullptr);
        EXPECT_TRUE(
            one_row_prefill_local
                ->supportsLazyPrefillGraphCapturePreflight())
            << "A restored one-row heterogeneous prefill must retain its "
               "captured explicit-routing continuation graph";
        EXPECT_TRUE(
            one_row_prefill_local
                ->supportsPaddedPrefillGraphCapturePreflight())
            << "The exact device live-row pointer is the typed distinction "
               "between this prefill bucket and ordinary one-row decode";

        for (const auto *graph : {&decode_graph, &prefill_graph})
        {
            EXPECT_EQ(
                countStagesOfType(*graph, ComputeStageType::MOE_EXPERT_FFN),
                1u);
            EXPECT_EQ(
                countStagesOfType(*graph, ComputeStageType::MOE_LOCAL_EXPERT),
                0u);
            EXPECT_EQ(
                countStagesOfType(
                    *graph, ComputeStageType::MOE_EXPERT_DISPATCH),
                0u);
            EXPECT_EQ(
                countStagesOfType(
                    *graph, ComputeStageType::MOE_SPARSE_DISPATCH),
                0u);
            EXPECT_EQ(
                countStagesOfType(
                    *graph, ComputeStageType::MOE_RANK_BATCH_DISPATCH),
                0u);
            EXPECT_EQ(
                countStagesOfType(
                    *graph, ComputeStageType::MOE_RANK_BATCH_RETURN_REDUCE),
                0u);
            EXPECT_EQ(
                countStagesOfType(
                    *graph,
                    ComputeStageType::MOE_OVERLAY_ACTIVATION_DISPATCH_PACK),
                1u);
            EXPECT_EQ(
                countStagesOfType(
                    *graph,
                    ComputeStageType::MOE_OVERLAY_ACTIVATION_RETURN_CONSUME),
                1u);

            const auto dispatch_names = stageNamesOfType(
                *graph,
                ComputeStageType::MOE_OVERLAY_ACTIVATION_DISPATCH_PACK);
            const auto return_names = stageNamesOfType(
                *graph,
                ComputeStageType::MOE_OVERLAY_ACTIVATION_RETURN_CONSUME);
            ASSERT_EQ(dispatch_names.size(), 1u);
            ASSERT_EQ(return_names.size(), 1u);
            EXPECT_TRUE(hasDependency(
                *graph, local_name, dispatch_names.front()));
            EXPECT_TRUE(hasDependency(
                *graph,
                "layer0_moe_overlay_continuation_routes_ordered_reduce",
                return_names.front()))
                << "The sole ordered reducer must wait until remote route "
                   "slots have been materialized";

            const auto *dispatch_node =
                graph->getNode(dispatch_names.front());
            ASSERT_NE(dispatch_node, nullptr);
            const auto *return_node =
                graph->getNode(return_names.front());
            ASSERT_NE(return_node, nullptr);
            const auto *ordered_reduce_node = graph->getNode(
                "layer0_moe_overlay_continuation_routes_ordered_reduce");
            ASSERT_NE(ordered_reduce_node, nullptr);
            const auto *ordered_reduce = dynamic_cast<
                const MoECanonicalRouteReduceStage *>(
                ordered_reduce_node->stage.get());
            ASSERT_NE(ordered_reduce, nullptr);
            const auto expected_route_slots = static_cast<std::uint32_t>(
                (graph == &decode_graph ? 1 : kSeqLen) * kTopK);
            EXPECT_TRUE(
                ordered_reduce->params()
                    .domain_route_assignment.validFor(expected_route_slots));
            EXPECT_TRUE(
                ordered_reduce->params().overlay_route_placement.valid());

            if (graph == &decode_graph)
            {
                const auto *dispatch_stage = dynamic_cast<
                    const MoEOverlayActivationDispatchPackStage *>(
                    dispatch_node->stage.get());
                const auto *return_stage = dynamic_cast<
                    const MoEOverlayActivationReturnConsumeStage *>(
                    return_node->stage.get());
                ASSERT_NE(dispatch_stage, nullptr);
                ASSERT_NE(return_stage, nullptr);
                EXPECT_EQ(
                    dispatch_stage->getParams().lane
                        .target_participant_id,
                    1);
                EXPECT_TRUE(
                    dispatch_stage->getParams().placement.valid());
                EXPECT_TRUE(
                    return_stage->getParams().lane.returned.valid());
                EXPECT_EQ(
                    return_stage->getParams()
                        .canonical_route_contributions,
                    ordered_reduce->params()
                        .canonical_route_contributions)
                    << "Return consumption only materializes canonical route "
                       "slots; the ordered reducer owns assignment and fold "
                       "authority";
                EXPECT_EQ(
                    dispatch_stage->getParams().active_row_count_device,
                    nullptr)
                    << "Decode owns one activation row; the sequence-length "
                       "scalar is the growing KV position, not a row count";
                EXPECT_TRUE(
                    dispatch_stage
                        ->supportsLazyPrefillGraphCapturePreflight());
                EXPECT_FALSE(
                    dispatch_stage
                        ->supportsPaddedPrefillGraphCapturePreflight())
                    << "One-row decode has no device live-row binding";
                EXPECT_TRUE(
                    return_stage
                        ->supportsLazyPrefillGraphCapturePreflight());
                EXPECT_TRUE(
                    return_stage
                        ->supportsPaddedPrefillGraphCapturePreflight());
                EXPECT_TRUE(
                    return_stage
                        ->supportsPaddedPrefillRealLengthContract());
            }
            else
            {
                const auto *dispatch_stage = dynamic_cast<
                    const MoEOverlayActivationDispatchPackBatchStage *>(
                    dispatch_node->stage.get());
                const auto *return_stage = dynamic_cast<
                    const MoEOverlayActivationReturnConsumeBatchStage *>(
                    return_node->stage.get());
                ASSERT_NE(dispatch_stage, nullptr);
                ASSERT_NE(return_stage, nullptr);
                const auto &transaction =
                    dispatch_stage->getParams().transaction;
                ASSERT_NE(transaction, nullptr);
                ASSERT_EQ(transaction->lanes().size(), 1u);
                EXPECT_EQ(
                    transaction->lanes().front().target_participant_id,
                    1);
                EXPECT_EQ(transaction->physicalRows(), kSeqLen);
                EXPECT_EQ(
                    return_stage->getParams().transaction,
                    transaction)
                    << "Prefill fork and join must retain one transaction";
                EXPECT_TRUE(
                    dispatch_stage->getParams().placement.valid());
                EXPECT_TRUE(
                    dispatch_stage
                        ->supportsLazyPrefillGraphCapturePreflight());
                EXPECT_TRUE(
                    dispatch_stage
                        ->supportsPaddedPrefillGraphCapturePreflight());
                EXPECT_TRUE(
                    dispatch_stage
                        ->supportsPaddedPrefillRealLengthContract());
                EXPECT_TRUE(
                    return_stage
                        ->supportsLazyPrefillGraphCapturePreflight());
                EXPECT_TRUE(
                    return_stage
                        ->supportsPaddedPrefillGraphCapturePreflight());
                EXPECT_TRUE(
                    return_stage
                        ->supportsPaddedPrefillRealLengthContract());
            }
            EXPECT_EQ(
                graph->nativeCaptureEnvelope(),
                GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
            EXPECT_FALSE(
                makeMoEOverlayRetainedParentPlan(*graph).has_value())
                << "Conditional attention and mapped packet edges must live "
                   "in one directly captured endpoint graph, never in CUDA "
                   "child graphs";
        }
    }

    /**
     * @brief A multi-device continuation has one MPI root and captured local experts.
     *
     * The two device graphs share one MPI rank, so allowing both to submit the
     * cross-rank sparse protocol is intrinsically invalid. This production
     * lowering proof requires each graph to execute only its own prepared
     * continuation experts, the logical root alone to exchange remote rows,
     * and both graphs to rejoin through the same rooted LocalTP publication.
     */
    /**
     * Prove one distributed LocalTP continuation under an explicit transport.
     * @param route_transport Native collective or mapped sparse publication.
     */
    void assertDistributedLocalTPContinuationCapture(
        MoEOverlayNodeLocalRouteTransport route_transport)
    {
        constexpr int kMTPSourceLayer = 1;
        auto plan = makeDistributedLocalTPContinuationPlan(
            /*layer_count=*/kMTPSourceLayer + 1);
        auto overlay_mpi =
            std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/2);
        overlay_mpi->set_topology(
            MockMPITopology::createSimple(
                /*rank=*/0,
                /*world_size=*/2,
                /*ranks_per_node=*/2));
        auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(
            plan,
            MoEExpertOverlayRuntimeResolverOptions{
                .current_world_rank = 0,
            });
        const auto owner_map = MoEExpertOwnerMap::build(*plan);

        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices(
            {GlobalDeviceAddress::rocm(0),
             GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);

        GraphConfig config0 = makeConfig(plan, /*layer_count=*/1);
        config0.max_activation_rows = kSeqLen;
        config0.default_device = DeviceId::rocm(0);
        config0.tp_ctx = &tp_ctx;
        config0.tp_device_idx = 0;
        config0.moe.overlay_mpi_ctx = overlay_mpi;
        config0.moe.expert_overlay_runtime_plan = runtime_plan;
        config0.moe.has_shared_expert = true;
        config0.moe.shared_intermediate_size = kIntermediate;

        config0.moe.node_local_route_transport = route_transport;
        if (route_transport ==
            MoEOverlayNodeLocalRouteTransport::MappedSparse)
        {
            /*
             * RankOrchestrator normally creates this process-local owner
             * before either participant graph. The direct builder fixture
             * installs the same shared lifetime explicitly; one fabric per
             * graph would split mapped epochs and acknowledgement authority.
             */
            config0.moe.node_local_route_exchange =
                std::make_shared<MoEOverlayNodeLocalRouteExchange>(
                    MoEOverlayNodeLocalRouteExchange::Config{
                        .devices = {
                            DeviceId::rocm(0), DeviceId::rocm(1)},
                        .root_device = DeviceId::rocm(0),
                        .identity = "continuation:continuation_domain",
                    });
        }

        GraphConfig config1 = config0;
        config1.default_device = DeviceId::rocm(1);
        config1.tp_device_idx = 1;

        auto model_ctx = ModelContext::createForTesting(
            "test.gguf", nullptr, 1, /*with_weight_manager=*/true);
        ASSERT_NE(model_ctx, nullptr);
        ASSERT_NE(model_ctx->concreteWeightManager(), nullptr);
        auto &registry =
            model_ctx->concreteWeightManager()->expertGemmRegistry();
        for (int layer_index = 0;
             layer_index <= kMTPSourceLayer;
             ++layer_index)
        {
            registerCompleteDomainExpertLayer(
                registry,
                "continuation_domain",
                DeviceId::rocm(0),
                layer_index);
            registerCompleteDomainExpertLayer(
                registry,
                "continuation_domain",
                DeviceId::rocm(1),
                layer_index);
        }

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        layer.shared_expert_gate =
            weight_arena.fp32({kIntermediate, kDModel});
        layer.shared_expert_up =
            weight_arena.fp32({kIntermediate, kDModel});
        layer.shared_expert_down =
            weight_arena.fp32({kDModel, kIntermediate});
        layer.shared_expert_gate_inp =
            weight_arena.fp32({1, kDModel});
        const MTPDepthWeights mtp_weights = makeMTPSidecarWeights(
            weight_arena, layer, kMTPSourceLayer);
        ModelWeights model_weights;
        model_weights.embedding_table =
            weight_arena.fp32({32u, static_cast<size_t>(kDModel)});
        model_weights.lm_head =
            weight_arena.fp32({32u, static_cast<size_t>(kDModel)});
        TensorArena activation_arena0;
        auto buffers0 = makeActivationBuffers(activation_arena0);
        MTPForwardOutput mtp_output0 =
            makeMTPSidecarOutput(activation_arena0);
        TensorArena activation_arena1;
        auto buffers1 = makeActivationBuffers(activation_arena1);
        MTPForwardOutput mtp_output1 =
            makeMTPSidecarOutput(activation_arena1);
        ScopedDevicePublicationStream stream0(DeviceId::rocm(0));
        ScopedDevicePublicationStream stream1(DeviceId::rocm(1));

        /*
         * Channel establishment is a two-rank setup protocol: each consumer
         * first-touches the payload direction that it will read before either
         * endpoint pins the complete shared mapping. This graph-lowering test
         * runs both logical ranks in one process, so model the remote rank's
         * setup half concurrently. The peer uses a CPU alias because this test
         * inspects continuation graph topology only; the dedicated CUDA/ROCm
         * packet integration suite executes both real GPU endpoints and checks
         * every returned byte.
         */
        auto peer_mpi =
            std::make_shared<MockMPIContext>(/*rank=*/1, /*world_size=*/2);
        peer_mpi->set_topology(
            MockMPITopology::createSimple(
                /*rank=*/1,
                /*world_size=*/2,
                /*ranks_per_node=*/2));
        /* The test loader has no GGUF metadata to discover. Install the exact
         * validated family that production derives from the model manifest so
         * both mapped endpoints retain main and MTP activation slots. */
        const MoEOverlayInferenceGraphFamilyIdentity graph_family{
            .graph_family_generation = 1,
            .main_layer_count = 1,
            .mtp_source_layers = {kMTPSourceLayer},
            .max_graph_rows = kSeqLen,
            .max_decode_rows = 1,
            .max_request_count = config0.max_request_count,
            .max_mtp_draft_depth = 1,
        };
        ASSERT_TRUE(graph_family.valid());
        const auto transaction_topology =
            makeMoEOverlayInferenceTopologyIdentity(
                owner_map,
                graph_family,
                /*source_world_rank=*/0,
                /*target_world_rank=*/1);
        const std::vector<int> remote_participants{2, 3, 4, 5};
        auto peer_workspace =
            std::make_shared<MoEOverlayRankBatchWireWorkspace>(
                MoEOverlayRankBatchWireWorkspace::Config{
                    .participant_ids = remote_participants,
                    .max_total_rows =
                        static_cast<std::size_t>(kSeqLen * kTopK),
                    .max_total_entries =
                        static_cast<std::size_t>(kSeqLen * kTopK),
                    .d_model = kDModel,
                    .top_k = kTopK,
                });
        std::shared_ptr<IMoEOverlayRankBatchTransport> peer_transport;
        std::exception_ptr peer_error;
        std::jthread peer_builder(
            [&]
            {
                try
                {
                    peer_transport = createMoEOverlayRankBatchTransport(
                        MoEOverlayRankBatchTransportConfig{
                            .mpi_ctx = peer_mpi,
                            .source_world_rank = 0,
                            .target_world_rank = 1,
                            .workspace = peer_workspace,
                            .max_rows_per_participant = kSeqLen,
                            .max_entries_per_participant =
                                static_cast<std::size_t>(kSeqLen * kTopK),
                            .d_model = kDModel,
                            .top_k = kTopK,
                            .tier_index = 1,
                            .domain_ordinal = 1,
                            .channel_identity =
                                "tier1#domain1#rank0to1#p2,3,4,5,",
                            .transaction_slot_count = 4096,
                            .transaction_topology = transaction_topology,
                            .source_endpoint = {
                                .world_rank = 0,
                                .participant_id = 0,
                                .tier_priority = 0,
                                .domain_ordinal = 0,
                            },
                            .target_tier_priority = 1,
                            .activation_graph_families =
                                makeMoEOverlayActivationGraphFamilyManifests(
                                    graph_family),
                            .local_lanes = {
                                {.participant_id = 2,
                                 .device = DeviceId::cpu()},
                                {.participant_id = 3,
                                 .device = DeviceId::cpu()},
                                {.participant_id = 4,
                                 .device = DeviceId::cpu()},
                                {.participant_id = 5,
                                 .device = DeviceId::cpu()},
                            },
                        });
                }
                catch (...)
                {
                    peer_error = std::current_exception();
                }
            });

        auto source_workspace =
            std::make_shared<MoEOverlayRankBatchWireWorkspace>(
                MoEOverlayRankBatchWireWorkspace::Config{
                    .participant_ids = remote_participants,
                    .max_total_rows =
                        static_cast<std::size_t>(kSeqLen * kTopK),
                    .max_total_entries =
                        static_cast<std::size_t>(kSeqLen * kTopK),
                    .d_model = kDModel,
                    .top_k = kTopK,
                });
        auto source_transport = createMoEOverlayRankBatchTransport(
            MoEOverlayRankBatchTransportConfig{
                .mpi_ctx = overlay_mpi,
                .source_world_rank = 0,
                .target_world_rank = 1,
                .workspace = std::move(source_workspace),
                .max_rows_per_participant = kSeqLen,
                .max_entries_per_participant =
                    static_cast<std::size_t>(kSeqLen * kTopK),
                .d_model = kDModel,
                .top_k = kTopK,
                .tier_index = 1,
                .domain_ordinal = 1,
                .channel_identity =
                    "tier1#domain1#rank0to1#p2,3,4,5,",
                .transaction_slot_count = 4096,
                .transaction_topology = transaction_topology,
                .source_endpoint = {
                    .world_rank = 0,
                    .participant_id = 0,
                    .tier_priority = 0,
                    .domain_ordinal = 0,
                },
                .target_tier_priority = 1,
                .activation_graph_families =
                    makeMoEOverlayActivationGraphFamilyManifests(
                        graph_family),
                .local_lanes = {
                    {.participant_id = 2,
                     .device = DeviceId::rocm(0)},
                    {.participant_id = 3,
                     .device = DeviceId::rocm(0)},
                    {.participant_id = 4,
                     .device = DeviceId::rocm(0)},
                    {.participant_id = 5,
                     .device = DeviceId::rocm(0)},
                },
            });
        peer_builder.join();
        if (peer_error)
            std::rethrow_exception(peer_error);
        ASSERT_NE(peer_transport, nullptr);
        auto channel_registry =
            std::make_shared<MoEOverlayRankBatchTransportRegistry>();
        channel_registry->install(
            "tier1#domain1#rank0to1#p2,3,4,5,",
            std::move(source_transport));
        config0.moe.rank_batch_transport_registry = channel_registry;
        config1.moe.rank_batch_transport_registry =
            std::move(channel_registry);

        const std::string local_name =
            "layer0_moe_expert_ffn_overlay_continuation_local";

        Qwen35MoEGraph builder0(model_ctx, overlay_mpi, config0);
        Qwen35MoEGraph builder1(model_ctx, overlay_mpi, config1);
        builder0.setWeights(model_weights);
        builder1.setWeights(model_weights);

        /*
         * Production constructs the one-token decode family before it needs
         * the ordinary-prefill family.  Build in that order so this test
         * cannot accidentally borrow a runtime table materialized by prefill.
         * Each continuation-local branch must receive and initialize its own
         * device table before capture preparation begins.
         */
        ComputeGraph decode_graph0 = builder0.buildFFNGraph(
            layer, buffers0, 0, /*seq_len=*/1, kBatchSize,
            DeviceId::rocm(0), stream0.get());
        ComputeGraph decode_graph1 = builder1.buildFFNGraph(
            layer, buffers1, 0, /*seq_len=*/1, kBatchSize,
            DeviceId::rocm(1), stream1.get());
        for (const auto *decode_graph : {&decode_graph0, &decode_graph1})
        {
            const auto *node = decode_graph->getNode(local_name);
            ASSERT_NE(node, nullptr);
            const auto *stage =
                dynamic_cast<const MoEExpertComputeStage *>(
                    node->stage.get());
            ASSERT_NE(stage, nullptr);
            EXPECT_TRUE(stage->hasMoERuntimeTableForTesting())
                << "A cold-built distributed continuation decode graph must "
                   "own its device runtime table";
            const auto *runtime_table =
                stage->moeRuntimeTableForTesting();
            ASSERT_NE(runtime_table, nullptr);
            ASSERT_FALSE(
                runtime_table->decodeRuntimePublicationRequired(0));
            const auto &runtime_state =
                runtime_table->hostLayerState(0);
            ASSERT_LE(runtime_state.active_bank, 1u);
            EXPECT_EQ(runtime_state.participant_count, 2u);
            const auto &runtime_bank =
                runtime_state.banks[runtime_state.active_bank];
            for (int expert = 0; expert < kNumExperts; ++expert)
            {
                const auto *const overlay_owner =
                    owner_map.ownerFor(/*layer_idx=*/0, expert);
                ASSERT_NE(overlay_owner, nullptr);
                EXPECT_EQ(
                    runtime_bank.overlay_route_participant[
                        static_cast<size_t>(expert)],
                    overlay_owner->owner_participant)
                    << "expert=" << expert
                    << " must retain its overlay-wide packet target even when "
                       "the domain-local runtime descriptor is external";
            }
            const auto placement_binding =
                runtime_table->overlayRoutePlacementBinding(0);
            EXPECT_TRUE(placement_binding.valid())
                << "The captured packet stage must receive both durable banks "
                   "and their exact request ticket as one runtime-table binding";
            EXPECT_EQ(
                placement_binding.expert_count,
                static_cast<uint32_t>(kNumExperts));
            for (int remote_expert = 3;
                 remote_expert < kNumExperts;
                 ++remote_expert)
            {
                const size_t expert_index =
                    static_cast<size_t>(remote_expert);
                EXPECT_EQ(
                    runtime_bank.experts[expert_index]
                        .owner_participant,
                    -1);
                EXPECT_EQ(
                    runtime_bank.resident_participant_mask[
                        expert_index],
                    0u);
                EXPECT_EQ(
                    runtime_bank.local_compute_mask[expert_index],
                    0u);
            }
            EXPECT_TRUE(
                stage->supportsGraphCaptureAfterLaunchPreparation())
                << stage->graphCaptureReadinessDebugString();
            EXPECT_EQ(
                stage->graphLaunchPreparationPolicy(),
                GraphLaunchPreparationPolicy::CaptureOnly);
        }

        /*
         * A predictor source layer lives immediately beyond the main-model
         * interval in Qwen3.5. Build that exact public sidecar shape after the
         * ordinary decode table exists, then prove the root publishes both
         * projections of its request-pinned route epoch. This is the fast
         * regression for the real-weight parity assertion: an `MTP0_` numeric
         * checkpoint without these views cannot certify which moved experts
         * actually contributed.
         */
        int draft_token = 1;
        int position_id = 0;
        MTPForwardInput mtp_input0;
        mtp_input0.draft_token_ids = &draft_token;
        mtp_input0.terminal_hidden = mtp_output0.projected;
        mtp_input0.position_ids = &position_id;
        mtp_input0.batch_size = 1;
        mtp_input0.seq_len = 1;
        mtp_input0.device_state_publication_stream = stream0.get();
        mtp_input0.device = DeviceId::rocm(0);
        MTPForwardInput mtp_input1 = mtp_input0;
        mtp_input1.terminal_hidden = mtp_output1.projected;
        mtp_input1.device_state_publication_stream = stream1.get();
        mtp_input1.device = DeviceId::rocm(1);

        ComputeGraph mtp_graph0 = builder0.buildMTPGraph(
            /*depth_idx=*/0, mtp_weights, mtp_input0, mtp_output0);
        ComputeGraph mtp_graph1 = builder1.buildMTPGraph(
            /*depth_idx=*/0, mtp_weights, mtp_input1, mtp_output1);
        const std::string mtp_reduce_name =
            "MTP0_moe_overlay_continuation_routes_ordered_reduce";
        const auto *mtp_reduce0 = dynamic_cast<
            const MoECanonicalRouteReduceStage *>(
            mtp_graph0.getNode(mtp_reduce_name)
                ? mtp_graph0.getNode(mtp_reduce_name)->stage.get()
                : nullptr);
        const auto *mtp_reduce1 = dynamic_cast<
            const MoECanonicalRouteReduceStage *>(
            mtp_graph1.getNode(mtp_reduce_name)
                ? mtp_graph1.getNode(mtp_reduce_name)->stage.get()
                : nullptr);
        ASSERT_NE(mtp_reduce0, nullptr);
        ASSERT_NE(mtp_reduce1, nullptr);
        const StageDumpInfo mtp_root_dump =
            mtp_reduce0->getDumpInfoSnapshot();
        const StageDumpInfo mtp_nonroot_dump =
            mtp_reduce1->getDumpInfoSnapshot();
        const auto has_mtp_route_output = [](
                                              const StageDumpInfo &dump,
                                              std::string_view name)
        {
            return std::any_of(
                dump.outputs.begin(),
                dump.outputs.end(),
                [name](const StageDumpInfo::OutputBuffer &output)
                {
                    return output.name &&
                           std::string_view(output.name) == name;
                });
        };
        for (const std::string_view output_name : {
                 "domain_route_participant_ids",
                 "runtime_route_weights",
                 "overlay_route_participants_bank0",
                 "overlay_route_bank0_epoch",
                 "overlay_route_participants_bank1",
                 "overlay_route_bank1_epoch",
                 "overlay_route_selected_bank",
             })
        {
            EXPECT_TRUE(has_mtp_route_output(mtp_root_dump, output_name))
                << output_name;
            EXPECT_FALSE(
                has_mtp_route_output(mtp_nonroot_dump, output_name))
                << output_name;
        }

        DeviceGraphExecutor::GraphSegmentCache root_capture_plan;
        DeviceGraphExecutor::GraphSegmentCache nonroot_capture_plan;
        const auto root_collectives =
            decode_graph0.collectiveNodeNames();
        const auto nonroot_collectives =
            decode_graph1.collectiveNodeNames();
        DeviceGraphCaptureController::buildCapturePlan(
            decode_graph0,
            root_capture_plan,
            &root_collectives,
            !root_collectives.empty(),
            /*collectives_graph_capturable=*/true,
            DeviceGraphExecutor::GraphReplayPlanPolicy::
                RequireFullGraph);
        DeviceGraphCaptureController::buildCapturePlan(
            decode_graph1,
            nonroot_capture_plan,
            &nonroot_collectives,
            !nonroot_collectives.empty(),
            /*collectives_graph_capturable=*/true,
            DeviceGraphExecutor::GraphReplayPlanPolicy::
                RequireFullGraph);

        EXPECT_EQ(root_capture_plan.segments.size(), 1u);
        EXPECT_EQ(nonroot_capture_plan.segments.size(), 1u);
        ASSERT_FALSE(root_capture_plan.segments.empty());
        ASSERT_FALSE(nonroot_capture_plan.segments.empty());
        EXPECT_TRUE(root_capture_plan.segments.front().capturable);
        EXPECT_TRUE(nonroot_capture_plan.segments.front().capturable);

        EXPECT_EQ(
            root_capture_plan.graph_replay_plan_policy,
            DeviceGraphExecutor::GraphReplayPlanPolicy::
                RequireFullGraph);
        EXPECT_EQ(
            nonroot_capture_plan.graph_replay_plan_policy,
            DeviceGraphExecutor::GraphReplayPlanPolicy::
                RequireFullGraph);
        EXPECT_EQ(
            decode_graph0.nativeCaptureEnvelope(),
            GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
        EXPECT_EQ(
            decode_graph1.nativeCaptureEnvelope(),
            GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
        EXPECT_FALSE(
            makeMoEOverlayRetainedParentPlan(decode_graph0).has_value())
            << "The continuation root must capture its mapped timeline edges "
               "inside the complete endpoint graph";
        EXPECT_FALSE(
            makeMoEOverlayRetainedParentPlan(decode_graph1).has_value())
            << "A continuation sibling likewise owns one complete endpoint graph";

        size_t root_dispatch_units = 0u;
        size_t root_return_units = 0u;
        for (const auto &segment : root_capture_plan.segments)
        {
            EXPECT_TRUE(segment.capturable);
            EXPECT_TRUE(segment.retained_parent_input_ids.empty())
                << "Direct full-graph capture has no child-frontier imports";
            for (const auto &stage_name : segment.stage_names)
            {
                const auto *node = decode_graph0.getNode(stage_name);
                ASSERT_NE(node, nullptr);
                if (node->stage->type() ==
                    ComputeStageType::MOE_OVERLAY_ACTIVATION_DISPATCH_PACK)
                {
                    ++root_dispatch_units;
                }
                if (node->stage->type() ==
                    ComputeStageType::MOE_OVERLAY_ACTIVATION_RETURN_CONSUME)
                {
                    ++root_return_units;
                }
            }
        }
        EXPECT_EQ(root_dispatch_units, 1u)
            << "One-row decode must publish every remote lane from one "
               "topology-sized kernel";
        EXPECT_EQ(root_return_units, 1u)
            << "One-row decode must join every remote lane concurrently before "
               "its deterministic planner-order fold";

        const auto *dispatch_batch_node = decode_graph0.getNode(
            "layer0_moe_overlay_activation_dispatch_pack_batch");
        ASSERT_NE(dispatch_batch_node, nullptr);
        const auto *dispatch_batch = dynamic_cast<
            const MoEOverlayActivationDispatchPackBatchStage *>(
            dispatch_batch_node->stage.get());
        ASSERT_NE(dispatch_batch, nullptr);
        const auto &dispatch_transaction =
            dispatch_batch->getParams().transaction;
        ASSERT_NE(dispatch_transaction, nullptr);
        ASSERT_EQ(dispatch_transaction->lanes().size(), 4u);
        EXPECT_EQ(
            dispatch_batch->getParams().device_id,
            DeviceId::rocm(0));
        EXPECT_EQ(
            dispatch_transaction->stageOrdinal(),
            0u);
        EXPECT_EQ(
            dispatch_transaction->modelLayerIndex(),
            0);
        EXPECT_TRUE(dispatch_batch->getParams().placement.valid());
        EXPECT_EQ(
            dispatch_batch->getParams().active_row_count_device,
            nullptr)
            << "A one-row multi-lane decode must not interpret the growing "
               "sequence length as sparse-packet live rows";
        EXPECT_EQ(
            dispatch_batch->graphLaunchPreparationPolicy(),
            GraphLaunchPreparationPolicy::CaptureOnly);

        const auto *return_batch_node = decode_graph0.getNode(
            "layer0_moe_overlay_activation_return_consume_batch");
        ASSERT_NE(return_batch_node, nullptr);
        const auto *return_batch = dynamic_cast<
            const MoEOverlayActivationReturnConsumeBatchStage *>(
            return_batch_node->stage.get());
        ASSERT_NE(return_batch, nullptr);
        const auto &return_transaction =
            return_batch->getParams().transaction;
        ASSERT_NE(return_transaction, nullptr);
        EXPECT_EQ(return_transaction, dispatch_transaction);
        ASSERT_EQ(return_transaction->lanes().size(), 4u);
        EXPECT_EQ(
            return_batch->getParams().device_id,
            DeviceId::rocm(0));
        EXPECT_EQ(
            return_batch->getParams().canonical_route_contributions,
            buffers0.extensions.at(
                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS));
        EXPECT_EQ(
            return_batch->graphLaunchPreparationPolicy(),
            GraphLaunchPreparationPolicy::CaptureOnly);

        std::vector<int> batched_participant_order;
        for (const auto &lane : dispatch_transaction->lanes())
        {
            EXPECT_EQ(lane.device, DeviceId::rocm(0));
            EXPECT_EQ(lane.graph_family_ordinal, 0u);
            batched_participant_order.push_back(
                lane.target_participant_id);
        }
        EXPECT_EQ(
            batched_participant_order,
            std::vector<int>({2, 3, 4, 5}));
        for (std::size_t lane = 0u;
             lane < return_transaction->lanes().size();
             ++lane)
        {
            EXPECT_EQ(
                return_transaction->lanes()[lane]
                    .target_participant_id,
                batched_participant_order[lane]);
        }
        for (const auto &segment : nonroot_capture_plan.segments)
        {
            EXPECT_TRUE(segment.retained_parent_input_ids.empty())
                << "Complete endpoint captures never weaken an arena frontier";
        }

        ComputeGraph graph0 = builder0.buildFFNGraph(
            layer, buffers0, 0, kSeqLen, kBatchSize,
            DeviceId::rocm(0), stream0.get());
        ComputeGraph graph1 = builder1.buildFFNGraph(
            layer, buffers1, 0, kSeqLen, kBatchSize,
            DeviceId::rocm(1), stream1.get());

        ASSERT_NE(graph0.getNode(local_name), nullptr);
        ASSERT_NE(graph1.getNode(local_name), nullptr);
        EXPECT_EQ(
            countStagesOfType(graph0, ComputeStageType::MOE_EXPERT_FFN),
            1u);
        EXPECT_EQ(
            countStagesOfType(graph1, ComputeStageType::MOE_EXPERT_FFN),
            1u);

        EXPECT_EQ(
            countStagesOfType(graph0, ComputeStageType::MOE_EXPERT_DISPATCH),
            0u);
        EXPECT_EQ(
            countStagesOfType(graph0, ComputeStageType::MOE_SPARSE_DISPATCH),
            0u);
        EXPECT_EQ(
            countStagesOfType(
                graph0, ComputeStageType::MOE_SPARSE_RETURN_REDUCE),
            0u);
        EXPECT_EQ(
            countStagesOfType(
                graph0, ComputeStageType::MOE_RANK_BATCH_DISPATCH),
            0u);
        EXPECT_EQ(
            countStagesOfType(
                graph0, ComputeStageType::MOE_RANK_BATCH_RETURN_REDUCE),
            0u);
        EXPECT_EQ(
            countStagesOfType(
                graph0, ComputeStageType::MOE_OVERLAY_TICKET_PUBLISH),
            0u);
        EXPECT_EQ(
            countStagesOfType(
                graph0, ComputeStageType::MOE_OVERLAY_TICKET_CONSUME),
            0u);
        EXPECT_EQ(
            countStagesOfType(
                graph0,
                ComputeStageType::MOE_OVERLAY_ACTIVATION_DISPATCH_PACK),
            1u);
        EXPECT_EQ(
            countStagesOfType(
                graph0,
                ComputeStageType::MOE_OVERLAY_ACTIVATION_RETURN_CONSUME),
            1u);

        EXPECT_EQ(
            countStagesOfType(graph1, ComputeStageType::MOE_EXPERT_DISPATCH),
            0u);
        EXPECT_EQ(
            countStagesOfType(graph1, ComputeStageType::MOE_SPARSE_DISPATCH),
            0u);
        EXPECT_EQ(
            countStagesOfType(
                graph1, ComputeStageType::MOE_SPARSE_RETURN_REDUCE),
            0u);
        EXPECT_EQ(
            countStagesOfType(
                graph1, ComputeStageType::MOE_RANK_BATCH_DISPATCH),
            0u);
        EXPECT_EQ(
            countStagesOfType(
                graph1, ComputeStageType::MOE_RANK_BATCH_RETURN_REDUCE),
            0u);
        EXPECT_EQ(
            countStagesOfType(
                graph1, ComputeStageType::MOE_OVERLAY_TICKET_PUBLISH),
            0u);
        EXPECT_EQ(
            countStagesOfType(
                graph1, ComputeStageType::MOE_OVERLAY_TICKET_CONSUME),
            0u);
        EXPECT_EQ(
            countStagesOfType(
                graph1,
                ComputeStageType::MOE_OVERLAY_ACTIVATION_DISPATCH_PACK),
            0u);
        EXPECT_EQ(
            countStagesOfType(
                graph1,
                ComputeStageType::MOE_OVERLAY_ACTIVATION_RETURN_CONSUME),
            0u);

        const std::string dispatch_name =
            "layer0_moe_overlay_activation_dispatch_pack_batch";
        const std::string return_name =
            "layer0_moe_overlay_activation_return_consume_batch";
        const auto *prefill_dispatch_node = graph0.getNode(dispatch_name);
        const auto *prefill_return_node = graph0.getNode(return_name);
        ASSERT_NE(prefill_dispatch_node, nullptr);
        ASSERT_NE(prefill_return_node, nullptr);
        const auto *prefill_dispatch = dynamic_cast<
            const MoEOverlayActivationDispatchPackBatchStage *>(
            prefill_dispatch_node->stage.get());
        const auto *prefill_return = dynamic_cast<
            const MoEOverlayActivationReturnConsumeBatchStage *>(
            prefill_return_node->stage.get());
        ASSERT_NE(prefill_dispatch, nullptr);
        ASSERT_NE(prefill_return, nullptr);

        const auto &prefill_transaction =
            prefill_dispatch->getParams().transaction;
        ASSERT_NE(prefill_transaction, nullptr);
        EXPECT_EQ(
            prefill_return->getParams().transaction,
            prefill_transaction)
            << "Fork and join must share one immutable layer transaction";
        ASSERT_EQ(prefill_transaction->lanes().size(), 4u);
        EXPECT_EQ(prefill_transaction->device(), DeviceId::rocm(0));
        EXPECT_EQ(prefill_transaction->physicalRows(), kSeqLen);
        EXPECT_EQ(prefill_transaction->stageOrdinal(), 0u);
        EXPECT_EQ(prefill_transaction->modelLayerIndex(), 0);
        EXPECT_EQ(prefill_dispatch->getParams().hidden, buffers0.normalized);
        EXPECT_EQ(
            prefill_dispatch->getParams().routing_indices,
            buffers0.extensions.at(BufferId::MOE_EXPERT_INDICES));
        EXPECT_EQ(
            prefill_dispatch->getParams().routing_weights,
            buffers0.extensions.at(BufferId::MOE_EXPERT_WEIGHTS));
        EXPECT_TRUE(prefill_dispatch->getParams().placement.valid());
        EXPECT_EQ(
            prefill_dispatch->getParams().placement.expert_count,
            static_cast<std::uint32_t>(kNumExperts));
        EXPECT_EQ(
            prefill_return->getParams().canonical_route_contributions,
            buffers0.extensions.at(
                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS));
        EXPECT_TRUE(
            prefill_dispatch->supportsGraphCaptureAfterLaunchPreparation());
        EXPECT_TRUE(
            prefill_dispatch->supportsLazyPrefillGraphCapturePreflight());
        EXPECT_TRUE(
            prefill_return->supportsGraphCaptureAfterLaunchPreparation());
        EXPECT_TRUE(
            prefill_return->supportsLazyPrefillGraphCapturePreflight());
        EXPECT_TRUE(
            prefill_return->supportsPaddedPrefillGraphCapturePreflight());
        EXPECT_EQ(
            prefill_dispatch->graphLaunchPreparationPolicy(),
            GraphLaunchPreparationPolicy::CaptureOnly);
        EXPECT_EQ(
            prefill_return->graphLaunchPreparationPolicy(),
            GraphLaunchPreparationPolicy::CaptureOnly);

        std::vector<int> prefill_participant_order;
        for (const auto &lane : prefill_transaction->lanes())
        {
            EXPECT_EQ(lane.device, DeviceId::rocm(0));
            EXPECT_EQ(lane.graph_family_ordinal, 0u);
            prefill_participant_order.push_back(
                lane.target_participant_id);
        }
        EXPECT_EQ(
            prefill_participant_order,
            std::vector<int>({2, 3, 4, 5}));
        EXPECT_EQ(
            graph0.nativeCaptureEnvelope(),
            GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
        EXPECT_EQ(
            graph1.nativeCaptureEnvelope(),
            GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
        EXPECT_FALSE(makeMoEOverlayRetainedParentPlan(graph0).has_value());
        EXPECT_FALSE(makeMoEOverlayRetainedParentPlan(graph1).has_value());

        const std::string ordered_name =
            "layer0_moe_overlay_continuation_routes_ordered_reduce";
        const std::string rooted_name =
            "layer0_moe_overlay_continuation_routes_reduce_to_root";
        const std::string merge_name =
            "layer0_moe_overlay_remote_routes_merge";
        const std::string legacy_broadcast_name =
            "layer0_moe_overlay_continuation_broadcast";
        const std::string shared_reduce_name =
            "layer0_shared_expert_reduce_to_overlay_root";
        const std::string shared_gate_name =
            "layer0_shared_expert_gate";
        const std::string combined_broadcast_name =
            "layer0_moe_overlay_combined_broadcast";
        EXPECT_TRUE(
            hasDependency(graph0, local_name, dispatch_name))
            << "The single fanout must publish before local overlap starts";
        EXPECT_TRUE(
            hasDependency(graph0, ordered_name, return_name))
            << "The continuation may fold canonical route slots only after "
               "every remote lane has materialized its disjoint rows";
        for (const auto *graph : {&graph0, &graph1})
        {
            ASSERT_NE(graph->getNode(ordered_name), nullptr);
            ASSERT_NE(graph->getNode(shared_reduce_name), nullptr);
            ASSERT_NE(graph->getNode(shared_gate_name), nullptr);
            ASSERT_NE(graph->getNode(combined_broadcast_name), nullptr);
            EXPECT_EQ(graph->getNode(legacy_broadcast_name), nullptr)
                << "The routed intermediate must not be broadcast separately";
            if (route_transport ==
                MoEOverlayNodeLocalRouteTransport::NativeCollective)
            {
                ASSERT_NE(graph->getNode(rooted_name), nullptr);
                EXPECT_TRUE(hasDependency(*graph, rooted_name, local_name));
                if (graph == &graph0)
                {
                    EXPECT_TRUE(hasDependency(
                        *graph, return_name, rooted_name));
                }
                else
                {
                    EXPECT_TRUE(hasDependency(
                        *graph, ordered_name, rooted_name));
                }
            }
            else
            {
                EXPECT_EQ(graph->getNode(rooted_name), nullptr);
                if (graph == &graph0)
                {
                    EXPECT_TRUE(hasDependency(
                        *graph, return_name, local_name));
                }
                else
                {
                    EXPECT_TRUE(hasDependency(
                        *graph, ordered_name, local_name));
                }
            }
            if (graph == &graph0)
            {
                EXPECT_TRUE(hasDependency(
                    *graph, ordered_name, return_name));
            }
            EXPECT_TRUE(hasDependency(
                *graph, shared_gate_name, shared_reduce_name));
            EXPECT_TRUE(hasDependency(
                *graph, shared_reduce_name, ordered_name))
                << "The rooted shared reduction must remain in the post-ticket "
                   "unit so every LocalTP participant launches its matching "
                   "collective in the same capture wave";
            EXPECT_FALSE(hasDependency(
                *graph, ordered_name, shared_reduce_name))
                << "An unmatched pre-ticket rooted reduction can occupy the GPU "
                   "while its peer waits at the next capture-wave rendezvous";
            EXPECT_TRUE(hasDependency(
                *graph, combined_broadcast_name, shared_gate_name));
            EXPECT_TRUE(hasDependency(
                *graph, shared_gate_name, ordered_name));
        }

        const auto *ordered_stage0 =
            dynamic_cast<const MoECanonicalRouteReduceStage *>(
                graph0.getNode(ordered_name)->stage.get());
        const auto *ordered_stage1 =
            dynamic_cast<const MoECanonicalRouteReduceStage *>(
                graph1.getNode(ordered_name)->stage.get());
        const auto *decode_ordered_stage0 =
            dynamic_cast<const MoECanonicalRouteReduceStage *>(
                decode_graph0.getNode(ordered_name)->stage.get());
        const auto *decode_ordered_stage1 =
            dynamic_cast<const MoECanonicalRouteReduceStage *>(
                decode_graph1.getNode(ordered_name)->stage.get());
        ASSERT_NE(ordered_stage0, nullptr);
        ASSERT_NE(ordered_stage1, nullptr);
        ASSERT_NE(decode_ordered_stage0, nullptr);
        ASSERT_NE(decode_ordered_stage1, nullptr);

        const auto assert_route_stage =
            [&](const MoECanonicalRouteReduceStage *stage,
                int expected_rows,
                int expected_participant,
                MoECanonicalRouteReductionRole expected_role)
        {
            const auto &params = stage->params();
            EXPECT_EQ(params.seq_len, expected_rows);
            EXPECT_EQ(params.top_k, kTopK);
            EXPECT_EQ(params.d_model, kDModel);
            EXPECT_EQ(params.reduction_role, expected_role);
            EXPECT_EQ(params.route_participant_id, expected_participant);
            {
                const bool uses_mapped_sparse_transport =
                    route_transport ==
                    MoEOverlayNodeLocalRouteTransport::MappedSparse;
                if (uses_mapped_sparse_transport)
                {
                    EXPECT_EQ(
                        params.node_local_route_exchange,
                        config0.moe.node_local_route_exchange)
                        << "Every graph and bucket must retain one sparse fabric owner";
                }
                else
                {
                    EXPECT_EQ(params.node_local_route_exchange, nullptr)
                        << "Native NCCL/RCCL owns row transport without a mapped fabric";
                }
                EXPECT_NE(
                    params.domain_route_assignment.participant_ids,
                    nullptr);
                EXPECT_TRUE(params.runtime_route_weights.validFor(
                    static_cast<std::uint32_t>(expected_rows),
                    static_cast<std::uint32_t>(kTopK)));
                EXPECT_EQ(
                    params.runtime_route_weights.projection,
                    expected_rows == 1
                        ? MoERuntimeRouteWeightProjection::DecodeTopK
                        : MoERuntimeRouteWeightProjection::GroupedRouteSlots);
                EXPECT_GE(
                    params.domain_route_assignment.capacity,
                    static_cast<std::uint32_t>(expected_rows * kTopK));
                EXPECT_TRUE(params.domain_route_assignment.validFor(
                    static_cast<std::uint32_t>(expected_rows * kTopK)));
                ASSERT_TRUE(params.overlay_route_placement.valid());
                EXPECT_EQ(
                    params.overlay_route_placement.expert_count,
                    static_cast<std::uint32_t>(kNumExperts));
                const bool root = expected_role ==
                                  MoECanonicalRouteReductionRole::RootOwner;
                if (uses_mapped_sparse_transport)
                {
                    EXPECT_EQ(
                        stage->coherencePolicy(), CoherencePolicy::FULL)
                        << "A mapped sparse publisher is active on every participant";
                    EXPECT_FALSE(stage->isPassiveGraphCaptureNoOp());
                }
                else
                {
                    EXPECT_EQ(
                        stage->coherencePolicy(),
                        root ? CoherencePolicy::FULL
                             : CoherencePolicy::NONE);
                    EXPECT_EQ(stage->isPassiveGraphCaptureNoOp(), !root);
                }

                const StageDumpInfo dump = stage->getDumpInfoSnapshot();
                const auto find_output =
                    [&](std::string_view name)
                    {
                        return std::find_if(
                            dump.outputs.begin(),
                            dump.outputs.end(),
                            [name](const StageDumpInfo::OutputBuffer &output)
                            {
                                return output.name &&
                                       std::string_view(output.name) == name;
                            });
                    };
                const auto assignment_dump = find_output(
                    "domain_route_participant_ids");
                const auto runtime_weights_dump = find_output(
                    "runtime_route_weights");
                const auto bank0_dump = find_output(
                    "overlay_route_participants_bank0");
                const auto bank0_epoch_dump = find_output(
                    "overlay_route_bank0_epoch");
                const auto bank1_dump = find_output(
                    "overlay_route_participants_bank1");
                const auto bank1_epoch_dump = find_output(
                    "overlay_route_bank1_epoch");
                const auto selected_bank_dump = find_output(
                    "overlay_route_selected_bank");
                if (expected_role ==
                    MoECanonicalRouteReductionRole::RootOwner)
                {
                    ASSERT_NE(assignment_dump, dump.outputs.end());
                    ASSERT_NE(runtime_weights_dump, dump.outputs.end());
                    ASSERT_NE(assignment_dump->tensor, nullptr);
                    ASSERT_NE(runtime_weights_dump->tensor, nullptr);
                    EXPECT_EQ(
                        assignment_dump->tensor->gpu_data_ptr(),
                        params.domain_route_assignment.participant_ids);
                    EXPECT_EQ(assignment_dump->rows,
                              static_cast<std::size_t>(expected_rows));
                    EXPECT_EQ(assignment_dump->cols,
                              static_cast<std::size_t>(kTopK));
                    EXPECT_STREQ(assignment_dump->dtype, "INT32");
                    EXPECT_EQ(
                        runtime_weights_dump->tensor->gpu_data_ptr(),
                        params.runtime_route_weights.weights);
                    EXPECT_EQ(runtime_weights_dump->rows,
                              static_cast<std::size_t>(expected_rows));
                    EXPECT_EQ(runtime_weights_dump->cols,
                              static_cast<std::size_t>(kTopK));
                    EXPECT_STREQ(runtime_weights_dump->dtype, "FP32");
                    ASSERT_NE(bank0_dump, dump.outputs.end());
                    ASSERT_NE(bank0_epoch_dump, dump.outputs.end());
                    ASSERT_NE(bank1_dump, dump.outputs.end());
                    ASSERT_NE(bank1_epoch_dump, dump.outputs.end());
                    ASSERT_NE(selected_bank_dump, dump.outputs.end());
                    ASSERT_NE(bank0_dump->tensor, nullptr);
                    ASSERT_NE(bank0_epoch_dump->tensor, nullptr);
                    ASSERT_NE(bank1_dump->tensor, nullptr);
                    ASSERT_NE(bank1_epoch_dump->tensor, nullptr);
                    ASSERT_NE(selected_bank_dump->tensor, nullptr);
                    EXPECT_EQ(
                        bank0_dump->tensor->gpu_data_ptr(),
                        params.overlay_route_placement.banks[0]
                            .route_participants);
                    EXPECT_EQ(
                        bank1_dump->tensor->gpu_data_ptr(),
                        params.overlay_route_placement.banks[1]
                            .route_participants);
                    EXPECT_EQ(
                        bank0_epoch_dump->tensor->gpu_data_ptr(),
                        params.overlay_route_placement.banks[0].epoch);
                    EXPECT_EQ(
                        bank1_epoch_dump->tensor->gpu_data_ptr(),
                        params.overlay_route_placement.banks[1].epoch);
                    EXPECT_EQ(
                        selected_bank_dump->tensor->gpu_data_ptr(),
                        &params.overlay_route_placement.status->bank);
                    EXPECT_EQ(bank0_dump->rows, 1u);
                    EXPECT_EQ(bank1_dump->rows, 1u);
                    EXPECT_EQ(bank0_dump->cols,
                              static_cast<std::size_t>(kNumExperts));
                    EXPECT_EQ(bank0_epoch_dump->rows, 1u);
                    EXPECT_EQ(bank0_epoch_dump->cols, 1u);
                    EXPECT_EQ(bank1_dump->cols,
                              static_cast<std::size_t>(kNumExperts));
                    EXPECT_EQ(bank1_epoch_dump->rows, 1u);
                    EXPECT_EQ(bank1_epoch_dump->cols, 1u);
                    EXPECT_EQ(selected_bank_dump->rows, 1u);
                    EXPECT_EQ(selected_bank_dump->cols, 1u);
                    EXPECT_STREQ(bank0_dump->dtype, "INT32");
                    EXPECT_STREQ(bank0_epoch_dump->dtype, "INT32");
                    EXPECT_STREQ(bank1_dump->dtype, "INT32");
                    EXPECT_STREQ(bank1_epoch_dump->dtype, "INT32");
                    EXPECT_STREQ(selected_bank_dump->dtype, "INT32");
                }
                else
                {
                    EXPECT_EQ(assignment_dump, dump.outputs.end())
                        << "Only the reduction owner publishes the canonical "
                           "domain route-assignment diagnostic";
                    EXPECT_EQ(runtime_weights_dump, dump.outputs.end());
                    EXPECT_EQ(bank0_dump, dump.outputs.end());
                    EXPECT_EQ(bank0_epoch_dump, dump.outputs.end());
                    EXPECT_EQ(bank1_dump, dump.outputs.end());
                    EXPECT_EQ(bank1_epoch_dump, dump.outputs.end());
                    EXPECT_EQ(selected_bank_dump, dump.outputs.end());
                }
            }
            EXPECT_TRUE(
                stage->supportsGraphCaptureAfterLaunchPreparation());
        };
        assert_route_stage(
            decode_ordered_stage0,
            /*expected_rows=*/1,
            /*expected_participant=*/0,
            MoECanonicalRouteReductionRole::RootOwner);
        assert_route_stage(
            decode_ordered_stage1,
            /*expected_rows=*/1,
            /*expected_participant=*/1,
            MoECanonicalRouteReductionRole::NonRootParticipant);
        assert_route_stage(
            ordered_stage0,
            /*expected_rows=*/kSeqLen,
            /*expected_participant=*/0,
            MoECanonicalRouteReductionRole::RootOwner);
        assert_route_stage(
            ordered_stage1,
            /*expected_rows=*/kSeqLen,
            /*expected_participant=*/1,
            MoECanonicalRouteReductionRole::NonRootParticipant);
        if (route_transport ==
            MoEOverlayNodeLocalRouteTransport::MappedSparse)
        {
            ASSERT_NE(config0.moe.node_local_route_exchange, nullptr);
            EXPECT_TRUE(config0.moe.node_local_route_exchange->materialized());
            EXPECT_EQ(
                config0.moe.node_local_route_exchange->routeCapacity(),
                static_cast<std::uint32_t>(kSeqLen * kTopK));
        }
        else
        {
            ASSERT_EQ(config0.moe.node_local_route_exchange, nullptr);
            const std::array<const ComputeGraph *, 4> native_graphs{
                &decode_graph0, &decode_graph1, &graph0, &graph1};
            for (std::size_t index = 0; index < native_graphs.size(); ++index)
            {
                const auto *rooted_stage = dynamic_cast<
                    const TPLocalRootedCollectiveStage *>(
                    native_graphs[index]
                        ->getNode(rooted_name)
                        ->stage.get());
                ASSERT_NE(rooted_stage, nullptr);
                const auto &params = rooted_stage->params();
                EXPECT_EQ(
                    params.operation,
                    TPLocalRootedCollectiveOperation::ReduceSum);
                EXPECT_EQ(params.root_device_index, 0);
                EXPECT_EQ(
                    params.participant_device_index,
                    static_cast<int>(index % 2u));
                const std::size_t rows = index < 2u ? 1u : kSeqLen;
                EXPECT_EQ(
                    params.count,
                    rows * static_cast<std::size_t>(kTopK) *
                        static_cast<std::size_t>(kDModel));
                ASSERT_TRUE(params.tensor_buffer_id.has_value());
                EXPECT_EQ(
                    *params.tensor_buffer_id,
                    BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
            }
        }
        EXPECT_EQ(graph0.getNode(merge_name), nullptr)
            << "Mapped returns fold directly into canonical MoE output";
        EXPECT_EQ(graph1.getNode(merge_name), nullptr);

        const auto *shared_reduce_stage0 =
            dynamic_cast<const TPLocalRootedCollectiveStage *>(
                graph0.getNode(shared_reduce_name)->stage.get());
        const auto *shared_reduce_stage1 =
            dynamic_cast<const TPLocalRootedCollectiveStage *>(
                graph1.getNode(shared_reduce_name)->stage.get());
        const auto *combined_broadcast_stage0 =
            dynamic_cast<const TPLocalRootedCollectiveStage *>(
                graph0.getNode(combined_broadcast_name)->stage.get());
        const auto *combined_broadcast_stage1 =
            dynamic_cast<const TPLocalRootedCollectiveStage *>(
                graph1.getNode(combined_broadcast_name)->stage.get());
        const auto *shared_gate_stage0 =
            dynamic_cast<const SharedExpertGateStage *>(
                graph0.getNode(shared_gate_name)->stage.get());
        const auto *shared_gate_stage1 =
            dynamic_cast<const SharedExpertGateStage *>(
                graph1.getNode(shared_gate_name)->stage.get());
        ASSERT_NE(shared_reduce_stage0, nullptr);
        ASSERT_NE(shared_reduce_stage1, nullptr);
        ASSERT_NE(combined_broadcast_stage0, nullptr);
        ASSERT_NE(combined_broadcast_stage1, nullptr);
        ASSERT_NE(shared_gate_stage0, nullptr);
        ASSERT_NE(shared_gate_stage1, nullptr);

        const std::array<const TPLocalRootedCollectiveStage *, 2>
            shared_reductions{
                shared_reduce_stage0,
                shared_reduce_stage1,
            };
        const std::array<const TPLocalRootedCollectiveStage *, 2>
            combined_broadcasts{
                combined_broadcast_stage0,
                combined_broadcast_stage1,
            };
        for (std::size_t participant = 0u;
             participant < shared_reductions.size();
             ++participant)
        {
            const auto &reduce = shared_reductions[participant]->params();
            const auto &broadcast =
                combined_broadcasts[participant]->params();
            EXPECT_EQ(
                reduce.operation,
                TPLocalRootedCollectiveOperation::ReduceSum);
            EXPECT_EQ(
                broadcast.operation,
                TPLocalRootedCollectiveOperation::Broadcast);
            EXPECT_EQ(reduce.root_device_index, 0);
            EXPECT_EQ(broadcast.root_device_index, 0);
            EXPECT_EQ(
                reduce.participant_device_index,
                static_cast<int>(participant));
            EXPECT_EQ(
                broadcast.participant_device_index,
                static_cast<int>(participant));
            EXPECT_EQ(
                reduce.count,
                static_cast<std::size_t>(kSeqLen * kDModel));
            EXPECT_EQ(broadcast.count, reduce.count);
            ASSERT_TRUE(reduce.tensor_buffer_id.has_value());
            ASSERT_TRUE(broadcast.tensor_buffer_id.has_value());
            EXPECT_EQ(
                *reduce.tensor_buffer_id,
                BufferId::MOE_SHARED_EXPERT_OUTPUT);
            EXPECT_EQ(
                *broadcast.tensor_buffer_id,
                BufferId::ATTN_PROJ);
            if (route_transport ==
                MoEOverlayNodeLocalRouteTransport::MappedSparse)
            {
                const bool mapped_dense =
                    shouldUseMoEOverlayMappedDensePublication(
                        route_transport,
                        broadcast.count * sizeof(float));
                EXPECT_EQ(
                    broadcast.mapped_dense_publication_exchange != nullptr,
                    mapped_dense)
                    << "No-P2P dense publication must honor the typed payload crossover";
                if (mapped_dense)
                {
                    EXPECT_EQ(
                        broadcast.mapped_dense_publication_exchange,
                        config0.moe.node_local_route_exchange);
                }
                EXPECT_TRUE(combined_broadcasts[participant]->isGraphCapturable());
            }
            else
            {
                EXPECT_EQ(
                    broadcast.mapped_dense_publication_exchange,
                    nullptr)
                    << "P2P-capable continuation output must remain on its native collective";
            }
        }
        EXPECT_EQ(
            shared_reduce_stage0->params().tensor,
            buffers0.extensions.at(BufferId::MOE_SHARED_EXPERT_OUTPUT));
        EXPECT_EQ(
            shared_reduce_stage1->params().tensor,
            buffers1.extensions.at(BufferId::MOE_SHARED_EXPERT_OUTPUT));
        EXPECT_EQ(
            combined_broadcast_stage0->params().tensor,
            buffers0.attn_proj);
        EXPECT_EQ(
            combined_broadcast_stage1->params().tensor,
            buffers1.attn_proj);
        EXPECT_EQ(
            shared_gate_stage0->executionRole(),
            SharedExpertGateStage::ExecutionRole::RootOwner);
        EXPECT_EQ(
            shared_gate_stage1->executionRole(),
            SharedExpertGateStage::ExecutionRole::NonRootObserver);
        EXPECT_EQ(
            shared_gate_stage1->coherencePolicy(),
            CoherencePolicy::NONE)
            << "A non-root observer must have no tensor ownership";
    }

    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         DistributedLocalTPContinuationUsesMappedSparseWithoutP2P)
    {
        assertDistributedLocalTPContinuationCapture(
            MoEOverlayNodeLocalRouteTransport::MappedSparse);
    }

    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         DistributedLocalTPContinuationKeepsNativeCollectiveWithP2P)
    {
        assertDistributedLocalTPContinuationCapture(
            MoEOverlayNodeLocalRouteTransport::NativeCollective);
    }

    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPApportionedRoutedComputeGpuPrefillUsesCapturableFastPathByDefault)
    {
        GraphConfig config = makeConfig(makeLocalTPApportionedHotPlan());
        config.default_device = DeviceId::rocm(0);
        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);
        tp_ctx.setRawAllgatherGraphCaptureSupported(true);
        config.tp_ctx = &tp_ctx;
        config.tp_device_idx = 0;

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        TensorArena activation_arena;
        auto buffers = makeActivationBuffers(activation_arena);

        auto model_ctx =
            makeTestingModelContextWithHotDomainExperts(config.n_layers);
        Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
        ScopedDevicePublicationStream publication_stream(DeviceId::rocm(0));
        ComputeGraph graph = graph_builder.buildFFNGraph(
            layer,
            buffers,
            0,
            kSeqLen,
            kBatchSize,
            DeviceId::rocm(0),
            publication_stream.get());

        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::MOE_EXPERT_DISPATCH), 0u)
            << "Homogeneous LocalTP GPU prefill must not lower through the host dispatch descriptor path";
        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::MOE_SPARSE_DISPATCH), 0u)
            << "Homogeneous LocalTP GPU prefill must use the fixed-topology grouped prefill path";
        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::MOE_SPARSE_RETURN_REDUCE), 0u);
        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::MOE_LOCAL_EXPERT), 0u);
        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::MOE_EXPERT_FFN), 1u);

        const auto *expert_node = graph.getNode("layer0_moe_expert_ffn_overlay_fast");
        ASSERT_NE(expert_node, nullptr);
        EXPECT_EQ(expert_node->device, DeviceId::rocm(0));
        const auto *expert_stage = dynamic_cast<const MoEExpertComputeStage *>(expert_node->stage.get());
        ASSERT_NE(expert_stage, nullptr);
        EXPECT_EQ(expert_stage->fixedTopologyPrefillExpertIdsForTesting(),
                  (std::vector<int>{0, 1, 2}));
        EXPECT_TRUE(expert_node->stage->supportsGraphCaptureAfterLaunchPreparation())
            << "Cold preflight must allow warmup to build MoE grouped-prefill capture resources";
        EXPECT_TRUE(expert_node->stage->supportsLazyPrefillGraphCapturePreflight())
            << "The fixed-topology grouped prefill path is the graph-capturable MoE dispatch contract";

        const auto *rooted_reduce_node =
            graph.getNode("layer0_moe_canonical_routes_reduce_to_root");
        ASSERT_NE(rooted_reduce_node, nullptr)
            << "Graph-local owner subsets must reduce router-ordered route slots to one fixed continuation root";
        const auto *rooted_reduce_stage =
            dynamic_cast<const TPLocalRootedCollectiveStage *>(
                rooted_reduce_node->stage.get());
        ASSERT_NE(rooted_reduce_stage, nullptr);
        EXPECT_EQ(
            rooted_reduce_stage->params().tensor,
            buffers.extensions.at(
                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS));
        EXPECT_EQ(
            rooted_reduce_stage->params().count,
            static_cast<size_t>(kSeqLen * kTopK * kDModel));
        EXPECT_EQ(
            rooted_reduce_stage->params().dtype,
            CollectiveDataType::FLOAT32);
        EXPECT_EQ(
            rooted_reduce_stage->params().operation,
            TPLocalRootedCollectiveOperation::ReduceSum);
        EXPECT_EQ(rooted_reduce_stage->params().participant_device_index, 0);
        EXPECT_EQ(rooted_reduce_stage->params().root_device_index, 0);
        EXPECT_EQ(
            rooted_reduce_stage->params().tensor_buffer_id,
            std::optional<BufferId>{
                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS});
        EXPECT_TRUE(hasDependency(
            graph,
            "layer0_moe_canonical_routes_reduce_to_root",
            "layer0_moe_expert_ffn_overlay_fast"));

        const auto *reduce_node =
            graph.getNode("layer0_moe_canonical_routes_reduce");
        ASSERT_NE(reduce_node, nullptr);
        EXPECT_EQ(
            reduce_node->stage->type(),
            ComputeStageType::MOE_CANONICAL_ROUTE_REDUCE);
        EXPECT_TRUE(hasDependency(
            graph,
            "layer0_moe_canonical_routes_reduce",
            "layer0_moe_canonical_routes_reduce_to_root"));
        const auto *reduce_stage =
            dynamic_cast<const MoECanonicalRouteReduceStage *>(
                reduce_node->stage.get());
        ASSERT_NE(reduce_stage, nullptr);
        EXPECT_EQ(
            reduce_stage->params().canonical_route_contributions,
            buffers.extensions.at(
                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS));
        EXPECT_EQ(
            reduce_stage->params().output,
            buffers.extensions.at(BufferId::MOE_COMBINED_OUTPUT));
        EXPECT_EQ(reduce_stage->params().seq_len, kSeqLen);
        EXPECT_EQ(reduce_stage->params().top_k, kTopK);
        EXPECT_EQ(reduce_stage->params().d_model, kDModel);
        EXPECT_EQ(
            reduce_stage->params().reduction_role,
            MoECanonicalRouteReductionRole::RootOwner);
        EXPECT_TRUE(reduce_stage->supportsGraphCaptureAfterLaunchPreparation());
        EXPECT_TRUE(reduce_stage->supportsLazyPrefillGraphCapturePreflight())
            << "A cold LocalTP MoE graph must admit the allocation-free canonical "
               "route reducer so eager warmup can bind its backend kernel wrapper";
        EXPECT_TRUE(reduce_stage->supportsPaddedPrefillGraphCapturePreflight())
            << "Canonical route reduction is row-independent and must not reject "
               "fixed padded prefill buckets";

        const auto *broadcast_node =
            graph.getNode("layer0_moe_canonical_routes_broadcast");
        ASSERT_NE(broadcast_node, nullptr)
            << "Only the compact routed result should be replicated after the root folds route slots";
        const auto *broadcast_stage =
            dynamic_cast<const TPLocalRootedCollectiveStage *>(
                broadcast_node->stage.get());
        ASSERT_NE(broadcast_stage, nullptr);
        EXPECT_EQ(
            broadcast_stage->params().tensor,
            buffers.extensions.at(BufferId::MOE_COMBINED_OUTPUT));
        EXPECT_EQ(
            broadcast_stage->params().count,
            static_cast<size_t>(kSeqLen * kDModel));
        EXPECT_EQ(
            broadcast_stage->params().operation,
            TPLocalRootedCollectiveOperation::Broadcast);
        EXPECT_EQ(broadcast_stage->params().participant_device_index, 0);
        EXPECT_EQ(broadcast_stage->params().root_device_index, 0);
        EXPECT_TRUE(hasDependency(
            graph,
            "layer0_moe_canonical_routes_broadcast",
            "layer0_moe_canonical_routes_reduce"));

        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::ALLREDUCE), 0u);
        EXPECT_EQ(
            countStagesOfType(graph, ComputeStageType::ROOTED_COLLECTIVE),
            2u);
        EXPECT_EQ(
            countStagesOfType(
                graph,
                ComputeStageType::MOE_CANONICAL_ROUTE_REDUCE),
            1u);
    }

    /**
     * @brief Require the optimized LocalTP graphs to assemble the initial RCU banks.
     *
     * The sparse generic overlay path publishes prepared engine lifetimes from
     * `MoELocalExpertStage` construction. Homogeneous LocalTP deliberately
     * bypasses those host-dispatch stages, so each participant's optimized graph
     * must perform the equivalent initial-bank publication itself. Building both
     * graph-local participants here reproduces one process owning two GPUs and
     * proves that startup cannot reach maintenance with an incomplete bank.
     */
    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPFastPathPublishesEveryInitialParticipantResidencyBank)
    {
        constexpr int kLayerCount = 1;
        auto plan = makeLocalTPApportionedHotPlan(kLayerCount);
        const auto owner_map = MoEExpertOwnerMap::build(*plan);
        GraphConfig config = makeConfig(plan, kLayerCount);

        auto authority = std::make_shared<MoEOverlayResidencyAuthority>(
            MoEOverlayResidencyAuthority::Config{
                .initial_plan = *plan,
                .model_metadata = {
                    .num_layers = kLayerCount,
                    .num_experts = kNumExperts,
                    .d_model = kDModel,
                    .routed_intermediate_size = kIntermediate,
                },
                .maintenance_mode = MoERebalanceRuntimeMode::Off,
                .histogram = nullptr,
                .perf_device = "rocm_localtp_test",
            });
        const auto snapshot = authority->snapshot();
        ASSERT_NE(snapshot, nullptr);
        ASSERT_TRUE(snapshot->valid());

        std::vector<int> local_participants;
        for (const auto &participant : owner_map.participants())
            local_participants.push_back(participant.participant_id);
        ASSERT_EQ(local_participants, (std::vector<int>{0, 1}));

        auto participant_residency =
            std::make_shared<MoEOverlayParticipantResidencyRegistry>(
                MoEOverlayParticipantResidencyRegistry::Config{
                    .owner_map = snapshot->owner_map,
                    .local_participant_ids = local_participants,
                    .num_layers = kLayerCount,
                    .num_experts = kNumExperts,
                    .initial_epoch = snapshot->epoch,
                });
        config.moe.expert_overlay_residency_authority = authority;
        config.moe.durable_residency_authority =
            MoEDurableResidencyAuthorityKind::ExpertOverlayRCU;
        config.moe.expert_overlay_participant_residency =
            participant_residency;

        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices(
            {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);
        tp_ctx.setRawAllgatherGraphCaptureSupported(true);
        config.tp_ctx = &tp_ctx;

        auto model_ctx =
            makeTestingModelContextWithHotDomainExperts(kLayerCount);
        registerOwnedParticipantExpertLayers(
            model_ctx->concreteWeightManager()->expertGemmRegistry(),
            owner_map,
            kLayerCount);

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        for (int participant = 0; participant < 2; ++participant)
        {
            const DeviceId device = DeviceId::rocm(participant);
            config.default_device = device;
            config.tp_device_idx = participant;

            TensorArena activation_arena;
            auto buffers = makeActivationBuffers(activation_arena);
            Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
            ScopedDevicePublicationStream publication_stream(device);
            const ComputeGraph graph = graph_builder.buildFFNGraph(
                layer,
                buffers,
                /*layer_idx=*/0,
                kSeqLen,
                kBatchSize,
                device,
                publication_stream.get());

            ASSERT_NE(
                graph.getNode("layer0_moe_expert_ffn_overlay_fast"),
                nullptr);
            EXPECT_EQ(
                countStagesOfType(
                    graph,
                    ComputeStageType::MOE_DEVICE_REBALANCE),
                0u)
                << "ExpertOverlayRCU must remain the sole durable placement "
                   "writer for a multi-participant GPU tier";
            const auto endpoint = participant_residency->endpoint(participant);
            ASSERT_NE(endpoint, nullptr);
            const auto bank = endpoint->acquire(snapshot->epoch);
            ASSERT_NE(bank, nullptr)
                << "participant=" << participant;
            EXPECT_EQ(bank->participant_id, participant);
            EXPECT_EQ(bank->device, device);

            if (participant == 0)
            {
                EXPECT_FALSE(participant_residency->allInitialBanksReady())
                    << "The second graph-local participant has not published yet";
            }
        }

        EXPECT_TRUE(participant_residency->allInitialBanksReady());
    }

    /**
     * @brief GPU shared-verifier nodes wait for their router-owned Q8 input rows.
     *
     * The production grouped router publishes a device-resident Q8 copy of each
     * normalized verifier row.  Both the routed experts and the always-active
     * shared expert reuse that publication.  NORMALIZED alone therefore does not
     * describe the shared stage's complete dependency set: without an edge from
     * `moe_routing`, graph topological ordering may run the shared sibling first
     * and consume a stale publication left by an earlier layer.
     *
     * This fixture supplies the same prepared expert-engine registry required by
     * production GPU lowering, but it deliberately creates no device context,
     * allocates no GPU memory, and executes no GPU work.  Unit coverage therefore
     * proves the CUDA and ROCm graph contract without violating the integration-
     * only rule for GPU execution.
     */
    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         GPUGroupedSharedVerifierDependsOnRouterQ8Producer)
    {
        auto model_ctx = ModelContext::createForTesting(
            "test.gguf",
            nullptr,
            /*layer_count=*/1,
            /*with_weight_manager=*/true);
        ASSERT_NE(model_ctx, nullptr);
        ASSERT_NE(model_ctx->concreteWeightManager(), nullptr);

        auto &registry = model_ctx->concreteWeightManager()->expertGemmRegistry();
        for (const DeviceId device : {DeviceId::cuda(0), DeviceId::rocm(0)})
            registerCompleteDeviceExpertLayer(registry, device, /*layer_idx=*/0);

        for (const DeviceId device : {DeviceId::cuda(0), DeviceId::rocm(0)})
        {
            SCOPED_TRACE("device=" + device.to_string());

            GraphConfig config = makeConfig(nullptr);
            config.default_device = device;
            config.grouped_mtp_verifier = true;
            config.compute_all_position_logits = true;
            config.moe.has_shared_expert = true;
            config.moe.shared_intermediate_size = kIntermediate;

            TensorArena weight_arena;
            auto layer = makeLayerWeights(weight_arena);
            layer.shared_expert_gate =
                weight_arena.fp32({kIntermediate, kDModel});
            layer.shared_expert_up =
                weight_arena.fp32({kIntermediate, kDModel});
            layer.shared_expert_down =
                weight_arena.fp32({kDModel, kIntermediate});
            layer.shared_expert_gate_inp =
                weight_arena.fp32({1, kDModel});

            TensorArena activation_arena;
            auto buffers = makeActivationBuffers(activation_arena);

            Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
            ScopedDevicePublicationStream publication_stream(device);
            const int32_t *device_row_count =
                publication_stream.publishRowCount(/*value=*/2);
            ComputeGraph graph = graph_builder.buildFFNGraph(
                layer,
                buffers,
                /*layer_idx=*/0,
                /*seq_len=*/2,
                kBatchSize,
                device,
                publication_stream.get(),
                device_row_count);

            const auto *router_node = graph.getNode("layer0_moe_routing");
            ASSERT_NE(router_node, nullptr);
            const auto *router_stage =
                dynamic_cast<const MoERoutingStage *>(router_node->stage.get());
            ASSERT_NE(router_stage, nullptr);
            EXPECT_EQ(
                router_stage->activeRowCountDeviceForTesting(),
                device_row_count);
            const auto *shared_node =
                graph.getNode("layer0_shared_expert_ffn");
            ASSERT_NE(shared_node, nullptr);
            const auto *shared_stage =
                dynamic_cast<const SharedExpertFFNStage *>(
                    shared_node->stage.get());
            ASSERT_NE(shared_stage, nullptr);
            EXPECT_TRUE(
                shared_stage->usesGroupedVerifierPrefillRouteForTesting());
            EXPECT_TRUE(
                shared_stage->requiresRouterQ8PublicationForTesting())
                << "The grouped shared verifier must fail closed without the "
                   "router's exact Q8 row publication";
            EXPECT_TRUE(hasDependency(
                graph,
                "layer0_shared_expert_ffn",
                "layer0_moe_routing"))
                << "The grouped shared verifier consumes the router's device Q8 publication";

            const auto *shared_gate_node =
                graph.getNode("layer0_shared_expert_gate");
            ASSERT_NE(shared_gate_node, nullptr);
            const auto *shared_gate_stage =
                dynamic_cast<const SharedExpertGateStage *>(
                    shared_gate_node->stage.get());
            ASSERT_NE(shared_gate_stage, nullptr);
            EXPECT_EQ(
                shared_gate_stage->activeRowCountDeviceForTesting(),
                device_row_count)
                << "Routing and shared gating must capture the same device geometry owner.";
        }
    }

    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPApportionedLeastLoadedTinyPrefillUsesStaticOwnerCostGate)
    {
        auto plan = makeLocalTPApportionedHotPlan();
        ASSERT_FALSE(plan->domains.empty());
        plan->domains[0].routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        GraphConfig config = makeConfig(plan);
        config.default_device = DeviceId::rocm(0);
        config.moe.routed_prefill_config
            .least_loaded_min_routed_rows = 8192;
        config.moe.routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);
        tp_ctx.setRawAllgatherGraphCaptureSupported(true);
        config.tp_ctx = &tp_ctx;
        config.tp_device_idx = 0;

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        TensorArena activation_arena;
        auto buffers = makeActivationBuffers(activation_arena);

        auto model_ctx =
            makeTestingModelContextWithHotDomainExperts(config.n_layers);
        Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
        ScopedDevicePublicationStream publication_stream(DeviceId::rocm(0));
        ComputeGraph graph = graph_builder.buildFFNGraph(
            layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::rocm(0),
            publication_stream.get());

        const auto *expert_node = graph.getNode("layer0_moe_expert_ffn_overlay_fast");
        ASSERT_NE(expert_node, nullptr);
        const auto *expert_stage = dynamic_cast<const MoEExpertComputeStage *>(expert_node->stage.get());
        ASSERT_NE(expert_stage, nullptr);
        EXPECT_EQ(expert_stage->routedExpertAssignmentPolicyForTesting(),
                  RoutedExpertAssignmentPolicy::StaticOwner)
            << "Tiny prefill below the LLEP routed-row gate must lower as standard "
               "apportioned-expert work instead of paying transfer-backed current-batch movement.";
        EXPECT_FALSE(expert_stage->usesRuntimeRowGroupingForTesting());
        EXPECT_FALSE(expert_stage->hasPrefillLLEPTPContextForTesting());
        EXPECT_FALSE(
            expert_stage->usesGraphPhasedCurrentBatchLLEPForTesting());
        EXPECT_TRUE(expert_stage->supportsRequestedRoutedAssignmentPolicyForTesting());
    }

    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPApportionedLeastLoadedAssignmentIsStampedOntoFastExpertStage)
    {
        auto plan = makeLocalTPApportionedHotPlan();
        ASSERT_FALSE(plan->domains.empty());
        plan->domains[0].routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        GraphConfig config = makeConfig(plan);
        config.default_device = DeviceId::rocm(0);
        config.moe.has_shared_expert = true;
        config.moe.shared_intermediate_size = kIntermediate;
        config.moe.routed_prefill_config
            .least_loaded_min_routed_rows = 0;
        config.moe.routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);
        tp_ctx.setRawAllgatherGraphCaptureSupported(true);
        config.tp_ctx = &tp_ctx;
        config.tp_device_idx = 0;

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        layer.shared_expert_gate =
            weight_arena.fp32({kIntermediate, kDModel});
        layer.shared_expert_up =
            weight_arena.fp32({kIntermediate, kDModel});
        layer.shared_expert_down =
            weight_arena.fp32({kDModel, kIntermediate});
        layer.shared_expert_gate_inp =
            weight_arena.fp32({1, kDModel});
        TensorArena activation_arena;
        auto buffers = makeActivationBuffers(activation_arena);

        auto model_ctx =
            makeTestingModelContextWithHotDomainExperts(config.n_layers);
        Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
        ScopedDevicePublicationStream publication_stream(DeviceId::rocm(0));
        ComputeGraph graph = graph_builder.buildFFNGraph(
            layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::rocm(0),
            publication_stream.get());

        const auto *expert_node = graph.getNode("layer0_moe_expert_ffn_overlay_fast");
        ASSERT_NE(expert_node, nullptr);
        const auto *expert_stage = dynamic_cast<const MoEExpertComputeStage *>(expert_node->stage.get());
        ASSERT_NE(expert_stage, nullptr);
        EXPECT_EQ(expert_stage->routedExpertAssignmentPolicyForTesting(),
                  RoutedExpertAssignmentPolicy::LeastLoadedResident)
            << "assignment=least-loaded-resident must reach the production expert stage; otherwise "
               "the graph can silently execute StaticOwner under an LLEP label.";
        EXPECT_TRUE(expert_stage->usesRuntimeRowGroupingForTesting())
            << "LLEP prefill must use runtime grouping so the device-side "
               "route-participant assignment kernel is reachable in production graphs.";
        EXPECT_TRUE(expert_stage->hasPrefillLLEPTPContextForTesting())
            << "LLEP prefill must carry its LocalTP context so full "
               "current-batch row exchange can use grouped NCCL/RCCL collectives.";
        EXPECT_EQ(
            expert_stage->prefillLLEPAssignmentModeForTesting(),
            PrefillLLEPAssignmentMode::GraphPhasedCurrentBatch);
        EXPECT_TRUE(
            expert_stage->usesGraphPhasedCurrentBatchLLEPForTesting())
            << "LLEP expert compute must consume the assignment published by "
               "the explicit plan/sideband/apply graph transaction.";
        EXPECT_TRUE(expert_stage->supportsRequestedRoutedAssignmentPolicyForTesting());

        const auto *child_runtime =
            dynamic_cast<const DeviceMoERuntimeTable *>(
                expert_stage->moeRuntimeTableForTesting());
        ASSERT_NE(child_runtime, nullptr);
        ASSERT_TRUE(child_runtime->usesOverlayEpochTicket());
        const auto *parent_runtime = child_runtime->overlayPlacementSource();
        ASSERT_NE(parent_runtime, nullptr)
            << "Current-batch LLEP must lower onto a request-local child of "
               "the canonical ExpertOverlay runtime table";
        EXPECT_NE(child_runtime, parent_runtime);
        EXPECT_TRUE(parent_runtime->usesOverlayEpochTicket());
        EXPECT_EQ(parent_runtime->overlayPlacementSource(), nullptr);
        EXPECT_EQ(child_runtime->overlayEpochTicket(),
                  parent_runtime->overlayEpochTicket())
            << "The child transaction must pin the exact durable epoch used "
               "by the main production graph";
        const auto &child_layer = child_runtime->hostLayerState(0);
        EXPECT_EQ(child_layer.overlay_placement_banks,
                  parent_runtime->devicePlacementBanks(0));
        EXPECT_EQ(child_layer.current_batch_llep_transient_bank_active, 0u)
            << "The child must begin by reading its canonical parent; only a "
               "successful payload apply may publish a private override";
    }

    /**
     * @brief Require current-batch LLEP payload movement to ride a real model collective.
     *
     * The router-dependent LLEP plan cannot sideband the preceding attention
     * allreduce, while the routed-result collective is too late because foreign
     * expert rows consume the imported weights. Qwen's input-parallel shared
     * expert supplies the causally valid anchor: plan and pack are explicit
     * graph work, its allreduce carries the packed allgather sideband, and a
     * separate apply node publishes executable assignments before routed FFN.
     * This test locks down that production topology without executing a fake
     * collective or accepting the former standalone allgather hidden inside
     * MoEExpertComputeStage.
     */
    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPLLEPPayloadSidebandsSharedExpertAllreduceBeforeRoutedCompute)
    {
        auto plan = makeLocalTPApportionedHotPlan();
        ASSERT_FALSE(plan->domains.empty());
        plan->domains[0].routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        GraphConfig config = makeConfig(plan);
        config.default_device = DeviceId::rocm(0);
        config.moe.has_shared_expert = true;
        config.moe.shared_intermediate_size = kIntermediate;
        config.moe.routed_prefill_config.least_loaded_min_routed_rows = 0;
        config.moe.routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices(
            {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);
        tp_ctx.setRawAllgatherGraphCaptureSupported(true);
        config.tp_ctx = &tp_ctx;
        config.tp_device_idx = 0;

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        layer.shared_expert_gate =
            weight_arena.fp32({kIntermediate, kDModel});
        layer.shared_expert_up =
            weight_arena.fp32({kIntermediate, kDModel});
        layer.shared_expert_down =
            weight_arena.fp32({kDModel, kIntermediate});
        layer.shared_expert_gate_inp =
            weight_arena.fp32({1, kDModel});

        TensorArena activation_arena;
        auto buffers = makeActivationBuffers(activation_arena);
        auto model_ctx =
            makeTestingModelContextWithHotDomainExperts(config.n_layers);
        Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
        ScopedDevicePublicationStream publication_stream(DeviceId::rocm(0));
        ComputeGraph graph = graph_builder.buildFFNGraph(
            layer,
            buffers,
            /*layer_idx=*/0,
            kSeqLen,
            kBatchSize,
            DeviceId::rocm(0),
            publication_stream.get());

        constexpr const char *kPlanNode =
            "layer0_moe_current_batch_llep_plan_pack";
        constexpr const char *kAnchorNode =
            "layer0_shared_expert_allreduce";
        constexpr const char *kApplyNode =
            "layer0_moe_current_batch_llep_unpack_apply_assign";
        constexpr const char *kExpertNode =
            "layer0_moe_expert_ffn_overlay_fast";

        ASSERT_NE(graph.getNode(kPlanNode), nullptr)
            << "Current-batch planning and payload packing must be an explicit graph producer";
        ASSERT_NE(graph.getNode(kAnchorNode), nullptr)
            << "LLEP must select independent shared publication so its payload has a causal anchor";
        ASSERT_NE(graph.getNode(kApplyNode), nullptr)
            << "Payload arrival and route assignment must be visible to graph scheduling";
        ASSERT_NE(graph.getNode(kExpertNode), nullptr);

        EXPECT_TRUE(hasDependency(graph, kPlanNode, "layer0_moe_routing"));
        EXPECT_TRUE(hasDependency(graph, kAnchorNode, kPlanNode));
        EXPECT_TRUE(hasDependency(
            graph, kAnchorNode, "layer0_shared_expert_ffn"));
        EXPECT_TRUE(hasDependency(graph, kApplyNode, kAnchorNode));
        EXPECT_TRUE(hasDependency(graph, kExpertNode, kApplyNode));

        const auto *anchor_stage = dynamic_cast<const TPAllreduceStage *>(
            graph.getNode(kAnchorNode)->stage.get());
        ASSERT_NE(anchor_stage, nullptr);
        const auto &sidebands =
            anchor_stage->sidebandWorkspaceBindings();
        const auto payload = std::find_if(
            sidebands.begin(),
            sidebands.end(),
            [](const TPAllreduceSidebandWorkspaceBinding &binding)
            {
                return binding.name ==
                       "moe_current_batch_llep_payload";
            });
        ASSERT_NE(payload, sidebands.end())
            << "Shared publication must carry the packed LLEP payload in its grouped collective";
        EXPECT_EQ(payload->kind, LocalTPCollectiveSidebandKind::Allgather);
        EXPECT_EQ(payload->dtype, CollectiveDataType::INT8);
        EXPECT_GT(payload->element_count, 0u);
        EXPECT_FALSE(payload->send_buffer_name.empty());
        EXPECT_FALSE(payload->recv_buffer_name.empty());

        EXPECT_EQ(
            graph.getNode("layer0_moe_canonical_publication_reduce_to_root"),
            nullptr)
            << "The routed and shared branches cannot share the late rank-bank collective when LLEP needs the shared branch as an earlier anchor";
        EXPECT_NE(
            graph.getNode("layer0_moe_canonical_routes_reduce_to_root"),
            nullptr)
            << "Routed contributions still require their exact ordered publication after expert compute";
    }

    /**
     * @brief Prove grouped verifier rows cannot inherit long-prefill migration.
     *
     * The production LLEP movement probe deliberately sets the routed-row
     * threshold to zero and requests full compact transport. That environment
     * must still leave MTP verifier rows on the canonical decode assignment:
     * moving an expert payload in every MoE layer for a three-row verifier batch
     * is not economical, and treating grouped verifier M as prefill obscures the
     * serial-row-equivalence contract. Prefix rehydration has an independent
     * graph-build transaction and is unaffected by this assignment assertion.
    */
    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPGroupedVerifierUsesDecodeAssignmentAndNeverPrefillLLEP)
    {
        auto plan = makeLocalTPApportionedHotPlan();
        ASSERT_FALSE(plan->domains.empty());
        plan->domains[0].routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        plan->domains[0].routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        GraphConfig config = makeConfig(plan);
        config.default_device = DeviceId::rocm(0);
        config.moe.has_shared_expert = true;
        config.moe.shared_intermediate_size = kIntermediate;
        config.moe.routed_prefill_config
            .least_loaded_min_routed_rows = 0;
        config.grouped_mtp_verifier = true;
        config.compute_all_position_logits = true;
        config.moe.routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        config.moe.routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices(
            {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);
        tp_ctx.setRawAllgatherGraphCaptureSupported(true);
        config.tp_ctx = &tp_ctx;
        config.tp_device_idx = 0;

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        TensorArena activation_arena;
        auto buffers = makeActivationBuffers(activation_arena);

        auto model_ctx =
            makeTestingModelContextWithHotDomainExperts(config.n_layers);
        Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
        ScopedDevicePublicationStream publication_stream(DeviceId::rocm(0));
        INT32Tensor absolute_positions(
            {static_cast<size_t>(kSeqLen)});
        for (int row = 0; row < kSeqLen; ++row)
            absolute_positions.mutable_int32_data()[row] = 1000 + row;
        ASSERT_TRUE(
            absolute_positions.ensureOnDevice(
                DeviceId::rocm(0),
                publication_stream.get()));
        const auto *absolute_positions_device =
            static_cast<const int32_t *>(
                absolute_positions.gpu_data_ptr());
        ASSERT_NE(absolute_positions_device, nullptr);
        ComputeGraph graph = graph_builder.buildFFNGraph(
            layer,
            buffers,
            /*layer_idx=*/0,
            /*seq_len=*/3,
            kBatchSize,
            DeviceId::rocm(0),
            publication_stream.get(),
            /*sequence_lengths_device=*/nullptr,
            absolute_positions_device);

        const auto *expert_node =
            graph.getNode("layer0_moe_expert_ffn_overlay_fast");
        ASSERT_NE(expert_node, nullptr);
        const auto *expert_stage =
            dynamic_cast<const MoEExpertComputeStage *>(
                expert_node->stage.get());
        ASSERT_NE(expert_stage, nullptr);

        EXPECT_EQ(
            expert_stage->routedExpertAssignmentPolicyForTesting(),
            RoutedExpertAssignmentPolicy::StaticOwner);
        EXPECT_TRUE(expert_stage->usesRuntimeRowGroupingForTesting())
            << "Grouped verifier rows must use the economical grouped runtime-table kernel; "
               "the static-owner assignment below is what keeps that kernel out of current-batch LLEP.";
        EXPECT_FALSE(expert_stage->hasPrefillLLEPTPContextForTesting());
        EXPECT_FALSE(
            expert_stage->usesGraphPhasedCurrentBatchLLEPForTesting())
            << "Grouped verifier rows must never enter current-batch prefill "
               "assignment or expert-payload transport, even when ordinary "
               "prefill uses least-loaded assignment.";
        EXPECT_TRUE(
            expert_stage->supportsRequestedRoutedAssignmentPolicyForTesting());
    }

    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPApportionedLeastLoadedPrefillTransferWorkspacesUseBoundedRollingLanes)
    {
        constexpr int kLayerCount = 3;
        auto plan = makeLocalTPApportionedHotPlan(kLayerCount);
        ASSERT_FALSE(plan->domains.empty());
        plan->domains[0].routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        GraphConfig config = makeConfig(plan, kLayerCount);
        config.default_device = DeviceId::rocm(0);
        config.moe.has_shared_expert = true;
        config.moe.shared_intermediate_size = kIntermediate;
        config.moe.routed_prefill_config
            .least_loaded_min_routed_rows = 0;
        config.moe.routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);
        tp_ctx.setRawAllgatherGraphCaptureSupported(true);
        config.tp_ctx = &tp_ctx;
        config.tp_device_idx = 0;

        TensorArena weight_arena;
        auto layer_weights = makeLayerWeights(weight_arena);
        layer_weights.shared_expert_gate =
            weight_arena.fp32({kIntermediate, kDModel});
        layer_weights.shared_expert_up =
            weight_arena.fp32({kIntermediate, kDModel});
        layer_weights.shared_expert_down =
            weight_arena.fp32({kDModel, kIntermediate});
        layer_weights.shared_expert_gate_inp =
            weight_arena.fp32({1, kDModel});
        TensorArena activation_arena0;
        auto buffers0 = makeActivationBuffers(activation_arena0);
        TensorArena activation_arena1;
        auto buffers1 = makeActivationBuffers(activation_arena1);
        TensorArena activation_arena2;
        auto buffers2 = makeActivationBuffers(activation_arena2);

        auto model_ctx = makeTestingModelContextWithHotDomainExperts(kLayerCount);
        Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
        ScopedDevicePublicationStream publication_stream(DeviceId::rocm(0));
        ComputeGraph graph0 = graph_builder.buildFFNGraph(
            layer_weights, buffers0, 0, kSeqLen, kBatchSize, DeviceId::rocm(0),
            publication_stream.get());
        ComputeGraph graph1 = graph_builder.buildFFNGraph(
            layer_weights, buffers1, 1, kSeqLen, kBatchSize, DeviceId::rocm(0),
            publication_stream.get());
        ComputeGraph graph2 = graph_builder.buildFFNGraph(
            layer_weights, buffers2, 2, kSeqLen, kBatchSize, DeviceId::rocm(0),
            publication_stream.get());

        const auto *stage0 = gpuCurrentBatchLLEPStage(
            graph0, "layer0_moe_current_batch_llep_plan_pack");
        const auto *stage1 = gpuCurrentBatchLLEPStage(
            graph1, "layer1_moe_current_batch_llep_plan_pack");
        const auto *stage2 = gpuCurrentBatchLLEPStage(
            graph2, "layer2_moe_current_batch_llep_plan_pack");
        const auto *apply0 = gpuCurrentBatchLLEPStage(
            graph0, "layer0_moe_current_batch_llep_unpack_apply_assign");
        const auto *apply1 = gpuCurrentBatchLLEPStage(
            graph1, "layer1_moe_current_batch_llep_unpack_apply_assign");
        const auto *apply2 = gpuCurrentBatchLLEPStage(
            graph2, "layer2_moe_current_batch_llep_unpack_apply_assign");
        ASSERT_NE(stage0, nullptr);
        ASSERT_NE(stage1, nullptr);
        ASSERT_NE(stage2, nullptr);
        ASSERT_NE(apply0, nullptr);
        ASSERT_NE(apply1, nullptr);
        ASSERT_NE(apply2, nullptr);
        EXPECT_EQ(stage0->params().phase, GPUCurrentBatchLLEPPhase::PlanAndPack);
        EXPECT_EQ(stage1->params().phase, GPUCurrentBatchLLEPPhase::PlanAndPack);
        EXPECT_EQ(stage2->params().phase, GPUCurrentBatchLLEPPhase::PlanAndPack);
        EXPECT_EQ(
            apply0->params().phase,
            GPUCurrentBatchLLEPPhase::UnpackApplyAndAssign);

        const std::string workspace0 = stage0->params().workspace_name;
        const std::string workspace1 = stage1->params().workspace_name;
        const std::string workspace2 = stage2->params().workspace_name;
        EXPECT_EQ(workspace0, apply0->params().workspace_name);
        EXPECT_EQ(workspace1, apply1->params().workspace_name);
        EXPECT_EQ(workspace2, apply2->params().workspace_name);
        EXPECT_NE(workspace0, workspace1)
            << "Adjacent layers retain independent payload lanes while their "
               "shared-expert collective transactions are in flight.";
        EXPECT_EQ(workspace0, workspace2)
            << "The graph must bound persistent transfer storage instead of "
               "allocating one expert payload arena per model layer.";
        EXPECT_NE(workspace0.find("prefill_lane=0"), std::string::npos);
        EXPECT_NE(workspace1.find("prefill_lane=1"), std::string::npos);
        EXPECT_NE(workspace2.find("prefill_lane=0"), std::string::npos);

        auto publication_buffer_name =
            [](const MoEGPUCurrentBatchLLEPStage *stage,
               const char *buffer_family) -> std::string
        {
            const auto requirements =
                stage->getWorkspaceRequirements(kSeqLen);
            const auto buffer = std::find_if(
                requirements.buffers.begin(),
                requirements.buffers.end(),
                [buffer_family](const WorkspaceDescriptor &descriptor)
                {
                    return descriptor.name.rfind(buffer_family, 0) == 0;
                });
            return buffer == requirements.buffers.end()
                       ? std::string{}
                       : buffer->name;
        };
        const std::string status0 =
            publication_buffer_name(
                stage0,
                MoEDeviceRebalanceStage::WS_STATUS);
        const std::string status1 =
            publication_buffer_name(
                stage1,
                MoEDeviceRebalanceStage::WS_STATUS);
        const std::string status2 =
            publication_buffer_name(
                stage2,
                MoEDeviceRebalanceStage::WS_STATUS);
        const std::string apply_status0 =
            publication_buffer_name(
                stage0,
                MoEDeviceRebalanceStage::WS_APPLY_STATUS);
        const std::string apply_status1 =
            publication_buffer_name(
                stage1,
                MoEDeviceRebalanceStage::WS_APPLY_STATUS);
        const std::string apply_status2 =
            publication_buffer_name(
                stage2,
                MoEDeviceRebalanceStage::WS_APPLY_STATUS);
        ASSERT_FALSE(status0.empty());
        ASSERT_FALSE(status1.empty());
        ASSERT_FALSE(status2.empty());
        ASSERT_FALSE(apply_status0.empty());
        ASSERT_FALSE(apply_status1.empty());
        ASSERT_FALSE(apply_status2.empty());
        EXPECT_NE(status0, status1);
        EXPECT_NE(status0, status2)
            << "Tiny transfer-status publications are layer-owned even when "
               "their bulk payload lanes roll over.";
        EXPECT_NE(status1, status2);
        EXPECT_NE(apply_status0, apply_status1);
        EXPECT_NE(apply_status0, apply_status2)
            << "Arrival publication must not alias across model layers.";
        EXPECT_NE(apply_status1, apply_status2);

        EXPECT_FALSE(stage0->isCollectiveStage());
        EXPECT_FALSE(stage1->isCollectiveStage());
        EXPECT_FALSE(stage2->isCollectiveStage())
            << "The explicit phases must not regain a hidden standalone allgather.";
        EXPECT_EQ(
            stage0->graphLaunchPreparationPolicy(),
            GraphLaunchPreparationPolicy::CaptureOnly);
        EXPECT_EQ(
            stage1->graphLaunchPreparationPolicy(),
            GraphLaunchPreparationPolicy::CaptureOnly);
        EXPECT_EQ(
            stage2->graphLaunchPreparationPolicy(),
            GraphLaunchPreparationPolicy::CaptureOnly)
            << "Every graph-phased prefill stage must preflight its backend "
               "kernel before CUDA/HIP capture begins.";
    }

    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPApportionedLeastLoadedSingleTokenDecodeIsSupported)
    {
        auto plan = makeLocalTPApportionedHotPlan();
        ASSERT_FALSE(plan->domains.empty());
        plan->domains[0].routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        GraphConfig config = makeConfig(plan);
        config.default_device = DeviceId::rocm(0);
        config.moe.routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);
        config.tp_ctx = &tp_ctx;
        config.tp_device_idx = 0;

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        TensorArena activation_arena;
        auto buffers = makeActivationBuffers(activation_arena);

        auto model_ctx = makeTestingModelContextWithHotDomainExperts();
        Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
        ScopedDevicePublicationStream publication_stream(DeviceId::rocm(0));
        ComputeGraph graph = graph_builder.buildFFNGraph(
            layer, buffers, 0, 1, kBatchSize, DeviceId::rocm(0),
            publication_stream.get());

        const auto *expert_node = graph.getNode("layer0_moe_expert_ffn_overlay_fast");
        ASSERT_NE(expert_node, nullptr);
        const auto *expert_stage = dynamic_cast<const MoEExpertComputeStage *>(expert_node->stage.get());
        ASSERT_NE(expert_stage, nullptr);
        EXPECT_EQ(expert_stage->routedExpertAssignmentPolicyForTesting(),
                  RoutedExpertAssignmentPolicy::LeastLoadedResident);
        EXPECT_FALSE(expert_stage->usesRuntimeRowGroupingForTesting())
            << "Single-token decode uses the runtime decode table, not the multi-token prefill grouper.";
        EXPECT_TRUE(expert_stage->hasMoERuntimeTableForTesting())
            << "LLEP decode must be given the runtime placement table; without it "
               "production decode fails closed before the device-routed path can run.";
        EXPECT_TRUE(expert_stage->supportsRequestedRoutedAssignmentPolicyForTesting())
            << "LLEP decode must be allowed to enter the existing device-routed "
               "runtime table path instead of failing before executeSingleToken().";
    }

    /**
     * @brief Prove rooted LocalTP MoE publication has total verifier-M lowering.
     *
     * Every participant must lower the same publish/reduce/finalize/broadcast
     * sequence for every grouped verifier row count. The rooted payload contains
     * both router-ordered route slots and rank-addressed shared banks, while the
     * root-only finalizer and compact output broadcast scale from the graph's
     * actual M. Retaining an M=1 count in a reused verifier graph would silently
     * publish stale rows even though collective ordering remained valid.
     *
     * M=1..16 covers serial decode and every currently supported speculative
     * depth.  M=31 is the same deeper sentinel used by the grouped-verifier
     * kernel sweeps to prove that sixteen rows are not an implementation limit.
     */
    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPApportionedRootedPublicationIsSymmetricAndMTotal)
    {
        auto plan = makeLocalTPApportionedHotPlan();
        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);

        GraphConfig config0 = makeConfig(plan);
        config0.default_device = DeviceId::rocm(0);
        config0.moe.has_shared_expert = true;
        config0.moe.shared_intermediate_size = kIntermediate;
        config0.tp_ctx = &tp_ctx;
        config0.tp_device_idx = 0;

        GraphConfig config1 = makeConfig(plan);
        config1.default_device = DeviceId::rocm(1);
        config1.moe.has_shared_expert = true;
        config1.moe.shared_intermediate_size = kIntermediate;
        config1.tp_ctx = &tp_ctx;
        config1.tp_device_idx = 1;

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        layer.shared_expert_gate =
            weight_arena.fp32({kIntermediate, kDModel});
        layer.shared_expert_up =
            weight_arena.fp32({kIntermediate, kDModel});
        layer.shared_expert_down =
            weight_arena.fp32({kDModel, kIntermediate});
        layer.shared_expert_gate_inp =
            weight_arena.fp32({1, kDModel});
        auto model_ctx = makeTestingModelContextWithHotDomainExperts();
        ScopedDevicePublicationStream publication_stream0(DeviceId::rocm(0));
        ScopedDevicePublicationStream publication_stream1(DeviceId::rocm(1));

        std::vector<int> verifier_rows;
        for (int m = 1; m <= 16; ++m)
            verifier_rows.push_back(m);
        verifier_rows.push_back(31);

        for (const int m : verifier_rows)
        {
            SCOPED_TRACE("verifier_rows=" + std::to_string(m));

            TensorArena activation_arena0;
            auto buffers0 = makeActivationBuffers(activation_arena0, m);
            TensorArena activation_arena1;
            auto buffers1 = makeActivationBuffers(activation_arena1, m);

            Qwen35MoEGraph graph_builder0(model_ctx, nullptr, config0);
            ComputeGraph graph0 = graph_builder0.buildFFNGraph(
                layer, buffers0, 0, m, kBatchSize, DeviceId::rocm(0),
                publication_stream0.get());

            Qwen35MoEGraph graph_builder1(model_ctx, nullptr, config1);
            ComputeGraph graph1 = graph_builder1.buildFFNGraph(
                layer, buffers1, 0, m, kBatchSize, DeviceId::rocm(1),
                publication_stream1.get());

            const auto rooted_collectives0 =
                stageNamesOfType(graph0, ComputeStageType::ROOTED_COLLECTIVE);
            const auto rooted_collectives1 =
                stageNamesOfType(graph1, ComputeStageType::ROOTED_COLLECTIVE);
            ASSERT_EQ(rooted_collectives0.size(), 2u);
            ASSERT_EQ(rooted_collectives1.size(), 2u);
            EXPECT_EQ(rooted_collectives0, rooted_collectives1)
                << "Every participant must enter identical rooted collectives";
            EXPECT_EQ(
                rooted_collectives0.front(),
                "layer0_moe_canonical_publication_reduce_to_root");
            EXPECT_EQ(
                rooted_collectives0.back(),
                "layer0_moe_canonical_publication_broadcast");
            EXPECT_TRUE(stageNamesOfType(graph0, ComputeStageType::ALLREDUCE).empty());
            EXPECT_TRUE(stageNamesOfType(graph1, ComputeStageType::ALLREDUCE).empty());
            EXPECT_EQ(graph0.getNode("layer0_shared_expert_allreduce"), nullptr);
            EXPECT_EQ(graph1.getNode("layer0_shared_expert_allreduce"), nullptr);
            EXPECT_EQ(graph0.getNode("layer0_shared_expert_gate"), nullptr);
            EXPECT_EQ(graph1.getNode("layer0_shared_expert_gate"), nullptr);
            EXPECT_EQ(graph0.getNode("layer0_moe_combine"), nullptr);
            EXPECT_EQ(graph1.getNode("layer0_moe_combine"), nullptr);

            const auto *rooted_reduce0 =
                dynamic_cast<const TPLocalRootedCollectiveStage *>(
                    graph0.getNode(
                              "layer0_moe_canonical_publication_reduce_to_root")
                        ->stage.get());
            const auto *rooted_reduce1 =
                dynamic_cast<const TPLocalRootedCollectiveStage *>(
                    graph1.getNode(
                              "layer0_moe_canonical_publication_reduce_to_root")
                        ->stage.get());
            const auto *broadcast0 =
                dynamic_cast<const TPLocalRootedCollectiveStage *>(
                    graph0.getNode(
                              "layer0_moe_canonical_publication_broadcast")
                        ->stage.get());
            const auto *broadcast1 =
                dynamic_cast<const TPLocalRootedCollectiveStage *>(
                    graph1.getNode(
                              "layer0_moe_canonical_publication_broadcast")
                        ->stage.get());
            ASSERT_NE(rooted_reduce0, nullptr);
            ASSERT_NE(rooted_reduce1, nullptr);
            ASSERT_NE(broadcast0, nullptr);
            ASSERT_NE(broadcast1, nullptr);

            const size_t expected_publication_elements =
                static_cast<size_t>(m) * (kTopK + 2u) * kDModel;
            const size_t expected_output_elements =
                static_cast<size_t>(m) * kDModel;
            EXPECT_EQ(
                rooted_reduce0->params().count,
                expected_publication_elements);
            EXPECT_EQ(
                rooted_reduce1->params().count,
                expected_publication_elements);
            EXPECT_EQ(broadcast0->params().count, expected_output_elements);
            EXPECT_EQ(broadcast1->params().count, expected_output_elements);
            EXPECT_EQ(
                broadcast0->params().tensor,
                buffers0.attn_proj);
            EXPECT_EQ(
                broadcast1->params().tensor,
                buffers1.attn_proj);
            EXPECT_EQ(rooted_reduce0->params().root_device_index, 0);
            EXPECT_EQ(rooted_reduce1->params().root_device_index, 0);
            EXPECT_EQ(broadcast0->params().root_device_index, 0);
            EXPECT_EQ(broadcast1->params().root_device_index, 0);

            const auto *publisher0 =
                dynamic_cast<const MoESharedExpertRankBankPublishStage *>(
                    graph0.getNode("layer0_moe_shared_rank_bank_publish")
                        ->stage.get());
            const auto *publisher1 =
                dynamic_cast<const MoESharedExpertRankBankPublishStage *>(
                    graph1.getNode("layer0_moe_shared_rank_bank_publish")
                        ->stage.get());
            ASSERT_NE(publisher0, nullptr);
            ASSERT_NE(publisher1, nullptr);
            EXPECT_EQ(publisher0->params().seq_len, m);
            EXPECT_EQ(publisher1->params().seq_len, m);
            EXPECT_EQ(publisher0->params().participant_device_index, 0);
            EXPECT_EQ(publisher1->params().participant_device_index, 1);
            EXPECT_EQ(publisher0->params().participant_count, 2);
            EXPECT_EQ(publisher1->params().participant_count, 2);
            EXPECT_TRUE(hasDependency(
                graph0,
                "layer0_moe_shared_rank_bank_publish",
                "layer0_shared_expert_ffn"));
            EXPECT_TRUE(hasDependency(
                graph0,
                "layer0_moe_shared_rank_bank_publish",
                "layer0_moe_expert_ffn_overlay_fast"));
            EXPECT_TRUE(hasDependency(
                graph0,
                "layer0_moe_canonical_publication_reduce_to_root",
                "layer0_moe_shared_rank_bank_publish"));

            const auto *finalizer0 =
                dynamic_cast<const MoECanonicalPublicationFinalizeStage *>(
                    graph0.getNode(
                              "layer0_moe_canonical_publication_finalize")
                        ->stage.get());
            const auto *finalizer1 =
                dynamic_cast<const MoECanonicalPublicationFinalizeStage *>(
                    graph1.getNode(
                              "layer0_moe_canonical_publication_finalize")
                        ->stage.get());
            ASSERT_NE(finalizer0, nullptr);
            ASSERT_NE(finalizer1, nullptr);
            EXPECT_EQ(finalizer0->params().seq_len, m);
            EXPECT_EQ(finalizer1->params().seq_len, m);
            EXPECT_EQ(finalizer0->params().participant_device_index, 0);
            EXPECT_EQ(finalizer1->params().participant_device_index, 1);
            EXPECT_EQ(finalizer0->params().root_device_index, 0);
            EXPECT_EQ(finalizer1->params().root_device_index, 0);
            EXPECT_EQ(finalizer0->params().participant_count, 2);
            EXPECT_EQ(finalizer1->params().participant_count, 2);
            EXPECT_FALSE(finalizer0->bufferContract().empty())
                << "The root graph owns the complete fixed-order epilogue";
            EXPECT_TRUE(finalizer1->bufferContract().empty())
                << "The non-root graph must not claim root-only output bytes";
            EXPECT_EQ(finalizer0->coherencePolicy(), CoherencePolicy::FULL);
            EXPECT_EQ(finalizer1->coherencePolicy(), CoherencePolicy::NONE);
            EXPECT_GT(finalizer0->estimatedFlops(), 0u);
            EXPECT_EQ(finalizer1->estimatedFlops(), 0u);
            EXPECT_TRUE(hasDependency(
                graph0,
                "layer0_moe_canonical_publication_finalize",
                "layer0_moe_canonical_publication_reduce_to_root"));
            EXPECT_TRUE(hasDependency(
                graph0,
                "layer0_moe_canonical_publication_broadcast",
                "layer0_moe_canonical_publication_finalize"));
            EXPECT_EQ(
                countStagesOfType(
                    graph0,
                    ComputeStageType::MOE_SHARED_RANK_BANK_PUBLISH),
                1u);
            EXPECT_EQ(
                countStagesOfType(
                    graph0,
                    ComputeStageType::MOE_CANONICAL_PUBLICATION_FINALIZE),
                1u);
            EXPECT_EQ(
                countStagesOfType(
                    graph0,
                    ComputeStageType::MOE_CANONICAL_ROUTE_REDUCE),
                0u);

            const auto *expert_stage0 =
                expertComputeStage(graph0, "layer0_moe_expert_ffn_overlay_fast");
            const auto *expert_stage1 =
                expertComputeStage(graph1, "layer0_moe_expert_ffn_overlay_fast");
            ASSERT_NE(expert_stage0, nullptr);
            ASSERT_NE(expert_stage1, nullptr);
            EXPECT_EQ(expert_stage0->fixedTopologyPrefillExpertIdsForTesting(),
                      (std::vector<int>{0, 1, 2}));
            EXPECT_EQ(expert_stage1->fixedTopologyPrefillExpertIdsForTesting(),
                      (std::vector<int>{3, 4, 5}));
        }
    }

} // namespace llaminar2::test
