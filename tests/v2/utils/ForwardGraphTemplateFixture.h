/**
 * @file ForwardGraphTemplateFixture.h
 * @brief Real tiny CPU Qwen construction inputs shared by tests and benchmark.
 *
 * All source and activation storage exists before timing. The fixture calls
 * the actual production builder without executing or initializing any stage;
 * it never substitutes a fabricated topology, compiler object or stage list.
 */
#pragma once
#include "utils/CPUForwardSamplingTestFixture.h"
#include "models/qwen35moe/Qwen35MoEGraph.h"
#include "execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/moe/MoELayeredExpertOwnership.h"
#include "backends/BackendManager.h"
#include "mocks/MockLocalTPContext.h"
#include "loaders/ExpertGemmRegistry.h"
#include "kernels/KernelFactory.h"
#include "memory/CPUSerialExpertStorageLayout.h"
#include <array>
#include <memory>
#include <vector>

namespace llaminar2::test
{
    /** @brief Own all real dense weights and buffers used by actual graph construction. */
    struct ForwardTemplateConstructionFixture
    {
        TinyQwenForwardFixture source{DeviceId::cpu(), KVCachePrecision::FP32};
        std::vector<std::unique_ptr<FP32Tensor>> storage;
        ModelBuffers buffers;
        std::array<int, 8> tokens{1, 2, 3, 4, 5, 6, 7, 8};
        std::array<int, 8> positions{0, 1, 2, 3, 4, 5, 6, 7};

        /** @brief Allocate fixture storage before the measured construction interval. */
        ForwardTemplateConstructionFixture()
        {
            initCPUBackend(-1);
            source.config.mtp.enabled = false;
            source.config.max_seq_len = 8;
            auto make = [&](std::size_t columns) {
                storage.push_back(TestTensorFactory::createFP32({8, columns}));
                return storage.back().get();
            };
            buffers.current_hidden = make(64);
            buffers.logits = make(128);
            auto &b = buffers.layer_buffers;
            b.current_hidden = buffers.current_hidden;
            b.residual = make(64); b.normalized = make(64);
            b.Q = make(64); b.K = make(32); b.V = make(32);
            b.K_full_prefill = make(32); b.V_full_prefill = make(32);
            b.attn_output = make(64); b.attn_proj = make(64);
            b.gate = make(128); b.up = make(128); b.ffn_output = make(128);
        }

        /** @return Original real layer tensors, including the complete FFN. */
        ModelWeights weights()
        {
            auto result = source.modelWeights();
            const auto original = result.get_layer_weights;
            result.get_layer_weights = [this, original](int layer) {
                auto value = original(layer);
                const auto &tensors = source.layers.at(static_cast<std::size_t>(layer));
                value.gate_proj = tensors.gate_proj.get();
                value.up_proj = tensors.up_proj.get();
                value.down_proj = tensors.down_proj.get();
                return value;
            };
            return result;
        }

        /** @return An independent real production builder with original source views. */
        std::unique_ptr<QwenStandardGraph> builder()
        {
            auto result = std::make_unique<QwenStandardGraph>(source.config, source.mpi);
            result->setWeights(weights()); result->setBuffers(buffers);
            return result;
        }

        /** @return Exact row specialization; it changes no immutable source authority. */
        ForwardInput input(int rows)
        {
            ForwardInput result;
            result.device = DeviceId::cpu(); result.batch_size = 1;
            result.seq_len = rows; result.real_seq_len = rows;
            result.token_ids = tokens.data(); result.position_ids = positions.data();
            result.execution_phase = rows == 1 ? ForwardExecutionPhase::Decode : ForwardExecutionPhase::Prefill;
            return result;
        }
    };

    /** @brief A real canonical whole-expert plan with deterministic/random owner policies. */
    inline std::shared_ptr<MoERoutedExpertPlacementPlan> forwardTemplatePlan(
        int layers = 4, int experts = 16, RoutedExpertOwnerOrder order = RoutedExpertOwnerOrder::Ordinal,
        int participants = 1)
    {
        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true; plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->continuation_domain = "cpu"; plan->shared_expert_domain = "cpu";
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->owner_order = order;
        RoutedExpertDomain domain;
        domain.name = "cpu";
        domain.scope = participants == 1 ? ExecutionDomainScope::SINGLE : ExecutionDomainScope::NODE_LOCAL;
        domain.backend = participants == 1 ? CollectiveBackendType::HOST : CollectiveBackendType::UPI;
        for (int rank = 0; rank < participants; ++rank)
        { domain.participants.push_back(GlobalDeviceAddress::cpu(rank)); domain.world_ranks.push_back(rank); }
        domain.owner_rank = 0;
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        plan->domains.push_back(domain);
        RoutedExpertTier tier; tier.name = "all"; tier.domain = "cpu"; tier.priority = 0;
        tier.fallback = true; plan->routed_tiers.push_back(tier);
        for (int layer = 0; layer < layers; ++layer)
            plan->placements.push_back({.layer = layer,
                .routed_expert_tier = std::vector<int>(static_cast<std::size_t>(experts), 0)});
        return plan;
    }
    /**
     * @brief Actual small replicated LocalTP overlay construction and real registry.
     * @details This follows the canonical Qwen35MoE graph fixture's CPU engine
     * preparation. No GPU, fake kernel, dense FFN replacement or layer-body
     * mock participates. The real graph producer resolves the original map
     * inside each routed layer, exposing any nested reconstruction in counts.
     */
    struct RoutedForwardTemplateFixture
    {
        GraphConfig config;
        std::shared_ptr<MockLocalTPContext> tp = std::make_shared<MockLocalTPContext>();
        std::shared_ptr<ModelContext> model;
        std::vector<std::shared_ptr<FP32Tensor>> tensors;
        LayerWeights layer;
        ActivationBuffers buffers;
        std::unique_ptr<PreparedWeightStore> serial_store;

        /** @brief Prepare original CPU weight engines and all scratch outside timing. */
        explicit RoutedForwardTemplateFixture(int layers = 2)
        {
            initCPUBackend(-1);
            config.n_layers = layers; config.total_n_layers = layers;
            config.d_model = 4; config.d_ff = 8;
            config.n_heads = 2; config.n_kv_heads = 2; config.head_dim = 2;
            config.max_seq_len = 4; config.max_activation_rows = 4;
            config.default_device = DeviceId::cpu();
            config.moe.num_experts = 2; config.moe.top_k = 1;
            config.moe.intermediate_size = 3; config.moe.has_shared_expert = false;
            config.moe.routed_compute_policy = RoutedExpertComputePolicy::Replicated;
            config.moe.local_expert_count = -1;
            tp->setDevices({GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)});
            tp->setBackend(CollectiveBackendType::HOST); config.tp_ctx = tp.get();
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>(*forwardTemplatePlan(layers, 2));
            auto &domain = plan->domains.front();
            domain.scope = ExecutionDomainScope::RANK_LOCAL;
            domain.participants = tp->devices(); domain.world_ranks.clear();
            domain.routed_compute_policy = RoutedExpertComputePolicy::Replicated;
            plan->base_model_domain = domain.name;
            plan->continuation_domain_spec.domain = domain.name;
            plan->continuation_domain_spec.logical_root_participant = 0;
            config.moe.routed_expert_plan = plan;
            config.moe.expert_overlay_runtime_plan = resolveMoEExpertOverlayRuntimePlan(
                plan, {.current_world_rank = 0, .validate_mvp_root_reachability = false});
            config.refreshMoEExecutionPolicy();
            model = ModelContext::createForTesting("forward-template-cpu.gguf", nullptr, layers, true);
            if (!model || !model->concreteWeightManager())
                throw std::runtime_error("Forward template fixture has no original model registry");
            auto make = [&](std::vector<std::size_t> shape) {
                tensors.push_back(std::make_shared<FP32Tensor>(std::move(shape)));
                std::fill_n(tensors.back()->mutable_data(), tensors.back()->numel(), 0.125f);
                return tensors.back().get();
            };
            layer.ffn_norm = make({4}); layer.moe_gate = make({2, 4});
            layer.moe_gate_exps = make({4, 3, 2});
            layer.moe_up_exps = make({4, 3, 2}); layer.moe_down_exps = make({3, 4, 2});
            buffers.current_hidden = make({4, 4}); buffers.attn_proj = make({4, 4});
            buffers.normalized = make({4, 4});
            for (auto [id, columns] : std::initializer_list<std::pair<BufferId, std::size_t>>{
                {BufferId::MOE_EXPERT_INDICES, 1}, {BufferId::MOE_EXPERT_WEIGHTS, 1},
                {BufferId::MOE_COMBINED_OUTPUT, 4}, {BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS, 4},
                {BufferId::MOE_GATE_SCRATCH, 2}, {BufferId::MOE_UP_SCRATCH, 2}})
                buffers.extensions[id] = make({4, columns});
            const auto owners = MoEExpertOwnerMap::build(*plan);
            auto &registry = model->concreteWeightManager()->expertGemmRegistry();
            for (const auto &participant : owners.participants())
            for (int index = 0; index < layers; ++index)
            for (int expert = 0; expert < 2; ++expert)
            for (const auto role : {ExpertGemmRegistry::WeightRole::GATE,
                    ExpertGemmRegistry::WeightRole::UP, ExpertGemmRegistry::WeightRole::DOWN})
            {
                auto source = std::make_shared<FP32Tensor>(role == ExpertGemmRegistry::WeightRole::DOWN
                    ? std::vector<std::size_t>{4, 3} : std::vector<std::size_t>{3, 4});
                std::fill_n(source->mutable_data(), source->numel(), 0.03125f * static_cast<float>(expert + 1));
                auto engine = llaminar::v2::kernels::KernelFactory::prepareExpertGemmLocal(
                    std::shared_ptr<const TensorBase>(source), DeviceId::cpu());
                if (!engine) throw std::runtime_error("Real CPU expert preparation failed");
                registry.registerEngineForParticipant(participant.domain_name, participant.device,
                    participant.world_rank_known ? participant.world_rank : -1,
                    participant.domain_participant_index, index, expert, role, engine.get(), engine);
                registry.registerEngineForDomain(participant.domain_name, participant.device,
                    index, expert, role, engine.get(), engine);
            }
        }

        /** @return Original static model layer views, retained by this fixture. */
        ModelWeights weights() const
        {
            ModelWeights result; result.get_layer_weights = [this](int) { return layer; };
            return result;
        }
        /** @return A real routed builder retaining the original model registry. */
        std::unique_ptr<Qwen35MoEGraph> builder()
        {
            auto result = std::make_unique<Qwen35MoEGraph>(model, nullptr, config);
            result->setWeights(weights());
            return result;
        }
        /**
         * @brief Build the actual general CPU overlay with admitted serial endpoint roots.
         * @details The canonical storage envelope owns packet/grouped scratch.
         * All expert engines still come from the original model registry; no
         * fake source or completion bypass is installed for this lifetime test.
         */
        std::unique_ptr<Qwen35MoEGraph> serialBuilder()
        {
            auto serial = config;
            serial.tp_ctx = nullptr;
            serial.moe.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            serial.moe.routed_expert_plan = forwardTemplatePlan(config.n_layers, 2);
            serial.moe.expert_overlay_runtime_plan = resolveMoEExpertOverlayRuntimePlan(
                serial.moe.routed_expert_plan,
                {.current_world_rank = 0, .validate_mvp_root_reachability = false});
            serial.refreshMoEExecutionPolicy();
            auto declaration = std::make_shared<CPUSerialExpertStorageAdmission>();
            declaration->allocator = {0, DeviceId::cpu()};
            declaration->participant_ids = {0}; declaration->controller_devices = {DeviceId::cpu()};
            declaration->envelope = std::make_shared<const CPUSerialExpertStorageEnvelope>(
                DeviceId::cpu(), 4, 1, config.d_model, config.moe.top_k,
                std::vector<int>{1, 2, 4}, 1, 8, true, CPUSerialExpertFamilyPolicy::LocalPowerOfTwo,
                config.moe.intermediate_size, config.moe.num_experts);
            serial.cpu_serial_expert_storage = {declaration};
            const auto packets = declaration->physicalBytes();
            const auto grouped = declaration->envelope->forController(DeviceId::cpu())->workspaceBytes();
            const auto total = CPUSerialExpertStorageLayout::add(packets, grouped);
            PhysicalMemoryPlanBuilder admission;
            const PhysicalMemoryResource resource{0, DeviceId::cpu(), total, total};
            admission.add(resource, PhysicalMemoryOwner::RoutedExpertWeights, 0);
            admission.add(resource, PhysicalMemoryOwner::ActivationArena, packets);
            admission.add(resource, PhysicalMemoryOwner::ExecutionWorkspace, grouped);
            serial_store = std::make_unique<PreparedWeightStore>();
            serial_store->installPhysicalMemoryAuthority(std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(admission.build()), 0));
            auto result = std::make_unique<Qwen35MoEGraph>(model, nullptr, serial);
            result->setWeights(weights()); result->setPreparedWeightStore(serial_store.get());
            ModelWeightBindings bindings; bindings.physical_memory_owner = PhysicalMemoryOwner::RoutedExpertWeights;
            result->setWeightBindings(bindings);
            return result;
        }

        /** @return One actual routed layer expansion with independent mutable stages. */
        ComputeGraph specialize(Qwen35MoEGraph &builder, int index, int rows)
        {
            return builder.buildFFNGraph(layer, buffers, index, rows, 1, DeviceId::cpu(), nullptr);
        }
    };

}
