/**
 * @file MoELocalExpertStage.cpp
 * @brief Implementation of participant-local graph-native sparse MoE expert compute.
 *
 * This stage consumes already-addressed sparse rows, filters them through its
 * immutable participant mask, invokes only its prepared expert engines, and
 * aggregates the resulting rows for the explicit return collective. It owns no
 * route selection or peer coordination, which keeps cross-tier ordering in the
 * graph and makes a participant's work independently observable.  The compact
 * route tensors may be stage-private or supplied by a serial graph-family
 * arena; the latter is immutable-address storage shared only after the graph
 * builder has established a non-concurrent execution contract.
 */

#include "MoELocalExpertStage.h"

#include "MoEExpertComputeStage.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../execution/moe/MoEExpertWeightService.h"
#include "../../../execution/moe/MoEExpertOverlayProfiler.h"
#include "../../../execution/moe/MoEWorkspaceRequirements.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../execution/local_execution/graph/ComputeGraph.h"
#include "../../../execution/local_execution/graph/DeviceGraphExecutor.h"
#include "../../../loaders/PreparedWeightStore.h"
#include "../../../memory/BufferArena.h"
#include "../../../tensors/Tensors.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../utils/VramBillOfMaterials.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        bool validateDeviceContext(IDeviceContext *ctx, DeviceId device, const char *stage_name)
        {
            if (!ctx)
            {
                LOG_ERROR("[" << stage_name << "] Null device context");
                return false;
            }
            (void)device;
            // Host-staged graph-native MoE stages can be scheduled inside a GPU graph
            // executor while executing participant-local CPU work. GPU local expert
            // stages must still run on the matching GPU executor; executing ROCm:1
            // kernels from a ROCm:0 context can corrupt device memory before the
            // stage reports an ordinary failure.
            if (device.is_gpu())
            {
                const DeviceId ctx_device = ctx->deviceId();
                if (!ctx_device.is_gpu() || ctx_device != device)
                {
                    LOG_ERROR("[" << stage_name << "] GPU local expert stage device "
                                  << device.to_string()
                                  << " does not match execution context "
                                  << ctx_device.to_string());
                    return false;
                }
            }
            return true;
        }

        bool validateSparseRows(const MoEOverlaySparseRows &rows, int top_k, int d_model)
        {
            if (rows.d_model != d_model || rows.top_k != top_k)
            {
                LOG_ERROR("[MoELocalExpertStage] Sparse row dimension mismatch: rows d_model="
                          << rows.d_model << " top_k=" << rows.top_k << ", stage d_model="
                          << d_model << " top_k=" << top_k);
                return false;
            }
            if (!rows.row_ids_host || !rows.entry_offsets_host || !rows.expert_ids_host ||
                !rows.route_weights_host || !rows.hidden_rows_fp32)
            {
                LOG_ERROR("[MoELocalExpertStage] Sparse input rows missing host buffers");
                return false;
            }
            if (rows.live_row_count > rows.row_capacity || rows.live_entry_count > rows.entry_capacity)
            {
                LOG_ERROR("[MoELocalExpertStage] Sparse live counts exceed capacity");
                return false;
            }
            if (rows.hidden_row_capacity == 0u ||
                !isValidMoEOverlayActivationHiddenPayloadLayout(
                    rows.hidden_payload_layout))
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Sparse input has no valid hidden-row address contract");
                return false;
            }
            if ((rows.live_row_count != 0 || rows.live_entry_count != 0) &&
                rows.residency_epoch == 0)
            {
                LOG_ERROR("[MoELocalExpertStage] Non-empty sparse input is missing its residency epoch");
                return false;
            }
            return true;
        }

        /// True when prepared_gate_gemm (size == num_experts) or slab refs + store are set.
        bool hasPreparedExpertState(const MoELocalExpertStage::Params &p)
        {
            const auto expected = static_cast<size_t>(std::max(p.num_experts, 0));
            if (expected > 0 &&
                p.prepared_gate_gemm.size() == expected &&
                p.prepared_up_gemm.size() == expected &&
                p.prepared_down_gemm.size() == expected)
                return true;
            if (p.prepared_store && p.gate_slab_ref.has_value() &&
                p.up_slab_ref.has_value() && p.down_slab_ref.has_value())
                return true;
            return false;
        }

        /** @brief Stable diagnostic spelling for one authenticated traffic phase. */
        const char *serviceSourceName(ExpertHistogramSource source) noexcept
        {
            switch (source)
            {
            case ExpertHistogramSource::DecodeToken:
                return "decode";
            case ExpertHistogramSource::PrefillChunk:
                return "prefill";
            case ExpertHistogramSource::GroupedVerifier:
                return "grouped_verifier";
            case ExpertHistogramSource::SyntheticTest:
                return "synthetic_test";
            }
            return "invalid";
        }

        /**
         * @brief Multiply two tensor dimensions without silently wrapping capacity.
         *
         * The arena allocation is model-lifetime graph identity.  Overflow
         * must fail while building that identity, not create an undersized
         * buffer that a later sparse route packet could overwrite.
         */
        size_t checkedMultiply(size_t lhs, size_t rhs, const char *what)
        {
            if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
            {
                throw std::overflow_error(
                    std::string("MoELocalExpertSerialBufferArena overflow while sizing ") +
                    what);
            }
            return lhs * rhs;
        }

        /** @brief Add tensor byte counts without silently wrapping diagnostics. */
        size_t checkedAdd(size_t lhs, size_t rhs, const char *what)
        {
            if (rhs > std::numeric_limits<size_t>::max() - lhs)
            {
                throw std::overflow_error(
                    std::string("MoELocalExpertSerialBufferArena overflow while summing ") +
                    what);
            }
            return lhs + rhs;
        }
    } // namespace

    namespace
    {
        /**
         * @brief First captured node that uploads one compact sparse packet.
         *
         * Host routing compacts directly into a family-owned pinned allocation.
         * This node embeds those fixed host addresses and the three stable
         * device tensor addresses in the retained CUDA/HIP graph. Replays
         * therefore submit all packet H2D work with the one graph launch that
         * also owns expert compute and output publication.
         */
        class MoELocalExpertInputPublishStage final : public IComputeStage
        {
        public:
            /** @brief Complete immutable input-publication identity. */
            struct Params
            {
                STAGE_PARAMS_COMMON_FIELDS;

                FP32Tensor *hidden = nullptr; ///< Compact hidden-row destination.
                FP32Tensor *routing_indices = nullptr; ///< Compact expert-id destination.
                FP32Tensor *routing_weights = nullptr; ///< Compact route-weight destination.
                std::shared_ptr<PinnedHostTransferBuffer>
                    pinned_transfer; ///< Stable H2D source and lifetime owner.
                size_t hidden_offset = 0; ///< Hidden bytes in the pinned family.
                size_t routing_indices_offset = 0; ///< Expert-id bytes in the family.
                size_t routing_weights_offset = 0; ///< Route-weight bytes in the family.
            };

            static_assert(StageParamsRequired<Params>);

            /**
             * @brief Construct a fixed-address captured input publication node.
             * @throws std::invalid_argument for inconsistent tensors/layout.
             */
            explicit MoELocalExpertInputPublishStage(Params params)
                : IComputeStage(params.device_id), params_(std::move(params))
            {
                if (!hasStaticContract())
                {
                    throw std::invalid_argument(
                        "Local expert input publisher requires three FP32 tensors and one complete pinned family layout");
                }
            }

            /** @brief Enqueue all captured H2D nodes on the exact graph stream. */
            bool execute(IDeviceContext *ctx) override
            {
                if (!ctx || ctx->deviceId() != params_.device_id ||
                    !hasFixedContract())
                {
                    LOG_ERROR(
                        "[MoELocalExpertInputPublishStage] Invalid exact device or fixed publication contract");
                    return false;
                }
                try
                {
                    TransferEngine &transfers = TransferEngine::instance();
                    void *const stream = requireGPUStream();
                    transfers.enqueuePinnedHostToDevice(
                        *params_.pinned_transfer,
                        params_.hidden_offset,
                        params_.hidden,
                        /*destination_offset=*/0,
                        params_.hidden->size_bytes(),
                        params_.device_id,
                        stream);
                    transfers.enqueuePinnedHostToDevice(
                        *params_.pinned_transfer,
                        params_.routing_indices_offset,
                        params_.routing_indices,
                        /*destination_offset=*/0,
                        params_.routing_indices->size_bytes(),
                        params_.device_id,
                        stream);
                    transfers.enqueuePinnedHostToDevice(
                        *params_.pinned_transfer,
                        params_.routing_weights_offset,
                        params_.routing_weights,
                        /*destination_offset=*/0,
                        params_.routing_weights->size_bytes(),
                        params_.device_id,
                        stream);
                    return true;
                }
                catch (const std::exception &ex)
                {
                    LOG_ERROR(
                        "[MoELocalExpertInputPublishStage] Failed to enqueue captured input publication: "
                        << ex.what());
                    return false;
                }
            }

            /** @brief Return the dedicated diagnostic operation type. */
            ComputeStageType type() const override
            {
                return ComputeStageType::MOE_LOCAL_EXPERT_INPUT_PUBLISH;
            }

            /** @brief Return the stable graph node name. */
            std::string name() const override
            {
                return "moe_local_expert_input_publish";
            }

            /** @brief Admit CUDA and ROCm as symmetric first-class implementations. */
            bool supportsBackend(ComputeBackendType backend) const override
            {
                return backend == ComputeBackendType::GPU_CUDA ||
                       backend == ComputeBackendType::GPU_ROCM;
            }

            /** @brief Report readiness only after every embedded address is fixed. */
            bool isGraphCapturable() const override
            {
                return hasFixedContract();
            }

            /** @brief Declare that setup binding completes capture readiness. */
            bool supportsGraphCaptureAfterLaunchPreparation() const override
            {
                return hasStaticContract();
            }

            /** @brief Bind and validate the executor-selected capture stream. */
            bool prepareGraphLaunch(
                IDeviceContext *ctx,
                void *stream) override
            {
                if (!ctx || ctx->deviceId() != params_.device_id || !stream ||
                    !hasFixedContract())
                {
                    LOG_ERROR(
                        "[MoELocalExpertInputPublishStage] Capture preparation requires fixed device and pinned addresses");
                    return false;
                }
                setGPUStream(stream);
                return true;
            }

            /** @brief Run launch preparation only for native graph capture. */
            GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
            {
                return GraphLaunchPreparationPolicy::CaptureOnly;
            }

            /** @brief Declare the three device tensors written by this node. */
            StageBufferContract bufferContract() const override
            {
                auto contract = StageBufferContract::build();
                contract.addOutput(BufferId::NORMALIZED, "FP32")
                    .addOutput(BufferId::MOE_EXPERT_INDICES, "FP32")
                    .addOutput(BufferId::MOE_EXPERT_WEIGHTS, "FP32");
                return contract;
            }

            /** @brief Expose the captured device destinations to diagnostics. */
            StageDumpInfo buildDumpInfoImpl() const override
            {
                StageDumpInfo info;
                info.addOutput(
                    "compact_hidden",
                    params_.hidden,
                    params_.hidden->numel(),
                    1u);
                info.addOutput(
                    "compact_routing_indices",
                    params_.routing_indices,
                    params_.routing_indices->numel(),
                    1u);
                info.addOutput(
                    "compact_routing_weights",
                    params_.routing_weights,
                    params_.routing_weights->numel(),
                    1u);
                return info;
            }

        private:
            /** @brief Validate immutable geometry without requiring live allocation. */
            [[nodiscard]] bool hasStaticContract() const noexcept
            {
                const auto fp32 = [](const FP32Tensor *tensor)
                {
                    return tensor && tensor->native_type() == TensorType::FP32 &&
                           tensor->size_bytes() > 0;
                };
                return params_.device_id.is_gpu() && fp32(params_.hidden) &&
                       fp32(params_.routing_indices) &&
                       fp32(params_.routing_weights) &&
                       params_.pinned_transfer &&
                       params_.pinned_transfer->registrationDevice() ==
                           params_.device_id &&
                       params_.pinned_transfer->contains(
                           params_.hidden_offset,
                           params_.hidden->size_bytes()) &&
                       params_.pinned_transfer->contains(
                           params_.routing_indices_offset,
                           params_.routing_indices->size_bytes()) &&
                       params_.pinned_transfer->contains(
                           params_.routing_weights_offset,
                           params_.routing_weights->size_bytes());
            }

            /** @brief Validate every device/host address embedded by capture. */
            [[nodiscard]] bool hasFixedContract() const noexcept
            {
                return hasStaticContract() &&
                       params_.pinned_transfer->isBound() &&
                       params_.hidden->gpu_data_ptr() &&
                       params_.routing_indices->gpu_data_ptr() &&
                       params_.routing_weights->gpu_data_ptr();
            }

            Params params_; ///< Complete immutable captured-copy identity.
        };

        /**
         * @brief Final captured node that publishes local expert rows to pinned memory.
         *
         * The node follows the grouped expert kernel inside the same retained
         * CUDA/HIP graph. It enqueues one fixed-capacity D2H transfer on the
         * exact capture stream; the later heterogeneous completion stage waits
         * on the graph event before reading the shared serial-family buffer.
         */
        class MoELocalExpertOutputPublishStage final : public IComputeStage
        {
        public:
            /** @brief Immutable graph construction inputs. */
            struct Params
            {
                STAGE_PARAMS_COMMON_FIELDS;

                FP32Tensor *output = nullptr; ///< Device output produced immediately before this node.
                size_t output_bytes = 0; ///< Fixed D2H byte count embedded in capture.
                size_t output_offset = 0; ///< Destination offset in the pinned family.
                std::shared_ptr<PinnedHostTransferBuffer>
                    pinned_transfer; ///< Stable host destination and lifetime owner.
            };

            static_assert(StageParamsRequired<Params>);

            /**
             * @brief Construct a fixed-address captured publication node.
             * @throws std::invalid_argument when static device or capacity identity is invalid.
             */
            explicit MoELocalExpertOutputPublishStage(Params params)
                : IComputeStage(params.device_id), params_(std::move(params))
            {
                if (!hasStaticContract())
                {
                    throw std::invalid_argument(
                        "Local expert output publisher requires a GPU, FP32 output, exact byte capacity, and pinned owner");
                }
            }

            /** @brief Enqueue the graph-captured D2H copy on the exact stage stream. */
            bool execute(IDeviceContext *ctx) override
            {
                if (!ctx || ctx->deviceId() != params_.device_id ||
                    !hasFixedContract())
                {
                    LOG_ERROR(
                        "[MoELocalExpertOutputPublishStage] Invalid exact device or fixed publication contract");
                    return false;
                }
                try
                {
                    TransferEngine::instance().enqueueDeviceToPinnedHost(
                        params_.output,
                        /*source_offset=*/0,
                        *params_.pinned_transfer,
                        params_.output_offset,
                        params_.output_bytes,
                        params_.device_id,
                        requireGPUStream());
                    return true;
                }
                catch (const std::exception &ex)
                {
                    LOG_ERROR(
                        "[MoELocalExpertOutputPublishStage] Failed to enqueue captured output publication: "
                        << ex.what());
                    return false;
                }
            }

            /** @brief Return the dedicated diagnostic operation type. */
            ComputeStageType type() const override
            {
                return ComputeStageType::MOE_LOCAL_EXPERT_OUTPUT_PUBLISH;
            }

            /** @brief Return the stable graph node name. */
            std::string name() const override
            {
                return "moe_local_expert_output_publish";
            }

            /** @brief Admit CUDA and ROCm as symmetric first-class implementations. */
            bool supportsBackend(ComputeBackendType backend) const override
            {
                return backend == ComputeBackendType::GPU_CUDA ||
                       backend == ComputeBackendType::GPU_ROCM;
            }

            /** @brief Report readiness only after every embedded address is fixed. */
            bool isGraphCapturable() const override
            {
                return hasFixedContract();
            }

            /** @brief Declare that setup-bound output storage completes capture readiness. */
            bool supportsGraphCaptureAfterLaunchPreparation() const override
            {
                return hasStaticContract();
            }

            /**
             * @brief Validate exact storage and bind the executor-selected stream.
             * @param ctx Exact participant device context.
             * @param stream Non-null borrowed capture stream.
             */
            bool prepareGraphLaunch(
                IDeviceContext *ctx,
                void *stream) override
            {
                if (!ctx || ctx->deviceId() != params_.device_id || !stream ||
                    !hasFixedContract())
                {
                    LOG_ERROR(
                        "[MoELocalExpertOutputPublishStage] Capture preparation requires fixed device and pinned addresses");
                    return false;
                }
                setGPUStream(stream);
                return true;
            }

            /** @brief Run launch preparation only for native graph capture. */
            GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
            {
                return GraphLaunchPreparationPolicy::CaptureOnly;
            }

            /** @brief Declare the exact internal output dependency to the graph planner. */
            StageBufferContract bufferContract() const override
            {
                auto contract = StageBufferContract::build();
                contract.addInput(BufferId::MOE_COMBINED_OUTPUT, "FP32");
                return contract;
            }

            /** @brief Expose the device source for optional graph diagnostics. */
            StageDumpInfo buildDumpInfoImpl() const override
            {
                StageDumpInfo info;
                info.addInput(
                    "local_expert_output",
                    params_.output,
                    params_.output->numel(),
                    1u);
                return info;
            }

        private:
            /** @brief Validate immutable construction geometry without requiring device storage. */
            [[nodiscard]] bool hasStaticContract() const noexcept
            {
                return params_.device_id.is_gpu() && params_.output &&
                       params_.output->native_type() == TensorType::FP32 &&
                       params_.output_bytes > 0 &&
                       params_.output_bytes == params_.output->size_bytes() &&
                       params_.pinned_transfer &&
                       params_.pinned_transfer->registrationDevice() ==
                           params_.device_id &&
                       params_.pinned_transfer->contains(
                           params_.output_offset,
                           params_.output_bytes);
            }

            /** @brief Validate every device and host address embedded by capture. */
            [[nodiscard]] bool hasFixedContract() const noexcept
            {
                return hasStaticContract() &&
                       params_.output->gpu_data_ptr() != nullptr &&
                       params_.pinned_transfer->isBound();
            }

            Params params_; ///< Complete immutable captured-copy identity.
        };
    } // namespace

    /**
     * @brief One immutable-capacity participant-local captured MoE transaction.
     *
     * Every member is model-lifetime state.  The graph embeds only the tensor
     * addresses owned by @ref tensor_family, while mutable expert placement is
     * published into backend-owned descriptor/mask storage before replay.  The
     * cache borrows the participant's context-owned worker stream and owns only
     * its auxiliary events; declaring it last ensures those resources are
     * fenced and released before the graph, arena, executor, or tensors they
     * protect.
     */
    struct MoELocalExpertStage::DeferredGPUReplayFamily
    {
        size_t row_capacity = 0;
        /** Fixed participant-local routes per row embedded in this graph. */
        int route_width = 0;
        const MoELocalExpertSerialBufferArena::TensorFamily *tensor_family =
            nullptr;
        ComputeGraph graph;
        BufferArena arena;
        std::unique_ptr<DeviceGraphExecutor> executor;
        MoEExpertComputeStage *compute_stage = nullptr;
        DeviceGraphExecutor::GraphSegmentCache cache;
    };

    MoELocalExpertSerialBufferArena::MoELocalExpertSerialBufferArena(
        Config config)
        : device_id_(config.device_id),
          row_capacity_(config.row_capacity),
          d_model_(config.d_model),
          routing_top_k_(config.routing_top_k),
          logical_participant_id_(config.logical_participant_id)
    {
        if (!device_id_.is_valid() || row_capacity_ == 0 || d_model_ <= 0 ||
            routing_top_k_ <= 0 ||
            (logical_participant_id_ && *logical_participant_id_ < 0))
        {
            throw std::invalid_argument(
                "MoELocalExpertSerialBufferArena requires a valid device, positive "
                "row capacity, d_model, routing_top_k, and non-negative participant id");
        }

        const std::string prefix =
            config.debug_name.empty()
                ? "moe_local_expert_serial@" + device_id_.to_string()
                : std::move(config.debug_name);

        /*
         * Every bucket owns a complete Tensor coherence state. This is more
         * memory than aliasing a short view into the maximum tensor, but it is
         * the only honest way to make a short H2D/D2H transfer independently
         * valid. Production bounds the ladder by one prefill segment rather
         * than the KV horizon, so its geometric sum stays below twice the
         * largest live family while every retained graph keeps exact ownership.
         */
        config.row_capacity_buckets.push_back(row_capacity_);
        std::sort(
            config.row_capacity_buckets.begin(),
            config.row_capacity_buckets.end());
        config.row_capacity_buckets.erase(
            std::unique(
                config.row_capacity_buckets.begin(),
                config.row_capacity_buckets.end()),
            config.row_capacity_buckets.end());

        const size_t d_model = static_cast<size_t>(d_model_);
        const size_t routing_top_k = static_cast<size_t>(routing_top_k_);
        families_.reserve(config.row_capacity_buckets.size());
        for (const size_t capacity : config.row_capacity_buckets)
        {
            if (capacity == 0 || capacity > row_capacity_)
            {
                throw std::invalid_argument(
                    "MoELocalExpertSerialBufferArena bucket capacity must be "
                    "positive and no larger than row_capacity");
            }

            const size_t hidden_elements =
                checkedMultiply(capacity, d_model, "compact hidden rows");
            const size_t routing_elements =
                checkedMultiply(capacity, routing_top_k, "compact routing rows");
            const size_t hidden_bytes =
                checkedMultiply(hidden_elements, sizeof(float), "compact hidden bytes");
            const size_t routing_bytes =
                checkedMultiply(routing_elements, sizeof(float), "compact routing bytes");
            const size_t family_bytes = checkedAdd(
                checkedAdd(hidden_bytes, routing_bytes, "compact input bytes"),
                checkedAdd(routing_bytes, hidden_bytes, "compact output bytes"),
                "compact family bytes");
            allocation_bytes_ = checkedAdd(
                allocation_bytes_, family_bytes, "compact arena bytes");

            TensorFamily family;
            family.row_capacity = capacity;
            family.allocation_bytes = family_bytes;
            if (device_id_.is_gpu())
            {
                /*
                 * The retained graph reads compact packet inputs and writes
                 * compact output through one immutable host-DMA allocation.
                 * Its byte layout exactly mirrors the four tensor payloads,
                 * so graph capture embeds no pageable pointer and model-time
                 * accounting can distinguish VRAM from pinned host capacity.
                 */
                family.pinned_transfer_bytes = family_bytes;
                family.pinned_hidden_offset = 0;
                family.pinned_routing_indices_offset = hidden_bytes;
                family.pinned_routing_weights_offset = checkedAdd(
                    hidden_bytes,
                    routing_bytes,
                    "pinned routing-weight offset");
                family.pinned_output_offset = checkedAdd(
                    family.pinned_routing_weights_offset,
                    routing_bytes,
                    "pinned output offset");
                pinned_transfer_bytes_ = checkedAdd(
                    pinned_transfer_bytes_,
                    family_bytes,
                    "pinned compact transfer bytes");
                family.pinned_transfer =
                    TransferEngine::instance().declarePinnedHostBuffer(
                        family_bytes,
                        device_id_);
            }
            family.hidden = std::make_shared<FP32Tensor>(
                std::vector<size_t>{capacity, d_model});
            family.routing_indices = std::make_shared<FP32Tensor>(
                std::vector<size_t>{capacity, routing_top_k});
            family.routing_weights = std::make_shared<FP32Tensor>(
                std::vector<size_t>{capacity, routing_top_k});
            family.output = std::make_shared<FP32Tensor>(
                std::vector<size_t>{capacity, d_model});

            const std::string family_prefix =
                prefix + ".rows" + std::to_string(capacity);
            family.hidden->setDebugName(family_prefix + ".compact_hidden");
            family.routing_indices->setDebugName(
                family_prefix + ".compact_routing_indices");
            family.routing_weights->setDebugName(
                family_prefix + ".compact_routing_weights");
            family.output->setDebugName(family_prefix + ".compact_output");
            families_.push_back(std::move(family));
        }
        if (families_.empty() ||
            families_.back().row_capacity != row_capacity_)
        {
            throw std::logic_error(
                "MoELocalExpertSerialBufferArena failed to materialize its maximum family");
        }

        const PerfStatsCollector::Tags tags{
            {"bytes", std::to_string(allocation_bytes_)},
            {"d_model", std::to_string(d_model_)},
            {"family_count", std::to_string(families_.size())},
            {"immutable", "true"},
            {"ownership", "per_device_participant_serial_graph_family"},
            {"participant",
             logical_participant_id_
                 ? std::to_string(*logical_participant_id_)
                 : "unbound"},
            {"row_capacity", std::to_string(row_capacity_)},
            {"routing_top_k", std::to_string(routing_top_k_)},
            {"pinned_transfer_bytes",
             std::to_string(pinned_transfer_bytes_)}};
        PerfStatsCollector::addCounter(
            "memory",
            "moe_serial_local_expert_buffer_arena_allocations",
            1.0,
            "model_setup",
            device_id_.to_string(),
            tags);
        PerfStatsCollector::addCounter(
            "memory",
            "moe_serial_local_expert_buffer_arena_bytes",
            static_cast<double>(allocation_bytes_),
            "model_setup",
            device_id_.to_string(),
            tags);
        if (pinned_transfer_bytes_ > 0)
        {
            PerfStatsCollector::addCounter(
                "memory",
                "moe_serial_local_expert_pinned_transfer_bytes",
                static_cast<double>(pinned_transfer_bytes_),
                "model_setup",
                device_id_.to_string(),
                tags);
        }
        logVramBomLine(
            "moe_serial_local_expert_buffer_arena",
            "device=" + device_id_.to_string() +
                " hidden_tensor=" + vramBomPointer(hidden().get()) +
                " ownership=per_device_participant_serial_graph_family"
                " immutable=true"
                " family_count=" + std::to_string(families_.size()) +
                " row_capacity=" + std::to_string(row_capacity_) +
                " d_model=" + std::to_string(d_model_) +
                " routing_top_k=" + std::to_string(routing_top_k_) +
                " pinned_transfer_bytes=" +
                std::to_string(pinned_transfer_bytes_) +
                " " + vramBomBytes(allocation_bytes_));
    }

    bool MoELocalExpertSerialBufferArena::supports(
        DeviceId device,
        size_t requested_row_capacity,
        int d_model,
        int routing_top_k) const noexcept
    {
        return device == device_id_ && requested_row_capacity <= row_capacity_ &&
               d_model == d_model_ && routing_top_k == routing_top_k_;
    }

    const MoELocalExpertSerialBufferArena::TensorFamily *
    MoELocalExpertSerialBufferArena::smallestFamilySupporting(
        size_t requested_row_capacity) const noexcept
    {
        if (requested_row_capacity == 0)
            return nullptr;
        const auto found = std::lower_bound(
            families_.begin(),
            families_.end(),
            requested_row_capacity,
            [](const TensorFamily &family, size_t requested)
            {
                return family.row_capacity < requested;
            });
        return found == families_.end() ? nullptr : &*found;
    }

    MoELocalExpertStage::MoELocalExpertStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (!params_.input_rows && params_.input_rows_lifetime)
            params_.input_rows = params_.input_rows_lifetime.get();
        if (!params_.output_rows && params_.output_rows_lifetime)
            params_.output_rows = params_.output_rows_lifetime.get();
        if (params_.input_rows)
        {
            const size_t packet_entry_capacity =
                params_.input_rows->entry_capacity > 0
                    ? params_.input_rows->entry_capacity
                    : params_.input_rows->row_capacity *
                          static_cast<size_t>(std::max(params_.top_k, 1));
            const size_t row_capacity =
                params_.graph_row_capacity > 0
                    ? params_.graph_row_capacity
                    : params_.input_rows->row_capacity;
            if (params_.top_k <= 0 || row_capacity == 0 ||
                row_capacity > params_.input_rows->row_capacity)
            {
                throw std::invalid_argument(
                    "MoELocalExpertStage graph row capacity exceeds its fixed packet view or has invalid top_k");
            }
            const size_t entry_capacity = checkedMultiply(
                row_capacity,
                static_cast<size_t>(params_.top_k),
                "graph sparse entry capacity");
            if (entry_capacity > packet_entry_capacity)
            {
                throw std::invalid_argument(
                    "MoELocalExpertStage graph entry capacity exceeds its fixed packet view");
            }
            if (!ensureCompactCapacity(row_capacity, params_.top_k))
            {
                throw std::runtime_error(
                    "MoELocalExpertStage failed to allocate its persistent compact row buffers during graph construction");
            }

            route_admission_capacity_ = entry_capacity;
            row_admission_capacity_ = row_capacity;
            active_routes_.reserve(route_admission_capacity_);
            row_output_slot_.resize(row_admission_capacity_, -1);
            validated_input_rows_.resize(row_admission_capacity_, 0u);
            output_input_rows_.reserve(row_admission_capacity_);
            compact_row_route_counts_.resize(row_admission_capacity_, 0u);
        }
        if (params_.moe_runtime_table && params_.layer_idx >= 0)
            (void)refreshRuntimePlacement();
        if (params_.cpu_current_batch_llep_state)
        {
            if (!params_.device_id.is_cpu() || params_.num_experts <= 0)
            {
                throw std::invalid_argument(
                    "CPU current-batch LLEP sparse consumer requires a positive CPU expert geometry");
            }
            cpu_current_batch_llep_transient_arrivals_.resize(
                static_cast<size_t>(params_.num_experts));
        }
        if (params_.num_experts > 0)
        {
            const size_t expert_count =
                static_cast<size_t>(params_.num_experts);
            invocation_gate_engines_.resize(expert_count, nullptr);
            invocation_up_engines_.resize(expert_count, nullptr);
            invocation_down_engines_.resize(expert_count, nullptr);
            invocation_expert_mask_.resize(expert_count, false);
        }

        if (params_.completion_policy ==
            CompletionPolicy::DeferredExplicitStage)
        {
            if (!params_.device_id.is_gpu() ||
                !params_.overlay_participant_residency ||
                params_.cpu_current_batch_llep_state ||
                params_.num_experts <= 0 ||
                params_.expert_mask.size() !=
                    static_cast<size_t>(params_.num_experts) ||
                route_admission_capacity_ == 0 ||
                row_admission_capacity_ == 0)
            {
                throw std::invalid_argument(
                    "Deferred MoELocalExpertStage completion requires a GPU, "
                    "epoch-indexed residency, complete expert mask, and fixed packet capacity");
            }
            createDeferredReplayFamilies();
        }
    }

    std::vector<size_t>
    MoELocalExpertSerialBufferArena::powerOfTwoRowBucketsThrough(
        size_t maximum_capacity)
    {
        std::vector<size_t> buckets;
        if (maximum_capacity == 0)
            return buckets;

        size_t capacity = 1;
        while (capacity < maximum_capacity)
        {
            buckets.push_back(capacity);
            if (capacity > std::numeric_limits<size_t>::max() / 2u)
                break;
            capacity *= 2u;
        }
        buckets.push_back(maximum_capacity);
        return buckets;
    }

    int MoELocalExpertSerialBufferArena::routeWidthBucketFor(
        int required_routes,
        int model_top_k) noexcept
    {
        if (required_routes <= 0 || model_top_k <= 0 ||
            required_routes > model_top_k)
        {
            return 0;
        }

        int bucket = 1;
        while (bucket < required_routes &&
               bucket <= model_top_k / 2)
        {
            bucket *= 2;
        }
        /* A non-power-of-two model top-k remains the exact final family. */
        return bucket >= required_routes ? std::min(bucket, model_top_k)
                                         : model_top_k;
    }

    MoELocalExpertStage::~MoELocalExpertStage()
    {
        if (deferred_lifecycle_ != DeferredLifecycle::Idle)
        {
            LOG_ERROR(
                "[MoELocalExpertStage] Destroyed with an uncompleted GPU sparse packet"
                << " layer=" << params_.layer_idx
                << " participant=" << params_.runtime_participant_index
                << " device=" << params_.device_id.to_string());
            std::terminate();
        }
    }

    bool MoELocalExpertStage::hasPendingDeferredOutput() const noexcept
    {
        return deferred_lifecycle_ != DeferredLifecycle::Idle;
    }

    bool MoELocalExpertStage::
        allDeferredReplayFamiliesUseServiceTelemetryForTesting() const
        noexcept
    {
        return !deferred_replay_families_.empty() &&
               std::all_of(
                   deferred_replay_families_.begin(),
                   deferred_replay_families_.end(),
                   [](const auto &replay)
                   {
                       return replay && replay->compute_stage &&
                              replay->compute_stage
                                  ->hasOverlayServiceTelemetryForTesting();
                   });
    }

    void MoELocalExpertStage::createDeferredReplayFamilies()
    {
        const size_t expert_count = static_cast<size_t>(params_.num_experts);
        if (params_.prepared_gate_gemm.size() != expert_count ||
            params_.prepared_up_gemm.size() != expert_count ||
            params_.prepared_down_gemm.size() != expert_count ||
            invocation_gate_engines_.size() != expert_count ||
            invocation_up_engines_.size() != expert_count ||
            invocation_down_engines_.size() != expert_count ||
            !params_.serial_compact_buffer_arena ||
            !compact_hidden_ || !compact_routing_indices_ ||
            !compact_routing_weights_ || !compact_output_)
        {
            throw std::invalid_argument(
                "Deferred MoELocalExpertStage requires complete construction-time prepared tables and an immutable compact-tensor arena");
        }
        std::copy(
            params_.prepared_gate_gemm.begin(),
            params_.prepared_gate_gemm.end(),
            invocation_gate_engines_.begin());
        std::copy(
            params_.prepared_up_gemm.begin(),
            params_.prepared_up_gemm.end(),
            invocation_up_engines_.begin());
        std::copy(
            params_.prepared_down_gemm.begin(),
            params_.prepared_down_gemm.end(),
            invocation_down_engines_.begin());
        std::copy(
            params_.expert_mask.begin(),
            params_.expert_mask.end(),
            invocation_expert_mask_.begin());

        const auto &families =
            params_.serial_compact_buffer_arena->families();
        std::vector<int> route_widths;
        for (int width = 1; width < params_.top_k;)
        {
            route_widths.push_back(width);
            if (width > params_.top_k / 2)
                break;
            width *= 2;
        }
        if (route_widths.empty() || route_widths.back() != params_.top_k)
            route_widths.push_back(params_.top_k);
        deferred_replay_families_.reserve(
            families.size() * route_widths.size());
        for (const auto &tensor_family : families)
        {
            if (tensor_family.row_capacity == 0 ||
                tensor_family.row_capacity >
                    static_cast<size_t>(std::numeric_limits<int>::max()) ||
                !tensor_family.hidden || !tensor_family.routing_indices ||
                !tensor_family.routing_weights || !tensor_family.output ||
                !tensor_family.pinned_transfer ||
                tensor_family.pinned_transfer_bytes !=
                    tensor_family.allocation_bytes ||
                tensor_family.pinned_transfer->sizeBytes() !=
                    tensor_family.pinned_transfer_bytes ||
                tensor_family.pinned_transfer->registrationDevice() !=
                    params_.device_id ||
                tensor_family.pinned_hidden_offset != 0 ||
                tensor_family.pinned_routing_indices_offset !=
                    tensor_family.hidden->size_bytes() ||
                tensor_family.pinned_routing_weights_offset !=
                    tensor_family.pinned_routing_indices_offset +
                        tensor_family.routing_indices->size_bytes() ||
                tensor_family.pinned_output_offset !=
                    tensor_family.pinned_routing_weights_offset +
                        tensor_family.routing_weights->size_bytes() ||
                tensor_family.pinned_output_offset +
                        tensor_family.output->size_bytes() !=
                    tensor_family.pinned_transfer_bytes)
            {
                throw std::invalid_argument(
                    "Deferred MoELocalExpertStage has an invalid immutable tensor family");
            }

            for (const int route_width : route_widths)
            {
            auto replay = std::make_unique<DeferredGPUReplayFamily>();
            replay->row_capacity = tensor_family.row_capacity;
            replay->route_width = route_width;
            replay->tensor_family = &tensor_family;

            MoEExpertComputeStage::Params compute_params;
            compute_params.device_id = params_.device_id;
            compute_params.input = tensor_family.hidden.get();
            compute_params.seq_len =
                static_cast<int>(tensor_family.row_capacity);
            compute_params.d_model = params_.d_model;
            compute_params.num_experts = params_.num_experts;
            compute_params.top_k = route_width;
            compute_params.gate_exps = params_.gate_exps;
            compute_params.up_exps = params_.up_exps;
            compute_params.down_exps = params_.down_exps;
            compute_params.expert_intermediate = params_.expert_intermediate;
            compute_params.layer_idx = params_.layer_idx;
            compute_params.expert_mask = invocation_expert_mask_;
            compute_params.routing_indices =
                tensor_family.routing_indices.get();
            compute_params.routing_weights =
                tensor_family.routing_weights.get();
            compute_params.output = tensor_family.output.get();
            compute_params.output_registered_in_arena = true;
            /*
             * A one-route bucket must still use the fixed-topology grouped
             * kernel.  Ordinary seq_len==1 selects runtime-table decode, whose
             * owner metadata is intentionally absent at this explicit sparse
             * boundary.  The forced grouped lane has identical route
             * arithmetic and is graph-capturable on both GPU backends.
             */
            compute_params.force_grouped_verifier_prefill_for_decode =
                tensor_family.row_capacity == 1u;
            compute_params.cpu_router_q8_input_publication =
                CPURouterQ8InputPublicationPolicy::RequireRouterStage;
            compute_params.prepared_gate_gemm = invocation_gate_engines_;
            compute_params.prepared_up_gemm = invocation_up_engines_;
            compute_params.prepared_down_gemm = invocation_down_engines_;
            compute_params.prepared_store = params_.prepared_store;
            compute_params.expert_registry = params_.expert_registry;
            compute_params.overlay_service_telemetry =
                params_.overlay_service_telemetry;
            compute_params.gate_slab_ref = params_.gate_slab_ref;
            compute_params.up_slab_ref = params_.up_slab_ref;
            compute_params.down_slab_ref = params_.down_slab_ref;

            auto compute_stage =
                std::make_unique<MoEExpertComputeStage>(
                    std::move(compute_params));
            replay->compute_stage = compute_stage.get();
            replay->compute_stage->releaseRawExpertWeights();

            /*
             * Prime fixed-size vectors and mark this as an externally prepared
             * sparse-overlay replay.  Later packets mutate only bytes behind
             * stable tensor/descriptor addresses.
             */
            if (!replay->compute_stage->bindSparseOverlayInvocation(
                    MoEExpertComputeStage::SparseOverlayInvocation{
                        .input = tensor_family.hidden.get(),
                        .routing_indices =
                            tensor_family.routing_indices.get(),
                        .routing_weights =
                            tensor_family.routing_weights.get(),
                        .output = tensor_family.output.get(),
                        .live_rows = static_cast<int>(
                            tensor_family.row_capacity),
                        .binding_kind =
                            MoEExpertComputeStage::
                                SparseOverlayBindingKind::SetupPriming,
                        .expert_mask = &invocation_expert_mask_,
                        .gate_engines = invocation_gate_engines_,
                        .up_engines = invocation_up_engines_,
                        .down_engines = invocation_down_engines_,
                    }))
            {
                throw std::runtime_error(
                    "Deferred MoELocalExpertStage could not prime a retained tensor-family executor");
            }

            if (!replay->arena.registerExternalBuffer(
                    BufferId::NORMALIZED,
                    tensor_family.hidden.get()) ||
                !replay->arena.registerExternalBuffer(
                    BufferId::MOE_EXPERT_INDICES,
                    tensor_family.routing_indices.get()) ||
                !replay->arena.registerExternalBuffer(
                    BufferId::MOE_EXPERT_WEIGHTS,
                    tensor_family.routing_weights.get()) ||
                !replay->arena.registerExternalBuffer(
                    BufferId::MOE_COMBINED_OUTPUT,
                    tensor_family.output.get()))
            {
                throw std::runtime_error(
                    "Deferred MoELocalExpertStage could not register a complete graph arena frontier");
            }

            GraphExecutorConfig executor_config;
            executor_config.default_device = params_.device_id;
            executor_config.worker_gpu_context_resolver =
                [worker = params_.worker_gpu_context,
                 expected_device = params_.device_id](DeviceId requested)
                    -> IWorkerGPUContext *
                {
                    return requested == expected_device ? worker : nullptr;
                };
            executor_config.worker_gpu_context_uses_process_pool = false;
            replay->executor =
                std::make_unique<DeviceGraphExecutor>(executor_config);
            replay->executor->setArena(&replay->arena);
            replay->cache.perf_context =
                "moe_overlay_local_layer_" +
                std::to_string(params_.layer_idx) + "_participant_" +
                std::to_string(params_.runtime_participant_index) +
                "_rows_" +
                std::to_string(tensor_family.row_capacity) + "_routes_" +
                std::to_string(route_width);
            /*
             * This family is immutable for the model lifetime and always owns
             * one complete captured endpoint transaction on one exact borrowed
             * participant stream. Select the typed retained host plan here,
             * where that topology is constructed, rather than rediscovering it
             * heuristically in the per-layer replay path.
             */
            replay->cache.steady_replay_host_policy =
                DeviceGraphExecutor::GraphSegmentCache::
                    SteadyReplayHostPolicy::RetainedFullGraph;
            replay->graph.addNode(
                "moe_overlay_local_expert_input_publish",
                std::make_unique<MoELocalExpertInputPublishStage>(
                    MoELocalExpertInputPublishStage::Params{
                        .device_id = params_.device_id,
                        .hidden = tensor_family.hidden.get(),
                        .routing_indices =
                            tensor_family.routing_indices.get(),
                        .routing_weights =
                            tensor_family.routing_weights.get(),
                        .pinned_transfer = tensor_family.pinned_transfer,
                        .hidden_offset =
                            tensor_family.pinned_hidden_offset,
                        .routing_indices_offset =
                            tensor_family.pinned_routing_indices_offset,
                        .routing_weights_offset =
                            tensor_family.pinned_routing_weights_offset,
                    }),
                params_.device_id);
            replay->graph.addNode(
                "moe_overlay_local_expert",
                std::move(compute_stage),
                params_.device_id);
            replay->graph.addNode(
                "moe_overlay_local_expert_output_publish",
                std::make_unique<MoELocalExpertOutputPublishStage>(
                    MoELocalExpertOutputPublishStage::Params{
                        .device_id = params_.device_id,
                        .output = tensor_family.output.get(),
                        .output_bytes = tensor_family.output->size_bytes(),
                        .output_offset =
                            tensor_family.pinned_output_offset,
                        .pinned_transfer = tensor_family.pinned_transfer,
                    }),
                params_.device_id);
            replay->graph.addDependency(
                "moe_overlay_local_expert",
                "moe_overlay_local_expert_input_publish");
            replay->graph.addDependency(
                "moe_overlay_local_expert_output_publish",
                "moe_overlay_local_expert");
            deferred_replay_families_.push_back(std::move(replay));
            }
        }

        if (deferred_replay_families_.empty())
        {
            throw std::runtime_error(
                "Deferred MoELocalExpertStage created no retained tensor-family graphs");
        }
    }

    MoELocalExpertStage::DeferredGPUReplayFamily *
    MoELocalExpertStage::currentDeferredReplayFamily() noexcept
    {
        const auto found = std::find_if(
            deferred_replay_families_.begin(),
            deferred_replay_families_.end(),
            [&](const auto &candidate)
            {
                const auto *family = candidate
                                         ? candidate->tensor_family
                                         : nullptr;
                return family &&
                       candidate->row_capacity == compact_capacity_ &&
                       candidate->route_width ==
                           compact_execution_top_k_ &&
                       family->hidden.get() == compact_hidden_.get() &&
                       family->routing_indices.get() ==
                           compact_routing_indices_.get() &&
                       family->routing_weights.get() ==
                           compact_routing_weights_.get() &&
                       family->output.get() == compact_output_.get();
            });
        return found == deferred_replay_families_.end()
                   ? nullptr
                   : found->get();
    }

    void MoELocalExpertStage::resetDeferredInvocation() noexcept
    {
        deferred_residency_bank_.reset();
        pending_deferred_replay_ = nullptr;
        deferred_lifecycle_ = DeferredLifecycle::Idle;
        pending_economy_timing_enabled_ = false;
        pending_endpoint_detail_enabled_ = false;
        pending_profiling_enabled_ = false;
        pending_compact_family_bytes_ = 0;
        compact_execution_top_k_ = 0;
        active_routes_.clear();
        output_input_rows_.clear();
    }

    ExpertPackedWeights
    MoELocalExpertStage::cloneCPUCurrentBatchLLEPPreparedExpert(
        int expert_id,
        uint64_t durable_parent_epoch) const
    {
        if (!params_.device_id.is_cpu() ||
            !params_.overlay_participant_residency ||
            durable_parent_epoch == 0 || expert_id < 0 ||
            expert_id >= params_.num_experts || params_.layer_idx < 0)
        {
            throw std::invalid_argument(
                "CPU LLEP packed clone has invalid endpoint, epoch, or expert identity");
        }

        const auto bank =
            params_.overlay_participant_residency->acquire(
                durable_parent_epoch);
        if (!bank ||
            static_cast<size_t>(params_.layer_idx) >= bank->layers.size())
        {
            throw std::runtime_error(
                "CPU LLEP packed clone cannot acquire its durable parent bank");
        }
        const auto &layer =
            bank->layers[static_cast<size_t>(params_.layer_idx)];
        const size_t expert_index = static_cast<size_t>(expert_id);
        if (!layer.valid(params_.num_experts) ||
            !layer.resident_mask[expert_index] ||
            !layer.experts[expert_index].complete())
        {
            throw std::runtime_error(
                "CPU LLEP packed clone requested an expert absent from its durable owner bank");
        }

        const auto &engines = layer.experts[expert_index];
        ExpertPackedWeights packed;
        packed.gate = engines.gate->cloneWeights();
        packed.up = engines.up->cloneWeights();
        packed.down = engines.down->cloneWeights();
        if (!packed.complete())
        {
            throw std::runtime_error(
                "CPU LLEP durable owner engines did not export complete packed clones");
        }
        return packed;
    }

    bool MoELocalExpertStage::installCPUCurrentBatchLLEPTransientResidency(
        const CPUCurrentBatchLLEPTransactionState &state,
        const std::unordered_map<int, PreparedExpertEngines> *arrivals)
    {
        if (!params_.cpu_current_batch_llep_state ||
            params_.cpu_current_batch_llep_state.get() != &state ||
            !params_.device_id.is_cpu() || state.active ||
            state.layer_idx != params_.layer_idx ||
            state.participant_id != params_.runtime_participant_index ||
            state.num_experts != params_.num_experts ||
            state.durable_parent_epoch == 0 ||
            state.owner_mask.size() !=
                static_cast<size_t>(params_.num_experts) ||
            state.transient_resident_mask.size() !=
                static_cast<size_t>(params_.num_experts) ||
            cpu_current_batch_llep_parent_epoch_ != 0 ||
            cpu_current_batch_llep_transient_arrivals_.size() !=
                static_cast<size_t>(params_.num_experts))
        {
            return false;
        }

        const auto parent =
            params_.overlay_participant_residency
                ? params_.overlay_participant_residency->acquire(
                      state.durable_parent_epoch)
                : MoEOverlayParticipantBankLease{};
        if (!parent ||
            static_cast<size_t>(params_.layer_idx) >= parent->layers.size())
        {
            return false;
        }
        const auto &parent_layer =
            parent->layers[static_cast<size_t>(params_.layer_idx)];
        if (!parent_layer.valid(params_.num_experts) ||
            parent_layer.resident_mask != state.owner_mask)
        {
            return false;
        }

        for (auto &entry : cpu_current_batch_llep_transient_arrivals_)
            entry = {};
        for (int expert = 0; expert < params_.num_experts; ++expert)
        {
            const size_t index = static_cast<size_t>(expert);
            if (!state.transient_resident_mask[index] ||
                state.owner_mask[index])
            {
                continue;
            }
            if (!arrivals)
                return false;
            const auto found = arrivals->find(expert);
            if (found == arrivals->end() || !found->second.complete())
                return false;
            cpu_current_batch_llep_transient_arrivals_[index] =
                found->second;
        }

        if (arrivals)
        {
            for (const auto &[expert, engines] : *arrivals)
            {
                if (expert < 0 || expert >= params_.num_experts ||
                    !engines.complete() ||
                    !state.transient_resident_mask[
                        static_cast<size_t>(expert)] ||
                    state.owner_mask[static_cast<size_t>(expert)])
                {
                    for (auto &entry :
                         cpu_current_batch_llep_transient_arrivals_)
                        entry = {};
                    return false;
                }
            }
        }

        cpu_current_batch_llep_parent_epoch_ =
            state.durable_parent_epoch;
        return true;
    }

    bool MoELocalExpertStage::discardCPUCurrentBatchLLEPTransientResidency(
        const CPUCurrentBatchLLEPTransactionState &state) noexcept
    {
        if (!params_.cpu_current_batch_llep_state ||
            params_.cpu_current_batch_llep_state.get() != &state ||
            state.layer_idx != params_.layer_idx ||
            state.participant_id != params_.runtime_participant_index)
        {
            return false;
        }
        for (auto &entry : cpu_current_batch_llep_transient_arrivals_)
            entry = {};
        cpu_current_batch_llep_parent_epoch_ = 0;
        return true;
    }

    bool MoELocalExpertStage::publishServiceMeasurement(
        ExpertHistogramSource source,
        uint64_t elapsed_nanoseconds,
        uint64_t activations)
    {
        if (!params_.overlay_participant_residency)
            return true;
        const auto status =
            params_.overlay_participant_residency
                ->recordServiceMeasurement(
                    params_.layer_idx,
                    source,
                    elapsed_nanoseconds,
                    activations);
        switch (status)
        {
        case MoEOverlayServiceMeasurementRecordStatus::Recorded:
        case MoEOverlayServiceMeasurementRecordStatus::Disabled:
        case MoEOverlayServiceMeasurementRecordStatus::Contended:
            return true;
        case MoEOverlayServiceMeasurementRecordStatus::Invalid:
            LOG_ERROR("[MoELocalExpertStage] Rejected invalid exact service "
                      "measurement for layer "
                      << params_.layer_idx);
            return false;
        case MoEOverlayServiceMeasurementRecordStatus::Overflow:
            LOG_ERROR("[MoELocalExpertStage] Exact service measurement "
                      "accumulator overflowed for layer "
                      << params_.layer_idx);
            return false;
        }
        return false;
    }

    bool MoELocalExpertStage::preparePersistentBuffers()
    {
        if (!params_.input_rows)
            return false;

        const size_t packet_entry_capacity =
            params_.input_rows->entry_capacity > 0
                ? params_.input_rows->entry_capacity
                : params_.input_rows->row_capacity *
                      static_cast<size_t>(std::max(params_.top_k, 1));
        const size_t row_capacity =
            params_.graph_row_capacity > 0
                ? params_.graph_row_capacity
                : params_.input_rows->row_capacity;
        if (params_.top_k <= 0 || row_capacity == 0 ||
            row_capacity > params_.input_rows->row_capacity)
        {
            return false;
        }
        const size_t entry_capacity = checkedMultiply(
            row_capacity,
            static_cast<size_t>(params_.top_k),
            "persistent graph sparse entry capacity");
        if (entry_capacity > packet_entry_capacity ||
            !ensureCompactCapacity(row_capacity, params_.top_k))
        {
            return false;
        }

        if (!params_.device_id.is_gpu())
            return true;
        if (!hasGPUStream())
        {
            throw std::runtime_error(
                "MoELocalExpertStage GPU persistent-buffer preparation "
                "requires an explicit non-null stream");
        }

        /*
         * Establish every bucket's device storage before inference. execute()
         * may choose a smaller family from the live sparse count, but it must
         * never turn that choice into a hot-path allocation. Repeated graph
         * stages sharing this arena only revisit already-prepared tensors.
         */
        const StageGPUExecution execution = gpuExecution();
        const auto prepare_family =
            [&](FP32Tensor *hidden,
                FP32Tensor *routing_indices,
                FP32Tensor *routing_weights,
                FP32Tensor *output)
        {
            execution.prepareInput(hidden);
            execution.prepareInput(routing_indices);
            execution.prepareInput(routing_weights);
            execution.prepareOutput(output);
        };
        if (params_.serial_compact_buffer_arena)
        {
            for (const auto &family :
                 params_.serial_compact_buffer_arena->families())
            {
                if (!family.pinned_transfer ||
                    family.pinned_transfer_bytes !=
                        family.allocation_bytes ||
                    family.pinned_transfer->sizeBytes() !=
                        family.pinned_transfer_bytes ||
                    family.pinned_transfer->registrationDevice() !=
                        params_.device_id)
                {
                    throw std::runtime_error(
                        "MoELocalExpertStage GPU serial family is missing its fixed pinned transfer owner");
                }
                TransferEngine::instance().bindPinnedHostBuffer(
                    *family.pinned_transfer);
                prepare_family(
                    family.hidden.get(),
                    family.routing_indices.get(),
                    family.routing_weights.get(),
                    family.output.get());
            }
        }
        else
        {
            prepare_family(
                compact_hidden_.get(),
                compact_routing_indices_.get(),
                compact_routing_weights_.get(),
                compact_output_.get());
        }
        return true;
    }

    bool MoELocalExpertStage::prepareExpertGemmEngines(Params &params)
    {
        const size_t expected =
            static_cast<size_t>(std::max(params.num_experts, 0));
        const auto complete_active_table = [&]()
        {
            if (expected == 0 ||
                params.prepared_gate_gemm.size() != expected ||
                params.prepared_up_gemm.size() != expected ||
                params.prepared_down_gemm.size() != expected)
            {
                return false;
            }

            bool has_active_expert = false;
            for (size_t expert = 0; expert < expected; ++expert)
            {
                const bool active =
                    params.expert_mask.empty() ||
                    (expert < params.expert_mask.size() &&
                     params.expert_mask[expert]);
                if (!active)
                    continue;
                has_active_expert = true;
                if (!params.prepared_gate_gemm[expert] ||
                    !params.prepared_up_gemm[expert] ||
                    !params.prepared_down_gemm[expert])
                {
                    return false;
                }
            }
            return has_active_expert;
        };

        if (complete_active_table())
            return true;

        if (params.expert_weight_resolution_policy ==
            ExpertWeightResolutionPolicy::RegistryOnly)
        {
            LOG_ERROR(
                "[MoELocalExpertStage] Registry-only ExpertOverlay stage has "
                "incomplete prepared engines for layer "
                << params.layer_idx << " on "
                << params.device_id.to_string()
                << "; refusing raw expert-parent fallback");
            return false;
        }

        MoEWeightContext context{
            params.device_id,
            params.num_experts,
            params.expert_intermediate,
            params.d_model,
            /*local_expert_start=*/0,
            /*local_expert_count=*/params.num_experts,
            params.layer_idx,
            params.expert_mask,
            params.gate_exps,
            params.up_exps,
            params.down_exps,
            params.expert_gate_views,
            params.expert_up_views,
            params.expert_down_views,
            params.prepared_gate_gemm,
            params.prepared_up_gemm,
            params.prepared_down_gemm,
            params.moe_owned_kernels,
            params.moe_packed_gate_lifetime,
            params.moe_packed_up_lifetime,
            params.moe_packed_down_lifetime,
            nullptr,
            params.prepared_store,
            params.expert_registry,
            params.gate_slab_ref,
            params.up_slab_ref,
            params.down_slab_ref,
            true,
            nullptr,
            CPUExpertNUMAPlacement::forDevice(params.device_id)};

        if (!MoEExpertWeightService::extractExpertViews(context))
            return false;
        const bool prepared = MoEExpertWeightService::prepareGemmEngines(context);
        params.gate_slab_ref = context.gate_slab_ref;
        params.up_slab_ref = context.up_slab_ref;
        params.down_slab_ref = context.down_slab_ref;
        return prepared && complete_active_table();
    }

    bool MoELocalExpertStage::refreshRuntimePlacement()
    {
        if (!params_.moe_runtime_table || params_.layer_idx < 0)
            return false;
        moe_runtime_layer_ = params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
        moe_runtime_table_initialized_ = runtimeTableHasActiveOverlayBank() || initializeMoERuntimePlacementBank();
        return moe_runtime_table_initialized_;
    }

    bool MoELocalExpertStage::initializeMoERuntimePlacementBank()
    {
        if (!params_.moe_runtime_table || params_.layer_idx < 0 || params_.num_experts <= 0)
            return false;
        if (params_.device_id.is_gpu() &&
            !params_.runtime_publication_stream)
        {
            throw std::invalid_argument(
                "[MoELocalExpertStage] GPU runtime placement publication "
                "requires an explicit non-null producer stream");
        }

        try
        {
            if (runtimeTableHasActiveOverlayBank())
                return true;

            const bool publication_required =
                params_.moe_runtime_table->decodeRuntimePublicationRequired(params_.layer_idx);
            const auto &state = params_.moe_runtime_table->hostLayerState(params_.layer_idx);
            if (!publication_required && state.active_epoch != 0)
            {
                LOG_ERROR("[MoELocalExpertStage] Invalid active MoE runtime placement bank for layer "
                          << params_.layer_idx << "; refusing to overwrite non-zero epoch "
                          << state.active_epoch);
                return false;
            }

            if (!params_.expert_mask.empty() &&
                params_.expert_mask.size() != static_cast<size_t>(params_.num_experts))
            {
                LOG_ERROR("[MoELocalExpertStage] Cannot initialize MoE runtime placement bank: expert_mask size "
                          << params_.expert_mask.size() << " != num_experts " << params_.num_experts);
                return false;
            }

            return publishMoERuntimePlacementBank(/*epoch=*/1u);
        }
        catch (const std::exception &ex)
        {
            LOG_ERROR("[MoELocalExpertStage] Failed to initialize MoE runtime placement bank for layer "
                      << params_.layer_idx << ": " << ex.what());
            return false;
        }
    }

    bool MoELocalExpertStage::publishMoERuntimePlacementBank(uint32_t epoch)
    {
        if (!params_.moe_runtime_table || params_.layer_idx < 0 ||
            params_.num_experts <= 0 || epoch == 0)
        {
            return false;
        }
        if (params_.device_id.is_gpu() && !params_.runtime_publication_stream)
        {
            throw std::invalid_argument(
                "[MoELocalExpertStage] GPU runtime placement publication "
                "requires an explicit non-null producer stream");
        }
        if (!params_.expert_mask.empty() &&
            params_.expert_mask.size() != static_cast<size_t>(params_.num_experts))
        {
            return false;
        }

        const bool has_prepared_vectors =
            params_.prepared_gate_gemm.size() == static_cast<size_t>(params_.num_experts) &&
            params_.prepared_up_gemm.size() == static_cast<size_t>(params_.num_experts) &&
            params_.prepared_down_gemm.size() == static_cast<size_t>(params_.num_experts);

        MoEPlacementUpdate update;
        update.epoch = epoch;
        update.expert_count = static_cast<uint32_t>(params_.num_experts);
        update.participant_id = params_.runtime_participant_index >= 0
                                    ? static_cast<uint32_t>(params_.runtime_participant_index)
                                    : 0u;
        update.participant_count = std::max<uint32_t>(update.participant_id + 1u, 1u);
        update.experts.resize(static_cast<size_t>(params_.num_experts));
        update.local_compute_mask.assign(static_cast<size_t>(params_.num_experts), 0u);
        update.replica_role.assign(
            static_cast<size_t>(params_.num_experts),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None));

        const uint32_t local_flags =
            toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                             DeviceMoEExpertFlags::Resident |
                             DeviceMoEExpertFlags::PreferredOwner |
                             DeviceMoEExpertFlags::LocalCompute);

        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            const bool local = params_.expert_mask.empty() ||
                               params_.expert_mask[static_cast<size_t>(expert_id)];
            if (!local)
                continue;

            if (!has_prepared_vectors ||
                !params_.prepared_gate_gemm[static_cast<size_t>(expert_id)] ||
                !params_.prepared_up_gemm[static_cast<size_t>(expert_id)] ||
                !params_.prepared_down_gemm[static_cast<size_t>(expert_id)])
            {
                LOG_ERROR("[MoELocalExpertStage] Cannot publish MoE runtime placement bank for layer "
                          << params_.layer_idx << " expert " << expert_id
                          << ": prepared gate/up/down GEMM engines are required for local experts");
                return false;
            }

            DeviceMoEExpertDescriptor desc;
            if (!params_.prepared_gate_gemm[static_cast<size_t>(expert_id)]
                     ->exportNativeVNNIMatrixDesc(desc.gate) ||
                !params_.prepared_up_gemm[static_cast<size_t>(expert_id)]
                     ->exportNativeVNNIMatrixDesc(desc.up) ||
                !params_.prepared_down_gemm[static_cast<size_t>(expert_id)]
                     ->exportNativeVNNIMatrixDesc(desc.down))
            {
                LOG_ERROR("[MoELocalExpertStage] Cannot publish MoE runtime placement bank for layer "
                          << params_.layer_idx << " expert " << expert_id
                          << ": prepared GEMM engines did not export native-VNNI descriptors");
                return false;
            }

            desc.logical_expert_id = expert_id;
            desc.owner_participant = params_.runtime_participant_index;
            desc.local_slot = expert_id;
            desc.flags = local_flags;
            update.experts[static_cast<size_t>(expert_id)] = desc;
            update.local_compute_mask[static_cast<size_t>(expert_id)] = 1u;
            update.replica_role[static_cast<size_t>(expert_id)] =
                static_cast<uint8_t>(DeviceMoEReplicaRole::Primary);
        }

        params_.moe_runtime_table->prepareInactiveBank(params_.layer_idx, update);
        params_.moe_runtime_table->flipActiveBank(
            params_.layer_idx,
            update.epoch,
            params_.runtime_publication_stream);
        moe_runtime_layer_ =
            params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
        return runtimeTableHasActiveOverlayBank();
    }

    ExpertWeightBlobs MoELocalExpertStage::serializeExpert(int expert_id) const
    {
        if (expert_id < 0 || expert_id >= params_.num_experts)
            throw std::out_of_range("MoELocalExpertStage expert id is outside layer geometry");
        auto context =
            const_cast<MoELocalExpertStage *>(this)->buildWeightContext();
        return MoEExpertWeightService::serializeExpert(context, expert_id);
    }

    std::vector<int> MoELocalExpertStage::missingPreparedExpertIds(
        const std::vector<int> &expert_ids) const
    {
        std::vector<int> missing;
        missing.reserve(expert_ids.size());
        for (const int expert_id : expert_ids)
        {
            if (expert_id < 0 || expert_id >= params_.num_experts)
                continue;
            const size_t index = static_cast<size_t>(expert_id);
            const bool complete =
                index < params_.prepared_gate_gemm.size() &&
                index < params_.prepared_up_gemm.size() &&
                index < params_.prepared_down_gemm.size() &&
                params_.prepared_gate_gemm[index] &&
                params_.prepared_up_gemm[index] &&
                params_.prepared_down_gemm[index];
            if (!complete)
                missing.push_back(expert_id);
        }
        return missing;
    }

    std::vector<const TensorBase *> MoELocalExpertStage::releaseDepartedExperts(
        const std::vector<bool> &new_mask)
    {
        auto context = buildWeightContext();
        return MoEExpertWeightService::releaseDepartedExperts(context, new_mask);
    }

    bool MoELocalExpertStage::registerAndPrepareNewExperts(
        const std::vector<bool> &new_mask,
        const std::unordered_map<int, ExpertWeightBlobs> *received_weights,
        const std::unordered_map<int, PreparedExpertEngines> *
            received_prepared_experts)
    {
        auto context = buildWeightContext();
        const bool prepared = MoEExpertWeightService::registerAndPrepareNewExperts(
            context,
            new_mask,
            received_weights,
            received_prepared_experts);
        params_.gate_slab_ref = context.gate_slab_ref;
        params_.up_slab_ref = context.up_slab_ref;
        params_.down_slab_ref = context.down_slab_ref;
        if (!prepared)
            return false;

        std::vector<int> active;
        active.reserve(new_mask.size());
        for (size_t expert_id = 0; expert_id < new_mask.size(); ++expert_id)
        {
            if (new_mask[expert_id])
                active.push_back(static_cast<int>(expert_id));
        }
        bindPreparedExpertEnginesForExperts(active);
        return true;
    }

    void MoELocalExpertStage::applyExpertMask(
        const std::vector<bool> &new_mask)
    {
        if (new_mask.size() != static_cast<size_t>(params_.num_experts))
        {
            throw std::invalid_argument(
                "MoELocalExpertStage expert mask must match num_experts");
        }
        std::vector<int> active;
        active.reserve(new_mask.size());
        for (size_t expert = 0; expert < new_mask.size(); ++expert)
        {
            if (new_mask[expert])
                active.push_back(static_cast<int>(expert));
        }
        if (!missingPreparedExpertIds(active).empty())
        {
            throw std::runtime_error(
                "MoELocalExpertStage cannot publish a mask with missing prepared experts");
        }

        const auto previous_mask = params_.expert_mask;
        params_.expert_mask = new_mask;
        moe_runtime_table_initialized_ = false;
        if (!params_.moe_runtime_table)
            return;

        try
        {
            const auto &state =
                params_.moe_runtime_table->hostLayerState(params_.layer_idx);
            if (!publishMoERuntimePlacementBank(state.active_epoch + 1u))
                throw std::runtime_error("runtime placement bank publication failed");
            moe_runtime_table_initialized_ = true;
        }
        catch (...)
        {
            params_.expert_mask = previous_mask;
            moe_runtime_table_initialized_ = runtimeTableHasActiveOverlayBank();
            throw;
        }
    }

    MoEWeightContext MoELocalExpertStage::buildWeightContext()
    {
        return MoEWeightContext{
            params_.device_id,
            params_.num_experts,
            params_.expert_intermediate,
            params_.d_model,
            /*local_expert_start=*/0,
            /*local_expert_count=*/params_.num_experts,
            params_.layer_idx,
            params_.expert_mask,
            params_.gate_exps,
            params_.up_exps,
            params_.down_exps,
            params_.expert_gate_views,
            params_.expert_up_views,
            params_.expert_down_views,
            params_.prepared_gate_gemm,
            params_.prepared_up_gemm,
            params_.prepared_down_gemm,
            params_.moe_owned_kernels,
            params_.moe_packed_gate_lifetime,
            params_.moe_packed_up_lifetime,
            params_.moe_packed_down_lifetime,
            payload_provider_,
            params_.prepared_store,
            params_.expert_registry,
            params_.gate_slab_ref,
            params_.up_slab_ref,
            params_.down_slab_ref,
            true,
            nullptr,
            CPUExpertNUMAPlacement::forDevice(params_.device_id)};
    }

    void MoELocalExpertStage::bindPreparedExpertEnginesForExperts(
        const std::vector<int> &expert_ids)
    {
        auto bind = [this](ITensorGemm *engine)
        {
            if (!engine)
                return;
            if (params_.device_id.is_gpu())
                bindStageStream(engine);
            if (bound_workspace_)
            {
                if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(engine);
                    consumer && !consumer->hasWorkspace())
                {
                    consumer->bindWorkspace(bound_workspace_);
                }
            }
        };

        for (const int expert_id : expert_ids)
        {
            if (expert_id < 0 || expert_id >= params_.num_experts)
                continue;
            const size_t index = static_cast<size_t>(expert_id);
            if (index < params_.prepared_gate_gemm.size())
                bind(params_.prepared_gate_gemm[index]);
            if (index < params_.prepared_up_gemm.size())
                bind(params_.prepared_up_gemm[index]);
            if (index < params_.prepared_down_gemm.size())
                bind(params_.prepared_down_gemm[index]);
        }
    }

    bool MoELocalExpertStage::runtimeTableHasActiveOverlayBank() const
    {
        if (!params_.moe_runtime_table || params_.layer_idx < 0 || params_.num_experts <= 0)
            return false;
        if (params_.moe_runtime_table->decodeRuntimePublicationRequired(params_.layer_idx))
            return false;

        const auto &state = params_.moe_runtime_table->hostLayerState(params_.layer_idx);
        if (state.active_bank > 1 ||
            state.active_epoch == 0 ||
            state.expert_count != static_cast<uint32_t>(params_.num_experts) ||
            state.top_k != static_cast<uint32_t>(params_.top_k))
        {
            return false;
        }

        const auto &bank = state.banks[state.active_bank];
        if (bank.epoch != state.active_epoch ||
            bank.expert_count != static_cast<uint32_t>(params_.num_experts))
        {
            return false;
        }

        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            const auto mask = bank.local_compute_mask[static_cast<size_t>(expert_id)];
            if (mask > 1u)
                return false;
            if (mask != 0u &&
                !params_.expert_mask.empty() &&
                params_.expert_mask.size() == static_cast<size_t>(params_.num_experts) &&
                !params_.expert_mask[static_cast<size_t>(expert_id)])
            {
                return false;
            }
            if (mask == 0u)
                continue;

            const auto &expert = bank.experts[static_cast<size_t>(expert_id)];
            if (expert.logical_expert_id != expert_id ||
                expert.local_slot < 0 ||
                !expert.weightsReady() ||
                !hasMoEExpertFlag(expert.flags, DeviceMoEExpertFlags::Valid) ||
                !hasMoEExpertFlag(expert.flags, DeviceMoEExpertFlags::Resident) ||
                !hasMoEExpertFlag(expert.flags, DeviceMoEExpertFlags::LocalCompute))
            {
                return false;
            }
        }

        return moe_runtime_layer_ != nullptr;
    }

    bool MoELocalExpertStage::runtimeLocalComputeEnabled(int expert_id) const
    {
        if (!params_.moe_runtime_table || expert_id < 0 || expert_id >= params_.num_experts)
            return false;
        if (params_.moe_runtime_table->decodeRuntimePublicationRequired(params_.layer_idx))
            return false;
        const auto &state = params_.moe_runtime_table->hostLayerState(params_.layer_idx);
        if (state.active_bank > 1 || state.active_epoch == 0)
            return false;
        const auto &bank = state.banks[state.active_bank];
        return bank.local_compute_mask[static_cast<size_t>(expert_id)] != 0u;
    }

    bool MoELocalExpertStage::staticExpertMaskDisablesAllExperts() const
    {
        return !params_.expert_mask.empty() &&
               params_.expert_mask.size() == static_cast<size_t>(params_.num_experts) &&
               std::none_of(params_.expert_mask.begin(), params_.expert_mask.end(),
                            [](bool enabled)
                            { return enabled; });
    }

    bool MoELocalExpertStage::hasRuntimeLocalWorkForInput(const MoEOverlaySparseRows &input) const
    {
        for (size_t entry = 0; entry < input.live_entry_count; ++entry)
        {
            const int expert_id = input.expert_ids_host[entry];
            if (expert_id < 0 || expert_id >= params_.num_experts)
                return true;
            if (runtimeLocalComputeEnabled(expert_id))
                return true;
        }
        return false;
    }

    bool MoELocalExpertStage::isExpertActiveForValidation(int expert_id) const
    {
        if (params_.moe_runtime_table && moe_runtime_table_initialized_)
            return runtimeLocalComputeEnabled(expert_id);
        return params_.expert_mask.empty() ||
               (static_cast<size_t>(expert_id) < params_.expert_mask.size() &&
                params_.expert_mask[static_cast<size_t>(expert_id)]);
    }

    bool MoELocalExpertStage::ensureCompactCapacity(size_t rows, int routing_top_k) const
    {
        if (routing_top_k <= 0)
        {
            LOG_ERROR("[MoELocalExpertStage] Invalid compact routing top_k=" << routing_top_k);
            return false;
        }
        const size_t capacity = std::max<size_t>(rows, 1u);

        if (params_.serial_compact_buffer_arena)
        {
            const auto &arena = *params_.serial_compact_buffer_arena;
            if (arena.logicalParticipantId() &&
                params_.runtime_participant_index != *arena.logicalParticipantId())
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Serial compact-buffer arena participant "
                    "does not match the stage: arena_participant="
                    << *arena.logicalParticipantId()
                    << " stage_participant=" << params_.runtime_participant_index);
                return false;
            }
            if (!arena.supports(
                    params_.device_id,
                    capacity,
                    params_.d_model,
                    routing_top_k))
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Serial compact-buffer arena does not "
                    "match participant-local route geometry: device="
                    << params_.device_id.to_string()
                    << " requested_capacity=" << capacity
                    << " requested_d_model=" << params_.d_model
                    << " requested_routing_top_k=" << routing_top_k
                    << " arena_device=" << arena.deviceId().to_string()
                    << " arena_capacity=" << arena.rowCapacity()
                    << " arena_d_model=" << arena.dModel()
                    << " arena_routing_top_k=" << arena.routingTopK());
                return false;
            }

            /*
             * Select a complete tensor family, never a prefix view into the
             * maximum allocation. Tensor coherence is allocation-wide, so a
             * short decode packet must have its own host/device validity state
             * if its transfer is to remain short. All families were allocated
             * and prepared during setup; this merely changes which stable
             * address the heterogeneous manual boundary submits.
             */
            const auto *family = arena.smallestFamilySupporting(capacity);
            if (!family || !family->hidden || !family->routing_indices ||
                !family->routing_weights || !family->output)
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Serial compact-buffer arena has no "
                    "complete family for requested_capacity="
                    << capacity);
                return false;
            }
            compact_hidden_ = family->hidden;
            compact_routing_indices_ = family->routing_indices;
            compact_routing_weights_ = family->routing_weights;
            compact_output_ = family->output;
            compact_capacity_ = family->row_capacity;
            compact_routing_top_k_ = arena.routingTopK();
            return compact_hidden_ && compact_routing_indices_ &&
                   compact_routing_weights_ && compact_output_;
        }

        if (compact_capacity_ >= capacity &&
            compact_routing_top_k_ == routing_top_k)
        {
            return true;
        }

        /*
         * A hand-built stage without the serial-family contract retains its
         * original exclusive allocation.  This is an explicit construction
         * choice, not a runtime fallback from shared ownership.
         */
        compact_hidden_ = std::make_shared<FP32Tensor>(std::vector<size_t>{capacity, static_cast<size_t>(params_.d_model)});
        compact_routing_indices_ = std::make_shared<FP32Tensor>(std::vector<size_t>{capacity, static_cast<size_t>(routing_top_k)});
        compact_routing_weights_ = std::make_shared<FP32Tensor>(std::vector<size_t>{capacity, static_cast<size_t>(routing_top_k)});
        compact_output_ = std::make_shared<FP32Tensor>(std::vector<size_t>{capacity, static_cast<size_t>(params_.d_model)});
        compact_capacity_ = capacity;
        compact_routing_top_k_ = routing_top_k;
        return true;
    }

    bool MoELocalExpertStage::execute(IDeviceContext *ctx)
    {
        return executePacketOnCurrentThread(ctx);
    }

    bool MoELocalExpertStage::executePacketOnCurrentThread(
        IDeviceContext *ctx)
    {
        if (!validateDeviceContext(ctx, params_.device_id, "MoELocalExpertStage"))
            return false;
        if (deferred_lifecycle_ != DeferredLifecycle::Idle)
        {
            LOG_ERROR(
                "[MoELocalExpertStage] A second sparse packet was submitted before explicit completion"
                << " layer=" << params_.layer_idx
                << " participant=" << params_.runtime_participant_index);
            return false;
        }
        if (!params_.input_rows || !params_.output_rows)
        {
            LOG_ERROR("[MoELocalExpertStage] Missing input or output row view");
            return false;
        }
        if (params_.num_experts <= 0 || params_.top_k <= 0 || params_.d_model <= 0 || params_.expert_intermediate <= 0)
        {
            LOG_ERROR("[MoELocalExpertStage] Invalid dimensions num_experts=" << params_.num_experts
                                                                              << " top_k=" << params_.top_k
                                                                              << " d_model=" << params_.d_model
                                                                              << " intermediate=" << params_.expert_intermediate);
            return false;
        }

        if (!params_.expert_mask.empty() && params_.expert_mask.size() != static_cast<size_t>(params_.num_experts))
        {
            LOG_ERROR("[MoELocalExpertStage] expert_mask size " << params_.expert_mask.size()
                                                                << " != num_experts " << params_.num_experts);
            return false;
        }
        const auto &input = *params_.input_rows;
        auto &output = *params_.output_rows;
        if (!validateSparseRows(input, params_.top_k, params_.d_model))
            return false;
        if (input.live_row_count > row_admission_capacity_ ||
            input.live_entry_count > route_admission_capacity_)
        {
            LOG_ERROR(
                "[MoELocalExpertStage] Sparse live prefix exceeds this graph's immutable admission"
                << " live_rows=" << input.live_row_count
                << " admitted_rows=" << row_admission_capacity_
                << " live_routes=" << input.live_entry_count
                << " admitted_routes=" << route_admission_capacity_);
            return false;
        }
        if (output.d_model != params_.d_model || !output.row_ids_host || !output.output_rows_fp32 ||
            output.row_capacity < input.live_row_count)
        {
            LOG_ERROR("[MoELocalExpertStage] Return row view cannot hold local expert output");
            return false;
        }

        output.key = input.key;
        /* The return packet certifies the exact bank used by this endpoint. */
        output.residency_epoch = input.residency_epoch;
        output.source_participant = input.target_participant;
        output.target_participant = input.source_participant;
        output.live_row_count = 0;

        const bool economy_timing_enabled =
            params_.overlay_participant_residency &&
            params_.overlay_participant_residency
                ->collectsEconomyServiceMeasurements();
        const bool endpoint_detail_enabled =
            PerfStatsCollector::isDomainEnabled("moe_overlay_endpoint");
        const bool validate_finite_values =
            debugEnv().validation.fail_on_nan;
        const auto service_start =
            economy_timing_enabled || endpoint_detail_enabled
                                       ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};

        MoEOverlayParticipantBankLease execution_residency_bank;
        const MoEOverlayParticipantLayerBank *execution_layer_bank = nullptr;
        const auto *cpu_llep_state =
            params_.cpu_current_batch_llep_state.get();
        const bool cpu_llep_transaction = cpu_llep_state != nullptr;
        if (cpu_llep_transaction)
        {
            if (!cpu_llep_state->active ||
                !cpu_llep_state->durable_epoch_lease.has_value() ||
                cpu_llep_state->durable_epoch_lease->purpose() !=
                    MoEOverlayResidencyAuthority::TicketLeasePurpose::
                        CurrentBatchLLEP ||
                cpu_llep_state->durable_parent_epoch == 0 ||
                cpu_llep_state->durable_parent_epoch !=
                    input.residency_epoch ||
                cpu_llep_state->durable_parent_epoch !=
                    cpu_current_batch_llep_parent_epoch_ ||
                cpu_llep_state->layer_idx != params_.layer_idx ||
                cpu_llep_state->participant_id !=
                    params_.runtime_participant_index ||
                cpu_llep_state->transient_resident_mask.size() !=
                    static_cast<size_t>(params_.num_experts))
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Sparse CPU LLEP packet does not match its active child transaction");
                return false;
            }
        }
        if (params_.overlay_participant_residency)
        {
            if (params_.moe_runtime_table)
            {
                LOG_ERROR("[MoELocalExpertStage] Epoch-indexed participant residency and a mutable runtime table cannot both own placement");
                return false;
            }
            if (params_.runtime_participant_index !=
                    params_.overlay_participant_residency->participantId() ||
                params_.device_id !=
                    params_.overlay_participant_residency->device() ||
                params_.num_experts !=
                    params_.overlay_participant_residency->numExperts() ||
                params_.layer_idx < 0 ||
                params_.layer_idx >=
                    params_.overlay_participant_residency->numLayers())
            {
                LOG_ERROR("[MoELocalExpertStage] Participant residency identity or geometry does not match the local stage");
                return false;
            }
            if (input.residency_epoch == 0)
            {
                LOG_ERROR("[MoELocalExpertStage] Epoch-indexed participant execution received an unversioned sparse packet");
                return false;
            }

            /*
             * Retain the complete immutable bank through GEMM completion. A
             * maintenance retirement may remove its map entry concurrently,
             * but cannot destroy these engines while this shared owner exists.
             */
            execution_residency_bank =
                params_.overlay_participant_residency->acquire(
                    input.residency_epoch);
            if (!execution_residency_bank ||
                execution_residency_bank->epoch != input.residency_epoch ||
                execution_residency_bank->participant_id !=
                    params_.runtime_participant_index ||
                execution_residency_bank->device != params_.device_id ||
                static_cast<size_t>(params_.layer_idx) >=
                    execution_residency_bank->layers.size())
            {
                LOG_ERROR("[MoELocalExpertStage] Sparse packet references absent or retired residency epoch "
                          << input.residency_epoch);
                return false;
            }
            execution_layer_bank =
                &execution_residency_bank->layers[
                    static_cast<size_t>(params_.layer_idx)];
            /*
             * prepareReadyBank() validates every layer, mask bit, and expert
             * triplet before installReadyBank() publishes the prebuilt node.
             * Repeating that O(num_experts) proof on every routed packet adds
             * no safety: callers cannot mutate the acquired const bank. The
             * exact epoch/endpoint/layer checks above are the complete runtime
             * ticket validation; publication tests own the exhaustive proof.
             */
            if (cpu_llep_transaction &&
                execution_layer_bank->resident_mask !=
                    cpu_llep_state->owner_mask)
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] CPU LLEP parent bank no longer matches the authority owner row");
                return false;
            }
        }

        /*
         * A graph-native overlay endpoint may legitimately receive no rows,
         * but it must still leave evidence that the participant stage ran.
         * Without this zero-work record, a missing owner is indistinguishable
         * from an idle one when two CPU/NUMA endpoints share one backend name.
         */
        const auto recordZeroLocalWork = [&]()
        {
            if (!MoEExpertOverlayProfiler::isEnabled())
                return;
            MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
                params_.layer_idx,
                input.key.tier_idx,
                params_.runtime_participant_index,
                params_.device_id.to_string(),
                params_.device_id.is_cpu(),
                input.live_row_count,
                /*active_routes=*/0,
                /*output_rows=*/0,
                {},
                /*compute_ms=*/0.0);
        };

        if (params_.moe_runtime_table)
        {
            if (!moe_runtime_table_initialized_ || !runtimeTableHasActiveOverlayBank())
            {
                LOG_ERROR("[MoELocalExpertStage] Invalid or uninitialized MoE runtime placement table for layer "
                          << params_.layer_idx << " on " << params_.device_id.to_string());
                return false;
            }
        }

        if (input.live_row_count == 0)
        {
            recordZeroLocalWork();
            if (usesDeferredCompletion())
                deferred_lifecycle_ = DeferredLifecycle::ReadyNoWork;
            return true;
        }

        if (params_.moe_runtime_table && !hasRuntimeLocalWorkForInput(input))
        {
            recordZeroLocalWork();
            if (usesDeferredCompletion())
                deferred_lifecycle_ = DeferredLifecycle::ReadyNoWork;
            return true;
        }

        if (!execution_layer_bank && !params_.moe_runtime_table &&
            staticExpertMaskDisablesAllExperts())
        {
            recordZeroLocalWork();
            if (usesDeferredCompletion())
                deferred_lifecycle_ = DeferredLifecycle::ReadyNoWork;
            return true;
        }

        const bool has_prepared = execution_layer_bank != nullptr ||
                                  hasPreparedExpertState(params_);
        if (!has_prepared)
        {
            throw std::runtime_error(
                "MoELocalExpertStage requires graph-build-time prepared expert "
                "state; inline expert extraction/preparation is forbidden");
        }
        std::string prepared_error;
        if (!execution_layer_bank && !validatePreparedWeights(&prepared_error))
        {
            LOG_ERROR("[MoELocalExpertStage] Prepared weight validation failed: " << prepared_error);
            return false;
        }

        active_routes_.clear();
        output_input_rows_.clear();
        std::fill_n(row_output_slot_.begin(), input.live_row_count, -1);
        std::fill_n(validated_input_rows_.begin(), input.live_row_count, 0u);

        for (size_t row = 0; row < input.live_row_count; ++row)
        {
            const int row_id = input.row_ids_host[row];
            if (row_id < 0)
            {
                LOG_ERROR("[MoELocalExpertStage] Negative sparse row id " << row_id
                                                                           << " at compact row " << row);
                return false;
            }

            const int32_t entry_begin = input.entry_offsets_host[row];
            const int32_t entry_end = input.entry_offsets_host[row + 1u];
            if (entry_begin < 0 || entry_end < entry_begin ||
                entry_end > static_cast<int32_t>(input.live_entry_count) ||
                entry_end - entry_begin > params_.top_k)
            {
                LOG_ERROR("[MoELocalExpertStage] Invalid entry offsets for compact row " << row);
                return false;
            }

            for (int32_t entry = entry_begin; entry < entry_end; ++entry)
            {
                const int expert_id = input.expert_ids_host[entry];
                const float weight = input.route_weights_host[entry];
                if (expert_id < 0 || expert_id >= params_.num_experts)
                {
                    LOG_ERROR("[MoELocalExpertStage] Expert id " << expert_id
                                                                 << " outside num_experts=" << params_.num_experts);
                    return false;
                }
                if (!std::isfinite(weight))
                {
                    LOG_ERROR("[MoELocalExpertStage] Non-finite route weight for expert "
                              << expert_id << " at compact row " << row);
                    return false;
                }
                const bool expert_is_active =
                    cpu_llep_transaction
                        ? cpu_llep_state->transient_resident_mask[
                              static_cast<size_t>(expert_id)]
                        : execution_layer_bank
                        ? execution_layer_bank->resident_mask[
                              static_cast<size_t>(expert_id)]
                        : isExpertActiveForValidation(expert_id);
                if (weight == 0.0f || !expert_is_active)
                    continue;

                if (validate_finite_values && !validated_input_rows_[row])
                {
                    const float *hidden_row =
                        input.hiddenRowForCompactIndex(row);
                    if (!hidden_row)
                    {
                        LOG_ERROR(
                            "[MoELocalExpertStage] Sparse hidden-row address exceeds the selected payload matrix"
                            << " layer=" << params_.layer_idx
                            << " participant="
                            << params_.runtime_participant_index
                            << " compact_row=" << row
                            << " row_id=" << row_id
                            << " hidden_row_capacity="
                            << input.hidden_row_capacity
                            << " layout="
                            << static_cast<int>(
                                   input.hidden_payload_layout));
                        return false;
                    }
                    for (int col = 0; col < params_.d_model; ++col)
                    {
                        if (!std::isfinite(hidden_row[col]))
                        {
                            LOG_ERROR("[MoELocalExpertStage] Non-finite sparse hidden value for layer "
                                      << params_.layer_idx
                                      << " participant=" << params_.runtime_participant_index
                                      << " compact_row=" << row
                                      << " row_id=" << row_id
                                      << " col=" << col);
                            return false;
                        }
                    }
                    validated_input_rows_[row] = 1u;
                }

                if (row_output_slot_[row] < 0)
                {
                    row_output_slot_[row] =
                        static_cast<int>(output_input_rows_.size());
                    output_input_rows_.push_back(row);
                }
                active_routes_.push_back(ActiveRoute{row, expert_id, weight});
            }
        }

        if (active_routes_.empty())
        {
            recordZeroLocalWork();
            if (usesDeferredCompletion())
                deferred_lifecycle_ = DeferredLifecycle::ReadyNoWork;
            return true;
        }

        const size_t compact_live_rows = output_input_rows_.size();
        if (compact_live_rows == 0)
            return false;

        /*
         * Count routes using the already established compact-row mapping, then
         * choose one setup-created 1/2/4/.../top-k graph. The route order inside
         * each row is unchanged; only invalid tail slots disappear from the
         * captured launch geometry. Tensor allocation retains the full model
         * top-k so every graph variant borrows the same stable addresses.
         */
        std::fill_n(
            compact_row_route_counts_.begin(), compact_live_rows, size_t{0});
        size_t maximum_local_routes_per_row = 0;
        for (const auto &route : active_routes_)
        {
            const int compact_row = row_output_slot_[route.input_row];
            if (compact_row < 0 ||
                static_cast<size_t>(compact_row) >= compact_live_rows)
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Active route lost its compact-row slot while selecting launch width");
                return false;
            }
            const size_t route_count =
                ++compact_row_route_counts_[static_cast<size_t>(compact_row)];
            if (route_count > static_cast<size_t>(params_.top_k))
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Participant-local routes exceed model top_k while selecting launch width");
                return false;
            }
            maximum_local_routes_per_row =
                std::max(maximum_local_routes_per_row, route_count);
        }
        const int compact_top_k =
            MoELocalExpertSerialBufferArena::routeWidthBucketFor(
                static_cast<int>(maximum_local_routes_per_row),
                params_.top_k);
        if (compact_top_k <= 0 ||
            !ensureCompactCapacity(compact_live_rows, params_.top_k))
        {
            return false;
        }
        compact_execution_top_k_ = compact_top_k;
        const size_t compact_family_bytes =
            compact_hidden_->size_bytes() +
            compact_routing_indices_->size_bytes() +
            compact_routing_weights_->size_bytes() +
            compact_output_->size_bytes();
        if (params_.device_id.is_gpu() &&
            PerfStatsCollector::isDomainEnabled("transfer"))
        {
            const size_t selected_h2d_bytes =
                compact_hidden_->size_bytes() +
                compact_routing_indices_->size_bytes() +
                compact_routing_weights_->size_bytes();
            const size_t selected_d2h_bytes = compact_output_->size_bytes();
            const size_t useful_hidden_bytes =
                checkedMultiply(
                    compact_live_rows,
                    static_cast<size_t>(params_.d_model) * sizeof(float),
                    "useful compact hidden bytes");
            const size_t useful_routing_bytes =
                checkedMultiply(
                    active_routes_.size(),
                    size_t{2} * sizeof(float),
                    "useful compact routing bytes");
            const size_t useful_h2d_bytes = checkedAdd(
                useful_hidden_bytes,
                useful_routing_bytes,
                "useful compact H2D bytes");
            const size_t useful_d2h_bytes = useful_hidden_bytes;
            const PerfStatsCollector::Tags transfer_tags{
                {"active_routes", std::to_string(active_routes_.size())},
                {"family_capacity", std::to_string(compact_capacity_)},
                {"layer", std::to_string(params_.layer_idx)},
                {"participant",
                 std::to_string(params_.runtime_participant_index)},
                {"route_width", std::to_string(compact_top_k)},
            };
            PerfStatsCollector::addCounter(
                "transfer",
                "moe_overlay_local_expert_h2d_bytes",
                static_cast<double>(selected_h2d_bytes),
                "sparse_endpoint",
                params_.device_id.to_string(),
                transfer_tags);
            PerfStatsCollector::addCounter(
                "transfer",
                "moe_overlay_local_expert_d2h_bytes",
                static_cast<double>(selected_d2h_bytes),
                "sparse_endpoint",
                params_.device_id.to_string(),
                transfer_tags);
            PerfStatsCollector::addCounter(
                "transfer",
                "moe_overlay_local_expert_padding_bytes",
                static_cast<double>(
                    selected_h2d_bytes + selected_d2h_bytes -
                    useful_h2d_bytes - useful_d2h_bytes),
                "sparse_endpoint",
                params_.device_id.to_string(),
                transfer_tags);
        }

        DeferredGPUReplayFamily *const compact_replay =
            usesDeferredCompletion() ? currentDeferredReplayFamily() : nullptr;
        const auto *const compact_family =
            compact_replay ? compact_replay->tensor_family : nullptr;
        if (usesDeferredCompletion() &&
            (!compact_family || !compact_family->pinned_transfer ||
             !compact_family->pinned_transfer->isBound()))
        {
            LOG_ERROR(
                "[MoELocalExpertStage] Retained sparse packet has no bound fixed pinned transfer family");
            return false;
        }

        /*
         * Retained GPU graphs consume a distinct backend-pinned staging family.
         * Writing it never asks Tensor coherence to download the preceding
         * device generation; the graph's first captured node overwrites the
         * complete device tensor capacity from these exact addresses.
         */
        float *hidden = compact_family
                            ? static_cast<float *>(
                                  compact_family->pinned_transfer->mutableData(
                                      compact_family->pinned_hidden_offset))
                            : compact_hidden_->mutable_data();
        float *routing_indices = compact_family
                                     ? static_cast<float *>(
                                           compact_family->pinned_transfer->mutableData(
                                               compact_family->pinned_routing_indices_offset))
                                     : compact_routing_indices_->mutable_data();
        float *routing_weights = compact_family
                                     ? static_cast<float *>(
                                           compact_family->pinned_transfer->mutableData(
                                               compact_family->pinned_routing_weights_offset))
                                     : compact_routing_weights_->mutable_data();
        /*
         * A retained GPU family publishes into a separate pinned destination,
         * so touching the tensor's stale host allocation here would first
         * download the preceding layer output. The fixed-capacity grouped
         * kernel overwrites every active and invalid-padded row. Inline CPU/GPU
         * execution still uses Tensor coherence and therefore clears its host
         * output before launch.
         */
        float *compact_output = usesDeferredCompletion()
                                    ? nullptr
                                    : compact_output_->mutable_data();
        const size_t compact_hidden_elements =
            compact_capacity_ * static_cast<size_t>(params_.d_model);
        const size_t compact_routing_elements =
            compact_capacity_ * static_cast<size_t>(compact_top_k);
        /*
         * A retained graph always executes its immutable bucket geometry.
         * Padding every unused row with an invalid expert and zero weight makes
         * the full-capacity launch mathematically identical to the live prefix,
         * Clearing hidden storage prevents stale bytes from obscuring an
         * invalid-route kernel defect. Inline execution also clears its tensor
         * output; retained GPU execution proves full overwrite by publishing
         * the complete fixed-capacity result from inside the captured graph.
         */
        /*
         * Every live hidden row is overwritten below and every padded row has
         * expert=-1/weight=0. Debug/integration validation clears the padding
         * so a backend defect is easy to diagnose; Release avoids writing a
         * capacity-wide 3,072-column host buffer whose bytes are mathematically
         * unreachable. The retained H2D node remains fixed-size and address
         * stable in both modes.
         */
        if (validate_finite_values)
            std::fill_n(hidden, compact_hidden_elements, 0.0f);
        std::fill_n(routing_indices, compact_routing_elements, -1.0f);
        std::fill_n(routing_weights, compact_routing_elements, 0.0f);
        if (compact_output)
            std::fill_n(compact_output, compact_hidden_elements, 0.0f);

        std::fill_n(
            compact_row_route_counts_.begin(), compact_live_rows, size_t{0});
        for (size_t compact_row = 0; compact_row < compact_live_rows; ++compact_row)
        {
            const size_t input_row = output_input_rows_[compact_row];
            const float *const source_hidden =
                input.hiddenRowForCompactIndex(input_row);
            if (!source_hidden)
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Active sparse row has no address in its selected hidden payload"
                    << " compact_row=" << input_row
                    << " row_id=" << input.row_ids_host[input_row]
                    << " hidden_row_capacity="
                    << input.hidden_row_capacity
                    << " layout="
                    << static_cast<int>(input.hidden_payload_layout));
                return false;
            }
            std::memcpy(
                hidden + compact_row * static_cast<size_t>(params_.d_model),
                source_hidden,
                static_cast<size_t>(params_.d_model) * sizeof(float));
        }
        for (const auto &route : active_routes_)
        {
            const int compact_row = row_output_slot_[route.input_row];
            if (compact_row < 0 ||
                static_cast<size_t>(compact_row) >= compact_live_rows)
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Active route lost its compact-row slot");
                return false;
            }
            size_t &route_slot =
                compact_row_route_counts_[static_cast<size_t>(compact_row)];
            if (route_slot >= static_cast<size_t>(compact_top_k))
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Participant-local routes exceed model top_k");
                return false;
            }
            const size_t flat_slot =
                static_cast<size_t>(compact_row) *
                    static_cast<size_t>(compact_top_k) +
                route_slot++;
            routing_indices[flat_slot] = static_cast<float>(route.expert_id);
            routing_weights[flat_slot] = route.weight;
        }

        /*
         * The optional endpoint trace divides the complete sparse service
         * interval. Deferred GPU execution publishes into pinned memory inside
         * its retained graph and the explicit completion node waits on that
         * graph event. Inline execution retains the tensor-owned host
         * materialization boundary. Timing reads add no second synchronization
         * beyond the required return-row boundary.
         */
        const auto stage_setup_start = endpoint_detail_enabled
                                           ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};
        const uint64_t engine_binding_generation = input.residency_epoch;
        const bool reuse_deferred_engine_binding =
            usesDeferredCompletion() && !cpu_llep_transaction &&
            invocation_engine_binding_generation_ ==
                engine_binding_generation;
        if (!reuse_deferred_engine_binding)
        {
            std::fill(
                invocation_gate_engines_.begin(),
                invocation_gate_engines_.end(),
                nullptr);
            std::fill(
                invocation_up_engines_.begin(),
                invocation_up_engines_.end(),
                nullptr);
            std::fill(
                invocation_down_engines_.begin(),
                invocation_down_engines_.end(),
                nullptr);
            std::fill(
                invocation_expert_mask_.begin(),
                invocation_expert_mask_.end(),
                false);

            // Resolve one complete construction-sized invocation table only
            // when the immutable authority publishes a different epoch.
            if (execution_layer_bank)
            {
                /*
                 * The nested grouped expert stage receives a complete immutable
                 * view of the selected epoch. It must not consult the globally
                 * current runtime bank after this packet selected an older one.
                 */
                const auto &execution_mask =
                    cpu_llep_transaction
                        ? cpu_llep_state->transient_resident_mask
                        : execution_layer_bank->resident_mask;
                std::copy(
                    execution_mask.begin(),
                    execution_mask.end(),
                    invocation_expert_mask_.begin());
                for (int expert = 0; expert < params_.num_experts; ++expert)
                {
                    const size_t expert_index = static_cast<size_t>(expert);
                    if (!invocation_expert_mask_[expert_index])
                    {
                        continue;
                    }
                    const PreparedExpertEngines *transient = nullptr;
                    if (cpu_llep_transaction &&
                        !execution_layer_bank->resident_mask[expert_index])
                    {
                        transient =
                            &cpu_current_batch_llep_transient_arrivals_[
                                expert_index];
                        if (!transient->complete())
                        {
                            LOG_ERROR(
                                "[MoELocalExpertStage] CPU LLEP transient resident has no complete arrival engines");
                            return false;
                        }
                    }
                    const auto &engines =
                        execution_layer_bank->experts[expert_index];
                    invocation_gate_engines_[expert_index] =
                        transient ? transient->gate.get() : engines.gate.get();
                    invocation_up_engines_[expert_index] =
                        transient ? transient->up.get() : engines.up.get();
                    invocation_down_engines_[expert_index] =
                        transient ? transient->down.get() : engines.down.get();
                }
            }
            else
            {
                if (params_.expert_mask.empty())
                {
                    std::fill(
                        invocation_expert_mask_.begin(),
                        invocation_expert_mask_.end(),
                        true);
                }
                else
                {
                    std::copy(
                        params_.expert_mask.begin(),
                        params_.expert_mask.end(),
                        invocation_expert_mask_.begin());
                }
                std::copy(
                    params_.prepared_gate_gemm.begin(),
                    params_.prepared_gate_gemm.end(),
                    invocation_gate_engines_.begin());
                std::copy(
                    params_.prepared_up_gemm.begin(),
                    params_.prepared_up_gemm.end(),
                    invocation_up_engines_.begin());
                std::copy(
                    params_.prepared_down_gemm.begin(),
                    params_.prepared_down_gemm.end(),
                    invocation_down_engines_.begin());
            }
        }

        // Inline GPU execution retains Tensor-aware placement. Retained sparse
        // execution performs all H2D/D2H work inside its one captured graph.
        const bool is_gpu = params_.device_id.is_gpu();
        if (is_gpu && !compact_replay)
        {
            const StageGPUExecution execution = gpuExecution();
            execution.prepareInput(compact_hidden_.get());
            execution.prepareInput(compact_routing_indices_.get());
            execution.prepareInput(compact_routing_weights_.get());
            execution.prepareOutput(compact_output_.get());
        }

        std::optional<MoEExpertComputeStage> inline_compute_stage;
        MoEExpertComputeStage *compute_stage = nullptr;
        DeferredGPUReplayFamily *deferred_replay = nullptr;
        if (usesDeferredCompletion())
        {
            deferred_replay = compact_replay;
            if (!deferred_replay || !deferred_replay->compute_stage ||
                !deferred_replay->executor)
            {
                return false;
            }
            if (!deferred_replay->compute_stage->bindSparseOverlayInvocation(
                    MoEExpertComputeStage::SparseOverlayInvocation{
                        .input = compact_hidden_.get(),
                        .routing_indices = compact_routing_indices_.get(),
                        .routing_weights = compact_routing_weights_.get(),
                        .output = compact_output_.get(),
                        .live_rows = static_cast<int>(compact_capacity_),
                        .engine_binding_generation =
                            engine_binding_generation,
                        .expert_mask = &invocation_expert_mask_,
                        .gate_engines = invocation_gate_engines_,
                        .up_engines = invocation_up_engines_,
                        .down_engines = invocation_down_engines_,
                    }))
            {
                return false;
            }
            invocation_engine_binding_generation_ =
                engine_binding_generation;
            compute_stage = deferred_replay->compute_stage;
        }
        else
        {
            MoEExpertComputeStage::Params compute_params;
            compute_params.device_id = params_.device_id;
            compute_params.input = compact_hidden_.get();
            compute_params.seq_len = static_cast<int>(compact_live_rows);
            compute_params.d_model = params_.d_model;
            compute_params.num_experts = params_.num_experts;
            compute_params.top_k = compact_top_k;
            compute_params.gate_exps = params_.gate_exps;
            compute_params.up_exps = params_.up_exps;
            compute_params.down_exps = params_.down_exps;
            compute_params.expert_intermediate = params_.expert_intermediate;
            compute_params.layer_idx = params_.layer_idx;
            compute_params.expert_mask = invocation_expert_mask_;
            compute_params.routing_indices = compact_routing_indices_.get();
            compute_params.routing_weights = compact_routing_weights_.get();
            compute_params.output = compact_output_.get();
            compute_params.output_registered_in_arena = false;
            compute_params.cpu_router_q8_input_publication =
                params_.device_id.is_cpu()
                    ? CPURouterQ8InputPublicationPolicy::PublishTransportedRows
                    : CPURouterQ8InputPublicationPolicy::RequireRouterStage;
            compute_params.prepared_gate_gemm = invocation_gate_engines_;
            compute_params.prepared_up_gemm = invocation_up_engines_;
            compute_params.prepared_down_gemm = invocation_down_engines_;
            compute_params.prepared_store = params_.prepared_store;
            compute_params.expert_registry = params_.expert_registry;
            compute_params.moe_runtime_table = execution_layer_bank
                                                   ? nullptr
                                                   : params_.moe_runtime_table;
            compute_params.overlay_service_telemetry =
                params_.overlay_service_telemetry;
            compute_params.gate_slab_ref = params_.gate_slab_ref;
            compute_params.up_slab_ref = params_.up_slab_ref;
            compute_params.down_slab_ref = params_.down_slab_ref;
            inline_compute_stage.emplace(std::move(compute_params));
            inline_compute_stage->releaseRawExpertWeights();
            compute_stage = &*inline_compute_stage;
        }

        if (bound_workspace_ && deferred_replay)
        {
            if (compute_stage->getWorkspace() != bound_workspace_)
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Retained sparse executor lost its setup-bound workspace");
                return false;
            }
        }
        else if (bound_workspace_)
        {
            compute_stage->bindWorkspace(bound_workspace_);
        }
        else if (is_gpu)
        {
            LOG_ERROR("[MoELocalExpertStage] GPU local expert compute requires a bound workspace");
            return false;
        }
        const bool profiling_enabled = MoEExpertOverlayProfiler::isEnabled();
        std::chrono::steady_clock::time_point t_compute_start;
        if (profiling_enabled || endpoint_detail_enabled)
            t_compute_start = std::chrono::steady_clock::now();

        if (deferred_replay)
        {
            IWorkerGPUContext *const worker =
                params_.worker_gpu_context;
            void *const producer_stream = requireGPUStream();
            if (!worker)
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Retained sparse endpoint graph requires its exact worker context and producer stream"
                    << " layer=" << params_.layer_idx
                    << " participant=" << params_.runtime_participant_index
                    << " device=" << params_.device_id.to_string());
                return false;
            }

            /*
             * This endpoint is already serialized on one participant stream.
             * Borrowing that exact stream makes the compact H2D copies the
             * graph's direct predecessors and lets host materialization observe
             * its event without two extra cross-stream edges per layer.
             */
            if (!deferred_replay->cache.capture_stream &&
                !deferred_replay->cache.bindBorrowedCaptureStream(
                    worker,
                    producer_stream,
                    params_.device_id,
                    /*context_from_process_pool=*/false))
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Could not bind retained sparse graph to the exact participant stream");
                return false;
            }

            const bool was_initialized = deferred_replay->cache.initialized;

            if (!deferred_replay->executor->executeWithCachedGraphReplay(
                    deferred_replay->graph,
                    ctx,
                    deferred_replay->cache,
                    producer_stream,
                    worker,
                    /*collective_nodes=*/nullptr,
                    /*collectives_graph_capturable=*/false,
                    /*force_recapture=*/false,
                    /*defer_final_sync=*/true,
                    {},
                    DeviceGraphExecutor::GraphReplayPlanPolicy::
                        RequireFullGraph))
            {
                return false;
            }
            if (!deferred_replay->cache.capture_stream)
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Retained sparse endpoint graph launched without an exact capture stream");
                return false;
            }

            if (PerfStatsCollector::isDomainEnabled("forward_graph"))
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    was_initialized
                        ? "moe_overlay_local_expert_graph_replays"
                        : "moe_overlay_local_expert_graph_captures",
                    1.0,
                    "moe_overlay",
                    params_.device_id.to_string(),
                    {{"family_capacity",
                      std::to_string(deferred_replay->row_capacity)},
                     {"layer", std::to_string(params_.layer_idx)},
                     {"participant",
                      std::to_string(params_.runtime_participant_index)},
                     {"route_width",
                      std::to_string(deferred_replay->route_width)}});
            }
            pending_deferred_replay_ = deferred_replay;
        }
        else
        {
            compute_stage->setGPUStream(gpuStream());
            if (!compute_stage->execute(ctx))
                return false;
        }

        const auto t_compute_end = profiling_enabled || endpoint_detail_enabled
                                       ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};

        pending_economy_timing_enabled_ = economy_timing_enabled;
        pending_endpoint_detail_enabled_ = endpoint_detail_enabled;
        pending_profiling_enabled_ = profiling_enabled;
        pending_compact_family_bytes_ = compact_family_bytes;
        pending_service_start_ = service_start;
        pending_stage_setup_start_ = stage_setup_start;
        pending_compute_start_ = t_compute_start;
        pending_compute_end_ = t_compute_end;
        deferred_residency_bank_ = std::move(execution_residency_bank);
        deferred_lifecycle_ = DeferredLifecycle::Submitted;

        if (usesDeferredCompletion())
            return true;
        return completeDeferredOutputOnCurrentThread(ctx);
    }

    bool MoELocalExpertStage::completeDeferredOutput(IDeviceContext *ctx)
    {
        return completeDeferredOutputOnCurrentThread(ctx);
    }

    bool MoELocalExpertStage::completeDeferredOutputOnCurrentThread(
        IDeviceContext *ctx)
    {
        if (!validateDeviceContext(
                ctx, params_.device_id, "MoELocalExpertCompletionStage"))
            return false;
        if (deferred_lifecycle_ == DeferredLifecycle::Idle)
        {
            LOG_ERROR(
                "[MoELocalExpertStage] Completion executed without a submitted sparse packet"
                << " layer=" << params_.layer_idx
                << " participant=" << params_.runtime_participant_index);
            return false;
        }
        const auto &input = *params_.input_rows;
        if (deferred_lifecycle_ == DeferredLifecycle::ReadyNoWork)
        {
            resetDeferredInvocation();
            if (PerfStatsCollector::isDomainEnabled("forward_graph"))
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "moe_overlay_local_expert_completions",
                    1.0,
                    "moe_overlay",
                    params_.device_id.to_string(),
                    {{"active_routes", "0"},
                     {"layer", std::to_string(params_.layer_idx)},
                     {"participant", std::to_string(params_.runtime_participant_index)},
                     {"service_source",
                      serviceSourceName(input.key.histogram_source)}});
            }
            return true;
        }

        auto &output = *params_.output_rows;
        const bool economy_timing_enabled =
            pending_economy_timing_enabled_;
        const bool endpoint_detail_enabled =
            pending_endpoint_detail_enabled_;
        const bool validate_finite_values =
            debugEnv().validation.fail_on_nan;
        const bool profiling_enabled = pending_profiling_enabled_;
        const size_t compact_family_bytes =
            pending_compact_family_bytes_;
        const auto service_start = pending_service_start_;
        const auto stage_setup_start = pending_stage_setup_start_;
        const auto t_compute_start = pending_compute_start_;
        const auto t_compute_end = pending_compute_end_;
        const double compute_ms = profiling_enabled
                                      ? std::chrono::duration<double, std::milli>(
                                            t_compute_end - t_compute_start)
                                            .count()
                                      : 0.0;
        const auto fail_completed_packet = [&]()
        {
            /* The host materialization boundary has already quiesced GPU work. */
            resetDeferredInvocation();
            return false;
        };

        const float *compact_result = nullptr;
        if (usesDeferredCompletion())
        {
            if (!pending_deferred_replay_)
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Submitted retained GPU packet lost its exact replay-family authority");
                std::terminate();
            }

            /*
             * This is the sole host wait for the participant transaction. The
             * retained graph has already enqueued compute followed by D2H into
             * its backend-pinned family buffer, so rank-batched callers can
             * submit every sibling GPU before reaching any one of these waits.
             */
            pending_deferred_replay_->cache.waitForCaptureStreamFence(
                DeviceGraphExecutor::GraphSegmentCache::HostFenceWaitPolicy::
                    ActiveProgress);
            const auto *const family =
                pending_deferred_replay_->tensor_family;
            if (!family || family->output.get() != compact_output_.get() ||
                !family->pinned_transfer ||
                !family->pinned_transfer->matches(
                    family->pinned_transfer_bytes,
                    params_.device_id) ||
                !family->pinned_transfer->contains(
                    family->pinned_output_offset,
                    compact_output_->size_bytes()))
            {
                LOG_ERROR(
                    "[MoELocalExpertStage] Captured output publication identity changed before completion");
                return fail_completed_packet();
            }
            compact_result = static_cast<const float *>(
                family->pinned_transfer->data(
                    family->pinned_output_offset));
        }
        else
        {
            compact_result = compact_output_->data();
        }
        if (!compact_result)
        {
            LOG_ERROR(
                "[MoELocalExpertStage] Host-visible compact output is null after publication");
            return fail_completed_packet();
        }
        const auto output_materialized_at = endpoint_detail_enabled
                                                ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
        if (validate_finite_values)
        {
            for (size_t compact_row = 0;
                 compact_row < output_input_rows_.size();
                 ++compact_row)
            {
                const float *src =
                    compact_result +
                    compact_row * static_cast<size_t>(params_.d_model);
                for (int col = 0; col < params_.d_model; ++col)
                {
                    if (!std::isfinite(src[col]))
                    {
                        LOG_ERROR("[MoELocalExpertStage] Non-finite local expert output for layer "
                                  << params_.layer_idx
                                  << " participant=" << params_.runtime_participant_index
                                  << " compact_row=" << compact_row
                                  << " input_row=" << output_input_rows_[compact_row]
                                  << " row_id="
                                  << input.row_ids_host[
                                         output_input_rows_[compact_row]]
                                  << " col=" << col);
                        return fail_completed_packet();
                    }
                }
            }
        }

        for (size_t output_row = 0; output_row < output_input_rows_.size(); ++output_row)
        {
            const size_t input_row = output_input_rows_[output_row];
            output.row_ids_host[output_row] = input.row_ids_host[input_row];
            std::memcpy(
                output.output_rows_fp32 +
                    output_row * static_cast<size_t>(params_.d_model),
                compact_result +
                    output_row * static_cast<size_t>(params_.d_model),
                static_cast<size_t>(params_.d_model) * sizeof(float));
        }
        if (validate_finite_values)
        {
            for (size_t output_row = 0;
                 output_row < output_input_rows_.size();
                 ++output_row)
            {
                const float *row =
                    output.output_rows_fp32 +
                    output_row * static_cast<size_t>(params_.d_model);
                for (int col = 0; col < params_.d_model; ++col)
                {
                    if (!std::isfinite(row[col]))
                    {
                        LOG_ERROR("[MoELocalExpertStage] Non-finite aggregated local expert output for layer "
                                  << params_.layer_idx
                                  << " participant=" << params_.runtime_participant_index
                                  << " output_row=" << output_row
                                  << " row_id=" << output.row_ids_host[output_row]
                                  << " col=" << col);
                        return fail_completed_packet();
                    }
                }
            }
        }
        output.live_row_count = output_input_rows_.size();

        const auto service_completed_at =
            economy_timing_enabled || endpoint_detail_enabled
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
        if (economy_timing_enabled)
        {
            /*
             * This is the cost the tier planner can actually save.  In
             * particular, a GPU endpoint is not just its expert kernel: its
             * heterogeneous packet service includes host compaction, the
             * intentional H2D boundary, device execution, host-visible output
             * materialization, validation, and return-row aggregation.
             * The captured pinned publication (or inline tensor materialization)
             * above already establishes the required completion edge, so
             * reading the host clock here adds no wait or device synchronization
             * to inference.
             */
            const auto elapsed_ns_signed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    service_completed_at - service_start)
                    .count();
            if (elapsed_ns_signed <= 0 ||
                !publishServiceMeasurement(
                    input.key.histogram_source,
                    static_cast<uint64_t>(elapsed_ns_signed),
                    static_cast<uint64_t>(active_routes_.size())))
            {
                return fail_completed_packet();
            }
        }

        if (endpoint_detail_enabled)
        {
            const auto completed_at = service_completed_at;
            const auto elapsedNs = [](auto begin, auto end) -> uint64_t
            {
                const auto value =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end - begin)
                        .count();
                return value > 0 ? static_cast<uint64_t>(value) : 1u;
            };
            std::set<int> traced_experts;
            for (const auto &route : active_routes_)
                traced_experts.insert(route.expert_id);
            std::ostringstream expert_ids;
            bool first_expert = true;
            for (const int expert : traced_experts)
            {
                if (!first_expert)
                    expert_ids << ':';
                expert_ids << expert;
                first_expert = false;
            }

            const PerfStatsCollector::Tags tags{
                {"active_expert_count", std::to_string(traced_experts.size())},
                {"active_experts", expert_ids.str()},
                {"active_routes", std::to_string(active_routes_.size())},
                {"compact_family_bytes", std::to_string(compact_family_bytes)},
                {"compact_row_capacity", std::to_string(compact_capacity_)},
                {"input_rows", std::to_string(input.live_row_count)},
                {"generation", std::to_string(input.key.generation_id)},
                {"layer", std::to_string(params_.layer_idx)},
                {"logical_step", std::to_string(input.key.step_id)},
                {"output_rows", std::to_string(output.live_row_count)},
                {"participant", std::to_string(params_.runtime_participant_index)},
                {"residency_epoch", std::to_string(input.residency_epoch)},
                {"route_width", std::to_string(compact_execution_top_k_)},
                {"tier", std::to_string(input.key.tier_idx)}};
            const std::string phase =
                serviceSourceName(input.key.histogram_source);
            const std::string device = params_.device_id.to_string();
            const auto record = [&](const char *name, auto begin, auto end)
            {
                PerfStatsCollector::recordTimingNs(
                    "moe_overlay_endpoint",
                    name,
                    elapsedNs(begin, end),
                    phase,
                    device,
                    tags);
            };
            record("packet_service", service_start, completed_at);
            record("route_validation_and_compaction", service_start, stage_setup_start);
            record("stage_setup_and_transfers", stage_setup_start, t_compute_start);
            record("compute_submission", t_compute_start, t_compute_end);
            record("output_materialization", t_compute_end, output_materialized_at);
            record("return_validation_and_aggregation", output_materialized_at, completed_at);
        }

        if (profiling_enabled)
        {
            std::set<int> active_expert_ids;
            for (const auto &route : active_routes_)
                active_expert_ids.insert(route.expert_id);
            MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
                params_.layer_idx,
                input.key.tier_idx,
                params_.runtime_participant_index,
                params_.device_id.to_string(),
                params_.device_id.is_cpu(),
                input.live_row_count,
                active_routes_.size(),
                output.live_row_count,
                std::vector<int>(
                    active_expert_ids.begin(), active_expert_ids.end()),
                compute_ms);
        }

        /*
         * PerfStats is the always-available production witness, whereas the
         * detailed overlay profiler is intentionally opt-in.  Publish this
         * record only after the prepared local GEMMs and compact return rows
         * have completed, so a route entry proves real participant work rather
         * than merely dispatch admission.  The sparse key supplies the same
         * root-published generation and logical step carried by the paired
         * dispatch/return collective; tests can therefore distinguish a live
         * CPU cold-tier route from an idle graph endpoint without maintaining a
         * host shadow of routing state.
         */
        if (PerfStatsCollector::isDomainEnabled("forward_graph"))
        {
            const char *const device_kind =
                params_.device_id.is_cpu()
                    ? "CPU"
                    : (params_.device_id.is_cuda() ? "CUDA" : "ROCm");
            PerfStatsCollector::addCounter(
                "forward_graph",
                "moe_overlay_local_expert_active_routes",
                static_cast<double>(active_routes_.size()),
                "moe_overlay",
                params_.device_id.to_string(),
                {{"completion", "local_expert_packet_complete"},
                 {"device_kind", device_kind},
                 {"generation", std::to_string(input.key.generation_id)},
                 {"identity_source", "sparse_collective_key"},
                 {"input_rows", std::to_string(input.live_row_count)},
                 {"compact_family_bytes", std::to_string(compact_family_bytes)},
                 {"compact_row_capacity", std::to_string(compact_capacity_)},
                 {"logical_step", std::to_string(input.key.step_id)},
                 {"output_rows", std::to_string(output.live_row_count)},
                 {"participant", std::to_string(params_.runtime_participant_index)},
                 {"service_source",
                  serviceSourceName(input.key.histogram_source)},
                 {"tier", std::to_string(input.key.tier_idx)}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                "moe_overlay_local_expert_completions",
                1.0,
                "moe_overlay",
                params_.device_id.to_string(),
                {{"active_routes", std::to_string(active_routes_.size())},
                 {"host_fence_policy",
                  usesDeferredCompletion() ? "active_progress_event" : "inline"},
                 {"host_execution",
                  "rank_graph_submit_completion_wave"},
                 {"layer", std::to_string(params_.layer_idx)},
                 {"participant", std::to_string(params_.runtime_participant_index)},
                 {"service_source",
                  serviceSourceName(input.key.histogram_source)}});
        }
        resetDeferredInvocation();
        return true;
    }

    bool MoELocalExpertStage::validatePreparedWeights(std::string *error) const
    {
        if (error)
            error->clear();

        const bool has_prepared_vectors =
            !params_.prepared_gate_gemm.empty() &&
            params_.prepared_gate_gemm.size() == static_cast<size_t>(params_.num_experts);
        const bool has_slab_refs = params_.prepared_store &&
                                   params_.gate_slab_ref.has_value() &&
                                   params_.up_slab_ref.has_value() &&
                                   params_.down_slab_ref.has_value();

        if (!has_prepared_vectors && !has_slab_refs)
        {
            // No prepared state — stage will use inline view extraction (legacy path).
            // This is only valid when raw tensors are present.
            if (params_.gate_exps && params_.up_exps && params_.down_exps)
                return true;
            if (error)
                *error = "[MoELocalExpertStage] No prepared expert state and no raw expert tensors";
            return false;
        }

        if (has_prepared_vectors)
        {
            // Verify every active expert has a non-null engine.
            const auto &gate_v = params_.prepared_gate_gemm;
            const auto &up_v = params_.prepared_up_gemm;
            const auto &down_v = params_.prepared_down_gemm;

            if (up_v.size() != gate_v.size() || down_v.size() != gate_v.size())
            {
                if (error)
                    *error = "[MoELocalExpertStage] Prepared engine vector size mismatch";
                return false;
            }

            for (int expert = 0; expert < params_.num_experts; ++expert)
            {
                const bool active = isExpertActiveForValidation(expert);
                if (!active)
                    continue;
                const bool ok = gate_v[expert] && up_v[expert] && down_v[expert];
                if (!ok)
                {
                    if (error)
                    {
                        std::ostringstream oss;
                        oss << "[MoELocalExpertStage] Layer " << params_.layer_idx
                            << " expert " << expert << " is active but prepared engine is null"
                            << " (gate=" << (bool)gate_v[expert]
                            << " up=" << (bool)up_v[expert]
                            << " down=" << (bool)down_v[expert] << ")";
                        *error = oss.str();
                    }
                    return false;
                }
            }
            return true;
        }

        // Slab-ref path: verify store contains every active expert in each slab.
        if (!has_slab_refs)
        {
            if (error)
                *error = "[MoELocalExpertStage] validatePreparedWeights reached slab path without all slab refs — internal logic error";
            return false;
        }

        auto validate_slab = [&](const char *name, const ExpertSlabRef &ref)
        {
            const auto availability = params_.prepared_store->expertAvailabilityMask(ref);
            if (availability.empty())
            {
                if (error)
                    *error = std::string("[MoELocalExpertStage] PreparedWeightStore missing expert slab for ") + name;
                return false;
            }
            if (availability.size() != static_cast<size_t>(params_.num_experts))
            {
                if (error)
                {
                    std::ostringstream oss;
                    oss << "[MoELocalExpertStage] PreparedWeightStore slab size mismatch for " << name
                        << ": got " << availability.size()
                        << ", expected " << params_.num_experts;
                    *error = oss.str();
                }
                return false;
            }

            for (int expert = 0; expert < params_.num_experts; ++expert)
            {
                const bool active = isExpertActiveForValidation(expert);
                if (!active)
                    continue;
                if (!availability[static_cast<size_t>(expert)])
                {
                    if (error)
                    {
                        std::ostringstream oss;
                        oss << "[MoELocalExpertStage] Layer " << params_.layer_idx
                            << " expert " << expert << " is active but missing in PreparedWeightStore slab "
                            << name;
                        *error = oss.str();
                    }
                    return false;
                }
            }
            return true;
        };

        if (!validate_slab("gate", *params_.gate_slab_ref) ||
            !validate_slab("up", *params_.up_slab_ref) ||
            !validate_slab("down", *params_.down_slab_ref))
        {
            return false;
        }
        return true;
    }

    size_t MoELocalExpertStage::estimatedFlops() const
    {
        const size_t rows = params_.input_rows ? params_.input_rows->live_row_count : 0;
        return rows * static_cast<size_t>(params_.top_k) *
               static_cast<size_t>(6) * static_cast<size_t>(params_.d_model) *
               static_cast<size_t>(params_.expert_intermediate);
    }

    bool MoELocalExpertStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU ||
               backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    StageBufferRequirements MoELocalExpertStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.gate_exps)
            reqs.addWeight("gate_exps", params_.gate_exps->shape(), toBufferTensorType(params_.gate_exps->native_type()));
        if (params_.up_exps)
            reqs.addWeight("up_exps", params_.up_exps->shape(), toBufferTensorType(params_.up_exps->native_type()));
        if (params_.down_exps)
            reqs.addWeight("down_exps", params_.down_exps->shape(), toBufferTensorType(params_.down_exps->native_type()));
        return reqs;
    }

    StageDumpInfo MoELocalExpertStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.gate_exps)
            info.addWeight("gate_exps", params_.gate_exps);
        if (params_.up_exps)
            info.addWeight("up_exps", params_.up_exps);
        if (params_.down_exps)
            info.addWeight("down_exps", params_.down_exps);
        info.addScalarInt("num_experts", params_.num_experts);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("expert_intermediate", params_.expert_intermediate);
        info.addScalarInt("layer_idx", params_.layer_idx);
        info.addScalarInt("live_rows", params_.input_rows ? static_cast<int>(params_.input_rows->live_row_count) : 0);
        return info;
    }

    WorkspaceRequirements MoELocalExpertStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        const int compact_rows_from_sparse_payload =
            params_.input_rows
                ? static_cast<int>(std::max<size_t>(params_.input_rows->entry_capacity, 1u))
                : 0;
        const int compact_rows_from_graph_hint =
            std::max(1, m) * std::max(1, params_.top_k);
        const int max_compact_rows =
            std::max(compact_rows_from_sparse_payload, compact_rows_from_graph_hint);

        WorkspaceRequirements reqs;
        if (params_.device_id.is_cuda())
        {
            /*
             * A local overlay endpoint still uses a full logical-expert
             * descriptor table. Its physical prepared slice can contain only
             * a subset of IDs, but the grouped CUDA kernel indexes table slots
             * by the global router ID and leaves non-owned slots blank. Asking
             * only for expertExecution() omitted these persistent descriptor
             * arenas, leaving a one-descriptor fallback allocation that failed
             * as soon as a 256-expert production graph published its table.
             */
            reqs.merge(MoEWorkspaceBuffers::cudaMoE(
                max_compact_rows,
                params_.d_model,
                params_.expert_intermediate,
                params_.num_experts,
                /*top_k=*/1));
        }
        else if (params_.device_id.is_rocm())
        {
            reqs.merge(MoEWorkspaceBuffers::rocmMoE(
                max_compact_rows,
                params_.d_model,
                params_.expert_intermediate,
                params_.num_experts,
                /*top_k=*/1));
        }

        /**
         * The local expert stage compacts sparse routes into one row per active
         * route before delegating to MoEExpertComputeStage.  The nested stage is
         * constructed at execute() time, so the graph allocator cannot discover
         * its GEMM scratch unless we advertise it here with the projected shapes.
         */
        auto mergeGemmRequirements = [&](const std::vector<ITensorGemm *> &engines,
                                         int out_features,
                                         int in_features)
        {
            for (auto *gemm : engines)
            {
                if (!gemm)
                    continue;
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer)
                    reqs.merge(consumer->getWorkspaceRequirements(
                        max_compact_rows,
                        out_features,
                        in_features));
            }
        };

        mergeGemmRequirements(params_.prepared_gate_gemm,
                              params_.expert_intermediate,
                              params_.d_model);
        mergeGemmRequirements(params_.prepared_up_gemm,
                              params_.expert_intermediate,
                              params_.d_model);
        mergeGemmRequirements(params_.prepared_down_gemm,
                              params_.d_model,
                              params_.expert_intermediate);

        (void)n;
        (void)k;
        return reqs;
    }

    void MoELocalExpertStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        auto bindAll = [workspace](const std::vector<ITensorGemm *> &engines)
        {
            for (auto *gemm : engines)
            {
                if (!gemm)
                    continue;
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer)
                    consumer->bindWorkspace(workspace);
            }
        };

        bindAll(params_.prepared_gate_gemm);
        bindAll(params_.prepared_up_gemm);
        bindAll(params_.prepared_down_gemm);
        for (auto &replay : deferred_replay_families_)
        {
            if (replay && replay->compute_stage)
                replay->compute_stage->bindWorkspace(workspace);
        }
        bound_workspace_ = workspace;
    }

    void MoELocalExpertStage::unbindWorkspace()
    {
        auto unbindAll = [](const std::vector<ITensorGemm *> &engines)
        {
            for (auto *gemm : engines)
            {
                if (!gemm)
                    continue;
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer)
                    consumer->unbindWorkspace();
            }
        };

        unbindAll(params_.prepared_gate_gemm);
        unbindAll(params_.prepared_up_gemm);
        unbindAll(params_.prepared_down_gemm);
        for (auto &replay : deferred_replay_families_)
        {
            if (replay && replay->compute_stage)
                replay->compute_stage->unbindWorkspace();
        }
        bound_workspace_ = nullptr;
    }

    MoELocalExpertCompletionStage::MoELocalExpertCompletionStage(
        Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (!params_.producer || !params_.device_id.is_gpu() ||
            params_.producer->params().device_id != params_.device_id ||
            !params_.producer->usesDeferredCompletion())
        {
            throw std::invalid_argument(
                "MoELocalExpertCompletionStage requires one matching deferred GPU producer");
        }
    }

    bool MoELocalExpertCompletionStage::execute(IDeviceContext *ctx)
    {
        if (!params_.producer ||
            !sharesExactGPUStreamWith(*params_.producer))
        {
            LOG_ERROR(
                "[MoELocalExpertCompletionStage] Completion and producer must share one exact non-null GPU stream");
            return false;
        }
        return params_.producer->completeDeferredOutput(ctx);
    }

    bool MoELocalExpertCompletionStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    StageDumpInfo MoELocalExpertCompletionStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.producer)
        {
            info.addScalarInt(
                "layer_idx", params_.producer->layerIndex());
            info.addScalarInt(
                "participant_idx",
                params_.producer->participantIndex());
            info.addScalarBool(
                "pending",
                params_.producer->hasPendingDeferredOutput());
        }
        return info;
    }

} // namespace llaminar2
