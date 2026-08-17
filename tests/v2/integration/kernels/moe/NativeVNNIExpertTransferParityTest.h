/**
 * @file NativeVNNIExpertTransferParityTest.h
 * @brief Shared CUDA/ROCm regression for transfer-backed grouped MoE parity.
 *
 * Current-batch LLEP changes both the physical expert descriptor and the
 * participant assigned to each grouped route. A grouped-kernel-only sweep can
 * therefore remain green while compact payload movement, transfer-slot format
 * retargeting, runtime-bank publication, or assignment-span consumption changes
 * the bytes produced by the real destination transaction. This helper drives
 * that complete production ABI with real GPU-prepared NativeVNNI weights.
 *
     * For every canonical quantized model format, every supported MTP depth, and
     * one prefill size beyond the small-group routing capacity, the test compares
     * two executions of the same routed transaction:
 *
 * 1. StaticOwner executes four source experts on participant zero.
 * 2. CurrentBatchLLEP keeps two experts resident on participant one, transfers
 *    two more into distinct stable slots through the production compact-payload
 *    ABI, consumes four assignment spans, and executes the same top-k=4 route
 *    transaction entirely on participant one.
 *
 * Equality is byte equality. The helper contains no row-replay arithmetic and
 * does not replace either backend's production grouped implementation.
 */

#pragma once

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/moe/DeviceMoERebalanceABI.h"
#include "execution/moe/DeviceMoETransferSlotDirectory.h"
#include "execution/moe/ExpertTierWeightStream.h"
#include "execution/moe/ExpertTierWeightTransferLane.h"
#include "execution/moe/LeastLoadedExpertAssignment.h"
#include "execution/moe/MoERuntimeTable.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"

#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/TestTensorFactory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace native_vnni_transfer_parity_detail
    {
        /**
         * @brief RAII owner for setup-only device allocations used by the test.
         *
         * Production owns these buffers through graph arenas. A focused kernel
         * integration test has no graph builder, so this owner models the same
         * stable-address lifetime while guaranteeing cleanup after assertions or
         * exceptions. Allocation occurs only during test setup, never inside a
         * captured or measured inference path.
         */
        class DeviceAllocation final
        {
        public:
            DeviceAllocation(IBackend *backend, int ordinal, size_t bytes)
                : backend_(backend), ordinal_(ordinal), bytes_(bytes)
            {
                if (!backend_ || bytes_ == 0u)
                    throw std::invalid_argument(
                        "NativeVNNI transfer parity allocation requires a backend and bytes");
                pointer_ = backend_->allocate(bytes_, ordinal_);
                if (!pointer_)
                    throw std::runtime_error(
                        "NativeVNNI transfer parity device allocation failed");
            }

            ~DeviceAllocation()
            {
                if (pointer_)
                    backend_->free(pointer_, ordinal_);
            }

            DeviceAllocation(const DeviceAllocation &) = delete;
            DeviceAllocation &operator=(const DeviceAllocation &) = delete;
            DeviceAllocation(DeviceAllocation &&) = delete;
            DeviceAllocation &operator=(DeviceAllocation &&) = delete;

            /** @brief Return the untyped stable device address. */
            [[nodiscard]] void *get() const noexcept { return pointer_; }

            /** @brief Return the stable address interpreted as @p T. */
            template <typename T>
            [[nodiscard]] T *as() const noexcept
            {
                return static_cast<T *>(pointer_);
            }

            /** @brief Return the allocation capacity in bytes. */
            [[nodiscard]] size_t bytes() const noexcept { return bytes_; }

        private:
            IBackend *backend_ = nullptr;
            int ordinal_ = -1;
            size_t bytes_ = 0u;
            void *pointer_ = nullptr;
        };

        /**
         * @brief Describe gate/up/down transfer capacity for one device codebook.
         */
        inline std::vector<DeviceMoETransferSlotDirectory::ProjectionSpec>
        projectionSpecsForFormat(
            const QuantizedVerifierFormatCase &format,
            int d_model,
            int intermediate)
        {
            uint8_t payload_bytes = 0u;
            uint8_t is_asymmetric = 0u;
            uint8_t has_emins = 0u;
            if (!deviceMoENativeVnniFormatForCodebook(
                    format.device_execution_codebook_id,
                    payload_bytes,
                    is_asymmetric,
                    has_emins))
            {
                throw std::runtime_error(
                    std::string("No transfer ABI for canonical format ") +
                    format.label);
            }

            const auto make = [&](const char *label, int n, int k)
            {
                return DeviceMoETransferSlotDirectory::ProjectionSpec{
                    .label = label,
                    .N = n,
                    .K = k,
                    .payload_bytes_per_block = payload_bytes,
                    .is_asymmetric = is_asymmetric != 0u,
                    .has_emins = has_emins != 0u,
                    .codebook_id = format.device_execution_codebook_id,
                    .format = ExpertWeightFormat::nativeVnni(
                        NativeVnniSourceIdentity{
                            .codebook_id = format.source_codebook_id,
                            .is_superblock = format.source_is_superblock,
                            .present = true,
                        }),
                };
            };
            return {
                make("gate", intermediate, d_model),
                make("up", intermediate, d_model),
                make("down", d_model, intermediate),
            };
        }

        /**
         * @brief Build one participant-local runtime table for an expert domain.
         *
         * Participant zero is the immutable owner of every expert. Participant
         * one may begin with a subset already resident; the transfer/apply
         * transaction must replace each missing descriptor with a pointer into a
         * distinct destination-owned transfer slot before grouped execution.
         */
        inline std::unique_ptr<DeviceMoERuntimeTable> makeRuntimeTable(
            DeviceId device,
            void *stream,
            const std::vector<DeviceMoEExpertDescriptor> &descriptors,
            uint32_t participant_id,
            const std::vector<uint8_t> &local_compute_mask,
            const std::vector<uint32_t> &resident_participant_mask,
            int num_layers,
            int top_k,
            int token_capacity)
        {
            if (descriptors.empty() ||
                descriptors.size() != local_compute_mask.size() ||
                descriptors.size() != resident_participant_mask.size() ||
                num_layers <= 0 ||
                top_k <= 0)
            {
                throw std::invalid_argument(
                    "NativeVNNI transfer parity runtime domain is inconsistent");
            }

            DeviceMoERuntimeTable::Config config{
                .device_id = device,
                .num_layers = num_layers,
                .num_experts = static_cast<int>(descriptors.size()),
                .top_k = top_k,
                .mirror_to_device = true,
                .prefill_token_capacity = token_capacity,
            };
            auto table = std::make_unique<DeviceMoERuntimeTable>(config);

            MoEPlacementUpdate update;
            update.epoch = 1u;
            update.expert_count = static_cast<uint32_t>(descriptors.size());
            update.participant_id = participant_id;
            update.participant_count = 2u;
            update.experts = descriptors;
            update.local_compute_mask = local_compute_mask;
            update.replica_role.resize(
                descriptors.size(),
                static_cast<uint8_t>(DeviceMoEReplicaRole::None));
            update.resident_participant_mask = resident_participant_mask;

            for (size_t expert = 0; expert < descriptors.size(); ++expert)
            {
                auto &published = update.experts[expert];
                const bool local_compute = local_compute_mask[expert] != 0u;
                published.local_slot =
                    local_compute ? static_cast<int32_t>(expert) : -1;
                published.flags = toMoEExpertFlags(
                    DeviceMoEExpertFlags::Valid |
                    DeviceMoEExpertFlags::Resident);
                if (local_compute)
                {
                    published.flags |=
                        toMoEExpertFlags(DeviceMoEExpertFlags::LocalCompute);
                    update.replica_role[expert] = static_cast<uint8_t>(
                        published.owner_participant ==
                                static_cast<int32_t>(participant_id)
                            ? DeviceMoEReplicaRole::Primary
                            : DeviceMoEReplicaRole::Replica);
                }
                if ((resident_participant_mask[expert] &
                     (resident_participant_mask[expert] - 1u)) != 0u)
                {
                    published.flags |=
                        toMoEExpertFlags(DeviceMoEExpertFlags::Replicated);
                }
            }

            for (int layer = 0; layer < num_layers; ++layer)
            {
                if (!table->prepareInactiveBank(layer, update) ||
                    !table->flipActiveBank(layer, update.epoch, stream))
                {
                    throw std::runtime_error(
                        "Failed to publish NativeVNNI transfer parity runtime table");
                }
            }
            return table;
        }

        /**
         * @brief Prepare one real source matrix through the production GPU loader.
         */
        inline DeviceNativeVNNIMatrixDesc prepareMatrixDescriptor(
            const QuantizedVerifierFormatCase &format,
            DeviceId device,
            void *stream,
            int rows,
            int columns,
            uint32_t seed,
            const std::string &name,
            ModelContextId model_id,
            std::vector<std::unique_ptr<TensorBase>> &weights,
            std::vector<GpuPreparedGemm> &prepared)
        {
            auto weight = format.create(
                {static_cast<size_t>(rows), static_cast<size_t>(columns)},
                seed);
            TensorBase *weight_ptr = weight.get();
            weights.push_back(std::move(weight));
            prepared.push_back(makeGpuPreparedGemm(
                weight_ptr,
                device,
                name,
                model_id));

            ITensorGemm *gemm = prepared.back().kernel;
            if (!gemm)
                throw std::runtime_error("Prepared NativeVNNI GEMM is null");
            if (auto *tensor_kernel = dynamic_cast<ITensorKernel *>(gemm))
                tensor_kernel->setGPUStream(stream);

            DeviceNativeVNNIMatrixDesc descriptor{};
            if (!gemm->exportNativeVNNIMatrixDesc(descriptor) ||
                descriptor.codebook_id != format.device_execution_codebook_id ||
                descriptor.n != rows || descriptor.k != columns)
            {
                throw std::runtime_error(
                    std::string("Prepared descriptor mismatch for ") +
                    format.label + " " + name);
            }
            return descriptor;
        }

        /**
         * @brief Create one deterministic, non-degenerate hidden-state matrix.
         */
        inline std::unique_ptr<TensorBase> makeHidden(
            int rows,
            int d_model,
            size_t format_index)
        {
            auto hidden = TestTensorFactory::createFP32(
                {static_cast<size_t>(rows), static_cast<size_t>(d_model)});
            float *values = hidden->mutable_data();
            for (size_t i = 0; i < hidden->numel(); ++i)
            {
                values[i] =
                    0.017f * std::sin(
                                 0.0041f * static_cast<float>(i + 11u + format_index)) -
                    0.009f * std::cos(
                                 0.0067f * static_cast<float>(i + 23u)) +
                    0.0007f * static_cast<float>(
                                  static_cast<int>(i % 31u) - 15);
            }
            return hidden;
        }

        /**
         * @brief Execute one already-published grouped runtime plan and observe it.
         *
         * The grouped expert producer writes canonical route contributions. The
         * fixed-order reducer owns the visible output. Keeping those two steps
         * separate mirrors the production LocalTP publication contract exactly.
         */
        inline std::vector<float> executePublishedPlan(
            IBackend *backend,
            IMoEKernel &kernel,
            DeviceMoERuntimeTable &runtime,
            DeviceId device,
            void *stream,
            ITensor *hidden,
            int gateup_table,
            int down_table,
            int rows,
            int d_model,
            int intermediate,
            int num_experts,
            int top_k,
            int layer_idx,
            std::vector<float> *canonical_output = nullptr)
        {
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(rows), static_cast<size_t>(d_model)});
            auto canonical = TestTensorFactory::createFP32(
                {static_cast<size_t>(rows),
                 static_cast<size_t>(top_k),
                 static_cast<size_t>(d_model)});
            if (!canonical->ensureOnDevice(device, stream))
                throw std::runtime_error("Failed to allocate canonical MoE output");

            if (!kernel.executeGroupedPrefillPipelineFromPublishedRuntimePlan(
                    runtime.deviceLayerState(layer_idx),
                    runtime.hostLayerState(layer_idx),
                    hidden,
                    output.get(),
                    gateup_table,
                    down_table,
                    rows,
                    d_model,
                    intermediate,
                    num_experts,
                    top_k,
                    canonical.get()))
            {
                throw std::runtime_error("Grouped runtime expert execution failed");
            }
            if (output->gpu_data_ptr() != nullptr)
            {
                throw std::runtime_error(
                    "Grouped producer wrote reducer-owned output storage");
            }
            if (!output->ensureOnDevice(device, stream) ||
                !kernel.reduceCanonicalRouteContributions(
                    canonical.get(), output.get(), rows, top_k, d_model))
            {
                throw std::runtime_error("Canonical MoE reduction failed");
            }
            if (!output->ensureOnHost(stream) ||
                (canonical_output && !canonical->ensureOnHost(stream)) ||
                !backend->synchronizeStream(stream, device.ordinal))
            {
                throw std::runtime_error("Failed to observe grouped MoE output");
            }
            if (canonical_output)
            {
                canonical_output->assign(
                    canonical->data(),
                    canonical->data() + canonical->numel());
            }
            return std::vector<float>(
                output->data(), output->data() + output->numel());
        }

        /**
         * @brief Publish the two device-owned current-batch plan counts.
         *
         * The runtime table allocates and owns the span/transfer arrays. Tests
         * may replace their contents between launches, but must update only the
         * two count words rather than uploading the stale host placement bank
         * over a transfer-applied device bank.
         */
        inline bool publishCurrentBatchPlanCounts(
            IBackend *backend,
            DeviceId device,
            void *stream,
            DeviceMoELayerRuntime *runtime,
            uint64_t span_count,
            uint64_t transfer_count)
        {
            if (!backend || !stream || !runtime)
                return false;
            const std::array<uint64_t, 2> counts = {
                span_count,
                transfer_count,
            };
            auto *count_words =
                reinterpret_cast<uint8_t *>(runtime) +
                offsetof(DeviceMoELayerRuntime, reserved_u64) +
                2u * sizeof(uint64_t);
            return backend->hostToDeviceOnStream(
                count_words,
                counts.data(),
                sizeof(counts),
                device.ordinal,
                stream);
        }

        /**
         * @brief Report the first unequal IEEE-754 word in two grouped outputs.
         */
        inline void expectByteEqual(
            const std::string &label,
            const std::vector<float> &actual,
            const std::vector<float> &expected,
            int d_model)
        {
            ASSERT_EQ(actual.size(), expected.size()) << label;
            if (std::memcmp(
                    actual.data(),
                    expected.data(),
                    actual.size() * sizeof(float)) == 0)
            {
                return;
            }

            size_t mismatch = 0u;
            while (mismatch < actual.size())
            {
                uint32_t actual_bits = 0u;
                uint32_t expected_bits = 0u;
                std::memcpy(&actual_bits, &actual[mismatch], sizeof(actual_bits));
                std::memcpy(&expected_bits, &expected[mismatch], sizeof(expected_bits));
                if (actual_bits != expected_bits)
                {
                    ADD_FAILURE()
                        << label
                        << " first_mismatch=" << mismatch
                        << " row=" << mismatch / static_cast<size_t>(d_model)
                        << " column=" << mismatch % static_cast<size_t>(d_model)
                        << " actual=" << actual[mismatch]
                        << " expected=" << expected[mismatch]
                        << " actual_bits=0x" << std::hex << actual_bits
                        << " expected_bits=0x" << expected_bits << std::dec;
                    return;
                }
                ++mismatch;
            }
        }
    } // namespace native_vnni_transfer_parity_detail

    /**
     * @brief Prove StaticOwner and transferred CurrentBatchLLEP grouped equality.
     *
     * @param backend_label Stable diagnostic label ("CUDA" or "ROCm").
     * @param device Physical GPU used to execute both simulated participants.
     * @param stream Explicit non-null backend stream owning every test operation.
     *
     * @throws std::runtime_error for setup or launch failures. GoogleTest catches
     *         the exception at the backend test boundary and reports the active
     *         format/M trace.
     */
    inline void runNativeVNNIExpertTransferGroupedParity(
        const char *backend_label,
        DeviceId device,
        void *stream)
    {
        using namespace native_vnni_transfer_parity_detail;

        if (!backend_label || !stream || !device.is_gpu())
            throw std::invalid_argument(
                "NativeVNNI transfer parity requires a GPU and explicit stream");
        IBackend *backend = getBackendFor(device);
        if (!backend)
            throw std::runtime_error("NativeVNNI transfer parity backend is unavailable");

        constexpr int kDModel = 2048;
        constexpr int kIntermediate = 512;
        constexpr int kMaxRows = 2560;
        constexpr int kNumExperts = 256;
        constexpr int kTopK = 8;
        constexpr uint32_t kPlanCapacity = 32u;
        /*
         * Collective payload lanes and physical transfer-directory slots are
         * different address spaces.  Keep the wire transaction compact while
         * forcing both arrivals beyond its width: production long-context LLEP
         * first diverged when layer 1 leased directory slots 32 through 35
         * after layer 0 had retained the lower slots.
         */
        constexpr uint32_t kPhysicalTransferSlotBase = kPlanCapacity;
        constexpr uint32_t kTransferSlotCount =
            kPhysicalTransferSlotBase + kPlanCapacity;
        constexpr uint32_t kCommandBufferCount = 1u;
        constexpr uint32_t kParticipantCount = 2u;
        constexpr uint32_t kDestinationParticipant = 1u;
        constexpr int kRuntimeLayerCount = 2;
        constexpr int kTargetLayer = 1;
        std::array<uint32_t, kPlanCapacity> transferred_experts{};
        for (uint32_t index = 0u; index < kPlanCapacity; ++index)
            transferred_experts[index] = index * 2u + 1u;
        /*
         * Cover the complete MTP regime plus both scalable-grouping boundaries
         * and the production long-context capture bucket that exposed the E2E
         * failure. The largest case is intentionally retained for every format:
         * small-M exactness cannot certify 32-bit count/offset arithmetic.
         */
        constexpr std::array<int, 8> kRows = {
            2, 4, 8, 15, 16, 65, 257, 2560};

        const auto &formats = quantizedMoEVerifierFormats();
        std::vector<std::vector<DeviceMoETransferSlotDirectory::ProjectionSpec>>
            format_specs;
        format_specs.reserve(formats.size());
        for (const auto &format : formats)
        {
            format_specs.push_back(
                projectionSpecsForFormat(format, kDModel, kIntermediate));
        }

        auto transfer_directory = DeviceMoETransferSlotDirectory::create(
            backend,
            device,
            device.ordinal,
            kDestinationParticipant,
            /*slot_count=*/kTransferSlotCount,
            DeviceMoETransferSlotDirectory::profileForLayerFormats(format_specs),
            /*vram_safety_margin_bytes=*/0u);
        ASSERT_NE(transfer_directory, nullptr);
        ASSERT_EQ(transfer_directory->slotCount(), kTransferSlotCount);

        auto kernel_owner =
            llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
        ASSERT_NE(kernel_owner, nullptr);
        kernel_owner->setGPUStream(stream);

        auto requirements = device.is_cuda()
                                ? MoEWorkspaceBuffers::cudaMoE(
                                      kMaxRows,
                                      kDModel,
                                      kIntermediate,
                                      kNumExperts,
                                      kTopK)
                                : MoEWorkspaceBuffers::rocmMoE(
                                      kMaxRows,
                                      kDModel,
                                      kIntermediate,
                                      kNumExperts,
                                      kTopK);
        DeviceWorkspaceManager workspace(
            device,
            requirements.total_bytes_with_alignment() + 4u * 1024u * 1024u);
        ASSERT_TRUE(workspace.allocate(requirements));
        auto *workspace_consumer =
            dynamic_cast<IWorkspaceConsumer *>(kernel_owner.get());
        ASSERT_NE(workspace_consumer, nullptr);
        workspace_consumer->bindWorkspace(&workspace);

        IMoEKernel &kernel = *kernel_owner;
        const MoEKernelLaunchContext launch{
            .stream = stream,
            .workspace = &workspace,
        };

        const uint64_t payload_slot_bytes =
            (sizeof(DeviceMoEExpertDirectoryEntry) +
             transfer_directory->wirePayloadBytes() + 255u) /
            256u * 256u;
        const size_t local_payload_bytes =
            kPlanCapacity * static_cast<size_t>(payload_slot_bytes);
        const size_t gathered_payload_bytes =
            kParticipantCount * local_payload_bytes;
        const size_t source_directory_entries =
            kParticipantCount * kCommandBufferCount * kPlanCapacity;

        DeviceAllocation d_plan(
            backend,
            device.ordinal,
            kPlanCapacity * sizeof(DeviceMoERebalancePlanEntry));
        DeviceAllocation d_plan_count(
            backend,
            device.ordinal,
            sizeof(uint32_t));
        DeviceAllocation d_command_header(
            backend,
            device.ordinal,
            sizeof(DeviceMoERebalanceCommandBufferHeader));
        DeviceAllocation d_source_descriptors(
            backend,
            device.ordinal,
            source_directory_entries * sizeof(DeviceMoEExpertDirectoryEntry));
        DeviceAllocation d_local_payload(
            backend,
            device.ordinal,
            local_payload_bytes);
        DeviceAllocation d_gathered_payload(
            backend,
            device.ordinal,
            gathered_payload_bytes);
        DeviceAllocation d_copy_status(
            backend,
            device.ordinal,
            sizeof(DeviceMoERebalanceApplyStatus));
        DeviceAllocation d_apply_status(
            backend,
            device.ordinal,
            sizeof(DeviceMoERebalanceApplyStatus));
        DeviceAllocation d_source_apply_status(
            backend,
            device.ordinal,
            sizeof(DeviceMoERebalanceApplyStatus));
        DeviceAllocation d_transfer_status(
            backend,
            device.ordinal,
            sizeof(DeviceMoERebalanceStatus));

        for (size_t format_index = 0;
             format_index < formats.size();
             ++format_index)
        {
            const auto &format = formats[format_index];
            SCOPED_TRACE(
                std::string(backend_label) + " format=" + format.label);

            std::vector<std::unique_ptr<TensorBase>> weights;
            std::vector<GpuPreparedGemm> prepared;
            weights.reserve(3u);
            prepared.reserve(3u);

            const uint64_t model_base =
                1900000u + static_cast<uint64_t>(format_index) * 64u +
                (device.is_cuda() ? 0u : 100000u);
            const std::string name_prefix =
                std::string("test.") + backend_label +
                ".native_vnni_transfer_parity." + format.label;

            std::vector<DeviceMoEExpertDescriptor> source_descriptors(kNumExperts);
            std::array<DeviceNativeVNNIMatrixDesc, kNumExperts> gate_descs{};
            std::array<DeviceNativeVNNIMatrixDesc, kNumExperts> up_descs{};
            std::array<DeviceNativeVNNIMatrixDesc, kNumExperts> down_descs{};

            /*
             * Cardinality, not unique payload bytes, is the subject of this
             * regression.  Prepare one real production-ABI matrix triplet per
             * format and alias it across 256 logical experts.  The grouped
             * kernels still execute every routed row and every expert group,
             * while setup remains light enough to sweep all codebooks and M.
             */
            const DeviceNativeVNNIMatrixDesc shared_gate =
                prepareMatrixDescriptor(
                    format,
                    device,
                    stream,
                    kIntermediate,
                    kDModel,
                    810001u,
                    name_prefix + ".shared.gate",
                    ModelContextId{model_base + 1u},
                    weights,
                    prepared);
            const DeviceNativeVNNIMatrixDesc shared_up =
                prepareMatrixDescriptor(
                    format,
                    device,
                    stream,
                    kIntermediate,
                    kDModel,
                    810002u,
                    name_prefix + ".shared.up",
                    ModelContextId{model_base + 2u},
                    weights,
                    prepared);
            const DeviceNativeVNNIMatrixDesc shared_down =
                prepareMatrixDescriptor(
                    format,
                    device,
                    stream,
                    kDModel,
                    kIntermediate,
                    810003u,
                    name_prefix + ".shared.down",
                    ModelContextId{model_base + 3u},
                    weights,
                    prepared);
            for (int expert = 0; expert < kNumExperts; ++expert)
            {
                auto &descriptor = source_descriptors[expert];
                descriptor.logical_expert_id = expert;
                descriptor.owner_participant = 0;
                descriptor.local_slot = expert;
                descriptor.flags = toMoEExpertFlags(
                    DeviceMoEExpertFlags::Valid |
                    DeviceMoEExpertFlags::Resident |
                    DeviceMoEExpertFlags::LocalCompute);

                descriptor.gate = shared_gate;
                descriptor.up = shared_up;
                descriptor.down = shared_down;
                gate_descs[expert] = descriptor.gate;
                up_descs[expert] = descriptor.up;
                down_descs[expert] = descriptor.down;
            }

            const int source_gateup_table =
                kernel.uploadGroupedExpertGateUpDescriptorTables(
                    gate_descs.data(),
                    up_descs.data(),
                    kNumExperts,
                    kDModel,
                    kIntermediate);
            const int source_down_table =
                kernel.uploadGroupedExpertDownDescriptorTable(
                    down_descs.data(),
                    kNumExperts,
                    kDModel,
                    kIntermediate);
            const int destination_gateup_table =
                kernel.uploadGroupedExpertGateUpDescriptorTables(
                    gate_descs.data(),
                    up_descs.data(),
                    kNumExperts,
                    kDModel,
                    kIntermediate);
            const int destination_down_table =
                kernel.uploadGroupedExpertDownDescriptorTable(
                    down_descs.data(),
                    kNumExperts,
                    kDModel,
                    kIntermediate);
            ASSERT_GE(source_gateup_table, 0);
            ASSERT_GE(source_down_table, 0);
            ASSERT_GE(destination_gateup_table, 0);
            ASSERT_GE(destination_down_table, 0);

            auto source_runtime = makeRuntimeTable(
                device,
                stream,
                source_descriptors,
                /*participant_id=*/0u,
                std::vector<uint8_t>(kNumExperts, 1u),
                std::vector<uint32_t>(kNumExperts, 0b01u),
                kRuntimeLayerCount,
                kTopK,
                kMaxRows);
            std::vector<uint8_t> destination_compute_mask(kNumExperts, 1u);
            std::vector<uint32_t> destination_resident_mask(kNumExperts, 0b11u);
            for (const uint32_t expert : transferred_experts)
            {
                destination_compute_mask[expert] = 0u;
                destination_resident_mask[expert] = 0b01u;
            }
            auto destination_runtime = makeRuntimeTable(
                device,
                stream,
                source_descriptors,
                kDestinationParticipant,
                destination_compute_mask,
                destination_resident_mask,
                kRuntimeLayerCount,
                kTopK,
                kMaxRows);

            /*
             * The host runtime object owns only immutable scratch addresses in
             * this setup. Publish the two current-batch counts once before GPU
             * work begins; all route values, placement mutation, and grouped
             * execution remain device-owned after this boundary.
             */
            auto &source_host = source_runtime->hostLayerState(kTargetLayer);
            auto &destination_host =
                destination_runtime->hostLayerState(kTargetLayer);
            destination_host.reserved_u64[2] = kNumExperts;
            destination_host.reserved_u64[3] = kPlanCapacity;
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                destination_runtime->deviceLayerState(kTargetLayer),
                &destination_host,
                sizeof(destination_host),
                device.ordinal,
                stream));

            DeviceMoERebalanceConfig base_config;
            base_config.num_layers = kRuntimeLayerCount;
            base_config.num_experts = kNumExperts;
            base_config.top_k = kTopK;
            base_config.participant_count = kParticipantCount;
            base_config.window_size_tokens = 1u;
            base_config.max_hot_replicas_per_participant = kPlanCapacity;
            base_config.active_transfer_slot_capacity = kTransferSlotCount;
            base_config.transfer_slot_directory_capacity = kTransferSlotCount;
            base_config.llep_enable_balanced_skip = 0u;

            std::array<DeviceMoERebalancePlanEntry, kPlanCapacity> plans{};
            for (uint32_t plan_index = 0u;
                 plan_index < kPlanCapacity;
                 ++plan_index)
            {
                auto &plan = plans[plan_index];
                plan.op = static_cast<uint32_t>(
                    DeviceMoERebalancePlanOp::ExpertPayloadArrival);
                plan.layer = kTargetLayer;
                plan.expert = transferred_experts[plan_index];
                plan.source_participant = 0u;
                plan.destination_participant = kDestinationParticipant;
                plan.source_resident_mask = 0b01u;
                plan.flags = moe_rebalance_abi::kPlanFlagCurrentBatchLLEP;
                plan.destination_slot =
                    kPhysicalTransferSlotBase + plan_index;
                plan.payload_slot = plan_index;
            }
            auto retained_layer_plans = plans;
            for (uint32_t plan_index = 0u;
                 plan_index < kPlanCapacity;
                 ++plan_index)
            {
                retained_layer_plans[plan_index].layer = 0u;
                retained_layer_plans[plan_index].destination_slot = plan_index;
            }

            DeviceMoERebalanceCommandBufferHeader source_header;
            source_header.epoch = static_cast<uint32_t>(format_index + 3u);
            source_header.phase = static_cast<uint32_t>(
                DeviceMoERebalancePipelinePhase::PlanAssignments);
            source_header.command_count = kPlanCapacity;
            source_header.command_capacity = kPlanCapacity;
            source_header.participant_id = 0u;
            source_header.participant_count = kParticipantCount;
            auto destination_header = source_header;
            destination_header.participant_id = kDestinationParticipant;
            auto retained_source_header = source_header;
            retained_source_header.epoch =
                static_cast<uint32_t>(format_index + 2u);
            auto retained_destination_header = retained_source_header;
            retained_destination_header.participant_id =
                kDestinationParticipant;

            const uint32_t plan_count = kPlanCapacity;
            auto source_config = base_config;
            source_config.participant_id = 0u;
            auto destination_config = base_config;
            destination_config.participant_id = kDestinationParticipant;
            const auto apply_config = prefillLLEPTransferConfig(
                destination_config,
                PrefillLLEPTransferPurpose::CurrentBatchMovement);

            ASSERT_TRUE(backend->memset(
                d_source_descriptors.get(),
                0,
                d_source_descriptors.bytes(),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->memset(
                d_local_payload.get(),
                0,
                d_local_payload.bytes(),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->memset(
                d_gathered_payload.get(),
                0,
                d_gathered_payload.bytes(),
                device.ordinal,
                stream));
            transfer_directory->resetRequestPublications(stream);

            /*
             * First retain one complete layer-0 wave in physical slots 0..31.
             * Layer 1 below must materialize into slots 32..63 without changing
             * any descriptor still published by this earlier layer.
             */
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_plan.get(),
                retained_layer_plans.data(),
                sizeof(retained_layer_plans),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_plan_count.get(),
                &plan_count,
                sizeof(plan_count),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_command_header.get(),
                &retained_source_header,
                sizeof(retained_source_header),
                device.ordinal,
                stream));
            ASSERT_TRUE(kernel.packDeviceRebalanceSourceDescriptors(
                launch,
                source_runtime->deviceLayerState(0),
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                kPlanCapacity,
                d_source_descriptors.as<DeviceMoEExpertDirectoryEntry>(),
                source_config,
                /*controller_state=*/nullptr,
                kCommandBufferCount));
            ASSERT_TRUE(kernel.packDeviceRebalanceCompactPayloads(
                launch,
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                kPlanCapacity,
                d_source_descriptors.as<DeviceMoEExpertDirectoryEntry>(),
                d_local_payload.as<uint8_t>(),
                /*local_payload_slot_count=*/kPlanCapacity,
                payload_slot_bytes,
                source_config,
                d_copy_status.as<DeviceMoERebalanceApplyStatus>(),
                /*controller_state=*/nullptr,
                kCommandBufferCount));
            ASSERT_TRUE(backend->deviceCopyAsync(
                d_gathered_payload.get(),
                d_local_payload.get(),
                local_payload_bytes,
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_command_header.get(),
                &retained_destination_header,
                sizeof(retained_destination_header),
                device.ordinal,
                stream));
            ASSERT_TRUE(kernel.unpackDeviceRebalanceCollectivePayloads(
                launch,
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_plan_count.as<uint32_t>(),
                kPlanCapacity,
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                d_gathered_payload.as<uint8_t>(),
                /*local_payload_slot_count=*/kPlanCapacity,
                payload_slot_bytes,
                transfer_directory->deviceEntries(),
                transfer_directory->slotCount(),
                destination_config,
                d_copy_status.as<DeviceMoERebalanceApplyStatus>(),
                /*controller_state=*/nullptr,
                kCommandBufferCount));
            ASSERT_TRUE(kernel.applyDeviceRebalanceArrivals(
                launch,
                destination_runtime->deviceLayerState(0),
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_plan_count.as<uint32_t>(),
                kPlanCapacity,
                transfer_directory->deviceEntries(),
                transfer_directory->slotCount(),
                apply_config,
                d_apply_status.as<DeviceMoERebalanceApplyStatus>(),
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                /*target_layer=*/0));

            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_plan.get(),
                plans.data(),
                sizeof(plans),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_plan_count.get(),
                &plan_count,
                sizeof(plan_count),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_command_header.get(),
                &source_header,
                sizeof(source_header),
                device.ordinal,
                stream));
            ASSERT_TRUE(kernel.packDeviceRebalanceSourceDescriptors(
                launch,
                source_runtime->deviceLayerState(0),
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                kPlanCapacity,
                d_source_descriptors.as<DeviceMoEExpertDirectoryEntry>(),
                source_config,
                /*controller_state=*/nullptr,
                kCommandBufferCount));
            ASSERT_TRUE(kernel.packDeviceRebalanceCompactPayloads(
                launch,
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                kPlanCapacity,
                d_source_descriptors.as<DeviceMoEExpertDirectoryEntry>(),
                d_local_payload.as<uint8_t>(),
                /*local_payload_slot_count=*/kPlanCapacity,
                payload_slot_bytes,
                source_config,
                d_copy_status.as<DeviceMoERebalanceApplyStatus>(),
                /*controller_state=*/nullptr,
                kCommandBufferCount));

            /*
             * A real two-rank graph uses one NCCL/RCCL allgather here. This
             * focused one-device regression preserves its byte layout and
             * ordering while replacing only the transport primitive with D2D.
             * Source participant zero occupies gathered lane zero.
             */
            ASSERT_TRUE(backend->deviceCopyAsync(
                d_gathered_payload.get(),
                d_local_payload.get(),
                local_payload_bytes,
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_command_header.get(),
                &destination_header,
                sizeof(destination_header),
                device.ordinal,
                stream));

            ASSERT_TRUE(kernel.unpackDeviceRebalanceCollectivePayloads(
                launch,
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_plan_count.as<uint32_t>(),
                kPlanCapacity,
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                d_gathered_payload.as<uint8_t>(),
                /*local_payload_slot_count=*/kPlanCapacity,
                payload_slot_bytes,
                transfer_directory->deviceEntries(),
                transfer_directory->slotCount(),
                destination_config,
                d_copy_status.as<DeviceMoERebalanceApplyStatus>(),
                /*controller_state=*/nullptr,
                kCommandBufferCount));

            ASSERT_TRUE(kernel.applyDeviceRebalanceArrivals(
                launch,
                destination_runtime->deviceLayerState(0),
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_plan_count.as<uint32_t>(),
                kPlanCapacity,
                transfer_directory->deviceEntries(),
                transfer_directory->slotCount(),
                apply_config,
                d_apply_status.as<DeviceMoERebalanceApplyStatus>(),
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                /*target_layer=*/kTargetLayer));

            /*
             * The source participates in the same domain-wide transaction even
             * though neither payload arrival targets participant zero. Publish
             * its independent completion record now; the split-row regression
             * below must consume the exact per-participant status object that a
             * real NCCL/RCCL graph would produce.
             */
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_command_header.get(),
                &source_header,
                sizeof(source_header),
                device.ordinal,
                stream));
            ASSERT_TRUE(kernel.applyDeviceRebalanceArrivals(
                launch,
                source_runtime->deviceLayerState(0),
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_plan_count.as<uint32_t>(),
                kPlanCapacity,
                transfer_directory->deviceEntries(),
                transfer_directory->slotCount(),
                prefillLLEPTransferConfig(
                    source_config,
                    PrefillLLEPTransferPurpose::CurrentBatchMovement),
                d_source_apply_status.as<DeviceMoERebalanceApplyStatus>(),
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                /*target_layer=*/kTargetLayer));

            DeviceMoERebalanceStatus transfer_status;
            transfer_status.planned_arrivals = kPlanCapacity;
            transfer_status.llep_assignment_span_count = kNumExperts;
            transfer_status.llep_weight_transfer_count = kPlanCapacity;
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_transfer_status.get(),
                &transfer_status,
                sizeof(transfer_status),
                device.ordinal,
                stream));

            DeviceMoERebalanceApplyStatus observed_apply{};
            DeviceMoELayerRuntime observed_retained_runtime{};
            DeviceMoELayerRuntime observed_runtime{};
            ASSERT_TRUE(backend->deviceToHostOnStream(
                &observed_apply,
                d_apply_status.get(),
                sizeof(observed_apply),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->deviceToHostOnStream(
                &observed_retained_runtime,
                destination_runtime->deviceLayerState(0),
                sizeof(observed_retained_runtime),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->deviceToHostOnStream(
                &observed_runtime,
                destination_runtime->deviceLayerState(kTargetLayer),
                sizeof(observed_runtime),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));

            ASSERT_EQ(observed_apply.status_code, 0u);
            ASSERT_EQ(observed_apply.invalid_plan_entries, 0u);
            ASSERT_EQ(observed_apply.missing_source_descriptors, 0u);
            ASSERT_EQ(observed_apply.missing_destination_slots, 0u);
            ASSERT_EQ(observed_apply.descriptor_mismatches, 0u);
            ASSERT_EQ(observed_apply.copy_incomplete, 0u);
            ASSERT_EQ(observed_apply.required_local_arrivals, kPlanCapacity);
            ASSERT_EQ(observed_apply.ready_local_arrivals, kPlanCapacity);
            ASSERT_EQ(observed_apply.applied_arrivals, kPlanCapacity);

            ASSERT_LE(observed_runtime.active_bank, 1u);
            const auto &destination_bank =
                observed_runtime.banks[observed_runtime.active_bank];
            for (int expert = 0; expert < kNumExperts; ++expert)
            {
                ASSERT_EQ(destination_bank.local_compute_mask[expert], 1u)
                    << "expert=" << expert;
                ASSERT_EQ(destination_bank.resident_participant_mask[expert], 0b11u)
                    << "expert=" << expert;
                ASSERT_EQ(
                    destination_bank.experts[expert].gate.codebook_id,
                    format.device_execution_codebook_id)
                    << "expert=" << expert;
            }
            const auto &slot_entries =
                transfer_directory->hostEntriesForTest();
            ASSERT_LE(observed_retained_runtime.active_bank, 1u);
            const auto &retained_bank = observed_retained_runtime.banks[
                observed_retained_runtime.active_bank];
            for (uint32_t plan_index = 0u;
                 plan_index < kPlanCapacity;
                 ++plan_index)
            {
                const uint32_t expert = transferred_experts[plan_index];
                const auto &retained_descriptor =
                    retained_bank.experts[expert];
                const auto &retained_slot_descriptor =
                    slot_entries[plan_index].descriptor;
                const uint32_t physical_slot =
                    kPhysicalTransferSlotBase + plan_index;
                const auto &destination_descriptor =
                    destination_bank.experts[expert];
                const auto &slot_descriptor =
                    slot_entries[physical_slot].descriptor;
                ASSERT_EQ(
                    retained_descriptor.local_slot,
                    static_cast<int32_t>(plan_index));
                ASSERT_EQ(
                    retained_descriptor.gate.payload,
                    retained_slot_descriptor.gate.payload);
                ASSERT_EQ(
                    retained_descriptor.up.payload,
                    retained_slot_descriptor.up.payload);
                ASSERT_EQ(
                    retained_descriptor.down.payload,
                    retained_slot_descriptor.down.payload);
                ASSERT_EQ(
                    destination_descriptor.local_slot,
                    static_cast<int32_t>(physical_slot));
                ASSERT_EQ(
                    destination_descriptor.gate.payload,
                    slot_descriptor.gate.payload);
                ASSERT_EQ(
                    destination_descriptor.up.payload,
                    slot_descriptor.up.payload);
                ASSERT_EQ(
                    destination_descriptor.down.payload,
                    slot_descriptor.down.payload);
            }
            ASSERT_EQ(observed_runtime.current_batch_llep_movement_observed, 1u);

            for (const int rows : kRows)
            {
                SCOPED_TRACE("M=" + std::to_string(rows));
                auto hidden = makeHidden(rows, kDModel, format_index);
                ASSERT_TRUE(hidden->ensureOnDevice(device, stream));

                auto routing_indices = TestTensorFactory::createFP32(
                    {static_cast<size_t>(rows), static_cast<size_t>(kTopK)});
                auto routing_weights = TestTensorFactory::createFP32(
                    {static_cast<size_t>(rows), static_cast<size_t>(kTopK)});
                constexpr std::array<float, kTopK> kRouteWeights = {
                    0.25f, 0.20f, 0.16f, 0.13f,
                    0.10f, 0.07f, 0.05f, 0.04f};
                for (int row = 0; row < rows; ++row)
                {
                    for (int route = 0; route < kTopK; ++route)
                    {
                        const size_t slot =
                            static_cast<size_t>(row) * kTopK + route;
                        routing_indices->mutable_data()[slot] =
                            static_cast<float>((row + route) % kNumExperts);
                        routing_weights->mutable_data()[slot] =
                            kRouteWeights[route];
                    }
                }
                ASSERT_TRUE(routing_indices->ensureOnDevice(device, stream));
                ASSERT_TRUE(routing_weights->ensureOnDevice(device, stream));

                ASSERT_TRUE(kernel.publishCompleteGroupedPrefillPlanFromRouter(
                    source_runtime->deviceLayerState(kTargetLayer),
                    routing_indices.get(),
                    routing_weights.get(),
                    rows,
                    rows,
                    kNumExperts,
                    kTopK,
                    source_gateup_table,
                    source_down_table,
                    /*filter_to_local_runtime_experts=*/true));
                std::vector<float> static_owner_canonical;
                const auto static_owner_output = executePublishedPlan(
                    backend,
                    kernel,
                    *source_runtime,
                    device,
                    stream,
                    hidden.get(),
                    source_gateup_table,
                    source_down_table,
                    rows,
                    kDModel,
                    kIntermediate,
                    kNumExperts,
                    kTopK,
                    kTargetLayer,
                    &static_owner_canonical);

                ASSERT_TRUE(kernel.groupPrefillRoutes(
                    destination_runtime->deviceLayerState(kTargetLayer),
                    routing_indices.get(),
                    routing_weights.get(),
                    rows,
                    rows,
                    kNumExperts,
                    kTopK,
                    /*filter_to_local_runtime_experts=*/false));

                std::array<least_loaded_ep::LeastLoadedExpertAssignmentSpan,
                           kNumExperts>
                    spans{};
                std::array<int32_t, kNumExperts * 2> span_bounds{};
                for (int expert = 0; expert < kNumExperts; ++expert)
                {
                    spans[expert] = {
                        .expert = static_cast<uint32_t>(expert),
                        .owner_participant = 0u,
                        .destination_participant = kDestinationParticipant,
                        .route_row_begin = 0u,
                        .route_row_end = static_cast<uint64_t>(rows),
                        .needs_foreign_weight = static_cast<uint8_t>(
                            (expert & 1) != 0 ? 1u : 0u),
                    };
                    span_bounds[2 * expert] = expert;
                    span_bounds[2 * expert + 1] = expert + 1;
                }
                ASSERT_TRUE(backend->hostToDeviceOnStream(
                    destination_host.reserved_ptrs[1],
                    spans.data(),
                    sizeof(spans),
                    device.ordinal,
                    stream));
                ASSERT_TRUE(backend->hostToDeviceOnStream(
                    destination_host.reserved_ptrs[0],
                    span_bounds.data(),
                    sizeof(span_bounds),
                    device.ordinal,
                    stream));
                ASSERT_TRUE(
                    kernel.assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
                        launch,
                        destination_runtime->deviceLayerState(kTargetLayer),
                        rows,
                        rows,
                        kNumExperts,
                        kTopK,
                        d_transfer_status.as<DeviceMoERebalanceStatus>(),
                        d_apply_status.as<DeviceMoERebalanceApplyStatus>()));

                std::vector<int32_t> assigned_participants(
                    static_cast<size_t>(rows) * kTopK,
                    -1);
                ASSERT_TRUE(backend->deviceToHostOnStream(
                    assigned_participants.data(),
                    destination_host.route_participant_ids,
                    assigned_participants.size() * sizeof(int32_t),
                    device.ordinal,
                    stream));
                ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                ASSERT_TRUE(std::all_of(
                    assigned_participants.begin(),
                    assigned_participants.end(),
                    [=](int32_t participant)
                    {
                        return participant ==
                               static_cast<int32_t>(kDestinationParticipant);
                    })) << "CurrentBatchLLEP did not assign every routed slot to the ready destination";

                ASSERT_TRUE(
                    kernel.publishCompleteGroupedPrefillPlanFromRuntimeAssignments(
                        destination_runtime->deviceLayerState(kTargetLayer),
                        rows,
                        rows,
                        kNumExperts,
                        kTopK,
                        destination_gateup_table,
                        destination_down_table));
                std::vector<float> llep_canonical;
                const auto llep_output = executePublishedPlan(
                    backend,
                    kernel,
                    *destination_runtime,
                    device,
                    stream,
                    hidden.get(),
                    destination_gateup_table,
                    destination_down_table,
                    rows,
                    kDModel,
                    kIntermediate,
                    kNumExperts,
                    kTopK,
                    kTargetLayer,
                    &llep_canonical);

                double norm_squared = 0.0;
                for (const float value : static_owner_output)
                {
                    ASSERT_TRUE(std::isfinite(value));
                    norm_squared += static_cast<double>(value) * value;
                }
                ASSERT_GT(norm_squared, 1.0e-14)
                    << format.label << " produced a degenerate all-zero witness";
                expectByteEqual(
                    std::string(backend_label) + " " + format.label +
                        " transferred CurrentBatchLLEP vs StaticOwner M=" +
                        std::to_string(rows),
                    llep_output,
                    static_owner_output,
                    kDModel);
                expectByteEqual(
                    std::string(backend_label) + " " + format.label +
                        " transferred CurrentBatchLLEP canonical routes vs StaticOwner M=" +
                        std::to_string(rows),
                    llep_canonical,
                    static_owner_canonical,
                    kDModel);

                /*
                 * Exercise the defining LLEP shape that the complete-expert
                 * comparison above cannot cover: every logical expert is split
                 * into two contiguous grouped-row spans and both participants
                 * execute one half. Production publishes those partials into
                 * the same canonical [row, route, column] coordinates before a
                 * rooted collective. Comparing the selected route slot before
                 * reduction isolates assignment/regrouping from collective
                 * arithmetic and proves that no row is duplicated, omitted, or
                 * associated with the wrong route weight.
                 */
                constexpr size_t kSplitSpanCount =
                    static_cast<size_t>(kNumExperts) * 2u;
                std::array<least_loaded_ep::LeastLoadedExpertAssignmentSpan,
                           kSplitSpanCount>
                    split_spans{};
                std::array<int32_t, kNumExperts * 2> split_span_bounds{};
                std::array<uint32_t, kNumExperts> grouped_route_counts{};
                for (int row = 0; row < rows; ++row)
                {
                    for (int route = 0; route < kTopK; ++route)
                    {
                        const int expert = (row + route) % kNumExperts;
                        ++grouped_route_counts[expert];
                    }
                }
                for (int expert = 0; expert < kNumExperts; ++expert)
                {
                    const size_t first = static_cast<size_t>(expert) * 2u;
                    const uint32_t route_count =
                        grouped_route_counts[expert];
                    const uint32_t split_row = (route_count + 1u) / 2u;
                    split_spans[first] = {
                        .expert = static_cast<uint32_t>(expert),
                        .owner_participant = 0u,
                        .destination_participant = 0u,
                        .route_row_begin = 0u,
                        .route_row_end = split_row,
                        .needs_foreign_weight = 0u,
                    };
                    split_spans[first + 1u] = {
                        .expert = static_cast<uint32_t>(expert),
                        .owner_participant = 0u,
                        .destination_participant = kDestinationParticipant,
                        .route_row_begin = split_row,
                        .route_row_end = route_count,
                        .needs_foreign_weight = static_cast<uint8_t>(
                            (expert & 1) != 0 ? 1u : 0u),
                    };
                    split_span_bounds[2 * expert] =
                        static_cast<int32_t>(first);
                    split_span_bounds[2 * expert + 1] =
                        static_cast<int32_t>(first + 2u);
                }

                for (auto *runtime : {source_runtime.get(),
                                      destination_runtime.get()})
                {
                    ASSERT_TRUE(kernel.groupPrefillRoutes(
                        runtime->deviceLayerState(kTargetLayer),
                        routing_indices.get(),
                        routing_weights.get(),
                        rows,
                        rows,
                        kNumExperts,
                        kTopK,
                        /*filter_to_local_runtime_experts=*/false));
                    ASSERT_TRUE(publishCurrentBatchPlanCounts(
                        backend,
                        device,
                        stream,
                        runtime->deviceLayerState(kTargetLayer),
                        kSplitSpanCount,
                        kPlanCapacity));
                    auto &host_runtime = runtime->hostLayerState(kTargetLayer);
                    ASSERT_TRUE(backend->hostToDeviceOnStream(
                        host_runtime.reserved_ptrs[1],
                        split_spans.data(),
                        sizeof(split_spans),
                        device.ordinal,
                        stream));
                    ASSERT_TRUE(backend->hostToDeviceOnStream(
                        host_runtime.reserved_ptrs[0],
                        split_span_bounds.data(),
                        sizeof(split_span_bounds),
                        device.ordinal,
                        stream));
                }

                transfer_status.llep_assignment_span_count =
                    static_cast<uint32_t>(kSplitSpanCount);
                ASSERT_TRUE(backend->hostToDeviceOnStream(
                    d_transfer_status.get(),
                    &transfer_status,
                    sizeof(transfer_status),
                    device.ordinal,
                    stream));
                ASSERT_TRUE(
                    kernel.assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
                        launch,
                        source_runtime->deviceLayerState(kTargetLayer),
                        rows,
                        rows,
                        kNumExperts,
                        kTopK,
                        d_transfer_status.as<DeviceMoERebalanceStatus>(),
                        d_source_apply_status.as<DeviceMoERebalanceApplyStatus>()));
                ASSERT_TRUE(
                    kernel.assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
                        launch,
                        destination_runtime->deviceLayerState(kTargetLayer),
                        rows,
                        rows,
                        kNumExperts,
                        kTopK,
                        d_transfer_status.as<DeviceMoERebalanceStatus>(),
                        d_apply_status.as<DeviceMoERebalanceApplyStatus>()));

                std::vector<int32_t> source_assignments(
                    static_cast<size_t>(rows) * kTopK,
                    -1);
                std::vector<int32_t> destination_assignments(
                    source_assignments.size(),
                    -1);
                ASSERT_TRUE(backend->deviceToHostOnStream(
                    source_assignments.data(),
                    source_host.route_participant_ids,
                    source_assignments.size() * sizeof(int32_t),
                    device.ordinal,
                    stream));
                ASSERT_TRUE(backend->deviceToHostOnStream(
                    destination_assignments.data(),
                    destination_host.route_participant_ids,
                    destination_assignments.size() * sizeof(int32_t),
                    device.ordinal,
                    stream));
                ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                ASSERT_EQ(source_assignments, destination_assignments)
                    << "LLEP participants consumed different assignment spans";
                ASSERT_TRUE(std::all_of(
                    source_assignments.begin(),
                    source_assignments.end(),
                    [](int32_t participant)
                    {
                        return participant == 0 || participant == 1;
                    }));
                ASSERT_NE(
                    std::find(source_assignments.begin(),
                              source_assignments.end(),
                              0),
                    source_assignments.end());
                ASSERT_NE(
                    std::find(source_assignments.begin(),
                              source_assignments.end(),
                              1),
                    source_assignments.end());

                ASSERT_TRUE(
                    kernel.publishCompleteGroupedPrefillPlanFromRuntimeAssignments(
                        source_runtime->deviceLayerState(kTargetLayer),
                        rows,
                        rows,
                        kNumExperts,
                        kTopK,
                        source_gateup_table,
                        source_down_table));

                /*
                 * A production participant owns a distinct kernel instance and
                 * grouping workspace. This focused fixture intentionally reuses
                 * one physical GPU and one kernel instance for both simulated
                 * participants, so each participant's publication and execution
                 * must remain one indivisible test transaction. Publishing the
                 * destination first would overwrite the source participant's
                 * inverse map in the shared fixture workspace and manufacture a
                 * cross-participant race that cannot exist between real ranks.
                 */
                std::vector<float> source_split_canonical;
                const auto source_split_output = executePublishedPlan(
                    backend,
                    kernel,
                    *source_runtime,
                    device,
                    stream,
                    hidden.get(),
                    source_gateup_table,
                    source_down_table,
                    rows,
                    kDModel,
                    kIntermediate,
                    kNumExperts,
                    kTopK,
                    kTargetLayer,
                    &source_split_canonical);

                ASSERT_TRUE(
                    kernel.publishCompleteGroupedPrefillPlanFromRuntimeAssignments(
                        destination_runtime->deviceLayerState(kTargetLayer),
                        rows,
                        rows,
                        kNumExperts,
                        kTopK,
                        destination_gateup_table,
                        destination_down_table));

                std::vector<float> destination_split_canonical;
                const auto destination_split_output = executePublishedPlan(
                    backend,
                    kernel,
                    *destination_runtime,
                    device,
                    stream,
                    hidden.get(),
                    destination_gateup_table,
                    destination_down_table,
                    rows,
                    kDModel,
                    kIntermediate,
                    kNumExperts,
                    kTopK,
                    kTargetLayer,
                    &destination_split_canonical);
                (void)source_split_output;
                (void)destination_split_output;

                ASSERT_EQ(source_split_canonical.size(),
                          static_owner_canonical.size());
                ASSERT_EQ(destination_split_canonical.size(),
                          static_owner_canonical.size());
                std::vector<float> reconstructed_canonical(
                    static_owner_canonical.size(),
                    0.0f);
                for (size_t route_slot = 0;
                     route_slot < source_assignments.size();
                     ++route_slot)
                {
                    const bool source_selected =
                        source_assignments[route_slot] == 0;
                    const auto &selected =
                        source_selected
                            ? source_split_canonical
                            : destination_split_canonical;
                    const auto &inactive =
                        source_selected
                            ? destination_split_canonical
                            : source_split_canonical;
                    const size_t begin =
                        route_slot * static_cast<size_t>(kDModel);
                    const size_t end = begin + static_cast<size_t>(kDModel);
                    std::copy(
                        selected.begin() + static_cast<std::ptrdiff_t>(begin),
                        selected.begin() + static_cast<std::ptrdiff_t>(end),
                        reconstructed_canonical.begin() +
                            static_cast<std::ptrdiff_t>(begin));
                    for (size_t element = begin; element < end; ++element)
                    {
                        uint32_t inactive_bits = 0u;
                        std::memcpy(
                            &inactive_bits,
                            &inactive[element],
                            sizeof(inactive_bits));
                        ASSERT_EQ(inactive_bits & 0x7fffffffu, 0u)
                            << "inactive participant wrote canonical route slot="
                            << route_slot << " element=" << element;
                    }
                }
                expectByteEqual(
                    std::string(backend_label) + " " + format.label +
                        " participant-split CurrentBatchLLEP canonical routes vs StaticOwner M=" +
                        std::to_string(rows),
                    reconstructed_canonical,
                    static_owner_canonical,
                    kDModel);
            }
        }

        ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
        workspace_consumer->unbindWorkspace();
    }

    namespace native_vnni_transfer_parity_detail
    {
        /**
         * @brief Own one CPU-promoted asymmetric projection on a GPU endpoint.
         *
         * The three allocation owners keep the descriptor stable while grouped
         * inference executes. `stats` is retained after the short-lived transfer
         * lane is destroyed so the caller can prove promotion did not introduce
         * an inference-stream wait or a blocking synchronization.
         */
        struct PromotedAsymmetricProjection final
        {
            std::unique_ptr<DeviceAllocation> payload;
            std::unique_ptr<DeviceAllocation> scales;
            std::unique_ptr<DeviceAllocation> mins;
            DeviceNativeVNNIMatrixDesc descriptor{};
            ExpertTierWeightTransferLaneStats stats{};
        };

        /**
         * @brief Stream one CPU-native asymmetric matrix into executable GPU form.
         *
         * @param backend Exact destination backend.
         * @param device CUDA or ROCm endpoint that owns the inactive allocation.
         * @param cpu_weights Real CPU prepared bytes retaining source provenance.
         * @param projection Stable gate/up/down role used in the transfer manifest.
         * @param identity Unique diagnostic identity for the transaction.
         * @return Stable codebook-23 descriptor and its device allocation owners.
         * @throws std::runtime_error when materialization, streaming, or format
         *         publication fails.
         */
        inline PromotedAsymmetricProjection promoteAsymmetricProjection(
            IBackend *backend,
            DeviceId device,
            const cpu::native_vnni::CPUNativeVNNIPackedWeights &cpu_weights,
            ExpertTierWeightProjection projection,
            uint64_t identity)
        {
            if (!backend || !device.is_gpu() ||
                !cpu_weights.usesExpandedInt8() ||
                !cpu_weights.is_asymmetric)
            {
                throw std::invalid_argument(
                    "Grouped promotion parity requires asymmetric expanded CPU weights");
            }

            const auto manifest = makeCpuToGpuExpertTierWeightStreamManifest(
                cpu_weights,
                identity,
                /*layer_id=*/0,
                /*expert_id=*/0,
                projection,
                /*maximum_units_per_chunk=*/1);
            const auto layout = manifest.deviceLayout();
            if (!layout.valid() ||
                layout.gpu_codebook_id !=
                    kNativeVnniExpandedInt8MinCodebook)
            {
                throw std::runtime_error(
                    "Asymmetric promotion did not select execution codebook 23");
            }

            const size_t blocks =
                static_cast<size_t>(layout.N) *
                static_cast<size_t>(layout.blocks_per_row);
            PromotedAsymmetricProjection result;
            result.payload = std::make_unique<DeviceAllocation>(
                backend,
                device.ordinal,
                blocks * layout.gpu_payload_bytes_per_block);
            result.scales = std::make_unique<DeviceAllocation>(
                backend,
                device.ordinal,
                blocks * sizeof(uint16_t));
            result.mins = std::make_unique<DeviceAllocation>(
                backend,
                device.ordinal,
                blocks * sizeof(uint16_t));

            ExpertTierGpuMutableProjectionView destination{
                .payload = result.payload->as<uint8_t>(),
                .scales = result.scales->as<uint16_t>(),
                .mins = result.mins->as<uint16_t>(),
                .emins = nullptr,
                .payload_bytes = result.payload->bytes(),
                .scales_bytes = result.scales->bytes(),
                .mins_bytes = result.mins->bytes(),
                .emins_bytes = 0u,
            };
            if (!destination.validFor(layout))
            {
                throw std::runtime_error(
                    "Asymmetric promotion allocation does not satisfy its layout");
            }

            std::string error;
            ExpertTierWeightTransferLane lane({
                .device = device,
                .staging_capacity_bytes = layout.chunkBytes(1),
                .lane_name = "grouped_asymmetric_promotion_" +
                             std::to_string(identity),
                .perf_device = device.to_string(),
                .collect_timing_measurements = true,
            });
            if (!lane.materialize(&error) ||
                !lane.startCpuToGpu(
                    layout,
                    cpu_weights.native_interleaved,
                    destination,
                    &error))
            {
                throw std::runtime_error(
                    "Failed to start grouped asymmetric promotion: " + error);
            }

            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(10);
            auto progress = lane.progress();
            while (progress == ExpertTierWeightTransferProgress::Pending &&
                   std::chrono::steady_clock::now() < deadline)
            {
                progress = lane.poll(&error);
                std::this_thread::yield();
            }
            if (progress != ExpertTierWeightTransferProgress::Ready)
            {
                throw std::runtime_error(
                    "Grouped asymmetric promotion did not complete: " + error);
            }
            result.stats = lane.stats();
            if (result.stats.transfers_completed != 1u ||
                result.stats.timing_measurement_failures != 0u ||
                !result.stats.last_measurement.valid() ||
                result.stats.last_measurement.device_nanoseconds == 0u ||
                result.stats.blocking_synchronizations != 0u ||
                result.stats.inference_stream_waits != 0u)
            {
                throw std::runtime_error(
                    "Grouped asymmetric promotion violated the async lane contract");
            }

            result.descriptor = DeviceNativeVNNIMatrixDesc{
                .payload = result.payload->as<uint8_t>(),
                .scales = result.scales->get(),
                .mins = result.mins->get(),
                .emins = nullptr,
                .n = layout.N,
                .k = layout.K,
                .blocks_per_row =
                    static_cast<uint32_t>(layout.blocks_per_row),
                .codebook_id = layout.gpu_codebook_id,
                .allocation_payload_bytes_per_block =
                    layout.gpu_payload_bytes_per_block,
                .allocation_has_mins =
                    static_cast<uint8_t>(layout.gpu_is_asymmetric),
                .allocation_has_emins =
                    static_cast<uint8_t>(layout.gpu_has_emins),
                .source_codebook_id = layout.gpu_source_codebook_id,
                .source_is_superblock = layout.gpu_source_is_superblock,
                .source_identity_present = 1u,
            };
            return result;
        }
    } // namespace native_vnni_transfer_parity_detail

    /**
     * @brief Prove promoted codebook-23 weights through grouped MoE execution.
     *
     * Three real Q5_1 projections are prepared twice: the ordinary compact GPU
     * load path publishes codebook 7, while the CPU-cold promotion path streams
     * the same mathematical weights into codebook 23. Both descriptor triplets
     * then execute the production grouped route plan at verifier and long-prefill
     * sizes. Final outputs and canonical route contributions must be byte equal.
     *
     * @param backend_label Stable diagnostic label (CUDA or ROCm).
     * @param device Physical backend endpoint.
     * @param stream Explicit non-null production stream.
     */
    inline void runExpandedAsymmetricGroupedMoEParity(
        const char *backend_label,
        DeviceId device,
        void *stream)
    {
        using namespace native_vnni_transfer_parity_detail;
        if (!backend_label || !device.is_gpu() || !stream)
            throw std::invalid_argument(
                "Grouped asymmetric parity requires a GPU and explicit stream");
        IBackend *backend = getBackendFor(device);
        if (!backend)
            throw std::runtime_error(
                "Grouped asymmetric parity backend is unavailable");

        constexpr int kDModel = 2048;
        constexpr int kIntermediate = 32;
        constexpr int kNumExperts = 2;
        constexpr int kTopK = 2;
        constexpr int kMaxRows = 65;
        constexpr int kLayer = 0;
        const auto &format = quantizedVerifierFormat("Q5_1");

        std::vector<std::unique_ptr<TensorBase>> weights;
        std::vector<GpuPreparedGemm> prepared;
        weights.reserve(3u);
        prepared.reserve(3u);
        const uint64_t model_base =
            device.is_cuda() ? 2910000u : 2920000u;
        const std::string prefix =
            std::string("test.") + backend_label +
            ".expanded_asymmetric_grouped";
        const DeviceNativeVNNIMatrixDesc compact_gate =
            prepareMatrixDescriptor(
                format,
                device,
                stream,
                kIntermediate,
                kDModel,
                99181u,
                prefix + ".gate",
                ModelContextId{model_base + 1u},
                weights,
                prepared);
        const DeviceNativeVNNIMatrixDesc compact_up =
            prepareMatrixDescriptor(
                format,
                device,
                stream,
                kIntermediate,
                kDModel,
                99182u,
                prefix + ".up",
                ModelContextId{model_base + 2u},
                weights,
                prepared);
        const DeviceNativeVNNIMatrixDesc compact_down =
            prepareMatrixDescriptor(
                format,
                device,
                stream,
                kDModel,
                kIntermediate,
                99183u,
                prefix + ".down",
                ModelContextId{model_base + 3u},
                weights,
                prepared);
        ASSERT_EQ(compact_gate.codebook_id, 7u);
        ASSERT_EQ(compact_up.codebook_id, 7u);
        ASSERT_EQ(compact_down.codebook_id, 7u);

        std::array<cpu::native_vnni::CPUNativeVNNIPackedWeights, 3>
            cpu_weights;
        for (size_t index = 0; index < cpu_weights.size(); ++index)
        {
            ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(
                weights[index].get(), cpu_weights[index]));
            ASSERT_TRUE(cpu_weights[index].usesExpandedInt8());
            ASSERT_TRUE(cpu_weights[index].is_asymmetric);
        }
        auto promoted_gate = promoteAsymmetricProjection(
            backend,
            device,
            cpu_weights[0],
            ExpertTierWeightProjection::Gate,
            model_base + 11u);
        auto promoted_up = promoteAsymmetricProjection(
            backend,
            device,
            cpu_weights[1],
            ExpertTierWeightProjection::Up,
            model_base + 12u);
        auto promoted_down = promoteAsymmetricProjection(
            backend,
            device,
            cpu_weights[2],
            ExpertTierWeightProjection::Down,
            model_base + 13u);

        const auto make_experts = [&](
            const DeviceNativeVNNIMatrixDesc &gate,
            const DeviceNativeVNNIMatrixDesc &up,
            const DeviceNativeVNNIMatrixDesc &down)
        {
            std::vector<DeviceMoEExpertDescriptor> descriptors(kNumExperts);
            for (int expert = 0; expert < kNumExperts; ++expert)
            {
                auto &descriptor = descriptors[expert];
                descriptor.logical_expert_id = expert;
                descriptor.owner_participant = 0;
                descriptor.local_slot = expert;
                descriptor.flags = toMoEExpertFlags(
                    DeviceMoEExpertFlags::Valid |
                    DeviceMoEExpertFlags::Resident |
                    DeviceMoEExpertFlags::LocalCompute);
                descriptor.gate = gate;
                descriptor.up = up;
                descriptor.down = down;
            }
            return descriptors;
        };
        const auto compact_experts =
            make_experts(compact_gate, compact_up, compact_down);
        const auto promoted_experts = make_experts(
            promoted_gate.descriptor,
            promoted_up.descriptor,
            promoted_down.descriptor);

        auto kernel_owner =
            llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
        ASSERT_NE(kernel_owner, nullptr);
        kernel_owner->setGPUStream(stream);
        auto requirements = device.is_cuda()
                                ? MoEWorkspaceBuffers::cudaMoE(
                                      kMaxRows,
                                      kDModel,
                                      kIntermediate,
                                      kNumExperts,
                                      kTopK)
                                : MoEWorkspaceBuffers::rocmMoE(
                                      kMaxRows,
                                      kDModel,
                                      kIntermediate,
                                      kNumExperts,
                                      kTopK);
        DeviceWorkspaceManager workspace(
            device,
            requirements.total_bytes_with_alignment() + 4u * 1024u * 1024u);
        ASSERT_TRUE(workspace.allocate(requirements));
        auto *workspace_consumer =
            dynamic_cast<IWorkspaceConsumer *>(kernel_owner.get());
        ASSERT_NE(workspace_consumer, nullptr);
        workspace_consumer->bindWorkspace(&workspace);
        IMoEKernel &kernel = *kernel_owner;

        std::array<DeviceNativeVNNIMatrixDesc, kNumExperts> compact_gates{};
        std::array<DeviceNativeVNNIMatrixDesc, kNumExperts> compact_ups{};
        std::array<DeviceNativeVNNIMatrixDesc, kNumExperts> compact_downs{};
        std::array<DeviceNativeVNNIMatrixDesc, kNumExperts> promoted_gates{};
        std::array<DeviceNativeVNNIMatrixDesc, kNumExperts> promoted_ups{};
        std::array<DeviceNativeVNNIMatrixDesc, kNumExperts> promoted_downs{};
        for (int expert = 0; expert < kNumExperts; ++expert)
        {
            compact_gates[expert] = compact_gate;
            compact_ups[expert] = compact_up;
            compact_downs[expert] = compact_down;
            promoted_gates[expert] = promoted_gate.descriptor;
            promoted_ups[expert] = promoted_up.descriptor;
            promoted_downs[expert] = promoted_down.descriptor;
        }
        const int compact_gateup_table =
            kernel.uploadGroupedExpertGateUpDescriptorTables(
                compact_gates.data(),
                compact_ups.data(),
                kNumExperts,
                kDModel,
                kIntermediate);
        const int compact_down_table =
            kernel.uploadGroupedExpertDownDescriptorTable(
                compact_downs.data(),
                kNumExperts,
                kDModel,
                kIntermediate);
        const int promoted_gateup_table =
            kernel.uploadGroupedExpertGateUpDescriptorTables(
                promoted_gates.data(),
                promoted_ups.data(),
                kNumExperts,
                kDModel,
                kIntermediate);
        const int promoted_down_table =
            kernel.uploadGroupedExpertDownDescriptorTable(
                promoted_downs.data(),
                kNumExperts,
                kDModel,
                kIntermediate);
        ASSERT_GE(compact_gateup_table, 0);
        ASSERT_GE(compact_down_table, 0);
        ASSERT_GE(promoted_gateup_table, 0);
        ASSERT_GE(promoted_down_table, 0);

        auto compact_runtime = makeRuntimeTable(
            device,
            stream,
            compact_experts,
            /*participant_id=*/0u,
            std::vector<uint8_t>(kNumExperts, 1u),
            std::vector<uint32_t>(kNumExperts, 0b01u),
            /*num_layers=*/1,
            kTopK,
            kMaxRows);
        auto promoted_runtime = makeRuntimeTable(
            device,
            stream,
            promoted_experts,
            /*participant_id=*/0u,
            std::vector<uint8_t>(kNumExperts, 1u),
            std::vector<uint32_t>(kNumExperts, 0b01u),
            /*num_layers=*/1,
            kTopK,
            kMaxRows);

        for (const int rows : {2, 4, 16, kMaxRows})
        {
            SCOPED_TRACE(
                std::string(backend_label) + " grouped promoted M=" +
                std::to_string(rows));
            auto hidden = makeHidden(rows, kDModel, 77u);
            auto routing_indices = TestTensorFactory::createFP32(
                {static_cast<size_t>(rows), kTopK});
            auto routing_weights = TestTensorFactory::createFP32(
                {static_cast<size_t>(rows), kTopK});
            for (int row = 0; row < rows; ++row)
            {
                routing_indices->mutable_data()[row * kTopK] =
                    static_cast<float>(row & 1);
                routing_indices->mutable_data()[row * kTopK + 1] =
                    static_cast<float>((row + 1) & 1);
                routing_weights->mutable_data()[row * kTopK] = 0.625f;
                routing_weights->mutable_data()[row * kTopK + 1] = 0.375f;
            }
            ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
            ASSERT_TRUE(routing_indices->ensureOnDevice(device, stream));
            ASSERT_TRUE(routing_weights->ensureOnDevice(device, stream));

            ASSERT_TRUE(kernel.publishCompleteGroupedPrefillPlanFromRouter(
                compact_runtime->deviceLayerState(kLayer),
                routing_indices.get(),
                routing_weights.get(),
                rows,
                rows,
                kNumExperts,
                kTopK,
                compact_gateup_table,
                compact_down_table,
                /*filter_to_local_runtime_experts=*/true));
            std::vector<float> compact_canonical;
            const auto compact_output = executePublishedPlan(
                backend,
                kernel,
                *compact_runtime,
                device,
                stream,
                hidden.get(),
                compact_gateup_table,
                compact_down_table,
                rows,
                kDModel,
                kIntermediate,
                kNumExperts,
                kTopK,
                kLayer,
                &compact_canonical);

            ASSERT_TRUE(kernel.publishCompleteGroupedPrefillPlanFromRouter(
                promoted_runtime->deviceLayerState(kLayer),
                routing_indices.get(),
                routing_weights.get(),
                rows,
                rows,
                kNumExperts,
                kTopK,
                promoted_gateup_table,
                promoted_down_table,
                /*filter_to_local_runtime_experts=*/true));
            std::vector<float> promoted_canonical;
            const auto promoted_output = executePublishedPlan(
                backend,
                kernel,
                *promoted_runtime,
                device,
                stream,
                hidden.get(),
                promoted_gateup_table,
                promoted_down_table,
                rows,
                kDModel,
                kIntermediate,
                kNumExperts,
                kTopK,
                kLayer,
                &promoted_canonical);

            double norm_squared = 0.0;
            for (const float value : compact_output)
            {
                ASSERT_TRUE(std::isfinite(value));
                norm_squared += static_cast<double>(value) * value;
            }
            ASSERT_GT(norm_squared, 1.0e-14);
            expectByteEqual(
                std::string(backend_label) +
                    " grouped promoted output M=" + std::to_string(rows),
                promoted_output,
                compact_output,
                kDModel);
            expectByteEqual(
                std::string(backend_label) +
                    " grouped promoted canonical M=" + std::to_string(rows),
                promoted_canonical,
                compact_canonical,
                kDModel);
        }

        ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
        workspace_consumer->unbindWorkspace();
    }
} // namespace llaminar2::test
