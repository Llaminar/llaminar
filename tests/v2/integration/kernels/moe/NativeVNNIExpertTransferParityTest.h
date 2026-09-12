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
 * For every canonical quantized model format and the shared grouped/prefill
 * geometry inventory, the arithmetic test compares the same routed transaction:
 *
 * 1. StaticOwner executes the canonical source experts on participant zero.
 * 2. CurrentBatchLLEP retains existing destination experts and transfers missing
 *    experts into disjoint layer-local slots through the compact-payload ABI.
 *    Transient child tables hold their canonical parent's acquired epoch until
 *    assignment-span consumption and grouped execution have completed.
 * A separate functional scope certifies every format's copied bytes and parent
 * immutability without paying for the full grouped arithmetic inventory.
 *
 * Equality is byte equality. The helper contains no row-replay arithmetic and
 * does not replace either backend's production grouped implementation.
 * Immutable router weights live for the entire workspace, just as they do in
 * a model instance. Changing expert inputs or formats must not manufacture a
 * new model router and exhaust the graph family's retained publications.
 */

#pragma once

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/DeviceMoERebalanceABI.h"
#include "execution/moe/DeviceMoEOverlayEpochArena.h"
#include "execution/moe/DeviceMoETransferSlotDirectory.h"
#include "execution/moe/CpuExpertSlotPool.h"
#include "execution/moe/ExpertTierWeightStream.h"
#include "execution/moe/ExpertTierWeightTransferLane.h"
#include "execution/moe/LeastLoadedExpertAssignment.h"
#include "execution/moe/MoEOverlayHostAuthorityDeviceBankPublisher.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/MoERuntimeTable.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"

#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/TestTensorFactory.h"

#include <algorithm>
#include <array>
#include <atomic>
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
         *
         * @param device Physical backend endpoint.
         * @param stream Explicit initialization and publication stream.
         * @param descriptors Canonical prepared projection identities.
         * @param participant_id This table's domain-local participant.
         * @param local_compute_mask Initial local eligibility, expert indexed.
         * @param resident_participant_mask Initial physical residency per expert.
         * @param num_layers Number of identical fixture layers to initialize.
         * @param top_k Configured routing width.
         * @param token_capacity Persistent grouped route-scratch capacity.
         * @param overlay_epoch_arena Optional production reader-epoch owner.
         * @param overlay_placement_source Canonical parent for a transient child;
         *        must outlive this table and share its exact epoch ticket.
         * @return Initialized participant table with stable device bindings.
         * @throws std::invalid_argument for inconsistent geometry or ownership.
         * @throws std::runtime_error when setup publication fails.
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
            int token_capacity,
            std::shared_ptr<DeviceMoEOverlayEpochArena> overlay_epoch_arena = {},
            DeviceMoERuntimeTable *overlay_placement_source = nullptr)
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
                .overlay_epoch_arena = std::move(overlay_epoch_arena),
                .overlay_placement_source = overlay_placement_source,
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

        /** @brief Distinct spatial distributions for immutable expert inputs. */
        enum class ExpertInputPattern { Smooth, Scrambled };

        /**
         * @brief Create deterministic hidden states without depending on a random library.
         * @param rows Logical token rows.
         * @param d_model Hidden width.
         * @param format_index Stable seed distinguishing operand families.
         * @param pattern Smooth low-amplitude inputs or scrambled unit-scale inputs.
         *
         * The scrambled case exercises realistic normalized magnitudes and
         * cancellation. Small smooth waves alone do not cover the nonlinear
         * SwiGLU or Q8 rounding boundaries reached by real expert activations.
         */
        inline std::unique_ptr<TensorBase> makeHidden(
            int rows,
            int d_model,
            size_t format_index,
            ExpertInputPattern pattern = ExpertInputPattern::Smooth)
        {
            auto hidden = TestTensorFactory::createFP32(
                {static_cast<size_t>(rows), static_cast<size_t>(d_model)});
            float *values = hidden->mutable_data();
            for (size_t i = 0; i < hidden->numel(); ++i)
            {
                if (pattern == ExpertInputPattern::Scrambled)
                {
                    // Integer mixing and an exact binary scale give both
                    // endpoints identical input words, including sign changes.
                    uint32_t bits = static_cast<uint32_t>(i) +
                                    static_cast<uint32_t>(format_index) * 0x9e3779b9u;
                    bits ^= bits >> 16;
                    bits *= 0x7feb352du;
                    bits ^= bits >> 15;
                    bits *= 0x846ca68bu;
                    bits ^= bits >> 16;
                    values[i] = static_cast<float>(static_cast<int>(bits & 0xffffu) - 32768) *
                                (1.0f / 16384.0f);
                    continue;
                }
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
            // Equal NaN payloads are not an arithmetic proof. Reject invalid
            // operands/results before the byte comparison can hide overflow.
            ASSERT_TRUE(std::all_of(actual.begin(), actual.end(),
                                    [](float value) { return std::isfinite(value); }))
                << label << " has non-finite actual output";
            ASSERT_TRUE(std::all_of(expected.begin(), expected.end(),
                                    [](float value) { return std::isfinite(value); }))
                << label << " has non-finite expected output";
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

        /**
         * @brief Own one CPU-promoted projection on a GPU endpoint.
         *
         * Optional metadata allocations follow the exact destination layout.
         * The owners keep every descriptor address stable until grouped
         * inference has consumed the promoted expert.
         */
        struct PromotedCPUProjection final
        {
            std::unique_ptr<DeviceAllocation> payload;
            std::unique_ptr<DeviceAllocation> scales;
            std::unique_ptr<DeviceAllocation> mins;
            std::unique_ptr<DeviceAllocation> emins;
            DeviceNativeVNNIMatrixDesc descriptor{};
            ExpertTierWeightTransferLaneStats stats{};
        };

        /**
         * @brief Stream final CPU execution bytes into their exact GPU form.
         * @param backend Exact destination backend.
         * @param device CUDA or ROCm endpoint that owns the inactive allocation.
         * @param cpu_weights Real CPU prepared bytes retaining source provenance.
         * @param projection Stable gate/up/down role used in the transfer manifest.
         * @param identity Unique diagnostic identity for the transaction.
         * @return Stable promoted descriptor and all of its allocation owners.
         */
        PromotedCPUProjection promoteCPUProjection(
            IBackend *backend,
            DeviceId device,
            const cpu::native_vnni::CPUNativeVNNIPackedWeights &cpu_weights,
            ExpertTierWeightProjection projection,
            uint64_t identity);

        /**
         * @brief Demote a GPU-prepared triplet into the real CPU inactive pool.
         * @param device Source device whose packed projections are immutable.
         * @param stream Exact source readiness stream.
         * @param format Original source-format identity.
         * @param descriptors Gate, up and down in projection order.
         * @return Lease retaining final CPU bytes and their executable engines.
         */
        CpuExpertSlotPool::Lease demoteGPUExpert(
            DeviceId device, void *stream,
            const QuantizedVerifierFormatCase &format,
            const std::array<DeviceNativeVNNIMatrixDesc, 3> &descriptors);

        /**
         * @brief Compare retained CPU sparse-packet execution with an independent row oracle.
         * @param engines Prepared gate/up/down engines, including real arrival slots.
         * @param hidden Compact oracle input rows; packet storage has nontrivial physical indices.
         * @param expected One unweighted expert output per input row.
         * @param rows Live rows, independent of retained scratch capacity.
         * @param d_model Input/output width.
         * @param intermediate Gate/up width.
         *
         * Eight logical experts share immutable weights. Narrow, wide and empty
         * route packets reuse the same executor and exercise every compact width
         * bucket, transported-input publication and retained scratch without
         * manufacturing another arithmetic oracle.
         */
        void expectCPUExpertPacketParity(
            const std::array<ITensorGemm *, 3> &engines,
            const float *hidden, const float *expected,
            int rows, int d_model, int intermediate);

        /**
         * @brief Extend the complete format inventory with numerical boundary operands.
         * @return All canonical quantized formats, plus a Q8 source whose half
         *         scales span zero, subnormal and small normal encodings.
         */
        std::vector<QuantizedVerifierFormatCase> crossTierArithmeticFormats();
    } // namespace native_vnni_transfer_parity_detail

    /**
     * @brief Prove one logical quantized expert is byte-stable across CPU/GPU.
     *
     * The transfer-only regressions above prove that every codebook survives
     * repacking, but movement correctness also requires the destination to
     * execute the same arithmetic program. For every canonical quantized
     * source format this test prepares the same gate/up/down tensors twice:
     * once as GPU-aligned CPU NativeVNNI engines and once through the production
     * CUDA/ROCm loader. The GPU endpoint runs its real grouped sparse-MoE plan;
     * the CPU endpoint runs the complete grouped FFN transaction. Hidden Q8,
     * gate/up, SwiGLU Q8, down, and canonical route publication are therefore
     * all inside the comparison rather than mocked by a format conversion.
     * The same operands cover both small and larger hidden states: tiny
     * activations alone can miss rounding differences in SwiGLU and its second
     * quantization boundary when a real expert changes device ownership.
     *
     * @param backend_label Stable diagnostic backend label.
     * @param device CUDA or ROCm endpoint paired with the CPU tier.
     * @param stream Explicit non-default GPU stream.
     * @param formats Source operands; registered gates use the complete default
     *        inventory. A focused diagnostic can supply actual GGUF triplets.
     * @param input_scale Positive diagnostic amplitude multiplier. Synthetic
     *        format fixtures use one; native weights can probe larger states
     *        without overflowing the synthetic fixtures' oversized scales.
     */
    inline void runCPUToGPUAllFormatExpertArithmeticParity(
        const char *backend_label,
        DeviceId device,
        void *stream,
        const std::vector<QuantizedVerifierFormatCase> &formats =
            native_vnni_transfer_parity_detail::crossTierArithmeticFormats(),
        float input_scale = 1.0f)
    {
        using namespace native_vnni_transfer_parity_detail;
        using CpuKernel =
            cpu::native_vnni::CPUNativeVNNIGemmKernel;
        using CpuDescriptor =
            CpuKernel::BatchedPrequantizedProjectionDesc;

        if (!backend_label || !device.is_gpu() || !stream ||
            !std::isfinite(input_scale) || input_scale <= 0.0f)
        {
            throw std::invalid_argument(
                "Cross-tier expert arithmetic parity requires a GPU and stream");
        }
        IBackend *backend = getBackendFor(device);
        if (!backend)
            throw std::runtime_error("Cross-tier parity backend is unavailable");

        // Qwen3.5-122B production expert geometry. Both K widths exercise the
        // complete GPU-aligned expert tree, while M=65 crosses from the
        // verifier family into the scalable prefill family.
        constexpr int kDModel = 3072;
        constexpr int kIntermediate = 1024;
        constexpr int kNumExperts = 1;
        constexpr int kTopK = 1;
        constexpr int kMaximumRows = 65;
        /** @brief One shape and amplitude of the immutable expert input. */
        struct InputCase
        {
            int rows;
            float amplitude;
            ExpertInputPattern pattern = ExpertInputPattern::Smooth;
        };
        constexpr std::array<InputCase, 12> kInputs = {{
            {1, 1.0f}, {2, 1.0f}, {3, 1.0f}, {15, 1.0f}, {65, 1.0f},
            {1, 4.0f}, {3, 4.0f}, {65, 4.0f},
            // Random format fixtures deliberately have much larger scales
            // than trained weights. Keep their composed Q8_1 intermediates
            // finite; native-GGUF diagnostics can request a larger input_scale.
            {1, 0.03125f, ExpertInputPattern::Scrambled},
            {1, 0.0625f, ExpertInputPattern::Scrambled},
            {1, 0.125f, ExpertInputPattern::Scrambled},
            {3, 0.0625f, ExpertInputPattern::Scrambled},
        }};

        // A workspace retains immutable router conversions for its entire
        // model lifetime. All cases use the same one-expert router; allocating
        // a new tensor per bank/input makes slot use depend on address reuse
        // and eventually exhausts the cache. Keep its source alive even after
        // the workspace/kernel retire. Expert weights and inputs still vary.
        auto router = TestTensorFactory::createFP32({kNumExperts, kDModel});
        std::fill_n(router->mutable_data(), router->numel(), 0.0f);
        ASSERT_TRUE(router->ensureOnDevice(device, stream));

        auto gpu_kernel =
            llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
        ASSERT_NE(gpu_kernel, nullptr);
        gpu_kernel->setGPUStream(stream);
        auto requirements = device.is_cuda()
                                ? MoEWorkspaceBuffers::cudaMoE(
                                      kMaximumRows,
                                      kDModel,
                                      kIntermediate,
                                      kNumExperts,
                                      kTopK)
                                : MoEWorkspaceBuffers::rocmMoE(
                                      kMaximumRows,
                                      kDModel,
                                      kIntermediate,
                                      kNumExperts,
                                      kTopK);
        DeviceWorkspaceManager workspace(
            device,
            requirements.total_bytes_with_alignment() +
                4u * 1024u * 1024u);
        ASSERT_TRUE(workspace.allocate(requirements));
        auto *workspace_consumer =
            dynamic_cast<IWorkspaceConsumer *>(gpu_kernel.get());
        ASSERT_NE(workspace_consumer, nullptr);
        workspace_consumer->bindWorkspace(&workspace);

        for (std::size_t format_index = 0;
             format_index < formats.size();
             ++format_index)
        {
            const auto &format = formats[format_index];
            SCOPED_TRACE(
                std::string(backend_label) + " cross-tier format=" +
                format.label);

            std::vector<std::unique_ptr<TensorBase>> weights;
            std::vector<GpuPreparedGemm> prepared;
            weights.reserve(3u);
            prepared.reserve(3u);
            const std::uint64_t model_base =
                3900000u + static_cast<std::uint64_t>(format_index) * 16u +
                (device.is_cuda() ? 0u : 100000u);
            const std::string prefix =
                std::string("test.") + backend_label +
                ".cross_tier_arithmetic." + format.label;
            const DeviceNativeVNNIMatrixDesc gate_descriptor =
                prepareMatrixDescriptor(
                    format,
                    device,
                    stream,
                    kIntermediate,
                    kDModel,
                    930001u + static_cast<std::uint32_t>(format_index),
                    prefix + ".gate",
                    ModelContextId{model_base + 1u},
                    weights,
                    prepared);
            const DeviceNativeVNNIMatrixDesc up_descriptor =
                prepareMatrixDescriptor(
                    format,
                    device,
                    stream,
                    kIntermediate,
                    kDModel,
                    940001u + static_cast<std::uint32_t>(format_index),
                    prefix + ".up",
                    ModelContextId{model_base + 2u},
                    weights,
                    prepared);
            const DeviceNativeVNNIMatrixDesc down_descriptor =
                prepareMatrixDescriptor(
                    format,
                    device,
                    stream,
                    kDModel,
                    kIntermediate,
                    950001u + static_cast<std::uint32_t>(format_index),
                    prefix + ".down",
                    ModelContextId{model_base + 3u},
                    weights,
                    prepared);

            // Production demotes an expert from a prepared MoE bank, not an
            // unpublished GEMM descriptor. Table preparation installs the IQ
            // decoder resources on this endpoint before the source-ready
            // event can authorize a transfer. Do not smuggle a raw test-only
            // table initializer into the demotion lane or its arithmetic.
            const int original_gate_up_table = gpu_kernel->uploadGroupedExpertGateUpDescriptorTables(
                &gate_descriptor, &up_descriptor, kNumExperts, kDModel,
                kIntermediate);
            const int original_down_table = gpu_kernel->uploadGroupedExpertDownDescriptorTable(
                &down_descriptor, kNumExperts, kDModel, kIntermediate);
            ASSERT_GE(original_gate_up_table, 0);
            ASSERT_GE(original_down_table, 0);

            std::array<std::unique_ptr<CpuKernel>, 3> cpu_kernels;
            for (std::size_t projection = 0;
                 projection < cpu_kernels.size();
                 ++projection)
            {
                cpu_kernels[projection] = std::make_unique<CpuKernel>(
                    weights[projection].get(),
                    0,
                    -1,
                    CPUProjectionNumericalPolicy::GPUAlignedExpert);
                ASSERT_TRUE(cpu_kernels[projection]->isValid());
            }
            auto demoted = demoteGPUExpert(
                device, stream, format,
                {gate_descriptor, up_descriptor, down_descriptor});
            ASSERT_EQ(demoted.projections.size(), cpu_kernels.size());
            for (size_t projection = 0; projection < cpu_kernels.size(); ++projection)
            {
                const auto *arrival = dynamic_cast<const CpuKernel *>(
                    demoted.projections[projection].engine.get());
                ASSERT_NE(arrival, nullptr);
                const auto &expected = cpu_kernels[projection]->packedWeights();
                const auto &actual = arrival->packedWeights();
                ASSERT_EQ(actual.numerical_policy, expected.numerical_policy);
                ASSERT_EQ(actual.native_interleaved.size(), expected.native_interleaved.size());
                // Equivalent encodings may differ in signed-zero metadata or
                // compensated payload representation. Executed FP32 rows,
                // not an accidental packing identity, own the proof below.
            }

            auto promoted_gate = promoteCPUProjection(
                backend,
                device,
                cpu_kernels[0]->packedWeights(),
                ExpertTierWeightProjection::Gate,
                model_base + 11u);
            auto promoted_up = promoteCPUProjection(
                backend,
                device,
                cpu_kernels[1]->packedWeights(),
                ExpertTierWeightProjection::Up,
                model_base + 12u);
            auto promoted_down = promoteCPUProjection(
                backend,
                device,
                cpu_kernels[2]->packedWeights(),
                ExpertTierWeightProjection::Down,
                model_base + 13u);

            DeviceMoEExpertDescriptor expert{};
            expert.logical_expert_id = 0;
            expert.owner_participant = 0;
            expert.local_slot = 0;
            expert.flags = toMoEExpertFlags(
                DeviceMoEExpertFlags::Valid |
                DeviceMoEExpertFlags::Resident |
                DeviceMoEExpertFlags::LocalCompute);
            expert.gate = promoted_gate.descriptor;
            expert.up = promoted_up.descriptor;
            expert.down = promoted_down.descriptor;
            std::vector<DeviceMoEExpertDescriptor> experts = {expert};

            const int gate_up_table =
                gpu_kernel->uploadGroupedExpertGateUpDescriptorTables(
                    &promoted_gate.descriptor,
                    &promoted_up.descriptor,
                    kNumExperts,
                    kDModel,
                    kIntermediate);
            const int down_table =
                gpu_kernel->uploadGroupedExpertDownDescriptorTable(
                    &promoted_down.descriptor,
                    kNumExperts,
                    kDModel,
                    kIntermediate);
            ASSERT_GE(gate_up_table, 0);
            ASSERT_GE(down_table, 0);

            auto runtime = makeRuntimeTable(
                device,
                stream,
                experts,
                /*participant_id=*/0u,
                std::vector<std::uint8_t>(kNumExperts, 1u),
                std::vector<std::uint32_t>(kNumExperts, 0b01u),
                /*num_layers=*/1,
                kTopK,
                kMaximumRows);

            // Certify original resident GPU bytes as well as a CPU promotion:
            // equivalent transfers alone cannot prove execution arithmetic.
            auto original_expert = expert;
            original_expert.gate = gate_descriptor;
            original_expert.up = up_descriptor;
            original_expert.down = down_descriptor;
            auto original_runtime = makeRuntimeTable(
                device, stream, {original_expert}, 0u,
                std::vector<std::uint8_t>(kNumExperts, 1u),
                std::vector<std::uint32_t>(kNumExperts, 0b01u),
                1, kTopK, kMaximumRows);

            for (const auto &input : kInputs)
            {
                const int rows = input.rows;
                SCOPED_TRACE("rows=" + std::to_string(rows) +
                             " amplitude=" + std::to_string(input.amplitude) +
                             " pattern=" + std::to_string(static_cast<int>(input.pattern)));
                auto hidden = makeHidden(rows, kDModel, format_index, input.pattern);
                // Scale exact shared inputs before either endpoint quantizes
                // them. This adds no alternate numerical or transfer path.
                for (size_t i = 0; i < hidden->numel(); ++i)
                    hidden->mutable_data()[i] *= input.amplitude * input_scale;
                const int hidden_blocks_per_row =
                    kDModel / static_cast<int>(Q8_1Block::BLOCK_SIZE);
                const int activation_blocks_per_row =
                    kIntermediate /
                    static_cast<int>(Q8_1Block::BLOCK_SIZE);
                std::vector<Q8_1Block> hidden_q8(
                    static_cast<std::size_t>(rows) *
                    hidden_blocks_per_row);
                std::vector<float> cpu_gate(
                    static_cast<std::size_t>(rows) * kIntermediate);
                std::vector<float> cpu_up(cpu_gate.size());
                std::vector<Q8_1Block> cpu_activation_q8(
                    static_cast<std::size_t>(rows) *
                    activation_blocks_per_row);
                std::vector<float> cpu_output(
                    static_cast<std::size_t>(rows) * kDModel);
                std::vector<float> cpu_serial_gate(cpu_gate.size());
                std::vector<float> cpu_serial_up(cpu_up.size());
                std::vector<float> cpu_serial_activation(cpu_up.size());
                std::vector<Q8_1Block> cpu_serial_activation_q8(
                    cpu_activation_q8.size());
                std::vector<float> cpu_serial_output(cpu_output.size());

                cpu::native_vnni::quantize_activations_to_q8_1(
                    hidden->data(),
                    hidden_q8.data(),
                    rows,
                    kDModel,
                    hidden_blocks_per_row,
                    CPUProjectionNumericalPolicy::GPUAlignedExpert);
                const std::array<CpuDescriptor, 2> gate_up{{
                    {
                        .kernel = cpu_kernels[0].get(),
                        .input_q8 = hidden_q8.data(),
                        .output = cpu_gate.data(),
                        .rows = rows,
                        .n = kIntermediate,
                        .ldc = kIntermediate,
                    },
                    {
                        .kernel = cpu_kernels[1].get(),
                        .input_q8 = hidden_q8.data(),
                        .output = cpu_up.data(),
                        .rows = rows,
                        .n = kIntermediate,
                        .ldc = kIntermediate,
                    },
                }};
                const std::array<CpuDescriptor, 1> down{{{
                    .kernel = cpu_kernels[2].get(),
                    .input_q8 = cpu_activation_q8.data(),
                    .output = cpu_output.data(),
                    .rows = rows,
                    .n = kDModel,
                    .ldc = kDModel,
                }}};
                ASSERT_TRUE(
                    CpuKernel::
                        execute_moe_grouped_ffn_transaction_preq_decode_equivalent(
                            gate_up.data(),
                            static_cast<int>(gate_up.size()),
                            kDModel,
                            cpu_gate.data(),
                            cpu_up.data(),
                            cpu_activation_q8.data(),
                            rows,
                            kIntermediate,
                            activation_blocks_per_row,
                            down.data(),
                            static_cast<int>(down.size())));

                /*
                 * Keep an independent M=1 oracle beside the layer-global CPU
                 * transaction. This is diagnostic and test-only: it localizes
                 * a broken grouped schedule before a downstream GPU mismatch
                 * obscures which arithmetic boundary changed. Production never
                 * replays expert rows.
                 */
                for (int row = 0; row < rows; ++row)
                {
                    const auto hidden_offset =
                        static_cast<std::size_t>(row) * hidden_blocks_per_row;
                    const auto intermediate_offset =
                        static_cast<std::size_t>(row) * kIntermediate;
                    const auto activation_offset =
                        static_cast<std::size_t>(row) *
                        activation_blocks_per_row;
                    const auto output_offset =
                        static_cast<std::size_t>(row) * kDModel;
                    cpu::native_vnni::gemv_native_vnni_preq(
                        cpu_kernels[0]->packedWeights(),
                        hidden_q8.data() + hidden_offset,
                        cpu_serial_gate.data() + intermediate_offset);
                    cpu::native_vnni::gemv_native_vnni_preq(
                        cpu_kernels[1]->packedWeights(),
                        hidden_q8.data() + hidden_offset,
                        cpu_serial_up.data() + intermediate_offset);
                    primitives::compute_swiglu_gpu_aligned_expert_serial(
                        cpu_serial_gate.data() + intermediate_offset,
                        cpu_serial_up.data() + intermediate_offset,
                        cpu_serial_activation.data() + intermediate_offset,
                        kIntermediate);
                    cpu::native_vnni::quantize_activations_to_q8_1(
                        cpu_serial_activation.data() + intermediate_offset,
                        cpu_serial_activation_q8.data() + activation_offset,
                        /*M=*/1,
                        kIntermediate,
                        activation_blocks_per_row,
                        CPUProjectionNumericalPolicy::GPUAlignedExpert);
                    cpu::native_vnni::gemv_native_vnni_preq(
                        cpu_kernels[2]->packedWeights(),
                        cpu_serial_activation_q8.data() + activation_offset,
                        cpu_serial_output.data() + output_offset);
                }
                expectByteEqual(
                    std::string(backend_label) + " " + format.label +
                        " CPU grouped vs independent M1 gate M=" +
                        std::to_string(rows),
                    cpu_gate,
                    cpu_serial_gate,
                    kIntermediate);
                expectByteEqual(
                    std::string(backend_label) + " " + format.label +
                        " CPU grouped vs independent M1 up M=" +
                        std::to_string(rows),
                    cpu_up,
                    cpu_serial_up,
                    kIntermediate);
                expectByteEqual(
                    std::string(backend_label) + " " + format.label +
                        " CPU grouped vs independent M1 full expert M=" +
                        std::to_string(rows),
                    cpu_output,
                    cpu_serial_output,
                    kDModel);

                auto demoted_gate_up = gate_up;
                auto demoted_down = down;
                demoted_gate_up[0].kernel = dynamic_cast<CpuKernel *>(
                    demoted.projections[0].engine.get());
                demoted_gate_up[1].kernel = dynamic_cast<CpuKernel *>(
                    demoted.projections[1].engine.get());
                demoted_down[0].kernel = dynamic_cast<CpuKernel *>(
                    demoted.projections[2].engine.get());
                ASSERT_TRUE(CpuKernel::execute_moe_grouped_ffn_transaction_preq_decode_equivalent(
                    demoted_gate_up.data(), static_cast<int>(demoted_gate_up.size()),
                    kDModel, cpu_gate.data(), cpu_up.data(), cpu_activation_q8.data(),
                    rows, kIntermediate, activation_blocks_per_row,
                    demoted_down.data(), static_cast<int>(demoted_down.size())));
                expectByteEqual(
                    std::string(backend_label) + " " + format.label +
                        " GPU-demoted CPU gate vs initial CPU M=" + std::to_string(rows),
                    cpu_gate, cpu_serial_gate, kIntermediate);
                expectByteEqual(
                    std::string(backend_label) + " " + format.label +
                        " GPU-demoted CPU up vs initial CPU M=" + std::to_string(rows),
                    cpu_up, cpu_serial_up, kIntermediate);
                expectByteEqual(
                    std::string(backend_label) + " " + format.label +
                        " GPU-demoted CPU full expert vs initial CPU M=" + std::to_string(rows),
                    cpu_output, cpu_serial_output, kDModel);

                // The production CPU ticket path adds retained scratch, sparse
                // row addressing and route publication around those same GEMMs.
                // Exercise it with genuine demoted engines, not mock outputs.
                expectCPUExpertPacketParity(
                    {demoted.projections[0].engine.get(),
                     demoted.projections[1].engine.get(),
                     demoted.projections[2].engine.get()},
                    hidden->data(), cpu_serial_output.data(),
                    rows, kDModel, kIntermediate);

                if (rows == 1)
                {
                    /*
                     * The tables now contain the CPU-promoted descriptors, so
                     * these are genuine before/after-movement checkpoints.
                     * They localize projection drift before SwiGLU and the
                     * second activation quantization can obscure its origin.
                     */
                    auto gpu_gate = TestTensorFactory::createFP32(
                        {1u, static_cast<std::size_t>(kIntermediate)});
                    auto gpu_up = TestTensorFactory::createFP32(
                        {1u, static_cast<std::size_t>(kIntermediate)});
                    ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
                    ASSERT_TRUE(gpu_gate->ensureOnDevice(device, stream));
                    ASSERT_TRUE(gpu_up->ensureOnDevice(device, stream));
                    ITensor *gate_outputs[1] = {gpu_gate.get()};
                    ITensor *up_outputs[1] = {gpu_up.get()};
                    const int expert_id = 0;
                    ASSERT_TRUE(gpu_kernel->groupedExpertGateUpDecodeFromTable(
                        hidden.get(),
                        &expert_id,
                        gate_up_table,
                        /*num_active=*/1,
                        gate_outputs,
                        up_outputs,
                        kDModel,
                        kIntermediate));
                    ASSERT_TRUE(gpu_gate->ensureOnHost(stream));
                    ASSERT_TRUE(gpu_up->ensureOnHost(stream));
                    ASSERT_TRUE(backend->synchronizeStream(
                        stream, device.ordinal));
                    expectByteEqual(
                        std::string(backend_label) + " " + format.label +
                            " CPU vs promoted-GPU gate checkpoint",
                        std::vector<float>(
                            gpu_gate->data(),
                            gpu_gate->data() + gpu_gate->numel()),
                        cpu_serial_gate,
                        kIntermediate);
                    expectByteEqual(
                        std::string(backend_label) + " " + format.label +
                            " CPU vs promoted-GPU up checkpoint",
                        std::vector<float>(
                            gpu_up->data(),
                            gpu_up->data() + gpu_up->numel()),
                        cpu_serial_up,
                        kIntermediate);
                }

                auto routing_indices = TestTensorFactory::createFP32(
                    {static_cast<std::size_t>(rows), kTopK});
                auto routing_weights = TestTensorFactory::createFP32(
                    {static_cast<std::size_t>(rows), kTopK});
                std::fill_n(
                    routing_indices->mutable_data(), rows * kTopK, 0.0f);
                std::vector<float> cpu_preweighted_output(
                    cpu_serial_output.size());
                for (int row = 0; row < rows; ++row)
                {
                    /*
                     * A unit route weight cannot distinguish raw expert rows
                     * from the preweighted publication contract used by the
                     * heterogeneous return lane. Vary non-dyadic weights by
                     * row so this proof covers the exact placement-sensitive
                     * multiply that follows a CPU/GPU ownership change.
                     */
                    const float route_weight =
                        0.137f + 0.019f * static_cast<float>(row % 11);
                    routing_weights->mutable_data()[row] = route_weight;
                    const size_t row_offset =
                        static_cast<size_t>(row) * kDModel;
                    for (int column = 0; column < kDModel; ++column)
                    {
                        cpu_preweighted_output[row_offset + column] =
                            device_fp32_contract::multiply(
                                route_weight,
                                cpu_serial_output[row_offset + column]);
                    }
                }
                ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
                ASSERT_TRUE(routing_indices->ensureOnDevice(device, stream));
                ASSERT_TRUE(routing_weights->ensureOnDevice(device, stream));
                ASSERT_TRUE(gpu_kernel->publishCompleteGroupedPrefillPlanFromRouter(
                    runtime->deviceLayerState(0),
                    routing_indices.get(),
                    routing_weights.get(),
                    rows,
                    rows,
                    kNumExperts,
                    kTopK,
                    gate_up_table,
                    down_table,
                    /*filter_to_local_runtime_experts=*/true));
                std::vector<float> gpu_canonical;
                const auto gpu_output = executePublishedPlan(
                    backend,
                    *gpu_kernel,
                    *runtime,
                    device,
                    stream,
                    hidden.get(),
                    gate_up_table,
                    down_table,
                    rows,
                    kDModel,
                    kIntermediate,
                    kNumExperts,
                    kTopK,
                    /*layer_idx=*/0,
                    &gpu_canonical);
                expectByteEqual(
                    std::string(backend_label) + " " + format.label +
                        " CPU vs promoted-GPU canonical expert rows M=" +
                        std::to_string(rows),
                    gpu_canonical,
                    cpu_preweighted_output,
                    kDModel);
                expectByteEqual(
                    std::string(backend_label) + " " + format.label +
                        " CPU vs promoted-GPU reduced expert output M=" +
                        std::to_string(rows),
                    gpu_output,
                    cpu_preweighted_output,
                    kDModel);

                if (rows == 1)
                {
                    /*
                     * M=1 in the grouped-prefill API above is not the fused
                     * runtime-decode implementation used by the model graph.
                     * Capture the real router -> decode -> canonical fold DAG
                     * too. One expert makes the softmax weight exactly one,
                     * so the independent CPU raw row is its expected result.
                     */
                    /** @brief Independent graph contracts for router-owned and transported rows. */
                    enum class DecodeRouteSource { RuntimeRouter, ExplicitTensorPacket };
                    /** @brief Exact descriptor identity and route producer for one encoding. */
                    struct DecodeBank
                    {
                        MoERuntimeTable *runtime;
                        int gate_up_table;
                        int down_table;
                        const char *label;
                        DecodeRouteSource route_source;
                    };
                    for (const auto &bank : std::array<DecodeBank, 4>{{
                             {runtime.get(), gate_up_table, down_table, "CPU-promoted",
                              DecodeRouteSource::RuntimeRouter},
                             {original_runtime.get(), original_gate_up_table,
                              original_down_table, "original GPU", DecodeRouteSource::RuntimeRouter},
                             {runtime.get(), gate_up_table, down_table, "CPU-promoted explicit packet",
                              DecodeRouteSource::ExplicitTensorPacket},
                             {original_runtime.get(), original_gate_up_table,
                              original_down_table, "original GPU explicit packet",
                              DecodeRouteSource::ExplicitTensorPacket}}})
                    {
                        SCOPED_TRACE(bank.label);
                        const bool explicit_packet = bank.route_source ==
                            DecodeRouteSource::ExplicitTensorPacket;
                        auto decode_output = TestTensorFactory::createFP32(
                            {1u, kDModel});
                        auto decode_routes = TestTensorFactory::createFP32(
                            {1u, kTopK, kDModel});
                        ASSERT_TRUE(decode_output->ensureOnDevice(device, stream));
                        ASSERT_TRUE(decode_routes->ensureOnDevice(device, stream));
                        ASSERT_TRUE(gpu_kernel->prepareRouteLaunch(router.get(), {
                            .kind = MoERouteLaunchKind::RuntimeDecode,
                            .physical_rows = 1,
                            .d_model = kDModel,
                            .num_experts = kNumExperts,
                            .top_k = kTopK,
                        }));
                        ASSERT_TRUE(gpu_kernel->prepareGroupedRuntimeDecodeLaunchState(
                            bank.gate_up_table, bank.down_table, kTopK, kDModel, kIntermediate,
                            MoEDecodeDescriptorSource::RuntimePlacementTable));
                        // Join setup publications before recording. The frozen
                        // graph owns every subsequent write on this same stream.
                        for (ITensor *tensor : std::array<ITensor *, 6>{
                                 hidden.get(), router.get(), decode_routes.get(),
                                 decode_output.get(), routing_indices.get(), routing_weights.get()})
                            TransferEngine::requireDeviceInput(tensor, device, stream);

                        auto &context = GPUDeviceContextPool::instance().getContext(device);
                        auto graph = context.createGraphCapture(stream);
                        ASSERT_NE(graph, nullptr);
                        ScopedBackendGraphCapture capture(
                            context, *graph, "cross-tier captured runtime decode");
                        ASSERT_TRUE(capture.begin());
                        const bool routed = explicit_packet || gpu_kernel->decodeRouteSelect(
                            bank.runtime->deviceLayerState(0), hidden.get(), router.get(),
                            kDModel, kNumExperts, kTopK,
                            /*normalize_topk=*/true, nullptr, nullptr,
                            /*write_legacy_outputs=*/false,
                            /*update_runtime_histogram=*/false,
                            /*absolute_position_ids_device=*/nullptr,
                            RoutedExpertRowExecutionPolicy::ParticipantAssigned);
                        // The colocated GPU+CPU model uses explicit tensor
                        // routes. That entry point resolves placement and
                        // quantizes hidden rows itself, unlike router-Q8 reuse.
                        const bool computed = explicit_packet
                            ? gpu_kernel->groupedExpertDecodeFromRouting(
                                hidden.get(), routing_indices.get(), routing_weights.get(),
                                bank.gate_up_table, bank.down_table, kTopK, decode_output.get(),
                                kDModel, kIntermediate, nullptr, decode_routes.get(),
                                bank.runtime->deviceLayerState(0),
                                MoEDecodeDescriptorSource::RuntimePlacementTable)
                            : gpu_kernel->groupedExpertDecodeFromRuntime(
                            bank.runtime->deviceLayerState(0), hidden.get(),
                            bank.gate_up_table, bank.down_table, kTopK, decode_output.get(),
                            kDModel, kIntermediate,
                            MoEDecodeDescriptorSource::RuntimePlacementTable,
                            decode_routes.get());
                        const bool reduced = gpu_kernel->reduceCanonicalRouteContributions(
                            decode_routes.get(), decode_output.get(), 1, kTopK, kDModel);
                        // RAII also closes an exceptional/abandoned recording.
                        capture.finish();
                        ASSERT_TRUE(routed);
                        ASSERT_TRUE(computed);
                        ASSERT_TRUE(reduced);
                        ASSERT_TRUE(graph->instantiate());
                        ASSERT_TRUE(graph->launch());
                        TransferEngine::publishDeviceWrite(
                            decode_output.get(), device, stream);
                        ASSERT_TRUE(decode_output->ensureOnHost(stream));
                        ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                        expectByteEqual(
                            std::string(backend_label) + " " + format.label +
                                " CPU vs captured " + bank.label + " runtime decode",
                            std::vector<float>(decode_output->data(),
                                               decode_output->data() + kDModel),
                            explicit_packet ? cpu_preweighted_output : cpu_serial_output, kDModel);
                    }
                }
            }
        }

        ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
        workspace_consumer->unbindWorkspace();
    }

    /** Functional copy certification and the broader grouped-math sweep. */
    enum class NativeExpertTransferProofScope
    {
        CopyPublication, ///< Every codebook's bytes, descriptors and epoch lease.
        GroupedArithmetic, ///< Also compare all grouped outputs with StaticOwner.
    };

    /**
     * @brief Prove StaticOwner and transferred CurrentBatchLLEP grouped equality.
     *
     * @param backend_label Stable diagnostic label ("CUDA" or "ROCm").
     * @param device Physical GPU used to execute both simulated participants.
     * @param stream Explicit non-null backend stream owning every test operation.
     * @param scope Copy/publication preflight or the complete grouped math sweep.
     *
     * @throws std::runtime_error for setup or launch failures. GoogleTest catches
     *         the exception at the backend test boundary and reports the active
     *         format/M trace.
     */
    void runNativeVNNIExpertTransferGroupedParity(
        const char *backend_label,
        DeviceId device,
        void *stream,
        NativeExpertTransferProofScope scope = NativeExpertTransferProofScope::GroupedArithmetic);

    namespace native_vnni_transfer_parity_detail
    {
        /**
         * @brief Stream one CPU-native matrix into executable GPU form.
         *
         * @param backend Exact destination backend.
         * @param device CUDA or ROCm endpoint that owns the inactive allocation.
         * @param cpu_weights Real CPU prepared bytes retaining source provenance.
         * @param projection Stable gate/up/down role used in the transfer manifest.
         * @param identity Unique diagnostic identity for the transaction.
         * @return Stable destination descriptor and its device allocation owners.
         * @throws std::runtime_error when materialization, streaming, or format
         *         publication fails.
         */
        inline PromotedCPUProjection promoteCPUProjection(
            IBackend *backend,
            DeviceId device,
            const cpu::native_vnni::CPUNativeVNNIPackedWeights &cpu_weights,
            ExpertTierWeightProjection projection,
            uint64_t identity)
        {
            if (!backend || !device.is_gpu())
            {
                throw std::invalid_argument(
                    "Grouped promotion parity requires a GPU destination");
            }

            const auto manifest = makeCpuToGpuExpertTierWeightStreamManifest(
                cpu_weights,
                identity,
                /*layer_id=*/0,
                /*expert_id=*/0,
                projection,
                /*maximum_units_per_chunk=*/1);
            const auto layout = manifest.deviceLayout();
            if (!layout.valid())
            {
                throw std::runtime_error(
                    "CPU promotion did not produce a valid device layout");
            }

            const size_t blocks =
                static_cast<size_t>(layout.N) *
                static_cast<size_t>(layout.blocks_per_row);
            PromotedCPUProjection result;
            result.payload = std::make_unique<DeviceAllocation>(
                backend,
                device.ordinal,
                blocks * layout.gpu_payload_bytes_per_block);
            result.scales = std::make_unique<DeviceAllocation>(
                backend,
                device.ordinal,
                blocks * sizeof(uint16_t));
            if (layout.gpu_is_asymmetric != 0u)
            {
                result.mins = std::make_unique<DeviceAllocation>(
                    backend,
                    device.ordinal,
                    blocks * sizeof(uint16_t));
            }
            if (layout.gpu_has_emins != 0u)
            {
                result.emins = std::make_unique<DeviceAllocation>(
                    backend,
                    device.ordinal,
                    blocks * sizeof(uint32_t));
            }

            ExpertTierGpuMutableProjectionView destination{
                .payload = result.payload->as<uint8_t>(),
                .scales = result.scales->as<uint16_t>(),
                .mins = result.mins
                            ? result.mins->as<uint16_t>()
                            : nullptr,
                .emins = result.emins
                             ? result.emins->as<uint32_t>()
                             : nullptr,
                .payload_bytes = result.payload->bytes(),
                .scales_bytes = result.scales->bytes(),
                .mins_bytes = result.mins ? result.mins->bytes() : 0u,
                .emins_bytes = result.emins ? result.emins->bytes() : 0u,
            };
            if (!destination.validFor(layout))
            {
                throw std::runtime_error(
                    "CPU promotion allocation does not satisfy its layout");
            }

            std::string error;
            ExpertTierWeightTransferLane lane({
                .device = device,
                .staging = TransferEngine::instance()
                               .allocatePersistentTransferStagingSlices(
                                   layout.chunkBytes(1),
                                   1u,
                                   device)
                               .front(),
                .execution = TransferEngine::instance()
                                 .allocatePersistentTransferExecutionLanes(
                                     1u,
                                     device,
                                     "grouped_cpu_promotion_" +
                                         std::to_string(identity))
                                 .front(),
                .progress = BackgroundTransferProgressBinding::nativeStream(),
                .lane_name = "grouped_cpu_promotion_" +
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
                    "Failed to start grouped CPU promotion: " + error);
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
                    "Grouped CPU promotion did not complete: " + error);
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
                    "Grouped CPU promotion violated the async lane contract");
            }

            result.descriptor = DeviceNativeVNNIMatrixDesc{
                .payload = result.payload->as<uint8_t>(),
                .scales = result.scales->get(),
                .mins = result.mins ? result.mins->get() : nullptr,
                .emins = result.emins ? result.emins->get() : nullptr,
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

        /**
         * @brief Minimal non-owning boundary required by host publication setup.
         *
         * The focused publisher regression has no concurrent inference graph
         * while maintenance prepares the candidate. The publisher nevertheless
         * requires the same typed model binding as production; this object makes
         * that absence explicit without introducing a stream synchronization or
         * pretending to own device execution state.
         */
        class QuiescentOverlayInferenceBoundary final
            : public IMoEOverlayDeviceInferenceBoundary,
              public IMoEOverlayDeviceInitialRuntimePublisher
        {
        public:
            /** The focused fixture has already published its sole runtime layer. */
            bool publishMoEOverlayDeviceInitialRuntime(
                void *controller_stream) override
            {
                if (!controller_stream)
                    return false;
                initial_runtime_published_.store(
                    true, std::memory_order_release);
                return true;
            }

            /** @return Whether setup invoked the typed initial publication edge. */
            [[nodiscard]] bool initialRuntimePublished() const noexcept
            {
                return initial_runtime_published_.load(
                    std::memory_order_acquire);
            }

            MoEOverlayInferenceBoundaryStatus
            enqueueMoEOverlayDeviceInferenceBoundary(
                void *maintenance_stream,
                MoEOverlayInferenceBoundaryRequest) override
            {
                return maintenance_stream
                           ? MoEOverlayInferenceBoundaryStatus::Submitted
                           : MoEOverlayInferenceBoundaryStatus::Failed;
            }

            bool installMoEOverlayTransferProgressEpoch(
                std::shared_ptr<MappedTransferProgressEpoch> epoch) override
            {
                return epoch != nullptr;
            }

        private:
            std::atomic<bool> initial_runtime_published_{false};
        };

        /** @brief Build one two-tier, one-participant-per-domain placement. */
        inline MoERoutedExpertPlacementPlan hostPublicationPlan(
            DeviceId device,
            bool candidate)
        {
            RoutedExpertDomain priority_zero_domain;
            priority_zero_domain.name = "priority_0_domain";
            priority_zero_domain.scope = ExecutionDomainScope::SINGLE;
            priority_zero_domain.backend = CollectiveBackendType::HOST;
            priority_zero_domain.participants = {
                device.is_cuda()
                    ? GlobalDeviceAddress::cuda(device.cuda_ordinal(), 0)
                    : GlobalDeviceAddress::rocm(device.rocm_ordinal(), 0)};
            priority_zero_domain.world_ranks = {0};
            priority_zero_domain.owner_rank = 0;
            priority_zero_domain.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;

            RoutedExpertDomain priority_one_domain;
            priority_one_domain.name = "priority_1_domain";
            priority_one_domain.scope = ExecutionDomainScope::SINGLE;
            priority_one_domain.backend = CollectiveBackendType::HOST;
            priority_one_domain.participants = {GlobalDeviceAddress::cpu(0)};
            priority_one_domain.world_ranks = {0};
            priority_one_domain.owner_rank = 0;
            priority_one_domain.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;

            RoutedExpertTier priority_zero;
            priority_zero.name = "priority_0";
            priority_zero.domain = priority_zero_domain.name;
            priority_zero.priority = 0;
            priority_zero.max_experts_per_layer = 1;

            RoutedExpertTier priority_one;
            priority_one.name = "priority_1";
            priority_one.domain = priority_one_domain.name;
            priority_one.priority = 1;
            priority_one.max_experts_per_layer = 1;
            priority_one.fallback = true;

            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = priority_zero_domain.name;
            plan.shared_expert_domain = priority_zero_domain.name;
            plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.owner_order = RoutedExpertOwnerOrder::Ordinal;
            plan.domains = {
                std::move(priority_zero_domain),
                std::move(priority_one_domain)};
            plan.routed_tiers = {
                std::move(priority_zero),
                std::move(priority_one)};
            plan.placements = {{
                .layer = 0,
                .routed_expert_tier = candidate
                                          ? std::vector<int>{1, 0}
                                          : std::vector<int>{0, 1},
            }};
            return plan;
        }

        /** @brief Materialize one immutable snapshot from its exact plan. */
        inline std::shared_ptr<const MoEOverlayResidencySnapshot>
        hostPublicationSnapshot(
            std::uint64_t epoch,
            MoERoutedExpertPlacementPlan plan)
        {
            auto retained_plan =
                std::make_shared<MoERoutedExpertPlacementPlan>(
                    std::move(plan));
            MoEExpertOwnerMap owner_map = MoEExpertOwnerMap::build(
                *retained_plan);
            auto snapshot = std::make_shared<MoEOverlayResidencySnapshot>();
            snapshot->epoch = epoch;
            snapshot->placement_plan = std::move(retained_plan);
            snapshot->owner_map = owner_map;
            snapshot->layered_ownership = owner_map.layeredOwnership(1, 2);
            if (!snapshot->valid())
                throw std::runtime_error(
                    "Host publication test produced an invalid snapshot");
            return snapshot;
        }

        /** @brief Create a prepared engine alias over promoted GPU storage. */
        inline std::shared_ptr<ITensorGemm> promotedEngine(
            DeviceId device,
            const DeviceNativeVNNIMatrixDesc &descriptor,
            const std::shared_ptr<void> &lifetime)
        {
            const NativeVnniSourceIdentity source_identity{
                .codebook_id = descriptor.source_codebook_id,
                .is_superblock = descriptor.source_is_superblock != 0u,
                .present = descriptor.source_identity_present != 0u,
            };
            const NativeVnniReusableDeviceAllocationFormat allocation{
                .payload_bytes_per_block =
                    descriptor.allocation_payload_bytes_per_block,
                .has_mins = descriptor.allocation_has_mins != 0u,
                .has_emins = descriptor.allocation_has_emins != 0u,
            };
#ifdef HAVE_CUDA
            if (device.is_cuda())
            {
                // The direct GEMM constructors predate the immutable device
                // descriptor ABI and still spell their borrowed weight views
                // as mutable pointers. They never write through these aliases;
                // retain constness everywhere except this legacy constructor
                // boundary so the publisher regression consumes the exact ABI
                // used by grouped execution.
                return std::make_shared<cuda::CUDAQuantisedGemmKernel>(
                    descriptor.n,
                    descriptor.k,
                    device.cuda_ordinal(),
                    const_cast<std::uint8_t *>(descriptor.payload),
                    const_cast<std::uint16_t *>(
                        static_cast<const std::uint16_t *>(descriptor.scales)),
                    const_cast<std::uint16_t *>(
                        static_cast<const std::uint16_t *>(descriptor.mins)),
                    const_cast<std::uint32_t *>(
                        static_cast<const std::uint32_t *>(descriptor.emins)),
                    descriptor.codebook_id,
                    descriptor.blocks_per_row,
                    lifetime,
                    source_identity,
                    allocation);
            }
#endif
#ifdef HAVE_ROCM
            if (device.is_rocm())
            {
                return std::make_shared<rocm::ROCmQuantisedGemmKernel>(
                    descriptor.n,
                    descriptor.k,
                    device.rocm_ordinal(),
                    const_cast<std::uint8_t *>(descriptor.payload),
                    const_cast<void *>(descriptor.scales),
                    const_cast<void *>(descriptor.mins),
                    const_cast<void *>(descriptor.emins),
                    descriptor.codebook_id,
                    descriptor.blocks_per_row,
                    lifetime,
                    source_identity,
                    allocation);
            }
#endif
            throw std::runtime_error(
                "Host publication test backend was not built");
        }

        /** @brief Poll one typed inactive-bank phase to its terminal. */
        template <typename Poll>
        inline MoEOverlayResidencyWaveProgress awaitPublicationPhase(
            Poll &&poll,
            std::string *error)
        {
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(10);
            auto progress = MoEOverlayResidencyWaveProgress::Pending;
            while (progress == MoEOverlayResidencyWaveProgress::Pending &&
                   std::chrono::steady_clock::now() < deadline)
            {
                progress = poll(error);
                if (progress == MoEOverlayResidencyWaveProgress::Pending)
                    std::this_thread::yield();
            }
            return progress;
        }
    } // namespace native_vnni_transfer_parity_detail

    /**
     * @brief Prove promoted codebook-23 weights through grouped MoE execution.
     *
         * Three real Q5_K projections are prepared twice: the ordinary compact GPU
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
        constexpr int kIntermediate = 512;
        constexpr int kNumExperts = 2;
        constexpr int kTopK = 2;
        constexpr int kMaxRows = 65;
        constexpr int kLayer = 0;
        const auto &format = quantizedVerifierFormat("Q5_K");

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
        auto promoted_gate = promoteCPUProjection(
            backend,
            device,
            cpu_weights[0],
            ExpertTierWeightProjection::Gate,
            model_base + 11u);
        auto promoted_up = promoteCPUProjection(
            backend,
            device,
            cpu_weights[1],
            ExpertTierWeightProjection::Up,
            model_base + 12u);
        auto promoted_down = promoteCPUProjection(
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

        /*
         * Close the last production-only gap: a heterogeneous host authority
         * does not call flipActiveBank() or upload a table assembled by this
         * test. It installs a complete immutable participant bank, exports the
         * retained engines into the canonical runtime table, DMA-publishes only
         * the inactive placement bank plus its selector, advances the device RCU
         * epoch, and lets inference acquire that epoch ticket. The descriptor
         * tables start with expert one empty, exactly as a graph captured before
         * promotion would. Runtime publication must therefore be the sole reason
         * the promoted expert becomes executable.
         */
        {
            constexpr int rows = 16;
            auto previous_snapshot = hostPublicationSnapshot(
                1u, hostPublicationPlan(device, /*candidate=*/false));
            auto candidate_snapshot = hostPublicationSnapshot(
                2u, hostPublicationPlan(device, /*candidate=*/true));
            const auto *gpu_participant =
                previous_snapshot->owner_map.participantForId(0);
            ASSERT_NE(gpu_participant, nullptr);
            ASSERT_EQ(gpu_participant->device, device);

            auto registry = std::make_shared<
                MoEOverlayParticipantResidencyRegistry>(
                MoEOverlayParticipantResidencyRegistry::Config{
                    .owner_map = previous_snapshot->owner_map,
                    .local_participant_ids = {0},
                    .num_layers = 1,
                    .num_experts = kNumExperts,
                    .initial_epoch = 1u,
                    .retained_epoch_capacity = 2u,
                });
            const auto borrowed = [](ITensorGemm *engine)
            {
                return std::shared_ptr<ITensorGemm>(
                    engine, [](ITensorGemm *) {});
            };
            MoEOverlayPreparedExpertTriplet compact_triplet{
                .gate = borrowed(prepared[0].kernel),
                .up = borrowed(prepared[1].kernel),
                .down = borrowed(prepared[2].kernel),
            };
            std::vector<MoEOverlayPreparedExpertTriplet> initial_triplets(
                kNumExperts);
            initial_triplets[0] = compact_triplet;
            std::string publication_error;
            ASSERT_TRUE(registry->registerInitialLayer(
                /*participant_id=*/0,
                /*layer_idx=*/0,
                std::vector<bool>{true, false},
                initial_triplets,
                &publication_error))
                << publication_error;
            ASSERT_TRUE(registry->allInitialBanksReady());

            auto promoted_lifetime = std::make_shared<int>(7319);
            MoEOverlayPreparedExpertTriplet promoted_triplet{
                .gate = promotedEngine(
                    device, promoted_gate.descriptor, promoted_lifetime),
                .up = promotedEngine(
                    device, promoted_up.descriptor, promoted_lifetime),
                .down = promotedEngine(
                    device, promoted_down.descriptor, promoted_lifetime),
            };
            auto endpoint = registry->endpoint(0);
            ASSERT_NE(endpoint, nullptr);
            auto candidate_bank = endpoint->cloneCandidate(1u, 2u);
            candidate_bank.layers[0].clearExpert(0);
            candidate_bank.layers[0].setResidentExpert(
                1, promoted_triplet);
            auto prepared_candidate = endpoint->prepareReadyBank(
                std::move(candidate_bank), &publication_error);
            ASSERT_TRUE(prepared_candidate.has_value()) << publication_error;
            ASSERT_EQ(
                endpoint->installReadyBank(
                    std::move(*prepared_candidate), &publication_error),
                MoEOverlayParticipantBankInstallStatus::Installed)
                << publication_error;

            auto epoch_arena = std::make_shared<DeviceMoEOverlayEpochArena>(
                DeviceMoEOverlayEpochArena::Config{
                    .device_id = device,
                    .initial_epoch = 1u,
                    // A freshly initialized runtime table prepares bank one.
                    .initial_bank = 1u,
                    .request_slot_capacity = 1u,
                });
            auto initial_runtime_descriptors = compact_experts;
            initial_runtime_descriptors[0].owner_participant = 0;
            initial_runtime_descriptors[1].owner_participant = 1;
            auto published_runtime = makeRuntimeTable(
                device,
                stream,
                initial_runtime_descriptors,
                /*participant_id=*/0u,
                std::vector<std::uint8_t>{1u, 0u},
                std::vector<std::uint32_t>{0b01u, 0b10u},
                /*num_layers=*/1,
                kTopK,
                kMaxRows,
                epoch_arena);
            ASSERT_EQ(
                published_runtime->hostLayerState(0).active_bank,
                1u);

            QuiescentOverlayInferenceBoundary inference_boundary;
            MoEOverlayDeviceControllerRuntimeBinding runtime_binding{
                .device = device,
                .runtime_layers_device =
                    published_runtime->deviceLayerState(0),
                .runtime_table_host = published_runtime.get(),
                .overlay_participant_id = 0,
                .domain_participant_id = 0u,
                .domain_participant_count = 1u,
                .layer_count = 1u,
                .expert_count = kNumExperts,
                .top_k = kTopK,
                .epoch_control = epoch_arena->control(),
                .maintenance_epoch = epoch_arena->maintenanceEpoch(),
                .maintenance_status = epoch_arena->maintenanceStatus(),
                .inference_boundary = &inference_boundary,
                .initial_runtime_publisher = &inference_boundary,
            };
            ASSERT_TRUE(runtime_binding.hostPublicationValid());
            auto publisher = std::make_shared<
                MoEOverlayHostAuthorityDeviceBankPublisher>(
                MoEOverlayHostAuthorityDeviceBankPublisher::Config{
                    .runtime_bindings = {runtime_binding},
                    .registry = registry,
                    .perf_device = device.to_string(),
                });
            ASSERT_TRUE(inference_boundary.initialRuntimePublished());
            auto transaction = publisher->createTransaction(
                previous_snapshot,
                candidate_snapshot,
                &publication_error);
            ASSERT_NE(transaction, nullptr) << publication_error;
            ASSERT_TRUE(transaction->beginPrepare(&publication_error))
                << publication_error;
            ASSERT_EQ(
                awaitPublicationPhase(
                    [&](std::string *error)
                    { return transaction->pollPrepare(error); },
                    &publication_error),
                MoEOverlayResidencyWaveProgress::Ready)
                << publication_error;

            /*
             * Preparation may install the immutable candidate bank, but it
             * must not change the live runtime selector.  Inference holding an
             * epoch-one ticket can overlap this observation and must continue
             * to see bank one until the explicit publication transition.
             */
            DeviceMoELayerRuntime prepared_runtime{};
            EXPECT_TRUE(backend->deviceToHostOnStream(
                &prepared_runtime,
                published_runtime->deviceLayerState(0),
                sizeof(prepared_runtime),
                device.ordinal,
                stream));
            EXPECT_TRUE(backend->synchronizeStream(stream, device.ordinal));
            EXPECT_EQ(prepared_runtime.active_bank, 1u);
            EXPECT_EQ(prepared_runtime.active_epoch, 1u);
            EXPECT_EQ(prepared_runtime.banks[0].epoch, 2u);
            EXPECT_TRUE(prepared_runtime.banks[0].experts[1].weightsReady());

            ASSERT_TRUE(transaction->beginPublication(&publication_error))
                << publication_error;
            ASSERT_EQ(
                awaitPublicationPhase(
                    [&](std::string *error)
                    { return transaction->pollPublication(error); },
                    &publication_error),
                MoEOverlayResidencyWaveProgress::Ready)
                << publication_error;

            const MoEKernelLaunchContext launch{.stream = stream};
            ASSERT_TRUE(kernel.acquireMoEOverlayEpoch(
                launch,
                epoch_arena->control(),
                epoch_arena->requestTicket(0u),
                epoch_arena->requestStatus(0u)));
            ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));

            /*
             * Observe the publication certificate before routing. These are
             * diagnostic-only D2H copies in an integration test; production
             * inference consumes the same records in place on device. Keeping
             * the assertions here makes the regression identify whether a
             * future break is in epoch admission or descriptor execution.
             */
            DeviceMoELayerRuntime observed_runtime{};
            DeviceMoEOverlayEpochTicket observed_ticket{};
            EXPECT_TRUE(backend->deviceToHostOnStream(
                &observed_runtime,
                published_runtime->deviceLayerState(0),
                sizeof(observed_runtime),
                device.ordinal,
                stream));
            EXPECT_TRUE(backend->deviceToHostOnStream(
                &observed_ticket,
                epoch_arena->requestTicket(0u),
                sizeof(observed_ticket),
                device.ordinal,
                stream));
            EXPECT_TRUE(backend->synchronizeStream(stream, device.ordinal));
            EXPECT_EQ(observed_runtime.active_bank, 0u);
            EXPECT_EQ(observed_runtime.active_epoch, 2u);
            EXPECT_EQ(observed_runtime.banks[0].epoch, 2u);
            EXPECT_EQ(observed_runtime.banks[0].expert_count,
                      static_cast<std::uint32_t>(kNumExperts));
            EXPECT_EQ(observed_runtime.banks[0].local_compute_mask[1], 1u);
            EXPECT_EQ(observed_runtime.banks[0].resident_participant_mask[1],
                      0b01u);
            EXPECT_TRUE(observed_runtime.banks[0].experts[1].weightsReady());
            EXPECT_EQ(observed_runtime.banks[0].experts[1].gate.payload,
                      promoted_gate.descriptor.payload);
            EXPECT_EQ(observed_ticket.epoch, 2u);
            EXPECT_EQ(observed_ticket.selector & 1u, 0u);
            EXPECT_GT(observed_ticket.selector >> 1u, 0u);

            std::array<DeviceNativeVNNIMatrixDesc, kNumExperts>
                sparse_bootstrap_gates{compact_gate, {}};
            std::array<DeviceNativeVNNIMatrixDesc, kNumExperts>
                sparse_bootstrap_ups{compact_up, {}};
            std::array<DeviceNativeVNNIMatrixDesc, kNumExperts>
                sparse_bootstrap_downs{compact_down, {}};
            const int published_gateup_table =
                kernel.uploadGroupedExpertGateUpDescriptorTables(
                    sparse_bootstrap_gates.data(),
                    sparse_bootstrap_ups.data(),
                    kNumExperts,
                    kDModel,
                    kIntermediate,
                    MoEDecodeDescriptorSource::RuntimePlacementTable);
            const int published_down_table =
                kernel.uploadGroupedExpertDownDescriptorTable(
                    sparse_bootstrap_downs.data(),
                    kNumExperts,
                    kDModel,
                    kIntermediate,
                    MoEDecodeDescriptorSource::RuntimePlacementTable);
            ASSERT_GE(published_gateup_table, 0);
            ASSERT_GE(published_down_table, 0);

            auto reference_descriptors = compact_experts;
            reference_descriptors[0].owner_participant = -1;
            reference_descriptors[1].owner_participant = 0;
            auto reference_runtime = makeRuntimeTable(
                device,
                stream,
                reference_descriptors,
                /*participant_id=*/0u,
                std::vector<std::uint8_t>{0u, 1u},
                std::vector<std::uint32_t>{0u, 0b01u},
                /*num_layers=*/1,
                kTopK,
                kMaxRows);

            auto hidden = makeHidden(rows, kDModel, 93u);
            auto routing_indices = TestTensorFactory::createFP32(
                {static_cast<std::size_t>(rows), kTopK});
            auto routing_weights = TestTensorFactory::createFP32(
                {static_cast<std::size_t>(rows), kTopK});
            for (int row = 0; row < rows; ++row)
            {
                routing_indices->mutable_data()[row * kTopK] = 1.0f;
                routing_indices->mutable_data()[row * kTopK + 1] = 0.0f;
                routing_weights->mutable_data()[row * kTopK] = 0.625f;
                routing_weights->mutable_data()[row * kTopK + 1] = 0.375f;
            }
            ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
            ASSERT_TRUE(routing_indices->ensureOnDevice(device, stream));
            ASSERT_TRUE(routing_weights->ensureOnDevice(device, stream));

            ASSERT_TRUE(kernel.publishCompleteGroupedPrefillPlanFromRouter(
                reference_runtime->deviceLayerState(0),
                routing_indices.get(),
                routing_weights.get(),
                rows,
                rows,
                kNumExperts,
                kTopK,
                compact_gateup_table,
                compact_down_table,
                /*filter_to_local_runtime_experts=*/true));
            std::vector<float> reference_canonical;
            const auto reference_output = executePublishedPlan(
                backend,
                kernel,
                *reference_runtime,
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
                /*layer_idx=*/0,
                &reference_canonical);

            ASSERT_TRUE(kernel.publishCompleteGroupedPrefillPlanFromRouter(
                published_runtime->deviceLayerState(0),
                routing_indices.get(),
                routing_weights.get(),
                rows,
                rows,
                kNumExperts,
                kTopK,
                published_gateup_table,
                published_down_table,
                /*filter_to_local_runtime_experts=*/true));

            std::array<std::int32_t, kNumExperts> observed_counts{};
            std::array<std::int32_t, rows * kTopK> observed_route_experts{};
            std::array<std::int32_t, rows * kTopK>
                observed_route_participants{};
            std::array<float, rows * kTopK> observed_route_weights{};
            EXPECT_TRUE(backend->deviceToHostOnStream(
                observed_counts.data(),
                observed_runtime.expert_counts,
                sizeof(observed_counts),
                device.ordinal,
                stream));
            EXPECT_TRUE(backend->deviceToHostOnStream(
                observed_route_experts.data(),
                observed_runtime.route_expert_ids,
                sizeof(observed_route_experts),
                device.ordinal,
                stream));
            EXPECT_TRUE(backend->deviceToHostOnStream(
                observed_route_participants.data(),
                observed_runtime.route_participant_ids,
                sizeof(observed_route_participants),
                device.ordinal,
                stream));
            EXPECT_TRUE(backend->deviceToHostOnStream(
                observed_route_weights.data(),
                observed_runtime.route_weights,
                sizeof(observed_route_weights),
                device.ordinal,
                stream));
            EXPECT_TRUE(backend->synchronizeStream(stream, device.ordinal));
            EXPECT_EQ(observed_counts[0], 0);
            EXPECT_EQ(observed_counts[1], rows);
            for (int row = 0; row < rows; ++row)
            {
                const auto local_slot =
                    static_cast<std::size_t>(row * kTopK);
                const auto remote_slot = local_slot + 1u;
                EXPECT_EQ(observed_route_experts[local_slot], 1);
                EXPECT_EQ(observed_route_participants[local_slot], 0);
                EXPECT_FLOAT_EQ(observed_route_weights[local_slot], 0.625f);
                EXPECT_EQ(observed_route_experts[remote_slot], 0);
                EXPECT_EQ(observed_route_participants[remote_slot], -1);
                EXPECT_FLOAT_EQ(observed_route_weights[remote_slot], 0.0f);
            }
            std::vector<float> published_canonical;
            const auto published_output = executePublishedPlan(
                backend,
                kernel,
                *published_runtime,
                device,
                stream,
                hidden.get(),
                published_gateup_table,
                published_down_table,
                rows,
                kDModel,
                kIntermediate,
                kNumExperts,
                kTopK,
                /*layer_idx=*/0,
                &published_canonical);
            expectByteEqual(
                std::string(backend_label) +
                    " host-published promoted output",
                published_output,
                reference_output,
                kDModel);
            expectByteEqual(
                std::string(backend_label) +
                    " host-published promoted canonical",
                published_canonical,
                reference_canonical,
                kDModel);

            ASSERT_TRUE(kernel.releaseMoEOverlayEpoch(
                launch,
                epoch_arena->control(),
                epoch_arena->requestTicket(0u),
                epoch_arena->requestStatus(0u)));
            ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
            ASSERT_EQ(
                awaitPublicationPhase(
                    [&](std::string *error)
                    { return transaction->pollRetirementFence(error); },
                    &publication_error),
                MoEOverlayResidencyWaveProgress::Ready)
                << publication_error;
            transaction->retirePrevious();
            transaction.reset();
        }

        ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
        workspace_consumer->unbindWorkspace();
    }
} // namespace llaminar2::test
