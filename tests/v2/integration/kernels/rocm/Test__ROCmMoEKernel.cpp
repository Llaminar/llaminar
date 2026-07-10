/**
 * @file Test__ROCmMoEKernel.cpp
 * @brief Integration tests for ROCm MoE kernel vs CPU reference
 *
 * Validates that ROCmMoEKernel produces numerically equivalent results to
 * CPUMoEKernel for all 5 IMoEKernel operations:
 *   1. route()           — gate logits + softmax + top-k
 *   2. gatherTokenBatch()— gather token rows to batch
 *   3. scatterAddWeighted() — weighted scatter-add
 *   4. sharedExpertGate()— sigmoid(dot) * scale
 *   5. swiGLU()          — silu(gate) * up
 *
 * Pass Criteria:
 * - Cosine similarity >= 0.999 for all operations
 * - No NaN/Inf in outputs
 * - Top-k expert selections match CPU reference
 *
 * Target Hardware: AMD MI50 (gfx906 / Vega 20)
 */

#include <gtest/gtest.h>

#include "tensors/Tensors.h"
#include "utils/Assertions.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "kernels/rocm/ROCmWeightPacker.h"
#include "kernels/cpu/moe/CPUMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/DecodeExpertHistogram.h"
#include "execution/moe/DeviceMoERebalanceController.h"
#include "execution/moe/MoEExpertWeightService.h"
#include "execution/moe/MoERuntimeTable.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "loaders/ModelContext.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#endif

#include "../../../utils/TestTensorFactory.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/QuantizedVerifierFormats.h"

#include <vector>
#include <array>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef HAVE_ROCM
extern "C" bool hipMoE_group_tokens_small_float(
    const float *routing_indices, const float *routing_weights,
    int *expert_counts, int *expert_offsets,
    int *grouped_token_indices, int *original_to_grouped,
    int *original_expert_ids,
    float *grouped_weights,
    int *active_expert_ids,
    int total_slots, int num_experts, int top_k,
    int max_active_experts,
    int device_idx, void *stream);

extern "C" bool hipMoE_scatter_tokens(
    const int *routing_indices, const float *routing_weights,
    const int *expert_offsets, int *write_heads,
    int *grouped_token_indices, int *original_to_grouped,
    float *grouped_weights,
    int total_slots, int num_experts, int top_k,
    int device_idx, void *stream);

extern "C" bool hipMoE_build_runtime_original_to_grouped(
    const void *runtime,
    int *original_to_grouped,
    int current_slots,
    int max_slots,
    int num_experts,
    int top_k,
    int device_idx,
    void *stream);

extern "C" bool hipMoE_gate_logits_small_m(
    const float *hidden, const float *gate_weights, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream);

extern "C" bool hipMoE_gate_logits_small_m_bf16_weights(
    const float *hidden, const void *gate_weights_bf16, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream);

extern "C" bool hipMoE_gate_logits_single_token(
    const float *hidden, const float *gate_weights, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream);

extern "C" bool hipMoE_gate_logits_single_token_bf16_weights(
    const float *hidden, const void *gate_weights_bf16, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream);

extern "C" bool hipMoE_softmax_topk(
    float *logits,
    int *expert_indices, float *expert_weights,
    int seq_len, int num_experts, int top_k,
    bool normalize_weights,
    int device_idx, void *stream,
    const int *device_effective_seq_len);

extern "C" bool hipMoE_materialize_runtime_prefill_descriptor_tables(
    const void *runtime,
    llaminar2::DeviceNativeVNNIMatrixDesc *gate_descs,
    llaminar2::DeviceNativeVNNIMatrixDesc *up_descs,
    llaminar2::DeviceNativeVNNIMatrixDesc *down_descs,
    int num_experts,
    int device_idx,
    void *stream);
#endif

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{

    // ============================================================================
    // ROCm Availability Check
    // ============================================================================

    std::unique_ptr<DeviceWorkspaceManager> bindDefaultMoEWorkspace(
        ROCmMoEKernel &kernel,
        int max_seq_len = 64,
        int d_model = 2048,
        int intermediate = 512,
        int num_experts = 256,
        int top_k = 16)
    {
        auto reqs = MoEWorkspaceBuffers::rocmMoE(
            max_seq_len,
            d_model,
            intermediate,
            num_experts,
            top_k);
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            DeviceId::rocm(0),
            reqs.total_bytes_with_alignment() + 4 * 1024 * 1024);
        EXPECT_TRUE(workspace->allocate(reqs));
        kernel.bindWorkspace(workspace.get());
        return workspace;
    }

    DeviceMoELayerRuntime makeAllLocalRuntime(
        int num_experts,
        int top_k,
        uint32_t epoch = 1)
    {
        DeviceMoELayerRuntime runtime{};
        runtime.active_bank = 0;
        runtime.active_epoch = epoch;
        runtime.expert_count = static_cast<uint32_t>(num_experts);
        runtime.top_k = static_cast<uint32_t>(top_k);
        for (auto &bank : runtime.banks)
        {
            bank.epoch = epoch;
            bank.expert_count = static_cast<uint32_t>(num_experts);
            const int capped_experts = std::min(num_experts, static_cast<int>(kDeviceMoEMaxExperts));
            for (int expert = 0; expert < capped_experts; ++expert)
                bank.local_compute_mask[expert] = 1;
        }
        return runtime;
    }

    void populateRuntimeDescriptors(
        DeviceMoELayerRuntime &runtime,
        const std::vector<DeviceNativeVNNIMatrixDesc> *gate_descs,
        const std::vector<DeviceNativeVNNIMatrixDesc> *up_descs,
        const std::vector<DeviceNativeVNNIMatrixDesc> *down_descs)
    {
        const int num_experts =
            std::min(static_cast<int>(runtime.expert_count), static_cast<int>(kDeviceMoEMaxExperts));
        for (auto &bank : runtime.banks)
        {
            bank.expert_count = runtime.expert_count;
            for (int expert = 0; expert < num_experts; ++expert)
            {
                bank.local_compute_mask[expert] = 1;
                bank.replica_role[expert] = static_cast<uint8_t>(DeviceMoEReplicaRole::Primary);
                bank.resident_participant_mask[expert] = 1u;

                auto &desc = bank.experts[expert];
                desc.logical_expert_id = expert;
                desc.owner_participant = 0;
                desc.local_slot = expert;
                desc.flags = toMoEExpertFlags(
                    DeviceMoEExpertFlags::Valid |
                    DeviceMoEExpertFlags::Resident |
                    DeviceMoEExpertFlags::LocalCompute);
                if (gate_descs)
                    desc.gate = (*gate_descs)[static_cast<size_t>(expert)];
                if (up_descs)
                    desc.up = (*up_descs)[static_cast<size_t>(expert)];
                if (down_descs)
                    desc.down = (*down_descs)[static_cast<size_t>(expert)];
            }
        }
    }

    struct MoEVerifierFormatCase
    {
        const char *name;
        std::function<std::unique_ptr<TensorBase>(const std::vector<size_t> &, int)> make;
    };

    /**
     * @brief NativeVNNI tensor formats that must preserve grouped verifier
     * decode equivalence.
     *
     * The MoE verifier pipeline uses the same NativeVNNI descriptor contract for
     * routed experts, shared experts, direct grouping, runtime grouping, and
     * masked LocalTP grouping.  Keep this list centralized so a newly supported
     * codebook cannot accidentally enter production without batch-invariance
     * coverage.
     */
    std::vector<MoEVerifierFormatCase> allMoEVerifierFormatCases()
    {
        std::vector<MoEVerifierFormatCase> formats;
        formats.reserve(quantizedVerifierFormats().size());
        for (const auto &format : quantizedVerifierFormats())
        {
            formats.push_back({
                format.label,
                [creator = format.create](const std::vector<size_t> &shape, int seed)
                {
                    return creator(shape, static_cast<uint32_t>(seed));
                }});
        }
        return formats;
    }

    DeviceNativeVNNIMatrixDesc fakeNativeVNNIDesc(uintptr_t base)
    {
        DeviceNativeVNNIMatrixDesc desc{};
        desc.payload = reinterpret_cast<const uint8_t *>(base);
        desc.scales = reinterpret_cast<const void *>(base + 0x100u);
        desc.mins = reinterpret_cast<const void *>(base + 0x200u);
        desc.n = 8;
        desc.k = 8;
        desc.blocks_per_row = 1;
        desc.codebook_id = 7;
        return desc;
    }

    void seedRuntimeBankExpert(
        DeviceMoEPlacementBank &bank,
        int expert,
        int owner_participant,
        bool local_ready)
    {
        auto &desc = bank.experts[expert];
        const uintptr_t base = 0x39000000u + static_cast<uintptr_t>(expert) * 0x4000u;
        desc.logical_expert_id = expert;
        desc.owner_participant = owner_participant;
        desc.local_slot = local_ready ? expert : -1;
        desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                      DeviceMoEExpertFlags::Resident);
        bank.local_compute_mask[expert] = local_ready ? 1u : 0u;
        bank.replica_role[expert] =
            local_ready
                ? static_cast<uint8_t>(DeviceMoEReplicaRole::Primary)
                : static_cast<uint8_t>(DeviceMoEReplicaRole::None);
        if (local_ready)
        {
            desc.gate = fakeNativeVNNIDesc(base + 0x10u);
            desc.up = fakeNativeVNNIDesc(base + 0x20u);
            desc.down = fakeNativeVNNIDesc(base + 0x30u);
            desc.flags |= toMoEExpertFlags(DeviceMoEExpertFlags::LocalCompute);
        }
    }

    /**
     * @brief Construct the clean copy/apply status header expected by rebalance kernels.
     *
     * Production transfer waves initialize this status in the payload pack
     * kernel and then let unpack accumulate destination-side counters into the
     * same record. Focused tests that invoke unpack directly use this helper to
     * seed the header without depending on a preceding pack.
     */
    DeviceMoERebalanceApplyStatus makeCleanRebalanceApplyStatus()
    {
        DeviceMoERebalanceApplyStatus status{};
        status.magic = kDeviceMoERebalanceMagic;
        status.version = kDeviceMoERebalanceVersion;
        status.status_code =
            static_cast<uint32_t>(DeviceMoERebalanceApplyStatusCode::Ok);
        return status;
    }

    DeviceNativeVNNIMatrixDesc makePrefillRuntimeDesc(
        uintptr_t base,
        int rows,
        int cols,
        uint8_t codebook_id)
    {
        DeviceNativeVNNIMatrixDesc desc{};
        desc.payload = reinterpret_cast<const uint8_t *>(base);
        desc.scales = reinterpret_cast<const void *>(base + 0x100u);
        desc.mins = reinterpret_cast<const void *>(base + 0x200u);
        desc.emins = reinterpret_cast<const void *>(base + 0x300u);
        desc.n = rows;
        desc.k = cols;
        desc.blocks_per_row = static_cast<uint32_t>(cols / 32);
        desc.codebook_id = codebook_id;
        return desc;
    }

    void expectSameNativeVNNIDesc(
        const DeviceNativeVNNIMatrixDesc &actual,
        const DeviceNativeVNNIMatrixDesc &expected)
    {
        EXPECT_EQ(reinterpret_cast<uintptr_t>(actual.payload),
                  reinterpret_cast<uintptr_t>(expected.payload));
        EXPECT_EQ(reinterpret_cast<uintptr_t>(actual.scales),
                  reinterpret_cast<uintptr_t>(expected.scales));
        EXPECT_EQ(reinterpret_cast<uintptr_t>(actual.mins),
                  reinterpret_cast<uintptr_t>(expected.mins));
        EXPECT_EQ(reinterpret_cast<uintptr_t>(actual.emins),
                  reinterpret_cast<uintptr_t>(expected.emins));
        EXPECT_EQ(actual.n, expected.n);
        EXPECT_EQ(actual.k, expected.k);
        EXPECT_EQ(actual.blocks_per_row, expected.blocks_per_row);
        EXPECT_EQ(actual.codebook_id, expected.codebook_id);
    }

    MoEPlacementUpdate makeParticipantOneBaseUpdate(uint32_t epoch)
    {
        constexpr int num_experts = 4;
        MoEPlacementUpdate update;
        update.epoch = epoch;
        update.expert_count = num_experts;
        update.participant_id = 1;
        update.participant_count = 3;
        update.experts.resize(num_experts);
        update.local_compute_mask = {0, 0, 1, 0};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        update.resident_participant_mask = {0b011u, 0b001u, 0b010u, 0b100u};
        const int owners[num_experts] = {0, 0, 1, 2};
        for (int expert = 0; expert < num_experts; ++expert)
        {
            auto &desc = update.experts[static_cast<size_t>(expert)];
            const uintptr_t base = 0x34000000u + static_cast<uintptr_t>(expert) * 0x1000u;
            desc.logical_expert_id = expert;
            desc.owner_participant = owners[expert];
            desc.local_slot = expert;
            desc.gate = fakeNativeVNNIDesc(base + 0x10u);
            desc.up = fakeNativeVNNIDesc(base + 0x20u);
            desc.down = fakeNativeVNNIDesc(base + 0x30u);
            desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                          DeviceMoEExpertFlags::Resident |
                                          DeviceMoEExpertFlags::PreferredOwner);
            if (update.local_compute_mask[static_cast<size_t>(expert)] != 0u)
                desc.flags |= toMoEExpertFlags(DeviceMoEExpertFlags::LocalCompute);
        }
        return update;
    }

    MoEPlacementUpdate makeDynamicOwnershipTransferUpdate(uint32_t epoch,
                                                          uint32_t participant_id)
    {
        auto update = makeParticipantOneBaseUpdate(epoch);
        update.participant_id = static_cast<int>(participant_id);
        update.participant_count = 2;
        update.resident_participant_mask = {0b01u, 0b01u, 0b10u, 0b10u};
        const int owners[] = {0, 0, 1, 1};
        update.local_compute_mask.assign(static_cast<size_t>(update.expert_count), 0u);
        update.replica_role.assign(
            static_cast<size_t>(update.expert_count),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None));
        for (int expert = 0; expert < update.expert_count; ++expert)
        {
            auto &desc = update.experts[static_cast<size_t>(expert)];
            desc.owner_participant = owners[expert];
            desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                          DeviceMoEExpertFlags::Resident |
                                          DeviceMoEExpertFlags::PreferredOwner);
            if (owners[expert] == static_cast<int>(participant_id))
            {
                update.local_compute_mask[static_cast<size_t>(expert)] = 1u;
                update.replica_role[static_cast<size_t>(expert)] =
                    static_cast<uint8_t>(DeviceMoEReplicaRole::Primary);
                desc.flags |= toMoEExpertFlags(DeviceMoEExpertFlags::LocalCompute);
            }
        }
        return update;
    }

    bool hasROCm()
    {
#ifdef HAVE_ROCM
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        return (err == hipSuccess && count > 0);
#else
        return false;
#endif
    }

#define SKIP_IF_NO_ROCM()                                           \
    do                                                              \
    {                                                               \
        if (!hasROCm())                                             \
        {                                                           \
            GTEST_SKIP() << "No ROCm GPU available, skipping test"; \
        }                                                           \
    } while (0)

    /**
     * @brief Own a non-default HIP stream on one explicit ROCm ordinal.
     *
     * Focused LocalTP regressions launch the same production kernel on two
     * participants.  Giving each participant an explicit stream mirrors the
     * graph executor contract and prevents the test from accidentally passing
     * because the HIP default stream serialized unrelated work.
     */
    class ScopedHipDeviceStream
    {
    public:
        explicit ScopedHipDeviceStream(int device_ordinal)
            : device_ordinal_(device_ordinal)
        {
            status_ = hipSetDevice(device_ordinal_);
            if (status_ == hipSuccess)
                status_ = hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking);
        }

        ~ScopedHipDeviceStream()
        {
            if (stream_)
            {
                (void)hipSetDevice(device_ordinal_);
                (void)hipStreamDestroy(stream_);
            }
        }

        ScopedHipDeviceStream(const ScopedHipDeviceStream &) = delete;
        ScopedHipDeviceStream &operator=(const ScopedHipDeviceStream &) = delete;

        hipError_t status() const { return status_; }
        hipStream_t get() const { return stream_; }

    private:
        int device_ordinal_ = 0;
        hipStream_t stream_ = nullptr;
        hipError_t status_ = hipSuccess;
    };

    /**
     * @brief Copy the device-authoritative FP32 bytes from a HIP tensor.
     *
     * Reading through the tensor host mirror can accidentally test coherence
     * bookkeeping instead of the bytes the next GPU stage consumes.  These MTP
     * verifier regressions copy directly from the resident device allocation.
     */
    std::vector<float> copyROCmFP32TensorToHost(
        const FP32Tensor *tensor,
        hipStream_t stream)
    {
        std::vector<float> host(tensor->numel());
        EXPECT_NE(tensor->gpu_data_ptr(), nullptr);
        EXPECT_EQ(hipMemcpyAsync(host.data(),
                                 tensor->gpu_data_ptr(),
                                 host.size() * sizeof(float),
                                 hipMemcpyDeviceToHost,
                                 stream),
                  hipSuccess);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        return host;
    }

    // ============================================================================
    // Similarity Utilities
    // ============================================================================

    double cosineSimilarity(const float *a, const float *b, size_t count)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        if (norm_a < 1e-30 && norm_b < 1e-30)
            return 1.0;
        if (norm_a < 1e-30 || norm_b < 1e-30)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    double relativeL2Error(const float *actual, const float *reference, size_t count)
    {
        double err_sq = 0.0, ref_sq = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            double diff = static_cast<double>(actual[i]) - reference[i];
            err_sq += diff * diff;
            ref_sq += static_cast<double>(reference[i]) * reference[i];
        }
        if (ref_sq < 1e-30)
            return err_sq < 1e-30 ? 0.0 : std::numeric_limits<double>::infinity();
        return std::sqrt(err_sq / ref_sq);
    }

    double maxAbsDiff(const float *actual, const float *reference, size_t count)
    {
        double max_diff = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            const double diff = std::fabs(static_cast<double>(actual[i]) - reference[i]);
            max_diff = std::max(max_diff, diff);
        }
        return max_diff;
    }

    struct StrictVerifierSimilarityMetrics
    {
        double cosine = 0.0;
        double relative_l2 = 0.0;
        double max_abs = 0.0;
        double min_row_cosine = 1.0;
        double max_row_relative_l2 = 0.0;
        double max_row_kl = 0.0;
        size_t worst_row = 0;
    };

    /**
     * @brief KL(reference || actual) after a stable row-wise softmax.
     *
     * Hidden-state vectors are not probabilities, but row-softmax KL is a useful
     * verifier guard: it catches shape changes in the largest coordinates that a
     * raw cosine can hide.  MTP verifier rows are short and near token decisions,
     * so a path that cannot satisfy this check is not decode-equivalent enough to
     * justify publishing accepted speculative state.
     */
    double rowSoftmaxKLDivergence(const float *actual, const float *reference, size_t row_width)
    {
        if (row_width == 0)
            return 0.0;

        double max_actual = -std::numeric_limits<double>::infinity();
        double max_reference = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < row_width; ++i)
        {
            max_actual = std::max(max_actual, static_cast<double>(actual[i]));
            max_reference = std::max(max_reference, static_cast<double>(reference[i]));
        }

        double sum_actual = 0.0;
        double sum_reference = 0.0;
        for (size_t i = 0; i < row_width; ++i)
        {
            sum_actual += std::exp(static_cast<double>(actual[i]) - max_actual);
            sum_reference += std::exp(static_cast<double>(reference[i]) - max_reference);
        }

        constexpr double kEps = 1.0e-30;
        double kl = 0.0;
        for (size_t i = 0; i < row_width; ++i)
        {
            const double p = std::exp(static_cast<double>(reference[i]) - max_reference) /
                             std::max(sum_reference, kEps);
            const double q = std::exp(static_cast<double>(actual[i]) - max_actual) /
                             std::max(sum_actual, kEps);
            kl += p * (std::log(std::max(p, kEps)) - std::log(std::max(q, kEps)));
        }
        return kl;
    }

    StrictVerifierSimilarityMetrics computeStrictVerifierSimilarity(
        const float *actual,
        const float *reference,
        size_t count,
        size_t row_width)
    {
        StrictVerifierSimilarityMetrics metrics;
        metrics.cosine = cosineSimilarity(actual, reference, count);
        metrics.relative_l2 = relativeL2Error(actual, reference, count);
        metrics.max_abs = maxAbsDiff(actual, reference, count);

        if (row_width == 0 || count % row_width != 0)
            row_width = count;
        const size_t rows = (row_width == 0) ? 0 : count / row_width;

        for (size_t row = 0; row < rows; ++row)
        {
            const float *row_actual = actual + row * row_width;
            const float *row_reference = reference + row * row_width;
            const double row_cosine = cosineSimilarity(row_actual, row_reference, row_width);
            const double row_rel_l2 = relativeL2Error(row_actual, row_reference, row_width);
            const double row_kl = rowSoftmaxKLDivergence(row_actual, row_reference, row_width);

            if (row_rel_l2 > metrics.max_row_relative_l2 ||
                row_kl > metrics.max_row_kl ||
                row_cosine < metrics.min_row_cosine)
            {
                metrics.worst_row = row;
            }
            metrics.min_row_cosine = std::min(metrics.min_row_cosine, row_cosine);
            metrics.max_row_relative_l2 = std::max(metrics.max_row_relative_l2, row_rel_l2);
            metrics.max_row_kl = std::max(metrics.max_row_kl, row_kl);
        }

        return metrics;
    }

    void expectStrictVerifierSimilarity(
        const char *label,
        const float *actual,
        const float *reference,
        size_t count,
        size_t row_width,
        double min_cosine = 0.9999,
        double max_relative_l2 = 0.006,
        double min_row_cosine = 0.9998,
        double max_row_relative_l2 = 0.008,
        double max_row_kl = 1.0e-4)
    {
        SCOPED_TRACE(label);
        ASSERT_NE(actual, nullptr);
        ASSERT_NE(reference, nullptr);
        ASSERT_GT(count, 0u);
        const auto metrics = computeStrictVerifierSimilarity(actual, reference, count, row_width);
        EXPECT_GE(metrics.cosine, min_cosine)
            << "relative_l2=" << metrics.relative_l2
            << " max_abs=" << metrics.max_abs
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " max_row_kl=" << metrics.max_row_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_LE(metrics.relative_l2, max_relative_l2)
            << "cosine=" << metrics.cosine
            << " max_abs=" << metrics.max_abs
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " max_row_kl=" << metrics.max_row_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_GE(metrics.min_row_cosine, min_row_cosine)
            << "cosine=" << metrics.cosine
            << " relative_l2=" << metrics.relative_l2
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " max_row_kl=" << metrics.max_row_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_LE(metrics.max_row_relative_l2, max_row_relative_l2)
            << "cosine=" << metrics.cosine
            << " relative_l2=" << metrics.relative_l2
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_kl=" << metrics.max_row_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_LE(metrics.max_row_kl, max_row_kl)
            << "cosine=" << metrics.cosine
            << " relative_l2=" << metrics.relative_l2
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " worst_row=" << metrics.worst_row;
    }

    /**
     * @brief Require grouped verifier rows to match serial decode exactly.
     *
     * MTP verifier publication is allowed to commit rows directly into live
     * decode state.  For these focused grouped-verifier regressions, a tiny
     * numerical tolerance is therefore the wrong contract: even sub-ulp drift
     * in an early MoE row can be amplified by later residual, GDN, and sampling
     * stages.  This helper treats the row-by-row M=1 decode path as the
     * canonical byte stream and reports similarity metrics only to make the
     * first failing kernel easier to diagnose.
     */
    void expectBitwiseVerifierRowsEqual(
        const char *label,
        const float *actual,
        const float *reference,
        size_t count,
        size_t row_width)
    {
        SCOPED_TRACE(label);
        ASSERT_NE(actual, nullptr);
        ASSERT_NE(reference, nullptr);
        ASSERT_GT(count, 0u);

        if (std::memcmp(actual, reference, count * sizeof(float)) == 0)
            return;

        size_t first_mismatch = 0;
        while (first_mismatch < count && actual[first_mismatch] == reference[first_mismatch])
            ++first_mismatch;

        uint32_t actual_bits = 0;
        uint32_t reference_bits = 0;
        if (first_mismatch < count)
        {
            std::memcpy(&actual_bits, actual + first_mismatch, sizeof(actual_bits));
            std::memcpy(&reference_bits, reference + first_mismatch, sizeof(reference_bits));
        }

        const auto metrics = computeStrictVerifierSimilarity(actual, reference, count, row_width);
        const size_t safe_row_width = row_width == 0 ? count : row_width;
        const size_t mismatch_row = first_mismatch / safe_row_width;
        const size_t mismatch_col = first_mismatch % safe_row_width;
        ADD_FAILURE()
            << "grouped verifier rows are not bitwise serial-decode equivalent"
            << " first_mismatch=" << first_mismatch
            << " row=" << mismatch_row
            << " col=" << mismatch_col
            << " actual=" << (first_mismatch < count ? actual[first_mismatch] : 0.0f)
            << " reference=" << (first_mismatch < count ? reference[first_mismatch] : 0.0f)
            << " actual_bits=0x" << std::hex << actual_bits
            << " reference_bits=0x" << reference_bits << std::dec
            << " cosine=" << metrics.cosine
            << " relative_l2=" << metrics.relative_l2
            << " max_abs=" << metrics.max_abs
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " max_row_kl=" << metrics.max_row_kl
            << " worst_row=" << metrics.worst_row;
    }

    double klDivergenceNormalized(const float *actual, const float *reference, size_t count)
    {
        constexpr double kEps = 1.0e-30;
        double kl = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            const double p = std::max(static_cast<double>(reference[i]), kEps);
            const double q = std::max(static_cast<double>(actual[i]), kEps);
            kl += p * (std::log(p) - std::log(q));
        }
        return kl;
    }

    /**
     * @brief Enforce decode-equivalent top-k routing for verifier-sized M=1..4 batches.
     */
    void expectStrictTopKRowsEquivalent(
        const std::vector<float> &batched_indices,
        const std::vector<float> &batched_weights,
        const std::vector<float> &rowwise_indices,
        const std::vector<float> &rowwise_weights,
        int seq_len,
        int top_k)
    {
        ASSERT_EQ(batched_indices.size(), rowwise_indices.size());
        ASSERT_EQ(batched_weights.size(), rowwise_weights.size());
        for (size_t i = 0; i < batched_indices.size(); ++i)
            ASSERT_EQ(static_cast<int>(batched_indices[i]), static_cast<int>(rowwise_indices[i]))
                << "top-k index mismatch at flattened slot " << i;

        expectBitwiseVerifierRowsEqual("top-k indices",
                                       batched_indices.data(),
                                       rowwise_indices.data(),
                                       batched_indices.size(),
                                       static_cast<size_t>(top_k));
        expectBitwiseVerifierRowsEqual("top-k weights",
                                       batched_weights.data(),
                                       rowwise_weights.data(),
                                       batched_weights.size(),
                                       static_cast<size_t>(top_k));
    }

    class ScopedEnvOverride
    {
    public:
        ScopedEnvOverride(const char *name, const char *value)
            : name_(name)
        {
            const char *old_value = std::getenv(name);
            if (old_value)
            {
                had_old_value_ = true;
                old_value_ = old_value;
            }
            ::setenv(name_.c_str(), value, 1);
        }

        ~ScopedEnvOverride()
        {
            if (had_old_value_)
                ::setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                ::unsetenv(name_.c_str());
        }

        ScopedEnvOverride(const ScopedEnvOverride &) = delete;
        ScopedEnvOverride &operator=(const ScopedEnvOverride &) = delete;

    private:
        std::string name_;
        bool had_old_value_ = false;
        std::string old_value_;
    };

    /// @brief Give synthetic K-quant blocks nonzero min terms so tests cover asymmetric correction.
    void injectNonZeroKQuantMins(TensorBase *tensor)
    {
        ASSERT_NE(tensor, nullptr);
        if (tensor->native_type() == TensorType::Q4_K)
        {
            auto *blocks = reinterpret_cast<Q4_KBlock *>(tensor->raw_mutable_data());
            const size_t block_count = tensor->size_bytes() / sizeof(Q4_KBlock);
            for (size_t block = 0; block < block_count; ++block)
                blocks[block].dmin = fp32_to_fp16(0.006f + 0.001f * static_cast<float>(block % 7));
            return;
        }
        if (tensor->native_type() == TensorType::Q5_K)
        {
            auto *blocks = reinterpret_cast<Q5_KBlock *>(tensor->raw_mutable_data());
            const size_t block_count = tensor->size_bytes() / sizeof(Q5_KBlock);
            for (size_t block = 0; block < block_count; ++block)
                blocks[block].dmin = fp32_to_fp16(0.005f + 0.001f * static_cast<float>(block % 5));
        }
    }

    bool hasNaNOrInf(const float *data, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            if (std::isnan(data[i]) || std::isinf(data[i]))
                return true;
        }
        return false;
    }

    // Fill vector with uniform random values
    void fillRandom(std::vector<float> &v, float lo, float hi, unsigned seed = 42)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(lo, hi);
        for (auto &x : v)
            x = dist(gen);
    }

} // namespace

#ifdef HAVE_ROCM

namespace
{
    class ScopedROCmEnvOverride
    {
    public:
        ScopedROCmEnvOverride(const char *name, const char *value)
            : name_(name)
        {
            const char *existing = std::getenv(name_);
            if (existing)
            {
                had_original_ = true;
                original_ = existing;
            }
            ::setenv(name_, value, 1);
            mutableDebugEnv().rocm.reload();
        }

        ~ScopedROCmEnvOverride()
        {
            if (had_original_)
                ::setenv(name_, original_.c_str(), 1);
            else
                ::unsetenv(name_);
            mutableDebugEnv().rocm.reload();
        }

        ScopedROCmEnvOverride(const ScopedROCmEnvOverride &) = delete;
        ScopedROCmEnvOverride &operator=(const ScopedROCmEnvOverride &) = delete;

    private:
        const char *name_ = nullptr;
        bool had_original_ = false;
        std::string original_;
    };

    /**
     * @brief Resolve the real Qwen3.6 MoE GGUF used by model-level MTP parity.
     *
     * Synthetic tensors are useful for all-format coverage, but the ROCm M=3
     * failure that motivated this regression first appeared only with the
     * checkpoint payload.  The environment override keeps the test portable
     * across developer workstations and CI model-cache layouts.
     */
    std::filesystem::path qwen36MoEModelPathForROCmMoEKernelTest()
    {
        if (const char *env = std::getenv("LLAMINAR_QWEN36_MOE_MODEL"))
            return std::filesystem::path(env);
        return std::filesystem::path("/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf");
    }

    /**
     * @brief Open a model context configured for native GPU-weight preparation.
     *
     * The focused kernel regressions load only their selected layer's expert
     * tensors, but they still use the same mmap/native-weight path as the
     * production graph builder so tensor views and quantized block payloads are
     * byte-for-byte the same objects the ROCm GEMM upload pipeline consumes.
     */
    std::shared_ptr<ModelContext> loadQwen36MoEModelForROCmExpertWeights(
        const std::filesystem::path &model_path)
    {
        ModelContextConfig config = ModelContextConfig::defaults();
        config.strategy = WeightDistributionStrategy::REPLICATED;
        config.weight_precision = WeightPrecision::NATIVE;
        config.use_mmap = true;
        config.target_is_gpu = true;
        return ModelContext::create(model_path.string(), config);
    }

    /**
     * @brief Rehydrate an FP32 value from a captured raw-snapshot bit pattern.
     */
    float f32FromBitsForROCmMoEKernelTest(uint32_t bits)
    {
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    /**
     * @brief Preserve the exact IEEE-754 representation of an FP32 value.
     *
     * Route weights are part of the byte-equivalence contract.  Converting a
     * captured weight through text, or comparing it with a tolerance, can hide
     * the one-bit differences that later change the ordered expert reduction.
     */
    uint32_t f32BitsForROCmMoEKernelTest(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    /// @brief Concatenates per-expert 2D tensors into the 3D GGUF-style parent tensor used by MoE views.
    template <typename TensorT>
    std::shared_ptr<TensorT> makeExpertParentTensor(
        const std::vector<std::unique_ptr<TensorBase>> &experts,
        const std::vector<size_t> &parent_shape)
    {
        std::vector<uint8_t> raw;
        for (const auto &expert : experts)
        {
            const auto *bytes = static_cast<const uint8_t *>(expert->raw_data());
            raw.insert(raw.end(), bytes, bytes + expert->size_bytes());
        }
        return std::make_shared<TensorT>(parent_shape, raw);
    }

    /// @brief Computes the routed MoE FFN output with dequantized FP32 weights as an independent reference.
    std::vector<float> computeCpuDequantMoEPrefillReference(
        const float *input,
        int seq_len,
        int d_model,
        int intermediate,
        int num_experts,
        int top_k,
        const std::vector<std::unique_ptr<TensorBase>> &gate_weight_tensors,
        const std::vector<std::unique_ptr<TensorBase>> &up_weight_tensors,
        const std::vector<std::unique_ptr<TensorBase>> &down_weight_tensors,
        const std::vector<int> &routing_indices,
        const std::vector<float> &routing_weights)
    {
        std::vector<std::vector<std::pair<int, float>>> routes_by_expert(static_cast<size_t>(num_experts));
        for (int token_idx = 0; token_idx < seq_len; ++token_idx)
        {
            for (int route_idx = 0; route_idx < top_k; ++route_idx)
            {
                const int slot_idx = token_idx * top_k + route_idx;
                routes_by_expert[static_cast<size_t>(routing_indices[static_cast<size_t>(slot_idx)])].push_back(
                    {token_idx, routing_weights[static_cast<size_t>(slot_idx)]});
            }
        }

        std::vector<float> output(static_cast<size_t>(seq_len) * d_model, 0.0f);
        std::vector<float> gate(static_cast<size_t>(intermediate));
        std::vector<float> up(static_cast<size_t>(intermediate));
        std::vector<float> swiglu(static_cast<size_t>(intermediate));

        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &routes = routes_by_expert[static_cast<size_t>(expert_id)];
            if (routes.empty())
                continue;

            const float *gate_weights = gate_weight_tensors[static_cast<size_t>(expert_id)]->data();
            const float *up_weights = up_weight_tensors[static_cast<size_t>(expert_id)]->data();
            const float *down_weights = down_weight_tensors[static_cast<size_t>(expert_id)]->data();

            for (const auto &[token_idx, route_weight] : routes)
            {
                const float *token_input = input + static_cast<size_t>(token_idx) * d_model;

                for (int hidden_idx = 0; hidden_idx < intermediate; ++hidden_idx)
                {
                    const float *gate_row = gate_weights + static_cast<size_t>(hidden_idx) * d_model;
                    const float *up_row = up_weights + static_cast<size_t>(hidden_idx) * d_model;
                    double gate_accum = 0.0;
                    double up_accum = 0.0;
                    for (int model_idx = 0; model_idx < d_model; ++model_idx)
                    {
                        const double activation = static_cast<double>(token_input[model_idx]);
                        gate_accum += activation * gate_row[model_idx];
                        up_accum += activation * up_row[model_idx];
                    }
                    gate[static_cast<size_t>(hidden_idx)] = static_cast<float>(gate_accum);
                    up[static_cast<size_t>(hidden_idx)] = static_cast<float>(up_accum);
                }

                for (int hidden_idx = 0; hidden_idx < intermediate; ++hidden_idx)
                {
                    const float gate_value = gate[static_cast<size_t>(hidden_idx)];
                    const float silu = gate_value / (1.0f + std::exp(-gate_value));
                    swiglu[static_cast<size_t>(hidden_idx)] = silu * up[static_cast<size_t>(hidden_idx)];
                }

                float *token_output = output.data() + static_cast<size_t>(token_idx) * d_model;
                for (int model_idx = 0; model_idx < d_model; ++model_idx)
                {
                    const float *down_row = down_weights + static_cast<size_t>(model_idx) * intermediate;
                    double down_accum = 0.0;
                    for (int hidden_idx = 0; hidden_idx < intermediate; ++hidden_idx)
                        down_accum += static_cast<double>(swiglu[static_cast<size_t>(hidden_idx)]) * down_row[hidden_idx];
                    token_output[model_idx] += route_weight * static_cast<float>(down_accum);
                }
            }
        }

        return output;
    }
}

TEST(Test__ROCmMoEKernel, UploadGroupedDescriptorTablesRejectInvalidDescriptors)
{
    SKIP_IF_NO_ROCM();

    const int d_model = 128;
    const int intermediate = 128;
    const int num_experts = 4;

    auto make_down_desc = [](int rows, int cols)
    {
        DeviceNativeVNNIMatrixDesc desc{};
        desc.payload = reinterpret_cast<const uint8_t *>(0x1000);
        desc.scales = reinterpret_cast<const void *>(0x2000);
        desc.n = rows;
        desc.k = cols;
        desc.blocks_per_row = static_cast<uint32_t>(cols / 32);
        desc.codebook_id = 0;
        return desc;
    };

    auto make_gateup_desc = [](int rows, int cols)
    {
        DeviceNativeVNNIMatrixDesc desc{};
        desc.payload = reinterpret_cast<const uint8_t *>(0x3000);
        desc.scales = reinterpret_cast<const void *>(0x4000);
        desc.n = rows;
        desc.k = cols;
        desc.blocks_per_row = static_cast<uint32_t>(cols / 32);
        desc.codebook_id = 0;
        return desc;
    };

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    std::vector<DeviceNativeVNNIMatrixDesc> down_descs;
    down_descs.reserve(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        down_descs.push_back(make_down_desc(d_model, intermediate));
    down_descs[2].payload = nullptr;
    EXPECT_EQ(moe_kernel.uploadGroupedExpertDownDescriptorTable(
                  down_descs.data(), num_experts, d_model, intermediate),
              -1);

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs;
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        gate_descs.push_back(make_gateup_desc(intermediate, d_model));
        up_descs.push_back(make_gateup_desc(intermediate, d_model));
    }
    up_descs[1].scales = nullptr;
    EXPECT_EQ(moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
                  gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate),
              -1);
}

TEST(Test__ROCmMoEKernel, RuntimePrefillDescriptorMaterializationUsesActiveRuntimeBank)
{
    SKIP_IF_NO_ROCM();

    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int d_model = 64;
    constexpr int intermediate = 96;
    constexpr uint32_t active_epoch = 17;

    DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, top_k, active_epoch);
    host_runtime.active_bank = 1;
    host_runtime.active_epoch = active_epoch;

    std::vector<DeviceNativeVNNIMatrixDesc> expected_gate(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> expected_up(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> expected_down(num_experts);

    for (int expert = 0; expert < num_experts; ++expert)
    {
        const uintptr_t inactive_base = 0x53000000u + static_cast<uintptr_t>(expert) * 0x10000u;
        host_runtime.banks[0].experts[expert].gate =
            makePrefillRuntimeDesc(inactive_base + 0x1000u, intermediate, d_model, 4);
        host_runtime.banks[0].experts[expert].up =
            makePrefillRuntimeDesc(inactive_base + 0x2000u, intermediate, d_model, 5);
        host_runtime.banks[0].experts[expert].down =
            makePrefillRuntimeDesc(inactive_base + 0x3000u, d_model, intermediate, 6);

        const uintptr_t active_base = 0x73000000u + static_cast<uintptr_t>(expert) * 0x10000u;
        expected_gate[static_cast<size_t>(expert)] =
            makePrefillRuntimeDesc(active_base + 0x1000u, intermediate, d_model,
                                   static_cast<uint8_t>(7 + expert));
        expected_up[static_cast<size_t>(expert)] =
            makePrefillRuntimeDesc(active_base + 0x2000u, intermediate, d_model,
                                   static_cast<uint8_t>(11 + expert));
        expected_down[static_cast<size_t>(expert)] =
            makePrefillRuntimeDesc(active_base + 0x3000u, d_model, intermediate,
                                   static_cast<uint8_t>(15 + expert));

        auto &active_desc = host_runtime.banks[1].experts[expert];
        active_desc.logical_expert_id = expert;
        active_desc.owner_participant = 0;
        active_desc.local_slot = expert;
        active_desc.flags = toMoEExpertFlags(
            DeviceMoEExpertFlags::Valid |
            DeviceMoEExpertFlags::Resident |
            DeviceMoEExpertFlags::LocalCompute);
        active_desc.gate = expected_gate[static_cast<size_t>(expert)];
        active_desc.up = expected_up[static_cast<size_t>(expert)];
        active_desc.down = expected_down[static_cast<size_t>(expert)];
    }

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoELayerRuntime *device_runtime = nullptr;
    DeviceNativeVNNIMatrixDesc *device_gate = nullptr;
    DeviceNativeVNNIMatrixDesc *device_up = nullptr;
    DeviceNativeVNNIMatrixDesc *device_down = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_runtime), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_gate),
                        num_experts * sizeof(DeviceNativeVNNIMatrixDesc)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_up),
                        num_experts * sizeof(DeviceNativeVNNIMatrixDesc)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_down),
                        num_experts * sizeof(DeviceNativeVNNIMatrixDesc)),
              hipSuccess);

    ASSERT_EQ(hipMemcpyAsync(device_runtime, &host_runtime, sizeof(host_runtime),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_TRUE(hipMoE_materialize_runtime_prefill_descriptor_tables(
        device_runtime,
        device_gate,
        device_up,
        device_down,
        num_experts,
        0,
        stream));
    EXPECT_FALSE(hipMoE_materialize_runtime_prefill_descriptor_tables(
        device_runtime,
        device_gate,
        device_up,
        device_down,
        num_experts,
        0,
        nullptr));

    std::vector<DeviceNativeVNNIMatrixDesc> actual_gate(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> actual_up(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> actual_down(num_experts);
    ASSERT_EQ(hipMemcpyAsync(actual_gate.data(), device_gate,
                             actual_gate.size() * sizeof(DeviceNativeVNNIMatrixDesc),
                             hipMemcpyDeviceToHost, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(actual_up.data(), device_up,
                             actual_up.size() * sizeof(DeviceNativeVNNIMatrixDesc),
                             hipMemcpyDeviceToHost, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(actual_down.data(), device_down,
                             actual_down.size() * sizeof(DeviceNativeVNNIMatrixDesc),
                             hipMemcpyDeviceToHost, stream),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    for (int expert = 0; expert < num_experts; ++expert)
    {
        SCOPED_TRACE(expert);
        expectSameNativeVNNIDesc(actual_gate[static_cast<size_t>(expert)],
                                 expected_gate[static_cast<size_t>(expert)]);
        expectSameNativeVNNIDesc(actual_up[static_cast<size_t>(expert)],
                                 expected_up[static_cast<size_t>(expert)]);
        expectSameNativeVNNIDesc(actual_down[static_cast<size_t>(expert)],
                                 expected_down[static_cast<size_t>(expert)]);
    }

    EXPECT_EQ(hipFree(device_down), hipSuccess);
    EXPECT_EQ(hipFree(device_up), hipSuccess);
    EXPECT_EQ(hipFree(device_gate), hipSuccess);
    EXPECT_EQ(hipFree(device_runtime), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, UploadGroupedDescriptorTablesAcceptAllNativeVNNICodebooks)
{
    SKIP_IF_NO_ROCM();

    struct CodebookCase
    {
        uint8_t codebook_id;
        bool requires_mins;
        bool requires_emins;
    };

    const std::vector<CodebookCase> codebooks = {
        {0, false, false},  {4, false, false},  {5, true, false},
        {6, false, false},  {7, true, false},   {8, true, false},
        {9, true, false},   {10, true, true},   {11, false, false},
        {12, false, false}, {13, true, false},  {14, true, false},
        {15, false, false}, {16, true, false},  {17, true, false},
        {19, false, false},
    };

    const int d_model = 128;
    const int intermediate = 128;
    const int num_experts = 2;

    auto make_desc = [](int rows, int cols, const CodebookCase &codebook, std::uintptr_t base)
    {
        DeviceNativeVNNIMatrixDesc desc{};
        desc.payload = reinterpret_cast<const uint8_t *>(base + 0x1000);
        desc.scales = reinterpret_cast<const void *>(base + 0x2000);
        desc.mins = codebook.requires_mins ? reinterpret_cast<const void *>(base + 0x3000) : nullptr;
        desc.emins = codebook.requires_emins ? reinterpret_cast<const void *>(base + 0x4000) : nullptr;
        desc.n = rows;
        desc.k = cols;
        desc.blocks_per_row = static_cast<uint32_t>(cols / 32);
        desc.codebook_id = codebook.codebook_id;
        return desc;
    };

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    for (size_t case_idx = 0; case_idx < codebooks.size(); ++case_idx)
    {
        const auto &codebook = codebooks[case_idx];
        const std::uintptr_t base = 0x100000 + case_idx * 0x10000;

        std::vector<DeviceNativeVNNIMatrixDesc> down_descs;
        std::vector<DeviceNativeVNNIMatrixDesc> gate_descs;
        std::vector<DeviceNativeVNNIMatrixDesc> up_descs;
        down_descs.reserve(num_experts);
        gate_descs.reserve(num_experts);
        up_descs.reserve(num_experts);

        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const std::uintptr_t expert_base = base + static_cast<std::uintptr_t>(expert_id) * 0x1000;
            down_descs.push_back(make_desc(d_model, intermediate, codebook, expert_base));
            gate_descs.push_back(make_desc(intermediate, d_model, codebook, expert_base + 0x5000));
            up_descs.push_back(make_desc(intermediate, d_model, codebook, expert_base + 0xA000));
        }

        EXPECT_GE(moe_kernel.uploadGroupedExpertDownDescriptorTable(
                      down_descs.data(), num_experts, d_model, intermediate),
                  0)
            << "down descriptor rejected codebook " << static_cast<int>(codebook.codebook_id);
        EXPECT_GE(moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
                      gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate),
                  0)
            << "gate/up descriptor rejected codebook " << static_cast<int>(codebook.codebook_id);
    }

    CodebookCase q2_k_missing_emins{10, true, false};
    std::vector<DeviceNativeVNNIMatrixDesc> missing_emins;
    missing_emins.reserve(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        const std::uintptr_t expert_base = 0x500000 + static_cast<std::uintptr_t>(expert_id) * 0x1000;
        missing_emins.push_back(make_desc(d_model, intermediate, q2_k_missing_emins, expert_base));
    }
    EXPECT_EQ(moe_kernel.uploadGroupedExpertDownDescriptorTable(
                  missing_emins.data(), num_experts, d_model, intermediate),
              -1);

    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST(Test__ROCmMoEKernel, UploadGroupedDescriptorTablesAcceptSparseBlankDescriptors)
{
    SKIP_IF_NO_ROCM();

    const int d_model = 128;
    const int intermediate = 128;
    const int num_experts = 4;

    auto make_desc = [](int rows, int cols, std::uintptr_t base)
    {
        DeviceNativeVNNIMatrixDesc desc{};
        desc.payload = reinterpret_cast<const uint8_t *>(base + 0x1000);
        desc.scales = reinterpret_cast<const void *>(base + 0x2000);
        desc.n = rows;
        desc.k = cols;
        desc.blocks_per_row = static_cast<uint32_t>(cols / 32);
        desc.codebook_id = 0;
        return desc;
    };

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    std::vector<DeviceNativeVNNIMatrixDesc> down_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(num_experts);
    down_descs[0] = make_desc(d_model, intermediate, 0x100000);
    down_descs[2] = make_desc(d_model, intermediate, 0x200000);
    gate_descs[0] = make_desc(intermediate, d_model, 0x300000);
    gate_descs[2] = make_desc(intermediate, d_model, 0x400000);
    up_descs[0] = make_desc(intermediate, d_model, 0x500000);
    up_descs[2] = make_desc(intermediate, d_model, 0x600000);

    EXPECT_GE(moe_kernel.uploadGroupedExpertDownDescriptorTable(
                  down_descs.data(), num_experts, d_model, intermediate),
              0);
    EXPECT_GE(moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
                  gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate),
              0);

    std::vector<DeviceNativeVNNIMatrixDesc> incomplete_gate_descs = gate_descs;
    std::vector<DeviceNativeVNNIMatrixDesc> incomplete_up_descs = up_descs;
    incomplete_gate_descs[1] = make_desc(intermediate, d_model, 0x700000);
    EXPECT_EQ(moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
                  incomplete_gate_descs.data(), incomplete_up_descs.data(),
                  num_experts, d_model, intermediate),
              -1);

    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST(Test__ROCmMoEKernel, RuntimePrefillRegroupFiltersRoutesByAssignedParticipant)
{
    SKIP_IF_NO_ROCM();

    const auto device = DeviceId::rocm(0);
    constexpr int seq_len = 3;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int total_slots = seq_len * top_k;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);

    DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(runtime_config);

    auto runtime_state = runtime_table.hostLayerState(0);
    runtime_state.participant_id = 1;
    runtime_state.participant_count = 2;
    ASSERT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(0),
                             &runtime_state,
                             sizeof(runtime_state),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    const float routing_indices_data[total_slots] = {
        0.0f, 1.0f,
        1.0f, 2.0f,
        2.0f, 3.0f};
    const float routing_weights_data[total_slots] = {
        0.50f, 0.10f,
        0.70f, 0.30f,
        0.20f, 0.80f};
    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots, routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots, routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(gpu_kernel.groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    const int32_t participant_assignments[total_slots] = {
        1, 0,
        1, 1,
        0, 1};
    ASSERT_EQ(hipMemcpyAsync(runtime_state.route_participant_ids,
                             participant_assignments,
                             sizeof(participant_assignments),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_TRUE(gpu_kernel.regroupPrefillRoutesFromRuntimeAssignments(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::array<int32_t, num_experts> counts{};
    std::array<int32_t, num_experts> offsets{};
    std::array<int32_t, total_slots> grouped_tokens{};
    std::array<float, total_slots> grouped_weights{};
    ASSERT_EQ(hipMemcpy(counts.data(), runtime_state.expert_counts,
                        sizeof(counts), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(offsets.data(), runtime_state.expert_offsets,
                        sizeof(offsets), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(grouped_tokens.data(), runtime_state.grouped_token_ids,
                        sizeof(grouped_tokens), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(grouped_weights.data(), runtime_state.grouped_route_weights,
                        sizeof(grouped_weights), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

    EXPECT_EQ(counts, (std::array<int32_t, num_experts>{1, 1, 1, 1}));
    EXPECT_EQ(offsets, (std::array<int32_t, num_experts>{0, 1, 2, 3}));
    EXPECT_EQ(grouped_tokens[0], 0);
    EXPECT_EQ(grouped_tokens[1], 1);
    EXPECT_EQ(grouped_tokens[2], 1);
    EXPECT_EQ(grouped_tokens[3], 2);
    EXPECT_EQ(grouped_tokens[4], 0);
    EXPECT_EQ(grouped_tokens[5], 0);
    EXPECT_NEAR(grouped_weights[0], 0.50f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[1], 0.70f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[2], 0.30f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[3], 0.80f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[4], 0.0f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[5], 0.0f, 1.0e-6f);
}

TEST(Test__ROCmMoEKernel, RuntimePrefillGroupingFiltersStaticOwnerLocalExperts)
{
    SKIP_IF_NO_ROCM();

    const auto device = DeviceId::rocm(0);
    constexpr int seq_len = 3;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int total_slots = seq_len * top_k;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);

    DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(runtime_config);

    auto runtime_state = runtime_table.hostLayerState(0);
    runtime_state.participant_id = 0;
    runtime_state.participant_count = 2;
    auto &bank = runtime_state.banks[runtime_state.active_bank];
    bank.expert_count = num_experts;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const bool local_ready = (expert == 0 || expert == 2);
        seedRuntimeBankExpert(bank, expert, local_ready ? 0 : 1, local_ready);
        bank.resident_participant_mask[expert] =
            local_ready ? 0b01u : 0b10u;
    }
    ASSERT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(0),
                             &runtime_state,
                             sizeof(runtime_state),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    const float routing_indices_data[total_slots] = {
        0.0f, 1.0f,
        2.0f, 3.0f,
        2.0f, 1.0f};
    const float routing_weights_data[total_slots] = {
        0.50f, 0.10f,
        0.70f, 0.30f,
        0.20f, 0.80f};
    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots, routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots, routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device, stream));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device, stream));

    ASSERT_TRUE(gpu_kernel.groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        /*filter_to_local_runtime_experts=*/true));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::array<int32_t, total_slots> route_experts{};
    std::array<int32_t, total_slots> route_participants{};
    std::array<int32_t, num_experts> counts{};
    std::array<int32_t, num_experts> offsets{};
    std::array<int32_t, total_slots> grouped_tokens{};
    std::array<float, total_slots> grouped_weights{};
    ASSERT_EQ(hipMemcpy(route_experts.data(), runtime_state.route_expert_ids,
                        sizeof(route_experts), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(route_participants.data(), runtime_state.route_participant_ids,
                        sizeof(route_participants), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(counts.data(), runtime_state.expert_counts,
                        sizeof(counts), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(offsets.data(), runtime_state.expert_offsets,
                        sizeof(offsets), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(grouped_tokens.data(), runtime_state.grouped_token_ids,
                        sizeof(grouped_tokens), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(grouped_weights.data(), runtime_state.grouped_route_weights,
                        sizeof(grouped_weights), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

    EXPECT_EQ(route_experts, (std::array<int32_t, total_slots>{0, -1, 2, -1, 2, -1}));
    EXPECT_EQ(route_participants, (std::array<int32_t, total_slots>{0, -1, 0, -1, 0, -1}));
    EXPECT_EQ(counts, (std::array<int32_t, num_experts>{1, 0, 2, 0}));
    EXPECT_EQ(offsets, (std::array<int32_t, num_experts>{0, 1, 1, 3}));
    EXPECT_EQ(grouped_tokens[0], 0);
    EXPECT_EQ(grouped_tokens[1], 2);
    EXPECT_EQ(grouped_tokens[2], 4);
    EXPECT_EQ(grouped_tokens[3], 0);
    EXPECT_EQ(grouped_tokens[4], 0);
    EXPECT_EQ(grouped_tokens[5], 0);
    EXPECT_NEAR(grouped_weights[0], 0.50f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[1], 0.70f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[2], 0.20f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[3], 0.0f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[4], 0.0f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[5], 0.0f, 1.0e-6f);
}

TEST(Test__ROCmMoEKernel, RuntimePrefillLeastLoadedResidentAssignmentBalancesHotReplicas)
{
    SKIP_IF_NO_ROCM();

    const auto device = DeviceId::rocm(0);
    constexpr int seq_len = 3;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int total_slots = seq_len * top_k;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);

    DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(runtime_config);

    auto runtime_state = runtime_table.hostLayerState(0);
    runtime_state.participant_id = 1;
    runtime_state.participant_count = 2;
    auto &bank = runtime_state.banks[runtime_state.active_bank];
    bank.expert_count = num_experts;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        bank.experts[expert].logical_expert_id = expert;
        bank.experts[expert].owner_participant = expert == 1 ? 1 : 0;
        bank.experts[expert].flags =
            toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                             DeviceMoEExpertFlags::Resident);
    }
    bank.resident_participant_mask[0] = 0b11u;
    bank.resident_participant_mask[1] = 0b10u;
    bank.resident_participant_mask[2] = 0b11u;
    bank.resident_participant_mask[3] = 0b01u;

    ASSERT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(0),
                             &runtime_state,
                             sizeof(runtime_state),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    const float routing_indices_data[total_slots] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        2.0f, 2.0f};
    const float routing_weights_data[total_slots] = {
        0.50f, 0.25f,
        0.75f, 0.10f,
        0.60f, 0.30f};
    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots, routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots, routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(gpu_kernel.groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_TRUE(gpu_kernel.assignPrefillRoutesLeastLoadedResident(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_TRUE(gpu_kernel.regroupPrefillRoutesFromRuntimeAssignments(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::array<int32_t, total_slots> route_participants{};
    std::array<int32_t, num_experts> counts{};
    std::array<int32_t, num_experts> offsets{};
    std::array<int32_t, total_slots> grouped_tokens{};
    std::array<float, total_slots> grouped_weights{};
    ASSERT_EQ(hipMemcpy(route_participants.data(), runtime_state.route_participant_ids,
                        sizeof(route_participants), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(counts.data(), runtime_state.expert_counts,
                        sizeof(counts), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(offsets.data(), runtime_state.expert_offsets,
                        sizeof(offsets), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(grouped_tokens.data(), runtime_state.grouped_token_ids,
                        sizeof(grouped_tokens), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(grouped_weights.data(), runtime_state.grouped_route_weights,
                        sizeof(grouped_weights), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

    std::array<std::array<int, 2>, num_experts> per_expert_participant_counts{};
    for (int slot = 0; slot < total_slots; ++slot)
    {
        const int expert = static_cast<int>(routing_indices_data[slot]);
        const int participant = route_participants[slot];
        ASSERT_GE(participant, 0);
        ASSERT_LT(participant, 2);
        ASSERT_NE(bank.resident_participant_mask[expert] & (1u << static_cast<uint32_t>(participant)), 0u);
        ++per_expert_participant_counts[expert][participant];
    }
    EXPECT_EQ(per_expert_participant_counts[0][0], 2);
    EXPECT_EQ(per_expert_participant_counts[0][1], 1);
    EXPECT_EQ(per_expert_participant_counts[1][0], 0);
    EXPECT_EQ(per_expert_participant_counts[1][1], 1);
    EXPECT_EQ(per_expert_participant_counts[2][0], 1);
    EXPECT_EQ(per_expert_participant_counts[2][1], 1);
    EXPECT_EQ(per_expert_participant_counts[3][0], 0);
    EXPECT_EQ(per_expert_participant_counts[3][1], 0);

    EXPECT_EQ(counts, (std::array<int32_t, num_experts>{1, 1, 1, 0}));
    EXPECT_EQ(offsets, (std::array<int32_t, num_experts>{0, 1, 2, 3}));
    std::vector<std::pair<int, float>> expected_local_grouped;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        for (int slot = 0; slot < total_slots; ++slot)
        {
            if (static_cast<int>(routing_indices_data[slot]) == expert &&
                route_participants[slot] == 1)
            {
                expected_local_grouped.emplace_back(slot / top_k, routing_weights_data[slot]);
            }
        }
    }
    ASSERT_EQ(expected_local_grouped.size(), 3u);
    EXPECT_EQ(grouped_tokens[0], expected_local_grouped[0].first);
    EXPECT_EQ(grouped_tokens[1], expected_local_grouped[1].first);
    EXPECT_EQ(grouped_tokens[2], expected_local_grouped[2].first);
    EXPECT_EQ(grouped_tokens[3], 0);
    EXPECT_EQ(grouped_tokens[4], 0);
    EXPECT_EQ(grouped_tokens[5], 0);
    EXPECT_NEAR(grouped_weights[0], expected_local_grouped[0].second, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[1], expected_local_grouped[1].second, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[2], expected_local_grouped[2].second, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[3], 0.0f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[4], 0.0f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[5], 0.0f, 1.0e-6f);
}

TEST(Test__ROCmMoEKernel, RuntimePrefillLeastLoadedCurrentBatchPlannerMaterializesSpans)
{
    SKIP_IF_NO_ROCM();

    const auto device = DeviceId::rocm(0);
    constexpr int seq_len = 8;
    constexpr int num_experts = 4;
    constexpr int top_k = 1;
    constexpr int total_slots = seq_len * top_k;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);

    DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(runtime_config);

    auto runtime_state = runtime_table.hostLayerState(0);
    runtime_state.participant_id = 0;
    runtime_state.participant_count = 2;
    auto &bank = runtime_state.banks[runtime_state.active_bank];
    bank.expert_count = num_experts;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int owner = expert < 2 ? 0 : 1;
        seedRuntimeBankExpert(bank, expert, owner, owner == static_cast<int>(runtime_state.participant_id));
        bank.resident_participant_mask[expert] = 1u << static_cast<uint32_t>(owner);
    }
    ASSERT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(0),
                             &runtime_state,
                             sizeof(runtime_state),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    const float routing_indices_data[total_slots] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 2.0f};
    const float routing_weights_data[total_slots] = {
        1.0f, 0.9f, 0.8f, 0.7f,
        0.6f, 0.5f, 0.4f, 0.3f};
    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots, routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots, routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(gpu_kernel.groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    least_loaded_ep::LeastLoadedExpertAssignmentConfig config;
    config.expert_count = num_experts;
    config.participant_count = 2;
    config.enable_balanced_skip = false;
    ASSERT_TRUE(gpu_kernel.planPrefillRoutesLeastLoadedCurrentBatch(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        config));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoELayerRuntime device_runtime_state{};
    ASSERT_EQ(hipMemcpy(&device_runtime_state,
                        runtime_table.deviceLayerState(0),
                        sizeof(device_runtime_state),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(device_runtime_state.reserved_u64[2], 4u);
    EXPECT_EQ(device_runtime_state.reserved_u64[3], 1u);

    std::array<least_loaded_ep::LeastLoadedExpertAssignmentSpan, 4> spans{};
    std::array<least_loaded_ep::LeastLoadedExpertWeightTransfer, 1> transfers{};
    std::array<int32_t, total_slots> route_participants{};
    ASSERT_EQ(hipMemcpy(spans.data(), runtime_state.reserved_ptrs[1],
                        sizeof(spans), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(transfers.data(), runtime_state.reserved_ptrs[2],
                        sizeof(transfers), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(route_participants.data(), runtime_state.route_participant_ids,
                        sizeof(route_participants), hipMemcpyDeviceToHost),
              hipSuccess);

    EXPECT_EQ(spans[0].expert, 0u);
    EXPECT_EQ(spans[0].owner_participant, 0u);
    EXPECT_EQ(spans[0].destination_participant, 0u);
    EXPECT_EQ(spans[0].route_row_begin, 0u);
    EXPECT_EQ(spans[0].route_row_end, 3u);
    EXPECT_EQ(spans[0].needs_foreign_weight, 0u);
    EXPECT_EQ(spans[1].expert, 0u);
    EXPECT_EQ(spans[1].owner_participant, 0u);
    EXPECT_EQ(spans[1].destination_participant, 1u);
    EXPECT_EQ(spans[1].route_row_begin, 3u);
    EXPECT_EQ(spans[1].route_row_end, 6u);
    EXPECT_EQ(spans[1].needs_foreign_weight, 1u);
    EXPECT_EQ(spans[2].expert, 1u);
    EXPECT_EQ(spans[2].destination_participant, 0u);
    EXPECT_EQ(spans[2].route_row_begin, 0u);
    EXPECT_EQ(spans[2].route_row_end, 1u);
    EXPECT_EQ(spans[3].expert, 2u);
    EXPECT_EQ(spans[3].destination_participant, 1u);
    EXPECT_EQ(spans[3].route_row_begin, 0u);
    EXPECT_EQ(spans[3].route_row_end, 1u);
    EXPECT_EQ(transfers[0].expert, 0u);
    EXPECT_EQ(transfers[0].source_participant, 0u);
    EXPECT_EQ(transfers[0].destination_participant, 1u);

    DeviceMoERebalanceConfig rebalance_config;
    rebalance_config.num_layers = 1;
    rebalance_config.num_experts = num_experts;
    rebalance_config.top_k = top_k;
    rebalance_config.participant_id = 0;
    rebalance_config.participant_count = 2;
    rebalance_config.root_participant = 0;
    rebalance_config.window_size_tokens = 256;

    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_header = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    DeviceMoERebalanceApplyStatus *d_apply_status = nullptr;
    ASSERT_EQ(hipMalloc(&d_plan, 4 * sizeof(DeviceMoERebalancePlanEntry)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(&d_plan_count, sizeof(uint32_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_header, sizeof(DeviceMoERebalanceCommandBufferHeader)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(&d_status, sizeof(DeviceMoERebalanceStatus)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(&d_apply_status, sizeof(DeviceMoERebalanceApplyStatus)),
              hipSuccess);
    ASSERT_TRUE(gpu_kernel.materializePrefillLeastLoadedTransferCommands(
        runtime_table.deviceLayerState(0),
        d_plan,
        d_plan_count,
        4,
        d_header,
        d_status,
        rebalance_config,
        2,
        0));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    uint32_t materialized_plan_count = 0;
    std::array<DeviceMoERebalancePlanEntry, 4> materialized_plan{};
    DeviceMoERebalanceCommandBufferHeader materialized_header{};
    DeviceMoERebalanceStatus materialized_status{};
    ASSERT_EQ(hipMemcpy(&materialized_plan_count, d_plan_count, sizeof(uint32_t),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(materialized_plan.data(), d_plan,
                        sizeof(materialized_plan), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&materialized_header, d_header, sizeof(materialized_header),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&materialized_status, d_status, sizeof(materialized_status),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(materialized_plan_count, 1u);
    EXPECT_EQ(materialized_header.magic, kDeviceMoERebalanceMagic);
    EXPECT_EQ(materialized_header.version, kDeviceMoERebalanceVersion);
    EXPECT_EQ(materialized_header.command_count, 1u);
    EXPECT_EQ(materialized_header.command_capacity, 4u);
    EXPECT_EQ(materialized_header.participant_id, 0u);
    EXPECT_EQ(materialized_header.participant_count, 2u);
    EXPECT_EQ(materialized_plan[0].op,
              static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival));
    EXPECT_EQ(materialized_plan[0].layer, 0u);
    EXPECT_EQ(materialized_plan[0].expert, 0u);
    EXPECT_EQ(materialized_plan[0].source_participant, 0u);
    EXPECT_EQ(materialized_plan[0].destination_participant, 1u);
    EXPECT_EQ(materialized_plan[0].destination_slot, 0u);
    EXPECT_EQ(materialized_plan[0].payload_slot, 0u);
    EXPECT_NE(materialized_plan[0].source_resident_mask & 0b01u, 0u);
    EXPECT_EQ(materialized_status.planned_arrivals, 1u);
    EXPECT_EQ(materialized_status.plan_overflow, 0u);
    EXPECT_EQ(materialized_status.payload_bucket_requested_slots, 1u);
    EXPECT_EQ(materialized_status.payload_bucket_slots, 1u);
    EXPECT_EQ(materialized_status.payload_source_participant_mask, 0b01u);
    EXPECT_EQ(materialized_status.payload_destination_participant_mask, 0b10u);
    EXPECT_EQ(materialized_status.payload_edge_mask,
              moe_rebalance_policy::directedParticipantEdgeBit(
                  0u, 1u, kDeviceMoEMaxParticipants));
    EXPECT_EQ(materialized_status.llep_assignment_span_count, 4u);
    EXPECT_EQ(materialized_status.llep_weight_transfer_count, 1u);
    DeviceMoERebalanceApplyStatus apply_status;
    apply_status.plan_entries_seen = materialized_status.planned_arrivals;
    apply_status.applied_arrivals = 0u;
    apply_status.changed_layers = materialized_status.planned_arrivals > 0u ? 1u : 0u;
    ASSERT_EQ(hipMemcpy(d_apply_status, &apply_status, sizeof(apply_status),
                        hipMemcpyHostToDevice),
              hipSuccess);

    ASSERT_TRUE(gpu_kernel.assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipMemcpy(route_participants.data(), runtime_state.route_participant_ids,
                        sizeof(route_participants), hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_TRUE(std::all_of(route_participants.begin(), route_participants.end(),
                            [](int32_t participant) { return participant == 0; }))
        << "guarded current-batch LLEP apply must not rewrite assignments while weight transfers are pending";

    ASSERT_EQ(hipMemcpy(&device_runtime_state,
                        runtime_table.deviceLayerState(0),
                        sizeof(device_runtime_state),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    device_runtime_state
        .banks[device_runtime_state.active_bank]
        .resident_participant_mask[0] |= 0b10u;
    ASSERT_EQ(hipMemcpy(runtime_table.deviceLayerState(0),
                        &device_runtime_state,
                        sizeof(device_runtime_state),
                        hipMemcpyHostToDevice),
              hipSuccess);

    ASSERT_TRUE(gpu_kernel.assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        d_status,
        d_apply_status));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipMemcpy(route_participants.data(), runtime_state.route_participant_ids,
                        sizeof(route_participants), hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(route_participants[0], 0);
    EXPECT_EQ(route_participants[1], 0);
    EXPECT_EQ(route_participants[2], 0);
    EXPECT_EQ(route_participants[3], 1);
    EXPECT_EQ(route_participants[4], 1);
    EXPECT_EQ(route_participants[5], 1);
    EXPECT_EQ(route_participants[6], 0);
    EXPECT_EQ(route_participants[7], 1);
    ASSERT_EQ(hipFree(d_apply_status), hipSuccess);
    ASSERT_EQ(hipFree(d_status), hipSuccess);
    ASSERT_EQ(hipFree(d_header), hipSuccess);
    ASSERT_EQ(hipFree(d_plan_count), hipSuccess);
    ASSERT_EQ(hipFree(d_plan), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, RuntimePrefillLeastLoadedCurrentBatchPlanUsesSymmetricResidency)
{
    SKIP_IF_NO_ROCM();

    const auto device = DeviceId::rocm(0);
    constexpr int seq_len = 12;
    constexpr int num_experts = 4;
    constexpr int top_k = 1;
    constexpr int total_slots = seq_len * top_k;
    constexpr size_t max_spans = 8;
    constexpr size_t max_transfers = 4;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);

    struct PlanSnapshot
    {
        uint64_t span_count = 0;
        uint64_t transfer_count = 0;
        std::array<least_loaded_ep::LeastLoadedExpertAssignmentSpan, max_spans> spans{};
        std::array<least_loaded_ep::LeastLoadedExpertWeightTransfer, max_transfers> transfers{};
    };

    const std::array<float, total_slots> routing_indices_data = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 2.0f, 2.0f, 3.0f};
    const std::array<float, total_slots> routing_weights_data = {
        1.0f, 0.9f, 0.8f, 0.7f,
        0.6f, 0.5f, 0.4f, 0.3f,
        0.2f, 0.2f, 0.1f, 0.1f};

    auto run_plan = [&](uint32_t participant_id) -> PlanSnapshot
    {
        DeviceMoERuntimeTable::Config runtime_config;
        runtime_config.device_id = device;
        runtime_config.num_layers = 1;
        runtime_config.num_experts = num_experts;
        runtime_config.top_k = top_k;
        runtime_config.mirror_to_device = true;
        runtime_config.prefill_token_capacity = seq_len;
        MoERuntimeTable runtime_table(runtime_config);

        auto runtime_state = runtime_table.hostLayerState(0);
        runtime_state.participant_id = participant_id;
        runtime_state.participant_count = 2;
        auto &bank = runtime_state.banks[runtime_state.active_bank];
        bank.expert_count = num_experts;
        const int owners[num_experts] = {0, 0, 1, 1};
        const uint32_t resident_masks[num_experts] = {0b11u, 0b01u, 0b10u, 0b11u};
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const bool locally_resident =
                (resident_masks[expert] & (1u << participant_id)) != 0u;
            const bool local_ready =
                locally_resident && !(participant_id == 0u && expert == 0);
            seedRuntimeBankExpert(bank, expert, owners[expert], local_ready);
            bank.local_compute_mask[expert] = local_ready ? 1u : 0u;
            bank.replica_role[expert] =
                local_ready
                    ? static_cast<uint8_t>(
                          owners[expert] == static_cast<int>(participant_id)
                              ? DeviceMoEReplicaRole::Primary
                              : DeviceMoEReplicaRole::Replica)
                    : static_cast<uint8_t>(DeviceMoEReplicaRole::None);
            bank.resident_participant_mask[expert] = resident_masks[expert];
        }
        EXPECT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(0),
                                 &runtime_state,
                                 sizeof(runtime_state),
                                 hipMemcpyHostToDevice,
                                 stream),
                  hipSuccess);

        auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
        auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
        std::copy(routing_indices_data.begin(), routing_indices_data.end(), routing_indices->mutable_data());
        std::copy(routing_weights_data.begin(), routing_weights_data.end(), routing_weights->mutable_data());
        EXPECT_TRUE(routing_indices->ensureOnDevice(device));
        EXPECT_TRUE(routing_weights->ensureOnDevice(device));

        EXPECT_TRUE(gpu_kernel.groupPrefillRoutes(
            runtime_table.deviceLayerState(0),
            routing_indices.get(),
            routing_weights.get(),
            seq_len,
            seq_len,
            num_experts,
            top_k));

        least_loaded_ep::LeastLoadedExpertAssignmentConfig config;
        config.expert_count = num_experts;
        config.participant_count = 2;
        config.enable_balanced_skip = false;
        EXPECT_TRUE(gpu_kernel.planPrefillRoutesLeastLoadedCurrentBatch(
            runtime_table.deviceLayerState(0),
            seq_len,
            seq_len,
            num_experts,
            top_k,
            config));
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        DeviceMoELayerRuntime device_runtime_state{};
        EXPECT_EQ(hipMemcpy(&device_runtime_state,
                            runtime_table.deviceLayerState(0),
                            sizeof(device_runtime_state),
                            hipMemcpyDeviceToHost),
                  hipSuccess);
        PlanSnapshot snapshot;
        snapshot.span_count = device_runtime_state.reserved_u64[2];
        snapshot.transfer_count = device_runtime_state.reserved_u64[3];
        EXPECT_LE(snapshot.span_count, max_spans);
        EXPECT_LE(snapshot.transfer_count, max_transfers);
        EXPECT_EQ(hipMemcpy(snapshot.spans.data(), runtime_state.reserved_ptrs[1],
                            sizeof(snapshot.spans), hipMemcpyDeviceToHost),
                  hipSuccess);
        EXPECT_EQ(hipMemcpy(snapshot.transfers.data(), runtime_state.reserved_ptrs[2],
                            sizeof(snapshot.transfers), hipMemcpyDeviceToHost),
                  hipSuccess);
        return snapshot;
    };

    const PlanSnapshot participant0 = run_plan(0);
    const PlanSnapshot participant1 = run_plan(1);
    ASSERT_GT(participant0.span_count, 0u);
    EXPECT_EQ(participant0.span_count, participant1.span_count);
    EXPECT_EQ(participant0.transfer_count, 0u);
    EXPECT_EQ(participant1.transfer_count, 0u);
    for (uint64_t i = 0; i < participant0.span_count; ++i)
    {
        EXPECT_EQ(participant0.spans[i].expert, participant1.spans[i].expert) << "span " << i;
        EXPECT_EQ(participant0.spans[i].owner_participant, participant1.spans[i].owner_participant) << "span " << i;
        EXPECT_EQ(participant0.spans[i].destination_participant, participant1.spans[i].destination_participant) << "span " << i;
        EXPECT_EQ(participant0.spans[i].route_row_begin, participant1.spans[i].route_row_begin) << "span " << i;
        EXPECT_EQ(participant0.spans[i].route_row_end, participant1.spans[i].route_row_end) << "span " << i;
        EXPECT_EQ(participant0.spans[i].needs_foreign_weight, participant1.spans[i].needs_foreign_weight) << "span " << i;
        EXPECT_EQ(participant0.spans[i].forced, participant1.spans[i].forced) << "span " << i;
    }
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, RuntimePrefillLeastLoadedTransferCommandsPreserveTransferOrder)
{
    SKIP_IF_NO_ROCM();

    const auto device = DeviceId::rocm(0);
    constexpr int seq_len = 30;
    constexpr int num_experts = 4;
    constexpr int top_k = 1;
    constexpr int total_slots = seq_len * top_k;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);

    DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(runtime_config);

    auto runtime_state = runtime_table.hostLayerState(0);
    runtime_state.participant_id = 0;
    runtime_state.participant_count = 3;
    auto &bank = runtime_state.banks[runtime_state.active_bank];
    bank.expert_count = num_experts;
    const int owners[num_experts] = {0, 0, 1, 2};
    for (int expert = 0; expert < num_experts; ++expert)
    {
        bank.experts[expert].logical_expert_id = expert;
        bank.experts[expert].owner_participant = owners[expert];
        bank.experts[expert].flags =
            toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                             DeviceMoEExpertFlags::Resident);
        bank.resident_participant_mask[expert] =
            1u << static_cast<uint32_t>(owners[expert]);
    }
    ASSERT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(0),
                             &runtime_state,
                             sizeof(runtime_state),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    std::vector<float> routing_indices_data(total_slots, 0.0f);
    std::vector<float> routing_weights_data(total_slots, 1.0f);
    for (int slot = 15; slot < total_slots; ++slot)
        routing_indices_data[static_cast<size_t>(slot)] = 1.0f;

    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data.begin(), routing_indices_data.end(),
              routing_indices->mutable_data());
    std::copy(routing_weights_data.begin(), routing_weights_data.end(),
              routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(gpu_kernel.groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    least_loaded_ep::LeastLoadedExpertAssignmentConfig config;
    config.expert_count = num_experts;
    config.participant_count = 3;
    config.enable_balanced_skip = false;

    DeviceMoERebalanceConfig rebalance_config;
    rebalance_config.num_layers = 1;
    rebalance_config.num_experts = num_experts;
    rebalance_config.top_k = top_k;
    rebalance_config.participant_id = 0;
    rebalance_config.participant_count = 3;
    rebalance_config.root_participant = 0;
    rebalance_config.window_size_tokens = 256;

    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_header = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(hipMalloc(&d_plan, 4 * sizeof(DeviceMoERebalancePlanEntry)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(&d_plan_count, sizeof(uint32_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_header, sizeof(DeviceMoERebalanceCommandBufferHeader)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(&d_status, sizeof(DeviceMoERebalanceStatus)),
              hipSuccess);

    config.max_weight_transfers = 1;
    ASSERT_TRUE(gpu_kernel.planPrefillRoutesLeastLoadedCurrentBatch(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        config));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(hipMemcpy(&runtime_state,
                        runtime_table.deviceLayerState(0),
                        sizeof(runtime_state),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(runtime_state.reserved_u64[3], 1u)
        << "planner must honor max_weight_transfers before compact payload materialization";

    ASSERT_TRUE(gpu_kernel.materializePrefillLeastLoadedTransferCommands(
        runtime_table.deviceLayerState(0),
        d_plan,
        d_plan_count,
        4,
        d_header,
        d_status,
        rebalance_config,
        1,
        0));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    uint32_t materialized_plan_count = 0;
    std::array<DeviceMoERebalancePlanEntry, 4> materialized_plan{};
    DeviceMoERebalanceStatus materialized_status{};
    ASSERT_EQ(hipMemcpy(&materialized_plan_count, d_plan_count, sizeof(uint32_t),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(materialized_plan.data(), d_plan,
                        sizeof(materialized_plan), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&materialized_status, d_status, sizeof(materialized_status),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(materialized_plan_count, 1u);
    EXPECT_EQ(materialized_status.planned_arrivals, 1u);
    EXPECT_EQ(materialized_status.plan_overflow, 0u);
    EXPECT_EQ(materialized_status.payload_bucket_requested_slots, 1u);
    EXPECT_EQ(materialized_status.payload_bucket_overflow, 0u);
    EXPECT_EQ(materialized_status.llep_weight_transfer_count, 1u);

    config.max_weight_transfers = 0;
    ASSERT_TRUE(gpu_kernel.planPrefillRoutesLeastLoadedCurrentBatch(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        config));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(hipMemcpy(&runtime_state,
                        runtime_table.deviceLayerState(0),
                        sizeof(runtime_state),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(runtime_state.reserved_u64[2], 4u);
    EXPECT_EQ(runtime_state.reserved_u64[3], 3u);
    ASSERT_TRUE(gpu_kernel.materializePrefillLeastLoadedTransferCommands(
        runtime_table.deviceLayerState(0),
        d_plan,
        d_plan_count,
        4,
        d_header,
        d_status,
        rebalance_config,
        4,
        0));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(hipMemcpy(&materialized_plan_count, d_plan_count, sizeof(uint32_t),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(materialized_plan.data(), d_plan,
                        sizeof(materialized_plan), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&materialized_status, d_status, sizeof(materialized_status),
                        hipMemcpyDeviceToHost),
              hipSuccess);

    ASSERT_EQ(materialized_plan_count, 3u);
    EXPECT_EQ(materialized_plan[0].expert, 0u);
    EXPECT_EQ(materialized_plan[0].source_participant, 0u);
    EXPECT_EQ(materialized_plan[0].destination_participant, 1u);
    EXPECT_EQ(materialized_plan[0].destination_slot, 0u);
    EXPECT_EQ(materialized_plan[0].payload_slot, 0u);
    EXPECT_EQ(materialized_plan[1].expert, 0u);
    EXPECT_EQ(materialized_plan[1].source_participant, 0u);
    EXPECT_EQ(materialized_plan[1].destination_participant, 2u);
    EXPECT_EQ(materialized_plan[1].destination_slot, 0u);
    EXPECT_EQ(materialized_plan[1].payload_slot, 1u);
    EXPECT_EQ(materialized_plan[2].expert, 1u);
    EXPECT_EQ(materialized_plan[2].source_participant, 0u);
    EXPECT_EQ(materialized_plan[2].destination_participant, 2u);
    EXPECT_EQ(materialized_plan[2].destination_slot, 1u);
    EXPECT_EQ(materialized_plan[2].payload_slot, 2u);
    EXPECT_EQ(materialized_status.planned_arrivals, 3u);
    EXPECT_EQ(materialized_status.plan_overflow, 0u);
    EXPECT_EQ(materialized_status.payload_bucket_requested_slots, 3u);
    EXPECT_EQ(materialized_status.payload_bucket_slots, 4u);
    EXPECT_EQ(materialized_status.payload_source_participant_mask, 0b001u);
    EXPECT_EQ(materialized_status.payload_destination_participant_mask, 0b110u);
    EXPECT_EQ(materialized_status.llep_assignment_span_count, 4u);
    EXPECT_EQ(materialized_status.llep_weight_transfer_count, 3u);

    ASSERT_EQ(hipFree(d_status), hipSuccess);
    ASSERT_EQ(hipFree(d_header), hipSuccess);
    ASSERT_EQ(hipFree(d_plan_count), hipSuccess);
    ASSERT_EQ(hipFree(d_plan), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, RuntimePrefillLeastLoadedStandardPlanAssignsOwnerRoutes)
{
    SKIP_IF_NO_ROCM();

    const auto device = DeviceId::rocm(0);
    constexpr int seq_len = 4;
    constexpr int num_experts = 2;
    constexpr int top_k = 1;
    constexpr int total_slots = seq_len * top_k;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);

    DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(runtime_config);

    auto runtime_state = runtime_table.hostLayerState(0);
    runtime_state.participant_id = 0;
    runtime_state.participant_count = 2;
    auto &bank = runtime_state.banks[runtime_state.active_bank];
    bank.expert_count = num_experts;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int owner = expert;
        seedRuntimeBankExpert(bank, expert, owner, owner == static_cast<int>(runtime_state.participant_id));
        bank.resident_participant_mask[expert] = 1u << static_cast<uint32_t>(owner);
    }
    ASSERT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(0),
                             &runtime_state,
                             sizeof(runtime_state),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    const float routing_indices_data[total_slots] = {0.0f, 0.0f, 1.0f, 1.0f};
    const float routing_weights_data[total_slots] = {1.0f, 0.9f, 0.8f, 0.7f};
    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots,
              routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots,
              routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(gpu_kernel.groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    least_loaded_ep::LeastLoadedExpertAssignmentConfig config;
    config.expert_count = num_experts;
    config.participant_count = 2;
    config.enable_balanced_skip = true;
    ASSERT_TRUE(gpu_kernel.planPrefillRoutesLeastLoadedCurrentBatch(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        config));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipMemcpy(&runtime_state,
                        runtime_table.deviceLayerState(0),
                        sizeof(runtime_state),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(runtime_state.reserved_u64[2], 0u);
    EXPECT_EQ(runtime_state.reserved_u64[3], 0u);

    ASSERT_TRUE(gpu_kernel.assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::array<int32_t, total_slots> route_participants{};
    ASSERT_EQ(hipMemcpy(route_participants.data(),
                        runtime_state.route_participant_ids,
                        sizeof(route_participants),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(route_participants[0], 0);
    EXPECT_EQ(route_participants[1], 0);
    EXPECT_EQ(route_participants[2], 1);
    EXPECT_EQ(route_participants[3], 1);

    ASSERT_TRUE(gpu_kernel.regroupPrefillRoutesFromRuntimeAssignments(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipMemcpy(&runtime_state,
                        runtime_table.deviceLayerState(0),
                        sizeof(runtime_state),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    std::array<int32_t, num_experts> expert_counts{};
    ASSERT_EQ(hipMemcpy(expert_counts.data(),
                        runtime_state.expert_counts,
                        sizeof(expert_counts),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(expert_counts[0], 2);
    EXPECT_EQ(expert_counts[1], 0);

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, RuntimePrefillLeastLoadedUncoveredSpanRowsFallBackToOwner)
{
    SKIP_IF_NO_ROCM();

    const auto device = DeviceId::rocm(0);
    constexpr int seq_len = 4;
    constexpr int num_experts = 2;
    constexpr int top_k = 1;
    constexpr int total_slots = seq_len * top_k;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);

    DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(runtime_config);

    auto runtime_state = runtime_table.hostLayerState(0);
    runtime_state.participant_id = 0;
    runtime_state.participant_count = 2;
    auto &bank = runtime_state.banks[runtime_state.active_bank];
    bank.expert_count = num_experts;
    seedRuntimeBankExpert(bank, 0, 0, true);
    seedRuntimeBankExpert(bank, 1, 1, false);
    bank.resident_participant_mask[0] = 0b01u;
    bank.resident_participant_mask[1] = 0b10u;
    ASSERT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(0),
                             &runtime_state,
                             sizeof(runtime_state),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    const float routing_indices_data[total_slots] = {0.0f, 1.0f, 0.0f, 1.0f};
    const float routing_weights_data[total_slots] = {1.0f, 0.9f, 0.8f, 0.7f};
    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots,
              routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots,
              routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(gpu_kernel.groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::array<least_loaded_ep::LeastLoadedExpertAssignmentSpan, 1> spans{};
    spans[0].expert = 0u;
    spans[0].owner_participant = 0u;
    spans[0].destination_participant = 0u;
    spans[0].route_row_begin = 0u;
    spans[0].route_row_end = 2u;
    spans[0].needs_foreign_weight = 0u;
    const std::array<int32_t, num_experts * 2> span_bounds = {0, 1, -1, -1};

    ASSERT_EQ(hipMemcpy(runtime_state.reserved_ptrs[1],
                        spans.data(),
                        sizeof(spans),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_state.reserved_ptrs[0],
                        span_bounds.data(),
                        sizeof(span_bounds),
                        hipMemcpyHostToDevice),
              hipSuccess);
    DeviceMoELayerRuntime device_runtime_state{};
    ASSERT_EQ(hipMemcpy(&device_runtime_state,
                        runtime_table.deviceLayerState(0),
                        sizeof(device_runtime_state),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    device_runtime_state.reserved_u64[2] = spans.size();
    device_runtime_state.reserved_u64[3] = 0u;
    ASSERT_EQ(hipMemcpy(runtime_table.deviceLayerState(0),
                        &device_runtime_state,
                        sizeof(device_runtime_state),
                        hipMemcpyHostToDevice),
              hipSuccess);

    ASSERT_TRUE(gpu_kernel.assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::array<int32_t, total_slots> route_participants{};
    ASSERT_EQ(hipMemcpy(route_participants.data(),
                        runtime_state.route_participant_ids,
                        sizeof(route_participants),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(route_participants[0], 0);
    EXPECT_EQ(route_participants[1], 1);
    EXPECT_EQ(route_participants[2], 0);
    EXPECT_EQ(route_participants[3], 1);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, RuntimePrefillLeastLoadedCurrentBatchNoTransferPlanAssignsRoutes)
{
    SKIP_IF_NO_ROCM();

    const auto device = DeviceId::rocm(0);
    constexpr int seq_len = 4;
    constexpr int num_experts = 4;
    constexpr int top_k = 1;
    constexpr int total_slots = seq_len * top_k;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);

    DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(runtime_config);

    auto runtime_state = runtime_table.hostLayerState(0);
    runtime_state.participant_id = 0;
    runtime_state.participant_count = 2;
    auto &bank = runtime_state.banks[runtime_state.active_bank];
    bank.expert_count = num_experts;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int owner = expert == 1 ? 1 : 0;
        seedRuntimeBankExpert(bank, expert, owner, owner == static_cast<int>(runtime_state.participant_id));
        bank.resident_participant_mask[expert] = 1u << static_cast<uint32_t>(owner);
    }
    ASSERT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(0),
                             &runtime_state,
                             sizeof(runtime_state),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    const float routing_indices_data[total_slots] = {0.0f, 1.0f, 0.0f, 1.0f};
    const float routing_weights_data[total_slots] = {1.0f, 0.9f, 0.8f, 0.7f};
    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots, routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots, routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(gpu_kernel.groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    least_loaded_ep::LeastLoadedExpertAssignmentConfig config;
    config.expert_count = num_experts;
    config.participant_count = 2;
    config.alpha_numerator = 2;
    config.alpha_denominator = 1;
    config.enable_balanced_skip = false;
    ASSERT_TRUE(gpu_kernel.planPrefillRoutesLeastLoadedCurrentBatch(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        config));
    ASSERT_TRUE(gpu_kernel.assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoELayerRuntime device_runtime_state{};
    ASSERT_EQ(hipMemcpy(&device_runtime_state,
                        runtime_table.deviceLayerState(0),
                        sizeof(device_runtime_state),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(device_runtime_state.reserved_u64[2], 2u);
    EXPECT_EQ(device_runtime_state.reserved_u64[3], 0u);

    std::array<int32_t, total_slots> route_participants{};
    ASSERT_EQ(hipMemcpy(route_participants.data(), runtime_state.route_participant_ids,
                        sizeof(route_participants), hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(route_participants[0], 0);
    EXPECT_EQ(route_participants[1], 1);
    EXPECT_EQ(route_participants[2], 0);
    EXPECT_EQ(route_participants[3], 1);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, RuntimePrefillLeastLoadedCurrentBatchResidentReplicaAvoidsTransfer)
{
    SKIP_IF_NO_ROCM();

    const auto device = DeviceId::rocm(0);
    constexpr int seq_len = 8;
    constexpr int num_experts = 4;
    constexpr int top_k = 1;
    constexpr int total_slots = seq_len * top_k;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);

    DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(runtime_config);

    auto runtime_state = runtime_table.hostLayerState(0);
    runtime_state.participant_id = 0;
    runtime_state.participant_count = 2;
    auto &bank = runtime_state.banks[runtime_state.active_bank];
    bank.expert_count = num_experts;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int owner = expert < 2 ? 0 : 1;
        seedRuntimeBankExpert(bank, expert, owner, owner == static_cast<int>(runtime_state.participant_id));
        bank.resident_participant_mask[expert] = 1u << static_cast<uint32_t>(owner);
    }
    bank.resident_participant_mask[0] = 0b11u;
    ASSERT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(0),
                             &runtime_state,
                             sizeof(runtime_state),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    const float routing_indices_data[total_slots] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 2.0f};
    const float routing_weights_data[total_slots] = {
        1.0f, 0.9f, 0.8f, 0.7f,
        0.6f, 0.5f, 0.4f, 0.3f};
    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots, routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots, routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(gpu_kernel.groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    least_loaded_ep::LeastLoadedExpertAssignmentConfig config;
    config.expert_count = num_experts;
    config.participant_count = 2;
    config.enable_balanced_skip = false;
    ASSERT_TRUE(gpu_kernel.planPrefillRoutesLeastLoadedCurrentBatch(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        config));
    ASSERT_TRUE(gpu_kernel.assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoELayerRuntime device_runtime_state{};
    ASSERT_EQ(hipMemcpy(&device_runtime_state,
                        runtime_table.deviceLayerState(0),
                        sizeof(device_runtime_state),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(device_runtime_state.reserved_u64[2], 4u);
    EXPECT_EQ(device_runtime_state.reserved_u64[3], 0u);

    std::array<int32_t, total_slots> route_participants{};
    ASSERT_EQ(hipMemcpy(route_participants.data(), runtime_state.route_participant_ids,
                        sizeof(route_participants), hipMemcpyDeviceToHost),
              hipSuccess);
    int expert0_on_participant0 = 0;
    int expert0_on_participant1 = 0;
    for (int slot = 0; slot < total_slots; ++slot)
    {
        if (static_cast<int>(routing_indices_data[slot]) != 0)
            continue;
        if (route_participants[slot] == 0)
            ++expert0_on_participant0;
        if (route_participants[slot] == 1)
            ++expert0_on_participant1;
    }
    EXPECT_EQ(expert0_on_participant0, 3);
    EXPECT_EQ(expert0_on_participant1, 3);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

// ============================================================================
// Test: route() — Gate logits + softmax + top-k
// ============================================================================

TEST(Test__ROCmMoEKernel, Route_DecodeSmall)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 1;
    const int d_model = 2048;
    const int num_experts = 64;
    const int top_k = 8;

    // Prepare host data
    std::vector<float> hidden(seq_len * d_model);
    std::vector<float> gate_weights(num_experts * d_model);
    fillRandom(hidden, -1.0f, 1.0f, 42);
    fillRandom(gate_weights, -0.1f, 0.1f, 123);

    // CPU reference
    CPUMoEKernel cpu_kernel;
    MoERoutingResult cpu_result;
    ASSERT_TRUE(cpu_kernel.route(hidden.data(), gate_weights.data(),
                                 seq_len, d_model, num_experts, top_k,
                                 true, cpu_result));

    // GPU execution
    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);

    // Upload data to device
    float *d_hidden = nullptr, *d_gate_weights = nullptr;
    (void)hipMalloc(&d_hidden, hidden.size() * sizeof(float));
    (void)hipMalloc(&d_gate_weights, gate_weights.size() * sizeof(float));
    (void)hipMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_gate_weights, gate_weights.data(), gate_weights.size() * sizeof(float), hipMemcpyHostToDevice);

    MoERoutingResult gpu_result;
    ASSERT_TRUE(gpu_kernel.route(d_hidden, d_gate_weights,
                                 seq_len, d_model, num_experts, top_k,
                                 true, gpu_result));

    (void)hipFree(d_hidden);
    (void)hipFree(d_gate_weights);

    // Verify router logits parity
    ASSERT_EQ(gpu_result.router_logits.size(), cpu_result.router_logits.size());
    ASSERT_FALSE(hasNaNOrInf(gpu_result.router_logits.data(), gpu_result.router_logits.size()));

    double logits_cosine = cosineSimilarity(
        gpu_result.router_logits.data(), cpu_result.router_logits.data(),
        gpu_result.router_logits.size());
    expectStrictVerifierSimilarity(
        "decode router logits should match CPU routing row-by-row",
        gpu_result.router_logits.data(),
        cpu_result.router_logits.data(),
        gpu_result.router_logits.size(),
        static_cast<size_t>(num_experts));

    // Verify top-k expert selections match
    ASSERT_EQ(gpu_result.expert_indices.size(), cpu_result.expert_indices.size());
    int matching_experts = 0;
    for (size_t i = 0; i < cpu_result.expert_indices.size(); ++i)
    {
        if (gpu_result.expert_indices[i] == cpu_result.expert_indices[i])
            ++matching_experts;
    }
    double expert_match_rate = static_cast<double>(matching_experts) / cpu_result.expert_indices.size();
    EXPECT_GE(expert_match_rate, 0.75)
        << "Expert selection match rate too low: " << expert_match_rate
        << " (" << matching_experts << "/" << cpu_result.expert_indices.size() << ")";

    // Verify top-k weights parity
    ASSERT_FALSE(hasNaNOrInf(gpu_result.expert_weights.data(), gpu_result.expert_weights.size()));
    double weights_cosine = cosineSimilarity(
        gpu_result.expert_weights.data(), cpu_result.expert_weights.data(),
        gpu_result.expert_weights.size());
    EXPECT_GE(weights_cosine, 0.999)
        << "Expert weights cosine similarity too low: " << weights_cosine;

    std::cout << "[Route_DecodeSmall] logits_cosine=" << std::fixed << std::setprecision(6)
              << logits_cosine
              << " expert_match=" << matching_experts << "/" << cpu_result.expert_indices.size()
              << " weights_cosine=" << weights_cosine << std::endl;
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectRuntimeStateUpdatesTopKAndHistogram)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int d_model = 256;
    const int num_experts = 16;
    const int top_k = 4;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::vector<float> hidden_host(static_cast<size_t>(d_model));
    std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
    fillRandom(hidden_host, -1.0f, 1.0f, 7201);
    fillRandom(gate_host, -0.1f, 0.1f, 7202);
    std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
    std::copy(gate_host.begin(), gate_host.end(), gate_weights->mutable_data());

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices->ensureOnDevice(device));
    ASSERT_TRUE(output_weights->ensureOnDevice(device));

    DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, top_k);

    DeviceMoELayerRuntime *device_runtime = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_runtime), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(device_runtime, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    EXPECT_FALSE(gpu_kernel.decodeRouteSelect(
        nullptr,
        hidden.get(), gate_weights.get(),
        d_model, num_experts, top_k,
        true,
        output_indices.get(), output_weights.get(),
        true, true));

    // Runtime-table semantic validation belongs to the graph-build/stage setup
    // boundary.  decodeRouteSelect() is part of the captured hot path, so this
    // test deliberately avoids expecting a D2H validation pass here.

    ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
        device_runtime,
        hidden.get(), gate_weights.get(),
        d_model, num_experts, top_k,
        true,
        output_indices.get(), output_weights.get(),
        true, true));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceMoELayerRuntime after{};
    ASSERT_EQ(hipMemcpy(&after, device_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipFree(device_runtime), hipSuccess);

    output_indices->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *legacy_indices = output_indices->data();
    const float *legacy_weights = output_weights->data();

    uint64_t histogram_sum = 0;
    for (int expert = 0; expert < num_experts; ++expert)
        histogram_sum += after.decode_histogram[expert];
    EXPECT_EQ(histogram_sum, static_cast<uint64_t>(top_k));

    for (int k = 0; k < top_k; ++k)
    {
        const int expert_id = after.topk_expert_ids[k];
        EXPECT_GE(expert_id, 0);
        EXPECT_LT(expert_id, num_experts);
        EXPECT_FLOAT_EQ(legacy_indices[k], static_cast<float>(expert_id));
        EXPECT_NEAR(legacy_weights[k], after.topk_weights[k], 1e-6f);
        EXPECT_GT(after.topk_weights[k], 0.0f);
        ASSERT_GE(expert_id, 0);
        ASSERT_LT(expert_id, num_experts);
        EXPECT_GE(after.decode_histogram[expert_id], 1u);
    }

    float weight_sum = 0.0f;
    for (int k = 0; k < top_k; ++k)
        weight_sum += after.topk_weights[k];
    EXPECT_NEAR(weight_sum, 1.0f, 1e-4f);
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectRuntimeAssignsReplicasOnceAcrossParticipants)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int num_layers = 1;
    constexpr int num_experts = 4;
    constexpr int top_k = 4;
    constexpr int d_model = 4;

    auto fake_desc = [](uintptr_t base, int n, int k)
    {
        DeviceNativeVNNIMatrixDesc desc;
        desc.payload = reinterpret_cast<const uint8_t *>(base);
        desc.scales = reinterpret_cast<const void *>(base + 0x100u);
        desc.n = n;
        desc.k = k;
        desc.blocks_per_row = 1;
        desc.codebook_id = 7;
        return desc;
    };

    auto make_update = [&](uint32_t participant_id)
    {
        MoEPlacementUpdate update;
        update.epoch = 1;
        update.expert_count = num_experts;
        update.participant_id = participant_id;
        update.participant_count = 3;
        update.experts.resize(num_experts);
        update.local_compute_mask.assign(num_experts, 0u);
        update.replica_role.assign(num_experts, static_cast<uint8_t>(DeviceMoEReplicaRole::None));
        update.resident_participant_mask.assign(num_experts, 0u);

        for (int expert = 0; expert < num_experts; ++expert)
        {
            const bool replicated = expert >= 2;
            const int owner = (expert == 3) ? 2 : (expert % 2);
            uint32_t resident_mask = 1u << static_cast<uint32_t>(owner);
            if (expert == 2)
                resident_mask |= 1u << 1u;
            if (expert == 3)
                resident_mask |= 1u << 0u;
            const bool local =
                (resident_mask & (1u << participant_id)) != 0u;

            DeviceMoEExpertDescriptor desc;
            desc.logical_expert_id = expert;
            desc.owner_participant = owner;
            desc.local_slot = local ? expert : -1;
            if (replicated)
                desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Replicated);
            update.resident_participant_mask[expert] = resident_mask;
            if (local)
            {
                const uintptr_t base = 0x20000000u + static_cast<uintptr_t>(participant_id) * 0x100000u +
                                       static_cast<uintptr_t>(expert) * 0x1000u;
                desc.gate = fake_desc(base + 0x10u, 8, d_model);
                desc.up = fake_desc(base + 0x20u, 8, d_model);
                desc.down = fake_desc(base + 0x30u, d_model, 8);
                DeviceMoEExpertFlags flags = DeviceMoEExpertFlags::Valid |
                                             DeviceMoEExpertFlags::Resident |
                                             DeviceMoEExpertFlags::LocalCompute;
                if (owner == static_cast<int>(participant_id))
                    flags |= DeviceMoEExpertFlags::PreferredOwner;
                if (replicated)
                    flags |= DeviceMoEExpertFlags::Replicated;
                desc.flags = toMoEExpertFlags(flags);
                update.local_compute_mask[expert] = 1u;
                update.replica_role[expert] = static_cast<uint8_t>(
                    replicated
                        ? (owner == static_cast<int>(participant_id) ? DeviceMoEReplicaRole::Primary
                                                                     : DeviceMoEReplicaRole::Replica)
                        : DeviceMoEReplicaRole::Primary);
            }
            update.experts[expert] = desc;
        }
        return update;
    };

    auto make_table = [&](uint32_t participant_id)
    {
        DeviceMoERuntimeTable::Config config;
        config.device_id = device;
        config.num_layers = num_layers;
        config.num_experts = num_experts;
        config.top_k = top_k;
        config.mirror_to_device = true;
        auto table = std::make_unique<MoERuntimeTable>(config);
        auto update = make_update(participant_id);
        EXPECT_TRUE(table->prepareInactiveBank(0, update));
        EXPECT_TRUE(table->flipActiveBank(0, update.epoch, nullptr));
        return table;
    };

    auto table0 = make_table(0);
    auto table1 = make_table(1);
    auto table2 = make_table(2);

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    const std::array<float, d_model> hidden_values{1.0f, 0.0f, 0.0f, 0.0f};
    const std::array<float, num_experts * d_model> gate_values{
        4.0f, 0.0f, 0.0f, 0.0f,
        3.0f, 0.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f};
    std::copy(hidden_values.begin(), hidden_values.end(), hidden->mutable_data());
    std::copy(gate_values.begin(), gate_values.end(), gate_weights->mutable_data());

    auto output_indices0 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights0 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_indices1 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights1 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_indices2 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights2 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices0->ensureOnDevice(device));
    ASSERT_TRUE(output_weights0->ensureOnDevice(device));
    ASSERT_TRUE(output_indices1->ensureOnDevice(device));
    ASSERT_TRUE(output_weights1->ensureOnDevice(device));
    ASSERT_TRUE(output_indices2->ensureOnDevice(device));
    ASSERT_TRUE(output_weights2->ensureOnDevice(device));

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(
        gpu_kernel,
        /*max_seq_len=*/1,
        /*d_model=*/d_model,
        /*intermediate=*/8,
        /*num_experts=*/num_experts,
        /*top_k=*/top_k);
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ScopedROCmEnvOverride wave_env("LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "0");
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            table0->deviceLayerState(0),
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            false,
            output_indices0.get(), output_weights0.get(),
            true, true));
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            table1->deviceLayerState(0),
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            false,
            output_indices1.get(), output_weights1.get(),
            true, true));
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            table2->deviceLayerState(0),
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            false,
            output_indices2.get(), output_weights2.get(),
            true, true));
    }

    auto copy_runtime = [&](MoERuntimeTable &table,
                            std::array<int32_t, top_k> &ids,
                            std::array<float, top_k> &weights)
    {
        auto *layer = table.deviceLayerState(0);
        const auto *base = reinterpret_cast<const char *>(layer);
        const auto *ids_device = reinterpret_cast<const int32_t *>(
            base + offsetof(DeviceMoELayerRuntime, topk_expert_ids));
        const auto *weights_device = reinterpret_cast<const float *>(
            base + offsetof(DeviceMoELayerRuntime, topk_weights));
        ASSERT_EQ(hipMemcpy(ids.data(), ids_device, ids.size() * sizeof(int32_t),
                            hipMemcpyDeviceToHost),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy(weights.data(), weights_device, weights.size() * sizeof(float),
                            hipMemcpyDeviceToHost),
                  hipSuccess);
    };

    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    std::array<int32_t, top_k> ids0{};
    std::array<int32_t, top_k> ids1{};
    std::array<int32_t, top_k> ids2{};
    std::array<float, top_k> runtime_weights0{};
    std::array<float, top_k> runtime_weights1{};
    std::array<float, top_k> runtime_weights2{};
    copy_runtime(*table0, ids0, runtime_weights0);
    copy_runtime(*table1, ids1, runtime_weights1);
    copy_runtime(*table2, ids2, runtime_weights2);

    const std::array<int32_t, top_k> expected0{0, -1, 2, -1};
    const std::array<int32_t, top_k> expected1{-1, 1, -1, -1};
    const std::array<int32_t, top_k> expected2{-1, -1, -1, 3};
    EXPECT_EQ(ids0, expected0);
    EXPECT_EQ(ids1, expected1);
    EXPECT_EQ(ids2, expected2);
    for (int slot = 0; slot < top_k; ++slot)
        EXPECT_EQ((ids0[slot] >= 0) + (ids1[slot] >= 0) + (ids2[slot] >= 0), 1) << "slot " << slot;

    output_indices0->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_indices1->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_indices2->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    for (int slot = 0; slot < top_k; ++slot)
    {
        EXPECT_EQ(output_indices0->data()[slot], static_cast<float>(slot));
        EXPECT_EQ(output_indices1->data()[slot], static_cast<float>(slot));
        EXPECT_EQ(output_indices2->data()[slot], static_cast<float>(slot));
    }
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectRuntimeRecordsHotCacheBalanceImprovement)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int num_layers = 1;
    constexpr int num_experts = 4;
    constexpr int top_k = 4;
    constexpr int d_model = 4;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    auto fake_desc = [](uintptr_t base, int n, int k)
    {
        DeviceNativeVNNIMatrixDesc desc;
        desc.payload = reinterpret_cast<const uint8_t *>(base);
        desc.scales = reinterpret_cast<const void *>(base + 0x100u);
        desc.n = n;
        desc.k = k;
        desc.blocks_per_row = 1;
        desc.codebook_id = 7;
        return desc;
    };

    auto make_update = [&](uint32_t participant_id)
    {
        MoEPlacementUpdate update;
        update.epoch = 1;
        update.expert_count = num_experts;
        update.participant_id = participant_id;
        update.participant_count = 3;
        update.experts.resize(num_experts);
        update.local_compute_mask.assign(num_experts, 0u);
        update.replica_role.assign(num_experts, static_cast<uint8_t>(DeviceMoEReplicaRole::None));
        update.resident_participant_mask = {0b011u, 0b011u, 0b101u, 0b101u};

        for (int expert = 0; expert < num_experts; ++expert)
        {
            const uint32_t resident_mask = update.resident_participant_mask[expert];
            const bool local = (resident_mask & (1u << participant_id)) != 0u;
            DeviceMoEExpertDescriptor desc;
            desc.logical_expert_id = expert;
            desc.owner_participant = 0;
            desc.local_slot = local ? expert : -1;
            desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Replicated);
            if (local)
            {
                const uintptr_t base = 0x43000000u + static_cast<uintptr_t>(participant_id) * 0x100000u +
                                       static_cast<uintptr_t>(expert) * 0x1000u;
                desc.gate = fake_desc(base + 0x10u, 8, d_model);
                desc.up = fake_desc(base + 0x20u, 8, d_model);
                desc.down = fake_desc(base + 0x30u, d_model, 8);
                DeviceMoEExpertFlags flags = DeviceMoEExpertFlags::Valid |
                                             DeviceMoEExpertFlags::Resident |
                                             DeviceMoEExpertFlags::LocalCompute |
                                             DeviceMoEExpertFlags::Replicated;
                if (participant_id == 0)
                    flags |= DeviceMoEExpertFlags::PreferredOwner;
                desc.flags = toMoEExpertFlags(flags);
                update.local_compute_mask[expert] = 1u;
                update.replica_role[expert] = static_cast<uint8_t>(
                    participant_id == 0 ? DeviceMoEReplicaRole::Primary
                                        : DeviceMoEReplicaRole::Replica);
            }
            update.experts[expert] = desc;
        }
        return update;
    };

    auto make_table = [&](uint32_t participant_id)
    {
        DeviceMoERuntimeTable::Config config;
        config.device_id = device;
        config.num_layers = num_layers;
        config.num_experts = num_experts;
        config.top_k = top_k;
        config.mirror_to_device = true;
        auto table = std::make_unique<MoERuntimeTable>(config);
        auto update = make_update(participant_id);
        EXPECT_TRUE(table->prepareInactiveBank(0, update));
        EXPECT_TRUE(table->flipActiveBank(0, update.epoch, stream));
        return table;
    };

    auto table0 = make_table(0);
    auto table1 = make_table(1);
    auto table2 = make_table(2);

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    const std::array<float, d_model> hidden_values{1.0f, 0.0f, 0.0f, 0.0f};
    const std::array<float, num_experts * d_model> gate_values{
        4.0f, 0.0f, 0.0f, 0.0f,
        3.0f, 0.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f};
    std::copy(hidden_values.begin(), hidden_values.end(), hidden->mutable_data());
    std::copy(gate_values.begin(), gate_values.end(), gate_weights->mutable_data());

    auto output_indices0 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights0 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_indices1 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights1 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_indices2 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights2 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices0->ensureOnDevice(device));
    ASSERT_TRUE(output_weights0->ensureOnDevice(device));
    ASSERT_TRUE(output_indices1->ensureOnDevice(device));
    ASSERT_TRUE(output_weights1->ensureOnDevice(device));
    ASSERT_TRUE(output_indices2->ensureOnDevice(device));
    ASSERT_TRUE(output_weights2->ensureOnDevice(device));

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(
        gpu_kernel,
        /*max_seq_len=*/1,
        /*d_model=*/d_model,
        /*intermediate=*/8,
        /*num_experts=*/num_experts,
        /*top_k=*/top_k);
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ScopedROCmEnvOverride wave_env("LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "0");
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            table0->deviceLayerState(0),
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            false,
            output_indices0.get(), output_weights0.get(),
            true, true));
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            table1->deviceLayerState(0),
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            false,
            output_indices1.get(), output_weights1.get(),
            true, true));
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            table2->deviceLayerState(0),
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            false,
            output_indices2.get(), output_weights2.get(),
            true, true));
    }

    DeviceMoELayerRuntime runtime0{};
    DeviceMoELayerRuntime runtime1{};
    DeviceMoELayerRuntime runtime2{};
    ASSERT_EQ(hipMemcpyAsync(&runtime0, table0->deviceLayerState(0), sizeof(runtime0),
                             hipMemcpyDeviceToHost, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(&runtime1, table1->deviceLayerState(0), sizeof(runtime1),
                             hipMemcpyDeviceToHost, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(&runtime2, table2->deviceLayerState(0), sizeof(runtime2),
                             hipMemcpyDeviceToHost, stream),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    const std::array<int32_t, top_k> expected0{0, -1, -1, 3};
    const std::array<int32_t, top_k> expected1{-1, 1, -1, -1};
    const std::array<int32_t, top_k> expected2{-1, -1, 2, -1};
    const std::array<int32_t, top_k> actual0{runtime0.topk_expert_ids[0],
                                             runtime0.topk_expert_ids[1],
                                             runtime0.topk_expert_ids[2],
                                             runtime0.topk_expert_ids[3]};
    const std::array<int32_t, top_k> actual1{runtime1.topk_expert_ids[0],
                                             runtime1.topk_expert_ids[1],
                                             runtime1.topk_expert_ids[2],
                                             runtime1.topk_expert_ids[3]};
    const std::array<int32_t, top_k> actual2{runtime2.topk_expert_ids[0],
                                             runtime2.topk_expert_ids[1],
                                             runtime2.topk_expert_ids[2],
                                             runtime2.topk_expert_ids[3]};
    EXPECT_EQ(actual0, expected0);
    EXPECT_EQ(actual1, expected1);
    EXPECT_EQ(actual2, expected2);

    for (const auto *runtime : {&runtime0, &runtime1, &runtime2})
    {
        EXPECT_EQ(runtime->router_hot_cache_eligible_dispatches, 1u);
        EXPECT_EQ(runtime->router_hot_cache_used_dispatches, 1u);
        EXPECT_EQ(runtime->router_hot_cache_improved_dispatches, 1u);
        EXPECT_EQ(runtime->router_hot_cache_default_load_spread_total, 4u);
        EXPECT_EQ(runtime->router_hot_cache_actual_load_spread_total, 1u);
        EXPECT_EQ(runtime->router_hot_cache_load_spread_improvement_total, 3u);
        EXPECT_EQ(runtime->router_hot_cache_active_dispatches, 1u);
        EXPECT_EQ(runtime->router_hot_cache_miss_dispatches, 0u);
        EXPECT_EQ(runtime->router_hot_cache_selected_expert_slots, 4u);
        EXPECT_EQ(runtime->router_hot_cache_replicated_selected_expert_slots, 4u);
    }

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, TokenRowPublicationTopK2SurvivesSnapshotSync)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int kRows = 2;
    constexpr int kTopK = 2;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<ITensorKernel &>(gpu_kernel).setGPUStream(stream);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);

    auto full_routes = TestTensorFactory::createFP32({kRows, kTopK});
    auto row0 = TestTensorFactory::createFP32({kTopK});
    auto row1 = TestTensorFactory::createFP32({kTopK});
    auto copied = TestTensorFactory::createFP32({kTopK});

    std::fill(full_routes->mutable_data(),
              full_routes->mutable_data() + kRows * kTopK,
              -99.0f);
    row0->mutable_data()[0] = 7.0f;
    row0->mutable_data()[1] = 11.0f;
    row1->mutable_data()[0] = 13.0f;
    row1->mutable_data()[1] = 17.0f;
    std::fill(copied->mutable_data(),
              copied->mutable_data() + kTopK,
              -1.0f);

    ASSERT_TRUE(full_routes->ensureOnDevice(device));
    ASSERT_TRUE(row0->ensureOnDevice(device));
    ASSERT_TRUE(row1->ensureOnDevice(device));
    ASSERT_TRUE(copied->ensureOnDevice(device));

    /*
     * These row helpers are diagnostic-only now: production MTP verifier rows
     * must stay grouped.  Keep this small ROCm test as a focused tensor
     * residency/coherence probe because the original bug was a stale top_k=2
     * route-row handoff, and that device/host mirror behavior is still worth
     * locking down independently from the grouped verifier path.
     */
    ASSERT_TRUE(gpu_kernel.writeTokenRowToTensor(full_routes.get(), row0.get(), 0, kTopK));
    ASSERT_TRUE(gpu_kernel.writeTokenRowToTensor(full_routes.get(), row1.get(), 1, kTopK));

    ASSERT_TRUE(full_routes->ensureOnHost(stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    const float *published = full_routes->data();
    ASSERT_NE(published, nullptr);
    EXPECT_FLOAT_EQ(published[0], 7.0f);
    EXPECT_FLOAT_EQ(published[1], 11.0f);
    EXPECT_FLOAT_EQ(published[2], 13.0f);
    EXPECT_FLOAT_EQ(published[3], 17.0f);

    ASSERT_TRUE(gpu_kernel.copyTokenRowFromTensor(full_routes.get(), copied.get(), 0, kTopK));
    ASSERT_TRUE(copied->ensureOnHost(stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    const float *copied_row = copied->data();
    ASSERT_NE(copied_row, nullptr);
    EXPECT_FLOAT_EQ(copied_row[0], 7.0f);
    EXPECT_FLOAT_EQ(copied_row[1], 11.0f);

    ASSERT_TRUE(gpu_kernel.copyTokenRowFromTensor(full_routes.get(), copied.get(), 1, kTopK));
    ASSERT_TRUE(copied->ensureOnHost(stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    copied_row = copied->data();
    ASSERT_NE(copied_row, nullptr);
    EXPECT_FLOAT_EQ(copied_row[0], 13.0f);
    EXPECT_FLOAT_EQ(copied_row[1], 17.0f);

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DecodeRuntimeHistogramSyncMatchesHostRecordAcrossTokensAndLayers)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int num_layers = 2;
    const int d_model = 128;
    const int num_experts = 16;
    const int top_k = 4;
    const int tokens = 5;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
    fillRandom(gate_host, -0.1f, 0.1f, 8102);
    std::copy(gate_host.begin(), gate_host.end(), gate_weights->mutable_data());

    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices->ensureOnDevice(device));
    ASSERT_TRUE(output_weights->ensureOnDevice(device));

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = num_layers;
    table_config.num_experts = num_experts;
    table_config.top_k = top_k;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    auto routing_only_update = [&](uint32_t epoch)
    {
        MoEPlacementUpdate update;
        update.epoch = epoch;
        update.expert_count = static_cast<uint32_t>(num_experts);
        update.experts.resize(static_cast<size_t>(num_experts));
        update.local_compute_mask.assign(static_cast<size_t>(num_experts), 0u);
        update.replica_role.assign(static_cast<size_t>(num_experts), 0u);
        return update;
    };

    for (int layer = 0; layer < num_layers; ++layer)
    {
        ASSERT_TRUE(runtime_table.prepareInactiveBank(layer, routing_only_update(1)));
        ASSERT_TRUE(runtime_table.flipActiveBank(layer, 1, stream));
    }

    DecodeExpertHistogramConfig hist_config;
    hist_config.num_layers = num_layers;
    hist_config.num_experts = num_experts;
    hist_config.top_k = top_k;
    hist_config.window_size = 32;
    hist_config.sockets = {DeviceId(DeviceType::CPU, 0), DeviceId(DeviceType::CPU, 1)};
    hist_config.expert_to_socket.resize(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
        hist_config.expert_to_socket[expert] = expert % 2;
    DecodeExpertHistogram host_record(hist_config);
    DecodeExpertHistogram runtime_merged(hist_config);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    std::vector<float> hidden_host(static_cast<size_t>(d_model));
    for (int token = 0; token < tokens; ++token)
    {
        for (int layer = 0; layer < num_layers; ++layer)
        {
            fillRandom(hidden_host, -1.0f, 1.0f, 8200 + token * 17 + layer);
            std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
            ASSERT_TRUE(hidden->ensureOnDevice(device, stream));

            ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
                runtime_table.deviceLayerState(layer),
                hidden.get(), gate_weights.get(),
                d_model, num_experts, top_k,
                true,
                output_indices.get(), output_weights.get(),
                true, true));

            output_indices->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            output_weights->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            const float *legacy_indices = output_indices->data();
            const float *legacy_weights = output_weights->data();
            int host_indices[top_k];
            float host_weights[top_k];
            for (int k = 0; k < top_k; ++k)
            {
                host_indices[k] = static_cast<int>(legacy_indices[k]);
                host_weights[k] = legacy_weights[k];
            }
            host_record.record(layer, host_indices, host_weights, top_k);
            runtime_merged.recordTokenBoundary(layer);
        }
    }

    ASSERT_TRUE(runtime_table.syncDecodeHistogramToHost(runtime_merged, stream));

    for (int layer = 0; layer < num_layers; ++layer)
        EXPECT_EQ(runtime_merged.layerHistogram(layer), host_record.layerHistogram(layer));
    EXPECT_EQ(runtime_merged.windowTokenCount(), host_record.windowTokenCount());

    for (int layer = 0; layer < num_layers; ++layer)
    {
        const auto &state = runtime_table.hostLayerState(layer);
        for (int expert = 0; expert < num_experts; ++expert)
            EXPECT_EQ(state.decode_histogram[expert], 0u);
    }

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceControllerPublishesRuntimeBankOnStream)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream));

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;

    std::vector<uint64_t> gathered(static_cast<size_t>(config.participant_count) *
                                       config.num_layers * config.num_experts,
                                   0);
    gathered[0] = 8; // participant 0, layer 0, expert 0 is hottest.

    uint64_t *d_gathered = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered),
                        gathered.size() * sizeof(uint64_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_status),
                        sizeof(DeviceMoERebalanceStatus)),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered, gathered.data(),
                             gathered.size() * sizeof(uint64_t),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.runDeviceRebalanceController(
        runtime_table.deviceLayerState(0), d_gathered, d_status, config));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceStatus status;
    ASSERT_EQ(hipMemcpy(&status, d_status, sizeof(status), hipMemcpyDeviceToHost), hipSuccess);
    DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(hipMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime),
                        hipMemcpyDeviceToHost),
              hipSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.windows_applied, 1u);
    EXPECT_EQ(runtime.active_epoch, 2u);
    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(bank.local_compute_mask[0], 1u);
    EXPECT_TRUE(hasMoEExpertFlag(bank.experts[0].flags, DeviceMoEExpertFlags::Replicated));
    EXPECT_EQ(bank.replica_role[0], static_cast<uint8_t>(DeviceMoEReplicaRole::Replica));
    EXPECT_EQ(bank.local_compute_mask[2], 1u);
    EXPECT_EQ(bank.replica_role[2], static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));
    for (int expert = 0; expert < 4; ++expert)
    {
        EXPECT_EQ(runtime.decode_histogram[expert], 0u);
        EXPECT_EQ(runtime.decode_local_histogram[expert], 0u);
    }

    EXPECT_EQ(hipFree(d_gathered), hipSuccess);
    EXPECT_EQ(hipFree(d_status), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceControllerPlansMissingReplicaArrivals)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    update.resident_participant_mask[0] = 0b001u;
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream));

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);

    std::vector<uint64_t> gathered(static_cast<size_t>(config.participant_count) *
                                       config.num_layers * config.num_experts,
                                   0);
    gathered[0] = 8; // participant 0, layer 0, expert 0 is hottest.

    uint64_t *d_gathered = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered),
                        gathered.size() * sizeof(uint64_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_status),
                        sizeof(DeviceMoERebalanceStatus)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan),
                        sizeof(DeviceMoERebalancePlanEntry)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_count),
                        sizeof(uint32_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_header),
                        sizeof(DeviceMoERebalanceCommandBufferHeader)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_wave_state),
                        sizeof(DeviceMoERebalanceWaveState)),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered, gathered.data(),
                             gathered.size() * sizeof(uint64_t),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.runDeviceRebalanceController(
        runtime_table.deviceLayerState(0),
        d_gathered,
        d_status,
        config,
        d_plan,
        d_plan_count,
        1,
        1,
        d_command_header,
        d_wave_state));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceStatus status;
    DeviceMoERebalancePlanEntry plan;
    DeviceMoERebalanceCommandBufferHeader command_header;
    DeviceMoERebalanceWaveState wave_state;
    uint32_t plan_count = 0;
    DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(hipMemcpy(&status, d_status, sizeof(status), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&plan, d_plan, sizeof(plan), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&command_header, d_command_header, sizeof(command_header), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&wave_state, d_wave_state, sizeof(wave_state), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&plan_count, d_plan_count, sizeof(plan_count), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime),
                        hipMemcpyDeviceToHost),
              hipSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.planned_arrivals, 1u);
    EXPECT_EQ(status.selected_replicas, 0u);
    EXPECT_EQ(status.pre_policy_load_total, 8u);
    EXPECT_EQ(status.pre_policy_load_min, 0u);
    EXPECT_EQ(status.pre_policy_load_max, 8u);
    EXPECT_EQ(status.post_policy_load_total, 8u);
    EXPECT_EQ(status.post_policy_load_min, 0u);
    EXPECT_EQ(status.post_policy_load_max, 4u)
        << "diagnostic post-policy load must count planned arrivals before transfer/apply publishes them";
    EXPECT_GT(status.pre_policy_imbalance_numerator,
              status.post_policy_imbalance_numerator);
    EXPECT_EQ(status.candidate_arrivals_considered, 1u);
    EXPECT_EQ(status.candidate_arrivals_below_floor, 0u);
    EXPECT_EQ(status.candidate_arrivals_pruned_by_count_bound, 0u);
    EXPECT_EQ(status.candidate_load_spread_improvement_total, 4u);
    EXPECT_EQ(status.candidate_load_spread_improvement_max, 4u);
    EXPECT_EQ(status.accepted_load_spread_improvement_total, 4u);
    EXPECT_EQ(status.accepted_load_spread_improvement_max, 4u);
    ASSERT_EQ(plan_count, 1u);
    EXPECT_EQ(command_header.magic, kDeviceMoERebalanceMagic);
    EXPECT_EQ(command_header.version, kDeviceMoERebalanceVersion);
    EXPECT_EQ(command_header.epoch, 2u);
    EXPECT_EQ(command_header.phase,
              static_cast<uint32_t>(DeviceMoERebalancePipelinePhase::PlanAssignments));
    EXPECT_EQ(command_header.command_count, 1u);
    EXPECT_EQ(command_header.command_capacity, 1u);
    EXPECT_EQ(command_header.participant_id, 1u);
    EXPECT_EQ(command_header.participant_count, 3u);
    EXPECT_EQ(wave_state.magic, kDeviceMoERebalanceMagic);
    EXPECT_EQ(wave_state.version, kDeviceMoERebalanceVersion);
    EXPECT_EQ(wave_state.epoch, 2u);
    EXPECT_EQ(wave_state.planned_start_layer, 0u);
    EXPECT_EQ(wave_state.planned_layer_count, 1u);
    EXPECT_EQ(wave_state.command_capacity, 1u);
    EXPECT_EQ(wave_state.participant_id, 1u);
    EXPECT_EQ(wave_state.participant_count, 3u);
    EXPECT_EQ(plan.op, static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival));
    EXPECT_EQ(plan.layer, 0u);
    EXPECT_EQ(plan.expert, 0u);
    EXPECT_EQ(plan.source_participant, 0u);
    EXPECT_EQ(plan.destination_participant, 1u);
    EXPECT_EQ(plan.source_resident_mask, 0b001u);
    EXPECT_EQ(plan.destination_slot, 0u)
        << "controller assigns graph-consumable transfer slots deterministically";
    EXPECT_EQ(plan.payload_slot, 0u)
        << "controller assigns dense sender payload slots independently from destination transfer slots";
    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(bank.local_compute_mask[0], 0u)
        << "planned arrivals must not become active until transfer/apply consumes the plan";
    EXPECT_FALSE(hasMoEExpertFlag(bank.experts[0].flags,
                                  DeviceMoEExpertFlags::Replicated));

    EXPECT_EQ(hipFree(d_gathered), hipSuccess);
    EXPECT_EQ(hipFree(d_status), hipSuccess);
    EXPECT_EQ(hipFree(d_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_plan_count), hipSuccess);
    EXPECT_EQ(hipFree(d_command_header), hipSuccess);
    EXPECT_EQ(hipFree(d_wave_state), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceControllerPlansDynamicOwnershipTransfersWithoutHotCache)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    auto update = makeDynamicOwnershipTransferUpdate(1, 0);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream));

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 0;
    config.participant_count = 2;
    config.window_size_tokens = 55;
    config.max_hot_replicas_per_participant = 0;
    config.dynamic_max_swaps_per_layer = 1;
    config.dynamic_max_plan_entries_per_wave = 2;
    config.flags = 0;
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply);

    std::vector<uint64_t> gathered(static_cast<size_t>(config.participant_count) *
                                       config.num_layers * config.num_experts,
                                   0);
    gathered[0] = 70; // participant 0 owns the heavy expert.
    gathered[1] = 30;
    gathered[config.num_experts + 2] = 1;
    gathered[config.num_experts + 3] = 9;

    uint64_t *d_gathered = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered),
                        gathered.size() * sizeof(uint64_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_status),
                        sizeof(DeviceMoERebalanceStatus)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan),
                        2 * sizeof(DeviceMoERebalancePlanEntry)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_count),
                        sizeof(uint32_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_header),
                        sizeof(DeviceMoERebalanceCommandBufferHeader)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_wave_state),
                        sizeof(DeviceMoERebalanceWaveState)),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered, gathered.data(),
                             gathered.size() * sizeof(uint64_t),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.runDeviceRebalanceController(
        runtime_table.deviceLayerState(0),
        d_gathered,
        d_status,
        config,
        d_plan,
        d_plan_count,
        2,
        1,
        d_command_header,
        d_wave_state));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceStatus status;
    std::array<DeviceMoERebalancePlanEntry, 2> plan{};
    DeviceMoERebalanceCommandBufferHeader command_header;
    DeviceMoERebalanceWaveState wave_state;
    uint32_t plan_count = 0;
    DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(hipMemcpy(&status, d_status, sizeof(status), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(plan.data(), d_plan, sizeof(plan), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&command_header, d_command_header, sizeof(command_header), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&wave_state, d_wave_state, sizeof(wave_state), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&plan_count, d_plan_count, sizeof(plan_count), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime),
                        hipMemcpyDeviceToHost),
              hipSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.planned_arrivals, 2u);
    EXPECT_EQ(status.selected_replicas, 0u);
    EXPECT_EQ(status.pre_policy_load_total, 110u);
    EXPECT_EQ(status.pre_policy_load_min, 10u);
    EXPECT_EQ(status.pre_policy_load_max, 100u);
    EXPECT_EQ(status.post_policy_load_total, 110u);
    EXPECT_EQ(status.post_policy_load_min, 31u);
    EXPECT_EQ(status.post_policy_load_max, 79u);
    EXPECT_EQ(status.accepted_load_spread_improvement_total, 42u);
    EXPECT_EQ(status.accepted_load_spread_improvement_max, 42u);
    ASSERT_EQ(plan_count, 2u);
    EXPECT_EQ(command_header.command_count, 2u);
    EXPECT_EQ(command_header.command_capacity, 2u);
    EXPECT_EQ(wave_state.command_capacity, 2u);

    EXPECT_EQ(plan[0].op, static_cast<uint32_t>(DeviceMoERebalancePlanOp::OwnershipTransfer));
    EXPECT_EQ(plan[0].expert, 0u);
    EXPECT_EQ(plan[0].source_participant, 0u);
    EXPECT_EQ(plan[0].destination_participant, 1u);
    EXPECT_EQ(plan[0].source_resident_mask, 0b01u);
    EXPECT_EQ(plan[0].destination_slot, 0u);
    EXPECT_EQ(plan[0].payload_slot, 0u);

    EXPECT_EQ(plan[1].op, static_cast<uint32_t>(DeviceMoERebalancePlanOp::OwnershipTransfer));
    EXPECT_EQ(plan[1].expert, 2u);
    EXPECT_EQ(plan[1].source_participant, 1u);
    EXPECT_EQ(plan[1].destination_participant, 0u);
    EXPECT_EQ(plan[1].source_resident_mask, 0b10u);
    EXPECT_EQ(plan[1].destination_slot, 0u);
    EXPECT_EQ(plan[1].payload_slot, 0u);

    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(bank.experts[0].owner_participant, 0)
        << "ownership transfer must not become active before async payload arrival apply";
    EXPECT_EQ(bank.experts[2].owner_participant, 1)
        << "ownership transfer must not remove the local primary before apply";

    EXPECT_EQ(hipFree(d_gathered), hipSuccess);
    EXPECT_EQ(hipFree(d_status), hipSuccess);
    EXPECT_EQ(hipFree(d_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_plan_count), hipSuccess);
    EXPECT_EQ(hipFree(d_command_header), hipSuccess);
    EXPECT_EQ(hipFree(d_wave_state), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceControllerRejectsOwnershipSwapFromNonResidentSource)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    auto update = makeDynamicOwnershipTransferUpdate(1, 0);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream));

    DeviceMoELayerRuntime seeded_runtime{};
    ASSERT_EQ(hipMemcpy(&seeded_runtime,
                        runtime_table.deviceLayerState(0),
                        sizeof(seeded_runtime),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    seeded_runtime.banks[seeded_runtime.active_bank].resident_participant_mask[2] = 0b01u;
    ASSERT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(0),
                             &seeded_runtime,
                             sizeof(seeded_runtime),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 0;
    config.participant_count = 2;
    config.window_size_tokens = 55;
    config.max_hot_replicas_per_participant = 0;
    config.dynamic_max_swaps_per_layer = 1;
    config.dynamic_max_plan_entries_per_wave = 2;
    config.flags = 0;
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply);

    std::vector<uint64_t> gathered(static_cast<size_t>(config.participant_count) *
                                       config.num_layers * config.num_experts,
                                   0);
    gathered[0] = 70;
    gathered[1] = 30;
    gathered[config.num_experts + 2] = 1;
    gathered[config.num_experts + 3] = 9;

    uint64_t *d_gathered = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered),
                        gathered.size() * sizeof(uint64_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_status),
                        sizeof(DeviceMoERebalanceStatus)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan),
                        2 * sizeof(DeviceMoERebalancePlanEntry)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_count),
                        sizeof(uint32_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_header),
                        sizeof(DeviceMoERebalanceCommandBufferHeader)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_wave_state),
                        sizeof(DeviceMoERebalanceWaveState)),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered,
                             gathered.data(),
                             gathered.size() * sizeof(uint64_t),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.runDeviceRebalanceController(
        runtime_table.deviceLayerState(0),
        d_gathered,
        d_status,
        config,
        d_plan,
        d_plan_count,
        2,
        1,
        d_command_header,
        d_wave_state));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceStatus status;
    DeviceMoERebalanceCommandBufferHeader command_header;
    uint32_t plan_count = 99;
    ASSERT_EQ(hipMemcpy(&status, d_status, sizeof(status), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&command_header, d_command_header, sizeof(command_header), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&plan_count, d_plan_count, sizeof(plan_count), hipMemcpyDeviceToHost), hipSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.dynamic_ownership_swap_attempts, 1u);
    EXPECT_EQ(status.dynamic_ownership_swap_accepts, 0u);
    EXPECT_EQ(status.dynamic_ownership_swap_rejections, 1u);
    EXPECT_EQ(status.skipped_no_resident, 1u);
    EXPECT_EQ(status.planned_arrivals, 0u);
    EXPECT_EQ(status.accepted_load_spread_improvement_total, 0u);
    EXPECT_EQ(plan_count, 0u);
    EXPECT_EQ(command_header.command_count, 0u);

    EXPECT_EQ(hipFree(d_gathered), hipSuccess);
    EXPECT_EQ(hipFree(d_status), hipSuccess);
    EXPECT_EQ(hipFree(d_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_plan_count), hipSuccess);
    EXPECT_EQ(hipFree(d_command_header), hipSuccess);
    EXPECT_EQ(hipFree(d_wave_state), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceControllerNarrowsDeferredApplySpanToCommandLayers)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 4;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    auto update = makeDynamicOwnershipTransferUpdate(1, 0);
    for (uint32_t layer = 0; layer < table_config.num_layers; ++layer)
    {
        ASSERT_TRUE(runtime_table.prepareInactiveBank(layer, update));
        ASSERT_TRUE(runtime_table.flipActiveBank(layer, update.epoch, stream));
    }

    DeviceMoERebalanceConfig config;
    config.num_layers = table_config.num_layers;
    config.num_experts = table_config.num_experts;
    config.top_k = table_config.top_k;
    config.participant_id = 0;
    config.participant_count = 2;
    config.window_size_tokens = 55;
    config.layer_wave_count = table_config.num_layers;
    config.max_hot_replicas_per_participant = 0;
    config.dynamic_max_swaps_per_layer = 1;
    config.dynamic_max_plan_entries_per_wave = 2;
    config.flags = 0;
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply);

    const size_t participant_stride =
        static_cast<size_t>(config.layer_wave_count) * config.num_experts;
    std::vector<uint64_t> gathered(
        static_cast<size_t>(config.participant_count) * participant_stride,
        0);
    auto set_count = [&](uint32_t participant, uint32_t wave_layer, uint32_t expert, uint64_t value) {
        gathered[static_cast<size_t>(participant) * participant_stride +
                 static_cast<size_t>(wave_layer) * config.num_experts +
                 expert] = value;
    };

    set_count(0, 1, 0, 70);
    set_count(0, 1, 1, 30);
    set_count(1, 1, 2, 1);
    set_count(1, 1, 3, 9);
    set_count(0, 3, 0, 55);
    set_count(1, 3, 2, 55);

    uint64_t *d_gathered = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered),
                        gathered.size() * sizeof(uint64_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_status),
                        sizeof(DeviceMoERebalanceStatus)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan),
                        2 * sizeof(DeviceMoERebalancePlanEntry)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_count),
                        sizeof(uint32_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_header),
                        sizeof(DeviceMoERebalanceCommandBufferHeader)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_wave_state),
                        sizeof(DeviceMoERebalanceWaveState)),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered, gathered.data(),
                             gathered.size() * sizeof(uint64_t),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.runDeviceRebalanceController(
        runtime_table.deviceLayerState(0),
        d_gathered,
        d_status,
        config,
        d_plan,
        d_plan_count,
        2,
        1,
        d_command_header,
        d_wave_state));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceStatus status;
    std::array<DeviceMoERebalancePlanEntry, 2> plan{};
    DeviceMoERebalanceCommandBufferHeader command_header;
    DeviceMoERebalanceWaveState wave_state;
    uint32_t plan_count = 0;
    ASSERT_EQ(hipMemcpy(&status, d_status, sizeof(status), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(plan.data(), d_plan, sizeof(plan), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&command_header, d_command_header, sizeof(command_header), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&wave_state, d_wave_state, sizeof(wave_state), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&plan_count, d_plan_count, sizeof(plan_count), hipMemcpyDeviceToHost), hipSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.changed_layers, 1u);
    ASSERT_EQ(plan_count, 2u);
    EXPECT_EQ(command_header.command_count, 2u);
    EXPECT_EQ(plan[0].layer, 1u);
    EXPECT_EQ(plan[1].layer, 1u);
    EXPECT_EQ(wave_state.planned_start_layer, 1u);
    EXPECT_EQ(wave_state.planned_layer_count, 1u);
    EXPECT_EQ(wave_state.next_start_layer, 0u);

    EXPECT_EQ(hipFree(d_gathered), hipSuccess);
    EXPECT_EQ(hipFree(d_status), hipSuccess);
    EXPECT_EQ(hipFree(d_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_plan_count), hipSuccess);
    EXPECT_EQ(hipFree(d_command_header), hipSuccess);
    EXPECT_EQ(hipFree(d_wave_state), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceControllerPlansLeastLoadedEPOwnershipTransfersWithoutHotCache)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    auto update = makeDynamicOwnershipTransferUpdate(1, 0);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream));

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 0;
    config.participant_count = 2;
    config.root_participant = 0;
    config.window_size_tokens = 50;
    config.max_hot_replicas_per_participant = 0;
    config.dynamic_max_swaps_per_layer = 0;
    config.dynamic_max_plan_entries_per_wave = 1;
    config.max_post_wave_load_spread_per_mille = 100;
    config.routed_assignment_policy = kDeviceMoERebalanceAssignmentLeastLoadedEP;
    config.flags = 0;
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply);

    std::vector<uint64_t> gathered(static_cast<size_t>(config.participant_count) *
                                       config.num_layers * config.num_experts,
                                   0);
    gathered[0] = 80;
    gathered[1] = 20;

    uint64_t *d_gathered = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered),
                        gathered.size() * sizeof(uint64_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_status),
                        sizeof(DeviceMoERebalanceStatus)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan),
                        sizeof(DeviceMoERebalancePlanEntry)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_count),
                        sizeof(uint32_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_header),
                        sizeof(DeviceMoERebalanceCommandBufferHeader)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_wave_state),
                        sizeof(DeviceMoERebalanceWaveState)),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered, gathered.data(),
                             gathered.size() * sizeof(uint64_t),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.runDeviceRebalanceController(
        runtime_table.deviceLayerState(0),
        d_gathered,
        d_status,
        config,
        d_plan,
        d_plan_count,
        1,
        1,
        d_command_header,
        d_wave_state));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceStatus status;
    DeviceMoERebalancePlanEntry plan;
    uint32_t plan_count = 0;
    ASSERT_EQ(hipMemcpy(&status, d_status, sizeof(status), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&plan, d_plan, sizeof(plan), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&plan_count, d_plan_count, sizeof(plan_count), hipMemcpyDeviceToHost), hipSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    ASSERT_EQ(plan_count, 1u);
    EXPECT_EQ(status.planned_arrivals, 1u);
    EXPECT_EQ(status.selected_replicas, 0u);
    EXPECT_EQ(status.candidate_arrivals_considered, 1u);
    EXPECT_EQ(status.candidate_load_spread_improvement_total, 100u)
        << "the shared LLEP span planner still reports the ideal row-span assignment";
    EXPECT_EQ(status.candidate_load_spread_improvement_max, 100u);
    EXPECT_EQ(status.accepted_load_spread_improvement_total, 40u)
        << "without HotExpertReplicaCache, LLEP must publish a durable whole-expert ownership move";
    EXPECT_EQ(status.accepted_load_spread_improvement_max, 40u);
    EXPECT_EQ(status.pre_policy_load_min, 0u);
    EXPECT_EQ(status.pre_policy_load_max, 100u);
    EXPECT_EQ(status.post_policy_load_min, 20u);
    EXPECT_EQ(status.post_policy_load_max, 80u);
    EXPECT_EQ(status.skipped_post_load_spread_ceiling, 0u)
        << "durable LLEP ownership movement must not be blocked by the hot-cache post-spread ceiling";
    EXPECT_EQ(plan.op, static_cast<uint32_t>(DeviceMoERebalancePlanOp::OwnershipTransfer));
    EXPECT_EQ(plan.layer, 0u);
    EXPECT_EQ(plan.expert, 0u);
    EXPECT_EQ(plan.source_participant, 0u);
    EXPECT_EQ(plan.destination_participant, 1u);
    EXPECT_EQ(plan.source_resident_mask, 0b01u);
    EXPECT_EQ(plan.destination_slot, 0u);
    EXPECT_EQ(plan.payload_slot, 0u);

    EXPECT_EQ(hipFree(d_gathered), hipSuccess);
    EXPECT_EQ(hipFree(d_status), hipSuccess);
    EXPECT_EQ(hipFree(d_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_plan_count), hipSuccess);
    EXPECT_EQ(hipFree(d_command_header), hipSuccess);
    EXPECT_EQ(hipFree(d_wave_state), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalancePublishNoopClearsStaleActiveWave)
{
    SKIP_IF_NO_ROCM();

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply);

    constexpr uint32_t command_buffer_count = 2;
    DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.next_epoch = 7;
    controller_state.active_wave = 1;
    controller_state.wave_count = command_buffer_count;
    auto &stale_wave = controller_state.waves[1];
    stale_wave.epoch = 6;
    stale_wave.state = static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::ReadyToApply);
    stale_wave.planned_start_layer = 0;
    stale_wave.planned_layer_count = 1;
    stale_wave.command_count = 2;
    stale_wave.copied_arrivals = 2;
    stale_wave.requested_payload_slots = 2;
    stale_wave.payload_bucket_slots = 2;
    stale_wave.payload_bucket_overflow = 1;

    std::array<DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    std::array<DeviceMoERebalanceWaveState, command_buffer_count> wave_states{};
    std::array<DeviceMoERebalancePlanEntry, command_buffer_count> plan_entries{};
    std::array<DeviceMoERebalanceApplyStatus, 3> gathered_copy_status{};
    for (uint32_t i = 0; i < command_buffer_count; ++i)
    {
        command_headers[i].command_capacity = 1;
        command_headers[i].participant_id = config.participant_id;
        command_headers[i].participant_count = config.participant_count;
        wave_states[i].command_capacity = 1;
        wave_states[i].participant_id = config.participant_id;
        wave_states[i].participant_count = config.participant_count;
    }
    command_headers[1].epoch = 0;
    command_headers[1].command_count = 0;

    DeviceMoERebalanceApplyStatus copy_status;
    copy_status.copied_arrivals = 9;

    DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    DeviceMoERebalanceWaveState *d_wave_states = nullptr;
    DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    DeviceMoERebalanceApplyStatus *d_copy_status = nullptr;
    DeviceMoERebalanceApplyStatus *d_gathered_copy_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_headers),
                        command_headers.size() * sizeof(command_headers[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_wave_states),
                        wave_states.size() * sizeof(wave_states[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_entries),
                        plan_entries.size() * sizeof(plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_copy_status), sizeof(copy_status)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_copy_status),
                        gathered_copy_status.size() * sizeof(gathered_copy_status[0])),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_controller_state, &controller_state, sizeof(controller_state),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_command_headers, command_headers.data(),
                             command_headers.size() * sizeof(command_headers[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_wave_states, wave_states.data(),
                             wave_states.size() * sizeof(wave_states[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_plan_entries, plan_entries.data(),
                             plan_entries.size() * sizeof(plan_entries[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_copy_status, &copy_status, sizeof(copy_status),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered_copy_status, gathered_copy_status.data(),
                             gathered_copy_status.size() * sizeof(gathered_copy_status[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.publishDeviceRebalanceTransferComplete(
        d_controller_state,
        d_command_headers,
        d_wave_states,
        d_copy_status,
        d_plan_entries,
        1,
        d_gathered_copy_status,
        config,
        command_buffer_count));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceGraphControllerState result;
    ASSERT_EQ(hipMemcpy(&result, d_controller_state, sizeof(result), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(result.maintenance_launches, 1u);
    const auto &cleared_wave = result.waves[1];
    EXPECT_EQ(cleared_wave.magic, kDeviceMoERebalanceMagic);
    EXPECT_EQ(cleared_wave.version, kDeviceMoERebalanceVersion);
    EXPECT_EQ(cleared_wave.epoch, 0u);
    EXPECT_EQ(cleared_wave.state,
              static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::Idle));
    EXPECT_EQ(cleared_wave.command_count, 0u);
    EXPECT_EQ(cleared_wave.copied_arrivals, 0u);
    EXPECT_EQ(cleared_wave.requested_payload_slots, 0u);
    EXPECT_EQ(cleared_wave.payload_bucket_slots, 0u);
    EXPECT_EQ(cleared_wave.payload_bucket_overflow, 0u);

    EXPECT_EQ(hipFree(d_controller_state), hipSuccess);
    EXPECT_EQ(hipFree(d_command_headers), hipSuccess);
    EXPECT_EQ(hipFree(d_wave_states), hipSuccess);
    EXPECT_EQ(hipFree(d_plan_entries), hipSuccess);
    EXPECT_EQ(hipFree(d_copy_status), hipSuccess);
    EXPECT_EQ(hipFree(d_gathered_copy_status), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalancePublishUsesProjectedWavePayloadBucket)
{
    SKIP_IF_NO_ROCM();

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERebalanceConfig config;
    config.num_layers = 4;
    config.num_experts = 8;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 2;
    config.window_size_tokens = 1;
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply);

    constexpr uint32_t command_buffer_count = 2;
    DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.active_wave = 0;
    controller_state.wave_count = command_buffer_count;
    controller_state.waves[0].requested_payload_slots = 0;
    controller_state.waves[0].payload_bucket_slots = 0;
    controller_state.waves[0].payload_bucket_index = 0;
    controller_state.waves[0].payload_bucket_overflow = 0;

    std::array<DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    std::array<DeviceMoERebalanceWaveState, command_buffer_count> wave_states{};
    std::array<DeviceMoERebalancePlanEntry, command_buffer_count * 4> plan_entries{};
    std::array<DeviceMoERebalanceApplyStatus, 2> gathered_copy_status{};
    for (uint32_t i = 0; i < command_buffer_count; ++i)
    {
        command_headers[i].command_capacity = 4;
        command_headers[i].participant_id = config.participant_id;
        command_headers[i].participant_count = config.participant_count;
        wave_states[i].command_capacity = 4;
        wave_states[i].participant_id = config.participant_id;
        wave_states[i].participant_count = config.participant_count;
    }
    command_headers[0].epoch = 9;
    command_headers[0].command_count = 1;
    plan_entries[0].op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    plan_entries[0].layer = 2;
    plan_entries[0].expert = 3;
    plan_entries[0].source_participant = 0;
    plan_entries[0].destination_participant = 1;
    plan_entries[0].source_resident_mask = 0b01u;
    plan_entries[0].destination_slot = 0;
    plan_entries[0].payload_slot = 0;
    wave_states[0].epoch = command_headers[0].epoch;
    wave_states[0].planned_start_layer = 2;
    wave_states[0].planned_layer_count = 1;
    wave_states[0].requested_payload_slots = 1;
    wave_states[0].payload_bucket_slots = 1;
    wave_states[0].payload_bucket_index = 0;
    wave_states[0].payload_bucket_overflow = 0;

    DeviceMoERebalanceApplyStatus copy_status;
    copy_status.copied_arrivals = 1;
    gathered_copy_status[1].copied_arrivals = 1;

    DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    DeviceMoERebalanceWaveState *d_wave_states = nullptr;
    DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    DeviceMoERebalanceApplyStatus *d_copy_status = nullptr;
    DeviceMoERebalanceApplyStatus *d_gathered_copy_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_headers),
                        command_headers.size() * sizeof(command_headers[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_wave_states),
                        wave_states.size() * sizeof(wave_states[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_entries),
                        plan_entries.size() * sizeof(plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_copy_status), sizeof(copy_status)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_copy_status),
                        gathered_copy_status.size() * sizeof(gathered_copy_status[0])),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_controller_state, &controller_state, sizeof(controller_state),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_command_headers, command_headers.data(),
                             command_headers.size() * sizeof(command_headers[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_wave_states, wave_states.data(),
                             wave_states.size() * sizeof(wave_states[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_plan_entries, plan_entries.data(),
                             plan_entries.size() * sizeof(plan_entries[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_copy_status, &copy_status, sizeof(copy_status),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered_copy_status, gathered_copy_status.data(),
                             gathered_copy_status.size() * sizeof(gathered_copy_status[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.publishDeviceRebalanceTransferComplete(
        d_controller_state,
        d_command_headers,
        d_wave_states,
        d_copy_status,
        d_plan_entries,
        4,
        d_gathered_copy_status,
        config,
        command_buffer_count));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceGraphControllerState result;
    ASSERT_EQ(hipMemcpy(&result, d_controller_state, sizeof(result), hipMemcpyDeviceToHost), hipSuccess);
    const auto &published_wave = result.waves[0];
    EXPECT_EQ(published_wave.epoch, command_headers[0].epoch);
    EXPECT_EQ(published_wave.state,
              static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::ReadyToApply));
    EXPECT_EQ(published_wave.planned_start_layer, wave_states[0].planned_start_layer);
    EXPECT_EQ(published_wave.planned_layer_count, wave_states[0].planned_layer_count);
    EXPECT_EQ(published_wave.command_count, command_headers[0].command_count);
    EXPECT_EQ(published_wave.copied_arrivals, copy_status.copied_arrivals);
    EXPECT_EQ(published_wave.requested_payload_slots, wave_states[0].requested_payload_slots);
    EXPECT_EQ(published_wave.payload_bucket_slots, wave_states[0].payload_bucket_slots);
    EXPECT_EQ(published_wave.payload_bucket_index, wave_states[0].payload_bucket_index);
    EXPECT_EQ(published_wave.payload_bucket_overflow, wave_states[0].payload_bucket_overflow);

    EXPECT_EQ(hipFree(d_controller_state), hipSuccess);
    EXPECT_EQ(hipFree(d_command_headers), hipSuccess);
    EXPECT_EQ(hipFree(d_wave_states), hipSuccess);
    EXPECT_EQ(hipFree(d_plan_entries), hipSuccess);
    EXPECT_EQ(hipFree(d_copy_status), hipSuccess);
    EXPECT_EQ(hipFree(d_gathered_copy_status), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalancePublishRequiresGatheredDestinationCopyStatus)
{
    SKIP_IF_NO_ROCM();

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 0;
    config.participant_count = 2;
    config.window_size_tokens = 1;
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply);

    constexpr uint32_t command_buffer_count = 2;
    constexpr uint32_t plan_capacity = 1;
    DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.active_wave = 0;
    controller_state.wave_count = command_buffer_count;

    std::array<DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    std::array<DeviceMoERebalanceWaveState, command_buffer_count> wave_states{};
    std::array<DeviceMoERebalancePlanEntry, command_buffer_count * plan_capacity> plan_entries{};
    std::array<DeviceMoERebalanceApplyStatus, 2> gathered_copy_status{};
    for (uint32_t i = 0; i < command_buffer_count; ++i)
    {
        command_headers[i].command_capacity = plan_capacity;
        command_headers[i].participant_id = config.participant_id;
        command_headers[i].participant_count = config.participant_count;
        wave_states[i].command_capacity = plan_capacity;
        wave_states[i].participant_id = config.participant_id;
        wave_states[i].participant_count = config.participant_count;
    }
    command_headers[0].epoch = 11;
    command_headers[0].command_count = 1;
    wave_states[0].epoch = command_headers[0].epoch;
    wave_states[0].planned_start_layer = 0;
    wave_states[0].planned_layer_count = 1;
    wave_states[0].requested_payload_slots = 1;
    wave_states[0].payload_bucket_slots = 1;
    plan_entries[0].op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    plan_entries[0].layer = 0;
    plan_entries[0].expert = 1;
    plan_entries[0].source_participant = 0;
    plan_entries[0].destination_participant = 1;
    plan_entries[0].source_resident_mask = 0b01u;
    plan_entries[0].destination_slot = 0;
    plan_entries[0].payload_slot = 0;

    DeviceMoERebalanceApplyStatus copy_status;
    copy_status.copied_arrivals = 0;
    gathered_copy_status[0].copied_arrivals = 0;
    gathered_copy_status[1].copied_arrivals = 0;

    DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    DeviceMoERebalanceWaveState *d_wave_states = nullptr;
    DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    DeviceMoERebalanceApplyStatus *d_copy_status = nullptr;
    DeviceMoERebalanceApplyStatus *d_gathered_copy_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_headers),
                        command_headers.size() * sizeof(command_headers[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_wave_states),
                        wave_states.size() * sizeof(wave_states[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_entries),
                        plan_entries.size() * sizeof(plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_copy_status), sizeof(copy_status)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_copy_status),
                        gathered_copy_status.size() * sizeof(gathered_copy_status[0])),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_controller_state, &controller_state, sizeof(controller_state),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_command_headers, command_headers.data(),
                             command_headers.size() * sizeof(command_headers[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_wave_states, wave_states.data(),
                             wave_states.size() * sizeof(wave_states[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_plan_entries, plan_entries.data(),
                             plan_entries.size() * sizeof(plan_entries[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_copy_status, &copy_status, sizeof(copy_status),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered_copy_status, gathered_copy_status.data(),
                             gathered_copy_status.size() * sizeof(gathered_copy_status[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.publishDeviceRebalanceTransferComplete(
        d_controller_state,
        d_command_headers,
        d_wave_states,
        d_copy_status,
        d_plan_entries,
        plan_capacity,
        d_gathered_copy_status,
        config,
        command_buffer_count));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceGraphControllerState result;
    ASSERT_EQ(hipMemcpy(&result, d_controller_state, sizeof(result), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(result.last_error_code,
              static_cast<uint32_t>(DeviceMoERebalanceStatusCode::MissingTransferCompletion));
    const auto &wave = result.waves[0];
    EXPECT_EQ(wave.state,
              static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::Error));
    EXPECT_NE(wave.state,
              static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::ReadyToApply));
    EXPECT_EQ(wave.error_code,
              static_cast<uint32_t>(DeviceMoERebalanceStatusCode::MissingTransferCompletion));

    EXPECT_EQ(hipFree(d_controller_state), hipSuccess);
    EXPECT_EQ(hipFree(d_command_headers), hipSuccess);
    EXPECT_EQ(hipFree(d_wave_states), hipSuccess);
    EXPECT_EQ(hipFree(d_plan_entries), hipSuccess);
    EXPECT_EQ(hipFree(d_copy_status), hipSuccess);
    EXPECT_EQ(hipFree(d_gathered_copy_status), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalancePublishRejectsSourceSideCopyErrors)
{
    SKIP_IF_NO_ROCM();

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 0;
    config.participant_count = 2;
    config.window_size_tokens = 1;
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply);

    constexpr uint32_t command_buffer_count = 2;
    constexpr uint32_t plan_capacity = 1;
    DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.active_wave = 0;
    controller_state.wave_count = command_buffer_count;

    std::array<DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    std::array<DeviceMoERebalanceWaveState, command_buffer_count> wave_states{};
    std::array<DeviceMoERebalancePlanEntry, command_buffer_count * plan_capacity> plan_entries{};
    std::array<DeviceMoERebalanceApplyStatus, 2> gathered_copy_status{};
    for (uint32_t i = 0; i < command_buffer_count; ++i)
    {
        command_headers[i].command_capacity = plan_capacity;
        command_headers[i].participant_id = config.participant_id;
        command_headers[i].participant_count = config.participant_count;
        wave_states[i].command_capacity = plan_capacity;
        wave_states[i].participant_id = config.participant_id;
        wave_states[i].participant_count = config.participant_count;
    }
    command_headers[0].epoch = 12;
    command_headers[0].command_count = 1;
    wave_states[0].epoch = command_headers[0].epoch;
    wave_states[0].planned_start_layer = 0;
    wave_states[0].planned_layer_count = 1;
    wave_states[0].requested_payload_slots = 1;
    wave_states[0].payload_bucket_slots = 1;
    plan_entries[0].op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    plan_entries[0].layer = 0;
    plan_entries[0].expert = 1;
    plan_entries[0].source_participant = 0;
    plan_entries[0].destination_participant = 1;
    plan_entries[0].source_resident_mask = 0b01u;
    plan_entries[0].destination_slot = 0;
    plan_entries[0].payload_slot = 0;

    /*
     * Regression for ROCm long-context Dynamic+MTP+prefix cache: the source
     * participant failed to pack a descriptor, but the destination participant
     * still reported enough copied arrivals.  Publication must treat the source
     * error counter as fatal instead of marking the transfer wave ready.
     */
    DeviceMoERebalanceApplyStatus copy_status;
    copy_status.missing_source_descriptors = 1;
    gathered_copy_status[0].missing_source_descriptors = 1;
    gathered_copy_status[1].copied_arrivals = 1;

    DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    DeviceMoERebalanceWaveState *d_wave_states = nullptr;
    DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    DeviceMoERebalanceApplyStatus *d_copy_status = nullptr;
    DeviceMoERebalanceApplyStatus *d_gathered_copy_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_headers),
                        command_headers.size() * sizeof(command_headers[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_wave_states),
                        wave_states.size() * sizeof(wave_states[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_entries),
                        plan_entries.size() * sizeof(plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_copy_status), sizeof(copy_status)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_copy_status),
                        gathered_copy_status.size() * sizeof(gathered_copy_status[0])),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_controller_state, &controller_state, sizeof(controller_state),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_command_headers, command_headers.data(),
                             command_headers.size() * sizeof(command_headers[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_wave_states, wave_states.data(),
                             wave_states.size() * sizeof(wave_states[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_plan_entries, plan_entries.data(),
                             plan_entries.size() * sizeof(plan_entries[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_copy_status, &copy_status, sizeof(copy_status),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered_copy_status, gathered_copy_status.data(),
                             gathered_copy_status.size() * sizeof(gathered_copy_status[0]),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.publishDeviceRebalanceTransferComplete(
        d_controller_state,
        d_command_headers,
        d_wave_states,
        d_copy_status,
        d_plan_entries,
        plan_capacity,
        d_gathered_copy_status,
        config,
        command_buffer_count));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceGraphControllerState result;
    ASSERT_EQ(hipMemcpy(&result, d_controller_state, sizeof(result), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(result.last_error_code,
              static_cast<uint32_t>(DeviceMoERebalanceStatusCode::MissingTransferCompletion));
    const auto &wave = result.waves[0];
    EXPECT_EQ(wave.state,
              static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::Error));
    EXPECT_NE(wave.state,
              static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::ReadyToApply));
    EXPECT_EQ(wave.error_code,
              static_cast<uint32_t>(DeviceMoERebalanceStatusCode::MissingTransferCompletion));

    EXPECT_EQ(hipFree(d_controller_state), hipSuccess);
    EXPECT_EQ(hipFree(d_command_headers), hipSuccess);
    EXPECT_EQ(hipFree(d_wave_states), hipSuccess);
    EXPECT_EQ(hipFree(d_plan_entries), hipSuccess);
    EXPECT_EQ(hipFree(d_copy_status), hipSuccess);
    EXPECT_EQ(hipFree(d_gathered_copy_status), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, ApplyReadyDeviceRebalanceWaveClearsAppliedCommandHeader)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream));

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;

    constexpr uint32_t plan_capacity = 4;
    constexpr uint32_t command_buffer_count = 2;
    std::vector<DeviceMoERebalancePlanEntry> plan_entries(
        static_cast<size_t>(plan_capacity) * command_buffer_count);
    auto &plan = plan_entries[0];
    plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ResidentExpertAssignment);
    plan.layer = 0;
    plan.expert = 0;
    plan.source_participant = 0;
    plan.destination_participant = config.participant_id;
    plan.source_resident_mask = 0b001u;
    plan.destination_slot = 0;
    plan.payload_slot = kDeviceMoEInvalidSlot;

    std::array<DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    for (uint32_t i = 0; i < command_buffer_count; ++i)
    {
        command_headers[i].command_capacity = plan_capacity;
        command_headers[i].participant_id = config.participant_id;
        command_headers[i].participant_count = config.participant_count;
    }
    command_headers[0].epoch = 7;
    command_headers[0].command_count = 1;

    DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.next_epoch = 8;
    controller_state.active_wave = 0;
    controller_state.wave_count = command_buffer_count;
    auto &wave = controller_state.waves[0];
    wave.epoch = command_headers[0].epoch;
    wave.state = static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::ReadyToApply);
    wave.planned_start_layer = 0;
    wave.planned_layer_count = 1;
    wave.command_count = 1;
    wave.copied_arrivals = 1;
    wave.requested_payload_slots = 0;
    wave.payload_bucket_slots = 0;

    DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    DeviceMoERebalanceApplyStatus *d_apply_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_entries),
                        plan_entries.size() * sizeof(plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_headers),
                        command_headers.size() * sizeof(command_headers[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_apply_status),
                        sizeof(DeviceMoERebalanceApplyStatus)),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_plan_entries,
                             plan_entries.data(),
                             plan_entries.size() * sizeof(plan_entries[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_command_headers,
                             command_headers.data(),
                             command_headers.size() * sizeof(command_headers[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_controller_state,
                             &controller_state,
                             sizeof(controller_state),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.applyReadyDeviceRebalanceWave(
        runtime_table.deviceLayerState(0),
        d_plan_entries,
        nullptr,
        plan_capacity,
        nullptr,
        0,
        config,
        d_apply_status,
        d_controller_state,
        d_command_headers,
        0,
        command_buffer_count));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceApplyStatus apply_status;
    DeviceMoERebalanceGraphControllerState result_state;
    std::array<DeviceMoERebalanceCommandBufferHeader, command_buffer_count> result_headers{};
    DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(hipMemcpy(&apply_status, d_apply_status, sizeof(apply_status), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&result_state, d_controller_state, sizeof(result_state), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(result_headers.data(),
                        d_command_headers,
                        result_headers.size() * sizeof(result_headers[0]),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&runtime,
                        runtime_table.deviceLayerState(0),
                        sizeof(runtime),
                        hipMemcpyDeviceToHost),
              hipSuccess);

    EXPECT_EQ(apply_status.status_code,
              static_cast<uint32_t>(DeviceMoERebalanceApplyStatusCode::Ok));
    EXPECT_EQ(apply_status.plan_entries_seen, 1u);
    EXPECT_EQ(apply_status.changed_layers, 1u);
    EXPECT_EQ(apply_status.applied_arrivals, 1u);
    EXPECT_EQ(apply_status.post_apply_multi_resident_experts, 1u);
    EXPECT_EQ(result_state.decode_apply_hits, 1u);
    EXPECT_EQ(result_state.active_wave, 1u);
    EXPECT_EQ(result_state.waves[0].state,
              static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::Applied));
    EXPECT_EQ(result_headers[0].epoch, 0u);
    EXPECT_EQ(result_headers[0].command_count, 0u);
    EXPECT_EQ(result_headers[0].command_capacity, plan_capacity);
    EXPECT_EQ(result_headers[0].participant_id, config.participant_id);
    EXPECT_EQ(result_headers[0].participant_count, config.participant_count);
    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(bank.local_compute_mask[0], 1u);
    EXPECT_EQ(bank.replica_role[0],
              static_cast<uint8_t>(DeviceMoEReplicaRole::Replica));
    EXPECT_EQ(bank.reserved[0], 1u);
    EXPECT_TRUE(hasMoEExpertFlag(bank.experts[0].flags,
                                 DeviceMoEExpertFlags::Replicated));

    EXPECT_EQ(hipFree(d_plan_entries), hipSuccess);
    EXPECT_EQ(hipFree(d_command_headers), hipSuccess);
    EXPECT_EQ(hipFree(d_controller_state), hipSuccess);
    EXPECT_EQ(hipFree(d_apply_status), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceOwnershipTransferUpdatesSourceOwner)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    auto update = makeDynamicOwnershipTransferUpdate(1, 0);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream));

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 0;
    config.participant_count = 2;
    config.window_size_tokens = 1;

    DeviceMoERebalancePlanEntry plan;
    plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::OwnershipTransfer);
    plan.layer = 0;
    plan.expert = 0;
    plan.source_participant = 0;
    plan.destination_participant = 1;
    plan.source_resident_mask = 0b01u;
    plan.destination_slot = 0;
    plan.payload_slot = 0;

    DeviceMoERebalanceCommandBufferHeader command_header;
    command_header.epoch = 2;
    command_header.phase =
        static_cast<uint32_t>(DeviceMoERebalancePipelinePhase::PlanAssignments);
    command_header.command_count = 1;
    command_header.command_capacity = 1;
    command_header.participant_id = config.participant_id;
    command_header.participant_count = config.participant_count;

    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    DeviceMoEExpertDirectoryEntry *d_transfer_slots = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    DeviceMoERebalanceApplyStatus *d_apply_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan), sizeof(plan)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_transfer_slots),
                        sizeof(DeviceMoEExpertDirectoryEntry)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_header), sizeof(command_header)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_apply_status),
                        sizeof(DeviceMoERebalanceApplyStatus)),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_plan, &plan, sizeof(plan), hipMemcpyHostToDevice, stream), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_command_header,
                             &command_header,
                             sizeof(command_header),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.applyDeviceRebalanceArrivals(
        runtime_table.deviceLayerState(0),
        d_plan,
        nullptr,
        1,
        d_transfer_slots,
        1,
        config,
        d_apply_status,
        d_command_header));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceApplyStatus apply_status;
    DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(hipMemcpy(&apply_status, d_apply_status, sizeof(apply_status), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime), hipMemcpyDeviceToHost), hipSuccess);

    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(apply_status.applied_arrivals, 0u)
        << "source-side ownership metadata apply does not copy a payload arrival";
    EXPECT_EQ(apply_status.changed_layers, 1u);
    EXPECT_EQ(bank.experts[0].owner_participant, 1)
        << "source participants must not resurrect stale ownership on the next runtime bank rebuild";
    EXPECT_EQ(bank.local_compute_mask[0], 0u);
    EXPECT_EQ(bank.replica_role[0], static_cast<uint8_t>(DeviceMoEReplicaRole::None));
    EXPECT_EQ(bank.resident_participant_mask[0], 0b10u);
    EXPECT_FALSE(hasMoEExpertFlag(bank.experts[0].flags,
                                  DeviceMoEExpertFlags::Resident));
    EXPECT_FALSE(hasMoEExpertFlag(bank.experts[0].flags,
                                  DeviceMoEExpertFlags::LocalCompute));
    EXPECT_FALSE(hasMoEExpertFlag(bank.experts[0].flags,
                                  DeviceMoEExpertFlags::Replicated));

    EXPECT_EQ(hipFree(d_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_transfer_slots), hipSuccess);
    EXPECT_EQ(hipFree(d_command_header), hipSuccess);
    EXPECT_EQ(hipFree(d_apply_status), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectWithReadyRebalanceApplyTargetAllRollsThroughLayers)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 2;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    for (int layer = 0; layer < 2; ++layer)
    {
        auto update = makeParticipantOneBaseUpdate(1);
        ASSERT_TRUE(runtime_table.prepareInactiveBank(layer, update));
        ASSERT_TRUE(runtime_table.flipActiveBank(layer, update.epoch, stream));
    }

    DeviceMoERebalanceConfig config;
    config.num_layers = 2;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply);

    constexpr uint32_t plan_capacity = 4;
    constexpr uint32_t command_buffer_count = 2;
    constexpr int d_model = 32;
    auto hidden = TestTensorFactory::createFP32({1, d_model});
    auto gate_weights = TestTensorFactory::createFP32({4, d_model});
    std::fill(hidden->mutable_data(), hidden->mutable_data() + hidden->numel(), 0.0f);
    std::fill(gate_weights->mutable_data(), gate_weights->mutable_data() + gate_weights->numel(), 0.0f);
    hidden->mutable_data()[0] = 1.0f;
    for (int expert = 0; expert < 4; ++expert)
        gate_weights->mutable_data()[static_cast<size_t>(expert) * d_model] =
            static_cast<float>(4 - expert);
    auto output_indices = TestTensorFactory::createFP32({2, 1});
    auto output_weights = TestTensorFactory::createFP32({2, 1});
    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices->ensureOnDevice(device));
    ASSERT_TRUE(output_weights->ensureOnDevice(device));

    std::vector<DeviceMoERebalancePlanEntry> plan_entries(
        static_cast<size_t>(plan_capacity) * command_buffer_count);
    for (uint32_t layer = 0; layer < 2; ++layer)
    {
        auto &plan = plan_entries[layer];
        plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ResidentExpertAssignment);
        plan.layer = layer;
        plan.expert = 0;
        plan.source_participant = 0;
        plan.destination_participant = config.participant_id;
        plan.source_resident_mask = 0b001u;
        plan.destination_slot = 0;
        plan.payload_slot = kDeviceMoEInvalidSlot;
    }

    std::array<DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    for (auto &header : command_headers)
    {
        header.command_capacity = plan_capacity;
        header.participant_id = config.participant_id;
        header.participant_count = config.participant_count;
    }
    command_headers[0].epoch = 7;
    command_headers[0].command_count = 2;

    DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.next_epoch = 8;
    controller_state.active_wave = 0;
    controller_state.wave_count = command_buffer_count;
    auto &wave = controller_state.waves[0];
    wave.epoch = command_headers[0].epoch;
    wave.state = static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::ReadyToApply);
    wave.planned_start_layer = 0;
    wave.planned_layer_count = 2;
    wave.command_count = 2;
    wave.copied_arrivals = 2;
    wave.requested_payload_slots = 0;
    wave.payload_bucket_slots = 0;

    DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    DeviceMoERebalanceApplyStatus *d_apply_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_entries),
                        plan_entries.size() * sizeof(plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_headers),
                        command_headers.size() * sizeof(command_headers[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_apply_status),
                        sizeof(DeviceMoERebalanceApplyStatus)),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_plan_entries,
                             plan_entries.data(),
                             plan_entries.size() * sizeof(plan_entries[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_command_headers,
                             command_headers.data(),
                             command_headers.size() * sizeof(command_headers[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_controller_state,
                             &controller_state,
                             sizeof(controller_state),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    ROCmMoEKernel kernel(0);
    static_cast<IMoEKernel &>(kernel).setGPUStream(stream);
    auto route_workspace = bindDefaultMoEWorkspace(
        kernel,
        /*max_seq_len=*/1,
        /*d_model=*/4,
        /*intermediate=*/8,
        /*num_experts=*/4,
        /*top_k=*/2);
    ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
    ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
    ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
    ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
    ScopedROCmEnvOverride wave_env("LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "0");

    ASSERT_TRUE(kernel.decodeRouteSelectWithReadyRebalanceApply(
        runtime_table.deviceLayerState(0),
        runtime_table.deviceLayerState(1),
        hidden.get(),
        gate_weights.get(),
        /*d_model=*/4,
        /*num_experts=*/4,
        /*top_k=*/2,
        /*normalize_weights=*/false,
        output_indices.get(),
        output_weights.get(),
        /*write_legacy_outputs=*/true,
        /*update_runtime_histogram=*/true,
        d_plan_entries,
        plan_capacity,
        d_command_headers,
        nullptr,
        0u,
        config,
        d_apply_status,
        d_controller_state,
        -1,
        command_buffer_count));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceGraphControllerState result_state;
    std::array<DeviceMoERebalanceCommandBufferHeader, command_buffer_count> result_headers{};
    ASSERT_EQ(hipMemcpy(&result_state, d_controller_state, sizeof(result_state), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(result_headers.data(),
                        d_command_headers,
                        result_headers.size() * sizeof(result_headers[0]),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(result_state.active_wave, 0u);
    EXPECT_EQ(result_state.waves[0].state,
              static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::ReadyToApply));
    EXPECT_EQ(result_state.waves[0].applied_layer_count, 1u);
    EXPECT_EQ(result_headers[0].epoch, 7u);
    EXPECT_EQ(result_headers[0].command_count, 2u);

    ASSERT_TRUE(kernel.decodeRouteSelectWithReadyRebalanceApply(
        runtime_table.deviceLayerState(0),
        runtime_table.deviceLayerState(1),
        hidden.get(),
        gate_weights.get(),
        /*d_model=*/4,
        /*num_experts=*/4,
        /*top_k=*/2,
        /*normalize_weights=*/false,
        output_indices.get(),
        output_weights.get(),
        /*write_legacy_outputs=*/true,
        /*update_runtime_histogram=*/true,
        d_plan_entries,
        plan_capacity,
        d_command_headers,
        nullptr,
        0u,
        config,
        d_apply_status,
        d_controller_state,
        -1,
        command_buffer_count));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(hipMemcpy(&result_state, d_controller_state, sizeof(result_state), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(result_headers.data(),
                        d_command_headers,
                        result_headers.size() * sizeof(result_headers[0]),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(result_state.active_wave, 1u);
    EXPECT_EQ(result_state.waves[0].state,
              static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::Applied));
    EXPECT_EQ(result_state.waves[0].applied_layer_count, 2u);
    EXPECT_EQ(result_headers[0].epoch, 0u);
    EXPECT_EQ(result_headers[0].command_count, 0u);

    EXPECT_EQ(hipFree(d_plan_entries), hipSuccess);
    EXPECT_EQ(hipFree(d_command_headers), hipSuccess);
    EXPECT_EQ(hipFree(d_controller_state), hipSuccess);
    EXPECT_EQ(hipFree(d_apply_status), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DecodeReadyRebalanceApplyDoesNotAdvanceIncompleteTransferSlotArrival)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    auto update = makeDynamicOwnershipTransferUpdate(1, 1);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream));

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 2;
    config.window_size_tokens = 1;
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply);

    constexpr uint32_t plan_capacity = 4;
    constexpr uint32_t command_buffer_count = 2;
    constexpr int d_model = 32;
    auto hidden = TestTensorFactory::createFP32({1, d_model});
    auto gate_weights = TestTensorFactory::createFP32({4, d_model});
    std::fill(hidden->mutable_data(), hidden->mutable_data() + hidden->numel(), 0.0f);
    std::fill(gate_weights->mutable_data(), gate_weights->mutable_data() + gate_weights->numel(), 0.0f);
    hidden->mutable_data()[0] = 1.0f;
    for (int expert = 0; expert < 4; ++expert)
        gate_weights->mutable_data()[static_cast<size_t>(expert) * d_model] =
            static_cast<float>(4 - expert);
    auto output_indices = TestTensorFactory::createFP32({1, 2});
    auto output_weights = TestTensorFactory::createFP32({1, 2});
    ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device, stream));
    ASSERT_TRUE(output_indices->ensureOnDevice(device, stream));
    ASSERT_TRUE(output_weights->ensureOnDevice(device, stream));

    std::vector<DeviceMoERebalancePlanEntry> plan_entries(
        static_cast<size_t>(plan_capacity) * command_buffer_count);
    auto &plan = plan_entries[0];
    plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::OwnershipTransfer);
    plan.layer = 0;
    plan.expert = 0;
    plan.source_participant = 0;
    plan.destination_participant = config.participant_id;
    plan.source_resident_mask = 0b01u;
    plan.destination_slot = 0;
    plan.payload_slot = 0;

    std::array<DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    for (auto &header : command_headers)
    {
        header.command_capacity = plan_capacity;
        header.participant_id = config.participant_id;
        header.participant_count = config.participant_count;
    }
    command_headers[0].epoch = 11;
    command_headers[0].command_count = 1;

    DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.next_epoch = 12;
    controller_state.active_wave = 0;
    controller_state.wave_count = command_buffer_count;
    auto &wave = controller_state.waves[0];
    wave.epoch = command_headers[0].epoch;
    wave.state = static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::ReadyToApply);
    wave.planned_start_layer = 0;
    wave.planned_layer_count = 1;
    wave.command_count = 1;
    wave.copied_arrivals = 0;
    wave.requested_payload_slots = 1;
    wave.payload_bucket_slots = 1;

    std::array<DeviceMoEExpertDirectoryEntry, 1> transfer_slots{};
    transfer_slots[0].participant = config.participant_id;
    transfer_slots[0].slot_index = 0;
    transfer_slots[0].flags =
        static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
        static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::TransferSlot);

    DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    DeviceMoERebalanceApplyStatus *d_apply_status = nullptr;
    DeviceMoEExpertDirectoryEntry *d_transfer_slots = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_entries),
                        plan_entries.size() * sizeof(plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_headers),
                        command_headers.size() * sizeof(command_headers[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_apply_status),
                        sizeof(DeviceMoERebalanceApplyStatus)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_transfer_slots),
                        transfer_slots.size() * sizeof(transfer_slots[0])),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_plan_entries,
                             plan_entries.data(),
                             plan_entries.size() * sizeof(plan_entries[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_command_headers,
                             command_headers.data(),
                             command_headers.size() * sizeof(command_headers[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_controller_state,
                             &controller_state,
                             sizeof(controller_state),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_transfer_slots,
                             transfer_slots.data(),
                             transfer_slots.size() * sizeof(transfer_slots[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    ROCmMoEKernel kernel(0);
    static_cast<IMoEKernel &>(kernel).setGPUStream(stream);
    auto workspace = bindDefaultMoEWorkspace(kernel, /*max_seq_len=*/64, /*d_model=*/d_model,
                                             /*intermediate=*/8, /*num_experts=*/4, /*top_k=*/2);

    ASSERT_TRUE(kernel.decodeRouteSelectWithReadyRebalanceApply(
        runtime_table.deviceLayerState(0),
        runtime_table.deviceLayerState(0),
        hidden.get(),
        gate_weights.get(),
        /*d_model=*/d_model,
        /*num_experts=*/4,
        /*top_k=*/2,
        /*normalize_weights=*/false,
        output_indices.get(),
        output_weights.get(),
        /*write_legacy_outputs=*/true,
        /*update_runtime_histogram=*/true,
        d_plan_entries,
        plan_capacity,
        d_command_headers,
        d_transfer_slots,
        static_cast<uint32_t>(transfer_slots.size()),
        config,
        d_apply_status,
        d_controller_state,
        -1,
        command_buffer_count));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceApplyStatus apply_status;
    DeviceMoERebalanceGraphControllerState result_state;
    std::array<DeviceMoERebalanceCommandBufferHeader, command_buffer_count> result_headers{};
    DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(hipMemcpy(&apply_status, d_apply_status, sizeof(apply_status), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&result_state, d_controller_state, sizeof(result_state), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(result_headers.data(),
                        d_command_headers,
                        result_headers.size() * sizeof(result_headers[0]),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&runtime,
                        runtime_table.deviceLayerState(0),
                        sizeof(runtime),
                        hipMemcpyDeviceToHost),
              hipSuccess);

    EXPECT_EQ(apply_status.copy_incomplete, 1u);
    EXPECT_EQ(apply_status.changed_layers, 0u);
    EXPECT_EQ(apply_status.applied_arrivals, 0u);
    EXPECT_EQ(result_state.active_wave, 0u);
    EXPECT_EQ(result_state.waves[0].state,
              static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::ReadyToApply));
    EXPECT_EQ(result_state.waves[0].applied_layer_count, 0u);
    EXPECT_EQ(result_headers[0].epoch, 11u);
    EXPECT_EQ(result_headers[0].command_count, 1u);
    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(bank.local_compute_mask[0], 0u);
    EXPECT_EQ(bank.experts[0].owner_participant, 0);
    EXPECT_EQ(bank.resident_participant_mask[0], 0b01u);

    EXPECT_EQ(hipFree(d_plan_entries), hipSuccess);
    EXPECT_EQ(hipFree(d_command_headers), hipSuccess);
    EXPECT_EQ(hipFree(d_controller_state), hipSuccess);
    EXPECT_EQ(hipFree(d_apply_status), hipSuccess);
    EXPECT_EQ(hipFree(d_transfer_slots), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DecodeReadyRebalanceApplyUsesOrderedLayerCursor)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 2;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    for (int layer = 0; layer < 2; ++layer)
    {
        auto update = makeParticipantOneBaseUpdate(1);
        ASSERT_TRUE(runtime_table.prepareInactiveBank(layer, update));
        ASSERT_TRUE(runtime_table.flipActiveBank(layer, update.epoch, stream));
    }

    DeviceMoERebalanceConfig config;
    config.num_layers = 2;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply);

    constexpr uint32_t plan_capacity = 4;
    constexpr uint32_t command_buffer_count = 2;
    constexpr int d_model = 32;
    auto hidden = TestTensorFactory::createFP32({1, d_model});
    auto gate_weights = TestTensorFactory::createFP32({4, d_model});
    std::fill(hidden->mutable_data(), hidden->mutable_data() + hidden->numel(), 0.0f);
    std::fill(gate_weights->mutable_data(), gate_weights->mutable_data() + gate_weights->numel(), 0.0f);
    hidden->mutable_data()[0] = 1.0f;
    for (int expert = 0; expert < 4; ++expert)
        gate_weights->mutable_data()[static_cast<size_t>(expert) * d_model] =
            static_cast<float>(4 - expert);
    auto output_indices = TestTensorFactory::createFP32({1, 2});
    auto output_weights = TestTensorFactory::createFP32({1, 2});
    ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device, stream));
    ASSERT_TRUE(output_indices->ensureOnDevice(device, stream));
    ASSERT_TRUE(output_weights->ensureOnDevice(device, stream));

    std::vector<DeviceMoERebalancePlanEntry> plan_entries(
        static_cast<size_t>(plan_capacity) * command_buffer_count);
    auto &plan = plan_entries[0];
    plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ResidentExpertAssignment);
    plan.layer = 0;
    plan.expert = 0;
    plan.source_participant = 0;
    plan.destination_participant = config.participant_id;
    plan.source_resident_mask = 0b001u;
    plan.destination_slot = 0;
    plan.payload_slot = kDeviceMoEInvalidSlot;

    std::array<DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    for (auto &header : command_headers)
    {
        header.command_capacity = plan_capacity;
        header.participant_id = config.participant_id;
        header.participant_count = config.participant_count;
    }
    command_headers[0].epoch = 17;
    command_headers[0].command_count = 1;

    DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.next_epoch = 18;
    controller_state.active_wave = 0;
    controller_state.wave_count = command_buffer_count;
    auto &wave = controller_state.waves[0];
    wave.epoch = command_headers[0].epoch;
    wave.state = static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::ReadyToApply);
    wave.planned_start_layer = 0;
    wave.planned_layer_count = 2;
    wave.command_count = 1;
    wave.copied_arrivals = 1;

    DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    DeviceMoERebalanceApplyStatus *d_apply_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_entries),
                        plan_entries.size() * sizeof(plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_headers),
                        command_headers.size() * sizeof(command_headers[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_apply_status),
                        sizeof(DeviceMoERebalanceApplyStatus)),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_plan_entries,
                             plan_entries.data(),
                             plan_entries.size() * sizeof(plan_entries[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_command_headers,
                             command_headers.data(),
                             command_headers.size() * sizeof(command_headers[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_controller_state,
                             &controller_state,
                             sizeof(controller_state),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    ROCmMoEKernel kernel(0);
    static_cast<IMoEKernel &>(kernel).setGPUStream(stream);
    auto workspace = bindDefaultMoEWorkspace(kernel, /*max_seq_len=*/64, /*d_model=*/d_model,
                                             /*intermediate=*/8, /*num_experts=*/4, /*top_k=*/2);

    auto run_route_apply = [&](int target_layer)
    {
        ASSERT_TRUE(kernel.decodeRouteSelectWithReadyRebalanceApply(
            runtime_table.deviceLayerState(0),
            runtime_table.deviceLayerState(1),
            hidden.get(),
            gate_weights.get(),
            /*d_model=*/d_model,
            /*num_experts=*/4,
            /*top_k=*/2,
            /*normalize_weights=*/false,
            output_indices.get(),
            output_weights.get(),
            /*write_legacy_outputs=*/true,
            /*update_runtime_histogram=*/true,
            d_plan_entries,
            plan_capacity,
            d_command_headers,
            nullptr,
            0u,
            config,
            d_apply_status,
            d_controller_state,
            target_layer,
            command_buffer_count));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    };

    DeviceMoERebalanceGraphControllerState result_state;
    std::array<DeviceMoERebalanceCommandBufferHeader, command_buffer_count> result_headers{};
    auto read_state = [&]()
    {
        ASSERT_EQ(hipMemcpy(&result_state, d_controller_state, sizeof(result_state), hipMemcpyDeviceToHost),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy(result_headers.data(),
                            d_command_headers,
                            result_headers.size() * sizeof(result_headers[0]),
                            hipMemcpyDeviceToHost),
                  hipSuccess);
    };

    run_route_apply(0);
    read_state();
    EXPECT_EQ(result_state.waves[0].state,
              static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::ReadyToApply));
    EXPECT_EQ(result_state.waves[0].applied_layer_count, 1u);
    EXPECT_EQ(result_headers[0].command_count, 1u);

    run_route_apply(0);
    read_state();
    EXPECT_EQ(result_state.waves[0].state,
              static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::ReadyToApply));
    EXPECT_EQ(result_state.waves[0].applied_layer_count, 1u)
        << "a previously applied layer must not consume the next rolling-wave slot";
    EXPECT_EQ(result_headers[0].command_count, 1u);

    run_route_apply(1);
    read_state();
    EXPECT_EQ(result_state.active_wave, 1u);
    EXPECT_EQ(result_state.waves[0].state,
              static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::Applied));
    EXPECT_EQ(result_state.waves[0].applied_layer_count, 2u);
    EXPECT_EQ(result_headers[0].command_count, 0u);

    EXPECT_EQ(hipFree(d_plan_entries), hipSuccess);
    EXPECT_EQ(hipFree(d_command_headers), hipSuccess);
    EXPECT_EQ(hipFree(d_controller_state), hipSuccess);
    EXPECT_EQ(hipFree(d_apply_status), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceControllerAggregatesRouterBenefitAcrossLayerWindow)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 2;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    for (int layer = 0; layer < 2; ++layer)
    {
        auto update = makeParticipantOneBaseUpdate(1);
        ASSERT_TRUE(runtime_table.prepareInactiveBank(layer, update));
        ASSERT_TRUE(runtime_table.flipActiveBank(layer, update.epoch, stream));
    }

    DeviceMoELayerRuntime seeded_layer{};
    ASSERT_EQ(hipMemcpy(&seeded_layer,
                        runtime_table.deviceLayerState(1),
                        sizeof(seeded_layer),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    seeded_layer.router_hot_cache_eligible_dispatches = 11;
    seeded_layer.router_hot_cache_used_dispatches = 7;
    seeded_layer.router_hot_cache_improved_dispatches = 5;
    seeded_layer.router_hot_cache_default_load_spread_total = 19;
    seeded_layer.router_hot_cache_actual_load_spread_total = 13;
    seeded_layer.router_hot_cache_load_spread_improvement_total = 6;
    seeded_layer.router_hot_cache_active_dispatches = 23;
    seeded_layer.router_hot_cache_miss_dispatches = 17;
    seeded_layer.router_hot_cache_selected_expert_slots = 46;
    seeded_layer.router_hot_cache_replicated_selected_expert_slots = 12;
    ASSERT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(1),
                             &seeded_layer,
                             sizeof(seeded_layer),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    DeviceMoERebalanceConfig config;
    config.num_layers = 2;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;
    config.layer_window_start = 0;
    config.layer_window_count = 2;
    config.layer_wave_count = 1;
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);

    std::vector<uint64_t> gathered(static_cast<size_t>(config.participant_count) *
                                       config.layer_wave_count * config.num_experts,
                                   0);
    gathered[0] = 2; // Keep the current one-layer wave ready while counters live on layer 1.
    uint64_t *d_gathered = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered),
                        gathered.size() * sizeof(uint64_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_status),
                        sizeof(DeviceMoERebalanceStatus)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan),
                        sizeof(DeviceMoERebalancePlanEntry)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_count),
                        sizeof(uint32_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_header),
                        sizeof(DeviceMoERebalanceCommandBufferHeader)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_wave_state),
                        sizeof(DeviceMoERebalanceWaveState)),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered,
                             gathered.data(),
                             gathered.size() * sizeof(uint64_t),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_status, 0, sizeof(DeviceMoERebalanceStatus), stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_plan_count, 0, sizeof(uint32_t), stream), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_command_header,
                             0,
                             sizeof(DeviceMoERebalanceCommandBufferHeader),
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_wave_state,
                             0,
                             sizeof(DeviceMoERebalanceWaveState),
                             stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.runDeviceRebalanceController(
        runtime_table.deviceLayerState(0),
        d_gathered,
        d_status,
        config,
        d_plan,
        d_plan_count,
        1,
        1,
        d_command_header,
        d_wave_state));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceStatus status;
    DeviceMoELayerRuntime result_layer{};
    ASSERT_EQ(hipMemcpy(&status, d_status, sizeof(status), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&result_layer,
                        runtime_table.deviceLayerState(1),
                        sizeof(result_layer),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(status.window_ready_slots, 2u);
    EXPECT_EQ(status.window_required_slots, 2u);
    EXPECT_EQ(status.skipped_busy_wave, 0u);
    EXPECT_EQ(status.skipped_not_ready, 0u);
    ASSERT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.router_hot_cache_eligible_dispatches, 11u);
    EXPECT_EQ(status.router_hot_cache_used_dispatches, 7u);
    EXPECT_EQ(status.router_hot_cache_improved_dispatches, 5u);
    EXPECT_EQ(status.router_hot_cache_default_load_spread_total, 19u);
    EXPECT_EQ(status.router_hot_cache_actual_load_spread_total, 13u);
    EXPECT_EQ(status.router_hot_cache_load_spread_improvement_total, 6u);
    EXPECT_EQ(status.router_hot_cache_active_dispatches, 23u);
    EXPECT_EQ(status.router_hot_cache_miss_dispatches, 17u);
    EXPECT_EQ(status.router_hot_cache_selected_expert_slots, 46u);
    EXPECT_EQ(status.router_hot_cache_replicated_selected_expert_slots, 12u);
    EXPECT_EQ(result_layer.router_hot_cache_active_dispatches, 0u);
    EXPECT_EQ(result_layer.router_hot_cache_eligible_dispatches, 0u);

    EXPECT_EQ(hipFree(d_gathered), hipSuccess);
    EXPECT_EQ(hipFree(d_status), hipSuccess);
    EXPECT_EQ(hipFree(d_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_plan_count), hipSuccess);
    EXPECT_EQ(hipFree(d_command_header), hipSuccess);
    EXPECT_EQ(hipFree(d_wave_state), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceControllerSkipsMissingArrivalThatWorsensImbalance)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    update.resident_participant_mask[0] = 0b001u;
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream));

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);

    std::vector<uint64_t> gathered(static_cast<size_t>(config.participant_count) *
                                       config.num_layers * config.num_experts,
                                   0);
    gathered[0] = 6; // participant 0, layer 0, expert 0.
    gathered[config.num_experts + 2] = 10; // participant 1 already carries a heavier local expert.

    uint64_t *d_gathered = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered),
                        gathered.size() * sizeof(uint64_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_status),
                        sizeof(DeviceMoERebalanceStatus)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan),
                        sizeof(DeviceMoERebalancePlanEntry)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_count),
                        sizeof(uint32_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_header),
                        sizeof(DeviceMoERebalanceCommandBufferHeader)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_wave_state),
                        sizeof(DeviceMoERebalanceWaveState)),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered, gathered.data(),
                             gathered.size() * sizeof(uint64_t),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.runDeviceRebalanceController(
        runtime_table.deviceLayerState(0),
        d_gathered,
        d_status,
        config,
        d_plan,
        d_plan_count,
        1,
        1,
        d_command_header,
        d_wave_state));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceStatus status;
    DeviceMoERebalanceCommandBufferHeader command_header;
    uint32_t plan_count = 99;
    DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(hipMemcpy(&status, d_status, sizeof(status), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&command_header, d_command_header, sizeof(command_header), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&plan_count, d_plan_count, sizeof(plan_count), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime),
                        hipMemcpyDeviceToHost),
              hipSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.planned_arrivals, 0u);
    EXPECT_EQ(status.selected_replicas, 0u);
    EXPECT_EQ(status.skipped_no_improvement, 1u);
    EXPECT_EQ(status.pre_policy_load_total, 16u);
    EXPECT_EQ(status.pre_policy_load_min, 0u);
    EXPECT_EQ(status.pre_policy_load_max, 10u);
    EXPECT_EQ(status.post_policy_load_total, 16u);
    EXPECT_EQ(status.post_policy_load_min, 0u);
    EXPECT_EQ(status.post_policy_load_max, 10u);
    EXPECT_EQ(status.pre_policy_imbalance_numerator,
              status.post_policy_imbalance_numerator);
    EXPECT_EQ(status.candidate_arrivals_considered, 1u);
    EXPECT_EQ(status.candidate_arrivals_below_floor, 1u);
    EXPECT_EQ(status.candidate_arrivals_pruned_by_count_bound, 0u);
    EXPECT_EQ(status.candidate_load_spread_improvement_total, 0u);
    EXPECT_EQ(status.candidate_load_spread_improvement_max, 0u);
    EXPECT_EQ(status.accepted_load_spread_improvement_total, 0u);
    EXPECT_EQ(status.accepted_load_spread_improvement_max, 0u);
    EXPECT_EQ(plan_count, 0u);
    EXPECT_EQ(command_header.command_count, 0u);
    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(bank.local_compute_mask[0], 0u);
    EXPECT_FALSE(hasMoEExpertFlag(bank.experts[0].flags,
                                  DeviceMoEExpertFlags::Replicated));

    EXPECT_EQ(hipFree(d_gathered), hipSuccess);
    EXPECT_EQ(hipFree(d_status), hipSuccess);
    EXPECT_EQ(hipFree(d_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_plan_count), hipSuccess);
    EXPECT_EQ(hipFree(d_command_header), hipSuccess);
    EXPECT_EQ(hipFree(d_wave_state), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceControllerPrunesMissingArrivalBelowCountBound)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    update.resident_participant_mask[0] = 0b001u;
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream));

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;
    config.min_load_spread_improvement = 8;
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);

    std::vector<uint64_t> gathered(static_cast<size_t>(config.participant_count) *
                                       config.num_layers * config.num_experts,
                                   0);
    gathered[0] = 6; // Window is ready, but this count cannot meet the configured floor.

    uint64_t *d_gathered = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered),
                        gathered.size() * sizeof(uint64_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_status),
                        sizeof(DeviceMoERebalanceStatus)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan),
                        sizeof(DeviceMoERebalancePlanEntry)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_count),
                        sizeof(uint32_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_header),
                        sizeof(DeviceMoERebalanceCommandBufferHeader)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_wave_state),
                        sizeof(DeviceMoERebalanceWaveState)),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered, gathered.data(),
                             gathered.size() * sizeof(uint64_t),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.runDeviceRebalanceController(
        runtime_table.deviceLayerState(0),
        d_gathered,
        d_status,
        config,
        d_plan,
        d_plan_count,
        1,
        1,
        d_command_header,
        d_wave_state));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceStatus status;
    DeviceMoERebalanceCommandBufferHeader command_header;
    uint32_t plan_count = 99;
    ASSERT_EQ(hipMemcpy(&status, d_status, sizeof(status), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&command_header, d_command_header, sizeof(command_header), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&plan_count, d_plan_count, sizeof(plan_count), hipMemcpyDeviceToHost), hipSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.planned_arrivals, 0u);
    EXPECT_EQ(status.selected_replicas, 0u);
    EXPECT_EQ(status.skipped_no_improvement, 1u);
    EXPECT_EQ(status.candidate_arrivals_considered, 0u);
    EXPECT_EQ(status.candidate_arrivals_below_floor, 0u);
    EXPECT_EQ(status.candidate_arrivals_pruned_by_count_bound, 1u);
    EXPECT_EQ(status.candidate_load_spread_improvement_total, 0u);
    EXPECT_EQ(status.candidate_load_spread_improvement_max, 0u);
    EXPECT_EQ(status.accepted_load_spread_improvement_total, 0u);
    EXPECT_EQ(status.accepted_load_spread_improvement_max, 0u);
    EXPECT_EQ(plan_count, 0u);
    EXPECT_EQ(command_header.command_count, 0u);

    EXPECT_EQ(hipFree(d_gathered), hipSuccess);
    EXPECT_EQ(hipFree(d_status), hipSuccess);
    EXPECT_EQ(hipFree(d_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_plan_count), hipSuccess);
    EXPECT_EQ(hipFree(d_command_header), hipSuccess);
    EXPECT_EQ(hipFree(d_wave_state), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalancePackDirectoryExportsOnlyLocalResidents)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    update.resident_participant_mask[0] = 0b001u;
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream));

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;

    DeviceMoEExpertDirectoryEntry *d_directory = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_directory),
                        config.num_experts * sizeof(DeviceMoEExpertDirectoryEntry)),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.packDeviceRebalanceDirectory(
        runtime_table.deviceLayerState(0),
        d_directory,
        config));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::vector<DeviceMoEExpertDirectoryEntry> directory(config.num_experts);
    ASSERT_EQ(hipMemcpy(directory.data(), d_directory,
                        directory.size() * sizeof(directory[0]),
                        hipMemcpyDeviceToHost),
              hipSuccess);

    EXPECT_FALSE(deviceMoEDirectoryEntryReady(directory[0], 1, 0, 0));
    ASSERT_TRUE(deviceMoEDirectoryEntryReady(directory[2], 1, 0, 2));
    EXPECT_EQ(directory[2].descriptor.logical_expert_id, 2);
    EXPECT_EQ(directory[2].slot_index, 2u);
    EXPECT_TRUE((directory[2].flags &
                 static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::LocalCompute)) != 0u);
    EXPECT_FALSE(deviceMoEDirectoryEntryReady(directory[3], 1, 0, 3));

    EXPECT_EQ(hipFree(d_directory), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceUnpackPreservesSourceSidePackErrors)
{
    SKIP_IF_NO_ROCM();

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 2;
    config.window_size_tokens = 1;

    constexpr uint32_t plan_capacity = 1;
    constexpr uint32_t command_buffer_count = 1;
    constexpr uint32_t local_payload_slot_count = 1;
    constexpr uint64_t payload_slot_bytes = 512;

    DeviceMoERebalancePlanEntry plan;
    plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    plan.layer = 0;
    plan.expert = 0;
    plan.source_participant = 1;
    plan.destination_participant = 0;
    plan.source_resident_mask = 0b10u;
    plan.destination_slot = 0;
    plan.payload_slot = 0;

    DeviceMoERebalanceCommandBufferHeader command_header;
    command_header.epoch = 11;
    command_header.command_count = 1;
    command_header.command_capacity = plan_capacity;
    command_header.participant_id = config.participant_id;
    command_header.participant_count = config.participant_count;

    const size_t source_descriptor_count =
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count) *
        static_cast<size_t>(plan_capacity);
    const size_t local_payload_bytes =
        static_cast<size_t>(local_payload_slot_count) *
        static_cast<size_t>(payload_slot_bytes);
    const size_t gathered_payload_bytes =
        static_cast<size_t>(config.participant_count) * local_payload_bytes;

    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    DeviceMoEExpertDirectoryEntry *d_local_source_descriptors = nullptr;
    uint8_t *d_local_payload = nullptr;
    uint8_t *d_gathered_payload = nullptr;
    DeviceMoEExpertDirectoryEntry *d_transfer_slots = nullptr;
    DeviceMoERebalanceApplyStatus *d_copy_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan), sizeof(plan)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_header), sizeof(command_header)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_source_descriptors),
                        source_descriptor_count * sizeof(DeviceMoEExpertDirectoryEntry)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_payload), local_payload_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_payload), gathered_payload_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_transfer_slots),
                        sizeof(DeviceMoEExpertDirectoryEntry)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_copy_status),
                        sizeof(DeviceMoERebalanceApplyStatus)),
              hipSuccess);

    ASSERT_EQ(hipMemcpyAsync(d_plan, &plan, sizeof(plan), hipMemcpyHostToDevice, stream), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_command_header,
                             &command_header,
                             sizeof(command_header),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_local_source_descriptors,
                             0,
                             source_descriptor_count * sizeof(DeviceMoEExpertDirectoryEntry),
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_local_payload, 0, local_payload_bytes, stream), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_gathered_payload, 0, gathered_payload_bytes, stream), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_transfer_slots,
                             0,
                             sizeof(DeviceMoEExpertDirectoryEntry),
                             stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.packDeviceRebalanceCompactPayloads(
        d_plan,
        d_command_header,
        plan_capacity,
        d_local_source_descriptors,
        d_local_payload,
        local_payload_slot_count,
        payload_slot_bytes,
        config,
        d_copy_status,
        nullptr,
        command_buffer_count));
    ASSERT_TRUE(gpu_kernel.unpackDeviceRebalanceCollectivePayloads(
        d_plan,
        nullptr,
        plan_capacity,
        d_command_header,
        d_gathered_payload,
        local_payload_slot_count,
        payload_slot_bytes,
        d_transfer_slots,
        1,
        config,
        d_copy_status,
        nullptr,
        command_buffer_count));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceApplyStatus copy_status{};
    ASSERT_EQ(hipMemcpy(&copy_status, d_copy_status, sizeof(copy_status), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(copy_status.status_code,
              static_cast<uint32_t>(DeviceMoERebalanceApplyStatusCode::Ok));
    EXPECT_EQ(copy_status.missing_source_descriptors, 1u)
        << "unpack must not clear source-side compact pack failures from the same transfer wave";
    EXPECT_EQ(copy_status.copied_arrivals, 0u);
    EXPECT_EQ(copy_status.skipped_wrong_destination, 1u);

    EXPECT_EQ(hipFree(d_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_command_header), hipSuccess);
    EXPECT_EQ(hipFree(d_local_source_descriptors), hipSuccess);
    EXPECT_EQ(hipFree(d_local_payload), hipSuccess);
    EXPECT_EQ(hipFree(d_gathered_payload), hipSuccess);
    EXPECT_EQ(hipFree(d_transfer_slots), hipSuccess);
    EXPECT_EQ(hipFree(d_copy_status), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceCompactPayloadRejectsResidentSourceWithoutLocalSlot)
{
    SKIP_IF_NO_ROCM();

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 2;
    config.window_size_tokens = 1;

    constexpr uint32_t plan_capacity = 1;
    constexpr uint32_t command_buffer_count = 1;
    constexpr uint32_t local_payload_slot_count = 1;
    constexpr uint64_t payload_slot_bytes = 512;
    constexpr uint32_t n = 2;
    constexpr uint32_t k = 32;
    constexpr uint32_t blocks_per_row = 1;
    constexpr size_t blocks = n * blocks_per_row;
    constexpr size_t projection_payload_bytes = blocks * 16;
    constexpr size_t projection_scales_bytes = blocks * sizeof(uint16_t);

    uint8_t *d_src_payload = nullptr;
    void *d_src_scales = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_src_payload), projection_payload_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_src_scales, projection_scales_bytes), hipSuccess);

    DeviceNativeVNNIMatrixDesc projection_desc;
    projection_desc.payload = d_src_payload;
    projection_desc.scales = d_src_scales;
    projection_desc.n = n;
    projection_desc.k = k;
    projection_desc.blocks_per_row = blocks_per_row;
    projection_desc.codebook_id = 0;

    DeviceMoEExpertDirectoryEntry source_entry;
    source_entry.descriptor.gate = projection_desc;
    source_entry.descriptor.up = projection_desc;
    source_entry.descriptor.down = projection_desc;
    source_entry.descriptor.logical_expert_id = 0;
    source_entry.descriptor.owner_participant = 1;
    source_entry.descriptor.local_slot = -1;
    source_entry.descriptor.flags =
        toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                         DeviceMoEExpertFlags::Resident);
    source_entry.layer = 0;
    source_entry.expert = 0;
    source_entry.participant = 1;
    source_entry.resident_mask = 0b10u;
    source_entry.slot_index = kDeviceMoEInvalidSlot;
    source_entry.flags =
        static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
        static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Resident);
    ASSERT_TRUE(deviceMoEPopulateDirectoryFormat(source_entry));

    const size_t source_descriptor_count =
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count) *
        static_cast<size_t>(plan_capacity);
    std::vector<DeviceMoEExpertDirectoryEntry> local_source_descriptors(
        source_descriptor_count);
    local_source_descriptors[0] = source_entry;

    DeviceMoERebalancePlanEntry plan;
    plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    plan.layer = 0;
    plan.expert = 0;
    plan.source_participant = 1;
    plan.destination_participant = 0;
    plan.source_resident_mask = 0b10u;
    plan.destination_slot = 0;
    plan.payload_slot = 0;

    DeviceMoERebalanceCommandBufferHeader command_header;
    command_header.epoch = 12;
    command_header.command_count = 1;
    command_header.command_capacity = plan_capacity;
    command_header.participant_id = config.participant_id;
    command_header.participant_count = config.participant_count;

    const size_t local_payload_bytes =
        static_cast<size_t>(local_payload_slot_count) *
        static_cast<size_t>(payload_slot_bytes);
    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    DeviceMoEExpertDirectoryEntry *d_local_source_descriptors = nullptr;
    uint8_t *d_local_payload = nullptr;
    DeviceMoERebalanceApplyStatus *d_copy_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan), sizeof(plan)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_header), sizeof(command_header)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_source_descriptors),
                        local_source_descriptors.size() * sizeof(local_source_descriptors[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_payload), local_payload_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_copy_status),
                        sizeof(DeviceMoERebalanceApplyStatus)),
              hipSuccess);

    ASSERT_EQ(hipMemcpyAsync(d_plan, &plan, sizeof(plan), hipMemcpyHostToDevice, stream), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_command_header,
                             &command_header,
                             sizeof(command_header),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_local_source_descriptors,
                             local_source_descriptors.data(),
                             local_source_descriptors.size() * sizeof(local_source_descriptors[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_local_payload, 0, local_payload_bytes, stream), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.packDeviceRebalanceCompactPayloads(
        d_plan,
        d_command_header,
        plan_capacity,
        d_local_source_descriptors,
        d_local_payload,
        local_payload_slot_count,
        payload_slot_bytes,
        config,
        d_copy_status,
        nullptr,
        command_buffer_count));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceApplyStatus copy_status{};
    DeviceMoEExpertDirectoryEntry payload_header{};
    ASSERT_EQ(hipMemcpy(&copy_status, d_copy_status, sizeof(copy_status), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&payload_header,
                        d_local_payload,
                        sizeof(payload_header),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(copy_status.missing_source_descriptors, 1u)
        << "a resident mask without a local slot is not a physical source for payload bytes";
    EXPECT_EQ(copy_status.invalid_plan_entries, 0u);
    EXPECT_EQ(payload_header.flags, 0u);

    EXPECT_EQ(hipFree(d_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_command_header), hipSuccess);
    EXPECT_EQ(hipFree(d_local_source_descriptors), hipSuccess);
    EXPECT_EQ(hipFree(d_local_payload), hipSuccess);
    EXPECT_EQ(hipFree(d_copy_status), hipSuccess);
    EXPECT_EQ(hipFree(d_src_payload), hipSuccess);
    EXPECT_EQ(hipFree(d_src_scales), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceDomainProjectionPublishesRootPayloadStatus)
{
    SKIP_IF_NO_ROCM();

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.root_participant = 0;
    config.window_size_tokens = 1;

    constexpr uint32_t plan_capacity = 4;
    constexpr uint32_t command_buffer_count = 2;
    constexpr uint32_t payload_slot_capacity = 4;
    std::vector<DeviceMoERebalancePlanEntry> gathered_plan_entries(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count) *
        static_cast<size_t>(plan_capacity));
    std::vector<DeviceMoERebalanceCommandBufferHeader> gathered_headers(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count));
    std::vector<DeviceMoERebalanceWaveState> gathered_wave_states(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count));

    DeviceMoERebalancePlanEntry root_plan;
    root_plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    root_plan.layer = 0;
    root_plan.expert = 2;
    root_plan.source_participant = 0;
    root_plan.destination_participant = 1;
    root_plan.source_resident_mask = 0b001u;
    root_plan.destination_slot = 0;
    root_plan.payload_slot = 1;
    gathered_plan_entries[0] = root_plan;

    auto &root_header = gathered_headers[0];
    root_header.epoch = 7;
    root_header.phase =
        static_cast<uint32_t>(DeviceMoERebalancePipelinePhase::PlanAssignments);
    root_header.command_count = 1;
    root_header.command_capacity = plan_capacity;
    root_header.participant_id = 0;
    root_header.participant_count = config.participant_count;

    auto &root_wave = gathered_wave_states[0];
    root_wave.epoch = root_header.epoch;
    root_wave.next_start_layer = 5;
    root_wave.planned_start_layer = 1;
    root_wave.planned_layer_count = 1;
    root_wave.command_capacity = plan_capacity;
    root_wave.participant_id = 0;
    root_wave.participant_count = config.participant_count;
    root_wave.requested_payload_slots = 2;
    root_wave.payload_bucket_slots = 2;
    root_wave.payload_bucket_index = 1;
    root_wave.payload_bucket_overflow = 0;

    DeviceMoERebalanceStatus status;
    std::vector<DeviceMoERebalancePlanEntry> local_plan_entries(
        static_cast<size_t>(command_buffer_count) * static_cast<size_t>(plan_capacity));
    std::vector<DeviceMoERebalanceCommandBufferHeader> local_headers(
        command_buffer_count);
    std::vector<DeviceMoERebalanceWaveState> local_wave_states(
        command_buffer_count);

    DeviceMoERebalancePlanEntry *d_gathered_plan = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_gathered_headers = nullptr;
    DeviceMoERebalanceWaveState *d_gathered_wave_states = nullptr;
    DeviceMoERebalancePlanEntry *d_local_plan = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_local_headers = nullptr;
    DeviceMoERebalanceWaveState *d_local_wave_states = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_plan),
                        gathered_plan_entries.size() * sizeof(gathered_plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_headers),
                        gathered_headers.size() * sizeof(gathered_headers[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_wave_states),
                        gathered_wave_states.size() * sizeof(gathered_wave_states[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_plan),
                        local_plan_entries.size() * sizeof(local_plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_headers),
                        local_headers.size() * sizeof(local_headers[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_wave_states),
                        local_wave_states.size() * sizeof(local_wave_states[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_status), sizeof(status)), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered_plan,
                             gathered_plan_entries.data(),
                             gathered_plan_entries.size() * sizeof(gathered_plan_entries[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered_headers,
                             gathered_headers.data(),
                             gathered_headers.size() * sizeof(gathered_headers[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered_wave_states,
                             gathered_wave_states.data(),
                             gathered_wave_states.size() * sizeof(gathered_wave_states[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_local_plan, 0, local_plan_entries.size() * sizeof(local_plan_entries[0]), stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_local_headers, 0, local_headers.size() * sizeof(local_headers[0]), stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_local_wave_states, 0, local_wave_states.size() * sizeof(local_wave_states[0]), stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_status, &status, sizeof(status), hipMemcpyHostToDevice, stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.projectDeviceRebalanceDomainCommands(
        d_gathered_plan,
        d_gathered_headers,
        plan_capacity,
        d_local_plan,
        d_local_headers,
        config,
        d_status,
        payload_slot_capacity,
        command_buffer_count,
        d_gathered_wave_states,
        d_local_wave_states));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(hipMemcpy(local_plan_entries.data(),
                        d_local_plan,
                        local_plan_entries.size() * sizeof(local_plan_entries[0]),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(local_headers.data(),
                        d_local_headers,
                        local_headers.size() * sizeof(local_headers[0]),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(local_wave_states.data(),
                        d_local_wave_states,
                        local_wave_states.size() * sizeof(local_wave_states[0]),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&status, d_status, sizeof(status), hipMemcpyDeviceToHost),
              hipSuccess);

    EXPECT_EQ(local_headers[0].participant_id, config.participant_id);
    EXPECT_EQ(local_headers[0].participant_count, config.participant_count);
    EXPECT_EQ(local_headers[0].epoch, root_header.epoch);
    EXPECT_EQ(local_headers[0].command_count, 1u);
    EXPECT_EQ(local_headers[1].command_count, 0u);
    EXPECT_EQ(local_plan_entries[0].source_participant, root_plan.source_participant);
    EXPECT_EQ(local_plan_entries[0].destination_participant, root_plan.destination_participant);
    EXPECT_EQ(local_plan_entries[0].payload_slot, root_plan.payload_slot);
    EXPECT_EQ(local_wave_states[0].participant_id, config.participant_id);
    EXPECT_EQ(local_wave_states[0].participant_count, config.participant_count);
    EXPECT_EQ(local_wave_states[0].epoch, root_wave.epoch);
    EXPECT_EQ(local_wave_states[0].next_start_layer, root_wave.next_start_layer);
    EXPECT_EQ(local_wave_states[0].planned_start_layer, root_wave.planned_start_layer);
    EXPECT_EQ(local_wave_states[0].planned_layer_count, root_wave.planned_layer_count);
    EXPECT_EQ(local_wave_states[0].command_capacity, plan_capacity);
    EXPECT_EQ(local_wave_states[0].requested_payload_slots, root_wave.requested_payload_slots);
    EXPECT_EQ(local_wave_states[0].payload_bucket_slots, root_wave.payload_bucket_slots);
    EXPECT_EQ(local_wave_states[0].payload_bucket_index, root_wave.payload_bucket_index);
    EXPECT_EQ(local_wave_states[0].payload_bucket_overflow, root_wave.payload_bucket_overflow);
    EXPECT_EQ(local_wave_states[1].epoch, 0u);
    EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.windows_observed, 1u);
    EXPECT_EQ(status.windows_applied, 1u);
    EXPECT_EQ(status.last_epoch, root_header.epoch);
    EXPECT_EQ(status.planned_arrivals, 1u);
    EXPECT_EQ(status.payload_bucket_requested_slots, 2u);
    EXPECT_EQ(status.payload_bucket_slots, 2u);
    EXPECT_EQ(status.payload_bucket_index, 1u);
    EXPECT_EQ(status.payload_bucket_overflow, 0u);

    EXPECT_EQ(hipFree(d_gathered_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_gathered_headers), hipSuccess);
    EXPECT_EQ(hipFree(d_gathered_wave_states), hipSuccess);
    EXPECT_EQ(hipFree(d_local_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_local_headers), hipSuccess);
    EXPECT_EQ(hipFree(d_local_wave_states), hipSuccess);
    EXPECT_EQ(hipFree(d_status), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceDomainProjectionRejectsNonResidentSourceParticipant)
{
    SKIP_IF_NO_ROCM();

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 0;
    config.participant_count = 3;
    config.root_participant = 0;
    config.window_size_tokens = 1;

    constexpr uint32_t plan_capacity = 4;
    constexpr uint32_t command_buffer_count = 2;
    constexpr uint32_t payload_slot_capacity = 4;
    std::vector<DeviceMoERebalancePlanEntry> gathered_plan_entries(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count) *
        static_cast<size_t>(plan_capacity));
    std::vector<DeviceMoERebalanceCommandBufferHeader> gathered_headers(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count));
    std::vector<DeviceMoERebalanceWaveState> gathered_wave_states(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count));

    DeviceMoERebalancePlanEntry root_plan;
    root_plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    root_plan.layer = 0;
    root_plan.expert = 2;
    root_plan.source_participant = 1;
    root_plan.destination_participant = 0;
    root_plan.source_resident_mask = 0b001u;
    root_plan.destination_slot = 0;
    root_plan.payload_slot = 0;
    gathered_plan_entries[0] = root_plan;

    auto &root_header = gathered_headers[0];
    root_header.epoch = 13;
    root_header.phase =
        static_cast<uint32_t>(DeviceMoERebalancePipelinePhase::PlanAssignments);
    root_header.command_count = 1;
    root_header.command_capacity = plan_capacity;
    root_header.participant_id = 0;
    root_header.participant_count = config.participant_count;

    auto &root_wave = gathered_wave_states[0];
    root_wave.epoch = root_header.epoch;
    root_wave.planned_start_layer = 0;
    root_wave.planned_layer_count = 1;
    root_wave.command_capacity = plan_capacity;
    root_wave.participant_id = 0;
    root_wave.participant_count = config.participant_count;
    root_wave.requested_payload_slots = 1;
    root_wave.payload_bucket_slots = 1;

    DeviceMoERebalanceStatus status;
    std::vector<DeviceMoERebalancePlanEntry> local_plan_entries(
        static_cast<size_t>(command_buffer_count) * static_cast<size_t>(plan_capacity));
    std::vector<DeviceMoERebalanceCommandBufferHeader> local_headers(
        command_buffer_count);
    std::vector<DeviceMoERebalanceWaveState> local_wave_states(
        command_buffer_count);

    DeviceMoERebalancePlanEntry *d_gathered_plan = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_gathered_headers = nullptr;
    DeviceMoERebalanceWaveState *d_gathered_wave_states = nullptr;
    DeviceMoERebalancePlanEntry *d_local_plan = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_local_headers = nullptr;
    DeviceMoERebalanceWaveState *d_local_wave_states = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_plan),
                        gathered_plan_entries.size() * sizeof(gathered_plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_headers),
                        gathered_headers.size() * sizeof(gathered_headers[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_wave_states),
                        gathered_wave_states.size() * sizeof(gathered_wave_states[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_plan),
                        local_plan_entries.size() * sizeof(local_plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_headers),
                        local_headers.size() * sizeof(local_headers[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_wave_states),
                        local_wave_states.size() * sizeof(local_wave_states[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_status), sizeof(status)), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered_plan,
                             gathered_plan_entries.data(),
                             gathered_plan_entries.size() * sizeof(gathered_plan_entries[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered_headers,
                             gathered_headers.data(),
                             gathered_headers.size() * sizeof(gathered_headers[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered_wave_states,
                             gathered_wave_states.data(),
                             gathered_wave_states.size() * sizeof(gathered_wave_states[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_local_plan, 0, local_plan_entries.size() * sizeof(local_plan_entries[0]), stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_local_headers, 0, local_headers.size() * sizeof(local_headers[0]), stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_local_wave_states, 0, local_wave_states.size() * sizeof(local_wave_states[0]), stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_status, &status, sizeof(status), hipMemcpyHostToDevice, stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.projectDeviceRebalanceDomainCommands(
        d_gathered_plan,
        d_gathered_headers,
        plan_capacity,
        d_local_plan,
        d_local_headers,
        config,
        d_status,
        payload_slot_capacity,
        command_buffer_count,
        d_gathered_wave_states,
        d_local_wave_states));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(hipMemcpy(local_plan_entries.data(),
                        d_local_plan,
                        local_plan_entries.size() * sizeof(local_plan_entries[0]),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(local_headers.data(),
                        d_local_headers,
                        local_headers.size() * sizeof(local_headers[0]),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(local_wave_states.data(),
                        d_local_wave_states,
                        local_wave_states.size() * sizeof(local_wave_states[0]),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&status, d_status, sizeof(status), hipMemcpyDeviceToHost),
              hipSuccess);

    EXPECT_EQ(local_headers[0].participant_id, config.participant_id);
    EXPECT_EQ(local_headers[0].participant_count, config.participant_count);
    EXPECT_EQ(local_headers[0].epoch, root_header.epoch);
    EXPECT_EQ(local_headers[0].command_count, 0u);
    EXPECT_EQ(local_plan_entries[0].op, 0u);
    EXPECT_EQ(local_wave_states[0].requested_payload_slots, 0u);
    EXPECT_EQ(local_wave_states[0].payload_bucket_slots, 0u);
    EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.windows_observed, 1u);
    EXPECT_EQ(status.windows_applied, 0u);
    EXPECT_EQ(status.planned_arrivals, 0u);
    EXPECT_EQ(status.invalid_runtime_layers, 1u);
    EXPECT_EQ(status.payload_bucket_requested_slots, 0u);
    EXPECT_EQ(status.payload_bucket_slots, 0u);
    EXPECT_EQ(status.payload_source_participant_mask, 0u);
    EXPECT_EQ(status.payload_destination_participant_mask, 0u);
    EXPECT_EQ(status.payload_edge_mask, 0ULL);

    EXPECT_EQ(hipFree(d_gathered_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_gathered_headers), hipSuccess);
    EXPECT_EQ(hipFree(d_gathered_wave_states), hipSuccess);
    EXPECT_EQ(hipFree(d_local_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_local_headers), hipSuccess);
    EXPECT_EQ(hipFree(d_local_wave_states), hipSuccess);
    EXPECT_EQ(hipFree(d_status), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceDomainProjectionStatusAggregatesRootPayloadAcrossWaves)
{
    SKIP_IF_NO_ROCM();

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.root_participant = 0;
    config.window_size_tokens = 1;

    constexpr uint32_t plan_capacity = 4;
    constexpr uint32_t command_buffer_count = 2;
    constexpr uint32_t payload_slot_capacity = 4;
    std::vector<DeviceMoERebalancePlanEntry> gathered_plan_entries(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count) *
        static_cast<size_t>(plan_capacity));
    std::vector<DeviceMoERebalanceCommandBufferHeader> gathered_headers(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count));

    DeviceMoERebalancePlanEntry stale_plan;
    stale_plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    stale_plan.layer = 0;
    stale_plan.expert = 2;
    stale_plan.source_participant = 0;
    stale_plan.destination_participant = 1;
    stale_plan.source_resident_mask = 0b001u;
    stale_plan.destination_slot = 0;
    stale_plan.payload_slot = 1;
    gathered_plan_entries[0] = stale_plan;

    auto &stale_header = gathered_headers[0];
    stale_header.epoch = 7;
    stale_header.phase =
        static_cast<uint32_t>(DeviceMoERebalancePipelinePhase::PlanAssignments);
    stale_header.command_count = 1;
    stale_header.command_capacity = plan_capacity;
    stale_header.participant_id = 0;
    stale_header.participant_count = config.participant_count;

    DeviceMoERebalanceStatus status;
    std::vector<DeviceMoERebalancePlanEntry> local_plan_entries(
        static_cast<size_t>(command_buffer_count) * static_cast<size_t>(plan_capacity));
    std::vector<DeviceMoERebalanceCommandBufferHeader> local_headers(
        command_buffer_count);

    DeviceMoERebalancePlanEntry *d_gathered_plan = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_gathered_headers = nullptr;
    DeviceMoERebalancePlanEntry *d_local_plan = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_local_headers = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_plan),
                        gathered_plan_entries.size() * sizeof(gathered_plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_headers),
                        gathered_headers.size() * sizeof(gathered_headers[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_plan),
                        local_plan_entries.size() * sizeof(local_plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_headers),
                        local_headers.size() * sizeof(local_headers[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_status), sizeof(status)), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered_plan,
                             gathered_plan_entries.data(),
                             gathered_plan_entries.size() * sizeof(gathered_plan_entries[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered_headers,
                             gathered_headers.data(),
                             gathered_headers.size() * sizeof(gathered_headers[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_local_plan, 0, local_plan_entries.size() * sizeof(local_plan_entries[0]), stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_local_headers, 0, local_headers.size() * sizeof(local_headers[0]), stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_status, &status, sizeof(status), hipMemcpyHostToDevice, stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.projectDeviceRebalanceDomainCommands(
        d_gathered_plan,
        d_gathered_headers,
        plan_capacity,
        d_local_plan,
        d_local_headers,
        config,
        d_status,
        payload_slot_capacity,
        command_buffer_count));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(hipMemcpy(local_headers.data(),
                        d_local_headers,
                        local_headers.size() * sizeof(local_headers[0]),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&status, d_status, sizeof(status), hipMemcpyDeviceToHost),
              hipSuccess);

    EXPECT_EQ(local_headers[0].command_count, 1u);
    EXPECT_EQ(local_headers[1].command_count, 0u);
    EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.planned_arrivals, 1u);
    EXPECT_EQ(status.payload_bucket_requested_slots, 2u);
    EXPECT_EQ(status.payload_bucket_slots, 2u);
    EXPECT_NE(status.payload_edge_mask, 0ULL);

    EXPECT_EQ(hipFree(d_gathered_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_gathered_headers), hipSuccess);
    EXPECT_EQ(hipFree(d_local_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_local_headers), hipSuccess);
    EXPECT_EQ(hipFree(d_status), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, PrefillLLEPDomainProjectionMergesDestinationLocalRequestsForSourcePacking)
{
    SKIP_IF_NO_ROCM();

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 8;
    config.top_k = 2;
    config.participant_id = 0;
    config.participant_count = 3;
    config.root_participant = 0;
    config.window_size_tokens = 1;

    constexpr uint32_t plan_capacity = 4;
    constexpr uint32_t command_buffer_count = 1;
    constexpr uint32_t payload_slot_capacity = 4;
    std::vector<DeviceMoERebalancePlanEntry> gathered_plan_entries(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count) *
        static_cast<size_t>(plan_capacity));
    std::vector<DeviceMoERebalanceCommandBufferHeader> gathered_headers(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count));

    auto make_header = [&](uint32_t participant, uint32_t count)
    {
        DeviceMoERebalanceCommandBufferHeader header;
        header.epoch = 11;
        header.phase =
            static_cast<uint32_t>(DeviceMoERebalancePipelinePhase::PlanAssignments);
        header.command_count = count;
        header.command_capacity = plan_capacity;
        header.participant_id = participant;
        header.participant_count = config.participant_count;
        return header;
    };
    gathered_headers[1] = make_header(1, 1);
    gathered_headers[2] = make_header(2, 1);

    auto make_plan = [&](uint32_t expert, uint32_t destination)
    {
        DeviceMoERebalancePlanEntry plan;
        plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
        plan.layer = 0;
        plan.expert = expert;
        plan.source_participant = 0;
        plan.destination_participant = destination;
        plan.source_resident_mask = 0b001u;
        plan.destination_slot = 0;
        plan.payload_slot = 0;
        return plan;
    };
    gathered_plan_entries[1 * plan_capacity] = make_plan(3, 1);
    gathered_plan_entries[2 * plan_capacity] = make_plan(5, 2);

    std::vector<DeviceMoERebalancePlanEntry> local_plan_entries(plan_capacity);
    uint32_t local_plan_count = 0;
    DeviceMoERebalanceCommandBufferHeader local_header;
    DeviceMoERebalanceStatus status;

    DeviceMoERebalancePlanEntry *d_gathered_plan = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_gathered_headers = nullptr;
    DeviceMoERebalancePlanEntry *d_local_plan = nullptr;
    uint32_t *d_local_plan_count = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_local_header = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_plan),
                        gathered_plan_entries.size() * sizeof(gathered_plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_headers),
                        gathered_headers.size() * sizeof(gathered_headers[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_plan),
                        local_plan_entries.size() * sizeof(local_plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_plan_count), sizeof(local_plan_count)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_header), sizeof(local_header)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_status), sizeof(status)), hipSuccess);

    ASSERT_EQ(hipMemcpyAsync(d_gathered_plan,
                             gathered_plan_entries.data(),
                             gathered_plan_entries.size() * sizeof(gathered_plan_entries[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered_headers,
                             gathered_headers.data(),
                             gathered_headers.size() * sizeof(gathered_headers[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_local_plan, 0, local_plan_entries.size() * sizeof(local_plan_entries[0]), stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_local_plan_count, 0, sizeof(local_plan_count), stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_local_header, 0, sizeof(local_header), stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_status, &status, sizeof(status), hipMemcpyHostToDevice, stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.projectPrefillLeastLoadedDomainCommands(
        d_gathered_plan,
        d_gathered_headers,
        plan_capacity,
        d_local_plan,
        d_local_plan_count,
        d_local_header,
        config,
        d_status,
        payload_slot_capacity,
        command_buffer_count));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(hipMemcpy(local_plan_entries.data(),
                        d_local_plan,
                        local_plan_entries.size() * sizeof(local_plan_entries[0]),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&local_plan_count, d_local_plan_count, sizeof(local_plan_count), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&local_header, d_local_header, sizeof(local_header), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&status, d_status, sizeof(status), hipMemcpyDeviceToHost),
              hipSuccess);

    EXPECT_EQ(local_header.participant_id, config.participant_id);
    EXPECT_EQ(local_header.participant_count, config.participant_count);
    EXPECT_EQ(local_header.command_count, 2u);
    EXPECT_EQ(local_plan_count, 2u);
    EXPECT_EQ(local_plan_entries[0].source_participant, 0u);
    EXPECT_EQ(local_plan_entries[0].destination_participant, 1u);
    EXPECT_EQ(local_plan_entries[0].payload_slot, 0u);
    EXPECT_EQ(local_plan_entries[0].destination_slot, 0u);
    EXPECT_EQ(local_plan_entries[1].source_participant, 0u);
    EXPECT_EQ(local_plan_entries[1].destination_participant, 2u);
    EXPECT_EQ(local_plan_entries[1].payload_slot, 1u)
        << "domain projection must remap colliding destination-local slots into source-local compact lanes";
    EXPECT_EQ(local_plan_entries[1].destination_slot, 0u);
    EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.planned_arrivals, 0u)
        << "projection must preserve local current-batch LLEP completion counters";
    EXPECT_EQ(status.candidate_arrivals_considered, 2u);
    EXPECT_EQ(status.payload_bucket_requested_slots, 2u);
    EXPECT_EQ(status.payload_bucket_slots, 2u);
    EXPECT_EQ(status.payload_source_participant_mask, 0b001u);
    EXPECT_EQ(status.payload_destination_participant_mask, 0b110u);

    EXPECT_EQ(hipFree(d_gathered_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_gathered_headers), hipSuccess);
    EXPECT_EQ(hipFree(d_local_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_local_plan_count), hipSuccess);
    EXPECT_EQ(hipFree(d_local_header), hipSuccess);
    EXPECT_EQ(hipFree(d_status), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalanceCopyAndApplyArrivalUsesTransferSlot)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    update.resident_participant_mask[0] = 0b001u;
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream));

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;

    constexpr uint32_t n = 2;
    constexpr uint32_t k = 32;
    constexpr uint32_t blocks_per_row = 1;
    constexpr size_t blocks = n * blocks_per_row;
    constexpr size_t payload_bytes = blocks * 16;
    constexpr size_t scales_bytes = blocks * sizeof(uint16_t);

    struct ProjectionBuffers
    {
        uint8_t *src_payload = nullptr;
        void *src_scales = nullptr;
        uint8_t *dst_payload = nullptr;
        void *dst_scales = nullptr;
        std::vector<uint8_t> expected_payload;
        std::vector<uint16_t> expected_scales;
    };

    std::array<ProjectionBuffers, 3> projections;
    for (size_t i = 0; i < projections.size(); ++i)
    {
        auto &projection = projections[i];
        projection.expected_payload.resize(payload_bytes);
        projection.expected_scales.resize(blocks);
        for (size_t byte = 0; byte < projection.expected_payload.size(); ++byte)
            projection.expected_payload[byte] = static_cast<uint8_t>(0x10u + i * 0x20u + byte);
        for (size_t scale = 0; scale < projection.expected_scales.size(); ++scale)
            projection.expected_scales[scale] = static_cast<uint16_t>(0x100u + i * 0x20u + scale);

        ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&projection.src_payload), payload_bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&projection.src_scales, scales_bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&projection.dst_payload), payload_bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&projection.dst_scales, scales_bytes), hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(projection.src_payload,
                                 projection.expected_payload.data(),
                                 payload_bytes,
                                 hipMemcpyHostToDevice,
                                 stream),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(projection.src_scales,
                                 projection.expected_scales.data(),
                                 scales_bytes,
                                 hipMemcpyHostToDevice,
                                 stream),
                  hipSuccess);
        ASSERT_EQ(hipMemsetAsync(projection.dst_payload, 0, payload_bytes, stream), hipSuccess);
        ASSERT_EQ(hipMemsetAsync(projection.dst_scales, 0, scales_bytes, stream), hipSuccess);
    }

    auto matrix_desc = [](const ProjectionBuffers &projection, bool destination)
    {
        DeviceNativeVNNIMatrixDesc desc;
        desc.payload = destination ? projection.dst_payload : projection.src_payload;
        desc.scales = destination ? projection.dst_scales : projection.src_scales;
        desc.n = n;
        desc.k = k;
        desc.blocks_per_row = blocks_per_row;
        desc.codebook_id = 0;
        return desc;
    };

    auto make_descriptor = [&](bool destination, int local_slot)
    {
        DeviceMoEExpertDescriptor desc;
        desc.gate = matrix_desc(projections[0], destination);
        desc.up = matrix_desc(projections[1], destination);
        desc.down = matrix_desc(projections[2], destination);
        desc.logical_expert_id = 0;
        desc.owner_participant = 0;
        desc.local_slot = local_slot;
        desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                      DeviceMoEExpertFlags::Resident |
                                      DeviceMoEExpertFlags::LocalCompute);
        return desc;
    };

    const uint32_t plan_capacity = 4;
    const uint32_t command_buffer_count = 1;
    auto source_descriptor_index = [&](uint32_t destination_participant,
                                       uint32_t command_buffer_index,
                                       uint32_t plan_index)
    {
        return (static_cast<size_t>(destination_participant) *
                    static_cast<size_t>(command_buffer_count) +
                static_cast<size_t>(command_buffer_index)) *
                   static_cast<size_t>(plan_capacity) +
               static_cast<size_t>(plan_index);
    };
    std::vector<DeviceMoEExpertDirectoryEntry> local_source_descriptors(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count) *
        static_cast<size_t>(plan_capacity));
    auto &source_entry = local_source_descriptors[
        source_descriptor_index(/*destination_participant=*/1,
                                /*command_buffer_index=*/0,
                                /*plan_index=*/0)];
    source_entry.descriptor = make_descriptor(false, 0);
    source_entry.layer = 0;
    source_entry.expert = 0;
    source_entry.participant = 0;
    source_entry.resident_mask = 0b001u;
    source_entry.slot_index = 0;
    source_entry.flags =
        static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
        static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Resident) |
        static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::LocalCompute);
    ASSERT_TRUE(deviceMoEPopulateDirectoryFormat(source_entry));

    std::vector<DeviceMoEExpertDirectoryEntry> transfer_slots(1);
    auto &slot = transfer_slots[0];
    slot.descriptor = make_descriptor(true, 7);
    slot.descriptor.logical_expert_id = -1;
    slot.layer = kDeviceMoEInvalidSlot;
    slot.expert = kDeviceMoEInvalidSlot;
    slot.participant = 1;
    slot.resident_mask = 0;
    slot.slot_index = 7;
    slot.flags =
        static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
        static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::TransferSlot);
    ASSERT_TRUE(deviceMoEPopulateDirectoryFormat(slot));

    DeviceMoERebalancePlanEntry plan;
    plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    plan.layer = 0;
    plan.expert = 0;
    plan.source_participant = 0;
    plan.destination_participant = 1;
    plan.source_resident_mask = 0b001u;
    plan.destination_slot = 0;
    plan.payload_slot = 0;
    std::vector<DeviceMoERebalancePlanEntry> plan_entries(plan_capacity);
    plan_entries[0] = plan;
    const uint32_t legacy_plan_count = 0;
    DeviceMoERebalanceCommandBufferHeader source_command_header;
    source_command_header.epoch = 2;
    source_command_header.phase =
        static_cast<uint32_t>(DeviceMoERebalancePipelinePhase::PlanAssignments);
    source_command_header.command_count = 1;
    source_command_header.command_capacity = plan_capacity;
    source_command_header.participant_id = 0;
    source_command_header.participant_count = 3;
    auto destination_command_header = source_command_header;
    destination_command_header.participant_id = 1;

    DeviceMoERebalanceConfig source_config = config;
    source_config.participant_id = 0;
    constexpr uint64_t payload_slot_bytes = 512;
    const size_t local_payload_slot_count = 1;
    const size_t local_payload_bytes =
        local_payload_slot_count * static_cast<size_t>(payload_slot_bytes);
    const size_t gathered_payload_bytes =
        static_cast<size_t>(config.participant_count) * local_payload_bytes;

    DeviceMoEExpertDirectoryEntry *d_local_source_descriptors = nullptr;
    uint8_t *d_local_payload = nullptr;
    uint8_t *d_gathered_payload = nullptr;
    DeviceMoEExpertDirectoryEntry *d_transfer_slots = nullptr;
    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    DeviceMoERebalanceApplyStatus *d_copy_status = nullptr;
    DeviceMoERebalanceApplyStatus *d_apply_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_source_descriptors),
                        local_source_descriptors.size() * sizeof(local_source_descriptors[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local_payload), local_payload_bytes),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered_payload), gathered_payload_bytes),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_transfer_slots),
                        transfer_slots.size() * sizeof(transfer_slots[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan),
                        plan_entries.size() * sizeof(plan_entries[0])),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_plan_count), sizeof(legacy_plan_count)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_command_header), sizeof(source_command_header)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_copy_status),
                        sizeof(DeviceMoERebalanceApplyStatus)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_apply_status),
                        sizeof(DeviceMoERebalanceApplyStatus)),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_local_source_descriptors,
                             local_source_descriptors.data(),
                             local_source_descriptors.size() * sizeof(local_source_descriptors[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_local_payload, 0, local_payload_bytes, stream), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_gathered_payload, 0, gathered_payload_bytes, stream), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_transfer_slots,
                             transfer_slots.data(),
                             transfer_slots.size() * sizeof(transfer_slots[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_plan,
                             plan_entries.data(),
                             plan_entries.size() * sizeof(plan_entries[0]),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_plan_count, &legacy_plan_count, sizeof(legacy_plan_count), hipMemcpyHostToDevice, stream), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_command_header,
                             &source_command_header,
                             sizeof(source_command_header),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.packDeviceRebalanceCompactPayloads(
        d_plan,
        d_command_header,
        plan_capacity,
        d_local_source_descriptors,
        d_local_payload,
        static_cast<uint32_t>(local_payload_slot_count),
        payload_slot_bytes,
        source_config,
        d_copy_status));
    ASSERT_EQ(hipMemcpyAsync(d_gathered_payload,
                             d_local_payload,
                             local_payload_bytes,
                             hipMemcpyDeviceToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_command_header,
                             &destination_command_header,
                             sizeof(destination_command_header),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_TRUE(gpu_kernel.unpackDeviceRebalanceCollectivePayloads(
        d_plan,
        d_plan_count,
        plan_capacity,
        d_command_header,
        d_gathered_payload,
        static_cast<uint32_t>(local_payload_slot_count),
        payload_slot_bytes,
        d_transfer_slots,
        static_cast<uint32_t>(transfer_slots.size()),
        config,
        d_copy_status));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceApplyStatus copy_status;
    ASSERT_EQ(hipMemcpy(&copy_status, d_copy_status, sizeof(copy_status), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(transfer_slots.data(), d_transfer_slots,
                        transfer_slots.size() * sizeof(transfer_slots[0]),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(copy_status.copied_arrivals, 1u);
    EXPECT_TRUE(deviceMoETransferSlotCopyComplete(transfer_slots[0], 1, 0, 0));
    EXPECT_EQ(transfer_slots[0].layer, 0u);
    EXPECT_EQ(transfer_slots[0].expert, 0u);
    EXPECT_EQ(transfer_slots[0].descriptor.logical_expert_id, 0);

    DeviceMoELayerRuntime seeded_runtime{};
    ASSERT_EQ(hipMemcpy(&seeded_runtime,
                        runtime_table.deviceLayerState(0),
                        sizeof(seeded_runtime),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    seeded_runtime.router_hot_cache_eligible_dispatches = 11;
    seeded_runtime.router_hot_cache_used_dispatches = 7;
    seeded_runtime.router_hot_cache_improved_dispatches = 5;
    seeded_runtime.router_hot_cache_default_load_spread_total = 19;
    seeded_runtime.router_hot_cache_actual_load_spread_total = 13;
    seeded_runtime.router_hot_cache_load_spread_improvement_total = 6;
    seeded_runtime.router_hot_cache_active_dispatches = 23;
    seeded_runtime.router_hot_cache_miss_dispatches = 17;
    seeded_runtime.router_hot_cache_selected_expert_slots = 46;
    seeded_runtime.router_hot_cache_replicated_selected_expert_slots = 12;
    ASSERT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(0),
                             &seeded_runtime,
                             sizeof(seeded_runtime),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    ASSERT_TRUE(gpu_kernel.applyDeviceRebalanceArrivals(
        runtime_table.deviceLayerState(0),
        d_plan,
        d_plan_count,
        plan_capacity,
        d_transfer_slots,
        static_cast<uint32_t>(transfer_slots.size()),
        config,
        d_apply_status,
        d_command_header));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceApplyStatus apply_status;
    DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(hipMemcpy(&apply_status, d_apply_status, sizeof(apply_status), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(apply_status.applied_arrivals, 1u);
    EXPECT_EQ(apply_status.changed_layers, 1u);
    EXPECT_EQ(apply_status.post_apply_multi_resident_experts, 1u);
    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(bank.local_compute_mask[0], 1u);
    EXPECT_EQ(bank.replica_role[0], static_cast<uint8_t>(DeviceMoEReplicaRole::Replica));
    EXPECT_EQ(bank.resident_participant_mask[0], 0b011u);
    EXPECT_EQ(bank.reserved[0], 1u)
        << "transfer-slot arrivals must refresh the multi-resident expert count used by decode routing";
    EXPECT_EQ(bank.experts[0].local_slot, 7);
    EXPECT_TRUE(hasMoEExpertFlag(bank.experts[0].flags, DeviceMoEExpertFlags::Replicated));
    EXPECT_EQ(runtime.router_hot_cache_eligible_dispatches, 11u)
        << "arrival apply starts a new histogram window but must leave router-benefit counters for controller export";
    EXPECT_EQ(runtime.router_hot_cache_used_dispatches, 7u);
    EXPECT_EQ(runtime.router_hot_cache_improved_dispatches, 5u);
    EXPECT_EQ(runtime.router_hot_cache_default_load_spread_total, 19u);
    EXPECT_EQ(runtime.router_hot_cache_actual_load_spread_total, 13u);
    EXPECT_EQ(runtime.router_hot_cache_load_spread_improvement_total, 6u);
    EXPECT_EQ(runtime.router_hot_cache_active_dispatches, 23u);
    EXPECT_EQ(runtime.router_hot_cache_miss_dispatches, 17u);
    EXPECT_EQ(runtime.router_hot_cache_selected_expert_slots, 46u);
    EXPECT_EQ(runtime.router_hot_cache_replicated_selected_expert_slots, 12u);

    auto hidden = TestTensorFactory::createFP32({1, 4});
    auto gate_weights = TestTensorFactory::createFP32({4, 4});
    const std::array<float, 4> hidden_values{1.0f, 0.0f, 0.0f, 0.0f};
    const std::array<float, 16> gate_values{
        4.0f, 0.0f, 0.0f, 0.0f,
        3.0f, 0.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f};
    std::copy(hidden_values.begin(), hidden_values.end(), hidden->mutable_data());
    std::copy(gate_values.begin(), gate_values.end(), gate_weights->mutable_data());
    auto output_indices = TestTensorFactory::createFP32({2, 1});
    auto output_weights = TestTensorFactory::createFP32({2, 1});
    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices->ensureOnDevice(device));
    ASSERT_TRUE(output_weights->ensureOnDevice(device));

    auto route_workspace = bindDefaultMoEWorkspace(
        gpu_kernel,
        /*max_seq_len=*/1,
        /*d_model=*/4,
        /*intermediate=*/8,
        /*num_experts=*/4,
        /*top_k=*/2);
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ScopedROCmEnvOverride wave_env("LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "1");
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            runtime_table.deviceLayerState(0),
            hidden.get(), gate_weights.get(),
            /*d_model=*/4,
            /*num_experts=*/4,
            /*top_k=*/2,
            /*normalize_weights=*/false,
            output_indices.get(), output_weights.get(),
            /*write_legacy_outputs=*/true,
            /*update_runtime_histogram=*/true));
    }
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(runtime.topk_expert_ids[0], 0);
    EXPECT_EQ(runtime.topk_expert_ids[1], -1)
        << "the transferred replica should take expert 0 locally while expert 1 stays remote";
    EXPECT_EQ(runtime.router_hot_cache_eligible_dispatches, 12u);
    EXPECT_EQ(runtime.router_hot_cache_used_dispatches, 8u);
    EXPECT_EQ(runtime.router_hot_cache_improved_dispatches, 6u);
    EXPECT_EQ(runtime.router_hot_cache_active_dispatches, 24u);
    EXPECT_EQ(runtime.router_hot_cache_miss_dispatches, 17u);
    EXPECT_EQ(runtime.router_hot_cache_selected_expert_slots, 48u);
    EXPECT_EQ(runtime.router_hot_cache_replicated_selected_expert_slots, 13u);

    DeviceMoERebalancePlanEntry resident_plan;
    resident_plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ResidentExpertAssignment);
    resident_plan.layer = 0;
    resident_plan.expert = 2;
    resident_plan.source_participant = 1;
    resident_plan.destination_participant = 1;
    resident_plan.source_resident_mask = 0b010u;
    resident_plan.destination_slot = 0;
    resident_plan.payload_slot = 0;
    destination_command_header.epoch = 3;
    destination_command_header.command_count = 1;
    ASSERT_EQ(hipMemcpyAsync(d_plan,
                             &resident_plan,
                             sizeof(resident_plan),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_command_header,
                             &destination_command_header,
                             sizeof(destination_command_header),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_TRUE(gpu_kernel.applyDeviceRebalanceArrivals(
        runtime_table.deviceLayerState(0),
        d_plan,
        d_plan_count,
        plan_capacity,
        d_transfer_slots,
        static_cast<uint32_t>(transfer_slots.size()),
        config,
        d_apply_status,
        d_command_header));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipMemcpy(&apply_status, d_apply_status, sizeof(apply_status), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime), hipMemcpyDeviceToHost), hipSuccess);
    const auto &rebuilt_bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(apply_status.applied_arrivals, 1u);
    EXPECT_EQ(apply_status.changed_layers, 1u);
    EXPECT_EQ(apply_status.post_apply_multi_resident_experts, 1u);
    EXPECT_EQ(rebuilt_bank.local_compute_mask[0], 1u)
        << "rebalance apply must preserve already-arrived hot replicas when another plan rebuilds the layer bank";
    EXPECT_EQ(rebuilt_bank.replica_role[0], static_cast<uint8_t>(DeviceMoEReplicaRole::Replica));
    EXPECT_EQ(rebuilt_bank.resident_participant_mask[0], 0b011u);
    EXPECT_TRUE(hasMoEExpertFlag(rebuilt_bank.experts[0].flags,
                                 DeviceMoEExpertFlags::Replicated));
    EXPECT_EQ(rebuilt_bank.reserved[0], 1u);

    destination_command_header.epoch = 4;
    destination_command_header.command_count = 1;
    ASSERT_EQ(hipMemcpyAsync(d_plan,
                             &plan,
                             sizeof(plan),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_command_header,
                             &destination_command_header,
                             sizeof(destination_command_header),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    for (const auto &projection : projections)
    {
        std::vector<uint8_t> actual_payload(payload_bytes);
        std::vector<uint16_t> actual_scales(blocks);
        ASSERT_EQ(hipMemcpy(actual_payload.data(),
                            projection.dst_payload,
                            payload_bytes,
                            hipMemcpyDeviceToHost),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy(actual_scales.data(),
                            projection.dst_scales,
                            scales_bytes,
                            hipMemcpyDeviceToHost),
                  hipSuccess);
        EXPECT_EQ(actual_payload, projection.expected_payload);
        EXPECT_EQ(actual_scales, projection.expected_scales);
    }

    ASSERT_EQ(hipMemsetAsync(d_gathered_payload,
                             0,
                             gathered_payload_bytes,
                             stream),
              hipSuccess);
    const auto clean_copy_status = makeCleanRebalanceApplyStatus();
    ASSERT_EQ(hipMemcpyAsync(d_copy_status,
                             &clean_copy_status,
                             sizeof(clean_copy_status),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_TRUE(gpu_kernel.unpackDeviceRebalanceCollectivePayloads(
        d_plan,
        d_plan_count,
        plan_capacity,
        d_command_header,
        d_gathered_payload,
        static_cast<uint32_t>(local_payload_slot_count),
        payload_slot_bytes,
        d_transfer_slots,
        static_cast<uint32_t>(transfer_slots.size()),
        config,
        d_copy_status));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipMemcpy(&copy_status, d_copy_status, sizeof(copy_status), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(transfer_slots.data(), d_transfer_slots,
                        transfer_slots.size() * sizeof(transfer_slots[0]),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(copy_status.copied_arrivals, 0u);
    EXPECT_EQ(copy_status.missing_source_descriptors, 1u);
    EXPECT_FALSE(deviceMoETransferSlotCopyComplete(transfer_slots[0], 1, 0, 0))
        << "a failed replay must clear stale CopyComplete metadata before apply";

    ASSERT_TRUE(gpu_kernel.applyDeviceRebalanceArrivals(
        runtime_table.deviceLayerState(0),
        d_plan,
        d_plan_count,
        plan_capacity,
        d_transfer_slots,
        static_cast<uint32_t>(transfer_slots.size()),
        config,
        d_apply_status,
        d_command_header));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipMemcpy(&apply_status, d_apply_status, sizeof(apply_status), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(apply_status.applied_arrivals, 0u);
    EXPECT_EQ(apply_status.copy_incomplete, 1u);
    EXPECT_EQ(apply_status.post_apply_multi_resident_experts, 0u);

    DeviceMoERuntimeTable source_runtime_table(table_config);
    auto source_update = makeParticipantOneBaseUpdate(10);
    source_update.participant_id = 0;
    source_update.local_compute_mask = {1, 0, 0, 0};
    source_update.replica_role = {
        static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
        static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        static_cast<uint8_t>(DeviceMoEReplicaRole::None),
    };
    source_update.resident_participant_mask = {0b001u, 0b001u, 0b010u, 0b100u};
    source_update.experts[0].owner_participant = 0;
    source_update.experts[0].flags =
        toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                         DeviceMoEExpertFlags::Resident |
                         DeviceMoEExpertFlags::LocalCompute);
    ASSERT_TRUE(source_runtime_table.prepareInactiveBank(0, source_update));
    ASSERT_TRUE(source_runtime_table.flipActiveBank(0, source_update.epoch, stream));

    source_command_header.epoch = 5;
    source_command_header.command_count = 1;
    source_command_header.participant_id = 0;
    ASSERT_EQ(hipMemcpyAsync(d_command_header,
                             &source_command_header,
                             sizeof(source_command_header),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_TRUE(gpu_kernel.applyDeviceRebalanceArrivals(
        source_runtime_table.deviceLayerState(0),
        d_plan,
        d_plan_count,
        plan_capacity,
        d_transfer_slots,
        static_cast<uint32_t>(transfer_slots.size()),
        source_config,
        d_apply_status,
        d_command_header));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    DeviceMoELayerRuntime source_runtime_state{};
    ASSERT_EQ(hipMemcpy(&apply_status, d_apply_status, sizeof(apply_status), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(&source_runtime_state,
                        source_runtime_table.deviceLayerState(0),
                        sizeof(source_runtime_state),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(apply_status.applied_arrivals, 0u);
    EXPECT_EQ(apply_status.changed_layers, 1u)
        << "source participants must learn destination residency so replica dispatch stays domain-consistent";
    EXPECT_EQ(apply_status.post_apply_multi_resident_experts, 1u);
    const auto &source_bank = source_runtime_state.banks[source_runtime_state.active_bank];
    EXPECT_EQ(source_bank.resident_participant_mask[0], 0b011u)
        << "payload arrivals are data-local on the destination but residency metadata is domain-wide";
    EXPECT_EQ(source_bank.local_compute_mask[0], 1u);
    EXPECT_TRUE(hasMoEExpertFlag(source_bank.experts[0].flags,
                                 DeviceMoEExpertFlags::Replicated));
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ScopedROCmEnvOverride wave_env("LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "1");
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            source_runtime_table.deviceLayerState(0),
            hidden.get(), gate_weights.get(),
            /*d_model=*/4,
            /*num_experts=*/4,
            /*top_k=*/2,
            /*normalize_weights=*/false,
            output_indices.get(), output_weights.get(),
            /*write_legacy_outputs=*/true,
            /*update_runtime_histogram=*/true));
    }
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipMemcpy(&source_runtime_state,
                        source_runtime_table.deviceLayerState(0),
                        sizeof(source_runtime_state),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(source_runtime_state.topk_expert_ids[0], -1)
        << "source and destination must make the same resident-copy assignment for replicated expert 0";
    EXPECT_EQ(source_runtime_state.topk_expert_ids[1], 1)
        << "source should still compute its non-replicated local expert";

    EXPECT_EQ(hipFree(d_local_source_descriptors), hipSuccess);
    EXPECT_EQ(hipFree(d_local_payload), hipSuccess);
    EXPECT_EQ(hipFree(d_gathered_payload), hipSuccess);
    EXPECT_EQ(hipFree(d_transfer_slots), hipSuccess);
    EXPECT_EQ(hipFree(d_plan), hipSuccess);
    EXPECT_EQ(hipFree(d_plan_count), hipSuccess);
    EXPECT_EQ(hipFree(d_command_header), hipSuccess);
    EXPECT_EQ(hipFree(d_copy_status), hipSuccess);
    EXPECT_EQ(hipFree(d_apply_status), hipSuccess);
    for (auto &projection : projections)
    {
        EXPECT_EQ(hipFree(projection.src_payload), hipSuccess);
        EXPECT_EQ(hipFree(projection.src_scales), hipSuccess);
        EXPECT_EQ(hipFree(projection.dst_payload), hipSuccess);
        EXPECT_EQ(hipFree(projection.dst_scales), hipSuccess);
    }
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DeviceRebalancePackHistogramsFeedsController)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream));

    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;

    constexpr size_t local_entries = 4;
    constexpr size_t gathered_entries = local_entries * 3;
    const uint64_t seeded_global_counts[local_entries] = {99, 99, 99, 99};
    const uint64_t seeded_local_counts[local_entries] = {8, 0, 0, 0};

    uint64_t *d_local = nullptr;
    uint64_t *d_gathered = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_local),
                        local_entries * sizeof(uint64_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_gathered),
                        gathered_entries * sizeof(uint64_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&d_status),
                        sizeof(DeviceMoERebalanceStatus)),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(0)->decode_histogram,
                             seeded_global_counts,
                             sizeof(seeded_global_counts),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(0)->decode_local_histogram,
                             seeded_local_counts,
                             sizeof(seeded_local_counts),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<IMoEKernel &>(gpu_kernel).setGPUStream(stream);
    ASSERT_TRUE(gpu_kernel.packDeviceRebalanceHistograms(
        runtime_table.deviceLayerState(0), d_local, config));

    uint64_t packed_counts[local_entries] = {};
    ASSERT_EQ(hipMemcpyAsync(packed_counts, d_local, sizeof(packed_counts),
                             hipMemcpyDeviceToHost, stream),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    EXPECT_EQ(packed_counts[0], 8u);
    EXPECT_EQ(packed_counts[1], 0u);
    EXPECT_EQ(packed_counts[2], 0u);
    EXPECT_EQ(packed_counts[3], 0u);

    ASSERT_EQ(hipMemsetAsync(d_gathered, 0,
                             gathered_entries * sizeof(uint64_t),
                             stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_gathered + config.participant_id * local_entries,
                             d_local,
                             local_entries * sizeof(uint64_t),
                             hipMemcpyDeviceToDevice,
                             stream),
              hipSuccess);

    ASSERT_TRUE(gpu_kernel.runDeviceRebalanceController(
        runtime_table.deviceLayerState(0), d_gathered, d_status, config));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    DeviceMoERebalanceStatus status;
    ASSERT_EQ(hipMemcpy(&status, d_status, sizeof(status), hipMemcpyDeviceToHost), hipSuccess);
    DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(hipMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime),
                        hipMemcpyDeviceToHost),
              hipSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.windows_applied, 1u);
    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(bank.local_compute_mask[0], 1u);
    EXPECT_TRUE(hasMoEExpertFlag(bank.experts[0].flags, DeviceMoEExpertFlags::Replicated));
    for (int expert = 0; expert < 4; ++expert)
    {
        EXPECT_EQ(runtime.decode_histogram[expert], 0u);
        EXPECT_EQ(runtime.decode_local_histogram[expert], 0u);
    }

    EXPECT_EQ(hipFree(d_local), hipSuccess);
    EXPECT_EQ(hipFree(d_gathered), hipSuccess);
    EXPECT_EQ(hipFree(d_status), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectWaveTopKMatchesDefaultRuntimeTopK)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int d_model = 512;
    const int num_experts = 64;
    const int top_k = 8;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices_default = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_default = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_indices_wave = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_wave = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::vector<float> hidden_host(static_cast<size_t>(d_model));
    std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
    fillRandom(hidden_host, -1.0f, 1.0f, 9401);
    fillRandom(gate_host, -0.2f, 0.2f, 9402);
    std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
    std::copy(gate_host.begin(), gate_host.end(), gate_weights->mutable_data());

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_default->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_default->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_wave->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_wave->ensureOnDevice(device));

    DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, top_k);

    DeviceMoELayerRuntime *runtime_default = nullptr;
    DeviceMoELayerRuntime *runtime_wave = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_default), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_wave), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_default, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_wave, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ScopedROCmEnvOverride wave_env("LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "0");
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            runtime_default,
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            true,
            output_indices_default.get(), output_weights_default.get(),
            true, true));
    }
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ScopedROCmEnvOverride wave_env("LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "1");
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            runtime_wave,
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            true,
            output_indices_wave.get(), output_weights_wave.get(),
            true, true));
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceMoELayerRuntime after_default{};
    DeviceMoELayerRuntime after_wave{};
    ASSERT_EQ(hipMemcpy(&after_default, runtime_default, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&after_wave, runtime_wave, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipFree(runtime_default), hipSuccess);
    ASSERT_EQ(hipFree(runtime_wave), hipSuccess);

    output_indices_default->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_default->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_indices_wave->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_wave->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *legacy_indices_default = output_indices_default->data();
    const float *legacy_weights_default = output_weights_default->data();
    const float *legacy_indices_wave = output_indices_wave->data();
    const float *legacy_weights_wave = output_weights_wave->data();

    double max_weight_diff = 0.0;
    for (int slot = 0; slot < top_k; ++slot)
    {
        EXPECT_EQ(after_wave.topk_expert_ids[slot], after_default.topk_expert_ids[slot]);
        EXPECT_FLOAT_EQ(legacy_indices_wave[slot], legacy_indices_default[slot]);
        EXPECT_FLOAT_EQ(legacy_indices_wave[slot], static_cast<float>(after_wave.topk_expert_ids[slot]));
        EXPECT_NEAR(after_wave.topk_weights[slot], after_default.topk_weights[slot], 2e-6f);
        EXPECT_NEAR(legacy_weights_wave[slot], legacy_weights_default[slot], 2e-6f);
        max_weight_diff = std::max(
            max_weight_diff,
            std::fabs(static_cast<double>(after_wave.topk_weights[slot]) -
                      static_cast<double>(after_default.topk_weights[slot])));
    }

    uint64_t histogram_sum_default = 0;
    uint64_t histogram_sum_wave = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        histogram_sum_default += after_default.decode_histogram[expert];
        histogram_sum_wave += after_wave.decode_histogram[expert];
    }
    EXPECT_EQ(histogram_sum_default, static_cast<uint64_t>(top_k));
    EXPECT_EQ(histogram_sum_wave, static_cast<uint64_t>(top_k));

    std::cout << "[DecodeRouteSelectWaveTopKMatchesDefaultRuntimeTopK] max_weight_diff="
              << max_weight_diff << "\n";
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectWaveTopKMatchesDefaultRuntimeTopKQwen35Shape)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int d_model = 512;
    const int num_experts = 256;
    const int top_k = 8;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices_default = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_default = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_indices_wave = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_wave = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::vector<float> hidden_host(static_cast<size_t>(d_model));
    std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
    fillRandom(hidden_host, -1.0f, 1.0f, 9501);
    fillRandom(gate_host, -0.2f, 0.2f, 9502);
    std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
    std::copy(gate_host.begin(), gate_host.end(), gate_weights->mutable_data());

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_default->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_default->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_wave->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_wave->ensureOnDevice(device));

    DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, top_k);

    DeviceMoELayerRuntime *runtime_default = nullptr;
    DeviceMoELayerRuntime *runtime_wave = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_default), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_wave), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_default, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_wave, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ScopedROCmEnvOverride wave_env("LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "0");
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            runtime_default,
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            true,
            output_indices_default.get(), output_weights_default.get(),
            true, true));
    }
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ScopedROCmEnvOverride wave_env("LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "1");
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            runtime_wave,
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            true,
            output_indices_wave.get(), output_weights_wave.get(),
            true, true));
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceMoELayerRuntime after_default{};
    DeviceMoELayerRuntime after_wave{};
    ASSERT_EQ(hipMemcpy(&after_default, runtime_default, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&after_wave, runtime_wave, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipFree(runtime_default), hipSuccess);
    ASSERT_EQ(hipFree(runtime_wave), hipSuccess);

    output_indices_default->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_default->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_indices_wave->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_wave->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *legacy_indices_default = output_indices_default->data();
    const float *legacy_weights_default = output_weights_default->data();
    const float *legacy_indices_wave = output_indices_wave->data();
    const float *legacy_weights_wave = output_weights_wave->data();

    double max_weight_diff = 0.0;
    for (int slot = 0; slot < top_k; ++slot)
    {
        EXPECT_EQ(after_wave.topk_expert_ids[slot], after_default.topk_expert_ids[slot]);
        EXPECT_FLOAT_EQ(legacy_indices_wave[slot], legacy_indices_default[slot]);
        EXPECT_FLOAT_EQ(legacy_indices_wave[slot], static_cast<float>(after_wave.topk_expert_ids[slot]));
        EXPECT_NEAR(after_wave.topk_weights[slot], after_default.topk_weights[slot], 2e-6f);
        EXPECT_NEAR(legacy_weights_wave[slot], legacy_weights_default[slot], 2e-6f);
        max_weight_diff = std::max(
            max_weight_diff,
            std::fabs(static_cast<double>(after_wave.topk_weights[slot]) -
                      static_cast<double>(after_default.topk_weights[slot])));
    }

    uint64_t histogram_sum_default = 0;
    uint64_t histogram_sum_wave = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        histogram_sum_default += after_default.decode_histogram[expert];
        histogram_sum_wave += after_wave.decode_histogram[expert];
    }
    EXPECT_EQ(histogram_sum_default, static_cast<uint64_t>(top_k));
    EXPECT_EQ(histogram_sum_wave, static_cast<uint64_t>(top_k));

    std::cout << "[DecodeRouteSelectWaveTopKMatchesDefaultRuntimeTopKQwen35Shape] max_weight_diff="
              << max_weight_diff << "\n";
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectFP16RouterMatchesFP32TopK)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int d_model = 1024;
    const int num_experts = 32;
    const int top_k = 6;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_indices_fp16 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_fp16 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::vector<float> hidden_host(static_cast<size_t>(d_model));
    fillRandom(hidden_host, -1.0f, 1.0f, 9301);
    const float hidden_norm_sq = std::inner_product(
        hidden_host.begin(), hidden_host.end(), hidden_host.begin(), 0.0f);

    std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
    std::mt19937 gen(9302);
    std::uniform_real_distribution<float> noise(-1.0e-5f, 1.0e-5f);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const float expert_score = 2.0f - 0.05f * static_cast<float>(expert);
        for (int i = 0; i < d_model; ++i)
        {
            gate_host[static_cast<size_t>(expert) * d_model + i] =
                expert_score * hidden_host[i] / hidden_norm_sq + noise(gen);
        }
    }

    std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
    std::copy(gate_host.begin(), gate_host.end(), gate_weights->mutable_data());

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_fp32->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_fp32->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_fp16->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_fp16->ensureOnDevice(device));

    DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, top_k);

    DeviceMoELayerRuntime *runtime_fp32 = nullptr;
    DeviceMoELayerRuntime *runtime_fp16 = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_fp32), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_fp16), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_fp32, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_fp16, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        {
            ScopedROCmEnvOverride router_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
            ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
                runtime_fp32,
                hidden.get(), gate_weights.get(),
                d_model, num_experts, top_k,
                true,
                output_indices_fp32.get(), output_weights_fp32.get(),
                true, false));
        }
        {
            ScopedROCmEnvOverride router_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "1");
            ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
                runtime_fp16,
                hidden.get(), gate_weights.get(),
                d_model, num_experts, top_k,
                true,
                output_indices_fp16.get(), output_weights_fp16.get(),
                true, false));
        }
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceMoELayerRuntime after_fp32{};
    DeviceMoELayerRuntime after_fp16{};
    ASSERT_EQ(hipMemcpy(&after_fp32, runtime_fp32, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&after_fp16, runtime_fp16, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipFree(runtime_fp32), hipSuccess);
    ASSERT_EQ(hipFree(runtime_fp16), hipSuccess);

    output_indices_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_indices_fp16->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_fp16->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *legacy_indices_fp32 = output_indices_fp32->data();
    const float *legacy_weights_fp32 = output_weights_fp32->data();
    const float *legacy_indices_fp16 = output_indices_fp16->data();
    const float *legacy_weights_fp16 = output_weights_fp16->data();

    float max_weight_diff = 0.0f;
    for (int k = 0; k < top_k; ++k)
    {
        EXPECT_EQ(after_fp16.topk_expert_ids[k], after_fp32.topk_expert_ids[k]);
        EXPECT_FLOAT_EQ(legacy_indices_fp32[k], static_cast<float>(after_fp32.topk_expert_ids[k]));
        EXPECT_FLOAT_EQ(legacy_indices_fp16[k], static_cast<float>(after_fp16.topk_expert_ids[k]));
        const float diff = std::fabs(after_fp16.topk_weights[k] - after_fp32.topk_weights[k]);
        max_weight_diff = std::max(max_weight_diff, diff);
        EXPECT_NEAR(after_fp16.topk_weights[k], after_fp32.topk_weights[k], 2.0e-3f);
        EXPECT_NEAR(legacy_weights_fp16[k], after_fp16.topk_weights[k], 1.0e-6f);
        EXPECT_NEAR(legacy_weights_fp32[k], after_fp32.topk_weights[k], 1.0e-6f);
    }

    std::cout << "[DecodeRouteSelectFP16RouterMatchesFP32TopK] max_weight_diff="
              << std::fixed << std::setprecision(8) << max_weight_diff << std::endl;
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectBF16RouterMatchesFP32TopK)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int d_model = 1024;
    const int num_experts = 32;
    const int top_k = 6;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto gate_weights_bf16 = TestTensorFactory::createBF16({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_indices_bf16 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_bf16 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::vector<float> hidden_host(static_cast<size_t>(d_model));
    fillRandom(hidden_host, -1.0f, 1.0f, 9361);
    const float hidden_norm_sq = std::inner_product(
        hidden_host.begin(), hidden_host.end(), hidden_host.begin(), 0.0f);

    std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
    std::mt19937 gen(9362);
    std::uniform_real_distribution<float> noise(-1.0e-5f, 1.0e-5f);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const float expert_score = 2.0f - 0.05f * static_cast<float>(expert);
        for (int i = 0; i < d_model; ++i)
        {
            gate_host[static_cast<size_t>(expert) * d_model + i] =
                expert_score * hidden_host[i] / hidden_norm_sq + noise(gen);
        }
    }

    std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
    std::copy(gate_host.begin(), gate_host.end(), gate_weights_fp32->mutable_data());
    gate_weights_bf16->from_fp32(gate_host.data(), gate_host.size());

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights_fp32->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights_bf16->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_fp32->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_fp32->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_bf16->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_bf16->ensureOnDevice(device));

    DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, top_k);

    DeviceMoELayerRuntime *runtime_fp32 = nullptr;
    DeviceMoELayerRuntime *runtime_bf16 = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_fp32), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_bf16), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_fp32, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_bf16, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride grouped_env("LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ScopedROCmEnvOverride wave_env("LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "0");

        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            runtime_fp32,
            hidden.get(), gate_weights_fp32.get(),
            d_model, num_experts, top_k,
            true,
            output_indices_fp32.get(), output_weights_fp32.get(),
            true, false));
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            runtime_bf16,
            hidden.get(), gate_weights_bf16.get(),
            d_model, num_experts, top_k,
            true,
            output_indices_bf16.get(), output_weights_bf16.get(),
            true, false));
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceMoELayerRuntime after_fp32{};
    DeviceMoELayerRuntime after_bf16{};
    ASSERT_EQ(hipMemcpy(&after_fp32, runtime_fp32, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&after_bf16, runtime_bf16, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipFree(runtime_fp32), hipSuccess);
    ASSERT_EQ(hipFree(runtime_bf16), hipSuccess);

    output_indices_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_indices_bf16->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_bf16->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *legacy_indices_fp32 = output_indices_fp32->data();
    const float *legacy_weights_fp32 = output_weights_fp32->data();
    const float *legacy_indices_bf16 = output_indices_bf16->data();
    const float *legacy_weights_bf16 = output_weights_bf16->data();

    float max_weight_diff = 0.0f;
    for (int k = 0; k < top_k; ++k)
    {
        EXPECT_EQ(after_bf16.topk_expert_ids[k], after_fp32.topk_expert_ids[k]);
        EXPECT_FLOAT_EQ(legacy_indices_fp32[k], static_cast<float>(after_fp32.topk_expert_ids[k]));
        EXPECT_FLOAT_EQ(legacy_indices_bf16[k], static_cast<float>(after_bf16.topk_expert_ids[k]));
        const float diff = std::fabs(after_bf16.topk_weights[k] - after_fp32.topk_weights[k]);
        max_weight_diff = std::max(max_weight_diff, diff);
        EXPECT_NEAR(after_bf16.topk_weights[k], after_fp32.topk_weights[k], 5.0e-3f);
        EXPECT_NEAR(legacy_weights_bf16[k], after_bf16.topk_weights[k], 1.0e-6f);
        EXPECT_NEAR(legacy_weights_fp32[k], after_fp32.topk_weights[k], 1.0e-6f);
    }

    std::cout << "[DecodeRouteSelectBF16RouterMatchesFP32TopK] max_weight_diff="
              << std::fixed << std::setprecision(8) << max_weight_diff << std::endl;
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectKPartRouterMatchesFP32TopK)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int d_model = 2048;
    const int num_experts = 64;
    const int top_k = 8;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::vector<float> hidden_host(static_cast<size_t>(d_model));
    fillRandom(hidden_host, -1.0f, 1.0f, 9401);
    const float hidden_norm_sq = std::inner_product(
        hidden_host.begin(), hidden_host.end(), hidden_host.begin(), 0.0f);

    std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
    std::mt19937 gen(9402);
    std::uniform_real_distribution<float> noise(-1.0e-6f, 1.0e-6f);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const float expert_score = 3.0f - 0.05f * static_cast<float>(expert);
        for (int i = 0; i < d_model; ++i)
        {
            gate_host[static_cast<size_t>(expert) * d_model + i] =
                expert_score * hidden_host[i] / hidden_norm_sq + noise(gen);
        }
    }

    std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
    std::copy(gate_host.begin(), gate_host.end(), gate_weights->mutable_data());

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_fp32->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_fp32->ensureOnDevice(device));

    DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, top_k);

    DeviceMoELayerRuntime *runtime_fp32 = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_fp32), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_fp32, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    auto expectHistogramMatchesTopK = [&](const DeviceMoELayerRuntime &runtime)
    {
        std::vector<uint64_t> expected(static_cast<size_t>(num_experts), 0);
        for (int k = 0; k < top_k; ++k)
        {
            const int expert_id = runtime.topk_expert_ids[k];
            EXPECT_GE(expert_id, 0);
            EXPECT_LT(expert_id, num_experts);
            if (expert_id >= 0 && expert_id < num_experts)
            {
                ++expected[static_cast<size_t>(expert_id)];
            }
        }
        for (int expert = 0; expert < num_experts; ++expert)
        {
            EXPECT_EQ(runtime.decode_histogram[expert], expected[static_cast<size_t>(expert)])
                << "expert=" << expert;
        }
    };

    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride grouped_env("LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            runtime_fp32,
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            true,
            output_indices_fp32.get(), output_weights_fp32.get(),
            true, true));
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceMoELayerRuntime after_fp32{};
    ASSERT_EQ(hipMemcpy(&after_fp32, runtime_fp32, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipFree(runtime_fp32), hipSuccess);

    output_indices_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *legacy_indices_fp32 = output_indices_fp32->data();
    const float *legacy_weights_fp32 = output_weights_fp32->data();
    for (int k = 0; k < top_k; ++k)
    {
        EXPECT_FLOAT_EQ(legacy_indices_fp32[k], static_cast<float>(after_fp32.topk_expert_ids[k]));
        EXPECT_NEAR(legacy_weights_fp32[k], after_fp32.topk_weights[k], 1.0e-6f);
    }
    expectHistogramMatchesTopK(after_fp32);

    for (int k_partitions : {2, 4, 8, 16})
    {
        auto output_indices_kpart = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
        auto output_weights_kpart = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
        ASSERT_TRUE(output_indices_kpart->ensureOnDevice(device));
        ASSERT_TRUE(output_weights_kpart->ensureOnDevice(device));

        DeviceMoELayerRuntime *runtime_kpart = nullptr;
        ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_kpart), sizeof(DeviceMoELayerRuntime)), hipSuccess);
        ASSERT_EQ(hipMemcpy(runtime_kpart, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

        const std::string kparts_value = std::to_string(k_partitions);
        {
            ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
            ScopedROCmEnvOverride grouped_env("LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER", "0");
            ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
            ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
            ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "1");
            ScopedROCmEnvOverride kparts_env("LLAMINAR_ROCM_MOE_ROUTER_KPARTS", kparts_value.c_str());
            ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
                runtime_kpart,
                hidden.get(), gate_weights.get(),
                d_model, num_experts, top_k,
                true,
                output_indices_kpart.get(), output_weights_kpart.get(),
                true, true));
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        DeviceMoELayerRuntime after_kpart{};
        ASSERT_EQ(hipMemcpy(&after_kpart, runtime_kpart, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipFree(runtime_kpart), hipSuccess);

        output_indices_kpart->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        output_weights_kpart->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        const float *legacy_indices_kpart = output_indices_kpart->data();
        const float *legacy_weights_kpart = output_weights_kpart->data();

        float max_weight_diff = 0.0f;
        for (int k = 0; k < top_k; ++k)
        {
            EXPECT_EQ(after_kpart.topk_expert_ids[k], after_fp32.topk_expert_ids[k])
                << "k_partitions=" << k_partitions << " slot=" << k;
            EXPECT_FLOAT_EQ(legacy_indices_kpart[k], static_cast<float>(after_kpart.topk_expert_ids[k]));
            const float diff = std::fabs(after_kpart.topk_weights[k] - after_fp32.topk_weights[k]);
            max_weight_diff = std::max(max_weight_diff, diff);
            EXPECT_NEAR(after_kpart.topk_weights[k], after_fp32.topk_weights[k], 1.0e-4f)
                << "k_partitions=" << k_partitions << " slot=" << k;
            EXPECT_NEAR(legacy_weights_kpart[k], after_kpart.topk_weights[k], 1.0e-6f);
        }
        expectHistogramMatchesTopK(after_kpart);

        std::cout << "[DecodeRouteSelectKPartRouterMatchesFP32TopK] kparts="
                  << k_partitions << " max_weight_diff="
                  << std::fixed << std::setprecision(8) << max_weight_diff << std::endl;
    }
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectQ8RouterMatchesFP32TopKAcrossSeeds)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int d_model = 2048;
    const int num_experts = 64;
    const int top_k = 8;
    const int case_count = 8;

    int drift_cases = 0;
    int drift_slots = 0;
    float max_weight_diff = 0.0f;

    for (int case_idx = 0; case_idx < case_count; ++case_idx)
    {
        auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
        auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
        auto output_indices_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
        auto output_weights_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
        auto output_indices_q8 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
        auto output_weights_q8 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

        std::vector<float> hidden_host(static_cast<size_t>(d_model));
        fillRandom(hidden_host, -1.0f, 1.0f, static_cast<unsigned>(9501 + case_idx));
        const float hidden_norm_sq = std::inner_product(
            hidden_host.begin(), hidden_host.end(), hidden_host.begin(), 0.0f);

        std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
        std::mt19937 gen(static_cast<unsigned>(9601 + case_idx));
        std::uniform_real_distribution<float> noise(-2.0e-4f, 2.0e-4f);
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const float expert_score = 5.0f - 0.16f * static_cast<float>(expert);
            for (int i = 0; i < d_model; ++i)
            {
                gate_host[static_cast<size_t>(expert) * d_model + i] =
                    expert_score * hidden_host[i] / hidden_norm_sq + noise(gen);
            }
        }

        std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
        std::copy(gate_host.begin(), gate_host.end(), gate_weights->mutable_data());

        ASSERT_TRUE(hidden->ensureOnDevice(device));
        ASSERT_TRUE(gate_weights->ensureOnDevice(device));
        ASSERT_TRUE(output_indices_fp32->ensureOnDevice(device));
        ASSERT_TRUE(output_weights_fp32->ensureOnDevice(device));
        ASSERT_TRUE(output_indices_q8->ensureOnDevice(device));
        ASSERT_TRUE(output_weights_q8->ensureOnDevice(device));

        DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, top_k);

        DeviceMoELayerRuntime *runtime_fp32 = nullptr;
        DeviceMoELayerRuntime *runtime_q8 = nullptr;
        ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_fp32), sizeof(DeviceMoELayerRuntime)), hipSuccess);
        ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_q8), sizeof(DeviceMoELayerRuntime)), hipSuccess);
        ASSERT_EQ(hipMemcpy(runtime_fp32, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(runtime_q8, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

        ROCmMoEKernel gpu_kernel(0);
        auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
        {
            ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
            ScopedROCmEnvOverride grouped_env("LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER", "0");
            ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
            ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
            ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
            ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
                runtime_fp32,
                hidden.get(), gate_weights.get(),
                d_model, num_experts, top_k,
                true,
                output_indices_fp32.get(), output_weights_fp32.get(),
                true, false));
        }
        {
            ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
            ScopedROCmEnvOverride grouped_env("LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER", "0");
            ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "1");
            ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
            ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
            ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
                runtime_q8,
                hidden.get(), gate_weights.get(),
                d_model, num_experts, top_k,
                true,
                output_indices_q8.get(), output_weights_q8.get(),
                true, false));
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        DeviceMoELayerRuntime after_fp32{};
        DeviceMoELayerRuntime after_q8{};
        ASSERT_EQ(hipMemcpy(&after_fp32, runtime_fp32, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipMemcpy(&after_q8, runtime_q8, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipFree(runtime_fp32), hipSuccess);
        ASSERT_EQ(hipFree(runtime_q8), hipSuccess);

        output_indices_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        output_weights_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        output_indices_q8->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        output_weights_q8->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        const float *legacy_indices_fp32 = output_indices_fp32->data();
        const float *legacy_weights_fp32 = output_weights_fp32->data();
        const float *legacy_indices_q8 = output_indices_q8->data();
        const float *legacy_weights_q8 = output_weights_q8->data();

        bool case_drifted = false;
        for (int k = 0; k < top_k; ++k)
        {
            EXPECT_FLOAT_EQ(legacy_indices_fp32[k], static_cast<float>(after_fp32.topk_expert_ids[k]));
            EXPECT_FLOAT_EQ(legacy_indices_q8[k], static_cast<float>(after_q8.topk_expert_ids[k]));
            EXPECT_NEAR(legacy_weights_fp32[k], after_fp32.topk_weights[k], 1.0e-6f);
            EXPECT_NEAR(legacy_weights_q8[k], after_q8.topk_weights[k], 1.0e-6f);

            if (after_q8.topk_expert_ids[k] != after_fp32.topk_expert_ids[k])
            {
                case_drifted = true;
                ++drift_slots;
            }
            EXPECT_EQ(after_q8.topk_expert_ids[k], after_fp32.topk_expert_ids[k])
                << "case=" << case_idx << " slot=" << k;

            const float diff = std::fabs(after_q8.topk_weights[k] - after_fp32.topk_weights[k]);
            max_weight_diff = std::max(max_weight_diff, diff);
            EXPECT_NEAR(after_q8.topk_weights[k], after_fp32.topk_weights[k], 2.0e-2f)
                << "case=" << case_idx << " slot=" << k;
        }
        if (case_drifted)
            ++drift_cases;
    }

    EXPECT_EQ(drift_cases, 0);
    EXPECT_EQ(drift_slots, 0);
    std::cout << "[DecodeRouteSelectQ8RouterMatchesFP32TopKAcrossSeeds] cases="
              << case_count << " drift_cases=" << drift_cases
              << " drift_slots=" << drift_slots << " max_weight_diff="
              << std::fixed << std::setprecision(8) << max_weight_diff << std::endl;
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectFailsOnQ8CacheMissDuringGraphCapture)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int d_model = 32;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::fill(hidden->mutable_data(), hidden->mutable_data() + hidden->numel(), 0.0f);
    hidden->mutable_data()[0] = 1.0f;
    std::fill(gate_weights->mutable_data(), gate_weights->mutable_data() + gate_weights->numel(), 0.0f);
    gate_weights->mutable_data()[0] = 4.0f;
    gate_weights->mutable_data()[d_model] = 3.0f;
    gate_weights->mutable_data()[2 * d_model] = 2.0f;
    gate_weights->mutable_data()[3 * d_model] = 1.0f;

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices->ensureOnDevice(device));
    ASSERT_TRUE(output_weights->ensureOnDevice(device));

    DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, top_k);
    DeviceMoELayerRuntime *runtime = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel, 64, d_model, 512, num_experts, top_k);
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride grouped_env("LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "1");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "1");
        GraphCaptureGuard guard;
        EXPECT_FALSE(gpu_kernel.decodeRouteSelect(
            runtime,
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            false,
            output_indices.get(), output_weights.get(),
            true, true));
    }

    ASSERT_EQ(hipFree(runtime), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectFailsWhenQ8RouterEnabledForUnalignedFP32Gate)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int d_model = 4;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::fill(hidden->mutable_data(), hidden->mutable_data() + hidden->numel(), 0.0f);
    hidden->mutable_data()[0] = 1.0f;
    std::fill(gate_weights->mutable_data(), gate_weights->mutable_data() + gate_weights->numel(), 0.0f);
    gate_weights->mutable_data()[0] = 4.0f;
    gate_weights->mutable_data()[d_model] = 3.0f;
    gate_weights->mutable_data()[2 * d_model] = 2.0f;
    gate_weights->mutable_data()[3 * d_model] = 1.0f;

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices->ensureOnDevice(device));
    ASSERT_TRUE(output_weights->ensureOnDevice(device));

    DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, top_k);
    DeviceMoELayerRuntime *runtime = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel, 64, d_model, 512, num_experts, top_k);
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride grouped_env("LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "1");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "1");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "1");
        EXPECT_FALSE(gpu_kernel.decodeRouteSelect(
            runtime,
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            false,
            output_indices.get(), output_weights.get(),
            true, true));
    }

    ASSERT_EQ(hipFree(runtime), hipSuccess);
}

TEST(Test__ROCmMoEKernel, Route_PrefillLarge)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 32;
    const int d_model = 2048;
    const int num_experts = 256;
    const int top_k = 8;

    std::vector<float> hidden(seq_len * d_model);
    std::vector<float> gate_weights(num_experts * d_model);
    fillRandom(hidden, -1.0f, 1.0f, 42);
    fillRandom(gate_weights, -0.1f, 0.1f, 123);

    // CPU reference
    CPUMoEKernel cpu_kernel;
    MoERoutingResult cpu_result;
    ASSERT_TRUE(cpu_kernel.route(hidden.data(), gate_weights.data(),
                                 seq_len, d_model, num_experts, top_k,
                                 true, cpu_result));

    // GPU
    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    float *d_hidden = nullptr, *d_gate = nullptr;
    (void)hipMalloc(&d_hidden, hidden.size() * sizeof(float));
    (void)hipMalloc(&d_gate, gate_weights.size() * sizeof(float));
    (void)hipMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_gate, gate_weights.data(), gate_weights.size() * sizeof(float), hipMemcpyHostToDevice);

    MoERoutingResult gpu_result;
    ASSERT_TRUE(gpu_kernel.route(d_hidden, d_gate,
                                 seq_len, d_model, num_experts, top_k,
                                 true, gpu_result));

    (void)hipFree(d_hidden);
    (void)hipFree(d_gate);

    // Verify per-token
    ASSERT_FALSE(hasNaNOrInf(gpu_result.router_logits.data(), gpu_result.router_logits.size()));
    double logits_cosine = cosineSimilarity(
        gpu_result.router_logits.data(), cpu_result.router_logits.data(),
        gpu_result.router_logits.size());
    expectStrictVerifierSimilarity(
        "prefill router logits should match CPU routing row-by-row",
        gpu_result.router_logits.data(),
        cpu_result.router_logits.data(),
        gpu_result.router_logits.size(),
        static_cast<size_t>(num_experts));

    // Count matching top-1 experts across all tokens
    int top1_matches = 0;
    for (int t = 0; t < seq_len; ++t)
    {
        if (gpu_result.expert_indices[t * top_k] == cpu_result.expert_indices[t * top_k])
            ++top1_matches;
    }
    double top1_rate = static_cast<double>(top1_matches) / seq_len;
    EXPECT_GE(top1_rate, 0.8)
        << "Top-1 expert match rate: " << top1_rate;

    std::cout << "[Route_PrefillLarge] logits_cosine=" << std::fixed << std::setprecision(6)
              << logits_cosine
              << " top1_match=" << top1_matches << "/" << seq_len << std::endl;
}

// ============================================================================
// Test: gatherTokenBatch()
// ============================================================================

TEST(Test__ROCmMoEKernel, GatherTokenBatch)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 64;
    const int d_model = 2048;
    const int num_tokens = 8;

    std::vector<float> hidden(seq_len * d_model);
    fillRandom(hidden, -1.0f, 1.0f, 42);

    // Select some token indices
    std::vector<int> token_indices = {0, 5, 12, 23, 31, 44, 50, 63};

    // CPU reference
    CPUMoEKernel cpu_kernel;
    std::vector<float> cpu_batch(num_tokens * d_model);
    cpu_kernel.gatherTokenBatch(hidden.data(), cpu_batch.data(),
                                token_indices.data(), num_tokens, d_model);

    // GPU
    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    float *d_hidden = nullptr, *d_batch = nullptr;
    int *d_indices = nullptr;
    (void)hipMalloc(&d_hidden, hidden.size() * sizeof(float));
    (void)hipMalloc(&d_batch, num_tokens * d_model * sizeof(float));
    (void)hipMalloc(&d_indices, num_tokens * sizeof(int));
    (void)hipMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_indices, token_indices.data(), num_tokens * sizeof(int), hipMemcpyHostToDevice);

    gpu_kernel.gatherTokenBatch(d_hidden, d_batch, d_indices, num_tokens, d_model);
    (void)hipDeviceSynchronize();

    std::vector<float> gpu_batch(num_tokens * d_model);
    (void)hipMemcpy(gpu_batch.data(), d_batch, gpu_batch.size() * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_hidden);
    (void)hipFree(d_batch);
    (void)hipFree(d_indices);

    // Exact match expected for gather (just copying)
    ASSERT_FALSE(hasNaNOrInf(gpu_batch.data(), gpu_batch.size()));
    double cosine = cosineSimilarity(gpu_batch.data(), cpu_batch.data(), gpu_batch.size());
    EXPECT_GE(cosine, 0.9999)
        << "Gather cosine: " << cosine;

    // Check element-wise equality (should be bit-exact for copies)
    int mismatches = 0;
    for (size_t i = 0; i < gpu_batch.size(); ++i)
    {
        if (gpu_batch[i] != cpu_batch[i])
            ++mismatches;
    }
    EXPECT_EQ(mismatches, 0) << "Gather had " << mismatches << " mismatches (should be exact copy)";

    std::cout << "[GatherTokenBatch] cosine=" << std::fixed << std::setprecision(6)
              << cosine << " mismatches=" << mismatches << std::endl;
}

// ============================================================================
// Test: scatterAddWeighted()
// ============================================================================

TEST(Test__ROCmMoEKernel, ScatterAddWeighted)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 32;
    const int d_model = 2048;
    const int num_tokens = 8;

    std::vector<float> expert_output(num_tokens * d_model);
    fillRandom(expert_output, -1.0f, 1.0f, 42);

    std::vector<int> token_indices = {0, 3, 7, 10, 15, 20, 25, 30};
    std::vector<float> weights = {0.15f, 0.12f, 0.18f, 0.10f, 0.13f, 0.11f, 0.14f, 0.07f};

    // CPU reference
    CPUMoEKernel cpu_kernel;
    std::vector<float> cpu_output(seq_len * d_model, 0.0f);
    cpu_kernel.scatterAddWeighted(cpu_output.data(), expert_output.data(),
                                  token_indices.data(), weights.data(),
                                  num_tokens, d_model);

    // GPU
    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    float *d_output = nullptr, *d_expert = nullptr, *d_weights = nullptr;
    int *d_indices = nullptr;
    (void)hipMalloc(&d_output, seq_len * d_model * sizeof(float));
    (void)hipMalloc(&d_expert, expert_output.size() * sizeof(float));
    (void)hipMalloc(&d_weights, weights.size() * sizeof(float));
    (void)hipMalloc(&d_indices, token_indices.size() * sizeof(int));
    (void)hipMemset(d_output, 0, seq_len * d_model * sizeof(float));
    (void)hipMemcpy(d_expert, expert_output.data(), expert_output.size() * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_indices, token_indices.data(), token_indices.size() * sizeof(int), hipMemcpyHostToDevice);

    gpu_kernel.scatterAddWeighted(d_output, d_expert, d_indices, d_weights, num_tokens, d_model);
    (void)hipDeviceSynchronize();

    std::vector<float> gpu_output(seq_len * d_model);
    (void)hipMemcpy(gpu_output.data(), d_output, gpu_output.size() * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_output);
    (void)hipFree(d_expert);
    (void)hipFree(d_weights);
    (void)hipFree(d_indices);

    // Verify
    ASSERT_FALSE(hasNaNOrInf(gpu_output.data(), gpu_output.size()));
    double cosine = cosineSimilarity(gpu_output.data(), cpu_output.data(), gpu_output.size());
    double l2_err = relativeL2Error(gpu_output.data(), cpu_output.data(), gpu_output.size());
    expectStrictVerifierSimilarity(
        "scatterAddWeighted should match CPU reference rows",
        gpu_output.data(),
        cpu_output.data(),
        gpu_output.size(),
        static_cast<size_t>(d_model));

    std::cout << "[ScatterAddWeighted] cosine=" << std::fixed << std::setprecision(6)
              << cosine << " l2_err=" << l2_err << std::endl;
}

// ============================================================================
// Test: sharedExpertGate()
// ============================================================================

TEST(Test__ROCmMoEKernel, SharedExpertGate_Decode)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 1;
    const int d_model = 2048;

    std::vector<float> input(seq_len * d_model);
    std::vector<float> gate_inp(d_model);
    std::vector<float> cpu_shared_output(seq_len * d_model);
    std::vector<float> gpu_shared_output_host(seq_len * d_model);
    fillRandom(input, -1.0f, 1.0f, 42);
    fillRandom(gate_inp, -0.5f, 0.5f, 123);
    fillRandom(cpu_shared_output, -2.0f, 2.0f, 456);
    // Copy same initial shared_output for GPU
    gpu_shared_output_host = cpu_shared_output;

    // CPU reference
    CPUMoEKernel cpu_kernel;
    cpu_kernel.sharedExpertGate(input.data(), gate_inp.data(),
                                cpu_shared_output.data(), seq_len, d_model);

    // GPU
    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    float *d_input = nullptr, *d_gate_inp = nullptr, *d_shared_output = nullptr;
    (void)hipMalloc(&d_input, input.size() * sizeof(float));
    (void)hipMalloc(&d_gate_inp, gate_inp.size() * sizeof(float));
    (void)hipMalloc(&d_shared_output, gpu_shared_output_host.size() * sizeof(float));
    (void)hipMemcpy(d_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_gate_inp, gate_inp.data(), gate_inp.size() * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_shared_output, gpu_shared_output_host.data(),
              gpu_shared_output_host.size() * sizeof(float), hipMemcpyHostToDevice);

    gpu_kernel.sharedExpertGate(d_input, d_gate_inp, d_shared_output, seq_len, d_model);
    (void)hipDeviceSynchronize();

    (void)hipMemcpy(gpu_shared_output_host.data(), d_shared_output,
              gpu_shared_output_host.size() * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_input);
    (void)hipFree(d_gate_inp);
    (void)hipFree(d_shared_output);

    // Verify
    ASSERT_FALSE(hasNaNOrInf(gpu_shared_output_host.data(), gpu_shared_output_host.size()));
    double cosine = cosineSimilarity(gpu_shared_output_host.data(), cpu_shared_output.data(),
                                     cpu_shared_output.size());
    double l2_err = relativeL2Error(gpu_shared_output_host.data(), cpu_shared_output.data(),
                                    cpu_shared_output.size());
    expectStrictVerifierSimilarity(
        "sharedExpertGate decode should match CPU reference rows",
        gpu_shared_output_host.data(),
        cpu_shared_output.data(),
        cpu_shared_output.size(),
        static_cast<size_t>(d_model));

    std::cout << "[SharedExpertGate_Decode_Fused] cosine=" << std::fixed << std::setprecision(6)
              << cosine << " l2_err=" << l2_err << std::endl;
}

TEST(Test__ROCmMoEKernel, SharedExpertGate_DecodeFusedFromTensorsMarksOutputDirty)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int seq_len = 1;
    const int d_model = 513;

    std::vector<float> input(seq_len * d_model);
    std::vector<float> gate_inp(d_model);
    std::vector<float> expected_shared_output(seq_len * d_model);
    fillRandom(input, -0.5f, 0.5f, 142);
    fillRandom(gate_inp, -0.25f, 0.25f, 223);
    fillRandom(expected_shared_output, -1.0f, 1.0f, 356);
    std::vector<float> initial_shared_output = expected_shared_output;

    CPUMoEKernel cpu_kernel;
    cpu_kernel.sharedExpertGate(input.data(), gate_inp.data(),
                                expected_shared_output.data(), seq_len, d_model);

    auto input_tensor = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto gate_tensor = TestTensorFactory::createFP32({static_cast<size_t>(d_model)});
    auto shared_output_tensor = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    std::copy(input.begin(), input.end(), input_tensor->mutable_data());
    std::copy(gate_inp.begin(), gate_inp.end(), gate_tensor->mutable_data());
    std::copy(initial_shared_output.begin(), initial_shared_output.end(), shared_output_tensor->mutable_data());

    ASSERT_TRUE(input_tensor->ensureOnDevice(device));
    ASSERT_TRUE(gate_tensor->ensureOnDevice(device));
    ASSERT_TRUE(shared_output_tensor->ensureOnDevice(device));

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    gpu_kernel.sharedExpertGateFromTensors(input_tensor.get(), gate_tensor.get(), shared_output_tensor.get(),
                                           seq_len, d_model);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(shared_output_tensor->coherenceState(), TensorCoherenceState::DEVICE_AUTHORITATIVE);

    const float *actual = shared_output_tensor->data();
    ASSERT_FALSE(hasNaNOrInf(actual, expected_shared_output.size()));
    double cosine = cosineSimilarity(actual, expected_shared_output.data(), expected_shared_output.size());
    double l2_err = relativeL2Error(actual, expected_shared_output.data(), expected_shared_output.size());
    expectStrictVerifierSimilarity(
        "sharedExpertGate tensor decode should match CPU reference rows",
        actual,
        expected_shared_output.data(),
        expected_shared_output.size(),
        static_cast<size_t>(d_model));

    std::cout << "[SharedExpertGate_DecodeFusedFromTensors] cosine=" << std::fixed << std::setprecision(6)
              << cosine << " l2_err=" << l2_err << std::endl;
}

TEST(Test__ROCmMoEKernel, SharedExpertGateAddFromTensorsMatchesCPU)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int seq_len = 3;
    const int d_model = 513;

    std::vector<float> input(seq_len * d_model);
    std::vector<float> gate_inp(d_model);
    std::vector<float> shared_output(seq_len * d_model);
    std::vector<float> routed_residual(seq_len * d_model);
    fillRandom(input, -0.3f, 0.3f, 641);
    fillRandom(gate_inp, -0.2f, 0.2f, 642);
    fillRandom(shared_output, -1.0f, 1.0f, 643);
    fillRandom(routed_residual, -0.5f, 0.5f, 644);

    auto input_cpu = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto input_gpu = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto gate_cpu = TestTensorFactory::createFP32({static_cast<size_t>(d_model)});
    auto gate_gpu = TestTensorFactory::createFP32({static_cast<size_t>(d_model)});
    auto shared_cpu = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto shared_gpu = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto residual_cpu = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto residual_gpu = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto combined_tensor = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto expected_tensor = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    std::copy(input.begin(), input.end(), input_cpu->mutable_data());
    std::copy(input.begin(), input.end(), input_gpu->mutable_data());
    std::copy(gate_inp.begin(), gate_inp.end(), gate_cpu->mutable_data());
    std::copy(gate_inp.begin(), gate_inp.end(), gate_gpu->mutable_data());
    std::copy(shared_output.begin(), shared_output.end(), shared_cpu->mutable_data());
    std::copy(shared_output.begin(), shared_output.end(), shared_gpu->mutable_data());
    std::copy(routed_residual.begin(), routed_residual.end(), residual_cpu->mutable_data());
    std::copy(routed_residual.begin(), routed_residual.end(), residual_gpu->mutable_data());
    std::fill(combined_tensor->mutable_data(), combined_tensor->mutable_data() + combined_tensor->numel(), 0.0f);
    std::fill(expected_tensor->mutable_data(), expected_tensor->mutable_data() + expected_tensor->numel(), 0.0f);

    CPUMoEKernel cpu_kernel;
    cpu_kernel.sharedExpertGateAddFromTensors(
        input_cpu.get(), gate_cpu.get(), shared_cpu.get(),
        residual_cpu.get(), expected_tensor.get(), seq_len, d_model);

    ASSERT_TRUE(input_gpu->ensureOnDevice(device));
    ASSERT_TRUE(gate_gpu->ensureOnDevice(device));
    ASSERT_TRUE(shared_gpu->ensureOnDevice(device));
    ASSERT_TRUE(residual_gpu->ensureOnDevice(device));
    ASSERT_TRUE(combined_tensor->ensureOnDevice(device));

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    gpu_kernel.sharedExpertGateAddFromTensors(
        input_gpu.get(), gate_gpu.get(), shared_gpu.get(),
        residual_gpu.get(), combined_tensor.get(), seq_len, d_model);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(combined_tensor->coherenceState(), TensorCoherenceState::DEVICE_AUTHORITATIVE);

    const float *actual = combined_tensor->data();
    ASSERT_FALSE(hasNaNOrInf(actual, expected_tensor->numel()));
    double cosine = cosineSimilarity(actual, expected_tensor->data(), expected_tensor->numel());
    double l2_err = relativeL2Error(actual, expected_tensor->data(), expected_tensor->numel());
    expectStrictVerifierSimilarity(
        "sharedExpertGateAdd tensor path should match CPU reference rows",
        actual,
        expected_tensor->data(),
        expected_tensor->numel(),
        static_cast<size_t>(d_model));

    std::cout << "[SharedExpertGateAddFromTensors] cosine=" << std::fixed << std::setprecision(6)
              << cosine << " l2_err=" << l2_err << std::endl;
}

TEST(Test__ROCmMoEKernel, SharedExpertGateVerifierRowsM234MatchSerialDecodeRows)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int d_model = 2048;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<ITensorKernel &>(gpu_kernel).setGPUStream(stream);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(
        gpu_kernel,
        /*max_seq_len=*/4,
        d_model,
        /*intermediate=*/512,
        /*num_experts=*/1,
        /*top_k=*/1);

    std::vector<float> gate_values(static_cast<size_t>(d_model));
    for (int i = 0; i < d_model; ++i)
    {
        gate_values[static_cast<size_t>(i)] =
            0.031f * std::sin(0.013f * static_cast<float>(i + 3)) -
            0.024f * std::cos(0.019f * static_cast<float>(i + 7)) +
            0.0005f * static_cast<float>((i % 29) - 14);
    }
    auto gate = TestTensorFactory::createFP32({static_cast<size_t>(d_model)});
    std::copy(gate_values.begin(), gate_values.end(), gate->mutable_data());
    ASSERT_TRUE(gate->ensureOnDevice(device, stream));

    for (int seq_len : {1, 2, 3, 4})
    {
        std::vector<float> input_values(static_cast<size_t>(seq_len) * d_model);
        std::vector<float> shared_values(static_cast<size_t>(seq_len) * d_model);
        std::vector<float> residual_values(static_cast<size_t>(seq_len) * d_model);
        for (size_t i = 0; i < input_values.size(); ++i)
        {
            input_values[i] =
                0.07f * std::sin(0.0067f * static_cast<float>(i + 11 + seq_len)) -
                0.041f * std::cos(0.0103f * static_cast<float>(i + 17)) +
                0.0009f * static_cast<float>(static_cast<int>(i % 23) - 11);
            shared_values[i] =
                0.13f * std::sin(0.0051f * static_cast<float>(i + 5)) +
                0.052f * std::cos(0.0147f * static_cast<float>(i + 13)) -
                0.0011f * static_cast<float>(static_cast<int>(i % 31) - 15);
            residual_values[i] =
                -0.09f * std::sin(0.0079f * static_cast<float>(i + 19)) +
                0.066f * std::cos(0.0111f * static_cast<float>(i + 29)) +
                0.0007f * static_cast<float>(static_cast<int>(i % 17) - 8);
        }

        auto grouped_input = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        auto grouped_shared_only = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        auto grouped_shared_add = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        auto grouped_residual = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        auto grouped_combined = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        std::copy(input_values.begin(), input_values.end(), grouped_input->mutable_data());
        std::copy(shared_values.begin(), shared_values.end(), grouped_shared_only->mutable_data());
        std::copy(shared_values.begin(), shared_values.end(), grouped_shared_add->mutable_data());
        std::copy(residual_values.begin(), residual_values.end(), grouped_residual->mutable_data());
        std::fill(grouped_combined->mutable_data(),
                  grouped_combined->mutable_data() + grouped_combined->numel(),
                  0.0f);
        ASSERT_TRUE(grouped_input->ensureOnDevice(device, stream));
        ASSERT_TRUE(grouped_shared_only->ensureOnDevice(device, stream));
        ASSERT_TRUE(grouped_shared_add->ensureOnDevice(device, stream));
        ASSERT_TRUE(grouped_residual->ensureOnDevice(device, stream));
        ASSERT_TRUE(grouped_combined->ensureOnDevice(device, stream));

        gpu_kernel.sharedExpertGateFromTensors(
            grouped_input.get(),
            gate.get(),
            grouped_shared_only.get(),
            seq_len,
            d_model);
        gpu_kernel.sharedExpertGateAddFromTensors(
            grouped_input.get(),
            gate.get(),
            grouped_shared_add.get(),
            grouped_residual.get(),
            grouped_combined.get(),
            seq_len,
            d_model);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        std::vector<float> serial_shared_only(static_cast<size_t>(seq_len) * d_model);
        std::vector<float> serial_shared_add(static_cast<size_t>(seq_len) * d_model);
        std::vector<float> serial_combined(static_cast<size_t>(seq_len) * d_model);

        for (int row = 0; row < seq_len; ++row)
        {
            const auto row_offset = static_cast<size_t>(row) * d_model;
            auto row_input = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
            auto row_shared_only = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
            auto row_shared_add = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
            auto row_residual = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
            auto row_combined = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
            std::copy_n(input_values.data() + row_offset, d_model, row_input->mutable_data());
            std::copy_n(shared_values.data() + row_offset, d_model, row_shared_only->mutable_data());
            std::copy_n(shared_values.data() + row_offset, d_model, row_shared_add->mutable_data());
            std::copy_n(residual_values.data() + row_offset, d_model, row_residual->mutable_data());
            std::fill(row_combined->mutable_data(),
                      row_combined->mutable_data() + row_combined->numel(),
                      0.0f);
            ASSERT_TRUE(row_input->ensureOnDevice(device, stream));
            ASSERT_TRUE(row_shared_only->ensureOnDevice(device, stream));
            ASSERT_TRUE(row_shared_add->ensureOnDevice(device, stream));
            ASSERT_TRUE(row_residual->ensureOnDevice(device, stream));
            ASSERT_TRUE(row_combined->ensureOnDevice(device, stream));

            gpu_kernel.sharedExpertGateFromTensors(
                row_input.get(),
                gate.get(),
                row_shared_only.get(),
                1,
                d_model);
            gpu_kernel.sharedExpertGateAddFromTensors(
                row_input.get(),
                gate.get(),
                row_shared_add.get(),
                row_residual.get(),
                row_combined.get(),
                1,
                d_model);
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

            row_shared_only->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
            row_shared_add->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
            row_combined->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
            std::copy_n(row_shared_only->data(), d_model, serial_shared_only.data() + row_offset);
            std::copy_n(row_shared_add->data(), d_model, serial_shared_add.data() + row_offset);
            std::copy_n(row_combined->data(), d_model, serial_combined.data() + row_offset);
        }

        grouped_shared_only->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        grouped_shared_add->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        grouped_combined->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);

        ASSERT_EQ(std::vector<float>(grouped_shared_only->data(),
                                     grouped_shared_only->data() + grouped_shared_only->numel()),
                  serial_shared_only)
            << "seq_len=" << seq_len;
        ASSERT_EQ(std::vector<float>(grouped_shared_add->data(),
                                     grouped_shared_add->data() + grouped_shared_add->numel()),
                  serial_shared_add)
            << "seq_len=" << seq_len;
        ASSERT_EQ(std::vector<float>(grouped_combined->data(),
                                     grouped_combined->data() + grouped_combined->numel()),
                  serial_combined)
            << "seq_len=" << seq_len;
    }

    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

/**
 * @brief Prove fused shared-gate LocalTP partials are grouped-decode invariant.
 *
 * The Qwen3.6 LocalTP MoE graph writes `combined = routed_partial +
 * gate * shared_partial` on every participant and then allreduces those
 * row-parallel partials.  This regression exercises two real ROCm participants
 * with production fused gate-add kernels, compares grouped M=2/3/4 verifier
 * rows to serial M=1 row replay, and requires byte-identical allreduced FP32
 * results.  There is intentionally no CPU oracle in this test: the contract is
 * backend serial-decode equivalence on the real GPU path.
 */
TEST(Test__ROCmMoEKernel, SharedExpertGateAddLocalTP2VerifierRowsMatchSerialAllreduce)
{
    SKIP_IF_NO_ROCM();

    int device_count = 0;
    ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
    if (device_count < 2)
        GTEST_SKIP() << "ROCm LocalTP2 regression needs two ROCm devices";

    constexpr int d_model = 2048;
    std::array<DeviceId, 2> devices = {DeviceId::rocm(0), DeviceId::rocm(1)};
    ScopedHipDeviceStream stream0(0);
    ScopedHipDeviceStream stream1(1);
    ASSERT_EQ(stream0.status(), hipSuccess);
    ASSERT_EQ(stream1.status(), hipSuccess);
    ASSERT_NE(stream0.get(), nullptr);
    ASSERT_NE(stream1.get(), nullptr);

    ROCmMoEKernel kernel0(0);
    ROCmMoEKernel kernel1(1);
    static_cast<ITensorKernel &>(kernel0).setGPUStream(stream0.get());
    static_cast<ITensorKernel &>(kernel1).setGPUStream(stream1.get());
    std::array<ROCmMoEKernel *, 2> kernels = {&kernel0, &kernel1};
    std::array<hipStream_t, 2> streams = {stream0.get(), stream1.get()};

    std::vector<float> gate_values(static_cast<size_t>(d_model));
    for (int i = 0; i < d_model; ++i)
    {
        gate_values[static_cast<size_t>(i)] =
            0.017f * std::sin(0.0091f * static_cast<float>(i + 5)) -
            0.011f * std::cos(0.0143f * static_cast<float>(i + 19)) +
            0.00031f * static_cast<float>((i % 37) - 18);
    }

    auto gate0 = TestTensorFactory::createFP32({static_cast<size_t>(d_model)});
    auto gate1 = TestTensorFactory::createFP32({static_cast<size_t>(d_model)});
    std::copy(gate_values.begin(), gate_values.end(), gate0->mutable_data());
    std::copy(gate_values.begin(), gate_values.end(), gate1->mutable_data());
    ASSERT_TRUE(gate0->ensureOnDevice(devices[0], streams[0]));
    ASSERT_TRUE(gate1->ensureOnDevice(devices[1], streams[1]));
    std::array<FP32Tensor *, 2> gates = {gate0.get(), gate1.get()};

    auto add_partials_in_order =
        [](std::vector<float> &sum, const std::vector<float> &partial)
    {
        ASSERT_EQ(sum.size(), partial.size());
        for (size_t i = 0; i < sum.size(); ++i)
            sum[i] += partial[i];
    };

    for (int seq_len : {2, 3, 4})
    {
        SCOPED_TRACE("seq_len=" + std::to_string(seq_len));
        const size_t element_count = static_cast<size_t>(seq_len) * d_model;
        std::vector<float> input_values(element_count);
        std::array<std::vector<float>, 2> shared_values;
        std::array<std::vector<float>, 2> residual_values;
        for (auto &values : shared_values)
            values.resize(element_count);
        for (auto &values : residual_values)
            values.resize(element_count);

        for (size_t i = 0; i < element_count; ++i)
        {
            input_values[i] =
                0.051f * std::sin(0.0047f * static_cast<float>(i + 11 + seq_len)) -
                0.037f * std::cos(0.0089f * static_cast<float>(i + 23)) +
                0.00063f * static_cast<float>(static_cast<int>(i % 41) - 20);
            for (int participant = 0; participant < 2; ++participant)
            {
                shared_values[static_cast<size_t>(participant)][i] =
                    (0.071f + 0.013f * participant) *
                        std::sin(0.0061f * static_cast<float>(i + 7 + 3 * participant)) +
                    (0.043f - 0.004f * participant) *
                        std::cos(0.0127f * static_cast<float>(i + 31)) -
                    0.00047f * static_cast<float>(static_cast<int>((i + participant) % 29) - 14);
                residual_values[static_cast<size_t>(participant)][i] =
                    (-0.059f + 0.006f * participant) *
                        std::sin(0.0053f * static_cast<float>(i + 17)) +
                    (0.049f + 0.003f * participant) *
                        std::cos(0.0101f * static_cast<float>(i + 13 + participant)) +
                    0.00052f * static_cast<float>(static_cast<int>((i + 5 * participant) % 23) - 11);
            }
        }

        std::vector<float> grouped_allreduced(element_count, 0.0f);
        for (int participant = 0; participant < 2; ++participant)
        {
            auto grouped_input = TestTensorFactory::createFP32(
                {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
            auto grouped_shared = TestTensorFactory::createFP32(
                {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
            auto grouped_residual = TestTensorFactory::createFP32(
                {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
            auto grouped_combined = TestTensorFactory::createFP32(
                {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
            std::copy(input_values.begin(), input_values.end(), grouped_input->mutable_data());
            std::copy(shared_values[static_cast<size_t>(participant)].begin(),
                      shared_values[static_cast<size_t>(participant)].end(),
                      grouped_shared->mutable_data());
            std::copy(residual_values[static_cast<size_t>(participant)].begin(),
                      residual_values[static_cast<size_t>(participant)].end(),
                      grouped_residual->mutable_data());
            std::fill(grouped_combined->mutable_data(),
                      grouped_combined->mutable_data() + grouped_combined->numel(),
                      0.0f);
            ASSERT_TRUE(grouped_input->ensureOnDevice(devices[static_cast<size_t>(participant)],
                                                      streams[static_cast<size_t>(participant)]));
            ASSERT_TRUE(grouped_shared->ensureOnDevice(devices[static_cast<size_t>(participant)],
                                                       streams[static_cast<size_t>(participant)]));
            ASSERT_TRUE(grouped_residual->ensureOnDevice(devices[static_cast<size_t>(participant)],
                                                         streams[static_cast<size_t>(participant)]));
            ASSERT_TRUE(grouped_combined->ensureOnDevice(devices[static_cast<size_t>(participant)],
                                                         streams[static_cast<size_t>(participant)]));

            kernels[static_cast<size_t>(participant)]->sharedExpertGateAddFromTensors(
                grouped_input.get(),
                gates[static_cast<size_t>(participant)],
                grouped_shared.get(),
                grouped_residual.get(),
                grouped_combined.get(),
                seq_len,
                d_model);
            ASSERT_EQ(hipStreamSynchronize(streams[static_cast<size_t>(participant)]), hipSuccess);

            add_partials_in_order(
                grouped_allreduced,
                copyROCmFP32TensorToHost(
                    grouped_combined.get(),
                    streams[static_cast<size_t>(participant)]));
        }

        std::vector<float> serial_allreduced(element_count, 0.0f);
        for (int row = 0; row < seq_len; ++row)
        {
            const size_t row_offset = static_cast<size_t>(row) * d_model;
            for (int participant = 0; participant < 2; ++participant)
            {
                std::vector<float> row_input(static_cast<size_t>(d_model));
                std::vector<float> row_shared(static_cast<size_t>(d_model));
                std::vector<float> row_residual(static_cast<size_t>(d_model));
                std::copy_n(input_values.data() + row_offset, d_model, row_input.data());
                std::copy_n(shared_values[static_cast<size_t>(participant)].data() + row_offset,
                            d_model,
                            row_shared.data());
                std::copy_n(residual_values[static_cast<size_t>(participant)].data() + row_offset,
                            d_model,
                            row_residual.data());

                auto row_input_tensor = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
                auto row_shared_tensor = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
                auto row_residual_tensor = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
                auto row_combined_tensor = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
                std::copy(row_input.begin(), row_input.end(), row_input_tensor->mutable_data());
                std::copy(row_shared.begin(), row_shared.end(), row_shared_tensor->mutable_data());
                std::copy(row_residual.begin(), row_residual.end(), row_residual_tensor->mutable_data());
                std::fill(row_combined_tensor->mutable_data(),
                          row_combined_tensor->mutable_data() + row_combined_tensor->numel(),
                          0.0f);
                ASSERT_TRUE(row_input_tensor->ensureOnDevice(devices[static_cast<size_t>(participant)],
                                                             streams[static_cast<size_t>(participant)]));
                ASSERT_TRUE(row_shared_tensor->ensureOnDevice(devices[static_cast<size_t>(participant)],
                                                              streams[static_cast<size_t>(participant)]));
                ASSERT_TRUE(row_residual_tensor->ensureOnDevice(devices[static_cast<size_t>(participant)],
                                                                streams[static_cast<size_t>(participant)]));
                ASSERT_TRUE(row_combined_tensor->ensureOnDevice(devices[static_cast<size_t>(participant)],
                                                                streams[static_cast<size_t>(participant)]));

                kernels[static_cast<size_t>(participant)]->sharedExpertGateAddFromTensors(
                    row_input_tensor.get(),
                    gates[static_cast<size_t>(participant)],
                    row_shared_tensor.get(),
                    row_residual_tensor.get(),
                    row_combined_tensor.get(),
                    1,
                    d_model);
                ASSERT_EQ(hipStreamSynchronize(streams[static_cast<size_t>(participant)]), hipSuccess);

                const auto row_partial = copyROCmFP32TensorToHost(
                    row_combined_tensor.get(),
                    streams[static_cast<size_t>(participant)]);
                for (int col = 0; col < d_model; ++col)
                    serial_allreduced[row_offset + static_cast<size_t>(col)] +=
                        row_partial[static_cast<size_t>(col)];
            }
        }

        expectBitwiseVerifierRowsEqual(
            "ROCm LocalTP2 fused shared gate-add allreduced verifier rows",
            grouped_allreduced.data(),
            serial_allreduced.data(),
            grouped_allreduced.size(),
            static_cast<size_t>(d_model));
    }
    EXPECT_EQ(hipSetDevice(0), hipSuccess);
}

TEST(Test__ROCmMoEKernel, SharedExpertGate_Prefill)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 64;
    const int d_model = 2048;

    std::vector<float> input(seq_len * d_model);
    std::vector<float> gate_inp(d_model);
    std::vector<float> cpu_shared_output(seq_len * d_model);
    std::vector<float> gpu_shared_output_host(seq_len * d_model);
    fillRandom(input, -1.0f, 1.0f, 42);
    fillRandom(gate_inp, -0.5f, 0.5f, 123);
    fillRandom(cpu_shared_output, -2.0f, 2.0f, 456);
    gpu_shared_output_host = cpu_shared_output;

    CPUMoEKernel cpu_kernel;
    cpu_kernel.sharedExpertGate(input.data(), gate_inp.data(),
                                cpu_shared_output.data(), seq_len, d_model);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    float *d_input = nullptr, *d_gate_inp = nullptr, *d_shared_output = nullptr;
    (void)hipMalloc(&d_input, input.size() * sizeof(float));
    (void)hipMalloc(&d_gate_inp, gate_inp.size() * sizeof(float));
    (void)hipMalloc(&d_shared_output, gpu_shared_output_host.size() * sizeof(float));
    (void)hipMemcpy(d_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_gate_inp, gate_inp.data(), gate_inp.size() * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_shared_output, gpu_shared_output_host.data(),
              gpu_shared_output_host.size() * sizeof(float), hipMemcpyHostToDevice);

    gpu_kernel.sharedExpertGate(d_input, d_gate_inp, d_shared_output, seq_len, d_model);
    (void)hipDeviceSynchronize();

    (void)hipMemcpy(gpu_shared_output_host.data(), d_shared_output,
              gpu_shared_output_host.size() * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_input);
    (void)hipFree(d_gate_inp);
    (void)hipFree(d_shared_output);

    ASSERT_FALSE(hasNaNOrInf(gpu_shared_output_host.data(), gpu_shared_output_host.size()));
    double cosine = cosineSimilarity(gpu_shared_output_host.data(), cpu_shared_output.data(),
                                     cpu_shared_output.size());
    expectStrictVerifierSimilarity(
        "sharedExpertGate prefill should match CPU reference rows",
        gpu_shared_output_host.data(),
        cpu_shared_output.data(),
        cpu_shared_output.size(),
        static_cast<size_t>(d_model));

    std::cout << "[SharedExpertGate_Prefill] cosine=" << std::fixed << std::setprecision(6)
              << cosine << std::endl;
}

// ============================================================================
// Test: swiGLU()
// ============================================================================

TEST(Test__ROCmMoEKernel, SwiGLU)
{
    SKIP_IF_NO_ROCM();

    const int count = 32 * 4864; // Typical MoE intermediate dim

    std::vector<float> gate(count), up(count);
    std::vector<float> cpu_gate(count), gpu_gate_host(count);
    fillRandom(gate, -2.0f, 2.0f, 42);
    fillRandom(up, -2.0f, 2.0f, 123);
    cpu_gate = gate;
    gpu_gate_host = gate;

    // CPU reference
    CPUMoEKernel cpu_kernel;
    cpu_kernel.swiGLU(cpu_gate.data(), up.data(), count);

    // GPU
    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    float *d_gate = nullptr, *d_up = nullptr;
    (void)hipMalloc(&d_gate, count * sizeof(float));
    (void)hipMalloc(&d_up, count * sizeof(float));
    (void)hipMemcpy(d_gate, gpu_gate_host.data(), count * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_up, up.data(), count * sizeof(float), hipMemcpyHostToDevice);

    gpu_kernel.swiGLU(d_gate, d_up, count);
    (void)hipDeviceSynchronize();

    (void)hipMemcpy(gpu_gate_host.data(), d_gate, count * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_gate);
    (void)hipFree(d_up);

    // Verify
    ASSERT_FALSE(hasNaNOrInf(gpu_gate_host.data(), gpu_gate_host.size()));
    double cosine = cosineSimilarity(gpu_gate_host.data(), cpu_gate.data(), count);
    double l2_err = relativeL2Error(gpu_gate_host.data(), cpu_gate.data(), count);
    expectStrictVerifierSimilarity(
        "swiGLU should match CPU reference rows",
        gpu_gate_host.data(),
        cpu_gate.data(),
        count,
        count);

    std::cout << "[SwiGLU] cosine=" << std::fixed << std::setprecision(6)
              << cosine << " l2_err=" << l2_err << std::endl;
}

// ============================================================================
// Test: grouped decode expert down path matches existing sequential ROCm path
// ============================================================================

TEST(Test__ROCmMoEKernel, GroupedExpertDownDecode_Q4_0MatchesSequential)
{
    SKIP_IF_NO_ROCM();

    const int d_model = 128;
    const int intermediate = 128;
    const int num_active = 4;
    const int expert_ids[num_active] = {2, 7, 3, 5};
    const float route_weights[num_active] = {0.39f, 0.27f, 0.21f, 0.13f};
    const DeviceId device = DeviceId::rocm(0);

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    std::vector<std::unique_ptr<TensorBase>> weight_tensors;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> down_kernels;
    weight_tensors.reserve(num_active);
    packed_weights.reserve(num_active);
    down_kernels.reserve(num_active);

    for (int i = 0; i < num_active; ++i)
    {
        auto weights = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)});
        auto packed = std::make_unique<rocm::ROCmPackedWeights>();
        ASSERT_TRUE(rocm::packWeightsToROCm(weights.get(), *packed))
            << "packWeightsToROCm failed for expert " << expert_ids[i];
        ASSERT_FALSE(packed->native_vnni_payload.empty());

        auto kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(packed.get(), 0);
        weight_tensors.push_back(std::move(weights));
        packed_weights.push_back(std::move(packed));
        down_kernels.push_back(std::move(kernel));
    }

    auto reqs = down_kernels[0]->getWorkspaceRequirements(1, d_model, intermediate);
    auto workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
    ASSERT_TRUE(workspace->allocate(reqs));
    for (auto &kernel : down_kernels)
    {
        kernel->bindWorkspace(workspace.get());
        kernel->prepareWeights();
    }

    std::vector<std::shared_ptr<FP32Tensor>> gate_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> up_tensors;
    gate_tensors.reserve(num_active);
    up_tensors.reserve(num_active);
    for (int i = 0; i < num_active; ++i)
    {
        auto gate = TestTensorFactory::createFP32Random(
            {1, static_cast<size_t>(intermediate)}, -2.0f, 2.0f, 100 + i);
        auto up = TestTensorFactory::createFP32Random(
            {1, static_cast<size_t>(intermediate)}, -2.0f, 2.0f, 200 + i);
        ASSERT_TRUE(gate->ensureOnDevice(device));
        ASSERT_TRUE(up->ensureOnDevice(device));
        gate_tensors.push_back(std::move(gate));
        up_tensors.push_back(std::move(up));
    }

    auto sequential_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto grouped_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto device_routed_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    ASSERT_TRUE(sequential_output->ensureOnDevice(device));
    ASSERT_TRUE(grouped_output->ensureOnDevice(device));
    ASSERT_TRUE(device_routed_output->ensureOnDevice(device));
    moe_kernel.zeroBuffer(sequential_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    moe_kernel.zeroBuffer(grouped_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    moe_kernel.zeroBuffer(device_routed_output.get(), static_cast<size_t>(d_model) * sizeof(float));

    for (int i = 0; i < num_active; ++i)
    {
        auto expert_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
        ASSERT_TRUE(expert_output->ensureOnDevice(device));
        ASSERT_TRUE(down_kernels[i]->multiply_tensor_with_fused_swiglu(
            gate_tensors[i].get(), up_tensors[i].get(), expert_output.get(),
            1, d_model, intermediate));
        expert_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        moe_kernel.weightedAddFromTensors(
            sequential_output.get(), expert_output.get(), route_weights[i], d_model);
    }
    sequential_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    ITensor *gate_ptrs[num_active] = {};
    ITensor *up_ptrs[num_active] = {};
    DeviceNativeVNNIMatrixDesc descs[num_active] = {};
    for (int i = 0; i < num_active; ++i)
    {
        gate_ptrs[i] = gate_tensors[i].get();
        up_ptrs[i] = up_tensors[i].get();
        ASSERT_TRUE(down_kernels[i]->exportNativeVNNIMatrixDesc(descs[i]));
        ASSERT_EQ(descs[i].n, d_model);
        ASSERT_EQ(descs[i].k, intermediate);
    }

    ASSERT_TRUE(moe_kernel.groupedExpertDownDecode(
        gate_ptrs, up_ptrs, expert_ids, route_weights, descs,
        num_active, grouped_output.get(), d_model, intermediate));
    (void)hipDeviceSynchronize();

    sequential_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    const float *seq = sequential_output->data();
    const float *grouped = grouped_output->data();
    ASSERT_FALSE(hasNaNOrInf(grouped, d_model));
    expectStrictVerifierSimilarity(
        "ROCm grouped expert down decode Q4_0 must match sequential decode",
        grouped,
        seq,
        static_cast<size_t>(d_model),
        static_cast<size_t>(d_model));
    const double cosine = cosineSimilarity(grouped, seq, d_model);
    const double l2_err = relativeL2Error(grouped, seq, d_model);

    std::cout << "[GroupedExpertDownDecode_Q4_0MatchesSequential] cosine="
              << std::fixed << std::setprecision(6) << cosine
              << " l2_err=" << l2_err << std::endl;
}

template <typename WeightFactory>
void runGroupedExpertDownDecodeFormatMatch(const char *label, WeightFactory create_weights)
{
    const int d_model = 128;
    const int intermediate = 256;
    const int num_experts = 8;
    const int num_active = 4;
    const int expert_ids[num_active] = {2, 7, 3, 5};
    const float route_weights[num_active] = {0.39f, 0.27f, 0.21f, 0.13f};
    const DeviceId device = DeviceId::rocm(0);

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    std::vector<std::unique_ptr<TensorBase>> weight_tensors;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> down_kernels;
    weight_tensors.reserve(num_experts);
    packed_weights.reserve(num_experts);
    down_kernels.reserve(num_experts);

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto weights = create_weights(static_cast<size_t>(d_model), static_cast<size_t>(intermediate));
        auto packed = std::make_unique<rocm::ROCmPackedWeights>();
        ASSERT_TRUE(rocm::packWeightsToROCm(weights.get(), *packed))
            << "packWeightsToROCm failed for " << label << " expert " << expert_id;
        ASSERT_FALSE(packed->native_vnni_payload.empty());

        auto kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(packed.get(), 0);
        weight_tensors.push_back(std::move(weights));
        packed_weights.push_back(std::move(packed));
        down_kernels.push_back(std::move(kernel));
    }

    auto reqs = down_kernels[0]->getWorkspaceRequirements(1, d_model, intermediate);
    auto workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
    ASSERT_TRUE(workspace->allocate(reqs));
    for (auto &kernel : down_kernels)
    {
        kernel->bindWorkspace(workspace.get());
        kernel->prepareWeights();
    }

    std::vector<std::shared_ptr<FP32Tensor>> gate_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> up_tensors;
    gate_tensors.reserve(num_active);
    up_tensors.reserve(num_active);
    for (int i = 0; i < num_active; ++i)
    {
        auto gate = TestTensorFactory::createFP32Random(
            {1, static_cast<size_t>(intermediate)}, -2.0f, 2.0f, 300 + i);
        auto up = TestTensorFactory::createFP32Random(
            {1, static_cast<size_t>(intermediate)}, -2.0f, 2.0f, 400 + i);
        ASSERT_TRUE(gate->ensureOnDevice(device));
        ASSERT_TRUE(up->ensureOnDevice(device));
        gate_tensors.push_back(std::move(gate));
        up_tensors.push_back(std::move(up));
    }

    auto sequential_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto grouped_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto device_routed_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto parallel_device_routed_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto runtime_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto parallel_runtime_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    ASSERT_TRUE(sequential_output->ensureOnDevice(device));
    ASSERT_TRUE(grouped_output->ensureOnDevice(device));
    ASSERT_TRUE(device_routed_output->ensureOnDevice(device));
    ASSERT_TRUE(parallel_device_routed_output->ensureOnDevice(device));
    ASSERT_TRUE(runtime_output->ensureOnDevice(device));
    ASSERT_TRUE(parallel_runtime_output->ensureOnDevice(device));
    moe_kernel.zeroBuffer(sequential_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    moe_kernel.zeroBuffer(grouped_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    moe_kernel.zeroBuffer(device_routed_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    moe_kernel.zeroBuffer(parallel_device_routed_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    moe_kernel.zeroBuffer(runtime_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    moe_kernel.zeroBuffer(parallel_runtime_output.get(), static_cast<size_t>(d_model) * sizeof(float));

    for (int i = 0; i < num_active; ++i)
    {
        auto expert_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
        ASSERT_TRUE(expert_output->ensureOnDevice(device));
        ASSERT_TRUE(down_kernels[expert_ids[i]]->multiply_tensor_with_fused_swiglu(
            gate_tensors[i].get(), up_tensors[i].get(), expert_output.get(),
            1, d_model, intermediate));
        expert_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        moe_kernel.weightedAddFromTensors(
            sequential_output.get(), expert_output.get(), route_weights[i], d_model);
    }
    sequential_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    ITensor *gate_ptrs[num_active] = {};
    ITensor *up_ptrs[num_active] = {};
    std::vector<DeviceNativeVNNIMatrixDesc> descs(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        ASSERT_TRUE(down_kernels[expert_id]->exportNativeVNNIMatrixDesc(descs[expert_id]));
        ASSERT_EQ(descs[expert_id].n, d_model);
        ASSERT_EQ(descs[expert_id].k, intermediate);
    }
    for (int i = 0; i < num_active; ++i)
    {
        gate_ptrs[i] = gate_tensors[i].get();
        up_ptrs[i] = up_tensors[i].get();
    }

    const int table_id = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(table_id, 0);

    ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromTable(
        gate_ptrs, up_ptrs, expert_ids, route_weights, table_id,
        num_active, grouped_output.get(), d_model, intermediate));

    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(num_active)});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_active)});
    for (int i = 0; i < num_active; ++i)
    {
        routing_indices->mutable_data()[i] = static_cast<float>(expert_ids[i]);
        routing_weights->mutable_data()[i] = route_weights[i];
    }
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    {
        ScopedROCmEnvOverride disable_parallel_down("LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE", "0");
        ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromRouting(
            gate_ptrs, up_ptrs, routing_indices.get(), routing_weights.get(), table_id,
            num_active, device_routed_output.get(), d_model, intermediate));
    }

    {
        ScopedROCmEnvOverride enable_parallel_down("LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE", "1");
        ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromRouting(
            gate_ptrs, up_ptrs, routing_indices.get(), routing_weights.get(), table_id,
            num_active, parallel_device_routed_output.get(), d_model, intermediate));
    }

    DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, num_active);
    populateRuntimeDescriptors(host_runtime, nullptr, nullptr, &descs);
    for (int i = 0; i < num_active; ++i)
    {
        host_runtime.topk_expert_ids[i] = expert_ids[i];
        host_runtime.topk_weights[i] = route_weights[i];
    }
    DeviceMoELayerRuntime *device_runtime = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_runtime), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(device_runtime, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    {
        ScopedROCmEnvOverride disable_parallel_down("LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE", "0");
        ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromRuntime(
            gate_ptrs, up_ptrs, device_runtime, table_id,
            num_active, runtime_output.get(), d_model, intermediate));
    }

    {
        ScopedROCmEnvOverride enable_parallel_down("LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE", "1");
        ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromRuntime(
            gate_ptrs, up_ptrs, device_runtime, table_id,
            num_active, parallel_runtime_output.get(), d_model, intermediate));
    }
    (void)hipDeviceSynchronize();
    ASSERT_EQ(hipFree(device_runtime), hipSuccess);

    sequential_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    device_routed_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    parallel_device_routed_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    runtime_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    parallel_runtime_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    const float *seq = sequential_output->data();
    const float *grouped = grouped_output->data();
    const float *device_routed = device_routed_output->data();
    const float *parallel_device_routed = parallel_device_routed_output->data();
    const float *runtime = runtime_output->data();
    const float *parallel_runtime = parallel_runtime_output->data();
    ASSERT_FALSE(hasNaNOrInf(grouped, d_model));
    ASSERT_FALSE(hasNaNOrInf(device_routed, d_model));
    ASSERT_FALSE(hasNaNOrInf(parallel_device_routed, d_model));
    ASSERT_FALSE(hasNaNOrInf(runtime, d_model));
    ASSERT_FALSE(hasNaNOrInf(parallel_runtime, d_model));
    const double cosine = cosineSimilarity(grouped, seq, d_model);
    const double l2_err = relativeL2Error(grouped, seq, d_model);
    const double device_cosine = cosineSimilarity(device_routed, seq, d_model);
    const double device_l2_err = relativeL2Error(device_routed, seq, d_model);
    const double runtime_cosine = cosineSimilarity(runtime, seq, d_model);
    const double runtime_l2_err = relativeL2Error(runtime, seq, d_model);
    const double parallel_runtime_cosine = cosineSimilarity(parallel_runtime, seq, d_model);
    const double parallel_runtime_l2_err = relativeL2Error(parallel_runtime, seq, d_model);
    const double parallel_vs_runtime_cosine = cosineSimilarity(parallel_runtime, runtime, d_model);
    const double parallel_vs_runtime_l2_err = relativeL2Error(parallel_runtime, runtime, d_model);
    const double parallel_vs_runtime_max_abs = maxAbsDiff(parallel_runtime, runtime, d_model);
    const double parallel_device_vs_runtime_cosine = cosineSimilarity(parallel_device_routed, parallel_runtime, d_model);
    const double parallel_device_vs_runtime_l2_err =
        relativeL2Error(parallel_device_routed, parallel_runtime, d_model);
    const double parallel_device_vs_runtime_max_abs =
        maxAbsDiff(parallel_device_routed, parallel_runtime, d_model);

    expectStrictVerifierSimilarity(
        (std::string(label) + " grouped expert down decode must match sequential").c_str(),
        grouped,
        seq,
        static_cast<size_t>(d_model),
        static_cast<size_t>(d_model));
    expectStrictVerifierSimilarity(
        (std::string(label) + " device-routed grouped expert down decode must match sequential").c_str(),
        device_routed,
        seq,
        static_cast<size_t>(d_model),
        static_cast<size_t>(d_model));
    expectStrictVerifierSimilarity(
        (std::string(label) + " runtime grouped expert down decode must match sequential").c_str(),
        runtime,
        seq,
        static_cast<size_t>(d_model),
        static_cast<size_t>(d_model));
    expectStrictVerifierSimilarity(
        (std::string(label) + " parallel device-routed grouped expert down decode must match sequential").c_str(),
        parallel_device_routed,
        seq,
        static_cast<size_t>(d_model),
        static_cast<size_t>(d_model));
    expectStrictVerifierSimilarity(
        (std::string(label) + " parallel runtime grouped expert down decode must match sequential").c_str(),
        parallel_runtime,
        seq,
        static_cast<size_t>(d_model),
        static_cast<size_t>(d_model));
    EXPECT_GE(parallel_vs_runtime_cosine, 0.99999)
        << label << " parallel runtime should closely match serial runtime, cosine=" << parallel_vs_runtime_cosine;
    EXPECT_LE(parallel_vs_runtime_l2_err, 0.002)
        << label << " parallel runtime should closely match serial runtime, relative L2=" << parallel_vs_runtime_l2_err;
    EXPECT_LE(parallel_vs_runtime_max_abs, 0.05)
        << label << " parallel runtime max abs diff vs serial runtime too high: " << parallel_vs_runtime_max_abs;
    EXPECT_GE(parallel_device_vs_runtime_cosine, 0.999999)
        << label << " parallel device-routed down should match parallel runtime, cosine="
        << parallel_device_vs_runtime_cosine;
    EXPECT_LE(parallel_device_vs_runtime_l2_err, 1.0e-5)
        << label << " parallel device-routed down should match parallel runtime, relative L2="
        << parallel_device_vs_runtime_l2_err;
    EXPECT_LE(parallel_device_vs_runtime_max_abs, 1.0e-4)
        << label << " parallel device-routed down max abs diff vs parallel runtime too high: "
        << parallel_device_vs_runtime_max_abs;

    std::cout << "[GroupedExpertDownDecode_" << label << "MatchesSequential] cosine="
              << std::fixed << std::setprecision(6) << cosine
              << " l2_err=" << l2_err
              << " device_cosine=" << device_cosine
              << " device_l2_err=" << device_l2_err
              << " runtime_cosine=" << runtime_cosine
              << " runtime_l2_err=" << runtime_l2_err
              << " parallel_runtime_cosine=" << parallel_runtime_cosine
              << " parallel_runtime_l2_err=" << parallel_runtime_l2_err
              << " parallel_vs_runtime_l2_err=" << parallel_vs_runtime_l2_err
              << " parallel_vs_runtime_max_abs=" << parallel_vs_runtime_max_abs
              << " parallel_device_vs_runtime_l2_err=" << parallel_device_vs_runtime_l2_err
              << " parallel_device_vs_runtime_max_abs=" << parallel_device_vs_runtime_max_abs << std::endl;
}

TEST(Test__ROCmMoEKernel, GroupedExpertDownDecode_Q5_KMatchesSequential)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertDownDecodeFormatMatch("Q5_K", [](size_t rows, size_t cols)
                                          {
                                              auto tensor = TestTensorFactory::createQ5_KRandom({rows, cols});
                                              injectNonZeroKQuantMins(tensor.get());
                                              return tensor;
                                          });
}

TEST(Test__ROCmMoEKernel, GroupedExpertDownDecode_Q4_KMatchesSequential)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertDownDecodeFormatMatch("Q4_K", [](size_t rows, size_t cols)
                                          {
                                              auto tensor = TestTensorFactory::createQ4_KRandom({rows, cols});
                                              injectNonZeroKQuantMins(tensor.get());
                                              return tensor;
                                          });
}

TEST(Test__ROCmMoEKernel, GroupedExpertDownDecode_Q6_KMatchesSequential)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertDownDecodeFormatMatch("Q6_K", [](size_t rows, size_t cols)
                                          { return TestTensorFactory::createQ6_KRandom({rows, cols}); });
}

template <typename WeightFactory>
void runGroupedExpertGateUpDecodeFormatMatch(const char *label, WeightFactory create_weights)
{
    const int d_model = 256;
    const int intermediate = 128;
    const int num_experts = 8;
    const int num_active = 4;
    const int expert_ids[num_active] = {2, 7, 3, 5};
    const DeviceId device = DeviceId::rocm(0);

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    std::vector<std::unique_ptr<TensorBase>> gate_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> up_weight_tensors;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> gate_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> up_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> gate_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> up_kernels;
    gate_weight_tensors.reserve(num_experts);
    up_weight_tensors.reserve(num_experts);
    gate_packed_weights.reserve(num_experts);
    up_packed_weights.reserve(num_experts);
    gate_kernels.reserve(num_experts);
    up_kernels.reserve(num_experts);

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto gate_weights = create_weights(static_cast<size_t>(intermediate), static_cast<size_t>(d_model));
        auto up_weights = create_weights(static_cast<size_t>(intermediate), static_cast<size_t>(d_model));
        auto gate_packed = std::make_unique<rocm::ROCmPackedWeights>();
        auto up_packed = std::make_unique<rocm::ROCmPackedWeights>();
        ASSERT_TRUE(rocm::packWeightsToROCm(gate_weights.get(), *gate_packed))
            << "packWeightsToROCm failed for " << label << " gate expert " << expert_id;
        ASSERT_TRUE(rocm::packWeightsToROCm(up_weights.get(), *up_packed))
            << "packWeightsToROCm failed for " << label << " up expert " << expert_id;
        ASSERT_FALSE(gate_packed->native_vnni_payload.empty());
        ASSERT_FALSE(up_packed->native_vnni_payload.empty());

        auto gate_kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(gate_packed.get(), 0);
        auto up_kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(up_packed.get(), 0);
        gate_weight_tensors.push_back(std::move(gate_weights));
        up_weight_tensors.push_back(std::move(up_weights));
        gate_packed_weights.push_back(std::move(gate_packed));
        up_packed_weights.push_back(std::move(up_packed));
        gate_kernels.push_back(std::move(gate_kernel));
        up_kernels.push_back(std::move(up_kernel));
    }

    auto reqs = gate_kernels[0]->getWorkspaceRequirements(1, intermediate, d_model);
    auto workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
    ASSERT_TRUE(workspace->allocate(reqs));
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        gate_kernels[expert_id]->bindWorkspace(workspace.get());
        up_kernels[expert_id]->bindWorkspace(workspace.get());
        gate_kernels[expert_id]->prepareWeights();
        up_kernels[expert_id]->prepareWeights();
    }

    auto input = TestTensorFactory::createFP32Random(
        {1, static_cast<size_t>(d_model)}, -2.0f, 2.0f, 900);
    ASSERT_TRUE(input->ensureOnDevice(device));

    std::vector<std::shared_ptr<FP32Tensor>> seq_gate_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> seq_up_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> grouped_gate_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> grouped_up_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> device_gate_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> device_up_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> runtime_gate_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> runtime_up_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> kpart_gate_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> kpart_up_tensors;
    seq_gate_tensors.reserve(num_active);
    seq_up_tensors.reserve(num_active);
    grouped_gate_tensors.reserve(num_active);
    grouped_up_tensors.reserve(num_active);
    device_gate_tensors.reserve(num_active);
    device_up_tensors.reserve(num_active);
    runtime_gate_tensors.reserve(num_active);
    runtime_up_tensors.reserve(num_active);
    kpart_gate_tensors.reserve(num_active);
    kpart_up_tensors.reserve(num_active);

    std::vector<ITensorGemm::TensorProjectionDesc> projections;
    projections.reserve(num_active * 2);
    for (int i = 0; i < num_active; ++i)
    {
        auto seq_gate = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto seq_up = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto grouped_gate = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto grouped_up = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto device_gate = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto device_up = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto runtime_gate = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto runtime_up = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto kpart_gate = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto kpart_up = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        ASSERT_TRUE(seq_gate->ensureOnDevice(device));
        ASSERT_TRUE(seq_up->ensureOnDevice(device));
        ASSERT_TRUE(grouped_gate->ensureOnDevice(device));
        ASSERT_TRUE(grouped_up->ensureOnDevice(device));
        ASSERT_TRUE(device_gate->ensureOnDevice(device));
        ASSERT_TRUE(device_up->ensureOnDevice(device));
        ASSERT_TRUE(runtime_gate->ensureOnDevice(device));
        ASSERT_TRUE(runtime_up->ensureOnDevice(device));
        ASSERT_TRUE(kpart_gate->ensureOnDevice(device));
        ASSERT_TRUE(kpart_up->ensureOnDevice(device));

        const int expert_id = expert_ids[i];
        projections.emplace_back(gate_kernels[expert_id].get(), seq_gate.get(), intermediate, nullptr, "gate");
        projections.emplace_back(up_kernels[expert_id].get(), seq_up.get(), intermediate, nullptr, "up");

        seq_gate_tensors.push_back(std::move(seq_gate));
        seq_up_tensors.push_back(std::move(seq_up));
        grouped_gate_tensors.push_back(std::move(grouped_gate));
        grouped_up_tensors.push_back(std::move(grouped_up));
        device_gate_tensors.push_back(std::move(device_gate));
        device_up_tensors.push_back(std::move(device_up));
        runtime_gate_tensors.push_back(std::move(runtime_gate));
        runtime_up_tensors.push_back(std::move(runtime_up));
        kpart_gate_tensors.push_back(std::move(kpart_gate));
        kpart_up_tensors.push_back(std::move(kpart_up));
    }

    ASSERT_TRUE(gate_kernels[expert_ids[0]]->multiply_fused_tensor(
        input.get(), projections, 1, d_model));
    (void)hipDeviceSynchronize();

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        ASSERT_TRUE(gate_kernels[expert_id]->exportNativeVNNIMatrixDesc(gate_descs[expert_id]));
        ASSERT_TRUE(up_kernels[expert_id]->exportNativeVNNIMatrixDesc(up_descs[expert_id]));
        ASSERT_EQ(gate_descs[expert_id].n, intermediate);
        ASSERT_EQ(gate_descs[expert_id].k, d_model);
        ASSERT_EQ(up_descs[expert_id].n, intermediate);
        ASSERT_EQ(up_descs[expert_id].k, d_model);
    }

    ITensor *grouped_gate_ptrs[num_active] = {};
    ITensor *grouped_up_ptrs[num_active] = {};
    ITensor *device_gate_ptrs[num_active] = {};
    ITensor *device_up_ptrs[num_active] = {};
    ITensor *runtime_gate_ptrs[num_active] = {};
    ITensor *runtime_up_ptrs[num_active] = {};
    ITensor *kpart_gate_ptrs[num_active] = {};
    ITensor *kpart_up_ptrs[num_active] = {};
    for (int i = 0; i < num_active; ++i)
    {
        grouped_gate_ptrs[i] = grouped_gate_tensors[i].get();
        grouped_up_ptrs[i] = grouped_up_tensors[i].get();
        device_gate_ptrs[i] = device_gate_tensors[i].get();
        device_up_ptrs[i] = device_up_tensors[i].get();
        runtime_gate_ptrs[i] = runtime_gate_tensors[i].get();
        runtime_up_ptrs[i] = runtime_up_tensors[i].get();
        kpart_gate_ptrs[i] = kpart_gate_tensors[i].get();
        kpart_up_ptrs[i] = kpart_up_tensors[i].get();
    }

    const int table_id = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(table_id, 0);

    ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromTable(
        input.get(), expert_ids, table_id, num_active,
        grouped_gate_ptrs, grouped_up_ptrs, d_model, intermediate));

    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(num_active)});
    for (int i = 0; i < num_active; ++i)
        routing_indices->mutable_data()[i] = static_cast<float>(expert_ids[i]);
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));

    ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromRouting(
        input.get(), routing_indices.get(), table_id, num_active,
        device_gate_ptrs, device_up_ptrs, d_model, intermediate));

    DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, num_active);
    populateRuntimeDescriptors(host_runtime, &gate_descs, &up_descs, nullptr);
    for (int i = 0; i < num_active; ++i)
        host_runtime.topk_expert_ids[i] = expert_ids[i];
    DeviceMoELayerRuntime *device_runtime = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_runtime), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(device_runtime, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    {
        ScopedROCmEnvOverride disable_kpart("LLAMINAR_ROCM_MOE_GATEUP_KPART_DECODE", "0");
        ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromRuntime(
            device_runtime, input.get(), table_id, num_active,
            runtime_gate_ptrs, runtime_up_ptrs, d_model, intermediate));
    }

    constexpr const char *kKPartCounts[] = {"2", "4", "8", "16"};
    for (const char *kparts : kKPartCounts)
    {
        ScopedROCmEnvOverride deterministic_off("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride enable_kpart("LLAMINAR_ROCM_MOE_GATEUP_KPART_DECODE", "1");
        ScopedROCmEnvOverride set_kparts("LLAMINAR_ROCM_MOE_GATEUP_KPARTS", kparts);
        ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromRuntime(
            device_runtime, input.get(), table_id, num_active,
            kpart_gate_ptrs, kpart_up_ptrs, d_model, intermediate))
            << label << " K-partition runtime gate/up failed for kparts=" << kparts;
        (void)hipDeviceSynchronize();

        for (int i = 0; i < num_active; ++i)
        {
            runtime_gate_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            runtime_up_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            kpart_gate_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            kpart_up_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

            const float *runtime_gate = runtime_gate_tensors[i]->data();
            const float *runtime_up = runtime_up_tensors[i]->data();
            const float *kpart_gate = kpart_gate_tensors[i]->data();
            const float *kpart_up = kpart_up_tensors[i]->data();

            const double kpart_gate_cosine = cosineSimilarity(kpart_gate, runtime_gate, intermediate);
            const double kpart_up_cosine = cosineSimilarity(kpart_up, runtime_up, intermediate);
            const double kpart_gate_l2 = relativeL2Error(kpart_gate, runtime_gate, intermediate);
            const double kpart_up_l2 = relativeL2Error(kpart_up, runtime_up, intermediate);
            const double kpart_gate_max_abs = maxAbsDiff(kpart_gate, runtime_gate, intermediate);
            const double kpart_up_max_abs = maxAbsDiff(kpart_up, runtime_up, intermediate);

            EXPECT_GE(kpart_gate_cosine, 0.99999)
                << label << " kpart gate cosine mismatch for active slot " << i << " kparts=" << kparts;
            EXPECT_GE(kpart_up_cosine, 0.99999)
                << label << " kpart up cosine mismatch for active slot " << i << " kparts=" << kparts;
            EXPECT_LE(kpart_gate_l2, 0.0015)
                << label << " kpart gate relative L2 too high for active slot " << i << " kparts=" << kparts;
            EXPECT_LE(kpart_up_l2, 0.0015)
                << label << " kpart up relative L2 too high for active slot " << i << " kparts=" << kparts;
            EXPECT_LE(kpart_gate_max_abs, 0.03)
                << label << " kpart gate max abs diff too high for active slot " << i << " kparts=" << kparts;
            EXPECT_LE(kpart_up_max_abs, 0.03)
                << label << " kpart up max abs diff too high for active slot " << i << " kparts=" << kparts;
        }
    }
    (void)hipDeviceSynchronize();
    ASSERT_EQ(hipFree(device_runtime), hipSuccess);

    for (int i = 0; i < num_active; ++i)
    {
        seq_gate_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        seq_up_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        grouped_gate_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        grouped_up_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        device_gate_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        device_up_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        runtime_gate_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        runtime_up_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        kpart_gate_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        kpart_up_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        const float *seq_gate = seq_gate_tensors[i]->data();
        const float *seq_up = seq_up_tensors[i]->data();
        const float *grouped_gate = grouped_gate_tensors[i]->data();
        const float *grouped_up = grouped_up_tensors[i]->data();
        const float *device_gate = device_gate_tensors[i]->data();
        const float *device_up = device_up_tensors[i]->data();
        const float *runtime_gate = runtime_gate_tensors[i]->data();
        const float *runtime_up = runtime_up_tensors[i]->data();
        const float *kpart_gate = kpart_gate_tensors[i]->data();
        const float *kpart_up = kpart_up_tensors[i]->data();

        ASSERT_FALSE(hasNaNOrInf(grouped_gate, intermediate));
        ASSERT_FALSE(hasNaNOrInf(grouped_up, intermediate));
        ASSERT_FALSE(hasNaNOrInf(device_gate, intermediate));
        ASSERT_FALSE(hasNaNOrInf(device_up, intermediate));
        ASSERT_FALSE(hasNaNOrInf(runtime_gate, intermediate));
        ASSERT_FALSE(hasNaNOrInf(runtime_up, intermediate));
        ASSERT_FALSE(hasNaNOrInf(kpart_gate, intermediate));
        ASSERT_FALSE(hasNaNOrInf(kpart_up, intermediate));

        const double gate_cosine = cosineSimilarity(grouped_gate, seq_gate, intermediate);
        const double up_cosine = cosineSimilarity(grouped_up, seq_up, intermediate);
        const double gate_l2 = relativeL2Error(grouped_gate, seq_gate, intermediate);
        const double up_l2 = relativeL2Error(grouped_up, seq_up, intermediate);
        const double device_gate_cosine = cosineSimilarity(device_gate, seq_gate, intermediate);
        const double device_up_cosine = cosineSimilarity(device_up, seq_up, intermediate);
        const double device_gate_l2 = relativeL2Error(device_gate, seq_gate, intermediate);
        const double device_up_l2 = relativeL2Error(device_up, seq_up, intermediate);
        const double runtime_gate_cosine = cosineSimilarity(runtime_gate, seq_gate, intermediate);
        const double runtime_up_cosine = cosineSimilarity(runtime_up, seq_up, intermediate);
        const double runtime_gate_l2 = relativeL2Error(runtime_gate, seq_gate, intermediate);
        const double runtime_up_l2 = relativeL2Error(runtime_up, seq_up, intermediate);
        const double kpart_gate_cosine = cosineSimilarity(kpart_gate, runtime_gate, intermediate);
        const double kpart_up_cosine = cosineSimilarity(kpart_up, runtime_up, intermediate);
        const double kpart_gate_l2 = relativeL2Error(kpart_gate, runtime_gate, intermediate);
        const double kpart_up_l2 = relativeL2Error(kpart_up, runtime_up, intermediate);

        expectStrictVerifierSimilarity(
            (std::string(label) + " grouped gate output must match sequential").c_str(),
            grouped_gate,
            seq_gate,
            static_cast<size_t>(intermediate),
            static_cast<size_t>(intermediate));
        expectStrictVerifierSimilarity(
            (std::string(label) + " grouped up output must match sequential").c_str(),
            grouped_up,
            seq_up,
            static_cast<size_t>(intermediate),
            static_cast<size_t>(intermediate));
        expectStrictVerifierSimilarity(
            (std::string(label) + " device-routed grouped gate output must match sequential").c_str(),
            device_gate,
            seq_gate,
            static_cast<size_t>(intermediate),
            static_cast<size_t>(intermediate));
        expectStrictVerifierSimilarity(
            (std::string(label) + " device-routed grouped up output must match sequential").c_str(),
            device_up,
            seq_up,
            static_cast<size_t>(intermediate),
            static_cast<size_t>(intermediate));
        expectStrictVerifierSimilarity(
            (std::string(label) + " runtime grouped gate output must match sequential").c_str(),
            runtime_gate,
            seq_gate,
            static_cast<size_t>(intermediate),
            static_cast<size_t>(intermediate));
        expectStrictVerifierSimilarity(
            (std::string(label) + " runtime grouped up output must match sequential").c_str(),
            runtime_up,
            seq_up,
            static_cast<size_t>(intermediate),
            static_cast<size_t>(intermediate));
        EXPECT_GE(kpart_gate_cosine, 0.99999)
            << label << " kpart grouped gate cosine mismatch vs runtime for active slot " << i;
        EXPECT_GE(kpart_up_cosine, 0.99999)
            << label << " kpart grouped up cosine mismatch vs runtime for active slot " << i;
        EXPECT_LE(kpart_gate_l2, 0.0015)
            << label << " kpart grouped gate relative L2 too high vs runtime for active slot " << i;
        EXPECT_LE(kpart_up_l2, 0.0015)
            << label << " kpart grouped up relative L2 too high vs runtime for active slot " << i;

        std::cout << "[GroupedExpertGateUpDecode_" << label
                  << "] slot=" << i
                  << " gate_cosine=" << std::fixed << std::setprecision(6) << gate_cosine
                  << " up_cosine=" << up_cosine
                  << " gate_l2=" << gate_l2
                  << " up_l2=" << up_l2
                  << " device_gate_cosine=" << device_gate_cosine
                  << " device_up_cosine=" << device_up_cosine
                  << " device_gate_l2=" << device_gate_l2
                  << " device_up_l2=" << device_up_l2
                  << " runtime_gate_cosine=" << runtime_gate_cosine
                  << " runtime_up_cosine=" << runtime_up_cosine
                  << " runtime_gate_l2=" << runtime_gate_l2
                  << " runtime_up_l2=" << runtime_up_l2
                  << " kpart_gate_cosine=" << kpart_gate_cosine
                  << " kpart_up_cosine=" << kpart_up_cosine
                  << " kpart_gate_l2=" << kpart_gate_l2
                  << " kpart_up_l2=" << kpart_up_l2 << std::endl;
    }
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_Q4_0KPartMatchesRuntime)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("Q4_0", [](size_t rows, size_t cols)
                                            { return TestTensorFactory::createQ4_0Random({rows, cols}); });
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_IQ4_NLKPartMatchesRuntime)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("IQ4_NL", [](size_t rows, size_t cols)
                                            { return TestTensorFactory::createIQ4_NLRandom({rows, cols}); });
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_Q4_1KPartMatchesRuntime)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("Q4_1", [](size_t rows, size_t cols)
                                            { return TestTensorFactory::createQ4_1Random({rows, cols}); });
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_Q5_0KPartMatchesRuntime)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("Q5_0", [](size_t rows, size_t cols)
                                            { return TestTensorFactory::createQ5_0Random({rows, cols}); });
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_Q5_1KPartMatchesRuntime)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("Q5_1", [](size_t rows, size_t cols)
                                            { return TestTensorFactory::createQ5_1Random({rows, cols}); });
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_Q5_KMatchesFusedProjection)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("Q5_K", [](size_t rows, size_t cols)
                                            {
                                                auto tensor = TestTensorFactory::createQ5_KRandom({rows, cols});
                                                injectNonZeroKQuantMins(tensor.get());
                                                return tensor;
                                            });
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_Q4_KMatchesFusedProjection)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("Q4_K", [](size_t rows, size_t cols)
                                            {
                                                auto tensor = TestTensorFactory::createQ4_KRandom({rows, cols});
                                                injectNonZeroKQuantMins(tensor.get());
                                                return tensor;
                                            });
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_Q6_KMatchesFusedProjection)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("Q6_K", [](size_t rows, size_t cols)
                                            { return TestTensorFactory::createQ6_KRandom({rows, cols}); });
}

TEST(Test__ROCmMoEKernel, GroupedSharedExpertDecodeCapturesWithStablePointerCache)
{
    SKIP_IF_NO_ROCM();

    constexpr int d_model = 128;
    constexpr int intermediate = 128;
    constexpr int num_experts = 1;
    constexpr int num_active = 1;
    constexpr int expert_ids[num_active] = {0};
    constexpr float expert_weights[num_active] = {1.0f};
    const DeviceId device = DeviceId::rocm(0);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    ROCmMoEKernel moe_kernel(0);
    static_cast<ITensorKernel &>(moe_kernel).setGPUStream(stream);
    auto moe_workspace = bindDefaultMoEWorkspace(
        moe_kernel,
        /*max_seq_len=*/1,
        d_model,
        intermediate,
        num_experts,
        /*top_k=*/1);

    auto make_kernel = [&](auto weights)
    {
        auto packed = std::make_unique<rocm::ROCmPackedWeights>();
        EXPECT_TRUE(rocm::packWeightsToROCm(weights.get(), *packed));
        auto kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(packed.get(), 0);
        static_cast<ITensorKernel *>(kernel.get())->setGPUStream(stream);
        return std::tuple{
            std::move(weights),
            std::move(packed),
            std::move(kernel)};
    };

    auto [gate_weights, gate_packed, gate_kernel] = make_kernel(
        TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}));
    auto [up_weights, up_packed, up_kernel] = make_kernel(
        TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}));
    auto [down_weights, down_packed, down_kernel] = make_kernel(
        TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}));

    WorkspaceRequirements gemm_reqs;
    gemm_reqs.merge(gate_kernel->getWorkspaceRequirements(1, intermediate, d_model));
    gemm_reqs.merge(up_kernel->getWorkspaceRequirements(1, intermediate, d_model));
    gemm_reqs.merge(down_kernel->getWorkspaceRequirements(1, d_model, intermediate));
    auto gemm_workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
    ASSERT_TRUE(gemm_workspace->allocate(gemm_reqs));
    for (auto *kernel : {gate_kernel.get(), up_kernel.get(), down_kernel.get()})
    {
        kernel->bindWorkspace(gemm_workspace.get());
        kernel->prepareWeights();
    }

    DeviceNativeVNNIMatrixDesc gate_desc{};
    DeviceNativeVNNIMatrixDesc up_desc{};
    DeviceNativeVNNIMatrixDesc down_desc{};
    ASSERT_TRUE(gate_kernel->exportNativeVNNIMatrixDesc(gate_desc));
    ASSERT_TRUE(up_kernel->exportNativeVNNIMatrixDesc(up_desc));
    ASSERT_TRUE(down_kernel->exportNativeVNNIMatrixDesc(down_desc));

    const int gateup_table = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        &gate_desc, &up_desc, num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        &down_desc, num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    auto hidden = TestTensorFactory::createFP32Random(
        {1, static_cast<size_t>(d_model)}, -0.5f, 0.5f, 5151);
    auto gate = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
    auto up = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
    auto output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
    ASSERT_TRUE(gate->ensureOnDevice(device, stream));
    ASSERT_TRUE(up->ensureOnDevice(device, stream));
    ASSERT_TRUE(output->ensureOnDevice(device, stream));

    std::array<ITensor *, num_active> gate_outputs = {gate.get()};
    std::array<ITensor *, num_active> up_outputs = {up.get()};

    ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromTable(
        hidden.get(), expert_ids, gateup_table, num_active,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate));
    ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromTable(
        gate_outputs.data(), up_outputs.data(), expert_ids, expert_weights,
        down_table, num_active, output.get(), d_model, intermediate));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
    const bool captured_gateup = moe_kernel.groupedExpertGateUpDecodeFromTable(
        hidden.get(), expert_ids, gateup_table, num_active,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate);
    const bool captured_down = moe_kernel.groupedExpertDownDecodeFromTable(
        gate_outputs.data(), up_outputs.data(), expert_ids, expert_weights,
        down_table, num_active, output.get(), d_model, intermediate);
    const hipError_t capture_status = hipStreamEndCapture(stream, &graph);
    EXPECT_TRUE(captured_gateup);
    EXPECT_TRUE(captured_down);
    ASSERT_EQ(capture_status, hipSuccess) << hipGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);

    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
    ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
    ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    ASSERT_FALSE(hasNaNOrInf(output->data(), static_cast<size_t>(d_model)));

    EXPECT_EQ(hipGraphExecDestroy(exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, RuntimeDecodeGraphReplayReadsDeviceDescriptorsWithoutTableRefresh)
{
    SKIP_IF_NO_ROCM();

    ScopedROCmEnvOverride disable_gateup_kpart("LLAMINAR_ROCM_MOE_GATEUP_KPART_DECODE", "0");
    ScopedROCmEnvOverride disable_parallel_down("LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE", "0");

    constexpr int num_experts = 2;
    constexpr int top_k = 1;
    constexpr int d_model = 128;
    constexpr int intermediate = 128;
    const DeviceId device = DeviceId::rocm(0);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    ROCmMoEKernel moe_kernel(0);
    static_cast<ITensorKernel &>(moe_kernel).setGPUStream(stream);
    auto moe_workspace = bindDefaultMoEWorkspace(
        moe_kernel,
        /*max_seq_len=*/1,
        d_model,
        intermediate,
        num_experts,
        top_k);

    std::vector<std::unique_ptr<TensorBase>> gate_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> up_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> down_weight_tensors;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> gate_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> up_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> down_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> gate_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> up_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> down_kernels;

    auto add_kernel = [&](std::vector<std::unique_ptr<TensorBase>> &weights,
                          std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> &packed,
                          std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> &kernels,
                          int rows,
                          int cols,
                          int seed)
    {
        auto weight = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(rows), static_cast<size_t>(cols)},
            static_cast<unsigned>(seed));
        auto packed_weight = std::make_unique<rocm::ROCmPackedWeights>();
        EXPECT_TRUE(rocm::packWeightsToROCm(weight.get(), *packed_weight))
            << "packWeightsToROCm failed for seed " << seed;
        auto kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(packed_weight.get(), 0);
        static_cast<ITensorKernel *>(kernel.get())->setGPUStream(stream);
        weights.push_back(std::move(weight));
        packed.push_back(std::move(packed_weight));
        kernels.push_back(std::move(kernel));
    };

    for (int expert = 0; expert < num_experts; ++expert)
    {
        add_kernel(gate_weight_tensors, gate_packed_weights, gate_kernels,
                   intermediate, d_model, 9100 + expert);
        add_kernel(up_weight_tensors, up_packed_weights, up_kernels,
                   intermediate, d_model, 9200 + expert);
        add_kernel(down_weight_tensors, down_packed_weights, down_kernels,
                   d_model, intermediate, 9300 + expert);
    }

    WorkspaceRequirements gemm_reqs;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        gemm_reqs.merge(gate_kernels[expert]->getWorkspaceRequirements(1, intermediate, d_model));
        gemm_reqs.merge(up_kernels[expert]->getWorkspaceRequirements(1, intermediate, d_model));
        gemm_reqs.merge(down_kernels[expert]->getWorkspaceRequirements(1, d_model, intermediate));
    }
    auto gemm_workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
    ASSERT_TRUE(gemm_workspace->allocate(gemm_reqs));
    for (int expert = 0; expert < num_experts; ++expert)
    {
        gate_kernels[expert]->bindWorkspace(gemm_workspace.get());
        up_kernels[expert]->bindWorkspace(gemm_workspace.get());
        down_kernels[expert]->bindWorkspace(gemm_workspace.get());
        gate_kernels[expert]->prepareWeights();
        up_kernels[expert]->prepareWeights();
        down_kernels[expert]->prepareWeights();
    }

    std::array<DeviceNativeVNNIMatrixDesc, num_experts> gate_descs = {};
    std::array<DeviceNativeVNNIMatrixDesc, num_experts> up_descs = {};
    std::array<DeviceNativeVNNIMatrixDesc, num_experts> down_descs = {};
    for (int expert = 0; expert < num_experts; ++expert)
    {
        ASSERT_TRUE(gate_kernels[expert]->exportNativeVNNIMatrixDesc(gate_descs[expert]));
        ASSERT_TRUE(up_kernels[expert]->exportNativeVNNIMatrixDesc(up_descs[expert]));
        ASSERT_TRUE(down_kernels[expert]->exportNativeVNNIMatrixDesc(down_descs[expert]));
        ASSERT_EQ(gate_descs[expert].n, intermediate);
        ASSERT_EQ(gate_descs[expert].k, d_model);
        ASSERT_EQ(up_descs[expert].n, intermediate);
        ASSERT_EQ(up_descs[expert].k, d_model);
        ASSERT_EQ(down_descs[expert].n, d_model);
        ASSERT_EQ(down_descs[expert].k, intermediate);
        ASSERT_EQ(gate_descs[expert].codebook_id, 0);
        ASSERT_EQ(up_descs[expert].codebook_id, 0);
        ASSERT_EQ(down_descs[expert].codebook_id, 0);
    }

    const std::array<DeviceNativeVNNIMatrixDesc, num_experts> stale_gate_descs = {
        gate_descs[0], gate_descs[0]};
    const std::array<DeviceNativeVNNIMatrixDesc, num_experts> stale_up_descs = {
        up_descs[0], up_descs[0]};
    const std::array<DeviceNativeVNNIMatrixDesc, num_experts> stale_down_descs = {
        down_descs[0], down_descs[0]};

    const int stale_gateup_table = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        stale_gate_descs.data(), stale_up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(stale_gateup_table, 0);
    const int stale_down_table = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        stale_down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(stale_down_table, 0);
    const int fresh_gateup_table = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(fresh_gateup_table, 0);
    const int fresh_down_table = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(fresh_down_table, 0);

    auto hidden = TestTensorFactory::createFP32Random(
        {1, static_cast<size_t>(d_model)}, -0.5f, 0.5f, 9411);
    ASSERT_TRUE(hidden->ensureOnDevice(device, stream));

    constexpr int expert_ids[top_k] = {1};
    constexpr float expert_weights[top_k] = {1.0f};
    auto run_table_reference = [&](int gateup_table, int down_table, ITensor *output)
    {
        auto gate = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto up = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        ASSERT_TRUE(gate->ensureOnDevice(device, stream));
        ASSERT_TRUE(up->ensureOnDevice(device, stream));
        auto *output_base = dynamic_cast<TensorBase *>(output);
        ASSERT_NE(output_base, nullptr);
        ASSERT_TRUE(output_base->ensureOnDevice(device, stream));
        std::array<ITensor *, top_k> gate_outputs = {gate.get()};
        std::array<ITensor *, top_k> up_outputs = {up.get()};

        ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromTable(
            hidden.get(), expert_ids, gateup_table, top_k,
            gate_outputs.data(), up_outputs.data(), d_model, intermediate));
        ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromTable(
            gate_outputs.data(), up_outputs.data(), expert_ids, expert_weights,
            down_table, top_k, output, d_model, intermediate));
    };

    auto stale_reference = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto fresh_reference = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    run_table_reference(stale_gateup_table, stale_down_table, stale_reference.get());
    run_table_reference(fresh_gateup_table, fresh_down_table, fresh_reference.get());
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    stale_reference->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    fresh_reference->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    ASSERT_GT(relativeL2Error(stale_reference->data(), fresh_reference->data(), static_cast<size_t>(d_model)), 0.01)
        << "descriptor-table A/B reference must be distinguishable for this regression";

    auto make_runtime = [&](const DeviceNativeVNNIMatrixDesc &gate,
                            const DeviceNativeVNNIMatrixDesc &up,
                            const DeviceNativeVNNIMatrixDesc &down)
    {
        DeviceMoELayerRuntime runtime{};
        runtime.active_bank = 0;
        runtime.active_epoch = 1;
        runtime.expert_count = num_experts;
        runtime.top_k = top_k;
        runtime.participant_id = 0;
        runtime.participant_count = 1;
        runtime.topk_expert_ids[0] = 1;
        runtime.topk_weights[0] = 1.0f;

        auto &bank = runtime.banks[0];
        bank.epoch = 1;
        bank.expert_count = num_experts;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            bank.local_compute_mask[expert] = 1;
            auto &desc = bank.experts[expert];
            desc.logical_expert_id = expert;
            desc.owner_participant = 0;
            desc.local_slot = expert;
            desc.flags = toMoEExpertFlags(
                DeviceMoEExpertFlags::Valid |
                DeviceMoEExpertFlags::Resident |
                DeviceMoEExpertFlags::LocalCompute);
            desc.gate = expert == 0 ? gate_descs[0] : gate;
            desc.up = expert == 0 ? up_descs[0] : up;
            desc.down = expert == 0 ? down_descs[0] : down;
        }
        return runtime;
    };

    const auto runtime_stale = make_runtime(gate_descs[0], up_descs[0], down_descs[0]);
    const auto runtime_fresh = make_runtime(gate_descs[1], up_descs[1], down_descs[1]);

    DeviceMoELayerRuntime *device_runtime = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_runtime), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_runtime, &runtime_stale, sizeof(runtime_stale),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);

    auto runtime_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    ASSERT_TRUE(runtime_output->ensureOnDevice(device, stream));
    ASSERT_TRUE(moe_kernel.groupedExpertDecodeFromRuntime(
        device_runtime, hidden.get(), stale_gateup_table, stale_down_table, top_k,
        runtime_output.get(), d_model, intermediate,
        MoEDecodeDescriptorSource::RuntimePlacementTable));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
    const bool captured = moe_kernel.groupedExpertDecodeFromRuntime(
        device_runtime, hidden.get(), stale_gateup_table, stale_down_table, top_k,
        runtime_output.get(), d_model, intermediate,
        MoEDecodeDescriptorSource::RuntimePlacementTable);
    const hipError_t capture_status = hipStreamEndCapture(stream, &graph);
    EXPECT_TRUE(captured);
    ASSERT_EQ(capture_status, hipSuccess) << hipGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);

    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_runtime, &runtime_fresh, sizeof(runtime_fresh),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    runtime_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    expectStrictVerifierSimilarity(
        "ROCm graph-replayed runtime decode must follow device runtime descriptors",
        runtime_output->data(),
        fresh_reference->data(),
        static_cast<size_t>(d_model),
        static_cast<size_t>(d_model));
    EXPECT_GT(relativeL2Error(runtime_output->data(), stale_reference->data(), static_cast<size_t>(d_model)), 0.01)
        << "graph replay followed the stale descriptor table instead of device runtime descriptors";

    EXPECT_EQ(hipGraphExecDestroy(exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipFree(device_runtime), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, RuntimeGroupedDecodeFusedPathMatchesTwoStepAndCaptures)
{
    SKIP_IF_NO_ROCM();

    ScopedEnvOverride perf_stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
    ScopedROCmEnvOverride enable_gateup_kpart("LLAMINAR_ROCM_MOE_GATEUP_KPART_DECODE", "1");
    ScopedROCmEnvOverride set_gateup_kparts("LLAMINAR_ROCM_MOE_GATEUP_KPARTS", "4");
    ScopedROCmEnvOverride disable_parallel_down("LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE", "0");
    ScopedROCmEnvOverride disable_fused_gateup_swiglu("LLAMINAR_ROCM_MOE_GATEUP_SWIGLU_QUANT_FUSED", "0");

    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    constexpr int num_experts = 8;
    constexpr int top_k = 8;
    const DeviceId device = DeviceId::rocm(0);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    ASSERT_TRUE(llaminar2::rocm::ensureIQGridTablesInitialized(0));

    ROCmMoEKernel moe_kernel(0);
    static_cast<ITensorKernel &>(moe_kernel).setGPUStream(stream);
    auto moe_workspace = bindDefaultMoEWorkspace(
        moe_kernel,
        /*max_seq_len=*/1,
        d_model,
        intermediate,
        num_experts,
        top_k);

    auto make_kernel = [&](auto weights)
    {
        auto packed = std::make_unique<rocm::ROCmPackedWeights>();
        EXPECT_TRUE(rocm::packWeightsToROCm(weights.get(), *packed));
        auto kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(packed.get(), 0);
        static_cast<ITensorKernel *>(kernel.get())->setGPUStream(stream);
        return std::tuple{
            std::move(weights),
            std::move(packed),
            std::move(kernel)};
    };

    std::vector<std::unique_ptr<TensorBase>> gate_weights;
    std::vector<std::unique_ptr<TensorBase>> up_weights;
    std::vector<std::unique_ptr<TensorBase>> down_weights;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> gate_packed;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> up_packed;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> down_packed;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> gate_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> up_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> down_kernels;
    gate_weights.reserve(num_experts);
    up_weights.reserve(num_experts);
    down_weights.reserve(num_experts);
    gate_packed.reserve(num_experts);
    up_packed.reserve(num_experts);
    down_packed.reserve(num_experts);
    gate_kernels.reserve(num_experts);
    up_kernels.reserve(num_experts);
    down_kernels.reserve(num_experts);

    WorkspaceRequirements gemm_reqs;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        auto [gate_w, gate_p, gate_k] = make_kernel(
            TestTensorFactory::createQ4_0Random(
                {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}));
        auto [up_w, up_p, up_k] = make_kernel(
            TestTensorFactory::createQ4_0Random(
                {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}));
        auto [down_w, down_p, down_k] = make_kernel(
            TestTensorFactory::createQ4_0Random(
                {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}));

        gemm_reqs.merge(gate_k->getWorkspaceRequirements(1, intermediate, d_model));
        gemm_reqs.merge(up_k->getWorkspaceRequirements(1, intermediate, d_model));
        gemm_reqs.merge(down_k->getWorkspaceRequirements(1, d_model, intermediate));

        gate_weights.push_back(std::move(gate_w));
        up_weights.push_back(std::move(up_w));
        down_weights.push_back(std::move(down_w));
        gate_packed.push_back(std::move(gate_p));
        up_packed.push_back(std::move(up_p));
        down_packed.push_back(std::move(down_p));
        gate_kernels.push_back(std::move(gate_k));
        up_kernels.push_back(std::move(up_k));
        down_kernels.push_back(std::move(down_k));
    }

    auto gemm_workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
    ASSERT_TRUE(gemm_workspace->allocate(gemm_reqs));
    for (int expert = 0; expert < num_experts; ++expert)
    {
        gate_kernels[expert]->bindWorkspace(gemm_workspace.get());
        up_kernels[expert]->bindWorkspace(gemm_workspace.get());
        down_kernels[expert]->bindWorkspace(gemm_workspace.get());
        gate_kernels[expert]->prepareWeights();
        up_kernels[expert]->prepareWeights();
        down_kernels[expert]->prepareWeights();
    }

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> down_descs(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        ASSERT_TRUE(gate_kernels[expert]->exportNativeVNNIMatrixDesc(gate_descs[expert]));
        ASSERT_TRUE(up_kernels[expert]->exportNativeVNNIMatrixDesc(up_descs[expert]));
        ASSERT_TRUE(down_kernels[expert]->exportNativeVNNIMatrixDesc(down_descs[expert]));
    }

    const int gateup_table = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    auto hidden = TestTensorFactory::createFP32Random(
        {1, static_cast<size_t>(d_model)}, -0.5f, 0.5f, 8181);
    auto two_step_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto fused_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto routing_tensor_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
    ASSERT_TRUE(two_step_output->ensureOnDevice(device, stream));
    ASSERT_TRUE(fused_output->ensureOnDevice(device, stream));
    ASSERT_TRUE(routing_tensor_output->ensureOnDevice(device, stream));

    std::array<std::shared_ptr<FP32Tensor>, top_k> gate_tensors;
    std::array<std::shared_ptr<FP32Tensor>, top_k> up_tensors;
    std::array<std::shared_ptr<FP32Tensor>, top_k> routing_gate_tensors;
    std::array<std::shared_ptr<FP32Tensor>, top_k> routing_up_tensors;
    std::array<ITensor *, top_k> gate_ptrs = {};
    std::array<ITensor *, top_k> up_ptrs = {};
    std::array<ITensor *, top_k> routing_gate_ptrs = {};
    std::array<ITensor *, top_k> routing_up_ptrs = {};
    for (int slot = 0; slot < top_k; ++slot)
    {
        gate_tensors[slot] = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        up_tensors[slot] = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        routing_gate_tensors[slot] = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        routing_up_tensors[slot] = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        ASSERT_TRUE(gate_tensors[slot]->ensureOnDevice(device, stream));
        ASSERT_TRUE(up_tensors[slot]->ensureOnDevice(device, stream));
        ASSERT_TRUE(routing_gate_tensors[slot]->ensureOnDevice(device, stream));
        ASSERT_TRUE(routing_up_tensors[slot]->ensureOnDevice(device, stream));
        gate_ptrs[slot] = gate_tensors[slot].get();
        up_ptrs[slot] = up_tensors[slot].get();
        routing_gate_ptrs[slot] = routing_gate_tensors[slot].get();
        routing_up_ptrs[slot] = routing_up_tensors[slot].get();
    }

    DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, top_k);
    populateRuntimeDescriptors(host_runtime, &gate_descs, &up_descs, &down_descs);
    for (int slot = 0; slot < top_k; ++slot)
    {
        const bool local_slot = (slot % 2) == 0;
        host_runtime.topk_expert_ids[slot] = local_slot ? slot : -1;
        host_runtime.topk_weights[slot] = local_slot ? (0.60f - 0.10f * static_cast<float>(slot)) : 0.0f;
    }
    DeviceMoELayerRuntime *device_runtime = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_runtime), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(
                  device_runtime, &host_runtime, sizeof(DeviceMoELayerRuntime),
                  hipMemcpyHostToDevice, stream),
              hipSuccess);

    moe_kernel.zeroBuffer(two_step_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    moe_kernel.zeroBuffer(fused_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromRuntime(
        device_runtime, hidden.get(), gateup_table, top_k,
        gate_ptrs.data(), up_ptrs.data(), d_model, intermediate));
    ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromRuntime(
        gate_ptrs.data(), up_ptrs.data(), device_runtime, down_table, top_k,
        two_step_output.get(), d_model, intermediate));

    PerfStatsCollector::reset();
    ASSERT_TRUE(moe_kernel.groupedExpertDecodeFromRuntime(
        device_runtime, hidden.get(), gateup_table, down_table, top_k,
        fused_output.get(), d_model, intermediate));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    two_step_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    fused_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    expectStrictVerifierSimilarity(
        "ROCm fused grouped decode must match two-step decode",
        fused_output->data(),
        two_step_output->data(),
        static_cast<size_t>(d_model),
        static_cast<size_t>(d_model));

    /*
     * MTP verifier row replay is not allowed to be a near miss of ordinary
     * serial decode.  The full-model long-context parity failure first showed
     * up as a tiny layer-7 MoE expert drift that later flipped routing.  Serial
     * decode reaches this fused runtime-table path; all-position verifier rows
     * currently replay from explicit routing tensors.  Keep the two backend
     * entry points locked together here so the model harness is not the first
     * place to notice a decode-equivalence break.
     */
    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(top_k)});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(top_k)});
    for (int slot = 0; slot < top_k; ++slot)
    {
        routing_indices->mutable_data()[slot] = static_cast<float>(host_runtime.topk_expert_ids[slot]);
        routing_weights->mutable_data()[slot] = host_runtime.topk_weights[slot];
    }
    ASSERT_TRUE(routing_indices->ensureOnDevice(device, stream));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device, stream));

    moe_kernel.zeroBuffer(routing_tensor_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromRouting(
        hidden.get(), routing_indices.get(), gateup_table, top_k,
        routing_gate_ptrs.data(), routing_up_ptrs.data(), d_model, intermediate));
    ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromRouting(
        routing_gate_ptrs.data(), routing_up_ptrs.data(),
        routing_indices.get(), routing_weights.get(), down_table, top_k,
        routing_tensor_output.get(), d_model, intermediate));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    routing_tensor_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    expectStrictVerifierSimilarity(
        "ROCm routing-tensor decode must match fused runtime-table decode",
        routing_tensor_output->data(),
        fused_output->data(),
        static_cast<size_t>(d_model),
        static_cast<size_t>(d_model));

    const auto counters = PerfStatsCollector::snapshot(
        {"kernel.rocm_moe_grouped_decode_fused_calls"});
    ASSERT_FALSE(counters.empty()) << PerfStatsCollector::summaryString(
        {"kernel.rocm_moe_grouped_decode_fused_calls"});

    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
    const bool captured_fused = moe_kernel.groupedExpertDecodeFromRuntime(
        device_runtime, hidden.get(), gateup_table, down_table, top_k,
        fused_output.get(), d_model, intermediate);
    const hipError_t capture_status = hipStreamEndCapture(stream, &graph);
    EXPECT_TRUE(captured_fused);
    ASSERT_EQ(capture_status, hipSuccess) << hipGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);

    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
    ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
    ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    EXPECT_EQ(hipGraphExecDestroy(exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    ASSERT_EQ(hipFree(device_runtime), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, GroupedPrefill_Q4KGateUp_Q5KDownMatchesSequentialGemm)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int seq_len = 4;
    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    constexpr int num_experts = 4;
    constexpr int top_k = 4;
    constexpr int total_slots = seq_len * top_k;

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    std::vector<std::unique_ptr<TensorBase>> gate_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> up_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> down_weight_tensors;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> gate_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> up_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> down_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> gate_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> up_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> down_kernels;

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto gate_weights = TestTensorFactory::createQ4_KRandom(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)});
        auto up_weights = TestTensorFactory::createQ4_KRandom(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)});
        auto down_weights = TestTensorFactory::createQ5_KRandom(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)});
        injectNonZeroKQuantMins(gate_weights.get());
        injectNonZeroKQuantMins(up_weights.get());
        injectNonZeroKQuantMins(down_weights.get());

        auto gate_packed = std::make_unique<rocm::ROCmPackedWeights>();
        auto up_packed = std::make_unique<rocm::ROCmPackedWeights>();
        auto down_packed = std::make_unique<rocm::ROCmPackedWeights>();
        ASSERT_TRUE(rocm::packWeightsToROCm(gate_weights.get(), *gate_packed));
        ASSERT_TRUE(rocm::packWeightsToROCm(up_weights.get(), *up_packed));
        ASSERT_TRUE(rocm::packWeightsToROCm(down_weights.get(), *down_packed));

        gate_kernels.push_back(std::make_unique<rocm::ROCmQuantisedGemmKernel>(gate_packed.get(), 0));
        up_kernels.push_back(std::make_unique<rocm::ROCmQuantisedGemmKernel>(up_packed.get(), 0));
        down_kernels.push_back(std::make_unique<rocm::ROCmQuantisedGemmKernel>(down_packed.get(), 0));
        gate_weight_tensors.push_back(std::move(gate_weights));
        up_weight_tensors.push_back(std::move(up_weights));
        down_weight_tensors.push_back(std::move(down_weights));
        gate_packed_weights.push_back(std::move(gate_packed));
        up_packed_weights.push_back(std::move(up_packed));
        down_packed_weights.push_back(std::move(down_packed));
    }

    auto workspace = std::make_unique<DeviceWorkspaceManager>(device, 128 * 1024 * 1024);
    ASSERT_TRUE(workspace->allocate(gate_kernels[0]->getWorkspaceRequirements(seq_len, intermediate, d_model)));
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        gate_kernels[expert_id]->bindWorkspace(workspace.get());
        up_kernels[expert_id]->bindWorkspace(workspace.get());
        down_kernels[expert_id]->bindWorkspace(workspace.get());
        gate_kernels[expert_id]->prepareWeights();
        up_kernels[expert_id]->prepareWeights();
        down_kernels[expert_id]->prepareWeights();
    }

    auto make_q4k_parent = [&](const std::vector<std::unique_ptr<TensorBase>> &experts)
    {
        std::vector<uint8_t> raw;
        for (const auto &expert : experts)
        {
            const auto *bytes = static_cast<const uint8_t *>(expert->raw_data());
            raw.insert(raw.end(), bytes, bytes + expert->size_bytes());
        }
        return std::make_shared<Q4_KTensor>(
            std::vector<size_t>{static_cast<size_t>(d_model), static_cast<size_t>(intermediate), static_cast<size_t>(num_experts)},
            raw);
    };
    auto make_q5k_parent = [&](const std::vector<std::unique_ptr<TensorBase>> &experts)
    {
        std::vector<uint8_t> raw;
        for (const auto &expert : experts)
        {
            const auto *bytes = static_cast<const uint8_t *>(expert->raw_data());
            raw.insert(raw.end(), bytes, bytes + expert->size_bytes());
        }
        return std::make_shared<Q5_KTensor>(
            std::vector<size_t>{static_cast<size_t>(intermediate), static_cast<size_t>(d_model), static_cast<size_t>(num_experts)},
            raw);
    };

    auto gate_parent = make_q4k_parent(gate_weight_tensors);
    auto up_parent = make_q4k_parent(up_weight_tensors);
    auto down_parent = make_q5k_parent(down_weight_tensors);
    std::vector<bool> expert_mask;
    std::vector<std::shared_ptr<TensorBase>> gpu_gate_views;
    std::vector<std::shared_ptr<TensorBase>> gpu_up_views;
    std::vector<std::shared_ptr<TensorBase>> gpu_down_views;
    std::vector<ITensorGemm *> gpu_gate_gemms;
    std::vector<ITensorGemm *> gpu_up_gemms;
    std::vector<ITensorGemm *> gpu_down_gemms;
    std::vector<std::shared_ptr<ITensorGemm>> gpu_owned_kernels;
    std::shared_ptr<void> gpu_gate_lifetime;
    std::shared_ptr<void> gpu_up_lifetime;
    std::shared_ptr<void> gpu_down_lifetime;
    MoEWeightContext gpu_ctx{
        device,
        num_experts,
        intermediate,
        d_model,
        0,
        num_experts,
        0,
        expert_mask,
        gate_parent.get(),
        up_parent.get(),
        down_parent.get(),
        gpu_gate_views,
        gpu_up_views,
        gpu_down_views,
        gpu_gate_gemms,
        gpu_up_gemms,
        gpu_down_gemms,
        gpu_owned_kernels,
        gpu_gate_lifetime,
        gpu_up_lifetime,
        gpu_down_lifetime};
    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(gpu_ctx));
    ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(gpu_ctx));

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)}, -1.0f, 1.0f, 901);
    const float *input_host = input->data();
    ASSERT_TRUE(input->ensureOnDevice(device));

    const int routing_indices_data[total_slots] = {
        0, 1, 2, 3,
        2, 3, 0, 1,
        0, 2, 1, 3,
        3, 1, 2, 0};
    const float routing_weights_data[total_slots] = {
        0.40f, 0.30f, 0.20f, 0.10f,
        0.35f, 0.25f, 0.25f, 0.15f,
        0.45f, 0.25f, 0.20f, 0.10f,
        0.50f, 0.20f, 0.20f, 0.10f};

    std::vector<float> expected(static_cast<size_t>(seq_len) * d_model, 0.0f);
    std::vector<std::vector<std::pair<int, float>>> tokens_by_expert(num_experts);
    for (int token = 0; token < seq_len; ++token)
    {
        for (int route = 0; route < top_k; ++route)
        {
            const int slot = token * top_k + route;
            tokens_by_expert[static_cast<size_t>(routing_indices_data[slot])].push_back(
                {token, routing_weights_data[slot]});
        }
    }

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        const auto &routes = tokens_by_expert[static_cast<size_t>(expert_id)];
        if (routes.empty())
            continue;

        const int count = static_cast<int>(routes.size());
        auto batch = TestTensorFactory::createFP32(
            {static_cast<size_t>(count), static_cast<size_t>(d_model)});
        for (int row = 0; row < count; ++row)
        {
            const int token = routes[static_cast<size_t>(row)].first;
            std::copy(input_host + static_cast<size_t>(token) * d_model,
                      input_host + static_cast<size_t>(token + 1) * d_model,
                      batch->mutable_data() + static_cast<size_t>(row) * d_model);
        }
        ASSERT_TRUE(batch->ensureOnDevice(device));

        auto gate = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(intermediate)});
        auto up = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(intermediate)});
        auto expert_out = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(d_model)});
        ASSERT_TRUE(gate->ensureOnDevice(device));
        ASSERT_TRUE(up->ensureOnDevice(device));
        ASSERT_TRUE(expert_out->ensureOnDevice(device));

        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {gate_kernels[expert_id].get(), gate.get(), intermediate, nullptr, "gate"},
            {up_kernels[expert_id].get(), up.get(), intermediate, nullptr, "up"}};
        ASSERT_TRUE(gate_kernels[expert_id]->multiply_fused_tensor(batch.get(), projections, count, d_model));
        gate->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        up->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        ASSERT_TRUE(down_kernels[expert_id]->multiply_tensor_with_fused_swiglu(
            gate.get(), up.get(), expert_out.get(), count, d_model, intermediate));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        expert_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        const float *expert_host = expert_out->data();
        for (int row = 0; row < count; ++row)
        {
            const int token = routes[static_cast<size_t>(row)].first;
            const float weight = routes[static_cast<size_t>(row)].second;
            for (int col = 0; col < d_model; ++col)
            {
                expected[static_cast<size_t>(token) * d_model + col] +=
                    weight * expert_host[static_cast<size_t>(row) * d_model + col];
            }
        }
    }

    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    for (int slot = 0; slot < total_slots; ++slot)
    {
        routing_indices->mutable_data()[slot] = static_cast<float>(routing_indices_data[slot]);
        routing_weights->mutable_data()[slot] = routing_weights_data[slot];
    }
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));
    ASSERT_TRUE(moe_kernel.prepareExpertGroupsAsync(
        routing_indices.get(), routing_weights.get(), seq_len, num_experts, top_k));

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> down_descs(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        ASSERT_TRUE(gpu_gate_gemms[expert_id]->exportNativeVNNIMatrixDesc(gate_descs[expert_id]));
        ASSERT_TRUE(gpu_up_gemms[expert_id]->exportNativeVNNIMatrixDesc(up_descs[expert_id]));
        ASSERT_TRUE(gpu_down_gemms[expert_id]->exportNativeVNNIMatrixDesc(down_descs[expert_id]));
    }

    const int gateup_table_id = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table_id, 0);
    const int down_table_id = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table_id, 0);

    /*
     * `router_reuse_decode_expected` is deliberately the live production M=1
     * oracle.  The captured hidden words, exact top-k route, real checkpoint
     * payloads, and preparation service above pin every input to this operation;
     * freezing its output would instead pin one historical Q8 implementation
     * and become stale whenever serial and grouped arithmetic are corrected
     * together.  Every grouped implementation below must match these current
     * serial bytes exactly.
     */
    auto grouped_output = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(grouped_output->ensureOnDevice(device));
    {
        ScopedROCmEnvOverride disable_parallel_down("LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE", "0");
        moe_kernel.zeroBuffer(grouped_output.get(), static_cast<size_t>(seq_len) * d_model * sizeof(float));
        ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
            input.get(), grouped_output.get(), gateup_table_id, down_table_id,
            seq_len, d_model, intermediate, num_experts, top_k));
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *grouped = grouped_output->data();
    expectStrictVerifierSimilarity(
        "ROCm grouped prefill must match sequential GPU rows",
        grouped,
        expected.data(),
        expected.size(),
        static_cast<size_t>(d_model));

    /*
     * ROCm serial decode normally uses the parallel down-projection kernel for
     * top-k > 1.  That path publishes one atomic contribution per original
     * route slot.  Grouped verifier prefill has already computed expert rows in
     * grouped-by-expert order, so this regression checks that its final scatter
     * switches to original-slot atomic publication when parallel-down decode is
     * enabled.  Otherwise the grouped verifier is close to the deterministic
     * sequential sum above, but not equivalent to default serial decode.
     */
    std::vector<float> parallel_reference(static_cast<size_t>(seq_len) * d_model, 0.0f);
    auto grouped_parallel_output = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(grouped_parallel_output->ensureOnDevice(device));
    {
        ScopedROCmEnvOverride enable_parallel_down("LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE", "1");
        ScopedROCmEnvOverride enable_gateup_kpart("LLAMINAR_ROCM_MOE_GATEUP_KPART_DECODE", "1");
        ScopedROCmEnvOverride set_gateup_kparts("LLAMINAR_ROCM_MOE_GATEUP_KPARTS", "4");

        std::array<std::shared_ptr<FP32Tensor>, top_k> gate_rows;
        std::array<std::shared_ptr<FP32Tensor>, top_k> up_rows;
        std::array<ITensor *, top_k> gate_ptrs = {};
        std::array<ITensor *, top_k> up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            gate_rows[slot] = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
            up_rows[slot] = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
            ASSERT_TRUE(gate_rows[slot]->ensureOnDevice(device));
            ASSERT_TRUE(up_rows[slot]->ensureOnDevice(device));
            gate_ptrs[slot] = gate_rows[slot].get();
            up_ptrs[slot] = up_rows[slot].get();
        }

        for (int token = 0; token < seq_len; ++token)
        {
            auto row_input = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
            std::copy(input_host + static_cast<size_t>(token) * d_model,
                      input_host + static_cast<size_t>(token + 1) * d_model,
                      row_input->mutable_data());
            ASSERT_TRUE(row_input->ensureOnDevice(device));

            auto row_indices = TestTensorFactory::createFP32({static_cast<size_t>(top_k)});
            auto row_weights = TestTensorFactory::createFP32({static_cast<size_t>(top_k)});
            for (int route = 0; route < top_k; ++route)
            {
                const int slot = token * top_k + route;
                row_indices->mutable_data()[route] = static_cast<float>(routing_indices_data[slot]);
                row_weights->mutable_data()[route] = routing_weights_data[slot];
            }
            ASSERT_TRUE(row_indices->ensureOnDevice(device));
            ASSERT_TRUE(row_weights->ensureOnDevice(device));

            auto row_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
            ASSERT_TRUE(row_output->ensureOnDevice(device));
            moe_kernel.zeroBuffer(row_output.get(), static_cast<size_t>(d_model) * sizeof(float));
            ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromRouting(
                row_input.get(), row_indices.get(), gateup_table_id, top_k,
                gate_ptrs.data(), up_ptrs.data(), d_model, intermediate));
            ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromRouting(
                gate_ptrs.data(), up_ptrs.data(),
                row_indices.get(), row_weights.get(), down_table_id, top_k,
                row_output.get(), d_model, intermediate));
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            row_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            std::copy(row_output->data(),
                      row_output->data() + d_model,
                      parallel_reference.data() + static_cast<size_t>(token) * d_model);
        }

        moe_kernel.zeroBuffer(grouped_parallel_output.get(), static_cast<size_t>(seq_len) * d_model * sizeof(float));
        ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
            input.get(), grouped_parallel_output.get(), gateup_table_id, down_table_id,
            seq_len, d_model, intermediate, num_experts, top_k));
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    grouped_parallel_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    expectStrictVerifierSimilarity(
        "ROCm grouped prefill parallel scatter must match row-by-row parallel decode",
        grouped_parallel_output->data(),
        parallel_reference.data(),
        parallel_reference.size(),
        static_cast<size_t>(d_model));
}

TEST(Test__ROCmMoEKernel, GroupedPrefillMaskedTopK8MatchesRowDecode)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int seq_len = 2;
    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    constexpr int num_experts = 16;
    constexpr int top_k = 8;
    constexpr int total_slots = seq_len * top_k;

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    std::vector<std::unique_ptr<TensorBase>> gate_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> up_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> down_weight_tensors;
    gate_weight_tensors.reserve(num_experts);
    up_weight_tensors.reserve(num_experts);
    down_weight_tensors.reserve(num_experts);

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto gate_weights = TestTensorFactory::createQ4_KRandom(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)},
            41000 + expert_id);
        auto up_weights = TestTensorFactory::createQ4_KRandom(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)},
            42000 + expert_id);
        auto down_weights = TestTensorFactory::createQ5_KRandom(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)},
            43000 + expert_id);
        injectNonZeroKQuantMins(gate_weights.get());
        injectNonZeroKQuantMins(up_weights.get());
        injectNonZeroKQuantMins(down_weights.get());
        gate_weight_tensors.push_back(std::move(gate_weights));
        up_weight_tensors.push_back(std::move(up_weights));
        down_weight_tensors.push_back(std::move(down_weights));
    }

    auto gate_parent = makeExpertParentTensor<Q4_KTensor>(
        gate_weight_tensors,
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate), static_cast<size_t>(num_experts)});
    auto up_parent = makeExpertParentTensor<Q4_KTensor>(
        up_weight_tensors,
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate), static_cast<size_t>(num_experts)});
    auto down_parent = makeExpertParentTensor<Q5_KTensor>(
        down_weight_tensors,
        {static_cast<size_t>(intermediate), static_cast<size_t>(d_model), static_cast<size_t>(num_experts)});

    std::vector<bool> expert_mask;
    std::vector<std::shared_ptr<TensorBase>> gpu_gate_views;
    std::vector<std::shared_ptr<TensorBase>> gpu_up_views;
    std::vector<std::shared_ptr<TensorBase>> gpu_down_views;
    std::vector<ITensorGemm *> gpu_gate_gemms;
    std::vector<ITensorGemm *> gpu_up_gemms;
    std::vector<ITensorGemm *> gpu_down_gemms;
    std::vector<std::shared_ptr<ITensorGemm>> gpu_owned_kernels;
    std::shared_ptr<void> gpu_gate_lifetime;
    std::shared_ptr<void> gpu_up_lifetime;
    std::shared_ptr<void> gpu_down_lifetime;
    MoEWeightContext gpu_ctx{
        device,
        num_experts,
        intermediate,
        d_model,
        0,
        num_experts,
        0,
        expert_mask,
        gate_parent.get(),
        up_parent.get(),
        down_parent.get(),
        gpu_gate_views,
        gpu_up_views,
        gpu_down_views,
        gpu_gate_gemms,
        gpu_up_gemms,
        gpu_down_gemms,
        gpu_owned_kernels,
        gpu_gate_lifetime,
        gpu_up_lifetime,
        gpu_down_lifetime};
    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(gpu_ctx));
    ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(gpu_ctx));

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> down_descs(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        ASSERT_TRUE(gpu_gate_gemms[expert_id]->exportNativeVNNIMatrixDesc(gate_descs[expert_id]));
        ASSERT_TRUE(gpu_up_gemms[expert_id]->exportNativeVNNIMatrixDesc(up_descs[expert_id]));
        ASSERT_TRUE(gpu_down_gemms[expert_id]->exportNativeVNNIMatrixDesc(down_descs[expert_id]));
    }

    const int gateup_table_id = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table_id, 0);
    const int down_table_id = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table_id, 0);

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)}, -0.05f, 0.05f, 44001);
    const float *input_host = input->data();
    ASSERT_TRUE(input->ensureOnDevice(device));

    /*
     * This mirrors a LocalTP expert overlay participant: the router still emits
     * a full top-k list, but this rank owns only part of the expert set.  The
     * masked grouped prefill path must therefore publish exactly the same local
     * contribution that serial decode would have produced before the TP
     * allreduce combines participants.
     */
    const int routing_indices_data[total_slots] = {
        0, 9, 1, 10, 2, 11, 3, 12,
        4, 13, 5, 14, 6, 15, 7, 8};
    const float routing_weights_data[total_slots] = {
        0.24f, 0.19f, 0.16f, 0.13f, 0.10f, 0.08f, 0.06f, 0.04f,
        0.23f, 0.18f, 0.15f, 0.13f, 0.11f, 0.09f, 0.07f, 0.04f};
    std::array<uint8_t, num_experts> local_expert_mask{};
    for (int expert_id = 0; expert_id < 8; ++expert_id)
        local_expert_mask[static_cast<size_t>(expert_id)] = 1u;

    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    for (int slot = 0; slot < total_slots; ++slot)
    {
        routing_indices->mutable_data()[slot] = static_cast<float>(routing_indices_data[slot]);
        routing_weights->mutable_data()[slot] = routing_weights_data[slot];
    }
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));
    ASSERT_TRUE(moe_kernel.prepareExpertGroupsAsyncMasked(
        routing_indices.get(), routing_weights.get(),
        seq_len, num_experts, top_k,
        local_expert_mask.data()));

    std::vector<float> row_by_row_reference(static_cast<size_t>(seq_len) * d_model, 0.0f);
    auto grouped_parallel_output = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(grouped_parallel_output->ensureOnDevice(device));

    ScopedROCmEnvOverride disable_parallel_down("LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE", "0");
    ScopedROCmEnvOverride disable_gateup_kpart("LLAMINAR_ROCM_MOE_GATEUP_KPART_DECODE", "0");
    ScopedROCmEnvOverride disable_fused_gateup_swiglu("LLAMINAR_ROCM_MOE_GATEUP_SWIGLU_QUANT_FUSED", "0");

    std::array<std::shared_ptr<FP32Tensor>, top_k> gate_rows;
    std::array<std::shared_ptr<FP32Tensor>, top_k> up_rows;
    std::array<ITensor *, top_k> gate_ptrs = {};
    std::array<ITensor *, top_k> up_ptrs = {};
    for (int route = 0; route < top_k; ++route)
    {
        gate_rows[route] = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        up_rows[route] = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        ASSERT_TRUE(gate_rows[route]->ensureOnDevice(device));
        ASSERT_TRUE(up_rows[route]->ensureOnDevice(device));
        gate_ptrs[route] = gate_rows[route].get();
        up_ptrs[route] = up_rows[route].get();
    }

    for (int token = 0; token < seq_len; ++token)
    {
        auto row_input = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
        std::copy(input_host + static_cast<size_t>(token) * d_model,
                  input_host + static_cast<size_t>(token + 1) * d_model,
                  row_input->mutable_data());
        ASSERT_TRUE(row_input->ensureOnDevice(device));

        auto row_indices = TestTensorFactory::createFP32({static_cast<size_t>(top_k)});
        auto row_weights = TestTensorFactory::createFP32({static_cast<size_t>(top_k)});
        for (int route = 0; route < top_k; ++route)
        {
            const int slot = token * top_k + route;
            const int expert = routing_indices_data[slot];
            const bool local = expert >= 0 && expert < num_experts &&
                               local_expert_mask[static_cast<size_t>(expert)] != 0u;
            row_indices->mutable_data()[route] = local ? static_cast<float>(expert) : -1.0f;
            row_weights->mutable_data()[route] = local ? routing_weights_data[slot] : 0.0f;
        }
        ASSERT_TRUE(row_indices->ensureOnDevice(device));
        ASSERT_TRUE(row_weights->ensureOnDevice(device));

        auto row_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
        ASSERT_TRUE(row_output->ensureOnDevice(device));
        moe_kernel.zeroBuffer(row_output.get(), static_cast<size_t>(d_model) * sizeof(float));
        ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromRouting(
            row_input.get(), row_indices.get(), gateup_table_id, top_k,
            gate_ptrs.data(), up_ptrs.data(), d_model, intermediate));
        ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromRouting(
            gate_ptrs.data(), up_ptrs.data(),
            row_indices.get(), row_weights.get(), down_table_id, top_k,
            row_output.get(), d_model, intermediate));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        row_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        std::copy(row_output->data(),
                  row_output->data() + d_model,
                  row_by_row_reference.data() + static_cast<size_t>(token) * d_model);
    }

    moe_kernel.zeroBuffer(grouped_parallel_output.get(), static_cast<size_t>(seq_len) * d_model * sizeof(float));
    ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
        input.get(), grouped_parallel_output.get(), gateup_table_id, down_table_id,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    grouped_parallel_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    expectBitwiseVerifierRowsEqual(
        "ROCm masked grouped prefill must match row-by-row local decode",
        grouped_parallel_output->data(),
        row_by_row_reference.data(),
        row_by_row_reference.size(),
        static_cast<size_t>(d_model));
}

TEST(Test__ROCmMoEKernel, GroupedPrefill_Qwen35RouteTable_Q4KQ5KMatchesCpuDequantReference)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int seq_len = 9;
    constexpr int d_model = 512;
    constexpr int intermediate = 256;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;
    constexpr int total_slots = seq_len * top_k;

    const std::vector<int> routing_indices = {
        112, 106, 107, 238, 181, 57, 200, 43,
        109, 41, 65, 24, 86, 29, 136, 2,
        134, 115, 220, 159, 41, 2, 224, 28,
        13, 131, 139, 47, 151, 159, 192, 207,
        13, 234, 242, 188, 144, 186, 217, 159,
        96, 187, 242, 117, 52, 133, 27, 250,
        181, 107, 82, 251, 80, 68, 59, 131,
        109, 41, 2, 24, 177, 131, 5, 225,
        139, 13, 159, 47, 116, 55, 220, 131};
    ASSERT_EQ(routing_indices.size(), static_cast<size_t>(total_slots));

    std::vector<float> routing_weights(static_cast<size_t>(total_slots));
    for (int token_idx = 0; token_idx < seq_len; ++token_idx)
    {
        float weight_sum = 0.0f;
        for (int route_idx = 0; route_idx < top_k; ++route_idx)
        {
            const int slot_idx = token_idx * top_k + route_idx;
            routing_weights[static_cast<size_t>(slot_idx)] =
                (1.0f / static_cast<float>(route_idx + 2)) + 0.003f * static_cast<float>(token_idx + 1);
            weight_sum += routing_weights[static_cast<size_t>(slot_idx)];
        }
        for (int route_idx = 0; route_idx < top_k; ++route_idx)
        {
            const int slot_idx = token_idx * top_k + route_idx;
            routing_weights[static_cast<size_t>(slot_idx)] /= weight_sum;
        }
    }

    std::vector<std::unique_ptr<TensorBase>> gate_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> up_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> down_weight_tensors;
    gate_weight_tensors.reserve(num_experts);
    up_weight_tensors.reserve(num_experts);
    down_weight_tensors.reserve(num_experts);

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto gate_weights = TestTensorFactory::createQ4_KRandom(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 10000 + expert_id);
        auto up_weights = TestTensorFactory::createQ4_KRandom(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 20000 + expert_id);
        auto down_weights = TestTensorFactory::createQ5_KRandom(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 30000 + expert_id);
        injectNonZeroKQuantMins(gate_weights.get());
        injectNonZeroKQuantMins(up_weights.get());
        injectNonZeroKQuantMins(down_weights.get());

        gate_weight_tensors.push_back(std::move(gate_weights));
        up_weight_tensors.push_back(std::move(up_weights));
        down_weight_tensors.push_back(std::move(down_weights));
    }

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)}, -0.05f, 0.05f, 3519);
    const float *input_host = input->data();

    const std::vector<float> reference = computeCpuDequantMoEPrefillReference(
        input_host, seq_len, d_model, intermediate, num_experts, top_k,
        gate_weight_tensors, up_weight_tensors, down_weight_tensors,
        routing_indices, routing_weights);

    auto gate_parent = makeExpertParentTensor<Q4_KTensor>(
        gate_weight_tensors,
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate), static_cast<size_t>(num_experts)});
    auto up_parent = makeExpertParentTensor<Q4_KTensor>(
        up_weight_tensors,
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate), static_cast<size_t>(num_experts)});
    auto down_parent = makeExpertParentTensor<Q5_KTensor>(
        down_weight_tensors,
        {static_cast<size_t>(intermediate), static_cast<size_t>(d_model), static_cast<size_t>(num_experts)});

    std::vector<bool> expert_mask;
    std::vector<std::shared_ptr<TensorBase>> gpu_gate_views;
    std::vector<std::shared_ptr<TensorBase>> gpu_up_views;
    std::vector<std::shared_ptr<TensorBase>> gpu_down_views;
    std::vector<ITensorGemm *> gpu_gate_gemms;
    std::vector<ITensorGemm *> gpu_up_gemms;
    std::vector<ITensorGemm *> gpu_down_gemms;
    std::vector<std::shared_ptr<ITensorGemm>> gpu_owned_kernels;
    std::shared_ptr<void> gpu_gate_lifetime;
    std::shared_ptr<void> gpu_up_lifetime;
    std::shared_ptr<void> gpu_down_lifetime;
    MoEWeightContext gpu_ctx{
        device,
        num_experts,
        intermediate,
        d_model,
        0,
        num_experts,
        0,
        expert_mask,
        gate_parent.get(),
        up_parent.get(),
        down_parent.get(),
        gpu_gate_views,
        gpu_up_views,
        gpu_down_views,
        gpu_gate_gemms,
        gpu_up_gemms,
        gpu_down_gemms,
        gpu_owned_kernels,
        gpu_gate_lifetime,
        gpu_up_lifetime,
        gpu_down_lifetime};
    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(gpu_ctx));
    ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(gpu_ctx));

    auto *gate0_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_gate_gemms[0]);
    auto *up0_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_up_gemms[0]);
    auto *down0_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_down_gemms[0]);
    ASSERT_NE(gate0_workspace, nullptr);
    ASSERT_NE(up0_workspace, nullptr);
    ASSERT_NE(down0_workspace, nullptr);

    WorkspaceRequirements sequential_reqs;
    sequential_reqs.merge(gate0_workspace->getWorkspaceRequirements(seq_len, intermediate, d_model));
    sequential_reqs.merge(up0_workspace->getWorkspaceRequirements(seq_len, intermediate, d_model));
    sequential_reqs.merge(down0_workspace->getWorkspaceRequirements(seq_len, d_model, intermediate));
    auto sequential_workspace = std::make_unique<DeviceWorkspaceManager>(device, 128 * 1024 * 1024);
    ASSERT_TRUE(sequential_workspace->allocate(sequential_reqs));
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto *gate_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_gate_gemms[expert_id]);
        auto *up_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_up_gemms[expert_id]);
        auto *down_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_down_gemms[expert_id]);
        ASSERT_NE(gate_workspace, nullptr);
        ASSERT_NE(up_workspace, nullptr);
        ASSERT_NE(down_workspace, nullptr);
        gate_workspace->bindWorkspace(sequential_workspace.get());
        up_workspace->bindWorkspace(sequential_workspace.get());
        down_workspace->bindWorkspace(sequential_workspace.get());
    }

    std::vector<float> sequential_output(static_cast<size_t>(seq_len) * d_model, 0.0f);
    std::vector<std::vector<std::pair<int, float>>> routes_by_expert(static_cast<size_t>(num_experts));
    for (int token_idx = 0; token_idx < seq_len; ++token_idx)
    {
        for (int route_idx = 0; route_idx < top_k; ++route_idx)
        {
            const int slot_idx = token_idx * top_k + route_idx;
            routes_by_expert[static_cast<size_t>(routing_indices[static_cast<size_t>(slot_idx)])].push_back(
                {token_idx, routing_weights[static_cast<size_t>(slot_idx)]});
        }
    }

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        const auto &routes = routes_by_expert[static_cast<size_t>(expert_id)];
        if (routes.empty())
            continue;

        const int count = static_cast<int>(routes.size());
        auto batch = TestTensorFactory::createFP32(
            {static_cast<size_t>(count), static_cast<size_t>(d_model)});
        for (int row = 0; row < count; ++row)
        {
            const int token_idx = routes[static_cast<size_t>(row)].first;
            std::copy(input_host + static_cast<size_t>(token_idx) * d_model,
                      input_host + static_cast<size_t>(token_idx + 1) * d_model,
                      batch->mutable_data() + static_cast<size_t>(row) * d_model);
        }
        ASSERT_TRUE(batch->ensureOnDevice(device));

        auto gate = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(intermediate)});
        auto up = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(intermediate)});
        auto expert_out = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(d_model)});
        ASSERT_TRUE(gate->ensureOnDevice(device));
        ASSERT_TRUE(up->ensureOnDevice(device));
        ASSERT_TRUE(expert_out->ensureOnDevice(device));

        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {gpu_gate_gemms[expert_id], gate.get(), intermediate, nullptr, "gate"},
            {gpu_up_gemms[expert_id], up.get(), intermediate, nullptr, "up"}};
        ASSERT_TRUE(gpu_gate_gemms[expert_id]->multiply_fused_tensor(
            batch.get(), projections, count, d_model));
        gate->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        up->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        ASSERT_TRUE(gpu_down_gemms[expert_id]->multiply_tensor_with_fused_swiglu(
            gate.get(), up.get(), expert_out.get(), count, d_model, intermediate));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        expert_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        const float *expert_host = expert_out->data();
        for (int row = 0; row < count; ++row)
        {
            const int token_idx = routes[static_cast<size_t>(row)].first;
            const float route_weight = routes[static_cast<size_t>(row)].second;
            for (int col = 0; col < d_model; ++col)
            {
                sequential_output[static_cast<size_t>(token_idx) * d_model + col] +=
                    route_weight * expert_host[static_cast<size_t>(row) * d_model + col];
            }
        }
    }

    const double sequential_cosine = cosineSimilarity(
        sequential_output.data(), reference.data(), reference.size());
    const double sequential_rel_l2 = relativeL2Error(
        sequential_output.data(), reference.data(), reference.size());
    const double sequential_max_abs = maxAbsDiff(
        sequential_output.data(), reference.data(), reference.size());
    EXPECT_GE(sequential_cosine, 0.960) << "relative L2=" << sequential_rel_l2
                                        << " max_abs=" << sequential_max_abs;
    EXPECT_LE(sequential_rel_l2, 0.35) << "cosine=" << sequential_cosine
                                       << " max_abs=" << sequential_max_abs;

    auto routing_indices_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    for (int slot_idx = 0; slot_idx < total_slots; ++slot_idx)
    {
        routing_indices_tensor->mutable_data()[slot_idx] =
            static_cast<float>(routing_indices[static_cast<size_t>(slot_idx)]);
        routing_weights_tensor->mutable_data()[slot_idx] = routing_weights[static_cast<size_t>(slot_idx)];
    }
    ASSERT_TRUE(input->ensureOnDevice(device));
    ASSERT_TRUE(routing_indices_tensor->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights_tensor->ensureOnDevice(device));

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);
    ASSERT_TRUE(moe_kernel.prepareExpertGroupsAsync(
        routing_indices_tensor.get(), routing_weights_tensor.get(), seq_len, num_experts, top_k));

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> down_descs(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        ASSERT_TRUE(gpu_gate_gemms[expert_id]->exportNativeVNNIMatrixDesc(gate_descs[expert_id]));
        ASSERT_TRUE(gpu_up_gemms[expert_id]->exportNativeVNNIMatrixDesc(up_descs[expert_id]));
        ASSERT_TRUE(gpu_down_gemms[expert_id]->exportNativeVNNIMatrixDesc(down_descs[expert_id]));
    }

    const int gateup_table_id = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table_id, 0);
    const int down_table_id = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table_id, 0);

    auto grouped_output = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(grouped_output->ensureOnDevice(device));
    moe_kernel.zeroBuffer(grouped_output.get(), static_cast<size_t>(seq_len) * d_model * sizeof(float));
    ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
        input.get(), grouped_output.get(), gateup_table_id, down_table_id,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *grouped = grouped_output->data();
    ASSERT_FALSE(hasNaNOrInf(grouped, reference.size()));

    const double cosine = cosineSimilarity(grouped, reference.data(), reference.size());
    const double rel_l2 = relativeL2Error(grouped, reference.data(), reference.size());
    const double max_abs = maxAbsDiff(grouped, reference.data(), reference.size());
    EXPECT_GE(cosine, 0.985) << "relative L2=" << rel_l2 << " max_abs=" << max_abs;
    EXPECT_LE(rel_l2, 0.20) << "cosine=" << cosine << " max_abs=" << max_abs;

    std::cout << "[GroupedPrefill_Qwen35RouteTable_Q4KQ5KMatchesCpuDequantReference] cosine="
              << std::fixed << std::setprecision(6) << cosine
              << " rel_l2=" << rel_l2
              << " max_abs=" << max_abs << std::endl;
}

TEST(Test__ROCmMoEKernel, GroupedPrefill_Qwen36RouteTable_IQ2SGateUp_IQ4XSDownMatchesCpuDequantReference)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int seq_len = 9;
    constexpr int d_model = 512;
    constexpr int intermediate = 256;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;
    constexpr int total_slots = seq_len * top_k;

    const std::vector<int> routing_indices = {
        112, 106, 107, 238, 181, 57, 200, 43,
        109, 41, 65, 24, 86, 29, 136, 2,
        134, 115, 220, 159, 41, 2, 224, 28,
        13, 131, 139, 47, 151, 159, 192, 207,
        13, 234, 242, 188, 144, 186, 217, 159,
        96, 187, 242, 117, 52, 133, 27, 250,
        181, 107, 82, 251, 80, 68, 59, 131,
        109, 41, 2, 24, 177, 131, 5, 225,
        139, 13, 159, 47, 116, 55, 220, 131};
    ASSERT_EQ(routing_indices.size(), static_cast<size_t>(total_slots));

    std::vector<float> routing_weights(static_cast<size_t>(total_slots));
    for (int token_idx = 0; token_idx < seq_len; ++token_idx)
    {
        float weight_sum = 0.0f;
        for (int route_idx = 0; route_idx < top_k; ++route_idx)
        {
            const int slot_idx = token_idx * top_k + route_idx;
            routing_weights[static_cast<size_t>(slot_idx)] =
                (1.0f / static_cast<float>(route_idx + 2)) + 0.003f * static_cast<float>(token_idx + 1);
            weight_sum += routing_weights[static_cast<size_t>(slot_idx)];
        }
        for (int route_idx = 0; route_idx < top_k; ++route_idx)
        {
            const int slot_idx = token_idx * top_k + route_idx;
            routing_weights[static_cast<size_t>(slot_idx)] /= weight_sum;
        }
    }

    std::vector<std::unique_ptr<TensorBase>> gate_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> up_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> down_weight_tensors;
    gate_weight_tensors.reserve(num_experts);
    up_weight_tensors.reserve(num_experts);
    down_weight_tensors.reserve(num_experts);

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        gate_weight_tensors.push_back(TestTensorFactory::createIQ2_SRandom(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 40000 + expert_id));
        up_weight_tensors.push_back(TestTensorFactory::createIQ2_SRandom(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 50000 + expert_id));
        down_weight_tensors.push_back(TestTensorFactory::createIQ4_XSRandom(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 60000 + expert_id));
    }

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)}, -0.05f, 0.05f, 3619);
    const float *input_host = input->data();

    const std::vector<float> reference = computeCpuDequantMoEPrefillReference(
        input_host, seq_len, d_model, intermediate, num_experts, top_k,
        gate_weight_tensors, up_weight_tensors, down_weight_tensors,
        routing_indices, routing_weights);

    auto gate_parent = makeExpertParentTensor<IQ2_STensor>(
        gate_weight_tensors,
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate), static_cast<size_t>(num_experts)});
    auto up_parent = makeExpertParentTensor<IQ2_STensor>(
        up_weight_tensors,
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate), static_cast<size_t>(num_experts)});
    auto down_parent = makeExpertParentTensor<IQ4_XSTensor>(
        down_weight_tensors,
        {static_cast<size_t>(intermediate), static_cast<size_t>(d_model), static_cast<size_t>(num_experts)});

    std::vector<bool> expert_mask;
    std::vector<std::shared_ptr<TensorBase>> gpu_gate_views;
    std::vector<std::shared_ptr<TensorBase>> gpu_up_views;
    std::vector<std::shared_ptr<TensorBase>> gpu_down_views;
    std::vector<ITensorGemm *> gpu_gate_gemms;
    std::vector<ITensorGemm *> gpu_up_gemms;
    std::vector<ITensorGemm *> gpu_down_gemms;
    std::vector<std::shared_ptr<ITensorGemm>> gpu_owned_kernels;
    std::shared_ptr<void> gpu_gate_lifetime;
    std::shared_ptr<void> gpu_up_lifetime;
    std::shared_ptr<void> gpu_down_lifetime;
    MoEWeightContext gpu_ctx{
        device,
        num_experts,
        intermediate,
        d_model,
        0,
        num_experts,
        0,
        expert_mask,
        gate_parent.get(),
        up_parent.get(),
        down_parent.get(),
        gpu_gate_views,
        gpu_up_views,
        gpu_down_views,
        gpu_gate_gemms,
        gpu_up_gemms,
        gpu_down_gemms,
        gpu_owned_kernels,
        gpu_gate_lifetime,
        gpu_up_lifetime,
        gpu_down_lifetime};
    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(gpu_ctx));
    ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(gpu_ctx));

    auto *gate0_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_gate_gemms[0]);
    auto *up0_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_up_gemms[0]);
    auto *down0_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_down_gemms[0]);
    ASSERT_NE(gate0_workspace, nullptr);
    ASSERT_NE(up0_workspace, nullptr);
    ASSERT_NE(down0_workspace, nullptr);

    WorkspaceRequirements sequential_reqs;
    sequential_reqs.merge(gate0_workspace->getWorkspaceRequirements(seq_len, intermediate, d_model));
    sequential_reqs.merge(up0_workspace->getWorkspaceRequirements(seq_len, intermediate, d_model));
    sequential_reqs.merge(down0_workspace->getWorkspaceRequirements(seq_len, d_model, intermediate));
    auto sequential_workspace = std::make_unique<DeviceWorkspaceManager>(device, 128 * 1024 * 1024);
    ASSERT_TRUE(sequential_workspace->allocate(sequential_reqs));
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto *gate_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_gate_gemms[expert_id]);
        auto *up_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_up_gemms[expert_id]);
        auto *down_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_down_gemms[expert_id]);
        ASSERT_NE(gate_workspace, nullptr);
        ASSERT_NE(up_workspace, nullptr);
        ASSERT_NE(down_workspace, nullptr);
        gate_workspace->bindWorkspace(sequential_workspace.get());
        up_workspace->bindWorkspace(sequential_workspace.get());
        down_workspace->bindWorkspace(sequential_workspace.get());
    }

    std::vector<float> sequential_output(static_cast<size_t>(seq_len) * d_model, 0.0f);
    std::vector<std::vector<std::pair<int, float>>> routes_by_expert(static_cast<size_t>(num_experts));
    for (int token_idx = 0; token_idx < seq_len; ++token_idx)
    {
        for (int route_idx = 0; route_idx < top_k; ++route_idx)
        {
            const int slot_idx = token_idx * top_k + route_idx;
            routes_by_expert[static_cast<size_t>(routing_indices[static_cast<size_t>(slot_idx)])].push_back(
                {token_idx, routing_weights[static_cast<size_t>(slot_idx)]});
        }
    }

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        const auto &routes = routes_by_expert[static_cast<size_t>(expert_id)];
        if (routes.empty())
            continue;

        const int count = static_cast<int>(routes.size());
        auto batch = TestTensorFactory::createFP32(
            {static_cast<size_t>(count), static_cast<size_t>(d_model)});
        for (int row = 0; row < count; ++row)
        {
            const int token_idx = routes[static_cast<size_t>(row)].first;
            std::copy(input_host + static_cast<size_t>(token_idx) * d_model,
                      input_host + static_cast<size_t>(token_idx + 1) * d_model,
                      batch->mutable_data() + static_cast<size_t>(row) * d_model);
        }
        ASSERT_TRUE(batch->ensureOnDevice(device));

        auto gate = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(intermediate)});
        auto up = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(intermediate)});
        auto expert_out = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(d_model)});
        ASSERT_TRUE(gate->ensureOnDevice(device));
        ASSERT_TRUE(up->ensureOnDevice(device));
        ASSERT_TRUE(expert_out->ensureOnDevice(device));

        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {gpu_gate_gemms[expert_id], gate.get(), intermediate, nullptr, "gate"},
            {gpu_up_gemms[expert_id], up.get(), intermediate, nullptr, "up"}};
        ASSERT_TRUE(gpu_gate_gemms[expert_id]->multiply_fused_tensor(
            batch.get(), projections, count, d_model));
        gate->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        up->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        ASSERT_TRUE(gpu_down_gemms[expert_id]->multiply_tensor_with_fused_swiglu(
            gate.get(), up.get(), expert_out.get(), count, d_model, intermediate));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        expert_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        const float *expert_host = expert_out->data();
        for (int row = 0; row < count; ++row)
        {
            const int token_idx = routes[static_cast<size_t>(row)].first;
            const float route_weight = routes[static_cast<size_t>(row)].second;
            for (int col = 0; col < d_model; ++col)
            {
                sequential_output[static_cast<size_t>(token_idx) * d_model + col] +=
                    route_weight * expert_host[static_cast<size_t>(row) * d_model + col];
            }
        }
    }

    const double sequential_cosine = cosineSimilarity(
        sequential_output.data(), reference.data(), reference.size());
    const double sequential_rel_l2 = relativeL2Error(
        sequential_output.data(), reference.data(), reference.size());
    const double sequential_max_abs = maxAbsDiff(
        sequential_output.data(), reference.data(), reference.size());
    EXPECT_GE(sequential_cosine, 0.960) << "relative L2=" << sequential_rel_l2
                                        << " max_abs=" << sequential_max_abs;
    EXPECT_LE(sequential_rel_l2, 0.35) << "cosine=" << sequential_cosine
                                       << " max_abs=" << sequential_max_abs;

    auto routing_indices_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    for (int slot_idx = 0; slot_idx < total_slots; ++slot_idx)
    {
        routing_indices_tensor->mutable_data()[slot_idx] =
            static_cast<float>(routing_indices[static_cast<size_t>(slot_idx)]);
        routing_weights_tensor->mutable_data()[slot_idx] = routing_weights[static_cast<size_t>(slot_idx)];
    }
    ASSERT_TRUE(input->ensureOnDevice(device));
    ASSERT_TRUE(routing_indices_tensor->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights_tensor->ensureOnDevice(device));

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);
    ASSERT_TRUE(moe_kernel.prepareExpertGroupsAsync(
        routing_indices_tensor.get(), routing_weights_tensor.get(), seq_len, num_experts, top_k));

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> down_descs(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        ASSERT_TRUE(gpu_gate_gemms[expert_id]->exportNativeVNNIMatrixDesc(gate_descs[expert_id]));
        ASSERT_TRUE(gpu_up_gemms[expert_id]->exportNativeVNNIMatrixDesc(up_descs[expert_id]));
        ASSERT_TRUE(gpu_down_gemms[expert_id]->exportNativeVNNIMatrixDesc(down_descs[expert_id]));
    }

    const int gateup_table_id = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table_id, 0);
    const int down_table_id = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table_id, 0);

    auto grouped_output = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(grouped_output->ensureOnDevice(device));
    moe_kernel.zeroBuffer(grouped_output.get(), static_cast<size_t>(seq_len) * d_model * sizeof(float));
    ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
        input.get(), grouped_output.get(), gateup_table_id, down_table_id,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *grouped = grouped_output->data();
    ASSERT_FALSE(hasNaNOrInf(grouped, reference.size()));

    const double cosine = cosineSimilarity(grouped, reference.data(), reference.size());
    const double rel_l2 = relativeL2Error(grouped, reference.data(), reference.size());
    const double max_abs = maxAbsDiff(grouped, reference.data(), reference.size());
    EXPECT_GE(cosine, 0.960) << "relative L2=" << rel_l2 << " max_abs=" << max_abs;
    EXPECT_LE(rel_l2, 0.35) << "cosine=" << cosine << " max_abs=" << max_abs;

    std::cout << "[GroupedPrefill_Qwen36RouteTable_IQ2SGateUp_IQ4XSDownMatchesCpuDequantReference] cosine="
              << std::fixed << std::setprecision(6) << cosine
              << " rel_l2=" << rel_l2
              << " max_abs=" << max_abs << std::endl;
}

// ============================================================================
// Test: KernelFactory dispatch creates ROCmMoEKernel for ROCm devices
// ============================================================================

TEST(Test__ROCmMoEKernel, KernelFactoryDispatch)
{
    SKIP_IF_NO_ROCM();

    using KernelFactory = llaminar::v2::kernels::KernelFactory;
    auto *kernel = KernelFactory::getOrCreateMoEKernel(DeviceId::rocm(0));
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(0))
        << "ROCm MoE kernel should support GPU device index";
    EXPECT_FALSE(kernel->supports_device(-1))
        << "ROCm MoE kernel should NOT support CPU device index";
}

// ============================================================================
// Phase 2 Tests: Device-Resident Histogram + Expert Mask
// ============================================================================

// ============================================================================
// Test: recordHistogramDevice() + syncHistogramToHost()
// ============================================================================

TEST(Test__ROCmMoEKernel, Histogram_RecordAndSync)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 8;
    const int top_k = 2;
    const int num_experts = 8;
    const int layer_idx = 0;

    // Known routing indices: each token picks 2 experts
    // Token 0: experts 0, 1
    // Token 1: experts 2, 3
    // Token 2: experts 0, 2
    // Token 3: experts 1, 3
    // Token 4: experts 4, 5
    // Token 5: experts 6, 7
    // Token 6: experts 0, 7
    // Token 7: experts 3, 5
    std::vector<int> routing_indices = {
        0, 1, 2, 3, 0, 2, 1, 3, 4, 5, 6, 7, 0, 7, 3, 5};

    // Expected histogram: count occurrences of each expert
    // Expert 0: 3 (tokens 0, 2, 6)
    // Expert 1: 2 (tokens 0, 3)
    // Expert 2: 2 (tokens 1, 2)
    // Expert 3: 3 (tokens 1, 3, 7)
    // Expert 4: 1 (token 4)
    // Expert 5: 2 (tokens 4, 7)
    // Expert 6: 1 (token 5)
    // Expert 7: 2 (tokens 5, 6)
    std::vector<uint64_t> expected_counts = {3, 2, 2, 3, 1, 2, 1, 2};

    // Upload routing indices to device
    int *d_indices = nullptr;
    (void)hipMalloc(&d_indices, routing_indices.size() * sizeof(int));
    (void)hipMemcpy(d_indices, routing_indices.data(),
              routing_indices.size() * sizeof(int), hipMemcpyHostToDevice);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    gpu_kernel.recordHistogramDevice(d_indices, seq_len, top_k, layer_idx);

    // Sync histogram to host
    std::vector<uint64_t> host_counts(num_experts, 0);
    gpu_kernel.syncHistogramToHost(host_counts.data(), layer_idx, num_experts);

    (void)hipFree(d_indices);

    // Verify counts
    uint64_t total_count = 0;
    for (int e = 0; e < num_experts; ++e)
    {
        total_count += host_counts[e];
        EXPECT_EQ(host_counts[e], expected_counts[e])
            << "Expert " << e << " count mismatch: got " << host_counts[e]
            << " expected " << expected_counts[e];
    }

    uint64_t expected_total = static_cast<uint64_t>(seq_len) * top_k;
    EXPECT_EQ(total_count, expected_total)
        << "Total histogram count mismatch";

    std::cout << "[Histogram_RecordAndSync] total_count=" << total_count
              << " expected=" << expected_total << std::endl;
}

// ============================================================================
// Test: resetHistogramDevice()
// ============================================================================

TEST(Test__ROCmMoEKernel, Histogram_Reset)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 4;
    const int top_k = 2;
    const int num_experts = 8;
    const int layer_idx = 0;

    // Record some histogram data
    std::vector<int> routing_indices = {0, 1, 2, 3, 4, 5, 6, 7};
    int *d_indices = nullptr;
    (void)hipMalloc(&d_indices, routing_indices.size() * sizeof(int));
    (void)hipMemcpy(d_indices, routing_indices.data(),
              routing_indices.size() * sizeof(int), hipMemcpyHostToDevice);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    gpu_kernel.recordHistogramDevice(d_indices, seq_len, top_k, layer_idx);

    // Verify something was recorded
    std::vector<uint64_t> counts_before(num_experts, 0);
    gpu_kernel.syncHistogramToHost(counts_before.data(), layer_idx, num_experts);
    uint64_t sum_before = 0;
    for (auto c : counts_before)
        sum_before += c;
    ASSERT_GT(sum_before, 0u) << "Histogram should have non-zero counts before reset";

    // Reset
    gpu_kernel.resetHistogramDevice(layer_idx, num_experts);

    // Sync and verify all zero
    std::vector<uint64_t> counts_after(num_experts, 99);
    gpu_kernel.syncHistogramToHost(counts_after.data(), layer_idx, num_experts);

    for (int e = 0; e < num_experts; ++e)
    {
        EXPECT_EQ(counts_after[e], 0u)
            << "Expert " << e << " should be 0 after reset, got " << counts_after[e];
    }

    (void)hipFree(d_indices);

    std::cout << "[Histogram_Reset] sum_before=" << sum_before
              << " sum_after=0 (all zeroed)" << std::endl;
}

// ============================================================================
// Test: Expert mask zeros out weights for inactive experts
// ============================================================================

TEST(Test__ROCmMoEKernel, ExpertMask_ApplyZerosWeights)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 8;
    const int top_k = 4;
    const int num_experts = 8;
    const int total_slots = seq_len * top_k;

    // Create routing indices — each token picks 4 experts
    std::vector<int> routing_indices(total_slots);
    std::vector<float> routing_weights(total_slots);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> expert_dist(0, num_experts - 1);
    std::uniform_real_distribution<float> weight_dist(0.05f, 0.5f);
    for (int i = 0; i < total_slots; ++i)
    {
        routing_indices[i] = expert_dist(gen);
        routing_weights[i] = weight_dist(gen);
    }

    // Save original weights for comparison
    std::vector<float> original_weights = routing_weights;

    // Expert mask: experts 0,1 active, experts 2-7 inactive
    std::vector<bool> mask(num_experts, false);
    mask[0] = true;
    mask[1] = true;

    // Upload to device
    int *d_indices = nullptr;
    float *d_weights = nullptr;
    (void)hipMalloc(&d_indices, total_slots * sizeof(int));
    (void)hipMalloc(&d_weights, total_slots * sizeof(float));
    (void)hipMemcpy(d_indices, routing_indices.data(), total_slots * sizeof(int), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_weights, routing_weights.data(), total_slots * sizeof(float), hipMemcpyHostToDevice);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    // Need to convert std::vector<bool> to a contiguous bool array
    std::vector<char> mask_bytes(num_experts);
    for (int i = 0; i < num_experts; ++i)
        mask_bytes[i] = mask[i] ? 1 : 0;
    gpu_kernel.updateExpertMaskDevice(reinterpret_cast<const bool *>(mask_bytes.data()), num_experts);
    gpu_kernel.applyExpertMaskDevice(d_weights, d_indices, seq_len, top_k);
    (void)hipDeviceSynchronize();

    // Read back weights
    std::vector<float> result_weights(total_slots);
    (void)hipMemcpy(result_weights.data(), d_weights, total_slots * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_indices);
    (void)hipFree(d_weights);

    // Verify
    int zeroed_count = 0;
    int unchanged_count = 0;
    for (int i = 0; i < total_slots; ++i)
    {
        int expert = routing_indices[i];
        if (expert == 0 || expert == 1)
        {
            EXPECT_FLOAT_EQ(result_weights[i], original_weights[i])
                << "Active expert " << expert << " weight at slot " << i
                << " should be unchanged";
            if (result_weights[i] == original_weights[i])
                ++unchanged_count;
        }
        else
        {
            EXPECT_FLOAT_EQ(result_weights[i], 0.0f)
                << "Inactive expert " << expert << " weight at slot " << i
                << " should be zeroed";
            if (result_weights[i] == 0.0f)
                ++zeroed_count;
        }
    }

    std::cout << "[ExpertMask_ApplyZerosWeights] zeroed=" << zeroed_count
              << " unchanged=" << unchanged_count
              << " total=" << total_slots << std::endl;
}

// ============================================================================
// Test: All-active expert mask leaves weights unchanged
// ============================================================================

TEST(Test__ROCmMoEKernel, ExpertMask_AllActiveNoChange)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 8;
    const int top_k = 2;
    const int num_experts = 8;
    const int total_slots = seq_len * top_k;

    // Create routing indices and weights
    std::vector<int> routing_indices(total_slots);
    std::vector<float> routing_weights(total_slots);
    std::mt19937 gen(123);
    std::uniform_int_distribution<int> expert_dist(0, num_experts - 1);
    std::uniform_real_distribution<float> weight_dist(0.05f, 0.5f);
    for (int i = 0; i < total_slots; ++i)
    {
        routing_indices[i] = expert_dist(gen);
        routing_weights[i] = weight_dist(gen);
    }
    std::vector<float> original_weights = routing_weights;

    // All experts active
    std::vector<char> mask_bytes(num_experts, 1);

    // Upload to device
    int *d_indices = nullptr;
    float *d_weights = nullptr;
    (void)hipMalloc(&d_indices, total_slots * sizeof(int));
    (void)hipMalloc(&d_weights, total_slots * sizeof(float));
    (void)hipMemcpy(d_indices, routing_indices.data(), total_slots * sizeof(int), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_weights, routing_weights.data(), total_slots * sizeof(float), hipMemcpyHostToDevice);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    gpu_kernel.updateExpertMaskDevice(reinterpret_cast<const bool *>(mask_bytes.data()), num_experts);
    gpu_kernel.applyExpertMaskDevice(d_weights, d_indices, seq_len, top_k);
    (void)hipDeviceSynchronize();

    // Read back
    std::vector<float> result_weights(total_slots);
    (void)hipMemcpy(result_weights.data(), d_weights, total_slots * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_indices);
    (void)hipFree(d_weights);

    // Verify all weights unchanged
    int mismatches = 0;
    for (int i = 0; i < total_slots; ++i)
    {
        if (result_weights[i] != original_weights[i])
        {
            ++mismatches;
            ADD_FAILURE() << "Weight at slot " << i << " changed: "
                          << original_weights[i] << " -> " << result_weights[i];
        }
    }

    EXPECT_EQ(mismatches, 0) << "All-active mask should leave all weights unchanged";

    std::cout << "[ExpertMask_AllActiveNoChange] mismatches=" << mismatches
              << " total_slots=" << total_slots << std::endl;
}

// ============================================================================
// Phase 3 Tests: Device-Side Token Grouping
// ============================================================================

// ============================================================================
// Test: groupTokensByExpertDevice() — basic correctness
// ============================================================================

TEST(Test__ROCmMoEKernel, GroupTokensByExpert_Basic)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 8;
    const int num_experts = 4;
    const int top_k = 2;
    const int total_slots = seq_len * top_k; // 16

    // Known routing indices (token → experts):
    // Token 0: experts {1, 3}
    // Token 1: experts {0, 2}
    // Token 2: experts {0, 1}
    // Token 3: experts {2, 3}
    // Token 4: experts {0, 3}
    // Token 5: experts {1, 2}
    // Token 6: experts {0, 1}
    // Token 7: experts {2, 3}
    std::vector<int> routing_indices = {
        1, 3, // token 0
        0, 2, // token 1
        0, 1, // token 2
        2, 3, // token 3
        0, 3, // token 4
        1, 2, // token 5
        0, 1, // token 6
        2, 3  // token 7
    };

    // Random routing weights
    std::vector<float> routing_weights(total_slots);
    fillRandom(routing_weights, 0.05f, 0.95f, 42);

    // Expected per-expert token sets:
    // Expert 0: tokens {1, 2, 4, 6} → count=4
    // Expert 1: tokens {0, 2, 5, 6} → count=4
    // Expert 2: tokens {1, 3, 5, 7} → count=4
    // Expert 3: tokens {0, 3, 4, 7} → count=4
    std::vector<int> expected_counts = {4, 4, 4, 4};

    // Build expected token→expert→weight mapping for verification
    // For each slot, record {expert_id, token_idx, weight}
    struct SlotInfo
    {
        int expert;
        int token;
        float weight;
    };
    std::vector<std::vector<SlotInfo>> expected_per_expert(num_experts);
    for (int s = 0; s < total_slots; ++s)
    {
        int token = s / top_k;
        int expert = routing_indices[s];
        expected_per_expert[expert].push_back({expert, token, routing_weights[s]});
    }

    // Upload to device
    int *d_routing_indices = nullptr;
    float *d_routing_weights = nullptr;
    (void)hipMalloc(&d_routing_indices, total_slots * sizeof(int));
    (void)hipMalloc(&d_routing_weights, total_slots * sizeof(float));
    (void)hipMemcpy(d_routing_indices, routing_indices.data(), total_slots * sizeof(int), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_routing_weights, routing_weights.data(), total_slots * sizeof(float), hipMemcpyHostToDevice);

    // Allocate output buffers
    int *d_expert_offsets = nullptr, *d_expert_counts = nullptr;
    int *d_grouped_indices = nullptr;
    float *d_grouped_weights = nullptr;
    (void)hipMalloc(&d_expert_offsets, num_experts * sizeof(int));
    (void)hipMalloc(&d_expert_counts, num_experts * sizeof(int));
    (void)hipMalloc(&d_grouped_indices, total_slots * sizeof(int));
    (void)hipMalloc(&d_grouped_weights, total_slots * sizeof(float));

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    bool ok = gpu_kernel.groupTokensByExpertDevice(
        d_routing_indices, d_routing_weights,
        seq_len, num_experts, top_k,
        d_expert_offsets, d_expert_counts,
        d_grouped_indices, d_grouped_weights);
    ASSERT_TRUE(ok) << "groupTokensByExpertDevice failed";

    (void)hipDeviceSynchronize();

    // D2H copy results
    std::vector<int> host_offsets(num_experts);
    std::vector<int> host_counts(num_experts);
    std::vector<int> host_grouped_indices(total_slots);
    std::vector<float> host_grouped_weights(total_slots);
    (void)hipMemcpy(host_offsets.data(), d_expert_offsets, num_experts * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipMemcpy(host_counts.data(), d_expert_counts, num_experts * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipMemcpy(host_grouped_indices.data(), d_grouped_indices, total_slots * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipMemcpy(host_grouped_weights.data(), d_grouped_weights, total_slots * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_routing_indices);
    (void)hipFree(d_routing_weights);
    (void)hipFree(d_expert_offsets);
    (void)hipFree(d_expert_counts);
    (void)hipFree(d_grouped_indices);
    (void)hipFree(d_grouped_weights);

    // Verify expert counts
    for (int e = 0; e < num_experts; ++e)
    {
        EXPECT_EQ(host_counts[e], expected_counts[e])
            << "Expert " << e << " count mismatch: got " << host_counts[e]
            << " expected " << expected_counts[e];
    }

    // Verify offsets are consistent with counts
    int running_offset = 0;
    for (int e = 0; e < num_experts; ++e)
    {
        EXPECT_EQ(host_offsets[e], running_offset)
            << "Expert " << e << " offset mismatch: got " << host_offsets[e]
            << " expected " << running_offset;
        running_offset += host_counts[e];
    }
    EXPECT_EQ(running_offset, total_slots) << "Total grouped slots mismatch";

    // Verify each expert's group contains the right tokens with the right weights
    bool all_matched = true;
    for (int e = 0; e < num_experts; ++e)
    {
        int offset = host_offsets[e];
        int count = host_counts[e];

        // Collect actual grouped tokens/weights for this expert
        std::vector<std::pair<int, float>> actual_group;
        for (int i = 0; i < count; ++i)
        {
            actual_group.push_back({host_grouped_indices[offset + i],
                                    host_grouped_weights[offset + i]});
        }

        // For each expected entry, verify it exists in the actual group
        for (const auto &expected : expected_per_expert[e])
        {
            bool found = false;
            for (auto &[tok, wt] : actual_group)
            {
                if (tok == expected.token && wt == expected.weight)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                ADD_FAILURE() << "Expert " << e << ": expected token " << expected.token
                              << " with weight " << expected.weight << " not found in group";
                all_matched = false;
            }
        }
    }

    std::cout << "[GroupTokensByExpert_Basic] expert_counts=["
              << host_counts[0] << "," << host_counts[1] << ","
              << host_counts[2] << "," << host_counts[3]
              << "] total=" << total_slots
              << " all_matched=" << (all_matched ? "true" : "false") << std::endl;
}

TEST(Test__ROCmMoEKernel, SmallFloatGrouping_VerifierSizedRoutesMatchHostGrouping)
{
    SKIP_IF_NO_ROCM();

    constexpr int seq_len = 2;
    constexpr int num_experts = 16;
    constexpr int top_k = 8;
    constexpr int total_slots = seq_len * top_k;

    std::vector<float> routing_indices(static_cast<size_t>(total_slots));
    std::vector<float> routing_weights(static_cast<size_t>(total_slots));
    std::vector<std::vector<std::pair<int, float>>> expected_per_expert(static_cast<size_t>(num_experts));
    for (int slot = 0; slot < total_slots; ++slot)
    {
        const int token = slot / top_k;
        const int expert = (slot * 5 + token * 3) % num_experts;
        const float weight = 0.03125f * static_cast<float>(slot + 1);
        routing_indices[static_cast<size_t>(slot)] = static_cast<float>(expert);
        routing_weights[static_cast<size_t>(slot)] = weight;
        expected_per_expert[static_cast<size_t>(expert)].push_back({token, weight});
    }

    float *d_routing_indices = nullptr;
    float *d_routing_weights = nullptr;
    int *d_expert_offsets = nullptr;
    int *d_expert_counts = nullptr;
    int *d_grouped_indices = nullptr;
    int *d_original_to_grouped = nullptr;
    int *d_original_expert_ids = nullptr;
    float *d_grouped_weights = nullptr;
    int *d_active_experts = nullptr;
    const int max_active_experts = std::min(total_slots, num_experts);

    ASSERT_EQ(hipMalloc(&d_routing_indices, total_slots * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_routing_weights, total_slots * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_expert_offsets, num_experts * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_expert_counts, num_experts * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_grouped_indices, total_slots * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_original_to_grouped, total_slots * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_original_expert_ids, total_slots * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_grouped_weights, total_slots * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_active_experts, max_active_experts * sizeof(int)), hipSuccess);

    ASSERT_EQ(hipMemcpy(d_routing_indices, routing_indices.data(),
                        total_slots * sizeof(float), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(d_routing_weights, routing_weights.data(),
                        total_slots * sizeof(float), hipMemcpyHostToDevice),
              hipSuccess);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    ASSERT_TRUE(hipMoE_group_tokens_small_float(
        d_routing_indices,
        d_routing_weights,
        d_expert_counts,
        d_expert_offsets,
        d_grouped_indices,
        d_original_to_grouped,
        d_original_expert_ids,
        d_grouped_weights,
        d_active_experts,
        total_slots,
        num_experts,
        top_k,
        max_active_experts,
        /*device_idx=*/0,
        stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

    std::vector<int> host_offsets(static_cast<size_t>(num_experts));
    std::vector<int> host_counts(static_cast<size_t>(num_experts));
    std::vector<int> host_grouped_indices(static_cast<size_t>(total_slots));
    std::vector<int> host_original_to_grouped(static_cast<size_t>(total_slots));
    std::vector<int> host_original_expert_ids(static_cast<size_t>(total_slots));
    std::vector<float> host_grouped_weights(static_cast<size_t>(total_slots));
    std::vector<int> host_active_experts(static_cast<size_t>(max_active_experts));
    ASSERT_EQ(hipMemcpy(host_offsets.data(), d_expert_offsets,
                        num_experts * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_counts.data(), d_expert_counts,
                        num_experts * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_grouped_indices.data(), d_grouped_indices,
                        total_slots * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_original_to_grouped.data(), d_original_to_grouped,
                        total_slots * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_original_expert_ids.data(), d_original_expert_ids,
                        total_slots * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_grouped_weights.data(), d_grouped_weights,
                        total_slots * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_active_experts.data(), d_active_experts,
                        max_active_experts * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);

    ASSERT_EQ(hipFree(d_routing_indices), hipSuccess);
    ASSERT_EQ(hipFree(d_routing_weights), hipSuccess);
    ASSERT_EQ(hipFree(d_expert_offsets), hipSuccess);
    ASSERT_EQ(hipFree(d_expert_counts), hipSuccess);
    ASSERT_EQ(hipFree(d_grouped_indices), hipSuccess);
    ASSERT_EQ(hipFree(d_original_to_grouped), hipSuccess);
    ASSERT_EQ(hipFree(d_original_expert_ids), hipSuccess);
    ASSERT_EQ(hipFree(d_grouped_weights), hipSuccess);
    ASSERT_EQ(hipFree(d_active_experts), hipSuccess);

    int running_offset = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const auto &expected = expected_per_expert[static_cast<size_t>(expert)];
        ASSERT_EQ(host_counts[static_cast<size_t>(expert)], static_cast<int>(expected.size()))
            << "expert=" << expert;
        ASSERT_EQ(host_offsets[static_cast<size_t>(expert)], running_offset)
            << "expert=" << expert;

        std::vector<std::pair<int, float>> actual;
        actual.reserve(expected.size());
        for (int i = 0; i < host_counts[static_cast<size_t>(expert)]; ++i)
        {
            const int grouped = running_offset + i;
            actual.push_back({host_grouped_indices[static_cast<size_t>(grouped)],
                              host_grouped_weights[static_cast<size_t>(grouped)]});
        }

        for (const auto &entry : expected)
        {
            EXPECT_NE(std::find(actual.begin(), actual.end(), entry), actual.end())
                << "expert=" << expert << " token=" << entry.first
                << " weight=" << entry.second;
        }

        running_offset += host_counts[static_cast<size_t>(expert)];
    }
    EXPECT_EQ(running_offset, total_slots);

    for (int slot = 0; slot < total_slots; ++slot)
    {
        const int token = slot / top_k;
        const int expert = static_cast<int>(routing_indices[static_cast<size_t>(slot)]);
        const int grouped = host_original_to_grouped[static_cast<size_t>(slot)];
        EXPECT_EQ(host_original_expert_ids[static_cast<size_t>(slot)], expert)
            << "slot=" << slot;
        ASSERT_GE(grouped, 0) << "slot=" << slot;
        ASSERT_LT(grouped, total_slots) << "slot=" << slot;
        EXPECT_GE(grouped, host_offsets[static_cast<size_t>(expert)])
            << "slot=" << slot << " expert=" << expert;
        EXPECT_LT(grouped, host_offsets[static_cast<size_t>(expert)] +
                               host_counts[static_cast<size_t>(expert)])
            << "slot=" << slot << " expert=" << expert;
        EXPECT_EQ(host_grouped_indices[static_cast<size_t>(grouped)], token)
            << "slot=" << slot;
        EXPECT_FLOAT_EQ(host_grouped_weights[static_cast<size_t>(grouped)],
                        routing_weights[static_cast<size_t>(slot)])
            << "slot=" << slot;
    }

    std::vector<int> expected_active_experts;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        if (!expected_per_expert[static_cast<size_t>(expert)].empty())
            expected_active_experts.push_back(expert);
    }
    ASSERT_LE(expected_active_experts.size(), host_active_experts.size());
    for (size_t i = 0; i < expected_active_experts.size(); ++i)
        EXPECT_EQ(host_active_experts[i], expected_active_experts[i]) << "active slot " << i;
    for (size_t i = expected_active_experts.size(); i < host_active_experts.size(); ++i)
        EXPECT_EQ(host_active_experts[i], -1) << "inactive slot " << i;
}

TEST(Test__ROCmMoEKernel, SmallFloatGrouping_RejectsNullStream)
{
    SKIP_IF_NO_ROCM();

    EXPECT_FALSE(hipMoE_group_tokens_small_float(
        /*routing_indices=*/nullptr,
        /*routing_weights=*/nullptr,
        /*expert_counts=*/nullptr,
        /*expert_offsets=*/nullptr,
        /*grouped_token_indices=*/nullptr,
        /*original_to_grouped=*/nullptr,
        /*original_expert_ids=*/nullptr,
        /*grouped_weights=*/nullptr,
        /*active_expert_ids=*/nullptr,
        /*total_slots=*/8,
        /*num_experts=*/16,
        /*top_k=*/8,
        /*max_active_experts=*/8,
        /*device_idx=*/0,
        /*stream=*/nullptr));
}

TEST(Test__ROCmMoEKernel, SmallMGateLogits_RejectsNullStream)
{
    SKIP_IF_NO_ROCM();

    float *d_hidden = nullptr;
    float *d_gate = nullptr;
    float *d_logits = nullptr;
    ASSERT_EQ(hipMalloc(&d_hidden, 2 * 32 * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_gate, 8 * 32 * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_logits, 2 * 8 * sizeof(float)), hipSuccess);

    EXPECT_FALSE(hipMoE_gate_logits_small_m(
        d_hidden,
        d_gate,
        d_logits,
        /*seq_len=*/2,
        /*d_model=*/32,
        /*num_experts=*/8,
        /*device_idx=*/0,
        /*stream=*/nullptr));

    (void)hipFree(d_hidden);
    (void)hipFree(d_gate);
    (void)hipFree(d_logits);
}

TEST(Test__ROCmMoEKernel, SmallMGateLogits_ModelShapeMatchesSingleTokenLaunches)
{
    SKIP_IF_NO_ROCM();

    constexpr int d_model = 2048;
    constexpr int num_experts = 256;
    const DeviceId device = DeviceId::rocm(0);

    auto gate_weights = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(num_experts), static_cast<size_t>(d_model)},
        -0.5f, 0.5f, 20260607);
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));

    for (int seq_len : {1, 2, 3, 4})
    {
        auto hidden = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
            -0.5f, 0.5f, 20260606 + seq_len);
        auto fused_logits = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(num_experts)});
        auto row_logits = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(num_experts)});

        ASSERT_TRUE(hidden->ensureOnDevice(device));
        ASSERT_TRUE(fused_logits->ensureOnDevice(device));
        ASSERT_TRUE(row_logits->ensureOnDevice(device));

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
        ASSERT_TRUE(hipMoE_gate_logits_small_m(
            static_cast<const float *>(hidden->gpu_data_ptr()),
            static_cast<const float *>(gate_weights->gpu_data_ptr()),
            static_cast<float *>(fused_logits->gpu_data_ptr()),
            seq_len, d_model, num_experts,
            /*device_idx=*/0,
            stream));
        for (int row = 0; row < seq_len; ++row)
        {
            ASSERT_TRUE(hipMoE_gate_logits_single_token(
                static_cast<const float *>(hidden->gpu_data_ptr()) + static_cast<size_t>(row) * d_model,
                static_cast<const float *>(gate_weights->gpu_data_ptr()),
                static_cast<float *>(row_logits->gpu_data_ptr()) + static_cast<size_t>(row) * num_experts,
                d_model, num_experts,
                /*device_idx=*/0,
                stream));
        }
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

        fused_logits->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        row_logits->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        const float *fused = fused_logits->data();
        const float *rowwise = row_logits->data();

        float max_abs_diff = 0.0f;
        for (int i = 0; i < seq_len * num_experts; ++i)
            max_abs_diff = std::max(max_abs_diff, std::fabs(fused[i] - rowwise[i]));
        EXPECT_LE(max_abs_diff, 1e-5f) << "seq_len=" << seq_len;
    }
}

TEST(Test__ROCmMoEKernel, SmallMBF16GateLogits_ModelShapeMatchesSingleTokenLaunches)
{
    SKIP_IF_NO_ROCM();

    constexpr int d_model = 2048;
    constexpr int num_experts = 256;
    const DeviceId device = DeviceId::rocm(0);

    std::vector<float> gate_weights_fp32(
        static_cast<size_t>(num_experts) * d_model);
    fillRandom(gate_weights_fp32, -0.5f, 0.5f, 20260608);
    auto gate_weights = TestTensorFactory::createBF16(
        {static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    gate_weights->from_fp32(gate_weights_fp32.data(), gate_weights_fp32.size());
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));

    for (int seq_len : {1, 2, 3, 4})
    {
        auto hidden = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
            -0.5f, 0.5f, 20260609 + seq_len);
        auto fused_logits = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(num_experts)});
        auto row_logits = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(num_experts)});

        ASSERT_TRUE(hidden->ensureOnDevice(device));
        ASSERT_TRUE(fused_logits->ensureOnDevice(device));
        ASSERT_TRUE(row_logits->ensureOnDevice(device));

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
        ASSERT_TRUE(hipMoE_gate_logits_small_m_bf16_weights(
            static_cast<const float *>(hidden->gpu_data_ptr()),
            gate_weights->gpu_data_ptr(),
            static_cast<float *>(fused_logits->gpu_data_ptr()),
            seq_len, d_model, num_experts,
            /*device_idx=*/0,
            stream));
        for (int row = 0; row < seq_len; ++row)
        {
            ASSERT_TRUE(hipMoE_gate_logits_single_token_bf16_weights(
                static_cast<const float *>(hidden->gpu_data_ptr()) +
                    static_cast<size_t>(row) * d_model,
                gate_weights->gpu_data_ptr(),
                static_cast<float *>(row_logits->gpu_data_ptr()) +
                    static_cast<size_t>(row) * num_experts,
                d_model, num_experts,
                /*device_idx=*/0,
                stream));
        }
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

        fused_logits->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        row_logits->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        const float *fused = fused_logits->data();
        const float *rowwise = row_logits->data();

        float max_abs_diff = 0.0f;
        for (int i = 0; i < seq_len * num_experts; ++i)
            max_abs_diff = std::max(max_abs_diff, std::fabs(fused[i] - rowwise[i]));
        EXPECT_LE(max_abs_diff, 1e-3f) << "seq_len=" << seq_len;
    }
}

TEST(Test__ROCmMoEKernel, VerifierRowsBF16RouteUsesDecodeEquivalentRouter)
{
    SKIP_IF_NO_ROCM();

    constexpr int seq_len = 4;
    constexpr int d_model = 64;
    constexpr int num_experts = 16;
    constexpr int top_k = 4;
    const DeviceId device = DeviceId::rocm(0);

    auto hidden = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
        -0.25f, 0.25f, 20260610);
    std::vector<float> gate_weights_fp32(
        static_cast<size_t>(num_experts) * d_model);
    fillRandom(gate_weights_fp32, -0.25f, 0.25f, 20260611);
    auto gate_weights = TestTensorFactory::createBF16(
        {static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    gate_weights->from_fp32(gate_weights_fp32.data(), gate_weights_fp32.size());
    auto routing_indices = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
    auto routing_weights = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ScopedEnvOverride perf_stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(
        moe_kernel,
        /*max_seq_len=*/seq_len,
        d_model,
        /*intermediate=*/128,
        num_experts,
        top_k);
    ASSERT_TRUE(moe_kernel.routeVerifierRowsDecodeEquivalent(
        hidden.get(),
        gate_weights.get(),
        seq_len,
        d_model,
        num_experts,
        top_k,
        /*normalize_weights=*/true,
        routing_indices.get(),
        routing_weights.get()));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    double decode_equivalent_router_calls = 0.0;
    bool saw_grouped_route = false;
    for (const auto &record : PerfStatsCollector::snapshot({"kernel.rocm_moe_decode_equivalent_small_m_router_calls"}))
    {
        decode_equivalent_router_calls += record.value;
        const auto route_it = record.tags.find("route");
        saw_grouped_route = saw_grouped_route ||
                            (route_it != record.tags.end() &&
                             route_it->second == "grouped_decode_equivalent");
    }
    EXPECT_GT(decode_equivalent_router_calls, 0.0)
        << "BF16 verifier-sized routing should use the decode-equivalent grouped router path";
    EXPECT_TRUE(saw_grouped_route)
        << "BF16 verifier routing must advertise the grouped decode-equivalent router path";
    PerfStatsCollector::reset();

    ASSERT_TRUE(routing_indices->ensureOnHost());
    ASSERT_TRUE(routing_weights->ensureOnHost());
    const float *indices = routing_indices->data();
    const float *weights = routing_weights->data();
    ASSERT_NE(indices, nullptr);
    ASSERT_NE(weights, nullptr);
    for (int slot = 0; slot < seq_len * top_k; ++slot)
    {
        const int expert = static_cast<int>(indices[slot]);
        EXPECT_GE(expert, 0);
        EXPECT_LT(expert, num_experts);
        EXPECT_TRUE(std::isfinite(weights[slot]));
    }
}

TEST(Test__ROCmMoEKernel, SoftmaxTopKParallelSelectionPreservesTieOrder)
{
    SKIP_IF_NO_ROCM();

    constexpr int seq_len = 2;
    constexpr int num_experts = 16;
    constexpr int top_k = 4;
    const DeviceId device = DeviceId::rocm(0);

    auto logits = TestTensorFactory::createFP32({seq_len, num_experts});
    auto indices = TestTensorFactory::createINT32({seq_len, top_k});
    auto weights = TestTensorFactory::createFP32({seq_len, top_k});

    // Row 0 has two equal winner pairs; row 1 is all ties. The expected expert
    // order therefore proves that the parallel reduction keeps the original
    // ascending serial-scan tie semantics.
    std::fill(logits->mutable_data(), logits->mutable_data() + seq_len * num_experts, 0.0f);
    logits->mutable_data()[2] = 5.0f;
    logits->mutable_data()[4] = 5.0f;
    logits->mutable_data()[1] = 4.0f;
    logits->mutable_data()[7] = 4.0f;
    for (int expert = 0; expert < num_experts; ++expert)
        logits->mutable_data()[num_experts + expert] = 1.0f;

    ASSERT_TRUE(logits->ensureOnDevice(device));
    ASSERT_TRUE(indices->ensureOnDevice(device));
    ASSERT_TRUE(weights->ensureOnDevice(device));

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    ASSERT_TRUE(hipMoE_softmax_topk(
        static_cast<float *>(logits->gpu_data_ptr()),
        static_cast<int *>(indices->gpu_data_ptr()),
        static_cast<float *>(weights->gpu_data_ptr()),
        seq_len,
        num_experts,
        top_k,
        /*normalize_weights=*/true,
        /*device_idx=*/0,
        stream,
        nullptr));
    indices->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    weights->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    ASSERT_TRUE(indices->ensureOnHost(stream));
    ASSERT_TRUE(weights->ensureOnHost(stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

    const int *actual_indices = indices->int32_data();
    const float *actual_weights = weights->data();

    const int expected_indices[seq_len * top_k] = {2, 4, 1, 7, 0, 1, 2, 3};
    for (int i = 0; i < seq_len * top_k; ++i)
        EXPECT_EQ(actual_indices[i], expected_indices[i]) << "slot=" << i;

    const float row0_pair_sum = 2.0f * std::exp(5.0f) + 2.0f * std::exp(4.0f);
    const float expected_row0[4] = {
        std::exp(5.0f) / row0_pair_sum,
        std::exp(5.0f) / row0_pair_sum,
        std::exp(4.0f) / row0_pair_sum,
        std::exp(4.0f) / row0_pair_sum};
    for (int k = 0; k < top_k; ++k)
        EXPECT_NEAR(actual_weights[k], expected_row0[k], 1e-6f) << "row0 k=" << k;
    for (int k = 0; k < top_k; ++k)
        EXPECT_NEAR(actual_weights[top_k + k], 0.25f, 1e-6f) << "row1 k=" << k;
}

TEST(Test__ROCmMoEKernel, VerifierRowsRouteUsesDecodeEquivalentRouterAndMatchesHostReference)
{
    SKIP_IF_NO_ROCM();

    ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");

    constexpr int seq_len = 4;
    constexpr int d_model = 32;
    constexpr int num_experts = 12;
    constexpr int top_k = 4;
    const DeviceId device = DeviceId::rocm(0);

    auto hidden = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
        -0.25f, 0.25f, 20260604);
    auto gate_weights = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(num_experts), static_cast<size_t>(d_model)},
        -0.25f, 0.25f, 20260605);
    auto routing_indices = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
    auto routing_weights = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ScopedEnvOverride perf_stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);
    ASSERT_TRUE(moe_kernel.routeVerifierRowsDecodeEquivalent(
        hidden.get(),
        gate_weights.get(),
        seq_len,
        d_model,
        num_experts,
        top_k,
        /*normalize_weights=*/true,
        routing_indices.get(),
        routing_weights.get()));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    double decode_equivalent_router_calls = 0.0;
    bool saw_grouped_route = false;
    for (const auto &record : PerfStatsCollector::snapshot({"kernel.rocm_moe_decode_equivalent_small_m_router_calls"}))
    {
        decode_equivalent_router_calls += record.value;
        const auto route_it = record.tags.find("route");
        saw_grouped_route = saw_grouped_route ||
                            (route_it != record.tags.end() &&
                             route_it->second == "grouped_decode_equivalent");
    }
    EXPECT_GT(decode_equivalent_router_calls, 0.0)
        << "verifier-sized routing should use the decode-equivalent grouped router path";
    EXPECT_TRUE(saw_grouped_route)
        << "FP32 verifier routing must advertise the grouped decode-equivalent router path";
    PerfStatsCollector::reset();

    ASSERT_TRUE(routing_indices->ensureOnHost());
    ASSERT_TRUE(routing_weights->ensureOnHost());
    const float *actual_indices = routing_indices->data();
    const float *actual_weights = routing_weights->data();
    ASSERT_NE(actual_indices, nullptr);
    ASSERT_NE(actual_weights, nullptr);

    const float *h = hidden->data();
    const float *g = gate_weights->data();
    for (int token = 0; token < seq_len; ++token)
    {
        std::vector<float> probs(static_cast<size_t>(num_experts));
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int expert = 0; expert < num_experts; ++expert)
        {
            float dot = 0.0f;
            for (int col = 0; col < d_model; ++col)
                dot += h[token * d_model + col] * g[expert * d_model + col];
            probs[static_cast<size_t>(expert)] = dot;
            max_logit = std::max(max_logit, dot);
        }

        float sum = 0.0f;
        for (float &prob : probs)
        {
            prob = std::exp(prob - max_logit);
            sum += prob;
        }
        for (float &prob : probs)
            prob /= sum;

        std::vector<int> expected_indices;
        std::vector<float> expected_weights;
        expected_indices.reserve(top_k);
        expected_weights.reserve(top_k);
        for (int k = 0; k < top_k; ++k)
        {
            int best_idx = 0;
            float best_val = -1.0f;
            for (int expert = 0; expert < num_experts; ++expert)
            {
                const bool already_picked =
                    std::find(expected_indices.begin(), expected_indices.end(), expert) != expected_indices.end();
                if (!already_picked && probs[static_cast<size_t>(expert)] > best_val)
                {
                    best_val = probs[static_cast<size_t>(expert)];
                    best_idx = expert;
                }
            }
            expected_indices.push_back(best_idx);
            expected_weights.push_back(best_val);
        }

        float topk_sum = 0.0f;
        for (float weight : expected_weights)
            topk_sum += weight;
        for (float &weight : expected_weights)
            weight /= topk_sum;

        for (int k = 0; k < top_k; ++k)
        {
            const int offset = token * top_k + k;
            EXPECT_EQ(static_cast<int>(actual_indices[static_cast<size_t>(offset)]),
                      expected_indices[static_cast<size_t>(k)])
                << "token=" << token << " k=" << k;
            EXPECT_NEAR(actual_weights[static_cast<size_t>(offset)],
                        expected_weights[static_cast<size_t>(k)],
                        1e-5f)
                << "token=" << token << " k=" << k;
        }
    }
}

TEST(Test__ROCmMoEKernel, VerifierRowsRouteMatchesSerialDecodeRouter)
{
    SKIP_IF_NO_ROCM();

    ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "1");
    ScopedEnvOverride perf_stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int d_model = 2048;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    ROCmMoEKernel moe_kernel(0);
    static_cast<ITensorKernel &>(moe_kernel).setGPUStream(stream);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    DeviceMoERuntimeTable::Config config;
    config.device_id = device;
    config.num_layers = 1;
    config.num_experts = num_experts;
    config.top_k = top_k;
    config.mirror_to_device = true;
    MoERuntimeTable runtime_table(config);

    MoEPlacementUpdate update;
    update.epoch = 1;
    update.expert_count = num_experts;
    update.experts.resize(num_experts);
    update.local_compute_mask.assign(num_experts, 0);
    update.replica_role.resize(num_experts, 0);
    for (int expert = 0; expert < num_experts; ++expert)
        update.experts[expert].logical_expert_id = expert;
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream));

    std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model);
    for (size_t i = 0; i < gate_values.size(); ++i)
        gate_values[i] = 0.035f * std::sin(0.011f * static_cast<float>(i + 13)) +
                         0.027f * std::cos(0.017f * static_cast<float>(i + 19)) +
                         0.0007f * static_cast<float>(static_cast<int>(i % 23) - 11);
    auto gate_weights = TestTensorFactory::createFP32(
        {static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    std::copy(gate_values.begin(), gate_values.end(), gate_weights->mutable_data());
    ASSERT_TRUE(gate_weights->ensureOnDevice(device, stream));

    for (int seq_len : {1, 2, 3, 4})
    {
        std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
        for (size_t i = 0; i < hidden_values.size(); ++i)
            hidden_values[i] = 0.05f * std::sin(0.007f * static_cast<float>(i + 1 + seq_len)) -
                               0.031f * std::cos(0.013f * static_cast<float>(i + 5)) +
                               0.0013f * static_cast<float>(static_cast<int>(i % 17) - 8);

        auto hidden = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        auto batched_indices = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
        auto batched_weights = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
        std::copy(hidden_values.begin(), hidden_values.end(), hidden->mutable_data());
        ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
        ASSERT_TRUE(batched_indices->allocateOnDevice(device, stream));
        ASSERT_TRUE(batched_weights->allocateOnDevice(device, stream));

        ASSERT_TRUE(moe_kernel.routeVerifierRowsDecodeEquivalent(
            hidden.get(),
            gate_weights.get(),
            seq_len,
            d_model,
            num_experts,
            top_k,
            /*normalize_weights=*/true,
            batched_indices.get(),
            batched_weights.get()))
            << "seq_len=" << seq_len;
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        std::vector<float> batched_indices_host(static_cast<size_t>(seq_len) * top_k);
        std::vector<float> batched_weights_host(static_cast<size_t>(seq_len) * top_k);
        ASSERT_EQ(hipMemcpyAsync(batched_indices_host.data(), batched_indices->gpu_data_ptr(),
                                 batched_indices_host.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(batched_weights_host.data(), batched_weights->gpu_data_ptr(),
                                 batched_weights_host.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        std::vector<float> serial_indices(static_cast<size_t>(seq_len) * top_k);
        std::vector<float> serial_weights(static_cast<size_t>(seq_len) * top_k);
        for (int row = 0; row < seq_len; ++row)
        {
            auto row_hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
            auto row_indices = TestTensorFactory::createFP32({1, static_cast<size_t>(top_k)});
            auto row_weights = TestTensorFactory::createFP32({1, static_cast<size_t>(top_k)});
            std::copy_n(hidden_values.data() + static_cast<size_t>(row) * d_model,
                        d_model,
                        row_hidden->mutable_data());
            ASSERT_TRUE(row_hidden->ensureOnDevice(device, stream));
            ASSERT_TRUE(row_indices->allocateOnDevice(device, stream));
            ASSERT_TRUE(row_weights->allocateOnDevice(device, stream));

            ASSERT_TRUE(moe_kernel.decodeRouteSelect(
                runtime_table.deviceLayerState(0),
                row_hidden.get(),
                gate_weights.get(),
                d_model,
                num_experts,
                top_k,
                /*normalize_weights=*/true,
                row_indices.get(),
                row_weights.get(),
                /*write_legacy_outputs=*/true,
                /*update_runtime_histogram=*/false))
                << "seq_len=" << seq_len << " row=" << row;

            ASSERT_EQ(hipMemcpyAsync(serial_indices.data() + static_cast<size_t>(row) * top_k,
                                     row_indices->gpu_data_ptr(),
                                     static_cast<size_t>(top_k) * sizeof(float),
                                     hipMemcpyDeviceToHost, stream),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(serial_weights.data() + static_cast<size_t>(row) * top_k,
                                     row_weights->gpu_data_ptr(),
                                     static_cast<size_t>(top_k) * sizeof(float),
                                     hipMemcpyDeviceToHost, stream),
                      hipSuccess);
        }
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        expectStrictTopKRowsEquivalent(
            batched_indices_host,
            batched_weights_host,
            serial_indices,
            serial_weights,
            seq_len,
            top_k);
    }

    bool saw_q8_grouped_route = false;
    for (const auto &record : PerfStatsCollector::snapshot({"kernel.rocm_moe_decode_equivalent_small_m_router_calls"}))
    {
        const auto route_it = record.tags.find("route");
        saw_q8_grouped_route = saw_q8_grouped_route ||
                               (route_it != record.tags.end() &&
                                route_it->second == "grouped_decode_equivalent_q8");
    }
    EXPECT_TRUE(saw_q8_grouped_route)
        << "Q8 verifier routing must exercise the economical grouped decode-equivalent router";
    PerfStatsCollector::reset();

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, GroupPrefillRoutesRuntimeState_DeterministicSmall)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int seq_len = 8;
    const int num_experts = 4;
    const int top_k = 2;
    const int total_slots = seq_len * top_k;

    std::vector<int> routing_indices = {
        1, 3,
        0, 2,
        0, 1,
        2, 3,
        0, 3,
        1, 2,
        0, 1,
        2, 3};
    std::vector<float> routing_indices_f32(static_cast<size_t>(total_slots));
    std::vector<float> routing_weights(static_cast<size_t>(total_slots));
    for (int i = 0; i < total_slots; ++i)
    {
        routing_indices_f32[static_cast<size_t>(i)] = static_cast<float>(routing_indices[static_cast<size_t>(i)]);
        routing_weights[static_cast<size_t>(i)] = 0.125f + 0.03125f * static_cast<float>(i);
    }

    auto routing_index_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weight_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_f32.begin(), routing_indices_f32.end(), routing_index_tensor->mutable_data());
    std::copy(routing_weights.begin(), routing_weights.end(), routing_weight_tensor->mutable_data());
    ASSERT_TRUE(routing_index_tensor->ensureOnDevice(device));
    ASSERT_TRUE(routing_weight_tensor->ensureOnDevice(device));

    DeviceMoERuntimeTable::Config config;
    config.device_id = device;
    config.num_layers = 1;
    config.num_experts = num_experts;
    config.top_k = top_k;
    config.mirror_to_device = true;
    config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(config);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    ASSERT_TRUE(gpu_kernel.groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_index_tensor.get(),
        routing_weight_tensor.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    const auto &state = runtime_table.hostLayerState(0);
    ASSERT_EQ(state.prefill_route_capacity, static_cast<uint32_t>(total_slots));

    std::vector<int> host_route_ids(total_slots);
    std::vector<float> host_route_weights(total_slots);
    std::vector<int> host_counts(num_experts);
    std::vector<int> host_offsets(num_experts);
    std::vector<int> host_grouped_ids(total_slots);
    std::vector<float> host_grouped_weights(total_slots);
    ASSERT_EQ(hipMemcpy(host_route_ids.data(), state.route_expert_ids,
                        host_route_ids.size() * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_route_weights.data(), state.route_weights,
                        host_route_weights.size() * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_counts.data(), state.expert_counts,
                        host_counts.size() * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_offsets.data(), state.expert_offsets,
                        host_offsets.size() * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_grouped_ids.data(), state.grouped_token_ids,
                        host_grouped_ids.size() * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_grouped_weights.data(), state.grouped_route_weights,
                        host_grouped_weights.size() * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);

    for (int slot = 0; slot < total_slots; ++slot)
    {
        EXPECT_EQ(host_route_ids[static_cast<size_t>(slot)], routing_indices[static_cast<size_t>(slot)]);
        EXPECT_FLOAT_EQ(host_route_weights[static_cast<size_t>(slot)], routing_weights[static_cast<size_t>(slot)]);
    }

    const std::vector<int> expected_counts = {4, 4, 4, 4};
    int running = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        EXPECT_EQ(host_counts[static_cast<size_t>(expert)], expected_counts[static_cast<size_t>(expert)]);
        EXPECT_EQ(host_offsets[static_cast<size_t>(expert)], running);
        running += expected_counts[static_cast<size_t>(expert)];
    }
    EXPECT_EQ(running, total_slots);

    std::vector<std::vector<std::pair<int, float>>> expected_per_expert(static_cast<size_t>(num_experts));
    for (int slot = 0; slot < total_slots; ++slot)
    {
        const int expert = routing_indices[static_cast<size_t>(slot)];
        expected_per_expert[static_cast<size_t>(expert)].push_back(
            {slot, routing_weights[static_cast<size_t>(slot)]});
    }

    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int offset = host_offsets[static_cast<size_t>(expert)];
        const auto &expected = expected_per_expert[static_cast<size_t>(expert)];
        ASSERT_EQ(static_cast<int>(expected.size()), host_counts[static_cast<size_t>(expert)]);
        for (int i = 0; i < static_cast<int>(expected.size()); ++i)
        {
            EXPECT_EQ(host_grouped_ids[static_cast<size_t>(offset + i)], expected[static_cast<size_t>(i)].first)
                << "expert=" << expert << " grouped row=" << i;
            EXPECT_FLOAT_EQ(host_grouped_weights[static_cast<size_t>(offset + i)],
                            expected[static_cast<size_t>(i)].second)
                << "expert=" << expert << " grouped row=" << i;
        }
    }
}

TEST(Test__ROCmMoEKernel, RuntimeOriginalToGroupedPreservesRouteSlotOrder)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int seq_len = 4;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int total_slots = seq_len * top_k;

    std::vector<float> routing_indices = {
        1, 3,
        0, 2,
        0, 1,
        2, 3};
    std::vector<float> routing_weights(static_cast<size_t>(total_slots));
    for (int slot = 0; slot < total_slots; ++slot)
        routing_weights[static_cast<size_t>(slot)] = 0.125f + 0.0625f * static_cast<float>(slot);

    auto routing_index_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weight_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices.begin(), routing_indices.end(), routing_index_tensor->mutable_data());
    std::copy(routing_weights.begin(), routing_weights.end(), routing_weight_tensor->mutable_data());
    ASSERT_TRUE(routing_index_tensor->ensureOnDevice(device));
    ASSERT_TRUE(routing_weight_tensor->ensureOnDevice(device));

    DeviceMoERuntimeTable::Config config;
    config.device_id = device;
    config.num_layers = 1;
    config.num_experts = num_experts;
    config.top_k = top_k;
    config.mirror_to_device = true;
    config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(config);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    ASSERT_TRUE(gpu_kernel.groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_index_tensor.get(),
        routing_weight_tensor.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    int *d_original_to_grouped = nullptr;
    ASSERT_EQ(hipMalloc(&d_original_to_grouped, total_slots * sizeof(int)), hipSuccess);
    ASSERT_TRUE(hipMoE_build_runtime_original_to_grouped(
        runtime_table.deviceLayerState(0),
        d_original_to_grouped,
        total_slots,
        total_slots,
        num_experts,
        top_k,
        0,
        gpu_kernel.getStream()));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<int> host_original_to_grouped(static_cast<size_t>(total_slots));
    std::vector<int> host_grouped_route_slots(static_cast<size_t>(total_slots));
    ASSERT_EQ(hipMemcpy(host_original_to_grouped.data(),
                        d_original_to_grouped,
                        host_original_to_grouped.size() * sizeof(int),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_grouped_route_slots.data(),
                        runtime_table.hostLayerState(0).grouped_token_ids,
                        host_grouped_route_slots.size() * sizeof(int),
                        hipMemcpyDeviceToHost),
              hipSuccess);

    std::vector<std::vector<int>> expected_route_slots_by_expert(static_cast<size_t>(num_experts));
    for (int slot = 0; slot < total_slots; ++slot)
    {
        const int expert = static_cast<int>(routing_indices[static_cast<size_t>(slot)]);
        expected_route_slots_by_expert[static_cast<size_t>(expert)].push_back(slot);
    }

    std::vector<int> expected_original_to_grouped(static_cast<size_t>(total_slots), -1);
    int grouped_slot = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        for (int route_slot : expected_route_slots_by_expert[static_cast<size_t>(expert)])
        {
            EXPECT_EQ(host_grouped_route_slots[static_cast<size_t>(grouped_slot)], route_slot)
                << "expert=" << expert << " grouped_slot=" << grouped_slot;
            expected_original_to_grouped[static_cast<size_t>(route_slot)] = grouped_slot;
            ++grouped_slot;
        }
    }
    ASSERT_EQ(grouped_slot, total_slots);

    for (int slot = 0; slot < total_slots; ++slot)
    {
        EXPECT_EQ(host_original_to_grouped[static_cast<size_t>(slot)],
                  expected_original_to_grouped[static_cast<size_t>(slot)])
            << "route slot " << slot;
    }

    EXPECT_EQ(hipFree(d_original_to_grouped), hipSuccess);
}

TEST(Test__ROCmMoEKernel, RuntimePrefillGatherScatter_ZeroCountExpertNoOps)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int seq_len = 4;
    const int d_model = 16;
    const int num_experts = 4;
    const int top_k = 2;
    const int total_slots = seq_len * top_k;

    std::vector<int> routing_indices = {
        0, 2,
        2, 0,
        0, 2,
        2, 0};
    std::vector<float> routing_indices_f32(static_cast<size_t>(total_slots));
    std::vector<float> routing_weights(static_cast<size_t>(total_slots), 0.5f);
    for (int i = 0; i < total_slots; ++i)
        routing_indices_f32[static_cast<size_t>(i)] = static_cast<float>(routing_indices[static_cast<size_t>(i)]);

    auto routing_index_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weight_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_f32.begin(), routing_indices_f32.end(), routing_index_tensor->mutable_data());
    std::copy(routing_weights.begin(), routing_weights.end(), routing_weight_tensor->mutable_data());
    ASSERT_TRUE(routing_index_tensor->ensureOnDevice(device));
    ASSERT_TRUE(routing_weight_tensor->ensureOnDevice(device));

    DeviceMoERuntimeTable::Config config;
    config.device_id = device;
    config.num_layers = 1;
    config.num_experts = num_experts;
    config.top_k = top_k;
    config.mirror_to_device = true;
    config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(config);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    ASSERT_TRUE(gpu_kernel.groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_index_tensor.get(),
        routing_weight_tensor.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    auto hidden = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    for (int i = 0; i < seq_len * d_model; ++i)
        hidden->mutable_data()[i] = 0.01f * static_cast<float>(i + 1);
    ASSERT_TRUE(hidden->ensureOnDevice(device));

    auto batch = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(batch->ensureOnDevice(device));
    ASSERT_TRUE(gpu_kernel.gatherPrefillExpertBatchFromRuntime(
        runtime_table.deviceLayerState(0), hidden.get(), batch.get(),
        /*expert_id=*/1, seq_len, d_model));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    batch->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *zero_batch = batch->data();
    for (int i = 0; i < seq_len * d_model; ++i)
        EXPECT_FLOAT_EQ(zero_batch[i], 0.0f) << "zero-count gather wrote row element " << i;

    auto output = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    std::fill(output->mutable_data(), output->mutable_data() + seq_len * d_model, 0.25f);
    ASSERT_TRUE(output->ensureOnDevice(device));
    auto expert_output = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    std::fill(expert_output->mutable_data(), expert_output->mutable_data() + seq_len * d_model, 7.0f);
    ASSERT_TRUE(expert_output->ensureOnDevice(device));

    ASSERT_TRUE(gpu_kernel.scatterPrefillExpertResultsFromRuntime(
        output.get(), expert_output.get(), runtime_table.deviceLayerState(0),
        /*expert_id=*/1, seq_len, d_model));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *no_op_output = output->data();
    for (int i = 0; i < seq_len * d_model; ++i)
        EXPECT_FLOAT_EQ(no_op_output[i], 0.25f) << "zero-count scatter changed output element " << i;
}

TEST(Test__ROCmMoEKernel, ScatterTokensWritesOriginalSlotMapForInvalidRoutes)
{
    SKIP_IF_NO_ROCM();

    constexpr int seq_len = 3;
    constexpr int top_k = 2;
    constexpr int total_slots = seq_len * top_k;
    constexpr int num_experts = 4;

    const std::array<int, total_slots> routing_indices = {
        0, -1,
        2, -1,
        0, 3};
    const std::array<float, total_slots> routing_weights = {
        0.75f, 0.0f,
        0.60f, 0.0f,
        0.25f, 0.40f};
    const std::array<int, num_experts> expert_offsets = {0, 2, 2, 3};
    const std::array<int, total_slots> expected_original_to_grouped = {
        0, -1,
        2, -1,
        1, 3};
    const std::array<int, 4> expected_grouped_tokens = {0, 2, 1, 2};
    const std::array<float, 4> expected_grouped_weights = {
        0.75f, 0.25f, 0.60f, 0.40f};

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    int *d_routing_indices = nullptr;
    float *d_routing_weights = nullptr;
    int *d_expert_offsets = nullptr;
    int *d_write_heads = nullptr;
    int *d_grouped_tokens = nullptr;
    int *d_original_to_grouped = nullptr;
    float *d_grouped_weights = nullptr;

    ASSERT_EQ(hipMalloc(&d_routing_indices, total_slots * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_routing_weights, total_slots * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_expert_offsets, num_experts * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_write_heads, num_experts * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_grouped_tokens, total_slots * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_original_to_grouped, total_slots * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_grouped_weights, total_slots * sizeof(float)), hipSuccess);

    ASSERT_EQ(hipMemcpyAsync(d_routing_indices, routing_indices.data(),
                             routing_indices.size() * sizeof(int),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_routing_weights, routing_weights.data(),
                             routing_weights.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_expert_offsets, expert_offsets.data(),
                             expert_offsets.size() * sizeof(int),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_write_heads, 0, num_experts * sizeof(int), stream), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_grouped_tokens, 0x7f, total_slots * sizeof(int), stream), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_original_to_grouped, 0xff, total_slots * sizeof(int), stream), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_grouped_weights, 0, total_slots * sizeof(float), stream), hipSuccess);

    ASSERT_TRUE(hipMoE_scatter_tokens(
        d_routing_indices,
        d_routing_weights,
        d_expert_offsets,
        d_write_heads,
        d_grouped_tokens,
        d_original_to_grouped,
        d_grouped_weights,
        total_slots,
        num_experts,
        top_k,
        0,
        stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::array<int, total_slots> host_original_to_grouped{};
    std::array<int, total_slots> host_grouped_tokens{};
    std::array<float, total_slots> host_grouped_weights{};
    ASSERT_EQ(hipMemcpy(host_original_to_grouped.data(), d_original_to_grouped,
                        host_original_to_grouped.size() * sizeof(int),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_grouped_tokens.data(), d_grouped_tokens,
                        host_grouped_tokens.size() * sizeof(int),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_grouped_weights.data(), d_grouped_weights,
                        host_grouped_weights.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              hipSuccess);

    for (int slot = 0; slot < total_slots; ++slot)
        EXPECT_EQ(host_original_to_grouped[slot], expected_original_to_grouped[slot])
            << "original route slot " << slot;
    for (int grouped = 0; grouped < static_cast<int>(expected_grouped_tokens.size()); ++grouped)
    {
        EXPECT_EQ(host_grouped_tokens[grouped], expected_grouped_tokens[grouped])
            << "grouped slot " << grouped;
        EXPECT_FLOAT_EQ(host_grouped_weights[grouped], expected_grouped_weights[grouped])
            << "grouped slot " << grouped;
    }

    EXPECT_EQ(hipFree(d_routing_indices), hipSuccess);
    EXPECT_EQ(hipFree(d_routing_weights), hipSuccess);
    EXPECT_EQ(hipFree(d_expert_offsets), hipSuccess);
    EXPECT_EQ(hipFree(d_write_heads), hipSuccess);
    EXPECT_EQ(hipFree(d_grouped_tokens), hipSuccess);
    EXPECT_EQ(hipFree(d_original_to_grouped), hipSuccess);
    EXPECT_EQ(hipFree(d_grouped_weights), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, FixedTopologyRuntimeGroupedPrefillMatchesExistingPrefillPath)
{
    SKIP_IF_NO_ROCM();
#ifdef ENABLE_PIPELINE_SNAPSHOTS
    GTEST_SKIP() << "Fixed-topology prefill capture path is release-only when snapshots are disabled";
#endif

    const DeviceId device = DeviceId::rocm(0);
    constexpr int seq_len = 2;
    constexpr int d_model = 64;
    constexpr int intermediate = 32;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int total_slots = seq_len * top_k;

    ScopedEnvOverride perf_stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    std::vector<std::unique_ptr<TensorBase>> gate_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> up_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> down_weight_tensors;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> gate_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> up_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> down_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> gate_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> up_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> down_kernels;

    gate_weight_tensors.reserve(num_experts);
    up_weight_tensors.reserve(num_experts);
    down_weight_tensors.reserve(num_experts);
    gate_packed_weights.reserve(num_experts);
    up_packed_weights.reserve(num_experts);
    down_packed_weights.reserve(num_experts);
    gate_kernels.reserve(num_experts);
    up_kernels.reserve(num_experts);
    down_kernels.reserve(num_experts);

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto gate_weights = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 100 + expert_id);
        auto up_weights = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 200 + expert_id);
        auto down_weights = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 300 + expert_id);

        auto gate_packed = std::make_unique<rocm::ROCmPackedWeights>();
        auto up_packed = std::make_unique<rocm::ROCmPackedWeights>();
        auto down_packed = std::make_unique<rocm::ROCmPackedWeights>();
        ASSERT_TRUE(rocm::packWeightsToROCm(gate_weights.get(), *gate_packed));
        ASSERT_TRUE(rocm::packWeightsToROCm(up_weights.get(), *up_packed));
        ASSERT_TRUE(rocm::packWeightsToROCm(down_weights.get(), *down_packed));

        auto gate_kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(gate_packed.get(), 0);
        auto up_kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(up_packed.get(), 0);
        auto down_kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(down_packed.get(), 0);

        gate_weight_tensors.push_back(std::move(gate_weights));
        up_weight_tensors.push_back(std::move(up_weights));
        down_weight_tensors.push_back(std::move(down_weights));
        gate_packed_weights.push_back(std::move(gate_packed));
        up_packed_weights.push_back(std::move(up_packed));
        down_packed_weights.push_back(std::move(down_packed));
        gate_kernels.push_back(std::move(gate_kernel));
        up_kernels.push_back(std::move(up_kernel));
        down_kernels.push_back(std::move(down_kernel));
    }

    auto workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
    ASSERT_TRUE(workspace->allocate(gate_kernels[0]->getWorkspaceRequirements(seq_len, intermediate, d_model)));
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        gate_kernels[expert_id]->bindWorkspace(workspace.get());
        up_kernels[expert_id]->bindWorkspace(workspace.get());
        down_kernels[expert_id]->bindWorkspace(workspace.get());
        gate_kernels[expert_id]->prepareWeights();
        up_kernels[expert_id]->prepareWeights();
        down_kernels[expert_id]->prepareWeights();
    }

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)}, -1.0f, 1.0f, 77);
    ASSERT_TRUE(input->ensureOnDevice(device));

    const float routing_indices_data[total_slots] = {
        0.0f, 1.0f,
        2.0f, 3.0f};
    const float routing_weights_data[total_slots] = {
        0.70f, 0.30f,
        0.60f, 0.40f};

    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots, routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots, routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    auto reference_output = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto fixed_output = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(reference_output->ensureOnDevice(device));
    ASSERT_TRUE(fixed_output->ensureOnDevice(device));

    auto gate_exps = TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(num_experts * intermediate), static_cast<size_t>(d_model)}, 401);
    auto up_exps = TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(num_experts * intermediate), static_cast<size_t>(d_model)}, 402);
    auto down_exps = TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(num_experts * d_model), static_cast<size_t>(intermediate)}, 403);

    auto make_params = [&](TensorBase *output, IMoERuntimeTable *runtime_table)
    {
        MoEExpertComputeStage::Params params;
        params.device_id = device;
        params.input = input.get();
        params.seq_len = seq_len;
        params.d_model = d_model;
        params.num_experts = num_experts;
        params.top_k = top_k;
        params.gate_exps = gate_exps.get();
        params.up_exps = up_exps.get();
        params.down_exps = down_exps.get();
        params.expert_intermediate = intermediate;
        params.local_expert_start = 0;
        params.local_expert_count = num_experts;
        params.layer_idx = 0;
        params.routing_indices = routing_indices.get();
        params.routing_weights = routing_weights.get();
        params.output = output;
        params.moe_runtime_table = runtime_table;
        params.use_runtime_prefill_grouping = runtime_table != nullptr;
        params.prepared_gate_gemm.assign(num_experts, nullptr);
        params.prepared_up_gemm.assign(num_experts, nullptr);
        params.prepared_down_gemm.assign(num_experts, nullptr);
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            params.prepared_gate_gemm[expert_id] = gate_kernels[expert_id].get();
            params.prepared_up_gemm[expert_id] = up_kernels[expert_id].get();
            params.prepared_down_gemm[expert_id] = down_kernels[expert_id].get();
        }
        return params;
    };

    ROCmDeviceContext ctx(device, 0);

    MoEExpertComputeStage reference_stage(make_params(reference_output.get(), nullptr));
    reference_stage.bindWorkspace(workspace.get());
    ASSERT_TRUE(reference_stage.execute(&ctx));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(runtime_config);

    MoEExpertComputeStage fixed_stage(make_params(fixed_output.get(), &runtime_table));
    ASSERT_TRUE(fixed_stage.isGraphCapturable());
    fixed_stage.bindWorkspace(workspace.get());
    ASSERT_TRUE(fixed_stage.execute(&ctx));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    {
        const auto records = PerfStatsCollector::snapshot({"kernel.rocm_moe_small_prefill_grouping_calls"});
        double small_grouping_calls = 0.0;
        for (const auto &record : records)
            small_grouping_calls += record.value;
        EXPECT_GT(small_grouping_calls, 0.0)
            << "verifier-sized fixed-topology prefill should use fused small-M grouping";
    }
    {
        const auto records = PerfStatsCollector::snapshot({"kernel.rocm_moe_grouped_prefill_active_expert_grid_calls"});
        double active_grid_calls = 0.0;
        bool saw_verifier_tile = false;
        for (const auto &record : records)
        {
            active_grid_calls += record.value;
            const auto tile_it = record.tags.find("tile_m");
            saw_verifier_tile = saw_verifier_tile ||
                                (tile_it != record.tags.end() && tile_it->second == "2");
        }
        EXPECT_GT(active_grid_calls, 0.0)
            << "verifier-sized fixed-topology prefill should launch over compact active experts";
        EXPECT_TRUE(saw_verifier_tile)
            << "ROCm two-row verifier grouped prefill should use the M=2 kernel bucket";
    }
    PerfStatsCollector::reset();

    reference_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    fixed_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *reference = reference_output->data();
    const float *fixed = fixed_output->data();
    const size_t output_count = static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
    expectStrictVerifierSimilarity(
        "fixed-topology runtime grouped prefill should match existing prefill path",
        fixed,
        reference,
        output_count,
        static_cast<size_t>(d_model),
        /*min_cosine=*/0.99999,
        /*max_relative_l2=*/0.001,
        /*min_row_cosine=*/0.99999,
        /*max_row_relative_l2=*/0.0015,
        /*max_row_kl=*/1.0e-6);
}

TEST(Test__ROCmMoEKernel, SharedExpertGroupedPrefillMatchesSequentialPath)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int seq_len = 4;
    constexpr int d_model = 64;
    constexpr int intermediate = 32;
    constexpr int num_experts = 1;
    constexpr int top_k = 1;

    ScopedEnvOverride perf_stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto gate_weights = TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 6101);
    auto up_weights = TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 6102);
    auto down_weights = TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 6103);

    auto gate_packed = std::make_unique<rocm::ROCmPackedWeights>();
    auto up_packed = std::make_unique<rocm::ROCmPackedWeights>();
    auto down_packed = std::make_unique<rocm::ROCmPackedWeights>();
    ASSERT_TRUE(rocm::packWeightsToROCm(gate_weights.get(), *gate_packed));
    ASSERT_TRUE(rocm::packWeightsToROCm(up_weights.get(), *up_packed));
    ASSERT_TRUE(rocm::packWeightsToROCm(down_weights.get(), *down_packed));

    rocm::ROCmQuantisedGemmKernel gate_kernel(gate_packed.get(), 0);
    rocm::ROCmQuantisedGemmKernel up_kernel(up_packed.get(), 0);
    rocm::ROCmQuantisedGemmKernel down_kernel(down_packed.get(), 0);

    WorkspaceRequirements gemm_reqs;
    gemm_reqs.merge(gate_kernel.getWorkspaceRequirements(seq_len, intermediate, d_model));
    gemm_reqs.merge(up_kernel.getWorkspaceRequirements(seq_len, intermediate, d_model));
    gemm_reqs.merge(down_kernel.getWorkspaceRequirements(seq_len, d_model, intermediate));
    auto gemm_workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
    ASSERT_TRUE(gemm_workspace->allocate(gemm_reqs));
    gate_kernel.bindWorkspace(gemm_workspace.get());
    up_kernel.bindWorkspace(gemm_workspace.get());
    down_kernel.bindWorkspace(gemm_workspace.get());
    gate_kernel.prepareWeights();
    up_kernel.prepareWeights();
    down_kernel.prepareWeights();

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)}, -0.5f, 0.5f, 6104);
    ASSERT_TRUE(input->ensureOnDevice(device));

    auto reference_gate = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(intermediate)});
    auto reference_up = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(intermediate)});
    auto reference_output = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(reference_gate->ensureOnDevice(device));
    ASSERT_TRUE(reference_up->ensureOnDevice(device));
    ASSERT_TRUE(reference_output->ensureOnDevice(device));

    std::vector<ITensorGemm::TensorProjectionDesc> projections = {
        {&gate_kernel, reference_gate.get(), intermediate, nullptr, "shared_gate"},
        {&up_kernel, reference_up.get(), intermediate, nullptr, "shared_up"}};
    ASSERT_TRUE(gate_kernel.multiply_fused_tensor(
        input.get(), projections, seq_len, d_model));
    reference_gate->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    reference_up->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    ASSERT_TRUE(down_kernel.multiply_tensor_with_fused_swiglu(
        reference_gate.get(), reference_up.get(), reference_output.get(),
        seq_len, d_model, intermediate));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceNativeVNNIMatrixDesc gate_desc;
    DeviceNativeVNNIMatrixDesc up_desc;
    DeviceNativeVNNIMatrixDesc down_desc;
    ASSERT_TRUE(gate_kernel.exportNativeVNNIMatrixDesc(gate_desc));
    ASSERT_TRUE(up_kernel.exportNativeVNNIMatrixDesc(up_desc));
    ASSERT_TRUE(down_kernel.exportNativeVNNIMatrixDesc(down_desc));

    ROCmMoEKernel moe_kernel(0);
    auto moe_workspace = bindDefaultMoEWorkspace(
        moe_kernel, seq_len, d_model, intermediate, num_experts, top_k);
    ASSERT_TRUE(moe_kernel.prepareSharedExpertPrefillGroup(seq_len));
    const int gateup_table = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        &gate_desc, &up_desc, num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        &down_desc, num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    auto grouped_output = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(grouped_output->ensureOnDevice(device));
    ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
        input.get(), grouped_output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    {
        const auto records = PerfStatsCollector::snapshot(
            {"kernel.rocm_moe_shared_expert_prefill_group_calls"});
        double calls = 0.0;
        for (const auto &record : records)
            calls += record.value;
        EXPECT_GT(calls, 0.0)
            << "shared expert verifier prefill must use the ROCm grouped preparation kernel";
    }

    reference_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    const size_t output_count = static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
    const float *reference = reference_output->data();
    const float *grouped = grouped_output->data();
    ASSERT_FALSE(hasNaNOrInf(grouped, output_count));

    expectStrictVerifierSimilarity(
        "ROCm shared-expert grouped verifier prefill must match sequential GEMV",
        grouped,
        reference,
        output_count,
        static_cast<size_t>(d_model));
}

template <typename GateFactory, typename UpFactory, typename DownFactory>
void runSharedExpertFFNStageVerifierRowsQwen36ShapeM234MatchSerialStageDecode(
    const char *format_label,
    GateFactory make_gate,
    UpFactory make_up,
    DownFactory make_down,
    int intermediate = 512)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const std::string format_name = format_label ? format_label : "unknown_format";
    constexpr int d_model = 2048;

    /*
     * This regression targets the optimized production contract itself.  Keep
     * grouped Q8 routing and its expert-input reuse enabled, while leaving
     * deterministic mode and every independent kernel tuning policy untouched.
     */
    ScopedROCmEnvOverride router_q8_env(
        "LLAMINAR_ROCM_MOE_ROUTER_Q8", "1");
    ScopedROCmEnvOverride router_q8_reuse_env(
        "LLAMINAR_ROCM_MOE_REUSE_ROUTER_Q8_HIDDEN", "1");
    ScopedEnvOverride perf_stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    auto gate_weights = make_gate(
        {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 728101);
    auto up_weights = make_up(
        {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 728102);
    auto down_weights = make_down(
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 728103);
    auto prepared = llaminar2::test::makeGpuPreparedFFNFixture(
        gate_weights.get(),
        up_weights.get(),
        down_weights.get(),
        device,
        "test.rocm_moe.qwen36_shared_stage_verifier_" + format_name,
        ModelContextId{728100});
    auto descriptor_probe_down_weights = make_down(
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 728104);
    auto descriptor_probe_down_prepared = llaminar2::test::makeGpuPreparedGemm(
        descriptor_probe_down_weights.get(),
        device,
        "test.rocm_moe.qwen36_shared_stage_verifier_" + format_name +
            ".routed_down_probe." + std::to_string(intermediate),
        ModelContextId{728104 + static_cast<uint64_t>(intermediate)});

    ROCmMoEKernel moe_kernel(0);
    static_cast<ITensorKernel &>(moe_kernel).setGPUStream(stream);

    auto make_params = [&](TensorBase *input,
                           TensorBase *output,
                           int seq_len,
                           bool grouped_verifier)
    {
        SharedExpertFFNStage::Params params;
        params.device_id = device;
        params.input = input;
        params.gate_w = gate_weights.get();
        params.up_w = up_weights.get();
        params.down_w = down_weights.get();
        params.output = output;
        params.seq_len = seq_len;
        params.d_model = d_model;
        params.intermediate = intermediate;
        /*
         * ROCm production verifier rows use the shared grouped table-prefill
         * route for M=2..4, and ordinary grouped shared decode for M=1.  This
         * stage-level regression intentionally follows graph wiring rather than
         * dense GEMM verifier hooks, because the latter can be numerically close
         * but not byte-identical to the serial decode path.
         */
        params.force_grouped_verifier_prefill_for_decode = grouped_verifier;
        params.force_decode_equivalent_verifier_prefill = false;
        params.disable_grouped_decode_shortcut = false;
        params.prepared_ref_gate = prepared.gate_ref;
        params.prepared_ref_up = prepared.up_ref;
        params.prepared_ref_down = prepared.down_ref;
        params.prepared_store = prepared.store.get();
        return params;
    };

    auto make_stage = [&](SharedExpertFFNStage::Params params)
    {
        auto stage = std::make_unique<SharedExpertFFNStage>(params);
        stage->setGPUStream(stream);
        stage->setMoEKernelForTesting(&moe_kernel);
        return stage;
    };

    auto planning_input = TestTensorFactory::createFP32(
        {4u, static_cast<size_t>(d_model)});
    auto planning_output = TestTensorFactory::createFP32(
        {4u, static_cast<size_t>(d_model)});
    auto planning_stage = make_stage(
        make_params(planning_input.get(), planning_output.get(), 4, true));
    auto reqs = planning_stage->getWorkspaceRequirements(4, d_model, intermediate);
    /*
     * A real Qwen3.6 graph places 256-entry routed tables and one-entry shared
     * tables in the same descriptor arena.  Reserve that production width in
     * this all-format sweep; a singleton-only workspace cannot expose slot
     * aliasing between those two table roles.
     */
    constexpr int routed_experts = 256;
    reqs.merge(MoEWorkspaceBuffers::rocmMoE(
        /*max_seq_len=*/4,
        d_model,
        intermediate,
        routed_experts,
        /*top_k=*/8));
    auto stage_workspace = std::make_unique<DeviceWorkspaceManager>(
        device,
        reqs.total_bytes_with_alignment() + 4 * 1024 * 1024);
    ASSERT_TRUE(stage_workspace->allocate(reqs))
        << "SharedExpertFFNStage ROCm Qwen3.6 verifier workspace";
    auto *moe_workspace = dynamic_cast<IWorkspaceConsumer *>(&moe_kernel);
    ASSERT_NE(moe_workspace, nullptr);
    moe_workspace->bindWorkspace(stage_workspace.get());

    /*
     * Qwen3.6 routes against 256 experts with top-k 8.  The shared expert does
     * not consume the selected expert IDs, but it does consume the rowwise Q8
     * activation publication produced as part of this exact routing operation.
     * Using the production router shape here prevents an all-format expert sweep
     * from accidentally proving only an isolated singleton pipeline.
     */
    constexpr int routed_top_k = 8;
    auto router_gate = TestTensorFactory::createFP32(
        {static_cast<size_t>(routed_experts), static_cast<size_t>(d_model)});
    for (size_t i = 0; i < router_gate->numel(); ++i)
    {
        router_gate->mutable_data()[i] =
            0.017f * std::sin(0.0017f * static_cast<float>(i + 29)) +
            0.011f * std::cos(0.0023f * static_cast<float>(i + 47));
    }
    ASSERT_TRUE(router_gate->ensureOnDevice(device, stream));

    DeviceNativeVNNIMatrixDesc serial_gate_desc{};
    DeviceNativeVNNIMatrixDesc serial_up_desc{};
    DeviceNativeVNNIMatrixDesc serial_down_desc{};
    DeviceNativeVNNIMatrixDesc routed_probe_down_desc{};
    ASSERT_TRUE(prepared.gate_kernel->exportNativeVNNIMatrixDesc(serial_gate_desc));
    ASSERT_TRUE(prepared.up_kernel->exportNativeVNNIMatrixDesc(serial_up_desc));
    ASSERT_TRUE(prepared.down_kernel->exportNativeVNNIMatrixDesc(serial_down_desc));
    ASSERT_TRUE(descriptor_probe_down_prepared.kernel->exportNativeVNNIMatrixDesc(
        routed_probe_down_desc));
    ASSERT_NE(routed_probe_down_desc.payload, serial_down_desc.payload)
        << "The routed down-table isolation probe needs a distinct valid descriptor";

    /*
     * Publish a full routed table first.  Expert 1 deliberately swaps the gate
     * and up descriptors, giving the probe an unmistakable byte signature while
     * retaining a valid shape and codebook.  Publishing the singleton shared
     * table below must not alter either routed descriptor.  Running this probe
     * inside the existing all-format helper makes the mixed-width registry
     * contract part of every NativeVNNI codebook gate.
     */
    std::vector<DeviceNativeVNNIMatrixDesc> routed_gate_descs(
        routed_experts,
        serial_gate_desc);
    std::vector<DeviceNativeVNNIMatrixDesc> routed_up_descs(
        routed_experts,
        serial_up_desc);
    constexpr int descriptor_probe_expert = 1;
    routed_gate_descs[descriptor_probe_expert] = serial_up_desc;
    routed_up_descs[descriptor_probe_expert] = serial_gate_desc;
    const int routed_gateup_table =
        moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
            routed_gate_descs.data(),
            routed_up_descs.data(),
            routed_experts,
            d_model,
            intermediate);
    ASSERT_GE(routed_gateup_table, 0)
        << "routed descriptor probe rejected format=" << format_name;
    std::vector<DeviceNativeVNNIMatrixDesc> routed_down_descs(
        routed_experts,
        serial_down_desc);
    routed_down_descs[descriptor_probe_expert] = routed_probe_down_desc;
    const int routed_down_table = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        routed_down_descs.data(),
        routed_experts,
        d_model,
        intermediate);
    ASSERT_GE(routed_down_table, 0)
        << "routed down descriptor probe rejected format=" << format_name;

    auto descriptor_probe_input = TestTensorFactory::createFP32(
        {1u, static_cast<size_t>(d_model)});
    for (size_t i = 0; i < descriptor_probe_input->numel(); ++i)
    {
        descriptor_probe_input->mutable_data()[i] =
            0.019f * std::sin(0.0071f * static_cast<float>(i + 5)) -
            0.013f * std::cos(0.0043f * static_cast<float>(i + 17));
    }
    ASSERT_TRUE(descriptor_probe_input->ensureOnDevice(device, stream));

    struct DescriptorProbeOutput
    {
        std::vector<float> gate;
        std::vector<float> up;
        std::vector<float> down;
    };
    auto execute_descriptor_probe = [&]() -> DescriptorProbeOutput
    {
        auto gate = TestTensorFactory::createFP32(
            {1u, static_cast<size_t>(intermediate)});
        auto up = TestTensorFactory::createFP32(
            {1u, static_cast<size_t>(intermediate)});
        auto down = TestTensorFactory::createFP32(
            {1u, static_cast<size_t>(d_model)});
        EXPECT_TRUE(gate->ensureOnDevice(device, stream));
        EXPECT_TRUE(up->ensureOnDevice(device, stream));
        EXPECT_TRUE(down->ensureOnDevice(device, stream));
        ITensor *gate_outputs[1] = {gate.get()};
        ITensor *up_outputs[1] = {up.get()};
        EXPECT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromTable(
            descriptor_probe_input.get(),
            &descriptor_probe_expert,
            routed_gateup_table,
            /*num_active=*/1,
            gate_outputs,
            up_outputs,
            d_model,
            intermediate));
        constexpr float descriptor_probe_weight = 1.0f;
        EXPECT_TRUE(moe_kernel.groupedExpertDownDecodeFromTable(
            gate_outputs,
            up_outputs,
            &descriptor_probe_expert,
            &descriptor_probe_weight,
            routed_down_table,
            /*num_active=*/1,
            down.get(),
            d_model,
            intermediate));
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        EXPECT_TRUE(gate->ensureOnHost(stream));
        EXPECT_TRUE(up->ensureOnHost(stream));
        EXPECT_TRUE(down->ensureOnHost(stream));
        return {
            std::vector<float>(gate->data(), gate->data() + gate->numel()),
            std::vector<float>(up->data(), up->data() + up->numel()),
            std::vector<float>(down->data(), down->data() + down->numel())};
    };
    const DescriptorProbeOutput routed_before_shared =
        execute_descriptor_probe();

    const int serial_gateup_table = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        &serial_gate_desc, &serial_up_desc, /*num_experts=*/1, d_model, intermediate);
    ASSERT_GE(serial_gateup_table, 0);
    const int serial_down_table = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        &serial_down_desc, /*num_experts=*/1, d_model, intermediate);
    ASSERT_GE(serial_down_table, 0);

    const DescriptorProbeOutput routed_after_shared =
        execute_descriptor_probe();
    expectBitwiseVerifierRowsEqual(
        ("ROCm " + format_name +
         " routed gate descriptors must survive shared-table publication")
            .c_str(),
        routed_after_shared.gate.data(),
        routed_before_shared.gate.data(),
        routed_before_shared.gate.size(),
        static_cast<size_t>(intermediate));
    expectBitwiseVerifierRowsEqual(
        ("ROCm " + format_name +
         " routed up descriptors must survive shared-table publication")
            .c_str(),
        routed_after_shared.up.data(),
        routed_before_shared.up.data(),
        routed_before_shared.up.size(),
        static_cast<size_t>(intermediate));
    expectBitwiseVerifierRowsEqual(
        ("ROCm " + format_name +
         " routed down descriptors must survive shared-table publication")
            .c_str(),
        routed_after_shared.down.data(),
        routed_before_shared.down.data(),
        routed_before_shared.down.size(),
        static_cast<size_t>(d_model));

    ROCmDeviceContext ctx(device, 0);

    static constexpr std::array<uint32_t, 2 * d_model>
        kLayer31ProductionM2HiddenBits = {
#include "../fixtures/Qwen36Layer31M2FFNNormBits.inc"
        };
    static_assert(
        kLayer31ProductionM2HiddenBits.size() ==
        static_cast<size_t>(2 * d_model));

    for (int seq_len : {2, 3, 4})
    {
        SCOPED_TRACE("format=" + format_name + " seq_len=" + std::to_string(seq_len));

        std::vector<float> input_values(static_cast<size_t>(seq_len) * d_model);
        for (size_t i = 0; i < input_values.size(); ++i)
        {
            if (i < kLayer31ProductionM2HiddenBits.size())
            {
                /*
                 * Rows 0 and 1 are the exact layer-31 normalized activations
                 * from the intermittent M=2 model failure.  Reuse them for every
                 * M bucket, then synthesize only the additional M=3/4 rows.  This
                 * turns one checkpoint-specific symptom into an all-codebook
                 * router-Q8/shared-expert regression.
                 */
                input_values[i] = f32FromBitsForROCmMoEKernelTest(
                    kLayer31ProductionM2HiddenBits[i]);
            }
            else
            {
                input_values[i] =
                    0.013f * std::sin(0.0037f * static_cast<float>(i + 17)) -
                    0.009f * std::cos(0.0059f * static_cast<float>(i + 31)) +
                    0.0006f *
                        static_cast<float>(static_cast<int>(i % 37) - 18);
            }
        }

        auto grouped_input = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        std::copy(input_values.begin(), input_values.end(), grouped_input->mutable_data());
        auto grouped_output = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        auto routed_output = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        auto grouped_routing_indices = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(routed_top_k)});
        auto grouped_routing_weights = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(routed_top_k)});
        ASSERT_TRUE(grouped_input->ensureOnDevice(device, stream));
        ASSERT_TRUE(grouped_output->ensureOnDevice(device, stream));
        ASSERT_TRUE(routed_output->ensureOnDevice(device, stream));
        ASSERT_TRUE(grouped_routing_indices->ensureOnDevice(device, stream));
        ASSERT_TRUE(grouped_routing_weights->ensureOnDevice(device, stream));

        ASSERT_TRUE(moe_kernel.routeVerifierRowsDecodeEquivalent(
            grouped_input.get(),
            router_gate.get(),
            seq_len,
            d_model,
            routed_experts,
            routed_top_k,
            /*normalize_weights=*/true,
            grouped_routing_indices.get(),
            grouped_routing_weights.get()))
            << "production grouped router failed before shared expert";
        ASSERT_TRUE(moe_kernel.prepareExpertGroupsAsync(
            grouped_routing_indices.get(),
            grouped_routing_weights.get(),
            seq_len,
            routed_experts,
            routed_top_k))
            << "production routed grouping failed before shared expert";
        ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
            grouped_input.get(),
            routed_output.get(),
            routed_gateup_table,
            routed_down_table,
            seq_len,
            d_model,
            intermediate,
            routed_experts,
            routed_top_k))
            << "production routed expert pipeline failed before shared expert";

        auto grouped_stage = make_stage(
            make_params(grouped_input.get(), grouped_output.get(), seq_len, true));
        grouped_stage->bindWorkspace(stage_workspace.get());
        ASSERT_FALSE(grouped_stage->usesDecodeEquivalentVerifierPrefillForTesting())
            << "ROCm grouped verifier rows must not exercise dense GEMM-hook verifier paths";
        ASSERT_TRUE(grouped_stage->usesGroupedVerifierPrefillRouteForTesting())
            << "ROCm shared verifier rows must use the grouped table-prefill stage path";
        ASSERT_TRUE(grouped_stage->execute(&ctx))
            << "grouped shared verifier stage failed";
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        ASSERT_TRUE(grouped_output->ensureOnHost(stream));
        const std::vector<float> grouped_host(
            grouped_output->data(),
            grouped_output->data() + grouped_output->numel());

        std::vector<float> serial_host(static_cast<size_t>(seq_len) * d_model);
        for (int row = 0; row < seq_len; ++row)
        {
            const auto row_offset = static_cast<size_t>(row) * d_model;
            auto row_input = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
            std::copy(input_values.data() + row_offset,
                      input_values.data() + row_offset + d_model,
                      row_input->mutable_data());
            auto row_gate = TestTensorFactory::createFP32({1u, static_cast<size_t>(intermediate)});
            auto row_up = TestTensorFactory::createFP32({1u, static_cast<size_t>(intermediate)});
            auto row_output = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
            auto row_routing_indices = TestTensorFactory::createFP32(
                {1u, static_cast<size_t>(routed_top_k)});
            auto row_routing_weights = TestTensorFactory::createFP32(
                {1u, static_cast<size_t>(routed_top_k)});
            ASSERT_TRUE(row_input->ensureOnDevice(device, stream));
            ASSERT_TRUE(row_gate->ensureOnDevice(device, stream));
            ASSERT_TRUE(row_up->ensureOnDevice(device, stream));
            ASSERT_TRUE(row_output->ensureOnDevice(device, stream));
            ASSERT_TRUE(row_routing_indices->ensureOnDevice(device, stream));
            ASSERT_TRUE(row_routing_weights->ensureOnDevice(device, stream));

            /*
             * Publish the same rowwise Q8 router state that precedes ordinary
             * serial decode.  The M=1 shared table kernel remains the oracle: it
             * performs its own row quantization, so equality proves the grouped
             * consumer may reuse the router's bytes without changing decode math.
             */
            ASSERT_TRUE(moe_kernel.routeVerifierRowsDecodeEquivalent(
                row_input.get(),
                router_gate.get(),
                /*seq_len=*/1,
                d_model,
                routed_experts,
                routed_top_k,
                /*normalize_weights=*/true,
                row_routing_indices.get(),
                row_routing_weights.get()))
                << "production serial router failed at row=" << row;

            constexpr int expert_id = 0;
            constexpr float expert_weight = 1.0f;
            ITensor *gate_outputs[1] = {row_gate.get()};
            ITensor *up_outputs[1] = {row_up.get()};
            ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromTable(
                row_input.get(), &expert_id, serial_gateup_table, 1,
                gate_outputs, up_outputs, d_model, intermediate))
                << "serial shared gate/up table decode failed at row=" << row;
            ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromTable(
                gate_outputs, up_outputs, &expert_id, &expert_weight,
                serial_down_table, 1, row_output.get(), d_model, intermediate))
                << "serial shared down table decode failed at row=" << row;
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            ASSERT_TRUE(row_output->ensureOnHost(stream));
            std::copy(row_output->data(),
                      row_output->data() + d_model,
            serial_host.begin() + row_offset);
        }

        expectBitwiseVerifierRowsEqual(
            ("ROCm SharedExpertFFNStage " + format_name +
             " grouped verifier rows must match serial stage decode").c_str(),
            grouped_host.data(),
            serial_host.data(),
            grouped_host.size(),
            static_cast<size_t>(d_model));

        /*
         * Record the real producer and consumer in one HIP graph.  The eager
         * pass above has already prepared router caches, descriptor tables,
         * pointer arrays, and workspace, so capture is allocation-free.  The
         * GraphCaptureGuard also marks the Q8 publication as capture-owned;
         * without it the consumer correctly refuses to treat not-yet-executed
         * recording work as an eager publication.
         */
        hipGraph_t graph = nullptr;
        ASSERT_EQ(
            hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
            hipSuccess);
        bool captured_router = false;
        bool captured_routed_grouping = false;
        bool captured_routed_experts = false;
        bool captured_shared = false;
        {
            GraphCaptureGuard capture_guard;
            captured_router = moe_kernel.routeVerifierRowsDecodeEquivalent(
                grouped_input.get(),
                router_gate.get(),
                seq_len,
                d_model,
                routed_experts,
                routed_top_k,
                /*normalize_weights=*/true,
                grouped_routing_indices.get(),
                grouped_routing_weights.get());
            captured_routed_grouping = moe_kernel.prepareExpertGroupsAsync(
                grouped_routing_indices.get(),
                grouped_routing_weights.get(),
                seq_len,
                routed_experts,
                routed_top_k);
            captured_routed_experts = moe_kernel.executeGroupedPrefillPipeline(
                grouped_input.get(),
                routed_output.get(),
                routed_gateup_table,
                routed_down_table,
                seq_len,
                d_model,
                intermediate,
                routed_experts,
                routed_top_k);
            captured_shared = grouped_stage->execute(&ctx);
        }
        const hipError_t capture_status = hipStreamEndCapture(stream, &graph);
        ASSERT_TRUE(captured_router)
            << "captured grouped router failed";
        ASSERT_TRUE(captured_routed_grouping)
            << "captured routed grouping failed";
        ASSERT_TRUE(captured_routed_experts)
            << "captured routed expert pipeline failed";
        ASSERT_TRUE(captured_shared)
            << "captured shared verifier stage failed";
        ASSERT_EQ(capture_status, hipSuccess)
            << hipGetErrorString(capture_status);
        ASSERT_NE(graph, nullptr);

        hipGraphExec_t graph_exec = nullptr;
        ASSERT_EQ(
            hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
            hipSuccess);
        ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        grouped_output->transitionTo(
            TensorCoherenceState::DEVICE_AUTHORITATIVE,
            device);
        ASSERT_TRUE(grouped_output->ensureOnHost(stream));
        expectBitwiseVerifierRowsEqual(
            ("ROCm graph-captured router-Q8 SharedExpertFFNStage " +
             format_name +
             " grouped verifier rows must match serial stage decode")
                .c_str(),
            grouped_output->data(),
            serial_host.data(),
            grouped_output->numel(),
            static_cast<size_t>(d_model));

        EXPECT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
        EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
        grouped_stage->unbindWorkspace();
    }

    const auto records = PerfStatsCollector::snapshot({"mtp"});
    double grouped_table_rows = 0.0;
    bool saw_grouped_table_route = false;
    for (const auto &record : records)
    {
        if (record.domain != "mtp" ||
            record.name != "moe_shared_grouped_decode_equivalent_verifier_prefill_rows")
        {
            continue;
        }
        const auto route_it = record.tags.find("route");
        const std::string route =
            route_it == record.tags.end() ? std::string() : route_it->second;
        if (route == "grouped_table_prefill")
        {
            grouped_table_rows += record.value;
            saw_grouped_table_route = true;
        }
    }
    EXPECT_EQ(static_cast<int>(grouped_table_rows), 18)
        << "Eager and captured M=2/3/4 production rows must be counted on the ROCm grouped table-prefill route\n"
        << PerfStatsCollector::summaryString({"mtp"});
    EXPECT_TRUE(saw_grouped_table_route)
        << "ROCm production stage verifier rows must exercise grouped table prefill\n"
        << PerfStatsCollector::summaryString({"mtp"});

    const auto reuse_records = PerfStatsCollector::snapshot(
        {"kernel.rocm_moe_grouped_prefill_router_q8_reuse_calls"});
    for (const int seq_len : {2, 3, 4})
    {
        double routed_reuse_calls = 0.0;
        double shared_reuse_calls = 0.0;
        for (const auto &record : reuse_records)
        {
            if (record.name !=
                "rocm_moe_grouped_prefill_router_q8_reuse_calls")
            {
                continue;
            }
            const auto seq_it = record.tags.find("seq_len");
            const auto top_k_it = record.tags.find("top_k");
            const auto source_it = record.tags.find("descriptor_source");
            if (seq_it != record.tags.end() &&
                seq_it->second == std::to_string(seq_len) &&
                top_k_it != record.tags.end() &&
                source_it != record.tags.end() &&
                source_it->second == "static_table")
            {
                if (top_k_it->second == std::to_string(routed_top_k))
                    routed_reuse_calls += record.value;
                else if (top_k_it->second == "1")
                    shared_reuse_calls += record.value;
            }
        }
        EXPECT_EQ(static_cast<int>(routed_reuse_calls), 2)
            << "M=" << seq_len
            << " routed experts must reuse router-owned Q8 rows in eager execution and capture recording\n"
            << PerfStatsCollector::summaryString({"kernel"});
        EXPECT_EQ(static_cast<int>(shared_reuse_calls), 2)
            << "M=" << seq_len
            << " shared expert must reuse router-owned Q8 rows after routed experts in eager execution and capture recording\n"
            << PerfStatsCollector::summaryString({"kernel"});
    }

    planning_stage->unbindWorkspace();
    PerfStatsCollector::reset();
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, SharedExpertFFNStageVerifierRows_Qwen36AllNativeVNNIFormats_M234MatchSerialStageDecode)
{
    for (const int intermediate : {512, 256})
    {
        SCOPED_TRACE("shared_intermediate=" + std::to_string(intermediate));
        for (const auto &format : allMoEVerifierFormatCases())
        {
            SCOPED_TRACE(format.name);
            runSharedExpertFFNStageVerifierRowsQwen36ShapeM234MatchSerialStageDecode(
                format.name,
                format.make,
                format.make,
                format.make,
                intermediate);
        }
    }
}

TEST(Test__ROCmMoEKernel, SharedExpertFFNStageVerifierRows_Qwen36ShapeIQ2SGateUpIQ4XSDown_M234MatchSerialStageDecode)
{
    /*
     * The model-level ROCm failure came from a mixed shared-expert projection:
     * gate/up packed as IQ2_S and down packed as IQ4_XS.  Keep that mixed layout
     * as an explicit stage-level regression in addition to the homogeneous
     * all-format sweep above.
     */
    runSharedExpertFFNStageVerifierRowsQwen36ShapeM234MatchSerialStageDecode(
        "IQ2S_gateup_IQ4XS_down",
        [](std::vector<size_t> shape, int seed) -> std::unique_ptr<TensorBase>
        {
            return TestTensorFactory::createIQ2_SRandom(shape, seed);
        },
        [](std::vector<size_t> shape, int seed) -> std::unique_ptr<TensorBase>
        {
            return TestTensorFactory::createIQ2_SRandom(shape, seed);
        },
        [](std::vector<size_t> shape, int seed) -> std::unique_ptr<TensorBase>
        {
            return TestTensorFactory::createIQ4_XSRandom(shape, seed);
        });
}

template <typename GateFactory, typename UpFactory, typename DownFactory>
void runSharedExpertVerifierPrefillQwen36ShapeM234MatchesRowByRowDecode(
    const char *format_label,
    GateFactory make_gate,
    UpFactory make_up,
    DownFactory make_down)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const std::string format_name = format_label ? format_label : "unknown_format";
    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    constexpr int num_experts = 1;
    constexpr int top_k = 1;
    constexpr int expert_id = 0;
    constexpr float expert_weight = 1.0f;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    auto gate_weights = make_gate(
        {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 718101);
    auto up_weights = make_up(
        {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 718102);
    auto down_weights = make_down(
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 718103);

    auto gate_prepared = llaminar2::test::makeGpuPreparedGemm(
        gate_weights.get(),
        device,
        "test.rocm_moe.shared_verifier_" + format_name + ".gate",
        ModelContextId{718101});
    auto up_prepared = llaminar2::test::makeGpuPreparedGemm(
        up_weights.get(),
        device,
        "test.rocm_moe.shared_verifier_" + format_name + ".up",
        ModelContextId{718102});
    auto down_prepared = llaminar2::test::makeGpuPreparedGemm(
        down_weights.get(),
        device,
        "test.rocm_moe.shared_verifier_" + format_name + ".down",
        ModelContextId{718103});

    auto *gate_kernel = gate_prepared.kernel;
    auto *up_kernel = up_prepared.kernel;
    auto *down_kernel = down_prepared.kernel;
    ASSERT_NE(gate_kernel, nullptr);
    ASSERT_NE(up_kernel, nullptr);
    ASSERT_NE(down_kernel, nullptr);
    for (ITensorGemm *kernel : {gate_kernel, up_kernel, down_kernel})
    {
        auto *tensor_kernel = dynamic_cast<ITensorKernel *>(kernel);
        ASSERT_NE(tensor_kernel, nullptr);
        tensor_kernel->setGPUStream(stream);
    }

    auto *gate_workspace = dynamic_cast<IWorkspaceConsumer *>(gate_kernel);
    auto *up_workspace = dynamic_cast<IWorkspaceConsumer *>(up_kernel);
    auto *down_workspace = dynamic_cast<IWorkspaceConsumer *>(down_kernel);
    ASSERT_NE(gate_workspace, nullptr);
    ASSERT_NE(up_workspace, nullptr);
    ASSERT_NE(down_workspace, nullptr);

    WorkspaceRequirements gemm_reqs;
    gemm_reqs.merge(gate_workspace->getWorkspaceRequirements(4, intermediate, d_model));
    gemm_reqs.merge(gate_workspace->getWorkspaceRequirements(1, intermediate, d_model));
    gemm_reqs.merge(up_workspace->getWorkspaceRequirements(4, intermediate, d_model));
    gemm_reqs.merge(up_workspace->getWorkspaceRequirements(1, intermediate, d_model));
    gemm_reqs.merge(down_workspace->getWorkspaceRequirements(4, d_model, intermediate));
    gemm_reqs.merge(down_workspace->getWorkspaceRequirements(1, d_model, intermediate));
    auto gemm_workspace = std::make_unique<DeviceWorkspaceManager>(device, 256 * 1024 * 1024);
    ASSERT_TRUE(gemm_workspace->allocate(gemm_reqs));
    gate_workspace->bindWorkspace(gemm_workspace.get());
    up_workspace->bindWorkspace(gemm_workspace.get());
    down_workspace->bindWorkspace(gemm_workspace.get());
    gate_kernel->prepareWeights();
    up_kernel->prepareWeights();
    down_kernel->prepareWeights();
    ASSERT_TRUE(gate_kernel->weights_converted());
    ASSERT_TRUE(up_kernel->weights_converted());
    ASSERT_TRUE(down_kernel->weights_converted());

    DeviceNativeVNNIMatrixDesc gate_desc{};
    DeviceNativeVNNIMatrixDesc up_desc{};
    DeviceNativeVNNIMatrixDesc down_desc{};
    ASSERT_TRUE(gate_kernel->exportNativeVNNIMatrixDesc(gate_desc));
    ASSERT_TRUE(up_kernel->exportNativeVNNIMatrixDesc(up_desc));
    ASSERT_TRUE(down_kernel->exportNativeVNNIMatrixDesc(down_desc));

    ROCmMoEKernel moe_kernel(0);
    static_cast<ITensorKernel &>(moe_kernel).setGPUStream(stream);
    auto moe_workspace = bindDefaultMoEWorkspace(
        moe_kernel,
        /*max_seq_len=*/4,
        d_model,
        intermediate,
        num_experts,
        top_k);

    const int gateup_table = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        &gate_desc, &up_desc, num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        &down_desc, num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    /**
     * @brief Compare grouped shared verifier publication against true serial
     * decode for each verifier row.
     *
     * Shared expert decode is routed through the ROCm MoE grouped table-decode
     * helpers even though there is only one expert.  This regression therefore
     * uses those M=1 helpers as the oracle rather than the generic M-row GEMM
     * path, which may choose a different reduction policy.
     */
    struct HiddenPattern
    {
        const char *name;
        float amplitude;
        int seed;
        bool pseudo_random;
    };

    const std::array<HiddenPattern, 4> hidden_patterns{{
        {"small_structured", 1.0f, 0, false},
        {"model_norm_structured", 18.0f, 0, false},
        {"model_norm_random", 12.0f, 718201, true},
        {"wide_random", 48.0f, 718202, true},
    }};

    auto fill_hidden_pattern = [&](TensorBase &hidden, int seq_len, const HiddenPattern &pattern)
    {
        std::mt19937 rng(static_cast<uint32_t>(pattern.seed + seq_len * 17));
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (size_t i = 0; i < hidden.numel(); ++i)
        {
            const int centered_col = static_cast<int>(i % 47) - 23;
            const int centered_block = static_cast<int>((i / 29) % 17) - 8;
            const float structured =
                0.011f * static_cast<float>(centered_col) +
                0.003f * static_cast<float>(centered_block);
            const float random_component = pattern.pseudo_random ? dist(rng) : 0.0f;

            /*
             * The full Qwen3.6 graph feeds shared-expert verifier rows after
             * RMSNorm and residual mixing, not after the tiny synthetic values
             * this regression originally used.  Keep the old pattern, but also
             * stress larger row scales so single-ULP expression-tree drift in
             * NativeVNNI dot or SwiGLU quantization cannot hide until model
             * parity loads the full checkpoint.
             */
            hidden.mutable_data()[i] =
                pattern.amplitude * (structured + 0.0075f * random_component);
        }
    };

    auto run_grouped_and_check = [&](int seq_len, const HiddenPattern &pattern)
    {
        auto hidden = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        fill_hidden_pattern(*hidden, seq_len, pattern);
        ASSERT_TRUE(hidden->ensureOnDevice(device, stream));

        std::vector<float> row_by_row_expected(
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model));
        for (int row = 0; row < seq_len; ++row)
        {
            auto hidden_row = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
            std::copy(hidden->data() + static_cast<size_t>(row) * d_model,
                      hidden->data() + static_cast<size_t>(row + 1) * d_model,
                      hidden_row->mutable_data());
            ASSERT_TRUE(hidden_row->ensureOnDevice(device, stream));

            auto gate = TestTensorFactory::createFP32({1u, static_cast<size_t>(intermediate)});
            auto up = TestTensorFactory::createFP32({1u, static_cast<size_t>(intermediate)});
            auto decode_output = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
            ASSERT_TRUE(gate->ensureOnDevice(device, stream));
            ASSERT_TRUE(up->ensureOnDevice(device, stream));
            ASSERT_TRUE(decode_output->ensureOnDevice(device, stream));

            ITensor *gate_outputs[1] = {gate.get()};
            ITensor *up_outputs[1] = {up.get()};
            ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromTable(
                hidden_row.get(), &expert_id, gateup_table, 1,
                gate_outputs, up_outputs, d_model, intermediate));
            ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromTable(
                gate_outputs, up_outputs, &expert_id, &expert_weight,
                down_table, 1, decode_output.get(), d_model, intermediate));
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

            decode_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
            std::copy(decode_output->data(),
                      decode_output->data() + d_model,
                      row_by_row_expected.begin() + static_cast<size_t>(row) * d_model);
        }

        auto grouped_output = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        ASSERT_TRUE(grouped_output->ensureOnDevice(device, stream));
        ASSERT_TRUE(moe_kernel.prepareSharedExpertPrefillGroup(seq_len));
        ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
            hidden.get(),
            grouped_output.get(),
            gateup_table,
            down_table,
            seq_len,
            d_model,
            intermediate,
            num_experts,
            top_k));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        ASSERT_FALSE(hasNaNOrInf(grouped_output->data(), grouped_output->numel()));
        expectBitwiseVerifierRowsEqual(
            ("ROCm Qwen3.6 shared " + format_name + " verifier M=" + std::to_string(seq_len) +
             " pattern=" + pattern.name +
             " must match row-by-row decode").c_str(),
            grouped_output->data(),
            row_by_row_expected.data(),
            grouped_output->numel(),
            static_cast<size_t>(d_model));
    };

    for (int seq_len : {1, 2, 3, 4})
    {
        for (const HiddenPattern &pattern : hidden_patterns)
        {
            SCOPED_TRACE(std::string("hidden_pattern=") + pattern.name);
            run_grouped_and_check(seq_len, pattern);
        }
    }

    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, SharedExpertVerifierPrefill_Qwen36AllNativeVNNIFormats_M234MatchesRowByRowDecode)
{
    for (const auto &format : allMoEVerifierFormatCases())
    {
        SCOPED_TRACE(format.name);
        runSharedExpertVerifierPrefillQwen36ShapeM234MatchesRowByRowDecode(
            format.name,
            format.make,
            format.make,
            format.make);
    }
}

TEST(Test__ROCmMoEKernel, SharedExpertVerifierPrefill_Qwen36ShapeIQ2SGateUpIQ4XSDown_M234MatchesRowByRowDecode)
{
    /*
     * Qwen3.6 MoE can pack shared gate/up and down projections with different
     * codebooks.  The all-format shared sweep proves every codebook family in a
     * homogeneous triplet, while this mixed-format case mirrors the model-level
     * IQ2_S gate/up plus IQ4_XS down layout that first exposed shared-expert drift.
     */
    runSharedExpertVerifierPrefillQwen36ShapeM234MatchesRowByRowDecode(
        "IQ2S_gateup_IQ4XS_down",
        [](std::vector<size_t> shape, int seed) -> std::unique_ptr<TensorBase>
        {
            return TestTensorFactory::createIQ2_SRandom(shape, seed);
        },
        [](std::vector<size_t> shape, int seed) -> std::unique_ptr<TensorBase>
        {
            return TestTensorFactory::createIQ2_SRandom(shape, seed);
        },
        [](std::vector<size_t> shape, int seed) -> std::unique_ptr<TensorBase>
        {
            return TestTensorFactory::createIQ4_XSRandom(shape, seed);
        });
}

template <typename GateFactory, typename UpFactory, typename DownFactory>
void runRoutedOnlyVerifierPrefillQwen36ShapeM234MatchesRowByRowDecode(
    const char *format_label,
    GateFactory make_gate,
    UpFactory make_up,
    DownFactory make_down,
    bool masked_local_tp = false,
    bool exercise_router_q8_publication = false)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const std::string format_name = format_label ? format_label : "unknown_format";
    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;
    constexpr int routed_variants = 16;
    std::array<uint8_t, num_experts> local_expert_mask{};
    if (masked_local_tp)
    {
        /*
         * Model-parity LocalTP participants see the router's full top-k list,
         * but each participant groups and publishes only the experts resident on
         * that shard before the TP allreduce combines local MoE contributions.
         * Owning the lower half gives every row below both local and non-local
         * route slots, so the regression proves original top-k order is
         * preserved while masked slots are skipped.
         */
        for (int expert = 0; expert < num_experts / 2; ++expert)
            local_expert_mask[static_cast<size_t>(expert)] = 1u;
    }

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    ROCmMoEKernel moe_kernel(0);
    static_cast<ITensorKernel &>(moe_kernel).setGPUStream(stream);
    auto moe_workspace = bindDefaultMoEWorkspace(
        moe_kernel,
        /*max_seq_len=*/4,
        d_model,
        intermediate,
        num_experts,
        top_k);

    std::vector<std::unique_ptr<TensorBase>> owned_weights;
    std::vector<llaminar2::test::GpuPreparedGemm> prepared_weights;
    owned_weights.reserve(static_cast<size_t>(routed_variants * 3));
    prepared_weights.reserve(static_cast<size_t>(routed_variants * 3));

    auto add_prepared = [&](std::unique_ptr<TensorBase> weight,
                            int seed,
                            const char *role) -> ITensorGemm *
    {
        auto *weight_ptr = weight.get();
        owned_weights.push_back(std::move(weight));
        prepared_weights.push_back(llaminar2::test::makeGpuPreparedGemm(
            weight_ptr,
            device,
            std::string("test.rocm_moe.qwen36_routed_verifier.") + format_name + "." + role +
                "." + std::to_string(seed),
            ModelContextId{890000 + static_cast<uint64_t>(seed)}));

        auto *kernel = prepared_weights.back().kernel;
        auto *tensor_kernel = dynamic_cast<ITensorKernel *>(kernel);
        if (!tensor_kernel)
            throw std::runtime_error("prepared ROCm routed verifier GEMM did not expose ITensorKernel");
        tensor_kernel->setGPUStream(stream);
        return kernel;
    };

    struct GemmTriplet
    {
        ITensorGemm *gate = nullptr;
        ITensorGemm *up = nullptr;
        ITensorGemm *down = nullptr;
    };

    std::array<GemmTriplet, routed_variants> routed{};
    for (int variant = 0; variant < routed_variants; ++variant)
    {
        routed[static_cast<size_t>(variant)].gate = add_prepared(
            make_gate(
                std::vector<size_t>{static_cast<size_t>(intermediate), static_cast<size_t>(d_model)},
                891000 + variant),
            891000 + variant,
            "routed_gate");
        routed[static_cast<size_t>(variant)].up = add_prepared(
            make_up(
                std::vector<size_t>{static_cast<size_t>(intermediate), static_cast<size_t>(d_model)},
                892000 + variant),
            892000 + variant,
            "routed_up");
        routed[static_cast<size_t>(variant)].down = add_prepared(
            make_down(
                std::vector<size_t>{static_cast<size_t>(d_model), static_cast<size_t>(intermediate)},
                893000 + variant),
            893000 + variant,
            "routed_down");
    }

    auto *workspace_probe = dynamic_cast<IWorkspaceConsumer *>(routed[0].gate);
    ASSERT_NE(workspace_probe, nullptr);
    WorkspaceRequirements gemm_reqs;
    gemm_reqs.merge(workspace_probe->getWorkspaceRequirements(4, intermediate, d_model));
    gemm_reqs.merge(workspace_probe->getWorkspaceRequirements(1, intermediate, d_model));
    if (auto *up_workspace = dynamic_cast<IWorkspaceConsumer *>(routed[0].up))
    {
        gemm_reqs.merge(up_workspace->getWorkspaceRequirements(4, intermediate, d_model));
        gemm_reqs.merge(up_workspace->getWorkspaceRequirements(1, intermediate, d_model));
    }
    if (auto *down_workspace = dynamic_cast<IWorkspaceConsumer *>(routed[0].down))
    {
        gemm_reqs.merge(down_workspace->getWorkspaceRequirements(4, d_model, intermediate));
        gemm_reqs.merge(down_workspace->getWorkspaceRequirements(1, d_model, intermediate));
    }
    auto gemm_workspace = std::make_unique<DeviceWorkspaceManager>(device, 256 * 1024 * 1024);
    ASSERT_TRUE(gemm_workspace->allocate(gemm_reqs));

    auto bind_gemm = [&](ITensorGemm *kernel)
    {
        auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
        ASSERT_NE(consumer, nullptr);
        consumer->bindWorkspace(gemm_workspace.get());
        kernel->prepareWeights();
        ASSERT_TRUE(kernel->weights_converted());
    };
    for (const auto &triplet : routed)
    {
        bind_gemm(triplet.gate);
        bind_gemm(triplet.up);
        bind_gemm(triplet.down);
    }

    auto variant_for_expert = [&](int expert_id) -> size_t
    {
        return static_cast<size_t>(expert_id % routed_variants);
    };

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(static_cast<size_t>(num_experts));
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(static_cast<size_t>(num_experts));
    std::vector<DeviceNativeVNNIMatrixDesc> down_descs(static_cast<size_t>(num_experts));
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const GemmTriplet &triplet = routed[variant_for_expert(expert)];
        ASSERT_TRUE(triplet.gate->exportNativeVNNIMatrixDesc(gate_descs[static_cast<size_t>(expert)]));
        ASSERT_TRUE(triplet.up->exportNativeVNNIMatrixDesc(up_descs[static_cast<size_t>(expert)]));
        ASSERT_TRUE(triplet.down->exportNativeVNNIMatrixDesc(down_descs[static_cast<size_t>(expert)]));
    }

    const int gateup_table = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    auto make_hidden = [](int seq_len)
    {
        auto hidden = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        for (size_t i = 0; i < hidden->numel(); ++i)
        {
            hidden->mutable_data()[i] =
                0.013f * static_cast<float>(static_cast<int>(i % 43) - 21) +
                0.004f * static_cast<float>(static_cast<int>((i / 17) % 19) - 9);
        }
        return hidden;
    };

    auto make_routes = [](int seq_len,
                          std::vector<float> &indices,
                          std::vector<float> &weights)
    {
        static constexpr std::array<int, top_k * 4> kExperts = {
            0, 13, 41, 96, 131, 159, 220, 238,
            3, 17, 42, 99, 144, 171, 221, 251,
            0, 17, 43, 96, 145, 159, 223, 251,
            5, 13, 42, 101, 131, 173, 220, 239};
        indices.resize(static_cast<size_t>(seq_len * top_k));
        weights.resize(static_cast<size_t>(seq_len * top_k));
        for (int row = 0; row < seq_len; ++row)
        {
            float sum = 0.0f;
            for (int route = 0; route < top_k; ++route)
            {
                const int slot = row * top_k + route;
                indices[static_cast<size_t>(slot)] =
                    static_cast<float>(kExperts[static_cast<size_t>(slot)]);
                weights[static_cast<size_t>(slot)] =
                    0.09f + 0.013f * static_cast<float>((slot * 5 + 3) % 11);
                sum += weights[static_cast<size_t>(slot)];
            }
            for (int route = 0; route < top_k; ++route)
            {
                const int slot = row * top_k + route;
                weights[static_cast<size_t>(slot)] /= sum;
            }
        }
    };

    /**
     * @brief Execute the grouped verifier path and compare it with serial
     * row-by-row decode for the same rows.
     *
     * The lambda is intentionally reused after a workspace rebind below.  ROCm
     * MoE keeps several grouped-verifier scratch pointers cached between calls;
     * a workspace handoff must invalidate those pointers before any capacity
     * check can short-circuit rebinding.
     */
    auto run_grouped_and_check = [&](int seq_len, const char *label)
    {
        auto hidden = make_hidden(seq_len);
        ASSERT_TRUE(hidden->ensureOnDevice(device, stream));

        std::vector<float> route_indices;
        std::vector<float> route_weights;
        make_routes(seq_len, route_indices, route_weights);
        auto routing_indices = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
        auto routing_weights = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
        std::copy(route_indices.begin(), route_indices.end(), routing_indices->mutable_data());
        std::copy(route_weights.begin(), route_weights.end(), routing_weights->mutable_data());
        ASSERT_TRUE(routing_indices->ensureOnDevice(device, stream));
        ASSERT_TRUE(routing_weights->ensureOnDevice(device, stream));

        std::vector<float> row_by_row_expected(
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model));
        for (int row = 0; row < seq_len; ++row)
        {
            auto hidden_row = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
            std::copy(hidden->data() + static_cast<size_t>(row) * d_model,
                      hidden->data() + static_cast<size_t>(row + 1) * d_model,
                      hidden_row->mutable_data());
            ASSERT_TRUE(hidden_row->ensureOnDevice(device, stream));

            std::array<int, top_k> expert_ids = {};
            std::array<float, top_k> expert_weights = {};
            for (int route = 0; route < top_k; ++route)
            {
                const int slot = row * top_k + route;
                const int expert =
                    static_cast<int>(route_indices[static_cast<size_t>(slot)]);
                const bool local =
                    !masked_local_tp ||
                    (expert >= 0 &&
                     expert < num_experts &&
                     local_expert_mask[static_cast<size_t>(expert)] != 0u);
                expert_ids[static_cast<size_t>(route)] =
                    local ? expert : -1;
                expert_weights[static_cast<size_t>(route)] =
                    local ? route_weights[static_cast<size_t>(slot)] : 0.0f;
            }

            std::array<std::shared_ptr<FP32Tensor>, top_k> gate_owned;
            std::array<std::shared_ptr<FP32Tensor>, top_k> up_owned;
            std::array<ITensor *, top_k> gate_outputs = {};
            std::array<ITensor *, top_k> up_outputs = {};
            for (int route = 0; route < top_k; ++route)
            {
                gate_owned[static_cast<size_t>(route)] =
                    TestTensorFactory::createFP32({1u, static_cast<size_t>(intermediate)});
                up_owned[static_cast<size_t>(route)] =
                    TestTensorFactory::createFP32({1u, static_cast<size_t>(intermediate)});
                ASSERT_TRUE(gate_owned[static_cast<size_t>(route)]->ensureOnDevice(device, stream));
                ASSERT_TRUE(up_owned[static_cast<size_t>(route)]->ensureOnDevice(device, stream));
                gate_outputs[static_cast<size_t>(route)] =
                    gate_owned[static_cast<size_t>(route)].get();
                up_outputs[static_cast<size_t>(route)] =
                    up_owned[static_cast<size_t>(route)].get();
            }

            auto decode_output = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
            ASSERT_TRUE(decode_output->ensureOnDevice(device, stream));
            if (masked_local_tp)
            {
                auto row_indices = TestTensorFactory::createFP32({static_cast<size_t>(top_k)});
                auto row_weights = TestTensorFactory::createFP32({static_cast<size_t>(top_k)});
                for (int route = 0; route < top_k; ++route)
                {
                    const int slot = row * top_k + route;
                    row_indices->mutable_data()[static_cast<size_t>(route)] =
                        route_indices[static_cast<size_t>(slot)];
                    row_weights->mutable_data()[static_cast<size_t>(route)] =
                        route_weights[static_cast<size_t>(slot)];
                }
                ASSERT_TRUE(row_indices->ensureOnDevice(device, stream));
                ASSERT_TRUE(row_weights->ensureOnDevice(device, stream));
                ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromRouting(
                    hidden_row.get(), row_indices.get(), gateup_table, top_k,
                    gate_outputs.data(), up_outputs.data(), d_model, intermediate,
                    local_expert_mask.data()));
                ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromRouting(
                    gate_outputs.data(), up_outputs.data(),
                    row_indices.get(), row_weights.get(),
                    down_table, top_k, decode_output.get(),
                    d_model, intermediate,
                    local_expert_mask.data()));
            }
            else
            {
                ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromTable(
                    hidden_row.get(), expert_ids.data(), gateup_table, top_k,
                    gate_outputs.data(), up_outputs.data(), d_model, intermediate));
                ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromTable(
                    gate_outputs.data(), up_outputs.data(), expert_ids.data(),
                    expert_weights.data(), down_table, top_k, decode_output.get(),
                    d_model, intermediate));
            }
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            decode_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
            std::copy(decode_output->data(),
                      decode_output->data() + d_model,
                      row_by_row_expected.begin() + static_cast<size_t>(row) * d_model);
        }

        auto grouped_output = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        ASSERT_TRUE(grouped_output->ensureOnDevice(device, stream));
        if (masked_local_tp)
        {
            ASSERT_TRUE(moe_kernel.prepareExpertGroupsAsyncMasked(
                routing_indices.get(),
                routing_weights.get(),
                seq_len,
                num_experts,
                top_k,
                local_expert_mask.data()));
        }
        else
        {
            ASSERT_TRUE(moe_kernel.prepareExpertGroupsAsync(
                routing_indices.get(), routing_weights.get(), seq_len, num_experts, top_k));
        }
        ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
            hidden.get(),
            grouped_output.get(),
            gateup_table,
            down_table,
            seq_len,
            d_model,
            intermediate,
            num_experts,
            top_k));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);

        expectBitwiseVerifierRowsEqual(
            ("ROCm Qwen3.6 " + format_name +
             (masked_local_tp ? " masked LocalTP" : " routed-only") +
             " verifier M=" + std::to_string(seq_len) +
             " (" + std::string(label) + ") must match row-by-row decode").c_str(),
            grouped_output->data(),
            row_by_row_expected.data(),
            grouped_output->numel(),
            static_cast<size_t>(d_model));

        if (masked_local_tp)
        {
            const auto records = PerfStatsCollector::snapshot(
                {"kernel.rocm_moe_masked_prefill_grouping_calls"});
            const std::string expected_seq_len = std::to_string(seq_len);
            const std::string expected_grid_slots =
                std::to_string(std::min(seq_len * top_k, num_experts / 2));
            const auto record = std::find_if(
                records.begin(),
                records.end(),
                [&](const PerfStatRecord &candidate)
                {
                    auto tag_equals = [&](const char *name, const std::string &value)
                    {
                        const auto it = candidate.tags.find(name);
                        return it != candidate.tags.end() && it->second == value;
                    };
                    return candidate.name ==
                               "rocm_moe_masked_prefill_grouping_calls" &&
                           tag_equals("seq_len", expected_seq_len) &&
                           tag_equals("mask_active_experts", "128") &&
                           tag_equals("expert_grid_slots", expected_grid_slots) &&
                           candidate.count > 0;
                });
            EXPECT_NE(record, records.end())
                << format_name << " M=" << seq_len
                << " must record the masked ROCm LocalTP grouping route\n"
                << PerfStatsCollector::summaryString(
                       {"kernel.rocm_moe_masked_prefill_grouping_calls"});
        }
    };

    for (int seq_len : {1, 2, 3, 4})
    {
        run_grouped_and_check(seq_len, "initial workspace");
    }

    auto rebound_workspace = bindDefaultMoEWorkspace(
        moe_kernel,
        /*max_seq_len=*/4,
        d_model,
        intermediate,
        num_experts,
        top_k);
    moe_workspace.reset();

    SCOPED_TRACE("ROCm grouped verifier scratch pointers must be rebound after workspace handoff");
    run_grouped_and_check(4, "after workspace rebind");

    if (exercise_router_q8_publication)
    {
        /*
         * The host-authored route cases above isolate expert math and masked
         * grouping.  This second phase deliberately enters through the
         * production grouped router so the all-format sweep also proves the
         * router's Q8 hidden publication is the exact byte stream consumed by
         * grouped gate/up.  Serial reference rows use the production M=1 router
         * and fused runtime decode; no row-replay implementation participates.
         */
        auto router_gate = TestTensorFactory::createFP32(
            {static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
        for (size_t i = 0; i < router_gate->numel(); ++i)
        {
            router_gate->mutable_data()[i] =
                0.021f * std::sin(0.0037f * static_cast<float>(i + 37)) +
                0.014f * std::cos(0.0051f * static_cast<float>(i + 19));
        }
        ASSERT_TRUE(router_gate->ensureOnDevice(device, stream));

        DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, top_k);
        populateRuntimeDescriptors(host_runtime, &gate_descs, &up_descs, &down_descs);
        DeviceMoELayerRuntime *device_runtime = nullptr;
        ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_runtime),
                            sizeof(DeviceMoELayerRuntime)),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_runtime,
                                 &host_runtime,
                                 sizeof(DeviceMoELayerRuntime),
                                 hipMemcpyHostToDevice,
                                 stream),
                  hipSuccess);

        DeviceMoERuntimeTable::Config runtime_config;
        runtime_config.device_id = device;
        runtime_config.num_layers = 1;
        runtime_config.num_experts = num_experts;
        runtime_config.top_k = top_k;
        runtime_config.mirror_to_device = true;
        runtime_config.prefill_token_capacity = 4;
        DeviceMoERuntimeTable runtime_table(runtime_config);
        auto &runtime_host_state = runtime_table.hostLayerState(0);
        populateRuntimeDescriptors(
            runtime_host_state, &gate_descs, &up_descs, &down_descs);
        ASSERT_EQ(hipMemcpyAsync(runtime_table.deviceLayerState(0),
                                 &runtime_host_state,
                                 sizeof(runtime_host_state),
                                 hipMemcpyHostToDevice,
                                 stream),
                  hipSuccess);

        PerfStatsCollector::reset();
        for (int seq_len : {2, 3, 4})
        {
            auto hidden = make_hidden(seq_len);
            ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
            auto routing_indices = TestTensorFactory::createFP32(
                {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
            auto routing_weights = TestTensorFactory::createFP32(
                {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
            auto grouped_output = TestTensorFactory::createFP32(
                {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
            auto runtime_grouped_output = TestTensorFactory::createFP32(
                {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
            ASSERT_TRUE(routing_indices->ensureOnDevice(device, stream));
            ASSERT_TRUE(routing_weights->ensureOnDevice(device, stream));
            ASSERT_TRUE(grouped_output->ensureOnDevice(device, stream));
            ASSERT_TRUE(runtime_grouped_output->ensureOnDevice(device, stream));

            ASSERT_TRUE(moe_kernel.routeVerifierRowsDecodeEquivalent(
                hidden.get(),
                router_gate.get(),
                seq_len,
                d_model,
                num_experts,
                top_k,
                /*normalize_weights=*/true,
                routing_indices.get(),
                routing_weights.get()))
                << format_name << " production grouped router M=" << seq_len;
            ASSERT_TRUE(moe_kernel.prepareExpertGroupsAsync(
                routing_indices.get(),
                routing_weights.get(),
                seq_len,
                num_experts,
                top_k));
            ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
                hidden.get(),
                grouped_output.get(),
                gateup_table,
                down_table,
                seq_len,
                d_model,
                intermediate,
                num_experts,
                top_k));

            ASSERT_TRUE(moe_kernel.groupPrefillRoutes(
                runtime_table.deviceLayerState(0),
                routing_indices.get(),
                routing_weights.get(),
                seq_len,
                runtime_config.prefill_token_capacity,
                num_experts,
                top_k))
                << format_name << " runtime route grouping M=" << seq_len;
            ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipelineFromRuntime(
                runtime_table.deviceLayerState(0),
                runtime_host_state,
                hidden.get(),
                runtime_grouped_output.get(),
                gateup_table,
                down_table,
                seq_len,
                d_model,
                intermediate,
                num_experts,
                top_k))
                << format_name << " runtime-placement grouped verifier M="
                << seq_len;
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            grouped_output->transitionTo(
                TensorCoherenceState::DEVICE_AUTHORITATIVE,
                device);
            runtime_grouped_output->transitionTo(
                TensorCoherenceState::DEVICE_AUTHORITATIVE,
                device);

            std::vector<float> serial_expected(
                static_cast<size_t>(seq_len) * static_cast<size_t>(d_model));
            for (int row = 0; row < seq_len; ++row)
            {
                auto row_hidden = TestTensorFactory::createFP32(
                    {1u, static_cast<size_t>(d_model)});
                std::copy_n(hidden->data() + static_cast<size_t>(row) * d_model,
                            d_model,
                            row_hidden->mutable_data());
                ASSERT_TRUE(row_hidden->ensureOnDevice(device, stream));

                ASSERT_TRUE(moe_kernel.decodeRouteSelect(
                    device_runtime,
                    row_hidden.get(),
                    router_gate.get(),
                    d_model,
                    num_experts,
                    top_k,
                    /*normalize_weights=*/true,
                    /*output_indices=*/nullptr,
                    /*output_weights=*/nullptr,
                    /*write_legacy_outputs=*/false,
                    /*update_runtime_histogram=*/false))
                    << format_name << " production serial router M=" << seq_len
                    << " row=" << row;

                auto row_output = TestTensorFactory::createFP32(
                    {1u, static_cast<size_t>(d_model)});
                ASSERT_TRUE(row_output->ensureOnDevice(device, stream));
                moe_kernel.zeroBuffer(
                    row_output.get(),
                    static_cast<size_t>(d_model) * sizeof(float));
                ASSERT_TRUE(moe_kernel.groupedExpertDecodeFromRuntime(
                    device_runtime,
                    row_hidden.get(),
                    gateup_table,
                    down_table,
                    top_k,
                    row_output.get(),
                    d_model,
                    intermediate,
                    MoEDecodeDescriptorSource::RuntimePlacementTable))
                    << format_name << " production serial expert decode M="
                    << seq_len << " row=" << row;
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
                row_output->transitionTo(
                    TensorCoherenceState::DEVICE_AUTHORITATIVE,
                    device);
                std::copy_n(row_output->data(),
                            d_model,
                            serial_expected.data() + static_cast<size_t>(row) * d_model);
            }

            expectBitwiseVerifierRowsEqual(
                ("ROCm " + format_name +
                 " production router-Q8 grouped verifier M=" +
                 std::to_string(seq_len) +
                 " must match production serial decode").c_str(),
                grouped_output->data(),
                serial_expected.data(),
                grouped_output->numel(),
                static_cast<size_t>(d_model));
            expectBitwiseVerifierRowsEqual(
                ("ROCm " + format_name +
                 " runtime-placement grouped verifier M=" +
                 std::to_string(seq_len) +
                 " must match runtime-placement serial decode")
                    .c_str(),
                runtime_grouped_output->data(),
                serial_expected.data(),
                runtime_grouped_output->numel(),
                static_cast<size_t>(d_model));

            const auto reuse_records = PerfStatsCollector::snapshot(
                {"kernel.rocm_moe_grouped_prefill_router_q8_reuse_calls"});
            const std::string expected_seq_len = std::to_string(seq_len);
            const auto reuse_record = std::find_if(
                reuse_records.begin(),
                reuse_records.end(),
                [&](const PerfStatRecord &record)
                {
                    const auto seq_it = record.tags.find("seq_len");
                    const auto source_it = record.tags.find("descriptor_source");
                    return record.name ==
                               "rocm_moe_grouped_prefill_router_q8_reuse_calls" &&
                           seq_it != record.tags.end() &&
                           seq_it->second == expected_seq_len &&
                           source_it != record.tags.end() &&
                           source_it->second == "static_table" &&
                           record.count > 0;
                });
            EXPECT_NE(reuse_record, reuse_records.end())
                << format_name << " M=" << seq_len
                << " must consume the grouped router's published Q8 rows";
            const auto runtime_reuse_record = std::find_if(
                reuse_records.begin(),
                reuse_records.end(),
                [&](const PerfStatRecord &record)
                {
                    const auto seq_it = record.tags.find("seq_len");
                    const auto source_it = record.tags.find("descriptor_source");
                    return record.name ==
                               "rocm_moe_grouped_prefill_router_q8_reuse_calls" &&
                           seq_it != record.tags.end() &&
                           seq_it->second == expected_seq_len &&
                           source_it != record.tags.end() &&
                           source_it->second == "runtime_table" &&
                           record.count > 0;
                });
            EXPECT_NE(runtime_reuse_record, reuse_records.end())
                << format_name << " M=" << seq_len
                << " must consume router-Q8 rows through runtime placement";
        }

        EXPECT_EQ(hipFree(device_runtime), hipSuccess);
        PerfStatsCollector::reset();
    }

    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}


TEST(Test__ROCmMoEKernel, RoutedOnlyVerifierPrefill_Qwen36AllNativeVNNIFormats_M234MatchesRowByRowDecode)
{
    ScopedEnvOverride perf_stats("LLAMINAR_PERF_STATS_SUMMARY", "1");
    ScopedROCmEnvOverride production_mode("LLAMINAR_DETERMINISTIC", "0");
    ScopedROCmEnvOverride q8_router("LLAMINAR_ROCM_MOE_ROUTER_Q8", "1");
    ScopedROCmEnvOverride q8_reuse(
        "LLAMINAR_ROCM_MOE_REUSE_ROUTER_Q8_HIDDEN", "1");
    for (const auto &format : allMoEVerifierFormatCases())
    {
        SCOPED_TRACE(format.name);
        runRoutedOnlyVerifierPrefillQwen36ShapeM234MatchesRowByRowDecode(
            format.name,
            format.make,
            format.make,
            format.make,
            /*masked_local_tp=*/false,
            /*exercise_router_q8_publication=*/true);
    }
}

TEST(Test__ROCmMoEKernel, MaskedLocalTPVerifierPrefill_Qwen36AllNativeVNNIFormats_M234MatchesRowByRowDecode)
{
    ScopedEnvOverride perf_stats("LLAMINAR_PERF_STATS_SUMMARY", "1");
    for (const auto &format : allMoEVerifierFormatCases())
    {
        SCOPED_TRACE(format.name);
        runRoutedOnlyVerifierPrefillQwen36ShapeM234MatchesRowByRowDecode(
            format.name,
            format.make,
            format.make,
            format.make,
            /*masked_local_tp=*/true);
    }
}

TEST(Test__ROCmMoEKernel, RoutedOnlyVerifierPrefill_Qwen36ShapeM234MatchesRowByRowDecode)
{
    runRoutedOnlyVerifierPrefillQwen36ShapeM234MatchesRowByRowDecode(
        "IQ2S_gateup_IQ4XS_down",
        [](std::vector<size_t> shape, int seed) -> std::unique_ptr<TensorBase>
        {
            return TestTensorFactory::createIQ2_SRandom(shape, seed);
        },
        [](std::vector<size_t> shape, int seed) -> std::unique_ptr<TensorBase>
        {
            return TestTensorFactory::createIQ2_SRandom(shape, seed);
        },
        [](std::vector<size_t> shape, int seed) -> std::unique_ptr<TensorBase>
        {
            return TestTensorFactory::createIQ4_XSRandom(shape, seed);
        });
}

TEST(Test__ROCmMoEKernel, MaskedLocalTPVerifierPrefill_Qwen36ShapeIQ2SGateUpIQ4XSDown_M234MatchesRowByRowDecode)
{
    /*
     * Qwen3.6 expert-overlay participants execute only the experts resident on
     * the local shard, while the full router top-k list still contains remote
     * experts.  The routed-only mixed-format test covers IQ2_S gate/up with
     * IQ4_XS down, but the masked LocalTP publication path has its own compact
     * grouping and ordered down-publish logic.  Keep this mixed-format case
     * explicit so the full parity harness is not the first place to discover
     * masked LocalTP drift.
     */
    runRoutedOnlyVerifierPrefillQwen36ShapeM234MatchesRowByRowDecode(
        "IQ2S_gateup_IQ4XS_down",
        [](std::vector<size_t> shape, int seed) -> std::unique_ptr<TensorBase>
        {
            return TestTensorFactory::createIQ2_SRandom(shape, seed);
        },
        [](std::vector<size_t> shape, int seed) -> std::unique_ptr<TensorBase>
        {
            return TestTensorFactory::createIQ2_SRandom(shape, seed);
        },
        [](std::vector<size_t> shape, int seed) -> std::unique_ptr<TensorBase>
        {
            return TestTensorFactory::createIQ4_XSRandom(shape, seed);
        },
        /*masked_local_tp=*/true);
}

/**
 * @brief Prove the grouped verifier pipeline matches ROCm production runtime decode.
 *
 * Earlier routed-only verifier tests compared grouped prefill with table-based
 * one-row decode helpers while explicitly disabling several ROCm decode
 * optimizations.  The Qwen3.6 model path does not use that reduced contract:
 * ordinary GPU decode reaches the runtime-table fused decode entry point, and
 * verifier rows reach the grouped prefill pipeline.  MTP publication requires
 * those two device-resident entry points to produce the same byte stream for
 * each verifier row, otherwise a small early MoE drift can be amplified by
 * later attention, GDN, and routing stages.
 */
TEST(Test__ROCmMoEKernel, RoutedOnlyVerifierPrefill_Qwen36IQ2SGateUpIQ4XSDown_M1234MatchesRuntimeDecode)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;
    constexpr int routed_variants = 16;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    ScopedROCmEnvOverride enable_parallel_down(
        "LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE", "1");

    ROCmMoEKernel moe_kernel(0);
    static_cast<ITensorKernel &>(moe_kernel).setGPUStream(stream);
    auto moe_workspace = bindDefaultMoEWorkspace(
        moe_kernel,
        /*max_seq_len=*/4,
        d_model,
        intermediate,
        num_experts,
        top_k);

    std::vector<std::unique_ptr<TensorBase>> owned_weights;
    std::vector<llaminar2::test::GpuPreparedGemm> prepared_weights;
    owned_weights.reserve(static_cast<size_t>(routed_variants * 3));
    prepared_weights.reserve(static_cast<size_t>(routed_variants * 3));

    auto add_prepared = [&](std::unique_ptr<TensorBase> weight,
                            int seed,
                            const char *role) -> ITensorGemm *
    {
        auto *weight_ptr = weight.get();
        owned_weights.push_back(std::move(weight));
        prepared_weights.push_back(llaminar2::test::makeGpuPreparedGemm(
            weight_ptr,
            device,
            std::string("test.rocm_moe.qwen36_runtime_vs_verifier.") + role +
                "." + std::to_string(seed),
            ModelContextId{920000 + static_cast<uint64_t>(seed)}));

        auto *kernel = prepared_weights.back().kernel;
        auto *tensor_kernel = dynamic_cast<ITensorKernel *>(kernel);
        if (!tensor_kernel)
        {
            ADD_FAILURE()
                << "prepared ROCm runtime/verifier GEMM must expose ITensorKernel";
            return static_cast<ITensorGemm *>(nullptr);
        }
        tensor_kernel->setGPUStream(stream);
        return kernel;
    };

    struct GemmTriplet
    {
        ITensorGemm *gate = nullptr;
        ITensorGemm *up = nullptr;
        ITensorGemm *down = nullptr;
    };

    std::array<GemmTriplet, routed_variants> routed{};
    for (int variant = 0; variant < routed_variants; ++variant)
    {
        routed[static_cast<size_t>(variant)].gate = add_prepared(
            TestTensorFactory::createIQ2_SRandom(
                {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)},
                921000 + variant),
            921000 + variant,
            "routed_gate_iq2s");
        routed[static_cast<size_t>(variant)].up = add_prepared(
            TestTensorFactory::createIQ2_SRandom(
                {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)},
                922000 + variant),
            922000 + variant,
            "routed_up_iq2s");
        routed[static_cast<size_t>(variant)].down = add_prepared(
            TestTensorFactory::createIQ4_XSRandom(
                {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)},
                923000 + variant),
            923000 + variant,
            "routed_down_iq4xs");
        ASSERT_NE(routed[static_cast<size_t>(variant)].gate, nullptr);
        ASSERT_NE(routed[static_cast<size_t>(variant)].up, nullptr);
        ASSERT_NE(routed[static_cast<size_t>(variant)].down, nullptr);
    }

    auto *workspace_probe = dynamic_cast<IWorkspaceConsumer *>(routed[0].gate);
    ASSERT_NE(workspace_probe, nullptr);
    WorkspaceRequirements gemm_reqs;
    gemm_reqs.merge(workspace_probe->getWorkspaceRequirements(4, intermediate, d_model));
    gemm_reqs.merge(workspace_probe->getWorkspaceRequirements(1, intermediate, d_model));
    if (auto *up_workspace = dynamic_cast<IWorkspaceConsumer *>(routed[0].up))
    {
        gemm_reqs.merge(up_workspace->getWorkspaceRequirements(4, intermediate, d_model));
        gemm_reqs.merge(up_workspace->getWorkspaceRequirements(1, intermediate, d_model));
    }
    if (auto *down_workspace = dynamic_cast<IWorkspaceConsumer *>(routed[0].down))
    {
        gemm_reqs.merge(down_workspace->getWorkspaceRequirements(4, d_model, intermediate));
        gemm_reqs.merge(down_workspace->getWorkspaceRequirements(1, d_model, intermediate));
    }
    auto gemm_workspace = std::make_unique<DeviceWorkspaceManager>(device, 256 * 1024 * 1024);
    ASSERT_TRUE(gemm_workspace->allocate(gemm_reqs));

    auto bind_gemm = [&](ITensorGemm *kernel)
    {
        auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
        ASSERT_NE(consumer, nullptr);
        consumer->bindWorkspace(gemm_workspace.get());
        kernel->prepareWeights();
        ASSERT_TRUE(kernel->weights_converted());
    };
    for (const auto &triplet : routed)
    {
        bind_gemm(triplet.gate);
        bind_gemm(triplet.up);
        bind_gemm(triplet.down);
    }

    auto variant_for_expert = [](int expert_id) -> size_t
    {
        return static_cast<size_t>(expert_id % routed_variants);
    };

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(static_cast<size_t>(num_experts));
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(static_cast<size_t>(num_experts));
    std::vector<DeviceNativeVNNIMatrixDesc> down_descs(static_cast<size_t>(num_experts));
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const GemmTriplet &triplet = routed[variant_for_expert(expert)];
        ASSERT_TRUE(triplet.gate->exportNativeVNNIMatrixDesc(gate_descs[static_cast<size_t>(expert)]));
        ASSERT_TRUE(triplet.up->exportNativeVNNIMatrixDesc(up_descs[static_cast<size_t>(expert)]));
        ASSERT_TRUE(triplet.down->exportNativeVNNIMatrixDesc(down_descs[static_cast<size_t>(expert)]));
    }

    const int gateup_table = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, top_k);
    populateRuntimeDescriptors(host_runtime, &gate_descs, &up_descs, &down_descs);
    DeviceMoELayerRuntime *device_runtime = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_runtime), sizeof(DeviceMoELayerRuntime)), hipSuccess);

    DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = 4;
    DeviceMoERuntimeTable prefill_runtime_table(runtime_config);
    auto &runtime_prefill_state = prefill_runtime_table.hostLayerState(0);
    populateRuntimeDescriptors(runtime_prefill_state, &gate_descs, &up_descs, &down_descs);
    ASSERT_EQ(hipMemcpyAsync(prefill_runtime_table.deviceLayerState(0),
                             &runtime_prefill_state,
                             sizeof(runtime_prefill_state),
                             hipMemcpyHostToDevice,
                             stream),
              hipSuccess);

    auto make_hidden = [](int seq_len)
    {
        auto hidden = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        for (size_t i = 0; i < hidden->numel(); ++i)
        {
            hidden->mutable_data()[i] =
                0.013f * static_cast<float>(static_cast<int>(i % 43) - 21) +
                0.004f * static_cast<float>(static_cast<int>((i / 17) % 19) - 9);
        }
        return hidden;
    };

    auto make_routes = [](int seq_len,
                          std::vector<float> &indices,
                          std::vector<float> &weights)
    {
        static constexpr std::array<int, top_k * 4> kExperts = {
            0, 13, 41, 96, 131, 159, 220, 238,
            3, 17, 42, 99, 144, 171, 221, 251,
            0, 17, 43, 96, 145, 159, 223, 251,
            5, 13, 42, 101, 131, 173, 220, 239};
        indices.resize(static_cast<size_t>(seq_len * top_k));
        weights.resize(static_cast<size_t>(seq_len * top_k));
        for (int row = 0; row < seq_len; ++row)
        {
            float sum = 0.0f;
            for (int route = 0; route < top_k; ++route)
            {
                const int slot = row * top_k + route;
                indices[static_cast<size_t>(slot)] =
                    static_cast<float>(kExperts[static_cast<size_t>(slot)]);
                weights[static_cast<size_t>(slot)] =
                    0.09f + 0.013f * static_cast<float>((slot * 5 + 3) % 11);
                sum += weights[static_cast<size_t>(slot)];
            }
            for (int route = 0; route < top_k; ++route)
            {
                const int slot = row * top_k + route;
                weights[static_cast<size_t>(slot)] /= sum;
            }
        }
    };

    for (int seq_len : {1, 2, 3, 4})
    {
        auto hidden = make_hidden(seq_len);
        ASSERT_TRUE(hidden->ensureOnDevice(device, stream));

        std::vector<float> route_indices;
        std::vector<float> route_weights;
        make_routes(seq_len, route_indices, route_weights);

        auto routing_indices = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
        auto routing_weights = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
        std::copy(route_indices.begin(), route_indices.end(), routing_indices->mutable_data());
        std::copy(route_weights.begin(), route_weights.end(), routing_weights->mutable_data());
        ASSERT_TRUE(routing_indices->ensureOnDevice(device, stream));
        ASSERT_TRUE(routing_weights->ensureOnDevice(device, stream));

        std::vector<float> static_table_decode_expected(
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model));
        std::vector<float> runtime_placement_decode_expected(
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model));
        for (int row = 0; row < seq_len; ++row)
        {
            auto row_hidden = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
            std::copy_n(hidden->data() + static_cast<size_t>(row) * d_model,
                        d_model,
                        row_hidden->mutable_data());
            ASSERT_TRUE(row_hidden->ensureOnDevice(device, stream));

            for (int slot = 0; slot < top_k; ++slot)
            {
                const int flat_slot = row * top_k + slot;
                host_runtime.topk_expert_ids[slot] =
                    static_cast<int32_t>(route_indices[static_cast<size_t>(flat_slot)]);
                host_runtime.topk_weights[slot] =
                    route_weights[static_cast<size_t>(flat_slot)];
            }

            auto run_serial_decode =
                [&](MoEDecodeDescriptorSource descriptor_source,
                    std::vector<float> *dst)
            {
                ASSERT_EQ(hipMemcpyAsync(device_runtime, &host_runtime, sizeof(host_runtime),
                                         hipMemcpyHostToDevice, stream),
                          hipSuccess);
                auto row_output = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
                ASSERT_TRUE(row_output->ensureOnDevice(device, stream));
                moe_kernel.zeroBuffer(row_output.get(), static_cast<size_t>(d_model) * sizeof(float));
                ASSERT_TRUE(moe_kernel.groupedExpertDecodeFromRuntime(
                    device_runtime,
                    row_hidden.get(),
                    gateup_table,
                    down_table,
                    top_k,
                    row_output.get(),
                    d_model,
                    intermediate,
                    descriptor_source));
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
                row_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
                std::copy_n(row_output->data(),
                            d_model,
                            dst->data() + static_cast<size_t>(row) * d_model);
            };

            run_serial_decode(MoEDecodeDescriptorSource::StaticDescriptorTable,
                              &static_table_decode_expected);
            run_serial_decode(MoEDecodeDescriptorSource::RuntimePlacementTable,
                              &runtime_placement_decode_expected);
        }

        expectBitwiseVerifierRowsEqual(
            ("ROCm Qwen3.6 IQ2_S/IQ4_XS static-table serial decode M=" +
             std::to_string(seq_len) +
             " must match runtime-placement serial decode").c_str(),
            static_table_decode_expected.data(),
            runtime_placement_decode_expected.data(),
            static_table_decode_expected.size(),
            static_cast<size_t>(d_model));

        auto grouped_output = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        ASSERT_TRUE(grouped_output->ensureOnDevice(device, stream));
        ASSERT_TRUE(moe_kernel.prepareExpertGroupsAsync(
            routing_indices.get(),
            routing_weights.get(),
            seq_len,
            num_experts,
            top_k));
        ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
            hidden.get(),
            grouped_output.get(),
            gateup_table,
            down_table,
            seq_len,
            d_model,
            intermediate,
            num_experts,
            top_k));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);

        expectBitwiseVerifierRowsEqual(
            ("ROCm Qwen3.6 IQ2_S/IQ4_XS grouped verifier M=" +
             std::to_string(seq_len) +
             " must match runtime-placement serial decode").c_str(),
            grouped_output->data(),
            runtime_placement_decode_expected.data(),
            grouped_output->numel(),
            static_cast<size_t>(d_model));

        /*
         * The model graph uses the runtime-table grouped prefill entry point
         * for LLEP / ExpertParallel verifier rows.  Keep this in the same
         * regression as the direct grouped pipeline so a future optimization
         * cannot accidentally re-enable original-slot atomic publication for
         * verifier-sized batches.
         */
        auto runtime_grouped_output = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        ASSERT_TRUE(runtime_grouped_output->ensureOnDevice(device, stream));
        ASSERT_TRUE(moe_kernel.groupPrefillRoutes(
            prefill_runtime_table.deviceLayerState(0),
            routing_indices.get(),
            routing_weights.get(),
            seq_len,
            runtime_config.prefill_token_capacity,
            num_experts,
            top_k));
        ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipelineFromRuntime(
            prefill_runtime_table.deviceLayerState(0),
            runtime_prefill_state,
            hidden.get(),
            runtime_grouped_output.get(),
            gateup_table,
            down_table,
            seq_len,
            d_model,
            intermediate,
            num_experts,
            top_k));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        runtime_grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);

        expectBitwiseVerifierRowsEqual(
            ("ROCm Qwen3.6 IQ2_S/IQ4_XS runtime grouped verifier M=" +
             std::to_string(seq_len) +
             " must match runtime-placement serial decode").c_str(),
            runtime_grouped_output->data(),
            runtime_placement_decode_expected.data(),
            runtime_grouped_output->numel(),
            static_cast<size_t>(d_model));
    }

    EXPECT_EQ(hipFree(device_runtime), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

/**
 * @brief Reproduce model-level ROCm M=3 routed-MoE drift with real Qwen3.6 weights.
 *
 * The synthetic all-format routed sweep proves that the ROCm grouped verifier
 * pipeline is structurally decode-equivalent for each codebook family, but it
 * previously failed to catch intermittent M=3 model parity breaks in `blk.3`
 * and `blk.27` of the real Qwen3.6 MoE checkpoint.  This regression closes
 * that gap by loading the exact expert tensor layout used by the model:
 *
 * - `blk.3.ffn_gate_exps.weight`: IQ2_S, `[2048, 512, 256]`
 * - `blk.3.ffn_up_exps.weight`: IQ2_S, `[2048, 512, 256]`
 * - `blk.3.ffn_down_exps.weight`: IQ4_XS, `[512, 2048, 256]`
 *
 * The selected layer's production grouped verifier path is then compared
 * byte-for-byte against the production one-row runtime-placement decode entry
 * point.  No serial row-replay fallback is exercised here: the oracle is the
 * existing device decode implementation that normal inference uses for M=1.
 */
void runRoutedOnlyVerifierPrefillQwen36RealWeightsM3MatchesRuntimeDecode(
    int layer_index)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;
    constexpr int seq_len = 3;

    const std::filesystem::path model_path = qwen36MoEModelPathForROCmMoEKernelTest();
    if (!std::filesystem::exists(model_path))
    {
        GTEST_SKIP() << "Qwen 3.6 MoE model not found at " << model_path
                     << "; set LLAMINAR_QWEN36_MOE_MODEL to run this real-weight regression";
    }

    auto model_ctx = loadQwen36MoEModelForROCmExpertWeights(model_path);
    ASSERT_NE(model_ctx, nullptr);
    const std::string layer_prefix = "blk." + std::to_string(layer_index);
    const std::string gate_tensor_name = layer_prefix + ".ffn_gate_exps.weight";
    const std::string up_tensor_name = layer_prefix + ".ffn_up_exps.weight";
    const std::string down_tensor_name = layer_prefix + ".ffn_down_exps.weight";
    const std::string router_tensor_name = layer_prefix + ".ffn_gate_inp.weight";
    ASSERT_TRUE(model_ctx->hasTensor(gate_tensor_name));
    ASSERT_TRUE(model_ctx->hasTensor(up_tensor_name));
    ASSERT_TRUE(model_ctx->hasTensor(down_tensor_name));
    ASSERT_TRUE(model_ctx->hasTensor(router_tensor_name));

    auto gate_parent = model_ctx->getWeightForDevice(gate_tensor_name, DeviceId::cpu());
    auto up_parent = model_ctx->getWeightForDevice(up_tensor_name, DeviceId::cpu());
    auto down_parent = model_ctx->getWeightForDevice(down_tensor_name, DeviceId::cpu());
    ASSERT_NE(gate_parent, nullptr);
    ASSERT_NE(up_parent, nullptr);
    ASSERT_NE(down_parent, nullptr);
    ASSERT_EQ(gate_parent->native_type(), TensorType::IQ2_S);
    ASSERT_EQ(up_parent->native_type(), TensorType::IQ2_S);
    ASSERT_EQ(down_parent->native_type(), TensorType::IQ4_XS);

    /*
     * Load the real router on the target device as well.  Production M=1
     * decode quantizes the normalized hidden row while computing router
     * logits, then lets the expert gate/up kernel reuse that device-owned Q8
     * row.  A focused expert test that starts after routing misses this
     * handoff and therefore cannot reproduce stale or non-equivalent router
     * scratch.  Keeping the router in this fixture makes the regression cover
     * the exact optimized dispatch used by the model graph.
     */
    auto router_weight =
        model_ctx->getWeightForDevice(router_tensor_name, device);
    ASSERT_NE(router_weight, nullptr);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    ScopedROCmEnvOverride enable_parallel_down(
        "LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE", "1");

    ROCmMoEKernel moe_kernel(0);
    static_cast<ITensorKernel &>(moe_kernel).setGPUStream(stream);
    auto moe_workspace = bindDefaultMoEWorkspace(
        moe_kernel,
        /*max_seq_len=*/seq_len,
        d_model,
        intermediate,
        num_experts,
        top_k);

    /*
     * Layer 27 retains the exact route captured from the original failure so
     * the production router itself remains under byte-level regression.  The
     * `experts` array is then replaced with the current production route and
     * becomes the authoritative preparation mask for both test layers.
     */
    std::array<int, seq_len * top_k> experts{};
    std::array<uint32_t, seq_len * top_k> captured_layer27_weight_bits{};
    if (layer_index == 27)
    {
        /*
         * Exact top-k route order and FP32 weights captured from the
         * intermittent model-level ROCm M=3 failure.  Row 0 was identical
         * through FFN norm and routing, then diverged in the routed expert
         * output; rows 1 and 2 inherited that state drift in layer 28.
         */
        experts = {
            54, 250, 121, 210, 177, 87, 135, 110,
            121, 206, 177, 220, 233, 230, 48, 238,
            87, 165, 88, 229, 48, 242, 185, 58};
        captured_layer27_weight_bits = {
            /*
             * Row zero was refreshed after router-hidden Q8 publication adopted
             * the reciprocal-multiply arithmetic used by ordinary NativeVNNI
             * M=1 activation quantization.  Rows one and two did not cross a
             * rounding boundary and retain their original captured bytes.
             */
            0x3e3d5903u, 0x3e29bed3u, 0x3df7b505u, 0x3df7143du,
            0x3ddafa51u, 0x3dd6786bu, 0x3dcfff59u, 0x3dc194f9u,
            0x3e48b871u, 0x3e33dc01u, 0x3e0f49bfu, 0x3ddea924u,
            0x3dd70439u, 0x3dc90252u, 0x3db5544au, 0x3db43fa8u,
            0x3e4cdf6au, 0x3e34c6d5u, 0x3e17d0a1u, 0x3dded8d6u,
            0x3dce15f2u, 0x3dcbf61cu, 0x3db203f8u, 0x3da22962u};
    }
    else if (layer_index != 3)
    {
        FAIL() << "No captured Qwen3.6 routed verifier fixture for layer "
               << layer_index;
    }

    /*
     * Route before preparing expert payloads.  The layer-27 snapshot has an
     * exact captured route that we validate below.  Layer 3 intentionally uses
     * a deterministic synthetic hidden tensor, so its old copied route IDs did
     * not belong to this input.  Preparing descriptors from those unrelated
     * IDs left production-routed experts resident-but-not-ready and correctly
     * tripped the runtime publication invariant.  Deriving the preparation
     * mask from the production router output makes both cases exercise the
     * same router-first ownership order as the model graph.
     */
    auto hidden = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    if (layer_index == 27)
    {
        static constexpr std::array<uint32_t, seq_len * d_model>
            kLayer27ProductionHiddenBits = {
#include "fixtures/Qwen36Layer27M3FFNNormBits.inc"
            };
        static_assert(kLayer27ProductionHiddenBits.size() ==
                      static_cast<size_t>(seq_len * d_model));
        for (size_t i = 0; i < hidden->numel(); ++i)
        {
            hidden->mutable_data()[i] =
                f32FromBitsForROCmMoEKernelTest(
                    kLayer27ProductionHiddenBits[i]);
        }
    }
    else
    {
        for (size_t i = 0; i < hidden->numel(); ++i)
        {
            const int col = static_cast<int>(i % d_model);
            const int row = static_cast<int>(i / d_model);
            const float low_frequency =
                0.39f * std::sin(0.007f * static_cast<float>(col + 17 * row));
            const float high_frequency =
                0.27f * std::cos(0.031f * static_cast<float>(col + 31));
            const float saw = 0.018f * static_cast<float>((col % 23) - 11);
            const float row_phase =
                0.11f * std::sin(
                            0.013f * static_cast<float>((row + 1) * (col + 5)));
            hidden->mutable_data()[i] =
                low_frequency + high_frequency + saw + row_phase;
        }
    }
    ASSERT_TRUE(hidden->ensureOnDevice(device, stream));

    auto routing_indices = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
    auto routing_weights = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
    ASSERT_TRUE(routing_indices->ensureOnDevice(device, stream));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device, stream));
    ASSERT_TRUE(router_weight->ensureOnDevice(device, stream));
    ASSERT_TRUE(moe_kernel.routeVerifierRowsDecodeEquivalent(
        hidden.get(),
        router_weight.get(),
        seq_len,
        d_model,
        num_experts,
        top_k,
        /*normalize_weights=*/true,
        routing_indices.get(),
        routing_weights.get()));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    routing_indices->transitionTo(
        TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    routing_weights->transitionTo(
        TensorCoherenceState::DEVICE_AUTHORITATIVE, device);

    for (size_t route = 0; route < experts.size(); ++route)
    {
        const int production_expert =
            static_cast<int>(routing_indices->data()[route]);
        const uint32_t production_weight_bits =
            f32BitsForROCmMoEKernelTest(routing_weights->data()[route]);
        if (layer_index == 27)
        {
            EXPECT_EQ(production_expert, experts[route])
                << "layer-27 captured expert route drift at slot " << route;
            EXPECT_EQ(
                production_weight_bits,
                captured_layer27_weight_bits[route])
                << "layer-27 captured route-weight drift at slot " << route;
        }
        experts[route] = production_expert;
    }

    std::vector<int> used_experts(experts.begin(), experts.end());
    std::sort(used_experts.begin(), used_experts.end());
    used_experts.erase(std::unique(used_experts.begin(), used_experts.end()), used_experts.end());

    struct PreparedExpertTriplet
    {
        ITensorGemm *gate = nullptr;
        ITensorGemm *up = nullptr;
        ITensorGemm *down = nullptr;
    };

    std::array<PreparedExpertTriplet, num_experts> routed{};
    std::vector<bool> expert_mask(static_cast<size_t>(num_experts), false);
    for (const int expert : used_experts)
    {
        expert_mask[static_cast<size_t>(expert)] = true;
    }

    std::vector<std::shared_ptr<TensorBase>> gate_views;
    std::vector<std::shared_ptr<TensorBase>> up_views;
    std::vector<std::shared_ptr<TensorBase>> down_views;
    std::vector<ITensorGemm *> gate_gemms;
    std::vector<ITensorGemm *> up_gemms;
    std::vector<ITensorGemm *> down_gemms;
    std::vector<std::shared_ptr<ITensorGemm>> owned_gemms;
    std::shared_ptr<void> gate_pool_lifetime;
    std::shared_ptr<void> up_pool_lifetime;
    std::shared_ptr<void> down_pool_lifetime;

    /*
     * Use the model graph's real expert preparation service.  It extracts all
     * 3D GGUF views and batches mask-active experts through one
     * LoadOrchestrator native-VNNI pool.  Preparing each view independently is
     * not a production-equivalent oracle: it changes packed-payload ownership,
     * slot alignment, and descriptor pointer layout.
     */
    MoEWeightContext weight_context{
        device,
        num_experts,
        intermediate,
        d_model,
        /*local_expert_start=*/0,
        /*local_expert_count=*/num_experts,
        layer_index,
        expert_mask,
        gate_parent.get(),
        up_parent.get(),
        down_parent.get(),
        gate_views,
        up_views,
        down_views,
        gate_gemms,
        up_gemms,
        down_gemms,
        owned_gemms,
        gate_pool_lifetime,
        up_pool_lifetime,
        down_pool_lifetime};
    weight_context.advise_raw_pages_after_prepare = false;
    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(weight_context));
    ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(weight_context));

    for (const int expert : used_experts)
    {
        routed[static_cast<size_t>(expert)].gate =
            gate_gemms[static_cast<size_t>(expert)];
        routed[static_cast<size_t>(expert)].up =
            up_gemms[static_cast<size_t>(expert)];
        routed[static_cast<size_t>(expert)].down =
            down_gemms[static_cast<size_t>(expert)];
        ASSERT_NE(routed[static_cast<size_t>(expert)].gate, nullptr);
        ASSERT_NE(routed[static_cast<size_t>(expert)].up, nullptr);
        ASSERT_NE(routed[static_cast<size_t>(expert)].down, nullptr);
    }

    WorkspaceRequirements gemm_reqs;
    auto merge_projection_reqs = [&](ITensorGemm *kernel, int N, int K)
    {
        auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
        ASSERT_NE(consumer, nullptr);
        gemm_reqs.merge(consumer->getWorkspaceRequirements(seq_len, N, K));
        gemm_reqs.merge(consumer->getWorkspaceRequirements(1, N, K));
    };
    for (const int expert : used_experts)
    {
        const PreparedExpertTriplet &triplet = routed[static_cast<size_t>(expert)];
        merge_projection_reqs(triplet.gate, intermediate, d_model);
        merge_projection_reqs(triplet.up, intermediate, d_model);
        merge_projection_reqs(triplet.down, d_model, intermediate);
    }
    auto gemm_workspace = std::make_unique<DeviceWorkspaceManager>(device, 512 * 1024 * 1024);
    ASSERT_TRUE(gemm_workspace->allocate(gemm_reqs));

    auto bind_gemm = [&](ITensorGemm *kernel)
    {
        auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
        ASSERT_NE(consumer, nullptr);
        consumer->bindWorkspace(gemm_workspace.get());
        kernel->prepareWeights();
        ASSERT_TRUE(kernel->weights_converted());
    };
    for (const int expert : used_experts)
    {
        const PreparedExpertTriplet &triplet = routed[static_cast<size_t>(expert)];
        bind_gemm(triplet.gate);
        bind_gemm(triplet.up);
        bind_gemm(triplet.down);
    }

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(static_cast<size_t>(num_experts));
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(static_cast<size_t>(num_experts));
    std::vector<DeviceNativeVNNIMatrixDesc> down_descs(static_cast<size_t>(num_experts));
    for (const int expert : used_experts)
    {
        const PreparedExpertTriplet &triplet = routed[static_cast<size_t>(expert)];
        ASSERT_TRUE(triplet.gate->exportNativeVNNIMatrixDesc(gate_descs[static_cast<size_t>(expert)]));
        ASSERT_TRUE(triplet.up->exportNativeVNNIMatrixDesc(up_descs[static_cast<size_t>(expert)]));
        ASSERT_TRUE(triplet.down->exportNativeVNNIMatrixDesc(down_descs[static_cast<size_t>(expert)]));
    }

    const int gateup_table = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    DeviceMoELayerRuntime host_runtime = makeAllLocalRuntime(num_experts, top_k);
    populateRuntimeDescriptors(host_runtime, &gate_descs, &up_descs, &down_descs);
    DeviceMoELayerRuntime *device_runtime = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_runtime), sizeof(DeviceMoELayerRuntime)), hipSuccess);

    DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    DeviceMoERuntimeTable prefill_runtime_table(runtime_config);
    auto &runtime_prefill_state = prefill_runtime_table.hostLayerState(0);
    populateRuntimeDescriptors(
        runtime_prefill_state,
        &gate_descs,
        &up_descs,
        &down_descs);
    ASSERT_EQ(
        hipMemcpyAsync(
            prefill_runtime_table.deviceLayerState(0),
            &runtime_prefill_state,
            sizeof(runtime_prefill_state),
            hipMemcpyHostToDevice,
            stream),
        hipSuccess);

    auto publish_grouped_router_q8 = [&]() -> bool
    {
        return moe_kernel.routeVerifierRowsDecodeEquivalent(
            hidden.get(),
            router_weight.get(),
            seq_len,
            d_model,
            num_experts,
            top_k,
            /*normalize_weights=*/true,
            routing_indices.get(),
            routing_weights.get());
    };

    /*
     * Match the model-level proof order: grouped verifier execution happens
     * before the restored serial M=1 oracle.  This warmup is intentionally not
     * used as an expected result; it detects backend dispatch/autotuner state
     * that leaks from M=3 into the later one-row production path.
     */
    auto grouped_before_serial = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(grouped_before_serial->ensureOnDevice(device, stream));
    ASSERT_TRUE(publish_grouped_router_q8());
    ASSERT_TRUE(moe_kernel.groupPrefillRoutes(
        prefill_runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        runtime_config.prefill_token_capacity,
        num_experts,
        top_k));
    ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipelineFromRuntime(
        prefill_runtime_table.deviceLayerState(0),
        runtime_prefill_state,
        hidden.get(),
        grouped_before_serial.get(),
        gateup_table,
        down_table,
        seq_len,
        d_model,
        intermediate,
        num_experts,
        top_k));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::vector<float> router_reuse_decode_expected(
        static_cast<size_t>(seq_len) * static_cast<size_t>(d_model));
    for (int row = 0; row < seq_len; ++row)
    {
        auto row_hidden = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
        std::copy_n(hidden->data() + static_cast<size_t>(row) * d_model,
                    d_model,
                    row_hidden->mutable_data());
        ASSERT_TRUE(row_hidden->ensureOnDevice(device, stream));

        /*
         * Exercise the complete production router-to-expert M=1 path.  The
         * router writes top-k state directly into the runtime layer and leaves
         * its freshly quantized hidden row resident in ROCmMoEKernel scratch;
         * groupedExpertDecodeFromRuntime must consume precisely those bytes.
         * Validate the route itself before comparing expert output so any
         * failure identifies either routing or the Q8 reuse handoff, rather
         * than presenting as an opaque downstream MoE mismatch.
         */
        ASSERT_EQ(hipMemcpyAsync(device_runtime, &host_runtime, sizeof(host_runtime),
                                 hipMemcpyHostToDevice, stream),
                  hipSuccess);
        auto production_route_indices = TestTensorFactory::createFP32(
            {static_cast<size_t>(top_k)});
        auto production_route_weights = TestTensorFactory::createFP32(
            {static_cast<size_t>(top_k)});
        ASSERT_TRUE(production_route_indices->ensureOnDevice(device, stream));
        ASSERT_TRUE(production_route_weights->ensureOnDevice(device, stream));
        ASSERT_TRUE(moe_kernel.decodeRouteSelect(
            device_runtime,
            row_hidden.get(),
            router_weight.get(),
            d_model,
            num_experts,
            top_k,
            /*normalize_weights=*/true,
            production_route_indices.get(),
            production_route_weights.get(),
            /*write_legacy_outputs=*/true,
            /*update_runtime_histogram=*/false));

        auto router_reuse_row_output = TestTensorFactory::createFP32(
            {1u, static_cast<size_t>(d_model)});
        ASSERT_TRUE(router_reuse_row_output->ensureOnDevice(device, stream));
        moe_kernel.zeroBuffer(
            router_reuse_row_output.get(),
            static_cast<size_t>(d_model) * sizeof(float));
        ASSERT_TRUE(moe_kernel.groupedExpertDecodeFromRuntime(
            device_runtime,
            row_hidden.get(),
            gateup_table,
            down_table,
            top_k,
            router_reuse_row_output.get(),
            d_model,
            intermediate,
            MoEDecodeDescriptorSource::RuntimePlacementTable));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        production_route_indices->transitionTo(
            TensorCoherenceState::DEVICE_AUTHORITATIVE,
            device);
        production_route_weights->transitionTo(
            TensorCoherenceState::DEVICE_AUTHORITATIVE,
            device);
        router_reuse_row_output->transitionTo(
            TensorCoherenceState::DEVICE_AUTHORITATIVE,
            device);
        for (int slot = 0; slot < top_k; ++slot)
        {
            const size_t flat_slot =
                static_cast<size_t>(row * top_k + slot);
            EXPECT_EQ(
                static_cast<int>(production_route_indices->data()[slot]),
                experts[flat_slot])
                << "production router expert mismatch at row=" << row
                << " slot=" << slot;
        }
        expectBitwiseVerifierRowsEqual(
            "ROCm Qwen3.6 production router weights must reproduce the captured serial route",
            production_route_weights->data(),
            routing_weights->data() + static_cast<size_t>(row) * top_k,
            static_cast<size_t>(top_k),
            static_cast<size_t>(top_k));
        std::copy_n(
            router_reuse_row_output->data(),
            d_model,
            router_reuse_decode_expected.data() +
                static_cast<size_t>(row) * d_model);
    }

    auto grouped_output = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(grouped_output->ensureOnDevice(device, stream));
    ASSERT_TRUE(publish_grouped_router_q8());
    ASSERT_TRUE(moe_kernel.prepareExpertGroupsAsync(
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        num_experts,
        top_k));
    ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
        hidden.get(),
        grouped_output.get(),
        gateup_table,
        down_table,
        seq_len,
        d_model,
        intermediate,
        num_experts,
        top_k));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);

    expectBitwiseVerifierRowsEqual(
        ("ROCm Qwen3.6 layer" + std::to_string(layer_index) +
         " real-weight routed grouped verifier M=3 must match router-Q8 serial decode").c_str(),
        grouped_output->data(),
        router_reuse_decode_expected.data(),
        grouped_output->numel(),
        static_cast<size_t>(d_model));

    auto runtime_grouped_output = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(runtime_grouped_output->ensureOnDevice(device, stream));
    ASSERT_TRUE(publish_grouped_router_q8());
    ASSERT_TRUE(moe_kernel.groupPrefillRoutes(
        prefill_runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        runtime_config.prefill_token_capacity,
        num_experts,
        top_k));
    ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipelineFromRuntime(
        prefill_runtime_table.deviceLayerState(0),
        runtime_prefill_state,
        hidden.get(),
        runtime_grouped_output.get(),
        gateup_table,
        down_table,
        seq_len,
        d_model,
        intermediate,
        num_experts,
        top_k));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    runtime_grouped_output->transitionTo(
        TensorCoherenceState::DEVICE_AUTHORITATIVE,
        device);

    expectBitwiseVerifierRowsEqual(
        ("ROCm Qwen3.6 layer" + std::to_string(layer_index) +
         " real-weight runtime-table grouped verifier M=3 must match router-Q8 serial decode").c_str(),
        runtime_grouped_output->data(),
        router_reuse_decode_expected.data(),
        runtime_grouped_output->numel(),
        static_cast<size_t>(d_model));

    auto captured_runtime_output = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(captured_runtime_output->ensureOnDevice(device, stream));
    hipGraph_t graph = nullptr;
    ASSERT_EQ(
        hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
        hipSuccess);
    const bool captured_routing = publish_grouped_router_q8();
    const bool captured_grouping = moe_kernel.groupPrefillRoutes(
        prefill_runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        runtime_config.prefill_token_capacity,
        num_experts,
        top_k);
    const bool captured_pipeline =
        moe_kernel.executeGroupedPrefillPipelineFromRuntime(
            prefill_runtime_table.deviceLayerState(0),
            runtime_prefill_state,
            hidden.get(),
            captured_runtime_output.get(),
            gateup_table,
            down_table,
            seq_len,
            d_model,
            intermediate,
            num_experts,
            top_k);
    const hipError_t capture_status = hipStreamEndCapture(stream, &graph);
    ASSERT_TRUE(captured_routing);
    ASSERT_TRUE(captured_grouping);
    ASSERT_TRUE(captured_pipeline);
    ASSERT_EQ(capture_status, hipSuccess) << hipGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);

    hipGraphExec_t graph_exec = nullptr;
    ASSERT_EQ(
        hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
        hipSuccess);
    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    captured_runtime_output->transitionTo(
        TensorCoherenceState::DEVICE_AUTHORITATIVE,
        device);

    expectBitwiseVerifierRowsEqual(
        ("ROCm Qwen3.6 layer" + std::to_string(layer_index) +
         " graph-captured real-weight runtime-table verifier M=3 must match router-Q8 serial decode").c_str(),
        captured_runtime_output->data(),
        router_reuse_decode_expected.data(),
        captured_runtime_output->numel(),
        static_cast<size_t>(d_model));

    EXPECT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);

    EXPECT_EQ(hipFree(device_runtime), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, RoutedOnlyVerifierPrefill_Qwen36Layer3RealWeights_M3MatchesRuntimeDecode)
{
    runRoutedOnlyVerifierPrefillQwen36RealWeightsM3MatchesRuntimeDecode(3);
}

TEST(Test__ROCmMoEKernel, RoutedOnlyVerifierPrefill_Qwen36Layer27RealWeights_M3MatchesRuntimeDecode)
{
    runRoutedOnlyVerifierPrefillQwen36RealWeightsM3MatchesRuntimeDecode(27);
}

// ============================================================================
// Test: groupTokensByExpertDevice() — prefill scale
// ============================================================================

TEST(Test__ROCmMoEKernel, GroupTokensByExpert_PrefillScale)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 32;
    const int num_experts = 8;
    const int top_k = 4;
    const int total_slots = seq_len * top_k; // 128

    // Random routing indices (uniform over experts)
    std::vector<int> routing_indices(total_slots);
    std::vector<float> routing_weights(total_slots);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> expert_dist(0, num_experts - 1);
    std::uniform_real_distribution<float> weight_dist(0.01f, 0.5f);
    for (int i = 0; i < total_slots; ++i)
    {
        routing_indices[i] = expert_dist(gen);
        routing_weights[i] = weight_dist(gen);
    }

    // Upload to device
    int *d_routing_indices = nullptr;
    float *d_routing_weights = nullptr;
    (void)hipMalloc(&d_routing_indices, total_slots * sizeof(int));
    (void)hipMalloc(&d_routing_weights, total_slots * sizeof(float));
    (void)hipMemcpy(d_routing_indices, routing_indices.data(), total_slots * sizeof(int), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_routing_weights, routing_weights.data(), total_slots * sizeof(float), hipMemcpyHostToDevice);

    // Allocate output buffers
    int *d_expert_offsets = nullptr, *d_expert_counts = nullptr;
    int *d_grouped_indices = nullptr;
    float *d_grouped_weights = nullptr;
    (void)hipMalloc(&d_expert_offsets, num_experts * sizeof(int));
    (void)hipMalloc(&d_expert_counts, num_experts * sizeof(int));
    (void)hipMalloc(&d_grouped_indices, total_slots * sizeof(int));
    (void)hipMalloc(&d_grouped_weights, total_slots * sizeof(float));

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    bool ok = gpu_kernel.groupTokensByExpertDevice(
        d_routing_indices, d_routing_weights,
        seq_len, num_experts, top_k,
        d_expert_offsets, d_expert_counts,
        d_grouped_indices, d_grouped_weights);
    ASSERT_TRUE(ok) << "groupTokensByExpertDevice failed";

    (void)hipDeviceSynchronize();

    // D2H copy results
    std::vector<int> host_offsets(num_experts);
    std::vector<int> host_counts(num_experts);
    std::vector<int> host_grouped_indices(total_slots);
    std::vector<float> host_grouped_weights(total_slots);
    (void)hipMemcpy(host_offsets.data(), d_expert_offsets, num_experts * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipMemcpy(host_counts.data(), d_expert_counts, num_experts * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipMemcpy(host_grouped_indices.data(), d_grouped_indices, total_slots * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipMemcpy(host_grouped_weights.data(), d_grouped_weights, total_slots * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_routing_indices);
    (void)hipFree(d_routing_weights);
    (void)hipFree(d_expert_offsets);
    (void)hipFree(d_expert_counts);
    (void)hipFree(d_grouped_indices);
    (void)hipFree(d_grouped_weights);

    // Verify: sum of all expert_counts == total_slots
    int sum_counts = 0;
    for (int e = 0; e < num_experts; ++e)
        sum_counts += host_counts[e];
    EXPECT_EQ(sum_counts, total_slots)
        << "Sum of expert counts should equal total_slots";

    // Verify: offsets are consistent
    int running = 0;
    for (int e = 0; e < num_experts; ++e)
    {
        EXPECT_EQ(host_offsets[e], running)
            << "Expert " << e << " offset mismatch";
        running += host_counts[e];
    }

    // Verify: each token appears exactly top_k times across all groups
    std::vector<int> token_appearances(seq_len, 0);
    for (int i = 0; i < total_slots; ++i)
    {
        int tok = host_grouped_indices[i];
        ASSERT_GE(tok, 0) << "Grouped token index " << i << " is negative";
        ASSERT_LT(tok, seq_len) << "Grouped token index " << i << " out of range: " << tok;
        token_appearances[tok]++;
    }

    bool all_tokens_accounted = true;
    for (int t = 0; t < seq_len; ++t)
    {
        if (token_appearances[t] != top_k)
        {
            ADD_FAILURE() << "Token " << t << " appears " << token_appearances[t]
                          << " times, expected " << top_k;
            all_tokens_accounted = false;
        }
    }

    // Verify: grouped weights match original weights
    // Build a reference: for each expert, collect the expected (token, weight) pairs
    std::vector<std::vector<std::pair<int, float>>> expected_per_expert(num_experts);
    for (int s = 0; s < total_slots; ++s)
    {
        int token = s / top_k;
        int expert = routing_indices[s];
        expected_per_expert[expert].push_back({token, routing_weights[s]});
    }

    bool weights_match = true;
    for (int e = 0; e < num_experts; ++e)
    {
        int offset = host_offsets[e];
        int count = host_counts[e];

        // Collect actual
        std::vector<std::pair<int, float>> actual;
        for (int i = 0; i < count; ++i)
            actual.push_back({host_grouped_indices[offset + i],
                              host_grouped_weights[offset + i]});

        // Check each expected entry exists
        for (const auto &[exp_tok, exp_wt] : expected_per_expert[e])
        {
            bool found = false;
            for (auto &[act_tok, act_wt] : actual)
            {
                if (act_tok == exp_tok && act_wt == exp_wt)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                weights_match = false;
                break;
            }
        }
        if (!weights_match)
            break;
    }
    EXPECT_TRUE(weights_match) << "Grouped weights don't match original routing weights";

    std::cout << "[GroupTokensByExpert_PrefillScale] total_slots=" << total_slots
              << " sum_counts=" << sum_counts
              << " all_tokens_accounted=" << (all_tokens_accounted ? "true" : "false")
              << " weights_match=" << (weights_match ? "true" : "false") << std::endl;
}

#else // !HAVE_ROCM

TEST(Test__ROCmMoEKernel, SkippedNoROCm)
{
    GTEST_SKIP() << "ROCm not available in this build";
}

#endif // HAVE_ROCM
