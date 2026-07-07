#include <gtest/gtest.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"

#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/compute_stages/ComputeStageUtils.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/DecodeExpertHistogram.h"
#include "execution/moe/DeviceMoERebalanceController.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "execution/moe/MoERuntimeTable.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/TestTensorFactory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
extern "C" bool cudaMoE_count_per_expert(
    const int *routing_indices,
    int *expert_counts,
    int total_slots,
    int num_experts,
    int device_idx,
    void *stream);

extern "C" bool cudaMoE_exclusive_scan(
    const int *expert_counts,
    int *expert_offsets,
    int num_experts,
    int device_idx,
    void *stream);

extern "C" bool cudaMoE_scatter_tokens_deterministic(
    const int *routing_indices,
    const float *routing_weights,
    const int *expert_offsets,
    const int *expert_counts,
    int *grouped_token_indices,
    int *original_to_grouped,
    int *original_expert_ids,
    float *grouped_weights,
    int total_slots,
    int top_k,
    int num_experts,
    int device_idx,
    void *stream);

extern "C" bool cudaMoE_group_tokens_small_float(
    const float *routing_indices,
    const float *routing_weights,
    int *expert_counts,
    int *expert_offsets,
    int *grouped_token_indices,
    int *original_to_grouped,
    int *original_expert_ids,
    float *grouped_weights,
    int *active_expert_ids,
    int total_slots,
    int num_experts,
    int top_k,
    int max_active_experts,
    int device_idx,
    void *stream);

extern "C" bool cudaMoE_group_tokens_small_float_with_shared_gate(
    const float *routing_indices,
    const float *routing_weights,
    const float *hidden,
    const float *shared_gate_inp,
    int *expert_counts,
    int *expert_offsets,
    int *grouped_token_indices,
    int *original_to_grouped,
    int *original_expert_ids,
    int *single_expert_ids,
    int *active_expert_ids,
    float *grouped_weights,
    int seq_len,
    int d_model,
    int num_routed_experts,
    int routed_top_k,
    int max_active_experts,
    int device_idx,
    void *stream);

extern "C" bool cudaMoE_route_logits(
    const float *hidden,
    const float *gate_weights,
    float *logits,
    int seq_len,
    int d_model,
    int num_experts,
    int device_idx,
    void *stream);

extern "C" bool cudaMoE_materialize_runtime_prefill_descriptor_tables(
    const void *runtime,
    llaminar2::DeviceNativeVNNIMatrixDesc *gate_descs,
    llaminar2::DeviceNativeVNNIMatrixDesc *up_descs,
    llaminar2::DeviceNativeVNNIMatrixDesc *down_descs,
    int num_experts,
    int device_idx,
    void *stream);

extern "C" bool cudaMoE_build_runtime_original_to_grouped(
    const void *runtime,
    int *original_to_grouped,
    int current_slots,
    int max_slots,
    int num_experts,
    int top_k,
    int device_idx,
    void *stream);

#endif

/**
 * @file Test__CUDAMoEKernel.cpp
 * @brief Focused CUDA MoE kernel parity tests against the CPU implementation.
 *
 * These tests exercise the public `IMoEKernel` surface used by compute stages:
 * tensor-aware routing, gather/scatter, fallback SwiGLU, zeroing, and the
 * device-side grouping path that drives per-expert CUDA prefill dispatch. The
 * fixtures use tiny deterministic tensors and skip cleanly when CUDA hardware
 * is unavailable.
 */

namespace
{
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    std::shared_ptr<llaminar2::FP32Tensor> makeTensor(const std::vector<size_t> &shape,
                                                      const std::vector<float> &values)
    {
        auto tensor = std::make_shared<llaminar2::FP32Tensor>(shape);
        float *data = tensor->mutable_data();
        std::copy(values.begin(), values.end(), data);
        return tensor;
    }

    std::shared_ptr<llaminar2::BF16Tensor> makeBF16Tensor(const std::vector<size_t> &shape,
                                                          const std::vector<float> &values)
    {
        auto tensor = std::make_shared<llaminar2::BF16Tensor>(shape);
        tensor->from_fp32(values.data(), values.size());
        return tensor;
    }

    std::shared_ptr<llaminar2::FP32Tensor> makeZeros(const std::vector<size_t> &shape)
    {
        auto tensor = std::make_shared<llaminar2::FP32Tensor>(shape);
        std::fill(tensor->mutable_data(), tensor->mutable_data() + tensor->numel(), 0.0f);
        return tensor;
    }

    llaminar2::DeviceNativeVNNIMatrixDesc fakeNativeVNNIDesc(uintptr_t base)
    {
        llaminar2::DeviceNativeVNNIMatrixDesc desc;
        desc.payload = reinterpret_cast<const uint8_t *>(base);
        desc.scales = reinterpret_cast<const void *>(base + 0x100u);
        desc.mins = reinterpret_cast<const void *>(base + 0x200u);
        desc.n = 8;
        desc.k = 8;
        desc.blocks_per_row = 1;
        desc.codebook_id = 7;
        return desc;
    }

    llaminar2::DeviceNativeVNNIMatrixDesc makePrefillRuntimeDesc(
        uintptr_t base,
        int rows,
        int cols,
        uint8_t codebook_id)
    {
        llaminar2::DeviceNativeVNNIMatrixDesc desc{};
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

    void seedRuntimeBankExpert(
        llaminar2::DeviceMoEPlacementBank &bank,
        int expert,
        int owner_participant,
        bool local_ready)
    {
        auto &desc = bank.experts[expert];
        const uintptr_t base = 0x39000000u + static_cast<uintptr_t>(expert) * 0x4000u;
        desc.logical_expert_id = expert;
        desc.owner_participant = owner_participant;
        desc.local_slot = local_ready ? expert : -1;
        desc.flags = llaminar2::toMoEExpertFlags(llaminar2::DeviceMoEExpertFlags::Valid |
                                                 llaminar2::DeviceMoEExpertFlags::Resident);
        bank.local_compute_mask[expert] = local_ready ? 1u : 0u;
        bank.replica_role[expert] =
            local_ready
                ? static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::Primary)
                : static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::None);
        if (local_ready)
        {
            desc.gate = fakeNativeVNNIDesc(base + 0x10u);
            desc.up = fakeNativeVNNIDesc(base + 0x20u);
            desc.down = fakeNativeVNNIDesc(base + 0x30u);
            desc.flags |= llaminar2::toMoEExpertFlags(
                llaminar2::DeviceMoEExpertFlags::LocalCompute);
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
    llaminar2::DeviceMoERebalanceApplyStatus makeCleanRebalanceApplyStatus()
    {
        llaminar2::DeviceMoERebalanceApplyStatus status{};
        status.magic = llaminar2::kDeviceMoERebalanceMagic;
        status.version = llaminar2::kDeviceMoERebalanceVersion;
        status.status_code =
            static_cast<uint32_t>(llaminar2::DeviceMoERebalanceApplyStatusCode::Ok);
        return status;
    }

    void expectSameNativeVNNIDesc(
        const llaminar2::DeviceNativeVNNIMatrixDesc &actual,
        const llaminar2::DeviceNativeVNNIMatrixDesc &expected)
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

    llaminar2::DeviceMoELayerRuntime makeAllLocalRuntime(
        int num_experts,
        int top_k,
        uint32_t epoch = 1)
    {
        llaminar2::DeviceMoELayerRuntime runtime{};
        runtime.active_bank = 0;
        runtime.active_epoch = epoch;
        runtime.expert_count = static_cast<uint32_t>(num_experts);
        runtime.top_k = static_cast<uint32_t>(top_k);
        for (auto &bank : runtime.banks)
        {
            bank.epoch = epoch;
            bank.expert_count = static_cast<uint32_t>(num_experts);
            const int capped_experts =
                std::min(num_experts, static_cast<int>(llaminar2::kDeviceMoEMaxExperts));
            for (int expert = 0; expert < capped_experts; ++expert)
            {
                bank.local_compute_mask[expert] = 1;
                bank.replica_role[expert] =
                    static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::Primary);
                bank.resident_participant_mask[expert] = 1u;
            }
        }
        return runtime;
    }

    void markAllExpertsLocalForRouting(llaminar2::MoEPlacementUpdate &update, int num_experts)
    {
        update.local_compute_mask.assign(num_experts, 1u);
        update.replica_role.assign(num_experts, static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::Primary));
        for (int expert = 0; expert < num_experts; ++expert)
        {
            auto &desc = update.experts[static_cast<size_t>(expert)];
            const uintptr_t base = 0x30000000u + static_cast<uintptr_t>(expert) * 0x1000u;
            desc.logical_expert_id = expert;
            desc.owner_participant = 0;
            desc.local_slot = expert;
            desc.gate = fakeNativeVNNIDesc(base + 0x10u);
            desc.up = fakeNativeVNNIDesc(base + 0x20u);
            desc.down = fakeNativeVNNIDesc(base + 0x30u);
            desc.flags = llaminar2::toMoEExpertFlags(llaminar2::DeviceMoEExpertFlags::Valid |
                                                     llaminar2::DeviceMoEExpertFlags::Resident |
                                                     llaminar2::DeviceMoEExpertFlags::PreferredOwner |
                                                     llaminar2::DeviceMoEExpertFlags::LocalCompute);
        }
    }

    llaminar2::MoEPlacementUpdate makeParticipantOneBaseUpdate(uint32_t epoch)
    {
        constexpr int num_experts = 4;
        llaminar2::MoEPlacementUpdate update;
        update.epoch = epoch;
        update.expert_count = num_experts;
        update.participant_id = 1;
        update.participant_count = 3;
        update.experts.resize(num_experts);
        update.local_compute_mask = {0, 0, 1, 0};
        update.replica_role = {
            static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::None),
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
            desc.flags = llaminar2::toMoEExpertFlags(llaminar2::DeviceMoEExpertFlags::Valid |
                                                     llaminar2::DeviceMoEExpertFlags::Resident |
                                                     llaminar2::DeviceMoEExpertFlags::PreferredOwner);
            if (update.local_compute_mask[static_cast<size_t>(expert)] != 0u)
                desc.flags |= llaminar2::toMoEExpertFlags(llaminar2::DeviceMoEExpertFlags::LocalCompute);
        }
        return update;
    }

    llaminar2::MoEPlacementUpdate makeDynamicOwnershipTransferUpdate(uint32_t epoch,
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
            static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::None));
        for (int expert = 0; expert < update.expert_count; ++expert)
        {
            auto &desc = update.experts[static_cast<size_t>(expert)];
            desc.owner_participant = owners[expert];
            desc.flags = llaminar2::toMoEExpertFlags(llaminar2::DeviceMoEExpertFlags::Valid |
                                                     llaminar2::DeviceMoEExpertFlags::Resident |
                                                     llaminar2::DeviceMoEExpertFlags::PreferredOwner);
            if (owners[expert] == static_cast<int>(participant_id))
            {
                update.local_compute_mask[static_cast<size_t>(expert)] = 1u;
                update.replica_role[static_cast<size_t>(expert)] =
                    static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::Primary);
                desc.flags |= llaminar2::toMoEExpertFlags(llaminar2::DeviceMoEExpertFlags::LocalCompute);
            }
        }
        return update;
    }

    void expectNearArray(const float *actual, const float *expected, size_t count, float tolerance = 1.0e-5f)
    {
        for (size_t i = 0; i < count; ++i)
            ASSERT_NEAR(actual[i], expected[i], tolerance) << "at element " << i;
    }

#ifdef HAVE_CUDA
    /**
     * @brief Copy a CUDA-resident FP32 tensor into host memory for strict parity checks.
     *
     * The MTP verifier bugs chased by these tests often involve tensor residency
     * and publication state.  Reading through the tensor's CPU mirror can hide
     * exactly that class of issue, so focused CUDA proofs fetch the bytes that
     * the next GPU stage would consume.
     */
    std::vector<float> copyCudaFP32TensorToHost(
        const std::shared_ptr<llaminar2::FP32Tensor> &tensor,
        cudaStream_t stream)
    {
        std::vector<float> host(tensor->numel());
        EXPECT_NE(tensor->gpu_data_ptr(), nullptr);
        EXPECT_EQ(cudaMemcpyAsync(host.data(),
                                  tensor->gpu_data_ptr(),
                                  host.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  stream),
                  cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        return host;
    }
#endif

    double cosineSimilarity(const float *a, const float *b, size_t count)
    {
        double dot = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (norm_a < 1.0e-30 && norm_b < 1.0e-30)
            return 1.0;
        if (norm_a < 1.0e-30 || norm_b < 1.0e-30)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    double relativeL2Error(const float *actual, const float *reference, size_t count)
    {
        double err_sq = 0.0;
        double ref_sq = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            const double diff = static_cast<double>(actual[i]) - static_cast<double>(reference[i]);
            err_sq += diff * diff;
            ref_sq += static_cast<double>(reference[i]) * static_cast<double>(reference[i]);
        }
        if (ref_sq < 1.0e-30)
            return err_sq < 1.0e-30 ? 0.0 : std::numeric_limits<double>::infinity();
        return std::sqrt(err_sq / ref_sq);
    }

    /**
     * @brief Euclidean norm for host-side verifier row witnesses.
     *
     * Grouped-vs-serial comparisons are only meaningful when the serial path did
     * real work.  The all-format MoE sweeps therefore assert nonzero serial and
     * grouped rows before accepting cosine/L2 agreement; this catches shared
     * descriptor or decode bugs where both paths accidentally publish zeroes.
     */
    double l2Norm(const float *values, size_t count)
    {
        double sum_sq = 0.0;
        for (size_t i = 0; i < count; ++i)
            sum_sq += static_cast<double>(values[i]) * static_cast<double>(values[i]);
        return std::sqrt(sum_sq);
    }

    double klDivergence(const float *actual, const float *reference, size_t count)
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
     * @brief Check that batched verifier top-k rows are numerically identical
     * enough to the row-wise decode router to be used as live MTP state.
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

        for (int row = 0; row < seq_len; ++row)
        {
            const float *actual = batched_weights.data() + static_cast<size_t>(row) * top_k;
            const float *reference = rowwise_weights.data() + static_cast<size_t>(row) * top_k;
            const double cosine = cosineSimilarity(actual, reference, static_cast<size_t>(top_k));
            const double rel_l2 = relativeL2Error(actual, reference, static_cast<size_t>(top_k));
            const double kl = klDivergence(actual, reference, static_cast<size_t>(top_k));
            for (int k = 0; k < top_k; ++k)
                ASSERT_NEAR(actual[k], reference[k], 1.0e-4f)
                    << "row=" << row << " slot=" << k;
            EXPECT_GT(cosine, 0.999999) << "row=" << row;
            EXPECT_LT(rel_l2, 1.0e-3) << "row=" << row;
            EXPECT_LT(kl, 1.0e-5) << "row=" << row;
        }
    }

    bool hasCudaDevice()
    {
#ifdef HAVE_CUDA
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
        return false;
#endif
    }

#ifdef HAVE_CUDA
    /// @brief RAII owner for tiny CUDA buffers used by descriptor-upload tests.
    class CudaAllocation
    {
    public:
        explicit CudaAllocation(size_t bytes)
        {
            EXPECT_EQ(cudaMalloc(&ptr_, bytes), cudaSuccess);
        }

        ~CudaAllocation()
        {
            if (ptr_)
                cudaFree(ptr_);
        }

        CudaAllocation(const CudaAllocation &) = delete;
        CudaAllocation &operator=(const CudaAllocation &) = delete;

        CudaAllocation(CudaAllocation &&other) noexcept
            : ptr_(other.ptr_)
        {
            other.ptr_ = nullptr;
        }

        CudaAllocation &operator=(CudaAllocation &&other) noexcept
        {
            if (this != &other)
            {
                if (ptr_)
                    cudaFree(ptr_);
                ptr_ = other.ptr_;
                other.ptr_ = nullptr;
            }
            return *this;
        }

        void *get() const { return ptr_; }

    private:
        void *ptr_ = nullptr;
    };

    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name)
        {
            const char *old = std::getenv(name);
            if (old)
            {
                had_old_ = true;
                old_value_ = old;
            }
            setenv(name, value, 1);
        }

        ~ScopedEnv()
        {
            if (had_old_)
                setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                unsetenv(name_.c_str());
        }

    private:
        std::string name_;
        bool had_old_ = false;
        std::string old_value_;
    };

    class ScopedCudaMoEGemmConfig
    {
    public:
        ScopedCudaMoEGemmConfig()
            : old_gateup_kpart_decode_(llaminar2::mutableDebugEnv().gemm.cuda_moe_gateup_kpart_decode),
              old_gateup_kparts_(llaminar2::mutableDebugEnv().gemm.cuda_moe_gateup_kparts),
              old_down_kpart_decode_(llaminar2::mutableDebugEnv().gemm.cuda_moe_down_kpart_decode),
              old_down_kparts_(llaminar2::mutableDebugEnv().gemm.cuda_moe_down_kparts)
        {
        }

        ~ScopedCudaMoEGemmConfig()
        {
            auto &gemm = llaminar2::mutableDebugEnv().gemm;
            gemm.cuda_moe_gateup_kpart_decode = old_gateup_kpart_decode_;
            gemm.cuda_moe_gateup_kparts = old_gateup_kparts_;
            gemm.cuda_moe_down_kpart_decode = old_down_kpart_decode_;
            gemm.cuda_moe_down_kparts = old_down_kparts_;
        }

        void set(bool gateup_kpart, int gateup_kparts, bool down_kpart, int down_kparts)
        {
            auto &gemm = llaminar2::mutableDebugEnv().gemm;
            gemm.cuda_moe_gateup_kpart_decode = gateup_kpart;
            gemm.cuda_moe_gateup_kparts = gateup_kparts;
            gemm.cuda_moe_down_kpart_decode = down_kpart;
            gemm.cuda_moe_down_kparts = down_kparts;
        }

    private:
        bool old_gateup_kpart_decode_ = true;
        int old_gateup_kparts_ = 16;
        bool old_down_kpart_decode_ = true;
        int old_down_kparts_ = 16;
    };

    class ScopedCudaMoEPrefillConfig
    {
    public:
        ScopedCudaMoEPrefillConfig()
            : old_tile_m_(llaminar2::mutableDebugEnv().gemm.cuda_moe_prefill_tile_m),
              old_fuse_swiglu_(llaminar2::mutableDebugEnv().gemm.cuda_moe_prefill_fuse_swiglu)
        {
        }

        ~ScopedCudaMoEPrefillConfig()
        {
            auto &gemm = llaminar2::mutableDebugEnv().gemm;
            gemm.cuda_moe_prefill_tile_m = old_tile_m_;
            gemm.cuda_moe_prefill_fuse_swiglu = old_fuse_swiglu_;
        }

        void set(int tile_m, bool fuse_swiglu)
        {
            auto &gemm = llaminar2::mutableDebugEnv().gemm;
            gemm.cuda_moe_prefill_tile_m = tile_m;
            gemm.cuda_moe_prefill_fuse_swiglu = fuse_swiglu;
        }

    private:
        int old_tile_m_ = 0;
        bool old_fuse_swiglu_ = true;
    };

    void expectGroupedDecodeCounter(
        const char *counter_name,
        const char *source,
        int active_slots,
        int d_model,
        int intermediate,
        const char *expected_route = nullptr)
    {
        const auto records =
            llaminar2::PerfStatsCollector::snapshot({std::string("kernel.") + counter_name});
        ASSERT_FALSE(records.empty()) << "missing perf counter " << counter_name;

        const auto expected_active_slots = std::to_string(active_slots);
        const auto expected_d_model = std::to_string(d_model);
        const auto expected_intermediate = std::to_string(intermediate);
        const auto match = std::find_if(
            records.begin(),
            records.end(),
            [&](const llaminar2::PerfStatRecord &record)
            {
                auto tag_equals = [&](const char *key, const std::string &value)
                {
                    const auto it = record.tags.find(key);
                    return it != record.tags.end() && it->second == value;
                };
                const auto route = record.tags.find("route");
                const bool route_matches =
                    route != record.tags.end() &&
                    (expected_route
                         ? route->second == expected_route
                         : (route->second == "kpart" || route->second == "serial" ||
                            route->second == "fused_kpart" || route->second == "fused_block_down"));
                return record.name == counter_name &&
                       tag_equals("source", source) &&
                       tag_equals("active_slots", expected_active_slots) &&
                       tag_equals("d_model", expected_d_model) &&
                       tag_equals("intermediate", expected_intermediate) &&
                       route_matches;
            });
        ASSERT_NE(match, records.end()) << "missing matching perf counter " << counter_name
                                        << " source=" << source;
        EXPECT_GE(match->count, 1u);
        EXPECT_GE(match->value, 1.0);
    }

    void expectFusedDecodeSubkernelTimer(
        const char *timer_name,
        int top_k,
        int d_model,
        int intermediate)
    {
        const auto records =
            llaminar2::PerfStatsCollector::snapshot({std::string("kernel_cuda.") + timer_name});
        ASSERT_FALSE(records.empty()) << "missing perf timer " << timer_name;

        const auto expected_top_k = std::to_string(top_k);
        const auto expected_d_model = std::to_string(d_model);
        const auto expected_intermediate = std::to_string(intermediate);
        const auto match = std::find_if(
            records.begin(),
            records.end(),
            [&](const llaminar2::PerfStatRecord &record)
            {
                auto tag_equals = [&](const char *key, const std::string &value)
                {
                    const auto it = record.tags.find(key);
                    return it != record.tags.end() && it->second == value;
                };
                return record.name == timer_name &&
                       record.domain == "kernel_cuda" &&
                       tag_equals("source", "fused_runtime") &&
                       tag_equals("stage_type", "MOE_EXPERT_FFN") &&
                       tag_equals("top_k", expected_top_k) &&
                       tag_equals("d_model", expected_d_model) &&
                       tag_equals("intermediate", expected_intermediate);
            });
        ASSERT_NE(match, records.end()) << "missing matching perf timer " << timer_name
                                        << "\n"
                                        << llaminar2::PerfStatsCollector::summaryString(
                                               {std::string("kernel_cuda.") + timer_name});
        EXPECT_EQ(match->count, 1u)
            << "captured graph replay must not add eager sub-kernel timing events";
        EXPECT_GT(match->total_ns, 0u);
    }

    void expectPrefillSwiGLUPathRecord(
        const char *swiglu_path,
        int seq_len,
        int top_k,
        int num_experts,
        int expected_tile_m,
        const char *expected_gateup_route = nullptr,
        const char *expected_down_route = nullptr,
        const char *expected_down_accumulation = nullptr,
        int expected_active_expert_slots = -1)
    {
        const auto records =
            llaminar2::PerfStatsCollector::snapshot({"kernel.cuda_moe_grouped_prefill_swiglu_path_calls"});
        const std::string expected_path = swiglu_path;
        const int total_slots = seq_len * top_k;
        const int active_slots = expected_active_expert_slots >= 0
                                     ? expected_active_expert_slots
                                     : std::min(total_slots, num_experts);
        const std::string expected_total_slots = std::to_string(total_slots);
        const std::string expected_active_slots = std::to_string(active_slots);
        const std::string expected_num_experts = std::to_string(num_experts);
        const std::string expected_tile = std::to_string(expected_tile_m);
        const std::string expected_tile_n =
            std::to_string((active_slots > 0 && seq_len <= 4) ? 64 : 128);
        const auto match = std::find_if(
            records.begin(),
            records.end(),
            [&](const llaminar2::PerfStatRecord &record)
            {
                auto tag_equals = [&](const char *key, const std::string &value)
                {
                    const auto it = record.tags.find(key);
                    return it != record.tags.end() && it->second == value;
                };
                return record.name == "cuda_moe_grouped_prefill_swiglu_path_calls" &&
                       tag_equals("swiglu_path", expected_path) &&
                       tag_equals("total_slots", expected_total_slots) &&
                       tag_equals("activation_quant_rows", std::to_string(seq_len)) &&
                       tag_equals("active_expert_slots", expected_active_slots) &&
                       tag_equals("num_experts", expected_num_experts) &&
                       tag_equals("tile_m", expected_tile) &&
                       tag_equals("tile_n", expected_tile_n) &&
                       (!expected_gateup_route ||
                        tag_equals("gateup_route", std::string(expected_gateup_route))) &&
                       (!expected_down_route ||
                        tag_equals("down_route", std::string(expected_down_route))) &&
                       (!expected_down_accumulation ||
                        tag_equals("down_accumulation", std::string(expected_down_accumulation)));
            });
        ASSERT_NE(match, records.end()) << "missing grouped prefill SwiGLU path counter path="
                                        << swiglu_path << " seq_len=" << seq_len
                                        << " tile_m=" << expected_tile_m << "\n"
                                        << llaminar2::PerfStatsCollector::jsonString(
                                               {"kernel.cuda_moe_grouped_prefill_swiglu_path_calls"});
        EXPECT_GE(match->count, 1u);
        EXPECT_GE(match->value, 1.0);
    }

    /**
     * @brief KL(reference || actual) after a stable row-wise softmax.
     *
     * The combined verifier tests compare hidden rows rather than final logits,
     * but the softmax KL still catches rank/shape drift in the largest row
     * coordinates.  That makes the focused kernel test much closer to the MTP
     * publication contract, where a small row-local error can flip a token.
     */
    double rowSoftmaxKLDivergence(const float *actual, const float *expected, size_t row_width)
    {
        double max_actual = -std::numeric_limits<double>::infinity();
        double max_expected = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < row_width; ++i)
        {
            max_actual = std::max(max_actual, static_cast<double>(actual[i]));
            max_expected = std::max(max_expected, static_cast<double>(expected[i]));
        }

        double sum_actual = 0.0;
        double sum_expected = 0.0;
        for (size_t i = 0; i < row_width; ++i)
        {
            sum_actual += std::exp(static_cast<double>(actual[i]) - max_actual);
            sum_expected += std::exp(static_cast<double>(expected[i]) - max_expected);
        }

        constexpr double kEps = 1.0e-30;
        double kl = 0.0;
        for (size_t i = 0; i < row_width; ++i)
        {
            const double p = std::exp(static_cast<double>(expected[i]) - max_expected) /
                             std::max(sum_expected, kEps);
            const double q = std::exp(static_cast<double>(actual[i]) - max_actual) /
                             std::max(sum_actual, kEps);
            kl += p * (std::log(std::max(p, kEps)) - std::log(std::max(q, kEps)));
        }
        return kl;
    }

    void expectVectorsClose(const std::vector<float> &actual,
                            const std::vector<float> &expected,
                            double min_cosine,
                            double max_relative_l2,
                            size_t row_width = 0,
                            double min_row_cosine = 0.0,
                            double max_row_relative_l2 = std::numeric_limits<double>::infinity(),
                            double max_row_kl = std::numeric_limits<double>::infinity())
    {
        ASSERT_EQ(actual.size(), expected.size());
        double dot = 0.0;
        double norm_actual = 0.0;
        double norm_expected = 0.0;
        double diff2 = 0.0;
        double max_abs = 0.0;
        size_t max_abs_index = 0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            ASSERT_TRUE(std::isfinite(actual[i])) << "actual element " << i;
            ASSERT_TRUE(std::isfinite(expected[i])) << "expected element " << i;
            const double a = actual[i];
            const double e = expected[i];
            const double diff = a - e;
            dot += a * e;
            norm_actual += a * a;
            norm_expected += e * e;
            diff2 += diff * diff;
            if (std::abs(diff) > max_abs)
            {
                max_abs = std::abs(diff);
                max_abs_index = i;
            }
        }

        const double cosine = (norm_actual < 1.0e-30 && norm_expected < 1.0e-30)
                                  ? 1.0
                                  : dot / (std::sqrt(norm_actual) * std::sqrt(norm_expected) + 1.0e-30);
        const double relative_l2 = (norm_expected < 1.0e-30)
                                       ? ((diff2 < 1.0e-30)
                                              ? 0.0
                                              : std::numeric_limits<double>::infinity())
                                       : std::sqrt(diff2) / std::sqrt(norm_expected);
        double min_observed_row_cosine = 1.0;
        double max_observed_row_relative_l2 = 0.0;
        double max_observed_row_kl = 0.0;
        size_t worst_row = 0;
        if (row_width != 0 && actual.size() % row_width == 0)
        {
            for (size_t row = 0; row < actual.size() / row_width; ++row)
            {
                const float *row_actual = actual.data() + row * row_width;
                const float *row_expected = expected.data() + row * row_width;
                double row_dot = 0.0;
                double row_norm_actual = 0.0;
                double row_norm_expected = 0.0;
                double row_diff2 = 0.0;
                for (size_t i = 0; i < row_width; ++i)
                {
                    const double a = row_actual[i];
                    const double e = row_expected[i];
                    const double diff = a - e;
                    row_dot += a * e;
                    row_norm_actual += a * a;
                    row_norm_expected += e * e;
                    row_diff2 += diff * diff;
                }
                const double row_cosine =
                    (row_norm_actual < 1.0e-30 && row_norm_expected < 1.0e-30)
                        ? 1.0
                        : row_dot / (std::sqrt(row_norm_actual) * std::sqrt(row_norm_expected) + 1.0e-30);
                const double row_relative_l2 =
                    (row_norm_expected < 1.0e-30)
                        ? ((row_diff2 < 1.0e-30)
                               ? 0.0
                               : std::numeric_limits<double>::infinity())
                        : std::sqrt(row_diff2) / std::sqrt(row_norm_expected);
                const double row_kl = rowSoftmaxKLDivergence(row_actual, row_expected, row_width);
                if (row_relative_l2 > max_observed_row_relative_l2 ||
                    row_kl > max_observed_row_kl ||
                    row_cosine < min_observed_row_cosine)
                {
                    worst_row = row;
                }
                min_observed_row_cosine = std::min(min_observed_row_cosine, row_cosine);
                max_observed_row_relative_l2 =
                    std::max(max_observed_row_relative_l2, row_relative_l2);
                max_observed_row_kl = std::max(max_observed_row_kl, row_kl);
            }
        }
        EXPECT_GE(cosine, min_cosine)
            << "max_abs=" << max_abs << " at index " << max_abs_index
            << " min_row_cosine=" << min_observed_row_cosine
            << " max_row_relative_l2=" << max_observed_row_relative_l2
            << " max_row_kl=" << max_observed_row_kl
            << " worst_row=" << worst_row;
        EXPECT_LE(relative_l2, max_relative_l2)
            << "max_abs=" << max_abs << " at index " << max_abs_index
            << " cosine=" << cosine
            << " min_row_cosine=" << min_observed_row_cosine
            << " max_row_relative_l2=" << max_observed_row_relative_l2
            << " max_row_kl=" << max_observed_row_kl
            << " worst_row=" << worst_row;
        if (row_width != 0)
        {
            EXPECT_GE(min_observed_row_cosine, min_row_cosine)
                << "cosine=" << cosine << " relative_l2=" << relative_l2
                << " max_row_relative_l2=" << max_observed_row_relative_l2
                << " max_row_kl=" << max_observed_row_kl
                << " worst_row=" << worst_row;
            EXPECT_LE(max_observed_row_relative_l2, max_row_relative_l2)
                << "cosine=" << cosine << " relative_l2=" << relative_l2
                << " min_row_cosine=" << min_observed_row_cosine
                << " max_row_kl=" << max_observed_row_kl
                << " worst_row=" << worst_row;
            EXPECT_LE(max_observed_row_kl, max_row_kl)
                << "cosine=" << cosine << " relative_l2=" << relative_l2
                << " min_row_cosine=" << min_observed_row_cosine
                << " max_row_relative_l2=" << max_observed_row_relative_l2
                << " worst_row=" << worst_row;
        }
    }

    /// @brief Build a minimal native-VNNI descriptor backed by CUDA device pointers.
    llaminar2::DeviceNativeVNNIMatrixDesc makeCudaNativeDesc(
        const CudaAllocation &payload,
        const CudaAllocation &scales,
        int rows,
        int cols)
    {
        llaminar2::DeviceNativeVNNIMatrixDesc desc{};
        desc.payload = static_cast<const uint8_t *>(payload.get());
        desc.scales = scales.get();
        desc.n = rows;
        desc.k = cols;
        desc.blocks_per_row = static_cast<uint32_t>(cols / 32);
        desc.codebook_id = 0;
        return desc;
    }
#endif

    using CUDAMoEWeightCreator = std::function<std::unique_ptr<llaminar2::TensorBase>(
        const std::vector<size_t> &shape,
        uint32_t seed)>;

    /**
     * @brief One tensor-format entry for CUDA grouped MoE verifier coverage.
     *
     * The grouped CUDA MoE kernels dispatch by native-VNNI codebook id, but the
     * descriptor metadata still differs across source tensor formats that share a
     * codebook.  This table is intentionally format-complete for the NativeVNNI
     * tensors that can reach grouped verifier prefill.
     */
    struct CUDAMoEFormatCase
    {
        const char *label;
        uint8_t codebook_id;
        CUDAMoEWeightCreator create;
    };

    /**
     * @brief Create bounded IQ4_XS weights for MoE grouped verifier tests.
     *
     * IQ4_XS encodes sub-block scale as `d * (ls - 32)`.  Keeping `ls` close to 32
     * gives stable, nonzero verifier rows while still exercising codebook 4 and
     * the IQ4_XS descriptor layout.
     */
    std::unique_ptr<llaminar2::TensorBase> createBoundedCudaMoEIQ4XS(
        const std::vector<size_t> &shape,
        uint32_t seed)
    {
        constexpr size_t block_size = llaminar2::IQ4_XSBlock::BLOCK_SIZE;
        const size_t rows = shape.at(0);
        const size_t cols = shape.at(1);
        const size_t blocks_per_row = (cols + block_size - 1) / block_size;
        const size_t total_blocks = rows * blocks_per_row;

        std::vector<uint8_t> raw_data(total_blocks * sizeof(llaminar2::IQ4_XSBlock));
        auto *blocks = reinterpret_cast<llaminar2::IQ4_XSBlock *>(raw_data.data());
        for (size_t block_idx = 0; block_idx < total_blocks; ++block_idx)
        {
            auto &block = blocks[block_idx];
            block.d = 0x2400; // FP16 0.015625.
            block.scales_h = 0;
            std::memset(block.scales_l, 0, sizeof(block.scales_l));

            for (int sub = 0; sub < 8; ++sub)
            {
                const uint16_t ls = static_cast<uint16_t>(
                    33u + ((seed + block_idx + static_cast<size_t>(sub)) & 0x3u));
                block.scales_l[sub / 2] |= static_cast<uint8_t>(
                    (ls & 0x0fu) << (4 * (sub & 1)));
                block.scales_h |= static_cast<uint16_t>(
                    ((ls >> 4) & 0x3u) << (2 * sub));
            }

            for (size_t j = 0; j < std::size(block.qs); ++j)
            {
                const uint8_t lo = static_cast<uint8_t>(
                    (seed + block_idx * 19u + j * 5u) & 0x0fu);
                const uint8_t hi = static_cast<uint8_t>(
                    ((seed >> 4) + block_idx * 23u + j * 7u) & 0x0fu);
                block.qs[j] = static_cast<uint8_t>(lo | (hi << 4));
            }
        }

        return std::make_unique<llaminar2::IQ4_XSTensor>(shape, raw_data);
    }

    /**
     * @brief Create deterministic IQ3_S weights with nonzero signs/high bits.
     *
     * The generic IQ3_S random factory is adequate for many smoke tests, but the
     * all-format grouped MoE sweep needs a stronger witness so serial and grouped
     * paths cannot both pass by producing zero rows.
     */
    std::unique_ptr<llaminar2::TensorBase> createNonzeroCudaMoEIQ3S(
        const std::vector<size_t> &shape,
        uint32_t seed)
    {
        constexpr size_t block_size = llaminar2::IQ3_SBlock::BLOCK_SIZE;
        const size_t rows = shape.at(0);
        const size_t cols = shape.at(1);
        const size_t blocks_per_row = (cols + block_size - 1) / block_size;
        const size_t total_blocks = rows * blocks_per_row;

        std::vector<uint8_t> raw_data(total_blocks * sizeof(llaminar2::IQ3_SBlock));
        auto *blocks = reinterpret_cast<llaminar2::IQ3_SBlock *>(raw_data.data());
        for (size_t block_idx = 0; block_idx < total_blocks; ++block_idx)
        {
            auto &block = blocks[block_idx];
            block.d = 0x3000; // FP16 0.125.

            for (size_t j = 0; j < std::size(block.qs); ++j)
            {
                block.qs[j] = static_cast<uint8_t>(
                    (seed + block_idx * 37u + j * 13u) & 0xffu);
            }
            for (size_t j = 0; j < std::size(block.qh); ++j)
            {
                block.qh[j] = static_cast<uint8_t>(
                    ((seed >> 3) + block_idx * 11u + j * 29u) & 0xffu);
            }
            for (size_t j = 0; j < std::size(block.signs); ++j)
            {
                block.signs[j] = static_cast<uint8_t>(
                    (0x5au ^ seed ^ (block_idx * 17u + j * 7u)) & 0xffu);
            }
            for (size_t j = 0; j < std::size(block.scales); ++j)
            {
                const uint8_t lo = static_cast<uint8_t>((1u + seed + block_idx + j) & 0x3u);
                const uint8_t hi = static_cast<uint8_t>((2u + (seed >> 2) + block_idx + j) & 0x3u);
                block.scales[j] = static_cast<uint8_t>(lo | (hi << 4));
            }
        }

        return std::make_unique<llaminar2::IQ3_STensor>(shape, raw_data);
    }

    std::vector<CUDAMoEFormatCase> cudaMoEGroupedNativeFormats()
    {
        return {
            {"Q4_0", 0, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createQ4_0Random(shape, seed); }},
            {"IQ4_NL", 4, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createIQ4_NLRandom(shape, seed); }},
            {"Q4_1", 5, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createQ4_1Random(shape, seed); }},
            {"IQ4_XS", 4, [](const std::vector<size_t> &shape, uint32_t seed)
             { return createBoundedCudaMoEIQ4XS(shape, seed); }},
            {"Q5_0", 6, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createQ5_0Random(shape, seed); }},
            {"Q5_1", 7, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createQ5_1Random(shape, seed); }},
            {"Q4_K", 5, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createQ4_KRandom(shape, seed); }},
            {"Q5_K", 7, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createQ5_KRandom(shape, seed); }},
            {"Q6_K", 8, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createQ6_KRandom(shape, seed); }},
            {"Q3_K", 9, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createQ3_KRandom(shape, seed); }},
            {"Q2_K", 10, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createQ2_KRandom(shape, seed); }},
            {"IQ3_S", 11, [](const std::vector<size_t> &shape, uint32_t seed)
             { return createNonzeroCudaMoEIQ3S(shape, seed); }},
            {"IQ3_XXS", 12, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createIQ3_XXSRandom(shape, seed); }},
            {"IQ2_S", 13, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createIQ2_SRandom(shape, seed); }},
            {"IQ2_XS", 14, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createIQ2_XSRandom(shape, seed); }},
            {"IQ2_XXS", 15, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createIQ2_XXSRandom(shape, seed); }},
            {"IQ1_S", 16, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createIQ1_SRandom(shape, seed); }},
            {"IQ1_M", 17, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createIQ1_MRandom(shape, seed); }},
            {"Q8_0", 19, [](const std::vector<size_t> &shape, uint32_t seed)
             { return llaminar2::test::TestTensorFactory::createQ8_0Random(shape, seed); }},
        };
    }

    class Test__CUDAMoEKernel : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
#ifndef HAVE_CUDA
            GTEST_SKIP() << "CUDA support not compiled";
#else
            if (!hasCudaDevice())
                GTEST_SKIP() << "No CUDA device available";
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            auto &gemm = llaminar2::mutableDebugEnv().gemm;
            old_cuda_moe_router_q8_ = gemm.cuda_moe_router_q8;
            old_cuda_moe_reuse_router_q8_hidden_ = gemm.cuda_moe_reuse_router_q8_hidden;
            gemm.cuda_moe_router_q8 = false;
            gemm.cuda_moe_reuse_router_q8_hidden = false;
            ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
            cuda_kernel_ = KernelFactory::getOrCreateMoEKernel(llaminar2::DeviceId::cuda(0));
            cpu_kernel_ = KernelFactory::getOrCreateMoEKernel(llaminar2::DeviceId::cpu());
            ASSERT_NE(cuda_kernel_, nullptr);
            ASSERT_NE(cpu_kernel_, nullptr);
            cuda_kernel_->setGPUStream(stream_);
            auto *workspace_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(cuda_kernel_);
            ASSERT_NE(workspace_consumer, nullptr);
            auto reqs = llaminar2::MoEWorkspaceBuffers::cudaMoE(
                /*max_seq_len=*/64,
                /*d_model=*/2048,
                /*intermediate=*/512,
                /*num_experts=*/256,
                /*top_k=*/16);
            reqs.merge(llaminar2::MoEWorkspaceBuffers::cudaMoE(
                /*max_seq_len=*/4,
                /*d_model=*/2048,
                /*intermediate=*/512,
                /*num_experts=*/256,
                /*top_k=*/16));
            reqs.merge(llaminar2::MoEWorkspaceBuffers::cudaMoE(
                /*max_seq_len=*/4,
                /*d_model=*/2048,
                /*intermediate=*/512,
                /*num_experts=*/257,
                /*top_k=*/16));
            reqs.merge(llaminar2::MoEWorkspaceBuffers::cudaMoE(
                /*max_seq_len=*/1536,
                /*d_model=*/2048,
                /*intermediate=*/512,
                /*num_experts=*/256,
                /*top_k=*/8));
            workspace_ = std::make_unique<llaminar2::DeviceWorkspaceManager>(
                llaminar2::DeviceId::cuda(0),
                reqs.total_bytes_with_alignment() + 4 * 1024 * 1024);
            ASSERT_TRUE(workspace_->allocate(reqs));
            workspace_consumer->bindWorkspace(workspace_.get());
#endif
        }

        void TearDown() override
        {
#ifdef HAVE_CUDA
            if (cuda_kernel_)
            {
                if (auto *workspace_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(cuda_kernel_))
                    workspace_consumer->unbindWorkspace();
            }
            workspace_.reset();
            if (stream_)
                cudaStreamDestroy(stream_);
            auto &gemm = llaminar2::mutableDebugEnv().gemm;
            gemm.cuda_moe_router_q8 = old_cuda_moe_router_q8_;
            gemm.cuda_moe_reuse_router_q8_hidden = old_cuda_moe_reuse_router_q8_hidden_;
#endif
        }

#ifdef HAVE_CUDA
        cudaStream_t stream_ = nullptr;
        std::unique_ptr<llaminar2::DeviceWorkspaceManager> workspace_;
        bool old_cuda_moe_router_q8_ = true;
        bool old_cuda_moe_reuse_router_q8_hidden_ = true;
#endif
        llaminar2::IMoEKernel *cuda_kernel_ = nullptr;
        llaminar2::IMoEKernel *cpu_kernel_ = nullptr;
    };
}

TEST_F(Test__CUDAMoEKernel, RouteLogitsHandlesMisalignedFP32Rows)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int seq_len = 2;
    constexpr int d_model = 12;
    constexpr int num_experts = 5;

    std::vector<float> hidden(static_cast<size_t>(seq_len) * d_model);
    std::vector<float> gate(static_cast<size_t>(num_experts) * d_model);
    for (int i = 0; i < seq_len * d_model; ++i)
        hidden[static_cast<size_t>(i)] = 0.05f * static_cast<float>((i % 7) - 3);
    for (int i = 0; i < num_experts * d_model; ++i)
        gate[static_cast<size_t>(i)] = 0.03f * static_cast<float>((i % 11) - 5);

    CudaAllocation hidden_alloc((hidden.size() + 1) * sizeof(float));
    CudaAllocation gate_alloc((gate.size() + 1) * sizeof(float));
    CudaAllocation logits_alloc(hidden.size() * sizeof(float));

    auto *d_hidden = static_cast<float *>(hidden_alloc.get()) + 1;
    auto *d_gate = static_cast<float *>(gate_alloc.get()) + 1;
    auto *d_logits = static_cast<float *>(logits_alloc.get());
    ASSERT_NE(reinterpret_cast<std::uintptr_t>(d_hidden) & 0x0fu, 0u);
    ASSERT_NE(reinterpret_cast<std::uintptr_t>(d_gate) & 0x0fu, 0u);

    ASSERT_EQ(cudaMemcpyAsync(d_hidden, hidden.data(), hidden.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gate, gate.data(), gate.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cudaMoE_route_logits(
        d_hidden, d_gate, d_logits, seq_len, d_model, num_experts, 0, stream_));

    std::vector<float> actual(static_cast<size_t>(seq_len) * num_experts);
    ASSERT_EQ(cudaMemcpyAsync(actual.data(), d_logits, actual.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    std::vector<float> expected(actual.size(), 0.0f);
    for (int token = 0; token < seq_len; ++token)
    {
        for (int expert = 0; expert < num_experts; ++expert)
        {
            float sum = 0.0f;
            for (int dim = 0; dim < d_model; ++dim)
            {
                sum += hidden[static_cast<size_t>(token) * d_model + dim] *
                       gate[static_cast<size_t>(expert) * d_model + dim];
            }
            expected[static_cast<size_t>(token) * num_experts + expert] = sum;
        }
    }

    expectNearArray(actual.data(), expected.data(), actual.size(), 1.0e-6f);
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimePrefillRegroupFiltersRoutesByAssignedParticipant)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    const auto device = llaminar2::DeviceId::cuda(0);
    constexpr int seq_len = 3;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int total_slots = seq_len * top_k;

    llaminar2::DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    llaminar2::MoERuntimeTable runtime_table(runtime_config);

    auto runtime_state = runtime_table.hostLayerState(0);
    runtime_state.participant_id = 1;
    runtime_state.participant_count = 2;
    ASSERT_EQ(cudaMemcpyAsync(runtime_table.deviceLayerState(0),
                              &runtime_state,
                              sizeof(runtime_state),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    const float routing_indices_data[total_slots] = {
        0.0f, 1.0f,
        1.0f, 2.0f,
        2.0f, 3.0f};
    const float routing_weights_data[total_slots] = {
        0.50f, 0.10f,
        0.70f, 0.30f,
        0.20f, 0.80f};
    auto routing_indices = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    auto routing_weights = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots, routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots, routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(cuda_kernel_->groupPrefillRoutes(
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
    ASSERT_EQ(cudaMemcpyAsync(runtime_state.route_participant_ids,
                              participant_assignments,
                              sizeof(participant_assignments),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_TRUE(cuda_kernel_->regroupPrefillRoutesFromRuntimeAssignments(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    std::array<int32_t, num_experts> counts{};
    std::array<int32_t, num_experts> offsets{};
    std::array<int32_t, total_slots> grouped_tokens{};
    std::array<float, total_slots> grouped_weights{};
    ASSERT_EQ(cudaMemcpy(counts.data(), runtime_state.expert_counts,
                         sizeof(counts), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(offsets.data(), runtime_state.expert_offsets,
                         sizeof(offsets), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(grouped_tokens.data(), runtime_state.grouped_token_ids,
                         sizeof(grouped_tokens), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(grouped_weights.data(), runtime_state.grouped_route_weights,
                         sizeof(grouped_weights), cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_EQ(counts, (std::array<int32_t, num_experts>{1, 1, 1, 1}));
    EXPECT_EQ(offsets, (std::array<int32_t, num_experts>{0, 1, 2, 3}));
    EXPECT_EQ(grouped_tokens[0], 0);
    EXPECT_EQ(grouped_tokens[1], 2);
    EXPECT_EQ(grouped_tokens[2], 3);
    EXPECT_EQ(grouped_tokens[3], 5);
    EXPECT_EQ(grouped_tokens[4], 0);
    EXPECT_EQ(grouped_tokens[5], 0);
    EXPECT_NEAR(grouped_weights[0], 0.50f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[1], 0.70f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[2], 0.30f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[3], 0.80f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[4], 0.0f, 1.0e-6f);
    EXPECT_NEAR(grouped_weights[5], 0.0f, 1.0e-6f);

    CudaAllocation original_to_grouped(sizeof(int) * total_slots);
    auto *d_original_to_grouped = static_cast<int *>(original_to_grouped.get());
    ASSERT_TRUE(cudaMoE_build_runtime_original_to_grouped(
        runtime_table.deviceLayerState(0),
        d_original_to_grouped,
        total_slots,
        total_slots,
        num_experts,
        top_k,
        0,
        stream_));
    std::array<int32_t, total_slots> ordered_map{};
    ASSERT_EQ(cudaMemcpyAsync(ordered_map.data(),
                              d_original_to_grouped,
                              sizeof(ordered_map),
                              cudaMemcpyDeviceToHost,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    EXPECT_EQ(ordered_map, (std::array<int32_t, total_slots>{0, -1, 1, 2, -1, 3}));
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimePrefillLeastLoadedResidentAssignmentBalancesHotReplicas)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    const auto device = llaminar2::DeviceId::cuda(0);
    constexpr int seq_len = 3;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int total_slots = seq_len * top_k;

    llaminar2::DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    llaminar2::MoERuntimeTable runtime_table(runtime_config);

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
            llaminar2::toMoEExpertFlags(llaminar2::DeviceMoEExpertFlags::Valid |
                                        llaminar2::DeviceMoEExpertFlags::Resident);
    }
    bank.resident_participant_mask[0] = 0b11u;
    bank.resident_participant_mask[1] = 0b10u;
    bank.resident_participant_mask[2] = 0b11u;
    bank.resident_participant_mask[3] = 0b01u;

    ASSERT_EQ(cudaMemcpyAsync(runtime_table.deviceLayerState(0),
                              &runtime_state,
                              sizeof(runtime_state),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    const float routing_indices_data[total_slots] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        2.0f, 2.0f};
    const float routing_weights_data[total_slots] = {
        0.50f, 0.25f,
        0.75f, 0.10f,
        0.60f, 0.30f};
    auto routing_indices = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    auto routing_weights = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots, routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots, routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(cuda_kernel_->groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_TRUE(cuda_kernel_->assignPrefillRoutesLeastLoadedResident(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_TRUE(cuda_kernel_->regroupPrefillRoutesFromRuntimeAssignments(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    std::array<int32_t, total_slots> route_participants{};
    std::array<int32_t, num_experts> counts{};
    std::array<int32_t, num_experts> offsets{};
    std::array<int32_t, total_slots> grouped_tokens{};
    std::array<float, total_slots> grouped_weights{};
    ASSERT_EQ(cudaMemcpy(route_participants.data(), runtime_state.route_participant_ids,
                         sizeof(route_participants), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(counts.data(), runtime_state.expert_counts,
                         sizeof(counts), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(offsets.data(), runtime_state.expert_offsets,
                         sizeof(offsets), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(grouped_tokens.data(), runtime_state.grouped_token_ids,
                         sizeof(grouped_tokens), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(grouped_weights.data(), runtime_state.grouped_route_weights,
                         sizeof(grouped_weights), cudaMemcpyDeviceToHost),
              cudaSuccess);

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
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimePrefillLeastLoadedCurrentBatchPlannerMaterializesSpans)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    const auto device = llaminar2::DeviceId::cuda(0);
    constexpr int seq_len = 8;
    constexpr int num_experts = 4;
    constexpr int top_k = 1;
    constexpr int total_slots = seq_len * top_k;

    llaminar2::DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    llaminar2::MoERuntimeTable runtime_table(runtime_config);

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
    ASSERT_EQ(cudaMemcpyAsync(runtime_table.deviceLayerState(0),
                              &runtime_state,
                              sizeof(runtime_state),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    const float routing_indices_data[total_slots] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 2.0f};
    const float routing_weights_data[total_slots] = {
        1.0f, 0.9f, 0.8f, 0.7f,
        0.6f, 0.5f, 0.4f, 0.3f};
    auto routing_indices = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    auto routing_weights = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots, routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots, routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(cuda_kernel_->groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentConfig config;
    config.expert_count = num_experts;
    config.participant_count = 2;
    config.enable_balanced_skip = false;
    ASSERT_TRUE(cuda_kernel_->planPrefillRoutesLeastLoadedCurrentBatch(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        config));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoELayerRuntime device_runtime_state{};
    ASSERT_EQ(cudaMemcpy(&device_runtime_state,
                         runtime_table.deviceLayerState(0),
                         sizeof(device_runtime_state),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(device_runtime_state.reserved_u64[2], 4u);
    EXPECT_EQ(device_runtime_state.reserved_u64[3], 1u);

    std::array<llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentSpan, 4> spans{};
    std::array<llaminar2::least_loaded_ep::LeastLoadedExpertWeightTransfer, 1> transfers{};
    std::array<int32_t, total_slots> route_participants{};
    ASSERT_EQ(cudaMemcpy(spans.data(), runtime_state.reserved_ptrs[1],
                         sizeof(spans), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(transfers.data(), runtime_state.reserved_ptrs[2],
                         sizeof(transfers), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(route_participants.data(), runtime_state.route_participant_ids,
                         sizeof(route_participants), cudaMemcpyDeviceToHost),
              cudaSuccess);

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

    llaminar2::DeviceMoERebalanceConfig rebalance_config;
    rebalance_config.num_layers = 1;
    rebalance_config.num_experts = num_experts;
    rebalance_config.top_k = top_k;
    rebalance_config.participant_id = 0;
    rebalance_config.participant_count = 2;
    rebalance_config.root_participant = 0;
    rebalance_config.window_size_tokens = 256;

    llaminar2::DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_header = nullptr;
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_apply_status = nullptr;
    ASSERT_EQ(cudaMalloc(&d_plan, 4 * sizeof(llaminar2::DeviceMoERebalancePlanEntry)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_plan_count, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_header, sizeof(llaminar2::DeviceMoERebalanceCommandBufferHeader)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_status, sizeof(llaminar2::DeviceMoERebalanceStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_apply_status, sizeof(llaminar2::DeviceMoERebalanceApplyStatus)),
              cudaSuccess);
    ASSERT_TRUE(cuda_kernel_->materializePrefillLeastLoadedTransferCommands(
        runtime_table.deviceLayerState(0),
        d_plan,
        d_plan_count,
        4,
        d_header,
        d_status,
        rebalance_config,
        2,
        0));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    uint32_t materialized_plan_count = 0;
    std::array<llaminar2::DeviceMoERebalancePlanEntry, 4> materialized_plan{};
    llaminar2::DeviceMoERebalanceCommandBufferHeader materialized_header{};
    llaminar2::DeviceMoERebalanceStatus materialized_status{};
    ASSERT_EQ(cudaMemcpy(&materialized_plan_count, d_plan_count, sizeof(uint32_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(materialized_plan.data(), d_plan,
                         sizeof(materialized_plan), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&materialized_header, d_header, sizeof(materialized_header),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&materialized_status, d_status, sizeof(materialized_status),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(materialized_plan_count, 1u);
    EXPECT_EQ(materialized_header.magic, llaminar2::kDeviceMoERebalanceMagic);
    EXPECT_EQ(materialized_header.version, llaminar2::kDeviceMoERebalanceVersion);
    EXPECT_EQ(materialized_header.command_count, 1u);
    EXPECT_EQ(materialized_header.command_capacity, 4u);
    EXPECT_EQ(materialized_header.participant_id, 0u);
    EXPECT_EQ(materialized_header.participant_count, 2u);
    EXPECT_EQ(materialized_plan[0].op,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ExpertPayloadArrival));
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
              llaminar2::moe_rebalance_policy::directedParticipantEdgeBit(
                  0u, 1u, llaminar2::kDeviceMoEMaxParticipants));
    EXPECT_EQ(materialized_status.llep_assignment_span_count, 4u);
    EXPECT_EQ(materialized_status.llep_weight_transfer_count, 1u);
    llaminar2::DeviceMoERebalanceApplyStatus apply_status;
    apply_status.plan_entries_seen = materialized_status.planned_arrivals;
    apply_status.applied_arrivals = 0u;
    apply_status.changed_layers = materialized_status.planned_arrivals > 0u ? 1u : 0u;
    ASSERT_EQ(cudaMemcpy(d_apply_status, &apply_status, sizeof(apply_status),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(route_participants.data(), runtime_state.route_participant_ids,
                         sizeof(route_participants), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_TRUE(std::all_of(route_participants.begin(), route_participants.end(),
                            [](int32_t participant) { return participant == 0; }))
        << "guarded current-batch LLEP apply must not rewrite assignments while weight transfers are pending";

    ASSERT_EQ(cudaMemcpy(&device_runtime_state,
                         runtime_table.deviceLayerState(0),
                         sizeof(device_runtime_state),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    device_runtime_state
        .banks[device_runtime_state.active_bank]
        .resident_participant_mask[0] |= 0b10u;
    ASSERT_EQ(cudaMemcpy(runtime_table.deviceLayerState(0),
                         &device_runtime_state,
                         sizeof(device_runtime_state),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        d_status,
        d_apply_status));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(route_participants.data(), runtime_state.route_participant_ids,
                         sizeof(route_participants), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(route_participants[0], 0);
    EXPECT_EQ(route_participants[1], 0);
    EXPECT_EQ(route_participants[2], 0);
    EXPECT_EQ(route_participants[3], 1);
    EXPECT_EQ(route_participants[4], 1);
    EXPECT_EQ(route_participants[5], 1);
    EXPECT_EQ(route_participants[6], 0);
    EXPECT_EQ(route_participants[7], 1);
    ASSERT_EQ(cudaFree(d_apply_status), cudaSuccess);
    ASSERT_EQ(cudaFree(d_status), cudaSuccess);
    ASSERT_EQ(cudaFree(d_header), cudaSuccess);
    ASSERT_EQ(cudaFree(d_plan_count), cudaSuccess);
    ASSERT_EQ(cudaFree(d_plan), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimePrefillLeastLoadedLongContextMaterializeStaysStreamHealthy)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    const auto device = llaminar2::DeviceId::cuda(0);
    constexpr int seq_len = 1801;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;
    constexpr int total_slots = seq_len * top_k;

    llaminar2::DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    llaminar2::MoERuntimeTable runtime_table(runtime_config);

    auto runtime_state = runtime_table.hostLayerState(0);
    runtime_state.participant_id = 0;
    runtime_state.participant_count = 2;
    auto &bank = runtime_state.banks[runtime_state.active_bank];
    bank.expert_count = num_experts;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int owner = expert < num_experts / 2 ? 0 : 1;
        seedRuntimeBankExpert(bank, expert, owner, owner == static_cast<int>(runtime_state.participant_id));
        bank.resident_participant_mask[expert] = 1u << static_cast<uint32_t>(owner);
    }
    ASSERT_EQ(cudaMemcpyAsync(runtime_table.deviceLayerState(0),
                              &runtime_state,
                              sizeof(runtime_state),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    auto routing_indices = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    auto routing_weights = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    for (int slot = 0; slot < total_slots; ++slot)
    {
        const int row = slot / top_k;
        const int kth = slot % top_k;
        const int expert =
            (kth == 0 && (row % 3) == 0)
                ? 128
                : (slot * 37 + row * 11 + kth * 7) % num_experts;
        routing_indices->mutable_data()[slot] = static_cast<float>(expert);
        routing_weights->mutable_data()[slot] = 1.0f / static_cast<float>(kth + 1);
    }
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(cuda_kernel_->groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentConfig llep_config;
    llep_config.expert_count = num_experts;
    llep_config.participant_count = 2;
    llep_config.enable_balanced_skip = false;
    llep_config.max_weight_transfers = 1;
    ASSERT_TRUE(cuda_kernel_->planPrefillRoutesLeastLoadedCurrentBatch(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        llep_config));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoELayerRuntime planned_runtime{};
    ASSERT_EQ(cudaMemcpy(&planned_runtime,
                         runtime_table.deviceLayerState(0),
                         sizeof(planned_runtime),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_GT(planned_runtime.reserved_u64[2], 0u);
    ASSERT_EQ(planned_runtime.reserved_u64[3], 1u);

    constexpr uint32_t plan_capacity = 32;
    constexpr uint32_t payload_slot_capacity = 1;
    llaminar2::DeviceMoERebalanceConfig rebalance_config;
    rebalance_config.num_layers = 1;
    rebalance_config.num_experts = num_experts;
    rebalance_config.top_k = top_k;
    rebalance_config.participant_id = 0;
    rebalance_config.participant_count = 2;
    rebalance_config.root_participant = 0;
    rebalance_config.window_size_tokens = 256;

    llaminar2::DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_header = nullptr;
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(cudaMalloc(&d_plan, plan_capacity * sizeof(llaminar2::DeviceMoERebalancePlanEntry)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_plan_count, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_header, sizeof(llaminar2::DeviceMoERebalanceCommandBufferHeader)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_status, sizeof(llaminar2::DeviceMoERebalanceStatus)),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->materializePrefillLeastLoadedTransferCommands(
        runtime_table.deviceLayerState(0),
        d_plan,
        d_plan_count,
        plan_capacity,
        d_header,
        d_status,
        rebalance_config,
        payload_slot_capacity,
        0));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    uint32_t materialized_plan_count = 0;
    llaminar2::DeviceMoERebalanceCommandBufferHeader materialized_header{};
    llaminar2::DeviceMoERebalanceStatus materialized_status{};
    ASSERT_EQ(cudaMemcpy(&materialized_plan_count, d_plan_count, sizeof(uint32_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&materialized_header, d_header, sizeof(materialized_header),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&materialized_status, d_status, sizeof(materialized_status),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(materialized_plan_count, 1u);
    EXPECT_EQ(materialized_header.command_count, 1u);
    EXPECT_EQ(materialized_status.status_code,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(materialized_status.planned_arrivals, 1u);
    EXPECT_EQ(materialized_status.llep_weight_transfer_count, 1u);
    EXPECT_EQ(materialized_status.llep_assignment_span_count,
              static_cast<uint32_t>(planned_runtime.reserved_u64[2]));

    ASSERT_EQ(cudaFree(d_status), cudaSuccess);
    ASSERT_EQ(cudaFree(d_header), cudaSuccess);
    ASSERT_EQ(cudaFree(d_plan_count), cudaSuccess);
    ASSERT_EQ(cudaFree(d_plan), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimePrefillLeastLoadedCurrentBatchPlanUsesSymmetricResidency)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    const auto device = llaminar2::DeviceId::cuda(0);
    constexpr int seq_len = 12;
    constexpr int num_experts = 4;
    constexpr int top_k = 1;
    constexpr int total_slots = seq_len * top_k;
    constexpr size_t max_spans = 8;
    constexpr size_t max_transfers = 4;

    struct PlanSnapshot
    {
        uint64_t span_count = 0;
        uint64_t transfer_count = 0;
        std::array<llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentSpan, max_spans> spans{};
        std::array<llaminar2::least_loaded_ep::LeastLoadedExpertWeightTransfer, max_transfers> transfers{};
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
        llaminar2::DeviceMoERuntimeTable::Config runtime_config;
        runtime_config.device_id = device;
        runtime_config.num_layers = 1;
        runtime_config.num_experts = num_experts;
        runtime_config.top_k = top_k;
        runtime_config.mirror_to_device = true;
        runtime_config.prefill_token_capacity = seq_len;
        llaminar2::MoERuntimeTable runtime_table(runtime_config);

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
                              ? llaminar2::DeviceMoEReplicaRole::Primary
                              : llaminar2::DeviceMoEReplicaRole::Replica)
                    : static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::None);
            bank.resident_participant_mask[expert] = resident_masks[expert];
        }
        EXPECT_EQ(cudaMemcpyAsync(runtime_table.deviceLayerState(0),
                                  &runtime_state,
                                  sizeof(runtime_state),
                                  cudaMemcpyHostToDevice,
                                  stream_),
                  cudaSuccess);

        auto routing_indices = llaminar2::test::TestTensorFactory::createFP32(
            {static_cast<size_t>(total_slots), 1});
        auto routing_weights = llaminar2::test::TestTensorFactory::createFP32(
            {static_cast<size_t>(total_slots), 1});
        std::copy(routing_indices_data.begin(), routing_indices_data.end(), routing_indices->mutable_data());
        std::copy(routing_weights_data.begin(), routing_weights_data.end(), routing_weights->mutable_data());
        EXPECT_TRUE(routing_indices->ensureOnDevice(device));
        EXPECT_TRUE(routing_weights->ensureOnDevice(device));

        EXPECT_TRUE(cuda_kernel_->groupPrefillRoutes(
            runtime_table.deviceLayerState(0),
            routing_indices.get(),
            routing_weights.get(),
            seq_len,
            seq_len,
            num_experts,
            top_k));

        llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentConfig config;
        config.expert_count = num_experts;
        config.participant_count = 2;
        config.enable_balanced_skip = false;
        EXPECT_TRUE(cuda_kernel_->planPrefillRoutesLeastLoadedCurrentBatch(
            runtime_table.deviceLayerState(0),
            seq_len,
            seq_len,
            num_experts,
            top_k,
            config));
        EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        llaminar2::DeviceMoELayerRuntime device_runtime_state{};
        EXPECT_EQ(cudaMemcpy(&device_runtime_state,
                             runtime_table.deviceLayerState(0),
                             sizeof(device_runtime_state),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        PlanSnapshot snapshot;
        snapshot.span_count = device_runtime_state.reserved_u64[2];
        snapshot.transfer_count = device_runtime_state.reserved_u64[3];
        EXPECT_LE(snapshot.span_count, max_spans);
        EXPECT_LE(snapshot.transfer_count, max_transfers);
        EXPECT_EQ(cudaMemcpy(snapshot.spans.data(), runtime_state.reserved_ptrs[1],
                             sizeof(snapshot.spans), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(cudaMemcpy(snapshot.transfers.data(), runtime_state.reserved_ptrs[2],
                             sizeof(snapshot.transfers), cudaMemcpyDeviceToHost),
                  cudaSuccess);
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
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimePrefillLeastLoadedTransferCommandsPreserveTransferOrder)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    const auto device = llaminar2::DeviceId::cuda(0);
    constexpr int seq_len = 30;
    constexpr int num_experts = 4;
    constexpr int top_k = 1;
    constexpr int total_slots = seq_len * top_k;

    llaminar2::DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    llaminar2::MoERuntimeTable runtime_table(runtime_config);

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
            llaminar2::toMoEExpertFlags(llaminar2::DeviceMoEExpertFlags::Valid |
                                        llaminar2::DeviceMoEExpertFlags::Resident);
        bank.resident_participant_mask[expert] =
            1u << static_cast<uint32_t>(owners[expert]);
    }
    ASSERT_EQ(cudaMemcpyAsync(runtime_table.deviceLayerState(0),
                              &runtime_state,
                              sizeof(runtime_state),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    std::vector<float> routing_indices_data(total_slots, 0.0f);
    std::vector<float> routing_weights_data(total_slots, 1.0f);
    for (int slot = 15; slot < total_slots; ++slot)
        routing_indices_data[static_cast<size_t>(slot)] = 1.0f;

    auto routing_indices = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    auto routing_weights = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data.begin(), routing_indices_data.end(),
              routing_indices->mutable_data());
    std::copy(routing_weights_data.begin(), routing_weights_data.end(),
              routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(cuda_kernel_->groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentConfig config;
    config.expert_count = num_experts;
    config.participant_count = 3;
    config.enable_balanced_skip = false;

    llaminar2::DeviceMoERebalanceConfig rebalance_config;
    rebalance_config.num_layers = 1;
    rebalance_config.num_experts = num_experts;
    rebalance_config.top_k = top_k;
    rebalance_config.participant_id = 0;
    rebalance_config.participant_count = 3;
    rebalance_config.root_participant = 0;
    rebalance_config.window_size_tokens = 256;

    llaminar2::DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_header = nullptr;
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(cudaMalloc(&d_plan, 4 * sizeof(llaminar2::DeviceMoERebalancePlanEntry)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_plan_count, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_header, sizeof(llaminar2::DeviceMoERebalanceCommandBufferHeader)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_status, sizeof(llaminar2::DeviceMoERebalanceStatus)),
              cudaSuccess);

    config.max_weight_transfers = 1;
    ASSERT_TRUE(cuda_kernel_->planPrefillRoutesLeastLoadedCurrentBatch(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        config));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&runtime_state,
                         runtime_table.deviceLayerState(0),
                         sizeof(runtime_state),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(runtime_state.reserved_u64[3], 1u)
        << "planner must honor max_weight_transfers before compact payload materialization";

    ASSERT_TRUE(cuda_kernel_->materializePrefillLeastLoadedTransferCommands(
        runtime_table.deviceLayerState(0),
        d_plan,
        d_plan_count,
        4,
        d_header,
        d_status,
        rebalance_config,
        1,
        0));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    uint32_t materialized_plan_count = 0;
    std::array<llaminar2::DeviceMoERebalancePlanEntry, 4> materialized_plan{};
    llaminar2::DeviceMoERebalanceStatus materialized_status{};
    ASSERT_EQ(cudaMemcpy(&materialized_plan_count, d_plan_count, sizeof(uint32_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(materialized_plan.data(), d_plan,
                         sizeof(materialized_plan), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&materialized_status, d_status, sizeof(materialized_status),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(materialized_plan_count, 1u);
    EXPECT_EQ(materialized_status.planned_arrivals, 1u);
    EXPECT_EQ(materialized_status.plan_overflow, 0u);
    EXPECT_EQ(materialized_status.payload_bucket_requested_slots, 1u);
    EXPECT_EQ(materialized_status.payload_bucket_overflow, 0u);
    EXPECT_EQ(materialized_status.llep_weight_transfer_count, 1u);

    config.max_weight_transfers = 0;
    ASSERT_TRUE(cuda_kernel_->planPrefillRoutesLeastLoadedCurrentBatch(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        config));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&runtime_state,
                         runtime_table.deviceLayerState(0),
                         sizeof(runtime_state),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(runtime_state.reserved_u64[2], 4u);
    EXPECT_EQ(runtime_state.reserved_u64[3], 3u);
    ASSERT_TRUE(cuda_kernel_->materializePrefillLeastLoadedTransferCommands(
        runtime_table.deviceLayerState(0),
        d_plan,
        d_plan_count,
        4,
        d_header,
        d_status,
        rebalance_config,
        4,
        0));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&materialized_plan_count, d_plan_count, sizeof(uint32_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(materialized_plan.data(), d_plan,
                         sizeof(materialized_plan), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&materialized_status, d_status, sizeof(materialized_status),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

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

    ASSERT_EQ(cudaFree(d_status), cudaSuccess);
    ASSERT_EQ(cudaFree(d_header), cudaSuccess);
    ASSERT_EQ(cudaFree(d_plan_count), cudaSuccess);
    ASSERT_EQ(cudaFree(d_plan), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimePrefillLeastLoadedStandardPlanAssignsOwnerRoutes)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    const auto device = llaminar2::DeviceId::cuda(0);
    constexpr int seq_len = 4;
    constexpr int num_experts = 2;
    constexpr int top_k = 1;
    constexpr int total_slots = seq_len * top_k;

    llaminar2::DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    llaminar2::MoERuntimeTable runtime_table(runtime_config);

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
    ASSERT_EQ(cudaMemcpyAsync(runtime_table.deviceLayerState(0),
                              &runtime_state,
                              sizeof(runtime_state),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    const float routing_indices_data[total_slots] = {0.0f, 0.0f, 1.0f, 1.0f};
    const float routing_weights_data[total_slots] = {1.0f, 0.9f, 0.8f, 0.7f};
    auto routing_indices = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    auto routing_weights = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots,
              routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots,
              routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(cuda_kernel_->groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentConfig config;
    config.expert_count = num_experts;
    config.participant_count = 2;
    config.enable_balanced_skip = true;
    ASSERT_TRUE(cuda_kernel_->planPrefillRoutesLeastLoadedCurrentBatch(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        config));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&runtime_state,
                         runtime_table.deviceLayerState(0),
                         sizeof(runtime_state),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(runtime_state.reserved_u64[2], 0u);
    EXPECT_EQ(runtime_state.reserved_u64[3], 0u);

    ASSERT_TRUE(cuda_kernel_->assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    std::array<int32_t, total_slots> route_participants{};
    ASSERT_EQ(cudaMemcpy(route_participants.data(),
                         runtime_state.route_participant_ids,
                         sizeof(route_participants),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(route_participants[0], 0);
    EXPECT_EQ(route_participants[1], 0);
    EXPECT_EQ(route_participants[2], 1);
    EXPECT_EQ(route_participants[3], 1);

    ASSERT_TRUE(cuda_kernel_->regroupPrefillRoutesFromRuntimeAssignments(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&runtime_state,
                         runtime_table.deviceLayerState(0),
                         sizeof(runtime_state),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    std::array<int32_t, num_experts> expert_counts{};
    ASSERT_EQ(cudaMemcpy(expert_counts.data(),
                         runtime_state.expert_counts,
                         sizeof(expert_counts),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(expert_counts[0], 2);
    EXPECT_EQ(expert_counts[1], 0);
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimePrefillLeastLoadedUncoveredSpanRowsFallBackToOwner)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    const auto device = llaminar2::DeviceId::cuda(0);
    constexpr int seq_len = 4;
    constexpr int num_experts = 2;
    constexpr int top_k = 1;
    constexpr int total_slots = seq_len * top_k;

    llaminar2::DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    llaminar2::MoERuntimeTable runtime_table(runtime_config);

    auto runtime_state = runtime_table.hostLayerState(0);
    runtime_state.participant_id = 0;
    runtime_state.participant_count = 2;
    auto &bank = runtime_state.banks[runtime_state.active_bank];
    bank.expert_count = num_experts;
    seedRuntimeBankExpert(bank, 0, 0, true);
    seedRuntimeBankExpert(bank, 1, 1, false);
    bank.resident_participant_mask[0] = 0b01u;
    bank.resident_participant_mask[1] = 0b10u;
    ASSERT_EQ(cudaMemcpyAsync(runtime_table.deviceLayerState(0),
                              &runtime_state,
                              sizeof(runtime_state),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    const float routing_indices_data[total_slots] = {0.0f, 1.0f, 0.0f, 1.0f};
    const float routing_weights_data[total_slots] = {1.0f, 0.9f, 0.8f, 0.7f};
    auto routing_indices = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    auto routing_weights = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots,
              routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots,
              routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(cuda_kernel_->groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    std::array<llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentSpan, 1> spans{};
    spans[0].expert = 0u;
    spans[0].owner_participant = 0u;
    spans[0].destination_participant = 0u;
    spans[0].route_row_begin = 0u;
    spans[0].route_row_end = 2u;
    spans[0].needs_foreign_weight = 0u;
    const std::array<int32_t, num_experts * 2> span_bounds = {0, 1, -1, -1};

    ASSERT_EQ(cudaMemcpy(runtime_state.reserved_ptrs[1],
                         spans.data(),
                         sizeof(spans),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(runtime_state.reserved_ptrs[0],
                         span_bounds.data(),
                         sizeof(span_bounds),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    llaminar2::DeviceMoELayerRuntime device_runtime_state{};
    ASSERT_EQ(cudaMemcpy(&device_runtime_state,
                         runtime_table.deviceLayerState(0),
                         sizeof(device_runtime_state),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    device_runtime_state.reserved_u64[2] = spans.size();
    device_runtime_state.reserved_u64[3] = 0u;
    ASSERT_EQ(cudaMemcpy(runtime_table.deviceLayerState(0),
                         &device_runtime_state,
                         sizeof(device_runtime_state),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    std::array<int32_t, total_slots> route_participants{};
    ASSERT_EQ(cudaMemcpy(route_participants.data(),
                         runtime_state.route_participant_ids,
                         sizeof(route_participants),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(route_participants[0], 0);
    EXPECT_EQ(route_participants[1], 1);
    EXPECT_EQ(route_participants[2], 0);
    EXPECT_EQ(route_participants[3], 1);
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimePrefillLeastLoadedCurrentBatchNoTransferPlanAssignsRoutes)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    const auto device = llaminar2::DeviceId::cuda(0);
    constexpr int seq_len = 4;
    constexpr int num_experts = 4;
    constexpr int top_k = 1;
    constexpr int total_slots = seq_len * top_k;

    llaminar2::DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    llaminar2::MoERuntimeTable runtime_table(runtime_config);

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
    ASSERT_EQ(cudaMemcpyAsync(runtime_table.deviceLayerState(0),
                              &runtime_state,
                              sizeof(runtime_state),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    const float routing_indices_data[total_slots] = {0.0f, 1.0f, 0.0f, 1.0f};
    const float routing_weights_data[total_slots] = {1.0f, 0.9f, 0.8f, 0.7f};
    auto routing_indices = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    auto routing_weights = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots, routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots, routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(cuda_kernel_->groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentConfig config;
    config.expert_count = num_experts;
    config.participant_count = 2;
    config.alpha_numerator = 2;
    config.alpha_denominator = 1;
    config.enable_balanced_skip = false;
    ASSERT_TRUE(cuda_kernel_->planPrefillRoutesLeastLoadedCurrentBatch(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        config));
    ASSERT_TRUE(cuda_kernel_->assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoELayerRuntime device_runtime_state{};
    ASSERT_EQ(cudaMemcpy(&device_runtime_state,
                         runtime_table.deviceLayerState(0),
                         sizeof(device_runtime_state),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(device_runtime_state.reserved_u64[2], 2u);
    EXPECT_EQ(device_runtime_state.reserved_u64[3], 0u);

    std::array<int32_t, total_slots> route_participants{};
    ASSERT_EQ(cudaMemcpy(route_participants.data(), runtime_state.route_participant_ids,
                         sizeof(route_participants), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(route_participants[0], 0);
    EXPECT_EQ(route_participants[1], 1);
    EXPECT_EQ(route_participants[2], 0);
    EXPECT_EQ(route_participants[3], 1);
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimePrefillLeastLoadedCurrentBatchResidentReplicaAvoidsTransfer)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    const auto device = llaminar2::DeviceId::cuda(0);
    constexpr int seq_len = 8;
    constexpr int num_experts = 4;
    constexpr int top_k = 1;
    constexpr int total_slots = seq_len * top_k;

    llaminar2::DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    llaminar2::MoERuntimeTable runtime_table(runtime_config);

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
    ASSERT_EQ(cudaMemcpyAsync(runtime_table.deviceLayerState(0),
                              &runtime_state,
                              sizeof(runtime_state),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    const float routing_indices_data[total_slots] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 2.0f};
    const float routing_weights_data[total_slots] = {
        1.0f, 0.9f, 0.8f, 0.7f,
        0.6f, 0.5f, 0.4f, 0.3f};
    auto routing_indices = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    auto routing_weights = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots, routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots, routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(cuda_kernel_->groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_indices.get(),
        routing_weights.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentConfig config;
    config.expert_count = num_experts;
    config.participant_count = 2;
    config.enable_balanced_skip = false;
    ASSERT_TRUE(cuda_kernel_->planPrefillRoutesLeastLoadedCurrentBatch(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k,
        config));
    ASSERT_TRUE(cuda_kernel_->assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
        runtime_table.deviceLayerState(0),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoELayerRuntime device_runtime_state{};
    ASSERT_EQ(cudaMemcpy(&device_runtime_state,
                         runtime_table.deviceLayerState(0),
                         sizeof(device_runtime_state),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(device_runtime_state.reserved_u64[2], 4u);
    EXPECT_EQ(device_runtime_state.reserved_u64[3], 0u);

    std::array<int32_t, total_slots> route_participants{};
    ASSERT_EQ(cudaMemcpy(route_participants.data(), runtime_state.route_participant_ids,
                         sizeof(route_participants), cudaMemcpyDeviceToHost),
              cudaSuccess);
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
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalanceDomainProjectionPublishesRootPayloadStatus)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    llaminar2::DeviceMoERebalanceConfig config;
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
    std::vector<llaminar2::DeviceMoERebalancePlanEntry> gathered_plan_entries(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count) *
        static_cast<size_t>(plan_capacity));
    std::vector<llaminar2::DeviceMoERebalanceCommandBufferHeader> gathered_headers(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count));
    std::vector<llaminar2::DeviceMoERebalanceWaveState> gathered_wave_states(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count));

    llaminar2::DeviceMoERebalancePlanEntry root_plan;
    root_plan.op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ExpertPayloadArrival);
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
        static_cast<uint32_t>(llaminar2::DeviceMoERebalancePipelinePhase::PlanAssignments);
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

    llaminar2::DeviceMoERebalanceStatus status;
    std::vector<llaminar2::DeviceMoERebalancePlanEntry> local_plan_entries(
        static_cast<size_t>(command_buffer_count) * static_cast<size_t>(plan_capacity));
    std::vector<llaminar2::DeviceMoERebalanceCommandBufferHeader> local_headers(
        command_buffer_count);
    std::vector<llaminar2::DeviceMoERebalanceWaveState> local_wave_states(
        command_buffer_count);

    llaminar2::DeviceMoERebalancePlanEntry *d_gathered_plan = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_gathered_headers = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_gathered_wave_states = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_local_plan = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_local_headers = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_local_wave_states = nullptr;
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_plan),
                         gathered_plan_entries.size() * sizeof(gathered_plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_headers),
                         gathered_headers.size() * sizeof(gathered_headers[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_wave_states),
                         gathered_wave_states.size() * sizeof(gathered_wave_states[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_plan),
                         local_plan_entries.size() * sizeof(local_plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_headers),
                         local_headers.size() * sizeof(local_headers[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_wave_states),
                         local_wave_states.size() * sizeof(local_wave_states[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_status), sizeof(status)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered_plan,
                              gathered_plan_entries.data(),
                              gathered_plan_entries.size() * sizeof(gathered_plan_entries[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered_headers,
                              gathered_headers.data(),
                              gathered_headers.size() * sizeof(gathered_headers[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered_wave_states,
                              gathered_wave_states.data(),
                              gathered_wave_states.size() * sizeof(gathered_wave_states[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_local_plan, 0, local_plan_entries.size() * sizeof(local_plan_entries[0]), stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_local_headers, 0, local_headers.size() * sizeof(local_headers[0]), stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_local_wave_states, 0, local_wave_states.size() * sizeof(local_wave_states[0]), stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_status, &status, sizeof(status), cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->projectDeviceRebalanceDomainCommands(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(local_plan_entries.data(),
                         d_local_plan,
                         local_plan_entries.size() * sizeof(local_plan_entries[0]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(local_headers.data(),
                         d_local_headers,
                         local_headers.size() * sizeof(local_headers[0]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(local_wave_states.data(),
                         d_local_wave_states,
                         local_wave_states.size() * sizeof(local_wave_states[0]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost),
              cudaSuccess);

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
    EXPECT_EQ(status.status_code, static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.windows_observed, 1u);
    EXPECT_EQ(status.windows_applied, 1u);
    EXPECT_EQ(status.last_epoch, root_header.epoch);
    EXPECT_EQ(status.planned_arrivals, 1u);
    EXPECT_EQ(status.payload_bucket_requested_slots, 2u);
    EXPECT_EQ(status.payload_bucket_slots, 2u);
    EXPECT_EQ(status.payload_bucket_index, 1u);
    EXPECT_EQ(status.payload_bucket_overflow, 0u);

    cudaFree(d_gathered_plan);
    cudaFree(d_gathered_headers);
    cudaFree(d_gathered_wave_states);
    cudaFree(d_local_plan);
    cudaFree(d_local_headers);
    cudaFree(d_local_wave_states);
    cudaFree(d_status);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalanceDomainProjectionRejectsNonResidentSourceParticipant)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    llaminar2::DeviceMoERebalanceConfig config;
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
    std::vector<llaminar2::DeviceMoERebalancePlanEntry> gathered_plan_entries(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count) *
        static_cast<size_t>(plan_capacity));
    std::vector<llaminar2::DeviceMoERebalanceCommandBufferHeader> gathered_headers(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count));
    std::vector<llaminar2::DeviceMoERebalanceWaveState> gathered_wave_states(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count));

    llaminar2::DeviceMoERebalancePlanEntry root_plan;
    root_plan.op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ExpertPayloadArrival);
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
        static_cast<uint32_t>(llaminar2::DeviceMoERebalancePipelinePhase::PlanAssignments);
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

    llaminar2::DeviceMoERebalanceStatus status;
    std::vector<llaminar2::DeviceMoERebalancePlanEntry> local_plan_entries(
        static_cast<size_t>(command_buffer_count) * static_cast<size_t>(plan_capacity));
    std::vector<llaminar2::DeviceMoERebalanceCommandBufferHeader> local_headers(
        command_buffer_count);
    std::vector<llaminar2::DeviceMoERebalanceWaveState> local_wave_states(
        command_buffer_count);

    llaminar2::DeviceMoERebalancePlanEntry *d_gathered_plan = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_gathered_headers = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_gathered_wave_states = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_local_plan = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_local_headers = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_local_wave_states = nullptr;
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_plan),
                         gathered_plan_entries.size() * sizeof(gathered_plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_headers),
                         gathered_headers.size() * sizeof(gathered_headers[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_wave_states),
                         gathered_wave_states.size() * sizeof(gathered_wave_states[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_plan),
                         local_plan_entries.size() * sizeof(local_plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_headers),
                         local_headers.size() * sizeof(local_headers[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_wave_states),
                         local_wave_states.size() * sizeof(local_wave_states[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_status), sizeof(status)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered_plan,
                              gathered_plan_entries.data(),
                              gathered_plan_entries.size() * sizeof(gathered_plan_entries[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered_headers,
                              gathered_headers.data(),
                              gathered_headers.size() * sizeof(gathered_headers[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered_wave_states,
                              gathered_wave_states.data(),
                              gathered_wave_states.size() * sizeof(gathered_wave_states[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_local_plan, 0, local_plan_entries.size() * sizeof(local_plan_entries[0]), stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_local_headers, 0, local_headers.size() * sizeof(local_headers[0]), stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_local_wave_states, 0, local_wave_states.size() * sizeof(local_wave_states[0]), stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_status, &status, sizeof(status), cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->projectDeviceRebalanceDomainCommands(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(local_plan_entries.data(),
                         d_local_plan,
                         local_plan_entries.size() * sizeof(local_plan_entries[0]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(local_headers.data(),
                         d_local_headers,
                         local_headers.size() * sizeof(local_headers[0]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(local_wave_states.data(),
                         d_local_wave_states,
                         local_wave_states.size() * sizeof(local_wave_states[0]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_EQ(local_headers[0].participant_id, config.participant_id);
    EXPECT_EQ(local_headers[0].participant_count, config.participant_count);
    EXPECT_EQ(local_headers[0].epoch, root_header.epoch);
    EXPECT_EQ(local_headers[0].command_count, 0u);
    EXPECT_EQ(local_plan_entries[0].op, 0u);
    EXPECT_EQ(local_wave_states[0].requested_payload_slots, 0u);
    EXPECT_EQ(local_wave_states[0].payload_bucket_slots, 0u);
    EXPECT_EQ(status.status_code, static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.windows_observed, 1u);
    EXPECT_EQ(status.windows_applied, 0u);
    EXPECT_EQ(status.planned_arrivals, 0u);
    EXPECT_EQ(status.invalid_runtime_layers, 1u);
    EXPECT_EQ(status.payload_bucket_requested_slots, 0u);
    EXPECT_EQ(status.payload_bucket_slots, 0u);
    EXPECT_EQ(status.payload_source_participant_mask, 0u);
    EXPECT_EQ(status.payload_destination_participant_mask, 0u);
    EXPECT_EQ(status.payload_edge_mask, 0ULL);

    cudaFree(d_gathered_plan);
    cudaFree(d_gathered_headers);
    cudaFree(d_gathered_wave_states);
    cudaFree(d_local_plan);
    cudaFree(d_local_headers);
    cudaFree(d_local_wave_states);
    cudaFree(d_status);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalanceDomainProjectionStatusAggregatesRootPayloadAcrossWaves)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    llaminar2::DeviceMoERebalanceConfig config;
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
    std::vector<llaminar2::DeviceMoERebalancePlanEntry> gathered_plan_entries(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count) *
        static_cast<size_t>(plan_capacity));
    std::vector<llaminar2::DeviceMoERebalanceCommandBufferHeader> gathered_headers(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count));

    llaminar2::DeviceMoERebalancePlanEntry stale_plan;
    stale_plan.op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ExpertPayloadArrival);
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
        static_cast<uint32_t>(llaminar2::DeviceMoERebalancePipelinePhase::PlanAssignments);
    stale_header.command_count = 1;
    stale_header.command_capacity = plan_capacity;
    stale_header.participant_id = 0;
    stale_header.participant_count = config.participant_count;

    llaminar2::DeviceMoERebalanceStatus status;
    std::vector<llaminar2::DeviceMoERebalancePlanEntry> local_plan_entries(
        static_cast<size_t>(command_buffer_count) * static_cast<size_t>(plan_capacity));
    std::vector<llaminar2::DeviceMoERebalanceCommandBufferHeader> local_headers(
        command_buffer_count);

    llaminar2::DeviceMoERebalancePlanEntry *d_gathered_plan = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_gathered_headers = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_local_plan = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_local_headers = nullptr;
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_plan),
                         gathered_plan_entries.size() * sizeof(gathered_plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_headers),
                         gathered_headers.size() * sizeof(gathered_headers[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_plan),
                         local_plan_entries.size() * sizeof(local_plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_headers),
                         local_headers.size() * sizeof(local_headers[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_status), sizeof(status)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered_plan,
                              gathered_plan_entries.data(),
                              gathered_plan_entries.size() * sizeof(gathered_plan_entries[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered_headers,
                              gathered_headers.data(),
                              gathered_headers.size() * sizeof(gathered_headers[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_local_plan, 0, local_plan_entries.size() * sizeof(local_plan_entries[0]), stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_local_headers, 0, local_headers.size() * sizeof(local_headers[0]), stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_status, &status, sizeof(status), cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->projectDeviceRebalanceDomainCommands(
        d_gathered_plan,
        d_gathered_headers,
        plan_capacity,
        d_local_plan,
        d_local_headers,
        config,
        d_status,
        payload_slot_capacity,
        command_buffer_count));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(local_headers.data(),
                         d_local_headers,
                         local_headers.size() * sizeof(local_headers[0]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_EQ(local_headers[0].command_count, 1u);
    EXPECT_EQ(local_headers[1].command_count, 0u);
    EXPECT_EQ(status.status_code, static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.planned_arrivals, 1u);
    EXPECT_EQ(status.payload_bucket_requested_slots, 2u);
    EXPECT_EQ(status.payload_bucket_slots, 2u);
    EXPECT_NE(status.payload_edge_mask, 0ULL);

    cudaFree(d_gathered_plan);
    cudaFree(d_gathered_headers);
    cudaFree(d_local_plan);
    cudaFree(d_local_headers);
    cudaFree(d_status);
#endif
}

TEST_F(Test__CUDAMoEKernel, PrefillLLEPDomainProjectionMergesDestinationLocalRequestsForSourcePacking)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    llaminar2::DeviceMoERebalanceConfig config;
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
    std::vector<llaminar2::DeviceMoERebalancePlanEntry> gathered_plan_entries(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count) *
        static_cast<size_t>(plan_capacity));
    std::vector<llaminar2::DeviceMoERebalanceCommandBufferHeader> gathered_headers(
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count));

    auto make_header = [&](uint32_t participant, uint32_t count)
    {
        llaminar2::DeviceMoERebalanceCommandBufferHeader header;
        header.epoch = 11;
        header.phase =
            static_cast<uint32_t>(llaminar2::DeviceMoERebalancePipelinePhase::PlanAssignments);
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
        llaminar2::DeviceMoERebalancePlanEntry plan;
        plan.op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ExpertPayloadArrival);
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

    std::vector<llaminar2::DeviceMoERebalancePlanEntry> local_plan_entries(plan_capacity);
    uint32_t local_plan_count = 0;
    llaminar2::DeviceMoERebalanceCommandBufferHeader local_header;
    llaminar2::DeviceMoERebalanceStatus status;

    llaminar2::DeviceMoERebalancePlanEntry *d_gathered_plan = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_gathered_headers = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_local_plan = nullptr;
    uint32_t *d_local_plan_count = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_local_header = nullptr;
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_plan),
                         gathered_plan_entries.size() * sizeof(gathered_plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_headers),
                         gathered_headers.size() * sizeof(gathered_headers[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_plan),
                         local_plan_entries.size() * sizeof(local_plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_plan_count), sizeof(local_plan_count)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_header), sizeof(local_header)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_status), sizeof(status)), cudaSuccess);

    ASSERT_EQ(cudaMemcpyAsync(d_gathered_plan,
                              gathered_plan_entries.data(),
                              gathered_plan_entries.size() * sizeof(gathered_plan_entries[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered_headers,
                              gathered_headers.data(),
                              gathered_headers.size() * sizeof(gathered_headers[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_local_plan, 0, local_plan_entries.size() * sizeof(local_plan_entries[0]), stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_local_plan_count, 0, sizeof(local_plan_count), stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_local_header, 0, sizeof(local_header), stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_status, &status, sizeof(status), cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->projectPrefillLeastLoadedDomainCommands(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(local_plan_entries.data(),
                         d_local_plan,
                         local_plan_entries.size() * sizeof(local_plan_entries[0]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&local_plan_count, d_local_plan_count, sizeof(local_plan_count), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&local_header, d_local_header, sizeof(local_header), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost),
              cudaSuccess);

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
    EXPECT_EQ(status.status_code, static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.planned_arrivals, 0u)
        << "projection must preserve local current-batch LLEP completion counters";
    EXPECT_EQ(status.candidate_arrivals_considered, 2u);
    EXPECT_EQ(status.payload_bucket_requested_slots, 2u);
    EXPECT_EQ(status.payload_bucket_slots, 2u);
    EXPECT_EQ(status.payload_source_participant_mask, 0b001u);
    EXPECT_EQ(status.payload_destination_participant_mask, 0b110u);

    cudaFree(d_gathered_plan);
    cudaFree(d_gathered_headers);
    cudaFree(d_local_plan);
    cudaFree(d_local_plan_count);
    cudaFree(d_local_header);
    cudaFree(d_status);
#endif
}

TEST_F(Test__CUDAMoEKernel, MoEWorkspaceBuffersAreDeviceAligned)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    auto reqs = llaminar2::MoEWorkspaceBuffers::cudaMoE(
        /*max_seq_len=*/4,
        /*d_model=*/2048,
        /*intermediate=*/512,
        /*num_experts=*/256,
        /*top_k=*/16);
    llaminar2::DeviceWorkspaceManager workspace(
        llaminar2::DeviceId::cuda(0),
        reqs.total_bytes_with_alignment() + 1024 * 1024);
    ASSERT_TRUE(workspace.allocate(reqs));

    for (const auto &desc : reqs.buffers)
    {
        void *ptr = workspace.getBuffer(desc.name);
        ASSERT_NE(ptr, nullptr) << desc.name;
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) & (desc.alignment - 1), 0u)
            << desc.name << " ptr=" << ptr << " alignment=" << desc.alignment;
    }
#endif
}

TEST_F(Test__CUDAMoEKernel, GroupPrefillRoutesRuntimeState_DeterministicSmall)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    const llaminar2::DeviceId device = llaminar2::DeviceId::cuda(0);
    constexpr int seq_len = 8;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int total_slots = seq_len * top_k;

    const std::vector<int> routing_indices = {
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
        routing_indices_f32[static_cast<size_t>(i)] =
            static_cast<float>(routing_indices[static_cast<size_t>(i)]);
        routing_weights[static_cast<size_t>(i)] =
            0.125f + 0.03125f * static_cast<float>(i);
    }

    auto routing_index_tensor =
        llaminar2::test::TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weight_tensor =
        llaminar2::test::TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_f32.begin(), routing_indices_f32.end(),
              routing_index_tensor->mutable_data());
    std::copy(routing_weights.begin(), routing_weights.end(),
              routing_weight_tensor->mutable_data());
    ASSERT_TRUE(routing_index_tensor->ensureOnDevice(device, stream_));
    ASSERT_TRUE(routing_weight_tensor->ensureOnDevice(device, stream_));

    llaminar2::DeviceMoERuntimeTable::Config config;
    config.device_id = device;
    config.num_layers = 1;
    config.num_experts = num_experts;
    config.top_k = top_k;
    config.mirror_to_device = true;
    config.prefill_token_capacity = seq_len;
    llaminar2::MoERuntimeTable runtime_table(config);

    ASSERT_TRUE(cuda_kernel_->groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_index_tensor.get(),
        routing_weight_tensor.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const auto &state = runtime_table.hostLayerState(0);
    ASSERT_EQ(state.prefill_route_capacity, static_cast<uint32_t>(total_slots));

    std::vector<int> host_route_ids(total_slots);
    std::vector<float> host_route_weights(total_slots);
    std::vector<int> host_counts(num_experts);
    std::vector<int> host_offsets(num_experts);
    std::vector<int> host_grouped_ids(total_slots);
    std::vector<float> host_grouped_weights(total_slots);
    ASSERT_EQ(cudaMemcpyAsync(host_route_ids.data(), state.route_expert_ids,
                              host_route_ids.size() * sizeof(int),
                              cudaMemcpyDeviceToHost, stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(host_route_weights.data(), state.route_weights,
                              host_route_weights.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(host_counts.data(), state.expert_counts,
                              host_counts.size() * sizeof(int),
                              cudaMemcpyDeviceToHost, stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(host_offsets.data(), state.expert_offsets,
                              host_offsets.size() * sizeof(int),
                              cudaMemcpyDeviceToHost, stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(host_grouped_ids.data(), state.grouped_token_ids,
                              host_grouped_ids.size() * sizeof(int),
                              cudaMemcpyDeviceToHost, stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(host_grouped_weights.data(), state.grouped_route_weights,
                              host_grouped_weights.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    for (int slot = 0; slot < total_slots; ++slot)
    {
        EXPECT_EQ(host_route_ids[static_cast<size_t>(slot)],
                  routing_indices[static_cast<size_t>(slot)]);
        EXPECT_FLOAT_EQ(host_route_weights[static_cast<size_t>(slot)],
                        routing_weights[static_cast<size_t>(slot)]);
    }

    const std::vector<int> expected_counts = {4, 4, 4, 4};
    int running = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        EXPECT_EQ(host_counts[static_cast<size_t>(expert)],
                  expected_counts[static_cast<size_t>(expert)]);
        EXPECT_EQ(host_offsets[static_cast<size_t>(expert)], running);
        running += expected_counts[static_cast<size_t>(expert)];
    }
    EXPECT_EQ(running, total_slots);

    std::vector<std::vector<std::pair<int, float>>> expected_per_expert(
        static_cast<size_t>(num_experts));
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
        ASSERT_EQ(static_cast<int>(expected.size()),
                  host_counts[static_cast<size_t>(expert)]);
        for (int i = 0; i < static_cast<int>(expected.size()); ++i)
        {
            EXPECT_EQ(host_grouped_ids[static_cast<size_t>(offset + i)],
                      expected[static_cast<size_t>(i)].first)
                << "expert=" << expert << " grouped row=" << i;
            EXPECT_FLOAT_EQ(host_grouped_weights[static_cast<size_t>(offset + i)],
                            expected[static_cast<size_t>(i)].second)
                << "expert=" << expert << " grouped row=" << i;
        }
    }
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimePrefillGatherScatter_ZeroCountExpertNoOps)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    const llaminar2::DeviceId device = llaminar2::DeviceId::cuda(0);
    constexpr int seq_len = 4;
    constexpr int d_model = 16;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int total_slots = seq_len * top_k;

    const std::vector<int> routing_indices = {
        0, 2,
        2, 0,
        0, 2,
        2, 0};
    std::vector<float> routing_indices_f32(static_cast<size_t>(total_slots));
    std::vector<float> routing_weights(static_cast<size_t>(total_slots), 0.5f);
    for (int i = 0; i < total_slots; ++i)
        routing_indices_f32[static_cast<size_t>(i)] =
            static_cast<float>(routing_indices[static_cast<size_t>(i)]);

    auto routing_index_tensor =
        llaminar2::test::TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weight_tensor =
        llaminar2::test::TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_f32.begin(), routing_indices_f32.end(),
              routing_index_tensor->mutable_data());
    std::copy(routing_weights.begin(), routing_weights.end(),
              routing_weight_tensor->mutable_data());
    ASSERT_TRUE(routing_index_tensor->ensureOnDevice(device, stream_));
    ASSERT_TRUE(routing_weight_tensor->ensureOnDevice(device, stream_));

    llaminar2::DeviceMoERuntimeTable::Config config;
    config.device_id = device;
    config.num_layers = 1;
    config.num_experts = num_experts;
    config.top_k = top_k;
    config.mirror_to_device = true;
    config.prefill_token_capacity = seq_len;
    llaminar2::MoERuntimeTable runtime_table(config);

    ASSERT_TRUE(cuda_kernel_->groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_index_tensor.get(),
        routing_weight_tensor.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    auto hidden =
        llaminar2::test::TestTensorFactory::createFP32({static_cast<size_t>(seq_len),
                                                       static_cast<size_t>(d_model)});
    for (int i = 0; i < seq_len * d_model; ++i)
        hidden->mutable_data()[i] = 0.01f * static_cast<float>(i + 1);
    ASSERT_TRUE(hidden->ensureOnDevice(device, stream_));

    auto batch =
        llaminar2::test::TestTensorFactory::createFP32({static_cast<size_t>(seq_len),
                                                       static_cast<size_t>(d_model)});
    ASSERT_TRUE(batch->ensureOnDevice(device, stream_));
    ASSERT_TRUE(cuda_kernel_->gatherPrefillExpertBatchFromRuntime(
        runtime_table.deviceLayerState(0), hidden.get(), batch.get(),
        /*expert_id=*/1, seq_len, d_model));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    batch->transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    const float *zero_batch = batch->data();
    for (int i = 0; i < seq_len * d_model; ++i)
        EXPECT_FLOAT_EQ(zero_batch[i], 0.0f)
            << "zero-count gather wrote row element " << i;

    auto output =
        llaminar2::test::TestTensorFactory::createFP32({static_cast<size_t>(seq_len),
                                                       static_cast<size_t>(d_model)});
    std::fill(output->mutable_data(), output->mutable_data() + seq_len * d_model, 0.25f);
    ASSERT_TRUE(output->ensureOnDevice(device, stream_));
    auto expert_output =
        llaminar2::test::TestTensorFactory::createFP32({static_cast<size_t>(seq_len),
                                                       static_cast<size_t>(d_model)});
    std::fill(expert_output->mutable_data(),
              expert_output->mutable_data() + seq_len * d_model, 7.0f);
    ASSERT_TRUE(expert_output->ensureOnDevice(device, stream_));

    ASSERT_TRUE(cuda_kernel_->scatterPrefillExpertResultsFromRuntime(
        output.get(), expert_output.get(), runtime_table.deviceLayerState(0),
        /*expert_id=*/1, seq_len, d_model));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    output->transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    const float *no_op_output = output->data();
    for (int i = 0; i < seq_len * d_model; ++i)
        EXPECT_FLOAT_EQ(no_op_output[i], 0.25f)
            << "zero-count scatter changed output element " << i;
#endif
}

    TEST_F(Test__CUDAMoEKernel, DecodeRouteSelectRuntimeTableMatchesCPU)
    {
#ifndef HAVE_CUDA
        GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    constexpr int num_layers = 1;
    constexpr int num_experts = 3;
    constexpr int top_k = 2;
    constexpr int d_model = 4;
    constexpr int seq_len = 1;

    DeviceMoERuntimeTable::Config cuda_config;
    cuda_config.device_id = llaminar2::DeviceId::cuda(0);
    cuda_config.num_layers = num_layers;
    cuda_config.num_experts = num_experts;
    cuda_config.top_k = top_k;
    cuda_config.mirror_to_device = true;
    DeviceMoERuntimeTable cuda_table(cuda_config);

    llaminar2::MoEPlacementUpdate update;
    update.epoch = 1;
    update.expert_count = num_experts;
    update.experts.resize(num_experts);
    update.local_compute_mask.assign(num_experts, 0);
    update.replica_role.resize(num_experts, 0);
    for (int i = 0; i < num_experts; ++i)
        update.experts[i].logical_expert_id = i;

    ASSERT_TRUE(cuda_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(cuda_table.flipActiveBank(0, update.epoch, stream_));

    // Deterministic hidden/gate tensors (1 token, d_model=4)
    auto hidden = makeTensor({seq_len, d_model}, {0.2f, -0.1f, 0.3f, 0.5f});
    auto gate = makeTensor({num_experts, d_model}, {0.1f, 0.2f, 0.3f, 0.4f,
                                                    -0.2f, 0.0f, 0.1f, 0.2f,
                                                    0.5f, -0.3f, 0.2f, -0.1f});
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    // CUDA decode uses the mirrored runtime table; CPU routeWithTensors is the
    // host reference for the same one-token top-k routing result.
    auto *cuda_layer = cuda_table.deviceLayerState(0);
    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        cuda_layer, hidden.get(), gate.get(), d_model, num_experts, top_k,
        true, cuda_indices.get(), cuda_weights.get(), true, true));
    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    // Compare legacy outputs
    expectNearArray(cuda_indices->data(), cpu_indices->data(), seq_len * top_k, 0.0f);
    expectNearArray(cuda_weights->data(), cpu_weights->data(), seq_len * top_k, 1e-5f);

    llaminar2::DecodeExpertHistogramConfig hist_config;
    hist_config.num_layers = num_layers;
    hist_config.num_experts = num_experts;
    hist_config.top_k = top_k;
    hist_config.window_size = 8;
    hist_config.sockets = {llaminar2::DeviceId(llaminar2::DeviceType::CPU, 0)};
    hist_config.expert_to_socket.assign(num_experts, 0);
    llaminar2::DecodeExpertHistogram runtime_histogram(hist_config);

    int expected_counts[num_experts] = {};
    const float *cpu_index_data = cpu_indices->data();
    for (int k = 0; k < top_k; ++k)
        ++expected_counts[static_cast<int>(cpu_index_data[k])];

    runtime_histogram.recordTokenBoundary(0);
    ASSERT_TRUE(cuda_table.syncDecodeHistogramToHost(runtime_histogram, stream_, true));
    EXPECT_EQ(runtime_histogram.windowTokenCount(), 1u);
    for (int expert = 0; expert < num_experts; ++expert)
        EXPECT_EQ(runtime_histogram.activationCount(0, expert), static_cast<uint64_t>(expected_counts[expert]));

    ASSERT_TRUE(cuda_table.syncDecodeHistogramToHost(runtime_histogram, stream_, true));
        for (int expert = 0; expert < num_experts; ++expert)
            EXPECT_EQ(runtime_histogram.activationCount(0, expert), static_cast<uint64_t>(expected_counts[expert]));
#endif
    }

    TEST_F(Test__CUDAMoEKernel, DecodeRouteSelectUsesQ8RouterForAlignedFP32Gate)
    {
#ifndef HAVE_CUDA
        GTEST_SKIP() << "CUDA support not compiled";
#else
        if (!hasCudaDevice())
            GTEST_SKIP() << "No CUDA device available";

        ScopedEnv perf_stats_env("LLAMINAR_PERF_STATS_JSON", "1");
        auto &gemm = llaminar2::mutableDebugEnv().gemm;
        const bool old_router_q8 = gemm.cuda_moe_router_q8;
        const bool old_reuse_hidden = gemm.cuda_moe_reuse_router_q8_hidden;
        gemm.cuda_moe_router_q8 = true;
        gemm.cuda_moe_reuse_router_q8_hidden = true;

        struct Restore
        {
            llaminar2::GemmConfig &gemm;
            bool router_q8;
            bool reuse_hidden;
            ~Restore()
            {
                gemm.cuda_moe_router_q8 = router_q8;
                gemm.cuda_moe_reuse_router_q8_hidden = reuse_hidden;
                llaminar2::PerfStatsCollector::reset();
            }
        } restore{gemm, old_router_q8, old_reuse_hidden};

        llaminar2::PerfStatsCollector::reset();

        using llaminar2::DeviceMoERuntimeTable;
        constexpr int num_layers = 1;
        constexpr int num_experts = 4;
        constexpr int top_k = 2;
        constexpr int d_model = 32;
        constexpr int seq_len = 1;

        DeviceMoERuntimeTable::Config cuda_config;
        cuda_config.device_id = llaminar2::DeviceId::cuda(0);
        cuda_config.num_layers = num_layers;
        cuda_config.num_experts = num_experts;
        cuda_config.top_k = top_k;
        cuda_config.mirror_to_device = true;
        DeviceMoERuntimeTable cuda_table(cuda_config);

        llaminar2::MoEPlacementUpdate update;
        update.epoch = 1;
        update.expert_count = num_experts;
        update.experts.resize(num_experts);
        update.local_compute_mask.assign(num_experts, 0);
        update.replica_role.resize(num_experts, 0);
        for (int i = 0; i < num_experts; ++i)
        {
            update.experts[i].logical_expert_id = i;
        }

        ASSERT_TRUE(cuda_table.prepareInactiveBank(0, update));
        ASSERT_TRUE(cuda_table.flipActiveBank(0, update.epoch, stream_));

        std::vector<float> hidden_values(d_model, 0.0f);
        hidden_values[0] = 1.0f;
        std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model, 0.0f);
        gate_values[0] = 4.0f;
        gate_values[d_model] = 3.0f;
        gate_values[2 * d_model] = 2.0f;
        gate_values[3 * d_model] = 1.0f;

        auto hidden = makeTensor({seq_len, d_model}, hidden_values);
        auto gate = makeTensor({num_experts, d_model}, gate_values);
        auto cuda_indices = makeZeros({seq_len, top_k});
        auto cuda_weights = makeZeros({seq_len, top_k});

        ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
            cuda_table.deviceLayerState(0), hidden.get(), gate.get(), d_model, num_experts, top_k,
            false, cuda_indices.get(), cuda_weights.get(), true, true));
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        EXPECT_EQ(static_cast<int>(cuda_indices->data()[0]), 0);
        EXPECT_EQ(static_cast<int>(cuda_indices->data()[1]), 1);

        const auto records =
            llaminar2::PerfStatsCollector::snapshot({"kernel.cuda_moe_router_q8_decode_calls"});
        const auto match = std::find_if(records.begin(), records.end(), [&](const llaminar2::PerfStatRecord &record)
                                        {
                                            auto tag_equals = [&](const char *key, const std::string &value)
                                            {
                                                const auto it = record.tags.find(key);
                                                return it != record.tags.end() && it->second == value;
                                            };
                                            return record.name == "cuda_moe_router_q8_decode_calls" &&
                                                   tag_equals("d_model", std::to_string(d_model)) &&
                                                   tag_equals("num_experts", std::to_string(num_experts));
                                        });
        ASSERT_NE(match, records.end()) << "CUDA decode route did not use the Q8 router path";
        EXPECT_GE(match->count, 1u);
        EXPECT_GE(match->value, 1.0);
#endif
    }

    TEST_F(Test__CUDAMoEKernel, DecodeRouteSelectFailsOnQ8CacheMissDuringGraphCapture)
    {
#ifndef HAVE_CUDA
        GTEST_SKIP() << "CUDA support not compiled";
#else
        if (!hasCudaDevice())
            GTEST_SKIP() << "No CUDA device available";

        auto &gemm = llaminar2::mutableDebugEnv().gemm;
        const bool old_router_q8 = gemm.cuda_moe_router_q8;
        gemm.cuda_moe_router_q8 = true;
        struct Restore
        {
            llaminar2::GemmConfig &gemm;
            bool router_q8;
            ~Restore()
            {
                gemm.cuda_moe_router_q8 = router_q8;
            }
        } restore{gemm, old_router_q8};

        using llaminar2::DeviceMoERuntimeTable;
        constexpr int num_layers = 1;
        constexpr int num_experts = 4;
        constexpr int top_k = 2;
        constexpr int d_model = 32;
        constexpr int seq_len = 1;

        DeviceMoERuntimeTable::Config cuda_config;
        cuda_config.device_id = llaminar2::DeviceId::cuda(0);
        cuda_config.num_layers = num_layers;
        cuda_config.num_experts = num_experts;
        cuda_config.top_k = top_k;
        cuda_config.mirror_to_device = true;
        DeviceMoERuntimeTable cuda_table(cuda_config);

        llaminar2::MoEPlacementUpdate update;
        update.epoch = 1;
        update.expert_count = num_experts;
        update.experts.resize(num_experts);
        update.local_compute_mask.assign(num_experts, 0);
        update.replica_role.resize(num_experts, 0);
        for (int expert = 0; expert < num_experts; ++expert)
            update.experts[expert].logical_expert_id = expert;

        ASSERT_TRUE(cuda_table.prepareInactiveBank(0, update));
        ASSERT_TRUE(cuda_table.flipActiveBank(0, update.epoch, stream_));

        std::vector<float> hidden_values(d_model, 0.0f);
        hidden_values[0] = 1.0f;
        std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model, 0.0f);
        gate_values[0] = 4.0f;
        gate_values[d_model] = 3.0f;
        gate_values[2 * d_model] = 2.0f;
        gate_values[3 * d_model] = 1.0f;

        auto hidden = makeTensor({seq_len, d_model}, hidden_values);
        auto gate = makeTensor({num_experts, d_model}, gate_values);
        auto cuda_indices = makeZeros({seq_len, top_k});
        auto cuda_weights = makeZeros({seq_len, top_k});

        const auto cuda_device = llaminar2::DeviceId::cuda(0);
        ASSERT_TRUE(hidden->ensureOnDevice(cuda_device, stream_));
        ASSERT_TRUE(gate->ensureOnDevice(cuda_device, stream_));
        ASSERT_TRUE(cuda_indices->ensureOnDevice(cuda_device, stream_));
        ASSERT_TRUE(cuda_weights->ensureOnDevice(cuda_device, stream_));
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        llaminar2::GraphCaptureGuard guard;
        EXPECT_FALSE(cuda_kernel_->decodeRouteSelect(
            cuda_table.deviceLayerState(0), hidden.get(), gate.get(), d_model, num_experts, top_k,
            false, cuda_indices.get(), cuda_weights.get(), true, true));
#endif
    }

    TEST_F(Test__CUDAMoEKernel, DecodeRouteSelectFailsWhenQ8RouterEnabledForUnalignedFP32Gate)
    {
#ifndef HAVE_CUDA
        GTEST_SKIP() << "CUDA support not compiled";
#else
        if (!hasCudaDevice())
            GTEST_SKIP() << "No CUDA device available";

        auto &gemm = llaminar2::mutableDebugEnv().gemm;
        const bool old_router_q8 = gemm.cuda_moe_router_q8;
        gemm.cuda_moe_router_q8 = true;
        struct Restore
        {
            llaminar2::GemmConfig &gemm;
            bool router_q8;
            ~Restore()
            {
                gemm.cuda_moe_router_q8 = router_q8;
            }
        } restore{gemm, old_router_q8};

        using llaminar2::DeviceMoERuntimeTable;
        constexpr int num_layers = 1;
        constexpr int num_experts = 4;
        constexpr int top_k = 2;
        constexpr int d_model = 4;
        constexpr int seq_len = 1;

        DeviceMoERuntimeTable::Config cuda_config;
        cuda_config.device_id = llaminar2::DeviceId::cuda(0);
        cuda_config.num_layers = num_layers;
        cuda_config.num_experts = num_experts;
        cuda_config.top_k = top_k;
        cuda_config.mirror_to_device = true;
        DeviceMoERuntimeTable cuda_table(cuda_config);

        llaminar2::MoEPlacementUpdate update;
        update.epoch = 1;
        update.expert_count = num_experts;
        update.experts.resize(num_experts);
        update.local_compute_mask.assign(num_experts, 0);
        update.replica_role.resize(num_experts, 0);
        for (int expert = 0; expert < num_experts; ++expert)
            update.experts[expert].logical_expert_id = expert;

        ASSERT_TRUE(cuda_table.prepareInactiveBank(0, update));
        ASSERT_TRUE(cuda_table.flipActiveBank(0, update.epoch, stream_));

        auto hidden = makeTensor({seq_len, d_model}, {1.0f, 0.0f, 0.0f, 0.0f});
        auto gate = makeTensor({num_experts, d_model}, {4.0f, 0.0f, 0.0f, 0.0f,
                                                        3.0f, 0.0f, 0.0f, 0.0f,
                                                        2.0f, 0.0f, 0.0f, 0.0f,
                                                        1.0f, 0.0f, 0.0f, 0.0f});
        auto cuda_indices = makeZeros({seq_len, top_k});
        auto cuda_weights = makeZeros({seq_len, top_k});

        EXPECT_FALSE(cuda_kernel_->decodeRouteSelect(
            cuda_table.deviceLayerState(0), hidden.get(), gate.get(), d_model, num_experts, top_k,
            false, cuda_indices.get(), cuda_weights.get(), true, true));
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
    }

    TEST_F(Test__CUDAMoEKernel, DeviceRebalanceControllerPublishesRuntimeBankOnStream)
    {
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream_));

    llaminar2::DeviceMoERebalanceConfig config;
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
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered),
                         gathered.size() * sizeof(uint64_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_status),
                         sizeof(llaminar2::DeviceMoERebalanceStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered, gathered.data(),
                              gathered.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->runDeviceRebalanceController(
        runtime_table.deviceLayerState(0), d_gathered, d_status, config));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceStatus status;
    ASSERT_EQ(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost), cudaSuccess);
    llaminar2::DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(cudaMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.windows_applied, 1u);
    EXPECT_EQ(runtime.active_epoch, 2u);
    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(bank.local_compute_mask[0], 1u);
    EXPECT_TRUE(llaminar2::hasMoEExpertFlag(bank.experts[0].flags,
                                            llaminar2::DeviceMoEExpertFlags::Replicated));
    EXPECT_EQ(bank.replica_role[0],
              static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::Replica));
    EXPECT_EQ(bank.local_compute_mask[2], 1u);
    EXPECT_EQ(bank.replica_role[2],
              static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::Primary));
    for (int expert = 0; expert < 4; ++expert)
    {
        EXPECT_EQ(runtime.decode_histogram[expert], 0u);
        EXPECT_EQ(runtime.decode_local_histogram[expert], 0u);
    }

    cudaFree(d_gathered);
    cudaFree(d_status);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalanceControllerPlansMissingReplicaArrivals)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    update.resident_participant_mask[0] = 0b001u;
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream_));

    llaminar2::DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::PlanMissingArrivals);
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::CollectLoadStats);

    std::vector<uint64_t> gathered(static_cast<size_t>(config.participant_count) *
                                       config.num_layers * config.num_experts,
                                   0);
    gathered[0] = 8; // participant 0, layer 0, expert 0 is hottest.

    uint64_t *d_gathered = nullptr;
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered),
                         gathered.size() * sizeof(uint64_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_status),
                         sizeof(llaminar2::DeviceMoERebalanceStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan),
                         sizeof(llaminar2::DeviceMoERebalancePlanEntry)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_count),
                         sizeof(uint32_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_header),
                         sizeof(llaminar2::DeviceMoERebalanceCommandBufferHeader)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_wave_state),
                         sizeof(llaminar2::DeviceMoERebalanceWaveState)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered, gathered.data(),
                              gathered.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->runDeviceRebalanceController(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceStatus status;
    llaminar2::DeviceMoERebalancePlanEntry plan;
    llaminar2::DeviceMoERebalanceCommandBufferHeader command_header;
    llaminar2::DeviceMoERebalanceWaveState wave_state;
    uint32_t plan_count = 0;
    llaminar2::DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&plan, d_plan, sizeof(plan), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&command_header, d_command_header, sizeof(command_header), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&wave_state, d_wave_state, sizeof(wave_state), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&plan_count, d_plan_count, sizeof(plan_count), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::Ok));
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
    EXPECT_EQ(command_header.magic, llaminar2::kDeviceMoERebalanceMagic);
    EXPECT_EQ(command_header.version, llaminar2::kDeviceMoERebalanceVersion);
    EXPECT_EQ(command_header.epoch, 2u);
    EXPECT_EQ(command_header.phase,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalancePipelinePhase::PlanAssignments));
    EXPECT_EQ(command_header.command_count, 1u);
    EXPECT_EQ(command_header.command_capacity, 1u);
    EXPECT_EQ(command_header.participant_id, 1u);
    EXPECT_EQ(command_header.participant_count, 3u);
    EXPECT_EQ(wave_state.magic, llaminar2::kDeviceMoERebalanceMagic);
    EXPECT_EQ(wave_state.version, llaminar2::kDeviceMoERebalanceVersion);
    EXPECT_EQ(wave_state.epoch, 2u);
    EXPECT_EQ(wave_state.planned_start_layer, 0u);
    EXPECT_EQ(wave_state.planned_layer_count, 1u);
    EXPECT_EQ(wave_state.command_capacity, 1u);
    EXPECT_EQ(wave_state.participant_id, 1u);
    EXPECT_EQ(wave_state.participant_count, 3u);
    EXPECT_EQ(plan.op, static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ExpertPayloadArrival));
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
    EXPECT_FALSE(llaminar2::hasMoEExpertFlag(bank.experts[0].flags,
                                             llaminar2::DeviceMoEExpertFlags::Replicated));

    cudaFree(d_gathered);
    cudaFree(d_status);
    cudaFree(d_plan);
    cudaFree(d_plan_count);
    cudaFree(d_command_header);
    cudaFree(d_wave_state);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalanceControllerPlansDynamicOwnershipTransfersWithoutHotCache)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    auto update = makeDynamicOwnershipTransferUpdate(1, 0);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream_));

    llaminar2::DeviceMoERebalanceConfig config;
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
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::PlanMissingArrivals);
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::CollectLoadStats);
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::DeferRuntimeApply);

    std::vector<uint64_t> gathered(static_cast<size_t>(config.participant_count) *
                                       config.num_layers * config.num_experts,
                                   0);
    gathered[0] = 70; // participant 0 owns the heavy expert.
    gathered[1] = 30;
    gathered[config.num_experts + 2] = 1;
    gathered[config.num_experts + 3] = 9;

    uint64_t *d_gathered = nullptr;
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered),
                         gathered.size() * sizeof(uint64_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_status),
                         sizeof(llaminar2::DeviceMoERebalanceStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan),
                         2 * sizeof(llaminar2::DeviceMoERebalancePlanEntry)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_count),
                         sizeof(uint32_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_header),
                         sizeof(llaminar2::DeviceMoERebalanceCommandBufferHeader)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_wave_state),
                         sizeof(llaminar2::DeviceMoERebalanceWaveState)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered, gathered.data(),
                              gathered.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->runDeviceRebalanceController(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceStatus status;
    std::array<llaminar2::DeviceMoERebalancePlanEntry, 2> plan{};
    llaminar2::DeviceMoERebalanceCommandBufferHeader command_header;
    llaminar2::DeviceMoERebalanceWaveState wave_state;
    uint32_t plan_count = 0;
    llaminar2::DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(plan.data(), d_plan, sizeof(plan), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&command_header, d_command_header, sizeof(command_header), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&wave_state, d_wave_state, sizeof(wave_state), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&plan_count, d_plan_count, sizeof(plan_count), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::Ok));
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

    EXPECT_EQ(plan[0].op, static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::OwnershipTransfer));
    EXPECT_EQ(plan[0].expert, 0u);
    EXPECT_EQ(plan[0].source_participant, 0u);
    EXPECT_EQ(plan[0].destination_participant, 1u);
    EXPECT_EQ(plan[0].source_resident_mask, 0b01u);
    EXPECT_EQ(plan[0].destination_slot, 0u);
    EXPECT_EQ(plan[0].payload_slot, 0u);

    EXPECT_EQ(plan[1].op, static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::OwnershipTransfer));
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

    cudaFree(d_gathered);
    cudaFree(d_status);
    cudaFree(d_plan);
    cudaFree(d_plan_count);
    cudaFree(d_command_header);
    cudaFree(d_wave_state);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalanceControllerRejectsOwnershipSwapFromNonResidentSource)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    auto update = makeDynamicOwnershipTransferUpdate(1, 0);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream_));

    llaminar2::DeviceMoELayerRuntime seeded_runtime{};
    ASSERT_EQ(cudaMemcpy(&seeded_runtime,
                         runtime_table.deviceLayerState(0),
                         sizeof(seeded_runtime),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    seeded_runtime.banks[seeded_runtime.active_bank].resident_participant_mask[2] = 0b01u;
    ASSERT_EQ(cudaMemcpyAsync(runtime_table.deviceLayerState(0),
                              &seeded_runtime,
                              sizeof(seeded_runtime),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    llaminar2::DeviceMoERebalanceConfig config;
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
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::PlanMissingArrivals);
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::CollectLoadStats);
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::DeferRuntimeApply);

    std::vector<uint64_t> gathered(static_cast<size_t>(config.participant_count) *
                                       config.num_layers * config.num_experts,
                                   0);
    gathered[0] = 70;
    gathered[1] = 30;
    gathered[config.num_experts + 2] = 1;
    gathered[config.num_experts + 3] = 9;

    uint64_t *d_gathered = nullptr;
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered),
                         gathered.size() * sizeof(uint64_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_status),
                         sizeof(llaminar2::DeviceMoERebalanceStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan),
                         2 * sizeof(llaminar2::DeviceMoERebalancePlanEntry)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_count),
                         sizeof(uint32_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_header),
                         sizeof(llaminar2::DeviceMoERebalanceCommandBufferHeader)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_wave_state),
                         sizeof(llaminar2::DeviceMoERebalanceWaveState)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered, gathered.data(),
                              gathered.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->runDeviceRebalanceController(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceStatus status;
    llaminar2::DeviceMoERebalanceCommandBufferHeader command_header;
    uint32_t plan_count = 99;
    ASSERT_EQ(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&command_header, d_command_header, sizeof(command_header), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&plan_count, d_plan_count, sizeof(plan_count), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.dynamic_ownership_swap_attempts, 1u);
    EXPECT_EQ(status.dynamic_ownership_swap_accepts, 0u);
    EXPECT_EQ(status.dynamic_ownership_swap_rejections, 1u);
    EXPECT_EQ(status.skipped_no_resident, 1u);
    EXPECT_EQ(status.planned_arrivals, 0u);
    EXPECT_EQ(status.accepted_load_spread_improvement_total, 0u);
    EXPECT_EQ(plan_count, 0u);
    EXPECT_EQ(command_header.command_count, 0u);

    cudaFree(d_gathered);
    cudaFree(d_status);
    cudaFree(d_plan);
    cudaFree(d_plan_count);
    cudaFree(d_command_header);
    cudaFree(d_wave_state);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalanceControllerNarrowsDeferredApplySpanToCommandLayers)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 4;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    auto update = makeDynamicOwnershipTransferUpdate(1, 0);
    for (uint32_t layer = 0; layer < table_config.num_layers; ++layer)
    {
        ASSERT_TRUE(runtime_table.prepareInactiveBank(layer, update));
        ASSERT_TRUE(runtime_table.flipActiveBank(layer, update.epoch, stream_));
    }

    llaminar2::DeviceMoERebalanceConfig config;
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
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::PlanMissingArrivals);
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::CollectLoadStats);
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::DeferRuntimeApply);

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
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered),
                         gathered.size() * sizeof(uint64_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_status),
                         sizeof(llaminar2::DeviceMoERebalanceStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan),
                         2 * sizeof(llaminar2::DeviceMoERebalancePlanEntry)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_count),
                         sizeof(uint32_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_header),
                         sizeof(llaminar2::DeviceMoERebalanceCommandBufferHeader)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_wave_state),
                         sizeof(llaminar2::DeviceMoERebalanceWaveState)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered, gathered.data(),
                              gathered.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->runDeviceRebalanceController(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceStatus status;
    std::array<llaminar2::DeviceMoERebalancePlanEntry, 2> plan{};
    llaminar2::DeviceMoERebalanceCommandBufferHeader command_header;
    llaminar2::DeviceMoERebalanceWaveState wave_state;
    uint32_t plan_count = 0;
    ASSERT_EQ(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(plan.data(), d_plan, sizeof(plan), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&command_header, d_command_header, sizeof(command_header), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&wave_state, d_wave_state, sizeof(wave_state), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&plan_count, d_plan_count, sizeof(plan_count), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.changed_layers, 1u);
    ASSERT_EQ(plan_count, 2u);
    EXPECT_EQ(command_header.command_count, 2u);
    EXPECT_EQ(plan[0].layer, 1u);
    EXPECT_EQ(plan[1].layer, 1u);
    EXPECT_EQ(wave_state.planned_start_layer, 1u);
    EXPECT_EQ(wave_state.planned_layer_count, 1u);
    EXPECT_EQ(wave_state.next_start_layer, 0u);

    cudaFree(d_gathered);
    cudaFree(d_status);
    cudaFree(d_plan);
    cudaFree(d_plan_count);
    cudaFree(d_command_header);
    cudaFree(d_wave_state);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalanceControllerPlansLeastLoadedEPOwnershipTransfersWithoutHotCache)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    auto update = makeDynamicOwnershipTransferUpdate(1, 0);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream_));

    llaminar2::DeviceMoERebalanceConfig config;
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
    config.routed_assignment_policy = llaminar2::kDeviceMoERebalanceAssignmentLeastLoadedEP;
    config.flags = 0;
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::PlanMissingArrivals);
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::CollectLoadStats);
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::DeferRuntimeApply);

    std::vector<uint64_t> gathered(static_cast<size_t>(config.participant_count) *
                                       config.num_layers * config.num_experts,
                                   0);
    gathered[0] = 80;
    gathered[1] = 20;

    uint64_t *d_gathered = nullptr;
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered),
                         gathered.size() * sizeof(uint64_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_status),
                         sizeof(llaminar2::DeviceMoERebalanceStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan),
                         sizeof(llaminar2::DeviceMoERebalancePlanEntry)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_count),
                         sizeof(uint32_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_header),
                         sizeof(llaminar2::DeviceMoERebalanceCommandBufferHeader)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_wave_state),
                         sizeof(llaminar2::DeviceMoERebalanceWaveState)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered, gathered.data(),
                              gathered.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->runDeviceRebalanceController(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceStatus status;
    llaminar2::DeviceMoERebalancePlanEntry plan;
    uint32_t plan_count = 0;
    ASSERT_EQ(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&plan, d_plan, sizeof(plan), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&plan_count, d_plan_count, sizeof(plan_count), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::Ok));
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
    EXPECT_EQ(plan.op, static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::OwnershipTransfer));
    EXPECT_EQ(plan.layer, 0u);
    EXPECT_EQ(plan.expert, 0u);
    EXPECT_EQ(plan.source_participant, 0u);
    EXPECT_EQ(plan.destination_participant, 1u);
    EXPECT_EQ(plan.source_resident_mask, 0b01u);
    EXPECT_EQ(plan.destination_slot, 0u);
    EXPECT_EQ(plan.payload_slot, 0u);

    cudaFree(d_gathered);
    cudaFree(d_status);
    cudaFree(d_plan);
    cudaFree(d_plan_count);
    cudaFree(d_command_header);
    cudaFree(d_wave_state);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalancePublishNoopClearsStaleActiveWave)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    llaminar2::DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::DeferRuntimeApply);

    constexpr uint32_t command_buffer_count = 2;
    llaminar2::DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.next_epoch = 7;
    controller_state.active_wave = 1;
    controller_state.wave_count = command_buffer_count;
    auto &stale_wave = controller_state.waves[1];
    stale_wave.epoch = 6;
    stale_wave.state = static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::ReadyToApply);
    stale_wave.planned_start_layer = 0;
    stale_wave.planned_layer_count = 1;
    stale_wave.command_count = 2;
    stale_wave.copied_arrivals = 2;
    stale_wave.requested_payload_slots = 2;
    stale_wave.payload_bucket_slots = 2;
    stale_wave.payload_bucket_overflow = 1;

    std::array<llaminar2::DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    std::array<llaminar2::DeviceMoERebalanceWaveState, command_buffer_count> wave_states{};
    std::array<llaminar2::DeviceMoERebalancePlanEntry, command_buffer_count> plan_entries{};
    std::array<llaminar2::DeviceMoERebalanceApplyStatus, 3> gathered_copy_status{};
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

    llaminar2::DeviceMoERebalanceApplyStatus copy_status;
    copy_status.copied_arrivals = 9;

    llaminar2::DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_wave_states = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_copy_status = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_gathered_copy_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_headers),
                         command_headers.size() * sizeof(command_headers[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_wave_states),
                         wave_states.size() * sizeof(wave_states[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_entries),
                         plan_entries.size() * sizeof(plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_copy_status), sizeof(copy_status)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_copy_status),
                         gathered_copy_status.size() * sizeof(gathered_copy_status[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_controller_state, &controller_state, sizeof(controller_state),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_command_headers, command_headers.data(),
                              command_headers.size() * sizeof(command_headers[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_wave_states, wave_states.data(),
                              wave_states.size() * sizeof(wave_states[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_plan_entries, plan_entries.data(),
                              plan_entries.size() * sizeof(plan_entries[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_copy_status, &copy_status, sizeof(copy_status),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered_copy_status, gathered_copy_status.data(),
                              gathered_copy_status.size() * sizeof(gathered_copy_status[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->publishDeviceRebalanceTransferComplete(
        d_controller_state,
        d_command_headers,
        d_wave_states,
        d_copy_status,
        d_plan_entries,
        1,
        d_gathered_copy_status,
        config,
        command_buffer_count));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceGraphControllerState result;
    ASSERT_EQ(cudaMemcpy(&result, d_controller_state, sizeof(result), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(result.maintenance_launches, 1u);
    const auto &cleared_wave = result.waves[1];
    EXPECT_EQ(cleared_wave.magic, llaminar2::kDeviceMoERebalanceMagic);
    EXPECT_EQ(cleared_wave.version, llaminar2::kDeviceMoERebalanceVersion);
    EXPECT_EQ(cleared_wave.epoch, 0u);
    EXPECT_EQ(cleared_wave.state,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::Idle));
    EXPECT_EQ(cleared_wave.command_count, 0u);
    EXPECT_EQ(cleared_wave.copied_arrivals, 0u);
    EXPECT_EQ(cleared_wave.requested_payload_slots, 0u);
    EXPECT_EQ(cleared_wave.payload_bucket_slots, 0u);
    EXPECT_EQ(cleared_wave.payload_bucket_overflow, 0u);

    cudaFree(d_controller_state);
    cudaFree(d_command_headers);
    cudaFree(d_wave_states);
    cudaFree(d_plan_entries);
    cudaFree(d_copy_status);
    cudaFree(d_gathered_copy_status);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalancePublishUsesProjectedWavePayloadBucket)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    llaminar2::DeviceMoERebalanceConfig config;
    config.num_layers = 4;
    config.num_experts = 8;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 2;
    config.window_size_tokens = 1;
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::DeferRuntimeApply);

    constexpr uint32_t command_buffer_count = 2;
    llaminar2::DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.active_wave = 0;
    controller_state.wave_count = command_buffer_count;
    controller_state.waves[0].requested_payload_slots = 0;
    controller_state.waves[0].payload_bucket_slots = 0;
    controller_state.waves[0].payload_bucket_index = 0;
    controller_state.waves[0].payload_bucket_overflow = 0;

    std::array<llaminar2::DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    std::array<llaminar2::DeviceMoERebalanceWaveState, command_buffer_count> wave_states{};
    std::array<llaminar2::DeviceMoERebalancePlanEntry, command_buffer_count * 4> plan_entries{};
    std::array<llaminar2::DeviceMoERebalanceApplyStatus, 2> gathered_copy_status{};
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
    plan_entries[0].op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ExpertPayloadArrival);
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

    llaminar2::DeviceMoERebalanceApplyStatus copy_status;
    copy_status.copied_arrivals = 1;
    gathered_copy_status[1].copied_arrivals = 1;

    llaminar2::DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_wave_states = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_copy_status = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_gathered_copy_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_headers),
                         command_headers.size() * sizeof(command_headers[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_wave_states),
                         wave_states.size() * sizeof(wave_states[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_entries),
                         plan_entries.size() * sizeof(plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_copy_status), sizeof(copy_status)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_copy_status),
                         gathered_copy_status.size() * sizeof(gathered_copy_status[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_controller_state, &controller_state, sizeof(controller_state),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_command_headers, command_headers.data(),
                              command_headers.size() * sizeof(command_headers[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_wave_states, wave_states.data(),
                              wave_states.size() * sizeof(wave_states[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_plan_entries, plan_entries.data(),
                              plan_entries.size() * sizeof(plan_entries[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_copy_status, &copy_status, sizeof(copy_status),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered_copy_status, gathered_copy_status.data(),
                              gathered_copy_status.size() * sizeof(gathered_copy_status[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->publishDeviceRebalanceTransferComplete(
        d_controller_state,
        d_command_headers,
        d_wave_states,
        d_copy_status,
        d_plan_entries,
        4,
        d_gathered_copy_status,
        config,
        command_buffer_count));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceGraphControllerState result;
    ASSERT_EQ(cudaMemcpy(&result, d_controller_state, sizeof(result), cudaMemcpyDeviceToHost), cudaSuccess);
    const auto &published_wave = result.waves[0];
    EXPECT_EQ(published_wave.epoch, command_headers[0].epoch);
    EXPECT_EQ(published_wave.state,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::ReadyToApply));
    EXPECT_EQ(published_wave.planned_start_layer, wave_states[0].planned_start_layer);
    EXPECT_EQ(published_wave.planned_layer_count, wave_states[0].planned_layer_count);
    EXPECT_EQ(published_wave.command_count, command_headers[0].command_count);
    EXPECT_EQ(published_wave.copied_arrivals, copy_status.copied_arrivals);
    EXPECT_EQ(published_wave.requested_payload_slots, wave_states[0].requested_payload_slots);
    EXPECT_EQ(published_wave.payload_bucket_slots, wave_states[0].payload_bucket_slots);
    EXPECT_EQ(published_wave.payload_bucket_index, wave_states[0].payload_bucket_index);
    EXPECT_EQ(published_wave.payload_bucket_overflow, wave_states[0].payload_bucket_overflow);

    cudaFree(d_controller_state);
    cudaFree(d_command_headers);
    cudaFree(d_wave_states);
    cudaFree(d_plan_entries);
    cudaFree(d_copy_status);
    cudaFree(d_gathered_copy_status);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalancePublishRequiresGatheredDestinationCopyStatus)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    llaminar2::DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 0;
    config.participant_count = 2;
    config.window_size_tokens = 1;
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::DeferRuntimeApply);

    constexpr uint32_t command_buffer_count = 2;
    constexpr uint32_t plan_capacity = 1;
    llaminar2::DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.active_wave = 0;
    controller_state.wave_count = command_buffer_count;

    std::array<llaminar2::DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    std::array<llaminar2::DeviceMoERebalanceWaveState, command_buffer_count> wave_states{};
    std::array<llaminar2::DeviceMoERebalancePlanEntry, command_buffer_count * plan_capacity> plan_entries{};
    std::array<llaminar2::DeviceMoERebalanceApplyStatus, 2> gathered_copy_status{};
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
    plan_entries[0].op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    plan_entries[0].layer = 0;
    plan_entries[0].expert = 1;
    plan_entries[0].source_participant = 0;
    plan_entries[0].destination_participant = 1;
    plan_entries[0].source_resident_mask = 0b01u;
    plan_entries[0].destination_slot = 0;
    plan_entries[0].payload_slot = 0;

    llaminar2::DeviceMoERebalanceApplyStatus copy_status;
    copy_status.copied_arrivals = 0;
    gathered_copy_status[0].copied_arrivals = 0;
    gathered_copy_status[1].copied_arrivals = 0;

    llaminar2::DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_wave_states = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_copy_status = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_gathered_copy_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_headers),
                         command_headers.size() * sizeof(command_headers[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_wave_states),
                         wave_states.size() * sizeof(wave_states[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_entries),
                         plan_entries.size() * sizeof(plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_copy_status), sizeof(copy_status)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_copy_status),
                         gathered_copy_status.size() * sizeof(gathered_copy_status[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_controller_state, &controller_state, sizeof(controller_state),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_command_headers, command_headers.data(),
                              command_headers.size() * sizeof(command_headers[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_wave_states, wave_states.data(),
                              wave_states.size() * sizeof(wave_states[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_plan_entries, plan_entries.data(),
                              plan_entries.size() * sizeof(plan_entries[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_copy_status, &copy_status, sizeof(copy_status),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered_copy_status, gathered_copy_status.data(),
                              gathered_copy_status.size() * sizeof(gathered_copy_status[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->publishDeviceRebalanceTransferComplete(
        d_controller_state,
        d_command_headers,
        d_wave_states,
        d_copy_status,
        d_plan_entries,
        plan_capacity,
        d_gathered_copy_status,
        config,
        command_buffer_count));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceGraphControllerState result;
    ASSERT_EQ(cudaMemcpy(&result, d_controller_state, sizeof(result), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(result.last_error_code,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::MissingTransferCompletion));
    const auto &wave = result.waves[0];
    EXPECT_EQ(wave.state,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::Error));
    EXPECT_NE(wave.state,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::ReadyToApply));
    EXPECT_EQ(wave.error_code,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::MissingTransferCompletion));

    cudaFree(d_controller_state);
    cudaFree(d_command_headers);
    cudaFree(d_wave_states);
    cudaFree(d_plan_entries);
    cudaFree(d_copy_status);
    cudaFree(d_gathered_copy_status);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalancePublishRejectsSourceSideCopyErrors)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    llaminar2::DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 0;
    config.participant_count = 2;
    config.window_size_tokens = 1;
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::DeferRuntimeApply);

    constexpr uint32_t command_buffer_count = 2;
    constexpr uint32_t plan_capacity = 1;
    llaminar2::DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.active_wave = 0;
    controller_state.wave_count = command_buffer_count;

    std::array<llaminar2::DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    std::array<llaminar2::DeviceMoERebalanceWaveState, command_buffer_count> wave_states{};
    std::array<llaminar2::DeviceMoERebalancePlanEntry, command_buffer_count * plan_capacity> plan_entries{};
    std::array<llaminar2::DeviceMoERebalanceApplyStatus, 2> gathered_copy_status{};
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
    plan_entries[0].op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ExpertPayloadArrival);
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
    llaminar2::DeviceMoERebalanceApplyStatus copy_status;
    copy_status.missing_source_descriptors = 1;
    gathered_copy_status[0].missing_source_descriptors = 1;
    gathered_copy_status[1].copied_arrivals = 1;

    llaminar2::DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_wave_states = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_copy_status = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_gathered_copy_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_headers),
                         command_headers.size() * sizeof(command_headers[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_wave_states),
                         wave_states.size() * sizeof(wave_states[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_entries),
                         plan_entries.size() * sizeof(plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_copy_status), sizeof(copy_status)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_copy_status),
                         gathered_copy_status.size() * sizeof(gathered_copy_status[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_controller_state, &controller_state, sizeof(controller_state),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_command_headers, command_headers.data(),
                              command_headers.size() * sizeof(command_headers[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_wave_states, wave_states.data(),
                              wave_states.size() * sizeof(wave_states[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_plan_entries, plan_entries.data(),
                              plan_entries.size() * sizeof(plan_entries[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_copy_status, &copy_status, sizeof(copy_status),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered_copy_status, gathered_copy_status.data(),
                              gathered_copy_status.size() * sizeof(gathered_copy_status[0]),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->publishDeviceRebalanceTransferComplete(
        d_controller_state,
        d_command_headers,
        d_wave_states,
        d_copy_status,
        d_plan_entries,
        plan_capacity,
        d_gathered_copy_status,
        config,
        command_buffer_count));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceGraphControllerState result;
    ASSERT_EQ(cudaMemcpy(&result, d_controller_state, sizeof(result), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(result.last_error_code,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::MissingTransferCompletion));
    const auto &wave = result.waves[0];
    EXPECT_EQ(wave.state,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::Error));
    EXPECT_NE(wave.state,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::ReadyToApply));
    EXPECT_EQ(wave.error_code,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::MissingTransferCompletion));

    cudaFree(d_controller_state);
    cudaFree(d_command_headers);
    cudaFree(d_wave_states);
    cudaFree(d_plan_entries);
    cudaFree(d_copy_status);
    cudaFree(d_gathered_copy_status);
#endif
}

TEST_F(Test__CUDAMoEKernel, ApplyReadyDeviceRebalanceWaveClearsAppliedCommandHeader)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream_));

    llaminar2::DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;

    constexpr uint32_t plan_capacity = 4;
    constexpr uint32_t command_buffer_count = 2;
    std::vector<llaminar2::DeviceMoERebalancePlanEntry> plan_entries(
        static_cast<size_t>(plan_capacity) * command_buffer_count);
    auto &plan = plan_entries[0];
    plan.op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ResidentExpertAssignment);
    plan.layer = 0;
    plan.expert = 0;
    plan.source_participant = 0;
    plan.destination_participant = config.participant_id;
    plan.source_resident_mask = 0b001u;
    plan.destination_slot = 0;
    plan.payload_slot = llaminar2::kDeviceMoEInvalidSlot;

    std::array<llaminar2::DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    for (uint32_t i = 0; i < command_buffer_count; ++i)
    {
        command_headers[i].command_capacity = plan_capacity;
        command_headers[i].participant_id = config.participant_id;
        command_headers[i].participant_count = config.participant_count;
    }
    command_headers[0].epoch = 7;
    command_headers[0].command_count = 1;

    llaminar2::DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.next_epoch = 8;
    controller_state.active_wave = 0;
    controller_state.wave_count = command_buffer_count;
    auto &wave = controller_state.waves[0];
    wave.epoch = command_headers[0].epoch;
    wave.state = static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::ReadyToApply);
    wave.planned_start_layer = 0;
    wave.planned_layer_count = 1;
    wave.command_count = 1;
    wave.copied_arrivals = 1;
    wave.requested_payload_slots = 0;
    wave.payload_bucket_slots = 0;

    llaminar2::DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    llaminar2::DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_apply_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_entries),
                         plan_entries.size() * sizeof(plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_headers),
                         command_headers.size() * sizeof(command_headers[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_apply_status),
                         sizeof(llaminar2::DeviceMoERebalanceApplyStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_plan_entries,
                              plan_entries.data(),
                              plan_entries.size() * sizeof(plan_entries[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_command_headers,
                              command_headers.data(),
                              command_headers.size() * sizeof(command_headers[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_controller_state,
                              &controller_state,
                              sizeof(controller_state),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->applyReadyDeviceRebalanceWave(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceApplyStatus apply_status;
    llaminar2::DeviceMoERebalanceGraphControllerState result_state;
    std::array<llaminar2::DeviceMoERebalanceCommandBufferHeader, command_buffer_count> result_headers{};
    llaminar2::DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(cudaMemcpy(&apply_status, d_apply_status, sizeof(apply_status), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&result_state, d_controller_state, sizeof(result_state), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(result_headers.data(),
                         d_command_headers,
                         result_headers.size() * sizeof(result_headers[0]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&runtime,
                         runtime_table.deviceLayerState(0),
                         sizeof(runtime),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_EQ(apply_status.status_code,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceApplyStatusCode::Ok));
    EXPECT_EQ(apply_status.plan_entries_seen, 1u);
    EXPECT_EQ(apply_status.changed_layers, 1u);
    EXPECT_EQ(apply_status.applied_arrivals, 1u);
    EXPECT_EQ(apply_status.post_apply_multi_resident_experts, 1u);
    EXPECT_EQ(result_state.decode_apply_hits, 1u);
    EXPECT_EQ(result_state.active_wave, 1u);
    EXPECT_EQ(result_state.waves[0].state,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::Applied));
    EXPECT_EQ(result_headers[0].epoch, 0u);
    EXPECT_EQ(result_headers[0].command_count, 0u);
    EXPECT_EQ(result_headers[0].command_capacity, plan_capacity);
    EXPECT_EQ(result_headers[0].participant_id, config.participant_id);
    EXPECT_EQ(result_headers[0].participant_count, config.participant_count);
    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(bank.local_compute_mask[0], 1u);
    EXPECT_EQ(bank.replica_role[0],
              static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::Replica));
    EXPECT_EQ(bank.reserved[0], 1u);
    EXPECT_TRUE(llaminar2::hasMoEExpertFlag(bank.experts[0].flags,
                                            llaminar2::DeviceMoEExpertFlags::Replicated));

    cudaFree(d_plan_entries);
    cudaFree(d_command_headers);
    cudaFree(d_controller_state);
    cudaFree(d_apply_status);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalanceOwnershipTransferUpdatesSourceOwner)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    auto update = makeDynamicOwnershipTransferUpdate(1, 0);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream_));

    llaminar2::DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 0;
    config.participant_count = 2;
    config.window_size_tokens = 1;

    llaminar2::DeviceMoERebalancePlanEntry plan;
    plan.op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::OwnershipTransfer);
    plan.layer = 0;
    plan.expert = 0;
    plan.source_participant = 0;
    plan.destination_participant = 1;
    plan.source_resident_mask = 0b01u;
    plan.destination_slot = 0;
    plan.payload_slot = 0;

    llaminar2::DeviceMoERebalanceCommandBufferHeader command_header;
    command_header.epoch = 2;
    command_header.phase =
        static_cast<uint32_t>(llaminar2::DeviceMoERebalancePipelinePhase::PlanAssignments);
    command_header.command_count = 1;
    command_header.command_capacity = 1;
    command_header.participant_id = config.participant_id;
    command_header.participant_count = config.participant_count;

    llaminar2::DeviceMoERebalancePlanEntry *d_plan = nullptr;
    llaminar2::DeviceMoEExpertDirectoryEntry *d_transfer_slots = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_apply_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan), sizeof(plan)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_transfer_slots),
                         sizeof(llaminar2::DeviceMoEExpertDirectoryEntry)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_header), sizeof(command_header)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_apply_status),
                         sizeof(llaminar2::DeviceMoERebalanceApplyStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_plan, &plan, sizeof(plan), cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_command_header,
                              &command_header,
                              sizeof(command_header),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->applyDeviceRebalanceArrivals(
        runtime_table.deviceLayerState(0),
        d_plan,
        nullptr,
        1,
        d_transfer_slots,
        1,
        config,
        d_apply_status,
        d_command_header));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceApplyStatus apply_status;
    llaminar2::DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(cudaMemcpy(&apply_status, d_apply_status, sizeof(apply_status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime), cudaMemcpyDeviceToHost), cudaSuccess);

    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(apply_status.applied_arrivals, 0u)
        << "source-side ownership metadata apply does not copy a payload arrival";
    EXPECT_EQ(apply_status.changed_layers, 1u);
    EXPECT_EQ(bank.experts[0].owner_participant, 1)
        << "source participants must not resurrect stale ownership on the next runtime bank rebuild";
    EXPECT_EQ(bank.local_compute_mask[0], 0u);
    EXPECT_EQ(bank.replica_role[0], static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::None));
    EXPECT_EQ(bank.resident_participant_mask[0], 0b10u);
    EXPECT_FALSE(llaminar2::hasMoEExpertFlag(bank.experts[0].flags,
                                             llaminar2::DeviceMoEExpertFlags::Resident));
    EXPECT_FALSE(llaminar2::hasMoEExpertFlag(bank.experts[0].flags,
                                             llaminar2::DeviceMoEExpertFlags::LocalCompute));
    EXPECT_FALSE(llaminar2::hasMoEExpertFlag(bank.experts[0].flags,
                                             llaminar2::DeviceMoEExpertFlags::Replicated));

    cudaFree(d_plan);
    cudaFree(d_transfer_slots);
    cudaFree(d_command_header);
    cudaFree(d_apply_status);
#endif
}

TEST_F(Test__CUDAMoEKernel, DecodeRouteSelectWithReadyRebalanceApplyTargetAllRollsThroughLayers)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 2;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    for (int layer = 0; layer < 2; ++layer)
    {
        auto update = makeParticipantOneBaseUpdate(1);
        ASSERT_TRUE(runtime_table.prepareInactiveBank(layer, update));
        ASSERT_TRUE(runtime_table.flipActiveBank(layer, update.epoch, stream_));
    }

    llaminar2::DeviceMoERebalanceConfig config;
    config.num_layers = 2;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::DeferRuntimeApply);

    constexpr uint32_t plan_capacity = 4;
    constexpr uint32_t command_buffer_count = 2;
    auto hidden = makeTensor({1, 4}, {1.0f, 0.0f, 0.0f, 0.0f});
    auto gate = makeTensor({4, 4}, {4.0f, 0.0f, 0.0f, 0.0f,
                                    3.0f, 0.0f, 0.0f, 0.0f,
                                    2.0f, 0.0f, 0.0f, 0.0f,
                                    1.0f, 0.0f, 0.0f, 0.0f});
    auto indices = makeZeros({1, 2});
    auto weights = makeZeros({1, 2});

    std::vector<llaminar2::DeviceMoERebalancePlanEntry> plan_entries(
        static_cast<size_t>(plan_capacity) * command_buffer_count);
    for (uint32_t layer = 0; layer < 2; ++layer)
    {
        auto &plan = plan_entries[layer];
        plan.op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ResidentExpertAssignment);
        plan.layer = layer;
        plan.expert = 0;
        plan.source_participant = 0;
        plan.destination_participant = config.participant_id;
        plan.source_resident_mask = 0b001u;
        plan.destination_slot = 0;
        plan.payload_slot = llaminar2::kDeviceMoEInvalidSlot;
    }

    std::array<llaminar2::DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    for (auto &header : command_headers)
    {
        header.command_capacity = plan_capacity;
        header.participant_id = config.participant_id;
        header.participant_count = config.participant_count;
    }
    command_headers[0].epoch = 7;
    command_headers[0].command_count = 2;

    llaminar2::DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.next_epoch = 8;
    controller_state.active_wave = 0;
    controller_state.wave_count = command_buffer_count;
    auto &wave = controller_state.waves[0];
    wave.epoch = command_headers[0].epoch;
    wave.state = static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::ReadyToApply);
    wave.planned_start_layer = 0;
    wave.planned_layer_count = 2;
    wave.command_count = 2;
    wave.copied_arrivals = 2;
    wave.requested_payload_slots = 0;
    wave.payload_bucket_slots = 0;

    llaminar2::DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    llaminar2::DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_apply_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_entries),
                         plan_entries.size() * sizeof(plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_headers),
                         command_headers.size() * sizeof(command_headers[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_apply_status),
                         sizeof(llaminar2::DeviceMoERebalanceApplyStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_plan_entries,
                              plan_entries.data(),
                              plan_entries.size() * sizeof(plan_entries[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_command_headers,
                              command_headers.data(),
                              command_headers.size() * sizeof(command_headers[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_controller_state,
                              &controller_state,
                              sizeof(controller_state),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->decodeRouteSelectWithReadyRebalanceApply(
        runtime_table.deviceLayerState(0),
        runtime_table.deviceLayerState(1),
        hidden.get(),
        gate.get(),
        4,
        4,
        2,
        false,
        indices.get(),
        weights.get(),
        true,
        true,
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceGraphControllerState result_state;
    std::array<llaminar2::DeviceMoERebalanceCommandBufferHeader, command_buffer_count> result_headers{};
    ASSERT_EQ(cudaMemcpy(&result_state, d_controller_state, sizeof(result_state), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(result_headers.data(),
                         d_command_headers,
                         result_headers.size() * sizeof(result_headers[0]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(result_state.active_wave, 0u);
    EXPECT_EQ(result_state.waves[0].state,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::ReadyToApply));
    EXPECT_EQ(result_state.waves[0].applied_layer_count, 1u);
    EXPECT_EQ(result_headers[0].epoch, 7u);
    EXPECT_EQ(result_headers[0].command_count, 2u);

    ASSERT_TRUE(cuda_kernel_->decodeRouteSelectWithReadyRebalanceApply(
        runtime_table.deviceLayerState(0),
        runtime_table.deviceLayerState(1),
        hidden.get(),
        gate.get(),
        4,
        4,
        2,
        false,
        indices.get(),
        weights.get(),
        true,
        true,
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&result_state, d_controller_state, sizeof(result_state), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(result_headers.data(),
                         d_command_headers,
                         result_headers.size() * sizeof(result_headers[0]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(result_state.active_wave, 1u);
    EXPECT_EQ(result_state.waves[0].state,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::Applied));
    EXPECT_EQ(result_state.waves[0].applied_layer_count, 2u);
    EXPECT_EQ(result_headers[0].epoch, 0u);
    EXPECT_EQ(result_headers[0].command_count, 0u);

    cudaFree(d_plan_entries);
    cudaFree(d_command_headers);
    cudaFree(d_controller_state);
    cudaFree(d_apply_status);
#endif
}

TEST_F(Test__CUDAMoEKernel, DecodeReadyRebalanceApplyDoesNotAdvanceIncompleteTransferSlotArrival)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    auto update = makeDynamicOwnershipTransferUpdate(1, 1);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream_));

    llaminar2::DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 2;
    config.window_size_tokens = 1;
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::DeferRuntimeApply);

    constexpr uint32_t plan_capacity = 4;
    constexpr uint32_t command_buffer_count = 2;
    auto hidden = makeTensor({1, 4}, {1.0f, 0.0f, 0.0f, 0.0f});
    auto gate = makeTensor({4, 4}, {4.0f, 0.0f, 0.0f, 0.0f,
                                    3.0f, 0.0f, 0.0f, 0.0f,
                                    2.0f, 0.0f, 0.0f, 0.0f,
                                    1.0f, 0.0f, 0.0f, 0.0f});
    auto indices = makeZeros({1, 2});
    auto weights = makeZeros({1, 2});

    std::vector<llaminar2::DeviceMoERebalancePlanEntry> plan_entries(
        static_cast<size_t>(plan_capacity) * command_buffer_count);
    auto &plan = plan_entries[0];
    plan.op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::OwnershipTransfer);
    plan.layer = 0;
    plan.expert = 0;
    plan.source_participant = 0;
    plan.destination_participant = config.participant_id;
    plan.source_resident_mask = 0b01u;
    plan.destination_slot = 0;
    plan.payload_slot = 0;

    std::array<llaminar2::DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    for (auto &header : command_headers)
    {
        header.command_capacity = plan_capacity;
        header.participant_id = config.participant_id;
        header.participant_count = config.participant_count;
    }
    command_headers[0].epoch = 11;
    command_headers[0].command_count = 1;

    llaminar2::DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.next_epoch = 12;
    controller_state.active_wave = 0;
    controller_state.wave_count = command_buffer_count;
    auto &wave = controller_state.waves[0];
    wave.epoch = command_headers[0].epoch;
    wave.state = static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::ReadyToApply);
    wave.planned_start_layer = 0;
    wave.planned_layer_count = 1;
    wave.command_count = 1;
    wave.copied_arrivals = 0;
    wave.requested_payload_slots = 1;
    wave.payload_bucket_slots = 1;

    std::array<llaminar2::DeviceMoEExpertDirectoryEntry, 1> transfer_slots{};
    transfer_slots[0].participant = config.participant_id;
    transfer_slots[0].slot_index = 0;
    transfer_slots[0].flags =
        static_cast<uint32_t>(llaminar2::DeviceMoERebalanceDirectoryFlags::Valid) |
        static_cast<uint32_t>(llaminar2::DeviceMoERebalanceDirectoryFlags::TransferSlot);

    llaminar2::DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    llaminar2::DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_apply_status = nullptr;
    llaminar2::DeviceMoEExpertDirectoryEntry *d_transfer_slots = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_entries),
                         plan_entries.size() * sizeof(plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_headers),
                         command_headers.size() * sizeof(command_headers[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_apply_status),
                         sizeof(llaminar2::DeviceMoERebalanceApplyStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_transfer_slots),
                         transfer_slots.size() * sizeof(transfer_slots[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_plan_entries,
                              plan_entries.data(),
                              plan_entries.size() * sizeof(plan_entries[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_command_headers,
                              command_headers.data(),
                              command_headers.size() * sizeof(command_headers[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_controller_state,
                              &controller_state,
                              sizeof(controller_state),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_transfer_slots,
                              transfer_slots.data(),
                              transfer_slots.size() * sizeof(transfer_slots[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->decodeRouteSelectWithReadyRebalanceApply(
        runtime_table.deviceLayerState(0),
        runtime_table.deviceLayerState(0),
        hidden.get(),
        gate.get(),
        4,
        4,
        2,
        false,
        indices.get(),
        weights.get(),
        true,
        true,
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceApplyStatus apply_status;
    llaminar2::DeviceMoERebalanceGraphControllerState result_state;
    std::array<llaminar2::DeviceMoERebalanceCommandBufferHeader, command_buffer_count> result_headers{};
    llaminar2::DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(cudaMemcpy(&apply_status, d_apply_status, sizeof(apply_status), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&result_state, d_controller_state, sizeof(result_state), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(result_headers.data(),
                         d_command_headers,
                         result_headers.size() * sizeof(result_headers[0]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&runtime,
                         runtime_table.deviceLayerState(0),
                         sizeof(runtime),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_EQ(apply_status.copy_incomplete, 1u);
    EXPECT_EQ(apply_status.changed_layers, 0u);
    EXPECT_EQ(apply_status.applied_arrivals, 0u);
    EXPECT_EQ(result_state.active_wave, 0u);
    EXPECT_EQ(result_state.waves[0].state,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::ReadyToApply));
    EXPECT_EQ(result_state.waves[0].applied_layer_count, 0u);
    EXPECT_EQ(result_headers[0].epoch, 11u);
    EXPECT_EQ(result_headers[0].command_count, 1u);
    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(bank.local_compute_mask[0], 0u);
    EXPECT_EQ(bank.experts[0].owner_participant, 0);
    EXPECT_EQ(bank.resident_participant_mask[0], 0b01u);

    cudaFree(d_plan_entries);
    cudaFree(d_command_headers);
    cudaFree(d_controller_state);
    cudaFree(d_apply_status);
    cudaFree(d_transfer_slots);
#endif
}

TEST_F(Test__CUDAMoEKernel, DecodeReadyRebalanceApplyUsesOrderedLayerCursor)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 2;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    for (int layer = 0; layer < 2; ++layer)
    {
        auto update = makeParticipantOneBaseUpdate(1);
        ASSERT_TRUE(runtime_table.prepareInactiveBank(layer, update));
        ASSERT_TRUE(runtime_table.flipActiveBank(layer, update.epoch, stream_));
    }

    llaminar2::DeviceMoERebalanceConfig config;
    config.num_layers = 2;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::DeferRuntimeApply);

    constexpr uint32_t plan_capacity = 4;
    constexpr uint32_t command_buffer_count = 2;
    auto hidden = makeTensor({1, 4}, {1.0f, 0.0f, 0.0f, 0.0f});
    auto gate = makeTensor({4, 4}, {4.0f, 0.0f, 0.0f, 0.0f,
                                    3.0f, 0.0f, 0.0f, 0.0f,
                                    2.0f, 0.0f, 0.0f, 0.0f,
                                    1.0f, 0.0f, 0.0f, 0.0f});
    auto indices = makeZeros({1, 2});
    auto weights = makeZeros({1, 2});

    std::vector<llaminar2::DeviceMoERebalancePlanEntry> plan_entries(
        static_cast<size_t>(plan_capacity) * command_buffer_count);
    auto &plan = plan_entries[0];
    plan.op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ResidentExpertAssignment);
    plan.layer = 0;
    plan.expert = 0;
    plan.source_participant = 0;
    plan.destination_participant = config.participant_id;
    plan.source_resident_mask = 0b001u;
    plan.destination_slot = 0;
    plan.payload_slot = llaminar2::kDeviceMoEInvalidSlot;

    std::array<llaminar2::DeviceMoERebalanceCommandBufferHeader, command_buffer_count> command_headers{};
    for (auto &header : command_headers)
    {
        header.command_capacity = plan_capacity;
        header.participant_id = config.participant_id;
        header.participant_count = config.participant_count;
    }
    command_headers[0].epoch = 17;
    command_headers[0].command_count = 1;

    llaminar2::DeviceMoERebalanceGraphControllerState controller_state;
    controller_state.participant_id = config.participant_id;
    controller_state.participant_count = config.participant_count;
    controller_state.next_epoch = 18;
    controller_state.active_wave = 0;
    controller_state.wave_count = command_buffer_count;
    auto &wave = controller_state.waves[0];
    wave.epoch = command_headers[0].epoch;
    wave.state = static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::ReadyToApply);
    wave.planned_start_layer = 0;
    wave.planned_layer_count = 2;
    wave.command_count = 1;
    wave.copied_arrivals = 1;

    llaminar2::DeviceMoERebalancePlanEntry *d_plan_entries = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_headers = nullptr;
    llaminar2::DeviceMoERebalanceGraphControllerState *d_controller_state = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_apply_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_entries),
                         plan_entries.size() * sizeof(plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_headers),
                         command_headers.size() * sizeof(command_headers[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_controller_state), sizeof(controller_state)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_apply_status),
                         sizeof(llaminar2::DeviceMoERebalanceApplyStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_plan_entries,
                              plan_entries.data(),
                              plan_entries.size() * sizeof(plan_entries[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_command_headers,
                              command_headers.data(),
                              command_headers.size() * sizeof(command_headers[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_controller_state,
                              &controller_state,
                              sizeof(controller_state),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    auto run_route_apply = [&](int target_layer)
    {
        ASSERT_TRUE(cuda_kernel_->decodeRouteSelectWithReadyRebalanceApply(
            runtime_table.deviceLayerState(0),
            runtime_table.deviceLayerState(1),
            hidden.get(),
            gate.get(),
            4,
            4,
            2,
            false,
            indices.get(),
            weights.get(),
            true,
            true,
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
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    };

    llaminar2::DeviceMoERebalanceGraphControllerState result_state;
    std::array<llaminar2::DeviceMoERebalanceCommandBufferHeader, command_buffer_count> result_headers{};
    auto read_state = [&]()
    {
        ASSERT_EQ(cudaMemcpy(&result_state, d_controller_state, sizeof(result_state), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(result_headers.data(),
                             d_command_headers,
                             result_headers.size() * sizeof(result_headers[0]),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
    };

    run_route_apply(0);
    read_state();
    EXPECT_EQ(result_state.waves[0].state,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::ReadyToApply));
    EXPECT_EQ(result_state.waves[0].applied_layer_count, 1u);
    EXPECT_EQ(result_headers[0].command_count, 1u);

    run_route_apply(0);
    read_state();
    EXPECT_EQ(result_state.waves[0].state,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::ReadyToApply));
    EXPECT_EQ(result_state.waves[0].applied_layer_count, 1u)
        << "a previously applied layer must not consume the next rolling-wave slot";
    EXPECT_EQ(result_headers[0].command_count, 1u);

    run_route_apply(1);
    read_state();
    EXPECT_EQ(result_state.active_wave, 1u);
    EXPECT_EQ(result_state.waves[0].state,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceWaveLifecycle::Applied));
    EXPECT_EQ(result_state.waves[0].applied_layer_count, 2u);
    EXPECT_EQ(result_headers[0].command_count, 0u);

    cudaFree(d_plan_entries);
    cudaFree(d_command_headers);
    cudaFree(d_controller_state);
    cudaFree(d_apply_status);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalanceControllerAggregatesRouterBenefitAcrossLayerWindow)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 2;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    for (int layer = 0; layer < 2; ++layer)
    {
        auto update = makeParticipantOneBaseUpdate(1);
        ASSERT_TRUE(runtime_table.prepareInactiveBank(layer, update));
        ASSERT_TRUE(runtime_table.flipActiveBank(layer, update.epoch, stream_));
    }

    llaminar2::DeviceMoELayerRuntime seeded_layer{};
    ASSERT_EQ(cudaMemcpy(&seeded_layer,
                         runtime_table.deviceLayerState(1),
                         sizeof(seeded_layer),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
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
    ASSERT_EQ(cudaMemcpyAsync(runtime_table.deviceLayerState(1),
                              &seeded_layer,
                              sizeof(seeded_layer),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    llaminar2::DeviceMoERebalanceConfig config;
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
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::CollectLoadStats);

    std::vector<uint64_t> gathered(static_cast<size_t>(config.participant_count) *
                                       config.layer_wave_count * config.num_experts,
                                   0);
    gathered[0] = 2; // Keep the current one-layer wave ready while counters live on layer 1.
    uint64_t *d_gathered = nullptr;
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered),
                         gathered.size() * sizeof(uint64_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_status),
                         sizeof(llaminar2::DeviceMoERebalanceStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan),
                         sizeof(llaminar2::DeviceMoERebalancePlanEntry)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_count),
                         sizeof(uint32_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_header),
                         sizeof(llaminar2::DeviceMoERebalanceCommandBufferHeader)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_wave_state),
                         sizeof(llaminar2::DeviceMoERebalanceWaveState)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered,
                              gathered.data(),
                              gathered.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_status, 0, sizeof(llaminar2::DeviceMoERebalanceStatus), stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_plan_count, 0, sizeof(uint32_t), stream_), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_command_header,
                              0,
                              sizeof(llaminar2::DeviceMoERebalanceCommandBufferHeader),
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_wave_state,
                              0,
                              sizeof(llaminar2::DeviceMoERebalanceWaveState),
                              stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->runDeviceRebalanceController(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceStatus status;
    llaminar2::DeviceMoELayerRuntime result_layer{};
    ASSERT_EQ(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&result_layer,
                         runtime_table.deviceLayerState(1),
                         sizeof(result_layer),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(status.window_ready_slots, 2u);
    EXPECT_EQ(status.window_required_slots, 2u);
    EXPECT_EQ(status.skipped_busy_wave, 0u);
    EXPECT_EQ(status.skipped_not_ready, 0u);
    ASSERT_EQ(status.status_code, static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::Ok));
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

    cudaFree(d_gathered);
    cudaFree(d_status);
    cudaFree(d_plan);
    cudaFree(d_plan_count);
    cudaFree(d_command_header);
    cudaFree(d_wave_state);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalanceControllerSkipsMissingArrivalThatWorsensImbalance)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    update.resident_participant_mask[0] = 0b001u;
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream_));

    llaminar2::DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::PlanMissingArrivals);
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::CollectLoadStats);

    std::vector<uint64_t> gathered(static_cast<size_t>(config.participant_count) *
                                       config.num_layers * config.num_experts,
                                   0);
    gathered[0] = 6; // participant 0, layer 0, expert 0.
    gathered[config.num_experts + 2] = 10; // participant 1 already carries a heavier local expert.

    uint64_t *d_gathered = nullptr;
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered),
                         gathered.size() * sizeof(uint64_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_status),
                         sizeof(llaminar2::DeviceMoERebalanceStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan),
                         sizeof(llaminar2::DeviceMoERebalancePlanEntry)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_count),
                         sizeof(uint32_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_header),
                         sizeof(llaminar2::DeviceMoERebalanceCommandBufferHeader)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_wave_state),
                         sizeof(llaminar2::DeviceMoERebalanceWaveState)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered, gathered.data(),
                              gathered.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->runDeviceRebalanceController(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceStatus status;
    llaminar2::DeviceMoERebalanceCommandBufferHeader command_header;
    uint32_t plan_count = 99;
    llaminar2::DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&command_header, d_command_header, sizeof(command_header), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&plan_count, d_plan_count, sizeof(plan_count), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::Ok));
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
    EXPECT_FALSE(llaminar2::hasMoEExpertFlag(bank.experts[0].flags,
                                             llaminar2::DeviceMoEExpertFlags::Replicated));

    cudaFree(d_gathered);
    cudaFree(d_status);
    cudaFree(d_plan);
    cudaFree(d_plan_count);
    cudaFree(d_command_header);
    cudaFree(d_wave_state);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalanceControllerPrunesMissingArrivalBelowCountBound)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    update.resident_participant_mask[0] = 0b001u;
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream_));

    llaminar2::DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;
    config.min_load_spread_improvement = 8;
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::PlanMissingArrivals);
    config.flags |= static_cast<uint32_t>(llaminar2::DeviceMoERebalanceFlags::CollectLoadStats);

    std::vector<uint64_t> gathered(static_cast<size_t>(config.participant_count) *
                                       config.num_layers * config.num_experts,
                                   0);
    gathered[0] = 6; // Window is ready, but this count cannot meet the configured floor.

    uint64_t *d_gathered = nullptr;
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    llaminar2::DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered),
                         gathered.size() * sizeof(uint64_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_status),
                         sizeof(llaminar2::DeviceMoERebalanceStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan),
                         sizeof(llaminar2::DeviceMoERebalancePlanEntry)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_count),
                         sizeof(uint32_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_header),
                         sizeof(llaminar2::DeviceMoERebalanceCommandBufferHeader)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_wave_state),
                         sizeof(llaminar2::DeviceMoERebalanceWaveState)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered, gathered.data(),
                              gathered.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->runDeviceRebalanceController(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceStatus status;
    llaminar2::DeviceMoERebalanceCommandBufferHeader command_header;
    uint32_t plan_count = 99;
    ASSERT_EQ(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&command_header, d_command_header, sizeof(command_header), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&plan_count, d_plan_count, sizeof(plan_count), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::Ok));
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

    cudaFree(d_gathered);
    cudaFree(d_status);
    cudaFree(d_plan);
    cudaFree(d_plan_count);
    cudaFree(d_command_header);
    cudaFree(d_wave_state);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalancePackDirectoryExportsOnlyLocalResidents)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    update.resident_participant_mask[0] = 0b001u;
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream_));

    llaminar2::DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 1;
    config.participant_count = 3;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;

    llaminar2::DeviceMoEExpertDirectoryEntry *d_directory = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_directory),
                         config.num_experts * sizeof(llaminar2::DeviceMoEExpertDirectoryEntry)),
              cudaSuccess);
    ASSERT_TRUE(cuda_kernel_->packDeviceRebalanceDirectory(
        runtime_table.deviceLayerState(0),
        d_directory,
        config));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    std::vector<llaminar2::DeviceMoEExpertDirectoryEntry> directory(config.num_experts);
    ASSERT_EQ(cudaMemcpy(directory.data(), d_directory,
                         directory.size() * sizeof(directory[0]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_FALSE(llaminar2::deviceMoEDirectoryEntryReady(directory[0], 1, 0, 0));
    ASSERT_TRUE(llaminar2::deviceMoEDirectoryEntryReady(directory[2], 1, 0, 2));
    EXPECT_EQ(directory[2].descriptor.logical_expert_id, 2);
    EXPECT_EQ(directory[2].slot_index, 2u);
    EXPECT_TRUE((directory[2].flags &
                 static_cast<uint32_t>(llaminar2::DeviceMoERebalanceDirectoryFlags::LocalCompute)) != 0u);
    EXPECT_FALSE(llaminar2::deviceMoEDirectoryEntryReady(directory[3], 1, 0, 3));

    cudaFree(d_directory);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalanceUnpackPreservesSourceSidePackErrors)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    llaminar2::DeviceMoERebalanceConfig config;
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

    llaminar2::DeviceMoERebalancePlanEntry plan;
    plan.op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    plan.layer = 0;
    plan.expert = 0;
    plan.source_participant = 1;
    plan.destination_participant = 0;
    plan.source_resident_mask = 0b10u;
    plan.destination_slot = 0;
    plan.payload_slot = 0;

    llaminar2::DeviceMoERebalanceCommandBufferHeader command_header;
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

    llaminar2::DeviceMoERebalancePlanEntry *d_plan = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    llaminar2::DeviceMoEExpertDirectoryEntry *d_local_source_descriptors = nullptr;
    uint8_t *d_local_payload = nullptr;
    uint8_t *d_gathered_payload = nullptr;
    llaminar2::DeviceMoEExpertDirectoryEntry *d_transfer_slots = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_copy_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan), sizeof(plan)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_header), sizeof(command_header)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_source_descriptors),
                         source_descriptor_count * sizeof(llaminar2::DeviceMoEExpertDirectoryEntry)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_payload), local_payload_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_payload), gathered_payload_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_transfer_slots),
                         sizeof(llaminar2::DeviceMoEExpertDirectoryEntry)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_copy_status),
                         sizeof(llaminar2::DeviceMoERebalanceApplyStatus)),
              cudaSuccess);

    ASSERT_EQ(cudaMemcpyAsync(d_plan, &plan, sizeof(plan), cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_command_header,
                              &command_header,
                              sizeof(command_header),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_local_source_descriptors,
                              0,
                              source_descriptor_count * sizeof(llaminar2::DeviceMoEExpertDirectoryEntry),
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_local_payload, 0, local_payload_bytes, stream_), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_gathered_payload, 0, gathered_payload_bytes, stream_), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_transfer_slots,
                              0,
                              sizeof(llaminar2::DeviceMoEExpertDirectoryEntry),
                              stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->packDeviceRebalanceCompactPayloads(
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
    ASSERT_TRUE(cuda_kernel_->unpackDeviceRebalanceCollectivePayloads(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceApplyStatus copy_status{};
    ASSERT_EQ(cudaMemcpy(&copy_status, d_copy_status, sizeof(copy_status), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(copy_status.status_code,
              static_cast<uint32_t>(llaminar2::DeviceMoERebalanceApplyStatusCode::Ok));
    EXPECT_EQ(copy_status.missing_source_descriptors, 1u)
        << "unpack must not clear source-side compact pack failures from the same transfer wave";
    EXPECT_EQ(copy_status.copied_arrivals, 0u);
    EXPECT_EQ(copy_status.skipped_wrong_destination, 1u);

    cudaFree(d_plan);
    cudaFree(d_command_header);
    cudaFree(d_local_source_descriptors);
    cudaFree(d_local_payload);
    cudaFree(d_gathered_payload);
    cudaFree(d_transfer_slots);
    cudaFree(d_copy_status);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalanceCompactPayloadRejectsResidentSourceWithoutLocalSlot)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    llaminar2::DeviceMoERebalanceConfig config;
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
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_src_payload), projection_payload_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_src_scales, projection_scales_bytes), cudaSuccess);

    llaminar2::DeviceNativeVNNIMatrixDesc projection_desc;
    projection_desc.payload = d_src_payload;
    projection_desc.scales = d_src_scales;
    projection_desc.n = n;
    projection_desc.k = k;
    projection_desc.blocks_per_row = blocks_per_row;
    projection_desc.codebook_id = 0;

    llaminar2::DeviceMoEExpertDirectoryEntry source_entry;
    source_entry.descriptor.gate = projection_desc;
    source_entry.descriptor.up = projection_desc;
    source_entry.descriptor.down = projection_desc;
    source_entry.descriptor.logical_expert_id = 0;
    source_entry.descriptor.owner_participant = 1;
    source_entry.descriptor.local_slot = -1;
    source_entry.descriptor.flags =
        llaminar2::toMoEExpertFlags(llaminar2::DeviceMoEExpertFlags::Valid |
                                    llaminar2::DeviceMoEExpertFlags::Resident);
    source_entry.layer = 0;
    source_entry.expert = 0;
    source_entry.participant = 1;
    source_entry.resident_mask = 0b10u;
    source_entry.slot_index = llaminar2::kDeviceMoEInvalidSlot;
    source_entry.flags =
        static_cast<uint32_t>(llaminar2::DeviceMoERebalanceDirectoryFlags::Valid) |
        static_cast<uint32_t>(llaminar2::DeviceMoERebalanceDirectoryFlags::Resident);
    ASSERT_TRUE(llaminar2::deviceMoEPopulateDirectoryFormat(source_entry));

    const size_t source_descriptor_count =
        static_cast<size_t>(config.participant_count) *
        static_cast<size_t>(command_buffer_count) *
        static_cast<size_t>(plan_capacity);
    std::vector<llaminar2::DeviceMoEExpertDirectoryEntry> local_source_descriptors(
        source_descriptor_count);
    local_source_descriptors[0] = source_entry;

    llaminar2::DeviceMoERebalancePlanEntry plan;
    plan.op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    plan.layer = 0;
    plan.expert = 0;
    plan.source_participant = 1;
    plan.destination_participant = 0;
    plan.source_resident_mask = 0b10u;
    plan.destination_slot = 0;
    plan.payload_slot = 0;

    llaminar2::DeviceMoERebalanceCommandBufferHeader command_header;
    command_header.epoch = 12;
    command_header.command_count = 1;
    command_header.command_capacity = plan_capacity;
    command_header.participant_id = config.participant_id;
    command_header.participant_count = config.participant_count;

    const size_t local_payload_bytes =
        static_cast<size_t>(local_payload_slot_count) *
        static_cast<size_t>(payload_slot_bytes);
    llaminar2::DeviceMoERebalancePlanEntry *d_plan = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    llaminar2::DeviceMoEExpertDirectoryEntry *d_local_source_descriptors = nullptr;
    uint8_t *d_local_payload = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_copy_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan), sizeof(plan)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_header), sizeof(command_header)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_source_descriptors),
                         local_source_descriptors.size() * sizeof(local_source_descriptors[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_payload), local_payload_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_copy_status),
                         sizeof(llaminar2::DeviceMoERebalanceApplyStatus)),
              cudaSuccess);

    ASSERT_EQ(cudaMemcpyAsync(d_plan, &plan, sizeof(plan), cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_command_header,
                              &command_header,
                              sizeof(command_header),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_local_source_descriptors,
                              local_source_descriptors.data(),
                              local_source_descriptors.size() * sizeof(local_source_descriptors[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_local_payload, 0, local_payload_bytes, stream_), cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->packDeviceRebalanceCompactPayloads(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceApplyStatus copy_status{};
    llaminar2::DeviceMoEExpertDirectoryEntry payload_header{};
    ASSERT_EQ(cudaMemcpy(&copy_status, d_copy_status, sizeof(copy_status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&payload_header,
                         d_local_payload,
                         sizeof(payload_header),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(copy_status.missing_source_descriptors, 1u)
        << "a resident mask without a local slot is not a physical source for payload bytes";
    EXPECT_EQ(copy_status.invalid_plan_entries, 0u);
    EXPECT_EQ(payload_header.flags, 0u);

    cudaFree(d_plan);
    cudaFree(d_command_header);
    cudaFree(d_local_source_descriptors);
    cudaFree(d_local_payload);
    cudaFree(d_copy_status);
    cudaFree(d_src_payload);
    cudaFree(d_src_scales);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalanceCopyAndApplyArrivalUsesTransferSlot)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    update.resident_participant_mask[0] = 0b001u;
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream_));

    llaminar2::DeviceMoERebalanceConfig config;
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

        ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&projection.src_payload), payload_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&projection.src_scales, scales_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&projection.dst_payload), payload_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&projection.dst_scales, scales_bytes), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(projection.src_payload,
                                  projection.expected_payload.data(),
                                  payload_bytes,
                                  cudaMemcpyHostToDevice,
                                  stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(projection.src_scales,
                                  projection.expected_scales.data(),
                                  scales_bytes,
                                  cudaMemcpyHostToDevice,
                                  stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(projection.dst_payload, 0, payload_bytes, stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(projection.dst_scales, 0, scales_bytes, stream_), cudaSuccess);
    }

    auto matrix_desc = [](const ProjectionBuffers &projection, bool destination)
    {
        llaminar2::DeviceNativeVNNIMatrixDesc desc;
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
        llaminar2::DeviceMoEExpertDescriptor desc;
        desc.gate = matrix_desc(projections[0], destination);
        desc.up = matrix_desc(projections[1], destination);
        desc.down = matrix_desc(projections[2], destination);
        desc.logical_expert_id = 0;
        desc.owner_participant = 0;
        desc.local_slot = local_slot;
        desc.flags = llaminar2::toMoEExpertFlags(llaminar2::DeviceMoEExpertFlags::Valid |
                                                 llaminar2::DeviceMoEExpertFlags::Resident |
                                                 llaminar2::DeviceMoEExpertFlags::LocalCompute);
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
    std::vector<llaminar2::DeviceMoEExpertDirectoryEntry> local_source_descriptors(
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
        static_cast<uint32_t>(llaminar2::DeviceMoERebalanceDirectoryFlags::Valid) |
        static_cast<uint32_t>(llaminar2::DeviceMoERebalanceDirectoryFlags::Resident) |
        static_cast<uint32_t>(llaminar2::DeviceMoERebalanceDirectoryFlags::LocalCompute);
    ASSERT_TRUE(llaminar2::deviceMoEPopulateDirectoryFormat(source_entry));

    std::vector<llaminar2::DeviceMoEExpertDirectoryEntry> transfer_slots(1);
    auto &slot = transfer_slots[0];
    slot.descriptor = make_descriptor(true, 7);
    slot.descriptor.logical_expert_id = -1;
    slot.layer = llaminar2::kDeviceMoEInvalidSlot;
    slot.expert = llaminar2::kDeviceMoEInvalidSlot;
    slot.participant = 1;
    slot.resident_mask = 0;
    slot.slot_index = 7;
    slot.flags =
        static_cast<uint32_t>(llaminar2::DeviceMoERebalanceDirectoryFlags::Valid) |
        static_cast<uint32_t>(llaminar2::DeviceMoERebalanceDirectoryFlags::TransferSlot);
    ASSERT_TRUE(llaminar2::deviceMoEPopulateDirectoryFormat(slot));

    llaminar2::DeviceMoERebalancePlanEntry plan;
    plan.op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    plan.layer = 0;
    plan.expert = 0;
    plan.source_participant = 0;
    plan.destination_participant = 1;
    plan.source_resident_mask = 0b001u;
    plan.destination_slot = 0;
    plan.payload_slot = 0;
    std::vector<llaminar2::DeviceMoERebalancePlanEntry> plan_entries(plan_capacity);
    plan_entries[0] = plan;
    const uint32_t legacy_plan_count = 0;
    llaminar2::DeviceMoERebalanceCommandBufferHeader source_command_header;
    source_command_header.epoch = 2;
    source_command_header.phase =
        static_cast<uint32_t>(llaminar2::DeviceMoERebalancePipelinePhase::PlanAssignments);
    source_command_header.command_count = 1;
    source_command_header.command_capacity = plan_capacity;
    source_command_header.participant_id = 0;
    source_command_header.participant_count = 3;
    auto destination_command_header = source_command_header;
    destination_command_header.participant_id = 1;

    llaminar2::DeviceMoERebalanceConfig source_config = config;
    source_config.participant_id = 0;
    constexpr uint64_t payload_slot_bytes = 512;
    const size_t local_payload_slot_count = 1;
    const size_t local_payload_bytes =
        local_payload_slot_count * static_cast<size_t>(payload_slot_bytes);
    const size_t gathered_payload_bytes =
        static_cast<size_t>(config.participant_count) * local_payload_bytes;

    llaminar2::DeviceMoEExpertDirectoryEntry *d_local_source_descriptors = nullptr;
    uint8_t *d_local_payload = nullptr;
    uint8_t *d_gathered_payload = nullptr;
    llaminar2::DeviceMoEExpertDirectoryEntry *d_transfer_slots = nullptr;
    llaminar2::DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    llaminar2::DeviceMoERebalanceCommandBufferHeader *d_command_header = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_copy_status = nullptr;
    llaminar2::DeviceMoERebalanceApplyStatus *d_apply_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_source_descriptors),
                         local_source_descriptors.size() * sizeof(local_source_descriptors[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local_payload), local_payload_bytes),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered_payload), gathered_payload_bytes),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_transfer_slots),
                         transfer_slots.size() * sizeof(transfer_slots[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan),
                         plan_entries.size() * sizeof(plan_entries[0])),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_plan_count), sizeof(legacy_plan_count)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_command_header), sizeof(source_command_header)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_copy_status),
                         sizeof(llaminar2::DeviceMoERebalanceApplyStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_apply_status),
                         sizeof(llaminar2::DeviceMoERebalanceApplyStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_local_source_descriptors,
                              local_source_descriptors.data(),
                              local_source_descriptors.size() * sizeof(local_source_descriptors[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_local_payload, 0, local_payload_bytes, stream_), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_gathered_payload, 0, gathered_payload_bytes, stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_transfer_slots,
                              transfer_slots.data(),
                              transfer_slots.size() * sizeof(transfer_slots[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_plan,
                              plan_entries.data(),
                              plan_entries.size() * sizeof(plan_entries[0]),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_plan_count, &legacy_plan_count, sizeof(legacy_plan_count), cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_command_header,
                              &source_command_header,
                              sizeof(source_command_header),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->packDeviceRebalanceCompactPayloads(
        d_plan,
        d_command_header,
        plan_capacity,
        d_local_source_descriptors,
        d_local_payload,
        static_cast<uint32_t>(local_payload_slot_count),
        payload_slot_bytes,
        source_config,
        d_copy_status));
    ASSERT_EQ(cudaMemcpyAsync(d_gathered_payload,
                              d_local_payload,
                              local_payload_bytes,
                              cudaMemcpyDeviceToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_command_header,
                              &destination_command_header,
                              sizeof(destination_command_header),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_TRUE(cuda_kernel_->unpackDeviceRebalanceCollectivePayloads(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceApplyStatus copy_status;
    ASSERT_EQ(cudaMemcpy(&copy_status, d_copy_status, sizeof(copy_status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(transfer_slots.data(), d_transfer_slots,
                         transfer_slots.size() * sizeof(transfer_slots[0]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(copy_status.copied_arrivals, 1u);
    EXPECT_TRUE(llaminar2::deviceMoETransferSlotCopyComplete(transfer_slots[0], 1, 0, 0));
    EXPECT_EQ(transfer_slots[0].layer, 0u);
    EXPECT_EQ(transfer_slots[0].expert, 0u);
    EXPECT_EQ(transfer_slots[0].descriptor.logical_expert_id, 0);

    llaminar2::DeviceMoELayerRuntime seeded_runtime{};
    ASSERT_EQ(cudaMemcpy(&seeded_runtime,
                         runtime_table.deviceLayerState(0),
                         sizeof(seeded_runtime),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
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
    ASSERT_EQ(cudaMemcpyAsync(runtime_table.deviceLayerState(0),
                              &seeded_runtime,
                              sizeof(seeded_runtime),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->applyDeviceRebalanceArrivals(
        runtime_table.deviceLayerState(0),
        d_plan,
        d_plan_count,
        plan_capacity,
        d_transfer_slots,
        static_cast<uint32_t>(transfer_slots.size()),
        config,
        d_apply_status,
        d_command_header));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceApplyStatus apply_status;
    llaminar2::DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(cudaMemcpy(&apply_status, d_apply_status, sizeof(apply_status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(apply_status.applied_arrivals, 1u);
    EXPECT_EQ(apply_status.changed_layers, 1u);
    EXPECT_EQ(apply_status.post_apply_multi_resident_experts, 1u);
    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(bank.local_compute_mask[0], 1u);
    EXPECT_EQ(bank.replica_role[0], static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::Replica));
    EXPECT_EQ(bank.resident_participant_mask[0], 0b011u);
    EXPECT_EQ(bank.reserved[0], 1u)
        << "transfer-slot arrivals must refresh the multi-resident expert count used by decode routing";
    EXPECT_EQ(bank.experts[0].local_slot, 7);
    EXPECT_TRUE(llaminar2::hasMoEExpertFlag(bank.experts[0].flags,
                                            llaminar2::DeviceMoEExpertFlags::Replicated));
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

    auto hidden = makeTensor({1, 4}, {1.0f, 0.0f, 0.0f, 0.0f});
    auto gate_weights = makeTensor({4, 4}, {
                                           4.0f, 0.0f, 0.0f, 0.0f,
                                           3.0f, 0.0f, 0.0f, 0.0f,
                                           2.0f, 0.0f, 0.0f, 0.0f,
                                           1.0f, 0.0f, 0.0f, 0.0f});
    auto output_indices = makeZeros({1, 2});
    auto output_weights = makeZeros({1, 2});
    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        runtime_table.deviceLayerState(0),
        hidden.get(), gate_weights.get(),
        /*d_model=*/4,
        /*num_experts=*/4,
        /*top_k=*/2,
        /*normalize_weights=*/false,
        output_indices.get(), output_weights.get(),
        /*write_legacy_outputs=*/true,
        /*update_runtime_histogram=*/true));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime), cudaMemcpyDeviceToHost), cudaSuccess);
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

    llaminar2::DeviceMoERebalancePlanEntry resident_plan;
    resident_plan.op = static_cast<uint32_t>(llaminar2::DeviceMoERebalancePlanOp::ResidentExpertAssignment);
    resident_plan.layer = 0;
    resident_plan.expert = 2;
    resident_plan.source_participant = 1;
    resident_plan.destination_participant = 1;
    resident_plan.source_resident_mask = 0b010u;
    resident_plan.destination_slot = 0;
    resident_plan.payload_slot = 0;
    destination_command_header.epoch = 3;
    destination_command_header.command_count = 1;
    ASSERT_EQ(cudaMemcpyAsync(d_plan,
                              &resident_plan,
                              sizeof(resident_plan),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_command_header,
                              &destination_command_header,
                              sizeof(destination_command_header),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_TRUE(cuda_kernel_->applyDeviceRebalanceArrivals(
        runtime_table.deviceLayerState(0),
        d_plan,
        d_plan_count,
        plan_capacity,
        d_transfer_slots,
        static_cast<uint32_t>(transfer_slots.size()),
        config,
        d_apply_status,
        d_command_header));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&apply_status, d_apply_status, sizeof(apply_status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime), cudaMemcpyDeviceToHost), cudaSuccess);
    const auto &rebuilt_bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(apply_status.applied_arrivals, 1u);
    EXPECT_EQ(apply_status.changed_layers, 1u);
    EXPECT_EQ(apply_status.post_apply_multi_resident_experts, 1u);
    EXPECT_EQ(rebuilt_bank.local_compute_mask[0], 1u)
        << "rebalance apply must preserve already-arrived hot replicas when another plan rebuilds the layer bank";
    EXPECT_EQ(rebuilt_bank.replica_role[0], static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::Replica));
    EXPECT_EQ(rebuilt_bank.resident_participant_mask[0], 0b011u);
    EXPECT_TRUE(llaminar2::hasMoEExpertFlag(rebuilt_bank.experts[0].flags,
                                            llaminar2::DeviceMoEExpertFlags::Replicated));
    EXPECT_EQ(rebuilt_bank.reserved[0], 1u);

    destination_command_header.epoch = 4;
    destination_command_header.command_count = 1;
    ASSERT_EQ(cudaMemcpyAsync(d_plan,
                              &plan,
                              sizeof(plan),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_command_header,
                              &destination_command_header,
                              sizeof(destination_command_header),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    for (const auto &projection : projections)
    {
        std::vector<uint8_t> actual_payload(payload_bytes);
        std::vector<uint16_t> actual_scales(blocks);
        ASSERT_EQ(cudaMemcpy(actual_payload.data(),
                             projection.dst_payload,
                             payload_bytes,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(actual_scales.data(),
                             projection.dst_scales,
                             scales_bytes,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(actual_payload, projection.expected_payload);
        EXPECT_EQ(actual_scales, projection.expected_scales);
    }

    ASSERT_EQ(cudaMemsetAsync(d_gathered_payload,
                              0,
                              gathered_payload_bytes,
                              stream_),
              cudaSuccess);
    const auto clean_copy_status = makeCleanRebalanceApplyStatus();
    ASSERT_EQ(cudaMemcpyAsync(d_copy_status,
                              &clean_copy_status,
                              sizeof(clean_copy_status),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_TRUE(cuda_kernel_->unpackDeviceRebalanceCollectivePayloads(
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
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&copy_status, d_copy_status, sizeof(copy_status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(transfer_slots.data(), d_transfer_slots,
                         transfer_slots.size() * sizeof(transfer_slots[0]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(copy_status.copied_arrivals, 0u);
    EXPECT_EQ(copy_status.missing_source_descriptors, 1u);
    EXPECT_FALSE(llaminar2::deviceMoETransferSlotCopyComplete(transfer_slots[0], 1, 0, 0))
        << "a failed replay must clear stale CopyComplete metadata before apply";

    ASSERT_TRUE(cuda_kernel_->applyDeviceRebalanceArrivals(
        runtime_table.deviceLayerState(0),
        d_plan,
        d_plan_count,
        plan_capacity,
        d_transfer_slots,
        static_cast<uint32_t>(transfer_slots.size()),
        config,
        d_apply_status,
        d_command_header));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&apply_status, d_apply_status, sizeof(apply_status), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(apply_status.applied_arrivals, 0u);
    EXPECT_EQ(apply_status.copy_incomplete, 1u);
    EXPECT_EQ(apply_status.post_apply_multi_resident_experts, 0u);

    DeviceMoERuntimeTable source_runtime_table(table_config);
    auto source_update = makeParticipantOneBaseUpdate(10);
    source_update.participant_id = 0;
    source_update.local_compute_mask = {1, 0, 0, 0};
    source_update.replica_role = {
        static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::Primary),
        static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::None),
        static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::None),
        static_cast<uint8_t>(llaminar2::DeviceMoEReplicaRole::None),
    };
    source_update.resident_participant_mask = {0b001u, 0b001u, 0b010u, 0b100u};
    source_update.experts[0].owner_participant = 0;
    source_update.experts[0].flags =
        llaminar2::toMoEExpertFlags(llaminar2::DeviceMoEExpertFlags::Valid |
                                    llaminar2::DeviceMoEExpertFlags::Resident |
                                    llaminar2::DeviceMoEExpertFlags::LocalCompute);
    ASSERT_TRUE(source_runtime_table.prepareInactiveBank(0, source_update));
    ASSERT_TRUE(source_runtime_table.flipActiveBank(0, source_update.epoch, stream_));

    source_command_header.epoch = 5;
    source_command_header.command_count = 1;
    source_command_header.participant_id = 0;
    ASSERT_EQ(cudaMemcpyAsync(d_command_header,
                              &source_command_header,
                              sizeof(source_command_header),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_TRUE(cuda_kernel_->applyDeviceRebalanceArrivals(
        source_runtime_table.deviceLayerState(0),
        d_plan,
        d_plan_count,
        plan_capacity,
        d_transfer_slots,
        static_cast<uint32_t>(transfer_slots.size()),
        source_config,
        d_apply_status,
        d_command_header));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    llaminar2::DeviceMoELayerRuntime source_runtime_state{};
    ASSERT_EQ(cudaMemcpy(&apply_status, d_apply_status, sizeof(apply_status), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&source_runtime_state,
                         source_runtime_table.deviceLayerState(0),
                         sizeof(source_runtime_state),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(apply_status.applied_arrivals, 0u);
    EXPECT_EQ(apply_status.changed_layers, 1u)
        << "source participants must learn destination residency so replica dispatch stays domain-consistent";
    EXPECT_EQ(apply_status.post_apply_multi_resident_experts, 1u);
    const auto &source_bank = source_runtime_state.banks[source_runtime_state.active_bank];
    EXPECT_EQ(source_bank.resident_participant_mask[0], 0b011u)
        << "payload arrivals are data-local on the destination but residency metadata is domain-wide";
    EXPECT_EQ(source_bank.local_compute_mask[0], 1u);
    EXPECT_TRUE(llaminar2::hasMoEExpertFlag(source_bank.experts[0].flags,
                                            llaminar2::DeviceMoEExpertFlags::Replicated));
    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        source_runtime_table.deviceLayerState(0),
        hidden.get(), gate_weights.get(),
        /*d_model=*/4,
        /*num_experts=*/4,
        /*top_k=*/2,
        /*normalize_weights=*/false,
        output_indices.get(), output_weights.get(),
        /*write_legacy_outputs=*/true,
        /*update_runtime_histogram=*/true));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&source_runtime_state,
                         source_runtime_table.deviceLayerState(0),
                         sizeof(source_runtime_state),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(source_runtime_state.topk_expert_ids[0], -1)
        << "source and destination must make the same resident-copy assignment for replicated expert 0";
    EXPECT_EQ(source_runtime_state.topk_expert_ids[1], 1)
        << "source should still compute its non-replicated local expert";

    cudaFree(d_local_source_descriptors);
    cudaFree(d_local_payload);
    cudaFree(d_gathered_payload);
    cudaFree(d_transfer_slots);
    cudaFree(d_plan);
    cudaFree(d_plan_count);
    cudaFree(d_command_header);
    cudaFree(d_copy_status);
    cudaFree(d_apply_status);
    for (auto &projection : projections)
    {
        cudaFree(projection.src_payload);
        cudaFree(projection.src_scales);
        cudaFree(projection.dst_payload);
        cudaFree(projection.dst_scales);
    }
#endif
}

TEST_F(Test__CUDAMoEKernel, DeviceRebalancePackHistogramsFeedsController)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = llaminar2::DeviceId::cuda(0);
    table_config.num_layers = 1;
    table_config.num_experts = 4;
    table_config.top_k = 2;
    table_config.mirror_to_device = true;
    DeviceMoERuntimeTable runtime_table(table_config);

    auto update = makeParticipantOneBaseUpdate(1);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream_));

    llaminar2::DeviceMoERebalanceConfig config;
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
    llaminar2::DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_local),
                         local_entries * sizeof(uint64_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_gathered),
                         gathered_entries * sizeof(uint64_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_status),
                         sizeof(llaminar2::DeviceMoERebalanceStatus)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(runtime_table.deviceLayerState(0)->decode_histogram,
                              seeded_global_counts,
                              sizeof(seeded_global_counts),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(runtime_table.deviceLayerState(0)->decode_local_histogram,
                              seeded_local_counts,
                              sizeof(seeded_local_counts),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_TRUE(cuda_kernel_->packDeviceRebalanceHistograms(
        runtime_table.deviceLayerState(0), d_local, config));

    uint64_t packed_counts[local_entries] = {};
    ASSERT_EQ(cudaMemcpyAsync(packed_counts, d_local, sizeof(packed_counts),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    EXPECT_EQ(packed_counts[0], 8u);
    EXPECT_EQ(packed_counts[1], 0u);
    EXPECT_EQ(packed_counts[2], 0u);
    EXPECT_EQ(packed_counts[3], 0u);

    ASSERT_EQ(cudaMemsetAsync(d_gathered, 0,
                              gathered_entries * sizeof(uint64_t),
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_gathered + config.participant_id * local_entries,
                              d_local,
                              local_entries * sizeof(uint64_t),
                              cudaMemcpyDeviceToDevice,
                              stream_),
              cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->runDeviceRebalanceController(
        runtime_table.deviceLayerState(0), d_gathered, d_status, config));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoERebalanceStatus status;
    ASSERT_EQ(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost), cudaSuccess);
    llaminar2::DeviceMoELayerRuntime runtime{};
    ASSERT_EQ(cudaMemcpy(&runtime, runtime_table.deviceLayerState(0), sizeof(runtime),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_EQ(status.status_code, static_cast<uint32_t>(llaminar2::DeviceMoERebalanceStatusCode::Ok));
    EXPECT_EQ(status.windows_applied, 1u);
    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(bank.local_compute_mask[0], 1u);
    EXPECT_TRUE(llaminar2::hasMoEExpertFlag(bank.experts[0].flags,
                                            llaminar2::DeviceMoEExpertFlags::Replicated));
    for (int expert = 0; expert < 4; ++expert)
    {
        EXPECT_EQ(runtime.decode_histogram[expert], 0u);
        EXPECT_EQ(runtime.decode_local_histogram[expert], 0u);
    }

    cudaFree(d_local);
    cudaFree(d_gathered);
    cudaFree(d_status);
#endif
}

TEST_F(Test__CUDAMoEKernel, DecodeRouteSelectBF16GateMatchesCPU)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    constexpr int num_layers = 1;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;
    constexpr int d_model = 2048;
    constexpr int seq_len = 1;

    DeviceMoERuntimeTable::Config cuda_config;
    cuda_config.device_id = llaminar2::DeviceId::cuda(0);
    cuda_config.num_layers = num_layers;
    cuda_config.num_experts = num_experts;
    cuda_config.top_k = top_k;
    cuda_config.mirror_to_device = true;
    DeviceMoERuntimeTable cuda_table(cuda_config);

    llaminar2::MoEPlacementUpdate update;
    update.epoch = 1;
    update.expert_count = num_experts;
    update.experts.resize(num_experts);
    markAllExpertsLocalForRouting(update, num_experts);

    ASSERT_TRUE(cuda_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(cuda_table.flipActiveBank(0, update.epoch, stream_));

    std::vector<float> hidden_values(static_cast<size_t>(d_model));
    std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model);
    for (int i = 0; i < d_model; ++i)
        hidden_values[static_cast<size_t>(i)] =
            0.07f * std::sin(static_cast<float>(i + 3) * 0.013f);
    for (int i = 0; i < num_experts * d_model; ++i)
        gate_values[static_cast<size_t>(i)] =
            0.04f * std::cos(static_cast<float>(i + 11) * 0.007f) +
            0.00002f * static_cast<float>(i % num_experts);

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto gate_bf16 = makeBF16Tensor({num_experts, d_model}, gate_values);
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        cuda_table.deviceLayerState(0), hidden.get(), gate_bf16.get(), d_model, num_experts, top_k,
        true, cuda_indices.get(), cuda_weights.get(), true, true));
    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate_bf16.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    expectNearArray(cuda_indices->data(), cpu_indices->data(), seq_len * top_k, 0.0f);
    expectNearArray(cuda_weights->data(), cpu_weights->data(), seq_len * top_k, 1.0e-4f);
#endif
}

TEST_F(Test__CUDAMoEKernel, DecodeRouteSelectRejectsUnsupportedGateType)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    constexpr int num_layers = 1;
    constexpr int num_experts = 16;
    constexpr int top_k = 4;
    constexpr int d_model = 32;
    constexpr int seq_len = 1;

    DeviceMoERuntimeTable::Config cuda_config;
    cuda_config.device_id = llaminar2::DeviceId::cuda(0);
    cuda_config.num_layers = num_layers;
    cuda_config.num_experts = num_experts;
    cuda_config.top_k = top_k;
    cuda_config.mirror_to_device = true;
    DeviceMoERuntimeTable cuda_table(cuda_config);

    llaminar2::MoEPlacementUpdate update;
    update.epoch = 1;
    update.expert_count = num_experts;
    update.experts.resize(num_experts);
    markAllExpertsLocalForRouting(update, num_experts);

    ASSERT_TRUE(cuda_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(cuda_table.flipActiveBank(0, update.epoch, stream_));

    std::vector<float> hidden_values(static_cast<size_t>(d_model));
    for (int i = 0; i < d_model; ++i)
        hidden_values[static_cast<size_t>(i)] = 0.01f * static_cast<float>(i - 8);

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto gate_q8 = llaminar2::test::TestTensorFactory::createQ8_0Random(
        {static_cast<size_t>(num_experts), static_cast<size_t>(d_model)}, /*seed=*/123);
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});

    EXPECT_FALSE(cuda_kernel_->decodeRouteSelect(
        cuda_table.deviceLayerState(0), hidden.get(), gate_q8.get(), d_model, num_experts, top_k,
        true, cuda_indices.get(), cuda_weights.get(), true, true));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, DecodeRouteSelectRuntimeOnlyDoesNotRequireLegacyOutputs)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoELayerRuntime;
    using llaminar2::DeviceMoERuntimeTable;
    constexpr int num_layers = 1;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int d_model = 4;
    constexpr int seq_len = 1;

    DeviceMoERuntimeTable::Config cuda_config;
    cuda_config.device_id = llaminar2::DeviceId::cuda(0);
    cuda_config.num_layers = num_layers;
    cuda_config.num_experts = num_experts;
    cuda_config.top_k = top_k;
    cuda_config.mirror_to_device = true;
    DeviceMoERuntimeTable cuda_table(cuda_config);

    llaminar2::MoEPlacementUpdate update;
    update.epoch = 1;
    update.expert_count = num_experts;
    update.experts.resize(num_experts);
    markAllExpertsLocalForRouting(update, num_experts);

    ASSERT_TRUE(cuda_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(cuda_table.flipActiveBank(0, update.epoch, stream_));

    auto hidden = makeTensor({seq_len, d_model}, {0.4f, -0.2f, 0.1f, 0.7f});
    auto gate = makeTensor({num_experts, d_model}, {0.2f, -0.1f, 0.5f, 0.3f,
                                                    -0.3f, 0.4f, 0.2f, 0.1f,
                                                    0.6f, 0.1f, -0.4f, 0.2f,
                                                    0.0f, -0.5f, 0.3f, 0.8f});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    auto *cuda_layer = cuda_table.deviceLayerState(0);
    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        cuda_layer, hidden.get(), gate.get(), d_model, num_experts, top_k,
        true, nullptr, nullptr, /*write_legacy_outputs=*/false, /*update_runtime_histogram=*/true));

    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));

    int32_t runtime_ids[top_k] = {};
    float runtime_weights[top_k] = {};
    const char *runtime_base = reinterpret_cast<const char *>(cuda_layer);
    const auto *ids_device = reinterpret_cast<const int32_t *>(
        runtime_base + offsetof(DeviceMoELayerRuntime, topk_expert_ids));
    const auto *weights_device = reinterpret_cast<const float *>(
        runtime_base + offsetof(DeviceMoELayerRuntime, topk_weights));

    ASSERT_EQ(cudaMemcpyAsync(runtime_ids, ids_device, sizeof(runtime_ids),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(runtime_weights, weights_device, sizeof(runtime_weights),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const float *cpu_index_data = cpu_indices->data();
    const float *cpu_weight_data = cpu_weights->data();
    int expected_counts[num_experts] = {};
    for (int k = 0; k < top_k; ++k)
    {
        EXPECT_EQ(runtime_ids[k], static_cast<int32_t>(cpu_index_data[k]));
        EXPECT_NEAR(runtime_weights[k], cpu_weight_data[k], 1.0e-5f);
        ++expected_counts[static_cast<int>(cpu_index_data[k])];
    }

    llaminar2::DecodeExpertHistogramConfig hist_config;
    hist_config.num_layers = num_layers;
    hist_config.num_experts = num_experts;
    hist_config.top_k = top_k;
    hist_config.window_size = 8;
    hist_config.sockets = {llaminar2::DeviceId(llaminar2::DeviceType::CPU, 0)};
    hist_config.expert_to_socket.assign(num_experts, 0);
    llaminar2::DecodeExpertHistogram runtime_histogram(hist_config);

    runtime_histogram.recordTokenBoundary(0);
    ASSERT_TRUE(cuda_table.syncDecodeHistogramToHost(runtime_histogram, stream_, true));
    EXPECT_EQ(runtime_histogram.windowTokenCount(), 1u);
    for (int expert = 0; expert < num_experts; ++expert)
        EXPECT_EQ(runtime_histogram.activationCount(0, expert), static_cast<uint64_t>(expected_counts[expert]));
#endif
}

TEST_F(Test__CUDAMoEKernel, DecodeRouteSelectRuntimeAssignsReplicasOnceAcrossParticipants)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoEExpertDescriptor;
    using llaminar2::DeviceMoEExpertFlags;
    using llaminar2::DeviceMoELayerRuntime;
    using llaminar2::DeviceMoEReplicaRole;
    using llaminar2::DeviceMoERuntimeTable;
    using llaminar2::DeviceNativeVNNIMatrixDesc;
    using llaminar2::MoEPlacementUpdate;

    constexpr int num_layers = 1;
    constexpr int num_experts = 4;
    constexpr int top_k = 4;
    constexpr int d_model = 4;
    constexpr int seq_len = 1;

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
                desc.flags = llaminar2::toMoEExpertFlags(DeviceMoEExpertFlags::Replicated);
            update.resident_participant_mask[expert] = resident_mask;
            if (local)
            {
                const uintptr_t base = 0x10000000u + static_cast<uintptr_t>(participant_id) * 0x100000u +
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
                desc.flags = llaminar2::toMoEExpertFlags(flags);
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
        config.device_id = llaminar2::DeviceId::cuda(0);
        config.num_layers = num_layers;
        config.num_experts = num_experts;
        config.top_k = top_k;
        config.mirror_to_device = true;
        auto table = std::make_unique<DeviceMoERuntimeTable>(config);
        auto update = make_update(participant_id);
        EXPECT_TRUE(table->prepareInactiveBank(0, update));
        EXPECT_TRUE(table->flipActiveBank(0, update.epoch, stream_));
        return table;
    };

    auto table0 = make_table(0);
    auto table1 = make_table(1);
    auto table2 = make_table(2);

    auto hidden = makeTensor({seq_len, d_model}, {1.0f, 0.0f, 0.0f, 0.0f});
    auto gate = makeTensor({num_experts, d_model}, {4.0f, 0.0f, 0.0f, 0.0f,
                                                    3.0f, 0.0f, 0.0f, 0.0f,
                                                    2.0f, 0.0f, 0.0f, 0.0f,
                                                    1.0f, 0.0f, 0.0f, 0.0f});
    auto indices0 = makeZeros({seq_len, top_k});
    auto weights0 = makeZeros({seq_len, top_k});
    auto indices1 = makeZeros({seq_len, top_k});
    auto weights1 = makeZeros({seq_len, top_k});
    auto indices2 = makeZeros({seq_len, top_k});
    auto weights2 = makeZeros({seq_len, top_k});

    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        table0->deviceLayerState(0), hidden.get(), gate.get(), d_model, num_experts, top_k,
        false, indices0.get(), weights0.get(), true, true));
    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        table1->deviceLayerState(0), hidden.get(), gate.get(), d_model, num_experts, top_k,
        false, indices1.get(), weights1.get(), true, true));
    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        table2->deviceLayerState(0), hidden.get(), gate.get(), d_model, num_experts, top_k,
        false, indices2.get(), weights2.get(), true, true));

    auto copy_runtime = [&](DeviceMoERuntimeTable &table,
                            std::array<int32_t, top_k> &ids,
                            std::array<float, top_k> &weights)
    {
        auto *layer = table.deviceLayerState(0);
        const auto *base = reinterpret_cast<const char *>(layer);
        const auto *ids_device = reinterpret_cast<const int32_t *>(
            base + offsetof(DeviceMoELayerRuntime, topk_expert_ids));
        const auto *weights_device = reinterpret_cast<const float *>(
            base + offsetof(DeviceMoELayerRuntime, topk_weights));
        ASSERT_EQ(cudaMemcpyAsync(ids.data(), ids_device, ids.size() * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost, stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(weights.data(), weights_device, weights.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_),
                  cudaSuccess);
    };

    std::array<int32_t, top_k> ids0{};
    std::array<int32_t, top_k> ids1{};
    std::array<int32_t, top_k> ids2{};
    std::array<float, top_k> runtime_weights0{};
    std::array<float, top_k> runtime_weights1{};
    std::array<float, top_k> runtime_weights2{};
    copy_runtime(*table0, ids0, runtime_weights0);
    copy_runtime(*table1, ids1, runtime_weights1);
    copy_runtime(*table2, ids2, runtime_weights2);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const std::array<int32_t, top_k> expected0{0, -1, 2, -1};
    const std::array<int32_t, top_k> expected1{-1, 1, -1, -1};
    const std::array<int32_t, top_k> expected2{-1, -1, -1, 3};
    EXPECT_EQ(ids0, expected0);
    EXPECT_EQ(ids1, expected1);
    EXPECT_EQ(ids2, expected2);
    for (int slot = 0; slot < top_k; ++slot)
    {
        EXPECT_EQ((ids0[slot] >= 0) + (ids1[slot] >= 0) + (ids2[slot] >= 0), 1) << "slot " << slot;
        EXPECT_EQ(indices0->data()[slot], static_cast<float>(slot));
        EXPECT_EQ(indices1->data()[slot], static_cast<float>(slot));
        EXPECT_EQ(indices2->data()[slot], static_cast<float>(slot));
    }
#endif
}

TEST_F(Test__CUDAMoEKernel, DecodeRouteSelectRuntimeRecordsHotCacheBalanceImprovement)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoEExpertDescriptor;
    using llaminar2::DeviceMoEExpertFlags;
    using llaminar2::DeviceMoELayerRuntime;
    using llaminar2::DeviceMoEReplicaRole;
    using llaminar2::DeviceMoERuntimeTable;
    using llaminar2::DeviceNativeVNNIMatrixDesc;
    using llaminar2::MoEPlacementUpdate;

    constexpr int num_layers = 1;
    constexpr int num_experts = 4;
    constexpr int top_k = 4;
    constexpr int d_model = 4;
    constexpr int seq_len = 1;

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
            desc.flags = llaminar2::toMoEExpertFlags(DeviceMoEExpertFlags::Replicated);
            if (local)
            {
                const uintptr_t base = 0x42000000u + static_cast<uintptr_t>(participant_id) * 0x100000u +
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
                desc.flags = llaminar2::toMoEExpertFlags(flags);
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
        config.device_id = llaminar2::DeviceId::cuda(0);
        config.num_layers = num_layers;
        config.num_experts = num_experts;
        config.top_k = top_k;
        config.mirror_to_device = true;
        auto table = std::make_unique<DeviceMoERuntimeTable>(config);
        auto update = make_update(participant_id);
        EXPECT_TRUE(table->prepareInactiveBank(0, update));
        EXPECT_TRUE(table->flipActiveBank(0, update.epoch, stream_));
        return table;
    };

    auto table0 = make_table(0);
    auto table1 = make_table(1);
    auto table2 = make_table(2);

    auto hidden = makeTensor({seq_len, d_model}, {1.0f, 0.0f, 0.0f, 0.0f});
    auto gate = makeTensor({num_experts, d_model}, {4.0f, 0.0f, 0.0f, 0.0f,
                                                    3.0f, 0.0f, 0.0f, 0.0f,
                                                    2.0f, 0.0f, 0.0f, 0.0f,
                                                    1.0f, 0.0f, 0.0f, 0.0f});
    auto indices0 = makeZeros({seq_len, top_k});
    auto weights0 = makeZeros({seq_len, top_k});
    auto indices1 = makeZeros({seq_len, top_k});
    auto weights1 = makeZeros({seq_len, top_k});
    auto indices2 = makeZeros({seq_len, top_k});
    auto weights2 = makeZeros({seq_len, top_k});

    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        table0->deviceLayerState(0), hidden.get(), gate.get(), d_model, num_experts, top_k,
        false, indices0.get(), weights0.get(), true, true));
    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        table1->deviceLayerState(0), hidden.get(), gate.get(), d_model, num_experts, top_k,
        false, indices1.get(), weights1.get(), true, true));
    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        table2->deviceLayerState(0), hidden.get(), gate.get(), d_model, num_experts, top_k,
        false, indices2.get(), weights2.get(), true, true));

    DeviceMoELayerRuntime runtime0{};
    DeviceMoELayerRuntime runtime1{};
    DeviceMoELayerRuntime runtime2{};
    ASSERT_EQ(cudaMemcpyAsync(&runtime0, table0->deviceLayerState(0), sizeof(runtime0),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(&runtime1, table1->deviceLayerState(0), sizeof(runtime1),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(&runtime2, table2->deviceLayerState(0), sizeof(runtime2),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

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
#endif
}

TEST_F(Test__CUDAMoEKernel, DecodeRouteSelectQwenScaleRuntimeTopKMatchesCPU)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoELayerRuntime;
    using llaminar2::DeviceMoERuntimeTable;
    constexpr int num_layers = 1;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;
    constexpr int d_model = 64;
    constexpr int seq_len = 1;

    DeviceMoERuntimeTable::Config cuda_config;
    cuda_config.device_id = llaminar2::DeviceId::cuda(0);
    cuda_config.num_layers = num_layers;
    cuda_config.num_experts = num_experts;
    cuda_config.top_k = top_k;
    cuda_config.mirror_to_device = true;
    DeviceMoERuntimeTable cuda_table(cuda_config);

    llaminar2::MoEPlacementUpdate update;
    update.epoch = 1;
    update.expert_count = num_experts;
    update.experts.resize(num_experts);
    markAllExpertsLocalForRouting(update, num_experts);

    ASSERT_TRUE(cuda_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(cuda_table.flipActiveBank(0, update.epoch, stream_));

    std::vector<float> hidden_values(static_cast<size_t>(d_model));
    for (int i = 0; i < d_model; ++i)
        hidden_values[static_cast<size_t>(i)] = (i == 0)
                                                    ? 1.0f
                                                    : 0.01f * std::sin(static_cast<float>(i) * 0.37f);

    std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        gate_values[static_cast<size_t>(expert) * d_model] =
            -1.5f + 0.015f * static_cast<float>(expert);
        for (int i = 1; i < d_model; ++i)
        {
            gate_values[static_cast<size_t>(expert) * d_model + i] =
                0.05f * std::cos(static_cast<float>((expert + 3) * (i + 5)) * 0.011f);
        }
    }

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto gate = makeTensor({num_experts, d_model}, gate_values);
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    auto *cuda_layer = cuda_table.deviceLayerState(0);
    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        cuda_layer, hidden.get(), gate.get(), d_model, num_experts, top_k,
        true, cuda_indices.get(), cuda_weights.get(), true, true));

    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));

    int32_t runtime_ids[top_k] = {};
    float runtime_weights[top_k] = {};
    const char *runtime_base = reinterpret_cast<const char *>(cuda_layer);
    const auto *ids_device = reinterpret_cast<const int32_t *>(
        runtime_base + offsetof(DeviceMoELayerRuntime, topk_expert_ids));
    const auto *weights_device = reinterpret_cast<const float *>(
        runtime_base + offsetof(DeviceMoELayerRuntime, topk_weights));
    ASSERT_EQ(cudaMemcpyAsync(runtime_ids, ids_device, sizeof(runtime_ids),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(runtime_weights, weights_device, sizeof(runtime_weights),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    expectNearArray(cuda_indices->data(), cpu_indices->data(), seq_len * top_k, 0.0f);
    expectNearArray(cuda_weights->data(), cpu_weights->data(), seq_len * top_k, 1.0e-5f);

    const float *cpu_index_data = cpu_indices->data();
    const float *cpu_weight_data = cpu_weights->data();
    for (int k = 0; k < top_k; ++k)
    {
        EXPECT_EQ(runtime_ids[k], static_cast<int32_t>(cpu_index_data[k]));
        EXPECT_NEAR(runtime_weights[k], cpu_weight_data[k], 1.0e-5f);
    }
#endif
}

TEST_F(Test__CUDAMoEKernel, DecodeRouteSelectUsesWorkspaceAcrossRebind)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoELayerRuntime;
    using llaminar2::DeviceMoERuntimeTable;
    constexpr int num_layers = 1;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;
    constexpr int d_model = 2048;
    constexpr int seq_len = 1;

    DeviceMoERuntimeTable::Config cuda_config;
    cuda_config.device_id = llaminar2::DeviceId::cuda(0);
    cuda_config.num_layers = num_layers;
    cuda_config.num_experts = num_experts;
    cuda_config.top_k = top_k;
    cuda_config.mirror_to_device = true;
    DeviceMoERuntimeTable cuda_table(cuda_config);

    llaminar2::MoEPlacementUpdate update;
    update.epoch = 1;
    update.expert_count = num_experts;
    update.experts.resize(num_experts);
    update.local_compute_mask.assign(num_experts, 0);
    update.replica_role.resize(num_experts, 0);
    for (int expert = 0; expert < num_experts; ++expert)
        update.experts[expert].logical_expert_id = expert;

    ASSERT_TRUE(cuda_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(cuda_table.flipActiveBank(0, update.epoch, stream_));

    std::vector<float> hidden_values(static_cast<size_t>(d_model));
    for (int i = 0; i < d_model; ++i)
        hidden_values[static_cast<size_t>(i)] =
            0.05f * std::sin(static_cast<float>(i) * 0.017f);

    std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        for (int i = 0; i < d_model; ++i)
        {
            gate_values[static_cast<size_t>(expert) * d_model + i] =
                0.02f * std::cos(static_cast<float>((expert + 11) * (i + 7)) * 0.003f) +
                0.0001f * static_cast<float>(expert);
        }
    }

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto gate = makeTensor({num_experts, d_model}, gate_values);
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    auto *workspace_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(cuda_kernel_);
    ASSERT_NE(workspace_consumer, nullptr);

    auto allocate_route_workspace = [&]()
    {
        auto reqs = llaminar2::MoEWorkspaceBuffers::routing(
            /*max_seq_len=*/seq_len,
            /*num_experts=*/num_experts,
            /*top_k=*/top_k);
        auto workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
            llaminar2::DeviceId::cuda(0),
            reqs.total_bytes_with_alignment() + 1024 * 1024);
        EXPECT_TRUE(workspace->allocate(reqs));
        return workspace;
    };

    auto first_workspace = allocate_route_workspace();
    ASSERT_NE(first_workspace, nullptr);
    workspace_consumer->bindWorkspace(first_workspace.get());

    auto *cuda_layer = cuda_table.deviceLayerState(0);
    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        cuda_layer, hidden.get(), gate.get(), d_model, num_experts, top_k,
        true, cuda_indices.get(), cuda_weights.get(), true, true));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    workspace_consumer->unbindWorkspace();
    first_workspace.reset();

    auto second_workspace = allocate_route_workspace();
    ASSERT_NE(second_workspace, nullptr);
    workspace_consumer->bindWorkspace(second_workspace.get());

    auto cuda_indices_second = makeZeros({seq_len, top_k});
    auto cuda_weights_second = makeZeros({seq_len, top_k});

    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        cuda_layer, hidden.get(), gate.get(), d_model, num_experts, top_k,
        true, cuda_indices_second.get(), cuda_weights_second.get(), true, true));

    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    expectNearArray(cuda_indices_second->data(), cpu_indices->data(), seq_len * top_k, 0.0f);
    expectNearArray(cuda_weights_second->data(), cpu_weights->data(), seq_len * top_k, 1.0e-5f);
#endif
}

TEST_F(Test__CUDAMoEKernel, DecodeRouteSelectRejectsGraphCaptureWithoutWorkspace)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    constexpr int num_layers = 1;
    constexpr int num_experts = 16;
    constexpr int top_k = 4;
    constexpr int d_model = 64;
    constexpr int seq_len = 1;

    auto *workspace_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(cuda_kernel_);
    ASSERT_NE(workspace_consumer, nullptr);
    workspace_consumer->unbindWorkspace();

    DeviceMoERuntimeTable::Config cuda_config;
    cuda_config.device_id = llaminar2::DeviceId::cuda(0);
    cuda_config.num_layers = num_layers;
    cuda_config.num_experts = num_experts;
    cuda_config.top_k = top_k;
    cuda_config.mirror_to_device = true;
    DeviceMoERuntimeTable cuda_table(cuda_config);

    llaminar2::MoEPlacementUpdate update;
    update.epoch = 1;
    update.expert_count = num_experts;
    update.experts.resize(num_experts);
    update.local_compute_mask.assign(num_experts, 0);
    update.replica_role.resize(num_experts, 0);
    for (int expert = 0; expert < num_experts; ++expert)
        update.experts[expert].logical_expert_id = expert;

    ASSERT_TRUE(cuda_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(cuda_table.flipActiveBank(0, update.epoch, stream_));

    std::vector<float> hidden_values(static_cast<size_t>(d_model));
    std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model);
    for (int i = 0; i < d_model; ++i)
        hidden_values[static_cast<size_t>(i)] = 0.01f * static_cast<float>(i + 1);
    for (int i = 0; i < num_experts * d_model; ++i)
        gate_values[static_cast<size_t>(i)] = 0.001f * static_cast<float>((i % 17) - 8);

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto gate = makeTensor({num_experts, d_model}, gate_values);
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});

    const auto cuda_device = llaminar2::DeviceId::cuda(0);
    ASSERT_TRUE(hidden->ensureOnDevice(cuda_device, stream_));
    ASSERT_TRUE(gate->ensureOnDevice(cuda_device, stream_));
    ASSERT_TRUE(cuda_indices->ensureOnDevice(cuda_device, stream_));
    ASSERT_TRUE(cuda_weights->ensureOnDevice(cuda_device, stream_));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::GraphCaptureGuard guard;
    EXPECT_FALSE(cuda_kernel_->decodeRouteSelect(
        cuda_table.deviceLayerState(0), hidden.get(), gate.get(), d_model, num_experts, top_k,
        true, cuda_indices.get(), cuda_weights.get(), true, true));
#endif
}

TEST_F(Test__CUDAMoEKernel, RouteWithTensorsMatchesCPU)
{
    constexpr int seq_len = 3;
    constexpr int d_model = 4;
    constexpr int num_experts = 5;
    constexpr int top_k = 2;

    auto hidden = makeTensor({seq_len, d_model}, {
                                                     0.25f,
                                                     -0.50f,
                                                     0.75f,
                                                     1.00f,
                                                     -1.00f,
                                                     0.50f,
                                                     0.25f,
                                                     -0.75f,
                                                     0.60f,
                                                     0.10f,
                                                     -0.30f,
                                                     0.90f,
                                                 });
    auto gate = makeTensor({num_experts, d_model}, {
                                                       0.10f,
                                                       0.20f,
                                                       -0.30f,
                                                       0.40f,
                                                       -0.40f,
                                                       0.10f,
                                                       0.30f,
                                                       0.20f,
                                                       0.50f,
                                                       -0.20f,
                                                       0.10f,
                                                       -0.10f,
                                                       -0.30f,
                                                       -0.60f,
                                                       0.40f,
                                                       0.70f,
                                                       0.20f,
                                                       0.80f,
                                                       -0.50f,
                                                       0.30f,
                                                   });
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    llaminar2::MoERoutingResult cuda_host_result;
    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cuda_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                               num_experts, top_k, true,
                                               cuda_indices.get(), cuda_weights.get(), cuda_host_result));
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif

    expectNearArray(cuda_indices->data(), cpu_indices->data(), cuda_indices->numel(), 0.0f);
    expectNearArray(cuda_weights->data(), cpu_weights->data(), cuda_weights->numel(), 1.0e-5f);

#ifdef ENABLE_PIPELINE_SNAPSHOTS
    ASSERT_EQ(cuda_host_result.router_logits.size(), cpu_host_result.router_logits.size());
    expectNearArray(cuda_host_result.router_logits.data(), cpu_host_result.router_logits.data(),
                    cuda_host_result.router_logits.size(), 1.0e-5f);
    for (int token = 0; token < seq_len; ++token)
    {
        float prob_sum = 0.0f;
        for (int expert = 0; expert < num_experts; ++expert)
            prob_sum += cuda_host_result.router_logits[static_cast<size_t>(token) * num_experts + expert];
        EXPECT_NEAR(prob_sum, 1.0f, 1.0e-5f);
    }
#endif
}

TEST_F(Test__CUDAMoEKernel, RouteWithTensorsAcceptsMappedActivationBuffers)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    constexpr int seq_len = 3;
    constexpr int d_model = 4;
    constexpr int num_experts = 5;
    constexpr int top_k = 2;
    const auto cuda_device = llaminar2::DeviceId::cuda(0);

    auto hidden = llaminar2::FP32Tensor::createMapped({seq_len, d_model}, cuda_device);
    auto cuda_indices = llaminar2::FP32Tensor::createMapped({seq_len, top_k}, cuda_device);
    auto cuda_weights = llaminar2::FP32Tensor::createMapped({seq_len, top_k}, cuda_device);
    ASSERT_NE(hidden, nullptr);
    ASSERT_NE(cuda_indices, nullptr);
    ASSERT_NE(cuda_weights, nullptr);
    ASSERT_TRUE(hidden->isMapped());
    ASSERT_TRUE(cuda_indices->isMapped());
    ASSERT_TRUE(cuda_weights->isMapped());

    const std::vector<float> hidden_values = {
        0.25f, -0.50f, 0.75f, 1.00f,
        -1.00f, 0.50f, 0.25f, -0.75f,
        0.60f, 0.10f, -0.30f, 0.90f};
    std::copy(hidden_values.begin(), hidden_values.end(), hidden->mutable_data());
    std::fill(cuda_indices->mutable_data(), cuda_indices->mutable_data() + cuda_indices->numel(), 0.0f);
    std::fill(cuda_weights->mutable_data(), cuda_weights->mutable_data() + cuda_weights->numel(), 0.0f);

    auto gate = makeTensor({num_experts, d_model}, {
                                                       0.10f,
                                                       0.20f,
                                                       -0.30f,
                                                       0.40f,
                                                       -0.40f,
                                                       0.10f,
                                                       0.30f,
                                                       0.20f,
                                                       0.50f,
                                                       -0.20f,
                                                       0.10f,
                                                       -0.10f,
                                                       -0.30f,
                                                       -0.60f,
                                                       0.40f,
                                                       0.70f,
                                                       0.20f,
                                                       0.80f,
                                                       -0.50f,
                                                       0.30f,
                                                   });
    auto cpu_hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    llaminar2::MoERoutingResult cuda_host_result;
    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cuda_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                               num_experts, top_k, true,
                                               cuda_indices.get(), cuda_weights.get(), cuda_host_result));
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(cpu_hidden.get(), gate.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    expectNearArray(cuda_indices->data(), cpu_indices->data(), cuda_indices->numel(), 0.0f);
    expectNearArray(cuda_weights->data(), cpu_weights->data(), cuda_weights->numel(), 1.0e-5f);
#endif
}

TEST_F(Test__CUDAMoEKernel, RouteWithTensorsTiledPrefillMatchesCPU)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    constexpr int seq_len = 32;
    constexpr int d_model = 64;
    constexpr int num_experts = 16;
    constexpr int top_k = 4;

    std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
    for (size_t i = 0; i < hidden_values.size(); ++i)
        hidden_values[i] = 0.125f * std::sin(static_cast<float>(i) * 0.17f) +
                           0.03125f * static_cast<float>(static_cast<int>(i % 7) - 3);

    std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model);
    for (size_t i = 0; i < gate_values.size(); ++i)
        gate_values[i] = 0.09375f * std::cos(static_cast<float>(i) * 0.11f) -
                         0.015625f * static_cast<float>(static_cast<int>(i % 5) - 2);

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto gate = makeTensor({num_experts, d_model}, gate_values);
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_JSON", "1");
    llaminar2::PerfStatsCollector::reset();

    llaminar2::MoERoutingResult cuda_host_result;
    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cuda_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                               num_experts, top_k, true,
                                               cuda_indices.get(), cuda_weights.get(), cuda_host_result));
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    expectNearArray(cuda_indices->data(), cpu_indices->data(), cuda_indices->numel(), 0.0f);
    expectNearArray(cuda_weights->data(), cpu_weights->data(), cuda_weights->numel(), 2.0e-5f);

#ifdef ENABLE_PIPELINE_SNAPSHOTS
    ASSERT_EQ(cuda_host_result.router_logits.size(), cpu_host_result.router_logits.size());
    expectNearArray(cuda_host_result.router_logits.data(), cpu_host_result.router_logits.data(),
                    cuda_host_result.router_logits.size(), 2.0e-5f);
#endif

    const auto records =
        llaminar2::PerfStatsCollector::snapshot({"kernel.cuda_moe_router_tiled_prefill_calls"});
    const auto match = std::find_if(records.begin(), records.end(), [&](const llaminar2::PerfStatRecord &record)
                                    {
                                        auto tag_equals = [&](const char *key, const std::string &value)
                                        {
                                            const auto it = record.tags.find(key);
                                            return it != record.tags.end() && it->second == value;
                                        };
                                        return record.name == "cuda_moe_router_tiled_prefill_calls" &&
                                               tag_equals("seq_len", std::to_string(seq_len)) &&
                                               tag_equals("d_model", std::to_string(d_model)) &&
                                               tag_equals("num_experts", std::to_string(num_experts));
                                    });
    ASSERT_NE(match, records.end()) << "tiled router perf counter missing; test did not exercise tiled prefill path";
    EXPECT_GE(match->count, 1u);
    EXPECT_GE(match->value, 1.0);
    llaminar2::PerfStatsCollector::reset();
#endif
}

TEST_F(Test__CUDAMoEKernel, RouteWithTensorsTiledPrefillCapturesAfterWarmup)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int seq_len = 32;
    constexpr int d_model = 64;
    constexpr int num_experts = 16;
    constexpr int top_k = 4;

    std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
    for (size_t i = 0; i < hidden_values.size(); ++i)
        hidden_values[i] = 0.125f * std::sin(static_cast<float>(i) * 0.17f) +
                           0.03125f * static_cast<float>(static_cast<int>(i % 7) - 3);

    std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model);
    for (size_t i = 0; i < gate_values.size(); ++i)
        gate_values[i] = 0.09375f * std::cos(static_cast<float>(i) * 0.11f) -
                         0.015625f * static_cast<float>(static_cast<int>(i % 5) - 2);

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto gate = makeTensor({num_experts, d_model}, gate_values);
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    llaminar2::MoERoutingResult warmup_host_result;
    ASSERT_TRUE(cuda_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                               num_experts, top_k, true,
                                               cuda_indices.get(), cuda_weights.get(), warmup_host_result));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(warmup_host_result.expert_indices.size(), static_cast<size_t>(seq_len * top_k));
    ASSERT_EQ(warmup_host_result.expert_weights.size(), static_cast<size_t>(seq_len * top_k));

    llaminar2::MoERoutingResult captured_host_result;
    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_route = cuda_kernel_->routeWithTensors(
        hidden.get(), gate.get(), seq_len, d_model,
        num_experts, top_k, true,
        cuda_indices.get(), cuda_weights.get(), captured_host_result);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_route);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);
    EXPECT_TRUE(captured_host_result.expert_indices.empty());
    EXPECT_TRUE(captured_host_result.expert_weights.empty());
    EXPECT_TRUE(captured_host_result.router_logits.empty());

    cudaGraphExec_t executable = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));

    std::vector<float> captured_indices(cuda_indices->numel());
    std::vector<float> captured_weights(cuda_weights->numel());
    ASSERT_EQ(cudaMemcpyAsync(captured_indices.data(), cuda_indices->gpu_data_ptr(),
                              captured_indices.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(captured_weights.data(), cuda_weights->gpu_data_ptr(),
                              captured_weights.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    expectNearArray(captured_indices.data(), cpu_indices->data(), captured_indices.size(), 0.0f);
    expectNearArray(captured_weights.data(), cpu_weights->data(), captured_weights.size(), 2.0e-5f);

    ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, RouteWithTensorsEffectiveSeqLenMasksPaddedRowsAcrossGraphReplay)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int bucket_seq_len = 8;
    constexpr int capture_real_seq_len = 6;
    constexpr int replay_real_seq_len = 5;
    constexpr int d_model = 64;
    constexpr int num_experts = 16;
    constexpr int top_k = 4;

    std::vector<float> hidden_values(static_cast<size_t>(bucket_seq_len) * d_model);
    for (size_t i = 0; i < hidden_values.size(); ++i)
        hidden_values[i] = 0.11f * std::sin(static_cast<float>(i) * 0.19f) +
                           0.017f * static_cast<float>(static_cast<int>(i % 11) - 5);

    std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model);
    for (size_t i = 0; i < gate_values.size(); ++i)
        gate_values[i] = 0.07f * std::cos(static_cast<float>(i) * 0.13f) -
                         0.021f * static_cast<float>(static_cast<int>(i % 7) - 3);

    auto hidden = makeTensor({bucket_seq_len, d_model}, hidden_values);
    auto gate = makeTensor({num_experts, d_model}, gate_values);
    auto cuda_indices = makeZeros({bucket_seq_len, top_k});
    auto cuda_weights = makeZeros({bucket_seq_len, top_k});
    auto cpu_indices = makeZeros({bucket_seq_len, top_k});
    auto cpu_weights = makeZeros({bucket_seq_len, top_k});

    int *device_effective_seq_len = nullptr;
    ASSERT_EQ(cudaMalloc(&device_effective_seq_len, sizeof(int)), cudaSuccess);
    auto free_effective = std::unique_ptr<int, void (*)(int *)>(
        device_effective_seq_len,
        [](int *ptr)
        {
            if (ptr)
                cudaFree(ptr);
        });

    ASSERT_EQ(cudaMemcpyAsync(device_effective_seq_len,
                              &capture_real_seq_len,
                              sizeof(int),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    llaminar2::MoERoutingResult warmup_result;
    ASSERT_TRUE(cuda_kernel_->routeWithTensorsEffectiveSeqLen(
        hidden.get(), gate.get(), bucket_seq_len, d_model,
        num_experts, top_k, true,
        cuda_indices.get(), cuda_weights.get(),
        warmup_result, device_effective_seq_len));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::MoERoutingResult captured_result;
    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_route = cuda_kernel_->routeWithTensorsEffectiveSeqLen(
        hidden.get(), gate.get(), bucket_seq_len, d_model,
        num_experts, top_k, true,
        cuda_indices.get(), cuda_weights.get(),
        captured_result, device_effective_seq_len);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_route);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);
    EXPECT_TRUE(captured_result.expert_indices.empty());
    EXPECT_TRUE(captured_result.expert_weights.empty());
    EXPECT_TRUE(captured_result.router_logits.empty());

    cudaGraphExec_t executable = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);

    ASSERT_EQ(cudaMemcpyAsync(device_effective_seq_len,
                              &replay_real_seq_len,
                              sizeof(int),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::MoERoutingResult cpu_result;
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate.get(), bucket_seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_result));

    std::vector<float> replay_indices(cuda_indices->numel());
    std::vector<float> replay_weights(cuda_weights->numel());
    ASSERT_EQ(cudaMemcpyAsync(replay_indices.data(), cuda_indices->gpu_data_ptr(),
                              replay_indices.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(replay_weights.data(), cuda_weights->gpu_data_ptr(),
                              replay_weights.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    for (int row = 0; row < bucket_seq_len; ++row)
    {
        for (int k = 0; k < top_k; ++k)
        {
            const size_t offset = static_cast<size_t>(row) * top_k + k;
            if (row < replay_real_seq_len)
            {
                EXPECT_FLOAT_EQ(replay_indices[offset], cpu_indices->data()[offset])
                    << "row=" << row << " k=" << k;
                EXPECT_NEAR(replay_weights[offset], cpu_weights->data()[offset], 2.0e-5f)
                    << "row=" << row << " k=" << k;
            }
            else
            {
                EXPECT_FLOAT_EQ(replay_indices[offset], -1.0f)
                    << "padded row=" << row << " k=" << k;
                EXPECT_FLOAT_EQ(replay_weights[offset], 0.0f)
                    << "padded row=" << row << " k=" << k;
            }
        }
    }

    ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, RouteWithTensorsSingleTokenQwenScalePopulatesSnapshotOutputs)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int seq_len = 1;
    constexpr int d_model = 2048;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;

    std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
    std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model);
    for (size_t i = 0; i < hidden_values.size(); ++i)
        hidden_values[i] = 0.07f * std::sin(0.013f * static_cast<float>(i + 1)) +
                           0.03f * std::cos(0.029f * static_cast<float>(i + 3));
    for (size_t i = 0; i < gate_values.size(); ++i)
        gate_values[i] = 0.05f * std::sin(0.017f * static_cast<float>(i + 5)) -
                         0.02f * std::cos(0.031f * static_cast<float>(i + 7));

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto gate = makeTensor({num_experts, d_model}, gate_values);
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    llaminar2::MoERoutingResult cuda_host_result;
    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cuda_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                               num_experts, top_k, true,
                                               cuda_indices.get(), cuda_weights.get(), cuda_host_result));
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    expectNearArray(cuda_indices->data(), cpu_indices->data(), cuda_indices->numel(), 0.0f);
    expectNearArray(cuda_weights->data(), cpu_weights->data(), cuda_weights->numel(), 1.0e-4f);

#ifdef ENABLE_PIPELINE_SNAPSHOTS
    ASSERT_EQ(cuda_host_result.router_logits.size(), static_cast<size_t>(num_experts));
    const int nonzero_logits = std::count_if(
        cuda_host_result.router_logits.begin(),
        cuda_host_result.router_logits.end(),
        [](float v)
        { return v != 0.0f; });
    EXPECT_GT(nonzero_logits, 0);
    expectNearArray(cuda_host_result.router_logits.data(), cpu_host_result.router_logits.data(),
                    cuda_host_result.router_logits.size(), 1.0e-4f);
#endif
#endif
}

TEST_F(Test__CUDAMoEKernel, RouteWithTensorsSingleTokenIsCudaGraphCapturableDeviceOnly)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int seq_len = 1;
    constexpr int d_model = 64;
    constexpr int num_experts = 8;
    constexpr int top_k = 2;

    std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
    std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model);
    for (size_t i = 0; i < hidden_values.size(); ++i)
        hidden_values[i] = 0.11f * std::sin(0.021f * static_cast<float>(i + 1)) -
                           0.04f * std::cos(0.037f * static_cast<float>(i + 2));
    for (size_t i = 0; i < gate_values.size(); ++i)
        gate_values[i] = 0.07f * std::sin(0.013f * static_cast<float>(i + 3)) +
                         0.05f * std::cos(0.029f * static_cast<float>(i + 4));

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto gate = makeTensor({num_experts, d_model}, gate_values);
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    llaminar2::MoERoutingResult warmup_host_result;
    ASSERT_TRUE(cuda_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                               num_experts, top_k, true,
                                               cuda_indices.get(), cuda_weights.get(), warmup_host_result));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(warmup_host_result.expert_indices.size(), static_cast<size_t>(seq_len * top_k));
    ASSERT_EQ(warmup_host_result.expert_weights.size(), static_cast<size_t>(seq_len * top_k));

    llaminar2::MoERoutingResult captured_host_result;
    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_route = cuda_kernel_->routeWithTensors(
        hidden.get(), gate.get(), seq_len, d_model,
        num_experts, top_k, true,
        cuda_indices.get(), cuda_weights.get(), captured_host_result);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_route);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);
    EXPECT_TRUE(captured_host_result.expert_indices.empty());
    EXPECT_TRUE(captured_host_result.expert_weights.empty());
    EXPECT_TRUE(captured_host_result.router_logits.empty());

    cudaGraphExec_t executable = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));

    std::vector<float> captured_indices(cuda_indices->numel());
    std::vector<float> captured_weights(cuda_weights->numel());
    ASSERT_EQ(cudaMemcpyAsync(captured_indices.data(), cuda_indices->gpu_data_ptr(),
                              captured_indices.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(captured_weights.data(), cuda_weights->gpu_data_ptr(),
                              captured_weights.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    expectNearArray(captured_indices.data(), cpu_indices->data(), captured_indices.size(), 0.0f);
    expectNearArray(captured_weights.data(), cpu_weights->data(), captured_weights.size(), 1.0e-5f);

    ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, RouteVerifierRowsDecodeEquivalentUsesDecodeKernelAndCaptures)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int seq_len = 2;
    constexpr int d_model = 2048;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;

    std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
    std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model);
    for (size_t i = 0; i < hidden_values.size(); ++i)
        hidden_values[i] = 0.06f * std::sin(0.009f * static_cast<float>(i + 1)) -
                           0.025f * std::cos(0.017f * static_cast<float>(i + 5)) +
                           0.002f * static_cast<float>(static_cast<int>(i % 11) - 5);
    for (size_t i = 0; i < gate_values.size(); ++i)
        gate_values[i] = 0.045f * std::sin(0.013f * static_cast<float>(i + 3)) +
                         0.035f * std::cos(0.019f * static_cast<float>(i + 7)) -
                         0.001f * static_cast<float>(static_cast<int>(i % 13) - 6);

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto gate = makeTensor({num_experts, d_model}, gate_values);
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_JSON", "1");
    llaminar2::PerfStatsCollector::reset();

    ASSERT_TRUE(cuda_kernel_->routeVerifierRowsDecodeEquivalent(
        hidden.get(), gate.get(), seq_len, d_model,
        num_experts, top_k, true,
        cuda_indices.get(), cuda_weights.get()));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_route = cuda_kernel_->routeVerifierRowsDecodeEquivalent(
        hidden.get(), gate.get(), seq_len, d_model,
        num_experts, top_k, true,
        cuda_indices.get(), cuda_weights.get());
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_route);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);

    cudaGraphExec_t executable = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));

    std::vector<float> captured_indices(cuda_indices->numel());
    std::vector<float> captured_weights(cuda_weights->numel());
    ASSERT_EQ(cudaMemcpyAsync(captured_indices.data(), cuda_indices->gpu_data_ptr(),
                              captured_indices.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(captured_weights.data(), cuda_weights->gpu_data_ptr(),
                              captured_weights.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    expectNearArray(captured_indices.data(), cpu_indices->data(), captured_indices.size(), 0.0f);
    expectNearArray(captured_weights.data(), cpu_weights->data(), captured_weights.size(), 1.0e-4f);

    const auto records =
        llaminar2::PerfStatsCollector::snapshot({"kernel.cuda_moe_router_decode_equivalent_small_m_calls"});
    const auto match = std::find_if(records.begin(), records.end(), [&](const llaminar2::PerfStatRecord &record)
                                    {
                                        auto tag_equals = [&](const char *key, const std::string &value)
                                        {
                                            const auto it = record.tags.find(key);
                                            return it != record.tags.end() && it->second == value;
                                        };
                                        return record.name == "cuda_moe_router_decode_equivalent_small_m_calls" &&
                                               tag_equals("seq_len", std::to_string(seq_len)) &&
                                               tag_equals("d_model", std::to_string(d_model)) &&
                                               tag_equals("num_experts", std::to_string(num_experts)) &&
                                               tag_equals("route", "grouped_decode_equivalent");
                                    });
    ASSERT_NE(match, records.end()) << "grouped decode-equivalent router counter missing; test did not exercise the verifier route";
    EXPECT_GE(match->count, 2u);
    EXPECT_GE(match->value, 2.0);
    llaminar2::PerfStatsCollector::reset();

    ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, RouteVerifierRowsDecodeEquivalentMatchesSerialDecodeRouter)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    auto &gemm = llaminar2::mutableDebugEnv().gemm;
    gemm.cuda_moe_router_q8 = true;

    constexpr int d_model = 2048;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_JSON", "1");
    llaminar2::PerfStatsCollector::reset();

    DeviceMoERuntimeTable::Config cuda_config;
    cuda_config.device_id = llaminar2::DeviceId::cuda(0);
    cuda_config.num_layers = 1;
    cuda_config.num_experts = num_experts;
    cuda_config.top_k = top_k;
    cuda_config.mirror_to_device = true;
    DeviceMoERuntimeTable cuda_table(cuda_config);

    llaminar2::MoEPlacementUpdate update;
    update.epoch = 1;
    update.expert_count = num_experts;
    update.experts.resize(num_experts);
    update.local_compute_mask.assign(num_experts, 0);
    update.replica_role.resize(num_experts, 0);
    for (int expert = 0; expert < num_experts; ++expert)
        update.experts[expert].logical_expert_id = expert;
    ASSERT_TRUE(cuda_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(cuda_table.flipActiveBank(0, update.epoch, stream_));

    std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model);
    for (size_t i = 0; i < gate_values.size(); ++i)
        gate_values[i] = 0.035f * std::sin(0.011f * static_cast<float>(i + 13)) +
                         0.027f * std::cos(0.017f * static_cast<float>(i + 19)) +
                         0.0007f * static_cast<float>(static_cast<int>(i % 23) - 11);
    auto gate = makeTensor({num_experts, d_model}, gate_values);

    for (int seq_len : {1, 2, 3, 4})
    {
        std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
        for (size_t i = 0; i < hidden_values.size(); ++i)
            hidden_values[i] = 0.05f * std::sin(0.007f * static_cast<float>(i + 1 + seq_len)) -
                               0.031f * std::cos(0.013f * static_cast<float>(i + 5)) +
                               0.0013f * static_cast<float>(static_cast<int>(i % 17) - 8);

        auto hidden = makeTensor(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
            hidden_values);
        auto batched_indices = makeZeros(
            {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
        auto batched_weights = makeZeros(
            {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});

        ASSERT_TRUE(cuda_kernel_->routeVerifierRowsDecodeEquivalent(
            hidden.get(), gate.get(), seq_len, d_model,
            num_experts, top_k, true,
            batched_indices.get(), batched_weights.get()))
            << "seq_len=" << seq_len;
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        std::vector<float> batched_indices_host(static_cast<size_t>(seq_len) * top_k);
        std::vector<float> batched_weights_host(static_cast<size_t>(seq_len) * top_k);
        ASSERT_EQ(cudaMemcpyAsync(batched_indices_host.data(), batched_indices->gpu_data_ptr(),
                                  batched_indices_host.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(batched_weights_host.data(), batched_weights->gpu_data_ptr(),
                                  batched_weights_host.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        std::vector<float> serial_indices(static_cast<size_t>(seq_len) * top_k);
        std::vector<float> serial_weights(static_cast<size_t>(seq_len) * top_k);
        for (int row = 0; row < seq_len; ++row)
        {
            std::vector<float> row_hidden_values(static_cast<size_t>(d_model));
            std::copy_n(hidden_values.data() + static_cast<size_t>(row) * d_model,
                        d_model,
                        row_hidden_values.data());
            auto row_hidden = makeTensor({1, d_model}, row_hidden_values);
            auto row_indices = makeZeros({1, top_k});
            auto row_weights = makeZeros({1, top_k});

            ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
                cuda_table.deviceLayerState(0),
                row_hidden.get(),
                gate.get(),
                d_model,
                num_experts,
                top_k,
                true,
                row_indices.get(),
                row_weights.get(),
                /*write_legacy_outputs=*/true,
                /*update_runtime_histogram=*/false))
                << "seq_len=" << seq_len << " row=" << row;

            ASSERT_EQ(cudaMemcpyAsync(serial_indices.data() + static_cast<size_t>(row) * top_k,
                                      row_indices->gpu_data_ptr(),
                                      static_cast<size_t>(top_k) * sizeof(float),
                                      cudaMemcpyDeviceToHost, stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(serial_weights.data() + static_cast<size_t>(row) * top_k,
                                      row_weights->gpu_data_ptr(),
                                      static_cast<size_t>(top_k) * sizeof(float),
                                      cudaMemcpyDeviceToHost, stream_),
                      cudaSuccess);
        }
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        expectStrictTopKRowsEquivalent(
            batched_indices_host,
            batched_weights_host,
            serial_indices,
            serial_weights,
            seq_len,
            top_k);
    }

    const auto records =
        llaminar2::PerfStatsCollector::snapshot({"kernel.cuda_moe_router_decode_equivalent_small_m_calls"});
    const auto q8_match = std::find_if(records.begin(), records.end(), [](const llaminar2::PerfStatRecord &record)
                                       {
                                           const auto route_it = record.tags.find("route");
                                           return record.name == "cuda_moe_router_decode_equivalent_small_m_calls" &&
                                                  route_it != record.tags.end() &&
                                                  route_it->second == "grouped_decode_equivalent_q8";
                                       });
    ASSERT_NE(q8_match, records.end())
        << "Q8 verifier routing must exercise the economical grouped decode-equivalent router";
    llaminar2::PerfStatsCollector::reset();
#endif
}

TEST_F(Test__CUDAMoEKernel, RouteWithTensorsBF16GateMatchesCPU)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int seq_len = 1;
    constexpr int d_model = 2048;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;

    std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
    std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model);
    for (size_t i = 0; i < hidden_values.size(); ++i)
        hidden_values[i] = 0.09f * std::sin(0.011f * static_cast<float>(i + 1)) -
                           0.04f * std::cos(0.023f * static_cast<float>(i + 2));
    for (size_t i = 0; i < gate_values.size(); ++i)
        gate_values[i] = 0.06f * std::sin(0.019f * static_cast<float>(i + 3)) +
                         0.03f * std::cos(0.037f * static_cast<float>(i + 4));

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto gate_bf16 = makeBF16Tensor({num_experts, d_model}, gate_values);
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    llaminar2::MoERoutingResult cuda_host_result;
    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cuda_kernel_->routeWithTensors(hidden.get(), gate_bf16.get(), seq_len, d_model,
                                               num_experts, top_k, true,
                                               cuda_indices.get(), cuda_weights.get(), cuda_host_result));
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate_bf16.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    expectNearArray(cuda_indices->data(), cpu_indices->data(), cuda_indices->numel(), 0.0f);
    expectNearArray(cuda_weights->data(), cpu_weights->data(), cuda_weights->numel(), 1.0e-4f);

#ifdef ENABLE_PIPELINE_SNAPSHOTS
    ASSERT_EQ(cuda_host_result.router_logits.size(), static_cast<size_t>(num_experts));
    ASSERT_EQ(cuda_host_result.router_logits.size(), cpu_host_result.router_logits.size());
    expectNearArray(cuda_host_result.router_logits.data(), cpu_host_result.router_logits.data(),
                    cuda_host_result.router_logits.size(), 1.0e-4f);
#endif
#endif
}

TEST_F(Test__CUDAMoEKernel, RouteWithTensorsRejectsInvalidTensorContractsBeforeLaunch)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int seq_len = 1;
    constexpr int d_model = 4;
    constexpr int num_experts = 3;
    constexpr int top_k = 2;

    auto hidden = makeTensor({seq_len, d_model}, {0.2f, -0.1f, 0.3f, 0.5f});
    auto hidden_bf16 = makeBF16Tensor({seq_len, d_model}, {0.2f, -0.1f, 0.3f, 0.5f});
    auto hidden_too_narrow = makeTensor({seq_len, d_model - 1}, {0.2f, -0.1f, 0.3f});
    auto gate = makeTensor({num_experts, d_model}, {0.1f, 0.2f, 0.3f, 0.4f,
                                                    -0.2f, 0.0f, 0.1f, 0.2f,
                                                    0.5f, -0.3f, 0.2f, -0.1f});
    auto gate_too_short = makeTensor({num_experts - 1, d_model}, {0.1f, 0.2f, 0.3f, 0.4f,
                                                                  -0.2f, 0.0f, 0.1f, 0.2f});
    auto indices = makeZeros({seq_len, top_k});
    auto weights = makeZeros({seq_len, top_k});
    auto short_indices = makeZeros({seq_len, top_k - 1});
    llaminar2::MoERoutingResult result;

    cudaGetLastError();
    EXPECT_FALSE(cuda_kernel_->routeWithTensors(hidden_bf16.get(), gate.get(), seq_len, d_model,
                                                num_experts, top_k, true,
                                                indices.get(), weights.get(), result));
    EXPECT_FALSE(cuda_kernel_->routeWithTensors(hidden_too_narrow.get(), gate.get(), seq_len, d_model,
                                                num_experts, top_k, true,
                                                indices.get(), weights.get(), result));
    EXPECT_FALSE(cuda_kernel_->routeWithTensors(hidden.get(), gate_too_short.get(), seq_len, d_model,
                                                num_experts, top_k, true,
                                                indices.get(), weights.get(), result));
    EXPECT_FALSE(cuda_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                                num_experts, top_k, true,
                                                short_indices.get(), weights.get(), result));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, GatherAndScatterMatchCPU)
{
    constexpr int rows = 4;
    constexpr int d_model = 5;
    const std::vector<int> token_indices = {3, 1, 0};
    const std::vector<float> weights = {0.25f, 0.50f, 0.75f};

    auto hidden = makeTensor({rows, d_model}, {
                                                  0.0f,
                                                  0.1f,
                                                  0.2f,
                                                  0.3f,
                                                  0.4f,
                                                  1.0f,
                                                  1.1f,
                                                  1.2f,
                                                  1.3f,
                                                  1.4f,
                                                  2.0f,
                                                  2.1f,
                                                  2.2f,
                                                  2.3f,
                                                  2.4f,
                                                  3.0f,
                                                  3.1f,
                                                  3.2f,
                                                  3.3f,
                                                  3.4f,
                                              });
    auto cuda_batch = makeZeros({token_indices.size(), d_model});
    auto cpu_batch = makeZeros({token_indices.size(), d_model});

    cuda_kernel_->gatherTokenBatchFromTensors(hidden.get(), cuda_batch.get(),
                                              token_indices.data(), static_cast<int>(token_indices.size()), d_model);
    cpu_kernel_->gatherTokenBatchFromTensors(hidden.get(), cpu_batch.get(),
                                             token_indices.data(), static_cast<int>(token_indices.size()), d_model);
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
    expectNearArray(cuda_batch->data(), cpu_batch->data(), cuda_batch->numel());

    auto cuda_output = makeZeros({rows, d_model});
    auto cpu_output = makeZeros({rows, d_model});
    cuda_kernel_->scatterAddWeightedFromTensors(cuda_output.get(), cuda_batch.get(),
                                                token_indices.data(), weights.data(),
                                                static_cast<int>(token_indices.size()), d_model);
    cpu_kernel_->scatterAddWeightedFromTensors(cpu_output.get(), cpu_batch.get(),
                                               token_indices.data(), weights.data(),
                                               static_cast<int>(token_indices.size()), d_model);
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
    expectNearArray(cuda_output->data(), cpu_output->data(), cuda_output->numel());
}

TEST_F(Test__CUDAMoEKernel, SwiGLUAndZeroMatchCPU)
{
    constexpr int count = 12;
    auto gate_cuda = makeTensor({count}, {-1.0f, -0.75f, -0.25f, 0.0f, 0.25f, 0.5f,
                                          0.75f, 1.0f, 1.25f, 1.5f, -1.5f, 2.0f});
    auto gate_cpu = makeTensor({count}, {-1.0f, -0.75f, -0.25f, 0.0f, 0.25f, 0.5f,
                                         0.75f, 1.0f, 1.25f, 1.5f, -1.5f, 2.0f});
    auto up = makeTensor({count}, {0.5f, -0.25f, 0.75f, 1.0f, -1.25f, 0.33f,
                                   1.2f, -0.8f, 0.9f, 0.4f, -0.6f, 0.1f});

    cuda_kernel_->swiGLUFromTensors(gate_cuda.get(), up.get(), count);
    cpu_kernel_->swiGLUFromTensors(gate_cpu.get(), up.get(), count);
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
    expectNearArray(gate_cuda->data(), gate_cpu->data(), count, 1.0e-5f);

    cuda_kernel_->zeroBuffer(gate_cuda.get(), gate_cuda->size_bytes());
    cpu_kernel_->zeroBuffer(gate_cpu.get(), gate_cpu->size_bytes());
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
    expectNearArray(gate_cuda->data(), gate_cpu->data(), count, 0.0f);
}

TEST_F(Test__CUDAMoEKernel, SharedExpertGateNegativeSaturationMatchesCPU)
{
    constexpr int seq_len = 1;
    constexpr int d_model = 8;

    std::vector<float> input_values(d_model, 10.0f);
    std::vector<float> gate_values(d_model, -10.0f);
    std::vector<float> shared_values = {
        3.0f, -2.0f, 5.0f, -7.0f, 0.5f, -0.25f, 4.0f, -9.0f};

    auto input_cuda = makeTensor({seq_len, d_model}, input_values);
    auto input_cpu = makeTensor({seq_len, d_model}, input_values);
    auto gate_cuda = makeTensor({d_model}, gate_values);
    auto gate_cpu = makeTensor({d_model}, gate_values);
    auto shared_cuda = makeTensor({seq_len, d_model}, shared_values);
    auto shared_cpu = makeTensor({seq_len, d_model}, shared_values);

    cuda_kernel_->sharedExpertGateFromTensors(
        input_cuda.get(), gate_cuda.get(), shared_cuda.get(),
        seq_len, d_model);
    cpu_kernel_->sharedExpertGateFromTensors(
        input_cpu.get(), gate_cpu.get(), shared_cpu.get(),
        seq_len, d_model);
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif

    expectNearArray(shared_cuda->data(), shared_cpu->data(), shared_cuda->numel(), 0.0f);
    for (size_t i = 0; i < shared_cuda->numel(); ++i)
        EXPECT_EQ(shared_cuda->data()[i], 0.0f) << "saturated gate output at element " << i;
}

TEST_F(Test__CUDAMoEKernel, SharedExpertGateAddFromTensorsMatchesCPU)
{
    constexpr int seq_len = 2;
    constexpr int d_model = 8;

    std::vector<float> input_values(seq_len * d_model);
    std::vector<float> gate_values(d_model);
    std::vector<float> shared_values(seq_len * d_model);
    std::vector<float> residual_values(seq_len * d_model);
    for (int i = 0; i < seq_len * d_model; ++i)
    {
        input_values[i] = 0.01f * static_cast<float>((i % d_model) + 1);
        shared_values[i] = -0.5f + 0.05f * static_cast<float>(i);
        residual_values[i] = 1.0f - 0.03f * static_cast<float>(i);
    }
    for (int i = 0; i < d_model; ++i)
        gate_values[i] = 0.02f * static_cast<float>(i - 3);

    auto input_cuda = makeTensor({seq_len, d_model}, input_values);
    auto input_cpu = makeTensor({seq_len, d_model}, input_values);
    auto gate_cuda = makeTensor({d_model}, gate_values);
    auto gate_cpu = makeTensor({d_model}, gate_values);
    auto shared_cuda = makeTensor({seq_len, d_model}, shared_values);
    auto shared_cpu = makeTensor({seq_len, d_model}, shared_values);
    auto residual_cuda = makeTensor({seq_len, d_model}, residual_values);
    auto residual_cpu = makeTensor({seq_len, d_model}, residual_values);
    auto combined_cuda = makeZeros({seq_len, d_model});
    auto combined_cpu = makeZeros({seq_len, d_model});

    cuda_kernel_->sharedExpertGateAddFromTensors(
        input_cuda.get(), gate_cuda.get(), shared_cuda.get(),
        residual_cuda.get(), combined_cuda.get(), seq_len, d_model);
    cpu_kernel_->sharedExpertGateAddFromTensors(
        input_cpu.get(), gate_cpu.get(), shared_cpu.get(),
        residual_cpu.get(), combined_cpu.get(), seq_len, d_model);
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif

    expectNearArray(combined_cuda->data(), combined_cpu->data(), combined_cuda->numel(), 1.0e-5f);
    expectNearArray(shared_cuda->data(), shared_cpu->data(), shared_cuda->numel(), 0.0f);
}

TEST_F(Test__CUDAMoEKernel, SharedExpertGateVerifierRowsM234MatchSerialDecodeRows)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int d_model = 2048;
    const auto device = llaminar2::DeviceId::cuda(0);

    std::vector<float> gate_values(static_cast<size_t>(d_model));
    for (int i = 0; i < d_model; ++i)
    {
        gate_values[static_cast<size_t>(i)] =
            0.031f * std::sin(0.013f * static_cast<float>(i + 3)) -
            0.024f * std::cos(0.019f * static_cast<float>(i + 7)) +
            0.0005f * static_cast<float>((i % 29) - 14);
    }
    auto gate = makeTensor({static_cast<size_t>(d_model)}, gate_values);
    ASSERT_TRUE(gate->ensureOnDevice(device, stream_));

    for (int seq_len : {2, 3, 4})
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

        auto grouped_input = makeTensor(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
            input_values);
        auto grouped_shared_only = makeTensor(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
            shared_values);
        auto grouped_shared_add = makeTensor(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
            shared_values);
        auto grouped_residual = makeTensor(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
            residual_values);
        auto grouped_combined = makeZeros(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

        cuda_kernel_->sharedExpertGateFromTensors(
            grouped_input.get(),
            gate.get(),
            grouped_shared_only.get(),
            seq_len,
            d_model);
        cuda_kernel_->sharedExpertGateAddFromTensors(
            grouped_input.get(),
            gate.get(),
            grouped_shared_add.get(),
            grouped_residual.get(),
            grouped_combined.get(),
            seq_len,
            d_model);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        std::vector<float> serial_shared_only(static_cast<size_t>(seq_len) * d_model);
        std::vector<float> serial_shared_add(static_cast<size_t>(seq_len) * d_model);
        std::vector<float> serial_combined(static_cast<size_t>(seq_len) * d_model);

        for (int row = 0; row < seq_len; ++row)
        {
            const auto row_offset = static_cast<size_t>(row) * d_model;
            std::vector<float> row_input(static_cast<size_t>(d_model));
            std::vector<float> row_shared(static_cast<size_t>(d_model));
            std::vector<float> row_residual(static_cast<size_t>(d_model));
            std::copy_n(input_values.data() + row_offset, d_model, row_input.data());
            std::copy_n(shared_values.data() + row_offset, d_model, row_shared.data());
            std::copy_n(residual_values.data() + row_offset, d_model, row_residual.data());

            auto row_input_tensor = makeTensor({1u, static_cast<size_t>(d_model)}, row_input);
            auto row_shared_only_tensor = makeTensor({1u, static_cast<size_t>(d_model)}, row_shared);
            auto row_shared_add_tensor = makeTensor({1u, static_cast<size_t>(d_model)}, row_shared);
            auto row_residual_tensor = makeTensor({1u, static_cast<size_t>(d_model)}, row_residual);
            auto row_combined_tensor = makeZeros({1u, static_cast<size_t>(d_model)});

            cuda_kernel_->sharedExpertGateFromTensors(
                row_input_tensor.get(),
                gate.get(),
                row_shared_only_tensor.get(),
                1,
                d_model);
            cuda_kernel_->sharedExpertGateAddFromTensors(
                row_input_tensor.get(),
                gate.get(),
                row_shared_add_tensor.get(),
                row_residual_tensor.get(),
                row_combined_tensor.get(),
                1,
                d_model);
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

            const auto row_shared_only_host = copyCudaFP32TensorToHost(row_shared_only_tensor, stream_);
            const auto row_shared_add_host = copyCudaFP32TensorToHost(row_shared_add_tensor, stream_);
            const auto row_combined_host = copyCudaFP32TensorToHost(row_combined_tensor, stream_);
            std::copy_n(row_shared_only_host.data(), d_model, serial_shared_only.data() + row_offset);
            std::copy_n(row_shared_add_host.data(), d_model, serial_shared_add.data() + row_offset);
            std::copy_n(row_combined_host.data(), d_model, serial_combined.data() + row_offset);
        }

        const auto grouped_shared_only_host = copyCudaFP32TensorToHost(grouped_shared_only, stream_);
        const auto grouped_shared_add_host = copyCudaFP32TensorToHost(grouped_shared_add, stream_);
        const auto grouped_combined_host = copyCudaFP32TensorToHost(grouped_combined, stream_);

        ASSERT_EQ(grouped_shared_only_host, serial_shared_only) << "seq_len=" << seq_len;
        ASSERT_EQ(grouped_shared_add_host, serial_shared_add) << "seq_len=" << seq_len;
        ASSERT_EQ(grouped_combined_host, serial_combined) << "seq_len=" << seq_len;
    }
#endif
}

/**
 * @brief Prove production shared-expert verifier rows match CUDA serial decode.
 *
 * The model-level Qwen3.6 MoE MTP verifier test compares grouped all-position
 * rows with ordinary one-token serial decode.  The shared expert is a dense
 * SwiGLU FFN, but CUDA serial decode intentionally reaches it through the MoE
 * grouped table decode shortcut so graph replay can reuse stable pointer-table
 * metadata.  This test exercises the exact `SharedExpertFFNStage` routes used
 * by the graph: M=2/3/4 grouped verifier publication on the left, and M=1
 * grouped table decode replay on the right.
 */
TEST_F(Test__CUDAMoEKernel, SharedExpertFFNStageVerifierRowsQwen36IQ3SMatchSerialGroupedDecode)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    const auto device = llaminar2::DeviceId::cuda(0);

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    ScopedCudaMoEGemmConfig gemm_config;
    gemm_config.set(
        /*gateup_kpart=*/true,
        /*gateup_kparts=*/16,
        /*down_kpart=*/true,
        /*down_kparts=*/16);
    llaminar2::PerfStatsCollector::reset();

    auto gate_w = llaminar2::test::TestTensorFactory::createIQ3_SRandom(
        {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 6201);
    auto up_w = llaminar2::test::TestTensorFactory::createIQ3_SRandom(
        {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 6202);
    auto down_w = llaminar2::test::TestTensorFactory::createIQ3_SRandom(
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 6203);
    auto prepared = llaminar2::test::makeGpuPreparedFFNFixture(
        gate_w.get(),
        up_w.get(),
        down_w.get(),
        device,
        "test.cuda_moe.qwen36_shared_stage_verifier",
        llaminar2::ModelContextId{6200});

    auto make_params = [&](llaminar2::TensorBase *input,
                           llaminar2::TensorBase *output,
                           int seq_len,
                           bool grouped_verifier)
    {
        llaminar2::SharedExpertFFNStage::Params params;
        params.device_id = device;
        params.input = input;
        params.gate_w = gate_w.get();
        params.up_w = up_w.get();
        params.down_w = down_w.get();
        params.output = output;
        params.seq_len = seq_len;
        params.d_model = d_model;
        params.intermediate = intermediate;
        params.force_grouped_verifier_prefill_for_decode = grouped_verifier;
        params.force_decode_equivalent_verifier_prefill = false;
        params.disable_grouped_decode_shortcut = false;
        params.prepared_ref_gate = prepared.gate_ref;
        params.prepared_ref_up = prepared.up_ref;
        params.prepared_ref_down = prepared.down_ref;
        params.prepared_store = prepared.store.get();
        return params;
    };

    auto make_stage = [&](llaminar2::SharedExpertFFNStage::Params params)
    {
        auto stage = std::make_unique<llaminar2::SharedExpertFFNStage>(params);
        stage->setGPUStream(stream_);
        stage->setMoEKernelForTesting(cuda_kernel_);
        return stage;
    };

    /*
     * Allocate the production stage's declared workspace once at the maximum
     * verifier bucket.  Both sides of the comparison reuse it so differences
     * cannot be explained by missing scratch, first-use allocation, or an
     * accidental change to a non-workspace-backed path.
     */
    auto planning_input = makeZeros({4u, static_cast<size_t>(d_model)});
    auto planning_output = makeZeros({4u, static_cast<size_t>(d_model)});
    auto planning_stage = make_stage(
        make_params(planning_input.get(), planning_output.get(), 4, true));
    auto reqs = planning_stage->getWorkspaceRequirements(4, d_model, intermediate);
    auto stage_workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
        device,
        reqs.total_bytes_with_alignment() + 4 * 1024 * 1024);
    ASSERT_TRUE(stage_workspace->allocate(reqs))
        << "SharedExpertFFNStage Qwen3.6 verifier workspace";

    llaminar2::CUDADeviceContext ctx(device, 0);

    auto expect_bitwise_equal =
        [](const std::vector<float> &actual,
           const std::vector<float> &expected,
           const std::string &label)
    {
        ASSERT_EQ(actual.size(), expected.size()) << label;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            if (actual[i] != expected[i])
            {
                ADD_FAILURE()
                    << label
                    << " first differing element=" << i
                    << " actual=" << actual[i]
                    << " expected=" << expected[i]
                    << " abs_diff=" << std::abs(actual[i] - expected[i]);
                return;
            }
        }
    };

    for (int seq_len : {2, 3, 4})
    {
        std::vector<float> input_values(static_cast<size_t>(seq_len) * d_model);
        for (size_t i = 0; i < input_values.size(); ++i)
        {
            input_values[i] =
                0.017f * std::sin(0.0041f * static_cast<float>(i + 13)) -
                0.011f * std::cos(0.0063f * static_cast<float>(i + 29)) +
                0.0008f * static_cast<float>(static_cast<int>(i % 31) - 15);
        }

        auto grouped_input = makeTensor(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
            input_values);
        auto grouped_output = makeZeros(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        ASSERT_TRUE(grouped_input->ensureOnDevice(device, stream_));
        ASSERT_TRUE(grouped_output->ensureOnDevice(device, stream_));

        auto grouped_stage = make_stage(
            make_params(grouped_input.get(), grouped_output.get(), seq_len, true));
        grouped_stage->bindWorkspace(stage_workspace.get());
        ASSERT_TRUE(grouped_stage->usesGroupedVerifierPrefillRouteForTesting())
            << "The left side must exercise the production grouped verifier route";
        ASSERT_TRUE(grouped_stage->execute(&ctx))
            << "grouped shared verifier stage failed at seq_len=" << seq_len;
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        const auto grouped_host = copyCudaFP32TensorToHost(grouped_output, stream_);

        std::vector<float> serial_host(static_cast<size_t>(seq_len) * d_model);
        for (int row = 0; row < seq_len; ++row)
        {
            const auto row_offset = static_cast<size_t>(row) * d_model;
            std::vector<float> row_values(static_cast<size_t>(d_model));
            std::copy_n(input_values.data() + row_offset, d_model, row_values.data());
            auto row_input = makeTensor({1u, static_cast<size_t>(d_model)}, row_values);
            auto row_output = makeZeros({1u, static_cast<size_t>(d_model)});
            ASSERT_TRUE(row_input->ensureOnDevice(device, stream_));
            ASSERT_TRUE(row_output->ensureOnDevice(device, stream_));

            auto row_stage = make_stage(
                make_params(row_input.get(), row_output.get(), 1, false));
            row_stage->bindWorkspace(stage_workspace.get());
            ASSERT_TRUE(row_stage->usesGroupedDecodeForTesting())
                << "The right side must exercise CUDA's ordinary M=1 grouped table decode route";
            ASSERT_TRUE(row_stage->execute(&ctx))
                << "serial grouped shared decode stage failed at seq_len="
                << seq_len << " row=" << row;
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

            const auto row_host = copyCudaFP32TensorToHost(row_output, stream_);
            std::copy_n(row_host.data(), d_model, serial_host.data() + row_offset);
            row_stage->unbindWorkspace();
        }

        expect_bitwise_equal(
            grouped_host,
            serial_host,
            "SharedExpertFFNStage Qwen3.6 IQ3_S grouped verifier vs serial grouped decode seq_len=" +
                std::to_string(seq_len));
        grouped_stage->unbindWorkspace();
    }

    const auto records =
        llaminar2::PerfStatsCollector::snapshot({"kernel", "mtp"});
    auto has_counter = [&](const char *domain,
                           const char *name,
                           const char *route,
                           const char *active_slots) -> bool
    {
        return std::any_of(
            records.begin(),
            records.end(),
            [&](const llaminar2::PerfStatRecord &record)
            {
                if (record.domain != domain || record.name != name)
                    return false;
                auto tag_equals = [&](const char *key, const char *value)
                {
                    const auto it = record.tags.find(key);
                    return it != record.tags.end() && it->second == value;
                };
                return (!route || tag_equals("route", route)) &&
                       (!active_slots || tag_equals("active_slots", active_slots));
            });
    };
    EXPECT_TRUE(has_counter(
        "mtp",
        "moe_shared_grouped_decode_equivalent_verifier_prefill_rows",
        "grouped_table_prefill",
        nullptr))
        << "The grouped side must not silently leave the grouped verifier route\n"
        << llaminar2::PerfStatsCollector::summaryString({"kernel", "mtp"});
    EXPECT_TRUE(has_counter(
        "kernel",
        "cuda_moe_grouped_decode_gateup_calls",
        "kpart",
        "1"))
        << "The serial side must exercise grouped table gate/up decode\n"
        << llaminar2::PerfStatsCollector::summaryString({"kernel", "mtp"});
    EXPECT_TRUE(has_counter(
        "kernel",
        "cuda_moe_grouped_decode_down_calls",
        "kpart",
        "1"))
        << "The serial side must exercise grouped table down decode\n"
        << llaminar2::PerfStatsCollector::summaryString({"kernel", "mtp"});

    planning_stage->unbindWorkspace();
    llaminar2::PerfStatsCollector::reset();
#endif
}

TEST_F(Test__CUDAMoEKernel, SharedExpertGateAddEffectiveSeqLenZeroesPaddedRowsAcrossGraphReplay)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int bucket_seq_len = 4;
    constexpr int replay_real_seq_len = 2;
    constexpr int d_model = 8;
    const auto device = llaminar2::DeviceId::cuda(0);

    std::vector<float> input_values(bucket_seq_len * d_model);
    std::vector<float> gate_values(d_model);
    std::vector<float> shared_values(bucket_seq_len * d_model);
    std::vector<float> residual_values(bucket_seq_len * d_model);
    for (int i = 0; i < bucket_seq_len * d_model; ++i)
    {
        input_values[static_cast<size_t>(i)] = 0.03f * static_cast<float>((i % 11) - 5);
        shared_values[static_cast<size_t>(i)] = -0.7f + 0.04f * static_cast<float>(i);
        residual_values[static_cast<size_t>(i)] = 0.9f - 0.02f * static_cast<float>(i);
    }
    for (int i = 0; i < d_model; ++i)
        gate_values[static_cast<size_t>(i)] = 0.025f * static_cast<float>(i - 3);

    auto input_cuda = makeTensor({bucket_seq_len, d_model}, input_values);
    auto gate_cuda = makeTensor({d_model}, gate_values);
    auto shared_cuda = makeTensor({bucket_seq_len, d_model}, shared_values);
    auto residual_cuda = makeTensor({bucket_seq_len, d_model}, residual_values);
    auto combined_cuda = makeTensor({bucket_seq_len, d_model},
                                    std::vector<float>(bucket_seq_len * d_model, 123.0f));

    auto input_cpu = makeTensor({bucket_seq_len, d_model}, input_values);
    auto gate_cpu = makeTensor({d_model}, gate_values);
    auto shared_cpu = makeTensor({bucket_seq_len, d_model}, shared_values);
    auto residual_cpu = makeTensor({bucket_seq_len, d_model}, residual_values);
    auto combined_cpu = makeTensor({bucket_seq_len, d_model},
                                   std::vector<float>(bucket_seq_len * d_model, 0.0f));
    cpu_kernel_->sharedExpertGateAddFromTensors(
        input_cpu.get(), gate_cpu.get(), shared_cpu.get(), residual_cpu.get(), combined_cpu.get(),
        replay_real_seq_len, d_model);

    ASSERT_TRUE(input_cuda->ensureOnDevice(device, stream_));
    ASSERT_TRUE(gate_cuda->ensureOnDevice(device, stream_));
    ASSERT_TRUE(shared_cuda->ensureOnDevice(device, stream_));
    ASSERT_TRUE(residual_cuda->ensureOnDevice(device, stream_));
    ASSERT_TRUE(combined_cuda->ensureOnDevice(device, stream_));

    int *device_effective_seq_len = nullptr;
    ASSERT_EQ(cudaMalloc(&device_effective_seq_len, sizeof(int)), cudaSuccess);
    int effective_seq_len = bucket_seq_len;
    ASSERT_EQ(cudaMemcpyAsync(device_effective_seq_len,
                              &effective_seq_len,
                              sizeof(int),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_gate = cuda_kernel_->sharedExpertGateAddFromTensorsEffectiveSeqLen(
        input_cuda.get(), gate_cuda.get(), shared_cuda.get(),
        residual_cuda.get(), combined_cuda.get(),
        bucket_seq_len, d_model, device_effective_seq_len);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_gate);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);

    cudaGraphExec_t executable = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);

    effective_seq_len = replay_real_seq_len;
    ASSERT_EQ(cudaMemcpyAsync(device_effective_seq_len,
                              &effective_seq_len,
                              sizeof(int),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);

    /*
     * Replay starts from nonzero tail values. The effective-length kernel must
     * overwrite padded rows to zero so stale full-bucket graph state cannot
     * escape into following stages.
     */
    std::vector<float> shared_replay = shared_values;
    std::vector<float> combined_replay(bucket_seq_len * d_model, 77.0f);
    ASSERT_EQ(cudaMemcpyAsync(shared_cuda->gpu_data_ptr(),
                              shared_replay.data(),
                              shared_replay.size() * sizeof(float),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(combined_cuda->gpu_data_ptr(),
                              combined_replay.data(),
                              combined_replay.size() * sizeof(float),
                              cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const float *shared_actual = shared_cuda->data();
    const float *combined_actual = combined_cuda->data();
    for (int row = 0; row < bucket_seq_len; ++row)
    {
        for (int col = 0; col < d_model; ++col)
        {
            const size_t idx = static_cast<size_t>(row) * d_model + col;
            if (row < replay_real_seq_len)
            {
                EXPECT_NEAR(shared_actual[idx], shared_cpu->data()[idx], 1.0e-5f)
                    << "shared row " << row << " col " << col;
                EXPECT_NEAR(combined_actual[idx], combined_cpu->data()[idx], 1.0e-5f)
                    << "combined row " << row << " col " << col;
            }
            else
            {
                EXPECT_EQ(shared_actual[idx], 0.0f)
                    << "padded shared row " << row << " col " << col;
                EXPECT_EQ(combined_actual[idx], 0.0f)
                    << "padded combined row " << row << " col " << col;
            }
        }
    }

    ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    ASSERT_EQ(cudaFree(device_effective_seq_len), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, PrepareExpertGroupsMatchesCPUGroups)
{
    constexpr int seq_len = 3;
    constexpr int top_k = 2;
    constexpr int num_experts = 4;
    constexpr int d_model = 3;

    auto routing_indices = makeTensor({seq_len, top_k}, {
                                                            1.0f,
                                                            2.0f,
                                                            0.0f,
                                                            1.0f,
                                                            3.0f,
                                                            1.0f,
                                                            2.0f,
                                                            0.0f,
                                                        });
    auto routing_weights = makeTensor({seq_len, top_k}, {
                                                            0.7f,
                                                            0.3f,
                                                            0.6f,
                                                            0.4f,
                                                            0.8f,
                                                            0.2f,
                                                            0.55f,
                                                            0.45f,
                                                        });
    auto hidden = makeTensor({seq_len, d_model}, {
                                                     1.0f,
                                                     1.1f,
                                                     1.2f,
                                                     2.0f,
                                                     2.1f,
                                                     2.2f,
                                                     3.0f,
                                                     3.1f,
                                                     3.2f,
                                                     4.0f,
                                                     4.1f,
                                                     4.2f,
                                                 });

    ASSERT_TRUE(cuda_kernel_->prepareExpertGroups(routing_indices.get(), routing_weights.get(),
                                                  seq_len, num_experts, top_k));
    ASSERT_TRUE(cpu_kernel_->prepareExpertGroups(routing_indices.get(), routing_weights.get(),
                                                 seq_len, num_experts, top_k));

    for (int expert = 0; expert < num_experts; ++expert)
        EXPECT_EQ(cuda_kernel_->getExpertTokenCount(expert), cpu_kernel_->getExpertTokenCount(expert));

    auto cuda_expert_one = makeZeros({static_cast<size_t>(cuda_kernel_->getExpertTokenCount(1)), d_model});
    auto cpu_expert_one = makeZeros({static_cast<size_t>(cpu_kernel_->getExpertTokenCount(1)), d_model});
    cuda_kernel_->gatherExpertBatch(hidden.get(), cuda_expert_one.get(), 1, d_model);
    cpu_kernel_->gatherExpertBatch(hidden.get(), cpu_expert_one.get(), 1, d_model);
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
    expectNearArray(cuda_expert_one->data(), cpu_expert_one->data(), cuda_expert_one->numel());
}

TEST_F(Test__CUDAMoEKernel, PrepareExpertGroupsAsyncAcceptsDeviceRoutingTensors)
{
    constexpr int seq_len = 4;
    constexpr int top_k = 2;
    constexpr int num_experts = 4;

    auto routing_indices = makeTensor({seq_len, top_k}, {
                                                            1.0f,
                                                            2.0f,
                                                            0.0f,
                                                            1.0f,
                                                            3.0f,
                                                            1.0f,
                                                            2.0f,
                                                            0.0f,
                                                        });
    auto routing_weights = makeTensor({seq_len, top_k}, {
                                                            0.7f,
                                                            0.3f,
                                                            0.6f,
                                                            0.4f,
                                                            0.8f,
                                                            0.2f,
                                                            0.55f,
                                                            0.45f,
                                                        });

    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(routing_indices.get(), routing_weights.get(),
                                                       seq_len, num_experts, top_k));
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, SmallFloatGroupingEmitsCompactActiveExperts)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int seq_len = 4;
    constexpr int top_k = 4;
    constexpr int num_experts = 16;
    constexpr int total_slots = seq_len * top_k;
    constexpr int max_active_experts = total_slots;

    const std::vector<float> routing_indices = {
        7.0f, 3.0f, 7.0f, 1.0f,
        5.0f, 1.0f, 9.0f, 3.0f,
        7.0f, 5.0f, 11.0f, 1.0f,
        3.0f, 9.0f, 11.0f, 5.0f,
    };
    std::vector<float> routing_weights(static_cast<size_t>(total_slots));
    for (int i = 0; i < total_slots; ++i)
        routing_weights[static_cast<size_t>(i)] = 0.05f * static_cast<float>(i + 1);

    float *d_indices = nullptr;
    float *d_weights = nullptr;
    int *d_counts = nullptr;
    int *d_offsets = nullptr;
    int *d_grouped_tokens = nullptr;
    int *d_original_to_grouped = nullptr;
    int *d_original_expert_ids = nullptr;
    float *d_grouped_weights = nullptr;
    int *d_active = nullptr;
    ASSERT_EQ(cudaMalloc(&d_indices, routing_indices.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_weights, routing_weights.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_counts, num_experts * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_offsets, num_experts * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_grouped_tokens, total_slots * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_original_to_grouped, total_slots * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_original_expert_ids, total_slots * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_grouped_weights, total_slots * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_active, max_active_experts * sizeof(int)), cudaSuccess);

    ASSERT_EQ(cudaMemcpyAsync(d_indices, routing_indices.data(),
                              routing_indices.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_weights, routing_weights.data(),
                              routing_weights.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cudaMoE_group_tokens_small_float(
        d_indices, d_weights, d_counts, d_offsets, d_grouped_tokens, d_original_to_grouped, d_original_expert_ids, d_grouped_weights,
        d_active, total_slots, num_experts, top_k, max_active_experts, 0, stream_));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    std::vector<int> counts(num_experts);
    std::vector<int> offsets(num_experts);
    std::vector<int> grouped_tokens(total_slots);
    std::vector<int> original_to_grouped(total_slots);
    std::vector<int> original_expert_ids(total_slots);
    std::vector<float> grouped_weights(total_slots);
    std::vector<int> active(max_active_experts);
    ASSERT_EQ(cudaMemcpy(counts.data(), d_counts, counts.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(offsets.data(), d_offsets, offsets.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(grouped_tokens.data(), d_grouped_tokens, grouped_tokens.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(original_to_grouped.data(), d_original_to_grouped, original_to_grouped.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(original_expert_ids.data(), d_original_expert_ids, original_expert_ids.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(grouped_weights.data(), d_grouped_weights, grouped_weights.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(active.data(), d_active, active.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    std::vector<int> expected_counts(num_experts, 0);
    for (float expert : routing_indices)
        ++expected_counts[static_cast<int>(expert)];

    std::vector<int> expected_offsets(num_experts, 0);
    int running = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        expected_offsets[expert] = running;
        running += expected_counts[expert];
    }
    EXPECT_EQ(counts, expected_counts);
    EXPECT_EQ(offsets, expected_offsets);

    std::vector<int> expected_active;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        if (expected_counts[expert] > 0)
            expected_active.push_back(expert);
    }
    ASSERT_LE(expected_active.size(), active.size());
    for (size_t i = 0; i < expected_active.size(); ++i)
        EXPECT_EQ(active[i], expected_active[i]) << "active slot " << i;
    for (size_t i = expected_active.size(); i < active.size(); ++i)
        EXPECT_EQ(active[i], -1) << "inactive slot " << i;

    for (int expert = 0; expert < num_experts; ++expert)
    {
        int local = 0;
        for (int slot = 0; slot < total_slots; ++slot)
        {
            if (static_cast<int>(routing_indices[static_cast<size_t>(slot)]) != expert)
                continue;
            const int dest = offsets[expert] + local++;
            ASSERT_GE(dest, 0);
            ASSERT_LT(dest, total_slots);
            EXPECT_EQ(original_to_grouped[slot], dest);
            EXPECT_EQ(original_expert_ids[slot], expert);
            EXPECT_EQ(grouped_tokens[dest], slot / top_k);
            EXPECT_FLOAT_EQ(grouped_weights[dest], routing_weights[static_cast<size_t>(slot)]);
        }
    }

    EXPECT_FALSE(cudaMoE_group_tokens_small_float(
        d_indices, d_weights, d_counts, d_offsets, d_grouped_tokens, d_original_to_grouped, d_original_expert_ids, d_grouped_weights,
        d_active, total_slots, num_experts, top_k, max_active_experts, 0, nullptr));

    cudaFree(d_indices);
    cudaFree(d_weights);
    cudaFree(d_counts);
    cudaFree(d_offsets);
    cudaFree(d_grouped_tokens);
    cudaFree(d_original_to_grouped);
    cudaFree(d_original_expert_ids);
    cudaFree(d_grouped_weights);
    cudaFree(d_active);
#endif
}

TEST_F(Test__CUDAMoEKernel, DeterministicScatterPreservesStableOrderAcrossChunks)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int seq_len = 80;
    constexpr int top_k = 8;
    constexpr int total_slots = seq_len * top_k;
    constexpr int num_experts = 40;

    std::vector<int> routing_indices(total_slots);
    std::vector<float> routing_weights(total_slots);
    for (int slot = 0; slot < total_slots; ++slot)
    {
        // Crosses multiple 256-thread chunks and leaves experts 32..39 empty,
        // exercising both stable within-chunk ordering and the zero-match skip.
        int expert = (slot * 17 + slot / 5) % 32;
        if (slot % 19 == 0)
            expert = 7;
        if (slot % 23 == 0)
            expert = 31;
        routing_indices[static_cast<size_t>(slot)] = expert;
        routing_weights[static_cast<size_t>(slot)] = 0.001f * static_cast<float>(slot + 3);
    }

    int *d_indices = nullptr;
    float *d_weights = nullptr;
    int *d_counts = nullptr;
    int *d_offsets = nullptr;
    int *d_grouped_tokens = nullptr;
    int *d_original_to_grouped = nullptr;
    int *d_original_expert_ids = nullptr;
    float *d_grouped_weights = nullptr;
    ASSERT_EQ(cudaMalloc(&d_indices, routing_indices.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_weights, routing_weights.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_counts, num_experts * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_offsets, num_experts * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_grouped_tokens, total_slots * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_original_to_grouped, total_slots * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_original_expert_ids, total_slots * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_grouped_weights, total_slots * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpyAsync(d_indices, routing_indices.data(),
                              routing_indices.size() * sizeof(int),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_weights, routing_weights.data(),
                              routing_weights.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_counts, 0, num_experts * sizeof(int), stream_), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_grouped_tokens, 0xff, total_slots * sizeof(int), stream_), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_original_to_grouped, 0xff, total_slots * sizeof(int), stream_), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_original_expert_ids, 0xff, total_slots * sizeof(int), stream_), cudaSuccess);

    ASSERT_TRUE(cudaMoE_count_per_expert(
        d_indices, d_counts, total_slots, num_experts, 0, stream_));
    ASSERT_TRUE(cudaMoE_exclusive_scan(
        d_counts, d_offsets, num_experts, 0, stream_));
    ASSERT_TRUE(cudaMoE_scatter_tokens_deterministic(
        d_indices, d_weights, d_offsets, d_counts,
        d_grouped_tokens, d_original_to_grouped, d_original_expert_ids, d_grouped_weights,
        total_slots, top_k, num_experts, 0, stream_));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    std::vector<int> counts(num_experts);
    std::vector<int> offsets(num_experts);
    std::vector<int> grouped_tokens(total_slots);
    std::vector<int> original_to_grouped(total_slots);
    std::vector<int> original_expert_ids(total_slots);
    std::vector<float> grouped_weights(total_slots);
    ASSERT_EQ(cudaMemcpy(counts.data(), d_counts, counts.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(offsets.data(), d_offsets, offsets.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(grouped_tokens.data(), d_grouped_tokens, grouped_tokens.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(original_to_grouped.data(), d_original_to_grouped, original_to_grouped.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(original_expert_ids.data(), d_original_expert_ids, original_expert_ids.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(grouped_weights.data(), d_grouped_weights, grouped_weights.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    std::vector<int> expected_counts(num_experts, 0);
    for (int expert : routing_indices)
        ++expected_counts[static_cast<size_t>(expert)];

    std::vector<int> expected_offsets(num_experts, 0);
    int running = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        expected_offsets[static_cast<size_t>(expert)] = running;
        running += expected_counts[static_cast<size_t>(expert)];
    }
    EXPECT_EQ(counts, expected_counts);
    EXPECT_EQ(offsets, expected_offsets);

    for (int expert = 0; expert < num_experts; ++expert)
    {
        int local = 0;
        for (int slot = 0; slot < total_slots; ++slot)
        {
            if (routing_indices[static_cast<size_t>(slot)] != expert)
                continue;
            const int dest = offsets[static_cast<size_t>(expert)] + local++;
            ASSERT_GE(dest, 0);
            ASSERT_LT(dest, total_slots);
            EXPECT_EQ(original_to_grouped[static_cast<size_t>(slot)], dest);
            EXPECT_EQ(original_expert_ids[static_cast<size_t>(slot)], expert);
            EXPECT_EQ(grouped_tokens[static_cast<size_t>(dest)], slot / top_k);
            EXPECT_FLOAT_EQ(grouped_weights[static_cast<size_t>(dest)],
                            routing_weights[static_cast<size_t>(slot)]);
        }
    }

    for (int expert = 32; expert < num_experts; ++expert)
        EXPECT_EQ(counts[static_cast<size_t>(expert)], 0);

    cudaFree(d_indices);
    cudaFree(d_weights);
    cudaFree(d_counts);
    cudaFree(d_offsets);
    cudaFree(d_grouped_tokens);
    cudaFree(d_original_to_grouped);
    cudaFree(d_original_expert_ids);
    cudaFree(d_grouped_weights);
#endif
}

TEST_F(Test__CUDAMoEKernel, UploadGroupedDescriptorTablesAcceptValidCudaDescriptors)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int d_model = 128;
    constexpr int intermediate = 128;
    constexpr int num_experts = 2;
    constexpr size_t descriptor_bytes = 4096;

    std::vector<CudaAllocation> payloads;
    std::vector<CudaAllocation> scales;
    payloads.reserve(static_cast<size_t>(num_experts * 3));
    scales.reserve(static_cast<size_t>(num_experts * 3));

    auto add_desc = [&](int rows, int cols)
    {
        payloads.emplace_back(descriptor_bytes);
        scales.emplace_back(descriptor_bytes);
        return makeCudaNativeDesc(payloads.back(), scales.back(), rows, cols);
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    down_descs.reserve(num_experts);
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        down_descs.push_back(add_desc(d_model, intermediate));
        gate_descs.push_back(add_desc(intermediate, d_model));
        up_descs.push_back(add_desc(intermediate, d_model));
    }

    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    EXPECT_GE(down_table, 0);

    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    EXPECT_GE(gateup_table, 0);

    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimePrefillDescriptorMaterializationUsesActiveRuntimeBank)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int d_model = 64;
    constexpr int intermediate = 96;
    constexpr uint32_t active_epoch = 17;

    llaminar2::DeviceMoELayerRuntime host_runtime =
        makeAllLocalRuntime(num_experts, top_k, active_epoch);
    host_runtime.active_bank = 1;
    host_runtime.active_epoch = active_epoch;

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> expected_gate(num_experts);
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> expected_up(num_experts);
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> expected_down(num_experts);

    for (int expert = 0; expert < num_experts; ++expert)
    {
        const uintptr_t inactive_base = 0x52000000u + static_cast<uintptr_t>(expert) * 0x10000u;
        host_runtime.banks[0].experts[expert].gate =
            makePrefillRuntimeDesc(inactive_base + 0x1000u, intermediate, d_model, 4);
        host_runtime.banks[0].experts[expert].up =
            makePrefillRuntimeDesc(inactive_base + 0x2000u, intermediate, d_model, 5);
        host_runtime.banks[0].experts[expert].down =
            makePrefillRuntimeDesc(inactive_base + 0x3000u, d_model, intermediate, 6);

        const uintptr_t active_base = 0x72000000u + static_cast<uintptr_t>(expert) * 0x10000u;
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
        active_desc.flags = llaminar2::toMoEExpertFlags(
            llaminar2::DeviceMoEExpertFlags::Valid |
            llaminar2::DeviceMoEExpertFlags::Resident |
            llaminar2::DeviceMoEExpertFlags::LocalCompute);
        active_desc.gate = expected_gate[static_cast<size_t>(expert)];
        active_desc.up = expected_up[static_cast<size_t>(expert)];
        active_desc.down = expected_down[static_cast<size_t>(expert)];
    }

    CudaAllocation device_runtime(sizeof(llaminar2::DeviceMoELayerRuntime));
    CudaAllocation device_gate(num_experts * sizeof(llaminar2::DeviceNativeVNNIMatrixDesc));
    CudaAllocation device_up(num_experts * sizeof(llaminar2::DeviceNativeVNNIMatrixDesc));
    CudaAllocation device_down(num_experts * sizeof(llaminar2::DeviceNativeVNNIMatrixDesc));

    ASSERT_EQ(cudaMemcpyAsync(device_runtime.get(), &host_runtime, sizeof(host_runtime),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_TRUE(cudaMoE_materialize_runtime_prefill_descriptor_tables(
        device_runtime.get(),
        static_cast<llaminar2::DeviceNativeVNNIMatrixDesc *>(device_gate.get()),
        static_cast<llaminar2::DeviceNativeVNNIMatrixDesc *>(device_up.get()),
        static_cast<llaminar2::DeviceNativeVNNIMatrixDesc *>(device_down.get()),
        num_experts,
        0,
        stream_));
    EXPECT_FALSE(cudaMoE_materialize_runtime_prefill_descriptor_tables(
        device_runtime.get(),
        static_cast<llaminar2::DeviceNativeVNNIMatrixDesc *>(device_gate.get()),
        static_cast<llaminar2::DeviceNativeVNNIMatrixDesc *>(device_up.get()),
        static_cast<llaminar2::DeviceNativeVNNIMatrixDesc *>(device_down.get()),
        num_experts,
        0,
        nullptr));

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> actual_gate(num_experts);
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> actual_up(num_experts);
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> actual_down(num_experts);
    ASSERT_EQ(cudaMemcpyAsync(actual_gate.data(), device_gate.get(),
                              actual_gate.size() * sizeof(llaminar2::DeviceNativeVNNIMatrixDesc),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(actual_up.data(), device_up.get(),
                              actual_up.size() * sizeof(llaminar2::DeviceNativeVNNIMatrixDesc),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(actual_down.data(), device_down.get(),
                              actual_down.size() * sizeof(llaminar2::DeviceNativeVNNIMatrixDesc),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

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
#endif
}

TEST_F(Test__CUDAMoEKernel, FixedTopologyRuntimeGroupedPrefillUsesCompactActiveExpertGrid)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    ScopedCudaMoEPrefillConfig prefill_config;
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);
    llaminar2::PerfStatsCollector::reset();

    constexpr int seq_len = 4;
    constexpr int top_k = 4;
    constexpr int num_experts = 16;
    constexpr int d_model = 32;
    constexpr int intermediate = 32;
    constexpr size_t descriptor_bytes = 4096;

    std::vector<CudaAllocation> payloads;
    std::vector<CudaAllocation> scales;
    payloads.reserve(static_cast<size_t>(num_experts * 3));
    scales.reserve(static_cast<size_t>(num_experts * 3));

    auto add_desc = [&](int rows, int cols)
    {
        payloads.emplace_back(descriptor_bytes);
        scales.emplace_back(descriptor_bytes);
        EXPECT_EQ(cudaMemsetAsync(payloads.back().get(), 0, descriptor_bytes, stream_), cudaSuccess);

        const int scale_count = rows * (cols / 32);
        std::vector<uint16_t> host_scales(static_cast<size_t>(scale_count), 0x3c00u);
        EXPECT_EQ(cudaMemcpyAsync(scales.back().get(), host_scales.data(),
                                  host_scales.size() * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        return makeCudaNativeDesc(payloads.back(), scales.back(), rows, cols);
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    down_descs.reserve(num_experts);
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        down_descs.push_back(add_desc(d_model, intermediate));
        gate_descs.push_back(add_desc(intermediate, d_model));
        up_descs.push_back(add_desc(intermediate, d_model));
    }

    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);
    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);

    std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
    for (size_t i = 0; i < hidden_values.size(); ++i)
        hidden_values[i] = 0.01f * static_cast<float>(static_cast<int>(i % 13) - 6);
    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto output = makeZeros({seq_len, d_model});

    auto routing_indices = makeTensor({seq_len, top_k}, {
                                                            7.0f,
                                                            3.0f,
                                                            7.0f,
                                                            1.0f,
                                                            5.0f,
                                                            1.0f,
                                                            9.0f,
                                                            3.0f,
                                                            7.0f,
                                                            5.0f,
                                                            11.0f,
                                                            1.0f,
                                                            3.0f,
                                                            9.0f,
                                                            11.0f,
                                                            5.0f,
                                                        });
    auto routing_weights = makeTensor({seq_len, top_k}, {
                                                            0.40f,
                                                            0.30f,
                                                            0.20f,
                                                            0.10f,
                                                            0.45f,
                                                            0.25f,
                                                            0.20f,
                                                            0.10f,
                                                            0.35f,
                                                            0.30f,
                                                            0.20f,
                                                            0.15f,
                                                            0.50f,
                                                            0.20f,
                                                            0.20f,
                                                            0.10f,
                                                        });

    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
        routing_indices.get(), routing_weights.get(), seq_len, num_experts, top_k));
    ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const float *output_data = output->data();
    for (size_t i = 0; i < output->numel(); ++i)
        ASSERT_TRUE(std::isfinite(output_data[i])) << "output element " << i;

    const auto grouping_records =
        llaminar2::PerfStatsCollector::snapshot({"kernel.cuda_moe_small_prefill_grouping_calls"});
    ASSERT_FALSE(grouping_records.empty());

    const auto grid_records =
        llaminar2::PerfStatsCollector::snapshot({"kernel.cuda_moe_grouped_prefill_active_expert_grid_calls"});
    ASSERT_FALSE(grid_records.empty());
    EXPECT_EQ(grid_records.front().tags.at("active_expert_slots"), std::to_string(seq_len * top_k));
    EXPECT_EQ(grid_records.front().tags.at("num_experts"), std::to_string(num_experts));
    EXPECT_EQ(grid_records.front().tags.at("tile_m"), "2")
        << "seq_len=4 verifier-style grouped prefill should use the tuned tiny-M tile by default";
    EXPECT_EQ(grid_records.front().tags.at("tile_n"), "64")
        << "compact verifier rows use the small-N expert tile while "
           "max_tokens_per_expert <= 4";
    expectPrefillSwiGLUPathRecord("fused", seq_len, top_k, num_experts, 2);

    llaminar2::PerfStatsCollector::reset();
#endif
}

TEST_F(Test__CUDAMoEKernel, FixedTopologyRuntimeGroupedPrefillMatchesExistingPrefillPath)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";
#ifdef ENABLE_PIPELINE_SNAPSHOTS
    GTEST_SKIP() << "Fixed-topology prefill capture path is release-only when snapshots are disabled";
#endif

    const auto device = llaminar2::DeviceId::cuda(0);
    constexpr int seq_len = 2;
    constexpr int d_model = 64;
    constexpr int intermediate = 32;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int total_slots = seq_len * top_k;

    struct ScopedGroupedPrefillFlag
    {
        bool old_value;
        explicit ScopedGroupedPrefillFlag(bool value)
            : old_value(llaminar2::mutableDebugEnv().gpu_moe.grouped_prefill)
        {
            llaminar2::mutableDebugEnv().gpu_moe.grouped_prefill = value;
        }
        ~ScopedGroupedPrefillFlag()
        {
            llaminar2::mutableDebugEnv().gpu_moe.grouped_prefill = old_value;
        }
    } grouped_prefill_flag(true);
    ScopedEnv perf_stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    llaminar2::PerfStatsCollector::reset();

    std::vector<std::unique_ptr<llaminar2::TensorBase>> gate_weight_tensors;
    std::vector<std::unique_ptr<llaminar2::TensorBase>> up_weight_tensors;
    std::vector<std::unique_ptr<llaminar2::TensorBase>> down_weight_tensors;
    std::vector<llaminar2::test::GpuPreparedGemm> gate_kernels;
    std::vector<llaminar2::test::GpuPreparedGemm> up_kernels;
    std::vector<llaminar2::test::GpuPreparedGemm> down_kernels;
    gate_weight_tensors.reserve(num_experts);
    up_weight_tensors.reserve(num_experts);
    down_weight_tensors.reserve(num_experts);
    gate_kernels.reserve(num_experts);
    up_kernels.reserve(num_experts);
    down_kernels.reserve(num_experts);

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto gate_weights = llaminar2::test::TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 100 + expert_id);
        auto up_weights = llaminar2::test::TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 200 + expert_id);
        auto down_weights = llaminar2::test::TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 300 + expert_id);

        gate_kernels.push_back(llaminar2::test::makeGpuPreparedGemm(
            gate_weights.get(), device,
            "test.cuda.runtime_prefill.gate." + std::to_string(expert_id),
            llaminar2::ModelContextId{9500}));
        up_kernels.push_back(llaminar2::test::makeGpuPreparedGemm(
            up_weights.get(), device,
            "test.cuda.runtime_prefill.up." + std::to_string(expert_id),
            llaminar2::ModelContextId{9500}));
        down_kernels.push_back(llaminar2::test::makeGpuPreparedGemm(
            down_weights.get(), device,
            "test.cuda.runtime_prefill.down." + std::to_string(expert_id),
            llaminar2::ModelContextId{9500}));

        gate_weight_tensors.push_back(std::move(gate_weights));
        up_weight_tensors.push_back(std::move(up_weights));
        down_weight_tensors.push_back(std::move(down_weights));
    }

    auto input = llaminar2::test::TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)}, -1.0f, 1.0f, 77);
    ASSERT_TRUE(input->ensureOnDevice(device));

    const float routing_indices_data[total_slots] = {
        0.0f, 1.0f,
        2.0f, 3.0f};
    const float routing_weights_data[total_slots] = {
        0.70f, 0.30f,
        0.60f, 0.40f};
    auto routing_indices = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    auto routing_weights = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots, routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots, routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    auto reference_output = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto runtime_output = llaminar2::test::TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(reference_output->ensureOnDevice(device));
    ASSERT_TRUE(runtime_output->ensureOnDevice(device));

    auto gate_exps = llaminar2::test::TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(num_experts * intermediate), static_cast<size_t>(d_model)}, 401);
    auto up_exps = llaminar2::test::TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(num_experts * intermediate), static_cast<size_t>(d_model)}, 402);
    auto down_exps = llaminar2::test::TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(num_experts * d_model), static_cast<size_t>(intermediate)}, 403);

    auto make_params = [&](llaminar2::TensorBase *output, llaminar2::IMoERuntimeTable *runtime_table)
    {
        llaminar2::MoEExpertComputeStage::Params params;
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
            params.prepared_gate_gemm[static_cast<size_t>(expert_id)] =
                gate_kernels[static_cast<size_t>(expert_id)].kernel;
            params.prepared_up_gemm[static_cast<size_t>(expert_id)] =
                up_kernels[static_cast<size_t>(expert_id)].kernel;
            params.prepared_down_gemm[static_cast<size_t>(expert_id)] =
                down_kernels[static_cast<size_t>(expert_id)].kernel;
        }
        return params;
    };

    auto stage_workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
        device, 64 * 1024 * 1024);
    llaminar2::CUDADeviceContext ctx(device, 0);

    llaminar2::mutableDebugEnv().gpu_moe.grouped_prefill = false;
    llaminar2::MoEExpertComputeStage reference_stage(make_params(reference_output.get(), nullptr));
    ASSERT_TRUE(stage_workspace->allocate(
        reference_stage.getWorkspaceRequirements(seq_len, d_model, intermediate)));
    reference_stage.bindWorkspace(stage_workspace.get());
    ASSERT_TRUE(reference_stage.execute(&ctx));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    llaminar2::DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    llaminar2::MoERuntimeTable runtime_table(runtime_config);

    llaminar2::mutableDebugEnv().gpu_moe.grouped_prefill = true;
    llaminar2::MoEExpertComputeStage runtime_stage(make_params(runtime_output.get(), &runtime_table));
    ASSERT_TRUE(runtime_stage.isGraphCapturable());
    runtime_stage.bindWorkspace(stage_workspace.get());
    ASSERT_TRUE(runtime_stage.execute(&ctx));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    {
        const auto records = llaminar2::PerfStatsCollector::snapshot(
            {"kernel.cuda_moe_small_prefill_grouping_calls"});
        double small_grouping_calls = 0.0;
        for (const auto &record : records)
            small_grouping_calls += record.value;
        EXPECT_GT(small_grouping_calls, 0.0)
            << "runtime-scratch fixed-topology prefill should use fused small-M grouping";
    }
    {
        const auto records = llaminar2::PerfStatsCollector::snapshot(
            {"kernel.cuda_moe_grouped_prefill_active_expert_grid_calls"});
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
            << "runtime-scratch fixed-topology prefill should launch over compact active experts";
        EXPECT_TRUE(saw_verifier_tile)
            << "CUDA two-row verifier grouped prefill should use the M=2 kernel bucket";
    }
    llaminar2::PerfStatsCollector::reset();

    reference_output->transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE);
    runtime_output->transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const size_t output_count = static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
    std::vector<float> reference_values(reference_output->data(), reference_output->data() + output_count);
    std::vector<float> runtime_values(runtime_output->data(), runtime_output->data() + output_count);
    expectVectorsClose(runtime_values, reference_values,
                       /*min_cosine=*/0.99999,
                       /*max_relative_l2=*/0.001,
                       static_cast<size_t>(d_model),
                       /*min_row_cosine=*/0.99999,
                       /*max_row_relative_l2=*/0.0015,
                       /*max_row_kl=*/1.0e-6);
#endif
}

TEST_F(Test__CUDAMoEKernel, GroupedDecodeMatchesGroupedPrefillForSingleTokenNativeWeights)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedCudaMoEGemmConfig gemm_config;
    gemm_config.set(
        /*gateup_kpart=*/true,
        /*gateup_kparts=*/16,
        /*down_kpart=*/true,
        /*down_kparts=*/16);
    ScopedCudaMoEPrefillConfig prefill_config;
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);
    ASSERT_TRUE(llaminar2::debugEnv().gemm.cuda_moe_prefill_fuse_swiglu)
        << "CUDA grouped prefill fused SwiGLU should stay enabled by default once "
           "production-shaped decode/prefill parity is covered";

    constexpr int seq_len = 1;
    constexpr int top_k = 8;
    constexpr int num_experts = 8;
    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    const auto device = llaminar2::DeviceId::cuda(0);

    std::vector<std::unique_ptr<llaminar2::TensorBase>> owned_weights;
    std::vector<llaminar2::test::GpuPreparedGemm> prepared_weights;
    owned_weights.reserve(static_cast<size_t>(num_experts * 3));
    prepared_weights.reserve(static_cast<size_t>(num_experts * 3));

    auto add_prepared_desc = [&](int rows, int cols, int seed, const char *role, int expected_codebook)
    {
        std::unique_ptr<llaminar2::TensorBase> weight;
        if (expected_codebook == 13)
        {
            weight = llaminar2::test::TestTensorFactory::createIQ2_SRandom(
                {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                static_cast<unsigned>(seed));
        }
        else
        {
            weight = llaminar2::test::TestTensorFactory::createIQ4_NLRandom(
                {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                static_cast<unsigned>(seed));
        }
        auto *weight_ptr = weight.get();
        owned_weights.push_back(std::move(weight));
        prepared_weights.push_back(llaminar2::test::makeGpuPreparedGemm(
            weight_ptr,
            device,
            std::string("test.cuda_moe.") + role + "." + std::to_string(seed),
            llaminar2::ModelContextId{123000 + static_cast<uint64_t>(seed)}));

        llaminar2::DeviceNativeVNNIMatrixDesc desc{};
        EXPECT_TRUE(prepared_weights.back().kernel->exportNativeVNNIMatrixDesc(desc))
            << "failed to export native descriptor for " << role;
        EXPECT_EQ(desc.n, rows);
        EXPECT_EQ(desc.k, cols);
        EXPECT_EQ(desc.codebook_id, expected_codebook);
        return desc;
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    down_descs.reserve(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        gate_descs.push_back(add_prepared_desc(intermediate, d_model, 3100 + expert, "gate", 13));
        up_descs.push_back(add_prepared_desc(intermediate, d_model, 3200 + expert, "up", 13));
        down_descs.push_back(add_prepared_desc(d_model, intermediate, 3300 + expert, "down", 4));
    }

    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    std::vector<float> hidden_values(static_cast<size_t>(d_model));
    for (int i = 0; i < d_model; ++i)
        hidden_values[static_cast<size_t>(i)] =
            0.017f * static_cast<float>((i % 19) - 9) +
            0.003f * static_cast<float>((i % 7) - 3);
    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto hidden_bf16 = makeBF16Tensor({seq_len, d_model}, hidden_values);
    std::vector<float> narrow_hidden_values(static_cast<size_t>(d_model - 1), 0.05f);
    auto hidden_too_narrow = makeTensor({seq_len, d_model - 1}, narrow_hidden_values);

    const std::array<int, top_k> expert_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    const std::array<float, top_k> expert_weights = {
        0.19f, 0.17f, 0.15f, 0.14f, 0.13f, 0.10f, 0.07f, 0.05f};
    auto routing_indices = makeTensor(
        {seq_len, top_k},
        {static_cast<float>(expert_ids[0]), static_cast<float>(expert_ids[1]),
         static_cast<float>(expert_ids[2]), static_cast<float>(expert_ids[3]),
         static_cast<float>(expert_ids[4]), static_cast<float>(expert_ids[5]),
         static_cast<float>(expert_ids[6]), static_cast<float>(expert_ids[7])});
    auto routing_weights = makeTensor(
        {seq_len, top_k},
        {expert_weights[0], expert_weights[1], expert_weights[2], expert_weights[3],
         expert_weights[4], expert_weights[5], expert_weights[6], expert_weights[7]});

    auto decode_gate0 = makeZeros({intermediate});
    auto decode_gate1 = makeZeros({intermediate});
    auto decode_gate2 = makeZeros({intermediate});
    auto decode_gate3 = makeZeros({intermediate});
    auto decode_gate4 = makeZeros({intermediate});
    auto decode_gate5 = makeZeros({intermediate});
    auto decode_gate6 = makeZeros({intermediate});
    auto decode_gate7 = makeZeros({intermediate});
    auto decode_up0 = makeZeros({intermediate});
    auto decode_up1 = makeZeros({intermediate});
    auto decode_up2 = makeZeros({intermediate});
    auto decode_up3 = makeZeros({intermediate});
    auto decode_up4 = makeZeros({intermediate});
    auto decode_up5 = makeZeros({intermediate});
    auto decode_up6 = makeZeros({intermediate});
    auto decode_up7 = makeZeros({intermediate});
    std::array<llaminar2::ITensor *, top_k> gate_outputs = {
        decode_gate0.get(), decode_gate1.get(), decode_gate2.get(), decode_gate3.get(),
        decode_gate4.get(), decode_gate5.get(), decode_gate6.get(), decode_gate7.get()};
    std::array<llaminar2::ITensor *, top_k> up_outputs = {
        decode_up0.get(), decode_up1.get(), decode_up2.get(), decode_up3.get(),
        decode_up4.get(), decode_up5.get(), decode_up6.get(), decode_up7.get()};
    auto decode_output = makeZeros({d_model});

    cudaGetLastError();
    EXPECT_FALSE(cuda_kernel_->groupedExpertGateUpDecodeFromTable(
        hidden_bf16.get(), expert_ids.data(), gateup_table, top_k,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate));
    EXPECT_FALSE(cuda_kernel_->groupedExpertGateUpDecodeFromTable(
        hidden_too_narrow.get(), expert_ids.data(), gateup_table, top_k,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    ASSERT_TRUE(cuda_kernel_->groupedExpertGateUpDecodeFromTable(
        hidden.get(), expert_ids.data(), gateup_table, top_k,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate));
    ASSERT_TRUE(cuda_kernel_->groupedExpertDownDecodeFromTable(
        gate_outputs.data(), up_outputs.data(), expert_ids.data(), expert_weights.data(),
        down_table, top_k, decode_output.get(), d_model, intermediate));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    std::vector<float> decode_values(
        decode_output->data(),
        decode_output->data() + decode_output->numel());

    gemm_config.set(
        /*gateup_kpart=*/false,
        /*gateup_kparts=*/16,
        /*down_kpart=*/false,
        /*down_kparts=*/16);
    auto serial_gate0 = makeZeros({intermediate});
    auto serial_gate1 = makeZeros({intermediate});
    auto serial_gate2 = makeZeros({intermediate});
    auto serial_gate3 = makeZeros({intermediate});
    auto serial_gate4 = makeZeros({intermediate});
    auto serial_gate5 = makeZeros({intermediate});
    auto serial_gate6 = makeZeros({intermediate});
    auto serial_gate7 = makeZeros({intermediate});
    auto serial_up0 = makeZeros({intermediate});
    auto serial_up1 = makeZeros({intermediate});
    auto serial_up2 = makeZeros({intermediate});
    auto serial_up3 = makeZeros({intermediate});
    auto serial_up4 = makeZeros({intermediate});
    auto serial_up5 = makeZeros({intermediate});
    auto serial_up6 = makeZeros({intermediate});
    auto serial_up7 = makeZeros({intermediate});
    std::array<llaminar2::ITensor *, top_k> serial_gate_outputs = {
        serial_gate0.get(), serial_gate1.get(), serial_gate2.get(), serial_gate3.get(),
        serial_gate4.get(), serial_gate5.get(), serial_gate6.get(), serial_gate7.get()};
    std::array<llaminar2::ITensor *, top_k> serial_up_outputs = {
        serial_up0.get(), serial_up1.get(), serial_up2.get(), serial_up3.get(),
        serial_up4.get(), serial_up5.get(), serial_up6.get(), serial_up7.get()};
    auto serial_decode_output = makeZeros({d_model});
    ASSERT_TRUE(cuda_kernel_->groupedExpertGateUpDecodeFromTable(
        hidden.get(), expert_ids.data(), gateup_table, top_k,
        serial_gate_outputs.data(), serial_up_outputs.data(), d_model, intermediate));
    ASSERT_TRUE(cuda_kernel_->groupedExpertDownDecodeFromTable(
        serial_gate_outputs.data(), serial_up_outputs.data(), expert_ids.data(), expert_weights.data(),
        down_table, top_k, serial_decode_output.get(), d_model, intermediate));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    std::vector<float> serial_decode_values(
        serial_decode_output->data(),
        serial_decode_output->data() + serial_decode_output->numel());

    gemm_config.set(
        /*gateup_kpart=*/true,
        /*gateup_kparts=*/16,
        /*down_kpart=*/true,
        /*down_kparts=*/16);

    auto prefill_output = makeZeros({seq_len, d_model});
    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
        routing_indices.get(), routing_weights.get(), seq_len, num_experts, top_k));
    ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), prefill_output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    std::vector<float> prefill_values(
        prefill_output->data(),
        prefill_output->data() + prefill_output->numel());

    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/false);
    auto split_prefill_output = makeZeros({seq_len, d_model});
    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
        routing_indices.get(), routing_weights.get(), seq_len, num_experts, top_k));
    ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), split_prefill_output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    std::vector<float> split_prefill_values(
        split_prefill_output->data(),
        split_prefill_output->data() + split_prefill_output->numel());

    expectVectorsClose(serial_decode_values, split_prefill_values,
                       0.9999, 0.006, /*row_width=*/d_model,
                       /*min_row_cosine=*/0.9998,
                       /*max_row_relative_l2=*/0.008,
                       /*max_row_kl=*/1.0e-4);
    expectVectorsClose(prefill_values, split_prefill_values,
                       0.9999, 0.006, /*row_width=*/d_model,
                       /*min_row_cosine=*/0.9998,
                       /*max_row_relative_l2=*/0.008,
                       /*max_row_kl=*/1.0e-4);
    expectVectorsClose(serial_decode_values, prefill_values,
                       0.9999, 0.006, /*row_width=*/d_model,
                       /*min_row_cosine=*/0.9998,
                       /*max_row_relative_l2=*/0.008,
                       /*max_row_kl=*/1.0e-4);
    expectVectorsClose(decode_values, serial_decode_values,
                       0.9999, 0.006, /*row_width=*/d_model,
                       /*min_row_cosine=*/0.9998,
                       /*max_row_relative_l2=*/0.008,
                       /*max_row_kl=*/1.0e-4);
    expectVectorsClose(decode_values, prefill_values,
                       0.9999, 0.006, /*row_width=*/d_model,
                       /*min_row_cosine=*/0.9998,
                       /*max_row_relative_l2=*/0.008,
                       /*max_row_kl=*/1.0e-4);
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimeGroupedDecodeFusedMatchesTwoStepAndGraphReplays)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    ScopedCudaMoEGemmConfig gemm_config;
    gemm_config.set(
        /*gateup_kpart=*/true,
        /*gateup_kparts=*/16,
        /*down_kpart=*/true,
        /*down_kparts=*/16);
    llaminar2::PerfStatsCollector::reset();

    using llaminar2::DeviceMoERuntimeTable;
    constexpr int num_layers = 1;
    constexpr int seq_len = 1;
    constexpr int top_k = 8;
    constexpr int num_experts = 8;
    constexpr int d_model = 512;
    constexpr int intermediate = 256;
    const auto device = llaminar2::DeviceId::cuda(0);

    std::vector<std::unique_ptr<llaminar2::TensorBase>> owned_weights;
    std::vector<llaminar2::test::GpuPreparedGemm> prepared_weights;
    owned_weights.reserve(static_cast<size_t>(num_experts * 3));
    prepared_weights.reserve(static_cast<size_t>(num_experts * 3));

    auto add_prepared_desc = [&](int rows, int cols, int seed, const char *role, int expected_codebook)
    {
        std::unique_ptr<llaminar2::TensorBase> weight;
        if (expected_codebook == 13)
        {
            weight = llaminar2::test::TestTensorFactory::createIQ2_SRandom(
                {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                static_cast<unsigned>(seed));
        }
        else
        {
            weight = llaminar2::test::TestTensorFactory::createIQ4_NLRandom(
                {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                static_cast<unsigned>(seed));
        }
        auto *weight_ptr = weight.get();
        owned_weights.push_back(std::move(weight));
        prepared_weights.push_back(llaminar2::test::makeGpuPreparedGemm(
            weight_ptr,
            device,
            std::string("test.cuda_moe.runtime_fused.") + role + "." + std::to_string(seed),
            llaminar2::ModelContextId{620000 + static_cast<uint64_t>(seed)}));

        llaminar2::DeviceNativeVNNIMatrixDesc desc{};
        EXPECT_TRUE(prepared_weights.back().kernel->exportNativeVNNIMatrixDesc(desc))
            << "failed to export native descriptor for " << role;
        EXPECT_EQ(desc.n, rows);
        EXPECT_EQ(desc.k, cols);
        EXPECT_EQ(desc.codebook_id, expected_codebook);
        return desc;
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    down_descs.reserve(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        gate_descs.push_back(add_prepared_desc(intermediate, d_model, 5100 + expert, "gate", 13));
        up_descs.push_back(add_prepared_desc(intermediate, d_model, 5200 + expert, "up", 13));
        down_descs.push_back(add_prepared_desc(d_model, intermediate, 5300 + expert, "down", 4));
    }

    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    DeviceMoERuntimeTable::Config cuda_config;
    cuda_config.device_id = device;
    cuda_config.num_layers = num_layers;
    cuda_config.num_experts = num_experts;
    cuda_config.top_k = top_k;
    cuda_config.mirror_to_device = true;
    DeviceMoERuntimeTable cuda_table(cuda_config);
    auto *runtime_layer = cuda_table.deviceLayerState(0);
    ASSERT_NE(runtime_layer, nullptr);

    std::vector<float> hidden_values(static_cast<size_t>(d_model));
    std::vector<float> router_values(static_cast<size_t>(num_experts) * static_cast<size_t>(d_model));
    for (int i = 0; i < d_model; ++i)
    {
        hidden_values[static_cast<size_t>(i)] =
            0.011f * static_cast<float>((i % 23) - 11) +
            0.002f * static_cast<float>((i % 5) - 2);
        for (int expert = 0; expert < num_experts; ++expert)
        {
            router_values[static_cast<size_t>(expert) * static_cast<size_t>(d_model) +
                          static_cast<size_t>(i)] =
                0.001f * static_cast<float>((i + 3 * expert) % 17) -
                0.007f * static_cast<float>(expert);
        }
    }

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto router = makeTensor({num_experts, d_model}, router_values);

    auto gate0 = makeZeros({intermediate});
    auto gate1 = makeZeros({intermediate});
    auto gate2 = makeZeros({intermediate});
    auto gate3 = makeZeros({intermediate});
    auto gate4 = makeZeros({intermediate});
    auto gate5 = makeZeros({intermediate});
    auto gate6 = makeZeros({intermediate});
    auto gate7 = makeZeros({intermediate});
    auto up0 = makeZeros({intermediate});
    auto up1 = makeZeros({intermediate});
    auto up2 = makeZeros({intermediate});
    auto up3 = makeZeros({intermediate});
    auto up4 = makeZeros({intermediate});
    auto up5 = makeZeros({intermediate});
    auto up6 = makeZeros({intermediate});
    auto up7 = makeZeros({intermediate});
    std::array<llaminar2::ITensor *, top_k> gate_outputs = {
        gate0.get(), gate1.get(), gate2.get(), gate3.get(),
        gate4.get(), gate5.get(), gate6.get(), gate7.get()};
    std::array<llaminar2::ITensor *, top_k> up_outputs = {
        up0.get(), up1.get(), up2.get(), up3.get(),
        up4.get(), up5.get(), up6.get(), up7.get()};
    auto two_step_output = makeZeros({d_model});
    auto fused_output = makeZeros({d_model});

    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        runtime_layer, hidden.get(), router.get(), d_model, num_experts, top_k,
        true, nullptr, nullptr, /*write_legacy_outputs=*/false, /*update_runtime_histogram=*/false));
    ASSERT_TRUE(cuda_kernel_->groupedExpertGateUpDecodeFromRuntime(
        runtime_layer, hidden.get(), gateup_table, top_k,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate));
    ASSERT_TRUE(cuda_kernel_->groupedExpertDownDecodeFromRuntime(
        gate_outputs.data(), up_outputs.data(), runtime_layer, down_table, top_k,
        two_step_output.get(), d_model, intermediate));
    ASSERT_TRUE(cuda_kernel_->groupedExpertDecodeFromRuntime(
        runtime_layer, hidden.get(), gateup_table, down_table, top_k,
        fused_output.get(), d_model, intermediate));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    std::vector<float> two_step_values(
        two_step_output->data(),
        two_step_output->data() + two_step_output->numel());
    std::vector<float> fused_values(
        fused_output->data(),
        fused_output->data() + fused_output->numel());
    expectVectorsClose(fused_values, two_step_values,
                       0.9999, 0.006, /*row_width=*/d_model,
                       /*min_row_cosine=*/0.9998,
                       /*max_row_relative_l2=*/0.008,
                       /*max_row_kl=*/1.0e-4);

    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_fused = cuda_kernel_->groupedExpertDecodeFromRuntime(
        runtime_layer, hidden.get(), gateup_table, down_table, top_k,
        fused_output.get(), d_model, intermediate);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_fused);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);

    cudaGraphExec_t graph_exec = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);

    std::vector<float> replay_values(
        fused_output->data(),
        fused_output->data() + fused_output->numel());
    expectVectorsClose(replay_values, two_step_values,
                       0.9999, 0.006, /*row_width=*/d_model,
                       /*min_row_cosine=*/0.9998,
                       /*max_row_relative_l2=*/0.008,
                       /*max_row_kl=*/1.0e-4);

    expectGroupedDecodeCounter(
        "cuda_moe_grouped_decode_fused_calls", "runtime", top_k, d_model, intermediate,
        "fused_block_down");
    expectFusedDecodeSubkernelTimer(
        "cuda_moe_fused_decode_hidden_quantize", top_k, d_model, intermediate);
    expectFusedDecodeSubkernelTimer(
        "cuda_moe_fused_decode_gateup_kpart", top_k, d_model, intermediate);
    expectFusedDecodeSubkernelTimer(
        "cuda_moe_fused_decode_swiglu_quantize", top_k, d_model, intermediate);
    expectFusedDecodeSubkernelTimer(
        "cuda_moe_fused_decode_down_warp_reduce", top_k, d_model, intermediate);
    llaminar2::PerfStatsCollector::reset();
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimeRouteSelectAndFusedDecodeCaptureWithLargeExpertTable)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    ScopedCudaMoEGemmConfig gemm_config;
    gemm_config.set(
        /*gateup_kpart=*/true,
        /*gateup_kparts=*/16,
        /*down_kpart=*/true,
        /*down_kparts=*/16);
    auto &gemm = llaminar2::mutableDebugEnv().gemm;
    const bool old_router_q8 = gemm.cuda_moe_router_q8;
    const bool old_reuse_router_q8_hidden = gemm.cuda_moe_reuse_router_q8_hidden;
    gemm.cuda_moe_router_q8 = true;
    gemm.cuda_moe_reuse_router_q8_hidden = true;
    struct RestoreRouterQ8
    {
        llaminar2::GemmConfig &gemm;
        bool router_q8;
        bool reuse_router_q8_hidden;
        ~RestoreRouterQ8()
        {
            gemm.cuda_moe_router_q8 = router_q8;
            gemm.cuda_moe_reuse_router_q8_hidden = reuse_router_q8_hidden;
            llaminar2::PerfStatsCollector::reset();
        }
    } restore_router_q8{gemm, old_router_q8, old_reuse_router_q8_hidden};
    llaminar2::PerfStatsCollector::reset();

    using llaminar2::DeviceMoERuntimeTable;
    constexpr int num_layers = 1;
    constexpr int seq_len = 1;
    constexpr int top_k = 8;
    constexpr int num_experts = 256;
    constexpr int d_model = 32;
    constexpr int intermediate = 32;
    constexpr size_t descriptor_bytes = 4096;
    const auto device = llaminar2::DeviceId::cuda(0);

    std::vector<CudaAllocation> payloads;
    std::vector<CudaAllocation> scales;
    payloads.reserve(static_cast<size_t>(num_experts * 3));
    scales.reserve(static_cast<size_t>(num_experts * 3));

    auto add_desc = [&](int rows, int cols)
    {
        payloads.emplace_back(descriptor_bytes);
        scales.emplace_back(descriptor_bytes);
        EXPECT_EQ(cudaMemsetAsync(payloads.back().get(), 0, descriptor_bytes, stream_), cudaSuccess);

        const int scale_count = rows * (cols / 32);
        std::vector<uint16_t> host_scales(static_cast<size_t>(scale_count), 0x3c00u);
        EXPECT_EQ(cudaMemcpyAsync(scales.back().get(), host_scales.data(),
                                  host_scales.size() * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        return makeCudaNativeDesc(payloads.back(), scales.back(), rows, cols);
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    down_descs.reserve(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        gate_descs.push_back(add_desc(intermediate, d_model));
        up_descs.push_back(add_desc(intermediate, d_model));
        down_descs.push_back(add_desc(d_model, intermediate));
    }

    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    DeviceMoERuntimeTable::Config cuda_config;
    cuda_config.device_id = device;
    cuda_config.num_layers = num_layers;
    cuda_config.num_experts = num_experts;
    cuda_config.top_k = top_k;
    cuda_config.mirror_to_device = true;
    DeviceMoERuntimeTable cuda_table(cuda_config);
    auto *runtime_layer = cuda_table.deviceLayerState(0);
    ASSERT_NE(runtime_layer, nullptr);

    std::vector<float> hidden_values(static_cast<size_t>(d_model), 1.0f);
    std::vector<float> router_values(static_cast<size_t>(num_experts) * static_cast<size_t>(d_model));
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const float value = static_cast<float>(expert + 1) * 0.001f;
        for (int i = 0; i < d_model; ++i)
            router_values[static_cast<size_t>(expert) * d_model + i] = value;
    }

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto router = makeTensor({num_experts, d_model}, router_values);
    auto route_indices = makeZeros({top_k});
    auto route_weights = makeZeros({top_k});
    auto output = makeZeros({d_model});

    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        runtime_layer, hidden.get(), router.get(), d_model, num_experts, top_k,
        true, route_indices.get(), route_weights.get(),
        /*write_legacy_outputs=*/true, /*update_runtime_histogram=*/true));
    ASSERT_TRUE(cuda_kernel_->groupedExpertDecodeFromRuntime(
        runtime_layer, hidden.get(), gateup_table, down_table, top_k,
        output.get(), d_model, intermediate));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_FALSE(llaminar2::PerfStatsCollector::snapshot(
                     {"kernel.cuda_moe_gateup_reused_router_q8_hidden_calls"})
                     .empty())
        << "eager route+fused decode should still reuse router Q8 hidden scratch";

    llaminar2::PerfStatsCollector::reset();
    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    bool captured_route = false;
    bool captured_fused = false;
    {
        llaminar2::GraphCaptureGuard capture_guard(/*host_bookkeeping=*/true);
        captured_route = cuda_kernel_->decodeRouteSelect(
            runtime_layer, hidden.get(), router.get(), d_model, num_experts, top_k,
            true, route_indices.get(), route_weights.get(),
            /*write_legacy_outputs=*/true, /*update_runtime_histogram=*/true);
        captured_fused = cuda_kernel_->groupedExpertDecodeFromRuntime(
            runtime_layer, hidden.get(), gateup_table, down_table, top_k,
            output.get(), d_model, intermediate);
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_route);
    EXPECT_TRUE(captured_fused);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);
    EXPECT_TRUE(llaminar2::PerfStatsCollector::snapshot(
                    {"kernel.cuda_moe_gateup_reused_router_q8_hidden_calls"})
                    .empty())
        << "graph capture must not mark recorded-only router Q8 scratch as reusable";

    cudaGraphExec_t graph_exec = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);
    for (int replay = 0; replay < 3; ++replay)
        ASSERT_EQ(cudaGraphLaunch(graph_exec, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    llaminar2::DeviceMoELayerRuntime host_runtime{};
    ASSERT_EQ(cudaMemcpy(&host_runtime, runtime_layer, sizeof(host_runtime), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int k = 0; k < top_k; ++k)
    {
        EXPECT_GE(host_runtime.topk_expert_ids[k], 0);
        EXPECT_LT(host_runtime.topk_expert_ids[k], num_experts);
    }

    const float *output_data = output->data();
    for (size_t i = 0; i < output->numel(); ++i)
        ASSERT_TRUE(std::isfinite(output_data[i])) << "output element " << i;

    ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimeDecodeGraphReplayReadsDeviceDescriptorsWithoutTableRefresh)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedCudaMoEGemmConfig gemm_config;
    gemm_config.set(
        /*gateup_kpart=*/false,
        /*gateup_kparts=*/16,
        /*down_kpart=*/false,
        /*down_kparts=*/16);

    constexpr int num_experts = 2;
    constexpr int top_k = 1;
    constexpr int d_model = 512;
    constexpr int intermediate = 256;
    const auto device = llaminar2::DeviceId::cuda(0);

    std::vector<std::unique_ptr<llaminar2::TensorBase>> owned_weights;
    std::vector<llaminar2::test::GpuPreparedGemm> prepared_weights;
    owned_weights.reserve(static_cast<size_t>(num_experts * 3));
    prepared_weights.reserve(static_cast<size_t>(num_experts * 3));

    auto add_prepared_desc = [&](int rows, int cols, int seed, const char *role)
    {
        auto weight = llaminar2::test::TestTensorFactory::createIQ4_NLRandom(
            {static_cast<size_t>(rows), static_cast<size_t>(cols)},
            static_cast<unsigned>(seed));
        auto *weight_ptr = weight.get();
        owned_weights.push_back(std::move(weight));
        prepared_weights.push_back(llaminar2::test::makeGpuPreparedGemm(
            weight_ptr,
            device,
            std::string("test.cuda_moe.runtime_descriptor_replay.") + role + "." +
                std::to_string(seed),
            llaminar2::ModelContextId{710000 + static_cast<uint64_t>(seed)}));

        llaminar2::DeviceNativeVNNIMatrixDesc desc{};
        EXPECT_TRUE(prepared_weights.back().kernel->exportNativeVNNIMatrixDesc(desc))
            << "failed to export native descriptor for " << role;
        EXPECT_EQ(desc.n, rows);
        EXPECT_EQ(desc.k, cols);
        EXPECT_EQ(desc.codebook_id, 4);
        return desc;
    };

    std::array<llaminar2::DeviceNativeVNNIMatrixDesc, num_experts> gate_descs = {
        add_prepared_desc(intermediate, d_model, 6100, "gate"),
        add_prepared_desc(intermediate, d_model, 6101, "gate"),
    };
    std::array<llaminar2::DeviceNativeVNNIMatrixDesc, num_experts> up_descs = {
        add_prepared_desc(intermediate, d_model, 6200, "up"),
        add_prepared_desc(intermediate, d_model, 6201, "up"),
    };
    std::array<llaminar2::DeviceNativeVNNIMatrixDesc, num_experts> down_descs = {
        add_prepared_desc(d_model, intermediate, 6300, "down"),
        add_prepared_desc(d_model, intermediate, 6301, "down"),
    };

    const std::array<llaminar2::DeviceNativeVNNIMatrixDesc, num_experts> stale_gate_descs = {
        gate_descs[0], gate_descs[0]};
    const std::array<llaminar2::DeviceNativeVNNIMatrixDesc, num_experts> stale_up_descs = {
        up_descs[0], up_descs[0]};
    const std::array<llaminar2::DeviceNativeVNNIMatrixDesc, num_experts> stale_down_descs = {
        down_descs[0], down_descs[0]};

    const int stale_gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        stale_gate_descs.data(), stale_up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(stale_gateup_table, 0);
    const int stale_down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        stale_down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(stale_down_table, 0);
    const int fresh_gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(fresh_gateup_table, 0);
    const int fresh_down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(fresh_down_table, 0);

    std::vector<float> hidden_values(static_cast<size_t>(d_model));
    for (int i = 0; i < d_model; ++i)
    {
        hidden_values[static_cast<size_t>(i)] =
            0.019f * static_cast<float>((i % 29) - 14) +
            0.003f * static_cast<float>((i % 7) - 3);
    }
    auto hidden = makeTensor({1, d_model}, hidden_values);

    const std::array<int, top_k> expert_ids = {1};
    const std::array<float, top_k> expert_weights = {1.0f};
    auto run_table_reference = [&](int gateup_table, int down_table, llaminar2::ITensor *output)
    {
        auto gate = makeZeros({intermediate});
        auto up = makeZeros({intermediate});
        std::array<llaminar2::ITensor *, top_k> gate_outputs = {gate.get()};
        std::array<llaminar2::ITensor *, top_k> up_outputs = {up.get()};

        ASSERT_TRUE(cuda_kernel_->groupedExpertGateUpDecodeFromTable(
            hidden.get(), expert_ids.data(), gateup_table, top_k,
            gate_outputs.data(), up_outputs.data(), d_model, intermediate));
        ASSERT_TRUE(cuda_kernel_->groupedExpertDownDecodeFromTable(
            gate_outputs.data(), up_outputs.data(), expert_ids.data(), expert_weights.data(),
            down_table, top_k, output, d_model, intermediate));
    };

    auto stale_reference = makeZeros({d_model});
    auto fresh_reference = makeZeros({d_model});
    run_table_reference(stale_gateup_table, stale_down_table, stale_reference.get());
    run_table_reference(fresh_gateup_table, fresh_down_table, fresh_reference.get());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const std::vector<float> stale_values(
        stale_reference->data(),
        stale_reference->data() + stale_reference->numel());
    const std::vector<float> fresh_values(
        fresh_reference->data(),
        fresh_reference->data() + fresh_reference->numel());
    ASSERT_GT(relativeL2Error(stale_values.data(), fresh_values.data(), fresh_values.size()), 0.01)
        << "descriptor-table A/B reference must be distinguishable for this regression";

    auto make_runtime = [&](const llaminar2::DeviceNativeVNNIMatrixDesc &gate,
                            const llaminar2::DeviceNativeVNNIMatrixDesc &up,
                            const llaminar2::DeviceNativeVNNIMatrixDesc &down)
    {
        llaminar2::DeviceMoELayerRuntime runtime{};
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
            desc.flags = llaminar2::toMoEExpertFlags(
                llaminar2::DeviceMoEExpertFlags::Valid |
                llaminar2::DeviceMoEExpertFlags::Resident |
                llaminar2::DeviceMoEExpertFlags::LocalCompute);
            desc.gate = expert == 0 ? gate_descs[0] : gate;
            desc.up = expert == 0 ? up_descs[0] : up;
            desc.down = expert == 0 ? down_descs[0] : down;
        }
        return runtime;
    };

    const auto runtime_stale = make_runtime(gate_descs[0], up_descs[0], down_descs[0]);
    const auto runtime_fresh = make_runtime(gate_descs[1], up_descs[1], down_descs[1]);

    llaminar2::DeviceMoELayerRuntime *device_runtime = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&device_runtime), sizeof(llaminar2::DeviceMoELayerRuntime)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(device_runtime, &runtime_stale, sizeof(runtime_stale),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    auto runtime_output = makeZeros({d_model});
    ASSERT_TRUE(cuda_kernel_->groupedExpertDecodeFromRuntime(
        device_runtime, hidden.get(), stale_gateup_table, stale_down_table, top_k,
        runtime_output.get(), d_model, intermediate,
        llaminar2::MoEDecodeDescriptorSource::RuntimePlacementTable));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured = cuda_kernel_->groupedExpertDecodeFromRuntime(
        device_runtime, hidden.get(), stale_gateup_table, stale_down_table, top_k,
        runtime_output.get(), d_model, intermediate,
        llaminar2::MoEDecodeDescriptorSource::RuntimePlacementTable);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);

    cudaGraphExec_t graph_exec = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(device_runtime, &runtime_fresh, sizeof(runtime_fresh),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const std::vector<float> runtime_values(
        runtime_output->data(),
        runtime_output->data() + runtime_output->numel());
    expectVectorsClose(runtime_values, fresh_values,
                       0.9999, 0.006, /*row_width=*/d_model,
                       /*min_row_cosine=*/0.9998,
                       /*max_row_relative_l2=*/0.008,
                       /*max_row_kl=*/1.0e-4);
    EXPECT_GT(relativeL2Error(runtime_values.data(), stale_values.data(), runtime_values.size()), 0.01)
        << "graph replay followed the stale descriptor table instead of device runtime descriptors";

    ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    ASSERT_EQ(cudaFree(device_runtime), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, VerifierSmallMPrefillM234MatchesDecodeRowsAndCaptures)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    ScopedCudaMoEPrefillConfig prefill_config;
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);
    ScopedCudaMoEGemmConfig gemm_config;
    gemm_config.set(
        /*gateup_kpart=*/true,
        /*gateup_kparts=*/16,
        /*down_kpart=*/true,
        /*down_kparts=*/16);
    llaminar2::PerfStatsCollector::reset();

    constexpr int top_k = 8;
    constexpr int num_experts = 32;
    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    const auto device = llaminar2::DeviceId::cuda(0);

    std::vector<std::unique_ptr<llaminar2::TensorBase>> owned_weights;
    std::vector<llaminar2::test::GpuPreparedGemm> prepared_weights;
    owned_weights.reserve(static_cast<size_t>(num_experts * 3));
    prepared_weights.reserve(static_cast<size_t>(num_experts * 3));

    auto add_prepared_desc = [&](int rows, int cols, int seed, const char *role, int expected_codebook)
    {
        std::unique_ptr<llaminar2::TensorBase> weight;
        if (expected_codebook == 13)
        {
            weight = llaminar2::test::TestTensorFactory::createIQ2_SRandom(
                {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                static_cast<unsigned>(seed));
        }
        else
        {
            weight = llaminar2::test::TestTensorFactory::createIQ4_XSRandom(
                {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                static_cast<unsigned>(seed));
        }

        auto *weight_ptr = weight.get();
        owned_weights.push_back(std::move(weight));
        prepared_weights.push_back(llaminar2::test::makeGpuPreparedGemm(
            weight_ptr,
            device,
            std::string("test.cuda_moe.verifier.") + role + "." + std::to_string(seed),
            llaminar2::ModelContextId{240000 + static_cast<uint64_t>(seed)}));

        llaminar2::DeviceNativeVNNIMatrixDesc desc{};
        EXPECT_TRUE(prepared_weights.back().kernel->exportNativeVNNIMatrixDesc(desc))
            << "failed to export native descriptor for " << role;
        EXPECT_EQ(desc.n, rows);
        EXPECT_EQ(desc.k, cols);
        EXPECT_EQ(desc.codebook_id, expected_codebook);
        return desc;
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    down_descs.reserve(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        gate_descs.push_back(add_prepared_desc(intermediate, d_model, 4100 + expert, "gate", 13));
        up_descs.push_back(add_prepared_desc(intermediate, d_model, 4200 + expert, "up", 13));
        down_descs.push_back(add_prepared_desc(d_model, intermediate, 4300 + expert, "down", 4));
    }

    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    auto make_hidden_values = [](int seq_len)
    {
        std::vector<float> values(static_cast<size_t>(seq_len) * d_model);
        for (size_t i = 0; i < values.size(); ++i)
        {
            values[i] =
                0.013f * static_cast<float>(static_cast<int>(i % 29) - 14) +
                0.002f * static_cast<float>(static_cast<int>((i / 7) % 11) - 5);
        }
        return values;
    };

    auto make_routing_indices = [](int seq_len)
    {
        std::vector<float> values(static_cast<size_t>(seq_len) * top_k);
        for (int row = 0; row < seq_len; ++row)
        {
            for (int k = 0; k < top_k; ++k)
            {
                // Keep each token's top-k expert ids unique, as the production
                // router does, while repeating experts across verifier rows.
                // That creates unused active-expert grid slots without relying
                // on impossible same-token duplicate top-k semantics.
                const int expert = (k + ((row & 1) ? 4 : 0)) % num_experts;
                values[static_cast<size_t>(row) * top_k + k] = static_cast<float>(expert);
            }
        }
        return values;
    };

    auto make_routing_weights = [](int seq_len)
    {
        std::vector<float> values(static_cast<size_t>(seq_len) * top_k);
        for (int row = 0; row < seq_len; ++row)
        {
            float sum = 0.0f;
            for (int k = 0; k < top_k; ++k)
            {
                const float weight = 0.05f + 0.01f * static_cast<float>((row + k) % top_k);
                values[static_cast<size_t>(row) * top_k + k] = weight;
                sum += weight;
            }
            for (int k = 0; k < top_k; ++k)
                values[static_cast<size_t>(row) * top_k + k] /= sum;
        }
        return values;
    };

    auto expect_prefill_record = [&](int seq_len, int expected_tile_m)
    {
        const auto records =
            llaminar2::PerfStatsCollector::snapshot({"kernel.cuda_moe_grouped_prefill_active_expert_grid_calls"});
        const int total_slots = seq_len * top_k;
        const int active_slots = std::min(total_slots, num_experts);
        const std::string expected_total_slots = std::to_string(total_slots);
        const std::string expected_active_slots = std::to_string(active_slots);
        const std::string expected_num_experts = std::to_string(num_experts);
        const std::string expected_tile = std::to_string(expected_tile_m);
        const std::string expected_tile_n =
            std::to_string((active_slots > 0 && seq_len <= 4) ? 64 : 128);
        const auto it = std::find_if(
            records.begin(),
            records.end(),
            [&](const llaminar2::PerfStatRecord &record)
            {
                auto tag_equals = [&](const char *key, const std::string &value)
                {
                    const auto tag = record.tags.find(key);
                    return tag != record.tags.end() && tag->second == value;
                };
                return tag_equals("total_slots", expected_total_slots) &&
                       tag_equals("active_expert_slots", expected_active_slots) &&
                       tag_equals("num_experts", expected_num_experts) &&
                       tag_equals("tile_m", expected_tile) &&
                       tag_equals("tile_n", expected_tile_n);
            });
        ASSERT_NE(it, records.end()) << "missing verifier small-M prefill counter for seq_len="
                                     << seq_len << " tile_m=" << expected_tile_m;
    };

    for (int seq_len : {2, 3, 4})
    {
        const auto hidden_values = make_hidden_values(seq_len);
        const auto routing_indices_values = make_routing_indices(seq_len);
        const auto routing_weights_values = make_routing_weights(seq_len);
        auto unique_routes = routing_indices_values;
        std::sort(unique_routes.begin(), unique_routes.end());
        unique_routes.erase(std::unique(unique_routes.begin(), unique_routes.end()), unique_routes.end());
        ASSERT_LT(unique_routes.size(), routing_indices_values.size())
            << "This regression must exercise repeated experts across rows and unused verifier grid slots";
        for (int row = 0; row < seq_len; ++row)
        {
            std::vector<float> row_routes(
                routing_indices_values.begin() + static_cast<ptrdiff_t>(row) * top_k,
                routing_indices_values.begin() + static_cast<ptrdiff_t>(row + 1) * top_k);
            std::sort(row_routes.begin(), row_routes.end());
            row_routes.erase(std::unique(row_routes.begin(), row_routes.end()), row_routes.end());
            ASSERT_EQ(row_routes.size(), static_cast<size_t>(top_k))
                << "Production top-k routes should be unique within a token row";
        }

        auto hidden = makeTensor({static_cast<size_t>(seq_len), d_model}, hidden_values);
        auto routing_indices = makeTensor({static_cast<size_t>(seq_len), top_k}, routing_indices_values);
        auto routing_weights = makeTensor({static_cast<size_t>(seq_len), top_k}, routing_weights_values);
        auto prefill_output = makeZeros({static_cast<size_t>(seq_len), d_model});
        auto split_prefill_output = makeZeros({static_cast<size_t>(seq_len), d_model});

        prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);
        ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
            routing_indices.get(), routing_weights.get(), seq_len, num_experts, top_k));
        ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
            hidden.get(), prefill_output.get(), gateup_table, down_table,
            seq_len, d_model, intermediate, num_experts, top_k));
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        std::vector<float> prefill_values(
            prefill_output->data(),
            prefill_output->data() + prefill_output->numel());

        prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/false);
        ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
            routing_indices.get(), routing_weights.get(), seq_len, num_experts, top_k));
        ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
            hidden.get(), split_prefill_output.get(), gateup_table, down_table,
            seq_len, d_model, intermediate, num_experts, top_k));
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        std::vector<float> split_prefill_values(
            split_prefill_output->data(),
            split_prefill_output->data() + split_prefill_output->numel());

        // The small-M verifier sweep compares two native-quantized prefill routes:
        // fused SwigLU+quantize and split SwigLU then quantize. They should stay
        // inside the same tolerance we require against row-wise decode.
        expectVectorsClose(prefill_values, split_prefill_values,
                           0.9999, 0.006,
                           /*row_width=*/d_model,
                           /*min_row_cosine=*/0.9998,
                           /*max_row_relative_l2=*/0.008,
                           /*max_row_kl=*/1.0e-4);
        expectPrefillSwiGLUPathRecord("split", seq_len, top_k, num_experts, 2);
        prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);

        std::vector<float> rowwise_decode_values;
        rowwise_decode_values.reserve(prefill_values.size());

        for (int row = 0; row < seq_len; ++row)
        {
            const auto row_begin = hidden_values.begin() + static_cast<ptrdiff_t>(row) * d_model;
            std::vector<float> row_hidden_values(row_begin, row_begin + d_model);
            auto row_hidden = makeTensor({1, d_model}, row_hidden_values);

            std::array<int, top_k> expert_ids = {};
            std::array<float, top_k> expert_weights = {};
            for (int k = 0; k < top_k; ++k)
            {
                const size_t slot = static_cast<size_t>(row) * top_k + k;
                expert_ids[static_cast<size_t>(k)] = static_cast<int>(routing_indices_values[slot]);
                expert_weights[static_cast<size_t>(k)] = routing_weights_values[slot];
            }

            std::array<std::shared_ptr<llaminar2::FP32Tensor>, top_k> gate_owned;
            std::array<std::shared_ptr<llaminar2::FP32Tensor>, top_k> up_owned;
            std::array<llaminar2::ITensor *, top_k> gate_outputs = {};
            std::array<llaminar2::ITensor *, top_k> up_outputs = {};
            for (int k = 0; k < top_k; ++k)
            {
                gate_owned[static_cast<size_t>(k)] = makeZeros({intermediate});
                up_owned[static_cast<size_t>(k)] = makeZeros({intermediate});
                gate_outputs[static_cast<size_t>(k)] = gate_owned[static_cast<size_t>(k)].get();
                up_outputs[static_cast<size_t>(k)] = up_owned[static_cast<size_t>(k)].get();
            }

            auto decode_output = makeZeros({d_model});
            ASSERT_TRUE(cuda_kernel_->groupedExpertGateUpDecodeFromTable(
                row_hidden.get(), expert_ids.data(), gateup_table, top_k,
                gate_outputs.data(), up_outputs.data(), d_model, intermediate));
            ASSERT_TRUE(cuda_kernel_->groupedExpertDownDecodeFromTable(
                gate_outputs.data(), up_outputs.data(), expert_ids.data(), expert_weights.data(),
                down_table, top_k, decode_output.get(), d_model, intermediate));
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

            rowwise_decode_values.insert(
                rowwise_decode_values.end(),
                decode_output->data(),
                decode_output->data() + decode_output->numel());
        }

        expectVectorsClose(prefill_values, rowwise_decode_values,
                           0.9999, 0.006,
                           /*row_width=*/d_model,
                           /*min_row_cosine=*/0.9998,
                           /*max_row_relative_l2=*/0.008,
                           /*max_row_kl=*/1.0e-4);

        ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
        const bool captured_grouping = cuda_kernel_->prepareExpertGroupsAsync(
            routing_indices.get(), routing_weights.get(), seq_len, num_experts, top_k);
        const bool captured_prefill = cuda_kernel_->executeGroupedPrefillPipeline(
            hidden.get(), prefill_output.get(), gateup_table, down_table,
            seq_len, d_model, intermediate, num_experts, top_k);
        cudaGraph_t graph = nullptr;
        const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
        EXPECT_TRUE(captured_grouping);
        EXPECT_TRUE(captured_prefill);
        ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
        ASSERT_NE(graph, nullptr);

        cudaGraphExec_t executable = nullptr;
        ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);

        const float *captured_output = prefill_output->data();
        for (size_t i = 0; i < prefill_output->numel(); ++i)
            ASSERT_TRUE(std::isfinite(captured_output[i])) << "captured output element " << i;

        expect_prefill_record(seq_len, 2);
        expectPrefillSwiGLUPathRecord("fused", seq_len, top_k, num_experts,
                                      2,
                                      "kpart_swiglu",
                                      "kpart_prefill",
                                      "token_direct");
    }

    const auto grouping_records =
        llaminar2::PerfStatsCollector::snapshot({"kernel.cuda_moe_small_prefill_grouping_calls"});
    ASSERT_FALSE(grouping_records.empty())
        << "MTP verifier-sized MoE prefill must keep the compact device grouping route";
    llaminar2::PerfStatsCollector::reset();
#endif
}

TEST_F(Test__CUDAMoEKernel, SharedExpertVerifierSmallMPrefillM234MatchesDecodeRowsAndCaptures)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    ScopedCudaMoEPrefillConfig prefill_config;
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);
    ScopedCudaMoEGemmConfig gemm_config;
    gemm_config.set(
        /*gateup_kpart=*/true,
        /*gateup_kparts=*/16,
        /*down_kpart=*/true,
        /*down_kparts=*/16);
    llaminar2::PerfStatsCollector::reset();

    constexpr int num_experts = 1;
    constexpr int top_k = 1;
    constexpr int d_model = 256;
    constexpr int intermediate = 256;
    const auto device = llaminar2::DeviceId::cuda(0);

    std::vector<std::unique_ptr<llaminar2::TensorBase>> owned_weights;
    std::vector<llaminar2::test::GpuPreparedGemm> prepared_weights;
    owned_weights.reserve(3);
    prepared_weights.reserve(3);

    auto add_prepared_desc = [&](int rows, int cols, int seed, const char *role, int expected_codebook)
    {
        std::unique_ptr<llaminar2::TensorBase> weight;
        if (expected_codebook == 13)
        {
            weight = llaminar2::test::TestTensorFactory::createIQ2_SRandom(
                {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                static_cast<unsigned>(seed));
        }
        else
        {
            weight = llaminar2::test::TestTensorFactory::createIQ4_XSRandom(
                {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                static_cast<unsigned>(seed));
        }

        auto *weight_ptr = weight.get();
        owned_weights.push_back(std::move(weight));
        prepared_weights.push_back(llaminar2::test::makeGpuPreparedGemm(
            weight_ptr,
            device,
            std::string("test.cuda_moe.shared_verifier.") + role,
            llaminar2::ModelContextId{250000 + static_cast<uint64_t>(seed)}));

        llaminar2::DeviceNativeVNNIMatrixDesc desc{};
        EXPECT_TRUE(prepared_weights.back().kernel->exportNativeVNNIMatrixDesc(desc))
            << "failed to export native descriptor for shared " << role;
        EXPECT_EQ(desc.n, rows);
        EXPECT_EQ(desc.k, cols);
        EXPECT_EQ(desc.codebook_id, expected_codebook);
        return desc;
    };

    std::array<llaminar2::DeviceNativeVNNIMatrixDesc, num_experts> gate_descs = {
        add_prepared_desc(intermediate, d_model, 5100, "gate", 13)};
    std::array<llaminar2::DeviceNativeVNNIMatrixDesc, num_experts> up_descs = {
        add_prepared_desc(intermediate, d_model, 5200, "up", 13)};
    std::array<llaminar2::DeviceNativeVNNIMatrixDesc, num_experts> down_descs = {
        add_prepared_desc(d_model, intermediate, 5300, "down", 4)};

    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    auto make_hidden_values = [](int seq_len)
    {
        std::vector<float> values(static_cast<size_t>(seq_len) * d_model);
        for (size_t i = 0; i < values.size(); ++i)
        {
            values[i] =
                0.011f * static_cast<float>(static_cast<int>(i % 31) - 15) +
                0.003f * static_cast<float>(static_cast<int>((i / 5) % 13) - 6);
        }
        return values;
    };

    auto expect_shared_group_record = [](int seq_len)
    {
        const auto records =
            llaminar2::PerfStatsCollector::snapshot({"kernel.cuda_moe_shared_expert_prefill_group_calls"});
        const std::string expected_seq_len = std::to_string(seq_len);
        const auto it = std::find_if(
            records.begin(),
            records.end(),
            [&](const llaminar2::PerfStatRecord &record)
            {
                auto tag_equals = [&](const char *key, const std::string &value)
                {
                    const auto tag = record.tags.find(key);
                    return tag != record.tags.end() && tag->second == value;
                };
                return record.name == "cuda_moe_shared_expert_prefill_group_calls" &&
                       tag_equals("seq_len", expected_seq_len) &&
                       tag_equals("active_expert_slots", "1") &&
                       tag_equals("top_k", "1");
            });
        ASSERT_NE(it, records.end()) << "missing shared expert prefill group counter seq_len="
                                     << seq_len;
    };

    for (int seq_len : {2, 3, 4})
    {
        const auto hidden_values = make_hidden_values(seq_len);
        auto hidden = makeTensor({static_cast<size_t>(seq_len), d_model}, hidden_values);
        auto prefill_output = makeZeros({static_cast<size_t>(seq_len), d_model});

        ASSERT_TRUE(cuda_kernel_->prepareSharedExpertPrefillGroup(seq_len));
        ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
            hidden.get(), prefill_output.get(), gateup_table, down_table,
            seq_len, d_model, intermediate, num_experts, top_k));
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        std::vector<float> prefill_values(
            prefill_output->data(),
            prefill_output->data() + prefill_output->numel());

        std::vector<float> rowwise_decode_values;
        rowwise_decode_values.reserve(prefill_values.size());
        constexpr int expert_id = 0;
        constexpr float expert_weight = 1.0f;

        for (int row = 0; row < seq_len; ++row)
        {
            const auto row_begin = hidden_values.begin() + static_cast<ptrdiff_t>(row) * d_model;
            std::vector<float> row_hidden_values(row_begin, row_begin + d_model);
            auto row_hidden = makeTensor({1, d_model}, row_hidden_values);
            auto gate = makeZeros({intermediate});
            auto up = makeZeros({intermediate});
            auto decode_output = makeZeros({d_model});

            std::array<llaminar2::ITensor *, top_k> gate_outputs = {gate.get()};
            std::array<llaminar2::ITensor *, top_k> up_outputs = {up.get()};

            ASSERT_TRUE(cuda_kernel_->groupedExpertGateUpDecodeFromTable(
                row_hidden.get(), &expert_id, gateup_table, top_k,
                gate_outputs.data(), up_outputs.data(), d_model, intermediate));
            ASSERT_TRUE(cuda_kernel_->groupedExpertDownDecodeFromTable(
                gate_outputs.data(), up_outputs.data(), &expert_id, &expert_weight,
                down_table, top_k, decode_output.get(), d_model, intermediate));
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

            rowwise_decode_values.insert(
                rowwise_decode_values.end(),
                decode_output->data(),
                decode_output->data() + decode_output->numel());
        }

        expectVectorsClose(prefill_values, rowwise_decode_values,
                           0.9999, 0.006,
                           /*row_width=*/d_model,
                           /*min_row_cosine=*/0.9998,
                           /*max_row_relative_l2=*/0.008,
                           /*max_row_kl=*/1.0e-4);

        ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
        const bool captured_grouping = cuda_kernel_->prepareSharedExpertPrefillGroup(seq_len);
        const bool captured_prefill = cuda_kernel_->executeGroupedPrefillPipeline(
            hidden.get(), prefill_output.get(), gateup_table, down_table,
            seq_len, d_model, intermediate, num_experts, top_k);
        cudaGraph_t graph = nullptr;
        const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
        EXPECT_TRUE(captured_grouping);
        EXPECT_TRUE(captured_prefill);
        ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
        ASSERT_NE(graph, nullptr);

        cudaGraphExec_t executable = nullptr;
        ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);

        expect_shared_group_record(seq_len);
        constexpr int kExpectedSharedExpertTileM = 2;
        expectPrefillSwiGLUPathRecord("fused", seq_len, top_k, num_experts,
                                      kExpectedSharedExpertTileM,
                                      "kpart_swiglu",
                                      "kpart_prefill",
                                      "token_direct",
                                      /*expected_active_expert_slots=*/1);
    }

    llaminar2::PerfStatsCollector::reset();
#endif
}

TEST_F(Test__CUDAMoEKernel, RoutedOnlyVerifierPrefill_IQ3S_M234MatchesRowByRowDecode)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    /*
     * This is the routed-only companion to the combined shared-expert IQ3_S
     * regression above.  Qwen3.6 MoE production MTP verifier rows often have no
     * shared expert path, so the grouped routed expert pipeline itself must be
     * decode-equivalent for M=2..4 before the graph can publish verifier rows
     * from it.
     */
    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;
    constexpr int routed_variants = 16;
    const auto device = llaminar2::DeviceId::cuda(0);

    ScopedCudaMoEPrefillConfig prefill_config;
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);
    ScopedCudaMoEGemmConfig gemm_config;
    gemm_config.set(
        /*gateup_kpart=*/true,
        /*gateup_kparts=*/16,
        /*down_kpart=*/true,
        /*down_kparts=*/16);

    std::vector<std::unique_ptr<llaminar2::TensorBase>> owned_weights;
    std::vector<llaminar2::test::GpuPreparedGemm> prepared_weights;
    owned_weights.reserve(static_cast<size_t>(routed_variants * 3));
    prepared_weights.reserve(static_cast<size_t>(routed_variants * 3));

    auto add_prepared_iq3 = [&](int rows,
                                int cols,
                                int seed,
                                const char *role) -> llaminar2::ITensorGemm *
    {
        auto weight = llaminar2::test::TestTensorFactory::createIQ3_SRandom(
            {static_cast<size_t>(rows), static_cast<size_t>(cols)},
            static_cast<unsigned>(seed));
        auto *weight_ptr = weight.get();
        owned_weights.push_back(std::move(weight));
        prepared_weights.push_back(llaminar2::test::makeGpuPreparedGemm(
            weight_ptr,
            device,
            std::string("test.cuda_moe.qwen36_iq3_routed_verifier.") + role +
                "." + std::to_string(seed),
            llaminar2::ModelContextId{910000 + static_cast<uint64_t>(seed)}));

        auto *kernel = prepared_weights.back().kernel;
        auto *tensor_kernel = dynamic_cast<llaminar2::ITensorKernel *>(kernel);
        if (!tensor_kernel)
            throw std::runtime_error(
                "prepared CUDA IQ3_S routed GEMM must expose an explicit stream contract");
        tensor_kernel->setGPUStream(stream_);
        return kernel;
    };

    struct GemmTriplet
    {
        llaminar2::ITensorGemm *gate = nullptr;
        llaminar2::ITensorGemm *up = nullptr;
        llaminar2::ITensorGemm *down = nullptr;
    };

    std::array<GemmTriplet, routed_variants> routed{};
    for (int variant = 0; variant < routed_variants; ++variant)
    {
        routed[static_cast<size_t>(variant)].gate =
            add_prepared_iq3(intermediate, d_model, 911000 + variant, "routed_gate");
        routed[static_cast<size_t>(variant)].up =
            add_prepared_iq3(intermediate, d_model, 912000 + variant, "routed_up");
        routed[static_cast<size_t>(variant)].down =
            add_prepared_iq3(d_model, intermediate, 913000 + variant, "routed_down");
    }

    auto variant_for_expert = [&](int expert_id) -> size_t
    {
        return static_cast<size_t>(expert_id % routed_variants);
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs(
        static_cast<size_t>(num_experts));
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs(
        static_cast<size_t>(num_experts));
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs(
        static_cast<size_t>(num_experts));
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const GemmTriplet &triplet = routed[variant_for_expert(expert)];
        ASSERT_TRUE(triplet.gate->exportNativeVNNIMatrixDesc(
            gate_descs[static_cast<size_t>(expert)]));
        ASSERT_TRUE(triplet.up->exportNativeVNNIMatrixDesc(
            up_descs[static_cast<size_t>(expert)]));
        ASSERT_TRUE(triplet.down->exportNativeVNNIMatrixDesc(
            down_descs[static_cast<size_t>(expert)]));
    }

    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    auto make_hidden = [](int seq_len)
    {
        auto hidden = llaminar2::test::TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        for (size_t i = 0; i < hidden->numel(); ++i)
        {
            hidden->mutable_data()[i] =
                0.013f * static_cast<float>(static_cast<int>(i % 43) - 21) +
                0.004f * static_cast<float>(static_cast<int>((i / 17) % 19) - 9);
        }
        return hidden;
    };

    auto make_routes = [](int seq_len, std::vector<float> &indices, std::vector<float> &weights)
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

    for (int seq_len : {2, 3, 4})
    {
        auto hidden = make_hidden(seq_len);
        ASSERT_TRUE(hidden->ensureOnDevice(device, stream_));

        std::vector<float> route_indices;
        std::vector<float> route_weights;
        make_routes(seq_len, route_indices, route_weights);
        auto routing_indices = llaminar2::test::TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
        auto routing_weights = llaminar2::test::TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
        std::copy(route_indices.begin(), route_indices.end(), routing_indices->mutable_data());
        std::copy(route_weights.begin(), route_weights.end(), routing_weights->mutable_data());
        ASSERT_TRUE(routing_indices->ensureOnDevice(device, stream_));
        ASSERT_TRUE(routing_weights->ensureOnDevice(device, stream_));

        std::vector<float> row_by_row_expected(
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model));
        for (int row = 0; row < seq_len; ++row)
        {
            auto hidden_row = llaminar2::test::TestTensorFactory::createFP32(
                {1u, static_cast<size_t>(d_model)});
            std::copy(hidden->data() + static_cast<size_t>(row) * d_model,
                      hidden->data() + static_cast<size_t>(row + 1) * d_model,
                      hidden_row->mutable_data());
            ASSERT_TRUE(hidden_row->ensureOnDevice(device, stream_));

            std::array<int, top_k> expert_ids = {};
            std::array<float, top_k> expert_weights = {};
            for (int route = 0; route < top_k; ++route)
            {
                const int slot = row * top_k + route;
                expert_ids[static_cast<size_t>(route)] =
                    static_cast<int>(route_indices[static_cast<size_t>(slot)]);
                expert_weights[static_cast<size_t>(route)] =
                    route_weights[static_cast<size_t>(slot)];
            }

            std::array<std::shared_ptr<llaminar2::FP32Tensor>, top_k> gate_owned;
            std::array<std::shared_ptr<llaminar2::FP32Tensor>, top_k> up_owned;
            std::array<llaminar2::ITensor *, top_k> gate_outputs = {};
            std::array<llaminar2::ITensor *, top_k> up_outputs = {};
            for (int route = 0; route < top_k; ++route)
            {
                gate_owned[static_cast<size_t>(route)] =
                    llaminar2::test::TestTensorFactory::createFP32(
                        {1u, static_cast<size_t>(intermediate)});
                up_owned[static_cast<size_t>(route)] =
                    llaminar2::test::TestTensorFactory::createFP32(
                        {1u, static_cast<size_t>(intermediate)});
                ASSERT_TRUE(gate_owned[static_cast<size_t>(route)]->ensureOnDevice(device, stream_));
                ASSERT_TRUE(up_owned[static_cast<size_t>(route)]->ensureOnDevice(device, stream_));
                gate_outputs[static_cast<size_t>(route)] = gate_owned[static_cast<size_t>(route)].get();
                up_outputs[static_cast<size_t>(route)] = up_owned[static_cast<size_t>(route)].get();
            }

            auto decode_output = llaminar2::test::TestTensorFactory::createFP32(
                {1u, static_cast<size_t>(d_model)});
            ASSERT_TRUE(decode_output->ensureOnDevice(device, stream_));
            ASSERT_TRUE(cuda_kernel_->groupedExpertGateUpDecodeFromTable(
                hidden_row.get(), expert_ids.data(), gateup_table, top_k,
                gate_outputs.data(), up_outputs.data(), d_model, intermediate));
            ASSERT_TRUE(cuda_kernel_->groupedExpertDownDecodeFromTable(
                gate_outputs.data(), up_outputs.data(), expert_ids.data(), expert_weights.data(),
                down_table, top_k, decode_output.get(), d_model, intermediate));
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
            decode_output->transitionTo(
                llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE,
                device);
            std::copy(decode_output->data(),
                      decode_output->data() + d_model,
                      row_by_row_expected.begin() + static_cast<size_t>(row) * d_model);
        }

        auto grouped_output = llaminar2::test::TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        ASSERT_TRUE(grouped_output->ensureOnDevice(device, stream_));
        ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
            routing_indices.get(), routing_weights.get(), seq_len, num_experts, top_k));
        ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
            hidden.get(),
            grouped_output.get(),
            gateup_table,
            down_table,
            seq_len,
            d_model,
            intermediate,
            num_experts,
            top_k));
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        grouped_output->transitionTo(
            llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE,
            device);

        expectVectorsClose(
            std::vector<float>(grouped_output->data(),
                               grouped_output->data() + grouped_output->numel()),
            row_by_row_expected,
            0.9999,
            0.006,
            /*row_width=*/d_model,
            /*min_row_cosine=*/0.9998,
            /*max_row_relative_l2=*/0.008,
            /*max_row_kl=*/1.0e-4);
    }
#endif
}

TEST_F(Test__CUDAMoEKernel, RoutedOnlyVerifierPrefill_AllNativeFormats_M234MatchRowByRowDecode)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    /*
     * This is the CUDA counterpart to the ROCm all-codegroup MoE sweep.  It runs
     * every NativeVNNI tensor format that can publish through grouped routed MoE
     * verifier prefill, including format aliases that share a codebook but carry
     * different min/scale descriptor metadata.  The reference is the production
     * row-by-row grouped decode entry point, not a CPU fallback.
     */
    constexpr int d_model = 256;
    constexpr int intermediate = 256;
    constexpr int num_experts = 4;
    constexpr int top_k = 4;
    const auto device = llaminar2::DeviceId::cuda(0);

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    ScopedCudaMoEPrefillConfig prefill_config;
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);
    ScopedCudaMoEGemmConfig gemm_config;
    gemm_config.set(
        /*gateup_kpart=*/true,
        /*gateup_kparts=*/4,
        /*down_kpart=*/true,
        /*down_kparts=*/4);

    for (const auto &format : cudaMoEGroupedNativeFormats())
    {
        SCOPED_TRACE(format.label);

        std::vector<std::unique_ptr<llaminar2::TensorBase>> owned_weights;
        std::vector<llaminar2::test::GpuPreparedGemm> prepared_weights;
        owned_weights.reserve(static_cast<size_t>(num_experts * 3));
        prepared_weights.reserve(static_cast<size_t>(num_experts * 3));

        auto add_prepared = [&](int rows,
                                int cols,
                                uint32_t seed,
                                const char *role) -> llaminar2::ITensorGemm *
        {
            auto weight = format.create(
                {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                seed);
            auto *weight_ptr = weight.get();
            owned_weights.push_back(std::move(weight));
            prepared_weights.push_back(llaminar2::test::makeGpuPreparedGemm(
                weight_ptr,
                device,
                std::string("test.cuda_moe.all_native_formats.") + format.label +
                    "." + role + "." + std::to_string(seed),
                llaminar2::ModelContextId{1320000 + static_cast<uint64_t>(seed)}));

            auto *kernel = prepared_weights.back().kernel;
            if (auto *tensor_kernel = dynamic_cast<llaminar2::ITensorKernel *>(kernel))
                tensor_kernel->setGPUStream(stream_);

            llaminar2::DeviceNativeVNNIMatrixDesc desc{};
            const bool exported = kernel->exportNativeVNNIMatrixDesc(desc);
            EXPECT_TRUE(exported) << format.label << " descriptor export for " << role;
            if (!exported)
                return nullptr;
            EXPECT_EQ(desc.codebook_id, format.codebook_id)
                << format.label << " must exercise the expected CUDA grouped MoE codebook";
            EXPECT_EQ(desc.n, rows);
            EXPECT_EQ(desc.k, cols);
            if (desc.codebook_id != format.codebook_id)
                return nullptr;
            return kernel;
        };

        struct ExpertTriplet
        {
            llaminar2::ITensorGemm *gate = nullptr;
            llaminar2::ITensorGemm *up = nullptr;
            llaminar2::ITensorGemm *down = nullptr;
        };

        std::array<ExpertTriplet, num_experts> experts{};
        for (int expert = 0; expert < num_experts; ++expert)
        {
            experts[static_cast<size_t>(expert)].gate =
                add_prepared(intermediate, d_model,
                             601000u + static_cast<uint32_t>(expert),
                             "gate");
            ASSERT_NE(experts[static_cast<size_t>(expert)].gate, nullptr);
            experts[static_cast<size_t>(expert)].up =
                add_prepared(intermediate, d_model,
                             602000u + static_cast<uint32_t>(expert),
                             "up");
            ASSERT_NE(experts[static_cast<size_t>(expert)].up, nullptr);
            experts[static_cast<size_t>(expert)].down =
                add_prepared(d_model, intermediate,
                             603000u + static_cast<uint32_t>(expert),
                             "down");
            ASSERT_NE(experts[static_cast<size_t>(expert)].down, nullptr);
        }

        std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs(num_experts);
        std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs(num_experts);
        std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs(num_experts);
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const auto &triplet = experts[static_cast<size_t>(expert)];
            ASSERT_TRUE(triplet.gate->exportNativeVNNIMatrixDesc(gate_descs[static_cast<size_t>(expert)]));
            ASSERT_TRUE(triplet.up->exportNativeVNNIMatrixDesc(up_descs[static_cast<size_t>(expert)]));
            ASSERT_TRUE(triplet.down->exportNativeVNNIMatrixDesc(down_descs[static_cast<size_t>(expert)]));
        }

        const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
            gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
        ASSERT_GE(gateup_table, 0) << format.label << " gate/up descriptor table";
        const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
            down_descs.data(), num_experts, d_model, intermediate);
        ASSERT_GE(down_table, 0) << format.label << " down descriptor table";

        for (int seq_len : {2, 3, 4})
        {
            auto hidden = llaminar2::test::TestTensorFactory::createFP32(
                {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
            for (size_t i = 0; i < hidden->numel(); ++i)
            {
                hidden->mutable_data()[i] =
                    0.017f * static_cast<float>(static_cast<int>(i % 37) - 18) +
                    0.003f * static_cast<float>(static_cast<int>((i / 11) % 23) - 11);
            }
            ASSERT_TRUE(hidden->ensureOnDevice(device, stream_));

            auto routing_indices = llaminar2::test::TestTensorFactory::createFP32(
                {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
            auto routing_weights = llaminar2::test::TestTensorFactory::createFP32(
                {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
            for (int row = 0; row < seq_len; ++row)
            {
                float sum = 0.0f;
                for (int route = 0; route < top_k; ++route)
                {
                    const int slot = row * top_k + route;
                    routing_indices->mutable_data()[static_cast<size_t>(slot)] =
                        static_cast<float>((row + route) % num_experts);
                    routing_weights->mutable_data()[static_cast<size_t>(slot)] =
                        0.10f + 0.017f *
                                    static_cast<float>((row * 7 + route * 3) % 9);
                    sum += routing_weights->mutable_data()[static_cast<size_t>(slot)];
                }
                for (int route = 0; route < top_k; ++route)
                {
                    routing_weights->mutable_data()[static_cast<size_t>(row * top_k + route)] /=
                        sum;
                }
            }
            ASSERT_TRUE(routing_indices->ensureOnDevice(device, stream_));
            ASSERT_TRUE(routing_weights->ensureOnDevice(device, stream_));

            std::vector<float> rowwise_expected(
                static_cast<size_t>(seq_len) * static_cast<size_t>(d_model));
            for (int row = 0; row < seq_len; ++row)
            {
                auto hidden_row = llaminar2::test::TestTensorFactory::createFP32(
                    {1u, static_cast<size_t>(d_model)});
                std::copy(hidden->data() + static_cast<size_t>(row) * d_model,
                          hidden->data() + static_cast<size_t>(row + 1) * d_model,
                          hidden_row->mutable_data());
                ASSERT_TRUE(hidden_row->ensureOnDevice(device, stream_));

                std::array<int, top_k> expert_ids = {};
                std::array<float, top_k> expert_weights = {};
                for (int route = 0; route < top_k; ++route)
                {
                    const int slot = row * top_k + route;
                    expert_ids[static_cast<size_t>(route)] =
                        static_cast<int>(routing_indices->data()[static_cast<size_t>(slot)]);
                    expert_weights[static_cast<size_t>(route)] =
                        routing_weights->data()[static_cast<size_t>(slot)];
                }

                std::array<std::shared_ptr<llaminar2::FP32Tensor>, top_k> gate_owned;
                std::array<std::shared_ptr<llaminar2::FP32Tensor>, top_k> up_owned;
                std::array<llaminar2::ITensor *, top_k> gate_outputs = {};
                std::array<llaminar2::ITensor *, top_k> up_outputs = {};
                for (int route = 0; route < top_k; ++route)
                {
                    gate_owned[static_cast<size_t>(route)] =
                        llaminar2::test::TestTensorFactory::createFP32(
                            {1u, static_cast<size_t>(intermediate)});
                    up_owned[static_cast<size_t>(route)] =
                        llaminar2::test::TestTensorFactory::createFP32(
                            {1u, static_cast<size_t>(intermediate)});
                    ASSERT_TRUE(gate_owned[static_cast<size_t>(route)]->ensureOnDevice(device, stream_));
                    ASSERT_TRUE(up_owned[static_cast<size_t>(route)]->ensureOnDevice(device, stream_));
                    gate_outputs[static_cast<size_t>(route)] = gate_owned[static_cast<size_t>(route)].get();
                    up_outputs[static_cast<size_t>(route)] = up_owned[static_cast<size_t>(route)].get();
                }

                auto decode_output = llaminar2::test::TestTensorFactory::createFP32(
                    {1u, static_cast<size_t>(d_model)});
                ASSERT_TRUE(decode_output->ensureOnDevice(device, stream_));
                ASSERT_TRUE(cuda_kernel_->groupedExpertGateUpDecodeFromTable(
                    hidden_row.get(), expert_ids.data(), gateup_table, top_k,
                    gate_outputs.data(), up_outputs.data(), d_model, intermediate))
                    << format.label << " rowwise gate/up M=" << seq_len << " row=" << row;
                ASSERT_TRUE(cuda_kernel_->groupedExpertDownDecodeFromTable(
                    gate_outputs.data(), up_outputs.data(), expert_ids.data(), expert_weights.data(),
                    down_table, top_k, decode_output.get(), d_model, intermediate))
                    << format.label << " rowwise down M=" << seq_len << " row=" << row;
                ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
                decode_output->transitionTo(
                    llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE,
                    device);
                std::copy(decode_output->data(),
                          decode_output->data() + d_model,
                          rowwise_expected.begin() + static_cast<size_t>(row) * d_model);
            }

            auto grouped_output = llaminar2::test::TestTensorFactory::createFP32(
                {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
            ASSERT_TRUE(grouped_output->ensureOnDevice(device, stream_));

            llaminar2::PerfStatsCollector::reset();
            ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
                routing_indices.get(), routing_weights.get(), seq_len, num_experts, top_k))
                << format.label << " grouped expert planning M=" << seq_len;
            ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
                hidden.get(),
                grouped_output.get(),
                gateup_table,
                down_table,
                seq_len,
                d_model,
                intermediate,
                num_experts,
                top_k))
                << format.label << " grouped verifier prefill M=" << seq_len;
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
            grouped_output->transitionTo(
                llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE,
                device);

            const std::vector<float> grouped_values(
                grouped_output->data(),
                grouped_output->data() + grouped_output->numel());
            ASSERT_GT(l2Norm(rowwise_expected.data(), rowwise_expected.size()), 1.0e-7)
                << format.label << " produced an all-zero serial decode witness";
            ASSERT_GT(l2Norm(grouped_values.data(), grouped_values.size()), 1.0e-7)
                << format.label << " produced an all-zero grouped decode witness";
            expectVectorsClose(
                grouped_values,
                rowwise_expected,
                0.9999,
                0.006,
                /*row_width=*/d_model,
                /*min_row_cosine=*/0.9998,
                /*max_row_relative_l2=*/0.008,
                /*max_row_kl=*/1.0e-4);
            expectPrefillSwiGLUPathRecord("fused", seq_len, top_k, num_experts, 2,
                                          "kpart_swiglu",
                                          "kpart_prefill",
                                          "token_direct",
                                          /*expected_active_expert_slots=*/num_experts);
        }
    }

    llaminar2::PerfStatsCollector::reset();
#endif
}

TEST_F(Test__CUDAMoEKernel, FixedTopologyRuntimeGroupedPrefillLargeAllExpertPathCapturesAndReplays)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    ScopedCudaMoEPrefillConfig prefill_config;
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);
    llaminar2::PerfStatsCollector::reset();

    constexpr int seq_len = 33;
    constexpr int top_k = 2;
    constexpr int num_experts = 128;
    constexpr int d_model = 32;
    constexpr int intermediate = 32;
    constexpr size_t descriptor_bytes = 4096;

    std::vector<CudaAllocation> payloads;
    std::vector<CudaAllocation> scales;
    payloads.reserve(static_cast<size_t>(num_experts * 3));
    scales.reserve(static_cast<size_t>(num_experts * 3));

    auto add_desc = [&](int rows, int cols)
    {
        payloads.emplace_back(descriptor_bytes);
        scales.emplace_back(descriptor_bytes);
        EXPECT_EQ(cudaMemsetAsync(payloads.back().get(), 0, descriptor_bytes, stream_), cudaSuccess);

        const int scale_count = rows * (cols / 32);
        std::vector<uint16_t> host_scales(static_cast<size_t>(scale_count), 0x3c00u);
        EXPECT_EQ(cudaMemcpyAsync(scales.back().get(), host_scales.data(),
                                  host_scales.size() * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        return makeCudaNativeDesc(payloads.back(), scales.back(), rows, cols);
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    down_descs.reserve(num_experts);
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        down_descs.push_back(add_desc(d_model, intermediate));
        gate_descs.push_back(add_desc(intermediate, d_model));
        up_descs.push_back(add_desc(intermediate, d_model));
    }

    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);
    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);

    std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
    for (size_t i = 0; i < hidden_values.size(); ++i)
        hidden_values[i] = 0.01f * static_cast<float>(static_cast<int>(i % 17) - 8);

    std::vector<float> routing_indices(static_cast<size_t>(seq_len) * top_k);
    std::vector<float> routing_weights(static_cast<size_t>(seq_len) * top_k);
    for (int slot = 0; slot < seq_len * top_k; ++slot)
    {
        routing_indices[static_cast<size_t>(slot)] =
            static_cast<float>((slot * 17 + 5) % num_experts);
        routing_weights[static_cast<size_t>(slot)] =
            (slot % top_k == 0) ? 0.625f : 0.375f;
    }

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto output = makeZeros({seq_len, d_model});
    auto routing_tensor = makeTensor({seq_len, top_k}, routing_indices);
    auto weights_tensor = makeTensor({seq_len, top_k}, routing_weights);

    // Warmup allocates grouping/prefill scratch and pins tensor residency before capture.
    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k));
    ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const auto active_grid_records =
        llaminar2::PerfStatsCollector::snapshot({"kernel.cuda_moe_grouped_prefill_active_expert_grid_calls"});
    EXPECT_FALSE(active_grid_records.empty())
        << "large prompt prefill should use the graph-capturable compact active-expert grid";
    expectPrefillSwiGLUPathRecord("fused", seq_len, top_k, num_experts, 16,
                                  "kpart_swiglu", "kpart_prefill", "token_direct");

    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_grouping = cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k);
    const bool captured_prefill = cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_grouping);
    EXPECT_TRUE(captured_prefill);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);

    cudaGraphExec_t executable = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const float *output_data = output->data();
    for (size_t i = 0; i < output->numel(); ++i)
        ASSERT_TRUE(std::isfinite(output_data[i])) << "output element " << i;

    ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, FixedTopologyRuntimeGroupedPrefillGraphReplayClearsInvalidPaddedRouteMappings)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedCudaMoEPrefillConfig prefill_config;
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);
    ScopedEnv perf_env("LLAMINAR_PERF_STATS_JSON", "1");
    llaminar2::PerfStatsCollector::reset();

    constexpr int seq_len = 33;
    constexpr int real_seq_len = 29;
    constexpr int top_k = 2;
    constexpr int num_experts = 128;
    constexpr int d_model = 32;
    constexpr int intermediate = 32;
    constexpr size_t descriptor_bytes = 4096;
    const auto device = llaminar2::DeviceId::cuda(0);
    static_assert(seq_len * top_k > 64, "test must use the all-expert prefill grouping path");

    std::vector<CudaAllocation> payloads;
    std::vector<CudaAllocation> scales;
    payloads.reserve(static_cast<size_t>(num_experts * 3));
    scales.reserve(static_cast<size_t>(num_experts * 3));

    auto add_desc = [&](int rows, int cols)
    {
        payloads.emplace_back(descriptor_bytes);
        scales.emplace_back(descriptor_bytes);
        EXPECT_EQ(cudaMemsetAsync(payloads.back().get(), 0x11, descriptor_bytes, stream_), cudaSuccess);

        const int scale_count = rows * (cols / 32);
        std::vector<uint16_t> host_scales(static_cast<size_t>(scale_count), 0x3c00u);
        EXPECT_EQ(cudaMemcpyAsync(scales.back().get(), host_scales.data(),
                                  host_scales.size() * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        return makeCudaNativeDesc(payloads.back(), scales.back(), rows, cols);
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    down_descs.reserve(num_experts);
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        down_descs.push_back(add_desc(d_model, intermediate));
        gate_descs.push_back(add_desc(intermediate, d_model));
        up_descs.push_back(add_desc(intermediate, d_model));
    }

    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);
    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);

    std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
    for (size_t i = 0; i < hidden_values.size(); ++i)
        hidden_values[i] = 0.015f * static_cast<float>(static_cast<int>(i % 23) - 11);

    std::vector<float> routing_indices(static_cast<size_t>(seq_len) * top_k);
    std::vector<float> routing_weights(static_cast<size_t>(seq_len) * top_k);
    auto fill_routes = [&](bool mask_tail)
    {
        for (int row = 0; row < seq_len; ++row)
        {
            for (int k = 0; k < top_k; ++k)
            {
                const int slot = row * top_k + k;
                if (mask_tail && row >= real_seq_len)
                {
                    routing_indices[static_cast<size_t>(slot)] = -1.0f;
                    routing_weights[static_cast<size_t>(slot)] = 0.0f;
                }
                else
                {
                    routing_indices[static_cast<size_t>(slot)] =
                        static_cast<float>((row * 17 + k * 29 + 3) % num_experts);
                    routing_weights[static_cast<size_t>(slot)] = (k == 0) ? 0.75f : 0.25f;
                }
            }
        }
    };

    fill_routes(/*mask_tail=*/false);
    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto routing_tensor = makeTensor({seq_len, top_k}, routing_indices);
    auto weights_tensor = makeTensor({seq_len, top_k}, routing_weights);
    auto output = makeZeros({seq_len, d_model});
    ASSERT_TRUE(hidden->ensureOnDevice(device, stream_));
    ASSERT_TRUE(routing_tensor->ensureOnDevice(device, stream_));
    ASSERT_TRUE(weights_tensor->ensureOnDevice(device, stream_));
    ASSERT_TRUE(output->ensureOnDevice(device, stream_));

    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k));
    ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_grouping = cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k);
    const bool captured_prefill = cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_grouping);
    EXPECT_TRUE(captured_prefill);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);

    cudaGraphExec_t executable = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    fill_routes(/*mask_tail=*/true);
    std::copy(routing_indices.begin(), routing_indices.end(), routing_tensor->mutable_data());
    std::copy(routing_weights.begin(), routing_weights.end(), weights_tensor->mutable_data());
    std::fill(output->mutable_data(), output->mutable_data() + output->numel(), 77.0f);
    ASSERT_TRUE(routing_tensor->ensureOnDevice(device, stream_));
    ASSERT_TRUE(weights_tensor->ensureOnDevice(device, stream_));
    ASSERT_TRUE(output->ensureOnDevice(device, stream_));

    ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    output->transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    expectPrefillSwiGLUPathRecord("fused", seq_len, top_k, num_experts, 16,
                                  "kpart_swiglu", "kpart_prefill", "token_direct");

    const float *output_data = output->data();
    int nonzero_real_rows = 0;
    for (int row = 0; row < real_seq_len; ++row)
    {
        double row_norm = 0.0;
        for (int col = 0; col < d_model; ++col)
            row_norm += std::abs(output_data[static_cast<size_t>(row) * d_model + col]);
        if (row_norm > 1.0e-6)
            ++nonzero_real_rows;
    }
    EXPECT_GT(nonzero_real_rows, 0)
        << "replayed graph did not produce any nonzero real rows";
    for (int row = real_seq_len; row < seq_len; ++row)
    {
        for (int col = 0; col < d_model; ++col)
        {
            const size_t idx = static_cast<size_t>(row) * d_model + col;
            EXPECT_EQ(output_data[idx], 0.0f)
                << "padded row " << row << " col " << col
                << " reused stale grouped route mapping";
        }
    }

    ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, FixedTopologyRuntimeGroupedPrefillQ8FusedMatchesSplitUnderGraphReplay)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    ScopedCudaMoEPrefillConfig prefill_config;
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);
    llaminar2::PerfStatsCollector::reset();

    constexpr int seq_len = 64;
    constexpr int top_k = 8;
    constexpr int num_experts = 256;
    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    constexpr int q8_codebook = 19;
    const auto device = llaminar2::DeviceId::cuda(0);

    std::vector<std::unique_ptr<llaminar2::TensorBase>> owned_weights;
    std::vector<llaminar2::test::GpuPreparedGemm> prepared_weights;
    owned_weights.reserve(3);
    prepared_weights.reserve(3);

    auto add_prepared_q8_desc = [&](int rows, int cols, int seed, const char *role)
    {
        auto weight = llaminar2::test::TestTensorFactory::createQ8_0Random(
            {static_cast<size_t>(rows), static_cast<size_t>(cols)},
            static_cast<unsigned>(seed));
        auto *weight_ptr = weight.get();
        owned_weights.push_back(std::move(weight));
        prepared_weights.push_back(llaminar2::test::makeGpuPreparedGemm(
            weight_ptr,
            device,
            std::string("test.cuda_moe.large_q8_prefill.") + role + "." + std::to_string(seed),
            llaminar2::ModelContextId{910000 + static_cast<uint64_t>(seed)}));

        llaminar2::DeviceNativeVNNIMatrixDesc desc{};
        EXPECT_TRUE(prepared_weights.back().kernel->exportNativeVNNIMatrixDesc(desc))
            << "failed to export Q8_0 native descriptor for " << role;
        EXPECT_EQ(desc.n, rows);
        EXPECT_EQ(desc.k, cols);
        EXPECT_EQ(desc.codebook_id, q8_codebook);
        return desc;
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    down_descs.reserve(num_experts);
    const auto gate_desc = add_prepared_q8_desc(intermediate, d_model, 5100, "gate");
    const auto up_desc = add_prepared_q8_desc(intermediate, d_model, 5200, "up");
    const auto down_desc = add_prepared_q8_desc(d_model, intermediate, 5300, "down");
    for (int expert = 0; expert < num_experts; ++expert)
    {
        gate_descs.push_back(gate_desc);
        up_descs.push_back(up_desc);
        down_descs.push_back(down_desc);
    }

    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
    for (size_t i = 0; i < hidden_values.size(); ++i)
    {
        hidden_values[i] =
            0.0105f * static_cast<float>(static_cast<int>(i % 31) - 15) +
            0.0025f * static_cast<float>(static_cast<int>((i / 13) % 17) - 8);
    }

    std::vector<float> routing_indices(static_cast<size_t>(seq_len) * top_k);
    std::vector<float> routing_weights(static_cast<size_t>(seq_len) * top_k);
    for (int row = 0; row < seq_len; ++row)
    {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k)
        {
            routing_indices[static_cast<size_t>(row) * top_k + k] =
                static_cast<float>((row * 13 + k * 17 + 3) % num_experts);
            const float weight = 0.04f + 0.01f * static_cast<float>((row + 3 * k) % top_k);
            routing_weights[static_cast<size_t>(row) * top_k + k] = weight;
            sum += weight;
        }
        for (int k = 0; k < top_k; ++k)
            routing_weights[static_cast<size_t>(row) * top_k + k] /= sum;
    }

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto routing_tensor = makeTensor({seq_len, top_k}, routing_indices);
    auto weights_tensor = makeTensor({seq_len, top_k}, routing_weights);

    auto split_output = makeZeros({seq_len, d_model});
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/false);
    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k));
    ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), split_output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    const std::vector<float> split_values(
        split_output->data(),
        split_output->data() + split_output->numel());

    auto fused_output = makeZeros({seq_len, d_model});
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);

    // Warm once outside capture to bind all workspace slices and tensor residency.
    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k));
    ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), fused_output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_grouping = cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k);
    const bool captured_prefill = cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), fused_output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_grouping);
    EXPECT_TRUE(captured_prefill);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);

    cudaGraphExec_t executable = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const std::vector<float> fused_values(
        fused_output->data(),
        fused_output->data() + fused_output->numel());

    // This is the large-prompt active-expert route used by bucketed prefill graphs.
    // It must be as decode-stable as the split gate/up -> SwiGLU oracle because
    // captured prefill buckets are reused across long-context E2E requests.
    expectVectorsClose(fused_values, split_values,
                       0.9999, 0.006,
                       /*row_width=*/d_model,
                       /*min_row_cosine=*/0.9998,
                       /*max_row_relative_l2=*/0.008,
                       /*max_row_kl=*/1.0e-4);
    expectPrefillSwiGLUPathRecord("fused", seq_len, top_k, num_experts,
                                  16,
                                  "kpart_swiglu",
                                  "kpart_prefill",
                                  "token_direct");

    ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    llaminar2::PerfStatsCollector::reset();
#endif
}

TEST_F(Test__CUDAMoEKernel, FixedTopologyRuntimeGroupedPrefillIQ3SFusedMatchesSplitUnderGraphReplay)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    ScopedCudaMoEPrefillConfig prefill_config;
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);
    llaminar2::PerfStatsCollector::reset();

    constexpr int seq_len = 1536;
    constexpr int top_k = 8;
    constexpr int num_experts = 256;
    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    constexpr int iq3s_codebook = 11;
    constexpr int expert_variants = 16;
    const auto device = llaminar2::DeviceId::cuda(0);

    /*
     * Qwen3.6 MoE bucketed prefill runs this all-expert path for long prompts.
     * The verifier-sized IQ3_S tests exercise a different active-expert kernel,
     * so keep a large enough token count to force the production prefill regime.
     */
    static_assert(seq_len * top_k > 64, "test must use the all-expert prefill path");

    std::vector<std::unique_ptr<llaminar2::TensorBase>> owned_weights;
    std::vector<llaminar2::test::GpuPreparedGemm> prepared_weights;
    owned_weights.reserve(static_cast<size_t>(expert_variants * 3));
    prepared_weights.reserve(static_cast<size_t>(expert_variants * 3));

    auto add_prepared_iq3_desc = [&](int rows,
                                     int cols,
                                     int seed,
                                     const char *role) -> llaminar2::DeviceNativeVNNIMatrixDesc
    {
        auto weight = llaminar2::test::TestTensorFactory::createIQ3_SRandom(
            {static_cast<size_t>(rows), static_cast<size_t>(cols)},
            static_cast<unsigned>(seed));
        auto *weight_ptr = weight.get();
        owned_weights.push_back(std::move(weight));
        prepared_weights.push_back(llaminar2::test::makeGpuPreparedGemm(
            weight_ptr,
            device,
            std::string("test.cuda_moe.large_iq3_prefill.") + role + "." + std::to_string(seed),
            llaminar2::ModelContextId{920000 + static_cast<uint64_t>(seed)}));

        auto *tensor_kernel = dynamic_cast<llaminar2::ITensorKernel *>(prepared_weights.back().kernel);
        if (tensor_kernel == nullptr)
        {
            throw std::runtime_error("prepared CUDA IQ3_S GEMM must expose an explicit stream contract");
        }
        tensor_kernel->setGPUStream(stream_);

        llaminar2::DeviceNativeVNNIMatrixDesc desc{};
        if (!prepared_weights.back().kernel->exportNativeVNNIMatrixDesc(desc))
        {
            throw std::runtime_error(std::string("failed to export IQ3_S native descriptor for ") + role);
        }
        EXPECT_EQ(desc.n, rows);
        EXPECT_EQ(desc.k, cols);
        EXPECT_EQ(desc.codebook_id, iq3s_codebook);
        return desc;
    };

    struct ExpertTriplet
    {
        llaminar2::DeviceNativeVNNIMatrixDesc gate{};
        llaminar2::DeviceNativeVNNIMatrixDesc up{};
        llaminar2::DeviceNativeVNNIMatrixDesc down{};
    };

    std::array<ExpertTriplet, expert_variants> variants{};
    for (int variant = 0; variant < expert_variants; ++variant)
    {
        variants[static_cast<size_t>(variant)].gate =
            add_prepared_iq3_desc(intermediate, d_model, 6100 + variant, "gate");
        variants[static_cast<size_t>(variant)].up =
            add_prepared_iq3_desc(intermediate, d_model, 6200 + variant, "up");
        variants[static_cast<size_t>(variant)].down =
            add_prepared_iq3_desc(d_model, intermediate, 6300 + variant, "down");
    }

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    down_descs.reserve(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const auto &variant = variants[static_cast<size_t>(expert % expert_variants)];
        gate_descs.push_back(variant.gate);
        up_descs.push_back(variant.up);
        down_descs.push_back(variant.down);
    }

    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
    auto fill_hidden_values = [&](int epoch)
    {
        for (size_t i = 0; i < hidden_values.size(); ++i)
        {
            hidden_values[i] =
                0.0095f * static_cast<float>(static_cast<int>((i + epoch * 13) % 37) - 18) +
                0.002f * static_cast<float>(static_cast<int>(((i / 19) + epoch * 7) % 23) - 11);
        }
    };

    std::vector<float> routing_indices(static_cast<size_t>(seq_len) * top_k);
    std::vector<float> routing_weights(static_cast<size_t>(seq_len) * top_k);
    auto fill_routing_values = [&](int epoch)
    {
        for (int row = 0; row < seq_len; ++row)
        {
            float sum = 0.0f;
            for (int k = 0; k < top_k; ++k)
            {
                const int slot = row * top_k + k;
                routing_indices[static_cast<size_t>(slot)] =
                    static_cast<float>((row * 29 + k * 17 + 7 + epoch * 31) % num_experts);
                const float weight = 0.03f + 0.007f * static_cast<float>((row * 3 + 5 * k + epoch * 11) % 13);
                routing_weights[static_cast<size_t>(slot)] = weight;
                sum += weight;
            }
            for (int k = 0; k < top_k; ++k)
                routing_weights[static_cast<size_t>(row) * top_k + k] /= sum;
        }
    };
    fill_hidden_values(/*epoch=*/0);
    fill_routing_values(/*epoch=*/0);

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto routing_tensor = makeTensor({seq_len, top_k}, routing_indices);
    auto weights_tensor = makeTensor({seq_len, top_k}, routing_weights);

    auto split_output = makeZeros({seq_len, d_model});
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/false);
    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k));
    ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), split_output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    const std::vector<float> split_values(
        split_output->data(),
        split_output->data() + split_output->numel());

    auto fused_output = makeZeros({seq_len, d_model});
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);

    // Warm once outside capture so descriptor tables, workspace-backed grouping
    // buffers, and tensor residency are all stable before graph recording.
    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k));
    ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), fused_output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_grouping = cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k);
    const bool captured_prefill = cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), fused_output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_grouping);
    EXPECT_TRUE(captured_prefill);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);

    cudaGraphExec_t executable = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);

    /*
     * Bucketed prefill graph cache captures once and then replays against new
     * request buffers at the same bucket size. Mutate and re-upload the tensors
     * after instantiation so this test catches stale graph-capture assumptions
     * rather than merely relaunching identical input.
     */
    fill_hidden_values(/*epoch=*/1);
    std::copy(hidden_values.begin(), hidden_values.end(), hidden->mutable_data());
    fill_routing_values(/*epoch=*/1);
    std::copy(routing_indices.begin(), routing_indices.end(), routing_tensor->mutable_data());
    std::copy(routing_weights.begin(), routing_weights.end(), weights_tensor->mutable_data());
    ASSERT_TRUE(hidden->ensureOnDevice(device, stream_));
    ASSERT_TRUE(routing_tensor->ensureOnDevice(device, stream_));
    ASSERT_TRUE(weights_tensor->ensureOnDevice(device, stream_));

    split_output = makeZeros({seq_len, d_model});
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/false);
    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k));
    ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), split_output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    const std::vector<float> replay_split_values(
        split_output->data(),
        split_output->data() + split_output->numel());

    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);
    ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const std::vector<float> fused_values(
        fused_output->data(),
        fused_output->data() + fused_output->numel());

    expectVectorsClose(fused_values, replay_split_values,
                       0.9999, 0.006,
                       /*row_width=*/d_model,
                       /*min_row_cosine=*/0.9998,
                       /*max_row_relative_l2=*/0.008,
                       /*max_row_kl=*/1.0e-4);
    expectPrefillSwiGLUPathRecord("fused", seq_len, top_k, num_experts,
                                  16,
                                  "kpart_swiglu",
                                  "kpart_prefill",
                                  "token_direct");

    ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    llaminar2::PerfStatsCollector::reset();
#endif
}

TEST_F(Test__CUDAMoEKernel, FixedTopologyRuntimeGroupedPrefillQ6KFusedMatchesSplitUnderGraphReplay)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    ScopedCudaMoEPrefillConfig prefill_config;
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);
    llaminar2::PerfStatsCollector::reset();

    constexpr int seq_len = 1536;
    constexpr int top_k = 8;
    constexpr int num_experts = 256;
    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    constexpr int q6k_codebook = 8;
    constexpr int expert_variants = 16;
    const auto device = llaminar2::DeviceId::cuda(0);

    /*
     * The Qwen3.6 MoE CUDA E2E flake was observed in the bucketed long-prefill
     * replay path.  Perf counters showed the routed expert matrices using
     * Q6_K/codebook 8, so this locks the real production codebook into a
     * fused-versus-split graph replay regression.
     */
    static_assert(seq_len * top_k > 64, "test must use the all-expert prefill path");

    std::vector<std::unique_ptr<llaminar2::TensorBase>> owned_weights;
    std::vector<llaminar2::test::GpuPreparedGemm> prepared_weights;
    owned_weights.reserve(static_cast<size_t>(expert_variants * 3));
    prepared_weights.reserve(static_cast<size_t>(expert_variants * 3));

    auto add_prepared_q6k_desc = [&](int rows,
                                     int cols,
                                     int seed,
                                     const char *role) -> llaminar2::DeviceNativeVNNIMatrixDesc
    {
        auto weight = llaminar2::test::TestTensorFactory::createQ6_KRandom(
            {static_cast<size_t>(rows), static_cast<size_t>(cols)},
            static_cast<unsigned>(seed));
        auto *weight_ptr = weight.get();
        owned_weights.push_back(std::move(weight));
        prepared_weights.push_back(llaminar2::test::makeGpuPreparedGemm(
            weight_ptr,
            device,
            std::string("test.cuda_moe.large_q6k_prefill.") + role + "." + std::to_string(seed),
            llaminar2::ModelContextId{930000 + static_cast<uint64_t>(seed)}));

        auto *tensor_kernel = dynamic_cast<llaminar2::ITensorKernel *>(prepared_weights.back().kernel);
        if (tensor_kernel == nullptr)
        {
            throw std::runtime_error("prepared CUDA Q6_K GEMM must expose an explicit stream contract");
        }
        tensor_kernel->setGPUStream(stream_);

        llaminar2::DeviceNativeVNNIMatrixDesc desc{};
        if (!prepared_weights.back().kernel->exportNativeVNNIMatrixDesc(desc))
        {
            throw std::runtime_error(std::string("failed to export Q6_K native descriptor for ") + role);
        }
        EXPECT_EQ(desc.n, rows);
        EXPECT_EQ(desc.k, cols);
        EXPECT_EQ(desc.codebook_id, q6k_codebook);
        return desc;
    };

    struct ExpertTriplet
    {
        llaminar2::DeviceNativeVNNIMatrixDesc gate{};
        llaminar2::DeviceNativeVNNIMatrixDesc up{};
        llaminar2::DeviceNativeVNNIMatrixDesc down{};
    };

    std::array<ExpertTriplet, expert_variants> variants{};
    for (int variant = 0; variant < expert_variants; ++variant)
    {
        variants[static_cast<size_t>(variant)].gate =
            add_prepared_q6k_desc(intermediate, d_model, 7100 + variant, "gate");
        variants[static_cast<size_t>(variant)].up =
            add_prepared_q6k_desc(intermediate, d_model, 7200 + variant, "up");
        variants[static_cast<size_t>(variant)].down =
            add_prepared_q6k_desc(d_model, intermediate, 7300 + variant, "down");
    }

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    down_descs.reserve(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const auto &variant = variants[static_cast<size_t>(expert % expert_variants)];
        gate_descs.push_back(variant.gate);
        up_descs.push_back(variant.up);
        down_descs.push_back(variant.down);
    }

    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
    auto fill_hidden_values = [&](int epoch)
    {
        for (size_t i = 0; i < hidden_values.size(); ++i)
        {
            hidden_values[i] =
                0.0085f * static_cast<float>(static_cast<int>((i + epoch * 17) % 41) - 20) +
                0.00225f * static_cast<float>(static_cast<int>(((i / 23) + epoch * 5) % 29) - 14);
        }
    };

    std::vector<float> routing_indices(static_cast<size_t>(seq_len) * top_k);
    std::vector<float> routing_weights(static_cast<size_t>(seq_len) * top_k);
    auto fill_routing_values = [&](int epoch)
    {
        for (int row = 0; row < seq_len; ++row)
        {
            float sum = 0.0f;
            for (int k = 0; k < top_k; ++k)
            {
                const int slot = row * top_k + k;
                routing_indices[static_cast<size_t>(slot)] =
                    static_cast<float>((row * 31 + k * 19 + 11 + epoch * 37) % num_experts);
                const float weight =
                    0.025f + 0.009f * static_cast<float>((row * 5 + 7 * k + epoch * 13) % 17);
                routing_weights[static_cast<size_t>(slot)] = weight;
                sum += weight;
            }
            for (int k = 0; k < top_k; ++k)
                routing_weights[static_cast<size_t>(row) * top_k + k] /= sum;
        }
    };
    fill_hidden_values(/*epoch=*/0);
    fill_routing_values(/*epoch=*/0);

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto routing_tensor = makeTensor({seq_len, top_k}, routing_indices);
    auto weights_tensor = makeTensor({seq_len, top_k}, routing_weights);

    auto split_output = makeZeros({seq_len, d_model});
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/false);
    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k));
    ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), split_output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    const std::vector<float> split_values(
        split_output->data(),
        split_output->data() + split_output->numel());

    auto fused_output = makeZeros({seq_len, d_model});
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);

    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k));
    ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), fused_output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_grouping = cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k);
    const bool captured_prefill = cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), fused_output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_grouping);
    EXPECT_TRUE(captured_prefill);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);

    cudaGraphExec_t executable = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);

    /*
     * Replay against a new request payload. If the graph captured stale host
     * assumptions or fused scratch from the warmup request, this strict oracle
     * will catch the same corruption class as the long-context server lane.
     */
    fill_hidden_values(/*epoch=*/1);
    std::copy(hidden_values.begin(), hidden_values.end(), hidden->mutable_data());
    fill_routing_values(/*epoch=*/1);
    std::copy(routing_indices.begin(), routing_indices.end(), routing_tensor->mutable_data());
    std::copy(routing_weights.begin(), routing_weights.end(), weights_tensor->mutable_data());
    ASSERT_TRUE(hidden->ensureOnDevice(device, stream_));
    ASSERT_TRUE(routing_tensor->ensureOnDevice(device, stream_));
    ASSERT_TRUE(weights_tensor->ensureOnDevice(device, stream_));

    split_output = makeZeros({seq_len, d_model});
    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/false);
    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k));
    ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), split_output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    const std::vector<float> replay_split_values(
        split_output->data(),
        split_output->data() + split_output->numel());

    prefill_config.set(/*tile_m=*/0, /*fuse_swiglu=*/true);
    ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const std::vector<float> fused_values(
        fused_output->data(),
        fused_output->data() + fused_output->numel());

    expectVectorsClose(fused_values, replay_split_values,
                       0.9999, 0.006,
                       /*row_width=*/d_model,
                       /*min_row_cosine=*/0.9998,
                       /*max_row_relative_l2=*/0.008,
                       /*max_row_kl=*/1.0e-4);
    expectPrefillSwiGLUPathRecord("fused", seq_len, top_k, num_experts,
                                  16,
                                  "kpart_swiglu",
                                  "kpart_prefill",
                                  "token_direct");

    ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    llaminar2::PerfStatsCollector::reset();
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimeGroupedDecodeDescriptorPathCapturesAfterWarmup)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    llaminar2::PerfStatsCollector::reset();

    using llaminar2::DeviceMoERuntimeTable;
    constexpr int num_layers = 1;
    constexpr int num_experts = 2;
    constexpr int top_k = 2;
    constexpr int d_model = 32;
    constexpr int intermediate = 32;
    constexpr int seq_len = 1;
    constexpr size_t descriptor_bytes = 4096;

    DeviceMoERuntimeTable::Config cuda_config;
    cuda_config.device_id = llaminar2::DeviceId::cuda(0);
    cuda_config.num_layers = num_layers;
    cuda_config.num_experts = num_experts;
    cuda_config.top_k = top_k;
    cuda_config.mirror_to_device = true;
    DeviceMoERuntimeTable cuda_table(cuda_config);

    std::vector<CudaAllocation> payloads;
    std::vector<CudaAllocation> scales;
    payloads.reserve(static_cast<size_t>(num_experts * 3));
    scales.reserve(static_cast<size_t>(num_experts * 3));

    auto add_desc = [&](int rows, int cols)
    {
        payloads.emplace_back(descriptor_bytes);
        scales.emplace_back(descriptor_bytes);
        EXPECT_EQ(cudaMemsetAsync(payloads.back().get(), 0, descriptor_bytes, stream_), cudaSuccess);

        const int scale_count = rows * (cols / 32);
        std::vector<uint16_t> host_scales(static_cast<size_t>(scale_count), 0x3c00u);
        EXPECT_EQ(cudaMemcpyAsync(scales.back().get(), host_scales.data(),
                                  host_scales.size() * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        return makeCudaNativeDesc(payloads.back(), scales.back(), rows, cols);
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    down_descs.reserve(num_experts);
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        down_descs.push_back(add_desc(d_model, intermediate));
        gate_descs.push_back(add_desc(intermediate, d_model));
        up_descs.push_back(add_desc(intermediate, d_model));
    }

    llaminar2::MoEPlacementUpdate update;
    update.epoch = 1;
    update.expert_count = num_experts;
    update.experts.resize(num_experts);
    update.local_compute_mask.assign(num_experts, 0);
    update.local_compute_mask[0] = 1;
    update.replica_role.resize(num_experts, 0);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        update.experts[expert].logical_expert_id = expert;
        update.experts[expert].gate = gate_descs[expert];
        update.experts[expert].up = up_descs[expert];
        update.experts[expert].down = down_descs[expert];
    }

    ASSERT_TRUE(cuda_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(cuda_table.flipActiveBank(0, update.epoch, stream_));

    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);
    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);

    std::vector<float> hidden_values(d_model);
    std::vector<float> router_values(static_cast<size_t>(num_experts) * d_model);
    for (int i = 0; i < d_model; ++i)
    {
        hidden_values[i] = 0.01f * static_cast<float>((i % 7) - 3);
        router_values[i] = 0.02f * static_cast<float>((i % 5) - 2);
        router_values[static_cast<size_t>(d_model) + i] = 0.03f * static_cast<float>((i % 3) - 1);
    }

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto router = makeTensor({num_experts, d_model}, router_values);
    auto gate0 = makeZeros({intermediate});
    auto gate1 = makeZeros({intermediate});
    auto up0 = makeZeros({intermediate});
    auto up1 = makeZeros({intermediate});
    auto output = makeZeros({d_model});

    std::array<llaminar2::ITensor *, top_k> gate_outputs = {gate0.get(), gate1.get()};
    std::array<llaminar2::ITensor *, top_k> up_outputs = {up0.get(), up1.get()};
    auto *runtime_layer = cuda_table.deviceLayerState(0);

    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        runtime_layer, hidden.get(), router.get(), d_model, num_experts, top_k,
        true, nullptr, nullptr, /*write_legacy_outputs=*/false, /*update_runtime_histogram=*/false));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    llaminar2::DeviceMoELayerRuntime host_runtime{};
    ASSERT_EQ(cudaMemcpy(&host_runtime, runtime_layer, sizeof(host_runtime), cudaMemcpyDeviceToHost),
              cudaSuccess);
    bool saw_masked = false;
    bool saw_local = false;
    for (int slot = 0; slot < top_k; ++slot)
    {
        saw_masked = saw_masked || host_runtime.topk_expert_ids[slot] < 0;
        saw_local = saw_local || host_runtime.topk_expert_ids[slot] >= 0;
    }
    EXPECT_TRUE(saw_masked)
        << "runtime route selection should mask the nonlocal expert";
    EXPECT_TRUE(saw_local)
        << "runtime route selection should keep the local expert";
    ASSERT_TRUE(cuda_kernel_->groupedExpertGateUpDecodeFromRuntime(
        runtime_layer, hidden.get(), gateup_table, top_k,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate));
    ASSERT_TRUE(cuda_kernel_->groupedExpertDownDecodeFromRuntime(
        gate_outputs.data(), up_outputs.data(), runtime_layer, down_table, top_k,
        output.get(), d_model, intermediate));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const float *output_data = output->data();
    for (int i = 0; i < d_model; ++i)
        ASSERT_TRUE(std::isfinite(output_data[i])) << "output element " << i;

    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_gateup = cuda_kernel_->groupedExpertGateUpDecodeFromRuntime(
        runtime_layer, hidden.get(), gateup_table, top_k,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate);
    const bool captured_down = cuda_kernel_->groupedExpertDownDecodeFromRuntime(
        gate_outputs.data(), up_outputs.data(), runtime_layer, down_table, top_k,
        output.get(), d_model, intermediate);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_gateup);
    EXPECT_TRUE(captured_down);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    if (graph)
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);

    expectGroupedDecodeCounter(
        "cuda_moe_grouped_decode_gateup_calls", "runtime", top_k, d_model, intermediate);
    expectGroupedDecodeCounter(
        "cuda_moe_grouped_decode_down_calls", "runtime", top_k, d_model, intermediate);
    llaminar2::PerfStatsCollector::reset();
#endif
}

TEST_F(Test__CUDAMoEKernel, StaticGroupedDecodeDescriptorPathCapturesAfterWarmup)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    llaminar2::PerfStatsCollector::reset();

    constexpr int num_experts = 1;
    constexpr int num_active = 1;
    constexpr int d_model = 32;
    constexpr int intermediate = 32;
    constexpr size_t descriptor_bytes = 4096;
    const int expert_ids[num_active] = {0};
    const float expert_weights[num_active] = {1.0f};

    std::vector<CudaAllocation> payloads;
    std::vector<CudaAllocation> scales;
    payloads.reserve(static_cast<size_t>(num_experts * 3));
    scales.reserve(static_cast<size_t>(num_experts * 3));

    auto add_desc = [&](int rows, int cols)
    {
        payloads.emplace_back(descriptor_bytes);
        scales.emplace_back(descriptor_bytes);
        EXPECT_EQ(cudaMemset(payloads.back().get(), 0, descriptor_bytes), cudaSuccess);

        const int scale_count = rows * (cols / 32);
        std::vector<uint16_t> host_scales(static_cast<size_t>(scale_count), 0x3c00u);
        EXPECT_EQ(cudaMemcpy(scales.back().get(), host_scales.data(),
                             host_scales.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
                  cudaSuccess);
        return makeCudaNativeDesc(payloads.back(), scales.back(), rows, cols);
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    down_descs.push_back(add_desc(d_model, intermediate));
    gate_descs.push_back(add_desc(intermediate, d_model));
    up_descs.push_back(add_desc(intermediate, d_model));

    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);
    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);

    std::vector<float> hidden_values(d_model);
    for (int i = 0; i < d_model; ++i)
        hidden_values[i] = 0.01f * static_cast<float>((i % 7) - 3);

    auto hidden = makeTensor({1, d_model}, hidden_values);
    auto gate = makeZeros({intermediate});
    auto up = makeZeros({intermediate});
    auto output = makeZeros({d_model});

    ASSERT_TRUE(hidden->ensureOnDevice(llaminar2::DeviceId::cuda(0), stream_));
    ASSERT_TRUE(gate->ensureOnDevice(llaminar2::DeviceId::cuda(0), stream_));
    ASSERT_TRUE(up->ensureOnDevice(llaminar2::DeviceId::cuda(0), stream_));
    ASSERT_TRUE(output->ensureOnDevice(llaminar2::DeviceId::cuda(0), stream_));

    std::array<llaminar2::ITensor *, num_active> gate_outputs = {gate.get()};
    std::array<llaminar2::ITensor *, num_active> up_outputs = {up.get()};

    ASSERT_TRUE(cuda_kernel_->groupedExpertGateUpDecodeFromTable(
        hidden.get(), expert_ids, gateup_table, num_active,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate));
    ASSERT_TRUE(cuda_kernel_->groupedExpertDownDecodeFromTable(
        gate_outputs.data(), up_outputs.data(), expert_ids, expert_weights,
        down_table, num_active, output.get(), d_model, intermediate));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const float *output_data = output->data();
    for (int i = 0; i < d_model; ++i)
        ASSERT_TRUE(std::isfinite(output_data[i])) << "output element " << i;

    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_gateup = cuda_kernel_->groupedExpertGateUpDecodeFromTable(
        hidden.get(), expert_ids, gateup_table, num_active,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate);
    const bool captured_down = cuda_kernel_->groupedExpertDownDecodeFromTable(
        gate_outputs.data(), up_outputs.data(), expert_ids, expert_weights,
        down_table, num_active, output.get(), d_model, intermediate);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_gateup);
    EXPECT_TRUE(captured_down);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    if (graph)
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);

    auto *workspace_consumer =
        dynamic_cast<llaminar2::IWorkspaceConsumer *>(cuda_kernel_);
    ASSERT_NE(workspace_consumer, nullptr);
    auto reqs = llaminar2::MoEWorkspaceBuffers::cudaMoE(
        /*max_seq_len=*/64,
        /*d_model=*/2048,
        /*intermediate=*/512,
        /*num_experts=*/256,
        /*top_k=*/16);
    reqs.merge(llaminar2::MoEWorkspaceBuffers::cudaMoE(
        /*max_seq_len=*/4,
        /*d_model=*/2048,
        /*intermediate=*/512,
        /*num_experts=*/257,
        /*top_k=*/16));
    auto second_workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
        llaminar2::DeviceId::cuda(0),
        reqs.total_bytes_with_alignment() + 4 * 1024 * 1024);
    ASSERT_TRUE(second_workspace->allocate(reqs));
    workspace_consumer->bindWorkspace(second_workspace.get());

    ASSERT_TRUE(cuda_kernel_->groupedExpertGateUpDecodeFromTable(
        hidden.get(), expert_ids, gateup_table, num_active,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate))
        << "Static grouped decode descriptor table ids must survive MoE workspace rebind.";
    ASSERT_TRUE(cuda_kernel_->groupedExpertDownDecodeFromTable(
        gate_outputs.data(), up_outputs.data(), expert_ids, expert_weights,
        down_table, num_active, output.get(), d_model, intermediate))
        << "Static grouped decode down table ids must survive MoE workspace rebind.";
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    workspace_consumer->bindWorkspace(workspace_.get());

    expectGroupedDecodeCounter(
        "cuda_moe_grouped_decode_gateup_calls", "table", num_active, d_model, intermediate);
    expectGroupedDecodeCounter(
        "cuda_moe_grouped_decode_down_calls", "table", num_active, d_model, intermediate);
    llaminar2::PerfStatsCollector::reset();
#endif
}
