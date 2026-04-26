/**
 * @file MoEFFNStage.cpp
 * @brief Implementation of unified MoE FFN, shared expert, and shared expert gate stages
 */

#include "MoEFFNStage.h"
#include "../../../execution/moe/DecodeExpertHistogram.h"
#include "../../../execution/moe/ExpertWeightTransfer.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../../kernels/PackedWeightsSerialization.h"
#include "../../../kernels/cpu/native_vnni/CPUPackedWeights.h"
#include "../../../kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
#include "../../../kernels/cpu/primitives/VectorPrimitives.h"
#include "../../../kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "../../../loaders/MmapRegion.h"
#include "../../../utils/Assertions.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"
#include <mpi.h>

#ifdef HAVE_CUDA
#include "../../../kernels/cuda/gemm/CUDAWeightPacker.h"
#include "../../../kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#endif

#ifdef HAVE_ROCM
#include "../../../kernels/rocm/ROCmWeightPacker.h"
#include "../../../kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <vector>

#ifdef __linux__
#include <unistd.h>   // syscall
#include <sys/syscall.h>
#include <numaif.h>    // move_pages
#include <sched.h>     // sched_getcpu
#endif

namespace llaminar2
{
    // Alias for fully-qualified KernelFactory access
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    namespace
    {
        /// Query the NUMA node of a virtual address using move_pages(2).
        /// Returns -1 if NUMA info unavailable (non-Linux or unmapped page).
        static int queryNUMANode(const void* ptr) {
#ifdef __linux__
            if (!ptr) return -1;
            void* pages[] = { const_cast<void*>(ptr) };
            int status[1] = { -1 };
            if (move_pages(0, 1, pages, nullptr, status, 0) == 0 && status[0] >= 0)
                return status[0];
#endif
            (void)ptr;
            return -1;
        }

        /// Get NUMA node of the CPU this thread is currently running on.
        static int currentCPUNode() {
#ifdef __linux__
            int cpu = sched_getcpu();
            if (cpu < 0) return -1;
            // /sys/devices/system/cpu/cpuN/topology/physical_package_id
            // approximates socket / NUMA node for typical 1-socket-per-NUMA configs.
            // Simpler: use move_pages on our own stack.
            unsigned char stack_byte;
            return queryNUMANode(&stack_byte);
#else
            return -1;
#endif
        }

        /// Audit NUMA placement of expert GEMM weights.
        /// Samples up to `max_sample` experts and logs the NUMA node of their
        /// native_interleaved data pointer, comparing against the expected node.
        /// @param experts    Expert IDs to audit
        /// @param gate_gemms Gate GEMM engines per expert
        /// @param label      Human-readable label for the log (e.g., "initial_pack", "rebalance")
        /// @param layer_idx  Layer index for log context
        /// @param max_sample Maximum experts to sample (0 = all)
        static void auditExpertNUMA(
            const std::vector<int>& experts,
            const std::vector<ITensorGemm*>& gate_gemms,
            const char* label,
            int layer_idx,
            int max_sample = 8)
        {
            if (experts.empty()) return;

            const int expected_node = currentCPUNode();
            if (expected_node < 0) {
                LOG_DEBUG("[MoEFFNStage][NUMA] Cannot determine current NUMA node, skipping audit");
                return;
            }

            int sampled = 0, local = 0, remote = 0, unmapped = 0;
            int first_remote_expert = -1;
            int first_remote_node = -1;

            const int step = (max_sample > 0 && static_cast<int>(experts.size()) > max_sample)
                           ? static_cast<int>(experts.size()) / max_sample : 1;

            for (size_t i = 0; i < experts.size(); i += step) {
                const int e = experts[i];
                if (e < 0 || e >= static_cast<int>(gate_gemms.size()) || !gate_gemms[e])
                    continue;

                auto* vnni_kernel = dynamic_cast<const cpu::native_vnni::CPUNativeVNNIGemmKernel*>(gate_gemms[e]);
                if (!vnni_kernel) continue;

                const auto& packed = vnni_kernel->packedWeights();
                const void* data_ptr = packed.interleavedBase();
                if (!data_ptr) continue;

                int node = queryNUMANode(data_ptr);
                ++sampled;

                if (node < 0) {
                    ++unmapped;
                } else if (node != expected_node) {
                    ++remote;
                    if (first_remote_expert < 0) {
                        first_remote_expert = e;
                        first_remote_node = node;
                    }
                } else {
                    ++local;
                }
            }

            if (sampled == 0) return;

            if (remote > 0) {
                LOG_WARN("[MoEFFNStage][NUMA] layer " << layer_idx << " " << label
                         << ": " << remote << "/" << sampled << " sampled experts on WRONG NUMA node"
                         << " (expected=" << expected_node
                         << ", first_remote: expert " << first_remote_expert
                         << " on node " << first_remote_node << ")"
                         << (unmapped > 0 ? (std::string(" [") + std::to_string(unmapped) + " unmapped]") : ""));
            } else {
                LOG_DEBUG("[MoEFFNStage][NUMA] layer " << layer_idx << " " << label
                         << ": all " << sampled << "/" << static_cast<int>(experts.size())
                         << " sampled experts on correct NUMA node " << expected_node
                         << (unmapped > 0 ? (std::string(" [") + std::to_string(unmapped) + " unmapped]") : ""));
            }
        }

        /// Execute SwiGLU activation + Down projection via fused kernel when available,
        /// falling back to IMoEKernel::swiGLU + separate GEMM when not (e.g., FP32 weights).
        /// @param gate_tensor [m, intermediate] — gate projection output (modified in-place on fallback)
        /// @param up_tensor [m, intermediate] — up projection output
        /// @param output [m, n] — final output
        /// @param down_gemm GEMM engine for down projection
        /// @param moe_kernel MoE kernel for SwiGLU fallback
        /// @param m sequence length / batch size
        /// @param n output dimension (d_model)
        /// @param intermediate intermediate dimension
        void fusedSwigluDown(
            FP32Tensor *gate_tensor, FP32Tensor *up_tensor, TensorBase *output,
            ITensorGemm *down_gemm, IMoEKernel *moe_kernel,
            int m, int n, int intermediate)
        {
            // Try fused path first (quantized GEMM engines support this)
            if (down_gemm->multiply_tensor_with_fused_swiglu(
                    gate_tensor, up_tensor, output,
                    m, n, intermediate))
            {
                return;
            }

            // Fallback: SwiGLU via device-agnostic kernel, then separate down GEMM
            float *g = gate_tensor->mutable_data();
            const float *u = up_tensor->data();
            const int count = m * intermediate;
            moe_kernel->swiGLU(g, u, count);
            down_gemm->multiply_tensor(
                gate_tensor, output,
                m, n, intermediate);
        }
    } // anonymous namespace

    // =========================================================================
    // MoEFFNStage — Unified Router + Expert FFN + Combine
    // =========================================================================

    MoEFFNStage::MoEFFNStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool MoEFFNStage::updateExpertMask(const std::vector<bool>& mask) {
        if (static_cast<int>(mask.size()) != params_.num_experts) {
            LOG_ERROR("[MoEFFNStage] Expert mask size " << mask.size()
                      << " != num_experts " << params_.num_experts);
            return false;
        }
        params_.expert_mask = mask;
        return true;
    }

    ExpertWeightBlobs MoEFFNStage::detachAndSerializeExpert(int expert_id) {
        ExpertWeightBlobs blobs;

        auto serialize_proj = [&](ITensorGemm* engine, const char* /*proj_name*/) -> std::vector<uint8_t> {
            if (!engine) return {};
            if (!engine->hasWeights()) return {};
            auto packed = engine->detachWeights();
            if (!packed) return {};
            return packed_weights_serialization::serialize(*packed);
        };

        blobs.gate = serialize_proj(params_.prepared_gate_gemm[expert_id], "gate");
        blobs.up   = serialize_proj(params_.prepared_up_gemm[expert_id], "up");
        blobs.down = serialize_proj(params_.prepared_down_gemm[expert_id], "down");

        return blobs;
    }

    ExpertWeightBlobs MoEFFNStage::serializeExpert(int expert_id) const {
        using namespace cpu::native_vnni;

        ExpertWeightBlobs blobs;

        auto serialize_proj = [](ITensorGemm* engine) -> std::vector<uint8_t> {
            if (!engine || !engine->hasWeights()) return {};
            auto* vnni_kernel = dynamic_cast<const CPUNativeVNNIGemmKernel*>(engine);
            if (!vnni_kernel) return {};
            // Wrap const ref — CPUPackedWeights const-ref ctor deep-copies.
            // serialize() only reads, so the copy is the data we send over MPI.
            CPUPackedWeights wrapper(vnni_kernel->packedWeights());
            return packed_weights_serialization::serialize(wrapper);
        };

        blobs.gate = serialize_proj(params_.prepared_gate_gemm[expert_id]);
        blobs.up   = serialize_proj(params_.prepared_up_gemm[expert_id]);
        blobs.down = serialize_proj(params_.prepared_down_gemm[expert_id]);

        return blobs;
    }

    bool MoEFFNStage::registerTransferredExpert(int expert_id, const ExpertWeightBlobs& blobs) {
        using namespace cpu::native_vnni;

        auto register_one = [&](const std::vector<uint8_t>& blob,
                                const std::shared_ptr<TensorBase>& view) -> bool {
            if (blob.empty() || !view) return false;

            auto packed_weights = packed_weights_serialization::deserialize(blob.data(), blob.size());
            if (!packed_weights) return false;

            auto* cpu_pw = dynamic_cast<CPUPackedWeights*>(packed_weights.get());
            if (!cpu_pw) return false;

            auto kernel = std::make_unique<CPUNativeVNNIGemmKernel>(cpu_pw->takePacked());
            return KernelFactory::registerPreparedGemmFromTransfer(
                view.get(), params_.device_id, std::move(kernel)) != nullptr;
        };

        bool gate_ok = register_one(blobs.gate, params_.expert_gate_views[expert_id]);
        bool up_ok   = register_one(blobs.up,   params_.expert_up_views[expert_id]);
        bool down_ok = register_one(blobs.down,  params_.expert_down_views[expert_id]);

        return gate_ok && up_ok && down_ok;
    }

    size_t MoEFFNStage::releaseRawExpertWeights() {
        size_t freed = 0;

        // Release heap-allocated raw data from 3D parent tensors.
        // For mmap-backed tensors, the madvise(MADV_DONTNEED) already happened
        // in prepareExpertGemmEngines(). For heap-allocated tensors, this frees
        // the raw_data_ vector — the only way to reclaim that memory.
        auto release_3d = [&](TensorBase* tensor, const char* name) {
            if (!tensor) return;
            if (tensor->is_mmap_data()) {
                // Already madvised at pack time — just count the bytes
                freed += tensor->size_bytes();
                LOG_DEBUG("[MoEFFNStage] " << name << ": mmap-backed ("
                          << (tensor->size_bytes() >> 20) << " MB) — already DONTNEED");
            } else if (!tensor->is_raw_data_released()) {
                size_t bytes = tensor->size_bytes();
                tensor->release_raw_data();
                freed += bytes;
                LOG_DEBUG("[MoEFFNStage] " << name << ": released "
                          << (bytes >> 20) << " MB heap data");
            }
        };

        release_3d(params_.gate_exps, "gate_exps");
        release_3d(params_.up_exps, "up_exps");
        release_3d(params_.down_exps, "down_exps");

        // Null out 3D pointers to prevent accidental fallback to raw repacking.
        // Views remain valid as KernelFactory cache keys.
        params_.gate_exps = nullptr;
        params_.up_exps = nullptr;
        params_.down_exps = nullptr;
        raw_weights_released_ = true;

        if (freed > 0) {
            LOG_INFO("[MoEFFNStage] Layer " << params_.layer_idx
                     << ": released " << (freed >> 20) << " MB raw expert weights"
                     << " (VNNI engines retain packed data)");
        }

        return freed;
    }

    bool MoEFFNStage::updateExpertMaskAndPrepareEngines(
        const std::vector<bool>& new_mask,
        const std::unordered_map<int, ExpertWeightBlobs>* received_weights) {
        if (static_cast<int>(new_mask.size()) != params_.num_experts) {
            LOG_ERROR("[MoEFFNStage] Expert mask size " << new_mask.size()
                      << " != num_experts " << params_.num_experts);
            return false;
        }

        // ── Phase 1: Release departed experts ────────────────────────────
        // Experts that were active (prepared) but are no longer in new_mask.
        // Release their packed weights to reclaim memory.
        {
            int departed_count = 0;
            for (int e = 0; e < params_.num_experts; ++e) {
                if (!new_mask[e] && params_.prepared_gate_gemm[e]) {
                    // Release packed weights from GEMM engines
                    params_.prepared_gate_gemm[e]->releaseWeights();
                    params_.prepared_up_gemm[e]->releaseWeights();
                    params_.prepared_down_gemm[e]->releaseWeights();

                    // Evict from KernelFactory cache so stale pointers are cleaned up
                    if (params_.expert_gate_views[e])
                        KernelFactory::clearPreparedGemmWeightsFor(params_.expert_gate_views[e].get());
                    if (params_.expert_up_views[e])
                        KernelFactory::clearPreparedGemmWeightsFor(params_.expert_up_views[e].get());
                    if (params_.expert_down_views[e])
                        KernelFactory::clearPreparedGemmWeightsFor(params_.expert_down_views[e].get());

                    // Null out the engine pointers
                    params_.prepared_gate_gemm[e] = nullptr;
                    params_.prepared_up_gemm[e] = nullptr;
                    params_.prepared_down_gemm[e] = nullptr;

                    ++departed_count;
                }
            }
            if (departed_count > 0) {
                LOG_DEBUG("[MoEFFNStage] Released " << departed_count
                         << " departed expert weight sets");
            }
        }

        // ── Phase 2: Prepare newly-acquired experts ──────────────────────
        // Find newly-acquired experts (true in new_mask, not previously prepared)
        std::vector<int> new_experts;
        for (int e = 0; e < params_.num_experts; ++e) {
            if (new_mask[e] && !params_.prepared_gate_gemm[e]) {
                new_experts.push_back(e);
            }
        }

        // Lazily prepare GEMM engines for newly-acquired experts
        if (!new_experts.empty()) {
            auto t_start = std::chrono::high_resolution_clock::now();
            int transferred_count = 0;

            LOG_DEBUG("[MoEFFNStage] Preparing " << new_experts.size()
                     << " new expert GEMM engines after rebalance");

            std::atomic<bool> error_flag{false};
            const int count = static_cast<int>(new_experts.size());

            // Phase 2a: Register transferred weights (single-threaded, fast deserialization)
            if (received_weights) {
                for (int idx = 0; idx < count; ++idx) {
                    const int e = new_experts[idx];
                    auto it = received_weights->find(e);
                    if (it != received_weights->end() && !it->second.empty()) {
                        if (registerTransferredExpert(e, it->second)) {
                            ++transferred_count;
                        }
                    }
                }
                if (transferred_count > 0) {
                    LOG_DEBUG("[MoEFFNStage] Registered " << transferred_count
                             << " experts from transferred weights (skipped VNNI packing)");
                }
            }

            // Phase 2b: Prepare engines (cache hit for transferred, full pack for rest)
            #pragma omp parallel for schedule(static)
            for (int idx = 0; idx < count; ++idx) {
                if (error_flag.load(std::memory_order_relaxed)) continue;
                const int e = new_experts[idx];

                if (!params_.expert_gate_views[e] || !params_.expert_up_views[e] ||
                    !params_.expert_down_views[e]) {
                    LOG_ERROR("[MoEFFNStage] Null view for new expert " << e);
                    error_flag.store(true, std::memory_order_relaxed);
                    continue;
                }

                auto *gp = KernelFactory::getOrCreatePreparedGemmWeights(
                    params_.expert_gate_views[e].get(), params_.device_id);
                auto *up = KernelFactory::getOrCreatePreparedGemmWeights(
                    params_.expert_up_views[e].get(), params_.device_id);
                auto *dp = KernelFactory::getOrCreatePreparedGemmWeights(
                    params_.expert_down_views[e].get(), params_.device_id);

                if (!gp || !up || !dp) {
                    LOG_ERROR("[MoEFFNStage] Failed GEMM weights for new expert " << e);
                    error_flag.store(true, std::memory_order_relaxed);
                    continue;
                }

                params_.prepared_gate_gemm[e] = KernelFactory::getOrCreateGemmEngine(gp);
                params_.prepared_up_gemm[e] = KernelFactory::getOrCreateGemmEngine(up);
                params_.prepared_down_gemm[e] = KernelFactory::getOrCreateGemmEngine(dp);

                if (!params_.prepared_gate_gemm[e] || !params_.prepared_up_gemm[e] ||
                    !params_.prepared_down_gemm[e]) {
                    LOG_ERROR("[MoEFFNStage] Failed GEMM engine for new expert " << e);
                    error_flag.store(true, std::memory_order_relaxed);
                    continue;
                }
            }

            if (error_flag.load()) return false;

            auto t_end = std::chrono::high_resolution_clock::now();
            double prep_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            LOG_DEBUG("[MoEFFNStage] VNNI prep for " << new_experts.size()
                      << " experts (" << transferred_count << " transferred): "
                      << std::fixed << std::setprecision(1) << prep_ms << " ms");

            // NUMA audit: verify rebalanced expert weights landed on the correct NUMA node.
            auditExpertNUMA(new_experts, params_.prepared_gate_gemm,
                            (transferred_count > 0 ? "rebalance_transferred" : "rebalance_repacked"),
                            params_.layer_idx);
        }

        params_.expert_mask = new_mask;

        // Invalidate cached engine vectors so ensureGemmEnginesCached()
        // re-copies from the updated params_ on next execute().
        cached_gate_gemm_.clear();
        cached_up_gemm_.clear();
        cached_down_gemm_.clear();

        return true;
    }

    // ── Phased rebalance API ─────────────────────────────────────────────

    std::vector<const TensorBase*> MoEFFNStage::releaseDepartedExperts(
        const std::vector<bool>& new_mask) {
        std::vector<const TensorBase*> evict_tensors;

        for (int e = 0; e < params_.num_experts; ++e) {
            if (!new_mask[e] && params_.prepared_gate_gemm[e]) {
                // Release packed weights from GEMM engines
                params_.prepared_gate_gemm[e]->releaseWeights();
                params_.prepared_up_gemm[e]->releaseWeights();
                params_.prepared_down_gemm[e]->releaseWeights();

                // Collect tensor views for batch cache eviction by the caller
                if (params_.expert_gate_views[e])
                    evict_tensors.push_back(params_.expert_gate_views[e].get());
                if (params_.expert_up_views[e])
                    evict_tensors.push_back(params_.expert_up_views[e].get());
                if (params_.expert_down_views[e])
                    evict_tensors.push_back(params_.expert_down_views[e].get());

                // Null out the engine pointers
                params_.prepared_gate_gemm[e] = nullptr;
                params_.prepared_up_gemm[e] = nullptr;
                params_.prepared_down_gemm[e] = nullptr;
            }
        }
        return evict_tensors;
    }

    bool MoEFFNStage::registerAndPrepareNewExperts(
        const std::vector<bool>& new_mask,
        const std::unordered_map<int, ExpertWeightBlobs>* received_weights) {

        // Find newly-acquired experts (true in new_mask, not previously prepared)
        std::vector<int> new_experts;
        for (int e = 0; e < params_.num_experts; ++e) {
            if (new_mask[e] && !params_.prepared_gate_gemm[e])
                new_experts.push_back(e);
        }
        if (new_experts.empty()) return true;

        auto t_start = std::chrono::high_resolution_clock::now();
        int transferred_count = 0;
        std::atomic<bool> error_flag{false};
        const int count = static_cast<int>(new_experts.size());

        // Register transferred weights (single-threaded, fast deserialization)
        if (received_weights) {
            for (int idx = 0; idx < count; ++idx) {
                const int e = new_experts[idx];
                auto it = received_weights->find(e);
                if (it != received_weights->end() && !it->second.empty()) {
                    if (registerTransferredExpert(e, it->second))
                        ++transferred_count;
                }
            }
        }

        // Prepare engines (cache hit for transferred, full pack for rest)
        for (int idx = 0; idx < count; ++idx) {
            if (error_flag.load(std::memory_order_relaxed)) break;
            const int e = new_experts[idx];

            if (!params_.expert_gate_views[e] || !params_.expert_up_views[e] ||
                !params_.expert_down_views[e]) {
                LOG_ERROR("[MoEFFNStage] Null view for new expert " << e);
                error_flag.store(true, std::memory_order_relaxed);
                continue;
            }

            auto *gp = KernelFactory::getOrCreatePreparedGemmWeights(
                params_.expert_gate_views[e].get(), params_.device_id);
            auto *up = KernelFactory::getOrCreatePreparedGemmWeights(
                params_.expert_up_views[e].get(), params_.device_id);
            auto *dp = KernelFactory::getOrCreatePreparedGemmWeights(
                params_.expert_down_views[e].get(), params_.device_id);

            if (!gp || !up || !dp) {
                LOG_ERROR("[MoEFFNStage] Failed GEMM weights for new expert " << e);
                error_flag.store(true, std::memory_order_relaxed);
                continue;
            }

            params_.prepared_gate_gemm[e] = KernelFactory::getOrCreateGemmEngine(gp);
            params_.prepared_up_gemm[e] = KernelFactory::getOrCreateGemmEngine(up);
            params_.prepared_down_gemm[e] = KernelFactory::getOrCreateGemmEngine(dp);

            if (!params_.prepared_gate_gemm[e] || !params_.prepared_up_gemm[e] ||
                !params_.prepared_down_gemm[e]) {
                LOG_ERROR("[MoEFFNStage] Failed GEMM engine for new expert " << e);
                error_flag.store(true, std::memory_order_relaxed);
                continue;
            }
        }

        if (error_flag.load()) return false;

        auto t_end = std::chrono::high_resolution_clock::now();
        double prep_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        LOG_DEBUG("[MoEFFNStage] Engine prep for " << new_experts.size()
                  << " experts (" << transferred_count << " transferred): "
                  << std::fixed << std::setprecision(1) << prep_ms << " ms");

        auditExpertNUMA(new_experts, params_.prepared_gate_gemm,
                        (transferred_count > 0 ? "rebalance_transferred" : "rebalance_repacked"),
                        params_.layer_idx);
        return true;
    }

    void MoEFFNStage::applyExpertMask(const std::vector<bool>& new_mask) {
        params_.expert_mask = new_mask;
        cached_gate_gemm_.clear();
        cached_up_gemm_.clear();
        cached_down_gemm_.clear();
    }

    IMoEKernel *MoEFFNStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
            moe_kernel_ = KernelFactory::getOrCreateMoEKernel(params_.device_id);
        return moe_kernel_;
    }

    void MoEFFNStage::stashRoutingResults(
        const std::vector<int> &expert_indices,
        const std::vector<float> &expert_weights,
        int seq_len, int top_k) const
    {
        const size_t n = static_cast<size_t>(seq_len) * top_k;
        routing_indices_f32_.resize(n);
        routing_weights_.resize(n);
        for (size_t i = 0; i < n; ++i)
            routing_indices_f32_[i] = static_cast<float>(expert_indices[i]);
        std::copy(expert_weights.begin(), expert_weights.end(), routing_weights_.begin());

        // Invalidate cached dump info so snapshot callback sees the routing data
        invalidateDumpInfoCache();
    }

    bool MoEFFNStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoEFFNStage] Null device context");
            return false;
        }

        if (!params_.input || !params_.gate_weights || !params_.output)
        {
            LOG_ERROR("[MoEFFNStage] Null input/gate/output tensor");
            return false;
        }

        if (!raw_weights_released_ && (!params_.gate_exps || !params_.up_exps || !params_.down_exps))
        {
            LOG_ERROR("[MoEFFNStage] Null expert weight tensors");
            return false;
        }

        // Fast path for decode (seq_len=1): eliminates gather/scatter overhead,
        // vector allocations, and expert_token_lists construction.
        if (params_.seq_len == 1)
        {
            return executeSingleToken(ctx);
        }

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int intermediate = params_.expert_intermediate;

        if (params_.expert_gate_views.empty())
        {
            LOG_ERROR("[MoEFFNStage] Requires pre-extracted expert views. "
                      "Call extractExpertViews() at graph build time.");
            return false;
        }

        const float *hidden = params_.input->data();
        const float *gate_w = params_.gate_weights->data();
        float *output = params_.output->mutable_data();

        // Get device-appropriate MoE kernel for routing/gather/scatter
        IMoEKernel *kernel = ensureMoEKernel();

        // Step 1: Routing — softmax top-k via device kernel
        MoERoutingResult routing;
        if (!kernel->route(hidden, gate_w, seq_len, d_model,
                           num_experts, top_k, params_.norm_topk_prob, routing))
        {
            LOG_ERROR("[MoEFFNStage] Routing failed");
            return false;
        }

        // Stash routing data for snapshot capture
        router_logits_ = std::move(routing.router_logits);
        stashRoutingResults(routing.expert_indices, routing.expert_weights, seq_len, top_k);

        // Step 2: Zero output
        std::memset(output, 0, static_cast<size_t>(seq_len) * d_model * sizeof(float));

        // Step 3: Group tokens by expert for batched GEMM execution.
        // With EP, we only process experts in our local range, but still
        // build the full routing map so scratch sizing is correct.
        const int local_start = params_.local_expert_start;
        const int local_count = (params_.local_expert_count < 0)
                                    ? num_experts
                                    : params_.local_expert_count;
        const int local_end = local_start + local_count;

        std::vector<std::vector<std::pair<int, float>>> expert_token_lists(num_experts);

        // During prefill, replicated experts must only run on their owner
        // socket.  If both sockets process the same expert, the allreduce
        // will double-count that expert's contribution (correctness bug)
        // and waste ~12% compute (performance bug).
        //
        // Use pre-built prefill mask when available (zero per-expert overhead).
        // Falls back to multi-branch check if mask isn't built yet.
        const bool has_prefill_mask = !params_.replica_set.prefill_mask.empty();
        const std::vector<bool>& prefill_mask_ref = params_.replica_set.prefill_mask;
        const bool has_replicas = params_.replica_set.num_replicated > 0;

        for (int t = 0; t < seq_len; ++t)
        {
            for (int k = 0; k < top_k; ++k)
            {
                int expert_id = routing.expert_indices[t * top_k + k];
                float weight = routing.expert_weights[t * top_k + k];
                // With EP or dynamic mask, only accumulate tokens for local experts
                bool is_local;
                if (has_prefill_mask)
                {
                    // Pre-built mask: single lookup, no branches
                    is_local = prefill_mask_ref[expert_id];
                }
                else if (!params_.expert_mask.empty())
                {
                    is_local = params_.expert_mask[expert_id];
                    // Replicated experts: only owner socket processes during prefill
                    if (is_local && has_replicas &&
                        params_.replica_set.is_replicated[expert_id] &&
                        params_.replica_set.owner_socket[expert_id] != params_.my_socket_id)
                    {
                        is_local = false;
                    }
                }
                else
                    is_local = (expert_id >= local_start && expert_id < local_end);
                if (is_local)
                    expert_token_lists[expert_id].emplace_back(t, weight);
            }
        }

        // Ensure GEMM engines are cached (lazy init on first call)
        ensureGemmEnginesCached();

        // Ensure scratch buffers have enough capacity for largest expert batch
        int max_batch = 0;
        for (const auto &tl : expert_token_lists)
            max_batch = std::max(max_batch, static_cast<int>(tl.size()));

        if (max_batch > 0 && max_batch > scratch_capacity_)
        {
            scratch_batch_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(max_batch), static_cast<size_t>(d_model)});
            scratch_gate_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(max_batch), static_cast<size_t>(intermediate)});
            scratch_up_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(max_batch), static_cast<size_t>(intermediate)});
            scratch_out_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(max_batch), static_cast<size_t>(d_model)});
            scratch_capacity_ = max_batch;
        }

        // Step 4: Execute each active expert (reusing cached engines + scratch)
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &token_list = expert_token_lists[expert_id];
            if (token_list.empty())
                continue;

            const int num_tokens = static_cast<int>(token_list.size());

            // Build token indices and weights arrays for kernel calls
            std::vector<int> token_indices(num_tokens);
            std::vector<float> token_weights(num_tokens);
            for (int i = 0; i < num_tokens; ++i)
            {
                token_indices[i] = token_list[i].first;
                token_weights[i] = token_list[i].second;
            }

            // Gather tokens into reusable scratch batch via kernel
            kernel->gatherTokenBatch(
                hidden, scratch_batch_->mutable_data(),
                token_indices.data(), num_tokens, d_model);

            // Use cached GEMM engines (device-agnostic via ITensorGemm)
            ITensorGemm *gate_gemm = cached_gate_gemm_[expert_id];
            ITensorGemm *up_gemm = cached_up_gemm_[expert_id];
            ITensorGemm *down_gemm = cached_down_gemm_[expert_id];

            // Gate+Up projections via fused multi-projection (quantizes input once)
            std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                {gate_gemm, scratch_gate_.get(), intermediate, nullptr, "gate"},
                {up_gemm, scratch_up_.get(), intermediate, nullptr, "up"}};
            gate_gemm->multiply_fused_tensor(
                scratch_batch_.get(), projections,
                num_tokens, d_model);

            // SwiGLU+Down via fused kernel with fallback through MoE kernel
            fusedSwigluDown(
                scratch_gate_.get(), scratch_up_.get(), scratch_out_.get(),
                down_gemm, kernel, num_tokens, d_model, intermediate);

            // Scatter weighted results back via kernel
            kernel->scatterAddWeighted(
                output, scratch_out_->data(),
                token_indices.data(), token_weights.data(),
                num_tokens, d_model);
        }

        LOG_TRACE("[MoEFFNStage] Processed " << seq_len << " tokens via GEMM kernels, "
                                             << top_k << " experts per token");
        return true;
    }

    // =========================================================================
    // MoEFFNStage::executeSingleToken — Optimized decode path (seq_len=1)
    //
    // Eliminates per-expert overhead:
    // - No gather (input IS the single token)
    // - No scatter (direct weighted accumulation into output)
    // - No vector allocations (stack arrays for top_k ≤ 16)
    // - No expert_token_lists grouping
    // - Reuses a single pair of scratch buffers across all experts
    // =========================================================================

    bool MoEFFNStage::executeSingleToken(IDeviceContext *ctx)
    {
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int intermediate = params_.expert_intermediate;

        if (params_.expert_gate_views.empty())
        {
            LOG_ERROR("[MoEFFNStage] Requires pre-extracted expert views.");
            return false;
        }

        const float *hidden = params_.input->data();
        const float *gate_w = params_.gate_weights->data();
        float *output = params_.output->mutable_data();

        // Routing — reuse pre-allocated cached_routing_ to avoid heap allocs
        IMoEKernel *kernel = ensureMoEKernel();
        if (!kernel->route(hidden, gate_w, /*seq_len=*/1, d_model,
                           num_experts, top_k, params_.norm_topk_prob, cached_routing_))
        {
            LOG_ERROR("[MoEFFNStage] Routing failed (single-token)");
            return false;
        }

#ifdef ENABLE_PIPELINE_SNAPSHOTS
        router_logits_ = std::move(cached_routing_.router_logits);
        stashRoutingResults(cached_routing_.expert_indices, cached_routing_.expert_weights, 1, top_k);
#endif

        // Record routing result in decode histogram (if tracking enabled)
        if (params_.decode_histogram && params_.layer_idx >= 0)
        {
            params_.decode_histogram->record(
                params_.layer_idx,
                cached_routing_.expert_indices.data(),
                cached_routing_.expert_weights.data(),
                top_k);
        }

        // Zero output
        std::memset(output, 0, static_cast<size_t>(d_model) * sizeof(float));

        // EP range
        const int local_start = params_.local_expert_start;
        const int local_count = (params_.local_expert_count < 0)
                                    ? num_experts
                                    : params_.local_expert_count;
        const int local_end = local_start + local_count;

        // Ensure GEMM engines are cached
        ensureGemmEnginesCached();

        // Ensure batch scratch buffers for gate+up (one per top-k expert).
        // All experts' gate+up are fused into a single OMP region, so we need
        // all outputs to exist simultaneously.
        if (static_cast<int>(scratch_gate_batch_.size()) < top_k)
        {
            scratch_gate_batch_.resize(top_k);
            scratch_up_batch_.resize(top_k);
            for (int i = 0; i < top_k; ++i)
            {
                scratch_gate_batch_[i] = std::make_shared<FP32Tensor>(
                    std::vector<size_t>{1, static_cast<size_t>(intermediate)});
                scratch_up_batch_[i] = std::make_shared<FP32Tensor>(
                    std::vector<size_t>{1, static_cast<size_t>(intermediate)});
            }
        }
        // Scratch for down projection output (reused per expert)
        if (!scratch_out_)
        {
            scratch_out_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{1, static_cast<size_t>(d_model)});
        }

        // Use input tensor directly (no gather needed for 1 token)
        const TensorBase *input_tensor = params_.input;

        // ---------------------------------------------------------------
        // Phase 1: Batch all experts' gate+up into ONE fused GEMV call.
        // This quantizes the input to Q8_1 once (not 8×) and uses a single
        // OMP parallel region (not 8×), saving ~7×(2µs quant + 6µs OMP)
        // = ~56µs per layer × 36 MoE layers = ~2ms per decode token.
        // ---------------------------------------------------------------
        struct ActiveExpert { int expert_id; float weight; int batch_idx; };
        ActiveExpert active_experts[16]; // stack-allocated, max top_k
        int num_active = 0;

        batch_projections_.clear();
        batch_projections_.reserve(top_k * 2);

        // Per-token dynamic dispatch for replicated experts.
        // When replicas are active, use ExpertReplicaSet::assignForToken()
        // to deterministically decide which socket computes each expert.
        bool compute_here[16]; // stack-allocated, max top_k

        if (params_.replica_set.num_replicated > 0)
        {
            params_.replica_set.assignForToken(
                cached_routing_.expert_indices.data(),
                cached_routing_.expert_weights.data(),
                top_k,
                params_.my_socket_id,
                params_.expert_mask,
                compute_here);
        }
        else
        {
            // No replicas — use simple mask/range check
            for (int k = 0; k < top_k; ++k)
            {
                const int expert_id = cached_routing_.expert_indices[k];
                if (!params_.expert_mask.empty())
                    compute_here[k] = params_.expert_mask[expert_id];
                else
                    compute_here[k] = (expert_id >= local_start && expert_id < local_end);
            }
        }

        for (int k = 0; k < top_k; ++k)
        {
            if (!compute_here[k]) continue;

            const int expert_id = cached_routing_.expert_indices[k];

            ITensorGemm *gate_gemm = cached_gate_gemm_[expert_id];
            ITensorGemm *up_gemm = cached_up_gemm_[expert_id];

            if (!gate_gemm || !up_gemm)
            {
                LOG_ERROR("[MoEFFNStage] FATAL: Null gate/up GEMM engine for expert "
                    << expert_id << " (layer " << params_.layer_idx
                    << ", mask=" << (params_.expert_mask.empty() ? -1 : (int)params_.expert_mask[expert_id])
                    << ", replicated=" << params_.replica_set.is_replicated[expert_id]
                    << ", prepared_gate=" << (bool)params_.prepared_gate_gemm[expert_id] << ")");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            batch_projections_.push_back(
                {gate_gemm, scratch_gate_batch_[num_active].get(), intermediate, nullptr, "gate"});
            batch_projections_.push_back(
                {up_gemm, scratch_up_batch_[num_active].get(), intermediate, nullptr, "up"});

            active_experts[num_active] = {expert_id, cached_routing_.expert_weights[k], num_active};
            num_active++;
        }

        // Single fused call: quantize once + single OMP region for all gate+up
        if (num_active > 0)
        {
            batch_projections_[0].kernel->multiply_fused_tensor(
                input_tensor, batch_projections_, /*m=*/1, d_model);
        }

        // ---------------------------------------------------------------
        // Phase 2: Fused SwiGLU + Down projection + weighted accumulate
        //
        // Strategy: apply SwiGLU for all experts first, then fuse all
        // down projections into a single OMP region. This saves
        // (num_active-1) OMP fork/join cycles (~8µs each) and improves
        // load balance via nowait (128 total chunks vs 4×32).
        // Falls back to sequential if fused path unavailable.
        // ---------------------------------------------------------------

        // Ensure per-expert output buffers for fused approach
        if (static_cast<int>(scratch_down_batch_.size()) < num_active)
        {
            scratch_down_batch_.resize(num_active);
            for (int i = 0; i < num_active; ++i)
            {
                if (!scratch_down_batch_[i])
                    scratch_down_batch_[i] = std::make_shared<FP32Tensor>(
                        std::vector<size_t>{1, static_cast<size_t>(d_model)});
            }
        }

        // Validate down GEMM engines
        for (int i = 0; i < num_active; ++i)
        {
            if (!cached_down_gemm_[active_experts[i].expert_id])
            {
                LOG_ERROR("[MoEFFNStage] FATAL: Null down GEMM engine for expert "
                    << active_experts[i].expert_id << " (layer " << params_.layer_idx << ")");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        // Phase 2a: Apply SwiGLU for all experts (serial, ~0.1µs each)
        for (int i = 0; i < num_active; ++i)
        {
            const auto &info = active_experts[i];
            const float *gate_fp32 = scratch_gate_batch_[info.batch_idx]->data();
            const float *up_fp32 = scratch_up_batch_[info.batch_idx]->data();
            swiglu_scratch_batch_.resize(std::max(swiglu_scratch_batch_.size(),
                                                  static_cast<size_t>(num_active)));
            if (static_cast<int>(swiglu_scratch_batch_[i].size()) < intermediate)
                swiglu_scratch_batch_[i].resize(intermediate);

            primitives::compute_swiglu_serial(gate_fp32, up_fp32,
                                              swiglu_scratch_batch_[i].data(), intermediate);
        }

        // Phase 2b: Try fused multi-input down projections
        bool fused_ok = false;
        if (num_active >= 2)
        {
            ITensorGemm::FusedExpertDownDesc down_descs[16];
            for (int i = 0; i < num_active && i < 16; ++i)
            {
                const auto &info = active_experts[i];
                down_descs[i].kernel = cached_down_gemm_[info.expert_id];
                down_descs[i].input = swiglu_scratch_batch_[i].data();
                down_descs[i].output = scratch_down_batch_[i]->mutable_data();
                down_descs[i].n = d_model;
            }
            fused_ok = cached_down_gemm_[active_experts[0].expert_id]
                           ->multiply_fused_expert_down(down_descs, num_active, 1, intermediate);
        }

        if (fused_ok)
        {
            // Phase 2c: Weighted accumulate all outputs
            for (int i = 0; i < num_active; ++i)
            {
                const auto &info = active_experts[i];
                primitives::vec_axpy(output, scratch_down_batch_[i]->data(),
                                     info.weight, d_model);
            }
        }
        else
        {
            // Fallback: sequential per-expert SwiGLU + Down + accumulate
            float *scratch_out_ptr = scratch_out_->mutable_data();
            for (int i = 0; i < num_active; ++i)
            {
                const auto &info = active_experts[i];
                ITensorGemm *down_gemm = cached_down_gemm_[info.expert_id];

                fusedSwigluDown(
                    scratch_gate_batch_[info.batch_idx].get(),
                    scratch_up_batch_[info.batch_idx].get(),
                    scratch_out_.get(),
                    down_gemm, kernel, /*m=*/1, d_model, intermediate);

                primitives::vec_axpy(output, scratch_out_ptr, info.weight, d_model);
            }
        }

        LOG_TRACE("[MoEFFNStage] Single-token decode (batched gate+up): " << num_active << " experts");
        return true;
    }

    void MoEFFNStage::ensureGemmEnginesCached() const
    {
        if (!cached_gate_gemm_.empty())
            return;

        const int num_experts = params_.num_experts;

        // Use pre-resolved engines from graph build time if available
        if (!params_.prepared_gate_gemm.empty())
        {
            cached_gate_gemm_ = params_.prepared_gate_gemm;
            cached_up_gemm_ = params_.prepared_up_gemm;
            cached_down_gemm_ = params_.prepared_down_gemm;
            return;
        }

        // Fallback: resolve on first call (triggers VNNI repacking — slow)
        LOG_WARN("[MoEFFNStage] GEMM engines not pre-resolved; "
                 "call prepareExpertGemmEngines() at graph build time for better perf");

        const int local_start = params_.local_expert_start;
        const int local_count = (params_.local_expert_count < 0)
                                    ? num_experts
                                    : params_.local_expert_count;
        const int local_end = local_start + local_count;

        cached_gate_gemm_.resize(num_experts, nullptr);
        cached_up_gemm_.resize(num_experts, nullptr);
        cached_down_gemm_.resize(num_experts, nullptr);

        for (int e = local_start; e < local_end; ++e)
        {
            if (!params_.expert_gate_views[e])
                continue;
            auto *gp = KernelFactory::getOrCreatePreparedGemmWeights(
                params_.expert_gate_views[e].get(), params_.device_id);
            auto *up = KernelFactory::getOrCreatePreparedGemmWeights(
                params_.expert_up_views[e].get(), params_.device_id);
            auto *dp = KernelFactory::getOrCreatePreparedGemmWeights(
                params_.expert_down_views[e].get(), params_.device_id);
            cached_gate_gemm_[e] = KernelFactory::getOrCreateGemmEngine(gp);
            cached_up_gemm_[e] = KernelFactory::getOrCreateGemmEngine(up);
            cached_down_gemm_[e] = KernelFactory::getOrCreateGemmEngine(dp);
        }
    }

    // =========================================================================
    // MoEFFNStage::extractExpertViews — Create 2D views from 3D packed tensors
    // =========================================================================

    bool MoEFFNStage::extractExpertViews(Params &params)
    {
        if (!params.gate_exps || !params.up_exps || !params.down_exps)
        {
            LOG_ERROR("[MoEFFNStage] Cannot extract views: null expert tensors");
            return false;
        }

        const int num_experts = params.num_experts;
        if (num_experts <= 0)
        {
            LOG_ERROR("[MoEFFNStage] Invalid num_experts: " << num_experts);
            return false;
        }

        // EP range: only extract views for local experts
        // Dynamic rebalancing: when expert_mask is set, expert data is replicated.
        // Extract views for ALL experts so GEMM engines can be prepared for all.
        // The mask only controls which experts are executed, not which views exist.
        const bool extract_all = !params.expert_mask.empty();
        const int local_start = params.local_expert_start;
        const int local_count = (params.local_expert_count < 0)
                                    ? num_experts
                                    : params.local_expert_count;
        const int local_end = local_start + local_count;

        params.expert_gate_views.resize(num_experts);
        params.expert_up_views.resize(num_experts);
        params.expert_down_views.resize(num_experts);

        // Extract 2D views for each expert.
        // GGUF 3D: shape = [ne[0], ne[1], ne[2]] = [cols, rows, num_experts_in_tensor]
        // Each expert's 2D slice is [rows, cols] at element offset within the tensor.
        //
        // With expert-parallel weight sharding, the 3D tensor may contain only
        // local experts (shape[2] == local_count) instead of all experts.
        // In that case, global expert index `e` maps to local tensor index
        // `e - local_start`. When the tensor has all experts (shape[2] == num_experts),
        // the offset uses the global index directly.
        auto extract_views = [local_start, local_end, extract_all](
                                 TensorBase *tensor_3d, int n_experts,
                                 std::vector<std::shared_ptr<TensorBase>> &views) -> bool
        {
            const auto &shape = tensor_3d->shape();
            if (shape.size() != 3)
            {
                LOG_ERROR("[MoE] Expert tensor must be 3D, got " << shape.size() << "D");
                return false;
            }

            // GGUF 3D: shape[0]=ne[0]=cols (fastest), shape[1]=ne[1]=rows, shape[2]=experts in tensor
            size_t cols = shape[0];
            size_t rows = shape[1];
            size_t tensor_expert_count = shape[2];
            size_t elements_per_expert = rows * cols;

            // Determine if the tensor was pre-sliced (expert-parallel sharding)
            // or contains all experts (replicated mode).
            const bool is_presliced = (static_cast<int>(tensor_expert_count) != n_experts);

            for (int e = 0; e < n_experts; ++e)
            {
                // Skip non-local experts under EP (unless extracting all for dynamic rebalancing)
                if (!extract_all && (e < local_start || e >= local_end))
                    continue;

                // For pre-sliced tensors: local expert `e` is at tensor index `e - local_start`
                // For full tensors: expert `e` is at tensor index `e`
                size_t tensor_idx = is_presliced ? static_cast<size_t>(e - local_start)
                                                 : static_cast<size_t>(e);
                size_t element_offset = tensor_idx * elements_per_expert;

                std::vector<size_t> view_shape = {rows, cols};
                auto view = tensor_3d->create_view(view_shape, element_offset);
                if (!view)
                {
                    LOG_ERROR("[MoE] Failed to create view for expert " << e
                              << " (tensor_idx=" << tensor_idx << ")");
                    return false;
                }
                views[e] = std::move(view);
            }
            return true;
        };

        if (!extract_views(params.gate_exps, num_experts, params.expert_gate_views))
            return false;
        if (!extract_views(params.up_exps, num_experts, params.expert_up_views))
            return false;
        if (!extract_views(params.down_exps, num_experts, params.expert_down_views))
            return false;

        LOG_DEBUG("[MoEFFNStage] Extracted " << (extract_all ? num_experts : local_count) << "/" << num_experts
                                             << " expert 2D views (EP range [" << local_start
                                             << ", " << local_end << ")"
                                            << (extract_all ? " extract_all=true" : "") << ")");
        return true;
    }

    bool MoEFFNStage::prepareExpertGemmEngines(Params &params)
    {
        const int num_experts = params.num_experts;
        if (params.expert_gate_views.empty() ||
            static_cast<int>(params.expert_gate_views.size()) != num_experts)
        {
            LOG_ERROR("[MoEFFNStage] prepareExpertGemmEngines: call extractExpertViews() first");
            return false;
        }

        // EP range
        // Dynamic rebalancing: when expert_mask is set, prepare ONLY mask-active
        // experts (not all). Views exist for all experts, but GEMM engines are
        // expensive (VNNI repacking). Newly-acquired experts get engines lazily
        // via updateExpertMaskAndPrepareEngines() after rebalancing.
        const bool use_mask = !params.expert_mask.empty();
        const int local_start = params.local_expert_start;
        const int local_count = (params.local_expert_count < 0)
                                    ? num_experts
                                    : params.local_expert_count;
        const int local_end = local_start + local_count;

        // Build list of experts to prepare
        std::vector<int> experts_to_prep;
        if (use_mask)
        {
            for (int e = 0; e < num_experts; ++e)
                if (params.expert_mask[e])
                    experts_to_prep.push_back(e);
        }
        else
        {
            for (int e = local_start; e < local_end; ++e)
                experts_to_prep.push_back(e);
        }
        const int prep_count = static_cast<int>(experts_to_prep.size());

        params.prepared_gate_gemm.resize(num_experts, nullptr);
        params.prepared_up_gemm.resize(num_experts, nullptr);
        params.prepared_down_gemm.resize(num_experts, nullptr);

        LOG_DEBUG("[MoEFFNStage] Preparing GEMM engines for " << prep_count << "/" << num_experts
                  << " experts (3 weights each = " << (prep_count * 3) << " total"
                  << (use_mask ? " [dynamic rebalance: mask-active only]" : "") << ")...");

#ifdef HAVE_CUDA
        if (params.device_id.is_cuda())
        {
            return prepareExpertGemmEnginesCUDA(params);
        }
#endif
#ifdef HAVE_ROCM
        if (params.device_id.is_rocm())
        {
            return prepareExpertGemmEnginesROCm(params);
        }
#endif

        // CPU path: parallelize expert GEMM engine preparation.
        // Each expert has unique tensors (unique raw_data() keys), so no cache
        // key collisions.  The heavy VNNI interleave runs lock-free; only the
        // KernelFactory registry insert takes a brief mutex.
        std::atomic<bool> error_flag{false};

        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < prep_count; ++idx)
        {
            if (error_flag.load(std::memory_order_relaxed)) continue;
            const int e = experts_to_prep[idx];

            if (!params.expert_gate_views[e] || !params.expert_up_views[e] || !params.expert_down_views[e])
            {
                LOG_ERROR("[MoEFFNStage] Null expert view for local expert " << e);
                error_flag.store(true, std::memory_order_relaxed);
                continue;
            }

            auto *gp = KernelFactory::getOrCreatePreparedGemmWeights(
                params.expert_gate_views[e].get(), params.device_id);
            auto *up = KernelFactory::getOrCreatePreparedGemmWeights(
                params.expert_up_views[e].get(), params.device_id);
            auto *dp = KernelFactory::getOrCreatePreparedGemmWeights(
                params.expert_down_views[e].get(), params.device_id);

            if (!gp || !up || !dp)
            {
                LOG_ERROR("[MoEFFNStage] Failed to prepare GEMM weights for expert " << e);
                error_flag.store(true, std::memory_order_relaxed);
                continue;
            }

            params.prepared_gate_gemm[e] = KernelFactory::getOrCreateGemmEngine(gp);
            params.prepared_up_gemm[e] = KernelFactory::getOrCreateGemmEngine(up);
            params.prepared_down_gemm[e] = KernelFactory::getOrCreateGemmEngine(dp);

            if (!params.prepared_gate_gemm[e] || !params.prepared_up_gemm[e] || !params.prepared_down_gemm[e])
            {
                LOG_ERROR("[MoEFFNStage] Failed to create GEMM engine for expert " << e);
                error_flag.store(true, std::memory_order_relaxed);
                continue;
            }
        }

        if (error_flag.load())
        {
            return false;
        }

        LOG_DEBUG("[MoEFFNStage] All " << (prep_count * 3) << " expert GEMM engines prepared (CPU/KernelFactory path)");

        // NUMA audit: verify packed weights landed on the correct NUMA node.
        auditExpertNUMA(experts_to_prep, params.prepared_gate_gemm,
                        "initial_pack", params.layer_idx);

        // Release mmap pages backing the raw expert weight data.
        // The VNNI interleaved engines now own their own copy — the original
        // mmap data is never accessed again. Releasing per-layer reduces peak RSS
        // by ~500 MB/layer instead of waiting for a bulk release at the end.
        // NOTE: Only safe for mmap-backed tensors. Expert-parallel sliced tensors
        // are heap-allocated copies — MADV_DONTNEED on heap memory corrupts malloc metadata.
        {
            size_t released = 0;
            if (params.gate_exps && params.gate_exps->is_mmap_data())
                released += MmapRegion::adviseDontneedRange(params.gate_exps->raw_data(), params.gate_exps->size_bytes());
            if (params.up_exps && params.up_exps->is_mmap_data())
                released += MmapRegion::adviseDontneedRange(params.up_exps->raw_data(), params.up_exps->size_bytes());
            if (params.down_exps && params.down_exps->is_mmap_data())
                released += MmapRegion::adviseDontneedRange(params.down_exps->raw_data(), params.down_exps->size_bytes());
            if (released > 0)
                LOG_DEBUG("[MoEFFNStage] Advised " << (released >> 20) << " MB of mmap pages DONTNEED after engine packing");
        }

        return true;
    }

    size_t MoEFFNStage::estimatedFlops() const
    {
        // Per token: top_k experts × (gate + up + down projections)
        // gate/up: d_model × intermediate
        // down: intermediate × d_model
        size_t per_expert = static_cast<size_t>(6) * params_.d_model * params_.expert_intermediate;
        return static_cast<size_t>(params_.seq_len) * params_.top_k * per_expert;
    }

    bool MoEFFNStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return !params_.expert_gate_views.empty();
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return !params_.expert_gate_views.empty();
#endif
        default:
            return false;
        }
    }

    StageBufferRequirements MoEFFNStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.input)
            reqs.addInput("input", params_.input->shape(), toBufferTensorType(params_.input->native_type()));
        if (params_.output)
            reqs.addOutput("output", params_.output->shape(), toBufferTensorType(params_.output->native_type()));
        return reqs;
    }

    StageDumpInfo MoEFFNStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.input)
            info.addInput("input", params_.input, params_.seq_len, params_.d_model);
        if (params_.gate_weights)
            info.addWeight("gate_weights", params_.gate_weights);
        if (params_.gate_exps)
            info.addWeight("gate_exps", params_.gate_exps);
        if (params_.up_exps)
            info.addWeight("up_exps", params_.up_exps);
        if (params_.down_exps)
            info.addWeight("down_exps", params_.down_exps);
        if (params_.output)
            info.addOutput("output", params_.output, params_.seq_len, params_.d_model);

        // Routing data (stashed during execute) — outputs[1..3]
        if (!router_logits_.empty())
            info.addOutput("router_logits", router_logits_.data(),
                           static_cast<size_t>(params_.seq_len),
                           static_cast<size_t>(params_.num_experts));
        if (!routing_indices_f32_.empty())
            info.addOutput("routing_indices", routing_indices_f32_.data(),
                           static_cast<size_t>(params_.seq_len),
                           static_cast<size_t>(params_.top_k));
        if (!routing_weights_.empty())
            info.addOutput("routing_weights", routing_weights_.data(),
                           static_cast<size_t>(params_.seq_len),
                           static_cast<size_t>(params_.top_k));

        info.addScalarInt("num_experts", params_.num_experts);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt("expert_intermediate", params_.expert_intermediate);
        info.addScalarInt("local_expert_start", params_.local_expert_start);
        info.addScalarInt("local_expert_count", params_.local_expert_count);
        return info;
    }

    // =========================================================================
    // SharedExpertFFNStage — Dense SwiGLU on shared expert
    // =========================================================================

    SharedExpertFFNStage::SharedExpertFFNStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    void SharedExpertFFNStage::ensureGemmEnginesCached() const
    {
        if (cached_gate_gemm_)
            return;
        auto *gp = KernelFactory::getOrCreatePreparedGemmWeights(params_.gate_w, params_.device_id);
        auto *up = KernelFactory::getOrCreatePreparedGemmWeights(params_.up_w, params_.device_id);
        auto *dp = KernelFactory::getOrCreatePreparedGemmWeights(params_.down_w, params_.device_id);
        cached_gate_gemm_ = KernelFactory::getOrCreateGemmEngine(gp);
        cached_up_gemm_ = KernelFactory::getOrCreateGemmEngine(up);
        cached_down_gemm_ = KernelFactory::getOrCreateGemmEngine(dp);
    }

    bool SharedExpertFFNStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[SharedExpertFFNStage] Null device context");
            return false;
        }

        if (!params_.input || !params_.gate_w || !params_.up_w || !params_.down_w || !params_.output)
        {
            LOG_ERROR("[SharedExpertFFNStage] Null tensor parameter");
            return false;
        }

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int intermediate = params_.intermediate;

        // Cache GEMM engines on first call
        ensureGemmEnginesCached();

        // Ensure scratch buffers are large enough
        if (seq_len > scratch_seq_len_)
        {
            scratch_gate_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(intermediate)});
            scratch_up_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(intermediate)});
            scratch_seq_len_ = seq_len;
        }

        // Gate+Up projections via fused multi-projection (quantizes input once)
        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {cached_gate_gemm_, scratch_gate_.get(), intermediate, nullptr, "shared_gate"},
            {cached_up_gemm_, scratch_up_.get(), intermediate, nullptr, "shared_up"}};
        cached_gate_gemm_->multiply_fused_tensor(
            params_.input, projections,
            seq_len, d_model);

        // SwiGLU+Down via fused kernel with MoE kernel fallback
        IMoEKernel *kernel = ensureMoEKernel();
        fusedSwigluDown(
            scratch_gate_.get(), scratch_up_.get(), params_.output,
            cached_down_gemm_, kernel, seq_len, d_model, intermediate);

        return true;
    }

    IMoEKernel *SharedExpertFFNStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
            moe_kernel_ = KernelFactory::getOrCreateMoEKernel(params_.device_id);
        return moe_kernel_;
    }

    size_t SharedExpertFFNStage::estimatedFlops() const
    {
        return static_cast<size_t>(6) * params_.seq_len * params_.d_model * params_.intermediate;
    }

    bool SharedExpertFFNStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    StageBufferRequirements SharedExpertFFNStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.input)
            reqs.addInput("input", params_.input->shape(), toBufferTensorType(params_.input->native_type()));
        if (params_.output)
            reqs.addOutput("output", params_.output->shape(), toBufferTensorType(params_.output->native_type()));
        return reqs;
    }

    StageDumpInfo SharedExpertFFNStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.input)
            info.addInput("input", params_.input, params_.seq_len, params_.d_model);
        if (params_.gate_w)
            info.addWeight("gate_w", params_.gate_w);
        if (params_.up_w)
            info.addWeight("up_w", params_.up_w);
        if (params_.down_w)
            info.addWeight("down_w", params_.down_w);
        if (params_.output)
            info.addOutput("output", params_.output, params_.seq_len, params_.d_model);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("intermediate", params_.intermediate);
        return info;
    }

    // =========================================================================
    // SharedExpertGateStage — Sigmoid gating on shared expert output
    // =========================================================================

    SharedExpertGateStage::SharedExpertGateStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool SharedExpertGateStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[SharedExpertGateStage] Null device context");
            return false;
        }

        if (!params_.input || !params_.gate_inp || !params_.shared_output)
        {
            LOG_ERROR("[SharedExpertGateStage] Null tensor parameter");
            return false;
        }

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;

        const float *input = params_.input->data();
        const float *gate_inp = params_.gate_inp->data();
        float *shared = params_.shared_output->mutable_data();

        // Delegate sigmoid gating to device-agnostic MoE kernel
        IMoEKernel *kernel = ensureMoEKernel();
        kernel->sharedExpertGate(input, gate_inp, shared, seq_len, d_model);

        return true;
    }

    IMoEKernel *SharedExpertGateStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
            moe_kernel_ = KernelFactory::getOrCreateMoEKernel(params_.device_id);
        return moe_kernel_;
    }

    size_t SharedExpertGateStage::estimatedFlops() const
    {
        // dot product + sigmoid + elementwise multiply
        return static_cast<size_t>(params_.seq_len) * (2 * params_.d_model + params_.d_model);
    }

    bool SharedExpertGateStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    StageBufferRequirements SharedExpertGateStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.input)
            reqs.addInput("input", params_.input->shape(), toBufferTensorType(params_.input->native_type()));
        if (params_.shared_output)
            reqs.addOutput("shared_output", params_.shared_output->shape(), toBufferTensorType(params_.shared_output->native_type()));
        return reqs;
    }

    StageDumpInfo SharedExpertGateStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.input)
            info.addInput("input", params_.input, params_.seq_len, params_.d_model);
        if (params_.gate_inp)
            info.addWeight("gate_inp", params_.gate_inp);
        if (params_.shared_output)
            info.addOutput("shared_output", params_.shared_output, params_.seq_len, params_.d_model);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        return info;
    }

    // =========================================================================
    // MoE batch packing helpers (CUDA / ROCm)
    // =========================================================================

#ifdef HAVE_CUDA
    bool MoEFFNStage::prepareExpertGemmEnginesCUDA(Params &params)
    {
        using namespace llaminar2::cuda;
        const int num_experts = params.num_experts;
        const int cuda_id = params.device_id.cuda_ordinal();

        // Helper: batch-pack one weight group and create per-expert kernels
        auto batchPackAndCreateKernels = [&](
            const std::vector<std::shared_ptr<TensorBase>> &views,
            std::vector<ITensorGemm *> &out_gemms,
            std::shared_ptr<void> &out_lifetime,
            const char *label) -> bool
        {
            const int rows = static_cast<int>(views[0]->rows());
            const int K = static_cast<int>(views[0]->cols());

            auto batch = packMoEExpertsCUDA(views, num_experts, rows);
            if (!batch)
            {
                LOG_ERROR("[MoEFFNStage::CUDA] Failed to batch-pack " << label);
                return false;
            }

            if (!batch->uploadToDevice(cuda_id))
            {
                LOG_ERROR("[MoEFFNStage::CUDA] Failed to upload " << label << " to device " << cuda_id);
                return false;
            }

            for (int e = 0; e < num_experts; ++e)
            {
                auto expert_ptrs = batch->getExpertDevicePointers(cuda_id, e);
                auto kernel = std::make_shared<llaminar2::cuda::CUDAQuantisedGemmKernel>(
                    rows, K, cuda_id,
                    expert_ptrs.d_vnni, expert_ptrs.d_scales,
                    expert_ptrs.d_mins, expert_ptrs.d_emins,
                    batch->codebook_id, static_cast<uint32_t>(batch->blocks_per_row),
                    batch);
                out_gemms[e] = kernel.get();
                params.moe_owned_kernels.push_back(std::move(kernel));
            }

            batch->freeHostBuffers();
            out_lifetime = std::move(batch);
            return true;
        };

        if (!batchPackAndCreateKernels(params.expert_gate_views, params.prepared_gate_gemm,
                                       params.moe_packed_gate_lifetime, "gate"))
            return false;
        if (!batchPackAndCreateKernels(params.expert_up_views, params.prepared_up_gemm,
                                       params.moe_packed_up_lifetime, "up"))
            return false;
        if (!batchPackAndCreateKernels(params.expert_down_views, params.prepared_down_gemm,
                                       params.moe_packed_down_lifetime, "down"))
            return false;

        LOG_DEBUG("[MoEFFNStage] All " << (num_experts * 3)
                  << " expert GEMM engines prepared (CUDA batch path, 3 GPU allocs)");

        // Release mmap pages for raw expert weights (now uploaded to GPU).
        // Only safe for mmap-backed tensors — EP-sliced tensors are heap-allocated.
        {
            size_t released = 0;
            if (params.gate_exps && params.gate_exps->is_mmap_data())
                released += MmapRegion::adviseDontneedRange(params.gate_exps->raw_data(), params.gate_exps->size_bytes());
            if (params.up_exps && params.up_exps->is_mmap_data())
                released += MmapRegion::adviseDontneedRange(params.up_exps->raw_data(), params.up_exps->size_bytes());
            if (params.down_exps && params.down_exps->is_mmap_data())
                released += MmapRegion::adviseDontneedRange(params.down_exps->raw_data(), params.down_exps->size_bytes());
            if (released > 0)
                LOG_DEBUG("[MoEFFNStage] Advised " << (released >> 20) << " MB of mmap pages DONTNEED after CUDA packing");
        }

        return true;
    }
#endif // HAVE_CUDA

#ifdef HAVE_ROCM
    bool MoEFFNStage::prepareExpertGemmEnginesROCm(Params &params)
    {
        using namespace llaminar2::rocm;
        const int num_experts = params.num_experts;
        const int rocm_id = params.device_id.rocm_ordinal();

        auto batchPackAndCreateKernels = [&](
            const std::vector<std::shared_ptr<TensorBase>> &views,
            std::vector<ITensorGemm *> &out_gemms,
            std::shared_ptr<void> &out_lifetime,
            const char *label) -> bool
        {
            const int rows = static_cast<int>(views[0]->rows());
            const int K = static_cast<int>(views[0]->cols());

            auto batch = packMoEExpertsROCm(views, num_experts, rows);
            if (!batch)
            {
                LOG_ERROR("[MoEFFNStage::ROCm] Failed to batch-pack " << label);
                return false;
            }

            if (!batch->uploadToDevice(rocm_id))
            {
                LOG_ERROR("[MoEFFNStage::ROCm] Failed to upload " << label << " to device " << rocm_id);
                return false;
            }

            for (int e = 0; e < num_experts; ++e)
            {
                auto expert_ptrs = batch->getExpertDevicePointers(rocm_id, e);
                auto kernel = std::make_shared<llaminar2::rocm::ROCmQuantisedGemmKernel>(
                    rows, K, rocm_id,
                    expert_ptrs.d_native_vnni,
                    expert_ptrs.d_native_scales,
                    expert_ptrs.d_native_mins,
                    expert_ptrs.d_native_emins,
                    batch->codebook_id, static_cast<uint32_t>(batch->blocks_per_row),
                    batch);
                out_gemms[e] = kernel.get();
                params.moe_owned_kernels.push_back(std::move(kernel));
            }

            batch->freeHostBuffers();
            out_lifetime = std::move(batch);
            return true;
        };

        if (!batchPackAndCreateKernels(params.expert_gate_views, params.prepared_gate_gemm,
                                       params.moe_packed_gate_lifetime, "gate"))
            return false;
        if (!batchPackAndCreateKernels(params.expert_up_views, params.prepared_up_gemm,
                                       params.moe_packed_up_lifetime, "up"))
            return false;
        if (!batchPackAndCreateKernels(params.expert_down_views, params.prepared_down_gemm,
                                       params.moe_packed_down_lifetime, "down"))
            return false;

        LOG_DEBUG("[MoEFFNStage] All " << (num_experts * 3)
                  << " expert GEMM engines prepared (ROCm batch path, 3 GPU allocs)");

        // Release mmap pages for raw expert weights (now uploaded to GPU).
        // Only safe for mmap-backed tensors — EP-sliced tensors are heap-allocated.
        {
            size_t released = 0;
            if (params.gate_exps && params.gate_exps->is_mmap_data())
                released += MmapRegion::adviseDontneedRange(params.gate_exps->raw_data(), params.gate_exps->size_bytes());
            if (params.up_exps && params.up_exps->is_mmap_data())
                released += MmapRegion::adviseDontneedRange(params.up_exps->raw_data(), params.up_exps->size_bytes());
            if (params.down_exps && params.down_exps->is_mmap_data())
                released += MmapRegion::adviseDontneedRange(params.down_exps->raw_data(), params.down_exps->size_bytes());
            if (released > 0)
                LOG_DEBUG("[MoEFFNStage] Advised " << (released >> 20) << " MB of mmap pages DONTNEED after ROCm packing");
        }

        return true;
    }
#endif // HAVE_ROCM

} // namespace llaminar2
