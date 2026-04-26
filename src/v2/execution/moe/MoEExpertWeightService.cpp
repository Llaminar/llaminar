/**
 * @file MoEExpertWeightService.cpp
 * @brief Implementation of MoE expert weight lifecycle service.
 *
 * Contains all weight preparation, serialization, and rebalancing logic
 * extracted from MoEExpertComputeStage.cpp. See MoEExpertWeightService.h for API docs.
 */

#include "MoEExpertWeightService.h"
#include "../../tensors/Tensors.h"
#include "../../tensors/BlockStructures.h"
#include "../../kernels/KernelFactory.h"
#include "../../kernels/PackedWeightsSerialization.h"
#include "../../kernels/cpu/native_vnni/CPUPackedWeights.h"
#include "../../kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
#include "../../loaders/MmapRegion.h"
#include "../../utils/Assertions.h"
#include "../../utils/Logger.h"
#include "../../utils/OpenMPUtils.h"

#ifdef HAVE_CUDA
#include "../../kernels/cuda/gemm/CUDAWeightPacker.h"
#include "../../kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#endif

#ifdef HAVE_ROCM
#include "../../kernels/rocm/ROCmWeightPacker.h"
#include "../../kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#endif

#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h>
#include <numaif.h>
#include <sched.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <vector>

namespace llaminar2 {

// Alias for fully-qualified KernelFactory access
using KernelFactory = llaminar::v2::kernels::KernelFactory;

namespace {

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
    unsigned char stack_byte;
    return queryNUMANode(&stack_byte);
#else
    return -1;
#endif
}

/// Audit NUMA placement of expert GEMM weights.
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
        LOG_DEBUG("[MoEWeightService][NUMA] Cannot determine current NUMA node, skipping audit");
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
        LOG_WARN("[MoEWeightService][NUMA] layer " << layer_idx << " " << label
                 << ": " << remote << "/" << sampled << " sampled experts on WRONG NUMA node"
                 << " (expected=" << expected_node
                 << ", first_remote: expert " << first_remote_expert
                 << " on node " << first_remote_node << ")"
                 << (unmapped > 0 ? (std::string(" [") + std::to_string(unmapped) + " unmapped]") : ""));
    } else {
        LOG_DEBUG("[MoEWeightService][NUMA] layer " << layer_idx << " " << label
                 << ": all " << sampled << "/" << static_cast<int>(experts.size())
                 << " sampled experts on correct NUMA node " << expected_node
                 << (unmapped > 0 ? (std::string(" [") + std::to_string(unmapped) + " unmapped]") : ""));
    }
}

} // anonymous namespace

// =========================================================================
// extractExpertViews — Create 2D views from 3D packed tensors
// =========================================================================

bool MoEExpertWeightService::extractExpertViews(MoEWeightContext& ctx)
{
    if (!ctx.gate_exps || !ctx.up_exps || !ctx.down_exps)
    {
        LOG_ERROR("[MoEWeightService] Cannot extract views: null expert tensors");
        return false;
    }

    const int num_experts = ctx.num_experts;
    if (num_experts <= 0)
    {
        LOG_ERROR("[MoEWeightService] Invalid num_experts: " << num_experts);
        return false;
    }

    // EP range: only extract views for local experts
    // Dynamic rebalancing: when expert_mask is set, expert data is replicated.
    // Extract views for ALL experts so GEMM engines can be prepared for all.
    // The mask only controls which experts are executed, not which views exist.
    const bool extract_all = !ctx.expert_mask.empty();
    const int local_start = ctx.local_expert_start;
    const int local_count = (ctx.local_expert_count < 0)
                                ? num_experts
                                : ctx.local_expert_count;
    const int local_end = local_start + local_count;

    ctx.expert_gate_views.resize(num_experts);
    ctx.expert_up_views.resize(num_experts);
    ctx.expert_down_views.resize(num_experts);

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

    if (!extract_views(ctx.gate_exps, num_experts, ctx.expert_gate_views))
        return false;
    if (!extract_views(ctx.up_exps, num_experts, ctx.expert_up_views))
        return false;
    if (!extract_views(ctx.down_exps, num_experts, ctx.expert_down_views))
        return false;

    LOG_DEBUG("[MoEWeightService] Extracted " << (extract_all ? num_experts : local_count) << "/" << num_experts
                                         << " expert 2D views (EP range [" << local_start
                                         << ", " << local_end << ")"
                                        << (extract_all ? " extract_all=true" : "") << ")");
    return true;
}

// =========================================================================
// prepareGemmEngines — Prepare GEMM engines for all expert views
// =========================================================================

bool MoEExpertWeightService::prepareGemmEngines(MoEWeightContext& ctx)
{
    const int num_experts = ctx.num_experts;
    if (ctx.expert_gate_views.empty() ||
        static_cast<int>(ctx.expert_gate_views.size()) != num_experts)
    {
        LOG_ERROR("[MoEWeightService] prepareGemmEngines: call extractExpertViews() first");
        return false;
    }

    // EP range
    // Dynamic rebalancing: when expert_mask is set, prepare ONLY mask-active
    // experts (not all). Views exist for all experts, but GEMM engines are
    // expensive (VNNI repacking). Newly-acquired experts get engines lazily
    // via registerAndPrepareNewExperts() after rebalancing.
    const bool use_mask = !ctx.expert_mask.empty();
    const int local_start = ctx.local_expert_start;
    const int local_count = (ctx.local_expert_count < 0)
                                ? num_experts
                                : ctx.local_expert_count;
    const int local_end = local_start + local_count;

    // Build list of experts to prepare
    std::vector<int> experts_to_prep;
    if (use_mask)
    {
        for (int e = 0; e < num_experts; ++e)
            if (ctx.expert_mask[e])
                experts_to_prep.push_back(e);
    }
    else
    {
        for (int e = local_start; e < local_end; ++e)
            experts_to_prep.push_back(e);
    }
    const int prep_count = static_cast<int>(experts_to_prep.size());

    ctx.prepared_gate_gemm.resize(num_experts, nullptr);
    ctx.prepared_up_gemm.resize(num_experts, nullptr);
    ctx.prepared_down_gemm.resize(num_experts, nullptr);

    LOG_DEBUG("[MoEWeightService] Preparing GEMM engines for " << prep_count << "/" << num_experts
              << " experts (3 weights each = " << (prep_count * 3) << " total"
              << (use_mask ? " [dynamic rebalance: mask-active only]" : "") << ")...");

#ifdef HAVE_CUDA
    if (ctx.device_id.is_cuda())
    {
        return prepareGemmEnginesCUDA(ctx);
    }
#endif
#ifdef HAVE_ROCM
    if (ctx.device_id.is_rocm())
    {
        return prepareGemmEnginesROCm(ctx);
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

        if (!ctx.expert_gate_views[e] || !ctx.expert_up_views[e] || !ctx.expert_down_views[e])
        {
            LOG_ERROR("[MoEWeightService] Null expert view for local expert " << e);
            error_flag.store(true, std::memory_order_relaxed);
            continue;
        }

        auto *gp = KernelFactory::getOrCreatePreparedGemmWeights(
            ctx.expert_gate_views[e].get(), ctx.device_id);
        auto *up = KernelFactory::getOrCreatePreparedGemmWeights(
            ctx.expert_up_views[e].get(), ctx.device_id);
        auto *dp = KernelFactory::getOrCreatePreparedGemmWeights(
            ctx.expert_down_views[e].get(), ctx.device_id);

        if (!gp || !up || !dp)
        {
            LOG_ERROR("[MoEWeightService] Failed to prepare GEMM weights for expert " << e);
            error_flag.store(true, std::memory_order_relaxed);
            continue;
        }

        ctx.prepared_gate_gemm[e] = KernelFactory::getOrCreateGemmEngine(gp);
        ctx.prepared_up_gemm[e] = KernelFactory::getOrCreateGemmEngine(up);
        ctx.prepared_down_gemm[e] = KernelFactory::getOrCreateGemmEngine(dp);

        if (!ctx.prepared_gate_gemm[e] || !ctx.prepared_up_gemm[e] || !ctx.prepared_down_gemm[e])
        {
            LOG_ERROR("[MoEWeightService] Failed to create GEMM engine for expert " << e);
            error_flag.store(true, std::memory_order_relaxed);
            continue;
        }
    }

    if (error_flag.load())
    {
        return false;
    }

    LOG_DEBUG("[MoEWeightService] All " << (prep_count * 3) << " expert GEMM engines prepared (CPU/KernelFactory path)");

    // NUMA audit: verify packed weights landed on the correct NUMA node.
    auditExpertNUMA(experts_to_prep, ctx.prepared_gate_gemm,
                    "initial_pack", ctx.layer_idx);

    // Release mmap pages backing the raw expert weight data.
    // The VNNI interleaved engines now own their own copy — the original
    // mmap data is never accessed again. Releasing per-layer reduces peak RSS
    // by ~500 MB/layer instead of waiting for a bulk release at the end.
    // NOTE: Only safe for mmap-backed tensors. Expert-parallel sliced tensors
    // are heap-allocated copies — MADV_DONTNEED on heap memory corrupts malloc metadata.
    {
        size_t released = 0;
        if (ctx.gate_exps && ctx.gate_exps->is_mmap_data())
            released += MmapRegion::adviseDontneedRange(ctx.gate_exps->raw_data(), ctx.gate_exps->size_bytes());
        if (ctx.up_exps && ctx.up_exps->is_mmap_data())
            released += MmapRegion::adviseDontneedRange(ctx.up_exps->raw_data(), ctx.up_exps->size_bytes());
        if (ctx.down_exps && ctx.down_exps->is_mmap_data())
            released += MmapRegion::adviseDontneedRange(ctx.down_exps->raw_data(), ctx.down_exps->size_bytes());
        if (released > 0)
            LOG_DEBUG("[MoEWeightService] Advised " << (released >> 20) << " MB of mmap pages DONTNEED after engine packing");
    }

    return true;
}

// =========================================================================
// releaseRawWeights — Free 3D parent weight tensors
// =========================================================================

size_t MoEExpertWeightService::releaseRawWeights(MoEWeightContext& ctx)
{
    size_t freed = 0;

    auto release_3d = [&](TensorBase*& tensor, const char* name) {
        if (!tensor) return;
        if (tensor->is_mmap_data()) {
            freed += tensor->size_bytes();
            LOG_DEBUG("[MoEWeightService] " << name << ": mmap-backed ("
                      << (tensor->size_bytes() >> 20) << " MB) — already DONTNEED");
        } else if (!tensor->is_raw_data_released()) {
            size_t bytes = tensor->size_bytes();
            tensor->release_raw_data();
            freed += bytes;
            LOG_DEBUG("[MoEWeightService] " << name << ": released "
                      << (bytes >> 20) << " MB heap data");
        }
    };

    release_3d(ctx.gate_exps, "gate_exps");
    release_3d(ctx.up_exps, "up_exps");
    release_3d(ctx.down_exps, "down_exps");

    // Null out 3D pointers to prevent accidental fallback to raw repacking.
    ctx.gate_exps = nullptr;
    ctx.up_exps = nullptr;
    ctx.down_exps = nullptr;

    if (freed > 0) {
        LOG_INFO("[MoEWeightService] Layer " << ctx.layer_idx
                 << ": released " << (freed >> 20) << " MB raw expert weights"
                 << " (VNNI engines retain packed data)");
    }

    return freed;
}

// =========================================================================
// Weight serialization (for MPI transfer)
// =========================================================================

ExpertWeightBlobs MoEExpertWeightService::detachAndSerializeExpert(MoEWeightContext& ctx, int expert_id)
{
    ExpertWeightBlobs blobs;

    auto serialize_proj = [&](ITensorGemm* engine, const char* /*proj_name*/) -> std::vector<uint8_t> {
        if (!engine) return {};
        if (!engine->hasWeights()) return {};
        auto packed = engine->detachWeights();
        if (!packed) return {};
        return packed_weights_serialization::serialize(*packed);
    };

    blobs.gate = serialize_proj(ctx.prepared_gate_gemm[expert_id], "gate");
    blobs.up   = serialize_proj(ctx.prepared_up_gemm[expert_id], "up");
    blobs.down = serialize_proj(ctx.prepared_down_gemm[expert_id], "down");

    return blobs;
}

ExpertWeightBlobs MoEExpertWeightService::serializeExpert(const MoEWeightContext& ctx, int expert_id)
{
    using namespace cpu::native_vnni;

    ExpertWeightBlobs blobs;

    auto serialize_proj = [](ITensorGemm* engine) -> std::vector<uint8_t> {
        if (!engine || !engine->hasWeights()) return {};
        auto* vnni_kernel = dynamic_cast<const CPUNativeVNNIGemmKernel*>(engine);
        if (!vnni_kernel) return {};
        CPUPackedWeights wrapper(vnni_kernel->packedWeights());
        return packed_weights_serialization::serialize(wrapper);
    };

    blobs.gate = serialize_proj(ctx.prepared_gate_gemm[expert_id]);
    blobs.up   = serialize_proj(ctx.prepared_up_gemm[expert_id]);
    blobs.down = serialize_proj(ctx.prepared_down_gemm[expert_id]);

    return blobs;
}

// =========================================================================
// Phased rebalance API
// =========================================================================

bool MoEExpertWeightService::registerTransferredExpert(MoEWeightContext& ctx, int expert_id, const ExpertWeightBlobs& blobs)
{
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
            view.get(), ctx.device_id, std::move(kernel)) != nullptr;
    };

    bool gate_ok = register_one(blobs.gate, ctx.expert_gate_views[expert_id]);
    bool up_ok   = register_one(blobs.up,   ctx.expert_up_views[expert_id]);
    bool down_ok = register_one(blobs.down,  ctx.expert_down_views[expert_id]);

    return gate_ok && up_ok && down_ok;
}

std::vector<const TensorBase*> MoEExpertWeightService::releaseDepartedExperts(
    MoEWeightContext& ctx, const std::vector<bool>& new_mask)
{
    std::vector<const TensorBase*> evict_tensors;

    for (int e = 0; e < ctx.num_experts; ++e) {
        if (!new_mask[e] && ctx.prepared_gate_gemm[e]) {
            // Release packed weights from GEMM engines
            ctx.prepared_gate_gemm[e]->releaseWeights();
            ctx.prepared_up_gemm[e]->releaseWeights();
            ctx.prepared_down_gemm[e]->releaseWeights();

            // Collect tensor views for batch cache eviction by the caller
            if (ctx.expert_gate_views[e])
                evict_tensors.push_back(ctx.expert_gate_views[e].get());
            if (ctx.expert_up_views[e])
                evict_tensors.push_back(ctx.expert_up_views[e].get());
            if (ctx.expert_down_views[e])
                evict_tensors.push_back(ctx.expert_down_views[e].get());

            // Null out the engine pointers
            ctx.prepared_gate_gemm[e] = nullptr;
            ctx.prepared_up_gemm[e] = nullptr;
            ctx.prepared_down_gemm[e] = nullptr;
        }
    }
    return evict_tensors;
}

bool MoEExpertWeightService::registerAndPrepareNewExperts(
    MoEWeightContext& ctx,
    const std::vector<bool>& new_mask,
    const std::unordered_map<int, ExpertWeightBlobs>* received_weights)
{
    // Find newly-acquired experts (true in new_mask, not previously prepared)
    std::vector<int> new_experts;
    for (int e = 0; e < ctx.num_experts; ++e) {
        if (new_mask[e] && !ctx.prepared_gate_gemm[e])
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
                if (registerTransferredExpert(ctx, e, it->second))
                    ++transferred_count;
            }
        }
    }

    // Prepare engines (cache hit for transferred, full pack for rest)
    for (int idx = 0; idx < count; ++idx) {
        if (error_flag.load(std::memory_order_relaxed)) break;
        const int e = new_experts[idx];

        if (!ctx.expert_gate_views[e] || !ctx.expert_up_views[e] ||
            !ctx.expert_down_views[e]) {
            LOG_ERROR("[MoEWeightService] Null view for new expert " << e);
            error_flag.store(true, std::memory_order_relaxed);
            continue;
        }

        auto *gp = KernelFactory::getOrCreatePreparedGemmWeights(
            ctx.expert_gate_views[e].get(), ctx.device_id);
        auto *up = KernelFactory::getOrCreatePreparedGemmWeights(
            ctx.expert_up_views[e].get(), ctx.device_id);
        auto *dp = KernelFactory::getOrCreatePreparedGemmWeights(
            ctx.expert_down_views[e].get(), ctx.device_id);

        if (!gp || !up || !dp) {
            LOG_ERROR("[MoEWeightService] Failed GEMM weights for new expert " << e);
            error_flag.store(true, std::memory_order_relaxed);
            continue;
        }

        ctx.prepared_gate_gemm[e] = KernelFactory::getOrCreateGemmEngine(gp);
        ctx.prepared_up_gemm[e] = KernelFactory::getOrCreateGemmEngine(up);
        ctx.prepared_down_gemm[e] = KernelFactory::getOrCreateGemmEngine(dp);

        if (!ctx.prepared_gate_gemm[e] || !ctx.prepared_up_gemm[e] ||
            !ctx.prepared_down_gemm[e]) {
            LOG_ERROR("[MoEWeightService] Failed GEMM engine for new expert " << e);
            error_flag.store(true, std::memory_order_relaxed);
            continue;
        }
    }

    if (error_flag.load()) return false;

    auto t_end = std::chrono::high_resolution_clock::now();
    double prep_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    LOG_DEBUG("[MoEWeightService] Engine prep for " << new_experts.size()
              << " experts (" << transferred_count << " transferred): "
              << std::fixed << std::setprecision(1) << prep_ms << " ms");

    auditExpertNUMA(new_experts, ctx.prepared_gate_gemm,
                    (transferred_count > 0 ? "rebalance_transferred" : "rebalance_repacked"),
                    ctx.layer_idx);
    return true;
}

// =========================================================================
// CUDA batch packing
// =========================================================================

#ifdef HAVE_CUDA
bool MoEExpertWeightService::prepareGemmEnginesCUDA(MoEWeightContext& ctx)
{
    using namespace llaminar2::cuda;
    const int num_experts = ctx.num_experts;
    const int cuda_id = ctx.device_id.cuda_ordinal();

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
            LOG_ERROR("[MoEWeightService::CUDA] Failed to batch-pack " << label);
            return false;
        }

        if (!batch->uploadToDevice(cuda_id))
        {
            LOG_ERROR("[MoEWeightService::CUDA] Failed to upload " << label << " to device " << cuda_id);
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
            ctx.moe_owned_kernels.push_back(std::move(kernel));
        }

        batch->freeHostBuffers();
        out_lifetime = std::move(batch);
        return true;
    };

    if (!batchPackAndCreateKernels(ctx.expert_gate_views, ctx.prepared_gate_gemm,
                                   ctx.moe_packed_gate_lifetime, "gate"))
        return false;
    if (!batchPackAndCreateKernels(ctx.expert_up_views, ctx.prepared_up_gemm,
                                   ctx.moe_packed_up_lifetime, "up"))
        return false;
    if (!batchPackAndCreateKernels(ctx.expert_down_views, ctx.prepared_down_gemm,
                                   ctx.moe_packed_down_lifetime, "down"))
        return false;

    LOG_DEBUG("[MoEWeightService] All " << (num_experts * 3)
              << " expert GEMM engines prepared (CUDA batch path, 3 GPU allocs)");

    // Release mmap pages for raw expert weights (now uploaded to GPU).
    {
        size_t released = 0;
        if (ctx.gate_exps && ctx.gate_exps->is_mmap_data())
            released += MmapRegion::adviseDontneedRange(ctx.gate_exps->raw_data(), ctx.gate_exps->size_bytes());
        if (ctx.up_exps && ctx.up_exps->is_mmap_data())
            released += MmapRegion::adviseDontneedRange(ctx.up_exps->raw_data(), ctx.up_exps->size_bytes());
        if (ctx.down_exps && ctx.down_exps->is_mmap_data())
            released += MmapRegion::adviseDontneedRange(ctx.down_exps->raw_data(), ctx.down_exps->size_bytes());
        if (released > 0)
            LOG_DEBUG("[MoEWeightService] Advised " << (released >> 20) << " MB of mmap pages DONTNEED after CUDA packing");
    }

    return true;
}
#endif // HAVE_CUDA

// =========================================================================
// ROCm batch packing
// =========================================================================

#ifdef HAVE_ROCM
bool MoEExpertWeightService::prepareGemmEnginesROCm(MoEWeightContext& ctx)
{
    using namespace llaminar2::rocm;
    const int num_experts = ctx.num_experts;
    const int rocm_id = ctx.device_id.rocm_ordinal();

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
            LOG_ERROR("[MoEWeightService::ROCm] Failed to batch-pack " << label);
            return false;
        }

        if (!batch->uploadToDevice(rocm_id))
        {
            LOG_ERROR("[MoEWeightService::ROCm] Failed to upload " << label << " to device " << rocm_id);
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
            ctx.moe_owned_kernels.push_back(std::move(kernel));
        }

        batch->freeHostBuffers();
        out_lifetime = std::move(batch);
        return true;
    };

    if (!batchPackAndCreateKernels(ctx.expert_gate_views, ctx.prepared_gate_gemm,
                                   ctx.moe_packed_gate_lifetime, "gate"))
        return false;
    if (!batchPackAndCreateKernels(ctx.expert_up_views, ctx.prepared_up_gemm,
                                   ctx.moe_packed_up_lifetime, "up"))
        return false;
    if (!batchPackAndCreateKernels(ctx.expert_down_views, ctx.prepared_down_gemm,
                                   ctx.moe_packed_down_lifetime, "down"))
        return false;

    LOG_DEBUG("[MoEWeightService] All " << (num_experts * 3)
              << " expert GEMM engines prepared (ROCm batch path, 3 GPU allocs)");

    // Release mmap pages for raw expert weights (now uploaded to GPU).
    {
        size_t released = 0;
        if (ctx.gate_exps && ctx.gate_exps->is_mmap_data())
            released += MmapRegion::adviseDontneedRange(ctx.gate_exps->raw_data(), ctx.gate_exps->size_bytes());
        if (ctx.up_exps && ctx.up_exps->is_mmap_data())
            released += MmapRegion::adviseDontneedRange(ctx.up_exps->raw_data(), ctx.up_exps->size_bytes());
        if (ctx.down_exps && ctx.down_exps->is_mmap_data())
            released += MmapRegion::adviseDontneedRange(ctx.down_exps->raw_data(), ctx.down_exps->size_bytes());
        if (released > 0)
            LOG_DEBUG("[MoEWeightService] Advised " << (released >> 20) << " MB of mmap pages DONTNEED after ROCm packing");
    }

    return true;
}
#endif // HAVE_ROCM

} // namespace llaminar2
