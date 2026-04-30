/**
 * @file MoEExpertWeightService.cpp
 * @brief Implementation of MoE expert weight lifecycle service.
 *
 * Contains all weight preparation, serialization, and rebalancing logic
 * extracted from MoEExpertComputeStage.cpp. See MoEExpertWeightService.h for API docs.
 */

#include "MoEExpertWeightService.h"
#include "GPUExpertTransfer.h"
#include "../../tensors/Tensors.h"
#include "../../tensors/BlockStructures.h"
#include "../../kernels/KernelFactory.h"
#include "../../kernels/PackedWeightsSerialization.h"
#include "../../kernels/cpu/native_vnni/CPUPackedWeights.h"
#include "../../kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
#include "../../loaders/MmapRegion.h"
#include "../../loaders/gpu_pipeline/LoadOrchestrator.h"
#include "../../backends/BackendManager.h"
#include "../../utils/Assertions.h"
#include "../../utils/DebugEnv.h"
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

static size_t quantizedViewRawBytes(const TensorBase& tensor)
{
    const size_t reported = tensor.size_bytes();
    if (reported > 0)
        return reported;

    const auto& shape = tensor.shape();
    if (shape.size() != 2)
        return 0;

    const size_t rows = shape[0];
    const size_t cols = shape[1];
    auto bytes_for = [rows, cols](size_t block_size, size_t block_bytes) -> size_t
    {
        const size_t blocks_per_row = (cols + block_size - 1) / block_size;
        return rows * blocks_per_row * block_bytes;
    };

    switch (tensor.native_type())
    {
    case TensorType::IQ4_NL: return bytes_for(IQ4_NLBlock::BLOCK_SIZE, sizeof(IQ4_NLBlock));
    case TensorType::IQ4_XS: return bytes_for(IQ4_XSBlock::BLOCK_SIZE, sizeof(IQ4_XSBlock));
    case TensorType::Q8_0: return bytes_for(Q8_0Block::BLOCK_SIZE, sizeof(Q8_0Block));
    case TensorType::Q4_0: return bytes_for(Q4_0Block::BLOCK_SIZE, sizeof(Q4_0Block));
    case TensorType::Q4_1: return bytes_for(Q4_1Block::BLOCK_SIZE, sizeof(Q4_1Block));
    case TensorType::Q5_0: return bytes_for(Q5_0Block::BLOCK_SIZE, sizeof(Q5_0Block));
    case TensorType::Q5_1: return bytes_for(Q5_1Block::BLOCK_SIZE, sizeof(Q5_1Block));
    case TensorType::Q2_K: return bytes_for(Q2_KBlock::BLOCK_SIZE, sizeof(Q2_KBlock));
    case TensorType::Q3_K: return bytes_for(Q3_KBlock::BLOCK_SIZE, sizeof(Q3_KBlock));
    case TensorType::Q4_K: return bytes_for(Q4_KBlock::BLOCK_SIZE, sizeof(Q4_KBlock));
    case TensorType::Q5_K: return bytes_for(Q5_KBlock::BLOCK_SIZE, sizeof(Q5_KBlock));
    case TensorType::Q6_K: return bytes_for(Q6_KBlock::BLOCK_SIZE, sizeof(Q6_KBlock));
    case TensorType::Q8_K: return bytes_for(Q8_KBlock::BLOCK_SIZE, sizeof(Q8_KBlock));
    case TensorType::IQ2_XXS: return bytes_for(IQ2_XXSBlock::BLOCK_SIZE, sizeof(IQ2_XXSBlock));
    case TensorType::IQ2_XS: return bytes_for(IQ2_XSBlock::BLOCK_SIZE, sizeof(IQ2_XSBlock));
    case TensorType::IQ2_S: return bytes_for(IQ2_SBlock::BLOCK_SIZE, sizeof(IQ2_SBlock));
    case TensorType::IQ3_XXS: return bytes_for(IQ3_XXSBlock::BLOCK_SIZE, sizeof(IQ3_XXSBlock));
    case TensorType::IQ3_S: return bytes_for(IQ3_SBlock::BLOCK_SIZE, sizeof(IQ3_SBlock));
    case TensorType::IQ1_S: return bytes_for(IQ1_SBlock::BLOCK_SIZE, sizeof(IQ1_SBlock));
    case TensorType::IQ1_M: return bytes_for(IQ1_MBlock::BLOCK_SIZE, sizeof(IQ1_MBlock));
    default: return 0;
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

    // GPU path: unified H2D + GPU repack via LoadOrchestrator pipeline.
    if (ctx.device_id.is_gpu())
    {
        return prepareGemmEnginesGPU(ctx);
    }

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

    // GPU path: raw H2D + GPU repack via LoadOrchestrator.
    // Ignores received_weights (CPU-serialized packed weights are useless on GPU).
    // Uses raw GGUF data from expert views — same pipeline as initial load.
    if (ctx.device_id.is_gpu()) {
        if (received_weights && !received_weights->empty()) {
            LOG_DEBUG("[MoEWeightService] Ignoring " << received_weights->size()
                      << " CPU-serialized weight blobs — using GPU repack pipeline instead");
        }
        return registerAndPrepareNewExpertsGPU(ctx, new_experts);
    }

    // CPU path: deserialize transferred weights, then KernelFactory pack
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
// GPU rebalance path (LoadOrchestrator: raw H2D + GPU repack for new experts)
// =========================================================================

bool MoEExpertWeightService::registerAndPrepareNewExpertsGPU(
    MoEWeightContext& ctx, const std::vector<int>& new_experts)
{
    using Clock = std::chrono::high_resolution_clock;
    const auto t_start = Clock::now();

    const int gpu_ordinal = ctx.device_id.is_cuda()
                                ? ctx.device_id.cuda_ordinal()
                                : ctx.device_id.rocm_ordinal();
    const int count = static_cast<int>(new_experts.size());

    IBackend* backend = getBackendFor(ctx.device_id);
    if (!backend)
    {
        LOG_ERROR("[MoEWeightService::GPU-rebalance] No backend for "
                  << ctx.device_id.to_string());
        return false;
    }

    // Weight groups: gate, up, down
    struct WeightGroup {
        const char* label;
        std::vector<std::shared_ptr<TensorBase>>& views;
        std::vector<ITensorGemm*>& out_gemms;
    };
    WeightGroup groups[] = {
        {"gate", ctx.expert_gate_views, ctx.prepared_gate_gemm},
        {"up",   ctx.expert_up_views,   ctx.prepared_up_gemm},
        {"down", ctx.expert_down_views,  ctx.prepared_down_gemm},
    };

    auto orchestrator = std::make_shared<LoadOrchestrator>(backend);
    orchestrator->addDevice(gpu_ordinal);

    size_t max_raw_bytes = 0;
    size_t total_planned = 0;

    // Phase 1: Plan weights for new experts
    for (auto& grp : groups)
    {
        for (int idx = 0; idx < count; ++idx)
        {
            const int e = new_experts[idx];
            const auto& view = grp.views[e];
            if (!view)
            {
                LOG_ERROR("[MoEWeightService::GPU-rebalance] Null view for expert "
                          << e << " in " << grp.label);
                return false;
            }

            auto* unpackable = dynamic_cast<IINT8Unpackable*>(view.get());
            const NativeVnniFormatInfo* vnni = unpackable ? unpackable->vnniFormatInfo() : nullptr;
            if (!vnni)
            {
                LOG_ERROR("[MoEWeightService::GPU-rebalance] Expert " << e << " "
                          << grp.label << " has no VNNI format info");
                return false;
            }

            const int N = static_cast<int>(view->rows());
            const int K = static_cast<int>(view->cols());
            const size_t raw_bytes = quantizedViewRawBytes(*view);
            if (raw_bytes == 0)
            {
                LOG_ERROR("[MoEWeightService::GPU-rebalance] Could not determine raw byte size for expert "
                          << e << " " << grp.label);
                return false;
            }
            const std::string slot_name = std::string(grp.label) + "_e" + std::to_string(e);

            orchestrator->planWeight(gpu_ordinal, slot_name, N, K,
                                     vnni->payload_bytes, vnni->is_asymmetric,
                                     vnni->has_emins, raw_bytes);
            max_raw_bytes = std::max(max_raw_bytes, raw_bytes);
            ++total_planned;
        }
    }

    LOG_DEBUG("[MoEWeightService::GPU-rebalance] Planned " << total_planned
              << " expert weights for " << ctx.device_id.to_string()
              << " (layer " << ctx.layer_idx << ")");

    // Phase 2: Allocate VRAM pool + pinned ring buffer
    const auto& rocm_cfg = debugEnv().rocm;
    orchestrator->allocate(max_raw_bytes, rocm_cfg.repack_streams);

    // Phase 3: Create weight jobs from raw GGUF data
    for (auto& grp : groups)
    {
        for (int idx = 0; idx < count; ++idx)
        {
            const int e = new_experts[idx];
            const auto& view = grp.views[e];

            auto* unpackable = dynamic_cast<IINT8Unpackable*>(view.get());
            const NativeVnniFormatInfo* vnni = unpackable->vnniFormatInfo();

            auto repack_fmt = codebookIdToRepackFormat(vnni->codebook_id, vnni->is_superblock);
            if (!repack_fmt)
            {
                LOG_ERROR("[MoEWeightService::GPU-rebalance] Unsupported repack format for expert "
                          << e << " " << grp.label
                          << " (codebook=" << static_cast<int>(vnni->codebook_id)
                          << ", superblock=" << vnni->is_superblock << ")");
                return false;
            }

            const std::string slot_name = std::string(grp.label) + "_e" + std::to_string(e);

            WeightJob job;
            job.name = slot_name;
            job.host_raw_data = view->raw_data();
            job.raw_bytes = quantizedViewRawBytes(*view);
            job.format = *repack_fmt;
            job.N = static_cast<int>(view->rows());
            job.K = static_cast<int>(view->cols());
            job.is_asymmetric = vnni->is_asymmetric;

            orchestrator->addWeightJob(gpu_ordinal, job);
        }
    }

    // Phase 4: Execute pipeline (pipelined H2D + GPU repack)
    orchestrator->load();

    auto* pool = orchestrator->getPool(gpu_ordinal);
    if (!pool)
    {
        LOG_ERROR("[MoEWeightService::GPU-rebalance] Pool not found after load");
        return false;
    }

    // Phase 5: Create per-expert GEMM kernels from pool slots
    for (auto& grp : groups)
    {
        for (int idx = 0; idx < count; ++idx)
        {
            const int e = new_experts[idx];
            const auto& view = grp.views[e];
            const std::string slot_name = std::string(grp.label) + "_e" + std::to_string(e);

            auto slot = pool->getSlot(slot_name);
            if (!slot)
            {
                LOG_ERROR("[MoEWeightService::GPU-rebalance] No slot for " << slot_name);
                return false;
            }

            auto* unpackable = dynamic_cast<IINT8Unpackable*>(view.get());
            const NativeVnniFormatInfo* vnni = unpackable->vnniFormatInfo();
            const int N = static_cast<int>(view->rows());
            const int K = static_cast<int>(view->cols());
            const uint32_t blocks_per_row = static_cast<uint32_t>(K / 32);

            std::shared_ptr<ITensorGemm> kernel;

#ifdef HAVE_CUDA
            if (ctx.device_id.is_cuda())
            {
                kernel = std::make_shared<llaminar2::cuda::CUDAQuantisedGemmKernel>(
                    N, K, gpu_ordinal,
                    slot->d_native_vnni_payload,
                    static_cast<uint16_t*>(slot->d_native_vnni_scales),
                    static_cast<uint16_t*>(slot->d_native_vnni_mins),
                    static_cast<uint32_t*>(slot->d_native_vnni_emins),
                    vnni->codebook_id, blocks_per_row,
                    orchestrator);
            }
#endif
#ifdef HAVE_ROCM
            if (ctx.device_id.is_rocm())
            {
                kernel = std::make_shared<llaminar2::rocm::ROCmQuantisedGemmKernel>(
                    N, K, gpu_ordinal,
                    slot->d_native_vnni_payload,
                    slot->d_native_vnni_scales,
                    slot->d_native_vnni_mins,
                    slot->d_native_vnni_emins,
                    vnni->codebook_id, blocks_per_row,
                    orchestrator);
            }
#endif

            if (!kernel)
            {
                LOG_ERROR("[MoEWeightService::GPU-rebalance] Failed to create kernel for " << slot_name);
                return false;
            }

            grp.out_gemms[e] = kernel.get();
            ctx.moe_owned_kernels.push_back(std::move(kernel));
        }
    }

    // Phase 6: Release staging resources; VRAM pool stays alive via kernel lifetime_owner_
    orchestrator->finalize();

    const auto elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
    LOG_INFO("[MoEWeightService::GPU-rebalance] " << (count * 3) << " expert GEMM engines "
             << "prepared via GPU pipeline in "
             << std::fixed << std::setprecision(1) << elapsed_ms << " ms"
             << " (layer " << ctx.layer_idx << ", "
             << ctx.device_id.to_string() << ")");

    return true;
}

// =========================================================================
// GPU pipeline path (LoadOrchestrator: raw H2D + GPU repack)
// =========================================================================

bool MoEExpertWeightService::prepareGemmEnginesGPU(MoEWeightContext& ctx)
{
    using Clock = std::chrono::high_resolution_clock;
    const auto t_start = Clock::now();

    const int num_experts = ctx.num_experts;
    const int gpu_ordinal = ctx.device_id.is_cuda()
                                ? ctx.device_id.cuda_ordinal()
                                : ctx.device_id.rocm_ordinal();

    // Build list of local expert indices (same as old batch path).
    std::vector<int> local_experts;
    {
        const bool use_mask = !ctx.expert_mask.empty();
        const int local_start = ctx.local_expert_start;
        const int local_count = (ctx.local_expert_count < 0)
                                    ? num_experts
                                    : ctx.local_expert_count;
        const int local_end = local_start + local_count;
        if (use_mask)
        {
            for (int e = 0; e < num_experts; ++e)
                if (ctx.expert_mask[e])
                    local_experts.push_back(e);
        }
        else
        {
            for (int e = local_start; e < local_end; ++e)
                local_experts.push_back(e);
        }
    }
    const int local_count = static_cast<int>(local_experts.size());
    if (local_count == 0)
    {
        LOG_WARN("[MoEWeightService::GPU] No local experts to prepare");
        return true;
    }

    // Get backend
    IBackend* backend = getBackendFor(ctx.device_id);
    if (!backend)
    {
        LOG_ERROR("[MoEWeightService::GPU] No backend for " << ctx.device_id.to_string());
        return false;
    }

    // Weight groups: gate, up, down — each with local_count expert views.
    struct WeightGroup {
        const char* label;
        std::vector<std::shared_ptr<TensorBase>>& views;
        std::vector<ITensorGemm*>& out_gemms;
        std::shared_ptr<void>& out_lifetime;
    };
    WeightGroup groups[] = {
        {"gate", ctx.expert_gate_views, ctx.prepared_gate_gemm, ctx.moe_packed_gate_lifetime},
        {"up",   ctx.expert_up_views,   ctx.prepared_up_gemm,   ctx.moe_packed_up_lifetime},
        {"down", ctx.expert_down_views,  ctx.prepared_down_gemm,  ctx.moe_packed_down_lifetime},
    };

    // Create one LoadOrchestrator for ALL expert weights (3 groups × local_count).
    // Single VRAM allocation, pipelined H2D + GPU repack.
    auto orchestrator = std::make_shared<LoadOrchestrator>(backend);
    orchestrator->addDevice(gpu_ordinal);

    size_t max_raw_bytes = 0;
    size_t total_planned = 0;

    // Phase 1: Plan all expert weights
    for (auto& grp : groups)
    {
        for (int idx = 0; idx < local_count; ++idx)
        {
            const int e = local_experts[idx];
            const auto& view = grp.views[e];
            if (!view)
            {
                LOG_ERROR("[MoEWeightService::GPU] Null view for expert " << e
                          << " in " << grp.label);
                return false;
            }

            auto* unpackable = dynamic_cast<IINT8Unpackable*>(view.get());
            const NativeVnniFormatInfo* vnni = unpackable ? unpackable->vnniFormatInfo() : nullptr;
            if (!vnni)
            {
                LOG_ERROR("[MoEWeightService::GPU] Expert " << e << " " << grp.label
                          << " has no VNNI format info — cannot use GPU repack");
                return false;
            }

            const int N = static_cast<int>(view->rows());
            const int K = static_cast<int>(view->cols());
            const size_t raw_bytes = quantizedViewRawBytes(*view);
            if (raw_bytes == 0)
            {
                LOG_ERROR("[MoEWeightService::GPU] Could not determine raw byte size for expert "
                          << e << " " << grp.label);
                return false;
            }

            // Unique name per expert per group
            const std::string slot_name = std::string(grp.label) + "_e" + std::to_string(e);

            orchestrator->planWeight(gpu_ordinal, slot_name, N, K,
                                     vnni->payload_bytes, vnni->is_asymmetric,
                                     vnni->has_emins, raw_bytes);
            max_raw_bytes = std::max(max_raw_bytes, raw_bytes);
            ++total_planned;
        }
    }

    LOG_INFO("[MoEWeightService::GPU] Planned " << total_planned << " expert weights for "
             << ctx.device_id.to_string() << " (layer " << ctx.layer_idx << ")");

    // Phase 2: Allocate VRAM pool + pinned ring buffer
    const auto& rocm_cfg = debugEnv().rocm;
    orchestrator->allocate(max_raw_bytes, rocm_cfg.repack_streams);

    // Phase 3: Create weight jobs
    for (auto& grp : groups)
    {
        for (int idx = 0; idx < local_count; ++idx)
        {
            const int e = local_experts[idx];
            const auto& view = grp.views[e];

            auto* unpackable = dynamic_cast<IINT8Unpackable*>(view.get());
            const NativeVnniFormatInfo* vnni = unpackable->vnniFormatInfo();

            auto repack_fmt = codebookIdToRepackFormat(vnni->codebook_id, vnni->is_superblock);
            if (!repack_fmt)
            {
                LOG_ERROR("[MoEWeightService::GPU] Unsupported repack format for expert " << e
                          << " " << grp.label << " (codebook=" << static_cast<int>(vnni->codebook_id)
                          << ", superblock=" << vnni->is_superblock << ")");
                return false;
            }

            const std::string slot_name = std::string(grp.label) + "_e" + std::to_string(e);

            WeightJob job;
            job.name = slot_name;
            job.host_raw_data = view->raw_data();
            job.raw_bytes = quantizedViewRawBytes(*view);
            job.format = *repack_fmt;
            job.N = static_cast<int>(view->rows());
            job.K = static_cast<int>(view->cols());
            job.is_asymmetric = vnni->is_asymmetric;

            orchestrator->addWeightJob(gpu_ordinal, job);
        }
    }

    // Phase 4: Execute pipeline (pipelined H2D + GPU repack)
    orchestrator->load();

    auto* pool = orchestrator->getPool(gpu_ordinal);
    if (!pool)
    {
        LOG_ERROR("[MoEWeightService::GPU] Pool not found after load");
        return false;
    }

    // Phase 5: Create per-expert GEMM kernels from pool slots
    for (auto& grp : groups)
    {
        for (int idx = 0; idx < local_count; ++idx)
        {
            const int e = local_experts[idx];
            const auto& view = grp.views[e];
            const std::string slot_name = std::string(grp.label) + "_e" + std::to_string(e);

            auto slot = pool->getSlot(slot_name);
            if (!slot)
            {
                LOG_ERROR("[MoEWeightService::GPU] No slot for " << slot_name);
                return false;
            }

            auto* unpackable = dynamic_cast<IINT8Unpackable*>(view.get());
            const NativeVnniFormatInfo* vnni = unpackable->vnniFormatInfo();
            const int N = static_cast<int>(view->rows());
            const int K = static_cast<int>(view->cols());
            const uint32_t blocks_per_row = static_cast<uint32_t>(K / 32);

            std::shared_ptr<ITensorGemm> kernel;

#ifdef HAVE_CUDA
            if (ctx.device_id.is_cuda())
            {
                kernel = std::make_shared<llaminar2::cuda::CUDAQuantisedGemmKernel>(
                    N, K, gpu_ordinal,
                    slot->d_native_vnni_payload,
                    static_cast<uint16_t*>(slot->d_native_vnni_scales),
                    static_cast<uint16_t*>(slot->d_native_vnni_mins),
                    static_cast<uint32_t*>(slot->d_native_vnni_emins),
                    vnni->codebook_id, blocks_per_row,
                    orchestrator); // lifetime: keeps VRAM pool alive
            }
#endif
#ifdef HAVE_ROCM
            if (ctx.device_id.is_rocm())
            {
                kernel = std::make_shared<llaminar2::rocm::ROCmQuantisedGemmKernel>(
                    N, K, gpu_ordinal,
                    slot->d_native_vnni_payload,
                    slot->d_native_vnni_scales,
                    slot->d_native_vnni_mins,
                    slot->d_native_vnni_emins,
                    vnni->codebook_id, blocks_per_row,
                    orchestrator); // lifetime: keeps VRAM pool alive
            }
#endif

            if (!kernel)
            {
                LOG_ERROR("[MoEWeightService::GPU] Failed to create kernel for " << slot_name);
                return false;
            }

            grp.out_gemms[e] = kernel.get();
            ctx.moe_owned_kernels.push_back(std::move(kernel));
        }
    }

    // Phase 6: Release orchestrator staging resources (pinned ring).
    // The VRAM pool stays alive via the shared_ptr in each kernel's lifetime_owner_.
    orchestrator->finalize();

    const auto elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
    LOG_INFO("[MoEWeightService::GPU] " << (local_count * 3) << "/" << (num_experts * 3)
             << " expert GEMM engines prepared via GPU pipeline in "
             << std::fixed << std::setprecision(1) << elapsed_ms << " ms"
             << " (layer " << ctx.layer_idx << ", "
             << ctx.device_id.to_string() << ")");

    // Release mmap pages for raw expert weights (now repacked on GPU).
    {
        size_t released = 0;
        if (ctx.gate_exps && ctx.gate_exps->is_mmap_data())
            released += MmapRegion::adviseDontneedRange(ctx.gate_exps->raw_data(), ctx.gate_exps->size_bytes());
        if (ctx.up_exps && ctx.up_exps->is_mmap_data())
            released += MmapRegion::adviseDontneedRange(ctx.up_exps->raw_data(), ctx.up_exps->size_bytes());
        if (ctx.down_exps && ctx.down_exps->is_mmap_data())
            released += MmapRegion::adviseDontneedRange(ctx.down_exps->raw_data(), ctx.down_exps->size_bytes());
        if (released > 0)
            LOG_DEBUG("[MoEWeightService] Advised " << (released >> 20) << " MB of mmap pages DONTNEED after GPU repack");
    }

    return true;
}


// =========================================================================
// GPU-direct expert transfer (GPU↔GPU, same packed format)
// =========================================================================

bool MoEExpertWeightService::transferExpertsGPUDirect(
    const MoEWeightContext& src_ctx,
    MoEWeightContext& dst_ctx,
    const std::vector<int>& expert_ids,
    int layer_idx)
{
#ifdef HAVE_ROCM
    if (!src_ctx.device_id.is_rocm() || !dst_ctx.device_id.is_rocm()) {
        LOG_DEBUG("[MoEWeightService] GPU-direct transfer requires both contexts on ROCm "
                  << "(src=" << src_ctx.device_id.to_string()
                  << " dst=" << dst_ctx.device_id.to_string() << ")");
        return false;
    }

    using namespace llaminar2::rocm;

    // Get source batch packed weights from lifetime pointers
    auto* src_gate_batch = static_cast<MoEBatchPackedWeightsROCm*>(src_ctx.moe_packed_gate_lifetime.get());
    auto* src_up_batch = static_cast<MoEBatchPackedWeightsROCm*>(src_ctx.moe_packed_up_lifetime.get());
    auto* src_down_batch = static_cast<MoEBatchPackedWeightsROCm*>(src_ctx.moe_packed_down_lifetime.get());

    if (!src_gate_batch || !src_up_batch || !src_down_batch) {
        LOG_DEBUG("[MoEWeightService] GPU-direct transfer: source batch packed weights not available");
        return false;
    }

    auto* dst_gate_batch = static_cast<MoEBatchPackedWeightsROCm*>(dst_ctx.moe_packed_gate_lifetime.get());
    auto* dst_up_batch = static_cast<MoEBatchPackedWeightsROCm*>(dst_ctx.moe_packed_up_lifetime.get());
    auto* dst_down_batch = static_cast<MoEBatchPackedWeightsROCm*>(dst_ctx.moe_packed_down_lifetime.get());

    if (!dst_gate_batch || !dst_up_batch || !dst_down_batch) {
        LOG_DEBUG("[MoEWeightService] GPU-direct transfer: destination batch packed weights not available");
        return false;
    }

    const int src_rocm = src_ctx.device_id.rocm_ordinal();
    const int dst_rocm = dst_ctx.device_id.rocm_ordinal();

    auto transferOneBatch = [&](
        MoEBatchPackedWeightsROCm* src_batch,
        MoEBatchPackedWeightsROCm* dst_batch,
        std::shared_ptr<void>& dst_lifetime,
        std::vector<ITensorGemm*>& dst_gemms,
        const char* label) -> bool
    {
        const size_t vnni_per_expert = src_batch->vnni_bytes_per_expert;
        const size_t scales_per_expert = src_batch->scales_per_expert * sizeof(uint16_t);
        const size_t mins_per_expert = src_batch->mins_per_expert * sizeof(uint16_t);
        const size_t emins_per_expert = src_batch->emins_per_expert * sizeof(uint32_t);

        for (int expert_id : expert_ids) {
            auto src_ptrs = src_batch->getExpertDevicePointers(src_rocm, expert_id);
            auto dst_ptrs_rocm = dst_batch->getExpertDevicePointers(dst_rocm, expert_id);

            GPUExpertPointers src_gep, dst_gep;
            src_gep.d_vnni = src_ptrs.d_native_vnni;
            src_gep.d_scales = src_ptrs.d_native_scales;
            src_gep.d_mins = src_ptrs.d_native_mins;
            src_gep.d_emins = src_ptrs.d_native_emins;
            dst_gep.d_vnni = dst_ptrs_rocm.d_native_vnni;
            dst_gep.d_scales = dst_ptrs_rocm.d_native_scales;
            dst_gep.d_mins = dst_ptrs_rocm.d_native_mins;
            dst_gep.d_emins = dst_ptrs_rocm.d_native_emins;

            if (!GPUExpertTransfer::transferExpert(
                    src_gep, dst_gep,
                    src_ctx.device_id, dst_ctx.device_id,
                    vnni_per_expert, scales_per_expert, mins_per_expert, emins_per_expert,
                    nullptr)) {
                LOG_ERROR("[MoEWeightService] GPU-direct transfer failed for "
                          << label << " expert " << expert_id
                          << " layer " << layer_idx);
                return false;
            }

            // Create GEMM engine for the destination device pointing to transferred data
            auto kernel = std::make_shared<ROCmQuantisedGemmKernel>(
                dst_batch->rows_per_expert, dst_batch->K, dst_rocm,
                dst_ptrs_rocm.d_native_vnni,
                dst_ptrs_rocm.d_native_scales,
                dst_ptrs_rocm.d_native_mins,
                dst_ptrs_rocm.d_native_emins,
                dst_batch->codebook_id, static_cast<uint32_t>(dst_batch->blocks_per_row),
                dst_lifetime);
            dst_gemms[expert_id] = kernel.get();
            dst_ctx.moe_owned_kernels.push_back(std::move(kernel));
        }
        return true;
    };

    if (!transferOneBatch(src_gate_batch, dst_gate_batch,
                          dst_ctx.moe_packed_gate_lifetime,
                          dst_ctx.prepared_gate_gemm, "gate"))
        return false;
    if (!transferOneBatch(src_up_batch, dst_up_batch,
                          dst_ctx.moe_packed_up_lifetime,
                          dst_ctx.prepared_up_gemm, "up"))
        return false;
    if (!transferOneBatch(src_down_batch, dst_down_batch,
                          dst_ctx.moe_packed_down_lifetime,
                          dst_ctx.prepared_down_gemm, "down"))
        return false;

    LOG_INFO("[MoEWeightService] GPU-direct transferred " << expert_ids.size()
             << " experts (3 weight types) ROCm:" << src_rocm
             << " → ROCm:" << dst_rocm << " layer " << layer_idx);
    return true;

#else
    (void)src_ctx; (void)dst_ctx; (void)expert_ids; (void)layer_idx;
    LOG_DEBUG("[MoEWeightService] GPU-direct transfer requires ROCm");
    return false;
#endif
}

} // namespace llaminar2
