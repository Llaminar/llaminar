/**
 * @file MoEExpertWeightService.cpp
 * @brief Implementation of MoE expert weight lifecycle service.
 *
 * Contains all weight preparation, serialization, and rebalancing logic
 * extracted from MoEExpertComputeStage.cpp. See MoEExpertWeightService.h for API docs.
 */

#include "MoEExpertWeightService.h"
#include "ExpertWeightPayloadProvider.h"
#include "GPUExpertTransfer.h"
#include "../../tensors/Tensors.h"
#include "../../tensors/BlockStructures.h"
#include "../../kernels/KernelFactory.h"
#include "../../kernels/PackedWeightsSerialization.h"
#include "../../kernels/cpu/native_vnni/CPUPackedWeights.h"
#include "../../kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
#include "../../loaders/MmapRegion.h"
#include "../../loaders/ExpertGemmRegistry.h"
#include "../../loaders/gpu_pipeline/LoadOrchestrator.h"
#include "../../backends/BackendManager.h"
#include "../../utils/Assertions.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/Logger.h"
#include "../../utils/OpenMPUtils.h"
#include "../../loaders/PreparedWeightStore.h"

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
#include <cstring>
#include <iomanip>
#include <optional>
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
    // expensive (VNNI repacking). Newly-acquired experts get engines from
    // serialized payloads via registerAndPrepareNewExperts() after rebalancing.
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
    // key collisions.  The heavy VNNI interleave runs lock-free.
    // Phase D: prepareExpertGemmLocal returns shared_ptr without global registry.
    std::atomic<bool> error_flag{false};

    // Per-expert engine storage for parallel assignment (avoids push_back race)
    std::vector<std::shared_ptr<ITensorGemm>> local_gate_engines(prep_count);
    std::vector<std::shared_ptr<ITensorGemm>> local_up_engines(prep_count);
    std::vector<std::shared_ptr<ITensorGemm>> local_down_engines(prep_count);

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

        auto gate_engine = KernelFactory::prepareExpertGemmLocal(
            ctx.expert_gate_views[e].get(), ctx.device_id);
        auto up_engine = KernelFactory::prepareExpertGemmLocal(
            ctx.expert_up_views[e].get(), ctx.device_id);
        auto down_engine = KernelFactory::prepareExpertGemmLocal(
            ctx.expert_down_views[e].get(), ctx.device_id);

        if (!gate_engine || !up_engine || !down_engine)
        {
            LOG_ERROR("[MoEWeightService] Failed to prepare GEMM weights for expert " << e);
            error_flag.store(true, std::memory_order_relaxed);
            continue;
        }

        ctx.prepared_gate_gemm[e] = gate_engine.get();
        ctx.prepared_up_gemm[e] = up_engine.get();
        ctx.prepared_down_gemm[e] = down_engine.get();

        // Store in per-index slot (no contention — each idx is unique)
        local_gate_engines[idx] = std::move(gate_engine);
        local_up_engines[idx] = std::move(up_engine);
        local_down_engines[idx] = std::move(down_engine);
    }

    // Transfer engine lifetimes to ctx after parallel region (single-threaded)
    if (!error_flag.load())
    {
        ctx.moe_owned_kernels.reserve(ctx.moe_owned_kernels.size() + prep_count * 3);
        for (int idx = 0; idx < prep_count; ++idx)
        {
            if (local_gate_engines[idx])
                ctx.moe_owned_kernels.push_back(std::move(local_gate_engines[idx]));
            if (local_up_engines[idx])
                ctx.moe_owned_kernels.push_back(std::move(local_up_engines[idx]));
            if (local_down_engines[idx])
                ctx.moe_owned_kernels.push_back(std::move(local_down_engines[idx]));
        }
    }

    if (error_flag.load())
    {
        return false;
    }

    LOG_DEBUG("[MoEWeightService] All " << (prep_count * 3) << " expert GEMM engines prepared (CPU/KernelFactory path)");

    // ── Phase B: Dual-path registration into PreparedWeightStore ─────────
    if (ctx.prepared_store)
    {
        // Register three slabs (gate, up, down) for this layer.
        // Use the 3D parent tensor identity for source_identity when available.
        auto register_slab = [&](WeightRole role, const std::vector<ITensorGemm*>& engines,
                                 const std::vector<std::shared_ptr<TensorBase>>& views,
                                 TensorBase* parent_3d) -> ExpertSlabRef
        {
            ExpertSlabDescriptor desc;
            desc.layer_idx = ctx.layer_idx;
            desc.role = role;
            desc.device = ctx.device_id;
            desc.num_experts = ctx.num_experts;
            desc.local_expert_start = ctx.local_expert_start;
            desc.local_expert_count = (ctx.local_expert_count < 0) ? ctx.num_experts : ctx.local_expert_count;
            desc.rows_per_expert = ctx.expert_intermediate;
            desc.cols_per_expert = ctx.d_model;
            // source_identity left default — TensorBase doesn't expose WeightIdentity.
            // Phase C will wire this through WeightManager lookup.
            (void)parent_3d;

            auto slab_ref = ctx.prepared_store->registerExpertSlab(desc);

            // Build arrivals for all prepared experts
            std::vector<ExpertArrival> arrivals;
            arrivals.reserve(experts_to_prep.size());
            for (int e : experts_to_prep)
            {
                if (!engines[e]) continue;
                ExpertArrival arrival;
                arrival.expert_id = e;
                arrival.engine = engines[e];
                // Find engine_lifetime from moe_owned_kernels
                for (const auto& owned : ctx.moe_owned_kernels)
                {
                    if (owned.get() == engines[e])
                    {
                        arrival.engine_lifetime = owned;
                        break;
                    }
                }
                arrival.view_lifetime = (e < static_cast<int>(views.size())) ? views[e] : nullptr;
                arrival.derivation = WeightDerivationKind::ExpertSlice;
                arrivals.push_back(std::move(arrival));
            }

            ctx.prepared_store->registerArrivedExperts(slab_ref, arrivals);
            return slab_ref;
        };

        ctx.gate_slab_ref = register_slab(WeightRole::MoEExpertGate, ctx.prepared_gate_gemm,
                                          ctx.expert_gate_views, ctx.gate_exps);
        ctx.up_slab_ref = register_slab(WeightRole::MoEExpertUp, ctx.prepared_up_gemm,
                                        ctx.expert_up_views, ctx.up_exps);
        ctx.down_slab_ref = register_slab(WeightRole::MoEExpertDown, ctx.prepared_down_gemm,
                                          ctx.expert_down_views, ctx.down_exps);

        LOG_DEBUG("[MoEWeightService] Phase B: Registered " << (prep_count * 3)
                  << " expert engines in PreparedWeightStore ("
                  << ctx.prepared_store->expertSlabCount() << " slabs total, "
                  << ctx.prepared_store->totalPopulatedExperts() << " populated experts)");
    }

    // NUMA audit: verify packed weights landed on the correct NUMA node.
    auditExpertNUMA(experts_to_prep, ctx.prepared_gate_gemm,
                    "initial_pack", ctx.layer_idx);

    // Mark experts as prepared in the payload provider (enables host data release)
    if (ctx.payload_provider)
    {
        for (int e : experts_to_prep)
            ctx.payload_provider->markExpertPrepared(ctx.layer_idx, e);
    }

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

    auto serialize_proj = [](ITensorGemm* engine, const std::shared_ptr<TensorBase>& view) -> std::vector<uint8_t> {
        if (!engine || !engine->hasWeights()) return {};
        auto packed = engine->cloneWeights();
        if (!packed) return {};

        if (packed->format() == PackedWeightsFormat::CPU_NATIVE_VNNI &&
            dynamic_cast<CPUPackedWeightsWithNativeBlocks*>(packed.get()) == nullptr &&
            view && view->raw_data())
        {
            auto* cpu_packed = dynamic_cast<CPUPackedWeights*>(packed.get());
            if (cpu_packed)
            {
                const size_t raw_bytes = quantizedViewRawBytes(*view);
                if (raw_bytes > 0)
                {
                    const auto* raw = static_cast<const uint8_t*>(view->raw_data());
                    std::vector<uint8_t> native_blocks(raw, raw + raw_bytes);
                    const size_t blocks = static_cast<size_t>(cpu_packed->packed().N)
                                      * static_cast<size_t>(std::max(1, cpu_packed->packed().blocks_per_row));
                    const size_t native_block_size = blocks > 0 ? raw_bytes / blocks : raw_bytes;
                    packed = std::make_unique<CPUPackedWeightsWithNativeBlocks>(
                        CPUNativeVNNIPackedWeights(cpu_packed->packed()),
                        std::move(native_blocks),
                        native_block_size);
                }
            }
        }

        return packed_weights_serialization::serialize(*packed);
    };

    blobs.gate = serialize_proj(ctx.prepared_gate_gemm[expert_id], ctx.expert_gate_views[expert_id]);
    blobs.up   = serialize_proj(ctx.prepared_up_gemm[expert_id],   ctx.expert_up_views[expert_id]);
    blobs.down = serialize_proj(ctx.prepared_down_gemm[expert_id],  ctx.expert_down_views[expert_id]);

    return blobs;
}

// =========================================================================
// Phased rebalance API
// =========================================================================

std::vector<const TensorBase*> MoEExpertWeightService::releaseDepartedExperts(
    MoEWeightContext& ctx, const std::vector<bool>& new_mask)
{
    std::vector<const TensorBase*> evict_tensors;
    std::vector<int> departed_ids;

    for (int e = 0; e < ctx.num_experts; ++e) {
        if (!new_mask[e] && ctx.prepared_gate_gemm[e]) {
            departed_ids.push_back(e);

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

    // Mirror departures to ExpertGemmRegistry (keeps registry in sync with stage state)
    if (ctx.expert_registry && !departed_ids.empty())
    {
        for (int e : departed_ids)
        {
            ctx.expert_registry->removeEngine(ctx.device_id, ctx.layer_idx, e, ExpertGemmRegistry::WeightRole::GATE);
            ctx.expert_registry->removeEngine(ctx.device_id, ctx.layer_idx, e, ExpertGemmRegistry::WeightRole::UP);
            ctx.expert_registry->removeEngine(ctx.device_id, ctx.layer_idx, e, ExpertGemmRegistry::WeightRole::DOWN);
        }
        LOG_DEBUG("[MoEWeightService] Removed " << departed_ids.size()
                  << " departed experts from ExpertGemmRegistry (layer " << ctx.layer_idx << ")");
    }

    // Phase C: Notify PreparedWeightStore of departed experts
    if (ctx.prepared_store && !departed_ids.empty())
    {
        if (ctx.gate_slab_ref.has_value())
            ctx.prepared_store->releaseDepartedExperts(*ctx.gate_slab_ref, departed_ids);
        if (ctx.up_slab_ref.has_value())
            ctx.prepared_store->releaseDepartedExperts(*ctx.up_slab_ref, departed_ids);
        if (ctx.down_slab_ref.has_value())
            ctx.prepared_store->releaseDepartedExperts(*ctx.down_slab_ref, departed_ids);

        LOG_DEBUG("[MoEWeightService] Phase C: Released " << departed_ids.size()
                  << " departed experts from PreparedWeightStore");
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

    // GPU path: prefer transferred packed/source-domain payloads, converting
    // CPU-native-with-blocks to GPU-repacked layout via the device pipeline.
    // Raw GGUF data is only a fallback for initial materialization before host
    // source bytes are released.
    if (ctx.device_id.is_gpu()) {
        return registerAndPrepareNewExpertsGPU(ctx, new_experts, received_weights);
    }

    // CPU path: deserialize transferred weights, then KernelFactory pack
    auto t_start = std::chrono::high_resolution_clock::now();
    int transferred_count = 0;
    std::atomic<bool> error_flag{false};
    const int count = static_cast<int>(new_experts.size());

    // Prepare engines: use transferred blobs directly (fast deserialization, no repack),
    // or fall back to prepareExpertGemmLocal (full VNNI repack from raw tensor data).
    for (int idx = 0; idx < count; ++idx) {
        if (error_flag.load(std::memory_order_relaxed)) break;
        const int e = new_experts[idx];

        if (!ctx.expert_gate_views[e] || !ctx.expert_up_views[e] ||
            !ctx.expert_down_views[e]) {
            LOG_ERROR("[MoEWeightService] Null view for new expert " << e);
            error_flag.store(true, std::memory_order_relaxed);
            continue;
        }

        std::shared_ptr<ITensorGemm> gate_engine, up_engine, down_engine;

        // Fast path: create kernels directly from pre-packed transferred blobs
        const ExpertWeightBlobs* blobs = nullptr;
        if (received_weights) {
            auto it = received_weights->find(e);
            if (it != received_weights->end() && !it->second.empty())
                blobs = &it->second;
        }

        if (blobs) {
            gate_engine = KernelFactory::createExpertGemmFromTransferBlob(blobs->gate);
            up_engine   = KernelFactory::createExpertGemmFromTransferBlob(blobs->up);
            down_engine = KernelFactory::createExpertGemmFromTransferBlob(blobs->down);

            if (gate_engine && up_engine && down_engine) {
                ++transferred_count;
            } else {
                // Fallback to full repack if any blob deserialization failed
                LOG_WARN("[MoEWeightService] Blob deserialization failed for expert " << e
                         << ", falling back to full repack");
                gate_engine.reset();
                up_engine.reset();
                down_engine.reset();
            }
        }

        // Slow path: full VNNI repack from raw tensor data
        if (!gate_engine) {
            gate_engine = KernelFactory::prepareExpertGemmLocal(
                ctx.expert_gate_views[e].get(), ctx.device_id);
        }
        if (!up_engine) {
            up_engine = KernelFactory::prepareExpertGemmLocal(
                ctx.expert_up_views[e].get(), ctx.device_id);
        }
        if (!down_engine) {
            down_engine = KernelFactory::prepareExpertGemmLocal(
                ctx.expert_down_views[e].get(), ctx.device_id);
        }

        if (!gate_engine || !up_engine || !down_engine) {
            LOG_ERROR("[MoEWeightService] Failed GEMM weights for new expert " << e);
            error_flag.store(true, std::memory_order_relaxed);
            continue;
        }

        ctx.prepared_gate_gemm[e] = gate_engine.get();
        ctx.prepared_up_gemm[e] = up_engine.get();
        ctx.prepared_down_gemm[e] = down_engine.get();

        ctx.moe_owned_kernels.push_back(std::move(gate_engine));
        ctx.moe_owned_kernels.push_back(std::move(up_engine));
        ctx.moe_owned_kernels.push_back(std::move(down_engine));
    }

    if (error_flag.load()) return false;

    // Mark experts as prepared in the payload provider
    if (ctx.payload_provider)
    {
        for (int e : new_experts)
            ctx.payload_provider->markExpertPrepared(ctx.layer_idx, e);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double prep_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    LOG_DEBUG("[MoEWeightService] Engine prep for " << new_experts.size()
              << " experts (" << transferred_count << " transferred): "
              << std::fixed << std::setprecision(1) << prep_ms << " ms");

    auditExpertNUMA(new_experts, ctx.prepared_gate_gemm,
                    (transferred_count > 0 ? "rebalance_transferred" : "rebalance_repacked"),
                    ctx.layer_idx);

    // Phase C: Register new arrivals in PreparedWeightStore using cached slab refs
    if (ctx.prepared_store && !new_experts.empty())
    {
        auto register_rebalance_arrivals = [&](const std::optional<ExpertSlabRef>& slab_ref,
                                               const std::vector<ITensorGemm*>& engines,
                                               const std::vector<std::shared_ptr<TensorBase>>& views)
        {
            if (!slab_ref.has_value()) return;

            std::vector<ExpertArrival> arrivals;
            for (int e : new_experts)
            {
                if (e < 0 || e >= static_cast<int>(engines.size()) || !engines[e]) continue;
                ExpertArrival arrival;
                arrival.expert_id = e;
                arrival.engine = engines[e];
                // Find engine_lifetime from moe_owned_kernels
                for (const auto& owned : ctx.moe_owned_kernels)
                {
                    if (owned.get() == engines[e])
                    {
                        arrival.engine_lifetime = owned;
                        break;
                    }
                }
                arrival.view_lifetime = (e < static_cast<int>(views.size())) ? views[e] : nullptr;
                arrival.derivation = WeightDerivationKind::RebalancedExpertReplica;
                arrivals.push_back(std::move(arrival));
            }

            if (!arrivals.empty())
                ctx.prepared_store->registerArrivedExperts(*slab_ref, arrivals);
        };

        register_rebalance_arrivals(ctx.gate_slab_ref, ctx.prepared_gate_gemm, ctx.expert_gate_views);
        register_rebalance_arrivals(ctx.up_slab_ref, ctx.prepared_up_gemm, ctx.expert_up_views);
        register_rebalance_arrivals(ctx.down_slab_ref, ctx.prepared_down_gemm, ctx.expert_down_views);

        LOG_DEBUG("[MoEWeightService] Phase C rebalance: Registered "
                  << new_experts.size() * 3 << " new expert engines in PreparedWeightStore"
                  << " (using cached slab refs)");
    }

    return true;
}

// =========================================================================
// GPU rebalance path (LoadOrchestrator: raw H2D + GPU repack for new experts)
// =========================================================================

bool MoEExpertWeightService::registerAndPrepareNewExpertsGPU(
    MoEWeightContext& ctx,
    const std::vector<int>& new_experts,
    const std::unordered_map<int, ExpertWeightBlobs>* received_weights)
{
    using Clock = std::chrono::high_resolution_clock;
    const auto t_start = Clock::now();

    const int gpu_ordinal = ctx.device_id.is_cuda()
                                ? ctx.device_id.cuda_ordinal()
                                : ctx.device_id.rocm_ordinal();
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

    struct TransferSource {
        std::unique_ptr<IPackedWeights> packed;
        const cpu::native_vnni::CPUPackedWeightsWithNativeBlocks* native = nullptr;
        RepackFormat format = RepackFormat::Q4_0;
    };
    std::vector<TransferSource> transfer_sources;
    std::unordered_map<int, ExpertWeightBlobs> provider_payload_cache;

    auto blobsForExpert = [&](int expert_id) -> const ExpertWeightBlobs* {
        if (received_weights)
        {
            auto it = received_weights->find(expert_id);
            if (it != received_weights->end() && !it->second.empty())
                return &it->second;
        }

        auto cached = provider_payload_cache.find(expert_id);
        if (cached != provider_payload_cache.end())
            return &cached->second;

        if (ctx.payload_provider)
        {
            auto payload = ctx.payload_provider->payloadFor(ctx.layer_idx, expert_id);
            if (payload && !payload->empty())
            {
                auto [it, inserted] = provider_payload_cache.emplace(expert_id, std::move(*payload));
                (void)inserted;
                return &it->second;
            }
        }

        return nullptr;
    };

    auto blobFor = [&](int expert_id, const char* label) -> const std::vector<uint8_t>* {
        const ExpertWeightBlobs* blobs = blobsForExpert(expert_id);
        if (!blobs) return nullptr;
        if (std::strcmp(label, "gate") == 0) return &blobs->gate;
        if (std::strcmp(label, "up") == 0) return &blobs->up;
        return &blobs->down;
    };

    auto hasCompleteTransferredBlobs = [&](int expert_id) {
        for (auto& grp : groups)
        {
            auto* blob = blobFor(expert_id, grp.label);
            if (!blob || blob->empty())
                return false;
        }
        return true;
    };

    auto makeTransferSource = [&](const std::vector<uint8_t>& blob,
                                  int expert_id,
                                  const char* label) -> std::optional<TransferSource> {
        if (blob.empty()) return std::nullopt;
        auto packed = packed_weights_serialization::deserialize(blob.data(), blob.size());
        if (!packed)
        {
            LOG_ERROR("[MoEWeightService::GPU-rebalance] Failed to deserialize transferred packed weights for expert "
                      << expert_id << " " << label << " on " << ctx.device_id.to_string());
            return std::nullopt;
        }

        auto* native = dynamic_cast<cpu::native_vnni::CPUPackedWeightsWithNativeBlocks*>(packed.get());
        if (!native)
        {
            LOG_ERROR("[MoEWeightService::GPU-rebalance] Transferred expert " << expert_id << " " << label
                      << " lacks native quantized blocks; CPU-packed-only conversion to GPU is not available yet");
            return std::nullopt;
        }

        const auto& cpu_packed = native->packed();
        auto repack_fmt = codebookIdToRepackFormat(cpu_packed.codebook_id, cpu_packed.is_superblock);
        if (!repack_fmt)
        {
            LOG_ERROR("[MoEWeightService::GPU-rebalance] Unsupported transferred packed format for expert "
                      << expert_id << " " << label
                      << " (codebook=" << static_cast<int>(cpu_packed.codebook_id)
                      << ", superblock=" << cpu_packed.is_superblock << ")");
            return std::nullopt;
        }

        TransferSource source;
        source.native = native;
        source.format = *repack_fmt;
        source.packed = std::move(packed);
        return source;
    };

    std::vector<int> experts_to_load;
    experts_to_load.reserve(new_experts.size());
    auto cached_engine_for = [&](const std::optional<ExpertSlabRef>& slab_ref, int expert_id) -> ITensorGemm*
    {
        if (!ctx.prepared_store || !slab_ref.has_value())
            return nullptr;
        return ctx.prepared_store->expertGemmKernel(*slab_ref, expert_id);
    };
    for (int e : new_experts)
    {
        ITensorGemm* cached_gate = cached_engine_for(ctx.gate_slab_ref, e);
        ITensorGemm* cached_up = cached_engine_for(ctx.up_slab_ref, e);
        ITensorGemm* cached_down = cached_engine_for(ctx.down_slab_ref, e);
        if (cached_gate && cached_up && cached_down)
        {
            groups[0].out_gemms[e] = cached_gate;
            groups[1].out_gemms[e] = cached_up;
            groups[2].out_gemms[e] = cached_down;
            continue;
        }

        if (!hasCompleteTransferredBlobs(e))
        {
            LOG_ERROR("[MoEWeightService::GPU-rebalance] Expert " << e
                      << " requires transferred/provider blobs for all gate/up/down weights on "
                      << ctx.device_id.to_string() << " (layer " << ctx.layer_idx
                      << "). Raw GGUF fallback is not allowed during GPU rebalance after host release.");
            return false;
        }

        experts_to_load.push_back(e);
    }

    const int count = static_cast<int>(experts_to_load.size());
    if (count == 0)
    {
        LOG_DEBUG("[MoEWeightService::GPU-rebalance] All " << new_experts.size()
                  << " requested experts already had prepared GPU handles on "
                  << ctx.device_id.to_string() << " (layer " << ctx.layer_idx << ")");
        return true;
    }

    // Phase 1: Plan weights for new experts
    for (auto& grp : groups)
    {
        for (int idx = 0; idx < count; ++idx)
        {
            const int e = experts_to_load[idx];
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

            auto* blob = blobFor(e, grp.label);
            if (!blob || blob->empty())
            {
                LOG_ERROR("[MoEWeightService::GPU-rebalance] Missing required transferred/provider blob for expert "
                          << e << " " << grp.label << " on " << ctx.device_id.to_string()
                          << " (layer " << ctx.layer_idx << ")");
                return false;
            }

            auto source = makeTransferSource(*blob, e, grp.label);
            if (!source) return false;
            const size_t source_bytes = source->native->nativeBlocks().size();
            transfer_sources.push_back(std::move(*source));

            orchestrator->planWeight(gpu_ordinal, slot_name, N, K,
                                     vnni->payload_bytes, vnni->is_asymmetric,
                                     vnni->has_emins, source_bytes);
            max_raw_bytes = std::max(max_raw_bytes, source_bytes);
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
    size_t transfer_source_idx = 0;
    for (auto& grp : groups)
    {
        for (int idx = 0; idx < count; ++idx)
        {
            const int e = experts_to_load[idx];
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
            job.format = *repack_fmt;
            job.N = static_cast<int>(view->rows());
            job.K = static_cast<int>(view->cols());
            job.is_asymmetric = vnni->is_asymmetric;

            if (transfer_source_idx >= transfer_sources.size())
            {
                LOG_ERROR("[MoEWeightService::GPU-rebalance] Internal transfer source accounting mismatch");
                return false;
            }
            const auto& source = transfer_sources[transfer_source_idx++];
            job.host_raw_data = source.native->nativeBlocks().data();
            job.raw_bytes = source.native->nativeBlocks().size();
            job.format = source.format;

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
            const int e = experts_to_load[idx];
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

    // Phase 7: Mark experts as prepared in the payload provider
    if (ctx.payload_provider)
    {
        for (int e : experts_to_load)
            ctx.payload_provider->markExpertPrepared(ctx.layer_idx, e);
    }

    // Mirror arrivals to ExpertGemmRegistry (keeps registry in sync with stage state)
    if (ctx.expert_registry)
    {
        for (int e : experts_to_load)
        {
            auto find_ownership = [&](ITensorGemm* raw) -> std::shared_ptr<ITensorGemm> {
                for (const auto& owned : ctx.moe_owned_kernels)
                    if (owned.get() == raw) return owned;
                return nullptr;
            };

            if (ctx.prepared_gate_gemm[e])
                ctx.expert_registry->replaceEngine(ctx.device_id, ctx.layer_idx, e,
                    ExpertGemmRegistry::WeightRole::GATE,
                    ctx.prepared_gate_gemm[e], find_ownership(ctx.prepared_gate_gemm[e]));
            if (ctx.prepared_up_gemm[e])
                ctx.expert_registry->replaceEngine(ctx.device_id, ctx.layer_idx, e,
                    ExpertGemmRegistry::WeightRole::UP,
                    ctx.prepared_up_gemm[e], find_ownership(ctx.prepared_up_gemm[e]));
            if (ctx.prepared_down_gemm[e])
                ctx.expert_registry->replaceEngine(ctx.device_id, ctx.layer_idx, e,
                    ExpertGemmRegistry::WeightRole::DOWN,
                    ctx.prepared_down_gemm[e], find_ownership(ctx.prepared_down_gemm[e]));
        }
        LOG_DEBUG("[MoEWeightService::GPU-rebalance] Registered " << experts_to_load.size()
                  << " arrived experts in ExpertGemmRegistry (layer " << ctx.layer_idx << ")");
    }

    const auto elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
    LOG_INFO("[MoEWeightService::GPU-rebalance] " << (count * 3) << " expert GEMM engines "
             << "prepared via GPU pipeline in "
             << std::fixed << std::setprecision(1) << elapsed_ms << " ms"
             << " (layer " << ctx.layer_idx << ", "
             << ctx.device_id.to_string() << ")");

    // Phase C: Register GPU rebalanced experts in PreparedWeightStore
    if (ctx.prepared_store)
    {
        auto register_gpu_rebalance = [&](const std::optional<ExpertSlabRef>& slab_ref,
                                          const std::vector<ITensorGemm*>& engines)
        {
            if (!slab_ref.has_value()) return;

            std::vector<ExpertArrival> arrivals;
            for (int e : experts_to_load)
            {
                if (!engines[e]) continue;
                ExpertArrival arrival;
                arrival.expert_id = e;
                arrival.engine = engines[e];
                // Find shared_ptr lifetime
                for (const auto& owned : ctx.moe_owned_kernels)
                {
                    if (owned.get() == engines[e])
                    {
                        arrival.engine_lifetime = owned;
                        break;
                    }
                }
                arrival.derivation = WeightDerivationKind::RebalancedExpertReplica;
                arrivals.push_back(std::move(arrival));
            }

            if (!arrivals.empty())
                ctx.prepared_store->registerArrivedExperts(*slab_ref, arrivals);
        };

        register_gpu_rebalance(ctx.gate_slab_ref, ctx.prepared_gate_gemm);
        register_gpu_rebalance(ctx.up_slab_ref, ctx.prepared_up_gemm);
        register_gpu_rebalance(ctx.down_slab_ref, ctx.prepared_down_gemm);

        LOG_DEBUG("[MoEWeightService::GPU-rebalance] Phase C: Registered "
                  << (count * 3) << " rebalanced expert engines in PreparedWeightStore");
    }

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
        LOG_DEBUG("[MoEWeightService::GPU] No local experts to prepare on "
                  << ctx.device_id.to_string() << " (layer " << ctx.layer_idx << ")");
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
            if (!job.host_raw_data)
            {
                LOG_ERROR("[MoEWeightService::GPU] Expert " << e << " "
                          << grp.label << " has null raw host data for " << job.raw_bytes
                          << " bytes on " << ctx.device_id.to_string()
                          << " (layer " << ctx.layer_idx << ")");
                return false;
            }
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

    // Phase 7: Mark experts as prepared in the payload provider
    if (ctx.payload_provider)
    {
        for (int e : local_experts)
            ctx.payload_provider->markExpertPrepared(ctx.layer_idx, e);
    }

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

    // ── Phase C: Register GPU expert engines in PreparedWeightStore ──────
    if (ctx.prepared_store)
    {
        auto register_gpu_slab = [&](WeightRole role,
                                     const std::vector<ITensorGemm*>& engines) -> ExpertSlabRef
        {
            ExpertSlabDescriptor desc;
            desc.layer_idx = ctx.layer_idx;
            desc.role = role;
            desc.device = ctx.device_id;
            desc.num_experts = ctx.num_experts;
            desc.local_expert_start = ctx.local_expert_start;
            desc.local_expert_count = (ctx.local_expert_count < 0) ? ctx.num_experts : ctx.local_expert_count;
            desc.rows_per_expert = ctx.expert_intermediate;
            desc.cols_per_expert = ctx.d_model;

            auto slab_ref = ctx.prepared_store->registerExpertSlab(desc);

            std::vector<ExpertArrival> arrivals;
            arrivals.reserve(local_experts.size());
            for (int e : local_experts)
            {
                if (!engines[e]) continue;
                ExpertArrival arrival;
                arrival.expert_id = e;
                arrival.engine = engines[e];
                // Find the shared_ptr lifetime in moe_owned_kernels
                for (const auto& owned : ctx.moe_owned_kernels)
                {
                    if (owned.get() == engines[e])
                    {
                        arrival.engine_lifetime = owned;
                        break;
                    }
                }
                arrival.derivation = WeightDerivationKind::ExpertSlice;
                arrivals.push_back(std::move(arrival));
            }

            ctx.prepared_store->registerArrivedExperts(slab_ref, arrivals);
            return slab_ref;
        };

        ctx.gate_slab_ref = register_gpu_slab(WeightRole::MoEExpertGate, ctx.prepared_gate_gemm);
        ctx.up_slab_ref = register_gpu_slab(WeightRole::MoEExpertUp, ctx.prepared_up_gemm);
        ctx.down_slab_ref = register_gpu_slab(WeightRole::MoEExpertDown, ctx.prepared_down_gemm);

        LOG_DEBUG("[MoEWeightService::GPU] Phase C: Registered " << (local_count * 3)
                  << " GPU expert engines in PreparedWeightStore ("
                  << ctx.prepared_store->expertSlabCount() << " slabs, "
                  << ctx.prepared_store->totalPopulatedExperts() << " populated)");
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
