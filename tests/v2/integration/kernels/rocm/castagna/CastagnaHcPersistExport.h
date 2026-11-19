/**
 * @file CastagnaHcPersistExport.h
 * @brief Diagnostic adapter for the unchanged pinned Castagna full T1 POST kernel.
 *
 * This is an external-comparator ABI, not a Llaminar execution mode. The caller
 * owns original encoded source buffers, stream, captured lifetime and exclusive
 * use of one module/global-state instance per physical GPU. No allocation or
 * transfer is hidden in launch. The full HC prefix is always included.
 */
#pragma once
#include <hip/hip_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace qfn_castagna_reference {
/** @brief One exact contiguous original-source interval, before any encoding/preparation. */
struct SourceShardWindow { std::uint64_t first, count; };
/**
 * @brief Equal-weight cumulative split from pinned llama-model.cpp:937-954.
 * @param extent Original global axis extent. @param granularity Exact source policy grain.
 * @param degree Number of original participants. @param participant Physical participant index.
 * @param rotation Original tensor placement rotation. @return Source interval, including valid empty intervals.
 * @details Each nonfinal cumulative boundary is floor(extent*j/degree), then
 * rounded down to granularity; the final participant receives the remainder.
 * The caller resolves the source policy grain (routed lcm(blocksize, configured
 * FFN grain), shared lcm(blocksize,64)). This helper does not admit a kernel or
 * implement weighted placement. Invalid degree/grain/index returns an empty range.
 */
constexpr SourceShardWindow equalWeightSourceShard(std::uint64_t extent,std::uint64_t granularity,
    std::uint32_t degree,std::uint32_t participant,std::uint32_t rotation=0) noexcept
{
    if(!degree||!granularity||participant>=degree)return {0,0};
    const std::uint64_t logical=(std::uint64_t(participant)+degree-(rotation%degree))%degree;
    const auto boundary=[=](std::uint64_t j) constexpr {
        if(j==degree)return extent;
        const std::uint64_t raw=(extent/degree)*j+((extent%degree)*j)/degree;
        return raw-raw%granularity;
    };
    const auto first=boundary(logical),last=boundary(logical+1);
    return {first,last-first};
}
/** @brief Original supported routed gate/up source families; never native K32 planes. */
enum class GateSource { Q4_K, Q5_K, Q8_0 };
/** @brief Original routed down source families. */
enum class DownSource { Q5_1, Q8_0 };
/** @brief Exact original source streams consumed by POST=true, STASH/INJ/AR4=false. */
struct PrefixInputs {
    const float* residual; ///< FP32[4][2560].
    const float* previous_output; ///< FP32[2560], preceding sublayer result, not precombined.
    const float* previous_gate_logits; ///< FP32[4], previous POST injection logits.
    const float* norm_weights; ///< FP32[4][2560].
    const char* hc_down_q8_0; ///< Original Q8_0[320][10240], contiguous encoded rows.
    const char* hc_up_q8_0; ///< Original Q8_0[10240][320], contiguous encoded rows.
    float epsilon, down_scale, pre_scale, post_scale1, post_scale2;
};
/** @brief Original contiguous raw GGUF matrices, with no reconstructed/requantized values. */
struct FFNInputs {
    const float* router; ///< FP32[512][2560].
    const char* shared_gate_q8_0; ///< Original Q8_0[S][2560].
    const char* shared_up_q8_0; ///< Original Q8_0[S][2560].
    const char* shared_down_q8_0; ///< Original Q8_0[2560][S].
    const float* shared_gate; ///< FP32[2560].
    const char* expert_gate; ///< Original selected GateSource[512][F][2560].
    const char* expert_up; ///< Same original source type and shape as gate.
    const char* expert_down; ///< Original selected DownSource[512][2560][F].
    GateSource gate_source;
    DownSource down_source;
    unsigned routed_columns, shared_columns, top_k;
    float selected_sum_clamp; ///< Original graph clamp argument before selected-weight division.
};
/** @brief Public full-kernel result planes; internal globals remain in the original module. */
struct Outputs {
    float* post_residual; ///< FP32[4][2560].
    float* normalized; ///< FP32[4][2560].
    float* mixed; ///< FP32[2560], actual HC prefix output/router input.
    float* router_logits; ///< FP32[512].
    float* ffn; ///< FP32[2560], before explicit TP sum.
    unsigned long long* phase_timestamps; ///< Optional original HTS[60][16]; null in canonical timing.
};
/** @brief Semantic request converted fieldwise into original private source structs. */
struct Request { PrefixInputs prefix; FFNInputs ffn; Outputs outputs; };
/** @brief Cold observed original launch admission; caller seals this with its request/graph. */
struct Admission {
    const void* function;
    int device;
    unsigned grid,threads,active_blocks_per_cu,compute_units;
    hipFuncAttributes resources;
    Request request; ///< Exact cold semantic identity; launch rejects any changed field.
};
/** @brief One original DSO-owned global byte range, borrowed only by this comparator. */
struct GlobalRegion { const char* name; void* data; std::size_t bytes; };
inline constexpr unsigned global_region_count=11;
/** @brief Actual original global-state addresses; no ownership or allocation is created. */
struct GlobalRegions { GlobalRegion regions[global_region_count]; std::size_t total_bytes; };
}

/** @brief Exact full T1 POST entry address, for native node/resource authentication. */
extern "C" const void* qfn_castagna_t1_post_ffn_kernel() noexcept;
/**
 * @brief Validate original source geometry and actual gfx906 residency before capture.
 * @param request Exact original source request. @param device Actual HIP ordinal.
 * @param admission Receives original function/resources and fixed original geometry.
 * @return HIP status; unsupported geometry/hardware is rejected, never padded/substituted.
 */
extern "C" hipError_t qfn_castagna_t1_post_ffn_prepare(
    const qfn_castagna_reference::Request* request,int device,qfn_castagna_reference::Admission* admission) noexcept;
/**
 * @brief Launch the unchanged original full HC+router/shared+routed FFN kernel.
 * @param request Same retained request used during preparation.
 * @param admission Same observed original entry/geometry. @param stream Exact nondefault stream.
 * @return Immediate HIP launch status. Completion is owned by the caller's native event.
 * @pre One exclusive comparator global-state lease; stream/context and every pointer
 * retained through completion. No external peer push or early projections are enabled.
 */
extern "C" hipError_t qfn_castagna_t1_post_ffn_launch(const qfn_castagna_reference::Request* request,
    const qfn_castagna_reference::Admission* admission,hipStream_t stream) noexcept;
/**
 * @brief Observe original global ranges cold, before request initialization/capture.
 * @param regions Receives exact module-owned pointers/bytes on the current device.
 * @return HIP status. The caller orders cold/reset writes using its transfer authority;
 * no global is reset by launch and no hot device/address queries are performed there.
 */
extern "C" hipError_t qfn_castagna_t1_post_ffn_globals(qfn_castagna_reference::GlobalRegions* regions) noexcept;

/** @brief Exact address of the separate untimed original-helper selection observer. */
extern "C" const void* qfn_castagna_t1_selection_observer_kernel() noexcept;
/**
 * @brief Observe original selection from the full kernel's retained raw router logits.
 * @param admission Original prepared full-kernel request/resources.
 * @param ids Caller-owned device output of at least top_k int32 values.
 * @param weights Caller-owned device output of at least top_k floats.
 * @param stream Exact original ordered stream, after full-kernel completion.
 * @return Immediate launch status. This separate1x64 kernel calls the unchanged
 * moe_topk_wave helper; it is excluded from the full-kernel timing interval.
 * No routes are forced, no CPU softmax is substituted, and no global state changes.
 */
extern "C" hipError_t qfn_castagna_t1_observe_selection(
    const qfn_castagna_reference::Admission* admission,int* ids,float* weights,hipStream_t stream) noexcept;
