/**
 * @file CastagnaNativeWeightExport.h
 * @brief Canonical-arithmetic fast-body diagnostic with explicit instance/protocol policies.
 *
 * The original Castagna A export is unchanged. Candidate numerical helpers and
 * loop ownership implement the current canonical HC/router/FFN contract inside
 * the retained layout. Cold typed bindings change scratch authority and scalar
 * synchronization only. No weight allocation, repack or production support
 * advertisement is owned here; movement/grouped integration is a separate gate.
 */
#pragma once
#include "CastagnaHcPersistExport.h"
#include "kernels/common/DeviceNativeQuantisedMatrixDesc.h"
#include <span>
namespace qfn_castagna_native {
using Matrix=llaminar2::DeviceNativeQuantisedMatrixDesc;
/** @brief Original prepared metadata passed directly, without packed-byte copies. */
struct NativeWeights {
    Matrix hc_down,hc_up,shared_gate,shared_up,shared_down;
    const Matrix* expert_gate=nullptr;
    const Matrix* expert_up=nullptr;
    const Matrix* expert_down=nullptr;
    const uint16_t* router_fp16=nullptr; ///< Borrowed actual prepared FP16 router; no conversion is owned here.
    bool normalize_weights=false; ///< Exact original router declaration, independent of clamp.
};
/** @brief Explicit untimed observation scope; never changes numerical arithmetic. */
enum class CanonicalPortObservationScope : unsigned char {
    Disabled, HC, FFN, Complete,
    CompleteEpoch ///< Publish the graph's sigmoid-weighted shared output.
};
/**
 * @brief Caller-owned diagnostic outputs for actual canonical-port phase publications.
 * @details All pointers borrow PMA storage retained before capture and terminal
 * retirement. T1 extents come from the sealed original request: H=2560, R=320,
 * K=4H, F=routed_columns, S=shared_columns, routes=top_k. Codes are int8;
 * scales are FP32 and sums are signed int32 per32 values. Observed and ordinary
 * entries must be distinct compile-time specializations; Events bind Disabled.
 * No observation pointer may be consumed as an execution input or weight cache.
 */
struct CanonicalPortObservation {
    CanonicalPortObservationScope scope=CanonicalPortObservationScope::Disabled;
    float *hc_down=nullptr,*hc_activated=nullptr,*hc_up=nullptr; ///< R,R,K public FP32 planes.
    int8_t *norm_codes=nullptr,*up_primary_codes=nullptr,*up_residual_codes=nullptr; ///< K,R,R codes.
    float *norm_scales=nullptr,*up_primary_scales=nullptr,*up_residual_scales=nullptr; ///< K/32,R/32,R/32.
    int8_t *hidden_codes=nullptr,*routed_codes=nullptr,*shared_codes=nullptr; ///< H,routes*F,S codes.
    float *hidden_scales=nullptr,*routed_scales=nullptr,*shared_scales=nullptr; ///< One scale per32 values.
    int32_t *hidden_sums=nullptr,*routed_sums=nullptr,*shared_sums=nullptr; ///< Exact code sums per32.
    float *routed_gate=nullptr,*routed_up=nullptr,*routed_activated=nullptr; ///< routes*F each.
    float *shared_gate=nullptr,*shared_up=nullptr,*shared_activated=nullptr; ///< S each.
    float* shared_gate_value=nullptr; ///< T actual sigmoid scalars; never recomputed by the host observer.
    float *router_probabilities=nullptr,*selected_weights=nullptr; ///< experts and routes actual canonical publications.
    int32_t *selected_ids=nullptr; ///< routes actual canonical IDs, never recomputed by the old selector.
    float *route_contributions=nullptr,*shared_output=nullptr,*routed_output=nullptr; ///< routes*H,H,H.
    /** @brief Seal every output pointer and scope without comparing struct padding. */
    bool operator==(const CanonicalPortObservation&) const = default;
};
/** @brief Explicit storage experiment, never an automatic fallback. */
enum class ScratchMode : unsigned char { ModuleGlobal, Instance };
/** @brief Explicit diagnostic lifetime protocol; no automatic recovery. */
enum class BarrierMode : unsigned char { MonotonicCounter, InstanceSense, PackedGeneration, WeightedGeneration };
/** @brief Cold caller-owned persistent storage; identity includes allocation generation. */
struct InstanceScratchBinding {
 void* data=nullptr;
 size_t bytes=0;
 uint64_t generation=0;
 bool operator==(const InstanceScratchBinding&) const = default;
};
/** @brief Single-source exported device layout, in bytes. */
struct InstanceScratchLayout { size_t bytes,alignment,prefix_offset,barrier_offset; };
/** @brief Same original external inputs/output roles with prepared matrix ownership. */
struct Request {
    qfn_castagna_reference::Request original;
    NativeWeights native;
    ScratchMode scratch_mode=ScratchMode::ModuleGlobal;
    BarrierMode barrier_mode=BarrierMode::MonotonicCounter;
    InstanceScratchBinding scratch;
    CanonicalPortObservation observation; ///< Optional diagnostic binding; disabled for timing.
};
/** @brief Complete cold B identity and actual function resource admission. */
struct Admission {
    const void* function=nullptr;
    int device=-1;
    unsigned grid=0,threads=0,active_blocks_per_cu=0;
    hipFuncAttributes resources{};
    Request request;
};
}
/** @return Actual B full-kernel function, separate from unchanged A. */
extern "C" const void* qfn_castagna_native_weight_kernel() noexcept;
/** @brief Validate every original prepared source metadata record cold.
 * @param request Complete B binding. @param gate Host metadata matching the existing uploaded512-table.
 * @param up Same up metadata. @param down Same down metadata. @param device Exact original GPU.
 * @param admission Receives exact B function/resources/semantic request snapshot.
 * @return HIP status; no format conversion, allocation, fallback or launch.
 */
extern "C" hipError_t qfn_castagna_native_weight_prepare(const qfn_castagna_native::Request* request,
    std::span<const qfn_castagna_native::Matrix> gate,std::span<const qfn_castagna_native::Matrix> up,
    std::span<const qfn_castagna_native::Matrix> down,int device,qfn_castagna_native::Admission* admission) noexcept;
/** @brief Submit B on the exact stream after immutable binding admission.
 * @param request Same retained request. @param admission Cold B authority. @param stream Nondefault stream.
 * @return Immediate HIP launch status. ModuleGlobal requires exclusive ownership; Instance seals its caller-owned allocation and generation.
 */
extern "C" hipError_t qfn_castagna_native_weight_launch(const qfn_castagna_native::Request* request,
    const qfn_castagna_native::Admission* admission,hipStream_t stream) noexcept;
/** @brief Observe B's separate original module-global storage, cold only. */
extern "C" hipError_t qfn_castagna_native_weight_globals(qfn_castagna_reference::GlobalRegions* regions) noexcept;
/** @brief Untimed original topK helper observer over B's retained raw logits. */
extern "C" hipError_t qfn_castagna_native_weight_observe_selection(const qfn_castagna_native::Admission* admission,
    int* ids,float* weights,hipStream_t stream) noexcept;

/** @return Exact separate untimed B observer entry for captured identity inspection. */
extern "C" const void* qfn_castagna_native_weight_selection_observer_kernel() noexcept;

/** @return Separate compile-time-observed candidate entry; never used by ordinary Events. */
extern "C" const void* qfn_castagna_native_weight_observed_kernel() noexcept;

/**
 * @brief Capture one actual scalar or vector FFN quantizer boundary witness.
 * @param input Caller-owned16-byte-aligned complete32-value blocks.
 * @param output40 Caller-owned40-byte records (scale,sum,codes), at least4-byte aligned.
 * @param blocks Exact positive live block count, bounded by the test owner.
 * @param vectorized Explicit diagnostic entry choice, never a production fallback.
 * @param stream Exact nondefault admitted stream, retained beyond capture.
 * @return Immediate HIP submission status; no query, allocation, transfer or join.
 */
extern "C" hipError_t qfn_castagna_native_weight_ffn_q8_probe(const float* input,
    void* output40,unsigned blocks,bool vectorized,hipStream_t stream) noexcept;

/** @brief Test-only captured scalar/vector HC norm Q8 boundary launcher; no production admission. */
extern "C" hipError_t qfn_castagna_native_hc_q8_boundary(const float* input,void* output40,
    unsigned blocks,bool vectorized,hipStream_t stream) noexcept;

/** @brief Captured same-input logical128 router differential; output uses retained40B diagnostic records. */
extern "C" hipError_t qfn_castagna_native_router_operand_boundary(const float*,void*,unsigned,bool,hipStream_t) noexcept;

namespace qfn_castagna_native {
/** @brief Immutable packed same-input selector geometry; initial test data only. */
/** @brief Explicit diagnostic input authority; no host selection math. */
enum class SelectorBoundaryInput : uint32_t { Logits, Probabilities };
struct SelectorBoundaryCase {
    uint32_t num_experts; ///< Exact live input/probability extent,1..512.
    uint32_t top_k; ///< Exact selected packet width,1..min(E,16).
    uint32_t normalize_weights; ///< Original policy encoded as0or1.
    uint32_t logits_offset; ///< Float offset into the packed original logit input.
    uint32_t output_offset; ///< Word offset: E probabilities, K IDs, K weights, then3guards.
    SelectorBoundaryInput input_kind=SelectorBoundaryInput::Logits;
};
static_assert(sizeof(SelectorBoundaryCase)==6*sizeof(uint32_t));
}
/**
 * @brief Capture the actual original logical256 selector over immutable cases.
 * @param logits Packed live FP32 logit words. @param cases Original per-case geometry.
 * @param output Packed exact probability/ID/weight words with caller-owned guards.
 * @param count Number of complete cases. @param stream Exact nondefault owner stream.
 * @return Immediate HIP launch status; unsupported device bodies trap rather than pass empty.
 */
extern "C" hipError_t qfn_castagna_selector_boundary_physical(const float* logits,
    const qfn_castagna_native::SelectorBoundaryCase* cases,uint32_t* output,unsigned count,hipStream_t stream) noexcept;
/**
 * @brief Capture row-native tournament transfers with the same actual arithmetic body.
 * @param logits Same packed live FP32 input. @param cases Same immutable case geometry.
 * @param output Same packed exact output ABI. @param count Complete case count.
 * @param stream Exact nondefault owner stream. @return Immediate HIP launch status.
 */
extern "C" hipError_t qfn_castagna_selector_boundary_logical(const float* logits,
    const qfn_castagna_native::SelectorBoundaryCase* cases,uint32_t* output,unsigned count,hipStream_t stream) noexcept;

/**
 * @brief Cold actual original/new norm plus Q8 differential; never used by timed inference.
 * @param post Complete E2560 retained POST rows. @param gain Complete original gain rows.
 * @param epsilon_bits Exact FP32 epsilon bits per row. @param output Caller-owned packed records.
 * @param cases Bounded positive row count. @param candidate Explicit diagnostic entry selection.
 * @param stream Exact nondefault stream. @return Immediate launch status.
 */
extern "C" hipError_t qfn_castagna_native_norm_q8_boundary(const float* post,const float* gain,
 const uint32_t* epsilon_bits,uint32_t* output,unsigned cases,bool candidate,hipStream_t stream) noexcept;

/**
 * @brief Cold captured literal-old versus joint-role fold/publication differential.
 * @param input Four-column P records. @param output Six exact words per record.
 * @param cases Positive bounded record count. @param candidate Explicit side.
 * @param stream Exact nondefault owner stream. @return Immediate launch status.
 */
extern "C" hipError_t qfn_castagna_native_shared_joint_role_probe(const float* input,unsigned* output,
 unsigned cases,bool candidate,hipStream_t stream) noexcept;
/**
 * @brief Cold actual-reader differential with distinct original role authorities.
 * @param gate Original gate descriptor. @param up Independent up descriptor.
 * @param activation Actual canonical40B Q8 row. @param first Real first row.
 * @param rows One or two rows valid for both descriptors. @param output Eleven words per lane.
 * @param candidate Explicit side. @param stream Exact owner stream. @return Immediate launch status.
 */
extern "C" hipError_t qfn_castagna_native_shared_role_reader_probe(qfn_castagna_native::Matrix gate,
 qfn_castagna_native::Matrix up,const void* activation,unsigned first,unsigned rows,unsigned* output,
 bool candidate,hipStream_t stream) noexcept;

/** @brief Cold immutable scratch layout; no device query or allocation. */
extern "C" qfn_castagna_native::InstanceScratchLayout qfn_castagna_native_scratch_layout() noexcept;
extern "C" const void* qfn_castagna_native_instance_kernel(bool observed,qfn_castagna_native::BarrierMode protocol) noexcept;
extern "C" hipError_t qfn_castagna_native_instance_regions(const qfn_castagna_native::Request*,qfn_castagna_reference::GlobalRegions*) noexcept;

/** @brief Cold actual-policy scalar barrier publication/weighted-order diagnostic. */
extern "C" hipError_t qfn_castagna_scalar_barrier_probe(unsigned*,const unsigned*,unsigned*,unsigned*,qfn_castagna_native::BarrierMode,hipStream_t) noexcept;
