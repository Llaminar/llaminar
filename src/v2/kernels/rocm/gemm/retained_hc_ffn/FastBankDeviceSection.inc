/**
 * @file CastagnaOrderedDownDeviceSection.inc
 * @brief Retained fast layout with canonical numerical helpers and typed scratch/joins.
 *
 * Numerical loop bodies and phase locations are shared by measured storage and
 * synchronization policies. Native gfx906/T1 scope is explicit; this diagnostic
 * does not advertise production grouped/movement or native gfx1100 support.
 * Historical upstream notes below describe the source lineage, not a changed
 * arithmetic or ownership authority for the selected canonical entry.
 */
// Persistent decode HC pre (qwen4exp, one token), see hc-persist.cuh. Prototype and measurements:
// mi50-engine/persistent-layer/hcpre (engine chain combine+norm / HC down / HC up+pre ~29 us -> ~18 us standalone).
//
// One 1024-thread block per CU (60). Wave 0 is the control wave: norm partial sums and the grid barrier.
// - HC down is tiled 10 row groups x 6 K groups: each block quantizes its K group's slice of y = x*w_norm (rstd is a
//   positive per-stream constant, so xn = y*rstd has the same int8 values; rstd folds into the reduction), dots it
//   with its 32 rows, and adds per-(row, stream) partials into int64 fixed-point accumulators (2^-32: exact, order
//   independent, deterministic). One grid barrier, then every block has lo and rstd.
// - gfx906 memory-pipe rules (measured): a CU's vector memory ops complete in issue order and __syncthreads() after a
//   global store drains all loads (s_waitcnt vmcnt(0)); so block syncs are LDS-only, the barrier runs on the scalar
//   memory path, and the HC up weights stream across the barrier.
// Weights are read in their raw q8_0 layout as dword-aligned block pairs (68 B): K groups start at even q8 blocks.

namespace qfn_retained_m1_hc_ffn_device {
using namespace qfn_castagna_native_impl; // Borrow the one unchanged canonical Q8 authority.
#include "NativeWeightReaders.inc"
#include "CanonicalOrderedDown.inc"
#include "CanonicalFFNArithmetic.inc"

// HCP_NT: threads per block. 1024 (16 waves per CU, 64 VGPRs) or 512 (8 waves per CU, 128 VGPRs: more loads in flight
// per lane, e.g. both HC weight streams); one block per CU either way
#ifndef HCP_NT
#define HCP_NT 1024
#endif
constexpr int E = 2560, HC = 4, K = HC*E, LR = 320, NT = HCP_NT;
constexpr int QB = K/32;              // q8 blocks of xn (320), 80 per stream
constexpr int PU = LR/64;             // q8_0 block pairs per HC up row (5)
constexpr int NB = 60;
constexpr int KG = 6, RG = NB/KG, RPG = LR/RG; // 10 row groups of 32 rows x 6 K groups
constexpr int RPW2 = RPG*64/NT/2;     // HC down row pairs per wave (1 at 1024 threads, 2 at 512)
constexpr int IU   = (HC*((E + NB - 1)/NB)*(LR/64) + NT - 1)/NT; // HC up (row, pair) items per thread (860 items: 1 or 2)
constexpr int PSTR = IU*NT;           // row stride of part[]: every HC up item
/**
 * @brief Original HC-up row owned by a native column-contiguous physical item.
 * @param physical Actual original thread0..1023. @return Original branch/column row0..223.
 * @details Each160-thread packet owns32 rows and five pair indices. The row
 * bounds predicate avoids retaining a full logical part index through float dots.
 */
__device__ __forceinline__ unsigned native_hc_up_row(unsigned physical) {
    static_assert(NT==1024&&IU==1&&PU==5,"HC-up map requires original admitted geometry");
    return (physical/160u)*32u+(physical&31u);
}
/** @brief Original pair0..4 in the same160-thread packet; no column is encoded. */
__device__ __forceinline__ unsigned native_hc_up_pair(unsigned physical) {
    return (physical>>5)%5u;
}
// HCP_EARLY_WU: issue the HC up weight loads at kernel entry, right after the HC down ones (both 3.5 MB streams in flight
// from the start; needs the 512-thread register budget), instead of behind the HC down atomics
#if HCP_NT == 1024
#define HCP_EARLY_WU 0
#define HCP_WU_VMCNT "6"
#elif HCP_NT == 512
#ifndef HCP_EARLY_WU
#define HCP_EARLY_WU 1
#endif
#if HCP_EARLY_WU
#define HCP_WU_VMCNT "0"
#else
#define HCP_WU_VMCNT "34"
#endif
#else
#error "HCP_NT must be 1024 or 512"
#endif
static_assert(RPW2 >= 1 && RPG == 2*RPW2*(NT/64), "HC down tile rows must split evenly over the waves");
struct q81 { half2 ds; int qs[8]; };

constexpr int TMAX = 4;              // tokens per call (decode 1, MTP verify up to 4)
constexpr int HCP_TS_N = 16;         // phase timestamps per block (GGML_CUDA_HCP_TS)
struct acc_t { long long lo[TMAX][HC][LR]; long long ss[TMAX][HC]; };
// inject logits stash (int64 fixed point, by barrier epoch parity): kernel e accumulates its own sublayer's inject
// into [e&1] (GGML_CUDA_HC_PERSIST=3), the next kernel reads [e&1] before its barrier instead of running the inject
// matmul, and every kernel clears the parity it could have read after its barrier
// kernel counter (its parity picks the stash / accumulator buffers; a kernel runs 1 or 3 grid barriers)
// FFN extension (GGML_CUDA_HC_PERSIST=4): the FFN-side kernel also computes the shared expert; the next kernel adds it
constexpr int FF = 640;               // shared expert width (max; -sm tensor: this device's shard, 320)
constexpr int NR = 512;               // router experts

// early matvecs (GGML_CUDA_HC_PERSIST=4): MUL_MATs of mixed found shortly after the chain (GDN qkv / z, attention
// projections) and the GDN gate group, computed after the second grid barrier straight into their output tensors (the
// host checks that nothing in between touches that memory); the graph then skips those nodes
constexpr int HCP_EARLY_Q8 = 4;
struct hc_early {
    const char * w[HCP_EARLY_Q8];  // q8_0 [rows][2560]
    float      * out[HCP_EARLY_Q8]; // [T][rows]
    int          rows[HCP_EARLY_Q8];
    int          nq, rows_total;
    const float * wa, * wb, * dt, * A;   // GDN gates: gate = softplus(wa.x + dt)*A, beta = sigmoid(wb.x)
    float       * gate, * beta;          // [T][H]
    int           H;
};

// fused tensor-parallel AllReduce (GGML_CUDA_HCP_AR=1, 2 GPUs): the AllReduce of the sublayer output is deferred into
// the persistent kernel that consumes it; block 0 pushes this GPU's partial into the peer's inbox (P2P) and raises the
// peer's flag, waits for its own, and every block reads out = own + inbox (both GPUs add in the same order as the
// AllReduce kernel, so the mirrored results stay bit identical). Device counters replace host tokens (HIP graphs).
// N GPUs (2 or 4): the inbox has one TMAX*E slot per peer in each half ([half][slot][TMAX*E]), a flag per slot; slot of
// source rank q on rank r: q < r ? q : q - 1. 4 GPUs add the partials by rank as (p0 + p1) + (p2 + p3) on every GPU.
constexpr int ARNS = 3;               // max peers
struct hc_ar {
    const float * inbox_mine;  // this GPU's inbox: 2 halves of ns*TMAX*E floats (uncached)
    const int   * flag_mine;   // uncached ints (one per peer slot), raised by the peers' pushes
    int ns;                    // peers: 1 (pair) or 3
    int rank;                  // this GPU's rank in the group
};
struct hc_push {               // this GPU's partial -> each peer's slot for it
    float * inbox[ARNS];       // the peer's inbox + this GPU's slot (half 0)
    int   * flag[ARNS];        // the peer's flag for this GPU's slot
    int n;                     // peers
    int hs;                    // floats per inbox half (ns*TMAX*E)
};

// wait half of the exchange: the peer's push kernel (queued at AllReduce time, so a host sync on this GPU cannot starve
// it) writes its partial into this GPU's inbox half s&1 and then raises flag_mine to s
__device__ __forceinline__ void hcp_ar_wait(const hc_ar & ar, unsigned int s) {
    if (threadIdx.x < ar.ns) {
        while ((int) (((const volatile int *) ar.flag_mine)[threadIdx.x] - (int) s) < 0) { __builtin_amdgcn_s_sleep(1); }
        __threadfence_system();
    }
    __syncthreads();
}

// the reduced value from this GPU's own partial and the inbox half: pairs own + peer (as the AllReduce kernel); 4 GPUs
// (p0 + p1) + (p2 + p3) by rank
template <bool AR4>
__device__ __forceinline__ float hcp_ar_sum(float own, const float * in, int ns, int rank, int idx) {
    if constexpr (!AR4) {
        (void) ns; (void) rank;
        return own + in[idx];
    }
    float v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        v[q] = q == rank ? own : in[(q < rank ? q : q - 1)*(TMAX*E) + idx];
    }
    return (v[0] + v[1]) + (v[2] + v[3]);
}

// routed experts (GGML_CUDA_HC_PERSIST=4, GGML_CUDA_HCP_MOE): after the router logits' barrier every block computes the
// top-k itself (one wave per token, bit-identical to topk_moe_sorted_gcn), then the q4_K gate/up rows of the selected
// experts (16 lanes per row) -> silu(gate)*up; after one more barrier the down rows (q5_1 / q8_0) of all selected experts,
// the routing weights, the shared expert's down rows and the shared gate give the FFN output directly
constexpr int NUMAX = 16;             // max experts per token
constexpr int FFRMAX = 640;           // max expert rows (gate/up) on this device
/** @brief Same physical banks, now optionally borrowed from one instance owner. */
struct alignas(16) InstanceOriginalScratch {
 acc_t        g_hcp_acc[2];
 unsigned int g_hcp_bar;
 long long    g_hcp_inj[2][TMAX][HC];
 unsigned int g_hcp_kern;
 alignas(16) float g_hcp_h[TMAX][FF];
 float        g_hcp_sg[TMAX];
 float        g_hcp_shexp[TMAX][E];
 unsigned int g_hcp_arseq;
 unsigned int g_hcp_pushed;
 unsigned int g_hcp_pushcnt;
 alignas(16) float g_hcp_mh[TMAX*NUMAX*FFRMAX];
};
static_assert(offsetof(InstanceOriginalScratch,g_hcp_h)%16==0);
static_assert(offsetof(InstanceOriginalScratch,g_hcp_mh)%16==0);
static_assert(offsetof(InstanceOriginalScratch,g_hcp_shexp)%16==0);

struct hc_moe {
    // Routed weights are immutable canonical descriptor tables.
    int          ffr;              // rows per expert (gate/up) = down K
    int          dq8;              // down is q8_0 (else q5_1)
    int          nused;            // selected experts per token
    float        clamp;            // weight sum clamp (norm)
    float      * out;              // FFN output [T][2560]
    int          on;
    int          dedup;            // T > 1: share each distinct expert's weight reads between its tokens
    int          gq5;              // gate / up type: 0 q4_K, 1 q5_K, 2 q8_0
    hc_push      push;             // fused AllReduce: the peers' inbox slots for this GPU (push.n == 0: none)
    llaminar2::rocm::native_runtime_abi::DeviceMoELayerRuntimeView* runtime;
    const int32_t* active_rows;
    float* public_ids;
    float* public_weights;
    llaminar2::MoEDecodeDescriptorPublication publication;
};

struct hc_ffn {
    const float * w_logits;  // router [512][2560] f32
    // Shared weights are immutable canonical descriptors.
    const float * w_sg;      // shared expert gate logit [2560] f32
    float       * logits;    // out [T][512]
    int           on;
    int           ff;        // shared expert rows on this device (640, or the -sm tensor shard)
    hc_moe        moe;
    NativeMatrix native_sgate,native_sup,native_sdown;
    const uint16_t* canonical_router_fp16;
    bool canonical_normalize_weights;
};

__device__ __forceinline__ int kslice0(int g) { return g == KG ? QB : 2*((g*QB/KG + 1)/2); } // even: 0 54 106 160 214 266
__device__ __forceinline__ long long to_fx(float v) { return __float2ll_rn(v*4294967296.0f); }
__device__ __forceinline__ float from_fx(long long v) { return (float) v*(1.0f/4294967296.0f); }
__device__ __forceinline__ void acc_add(long long * p, float v) {
    __hip_atomic_fetch_add((unsigned long long *) p, (unsigned long long) to_fx(v), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void lds_sync() { __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }

// full grid barrier (all of this block's global stores done first), counter on the scalar memory path
/**
 * @brief Original grid protocol with a signaling-wave-only publication hook.
 * @param before_arrival Inline callback executed by all64 lanes of wave0 after
 * the original entry join. It must drain any new GLOBAL stores before returning.
 * @details Other waves can publish nothing after entry; they proceed to the
 * original exit join. The scalar counter, poll, invalidation and both joins are
 * unchanged. Empty callbacks preserve every other original grid boundary.
 */
template<class BeforeArrival>
__device__ __forceinline__ void native_grid_sync_before_arrival(const BeforeArrival& before_arrival,unsigned* counter) {
    __asm__ volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    lds_sync();
    if (threadIdx.x < 64) {
        before_arrival();
        unsigned int old;
        __asm__ volatile("s_atomic_add %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(old) : "s"(counter), "0"(1u) : "memory");
        const unsigned int target = (old/NB + 1)*NB;
        while (true) {
            unsigned int cur;
            __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(cur) : "s"(counter) : "memory");
            if (cur >= target) { break; }
            __builtin_amdgcn_s_sleep(0);
        }
        __asm__ volatile("buffer_wbinvl1_vol" ::: "memory");
    }
    lds_sync();
}

/** @brief Preserve original grid boundaries that have no new publication. */
__device__ __forceinline__ void grid_sync_all(unsigned* counter) {
    native_grid_sync_before_arrival([] {},counter);
}

#include "ScalarInstanceSenseBarrier.inc"
#include "ScalarInstancePackedBarrier.inc"
#include "ScalarInstanceWeightedBarrier.inc"
/** @brief The unchanged counter protocol with explicit bound address. */
struct CounterInstanceJoin { unsigned* counter; __device__ __forceinline__ void sync(){grid_sync_all(counter);} };
/** @brief Select an explicit join policy at compile time, keeping original call locations. */
template<class Join> __device__ __forceinline__ void grid_sync_all(Join& join){join.sync();}

// q8_0 block pair (17 dwords) . two q8_1 blocks
__device__ __forceinline__ float pair_dot(const int w[17], const q81 & y0, const q81 & y1) {
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        s0 = __builtin_amdgcn_sdot4(w[j], y0.qs[j], s0, false);
        s1 = __builtin_amdgcn_sdot4(w[8 + j], y1.qs[j], s1, false);
    }
    const float d0 = __half2float(__ushort_as_half((unsigned short) (w[16] & 0xFFFF)));
    const float d1 = __half2float(__ushort_as_half((unsigned short) ((unsigned int) w[16] >> 16)));
    return d0*__low2float(y0.ds)*(float) s0 + d1*__low2float(y1.ds)*(float) s1;
}

// q8_0 block pair (17 dwords) . 64 f32 values (two 32-value groups at a0 and a1: padded LDS rows)
__device__ __forceinline__ float pair_dot_f32(const int w[17], const float * a0, const float * a1) {
    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q0 = w[j], q1 = w[8 + j];
        s0 += (float) (int8_t) q0*a0[4*j] + (float) (int8_t) (q0 >> 8)*a0[4*j + 1] + (float) (int8_t) (q0 >> 16)*a0[4*j + 2] + (float) (int8_t) (q0 >> 24)*a0[4*j + 3];
        s1 += (float) (int8_t) q1*a1[4*j] + (float) (int8_t) (q1 >> 8)*a1[4*j + 1] + (float) (int8_t) (q1 >> 16)*a1[4*j + 2] + (float) (int8_t) (q1 >> 24)*a1[4*j + 3];
    }
    const float d0 = __half2float(__ushort_as_half((unsigned short) (w[16] & 0xFFFF)));
    const float d1 = __half2float(__ushort_as_half((unsigned short) ((unsigned int) w[16] >> 16)));
    return d0*s0 + d1*s1;
}

// the residual streams: either x (the combine already ran) or, fused (POST), the combine out*pw + res computed on
// the fly (pw = ps2*sigmoid(ps1*raw) per stream and token, as dsv4_hc_post_f32)
struct hc_src {
    const float * x;     // [T][4][2560], or the residual when POST
    const float * out;   // POST: sublayer output [T][2560]
    const float * raw;   // POST: inject logits [T][4] (unless read from the stash)
    const float * w_inj; // INJ: inject weights [4][10240]
    float ps1, ps2;
    int shx;             // POST: out += the shared expert stash (the FFN output is moe_out + shexp)
    const float * ar_in; // POST, fused AllReduce: this exchange's inbox half (the peers' partials of out), else null
    int ar_ns, ar_rank;  // peers, this GPU's rank
};

template <bool POST, bool AR4>
__device__ __forceinline__ float4 load_x4(const hc_src & a, const float * pw, int t, int j4,const float (*shared_output)[E]) {
    const float4 r = ((const float4 *) a.x)[t*(K/4) + j4];
    if (!POST) {
        return r;
    }
    float4 o = ((const float4 *) a.out)[t*(E/4) + j4 % (E/4)];
    if (a.ar_in) {
        if constexpr (!AR4) {
            const float4 pr = ((const float4 *) a.ar_in)[t*(E/4) + j4 % (E/4)];
            o = make_float4(o.x + pr.x, o.y + pr.y, o.z + pr.z, o.w + pr.w);
        } else {
            const int b = t*E + 4*(j4 % (E/4));
            o = make_float4(hcp_ar_sum<true>(o.x, a.ar_in, a.ar_ns, a.ar_rank, b), hcp_ar_sum<true>(o.y, a.ar_in, a.ar_ns, a.ar_rank, b + 1),
                            hcp_ar_sum<true>(o.z, a.ar_in, a.ar_ns, a.ar_rank, b + 2), hcp_ar_sum<true>(o.w, a.ar_in, a.ar_ns, a.ar_rank, b + 3));
        }
    }
    if (a.shx) {
        const float4 sh = ((const float4 *) shared_output[t])[j4 % (E/4)];
        o = make_float4(o.x + sh.x, o.y + sh.y, o.z + sh.z, o.w + sh.w);
    }
    const float w = pw[t*HC + (4*j4) / E];
    return make_float4(o.x*w + r.x, o.y*w + r.y, o.z*w + r.z, o.w*w + r.w);
}

// q8_0 block pair . T f32 vectors at once (each int8 is converted once; converting per token let the compiler keep
// all 64 converted weights live across the token loop and spill)
template <int T>
__device__ __forceinline__ void pair_dot_f32_t(const int w[17], const float (*lo)[(LR/32)*33], int pb, float out[T]) {
    float s0[T], s1[T];
#pragma unroll
    for (int t = 0; t < T; ++t) { s0[t] = 0.0f; s1[t] = 0.0f; }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int q0 = w[j], q1 = w[8 + j];
        // B-only scheduling edge: expand this unchanged packed pair only
        // after its preceding original dot prefix, not all future bytes early.
        // The identity changes no bits and carries no memory-ordering claim.
        asm volatile("" : "+v"(q0), "+v"(q1) : "v"(s0[0]), "v"(s1[0]));
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const float f0 = (float) (int8_t) (q0 >> (8*b)), f1 = (float) (int8_t) (q1 >> (8*b));
#pragma unroll
            for (int t = 0; t < T; ++t) {
                s0[t] += f0*lo[t][33*pb + 4*j + b];
                s1[t] += f1*lo[t][33*(pb + 1) + 4*j + b];
            }
        }
    }
    const float d0 = __half2float(__ushort_as_half((unsigned short) (w[16] & 0xFFFF)));
    const float d1 = __half2float(__ushort_as_half((unsigned short) ((unsigned int) w[16] >> 16)));
#pragma unroll
    for (int t = 0; t < T; ++t) { out[t] = d0*s0[t] + d1*s1[t]; }
}

// ---- routed experts helpers
template <int ctrl, int row_mask = 0xF>
__device__ __forceinline__ uint32_t moe_dpp(const uint32_t v) {
    return (uint32_t) __builtin_amdgcn_update_dpp((int) v, (int) v, ctrl, row_mask, 0xF, false);
}
template <bool MAX>
__device__ __forceinline__ uint32_t moe_wave_reduce(uint32_t v) {
    const auto op = [](uint32_t a, uint32_t b) { return MAX ? max(a, b) : min(a, b); };
    v = op(v, moe_dpp<0xB1>(v));
    v = op(v, moe_dpp<0x4E>(v));
    v = op(v, moe_dpp<0x141>(v));
    v = op(v, moe_dpp<0x140>(v));
    v = op(v, moe_dpp<0x142, 0xA>(v));
    v = op(v, moe_dpp<0x143, 0xC>(v));
    return (uint32_t) __builtin_amdgcn_readlane((int) v, 63);
}

// sums over DPP rows (no LDS): every lane of each 16-lane row / 8-lane half row gets its group's sum
template <int ctrl>
__device__ __forceinline__ float dppf(const float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), ctrl, 0xF, 0xF, false));
}
__device__ __forceinline__ float row8_sum(float v) {
    v += dppf<0xB1>(v);    // quad_perm [1,0,3,2]
    v += dppf<0x4E>(v);    // quad_perm [2,3,0,1]
    v += dppf<0x141>(v);   // row_half_mirror
    return v;
}
__device__ __forceinline__ float row16_sum(float v) {
    v = row8_sum(v);
    v += dppf<0x140>(v);   // row_mirror
    return v;
}
__device__ __forceinline__ float row8_max(float v) {
    v = fmaxf(v, dppf<0xB1>(v));
    v = fmaxf(v, dppf<0x4E>(v));
    v = fmaxf(v, dppf<0x141>(v));
    return v;
}
// sum over each 32-lane half: the result is valid in the upper row of each half (lanes 16-31 and 48-63)
__device__ __forceinline__ float half32_sum(float v) {
    v = row16_sum(v);
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x142, 0xA, 0xF, false));   // row_bcast15
    return v;
}
// wave sum (any order), result in every lane
__device__ __forceinline__ float wave_sum_f(float v) {
    v = half32_sum(v);
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x143, 0xC, 0xF, false));   // row_bcast31
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 63));
}
// wave maximum (any order), result in every lane
__device__ __forceinline__ float wave_max_f(float v) {
    v = row8_max(v);
    v = fmaxf(v, dppf<0x140>(v));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(v), __float_as_int(v), 0x142, 0xA, 0xF, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(v), __float_as_int(v), 0x143, 0xC, 0xF, false)));
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 63));
}



// scales / mins of q4_K sub-blocks 2p and 2p + 1 (get_scale_min_k4) from the 12 scale bytes (w0 w1 w2)
// QK effective coefficients are provided by the native prepared scale/min planes.

// one q4_K item of a row: the lane's 16-byte qs chunk qv (sub-blocks 2p, 2p+1 at positions 16h..16h+15, hd: the
// super-block's d/dmin and scales) . the q8 x ints xa (block 2p) / xb (block 2p+1) with their scales and int sums
// (q5_K: qh holds the 16 qh bytes of the same positions; bit 2p / 2p + 1 is the 5th bit of sub-block 2p / 2p + 1)
template <bool Q5>
__device__ __forceinline__ float moe_q4k_item(NativeQKItem item, const int xa[4], const int xb[4],
                                              const int p, const int half, const float d8a, const float d8b, const int sA, const int sB) {
    const int4 qv=item.q,qh=item.high;
    const int q[4] = { qv.x, qv.y, qv.z, qv.w }, hb[4] = { qh.x, qh.y, qh.z, qh.w };
    int dA = 0, dB = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int lo,hi;
        if constexpr (Q5) {
            lo = q[j] & 0x0F0F0F0F;
            hi = (q[j] >> 4) & 0x0F0F0F0F;
            lo |= ((hb[j] >> (2*p)) & 0x01010101) << 4;
            hi |= ((hb[j] >> (2*p + 1)) & 0x01010101) << 4;
        } else {
            // This half owns one complete canonical K32 block. Compute its
            // two original 16-value chains before exchanging one integer.
            lo=unsigned(q[j])&0x0F0F0F0F;
            hi=(unsigned(q[j])>>4)&0x0F0F0F0F;
        }
        dA = __builtin_amdgcn_sdot4(lo, xa[j], dA, false);
        dB = __builtin_amdgcn_sdot4(hi, xb[j], dB, false);
    }
    if constexpr (!Q5) {
        const int local_low=dA,local_high=dB;
        const int peer_partial=__builtin_bit_cast(int,native_q4_half_peer(__builtin_bit_cast(unsigned,half ? local_low : local_high)));
        dA=half ? peer_partial : local_low;
        dB=half ? local_high : peer_partial;
        const unsigned local=unsigned(item.high.x);
        const unsigned peer=native_q4_half_peer(local);
        const unsigned a=half ? peer : local,b=half ? local : peer;
        item.scale_a=__half2float(__ushort_as_half(static_cast<unsigned short>(a)));
        item.scale_b=__half2float(__ushort_as_half(static_cast<unsigned short>(b)));
        item.min_a=__half2float(__ushort_as_half(static_cast<unsigned short>(a>>16)));
        item.min_b=__half2float(__ushort_as_half(static_cast<unsigned short>(b>>16)));
    }
    if constexpr (!Q5) {
        // Preserve the accepted native-Q4 minimum term's emitted orientation:
        // round the B product, then fuse A into that value. Moving coefficient
        // bits through the peer lane must not choose the opposite contraction.
        float minimum_a,minimum_b,minimum_b_product;
        {
#pragma clang fp contract(off)
            minimum_a=d8a*(float)sA;
            minimum_b=d8b*(float)sB;
            minimum_b_product=minimum_b*item.min_b;
        }
        const float minimum=__builtin_fmaf(minimum_a,item.min_a,minimum_b_product);
        return (d8a*(float)dA*item.scale_a + d8b*(float)dB*item.scale_b) + minimum;
    }
    return (d8a*(float)dA*item.scale_a + d8b*(float)dB*item.scale_b) +
           (d8a*(float)sA*item.min_a + d8b*(float)sB*item.min_b);
}

// q5_1 block (6 dwords) . q8_1 block (ds = d, sum)
__device__ __forceinline__ float moe_q51_dot(const int * w, const q81 & x) {
    const unsigned qh = w[1];
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = w[2 + j];
        const unsigned hb = qh >> (4*j), hb2 = qh >> (4*j + 16);
        const int lo = (q & 0x0F0F0F0F)        | ((hb & 1) << 4)  | ((hb & 2) << 11)  | ((hb & 4) << 18)  | ((hb & 8) << 25);
        const int hi = ((q >> 4) & 0x0F0F0F0F) | ((hb2 & 1) << 4) | ((hb2 & 2) << 11) | ((hb2 & 4) << 18) | ((hb2 & 8) << 25);
        sumi = __builtin_amdgcn_sdot4(lo, x.qs[j], sumi, false);
        sumi = __builtin_amdgcn_sdot4(hi, x.qs[4 + j], sumi, false);
    }
    const float2 dm = __half22float2(__builtin_bit_cast(half2, w[0]));
    const float2 ds = __half22float2(x.ds);
    return dm.x*ds.x*(float) sumi + dm.y*ds.y;
}

/**
 * @brief Original HC-down pair owned by a row-wide physical lane.
 * @param wave Original wave0..15. @param lane Physical lane0..63.
 * @return Original pair0..31; each half-wave owns32 adjacent output rows.
 */
__device__ __forceinline__ unsigned native_hc_down_pair(unsigned wave,unsigned lane) {
    return 2u*wave+(lane>>5);
}
/**
 * @brief Original local output row in the fixed32-row HC-down CTA tile.
 * @param lane Physical lane0..63. @return Local row0..31.
 */
__device__ __forceinline__ unsigned native_hc_down_row(unsigned lane) {
    return lane&31u;
}
/**
 * @brief Address one exact original leaf in the temporary phase-union plane.
 * @param row Original local row0..31. @param pair Original partition0..31.
 * @return Padded float index; row-wide writes and half32 reads permute LDS banks.
 * @details The pad is storage only. No padded value is read or communicated.
 */
__device__ __forceinline__ unsigned native_hc_down_leaf(unsigned row,unsigned pair) {
    return pair*33u+row;
}

// 8 consecutive threads quantize 32 values (4 each) into a q8_1 block with ds = (d, sum), as quantize_q8_1
__device__ __forceinline__ void moe_quant4(const float4 v, q81 & b, const int sub) {
    const float amax = row8_max(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    const float sum  = row8_sum(v.x + v.y + v.z + v.w);
    const float d = amax/127.0f, dd = amax == 0.0f ? 1.0f : d;
    char4 q;
    const float id = __builtin_amdgcn_rcpf(dd); // one reciprocal instead of 4 IEEE divisions (x/d rounds the same
                                                // except within an ulp of a .5 boundary)
    q.x = roundf(v.x*id); q.y = roundf(v.y*id); q.z = roundf(v.z*id); q.w = roundf(v.w*id);
    b.qs[sub] = *(int *) &q;
    if (sub == 0) { b.ds = make_half2(d, sum); }
}

#include "CurrentLlaminarHCPrefix.inc"

template <int T, bool POST, bool STASH, bool INJ, bool AR4, qfn_castagna_reference::GateSource GateFormat, bool ObserveCanonical=false,bool Instance=false,bool Sense=false,bool Packed=false,bool Weighted=false,bool PublishRoleTables=true,unsigned RoutedDownCodebook=7,bool CollectHistogram=true>
__launch_bounds__(NT)
#if HCP_NT == 512
__attribute__((amdgpu_waves_per_eu(1, 2)))
#endif
__global__ void hc_persist_kernel(const hc_src src_in, const float * __restrict__ w_norm, const NativeMatrix w_down,
                                  const NativeMatrix w_up, const float eps, const float lo_scale, const float pre_scale,
                                  float * __restrict__ xpost, float * __restrict__ xn, float * __restrict__ mixed, const hc_ffn ffn,
                                  const hc_ar ar,
                                  qfn_castagna_native::CanonicalPortObservation observation={}, qfn_castagna_native::InstanceScratchBinding binding={}) {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX9__) // GCN only (hc_persist_chain): other targets compile an empty kernel

    // This is the retained closed/no-observation entry admitted by graph sealing.
    // Only this entry uses the gfx906 native-half descriptor and packed selector.
    constexpr bool ClosedRetainedPolicy=!PublishRoleTables&&!ObserveCanonical;
    constexpr hc_early early{};
    auto* instance_original=static_cast<InstanceOriginalScratch*>(binding.data);
    CurrentHCPrefixState* instance_prefix=nullptr;
    if constexpr(Instance)instance_prefix=reinterpret_cast<CurrentHCPrefixState*>(static_cast<char*>(binding.data)+((sizeof(InstanceOriginalScratch)+15)&~size_t(15)));
    auto& g_hcp_acc=instance_original->g_hcp_acc;
    auto& g_hcp_bar=instance_original->g_hcp_bar;
    auto& g_hcp_inj=instance_original->g_hcp_inj;
    auto& g_hcp_kern=instance_original->g_hcp_kern;
    auto& g_hcp_h=instance_original->g_hcp_h;
    auto& g_hcp_sg=instance_original->g_hcp_sg;
    auto& g_hcp_shexp=instance_original->g_hcp_shexp;
    auto& g_hcp_arseq=instance_original->g_hcp_arseq;
    auto& g_hcp_pushed=instance_original->g_hcp_pushed;
    auto& g_hcp_pushcnt=instance_original->g_hcp_pushcnt;
    auto& g_hcp_mh=instance_original->g_hcp_mh;
    // The one-row closed graph has no consumer of the intermediate FP32 layout.
    // Fold shared and routed activation into one contiguous input to its Q8 read.
    // Other entries retain their original row stride and shared plane.
    static_assert(!ClosedRetainedPolicy || (T==1 && NUMAX*FFRMAX+FF<=TMAX*NUMAX*FFRMAX));
    const unsigned routed_stride=ClosedRetainedPolicy ? unsigned(ffn.moe.ffr) : FFRMAX;
    const unsigned shared_start=unsigned(ffn.moe.nused)*routed_stride;
    static_assert(Instance&&Weighted,"The retained entry requires its admitted instance and weighted join");
    auto& prefix_state=*instance_prefix;
    constexpr size_t barrier_offset=((sizeof(InstanceOriginalScratch)+15)&~size_t(15))+sizeof(CurrentHCPrefixState);
    using Join=std::conditional_t<Weighted,ScalarInstanceWeightedJoin,std::conditional_t<Packed,ScalarInstancePackedJoin,std::conditional_t<Sense,ScalarInstanceSenseJoin,CounterInstanceJoin>>>;
    Join join{};
    if constexpr(Sense)join.words=reinterpret_cast<unsigned*>(static_cast<char*>(binding.data)+barrier_offset);
    else join.counter=&g_hcp_bar;
static_assert(!Weighted||(Packed&&Sense&&Instance),"Weighted protocol requires the same instance phase bank");
static_assert(!Packed||(Sense&&Instance),"Packed lifetime protocol requires instance-owned phase storage");
#define HTS(k) {}
    // HC and FFN activation lifetimes are separated by mixed-ready. Keep the
    // HC name available for unselected original branches; current suffix uses
    // the original production code/scale/intsum representation exclusively.
    __shared__ union { q81 hc[T][80]; CanonicalFFNQ8Block ffn[T][80]; } activation_storage;
    auto& xqk=activation_storage.hc;
    auto& canonical_hidden=activation_storage.ffn;
    __shared__ float lo2[T][(LR/32)*33];  // silu(lo_scale*lo) in f32 like the engine's fused HC up (padded rows)
    // part[T][NT] (HC up / shared expert down partials), and in the routed expert phase the q8_1 staging of the down
    // inputs (T*(nused*ffr + ff)/32 blocks)
    // (T <= 3: also mixed in f32 for the E phase's f32 matvecs, T*10 KB)
    constexpr int ARENA = T < 4 ? T*E*4 : 4*4800;
    // Affine GU owns its existing temporary64-row effective half coefficient panel in this phase union.
    // Earlier mixed/HC users and later down Q8 users retain their original edges.
    constexpr int NATIVE_GU_PANEL_BYTES = 2*(E/32)*NativeQ4PanelContract::row_capacity*sizeof(unsigned);
    constexpr int PREFIX_PHASE_BYTES = (256+2*E)*sizeof(float);
    constexpr int NATIVE_BODY_BYTES = ARENA > T*PSTR*4 ? ARENA : T*PSTR*4;
    constexpr int NATIVE_PHASE_BYTES = NATIVE_BODY_BYTES > PREFIX_PHASE_BYTES ? NATIVE_BODY_BYTES : PREFIX_PHASE_BYTES;
    __shared__ __attribute__((aligned(16))) char arena[NATIVE_PHASE_BYTES > NATIVE_GU_PANEL_BYTES ? NATIVE_PHASE_BYTES : NATIVE_GU_PANEL_BYTES];
    static_assert(sizeof(arena)>=(256+2*E)*sizeof(float),
        "Wide HC norm staging must fit the existing phase arena");
    float (*part)[PSTR] = (float (*)[PSTR]) arena;
    __shared__ int   sel_e[T][NUMAX];
    // Separate live extent: T1's original sorting arrays can be eliminated.
    // No alias with grouping state and no claim that these bytes are free.
    __shared__ uint32_t native_route_masks[T];
    __shared__ int   d_ex[T*NUMAX];       // routed experts: distinct experts of the call (first-appearance order)
    __shared__ int   d_ent[T*NUMAX][T];   // their users: entries t*nused + slot
    __shared__ int   d_tok[T*NUMAX][T];   // their tokens
    __shared__ int   f_ent[T*NUMAX];      // all entries grouped by expert (distinct order, users in entry order)
    __shared__ int   f_tok[T*NUMAX];      // their tokens
    __shared__ int   d_n[T*NUMAX];
    __shared__ int   d_cnt;
    __shared__ float sel_w[T][NUMAX];
    __shared__ float gl[T][HC][64];
    __shared__ float xcw[T][HC][64];
    __shared__ float rstd[T][HC];
    __shared__ float red[2*T][NT/64];
    __shared__ float pw[T*HC];
    __shared__ float injp[HC][TMAX*44];
    __shared__ float4 xsl[TMAX*44];      // this block's slice of the residual streams (read before the barrier)
    __shared__ float4 wil[HC][44];       // INJ: inject weights of the slice
    __shared__ int   epoch_s;
    const int tid = threadIdx.x, bid = blockIdx.x, lane = tid % 64, wave = tid / 64;
    HTS(0)
    static_assert(POST && !STASH && !INJ && !AR4,"Current HC diagnostic is explicit ordinary POST, without injection/peer work");
    const int e0=bid*E/NB,e1=(bid+1)*E/NB,n=e1-e0;
    static_assert(sizeof(arena)>=2u*(LR/32)*HC*((E+NB-1)/NB)*sizeof(float),
        "Whole-CTA HC-up contribution panel must fit the actual retired phase arena");
    static_assert(sizeof(arena)>=(256u+E)*sizeof(float),
        "HC normalized publication must fit the actual phase arena");
    static_assert(sizeof(lo2)>=2u*LR+2u*(LR/32)*sizeof(float),
        "Local HC activation banks must fit the existing lo2 interval");
    static_assert(sizeof(arena)>=LR*sizeof(float),
        "Local activated diagnostic values must fit the retired phase arena");
    static_assert(sizeof(arena)>=16u*(20u+1u)*((LR+NB-1)/NB)*sizeof(float),
        "Output-owned HC down leaves and completed partitions must fit phase arena");
    const bool live=llaminar2::rocm::router_native::runtime_decode_row_active(ffn.moe.active_rows);
    current_llaminar_hc_prefix<T,::llaminar2::QuantizedActivationOperandPrecision::TwoTermQ8,
        ObserveCanonical,ClosedRetainedPolicy>(
        src_in,w_norm,w_down,w_up,eps,xpost,xn,mixed,
        reinterpret_cast<float*>(arena),reinterpret_cast<float*>(lo2),gl,observation,prefix_state,join,live);
    HTS(4)
    // The HC prefix is observable even for count0. Only its completed publication
    // edge may precede the original runtime's inactive routing transaction.
    if(!live) {
        using Runtime=llaminar2::rocm::native_runtime_abi::DeviceMoELayerRuntimeView;
        using Acquired=llaminar2::rocm::router_native::AcquiredRuntimeDecodeBank<Runtime>;
        const auto inactive_bank=Acquired::acquire(ffn.moe.runtime);
        if(bid==0)llaminar2::rocm::router_native::runtime_publish_independent_decode_dispatch_from_bank<CollectHistogram,
            llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal,PublishRoleTables,Runtime,
            llaminar2::rocm::router_native::RuntimeDecodePublicationValidation::ValidatedOriginalFragment>(inactive_bank,
                nullptr,nullptr,NR,ffn.moe.nused,ffn.moe.public_ids,ffn.moe.public_weights,true,
                ffn.moe.active_rows,ffn.moe.publication);
        // Original exact(false) has no live suffix scratch extent. The combined
        // public plane remains defined, with one cardinal writer for every word.
        for(unsigned column=unsigned(e0)+unsigned(tid);column<unsigned(e1);column+=NT)
            ffn.moe.out[column]=0.0f;
        return;
    }
    // Preserve the row-wide mixed staging for floating consumers. Quantization
    // delegates to the sole current32-lane production implementation.
    constexpr bool MIX_LDS=T<4;
    float4* mixl=reinterpret_cast<float4*>(arena);
    // One aligned packet serves both the existing mixed plane and exact Q8.
    for(int quad=tid;quad<T*(E/4);quad+=NT) {
        const float4 values=reinterpret_cast<const float4*>(mixed)[quad];
        if(MIX_LDS)mixl[quad]=values;
        const int token=quad/(E/4),column=4*(quad%(E/4));
        canonicalFFNQuantizeFour<ClosedRetainedPolicy>(values,size_t(column),canonical_hidden[token]);
    }
    lds_sync();
    if constexpr(ObserveCanonical) {
        if(bid==0) {
            for(int index=tid;index<T*E;index+=NT) {
                const auto& block=canonical_hidden[index/E][(index%E)/32];
                if(observation.hidden_codes)observation.hidden_codes[index]=reinterpret_cast<const int8_t*>(block.qs)[index%32];
            }
            for(int index=tid;index<T*(E/32);index+=NT) {
                const auto& block=canonical_hidden[index/(E/32)][index%(E/32)];
                if(observation.hidden_scales)observation.hidden_scales[index]=block.scale;
                if(observation.hidden_sums)observation.hidden_sums[index]=block.sum;
            }
        }
    }
    if (early.nq > 0) {
        // early q8_0 matrices: this block's share of all their rows. The first NT/8 rows take 8 lanes each (5 block pairs
        // per lane, one round); rows beyond that take a whole wave each (one pair per lane), so the leftover rows cost one
        // load deep instead of a second 5-deep round
        const int g0 = bid*early.rows_total/NB, g1 = (bid + 1)*early.rows_total/NB;
        const int l8 = lane % 8;
        const int gsplit = min(g1, g0 + NT/8);
        for (int g = g0 + tid/8; g < gsplit; g += NT/8) {
            int m = 0, r = g;
            while (m + 1 < early.nq && r >= early.rows[m]) { r -= early.rows[m]; ++m; }
            const int * wrow = (const int *) (early.w[m] + (size_t) r*(E/32*34));
            float acc[T];
#pragma unroll
            for (int t = 0; t < T; ++t) { acc[t] = 0.0f; }
#pragma unroll 1
            for (int k = 0; k < 5; ++k) {
                const int p = l8 + 8*k;
                int w[17];
#pragma unroll
                for (int q = 0; q < 17; ++q) { w[q] = wrow[17*p + q]; }
#pragma unroll
                for (int t = 0; t < T; ++t) { acc[t] += pair_dot(w, xqk[t][2*p], xqk[t][2*p + 1]); }
            }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                float v = acc[t];
                v = row8_sum(v);
                if (l8 == 0) { early.out[m][t*early.rows[m] + r] = v; }
            }
        }
#pragma unroll 1
        for (int g = gsplit + wave; g < g1; g += NT/64) {
            int m = 0, r = g;
            while (m + 1 < early.nq && r >= early.rows[m]) { r -= early.rows[m]; ++m; }
            const int p = min(lane, E/64 - 1);
            const int * wsrc = (const int *) (early.w[m] + (size_t) r*(E/32*34) + (size_t) p*68);
            int w[17];
#pragma unroll
            for (int q = 0; q < 17; ++q) { w[q] = wsrc[q]; }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                float v = lane < E/64 ? pair_dot(w, xqk[t][2*p], xqk[t][2*p + 1]) : 0.0f;
                v = wave_sum_f(v);
                if (lane == 0) { early.out[m][t*early.rows[m] + r] = v; }
            }
        }
    }
    HTS(14)
    // The same original shared row authority is also needed by wave0 at arrival.
    const int r0 = bid*ffn.ff/NB, ns = ffn.on ? (bid + 1)*ffn.ff/NB - r0 : 0;
    if (ffn.on || early.H > 0) {
        const int l0 = bid*NR/NB, nl = ffn.on ? (bid + 1)*NR/NB - l0 : 0;   // router rows
        const int a0 = bid*2*early.H/NB, na = (bid + 1)*2*early.H/NB - a0;          // GDN gate rows (alpha, then beta)
        const int nsg = ffn.on && bid == NB - 1;
        // T1 one wave owns two rows and both roles. Other templates retain
        // the original one-row task extent and source body.
        const int shared_tasks=T==1?(ns+1)/2:ns;
        const int ntask = nl + shared_tasks + nsg + na;
        // A physical wave has one task. Make that existing ownership explicit so
        // role-dependent descriptor addresses remain scalar, not lane-private state.
        const int uniform_task_wave=__builtin_amdgcn_readfirstlane(wave);
#pragma unroll 1
        for (int task = uniform_task_wave; task < ntask; task += NT/64) {
            if (task < nl || task >= nl + shared_tasks) {
                // f32 row . f32 mixed: router logits, the shared expert gate logit, or a GDN gate row
                const int ab = task - nl - shared_tasks - nsg;   // >= 0: GDN gate row a0 + ab
                if(task<nl) {
#pragma unroll
                    for(int token=0;token<T;++token) {
                        const float value=canonicalRouter128<true>(mixed+token*E,ffn.canonical_router_fp16,E,NR,l0+task,unsigned(lane));
                        if(lane==0)ffn.logits[token*NR+l0+task]=value;
                    }
                    continue;
                }
                if(ab<0) {
#pragma unroll
                    for(int token=0;token<T;++token) {
                        const float value=canonicalSharedGate256(mixed+token*E,ffn.w_sg,E,unsigned(lane));
                        if(lane==0)g_hcp_sg[token]=value;
                    }
                    continue;
                }
                const int hr = a0 + ab, is_b = hr >= early.H, h = is_b ? hr - early.H : hr;
                const float4 * w4 = task < nl ? (const float4 *) (ffn.w_logits + (size_t) (l0 + task)*E) :
                                    ab < 0    ? (const float4 *) ffn.w_sg :
                                                (const float4 *) ((is_b ? early.wb : early.wa) + (size_t) h*E);
                float acc[T];
#pragma unroll
                for (int t = 0; t < T; ++t) { acc[t] = 0.0f; }
                // 10 float4 per lane: batches of FB weight loads in flight (one load per iteration left the row 10 memory
                // round trips deep); mixed from LDS (T <= 3) or L2
                constexpr int FB = MIX_LDS ? 5 : 2;
#pragma unroll 1
                for (int k0 = lane; k0 < E/4; k0 += FB*64) {
                    float4 w[FB];
#pragma unroll
                    for (int b = 0; b < FB; ++b) { w[b] = w4[k0 + b*64]; }
#pragma unroll
                    for (int b = 0; b < FB; ++b) {
                        const int k = k0 + b*64;
#pragma unroll
                        for (int t = 0; t < T; ++t) {
                            const float4 m = MIX_LDS ? mixl[t*(E/4) + k] : ((const float4 *) mixed)[t*(E/4) + k];
                            acc[t] += w[b].x*m.x + w[b].y*m.y + w[b].z*m.z + w[b].w*m.w;
                        }
                    }
                }
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float v = wave_sum_f(acc[t]);
                    if (lane == 0) {
                        if (task < nl) {
                            ffn.logits[t*NR + l0 + task] = v;
                        } else if (ab < 0) {
                            g_hcp_sg[t] = v;
                        } else if (is_b) {
                            early.beta[t*early.H + h] = 1.0f/(1.0f + expf(-v));
                        } else {
                            const float u = v + early.dt[h];
                            early.gate[t*early.H + h] = (u > 20.0f ? u : logf(1.0f + expf(u)))*early.A[h]; // as ggml softplus
                        }
                    }
                }
            } else {
                const int shared_task=task-nl;
                if constexpr(T==1) {
                    constexpr unsigned parts=llaminar2::MoEProjectionNumericalContract::orderedKPartitionsForWidth(E);
                    const unsigned local_first=2*unsigned(shared_task);
                    const bool paired=local_first+1<unsigned(ns); // Wave-uniform real row count.
                    const unsigned column=unsigned(lane)&3u;
                    const unsigned local_row=local_first+(column&1u);
                    const auto role=(column&2u)?CanonicalSharedQ8Role::Up:CanonicalSharedQ8Role::Gate;
                    // Both descriptor records are borrowed before lane-role
                    // selection; only raw fields below may vary by lane.
                    const CanonicalSharedQ8RolePair roles(ffn.native_sgate,ffn.native_sup);
                    const auto fields=roles.select(role);
                    float partial=0.f;
                    if(local_row<unsigned(ns))
                        partial=canonicalSharedQ8Partial(fields,unsigned(r0)+local_row,
                            unsigned(lane)>>2,E,canonical_hidden[0]);
                    // Every lane reconverges before all original P transfers.
                    const float2 folded=canonicalSharedQ8Fold(partial,parts,unsigned(lane),paired);
                    const float gate=folded.x,up=folded.y;
                    if(lane<2&&local_first+unsigned(lane)<unsigned(ns)) {
                        const unsigned row=unsigned(r0)+local_first+unsigned(lane);
                        const float value=llaminar2::rocm::decode_equivalent::swigluValue(gate,up);
                        if constexpr(ClosedRetainedPolicy)g_hcp_mh[shared_start+row]=value;
                        else g_hcp_h[0][row]=value;
                        if constexpr(ObserveCanonical) {
                            if(observation.shared_gate)observation.shared_gate[row]=gate;
                            if(observation.shared_up)observation.shared_up[row]=up;
                            if(observation.shared_activated)observation.shared_activated[row]=value;
                        }
                    }
                    continue;
                }
                // shared expert gate row, then up row (same registers), r -> silu(gate)*up
                const int r = r0 + task - nl, p = min(lane, 39);
                int a[17];
#pragma unroll 1
                for (int h = 0; h < 2; ++h) {
                    native_q8_pair(h == 0 ? ffn.native_sgate : ffn.native_sup,r,2*p,a);
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        const float v = wave_sum_f(lane < 40 ? pair_dot(a, xqk[t][2*p], xqk[t][2*p + 1]) : 0.0f);
                        if (lane == 0) {
                            if (h == 0) { gl[t][0][wave] = v; }
                            else { const float g = gl[t][0][wave]; const float value=g/(1.0f + expf(-g))*v;
                                if constexpr(ClosedRetainedPolicy)g_hcp_mh[shared_start+r]=value;
                                else g_hcp_h[t][r]=value; }
                        }
                    }
                }
            }
        }
    }
    HTS(5)
    if (!ffn.on) {
        HTS(7)
        return;
    }
    // Every shared writer now finishes before the original all-wave entry
    // VM0/LGKM0 + CTA join. Default grid_sync_all drains those GLOBAL/public
    // stores before scalar arrival, with the same counter/poll/exit join. No
    // callback reads gl or publishes after entry in the T1 path.
    grid_sync_all(join);
    HTS(6)
    if (ffn.moe.on) {
        const hc_moe & mo = ffn.moe;
        static_assert(T==1,"The first acquired-bank adapter is explicitly M1 only");
        using Runtime=llaminar2::rocm::native_runtime_abi::DeviceMoELayerRuntimeView;
        using Acquired=llaminar2::rocm::router_native::AcquiredRuntimeDecodeBank<Runtime>;
        const auto acquired=Acquired::acquire(mo.runtime);
        using Fragment=llaminar2::rocm::router_native::ValidatedOriginalFragmentBank<Runtime,NR>;
        using Tables=llaminar2::rocm::fast_native::AcquiredNativeExpertTables<Acquired>;
        const auto fragment=Fragment::bind(acquired,mo.nused);
        const auto tables=Tables::fromBoundFragment(fragment);
        const auto native_gate=tables.gate();
        const auto native_up=tables.up();
        const auto native_down=tables.down();
        unsigned int push_s = 0;
        long long pofs = 0;              // this exchange's half in every peer inbox
        if (mo.push.n) {
            unsigned int s0;
            __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(s0) : "s"(&g_hcp_arseq) : "memory");
            push_s = s0 + 1;
            pofs = (long long) (push_s & 1)*mo.push.hs;
        }
        const int NU = mo.nused, ffr = mo.ffr, NBK = ffr/32;
        const int wu_ = __builtin_amdgcn_readfirstlane(wave);   // wave-uniform in SGPRs
        constexpr bool OwnedSlotPrefix=!ObserveCanonical&&!PublishRoleTables;
        static_assert(!OwnedSlotPrefix||llaminar2::rocm::router_native::kDeviceMoEMaxTopK<=NB);
        // A peer CTA owns one original gate/up slot. Only CTA0 needs the complete
        // selected row before the existing GU grid join to publish global state.
        const unsigned owned_slot=((unsigned(bid)+1)*unsigned(NU)-1)/NB;
        const int selection_extent=OwnedSlotPrefix&&bid!=0?int(owned_slot+1):NU;
        const bool normalize_here=ffn.canonical_normalize_weights&&(!OwnedSlotPrefix||bid==0);
        // All threads enter the ACTUAL current logical256 selector joins.
        // Its scratch borrows the retired mixed plane, before coefficient-panel use.
        for(int token=0;token<T;++token) {
            auto& selection=*reinterpret_cast<llaminar2::rocm::router_native::RouterSelectionShared*>(arena);
            llaminar2::rocm::router_native::selectProbabilities<false,256,false,
                llaminar2::rocm::WorkgroupJoinMemory::LDSOnly,
                llaminar2::rocm::router_native::TopKTransferPolicy::RowNativeRoot,
                llaminar2::rocm::router_native::SelectorIntermediateStorage::ThreadPair,!ObserveCanonical,ClosedRetainedPolicy>(ffn.logits+token*NR,nullptr,NR,selection_extent,
                    normalize_here,selection);
            if(tid<NU&&(!OwnedSlotPrefix||bid==0||unsigned(tid)==owned_slot)) {
                sel_e[token][tid]=selection.selected_ids[tid];
                if constexpr(!OwnedSlotPrefix)sel_w[token][tid]=selection.selected_weights[tid];
            }
            if constexpr(ObserveCanonical) {
                if(bid==0) {
                    if(tid<NR&&observation.router_probabilities)observation.router_probabilities[token*NR+tid]=selection.expert_probs[tid];
                    if(tid<NU){
                        if(observation.selected_ids)observation.selected_ids[token*NU+tid]=selection.selected_ids[tid];
                        if(observation.selected_weights)observation.selected_weights[token*NU+tid]=selection.selected_weights[tid];
                    }
                }
            }
            // Retain the original runtime publisher, once per logical slot.
            // Other CTAs resolve only their job mask, never public state/counters.
            if(tid<64) {
                bool eligible=false;
                if(tid<NU&&(!OwnedSlotPrefix||bid==0||unsigned(tid)==owned_slot)) {
                    const int expert=selection.selected_ids[tid];
                    bool local;
                    if(bid==0) {
                        local=llaminar2::rocm::router_native::runtime_publish_bound_original_fragment_dispatch<false,CollectHistogram&&!PublishRoleTables&&!ObserveCanonical,CollectHistogram>(
                            fragment,selection.selected_ids,selection.selected_weights,
                            mo.public_ids,mo.public_weights,mo.active_rows,mo.publication);
                    } else {
                        const auto route=fragment.select(expert);
                        local=route.disposition==llaminar2::MoEIntermediateShardRowDisposition::Local;
                    }
                    eligible=local&&tables.requireOriginalNativeLocalFormat(expert);
                }
                const auto mask=__ballot(eligible);
                if(lane==0)native_route_masks[token]=static_cast<uint32_t>(mask);
            }
            lds_sync();
        }
        HTS(11)
        // distinct experts over the call's T*NU picks (wave 0, identical in every block): entry i = t*NU + slot;
        // an entry is the first of its expert if no lower entry has it; its users are in entry order
        // One-token entries already are selected slots with one user. The
        // original selected IDs/weights remain their sole LDS authority.
        if constexpr (T>1) {
            if (wu_==0) {
                const int ne = T*NU;
                const int x = lane < ne ? (&sel_e[0][0])[(lane / NU)*NUMAX + lane % NU] : -1 - lane;
                unsigned long long same = 1ull << lane;
                if (T > 1 && mo.dedup) {
                    for (int j = 0; j < ne; ++j) {
                        const int xj = __builtin_amdgcn_readlane(x, j);
                        same |= xj == x ? 1ull << j : 0ull;
                    }
                }
                const unsigned long long lt = (1ull << lane) - 1;
                const bool first = lane < ne && (same & lt) == 0;
                const unsigned long long fm = __ballot(first);
                const int fj = __ffsll((unsigned long long) same) - 1;               // first entry of my expert
                const int di = __popcll(fm & ((1ull << fj) - 1));                   // its distinct index
                const int k  = __popcll(same & lt);                                 // my rank among its users
                // position in the expert-grouped entry list: the entries of experts that appear earlier, then my rank
                int before = 0;
                for (int j = 0; j < ne; ++j) { before += __builtin_amdgcn_readlane(fj, j) < fj ? 1 : 0; }
                if (lane < ne) {
                    f_ent[before + k] = lane;
                    f_tok[before + k] = lane / NU;
                    d_ent[di][k] = lane;
                    d_tok[di][k] = lane / NU;
                    if (first) { d_ex[di] = x; d_n[di] = __popcll(same); }
                }
                if (lane == 0) { d_cnt = __popcll(fm); }
            }
        }
        lds_sync();
        const int ND = T==1 ? NU : d_cnt;
        HTS(8)
        // Assign one selected entry to each CTA cohort. Cardinal aligned
        // chunks balance real output rows across the cohort; cyclic entry
        // rounds cover more entries than CTAs without a global partial bank.
        {
            constexpr bool Q4=GateFormat==qfn_castagna_reference::GateSource::Q4_K;
            static_assert(Q4 || GateFormat==qfn_castagna_reference::GateSource::Q5_K,
                "The retained entry uses canonical native affine Q4 or Q5 gate/up");
            constexpr unsigned WaveRows=NativeQ4PanelContract::rows_per_wave;
            constexpr unsigned Capacity=NativeQ4PanelContract::row_capacity;
            static_assert(NT%64==0 && Capacity==WaveRows*(NT/64) && E%512==0,
                "The original coefficient panel must cover every native wave");
            const unsigned entries=unsigned(T)*unsigned(NU);
            const unsigned cohorts=entries<unsigned(NB)?entries:unsigned(NB);
            if(cohorts>0) {
                const unsigned cohort=((unsigned(bid)+1)*cohorts-1)/unsigned(NB);
                const unsigned cohort_first=cohort*unsigned(NB)/cohorts;
                const unsigned cohort_end=(cohort+1)*unsigned(NB)/cohorts;
                const unsigned owners=cohort_end-cohort_first,rank=unsigned(bid)-cohort_first;
                const unsigned grain=native_gu_row_grain(unsigned(ffr),owners);
                const unsigned chunks=(unsigned(ffr)+grain-1)/grain;
                const unsigned first=(rank*chunks/owners)*grain;
                const unsigned end_chunk=((rank+1)*chunks/owners)*grain;
                const unsigned last=end_chunk<unsigned(ffr)?end_chunk:unsigned(ffr);
                if(first<last) {
                    for(unsigned f=cohort;f<entries;f+=cohorts) {
                        const int slot=T==1?int(f):__builtin_amdgcn_readfirstlane(f_ent[f]);
                        const int t=T==1?0:__builtin_amdgcn_readfirstlane(f_tok[f]),e=slot-t*NU;
                        const auto selected_job=llaminar2::rocm::fast_native::PublishedNativeRoute::borrow(
                            sel_e[t],native_route_masks[t],unsigned(e));
                        const bool native_job=bool(selected_job);
                        if(!native_job) {
                            // A legitimate unavailable route has no source access.
                            // Publish the defined empty local contribution instead.
                            for(unsigned output=first+unsigned(tid);output<last;output+=NT) {
                                g_hcp_mh[unsigned(slot)*routed_stride+output]=0.0f;
                                if constexpr(ObserveCanonical) {
                                    const unsigned index=unsigned(slot*ffr)+output;
                                    if(observation.routed_gate)observation.routed_gate[index]=0.0f;
                                    if(observation.routed_up)observation.routed_up[index]=0.0f;
                                    if(observation.routed_activated)observation.routed_activated[index]=0.0f;
                                }
                            }
                            // This condition is CTA-uniform. No peer needs a
                            // panel join because this job never touches a panel.
                            continue;
                        }
                        const NativeMatrix& gate_matrix=native_gate[selected_job];
                        const NativeMatrix& up_matrix=native_up[selected_job];
#pragma unroll 1
                        for(unsigned tile=first;tile<last;tile+=Capacity) {
                            const unsigned count=last-tile<Capacity?last-tile:Capacity;
                            auto* panel=reinterpret_cast<unsigned*>(arena);
                            // All1024 workers reach both panel calls and this
                            // join, including inactive output waves and columns.
                            native_q4_panel_publish(&gate_matrix,tile,count,unsigned(tid),NT,E/32,panel);
                            native_q4_panel_publish(&up_matrix,tile,count,unsigned(tid),NT,E/32,panel+(E/32)*Capacity);
                            lds_sync();
                            const unsigned wave_row=unsigned(wu_)*WaveRows;
                            if(wave_row<count) {
                                const unsigned local_row=wave_row+(unsigned(lane)&(WaveRows-1));
                                const int r=int(tile+local_row);
                                constexpr unsigned parts=llaminar2::MoEProjectionNumericalContract::orderedKPartitionsForWidth(E);
                                constexpr unsigned blocks=E/32,partition_width=(blocks+parts-1)/parts;
                                const unsigned part=unsigned(lane>>2);
                                float gg=0.f,uu=0.f;
                                const NativeTaskFields gate_fields(&gate_matrix);
                                if(local_row<count) {
#pragma unroll 1
                                    for(unsigned block=part*partition_width;block<blocks&&block<(part+1)*partition_width;++block) {
                                        const auto& activation=canonical_hidden[t][block];
                                        if constexpr(Q4) {
                                            const auto wg=native_qk_item<false,true>(gate_fields,r,int(block/8),int((block%8)/2),int(block&1),panel,tile);
                                            const auto wu=native_qk_item<false,true>(&up_matrix,r,int(block/8),int((block%8)/2),int(block&1),panel+(E/32)*Capacity,tile);
                                            canonicalQ4PairCommit(wg,wu,activation,gg,uu);
                                        } else {
                                            // Native Q5_K uses the same executable K32 block as Q5_1.
                                            // Source-superblock provenance was authenticated before publication.
                                            namespace numeric=llaminar2::device_native_quantised_contract;
                                            // Q5 uses the identical affine metadata planes. Its payload
                                            // and high bits remain eight original native dot4 operands.
                                            const auto field=native_q4_panel_index(local_row,block);
                                            int words[6];
                                            native_q51_block<false>(gate_fields,r,int(block),words,panel[field]);
                                            gg=numeric::accumulate(gg,currentDownQ51Block(words,activation));
                                            native_q51_block<false>(&up_matrix,r,int(block),words,panel[(E/32)*Capacity+field]);
                                            uu=numeric::accumulate(uu,currentDownQ51Block(words,activation));
                                        }
                                    }
                                }
                                // Reconverge all64 peers before the original
                                // ordered P roots. Missing columns have no read
                                // or publication; real columns keep all P leaves.
                                gg=canonical_gu::foldFourColumns(gg,parts,unsigned(lane));
                                uu=canonical_gu::foldFourColumns(uu,parts,unsigned(lane));
                                if(lane<int(WaveRows) && wave_row+unsigned(lane)<count) {
                                    const unsigned output=tile+wave_row+unsigned(lane);
                                    const float value=llaminar2::rocm::decode_equivalent::swigluValue(gg,uu);
                                    g_hcp_mh[unsigned(slot)*routed_stride+output]=value;
                                    if constexpr(ObserveCanonical) {
                                        const unsigned index=unsigned(slot*ffr)+output;
                                        if(observation.routed_gate)observation.routed_gate[index]=gg;
                                        if(observation.routed_up)observation.routed_up[index]=uu;
                                        if(observation.routed_activated)observation.routed_activated[index]=value;
                                    }
                                }
                            }
                            // Retire every local consumer before this CTA's
                            // next tile or entry. The unchanged final grid join
                            // retires the last panel; other CTAs need not align
                            // their numbers of local rounds.
                            if(last-tile>Capacity || entries-f>cohorts)lds_sync();
                        }
                    }
                }
            }
        }
        HTS(9)
        grid_sync_all(join);
        HTS(10)
        if constexpr(OwnedSlotPrefix) {
            // The original joined CTA0 stores are the only new source of the
            // complete local row. Nonlocal slots carry the original -1/0.
            // The last physical wave is outside the cold tuple's 688 Q8
            // quads. Overlap this original ledger read with canonical Q8 and
            // publish the copied row at the existing Q8 LDS join.
            if(tid>=NT-64&&lane<NU) {
                const auto* ledger=static_cast<const volatile Runtime*>(mo.runtime);
                sel_e[0][lane]=ledger->topk_expert_ids[lane];
                sel_w[0][lane]=ledger->topk_weights[lane];
            }
        }
        // Canonical Q8 publication keeps its exact32-lane reduction trees.
        // Each block writes40B of original codes/scale/integer-sum metadata;
        // its activation values and consumed live extent are unchanged.
        auto* dq = reinterpret_cast<CanonicalFFNQ8Block*>(arena);
        const unsigned routed_values = unsigned(T * NU * ffr);
        const unsigned shared_values = unsigned(T * ffn.ff);
        if constexpr (ObserveCanonical) if (blockIdx.x == 0) {
            for (unsigned item = unsigned(tid); item < unsigned(T * NU); item += unsigned(NT)) {
                const unsigned token = item / unsigned(NU), route = item % unsigned(NU);
                if (observation.selected_ids) observation.selected_ids[item] = sel_e[token][route];
                if (observation.selected_weights) observation.selected_weights[item] = sel_w[token][route];
            }
        }
        for (unsigned quad = unsigned(tid); quad < (routed_values + shared_values)/4;
             quad += unsigned(NT)) {
            const unsigned linear=quad*4;
            const float* source=ClosedRetainedPolicy ? &g_hcp_mh[linear] : linear<routed_values
                ? &g_hcp_mh[(linear/unsigned(ffr))*FFRMAX + linear%unsigned(ffr)]
                : &g_hcp_h[(linear-routed_values)/unsigned(ffn.ff)]
                           [(linear-routed_values)%unsigned(ffn.ff)];
            const float4 values=*reinterpret_cast<const float4*>(source);
            const unsigned packed=canonicalFFNQuantizeFour<ClosedRetainedPolicy>(values,linear,dq);
            if constexpr (ObserveCanonical) if (blockIdx.x == 0) {
                const bool routed = linear < routed_values;
                const unsigned coordinate = routed ? linear : linear - routed_values;
                auto* codes = routed ? observation.routed_codes : observation.shared_codes;
                auto* scales = routed ? observation.routed_scales : observation.shared_scales;
                auto* sums = routed ? observation.routed_sums : observation.shared_sums;
                const auto& block = dq[linear / 32];
                if(codes)for(unsigned component=0;component<4;++component)
                    codes[coordinate+component]=__builtin_bit_cast(int8_t,uint8_t(packed>>(component*8)));
                if ((linear % 32) == 0) {
                    if (scales) scales[coordinate / 32] = block.scale;
                    if (sums) sums[coordinate / 32] = block.sum;
                }
            }
        }
        if (tid < T) pw[tid] = 1.0f / (1.0f + expf(-g_hcp_sg[tid]));
        if constexpr (ObserveCanonical)
            if (blockIdx.x == 0 && tid < T && observation.shared_gate_value)
                observation.shared_gate_value[tid] = pw[tid];
        lds_sync();
        HTS(12)
        const auto* sq = dq + T * NU * NBK;
        // The closed Q8-down tuple seals both down matrices at H.
        // Narrow their address stride without changing either K partition.
        constexpr bool FixedQ8Down=ClosedRetainedPolicy &&
            RoutedDownCodebook==19;
        // Route/column owners retain each expert's complete K fold in
        // registers. A small LDS bank exposes only finished weighted routes,
        // permitting independent experts to use different physical waves.
        // One CTA publication edge replaces all intra-dot K shuffles.
        // Every positive native slice is 32-column aligned. The widest
        // retained source can fill all routed slots plus the shared bank.
        // Closed Q8 down is admitted only for H2560 on exactly 60 CTAs,
        // with one live row and ten selected slots. Every CTA owns up to
        // one wave of columns, so each of the eleven waves owns one route.
        constexpr bool OneWavePerRoute = FixedQ8Down;
        static_assert(!OneWavePerRoute || (E==2560 && NB==60 && T==1 && (E+NB-1)/NB<=64));
        constexpr int max_q8_blocks=T*(NUMAX*FFRMAX+FF)/32;
        constexpr int max_output_columns=(E+NB-1)/NB;
        const unsigned first_column=unsigned(e0);
        const unsigned columns=unsigned(n);
        static_assert(sizeof(arena)>=max_q8_blocks*sizeof(CanonicalFFNQ8Block)+
            T*(NUMAX+1)*max_output_columns*sizeof(float),
            "Retained native quantized inputs and finished route rows must fit the phase LDS");
        auto* completed = reinterpret_cast<float*>(dq + T * (NU * NBK + unsigned(ffn.ff / 32)));
        const unsigned columns_per_wave = 64;
        const unsigned tiles = OneWavePerRoute ? 1u : (columns + columns_per_wave - 1) / columns_per_wave;
        const unsigned route_count = OneWavePerRoute ? ::llaminar2::rocm::retainedM1SelectedExperts + 1 : unsigned(NU + 1);
        const unsigned wave = unsigned(tid) / columns_per_wave;
        const unsigned lane_column = unsigned(tid) % columns_per_wave;
        const unsigned tasks = unsigned(T) * route_count * tiles;
        for (unsigned task = wave; task < tasks; task += unsigned(NT) / columns_per_wave) {
            const unsigned tile = task % tiles;
            const unsigned row_route = task / tiles;
            const unsigned token = OneWavePerRoute ? 0u : row_route / route_count;
            const unsigned route = OneWavePerRoute ? task : row_route % route_count;
            const unsigned local = tile * columns_per_wave + lane_column;
            if (local < columns) {
                const unsigned column = first_column + local;
                float value;
                if (route < unsigned(NU)) {
                    using NativeRoute=llaminar2::rocm::fast_native::PublishedNativeRoute;
                    const auto selected_job=[&] {
                        if constexpr(OwnedSlotPrefix)return NativeRoute::borrowJoinedLocalDispatch(sel_e[token],route);
                        else return NativeRoute::borrow(sel_e[token],native_route_masks[token],route);
                    }();
                    if(selected_job) {
                        const float down = currentDownRow<RoutedDownCodebook,
                            ((RoutedDownCodebook==7&&!ObserveCanonical&&!PublishRoleTables)||FixedQ8Down)?E:0>(native_down + selected_job, column,
                            unsigned(ffr), dq + (token * unsigned(NU) + route) * unsigned(NBK));
                        value = ::llaminar2::device_native_quantised_contract::weightRoute(
                            sel_w[token][route], down);
                    } else value=0.0f;
                    if constexpr (ObserveCanonical)
                        if (observation.route_contributions)
                            observation.route_contributions[
                                (size_t(token) * NU + route) * E + column] = value;
                } else {
                    const float down = currentDownRow<19,FixedQ8Down?E:0>(ffn.native_sdown, column,
                        unsigned(ffn.ff), sq + token * unsigned(ffn.ff / 32));
                    value = ::llaminar2::device_native_quantised_contract::accumulate(0.0f,
                        ::llaminar2::device_native_quantised_contract::weightRoute(1.0f, down));
                }
                completed[(token * route_count + route) * columns + local] = value;
            }
        }
        lds_sync();
        for (unsigned local = unsigned(tid); local < columns; local += unsigned(NT)) {
#pragma unroll
            for (unsigned token = 0; token < unsigned(T); ++token) {
                float routed = 0.0f;
                for (unsigned route = 0; route < unsigned(NU); ++route)
                    routed = ::llaminar2::device_native_quantised_contract::accumulate(routed,
                        completed[(token * route_count + route) * columns + local]);
                const float shared = ::llaminar2::device_fp32_contract::multiply(pw[token],
                    completed[(token * route_count + unsigned(NU)) * columns + local]);
                const float value = ::llaminar2::device_fp32_contract::add(routed, shared);
                const unsigned column = first_column + local;
                if constexpr (ObserveCanonical)
                    if (observation.routed_output)
                        observation.routed_output[size_t(token) * E + column] = routed;
                if constexpr (ObserveCanonical)
                    if (observation.shared_output)
                        observation.shared_output[size_t(token) * E + column] = shared;
                mo.out[token * E + column] = value;
#pragma unroll
                for (int peer = 0; peer < (AR4 ? ARNS : 1); ++peer)
                    if (peer < mo.push.n)
                        mo.push.inbox[peer][pofs + token * E + column] = value;
            }
        }
        if constexpr(PublishRoleTables) {
            // Completes in this original kernel before the node/stream and optional
            // peer signal. Numerical roles consumed the same leased bank directly.
            // All three original NativeQuantised outputs remain required.
            // Give each role its own 16 lanes of the existing selector wave.
            // Every local slot passed the original format and bank checks
            // before this ballot. Each admitted slot/role has one writer.
            constexpr unsigned RoleSlots=llaminar2::rocm::router_native::kDeviceMoEMaxTopK;
            static_assert(RoleSlots==16);
            const unsigned role=unsigned(lane)/RoleSlots;
            const unsigned slot=unsigned(lane)%RoleSlots;
            if(bid==0&&tid<64&&role<3&&((native_route_masks[0]>>slot)&1u)) {
                const auto& packet=mo.publication;
                if(packet.format!=llaminar2::DeviceMoEWeightFormat::NativeQuantised||
                   packet.projections!=llaminar2::DeviceMoEProjectionSet::IntermediateShard||
                   packet.expert_count!=NR||!packet.gate||!packet.up||!packet.down)
                    __builtin_trap();
                const unsigned expert=unsigned(sel_e[0][slot]);
                const auto& record=fragment.original().bank().experts[expert];
                const auto* source=role==0?&record.gate:role==1?&record.up:&record.down;
                auto* destination=role==0?packet.gate:role==1?packet.up:packet.down;
                __builtin_memcpy(destination+expert,source,sizeof(*source));
            }
        }
        if (mo.push.n) {
            // fused AllReduce push: this block's rows are in the peers' inboxes; the last block raises the peers' flags
            __threadfence_system();
            __asm__ volatile("s_waitcnt vmcnt(0)" ::: "memory");
            lds_sync();
            if (tid == 0) {
                const unsigned int old = atomicAdd(&g_hcp_pushcnt, 1u);
                if (old % NB == NB - 1) {
                    __threadfence_system();
                    for (int k = 0; k < mo.push.n; ++k) { *(volatile int *) mo.push.flag[k] = (int) push_s; }
                    __threadfence_system();
                    atomicAdd(&g_hcp_arseq, 1u);
                    g_hcp_pushed = 1u;
                }
            }
        }
        HTS(7)
        return;
    }
    // shared expert down: h -> q8_1 in LDS (xqk rows reused), rows [e0, e1) x 10 block pairs, times sigmoid(sg)
    const int ff4 = ffn.ff/4;
    for (int i4 = tid; i4 < T*ff4; i4 += NT) {
        const int t = i4 / ff4, j = i4 % ff4;
        const float4 v = ((const float4 *) g_hcp_h[t])[j];
        const float amax = row8_max(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        const float d = amax/127.0f, dd = amax == 0.0f ? 1.0f : d; // as quantize_q8_1: q = round(x/d)
        char4 q;
        const float id = __builtin_amdgcn_rcpf(dd); // one reciprocal instead of 4 IEEE divisions (x/d rounds the same
            // except within an ulp of a .5 boundary)
            q.x = roundf(v.x*id); q.y = roundf(v.y*id); q.z = roundf(v.z*id); q.w = roundf(v.w*id);
        xqk[t][j/8].qs[j % 8] = *(int *) &q;
        if (j % 8 == 0) { xqk[t][j/8].ds = make_half2(d, 0.0f); }
    }
    if (tid < T) { pw[tid] = 1.0f/(1.0f + expf(-g_hcp_sg[tid])); }
    lds_sync();
    const int PD = ffn.ff/64;             // block pairs per down row (10, or 5 for a shard)
    if (tid < n*PD) {
        const int i = tid / PD, p = tid % PD;
        int w[17];
        native_q8_pair(ffn.native_sdown,e0+i,2*p,w);
#pragma unroll
        for (int t = 0; t < T; ++t) { part[t][tid] = pair_dot(w, xqk[t][2*p], xqk[t][2*p + 1]); }
    }
    lds_sync();
    for (int it = tid; it < T*n; it += NT) {
        const int t = it / n, i = it % n;
        float s2 = 0.0f;
        for (int p = 0; p < PD; ++p) { s2 += part[t][i*PD + p]; }
        g_hcp_shexp[t][e0 + i] = s2*pw[t];
    }
    HTS(7)
#undef HTS
#endif // gfx9 device code
}

} // namespace
